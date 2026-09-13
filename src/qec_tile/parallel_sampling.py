"""시드로 재현되는 병렬 shot 수집.

``sinter_sampling.collect``의 대안이다. sinter는 시드를 넘길 자리가 없어서
(``_stim_then_decode_sampler``가 ``compile_detector_sampler()``를 인자 없이 부른다)
실행마다 다른 shot을 뽑고, 배치 크기도 벽시계 시간으로 정해진다. 재현이 필요하면
이 경로를 쓴다.

결과는 ``(circuit, shots, decoder, seed, chunk, max_errors)``만으로 정해진다 --
``workers``는 들어가지 않는다. shot을 고정 크기 ``chunk``로 쪼개고 조각 인덱스에서
시드를 유도하므로, 코어 수가 다른 머신에서도 같은 데이터가 나온다.
"""
from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import time

import numpy as np
import stim
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices

from .decode import DECODERS, OBSERVABLE_DECODERS, FailureCounts


def _format_duration(seconds: float) -> str:
    """5s / 12m30s / 3.4h."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m{seconds % 60:02.0f}s"
    return f"{seconds / 3600:.1f}h"


def _chunk_seed(seed: int, index: int) -> int:
    """조각마다 다른 샘플러 시드.

    ``SeedSequence``가 섞어주므로 인접한 조각이 상관되지 않는다. 모든 조각이 같은
    시드를 쓰면 같은 shot을 조각 수만큼 반복해 통계가 망가진다.
    """
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


# worker가 조각마다 DEM을 다시 짓지 않도록 프로세스 전역에 한 번 만든다.
_WORKER: dict = {}


def _init_worker(circuit_text: str, decoder: str,
                 max_iter: int | None) -> None:
    """Pool 초기화: 회로·DEM 행렬·디코더를 프로세스당 한 번 만든다.

    조각마다 디코더를 만들면 무거운 디코더에서 재생성 비용이 조각 수만큼 붙는다 --
    sinter 경로가 배치마다 앙상블을 다시 지어 ``MemoryError``까지 갔던 그 문제다.
    """
    # Ctrl+C는 부모만 처리한다. worker마다 traceback을 찍으면 24개분이 쏟아진다.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    circuit = stim.Circuit(circuit_text)
    matrices = detector_error_model_to_check_matrices(
        circuit.detector_error_model(), allow_undecomposed_hyperedges=True)
    observables = matrices.observables_matrix.toarray().astype(np.uint8)
    if decoder in DECODERS:
        build = DECODERS[decoder]
        args = (matrices.check_matrix, matrices.priors)
    elif decoder in OBSERVABLE_DECODERS:            # coset 판별에 O가 필요하다
        build = OBSERVABLE_DECODERS[decoder]
        args = (matrices.check_matrix, matrices.priors, observables)
    else:
        raise ValueError(f"unknown decoder {decoder!r}; have "
                         f"{sorted(set(DECODERS) | set(OBSERVABLE_DECODERS))}")
    # max_iter가 None이면 인자를 아예 안 넘긴다 -- 넘기면 람다의 기본값이 안 먹고
    # None이 그대로 디코더까지 흘러간다.
    _WORKER.update(
        circuit=circuit, observables=observables,
        decoder=(build(*args) if max_iter is None
                 else build(*args, max_iter)))


def _run_chunk(task: tuple[int, int, int]) -> tuple[int, int, int]:
    """조각 하나를 샘플링해 디코딩하고 ``(shots, block_fails, flips)``.

    채점은 ``circuit.circuit_failure_counts``와 같다: 예측 observable을 참값과
    비교해 하나라도 다르면 블록 실패이고, 다른 개수만큼 flip에 더한다.
    """
    index, size, seed = task
    circuit = _WORKER["circuit"]
    observables = _WORKER["observables"]
    decoder = _WORKER["decoder"]
    detections, observed = circuit.compile_detector_sampler(
        seed=_chunk_seed(seed, index)).sample(size, separate_observables=True)
    block_fails = flips = 0
    for detection, actual in zip(detections, observed):
        prediction = (observables @ decoder.decode(
            detection.astype(np.uint8))) % 2
        mismatch = prediction != actual
        block_fails += bool(mismatch.any())
        flips += int(mismatch.sum())
    return size, block_fails, flips


def parallel_failure_counts(circuit, shots: int, decoder: str, *,
                            seed: int = 42, workers: int = 8,
                            chunk: int = 100,
                            max_errors: int | None = None,
                            max_iter: int | None = None,
                            progress: bool = True, start_chunk: int = 0,
                            start_counts: tuple[int, int, int] | None = None,
                            checkpoint=None) -> FailureCounts:
    """``shots``개를 병렬로 디코딩한다. 시드만 같으면 재현된다.

    ``max_iter``가 None이면 디코더의 레지스트리 기본값을 쓴다.

    ``max_errors``는 조각 경계에서만 멈춘다 -- 조각을 인덱스 순서대로(``imap``)
    소비하므로 어느 worker가 먼저 끝냈는지와 무관하게 멈추는 지점이 결정적이다.
    ``imap_unordered``로 바꾸면 그 성질이 사라진다.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    k = circuit.num_observables
    sizes = [min(chunk, shots - start) for start in range(0, shots, chunk)]
    total, block_fails, flips = start_counts or (0, 0, 0)
    # 조각 번호는 그대로 두고 앞부분만 잘라낸다 -- 번호가 시드를 정하므로 다시
    # 매기면 이어받은 조각이 처음부터 돌렸을 때와 다른 shot을 뽑는다.
    tasks = [(i, size, seed) for i, size in enumerate(sizes)][start_chunk:]
    if not tasks or (max_errors is not None and block_fails >= max_errors):
        return FailureCounts(total, block_fails, flips, k)

    done = start_chunk
    session = 0                      # 이번 실행에서 민 shot (ETA는 이 기준이다)
    start = time.monotonic()
    last = 0.0
    # fork는 스레드를 띄운 부모(numpy/BLAS)에서 자식이 교착에 빠질 수 있고,
    # Windows는 어차피 spawn이다. 명시해서 두 플랫폼 동작을 같게 만든다.
    # initargs가 (str, str, int|None)뿐이라 pickle도 문제없다.
    if progress:
        # 조각이 크면 첫 진행 줄이 수십 분 뒤에 나온다. 시작했다는 것과 조각 수를
        # 먼저 알려야 죽은 건지 도는 건지 알 수 있다.
        print(f"  collecting {shots:,} shots, {len(tasks):,} chunks of {chunk}, "
              f"{workers} workers", file=sys.stderr, flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(str(circuit), decoder, max_iter)) as pool:
        for size, fails, chunk_flips in pool.imap(_run_chunk, tasks):
            total += size
            session += size
            block_fails += fails
            flips += chunk_flips
            done += 1
            if checkpoint is not None:      # 쓰기 빈도는 호출자가 정한다
                checkpoint(done, total, block_fails, flips)
            now = time.monotonic()
            # 조각이 수천 개라 매번 찍으면 출력이 병목이 된다. stderr로 보내고
            # \r로 한 줄을 덮어쓴다 (CSV/stdout은 깨끗하게 둔다).
            if progress and (now - last > 1.0 or total == shots):
                last = now
                elapsed = now - start
                # total에는 이전 실행분이 섞여 있어 속도 계산에 쓰면 안 된다.
                eta = (shots - total) * elapsed / session
                if max_errors is not None and block_fails:
                    # --max-errors면 보통 오류 쪽이 먼저 찬다. shot 기준만 쓰면
                    # "1000만까지 며칠"이라는 쓸모없는 숫자가 나온다.
                    eta = min(eta, (max_errors - block_fails) * elapsed
                              / block_fails)
                errs = ("" if max_errors is None
                        else f" errors {block_fails}/{max_errors}")
                print(f"\r  {total:,}/{shots:,} shots{errs}  "
                      f"{_format_duration(elapsed)} elapsed  "
                      f"eta {_format_duration(max(eta, 0))}   ",
                      end="", file=sys.stderr, flush=True)
            if max_errors is not None and block_fails >= max_errors:
                break                  # 조각 단위로만 멈춘다 -- 결정적이다
    if progress:
        print(file=sys.stderr)
    return FailureCounts(total, block_fails, flips, k)
