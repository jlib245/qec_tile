"""시드로 재현되는 병렬 shot 수집.

sinter는 ``compile_detector_sampler()``에 시드를 넘길 자리가 없어서 실행마다 다른
shot을 뽑는다. 이 모듈은 그것을 대체하므로, 재현성과 worker 수 독립성이 전부다.
"""
import stim

from qec_tile.parallel_sampling import _chunk_seed, parallel_failure_counts


def small_circuit():
    """초 단위로 끝나는 회로. d=3, 3라운드.

    p=0.05는 200 shot에 오류 56개를 낸다 (p=0.02면 10개). 오류가 적으면 두 실행이
    서로 다른 shot을 봐도 카운트가 우연히 일치해 재현성 테스트가 헛돈다.
    """
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=3, rounds=3,
                                  after_clifford_depolarization=0.05)


def chunk_trace(circuit, **kwargs):
    """조각마다의 누적 ``(done, shots, fails, flips)`` 수열.

    총합 하나만 비교하면 우연 일치가 10%대다 (오류 56개면 표준편차가 6). 조각
    4개의 수열을 비교하면 그 확률이 자릿수로 떨어진다.
    """
    seen = []
    parallel_failure_counts(circuit, 200, "bposd_0", seed=3, chunk=50,
                            progress=False,
                            checkpoint=lambda *a: seen.append(a), **kwargs)
    return seen


def test_counts_are_reproducible():
    """같은 인자면 조각 수열까지 완전히 같다 — sinter가 못 하는 것."""
    circuit = small_circuit()
    assert chunk_trace(circuit, workers=2) == chunk_trace(circuit, workers=2)


def test_the_worker_count_does_not_change_the_data():
    """worker 수는 결과에 들어가지 않는다.

    조각을 worker 수로 나누면 코어 수가 데이터를 바꿔 머신 간 재현이 깨진다.
    고정 ``chunk``를 쓰는 설계가 여기에 걸려 있다.
    """
    circuit = small_circuit()
    assert chunk_trace(circuit, workers=2) == chunk_trace(circuit, workers=4)


def test_chunk_seeds_are_distinct():
    """조각마다, 그리고 시드마다 다른 샘플러 시드.

    모든 조각이 같은 시드를 쓰면 같은 shot을 조각 수만큼 반복해 통계가 망가진다 --
    에러 없이 조용히 틀리는 종류다. 카운트 비교로 테스트하면 우연히 일치해 flaky해
    지므로 시드 유도 자체를 본다.
    """
    assert _chunk_seed(3, 0) != _chunk_seed(4, 0)
    assert _chunk_seed(3, 0) != _chunk_seed(3, 1)


def test_max_errors_stops_at_a_chunk_boundary():
    """조각을 인덱스 순서대로 소비하며 멈추므로 멈추는 지점이 결정적이다.

    완료 순서대로 멈추면 총 shot이 실행마다 달라진다 (sinter에서 26,403과 33,708로
    갈린 원인).
    """
    circuit = small_circuit()
    first = parallel_failure_counts(circuit, 2000, "bposd_0", seed=5,
                                    workers=4, chunk=50, max_errors=20,
                                    progress=False)
    second = parallel_failure_counts(circuit, 2000, "bposd_0", seed=5,
                                     workers=2, chunk=50, max_errors=20,
                                     progress=False)
    assert first == second
    assert first.shots % 50 == 0 or first.shots == 2000
    assert first.block_fails >= 20


def test_resuming_matches_running_from_the_start():
    """조각 k부터 이어간 결과가 처음부터 완주한 것과 같다.

    이어받기의 근거다. tasks의 조각 번호를 다시 매기면 이어받은 조각이 다른
    시드를 받아 같은 shot을 두 번 세게 되고, 여기서 갈린다.
    """
    circuit = small_circuit()
    whole = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                    workers=2, chunk=50, progress=False)
    head = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                   workers=2, chunk=50, progress=False,
                                   start_chunk=0, start_counts=(0, 0, 0))
    part = parallel_failure_counts(circuit, 100, "bposd_0", seed=3,
                                   workers=2, chunk=50, progress=False)
    rest = parallel_failure_counts(
        circuit, 200, "bposd_0", seed=3, workers=2, chunk=50, progress=False,
        start_chunk=2,
        start_counts=(part.shots, part.block_fails, part.logical_flips))
    assert head == whole
    assert rest == whole


def test_checkpoint_reports_every_chunk():
    """조각마다 (done, shots, fails, flips)를 보고한다 -- 상태 파일의 재료다.

    done이 건너뛰면 이어받기 지점이 틀어지고, 마지막 조각이 빠지면 상태 파일이
    항상 한 조각 뒤처진다.
    """
    circuit = small_circuit()
    seen = []
    result = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                     workers=2, chunk=50, progress=False,
                                     checkpoint=lambda *a: seen.append(a))
    assert [done for done, *_ in seen] == [1, 2, 3, 4]
    assert seen[-1][1:] == (result.shots, result.block_fails,
                            result.logical_flips)
