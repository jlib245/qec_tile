"""sinter를 통한 병렬 shot 수집.

``sinter.collect``를 감싸서, ``circuit.circuit_failure_rate``의 한 발씩 돌리는
루프 대신 여러 circuit을 worker 프로세스에 흩어 한꺼번에 디코딩한다. 두 경로
모두 같은 BP+OSD 설정을 쓴다.

worker 스케줄링은 비결정적이라 실행마다 count가 달라진다 — 재현 가능한(시드된)
숫자가 필요하면 serial인 ``circuit_failure_rate``를 쓴다.
"""
from __future__ import annotations

import numpy as np
import sinter
import stim
from ldpc import SinterBpOsdDecoder
from ldpc.ckt_noise.dem_matrices import detector_error_model_to_check_matrices
from ldpc.sinter_decoders import SinterLsdDecoder

from .decode import FailureCounts, VibeCosetDecoder, VibeLsdDecoder


def _osd(osd_method: str, osd_order: int, max_iter: int = 50):
    # decode.make_decoder와 맞춰둔다 — serial과 parallel 결과를 견줄 수 있게.
    return lambda: SinterBpOsdDecoder(
        bp_method="minimum_sum", max_iter=max_iter, ms_scaling_factor=1.0,
        osd_method=osd_method, osd_order=osd_order)


class SinterVibeLsdDecoder(sinter.Decoder):
    """``VibeLsdDecoder``를 sinter worker에서 돌리는 래퍼 -- ldpc의 SinterLsdDecoder와
    같은 골격.

    DEM은 worker가 파일로 받으므로 행렬과 앙상블은 여기서 짓고, shot은 b8로 읽고 쓴다.
    """

    def __init__(self, *, coset: bool = False, **kwargs):
        self.coset = coset                         # coset 합산 변형을 쓸지
        self.kwargs = kwargs                       # VibeLsdDecoder 손잡이 그대로

    def decode_via_files(self, *, num_shots, num_dets, num_obs, dem_path,
                         dets_b8_in_path, obs_predictions_b8_out_path,
                         tmp_dir):
        dem = stim.DetectorErrorModel.from_file(dem_path)
        matrices = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True)
        observables = matrices.observables_matrix.toarray().astype(np.uint8)
        if self.coset:                             # 디코딩에도 observable을 쓴다
            decoder = VibeCosetDecoder(matrices.check_matrix.toarray(),
                                       matrices.priors, observables,
                                       **self.kwargs)
        else:
            decoder = VibeLsdDecoder(matrices.check_matrix.toarray(),
                                     matrices.priors, **self.kwargs)
        shots = stim.read_shot_data_file(path=dets_b8_in_path, format="b8",
                                         num_detectors=num_dets)
        predictions = np.zeros((num_shots, num_obs), dtype=bool)
        for i in range(num_shots):
            correction = decoder.decode(shots[i].astype(np.uint8))
            predictions[i] = (observables @ correction) % 2
        stim.write_shot_data_file(data=predictions,
                                  path=obs_predictions_b8_out_path,
                                  format="b8", num_observables=num_obs)


# 인스턴스가 아니라 factory다: sinter는 디코더를 각 worker로 pickle하고,
# collect()마다 새 객체를 쓰면 실행 간에 상태가 섞이지 않는다. decode.DECODERS와
# 짝을 맞췄고 bplsd_cs7만 빠졌다 — SinterLsdDecoder에는 lsd_method 손잡이가
# 없어서 combination sweep LSD는 serial 전용으로 남는다.
SINTER_DECODERS = {
    "bposd_cs7": lambda max_iter=50: _osd("osd_cs", 7, max_iter)(),
    "bposd_0": lambda max_iter=50: _osd("osd_0", 0, max_iter)(),
    "bplsd_0": lambda max_iter=30: SinterLsdDecoder(   # Hillmann et al. 설정
        bp_method="minimum_sum", max_iter=max_iter,
        ms_scaling_factor=0.625, lsd_order=0),
    "vibelsd_32": lambda max_iter=20: SinterVibeLsdDecoder(
        ensemble=32, max_iter=max_iter),
    "vibelsd_200": lambda max_iter=15: SinterVibeLsdDecoder(
        ensemble=200, max_iter=max_iter),
    # vibelsd_200과 ensemble/max_iter/converged가 같다 -- 최종 선택 규칙만 다른 대조군.
    "vibecoset_200": lambda max_iter=15: SinterVibeLsdDecoder(
        coset=True, ensemble=200, max_iter=max_iter),
}


def collect(circuits: dict[object, stim.Circuit], decoder: str,
            max_shots: int, max_errors: int | None = None,
            workers: int = 8, progress: bool = True,
            max_iter: int | None = None) -> dict[object, FailureCounts]:
    """모든 circuit을 병렬로 디코딩 -> ``{key: FailureCounts}``.

    sinter의 ``count_observable_error_combos``가 shot마다 뒤집힌 observable 조합을
    ``obs_mistake_mask=E_E_``(E = 뒤집힘) 키로 세어 주므로, 거기서 ``logical_flips``
    를 직접 합산한다 -- ``circuit_failure_counts``와 같은 집계다.

    task가 metadata에 인덱스를 지고 다니는 이유: sinter는 완료 순서로 stat을
    돌려주고, JSON은 tuple 키를 망친다.
    """
    build = SINTER_DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(SINTER_DECODERS)}")

    # DEM에 error mechanism이 없는 circuit은 절대 실패할 수 없는데, 디코더는
    # 빈 문제에서 멈춰버린다 — 그냥 자명한 답을 준다.
    results: dict = {}
    noisy: dict = {}
    for key, circuit in circuits.items():
        dem = circuit.detector_error_model()
        if any(inst.type == "error" for inst in dem.flattened()):
            noisy[key] = circuit
        else:
            results[key] = FailureCounts(max_shots, 0, 0,
                                         circuit.num_observables)
    if not noisy:
        return results

    keys = list(noisy)
    tasks = [sinter.Task(circuit=noisy[key], json_metadata={"index": i})
             for i, key in enumerate(keys)]
    task_stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=[decoder],
        # None이면 인자를 아예 안 넘긴다 -- 항목마다 다른 기본값이 살아야 한다
        custom_decoders={decoder: build() if max_iter is None
                         else build(max_iter)},
        max_shots=max_shots,
        max_errors=max_errors,
        count_observable_error_combos=True,
        print_progress=progress,     # stderr로 나가니 CSV/stdout은 깨끗하다
    )
    for stat in task_stats:
        key = keys[stat.json_metadata["index"]]
        flips = sum(count * name.split("=", 1)[1].count("E")
                    for name, count in stat.custom_counts.items()
                    if name.startswith("obs_mistake_mask="))
        results[key] = FailureCounts(stat.shots, stat.errors, flips,
                                     noisy[key].num_observables)
    return results
