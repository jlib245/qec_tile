"""sinter를 통한 병렬 shot 수집.

``sinter.collect``를 감싸서, ``circuit.circuit_failure_rate``의 한 발씩 돌리는
루프 대신 여러 circuit을 worker 프로세스에 흩어 한꺼번에 디코딩한다. 두 경로
모두 같은 BP+OSD 설정을 쓴다.

worker 스케줄링은 비결정적이라 실행마다 count가 달라진다 — 재현 가능한(시드된)
숫자가 필요하면 serial인 ``circuit_failure_rate``를 쓴다.
"""
from __future__ import annotations

import sinter
import stim
from ldpc import SinterBpOsdDecoder
from ldpc.sinter_decoders import SinterLsdDecoder


def _osd(osd_method: str, osd_order: int):
    # decode.make_decoder와 맞춰둔다 — serial과 parallel 결과를 견줄 수 있게.
    return lambda: SinterBpOsdDecoder(
        bp_method="minimum_sum", max_iter=50, ms_scaling_factor=1.0,
        osd_method=osd_method, osd_order=osd_order)


# 인스턴스가 아니라 factory다: sinter는 디코더를 각 worker로 pickle하고,
# collect()마다 새 객체를 쓰면 실행 간에 상태가 섞이지 않는다. decode.DECODERS와
# 짝을 맞췄고 bplsd_cs7만 빠졌다 — SinterLsdDecoder에는 lsd_method 손잡이가
# 없어서 combination sweep LSD는 serial 전용으로 남는다.
SINTER_DECODERS = {
    "bposd_cs7": _osd("osd_cs", 7),
    "bposd_0": _osd("osd_0", 0),
    "bplsd_0": lambda: SinterLsdDecoder(          # Hillmann et al. 설정
        bp_method="minimum_sum", max_iter=30,
        ms_scaling_factor=0.625, lsd_order=0),
}


def collect(circuits: dict[object, stim.Circuit], decoder: str,
            max_shots: int, max_errors: int | None = None,
            workers: int = 8,
            progress: bool = True) -> dict[object, tuple[int, int]]:
    """모든 circuit을 병렬로 디코딩 -> ``{key: (shots, errors)}``.

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
            results[key] = (max_shots, 0)
    if not noisy:
        return results

    keys = list(noisy)
    tasks = [sinter.Task(circuit=noisy[key], json_metadata={"index": i})
             for i, key in enumerate(keys)]
    task_stats = sinter.collect(
        num_workers=workers,
        tasks=tasks,
        decoders=[decoder],
        custom_decoders={decoder: build()},
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=progress,     # stderr로 나가니 CSV/stdout은 깨끗하다
    )
    results.update({keys[stat.json_metadata["index"]]:
                    (stat.shots, stat.errors) for stat in task_stats})
    return results
