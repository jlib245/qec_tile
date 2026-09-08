"""tile code의 circuit-level memory-Z 실험. stim으로 짓는다.

한 라운드는 X stabilizer 전부, 이어서 Z stabilizer 전부를 각자의 ancilla로
측정한다. CNOT 스케줄은 병진 대칭성을 이용한다: tile 오프셋의 전역 순서를 하나
고정하고, 시간 슬롯 ``o``에서 모든 check가 ``anchor + o``의 qubit을 건드린다.
``anchor + o``가 anchor를 결정하므로 같은 타입의 check 둘이 한 슬롯에서 충돌하는
일은 없다. 잘려나간 경계 check는 잃어버린 슬롯을 그냥 건너뛴다. X 층과 Z 층은
겹치지 않는 슬롯에 두어(depth ~ 2w), 깊이를 내주고 스케줄링의 단순함을 얻는다.

``memory_z_base``는 moment 사이에 TICK을 넣어 잡음 없는 회로를 뱉는다. 잡음은
``noise_model.NoiseModel``(Gidney 구현을 vendored)을 통한 후처리 단계이고, gate
잡음과 moment별 idle 잡음을 주입한다. ``memory_z_circuit``은 단순한 균일 ``p``
모델을 wrapper로 남겨둔 것이고, ``NoiseModel.SI1000(p)``이 초전도 기반의 표준
모델을 준다.

detector: data가 |0>에서 시작하므로 Z-check 결과는 첫 라운드에서 결정적이고 그
뒤로는 라운드끼리 비교한다. X-check 결과는 두 번째 라운드부터만 쓴다. 마지막의
transversal Z readout이 각 Z check를 한 번 더 재구성한다. k개 observable은 그 최종
readout에 LZ 행을 적용한 것이다.
"""
from __future__ import annotations

import numpy as np
import stim
from ldpc.ckt_noise import detector_error_model_to_check_matrices

from .decode import DECODERS, FailureCounts
from .noise_model import NoiseModel


def _schedule(H, anchors, qubits) -> tuple[list, list[dict]]:
    """전역 시간 슬롯(tile 오프셋)과, check마다 오프셋 -> data 열."""
    per_check: list[dict] = []
    slots = set()
    for row, (anchor_x, anchor_y) in zip(np.asarray(H), anchors):
        offsets = {}
        for col in np.flatnonzero(row):
            orient, x, y = qubits[col]
            offset = (orient, x - anchor_x, y - anchor_y)
            offsets[offset] = int(col)
            slots.add(offset)
        per_check.append(offsets)
    return sorted(slots), per_check


def memory_z_base(code, rounds: int) -> stim.Circuit:
    """``rounds``번의 syndrome 라운드에 대한 잡음 없는 memory-Z 회로.

    moment를 TICK으로 갈라놓아 ``noise_model.NoiseModel``이 후처리로 gate 잡음과,
    무엇보다 moment별 idle 잡음을 주입할 수 있게 한다.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    n = code.n
    mx, mz = code.HX.shape[0], code.HZ.shape[0]
    data = list(range(n))
    x_anc = [n + i for i in range(mx)]
    z_anc = [n + mx + j for j in range(mz)]
    x_slots, x_checks = _schedule(code.HX, code.x_anchors, code.qubits)
    z_slots, z_checks = _schedule(code.HZ, code.z_anchors, code.qubits)
    _, LZ = code.logicals()

    circuit = stim.Circuit()
    # 좌표를 붙여두면 stim의 timeslice 그림이 실제 격자를 그린다: data qubit은 edge
    # 중점에, ancilla는 자기 anchor의 box 안에 (0.25/0.75 오프셋이 bulk의 X와 Z
    # ancilla를 떼어놓는다).
    for col, (orient, x, y) in enumerate(code.qubits):
        xy = (x + 0.5, y) if orient == "H" else (x, y + 0.5)
        circuit.append("QUBIT_COORDS", [col], xy)
    for i, (anchor_x, anchor_y) in enumerate(code.x_anchors):
        circuit.append("QUBIT_COORDS", [x_anc[i]],
                       (anchor_x + 0.25, anchor_y + 0.25))
    for j, (anchor_x, anchor_y) in enumerate(code.z_anchors):
        circuit.append("QUBIT_COORDS", [z_anc[j]],
                       (anchor_x + 0.75, anchor_y + 0.75))

    circuit.append("R", data + z_anc)          # |0> data와 Z ancilla
    circuit.append("RX", x_anc)                # |+> X ancilla
    circuit.append("TICK")

    for round_index in range(rounds):
        # X 층: ancilla가 control이다 (자기 support에서 X를 측정).
        for slot in x_slots:
            pairs = [q for i, offsets in enumerate(x_checks)
                     if slot in offsets for q in (x_anc[i], offsets[slot])]
            circuit.append("CX", pairs)
            circuit.append("TICK")
        # Z 층: data가 control이다.
        for slot in z_slots:
            pairs = [q for j, offsets in enumerate(z_checks)
                     if slot in offsets for q in (offsets[slot], z_anc[j])]
            circuit.append("CX", pairs)
            circuit.append("TICK")

        # ancilla를 측정하고 reset한다 (mx개 결과, 그다음 mz개).
        circuit.append("MRX", x_anc)
        circuit.append("MR", z_anc)

        stride = mx + mz                       # 라운드당 측정 수
        for j in range(mz):
            current = -(mz - j)
            if round_index == 0:               # |0>에 대해 결정적
                circuit.append("DETECTOR", [stim.target_rec(current)])
            else:
                circuit.append("DETECTOR", [stim.target_rec(current),
                                            stim.target_rec(current - stride)])
        if round_index > 0:
            for i in range(mx):
                current = -(mz + mx - i)
                circuit.append("DETECTOR", [stim.target_rec(current),
                                            stim.target_rec(current - stride)])
        circuit.append("TICK")

    # data의 마지막 transversal Z readout이 각 Z check를 재구성한다.
    circuit.append("M", data)
    for j in range(mz):
        targets = [stim.target_rec(-(n - col))
                   for col in np.flatnonzero(code.HZ[j])]
        targets.append(stim.target_rec(-(n + mz - j)))
        circuit.append("DETECTOR", targets)
    for l, logical in enumerate(LZ):
        targets = [stim.target_rec(-(n - col))
                   for col in np.flatnonzero(logical)]
        circuit.append("OBSERVABLE_INCLUDE", targets, l)
    return circuit


def memory_z_circuit(code, rounds: int, p: float) -> stim.Circuit:
    """균일 잡음 아래의 memory-Z 회로: 모든 gate, 측정, reset이 확률 ``p``로
    실패하고 idle 잡음은 없다.

    "circuit" 벤치마크 축을 위해 남겨둔다. 표준 대안은
    ``NoiseModel.SI1000(p).noisy_circuit(memory_z_base(code, rounds))``다.
    """
    uniform = NoiseModel(
        idle=0.0,
        measure_reset_idle=0.0,
        noisy_gates={"CX": p, "R": p, "RX": p, "M": p, "MR": p, "MRX": p},
    )
    return uniform.noisy_circuit(memory_z_base(code, rounds))


def circuit_failure_counts(circuit: stim.Circuit, shots: int, decoder: str,
                           seed: int | None = None) -> FailureCounts:
    """회로를 샘플링해 detection event를 디코딩하고 두 종류의 실패를 집계한다.

    디코더는 detector error model의 행렬 위에서 일하지만, event는 stim이 실제 회로를
    샘플링해서 나온다 — DEM은 디코더의 지도로 쓰이고 잡음원이 아니다. 디코딩된
    오류에서 예측한 observable 뒤집힘과 샘플된 것을 비교해, 하나라도 다르면 블록
    실패이고 다른 개수만큼 ``logical_flips``에 더한다 (참 오류를 몰라도 되는,
    residual 판정과 동등한 검사). 후자는 observable basis(``OBSERVABLE_INCLUDE``에
    넣은 논리 연산자)에 의존한다.
    """
    dem = circuit.detector_error_model()
    matrices = detector_error_model_to_check_matrices(
        dem, allow_undecomposed_hyperedges=True)   # BP+OSD는 hyperedge를 받는다
    k = circuit.num_observables
    if matrices.check_matrix.shape[1] == 0:        # 잡음 없음: 실패할 것이 없다
        return FailureCounts(shots, 0, 0, k)

    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    decoder_obj = build(matrices.check_matrix, matrices.priors)

    sampler = circuit.compile_detector_sampler(seed=seed)
    detections, observed = sampler.sample(shots, separate_observables=True)

    A = matrices.observables_matrix.toarray().astype(np.uint8)
    block_fails = 0
    logical_flips = 0
    for detection, actual in zip(detections, observed):
        x_hat = decoder_obj.decode(detection.astype(np.uint8))
        mismatch = ((A @ x_hat) % 2) != actual
        block_fails += bool(mismatch.any())
        logical_flips += int(mismatch.sum())
    return FailureCounts(shots, block_fails, logical_flips, k)


def circuit_failure_rate(circuit: stim.Circuit, shots: int, decoder: str,
                         seed: int | None = None) -> float:
    """블록 실패율 -- ``circuit_failure_counts``의 ``block_fails / shots``."""
    counts = circuit_failure_counts(circuit, shots, decoder, seed)
    return counts.block_fails / shots
