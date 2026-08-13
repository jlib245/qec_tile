"""Tanner graph의 짧은 cycle.

BP는 트리에서 정확하므로, 성능을 깎는 것은 그래프에 실제로 있는 cycle이다 —
메시지가 자기를 보낸 쪽과 상관된 채로 돌아온다. girth 하나로는 요약이 약하고
디코더가 체감하는 것은 짧은 cycle이 *몇 개*냐이므로, 길이별로 센다.

그래프는 이분(한쪽은 check, 다른쪽은 qubit)이라 모든 cycle은 길이가 짝수이고
가장 짧은 것은 4다: 두 check가 두 qubit을 공유하는 경우. 길이 ``2k``의 cycle은
check ``k``개와 qubit ``k``개를 번갈아 지나며, 모두 서로 다르다.
"""
from __future__ import annotations

import numpy as np


def cycle_counts(H: np.ndarray, max_length: int = 8) -> dict[int, int]:
    """Tanner graph에서 ``max_length``까지 각 짝수 길이의 cycle 수.

    ``H``는 parity check 행렬 — 행이 check, 열이 qubit. ``{4: n4, 6: n6, ...}``를
    돌려준다. girth는 count가 0이 아닌 첫 길이다.

    각 cycle은 가장 작은 인덱스의 check에서만 걷는다. 이러면 시작점이 고정되고,
    한 바퀴 도는 두 방향은 마지막에 나눠 없앤다.
    """
    if max_length < 4 or max_length % 2:
        raise ValueError("max_length must be an even number >= 4")

    H = np.ascontiguousarray(H, dtype=np.uint8)
    qubits_of_check = [np.flatnonzero(row).tolist() for row in H]
    checks_of_qubit = [np.flatnonzero(column).tolist() for column in H.T]

    counts = {length: 0 for length in range(4, max_length + 1, 2)}
    max_checks = max_length // 2
    used_checks: set[int] = set()
    used_qubits: set[int] = set()

    def walk(start_check: int, current_check: int) -> None:
        for qubit in qubits_of_check[current_check]:
            if qubit in used_qubits:        # 방금 타고 온 qubit도 걸린다
                continue
            used_qubits.add(qubit)
            for next_check in checks_of_qubit[qubit]:
                if next_check == start_check:
                    if len(used_checks) >= 2:   # 1이면 check-qubit-check일 뿐
                        counts[2 * len(used_checks)] += 1
                elif (next_check > start_check
                        and next_check not in used_checks
                        and len(used_checks) < max_checks):
                    used_checks.add(next_check)
                    walk(start_check, next_check)
                    used_checks.discard(next_check)
            used_qubits.discard(qubit)

    for start_check in range(H.shape[0]):
        used_checks.add(start_check)
        walk(start_check, start_check)
        used_checks.discard(start_check)

    return {length: count // 2 for length, count in counts.items()}


def girth(H: np.ndarray, max_length: int = 8) -> int | None:
    """가장 짧은 cycle의 길이, 그만큼 짧은 것이 없으면 ``None``."""
    for length, count in sorted(cycle_counts(H, max_length).items()):
        if count:
            return length
    return None
