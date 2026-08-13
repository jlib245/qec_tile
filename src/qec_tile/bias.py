"""Bias-tailored frame — qubit 일부에 Hadamard를 걸어 잡음의 bias 방향을 재라벨링한다.

Bonilla Ataides, Tuckett, Bartlett, Flammia & Brown, "The XZZX surface code",
Nat. Commun. 12, 2172 (2021), arXiv:2009.07851 의 XZZX surface code는 CSS surface
code에 sublattice Hadamard를 건 것이다. Roffe, Cohen, Quintavalle, Chandra &
Campbell, "Bias-tailored quantum LDPC codes", Quantum 7, 1005 (2023),
arXiv:2202.01702 이 그것을 LDPC로 옮겼다: hypergraph product의 두 sector 중 한쪽에만
Hadamard를 걸면 biased noise 아래 성능이 오른다.

tile code에서 그 이분에 해당하는 것이 H edge qubit과 V edge qubit이다.

Hadamard를 건 qubit에서는 lab frame의 Z가 code frame의 X가 된다. depolarizing은 이
변환에 불변이므로, bias가 없으면 frame을 바꿔도 아무것도 달라지지 않는다. 이 모듈이
값을 내는 것은 잡음이 한쪽 Pauli로 기울었을 때다.
"""
from __future__ import annotations

import numpy as np


def edge_mask(code, orient: str = "V") -> np.ndarray:
    """``orient`` edge에 앉은 qubit 열의 boolean mask — Hadamard를 걸 자리.

    tile code의 qubit은 edge에 앉으므로 방향이 자연스러운 이분을 준다. 이것이
    bias-tailored hypergraph product의 두 sector에 대응한다.
    """
    if orient not in ("H", "V"):
        raise ValueError(f"orient must be 'H' or 'V', not {orient!r}")
    return np.array([edge_orient == orient
                     for edge_orient, _, _ in code.qubits], dtype=bool)


def pure_z_matrices(code, hadamard: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """lab frame의 순수 Z 잡음이 보는 ``(H_detect, L_pair)``.

    ``hadamard``가 True인 qubit에서는 lab의 Z가 code frame의 X가 되어 Z-check가
    본다. 나머지에서는 Z로 남아 X-check가 본다. 두 블록을 열만 지운 채 쌓으면
    ``n``열짜리 고전 문제 하나가 되어 ``distance.classical_distance_upper_bound``에
    그대로 들어간다.

    ``hadamard=None``이 지금의 CSS frame이고, 순수 X bias는 mask를 뒤집은 것이다 --
    ``pure_z_matrices(code, ~hadamard)``.
    """
    hadamard = (np.zeros(code.n, dtype=bool) if hadamard is None
                else np.asarray(hadamard, dtype=bool))
    if hadamard.shape != (code.n,):
        raise ValueError(f"hadamard mask has shape {hadamard.shape}, "
                         f"expected ({code.n},)")
    LX, LZ = code.logicals()
    stays_z, becomes_x = ~hadamard, hadamard
    H_detect = np.vstack([code.HX * stays_z, code.HZ * becomes_x])
    L_pair = np.vstack([LX * stays_z, LZ * becomes_x])
    return H_detect, L_pair
