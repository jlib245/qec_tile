"""Phenomenological 잡음 — 시공간 디코딩 행렬 (X sector).

syndrome 측정을 ``T``번 반복하고, 측정된 각 syndrome 비트는 ``meas_error``
(문헌의 ``q``) 확률로 뒤집힌다. 디코딩은 syndrome의 차분(detector)
``D_t = sigma_t xor sigma_{t-1}`` 위에서 이뤄진다:

    D_t = HZ e_t  xor  u_{t-1} xor u_t

그래서 data 오류는 detector를 한 라운드, 측정 오류는 두 라운드 건드린다
(DKLP의 spacelike/timelike edge, quant-ph/0110143). 라운드로 쌓으면 하나의
Kronecker 곱 check 행렬이 된다

    H = [ I_T (x) HZ  |  L_T (x) I_m ]      L_T = 시간 방향 bidiagonal
    L = [ 1_T (x) LZ  |  0 ]                (observable은 data만 본다)

관례: 마지막 라운드는 완벽하다고 둔다(``u_T = 0``) — data qubit을 직접 읽는
최종 readout을 대신하며, 그래서 측정 오류 열은 T-1개다. ``rounds=1``이면 전부
code-capacity 짝 (HZ, LZ)으로 줄어든다.
"""
from __future__ import annotations

import numpy as np


def spacetime_matrices(code, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """X sector를 ``rounds`` 라운드 측정할 때의 ``(H, L)``."""
    m, n = code.HZ.shape
    _, LZ = code.logicals()

    data = np.kron(np.eye(rounds, dtype=np.uint8), code.HZ)

    time_pairs = np.zeros((rounds, rounds - 1), dtype=np.uint8)
    for t in range(rounds - 1):                 # u_t는 detector t와 t+1을 건드린다
        time_pairs[t, t] = 1
        time_pairs[t + 1, t] = 1
    measurement = np.kron(time_pairs, np.eye(m, dtype=np.uint8))

    H = np.hstack([data, measurement])
    L = np.hstack([np.tile(LZ, rounds),
                   np.zeros((LZ.shape[0], (rounds - 1) * m), dtype=np.uint8)])
    return H, L


def spacetime_channel(code, rounds: int, p: float,
                      meas_error: float) -> np.ndarray:
    """열별 prior: data 열에는 ``p``, 측정 열에는 ``meas_error``(문헌의 ``q``)."""
    m, n = code.HZ.shape
    return np.concatenate([np.full(rounds * n, p),
                           np.full((rounds - 1) * m, meas_error)])
