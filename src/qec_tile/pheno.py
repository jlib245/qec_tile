"""Phenomenological noise — the space-time decoding matrices (X sector).

Syndrome measurement is repeated ``T`` times and each measured syndrome bit
flips with probability ``meas_error`` (the literature's ``q``).  Decoding
works on syndrome differences
(detectors) ``D_t = sigma_t xor sigma_{t-1}``:

    D_t = HZ e_t  xor  u_{t-1} xor u_t

so a data error hits one detector round and a measurement error hits two
(DKLP's spacelike and timelike edges, quant-ph/0110143).  Stacked over rounds
this is one Kronecker-product check matrix

    H = [ I_T (x) HZ  |  L_T (x) I_m ]      L_T = bidiagonal in time
    L = [ 1_T (x) LZ  |  0 ]                (the observable sees data only)

Convention: the last round is perfect (``u_T = 0``), standing in for a final
direct readout of the data qubits, so there are T-1 measurement-error columns.
With ``rounds=1`` everything reduces to the code-capacity pair (HZ, LZ).
"""
from __future__ import annotations

import numpy as np


def spacetime_matrices(code, rounds: int) -> tuple[np.ndarray, np.ndarray]:
    """``(H, L)`` for ``rounds`` measurement rounds of the X sector."""
    m, n = code.HZ.shape
    _, LZ = code.logicals()

    data = np.kron(np.eye(rounds, dtype=np.uint8), code.HZ)

    time_pairs = np.zeros((rounds, rounds - 1), dtype=np.uint8)
    for t in range(rounds - 1):                 # u_t hits detectors t and t+1
        time_pairs[t, t] = 1
        time_pairs[t + 1, t] = 1
    measurement = np.kron(time_pairs, np.eye(m, dtype=np.uint8))

    H = np.hstack([data, measurement])
    L = np.hstack([np.tile(LZ, rounds),
                   np.zeros((LZ.shape[0], (rounds - 1) * m), dtype=np.uint8)])
    return H, L


def spacetime_channel(code, rounds: int, p: float,
                      meas_error: float) -> np.ndarray:
    """Per-column priors: ``p`` on data columns, ``meas_error`` on measurement
    ones (the literature's ``q``)."""
    m, n = code.HZ.shape
    return np.concatenate([np.full(rounds * n, p),
                           np.full((rounds - 1) * m, meas_error)])
