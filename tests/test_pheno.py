"""Phenomenological 잡음 — 시공간 check 행렬과 디코딩."""
import numpy as np

from qec_tile.decode import block_failure_rate
from qec_tile.pheno import spacetime_channel, spacetime_matrices
from qec_tile.tile import paper_code

SMALL = ("b3w6", 4, 4)


def test_shapes():
    code = paper_code(*SMALL)
    T = 3
    m, n = code.HZ.shape
    H, L = spacetime_matrices(code, rounds=T)
    assert H.shape == (T * m, T * n + (T - 1) * m)
    assert L.shape == (code.k, T * n + (T - 1) * m)


def test_one_round_reduces_to_code_capacity():
    """(완벽한) 라운드 하나면 새로울 게 없다: H가 곧 HZ다."""
    code = paper_code(*SMALL)
    H, L = spacetime_matrices(code, rounds=1)
    _, LZ = code.logicals()
    assert (H == code.HZ).all()
    assert (L == LZ).all()


def test_data_columns_are_block_diagonal():
    """라운드 t의 data 오류는 라운드 t의 detector만 건드린다."""
    code = paper_code(*SMALL)
    T = 3
    m, n = code.HZ.shape
    H, _ = spacetime_matrices(code, rounds=T)
    for t in range(T):
        block = H[:, t * n:(t + 1) * n]
        assert (block[t * m:(t + 1) * m] == code.HZ).all()
        rest = np.delete(block, np.s_[t * m:(t + 1) * m], axis=0)
        assert not rest.any()


def test_measurement_columns_are_time_dominoes():
    """u_t는 같은 check의 detector를 라운드 t와 t+1에서 뒤집는다 — weight 2."""
    code = paper_code(*SMALL)
    T = 4
    m, n = code.HZ.shape
    H, _ = spacetime_matrices(code, rounds=T)
    meas = H[:, T * n:]
    assert (meas.sum(0) == 2).all()
    for s in range(T - 1):
        for i in range(m):
            col = meas[:, s * m + i]
            assert col[s * m + i] == 1 and col[(s + 1) * m + i] == 1


def test_logical_ignores_measurement_errors():
    """측정 오류는 observable을 절대 뒤집지 않는다. data 오류만 뒤집는다."""
    code = paper_code(*SMALL)
    T = 3
    m, n = code.HZ.shape
    _, L = spacetime_matrices(code, rounds=T)
    _, LZ = code.logicals()
    for t in range(T):
        assert (L[:, t * n:(t + 1) * n] == LZ).all()
    assert not L[:, T * n:].any()


def test_channel_layout():
    code = paper_code(*SMALL)
    T = 3
    m, n = code.HZ.shape
    channel = spacetime_channel(code, rounds=T, p=0.01, meas_error=0.02)
    assert channel.shape == (T * n + (T - 1) * m,)
    assert (channel[:T * n] == 0.01).all()
    assert (channel[T * n:] == 0.02).all()


def test_zero_noise_never_fails():
    code = paper_code(*SMALL)
    H, L = spacetime_matrices(code, rounds=3)
    channel = spacetime_channel(code, rounds=3, p=0.0, meas_error=0.0)
    assert block_failure_rate(H, L, channel, shots=50,
                              decoder="bposd_cs7", seed=0) == 0.0


def test_seed_is_deterministic():
    code = paper_code(*SMALL)
    H, L = spacetime_matrices(code, rounds=3)
    channel = spacetime_channel(code, rounds=3, p=0.03, meas_error=0.03)
    a = block_failure_rate(H, L, channel, shots=100,
                           decoder="bposd_cs7", seed=5)
    b = block_failure_rate(H, L, channel, shots=100,
                           decoder="bposd_cs7", seed=5)
    assert a == b


def test_measurement_noise_hurts():
    """같은 p에서 측정 잡음을 더하는 것이 도움이 될 수는 없다."""
    code = paper_code(*SMALL)
    H, L = spacetime_matrices(code, rounds=4)
    quiet = spacetime_channel(code, rounds=4, p=0.04, meas_error=0.0)
    noisy = spacetime_channel(code, rounds=4, p=0.04, meas_error=0.08)
    r_quiet = block_failure_rate(H, L, quiet, shots=600,
                                 decoder="bposd_cs7", seed=1)
    r_noisy = block_failure_rate(H, L, noisy, shots=600,
                                 decoder="bposd_cs7", seed=1)
    assert r_quiet <= r_noisy
