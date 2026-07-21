"""Linear algebra over GF(2)."""
import numpy as np

from qec_tile.gf2 import nullspace2, quotient_basis, rank2, rref2


def test_rank2_small_by_hand():
    M = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]], dtype=np.uint8)
    assert rank2(M) == 2                       # row 3 = row 1 XOR row 2
    assert np.linalg.matrix_rank(M.astype(float)) == 3   # but rank 3 over R


def test_rank2_edge_cases():
    assert rank2(np.zeros((4, 5), dtype=np.uint8)) == 0
    assert rank2(np.eye(5, dtype=np.uint8)) == 5


def test_rref2_is_reduced():
    rng = np.random.default_rng(0)
    M = (rng.random((12, 20)) < 0.4).astype(np.uint8)
    reduced, pivot_cols = rref2(M)
    assert reduced.shape[0] == len(pivot_cols) == rank2(M)
    for i, col in enumerate(pivot_cols):
        assert reduced[i, col] == 1 and reduced[:, col].sum() == 1


def test_nullspace2():
    rng = np.random.default_rng(1)
    M = (rng.random((10, 25)) < 0.4).astype(np.uint8)
    basis = nullspace2(M)
    assert not ((basis @ M.T) % 2).any()
    assert basis.shape[0] == M.shape[1] - rank2(M)
    assert rank2(basis) == basis.shape[0]


def test_quotient_basis_counts_and_independence():
    rng = np.random.default_rng(2)
    subspace = (rng.random((5, 30)) < 0.3).astype(np.uint8)
    candidates = np.vstack([subspace,
                            (rng.random((7, 30)) < 0.3).astype(np.uint8)])
    extra = quotient_basis(subspace, candidates)
    assert rank2(np.vstack([subspace, extra])) == rank2(subspace) + extra.shape[0]
    assert (rank2(np.vstack([subspace, candidates]))
            == rank2(subspace) + extra.shape[0])
