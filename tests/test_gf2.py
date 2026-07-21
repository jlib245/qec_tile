"""GF(2) 선형대수 — mod 2 가우스 소거."""
import numpy as np

from qec_tile.gf2 import nullspace2, quotient_basis, rank2, rref2


def test_rank2_small_by_hand():
    # 3행이지만 r3 = r1 XOR r2 이므로 GF(2) 랭크는 2
    M = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]], dtype=np.uint8)
    assert rank2(M) == 2
    # 같은 행렬이 실수 위에서는 랭크 3 — 1+1=0 이라서 생기는 차이
    assert np.linalg.matrix_rank(M.astype(float)) == 3


def test_rank2_edge_cases():
    assert rank2(np.zeros((4, 5), dtype=np.uint8)) == 0
    assert rank2(np.eye(5, dtype=np.uint8)) == 5


def test_rref2_is_reduced():
    rng = np.random.default_rng(0)
    M = (rng.random((12, 20)) < 0.4).astype(np.uint8)
    R, piv = rref2(M)
    assert R.shape[0] == len(piv) == rank2(M)
    for i, c in enumerate(piv):
        col = R[:, c]
        assert col[i] == 1 and col.sum() == 1      # 각 피벗 열은 단위 벡터


def test_nullspace2():
    rng = np.random.default_rng(1)
    M = (rng.random((10, 25)) < 0.4).astype(np.uint8)
    K = nullspace2(M)
    assert not ((K @ M.T) % 2).any()               # 실제로 커널에 있고
    assert K.shape[0] == M.shape[1] - rank2(M)     # 차원 정리를 만족하고
    assert rank2(K) == K.shape[0]                  # 서로 독립이다


def test_quotient_basis_counts_and_independence():
    rng = np.random.default_rng(2)
    sub = (rng.random((5, 30)) < 0.3).astype(np.uint8)
    whole = np.vstack([sub, (rng.random((7, 30)) < 0.3).astype(np.uint8)])
    q = quotient_basis(sub, whole)
    assert rank2(np.vstack([sub, q])) == rank2(sub) + q.shape[0]
    assert rank2(np.vstack([sub, whole])) == rank2(sub) + q.shape[0]
