"""GF(2) 선형대수 — 덧셈이 XOR인 체.

여기의 모든 루틴은 0/1 ``uint8`` 배열을 받고 돌려준다. 행 소거는 그냥
``row ^= pivot_row``다 — GF(2)에서 0이 아닌 스칼라는 1뿐이라 나눗셈도
스케일링도 필요하지 않다.

CSS 코드는 부호 이론의 질문을 이 연산들로 바꿔놓는다. ``k``는 rank 결손이고,
"어떤 오류가 모든 check를 통과하는가"는 nullspace이며, "그중 어느 것이 단지
stabilizer일 뿐인가"는 몫공간이다.
"""
from __future__ import annotations

import numpy as np


def rank2(matrix: np.ndarray) -> int:
    """0/1 행렬의 GF(2) 위 rank."""
    work = np.ascontiguousarray(matrix, dtype=np.uint8).copy()
    n_rows, n_cols = work.shape
    rank = 0                                # 지금까지 잡은 pivot 수, 곧 다음 pivot 행
    for col in range(n_cols):
        below = np.flatnonzero(work[rank:, col])
        if below.size == 0:                 # 이 열에는 pivot으로 쓸 것이 없다
            continue
        pivot_row = rank + below[0]
        if pivot_row != rank:
            work[[rank, pivot_row]] = work[[pivot_row, rank]]
        to_clear = np.flatnonzero(work[:, col])
        to_clear = to_clear[to_clear != rank]
        if to_clear.size:
            work[to_clear] ^= work[rank]
        rank += 1
        if rank == n_rows:
            break
    return rank


def rref2(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """GF(2) 위 기약 행 사다리꼴(RREF), pivot 열과 함께."""
    work = np.ascontiguousarray(matrix, dtype=np.uint8).copy()
    n_rows, n_cols = work.shape
    pivot_cols: list[int] = []
    rank = 0
    for col in range(n_cols):
        below = np.flatnonzero(work[rank:, col])
        if below.size == 0:
            continue
        pivot_row = rank + below[0]
        if pivot_row != rank:
            work[[rank, pivot_row]] = work[[pivot_row, rank]]
        to_clear = np.flatnonzero(work[:, col])
        to_clear = to_clear[to_clear != rank]
        if to_clear.size:
            work[to_clear] ^= work[rank]
        pivot_cols.append(col)
        rank += 1
        if rank == n_rows:
            break
    return work[:rank], pivot_cols


def nullspace2(matrix: np.ndarray) -> np.ndarray:
    """{v : matrix v = 0}의 GF(2) 위 기저, 한 행이 한 벡터.

    free 열 하나가 기저 벡터 하나를 낳는다. 그 열을 1로 두면 pivot 열들은
    RREF에서 그 열이 갖는 값으로 강제된다. GF(2)에는 뒤집을 부호가 없으니
    RREF 성분을 그대로 베껴 쓴다.
    """
    n_cols = matrix.shape[1]
    rref, pivot_cols = rref2(matrix)
    free_cols = [col for col in range(n_cols) if col not in set(pivot_cols)]
    basis = np.zeros((len(free_cols), n_cols), dtype=np.uint8)
    for i, free_col in enumerate(free_cols):
        basis[i, free_col] = 1
        basis[i, pivot_cols] = rref[:, free_col]
    return basis


def quotient_basis(subspace: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """``subspace``의 행공간을 법으로 독립인 ``candidates``의 행들.

    행마다 rank를 다시 재지 않고 소거를 한 번만 한다. 먼저 subspace를
    ``pivots``에 소거해 넣고, 그다음 후보를 하나씩 그에 대해 줄인다. 0으로
    줄어든 후보는 이미 span 안에 있다. 그렇지 않은 후보는 남기고 *동시에*
    등록하므로, 뒤따르는 후보는 앞서 받아들여진 것들에 대해서도 걸러진다.
    """
    pivots: dict[int, np.ndarray] = {}      # pivot 열 -> 줄여진 그 행

    def reduce(row):
        """등록된 pivot들이 설명할 수 있는 만큼을 ``row``에서 뺀 나머지."""
        residual = row.copy()
        for pivot_col in sorted(pivots):
            if residual[pivot_col]:
                residual ^= pivots[pivot_col]
        return residual

    for row in subspace:
        residual = reduce(row)
        nonzero = np.flatnonzero(residual)
        if nonzero.size:
            pivots[int(nonzero[0])] = residual

    kept = []
    for row in candidates:
        residual = reduce(row)
        nonzero = np.flatnonzero(residual)
        if nonzero.size:                     # 앞선 것들이 span하지 못한다
            pivots[int(nonzero[0])] = residual   # 등록은 줄여진 형태로,
            kept.append(row)                     # 반환은 원래 행으로
    return (np.array(kept, dtype=np.uint8) if kept
            else np.zeros((0, candidates.shape[1]), dtype=np.uint8))
