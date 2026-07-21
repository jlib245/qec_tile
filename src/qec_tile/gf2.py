"""Linear algebra over GF(2) — the field where addition is XOR.

Every routine here takes and returns 0/1 ``uint8`` arrays, and row reduction
is plain ``row ^= pivot_row``: over GF(2) the only nonzero scalar is 1, so
elimination needs no division and no scaling.

CSS codes turn coding questions into these: ``k`` is a rank deficiency,
"which errors pass every check" is a nullspace, and "which of those are
merely stabilizers" is a quotient.
"""
from __future__ import annotations

import numpy as np


def rank2(matrix: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2)."""
    work = np.ascontiguousarray(matrix, dtype=np.uint8).copy()
    n_rows, n_cols = work.shape
    rank = 0                                # pivots so far, and the next row
    for col in range(n_cols):
        below = np.flatnonzero(work[rank:, col])
        if below.size == 0:                 # nothing to pivot on in this column
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
    """Reduced row echelon form over GF(2), with the pivot columns."""
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
    """Basis of {v : matrix v = 0} over GF(2), one vector per row.

    One basis vector per free column: set that column to 1, and the pivot
    columns are then forced to whatever that column holds in the RREF.  Over
    GF(2) there is no sign to flip, so the RREF entries are copied as they are.
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
    """Rows of ``candidates`` independent modulo the row space of ``subspace``.

    One incremental elimination rather than a rank recomputation per row: the
    subspace is eliminated into ``pivots`` first, then every candidate is
    reduced against it.  A candidate reducing to zero is already spanned; one
    that does not is kept *and* registered, so the candidates that follow are
    tested against the accepted ones too.
    """
    pivots: dict[int, np.ndarray] = {}      # pivot column -> its reduced row

    def reduce(row):
        """``row`` minus everything the registered pivots can account for."""
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
        if nonzero.size:                     # not spanned by what came before
            pivots[int(nonzero[0])] = residual   # register the reduced form,
            kept.append(row)                     # but return the original row
    return (np.array(kept, dtype=np.uint8) if kept
            else np.zeros((0, candidates.shape[1]), dtype=np.uint8))
