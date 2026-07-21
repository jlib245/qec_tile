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


def rank2(M: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2)."""
    A = np.ascontiguousarray(M, dtype=np.uint8).copy()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        nz = np.flatnonzero(A[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        hit = np.flatnonzero(A[:, c])
        hit = hit[hit != r]
        if hit.size:
            A[hit] ^= A[r]
        r += 1
        if r == rows:
            break
    return r


def rref2(M: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2), with the pivot columns."""
    A = np.ascontiguousarray(M, dtype=np.uint8).copy()
    rows, cols = A.shape
    piv, r = [], 0
    for c in range(cols):
        nz = np.flatnonzero(A[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            A[[r, p]] = A[[p, r]]
        hit = np.flatnonzero(A[:, c])
        hit = hit[hit != r]
        if hit.size:
            A[hit] ^= A[r]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return A[:r], piv


def nullspace2(M: np.ndarray) -> np.ndarray:
    """Basis of {v : M v = 0} over GF(2), one vector per row."""
    n = M.shape[1]
    R, piv = rref2(M)
    free = [c for c in range(n) if c not in set(piv)]
    out = np.zeros((len(free), n), dtype=np.uint8)
    for i, f in enumerate(free):
        out[i, f] = 1
        out[i, piv] = R[:, f]
    return out


def quotient_basis(sub: np.ndarray, whole: np.ndarray) -> np.ndarray:
    """Rows of ``whole`` that are independent modulo the row space of ``sub``.

    One incremental elimination rather than a rank recomputation per row.
    """
    pivots: dict[int, np.ndarray] = {}

    def reduce(v):
        v = v.copy()
        for c in sorted(pivots):
            if v[c]:
                v ^= pivots[c]
        return v

    for row in sub:
        v = reduce(row)
        nz = np.flatnonzero(v)
        if nz.size:
            pivots[int(nz[0])] = v

    keep = []
    for row in whole:
        v = reduce(row)
        nz = np.flatnonzero(v)
        if nz.size:
            pivots[int(nz[0])] = v
            keep.append(row)
    return (np.array(keep, dtype=np.uint8) if keep
            else np.zeros((0, whole.shape[1]), dtype=np.uint8))
