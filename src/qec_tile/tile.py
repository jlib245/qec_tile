"""Tile codes — CSS codes that are O(1)-local on a planar lattice with boundary.

Construction follows Steffan, Choe, Breuckmann, Fernandes Pereira & Eberhardt,
"Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary"
(arXiv:2504.09171).

Geometry
--------
Qubits sit on edges of the square lattice.  Vertices are at integer ``(x, y)``::

    H(x, y):  (x, y) -- (x+1, y)      horizontal edge
    V(x, y):  (x, y) -- (x, y+1)      vertical edge

A ``B x B`` box is B x B *cells*.  Its usable edges are those not on the top
row or the rightmost column, i.e. ``H(x, y)`` and ``V(x, y)`` for
``x, y in [0, B)`` — ``2*B**2`` candidates per box.

A tile is a subset of those. Condition (T2) fixes the Z-tile from the X-tile
(180-degree rotation plus H<->V swap), which makes every relative overlap even::

    Z_V = {(B-1-x, B-1-y) for (x, y) in X_H}
    Z_H = {(B-1-x, B-1-y) for (x, y) in X_V}

Layout
------
Anchors sit at their box's lower-left corner; with bulk block ``L1 x L2`` and
``g = B - 1``::

    bulk       (X and Z):  i in [0, L1),              j in [0, L2)
    x_boundary (X only):   i in [0, L1),              j in [-g, 0) u [L2, L2+g)
    z_boundary (Z only):   i in [-g, 0) u [L1, L1+g), j in [0, L2)

The paper's figures color these black, red and blue respectively.

Qubits are the union of the boxes over bulk anchors only, so bulk tiles are
never truncated while the boundary tiles hang off the lattice and get cut.
What an x_boundary tile loses is always out of range in ``y`` and what a
z_boundary tile loses is out of range in ``x``, so a cut qubit never sits in a
tile of the opposite type — the (T2) overlap parity survives truncation.

Hence ``n = 2*(L1+g)*(L2+g)``, and every check is independent so ``k = 2*g**2``
whatever the layout size.

The paper closes with a pruning pass: drop every qubit that no X-stabilizer or
no Z-stabilizer acts on, then drop the stabilizers left empty by that.  For the
paper's own tiles nothing is dropped; it matters for tiles that do not span the
whole box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import rank2

Edge = tuple[str, int, int]


def dual_tile(x_h, x_v, B: int) -> tuple[list, list]:
    """(T2): the Z-tile determined by the X-tile."""
    z_v = [(B - 1 - x, B - 1 - y) for (x, y) in x_h]
    z_h = [(B - 1 - x, B - 1 - y) for (x, y) in x_v]
    return z_h, z_v


@dataclass
class TileCode:
    HX: np.ndarray                 # (mx, n) uint8
    HZ: np.ndarray                 # (mz, n) uint8
    qubits: list[Edge]             # column index -> ('H'|'V', x, y)
    x_anchors: list[tuple[int, int]]   # row index of HX -> anchor
    z_anchors: list[tuple[int, int]]
    B: int
    L1: int
    L2: int

    @property
    def n(self) -> int:
        return len(self.qubits)

    @property
    def k(self) -> int:
        return self.n - rank2(self.HX) - rank2(self.HZ)

    def __str__(self) -> str:
        return (f"TileCode(B={self.B}, layout={self.L1}x{self.L2}, "
                f"n={self.n}, k={self.k}, "
                f"w={int(self.HX.sum(1).max())}/{int(self.HZ.sum(1).max())})")


def build_tile_code(x_h, x_v, B: int, L1: int, L2: int) -> TileCode:
    """Build the tile code with X-tile ``x_h`` (horizontal) + ``x_v`` (vertical).

    Coordinates are box-relative offsets in ``[0, B)^2``.
    """
    for (x, y) in list(x_h) + list(x_v):
        if not (0 <= x < B and 0 <= y < B):
            raise ValueError(f"offset ({x},{y}) outside the {B}x{B} box")

    z_h, z_v = dual_tile(x_h, x_v, B)
    x_tile = [("H", x, y) for x, y in x_h] + [("V", x, y) for x, y in x_v]
    z_tile = [("H", x, y) for x, y in z_h] + [("V", x, y) for x, y in z_v]

    g = B - 1
    qubits = sorted((o, x, y)
                    for o in "HV"
                    for x in range(L1 + g)
                    for y in range(L2 + g))
    idx = {q: i for i, q in enumerate(qubits)}

    bulk = [(i, j) for i in range(L1) for j in range(L2)]
    x_boundary = [(i, j) for i in range(L1)
                  for j in [*range(-g, 0), *range(L2, L2 + g)]]
    z_boundary = [(i, j) for j in range(L2)
                  for i in [*range(-g, 0), *range(L1, L1 + g)]]

    def assemble(tile, anchors):
        rows, kept = [], []
        for (ax, ay) in anchors:
            r = np.zeros(len(qubits), dtype=np.uint8)
            for (o, dx, dy) in tile:
                q = (o, ax + dx, ay + dy)
                if q in idx:                 # truncate to available qubits
                    r[idx[q]] ^= 1
            if r.any():                      # a tile entirely off the lattice
                rows.append(r)
                kept.append((ax, ay))
        return (np.array(rows, dtype=np.uint8) if rows
                else np.zeros((0, len(qubits)), dtype=np.uint8)), kept

    HX, xa = assemble(x_tile, bulk + x_boundary)
    HZ, za = assemble(z_tile, bulk + z_boundary)

    # Paper's final pass: a qubit no X-check (or no Z-check) touches leaves no
    # syndrome, so it is dropped; then the checks that are now empty go too.
    # One pass suffices — an emptied check held none of the surviving qubits,
    # so removing it cannot uncover any of them.
    live = (HX.sum(0) > 0) & (HZ.sum(0) > 0)
    if not live.all():
        qubits = [q for q, keep in zip(qubits, live) if keep]
        HX, HZ = HX[:, live], HZ[:, live]
        HX, xa = drop_empty_rows(HX, xa)
        HZ, za = drop_empty_rows(HZ, za)

    if ((HX @ HZ.T) % 2).any():
        raise ValueError("stabilizers do not commute — X-tile violates (T2)")
    return TileCode(HX, HZ, qubits, xa, za, B, L1, L2)


def drop_empty_rows(H: np.ndarray, anchors: list) -> tuple[np.ndarray, list]:
    """Rows of ``H`` with any support, and the matching anchors."""
    keep = H.any(axis=1)
    return H[keep], [a for a, k in zip(anchors, keep) if k]
