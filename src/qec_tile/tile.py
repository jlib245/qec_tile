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

The anchor sets above are the paper's (unrotated) square layout.  The qubit
set is built as a literal union of boxes, so swapping in another bulk layout
only means changing those three lists.

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

from .gf2 import nullspace2, quotient_basis, rank2

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

    def logicals(self) -> tuple[np.ndarray, np.ndarray]:
        """``(LX, LZ)``, each ``(k, n)`` over GF(2).

        X-type logicals are ker(H_Z) modulo the X-stabilizers, and vice versa.
        An X-type residual error ``r`` (one that already matches the Z-check
        syndrome) is a logical failure iff ``LZ @ r != 0``.
        """
        LX = quotient_basis(self.HX, nullspace2(self.HZ))
        LZ = quotient_basis(self.HZ, nullspace2(self.HX))
        return LX, LZ

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
    bulk = [(i, j) for i in range(L1) for j in range(L2)]
    x_boundary = [(i, j) for i in range(L1)
                  for j in [*range(-g, 0), *range(L2, L2 + g)]]
    z_boundary = [(i, j) for j in range(L2)
                  for i in [*range(-g, 0), *range(L1, L1 + g)]]

    # Qubits are the union of the bulk anchors' B x B boxes.  For the square
    # layout that union is the rectangle [0, L1+g) x [0, L2+g), but taking it
    # literally keeps the rest of the construction layout-agnostic.
    qubits = sorted({(orient, i + dx, j + dy)
                     for orient in "HV"
                     for (i, j) in bulk
                     for dx in range(B)
                     for dy in range(B)})
    col_of = {qubit: col for col, qubit in enumerate(qubits)}

    def assemble(tile, anchors):
        """One check per anchor: stamp the tile, cut whatever misses a qubit."""
        checks, kept_anchors = [], []
        for (anchor_x, anchor_y) in anchors:
            check = np.zeros(len(qubits), dtype=np.uint8)
            for (orient, dx, dy) in tile:
                qubit = (orient, anchor_x + dx, anchor_y + dy)
                if qubit in col_of:          # truncate to available qubits
                    check[col_of[qubit]] ^= 1
            if check.any():                  # a tile entirely off the lattice
                checks.append(check)
                kept_anchors.append((anchor_x, anchor_y))
        return (np.array(checks, dtype=np.uint8) if checks
                else np.zeros((0, len(qubits)), dtype=np.uint8)), kept_anchors

    HX, x_anchors = assemble(x_tile, bulk + x_boundary)
    HZ, z_anchors = assemble(z_tile, bulk + z_boundary)

    # Paper's final pass: a qubit no X-check (or no Z-check) touches leaves no
    # syndrome, so it is dropped; then the checks that are now empty go too.
    # One pass suffices — an emptied check held none of the surviving qubits,
    # so removing it cannot uncover any of them.
    covered = (HX.sum(0) > 0) & (HZ.sum(0) > 0)
    if not covered.all():
        qubits = [qubit for qubit, is_covered in zip(qubits, covered)
                  if is_covered]
        HX, HZ = HX[:, covered], HZ[:, covered]
        HX, x_anchors = drop_empty_checks(HX, x_anchors)
        HZ, z_anchors = drop_empty_checks(HZ, z_anchors)

    if ((HX @ HZ.T) % 2).any():
        raise ValueError("stabilizers do not commute — X-tile violates (T2)")
    return TileCode(HX, HZ, qubits, x_anchors, z_anchors, B, L1, L2)


def drop_empty_checks(checks: np.ndarray,
                      anchors: list) -> tuple[np.ndarray, list]:
    """Rows of ``checks`` with any support left, and the matching anchors."""
    nonempty = checks.any(axis=1)
    return checks[nonempty], [anchor for anchor, is_kept
                              in zip(anchors, nonempty) if is_kept]


# Tiles from the paper, as (X_H, X_V).  Table 1 rows 3 and 4 share a tile and
# differ only in layout, so the name records the tile, not the code.
TILES: dict[str, tuple[list, list]] = {
    # Table 1
    "b3w6": ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    "b3w8": ([(0, 0), (0, 1), (0, 2), (2, 0)],
             [(0, 0), (0, 2), (1, 1), (2, 2)]),
    "b4w8": ([(0, 0), (0, 3), (2, 2), (3, 0)],
             [(0, 1), (1, 0), (1, 1), (3, 3)]),
    "b4w10": ([(0, 0), (1, 0), (2, 1), (2, 3), (3, 0)],
              [(0, 3), (1, 0), (3, 1), (3, 2), (3, 3)]),
}

# Table 2: all eight depicted weight-6 B=3 X-tiles giving [[288,8,12]] at 10x10.
# The paper's full count of 16 is these plus their X<->Z swaps.
TABLE2: list[tuple[list, list]] = [
    ([(0, 0), (0, 1), (2, 2)], [(0, 2), (1, 0), (2, 0)]),
    ([(0, 0), (0, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    ([(0, 0), (1, 0), (2, 2)], [(0, 1), (0, 2), (2, 0)]),
    ([(0, 0), (1, 0), (2, 2)], [(0, 2), (2, 0), (2, 1)]),
    ([(0, 0), (1, 2), (2, 2)], [(0, 1), (0, 2), (2, 0)]),
    ([(0, 0), (1, 2), (2, 2)], [(0, 2), (2, 0), (2, 1)]),
    ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 0), (2, 0)]),
    ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
]


def paper_code(name: str, L1: int, L2: int) -> TileCode:
    """Convenience: build one of the paper's tiles at a given layout size."""
    x_h, x_v = TILES[name]
    B = 3 if name.startswith("b3") else 4
    return build_tile_code(x_h, x_v, B, L1, L2)
