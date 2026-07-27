"""Directional tile codes — tile codes whose tile is traced by a compass walk.

Construction follows Gu, Noszko, Steffan, Eberhardt, Roffe, Eisert &
Koutsioumpas, "Nearest-neighbour gates are all you need: High-rate quantum
low-density parity-check codes on a planar grid" (arXiv:2606.19482).

Definition 1: "A pair of XX- and ZZ-tiles is called directional if the tiles
satisfy the mutual condition, one tile forms an ordered connected string
labelled by a directional word D = d_1 d_2 ... d_w, and the other tile
contains the same string on the dual lattice.  In addition, every displacement
vector of the ordered connected string with odd vertical displacement must
occur with even multiplicity."

The mutual condition -- "for each horizontal, respectively vertical, edge of
the XX-tile with coordinate (a,b), the ZZ-tile contains the vertical,
respectively horizontal, edge with coordinate (B-1-a, B-1-b)" -- is (T2) of
the original tile-code paper, so ``tile.z_tile_from_x`` already implements it
and a directional code is a tile code with a restricted X-tile.

The walk
--------
The string is a string of *edges*, so one letter is one edge and hence one
data qubit: a word of weight w gives a weight-w stabilizer.  The check qubit
walks the vertices of the code lattice and eats the edge it crosses::

    (x, y) --N--> V(x,   y  ), arriving at (x,   y+1)
    (x, y) --S--> V(x,   y-1), arriving at (x,   y-1)
    (x, y) --E--> H(x,   y  ), arriving at (x+1, y  )
    (x, y) --W--> H(x-1, y  ), arriving at (x-1, y  )

On hardware the square grid is this lattice at twice the scale, so every
integer site ``(X, Y)`` is one of four things::

    (even, even)  vertex       -- XX-check anchor
    (odd,  even)  H((X-1)/2, Y/2)   data qubit
    (even, odd )  V(X/2, (Y-1)/2)   data qubit
    (odd,  odd )  face centre  -- ZZ-check anchor on the dual lattice

Consecutive edges of the string are two hardware steps apart with a vertex or
face centre in between; that in-between site is the routing qubit the check
SWAPs through between CXSWAPs.  XX-checks start on vertices and ZZ-checks on
face centres, so the two walks never collide.
"""
from __future__ import annotations

import itertools
from collections import Counter

from .tile import TileCode, build_tile_code

# Compass steps on the hardware grid: N is +y, E is +x.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

Edge = tuple[str, int, int]


def parse_directional_word(word: str) -> list[tuple[int, int]]:
    """Expand a directional word into its unit steps.

    The paper writes words with superscripts (``N^2 E^2 S E S E^2 N^2``); here
    the repeat count follows the letter as plain digits, read greedily, so the
    constant reads like the printed word: ``"N2E2SESE2N2"``.
    """
    steps: list[tuple[int, int]] = []
    position = 0
    compact = "".join(word.split())
    while position < len(compact):
        letter = compact[position]
        if letter not in DIRECTIONS:
            raise ValueError(
                f"{word!r}: expected one of {sorted(DIRECTIONS)} at index "
                f"{position}, got {letter!r}")
        position += 1
        start = position
        while position < len(compact) and compact[position].isdigit():
            position += 1
        repeat = int(compact[start:position]) if position > start else 1
        if repeat < 1:
            raise ValueError(f"{word!r}: repeat count must be >= 1")
        steps.extend([DIRECTIONS[letter]] * repeat)
    if not steps:
        raise ValueError("directional word is empty")
    return steps


def walk_edges(steps: list[tuple[int, int]]) -> list[Edge]:
    """Walk the lattice vertices and name the edge each step crosses.

    One letter is one edge, so the result has one data qubit per step and the
    walk order is the order the syndrome extraction visits them.

    Coordinates are relative to the anchor and may be negative; normalising
    them into a ``B x B`` box is the caller's job.  A step in the negative
    direction crosses the edge *behind* the vertex -- going ``W`` from
    ``(0, 0)`` crosses ``H(-1, 0)``, not ``H(0, 0)``.
    """
    edges: list[Edge] = []
    x, y = 0, 0
    for (step_x, step_y) in steps:
        if step_x:
            edges.append(("H", x if step_x > 0 else x - 1, y))
        else:
            edges.append(("V", x, y if step_y > 0 else y - 1))
        x, y = x + step_x, y + step_y
    return edges


def hardware_site(edge: Edge) -> tuple[int, int]:
    """The edge's midpoint on the hardware grid, which is the lattice doubled.

    Horizontal edges land on even hardware y and vertical ones on odd, so a
    displacement has odd vertical component exactly when it joins an H edge to
    a V edge.
    """
    orient, x, y = edge
    return (2 * x + 1, 2 * y) if orient == "H" else (2 * x, 2 * y + 1)


def displacement_vectors(edges: list[Edge]) -> Counter[tuple[int, int]]:
    """How often each vector between two edges of the string occurs.

    Every ordered pair ``i < j`` of the walk contributes ``site_j - site_i``,
    which is what Figure 5 draws "for each starting point".
    """
    sites = [hardware_site(edge) for edge in edges]
    return Counter((later_x - earlier_x, later_y - earlier_y)
                   for (earlier_x, earlier_y), (later_x, later_y)
                   in itertools.combinations(sites, 2))


def satisfies_parity_condition(edges: list[Edge]) -> bool:
    """Definition 1: "every displacement vector of the ordered connected string
    with odd vertical displacement must occur an even number of times".

    The mutual condition alone makes the stabilizers commute; this is the extra
    condition that makes the syndrome-extraction circuit deterministic, so it
    constrains the walk as a schedule rather than the tile as a set.
    """
    return all(count % 2 == 0
               for vector, count in displacement_vectors(edges).items()
               if vector[1] % 2)


def tile_from_word(word: str,
                   B: int | None = None) -> tuple[list[tuple[int, int]],
                                                  list[tuple[int, int]], int]:
    """The X-tile ``(x_h, x_v, B)`` traced by ``word``, ready for the assembler.

    Offsets are translated so the tile hugs the origin, as ``build_tile_code``
    requires.  ``B`` defaults to the smallest square box holding the walk; the
    paper fixes the layout as an ``(M+B-1) x (N+B-1)`` grid but never states
    which ``B`` its own codes use, so it stays overridable.
    """
    edges = walk_edges(parse_directional_word(word))
    if len(set(edges)) != len(edges):
        raise ValueError(f"{word!r}: the walk crosses an edge twice, so it is "
                         f"not an ordered connected string")

    min_x = min(x for _, x, _ in edges)
    min_y = min(y for _, _, y in edges)
    span = max(max(x for _, x, _ in edges) - min_x,
               max(y for _, _, y in edges) - min_y) + 1
    if B is None:
        B = span
    elif B < span:
        raise ValueError(f"{word!r} spans {span}, does not fit a {B}x{B} box")

    x_h = sorted((x - min_x, y - min_y) for orient, x, y in edges
                 if orient == "H")
    x_v = sorted((x - min_x, y - min_y) for orient, x, y in edges
                 if orient == "V")
    return x_h, x_v, B


def build_directional_code(word: str, M: int, N: int,
                           B: int | None = None) -> TileCode:
    """The directional tile code for ``word`` on the paper's M x N anchor grid.

    Assembly is the original tile-code assembly: the paper tessellates "on an
    (M+B-1) x (N+B-1) rectangular grid" with "the vertices of an M x N subgrid
    as anchors", which is what ``build_tile_code`` already lays out, and the
    mutual condition is (T2), which ``z_tile_from_x`` already imposes.  What
    is new in a directional code is how the tile is drawn, not how it is
    stamped.
    """
    x_h, x_v, B = tile_from_word(word, B)
    return build_tile_code(x_h, x_v, B, M, N)


# Table 2 of arXiv:2606.19482, as (word, M, N, n, k, d).  The paper prints the
# word and [[n,k,d]] but neither B nor the M x N anchor grid, so both were
# recovered by search.  (n,k) alone did not pin the layout down -- three rows
# had a second layout with the same (n,k); the distance separated them, and the
# rejected ones were far off (d = 3, 2, 6 against 7, 15, 11).  B is the minimal
# box holding the walk, which costs nothing: the code does not depend on it.
PAPER_CODES: list[tuple[str, int, int, int, int, int]] = [
    ("N2ESEN2",       4, 4,  60,  4,  5),
    ("N2ESEN2",       8, 8, 180,  4,  9),
    ("N2E2SE2N2",    11, 6, 217, 10,  7),
    ("N2E2SE2N2",    15, 8, 351, 10,  9),
    ("N2E2SESE2N2",  12, 4, 182, 14, 10),
    ("N2E2SESE2N2",  17, 6, 323, 14, 15),
    ("N2E2SE3SE2N2", 16, 4, 248, 20, 11),
]
