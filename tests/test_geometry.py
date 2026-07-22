"""Tile geometry — the symmetry condition (T2) buys."""
import numpy as np
import pytest

from qec_tile.tile import z_tile_from_x

# (B, X_H, X_V) — the paper's b3w6 and b4w8 tiles
TILES = [
    (3, [(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    (4, [(0, 0), (0, 3), (2, 2), (3, 0)], [(0, 1), (1, 0), (1, 1), (3, 3)]),
]


def as_edges(horizontal, vertical):
    return ({("H", x, y) for x, y in horizontal}
            | {("V", x, y) for x, y in vertical})


def overlap(a: set, b: set, dx: int, dy: int) -> int:
    """How many qubits ``a`` shares with ``b`` shifted by (dx, dy)."""
    return len(a & {(orient, x + dx, y + dy) for (orient, x, y) in b})


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_is_an_involution(B, x_h, x_v):
    z_h, z_v = z_tile_from_x(x_h, x_v, B)
    back_h, back_v = z_tile_from_x(z_h, z_v, B)
    assert set(back_h) == set(x_h) and set(back_v) == set(x_v)


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_stays_inside_the_box(B, x_h, x_v):
    z_h, z_v = z_tile_from_x(x_h, x_v, B)
    assert all(0 <= x < B and 0 <= y < B for x, y in z_h + z_v)


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_preserves_weight(B, x_h, x_v):
    z_h, z_v = z_tile_from_x(x_h, x_v, B)
    assert (len(z_h), len(z_v)) == (len(x_v), len(x_h))


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_every_relative_overlap_is_even(B, x_h, x_v):
    """Why (T2) exists: even overlap is what makes X and Z commute."""
    X = as_edges(x_h, x_v)
    Z = as_edges(*z_tile_from_x(x_h, x_v, B))
    for dx in range(-B, B + 1):
        for dy in range(-B, B + 1):
            assert overlap(X, Z, dx, dy) % 2 == 0, f"shift ({dx},{dy})"


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_h_and_v_overlaps_are_equal(B, x_h, x_v):
    """The mechanism: the H and V overlaps match, so the total is twice one."""
    z_h, z_v = z_tile_from_x(x_h, x_v, B)
    XH, XV = as_edges(x_h, []), as_edges([], x_v)
    ZH, ZV = as_edges(z_h, []), as_edges([], z_v)
    for dx in range(-B, B + 1):
        for dy in range(-B, B + 1):
            assert overlap(XH, ZH, dx, dy) == overlap(XV, ZV, dx, dy)


def test_even_overlap_holds_for_random_tiles():
    """Not luck of the two tiles above — (T2) alone forces it."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        B = int(rng.integers(2, 6))
        cells = [(int(x), int(y)) for x in range(B) for y in range(B)]
        pick = lambda: [cells[i] for i in
                        rng.choice(len(cells), size=int(rng.integers(1, B * B)),
                                   replace=False)]
        x_h, x_v = pick(), pick()
        X = as_edges(x_h, x_v)
        Z = as_edges(*z_tile_from_x(x_h, x_v, B))
        for dx in range(-B, B + 1):
            for dy in range(-B, B + 1):
                assert overlap(X, Z, dx, dy) % 2 == 0
