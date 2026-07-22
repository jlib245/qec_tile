"""Tile code assembly — anchor placement, truncation and pruning."""
import numpy as np
import pytest

from qec_tile.gf2 import rank2
from qec_tile.tile import TABLE2, build_tile_code, paper_code

B3W6 = ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)])
B4W8 = ([(0, 0), (0, 3), (2, 2), (3, 0)], [(0, 1), (1, 0), (1, 1), (3, 3)])

# dy only spans {0, 1}, so the box is not fully used and qubits go uncovered.
FLAT = ([(0, 0), (1, 1)], [(0, 1), (2, 0)])


def test_qubits_are_unique_and_counted_by_the_formula():
    c = build_tile_code(*B3W6, 3, 4, 5)
    assert c.n == len(c.qubits) == 2 * (4 + 3 - 1) * (5 + 3 - 1)
    assert len(set(c.qubits)) == c.n
    assert all(orient in "HV" for orient, _, _ in c.qubits)


@pytest.mark.parametrize("B,x_h,x_v,L1,L2",
                         [(3, *B3W6, 4, 4), (3, *B3W6, 5, 3), (4, *B4W8, 5, 3)])
def test_qubits_are_the_union_of_bulk_boxes(B, x_h, x_v, L1, L2):
    """A rectangular layout makes that union the plain range the code uses."""
    c = build_tile_code(x_h, x_v, B, L1, L2)
    union = {(orient, i + dx, j + dy)
             for orient in "HV"
             for i in range(L1) for j in range(L2)
             for dx in range(B) for dy in range(B)}
    assert set(c.qubits) == union


def test_stabilizers_commute():
    c = build_tile_code(*B3W6, 3, 5, 5)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_checks_are_independent():
    """Unlike toric/BB codes, tile codes have no check dependencies at all."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    assert rank2(c.HX) == c.HX.shape[0] == 140      # 100 bulk + 40 boundary
    assert rank2(c.HZ) == c.HZ.shape[0] == 140


@pytest.mark.parametrize("B,L1,L2", [(3, 5, 5), (3, 6, 9), (4, 5, 5), (4, 7, 6)])
def test_k_is_2g_squared_regardless_of_layout(B, L1, L2):
    x_h, x_v = B3W6 if B == 3 else B4W8
    c = build_tile_code(x_h, x_v, B, L1, L2)
    assert c.n == 2 * (L1 + B - 1) * (L2 + B - 1)
    assert c.k == 2 * (B - 1) ** 2


def test_bulk_checks_are_untruncated_and_uniform():
    """Every bulk anchor carries a full-weight tile — the stencil premise."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    weights = [w for anchor, w in zip(c.x_anchors, c.HX.sum(1))
               if 0 <= anchor[0] < 10 and 0 <= anchor[1] < 10]
    assert len(weights) == 100 and set(weights) == {6}


def test_boundary_checks_are_truncated():
    c = build_tile_code(*B3W6, 3, 10, 10)
    weights = [w for anchor, w in zip(c.x_anchors, c.HX.sum(1))
               if not (0 <= anchor[0] < 10 and 0 <= anchor[1] < 10)]
    assert len(weights) == 40
    assert max(weights) <= 6 and min(weights) < 6


def test_empty_checks_are_dropped():
    """A tile landing entirely off the lattice yields no check at all."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    assert c.HX.shape[0] == len(c.x_anchors)
    assert c.HZ.shape[0] == len(c.z_anchors)
    assert c.HX.any(axis=1).all()
    assert c.HZ.any(axis=1).all()


@pytest.mark.parametrize("B,x_h,x_v", [(3, *B3W6), (4, *B4W8)])
def test_pruning_is_a_noop_for_paper_tiles(B, x_h, x_v):
    c = build_tile_code(x_h, x_v, B, 6, 6)
    assert c.n == 2 * (6 + B - 1) ** 2


def test_uncovered_qubits_are_removed():
    """A qubit with no X (or no Z) check leaves no syndrome to decode from."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert c.n < 2 * (6 + 2) ** 2
    assert (c.HX.sum(0) > 0).all()
    assert (c.HZ.sum(0) > 0).all()


def test_one_pass_pruning_is_already_a_fixpoint():
    """An emptied check held no surviving qubit, so it uncovers none."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX.sum(0) == 0) | (c.HZ.sum(0) == 0)).any()
    assert c.HX.any(axis=1).all() and c.HZ.any(axis=1).all()
    assert c.HX.shape[0] == len(c.x_anchors)
    assert c.HZ.shape[0] == len(c.z_anchors)


def test_pruning_preserves_commutation():
    """A dropped qubit was absent from one type entirely, so overlaps hold."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_offsets_outside_the_box_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        build_tile_code([(0, 0), (3, 1)], [(0, 2)], 3, 6, 6)


# --- the paper's tables ---------------------------------------------------

# (tile name, layout, n, k) — Table 1.  Rows 3 and 4 share the b4w8 tile and
# differ only in layout.  The distances (12/14/13/19) are not checked: computing
# them is NP-hard.
TABLE1 = [
    ("b3w6", 10, 288, 8),     # [[288,8,12]]
    ("b3w8", 10, 288, 8),     # [[288,8,14]]
    ("b4w8", 9, 288, 18),     # [[288,18,13]]
    ("b4w8", 13, 512, 18),    # [[512,18,19]]
    ("b4w10", 13, 512, 18),   # appendix, randomized search
]


@pytest.mark.parametrize("name,L,n,k", TABLE1)
def test_table1(name, L, n, k):
    c = paper_code(name, L, L)
    assert (c.n, c.k) == (n, k)


@pytest.mark.parametrize("x_h,x_v", TABLE2)
def test_table2_all_give_288_8(x_h, x_v):
    c = build_tile_code(x_h, x_v, 3, 10, 10)
    assert (c.n, c.k) == (288, 8)
    assert c.HX.sum(1).max() == 6 and c.HZ.sum(1).max() == 6
