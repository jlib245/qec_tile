"""tile code 조립 — anchor 배치, 절단, pruning."""
import numpy as np
import pytest

from qec_tile.gf2 import rank2
from qec_tile.tile import TABLE2, build_tile_code, paper_code

B3W6 = ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)])
B4W8 = ([(0, 0), (0, 3), (2, 2), (3, 0)], [(0, 1), (1, 0), (1, 1), (3, 3)])

# dy가 {0, 1}만 걸치므로 box를 다 쓰지 않고 덮이지 않는 qubit이 생긴다.
FLAT = ([(0, 0), (1, 1)], [(0, 1), (2, 0)])


def test_qubits_are_unique_and_counted_by_the_formula():
    c = build_tile_code(*B3W6, 3, 4, 5)
    assert c.n == len(c.qubits) == 2 * (4 + 3 - 1) * (5 + 3 - 1)
    assert len(set(c.qubits)) == c.n
    assert all(orient in "HV" for orient, _, _ in c.qubits)


@pytest.mark.parametrize("B,x_h,x_v,L1,L2",
                         [(3, *B3W6, 4, 4), (3, *B3W6, 5, 3), (4, *B4W8, 5, 3)])
def test_qubits_are_the_union_of_bulk_boxes(B, x_h, x_v, L1, L2):
    """직사각 배치에서는 그 합집합이 코드가 쓰는 평범한 range가 된다."""
    c = build_tile_code(x_h, x_v, B, L1, L2)
    union = {(orient, i + dx, j + dy)
             for orient in "HV"
             for i in range(L1) for j in range(L2)
             for dx in range(B) for dy in range(B)}
    assert set(c.qubits) == union


def test_anchors_are_enumerated_by_one_rule():
    """(i, j) 순서로 한 번만 훑는다. bulk 먼저, 경계 나중으로 쪼개지 않는다.

    qubit 열이 x-major라서 j를 한 칸 옮기면 check의 support가 한 열, i를 한 칸
    옮기면 격자 열 하나만큼 밀린다. j를 가장 안쪽에 두고 anchor를 열거하는 것이
    연속한 check를 행렬 위에서 건너뛰지 않고 겹치게 만든다. 훑기를 bulk 패스와 경계
    패스로 쪼개면 이음매에서 깨졌다 -- b3w6 L=5에서 HZ의 경계 행이 bulk의 +1, +1, +1에
    비해 +0, +37, +7, -43으로 뛰었다.
    """
    code = build_tile_code(*B3W6, 3, 5, 5)
    assert code.x_anchors == sorted(code.x_anchors)
    assert code.z_anchors == sorted(code.z_anchors)


def test_anchor_sets_are_the_paper_rectangles():
    """X anchor는 j로, Z anchor는 i로 격자 밖으로 B-1씩 나간다."""
    B, L1, L2 = 3, 5, 4
    code = build_tile_code(*B3W6, B, L1, L2)
    assert set(code.x_anchors) == {(i, j) for i in range(L1)
                                   for j in range(-(B - 1), L2 + B - 1)}
    assert set(code.z_anchors) == {(i, j) for i in range(-(B - 1), L1 + B - 1)
                                   for j in range(L2)}


def test_stabilizers_commute():
    c = build_tile_code(*B3W6, 3, 5, 5)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_checks_are_independent():
    """toric/BB 코드와 달리 tile code에는 check 의존성이 전혀 없다."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    assert rank2(c.HX) == c.HX.shape[0] == 140      # bulk 100 + 경계 40
    assert rank2(c.HZ) == c.HZ.shape[0] == 140


@pytest.mark.parametrize("B,L1,L2", [(3, 5, 5), (3, 6, 9), (4, 5, 5), (4, 7, 6)])
def test_k_is_2g_squared_regardless_of_layout(B, L1, L2):
    x_h, x_v = B3W6 if B == 3 else B4W8
    c = build_tile_code(x_h, x_v, B, L1, L2)
    assert c.n == 2 * (L1 + B - 1) * (L2 + B - 1)
    assert c.k == 2 * (B - 1) ** 2


def test_bulk_checks_are_untruncated_and_uniform():
    """모든 bulk anchor가 full-weight tile을 진다 — stencil의 전제."""
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
    """tile이 통째로 격자 밖에 떨어지면 check가 아예 생기지 않는다."""
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
    """X(또는 Z) check가 없는 qubit은 디코딩할 syndrome을 남기지 않는다."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert c.n < 2 * (6 + 2) ** 2
    assert (c.HX.sum(0) > 0).all()
    assert (c.HZ.sum(0) > 0).all()


def test_one_pass_pruning_is_already_a_fixpoint():
    """비워진 check는 살아남은 qubit을 갖고 있지 않았으므로 아무것도 드러내지 않는다."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX.sum(0) == 0) | (c.HZ.sum(0) == 0)).any()
    assert c.HX.any(axis=1).all() and c.HZ.any(axis=1).all()
    assert c.HX.shape[0] == len(c.x_anchors)
    assert c.HZ.shape[0] == len(c.z_anchors)


def test_pruning_preserves_commutation():
    """버려진 qubit은 한 타입에서 아예 없었으므로 overlap은 유지된다."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_offsets_outside_the_box_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        build_tile_code([(0, 0), (3, 1)], [(0, 2)], 3, 6, 6)


# --- 논문의 표 --------------------------------------------------------------

# (tile 이름, 배치, n, k) — Table 1. 3행과 4행은 b4w8 tile을 공유하고 배치만
# 다르다. 거리(12/14/13/19)는 검사하지 않는다: 계산이 NP-hard다.
TABLE1 = [
    ("b3w6", 10, 288, 8),     # [[288,8,12]]
    ("b3w8", 10, 288, 8),     # [[288,8,14]]
    ("b4w8", 9, 288, 18),     # [[288,18,13]]
    ("b4w8", 13, 512, 18),    # [[512,18,19]]
    ("b4w10", 13, 512, 18),   # 부록, 무작위 탐색
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
