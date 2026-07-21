"""타일 코드 조립 — 앵커 배치와 절단(truncation)."""
import numpy as np
import pytest

from qec_tile.gf2 import rank2
from qec_tile.tile import build_tile_code

# 논문 타일 두 개. 이름표(TILES 딕셔너리)는 Step 4에서 붙인다.
B3W6 = ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)])
B4W8 = ([(0, 0), (0, 3), (2, 2), (3, 0)], [(0, 1), (1, 0), (1, 1), (3, 3)])

# dy가 {0,1}뿐이라 박스 세로를 다 안 쓰는 타일 — 미커버 큐빗이 생긴다.
FLAT = ([(0, 0), (1, 1)], [(0, 1), (2, 0)])


def test_qubits_are_unique_and_counted_by_the_formula():
    """n = 2(L1+g)(L2+g) — 큐빗은 black 박스들의 합집합."""
    c = build_tile_code(*B3W6, 3, 4, 5)
    g = 3 - 1
    assert c.n == len(c.qubits) == 2 * (4 + g) * (5 + g)
    assert len(set(c.qubits)) == c.n            # 중복 없음
    assert all(o in "HV" for o, _, _ in c.qubits)


def test_stabilizers_commute():
    """(T2)로 만든 Z-타일이므로 HX·HZ^T는 mod 2로 0."""
    c = build_tile_code(*B3W6, 3, 5, 5)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_checks_are_independent():
    """토릭/BB 코드와 달리 타일 코드는 검사 간 종속성이 전혀 없다."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    assert rank2(c.HX) == c.HX.shape[0] == 140      # bulk 100 + 경계 40
    assert rank2(c.HZ) == c.HZ.shape[0] == 140


@pytest.mark.parametrize("B,L1,L2", [(3, 5, 5), (3, 6, 9), (4, 5, 5), (4, 7, 6)])
def test_k_is_2g_squared_regardless_of_layout(B, L1, L2):
    """k = 2(B-1)^2. 레이아웃을 키워도 논리 큐빗 수는 그대로다."""
    x_h, x_v = B3W6 if B == 3 else B4W8
    c = build_tile_code(x_h, x_v, B, L1, L2)
    g = B - 1
    assert c.n == 2 * (L1 + g) * (L2 + g)
    assert c.k == 2 * g * g


def test_bulk_checks_are_untruncated_and_uniform():
    """bulk 앵커는 전부 온전한 타일을 얹는다 — 벌크가 균일하다는 뜻."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    bulk = [w for a, w in zip(c.x_anchors, c.HX.sum(1))
            if 0 <= a[0] < 10 and 0 <= a[1] < 10]
    assert len(bulk) == 100 and set(bulk) == {6}


def test_boundary_checks_are_truncated():
    """경계 앵커는 격자 밖으로 삐져나가 잘린다 — 경계가 생기는 지점."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    edge = [w for a, w in zip(c.x_anchors, c.HX.sum(1))
            if not (0 <= a[0] < 10 and 0 <= a[1] < 10)]
    assert len(edge) == 40
    assert max(edge) <= 6 and min(edge) < 6      # 최소 하나는 잘려 있다


def test_empty_checks_are_dropped():
    """타일이 통째로 격자 밖이면 그 검사는 아예 버린다 (논문 step 4)."""
    c = build_tile_code(*B3W6, 3, 10, 10)
    assert c.HX.shape[0] == len(c.x_anchors)     # 행과 앵커가 일대일
    assert c.HZ.shape[0] == len(c.z_anchors)
    assert c.HX.any(axis=1).all()                # 0인 행이 없다
    assert c.HZ.any(axis=1).all()


@pytest.mark.parametrize("B,x_h,x_v", [(3, *B3W6), (4, *B4W8)])
def test_pruning_is_a_noop_for_paper_tiles(B, x_h, x_v):
    """논문 타일은 모든 큐빗이 X도 Z도 받으므로 지울 게 없다."""
    c = build_tile_code(x_h, x_v, B, 6, 6)
    assert c.n == 2 * (6 + B - 1) ** 2


def test_uncovered_qubits_are_removed():
    """논문 마지막 단계: X 또는 Z를 하나도 못 받는 큐빗은 제거한다.

    그런 큐빗은 오류가 나도 신드롬을 남기지 않아 디코더가 손댈 수 없다.
    """
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert c.n < 2 * (6 + 2) ** 2                  # 실제로 줄었고
    assert (c.HX.sum(0) > 0).all()                 # 남은 큐빗은 전부
    assert (c.HZ.sum(0) > 0).all()                 # X도 Z도 받는다


def test_one_pass_pruning_is_already_a_fixpoint():
    """큐빗 제거 -> 검사 제거를 반복할 필요가 없다.

    비게 된 검사는 살아남은 큐빗을 하나도 안 건드리므로, 그 검사를 지워도
    남은 큐빗의 커버 수는 줄지 않는다. 그래서 1패스가 곧 고정점이다.
    """
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX.sum(0) == 0) | (c.HZ.sum(0) == 0)).any()
    assert c.HX.any(axis=1).all() and c.HZ.any(axis=1).all()
    assert c.HX.shape[0] == len(c.x_anchors)
    assert c.HZ.shape[0] == len(c.z_anchors)


def test_pruning_preserves_commutation():
    """제거되는 큐빗은 한쪽 타입의 검사에 아예 안 들어 있으므로,
    열을 지워도 X-Z 겹침 수는 변하지 않는다."""
    c = build_tile_code(*FLAT, 3, 6, 6)
    assert not ((c.HX @ c.HZ.T) % 2).any()


def test_offsets_outside_the_box_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        build_tile_code([(0, 0), (3, 1)], [(0, 2)], 3, 6, 6)
