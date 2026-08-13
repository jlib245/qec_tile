"""Bias-tailored frame — Hadamard를 걸 qubit 집합과, 순수 Z 잡음이 보는 행렬."""
import numpy as np
import pytest

from qec_tile.bias import edge_mask, pure_z_matrices
from qec_tile.directional import build_directional_code
from qec_tile.distance import (classical_distance_upper_bound,
                               distance_upper_bound)
from qec_tile.tile import paper_code


def test_mask_splits_a_tile_code_in_half():
    """pruning이 없는 tile code의 qubit은 방향별로 정확히 절반이다.

    qubit 집합이 {H,V} x [0, L1+B-1) x [0, L2+B-1) = 2 x 5 x 5 이고, 논문 자신의
    tile은 box를 다 채워 pruning이 아무것도 지우지 않는다.
    """
    code = paper_code("b3w6", 3, 3)
    assert code.n == 50
    assert edge_mask(code, "V").sum() == 25


def test_mask_selects_the_named_orientation():
    """절반 검사는 25/25라 mask가 뒤집혀도 통과한다. 그걸 잡는다."""
    code = paper_code("b3w6", 3, 3)
    mask = edge_mask(code, "V")
    assert all(code.qubits[col][0] == "V" for col in np.flatnonzero(mask))


def test_the_two_orientations_partition_the_qubits():
    """directional code는 pruning이 돌아 절반이 아닐 수 있다. 분할인 것은 항상이다."""
    code = build_directional_code("N2ESEN2", 4, 4)
    h, v = edge_mask(code, "H"), edge_mask(code, "V")
    assert (h | v).all()
    assert not (h & v).any()


def test_unknown_orientation_is_rejected():
    """Pauli 이름을 잘못 넣으면 전부 False인 mask가 나와 "지금 frame"과 구별이 안 된다."""
    with pytest.raises(ValueError, match="orient"):
        edge_mask(paper_code("b3w6", 3, 3), "X")


# --- 순수 Z 잡음이 보는 행렬 --------------------------------------------------

def test_no_hadamard_reduces_to_the_z_sector():
    """mask가 비면 H_detect = [HX; 0], L_pair = [LX; 0] — z sector 그 자체다.

    0으로 채워진 행은 rref2가 떨어뜨리고 L_pair의 0 행은 어떤 벡터와도 짝이 안
    되므로, 문제가 정확히 (HX, LX)로 줄어든다.
    """
    code = paper_code("b3w6", 3, 3)
    H_detect, L_pair = pure_z_matrices(code)
    assert (classical_distance_upper_bound(H_detect, L_pair,
                                           trials=200, seed=0)
            == distance_upper_bound(code, trials=200, seed=0, sector="z"))


def test_hadamard_everywhere_is_the_x_sector():
    """모든 qubit에 Hadamard를 걸면 lab의 Z가 전부 code frame의 X가 된다.

    HX/HZ를 반대로 짝지어도 mask가 빈 경우는 그대로 통과한다. mask의 양 끝에서
    서로 다른 sector가 나와야 방향이 고정된다.
    """
    code = paper_code("b3w6", 3, 3)
    H_detect, L_pair = pure_z_matrices(code, np.ones(code.n, dtype=bool))
    assert (classical_distance_upper_bound(H_detect, L_pair,
                                           trials=200, seed=0)
            == distance_upper_bound(code, trials=200, seed=0, sector="x"))


def test_the_tailored_frame_decouples_the_two_orientations():
    """어떤 행도 H qubit과 V qubit을 동시에 보지 않는다.

    무한 bias에서 코드가 두 덩어리로 쪼개진다는 주장. 지금 frame에서는 HX 행이 둘을
    동시에 보므로 tailored frame에만 있는 성질이다.
    """
    code = paper_code("b3w6", 3, 3)
    v = edge_mask(code, "V")
    H_detect, _ = pure_z_matrices(code, v)
    touches_h = H_detect[:, ~v].any(axis=1)
    touches_v = H_detect[:, v].any(axis=1)
    assert not (touches_h & touches_v).any()


def test_mask_shape_is_checked():
    """길이 1이면 numpy가 조용히 broadcast해서 전체 열에 적용해버린다."""
    with pytest.raises(ValueError, match="shape"):
        pure_z_matrices(paper_code("b3w6", 3, 3), np.ones(1, dtype=bool))
