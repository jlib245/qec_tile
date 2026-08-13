"""Tanner graph의 짧은 cycle — 길이 4, 6, 8에서의 개수."""
from math import comb

import numpy as np
import pytest

from qec_tile.cycles import cycle_counts
from qec_tile.tile import paper_code


def test_no_cycles_in_a_forest():
    """check마다 qubit 하나면 공유가 없으니 cycle이 닫힐 수 없다."""
    assert cycle_counts(np.eye(3, dtype=np.uint8)) == {4: 0, 6: 0, 8: 0}


def test_k33_counts():
    """K(3,3). 모든 개수를 손으로 알 수 있다.

    4-cycle은 check 2개와 qubit 2개의 아무 조합이라 C(3,2)*C(3,2) = 9다. 6-cycle은
    양쪽을 셋 다 쓰는 K(3,3)의 Hamiltonian cycle이고 6개다. 8-cycle은 check 4개가
    필요한데 3개뿐이다.
    """
    assert cycle_counts(np.ones((3, 3), dtype=np.uint8)) == {4: 9, 6: 6, 8: 0}


def test_four_cycles_match_the_matrix_formula():
    """독립적인 계수: qubit g개를 공유하는 check 둘이 C(g, 2)개의 4-cycle을 닫는다.

    walk의 중복 계수를 잡는다 — 이 공식은 경로를 아예 보지 않는다.
    """
    HZ = paper_code("b3w6", 4, 4).HZ.astype(int)
    overlaps = HZ @ HZ.T
    expected = sum(comb(int(overlaps[i, j]), 2)
                   for i in range(overlaps.shape[0])
                   for j in range(i + 1, overlaps.shape[0]))
    assert cycle_counts(HZ, max_length=4) == {4: expected}


def test_a_longer_walk_leaves_the_short_counts_alone():
    """max_length를 올리면 항목이 늘 뿐, 아래쪽 값이 바뀌면 안 된다."""
    HZ = paper_code("b3w6", 4, 4).HZ
    short = cycle_counts(HZ, max_length=6)
    long = cycle_counts(HZ, max_length=8)
    assert all(long[length] == count for length, count in short.items())


@pytest.mark.parametrize("max_length", [0, 2, 3, 5, 7])
def test_max_length_is_validated(max_length):
    """이분 그래프에는 홀수 cycle이 없고 4보다 짧은 것도 없다."""
    with pytest.raises(ValueError, match="max_length"):
        cycle_counts(np.ones((3, 3), dtype=np.uint8), max_length=max_length)
