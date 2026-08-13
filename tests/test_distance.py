"""부호 거리 — 정확한 열거와 무작위 상한."""
import numpy as np
import pytest

from qec_tile.distance import (classical_distance_upper_bound,
                               distance_bruteforce, distance_ilp,
                               distance_upper_bound)
from qec_tile.tile import paper_code

# C(50, 3) 열거가 즉시 끝날 만큼 작다. 거리는 3이다: weight 1이나 2인 오류는
# 검출 불가이면서 stabilizer가 아닌 경우가 없다.
SMALL = ("b3w6", 3, 3)


def test_bruteforce_finds_the_known_distance():
    assert distance_bruteforce(paper_code(*SMALL)) == 3


def test_bruteforce_gives_up_past_max_weight():
    assert distance_bruteforce(paper_code(*SMALL), max_weight=2) is None


def test_distance_is_the_min_over_both_sectors():
    code = paper_code(*SMALL)
    d_x = distance_bruteforce(code, sector="x")
    d_z = distance_bruteforce(code, sector="z")
    assert distance_bruteforce(code) == min(d_x, d_z)


def test_upper_bound_reaches_the_exact_value_here():
    assert distance_upper_bound(paper_code(*SMALL), trials=20, seed=0) == 3


@pytest.mark.parametrize("seed", range(5))
def test_upper_bound_never_undershoots(seed):
    """실제 logical의 weight를 보고하므로 너무 낮게 나올 수 없다."""
    code = paper_code(*SMALL)
    assert distance_upper_bound(code, trials=1, seed=seed) >= 3


def test_upper_bound_is_deterministic_given_a_seed():
    code = paper_code(*SMALL)
    assert (distance_upper_bound(code, trials=3, seed=7)
            == distance_upper_bound(code, trials=3, seed=7))


def test_ilp_agrees_with_bruteforce():
    code = paper_code(*SMALL)
    assert distance_ilp(code) == distance_bruteforce(code) == 3


def test_ilp_accepts_a_bound_from_the_random_search():
    """흔한 작업 흐름: 싸게 상한을 잡고, ILP가 확정하게 한다."""
    code = paper_code(*SMALL)
    bound = distance_upper_bound(code, trials=20, seed=0)
    assert distance_ilp(code, upper_bound=bound) == 3


def test_ilp_proves_absence_below_the_distance():
    """weight <= 2인 logical은 없으므로 상한을 건 문제가 실행 불가능하다."""
    assert distance_ilp(paper_code(*SMALL), upper_bound=2) is None


def test_ilp_is_the_min_over_both_sectors():
    code = paper_code(*SMALL)
    assert distance_ilp(code) == min(distance_ilp(code, sector="x"),
                                     distance_ilp(code, sector="z"))


def test_unknown_sector_is_rejected():
    with pytest.raises(ValueError, match="sector"):
        distance_bruteforce(paper_code(*SMALL), sector="y")


# --- 행렬을 직접 받는 고전판 --------------------------------------------------

def test_classical_distance_of_a_repetition_code():
    """r-bit repetition code의 거리는 r.

    ``H``가 인접 bit 비교라 ``ker(H) = {00000, 11111}``이다. nontrivial codeword가
    weight 5짜리 하나뿐이라 상한이 정확히 5로 떨어진다 — 추측이 아니라 nullspace가
    2차원이라서 나오는 값이다. generator를 nullspace가 아니라 ``H`` 자신으로 넘기면
    weight 2가 나온다.
    """
    r = 5
    H = np.zeros((r - 1, r), dtype=np.uint8)
    for i in range(r - 1):
        H[i, i] = H[i, i + 1] = 1
    L = np.ones((1, r), dtype=np.uint8)
    assert classical_distance_upper_bound(H, L, trials=20, seed=0) == r


def test_classical_upper_bound_reproduces_the_css_z_sector():
    """같은 문제를 두 경로로 — sector 헬퍼 경유와 행렬 직접 입력.

    등호는 우연이 아니다. ``_sectors["z"]``가 쓰는 ``[HZ; LZ]``의 행공간이
    ``ker(HX)``이고 ``nullspace2(HX)``도 같은 공간인데, ``rref2``의 결과는 열 순서만
    같으면 생성집합과 무관하게 유일하다. 그래서 같은 seed에서 두 경로가 같은 행을
    본다. 이게 성립해야 뒤에서 frame 둘을 같은 저울로 비교했다고 말할 수 있다.
    """
    code = paper_code(*SMALL)
    LX, _ = code.logicals()
    assert (classical_distance_upper_bound(code.HX, LX, trials=200, seed=0)
            == distance_upper_bound(code, trials=200, seed=0, sector="z"))


@pytest.mark.slow
def test_paper_288_8_12_distance():
    """Table 1의 [[288, 8, 12]] 재현: 샘플링으로 상한을 잡고 ILP로 증명한다.

    ~30분: ILP만으로도 각 몇 분씩인 부분문제 16개다.
    """
    code = paper_code("b3w6", 10, 10)
    bound = distance_upper_bound(code, trials=200, seed=1)
    assert bound == 12                              # weight 12인 실제 logical
    assert distance_ilp(code, upper_bound=bound - 1) is None   # 더 가벼운 것은 없다
