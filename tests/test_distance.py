"""Code distance — exact enumeration and randomised upper bound."""
import pytest

from qec_tile.distance import (distance_bruteforce, distance_ilp,
                               distance_upper_bound)
from qec_tile.tile import paper_code

# Small enough that C(50, 3) enumeration is instant.  Its distance is 3:
# no weight-1 or weight-2 error is both undetectable and non-stabilizer.
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
    """It reports the weight of an actual logical, so it cannot be too low."""
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
    """The usual workflow: bound it cheaply, then let the ILP confirm."""
    code = paper_code(*SMALL)
    bound = distance_upper_bound(code, trials=20, seed=0)
    assert distance_ilp(code, upper_bound=bound) == 3


def test_ilp_proves_absence_below_the_distance():
    """No logical of weight <= 2 exists, so the bounded problem is infeasible."""
    assert distance_ilp(paper_code(*SMALL), upper_bound=2) is None


def test_ilp_is_the_min_over_both_sectors():
    code = paper_code(*SMALL)
    assert distance_ilp(code) == min(distance_ilp(code, sector="x"),
                                     distance_ilp(code, sector="z"))


def test_unknown_sector_is_rejected():
    with pytest.raises(ValueError, match="sector"):
        distance_bruteforce(paper_code(*SMALL), sector="y")


@pytest.mark.slow
def test_paper_288_8_12_distance():
    """Reproduce Table 1's [[288, 8, 12]]: bound with sampling, prove with ILP.

    ~30 min: the ILP alone is 16 subproblems of a couple of minutes each.
    """
    code = paper_code("b3w6", 10, 10)
    bound = distance_upper_bound(code, trials=200, seed=1)
    assert bound == 12                              # a real logical of weight 12
    assert distance_ilp(code, upper_bound=bound - 1) is None   # none lighter

