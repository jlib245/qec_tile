"""Directional tile codes — word parsing, walk geometry, code assembly."""
import pytest

from qec_tile.directional import (PAPER_CODES, build_directional_code,
                                  displacement_vectors, parse_directional_word,
                                  satisfies_parity_condition, tile_from_word,
                                  walk_edges)
from qec_tile.distance import distance_upper_bound
from qec_tile.gf2 import rank2

# Derived, never listed by hand: adding a code to PAPER_CODES has to pull it
# into the word-level checks too, or it silently escapes them.
PAPER_WORDS = sorted({word for word, *_ in PAPER_CODES})


def test_single_letters_map_to_unit_steps():
    """Compass convention: N is +y and E is +x on the hardware grid."""
    assert parse_directional_word("N") == [(0, 1)]
    assert parse_directional_word("E") == [(1, 0)]
    assert parse_directional_word("S") == [(0, -1)]
    assert parse_directional_word("W") == [(-1, 0)]


def test_digits_repeat_the_step():
    assert parse_directional_word("N2") == [(0, 1), (0, 1)]
    assert parse_directional_word("E3") == [(1, 0)] * 3


def test_repeat_counts_are_read_greedily():
    """``N12`` is twelve steps, not one step followed by a stray ``2``."""
    assert parse_directional_word("N12") == [(0, 1)] * 12


def test_paper_words_have_the_printed_length():
    """Table 2 prints the word's letter count as its weight w."""
    assert len(parse_directional_word("N2ESEN2")) == 7
    assert len(parse_directional_word("N2E2SE2N2")) == 9
    assert len(parse_directional_word("N2E2SESE2N2")) == 11
    assert len(parse_directional_word("N2E2SE3SE2N2")) == 13


def test_whitespace_is_ignored():
    assert parse_directional_word("N2 E2 S") == parse_directional_word("N2E2S")


def test_repeat_of_one_is_allowed():
    assert parse_directional_word("N1E") == parse_directional_word("NE")


@pytest.mark.parametrize("word", ["NX", "2N", "N0", "", "   "])
def test_malformed_words_are_rejected(word):
    """A typo must not silently parse into a shorter walk."""
    with pytest.raises(ValueError):
        parse_directional_word(word)


# --- the walk -------------------------------------------------------------

@pytest.mark.parametrize("word", PAPER_WORDS)
def test_paper_words_have_weight_w(word):
    """One letter is one edge, so Table 2's weight column is the walk length."""
    steps = parse_directional_word(word)
    assert len(walk_edges(steps)) == len(steps)


def test_w7_word_walk():
    """N2ESEN2 traced by hand on the code lattice, in walk order."""
    assert walk_edges(parse_directional_word("N2ESEN2")) == [
        ("V", 0, 0), ("V", 0, 1), ("H", 0, 2), ("V", 1, 1), ("H", 1, 1),
        ("V", 2, 1), ("V", 2, 2)]


def test_w11_word_walk():
    """N2E2SESE2N2 — the word behind the paper's [[323,14,15]] code."""
    assert walk_edges(parse_directional_word("N2E2SESE2N2")) == [
        ("V", 0, 0), ("V", 0, 1), ("H", 0, 2), ("H", 1, 2), ("V", 2, 1),
        ("H", 2, 1), ("V", 3, 0), ("H", 3, 0), ("H", 4, 0), ("V", 5, 0),
        ("V", 5, 1)]


def test_backward_steps_cross_the_edge_behind_the_vertex():
    """Going W from (0,0) crosses H(-1,0); reading it as H(0,0) shifts the tile."""
    assert walk_edges(parse_directional_word("S")) == [("V", 0, -1)]
    assert walk_edges(parse_directional_word("W")) == [("H", -1, 0)]


def test_a_step_and_its_reverse_cross_the_same_edge():
    """N then S returns along the edge it came up, so the walk is not a string."""
    assert walk_edges(parse_directional_word("NS")) == [("V", 0, 0), ("V", 0, 0)]


# --- the tile -------------------------------------------------------------

def test_w7_tile():
    """The w=7 walk, split by orientation and boxed."""
    assert tile_from_word("N2ESEN2") == (
        [(0, 2), (1, 1)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], 3)


def test_w11_tile():
    """The word behind the paper's [[323,14,15]] code."""
    assert tile_from_word("N2E2SESE2N2") == (
        [(0, 2), (1, 2), (2, 1), (3, 0), (4, 0)],
        [(0, 0), (0, 1), (2, 1), (3, 0), (5, 0), (5, 1)], 6)


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_offsets_are_normalised_to_the_origin(word):
    """build_tile_code rejects anything outside [0,B), so the box must be hugged."""
    x_h, x_v, _ = tile_from_word(word)
    offsets = x_h + x_v
    assert min(x for x, _ in offsets) == 0
    assert min(y for _, y in offsets) == 0


def test_southward_word_is_shifted_into_the_box():
    """A walk into negative y must be translated, not clipped."""
    assert tile_from_word("S2ESES2") == (
        [(0, 3), (1, 2)], [(0, 3), (0, 4), (1, 2), (2, 0), (2, 1)], 5)


def test_B_is_the_larger_span():
    """S2ESES2 spans 3 in x but 5 in y; a square box has to take the larger."""
    _, _, B = tile_from_word("S2ESES2")
    assert B == 5


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_tile_fits_the_box(word):
    x_h, x_v, B = tile_from_word(word)
    assert all(0 <= x < B and 0 <= y < B for x, y in x_h + x_v)


def test_repeated_edge_is_rejected():
    """NSN walks V(0,0) twice — not a string, and not a valid CXSWAP order."""
    with pytest.raises(ValueError, match="twice"):
        tile_from_word("NSN")


def test_explicit_B_is_honoured():
    """B is a free parameter: the paper never states it for its own codes."""
    assert tile_from_word("N2ESEN2", B=4) == (
        [(0, 2), (1, 1)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], 4)


def test_too_small_B_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        tile_from_word("N2ESEN2", B=1)


# --- the parity condition -------------------------------------------------

def test_fig5_word_displacement_vectors():
    """Figure 5's own example, NESEN: three yellow pairs and four green.

    "Green vectors have even vertical displacement, while the different shades
    of yellow indicate pairs of vectors with odd vertical displacement."
    """
    vectors = displacement_vectors(walk_edges(parse_directional_word("NESEN")))
    odd = {vector: count for vector, count in vectors.items() if vector[1] % 2}
    assert odd == {(1, 1): 2, (3, -1): 2, (1, -1): 2}
    assert sum(count for vector, count in vectors.items()
               if not vector[1] % 2) == 4


def test_odd_vertical_displacement_means_mixed_orientation():
    """H sites sit at even hardware y and V sites at odd, so Δy parity is the
    orientation test — the fact that makes the condition cheap to reason about."""
    for word in PAPER_WORDS:
        edges = walk_edges(parse_directional_word(word))
        for i, first in enumerate(edges):
            for second in edges[i + 1:]:
                vector = displacement_vectors([first, second])
                (delta,) = vector
                assert bool(delta[1] % 2) == (first[0] != second[0])


def test_a_word_violating_the_condition_is_detected():
    """Without this the checker could just return True and every test above passes.

    NE has one displacement vector, (1,1): odd vertical, multiplicity one.
    """
    assert not satisfies_parity_condition(
        walk_edges(parse_directional_word("NE")))


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_tile_weight_is_the_word_length(word):
    """The tile itself, before any assembly or pruning: 7, 9, 11, 13."""
    x_h, x_v, _ = tile_from_word(word)
    assert len(x_h) + len(x_v) == len(parse_directional_word(word))


# --- the paper's codes ------------------------------------------------------
#
# Everything below runs on the codes we actually build, benchmark and decode,
# so adding a row to PAPER_CODES subjects it to the whole contract at once.

@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_commute(word, M, N, n, k, d):
    """Definition 1's mutual condition is (T2), so the walk cannot break it."""
    code = build_directional_code(word, M, N)
    assert not ((code.HX @ code.HZ.T) % 2).any()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_a_deterministic_schedule(word, M, N, n, k, d):
    """Definition 1's parity condition, on the codes that will carry a circuit.

    It depends on the word alone, so this repeats a word across its layouts on
    purpose: what must never happen is a code reaching the decoder without it.
    """
    assert satisfies_parity_condition(walk_edges(parse_directional_word(word)))


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_no_overweight_checks(word, M, N, n, k, d):
    """Truncation and pruning only ever remove qubits from a stamped tile."""
    weight = len(parse_directional_word(word))
    code = build_directional_code(word, M, N)
    assert code.HX.sum(1).max() <= weight
    assert code.HZ.sum(1).max() <= weight


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_no_empty_checks(word, M, N, n, k, d):
    """A tile stamped entirely off the lattice must leave no check behind."""
    code = build_directional_code(word, M, N)
    assert code.HX.shape[0] == len(code.x_anchors)
    assert code.HZ.shape[0] == len(code.z_anchors)
    assert code.HX.any(axis=1).all()
    assert code.HZ.any(axis=1).all()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_reach_the_pruning_fixpoint(word, M, N, n, k, d):
    """Every surviving qubit is seen by both check types after one pass.

    Worth more here than for the paper's original tiles, where pruning removes
    nothing and the property holds for free — a directional tile does not fill
    its box, so the pass actually runs.
    """
    code = build_directional_code(word, M, N)
    assert (code.HX.sum(0) > 0).all()
    assert (code.HZ.sum(0) > 0).all()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_independent_checks(word, M, N, n, k, d):
    """Tile codes have no check dependencies; pruning here must not create any.

    If it holds then k = n - mx - mz, which is what makes k a property of the
    word rather than of the layout.
    """
    code = build_directional_code(word, M, N)
    assert (rank2(code.HX), rank2(code.HZ)) == (code.HX.shape[0],
                                                code.HZ.shape[0])


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_are_invariant_under_larger_B(word, M, N, n, k, d):
    """Why the paper never states B: a roomier box prunes back to the same code.

    Raising B shifts the Z-tile by (1,1) — a relabelling of the Z anchors —
    and widens the qubit grid, but the tile never reaches the extra qubits so
    pruning drops them again.
    """
    _, _, minimal = tile_from_word(word)
    codes = [build_directional_code(word, M, N, B)
             for B in (minimal, minimal + 1, minimal + 2)]
    assert len({(code.n, code.k) for code in codes}) == 1


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_the_printed_n_and_k(word, M, N, n, k, d):
    """Every row of Table 2, from the word alone.

    This is the check that fixes the whole geometry: the walk convention, the
    anchor layout and the pruning pass all have to be right at once for seven
    independent (n,k) pairs to land.
    """
    code = build_directional_code(word, M, N)
    assert (code.n, code.k) == (n, k)


@pytest.mark.slow
@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_reach_the_printed_distance(word, M, N, n, k, d):
    """The distances too — an upper bound, not a proof.

    ``distance_upper_bound`` reports the weight of a logical operator it
    actually found, so it can never sit below d.  Landing exactly on the
    paper's value means such an operator exists and nothing lighter turned up.
    Proving the bound tight needs ``distance_ilp``, which does not finish at
    n = 351.  This is what chose the layouts: the discarded candidates shared
    (n,k) but came back at d = 3, 2 and 6.
    """
    code = build_directional_code(word, M, N)
    assert distance_upper_bound(code, trials=400, seed=0) == d
