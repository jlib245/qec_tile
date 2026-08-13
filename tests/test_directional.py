"""Directional tile code — word 파싱, walk 기하, 코드 조립."""
import pytest

from qec_tile.directional import (PAPER_CODES, build_directional_code,
                                  displacement_vectors, parse_directional_word,
                                  satisfies_parity_condition, tile_from_word,
                                  walk_edges)
from qec_tile.distance import distance_upper_bound
from qec_tile.gf2 import rank2

# 손으로 적지 않고 파생시킨다: PAPER_CODES에 코드를 추가하면 word 수준 검사에도
# 끌려들어와야 하고, 그러지 않으면 조용히 빠져나간다.
PAPER_WORDS = sorted({word for word, *_ in PAPER_CODES})


def test_single_letters_map_to_unit_steps():
    """compass 관례: 하드웨어 격자에서 N은 +y, E는 +x."""
    assert parse_directional_word("N") == [(0, 1)]
    assert parse_directional_word("E") == [(1, 0)]
    assert parse_directional_word("S") == [(0, -1)]
    assert parse_directional_word("W") == [(-1, 0)]


def test_digits_repeat_the_step():
    assert parse_directional_word("N2") == [(0, 1), (0, 1)]
    assert parse_directional_word("E3") == [(1, 0)] * 3


def test_repeat_counts_are_read_greedily():
    """``N12``는 열두 걸음이고, 한 걸음 뒤에 엉뚱한 ``2``가 붙은 게 아니다."""
    assert parse_directional_word("N12") == [(0, 1)] * 12


def test_paper_words_have_the_printed_length():
    """Table 2는 word의 글자 수를 weight w로 인쇄한다."""
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
    """오타가 조용히 더 짧은 walk로 파싱되면 안 된다."""
    with pytest.raises(ValueError):
        parse_directional_word(word)


# --- walk -----------------------------------------------------------------

@pytest.mark.parametrize("word", PAPER_WORDS)
def test_paper_words_have_weight_w(word):
    """한 글자가 한 edge이므로 Table 2의 weight 열이 곧 walk 길이다."""
    steps = parse_directional_word(word)
    assert len(walk_edges(steps)) == len(steps)


def test_w7_word_walk():
    """N2ESEN2를 코드 격자 위에서 손으로 따라간 것, walk 순서로."""
    assert walk_edges(parse_directional_word("N2ESEN2")) == [
        ("V", 0, 0), ("V", 0, 1), ("H", 0, 2), ("V", 1, 1), ("H", 1, 1),
        ("V", 2, 1), ("V", 2, 2)]


def test_w11_word_walk():
    """N2E2SESE2N2 — 논문의 [[323,14,15]] 코드 뒤에 있는 word."""
    assert walk_edges(parse_directional_word("N2E2SESE2N2")) == [
        ("V", 0, 0), ("V", 0, 1), ("H", 0, 2), ("H", 1, 2), ("V", 2, 1),
        ("H", 2, 1), ("V", 3, 0), ("H", 3, 0), ("H", 4, 0), ("V", 5, 0),
        ("V", 5, 1)]


def test_backward_steps_cross_the_edge_behind_the_vertex():
    """(0,0)에서 W로 가면 H(-1,0)을 건넌다. H(0,0)으로 읽으면 tile이 밀린다."""
    assert walk_edges(parse_directional_word("S")) == [("V", 0, -1)]
    assert walk_edges(parse_directional_word("W")) == [("H", -1, 0)]


def test_a_step_and_its_reverse_cross_the_same_edge():
    """N 다음 S는 올라온 edge로 되돌아오므로, 그 walk는 string이 아니다."""
    assert walk_edges(parse_directional_word("NS")) == [("V", 0, 0), ("V", 0, 0)]


# --- tile -----------------------------------------------------------------

def test_w7_tile():
    """w=7 walk를 방향별로 갈라 box에 담은 것."""
    assert tile_from_word("N2ESEN2") == (
        [(0, 2), (1, 1)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], 3)


def test_w11_tile():
    """논문의 [[323,14,15]] 코드 뒤에 있는 word."""
    assert tile_from_word("N2E2SESE2N2") == (
        [(0, 2), (1, 2), (2, 1), (3, 0), (4, 0)],
        [(0, 0), (0, 1), (2, 1), (3, 0), (5, 0), (5, 1)], 6)


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_offsets_are_normalised_to_the_origin(word):
    """build_tile_code는 [0,B) 밖을 거부하므로 box에 딱 붙어야 한다."""
    x_h, x_v, _ = tile_from_word(word)
    offsets = x_h + x_v
    assert min(x for x, _ in offsets) == 0
    assert min(y for _, y in offsets) == 0


def test_southward_word_is_shifted_into_the_box():
    """음의 y로 가는 walk는 잘리지 말고 평행이동돼야 한다."""
    assert tile_from_word("S2ESES2") == (
        [(0, 3), (1, 2)], [(0, 3), (0, 4), (1, 2), (2, 0), (2, 1)], 5)


def test_B_is_the_larger_span():
    """S2ESES2는 x로 3, y로 5를 걸친다. 정사각 box는 큰 쪽을 잡아야 한다."""
    _, _, B = tile_from_word("S2ESES2")
    assert B == 5


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_tile_fits_the_box(word):
    x_h, x_v, B = tile_from_word(word)
    assert all(0 <= x < B and 0 <= y < B for x, y in x_h + x_v)


def test_repeated_edge_is_rejected():
    """NSN은 V(0,0)을 두 번 걷는다 — string이 아니고, 유효한 CXSWAP 순서도 아니다."""
    with pytest.raises(ValueError, match="twice"):
        tile_from_word("NSN")


def test_explicit_B_is_honoured():
    """B는 자유 파라미터다: 논문은 자기 코드에 대해 B를 밝히지 않는다."""
    assert tile_from_word("N2ESEN2", B=4) == (
        [(0, 2), (1, 1)], [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)], 4)


def test_too_small_B_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        tile_from_word("N2ESEN2", B=1)


# --- parity 조건 ----------------------------------------------------------

def test_fig5_word_displacement_vectors():
    """Figure 5 자신의 예시 NESEN: 노란 쌍 셋과 초록 넷.

    "Green vectors have even vertical displacement, while the different shades
    of yellow indicate pairs of vectors with odd vertical displacement."
    """
    vectors = displacement_vectors(walk_edges(parse_directional_word("NESEN")))
    odd = {vector: count for vector, count in vectors.items() if vector[1] % 2}
    assert odd == {(1, 1): 2, (3, -1): 2, (1, -1): 2}
    assert sum(count for vector, count in vectors.items()
               if not vector[1] % 2) == 4


def test_odd_vertical_displacement_means_mixed_orientation():
    """H site는 하드웨어 y가 짝수, V site는 홀수에 앉으므로 Δy의 parity가 곧 방향
    판정이다 — 이 조건을 값싸게 따질 수 있게 해주는 사실."""
    for word in PAPER_WORDS:
        edges = walk_edges(parse_directional_word(word))
        for i, first in enumerate(edges):
            for second in edges[i + 1:]:
                vector = displacement_vectors([first, second])
                (delta,) = vector
                assert bool(delta[1] % 2) == (first[0] != second[0])


def test_a_word_violating_the_condition_is_detected():
    """이게 없으면 checker가 그냥 True를 돌려줘도 위의 모든 테스트가 통과한다.

    NE의 변위 벡터는 (1,1) 하나다: 수직이 홀수이고 중복도는 1.
    """
    assert not satisfies_parity_condition(
        walk_edges(parse_directional_word("NE")))


@pytest.mark.parametrize("word", PAPER_WORDS)
def test_tile_weight_is_the_word_length(word):
    """tile 자체, 조립이나 pruning 전: 7, 9, 11, 13."""
    x_h, x_v, _ = tile_from_word(word)
    assert len(x_h) + len(x_v) == len(parse_directional_word(word))


# --- 논문의 코드들 ----------------------------------------------------------
#
# 아래 전부는 우리가 실제로 짓고, 벤치마크하고, 디코딩하는 코드 위에서 돌아간다.
# 그래서 PAPER_CODES에 한 행을 추가하면 그 계약 전체가 한꺼번에 걸린다.

@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_commute(word, M, N, n, k, d):
    """Definition 1의 mutual condition은 (T2)이므로 walk가 그것을 깰 수 없다."""
    code = build_directional_code(word, M, N)
    assert not ((code.HX @ code.HZ.T) % 2).any()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_a_deterministic_schedule(word, M, N, n, k, d):
    """Definition 1의 parity 조건. 회로를 지게 될 코드들에 대해.

    word에만 의존하므로 한 word를 여러 배치에 걸쳐 일부러 반복한다: 절대 일어나면 안
    되는 일은 이것 없이 코드가 디코더까지 가는 것이다.
    """
    assert satisfies_parity_condition(walk_edges(parse_directional_word(word)))


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_no_overweight_checks(word, M, N, n, k, d):
    """절단과 pruning은 찍어낸 tile에서 qubit을 덜어낼 뿐이다."""
    weight = len(parse_directional_word(word))
    code = build_directional_code(word, M, N)
    assert code.HX.sum(1).max() <= weight
    assert code.HZ.sum(1).max() <= weight


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_no_empty_checks(word, M, N, n, k, d):
    """tile이 통째로 격자 밖에 찍히면 check가 남아서는 안 된다."""
    code = build_directional_code(word, M, N)
    assert code.HX.shape[0] == len(code.x_anchors)
    assert code.HZ.shape[0] == len(code.z_anchors)
    assert code.HX.any(axis=1).all()
    assert code.HZ.any(axis=1).all()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_reach_the_pruning_fixpoint(word, M, N, n, k, d):
    """한 번의 패스 뒤에 살아남은 모든 qubit이 두 check 타입에 보인다.

    논문의 원래 tile보다 여기서 더 값지다. 거기서는 pruning이 아무것도 지우지 않아 이
    성질이 공짜로 성립하지만, directional tile은 box를 다 채우지 않아 패스가 실제로
    돈다.
    """
    code = build_directional_code(word, M, N)
    assert (code.HX.sum(0) > 0).all()
    assert (code.HZ.sum(0) > 0).all()


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_independent_checks(word, M, N, n, k, d):
    """tile code에는 check 의존성이 없다. 여기의 pruning이 그것을 만들어내면 안 된다.

    성립하면 k = n - mx - mz이고, 이것이 k를 배치가 아니라 word의 성질로 만든다.
    """
    code = build_directional_code(word, M, N)
    assert (rank2(code.HX), rank2(code.HZ)) == (code.HX.shape[0],
                                                code.HZ.shape[0])


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_are_invariant_under_larger_B(word, M, N, n, k, d):
    """논문이 B를 밝히지 않는 이유: box를 넉넉히 잡아도 pruning이 같은 코드로 되돌린다.

    B를 올리면 Z-tile이 (1,1)만큼 밀리고(Z anchor의 재labelling) qubit 격자가
    넓어지는데, tile이 늘어난 qubit에 닿지 않으므로 pruning이 다시 버린다.
    """
    _, _, minimal = tile_from_word(word)
    codes = [build_directional_code(word, M, N, B)
             for B in (minimal, minimal + 1, minimal + 2)]
    assert len({(code.n, code.k) for code in codes}) == 1


@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_have_the_printed_n_and_k(word, M, N, n, k, d):
    """Table 2의 모든 행을, word 하나만으로.

    기하 전체를 고정하는 검사다: walk 관례, anchor 배치, pruning 패스가 동시에 다
    맞아야 독립적인 (n,k) 짝 일곱 개가 들어맞는다.
    """
    code = build_directional_code(word, M, N)
    assert (code.n, code.k) == (n, k)


@pytest.mark.slow
@pytest.mark.parametrize("word,M,N,n,k,d", PAPER_CODES)
def test_paper_codes_reach_the_printed_distance(word, M, N, n, k, d):
    """거리도 — 증명이 아니라 상한이다.

    ``distance_upper_bound``는 자기가 실제로 찾아낸 logical 연산자의 weight를
    보고하므로 d 아래에 앉을 수 없다. 논문 값에 정확히 닿는다는 것은 그런 연산자가
    존재하고 더 가벼운 것은 나오지 않았다는 뜻이다. 상한이 딱 맞음을 증명하려면
    ``distance_ilp``가 필요한데, n = 351에서는 끝나지 않는다. 배치를 고른 근거가
    이것이다: 버려진 후보들은 (n,k)는 같았지만 d = 3, 2, 6으로 돌아왔다.
    """
    code = build_directional_code(word, M, N)
    assert distance_upper_bound(code, trials=400, seed=0) == d
