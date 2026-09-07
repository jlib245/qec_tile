"""Nearest-neighbour syndrome extraction, 두 번째 구성 -- 걸음의 기하부터."""
import numpy as np
import pytest

from qec_tile.directional import (DIRECTIONS, build_directional_code,
                                  hardware_site, parse_directional_word,
                                  walk_edges)
from qec_tile.noise_model import NoiseModel
from qec_tile.walk2 import (CHECK_X, CHECK_Z, DATA, ROUTING, ROUNDS_TESTED,
                            Qubit, births_for, check_starts, detours,
                            optimise_routing, route_windows, flow_step,
                            partial_sums, trace_pruning, route_window_shortening,
                            reverse_steps, round_births, walk_crossings,
                            walk_layout, walk_memory_z_base, walk_round,
                            walk_schedule)

# 논문 Table 2의 word 셋. 아래 세 word 모두 Definition 1을 만족한다.
WORDS = ["NESEN", "N2ESEN2", "N2E2SESE2N2"]


def test_partial_sums_starts_at_the_origin_and_ends_at_the_total_displacement():
    """``NESEN``의 부분합. 걸음을 손으로 누적한 것이 기대값이다.

    ``S_0 = (0,0)``을 빠뜨리는 off-by-one을 잡는다 -- 궤적에 출발점이 없으면
    ``walk_layout``이 출발점 자리를 안 깔아 배치에 구멍이 난다.
    """
    steps = parse_directional_word("NESEN")
    assert partial_sums(steps) == [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1)]
    assert len(partial_sums(steps)) == len(steps) + 1


def test_reversing_the_steps_undoes_the_displacement():
    """역 word는 순서와 부호를 **둘 다** 뒤집은 것이다.

    총 변위만 보면 부호만 뒤집은 것도 통과하므로, 리스트 자체를 대조한다.
    """
    steps = parse_directional_word("NESEN")
    assert reverse_steps(steps) == [(0, -1), (-1, 0), (0, 1), (-1, 0), (0, -1)]
    assert reverse_steps(reverse_steps(steps)) == steps


def test_the_reverse_round_returns_every_walker_home():
    """word와 역 word를 이어 걸으면 변위가 0이다.

    "Consecutive rounds are alternated between the word D and the inverse word,
    so that the physical layout is restored" -- 배치를 만들지 않고 기하만으로
    그 전제를 고정한다.
    """
    steps = parse_directional_word("N2E2SESE2N2")
    (end_x, end_y) = partial_sums(steps)[-1]
    (back_x, back_y) = partial_sums(reverse_steps(steps))[-1]
    assert (end_x + back_x, end_y + back_y) == (0, 0)


# --- 건너는 edge -----------------------------------------------------------

@pytest.mark.parametrize("word", WORDS)
def test_the_crossings_are_the_hardware_sites_of_the_walked_edges(word):
    """``h_m``이 곧 walk의 ``m``번째 edge의 하드웨어 좌표다.

    왼쪽은 하드웨어 변위를 누적한 것이고 오른쪽은 코드 격자에서 edge에 이름을 붙인
    것이라 서로 독립인 유도다. off-by-one(``S_m``만 쓰면 만나는 *위치*가 나온다),
    부호 뒤집힘, 짝 어긋남이 전부 여기서 갈린다.
    """
    steps = parse_directional_word(word)
    assert walk_crossings(steps) == [hardware_site(edge)
                                     for edge in walk_edges(steps)]


def test_the_crossings_of_the_figure_5_word_by_hand():
    """``NESEN``은 손으로 누적한 값과 대조한다 -- 위 테스트의 두 변이 같이 틀리는 경우."""
    steps = parse_directional_word("NESEN")
    assert walk_crossings(steps) == [(0, 1), (1, 2), (2, 1), (3, 0), (4, 1)]


@pytest.mark.parametrize("word", WORDS)
def test_every_crossing_lands_on_a_data_site(word):
    """모든 ``h_m``의 좌표합이 홀수. data는 홀수합 자리에만 앉는다.

    짝수합이 나오면 check가 data가 아니라 다른 check나 routing을 먹는다는 뜻이다.
    """
    crossings = walk_crossings(parse_directional_word(word))
    assert all((cross_x + cross_y) % 2 == 1 for cross_x, cross_y in crossings)


@pytest.mark.parametrize("word", WORDS)
def test_there_is_one_crossing_per_letter(word):
    """한 글자가 한 층, 한 층이 data 하나 -- 라운드 깊이가 ``w``라는 주장의 뼈대다."""
    steps = parse_directional_word(word)
    assert len(walk_crossings(steps)) == len(steps)


@pytest.mark.parametrize("word", WORDS)
def test_the_reverse_round_meets_the_same_data_in_reverse_order(word):
    """역 라운드는 같은 data를 역순으로 먹는다. ``2*S_w``만 되돌려주면 된다.

    ``d'_j = -d_{w+1-j}`` => ``S'_j = S_{w-j} - S_w`` => ``h'_j = h_{w-j+1} - 2*S_w``.

    홀수 라운드의 창을 ``start + 2*S_w`` 기준으로 재는 근거가 이 대수다. 회로까지
    가서 검증하면 어긋나도 조용히 지나가므로 (detector는 멀쩡한 채로 다른
    stabilizer를 재게 된다) 순수 기하 단계에서 못 박는다.
    """
    steps = parse_directional_word(word)
    crossings = walk_crossings(steps)
    (end_x, end_y) = partial_sums(steps)[-1]
    back = [(2 * end_x + cross_x, 2 * end_y + cross_y)
            for cross_x, cross_y in walk_crossings(reverse_steps(steps))]
    assert back == list(reversed(crossings))


# --- 한 층 ------------------------------------------------------------------

NORTH = DIRECTIONS["N"]


def test_an_x_check_takes_the_data_it_walks_onto():
    """X check가 걸어 들어간 자리의 data와 CXSWAP하고, 둘이 자리를 바꾼다.

    control이 뒤집히면 data가 control이 되어 X stabilizer가 아닌 것을 재게 된다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(DATA, 0, (0, 1))}
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == [("CXSWAP", (0, 0), (0, 1))]
    assert measured == []
    assert qubit_at[(0, 1)].role == CHECK_X      # check가 앞으로 갔고
    assert qubit_at[(0, 0)].role == DATA         # data가 뒤로 밀렸다


def test_a_z_check_lets_the_data_control():
    """Z check는 data가 control이다. 뒤집으면 Z stabilizer가 안 나온다."""
    qubit_at = {(1, 1): Qubit(CHECK_Z, 0, (1, 1)),
                (1, 2): Qubit(DATA, 0, (1, 2))}
    gates, _ = flow_step(qubit_at, NORTH)
    assert gates == [("CXSWAP", (1, 2), (1, 1))]


def test_a_check_swaps_through_routing():
    """check가 routing을 만나면 CXSWAP이 아니라 SWAP이다.

    CXSWAP이면 CX 부분이 ``|+>`` check와 ``|0>`` routing을 얽어버려 뒤이은 측정이
    무작위가 된다 -- 잡음이 없어도 detector가 우는 실패 모드다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(ROUTING, 0, (0, 1))}
    gates, _ = flow_step(qubit_at, NORTH)
    assert gates == [("SWAP", (0, 0), (0, 1))]


def test_routing_pushes_data():
    """routing이 data를 만나면 SWAP으로 밀어준다 -- Algorithm 1의 16-19행.

    빠지면 경계 data가 check 앞으로 배달되지 않아 check가 열 하나를 덜 먹는다.
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0)),
                (0, 1): Qubit(DATA, 0, (0, 1))}
    gates, _ = flow_step(qubit_at, NORTH)
    assert gates == [("SWAP", (0, 0), (0, 1))]
    assert qubit_at[(0, 1)].role == ROUTING


def test_two_routing_qubits_do_nothing():
    """routing끼리는 gate도 없고 자리도 안 바뀐다.

    (``(0,1)``은 갈 자리가 없어 따로 잘려 나간다 -- 그건 아래 테스트가 본다.)
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0)),
                (0, 1): Qubit(ROUTING, 1, (0, 1))}
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == []
    assert measured == []
    assert qubit_at[(0, 0)] == Qubit(ROUTING, 0, (0, 0))


def test_data_never_moves_on_its_own():
    """data는 스스로 걷지 않는다. 밀어줄 이가 없으면 그냥 그 자리다."""
    qubit_at = {(0, 1): Qubit(DATA, 0, (0, 1))}
    before = dict(qubit_at)
    assert flow_step(qubit_at, NORTH) == ([], [])
    assert qubit_at == before


@pytest.mark.parametrize("letter", ["N", "E", "S", "W"])
def test_every_letter_steps_the_walkers_once(letter):
    """N/E/S/W 네 걸음의 계약. 부호가 뒤집히면 반대쪽 edge를 먹는다."""
    step = DIRECTIONS[letter]
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                step: Qubit(DATA, 0, step)}
    gates, _ = flow_step(qubit_at, step)
    assert gates == [("CXSWAP", (0, 0), step)]
    assert qubit_at[step].role == CHECK_X


def test_a_check_with_nowhere_to_go_waits_until_it_is_done():
    """갈 자리가 없으면 check는 기다린다. support를 다 먹은 뒤라면 거기서 읽힌다.

    Algorithm 1의 if/else if에는 else가 없다 -- 앞이 ``D``도 ``R``도 아니면 아무것도
    안 한다. 창 중간에서 빈 자리를 만난 check를 측정해 버리면 논문 Figure 7의
    배치(bottom X가 층 3에서 한 층 기다린다)가 거절된다. 다 먹은 check는 지금처럼
    그 자리에서 읽혀 ``|0>``이 된다 -- 창 뒤쪽 절단이 이것이다.
    """
    check = Qubit(CHECK_X, 3, (0, 0))
    qubit_at = {(0, 0): check}
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == []
    assert measured == []
    assert qubit_at[(0, 0)] == check              # 제자리에서 기다린다

    gates, measured = flow_step(qubit_at, NORTH, done={(CHECK_X, 3)})
    assert gates == []
    assert measured == [(check, (0, 0))]
    assert qubit_at[(0, 0)].role == ROUTING      # MRX 뒤에는 |0>이다


def test_routing_with_nowhere_to_go_just_waits():
    """routing은 그냥 선다 -- Algorithm 1이 16-20행의 ``end if``로 비워둔 분기다.

    읽을 것이 없으니 ``measured``에 들어가면 안 되고, 그 자리의 물리 qubit은 여전히
    거기 있으니 배치에서 빠져도 안 된다.
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0))}
    before = dict(qubit_at)
    assert flow_step(qubit_at, NORTH) == ([], [])
    assert qubit_at == before


def test_a_waiting_routing_still_carries_a_check_past_it():
    """선 routing은 뒤따르는 check의 디딤돌로 남는다.

    ``(0,1)``의 routing은 갈 자리가 없어 서고, check는 그 자리를 밟고 계속 걷는다.
    routing을 배치에서 지우는 구현이면 check가 밟을 자리를 잃어 ``(0,0)``에서
    읽혀버린다 -- 앞쪽에 support가 남아 있었다면 그만큼 덜 재게 된다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(ROUTING, 0, (0, 1))}
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == [("SWAP", (0, 0), (0, 1))]
    assert measured == []
    assert qubit_at[(0, 1)].role == CHECK_X


def test_routing_facing_a_check_does_nothing():
    """``R``은 ``D``를 만날 때만 움직인다. 앞이 check면 그냥 선다.

    두 check 사이에 낀 ``R``이 이 경우다. 여기서 ``R``까지 걸으면 뒤의 check와 같은
    자리를 두고 겹쳐서, 경계에 check가 둘 나란한 배치가 통째로 성립하지 않는다.
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0)),
                (0, 1): Qubit(CHECK_Z, 0, (0, 1)),
                (0, 2): Qubit(DATA, 0, (0, 2))}   # check는 잘리지 않고 계속 걷는다
    routing = qubit_at[(0, 0)]
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == [("CXSWAP", (0, 2), (0, 1))]
    assert measured == []
    assert qubit_at[(0, 0)] == routing        # R은 제자리


def test_a_check_behind_another_check_waits():
    """check가 check를 마주치면 기다린다. 앞 check만 걷고 뒤는 제자리다.

    에러로 처리하면 논문 Figure 7의 배치(bottom X가 층 5에서 앞 check 뒤에서 한 층
    기다린 뒤 층 6에 마지막 data를 만난다)가 거절된다. Algorithm 1에는 이 경우의
    분기가 없다.
    """
    behind = Qubit(CHECK_X, 0, (0, 0))
    qubit_at = {(0, 0): behind,
                (0, 1): Qubit(CHECK_Z, 0, (0, 1)),
                (0, 2): Qubit(DATA, 0, (0, 2))}
    gates, measured = flow_step(qubit_at, NORTH)
    assert gates == [("CXSWAP", (0, 2), (0, 1))]
    assert measured == []
    assert qubit_at[(0, 0)] == behind


def test_one_site_cannot_take_two_gates():
    """한 site가 한 층에 두 gate에 들어가면 stim의 moment 규칙 위반이다.

    check가 ``(0,1)``을 밟는 동시에 그 routing이 ``(0,2)``의 data를 민다. 올바른
    배치에서는 생길 수 없어야 하므로, 걸린다면 배치 규칙이 깨졌다는 신호다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(ROUTING, 0, (0, 1)),
                (0, 2): Qubit(DATA, 0, (0, 2))}
    with pytest.raises(ValueError, match="two gates"):
        flow_step(qubit_at, NORTH)


def test_the_layer_moves_all_at_once():
    """층 안의 이동은 동시다. 각 qubit이 정확히 한 칸씩만 간다.

    제자리 갱신을 루프 안에서 해버리면 방금 밀린 data를 다음 walker가 또 민다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(DATA, 0, (0, 1)),
                (0, 2): Qubit(ROUTING, 0, (0, 2)),
                (0, 3): Qubit(DATA, 1, (0, 3))}
    gates, _ = flow_step(qubit_at, NORTH)
    assert gates == [("CXSWAP", (0, 0), (0, 1)), ("SWAP", (0, 2), (0, 3))]
    assert qubit_at[(0, 1)].role == CHECK_X
    assert qubit_at[(0, 0)] == Qubit(DATA, 0, (0, 1))    # 한 칸만 밀렸다
    assert qubit_at[(0, 3)].role == ROUTING
    assert qubit_at[(0, 2)] == Qubit(DATA, 1, (0, 3))


def test_a_step_and_its_reverse_restore_the_layout():
    """N 다음 S면 배치가 원래대로 -- 라운드 교대로 배치가 복원된다는 것의 최소 단위."""
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(DATA, 0, (0, 1))}
    before = dict(qubit_at)
    flow_step(qubit_at, NORTH)
    flow_step(qubit_at, DIRECTIONS["S"])
    assert qubit_at == before


# --- check 출발점 -----------------------------------------------------------

def test_x_checks_start_at_the_walk_origin():
    """``NESEN``은 walk가 원점에 붙어 있어(``min = 0``) 보정이 없다.

    ``min``의 부호가 뒤집히면 X check가 tile 하나만큼 어긋난 자리에서 출발한다.
    """
    code = build_directional_code("NESEN", 4, 4)
    x_starts, _ = check_starts(code, "NESEN")
    assert x_starts == [(2 * i, 2 * j) for i, j in code.x_anchors]


@pytest.mark.parametrize("word,L1,L2,B", [
    ("NESEN", 4, 4, None),
    ("NESEN", 4, 4, 4),            # B는 자유 파라미터. Z 출발점이 B와 함께 움직인다
    ("N2ESEN2", 4, 4, None),
    ("N2E2SESE2N2", 12, 4, None),
])
def test_the_walk_reaches_exactly_the_check_support(word, L1, L2, B):
    """핵심 계약: 모든 check에 대해 ``{start + h_m}`` 중 data인 것 == 그 행의 support.

    왼쪽은 흐름의 기하에서, 오른쪽은 ``tile.py``의 조립에서 나오므로 독립이다.
    출발점, 걸음 부호, ``code.B``, 경계 절단, pruning이 동시에 맞아야만 성립한다.
    """
    code = build_directional_code(word, L1, L2, B)
    crossings = walk_crossings(parse_directional_word(word))
    col_of = {hardware_site(edge): col for col, edge in enumerate(code.qubits)}
    x_starts, z_starts = check_starts(code, word)
    for checks, starts in ((code.HX, x_starts), (code.HZ, z_starts)):
        for row, (start_x, start_y) in zip(checks, starts):
            reached = {col_of[site]
                       for site in ((start_x + cross_x, start_y + cross_y)
                                    for cross_x, cross_y in crossings)
                       if site in col_of}
            assert reached == set(np.flatnonzero(row))


def test_a_word_whose_z_tile_is_not_reachable_is_rejected():
    """``NESE``는 Definition 1의 parity 조건을 어긴다.

    코드 자체는 지어지지만 Z check가 같은 word로는 자기 tile에 닿지 못하므로,
    출발점을 정하는 시점에 걸려야 한다.
    """
    code = build_directional_code("NESE", 4, 4)
    with pytest.raises(ValueError, match="parity"):
        check_starts(code, "NESE")


# --- 배치 ------------------------------------------------------------------

LAYOUTS = [("NESEN", 2, 2), ("NESEN", 4, 4), ("N2ESEN2", 4, 4)]


def small_layout():
    """``NESEN`` 2x2 -- 논문 Figure 4의 코드 ``[[18,2,dx=3,dz=2]]``."""
    code = build_directional_code("NESEN", 2, 2)
    return code, walk_layout(code, "NESEN")


def test_every_data_column_sits_at_its_edge_midpoint():
    """모든 열이 정확히 한 번, 자기 edge 중점에 data로 앉는다."""
    code, qubit_at = small_layout()
    seats = {qubit.index: site for site, qubit in qubit_at.items()
             if qubit.role == DATA}
    assert len(seats) == code.n
    assert all(seats[col] == hardware_site(edge)
               for col, edge in enumerate(code.qubits))


def test_checks_sit_at_their_starts():
    """역할과 index가 ``HX``/``HZ`` 행과 정렬된다.

    어긋나면 detector가 엉뚱한 행끼리 비교하게 되는데, 그건 무잡음 회로에서도
    조용히 지나간다.
    """
    code, qubit_at = small_layout()
    x_starts, z_starts = check_starts(code, "NESEN")
    for row, site in enumerate(x_starts):
        assert qubit_at[site] == Qubit(CHECK_X, row, site)
    for row, site in enumerate(z_starts):
        assert qubit_at[site] == Qubit(CHECK_Z, row, site)


def test_every_check_is_present():
    """하나라도 빠지면 그 stabilizer를 아예 안 재는 회로가 된다."""
    code, qubit_at = small_layout()
    roles = [qubit.role for qubit in qubit_at.values()]
    assert roles.count(CHECK_X) == code.HX.shape[0]
    assert roles.count(CHECK_Z) == code.HZ.shape[0]


@pytest.mark.parametrize("word,L1,L2", LAYOUTS)
def test_both_paths_are_covered(word, L1, L2):
    """check가 갈 경로와 data가 갈 경로에 빈 칸이 없다 -- 배치 규칙 그 자체.

    규칙을 되읽는 것이라 약하지만, 두 항 중 하나를 빠뜨리는 회귀를 잡는다.
    """
    code = build_directional_code(word, L1, L2)
    qubit_at = walk_layout(code, word)
    steps = parse_directional_word(word)
    x_starts, z_starts = check_starts(code, word)
    for starts, trajectory in ((x_starts + z_starts, partial_sums(steps)),
                               ([hardware_site(edge) for edge in code.qubits],
                                partial_sums(reverse_steps(steps)))):
        for start_x, start_y in starts:
            for offset_x, offset_y in trajectory:
                assert (start_x + offset_x, start_y + offset_y) in qubit_at


def test_routing_grows_with_the_boundary_not_the_area():
    """routing은 둘레만큼만 자란다 -- 논문 Appendix C의 ``n_r = O(sqrt n)``.

    실측하면 ``routing / sqrt(n)``이 2x2부터 8x8까지 6.60, 6.79, 6.87, 6.91로
    거의 상수다. 면적으로 자라는 규칙(격자를 통째로 채우는 것)은 여기서 깨진다.
    """
    ratios = []
    for size in (2, 4, 6, 8):
        code = build_directional_code("NESEN", size, size)
        routing = sum(1 for qubit in walk_layout(code, "NESEN").values()
                      if qubit.role == ROUTING)
        ratios.append(routing / code.n ** 0.5)
    assert max(ratios) / min(ratios) < 1.1


@pytest.mark.parametrize("word,L1,L2", LAYOUTS)
def test_a_full_round_measures_exactly_the_stabilisers(word, L1, L2):
    """핵심 계약: 배치에 ``flow_step``을 ``w``번 돌리면 각 check가 자기 행을 먹는다.

    왼쪽은 흐름을 실제로 굴린 결과이고 오른쪽은 ``tile.py``의 조립이라 독립이다.
    배치가 한 자리라도 모자라면 check가 잘리거나 열을 덜 먹어 여기서 드러난다.
    """
    code = build_directional_code(word, L1, L2)
    qubit_at = walk_layout(code, word)
    collected: dict = {}
    for step in parse_directional_word(word):
        gates, measured = flow_step(qubit_at, step)
        assert measured == []                  # 잘리는 check가 없어야 한다
        for name, first, second in gates:
            if name != "CXSWAP":
                continue
            one, other = qubit_at[first], qubit_at[second]
            check, data = (one, other) if one.role != DATA else (other, one)
            collected.setdefault((check.role, check.index), set()).add(data.index)
    for checks, role in ((code.HX, CHECK_X), (code.HZ, CHECK_Z)):
        for row in range(checks.shape[0]):
            assert collected.get((role, row), set()) == set(
                np.flatnonzero(checks[row]))


# --- 한 라운드 --------------------------------------------------------------

def test_the_round_has_one_layer_per_letter():
    """한 글자가 한 층 -- 라운드 깊이가 ``w``이고 코드 크기와 무관하다는 주장."""
    code, qubit_at = small_layout()
    steps = parse_directional_word("NESEN")
    layers, _ = walk_round(code, qubit_at, steps)
    assert len(layers) == len(steps) == 5


def test_the_round_measures_exactly_the_stabilisers():
    """``walk_round``가 안에서 ``HX``/``HZ`` 행과 대조한다.

    여기서는 다른 각도로 한 번 더 건다: ``CXSWAP`` 총수가 support 무게의 총합과
    같아야 한다. 남의 열을 먹었으면 안쪽 대조가, 덜 먹었으면 이 개수가 잡는다.
    """
    code, qubit_at = small_layout()
    layers, _ = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    cxswaps = sum(1 for layer in layers for name, _, _ in layer
                  if name == "CXSWAP")
    assert cxswaps == int(code.HX.sum() + code.HZ.sum()) == 60


def test_every_gate_is_nearest_neighbour():
    """논문 제목이 주장하는 것. 모든 2q gate가 격자에서 한 칸 떨어진 두 자리에 걸린다."""
    code, qubit_at = small_layout()
    layers, _ = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    for layer in layers:
        for _, (first_x, first_y), (second_x, second_y) in layer:
            assert abs(first_x - second_x) + abs(first_y - second_y) == 1


def test_the_window_is_where_the_check_touches_data():
    """창 = data를 처음 만나는 층부터 마지막으로 만나는 층까지 (0부터 셈).

    회로가 reset과 측정을 여기에 맞춰 놓는다. X check는 여섯 개가 다섯 층을 꽉
    채우지만, Z check는 경계에서 tile이 잘려 창이 짧다 -- 실측값이다.
    """
    code, qubit_at = small_layout()
    _, windows = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    assert len(windows) == code.HX.shape[0] + code.HZ.shape[0] == 16
    assert all(windows[(CHECK_X, row)] == (0, 4)
               for row in range(code.HX.shape[0]))
    assert sorted(windows[(CHECK_Z, row)]
                  for row in range(code.HZ.shape[0])) == [
        (0, 1), (0, 1), (0, 3), (0, 3), (0, 4),
        (0, 4), (1, 4), (1, 4), (3, 4), (3, 4)]


def test_a_check_carrying_the_wrong_row_is_caught():
    """안쪽 대조에 이빨이 있는지. 0번 X check에 1번 행을 붙이면 먹는 열이 달라진다."""
    code, qubit_at = small_layout()
    x_starts, _ = check_starts(code, "NESEN")
    qubit_at[x_starts[0]] = Qubit(CHECK_X, 1, x_starts[0])
    with pytest.raises(ValueError, match="support"):
        walk_round(code, qubit_at, parse_directional_word("NESEN"))


def test_the_inverse_word_restores_the_layout():
    """"Consecutive rounds are alternated between the word D and the inverse word,
    so that the physical layout is restored without introducing long-range
    operations."
    """
    def fingerprint(layout):                 # routing끼리는 구별되지 않는다
        return {site: (qubit.role, qubit.index
                       if qubit.role != ROUTING else None)
                for site, qubit in layout.items()}

    code, qubit_at = small_layout()
    before = fingerprint(qubit_at)
    steps = parse_directional_word("NESEN")
    walk_round(code, qubit_at, steps)
    assert fingerprint(qubit_at) != before          # 한 라운드 뒤에는 옮겨져 있고
    walk_round(code, qubit_at, reverse_steps(steps))
    assert fingerprint(qubit_at) == before          # 역 word가 되돌린다


def test_the_larger_layout_measures_its_stabilisers_too():
    """경계 tile이 훨씬 많은 배치에서도 같은 계약."""
    code = build_directional_code("N2ESEN2", 4, 4)
    walk_round(code, walk_layout(code, "N2ESEN2"),
               parse_directional_word("N2ESEN2"))


# --- routing 최적화 ---------------------------------------------------------

PRUNED = [("NESEN", 2, 2), ("NESEN", 4, 4), ("N2ESEN2", 4, 4),
          ("N2E2SESE2N2", 12, 4)]


@pytest.mark.parametrize("word,L1,L2", LAYOUTS)
def test_the_windows_match_what_the_round_measures(word, L1, L2):
    """``route_windows``의 산술이 흐름을 실제로 돌린 결과와 같다.

    왼쪽은 ``h_m``으로 바로 낸 것이고 오른쪽은 ``flow_step``을 ``w``번 굴린 결과라
    서로 독립이다. 창이 틀리면 reset·측정이 엉뚱한 moment에 놓인다.
    """
    code = build_directional_code(word, L1, L2)
    _, windows = walk_round(code, walk_layout(code, word),
                            parse_directional_word(word))
    assert route_windows(code, word) == windows


@pytest.mark.parametrize("word,L1,L2", PRUNED)
def test_pruning_keeps_the_stabilisers(word, L1, L2):
    """줄인 배치로도 각 check가 자기 행을 그대로 먹는다.

    ``walk_round``가 안에서 ``HX``/``HZ``와 대조하므로, 잘라낸 자리가 실은 필요한
    것이었으면 여기서 터진다. 논문의 채택 기준과 같다: "The optimisation preserves
    the measured stabiliser supports, detectors and logical observables".
    """
    code = build_directional_code(word, L1, L2)
    layout, shifts = route_window_shortening(code, word)
    walk_round(code, layout, parse_directional_word(word),
               births_for(code, word, 0, layout, shifts))


def test_pruning_removes_routing():
    """실측: routing이 28->18, 48->30, 84->53으로 준다.

    죽은 앞뒤 구간의 자리와 옮긴 것의 원래 자리가 같이 빠진다. 출발점을 한 칸 옮길
    때마다 배치를 다시 지어 검증하므로, 필요 없어진 앞 구간이 통째로 빠진다.
    """
    for (word, L1, L2), before, after in (
            (("NESEN", 2, 2), 28, 18), (("NESEN", 4, 4), 48, 30),
            (("N2ESEN2", 4, 4), 84, 53)):
        code = build_directional_code(word, L1, L2)
        plain = sum(1 for qubit in walk_layout(code, word).values()
                    if qubit.role == ROUTING)
        layout, _ = route_window_shortening(code, word)
        pruned = sum(1 for qubit in layout.values() if qubit.role == ROUTING)
        assert (plain, pruned) == (before, after)


@pytest.mark.parametrize("word,L1,L2", PRUNED)
def test_a_birth_seat_is_left_empty(word, L1, L2):
    """태어날 자리는 배치에 없어야 한다.

    ``flow_step``은 교환만 하므로 점유 여부가 라운드 내내 안 바뀐다 -- 비워둬야 층
    ``first``까지 비어 있고, 거기 check를 넣어도 지우는 것이 없다. routing을 두면
    앞에 data가 있을 때 걸어가 버려 흐름에 없던 walker가 하나 늘어난다.
    """
    code = build_directional_code(word, L1, L2)
    layout, shifts = route_window_shortening(code, word)
    for _, _, _, seat, _, _ in round_births(code, word, 0, shifts):
        assert seat not in layout


def test_no_check_moves_past_its_window():
    """출발점은 창이 열리는 층을 넘지 못한다 -- 넘으면 첫 상호작용을 건너뛴다.

    창이 층 0에 열리는 check는 아예 못 옮기고(그래서 ``shifts``에 없고), 나머지는
    ``0 < j <= first``다. ``N2ESEN2`` 4x4는 창이 늦게 열리는 check가 16개이고 이제
    전부 옮겨진다 -- 단수 조건 대신 검증으로 판단하기 때문이다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    layout, shifts = route_window_shortening(code, "N2ESEN2")
    windows = route_windows(code, "N2ESEN2")
    late = {key for key, (first, _) in windows.items() if first > 0}
    assert len(late) == 16
    assert set(shifts) <= late               # 창이 층 0이면 못 옮긴다
    for key, j in shifts.items():
        assert 0 < j <= windows[key][0]


def two_rounds(code, word, layout, shifts=()):
    """word와 역 word 한 번씩. ``walk_round``가 안에서 support를 대조한다."""
    steps = parse_directional_word(word)
    qubit_at = dict(layout)
    for round_index, these in enumerate((steps, reverse_steps(steps))):
        walk_round(code, qubit_at, these,
                   births_for(code, word, round_index, qubit_at, shifts))
    return qubit_at


def fully_pruned(code, word):
    """구성적 절단 위에 후보 이동 탐색까지 -- 최종 배치."""
    return trace_pruning(code, word, *route_window_shortening(code, word))


def test_generating_and_testing_removes_more():
    """창 절단만으로는 최소가 아니다. 후보 이동을 더 시도하면 빠진다 -- 실측
    18->14, 30->24, 53->47.

    기본값은 data 이동을 끈 상태다 (``move_data=False``). 켜면 각각 12/20/44까지 가지만
    data가 논리 자리에서 깊이 4~5까지 끌려 들어간다.

    논문 Appendix C의 trace pruning이 이것이다. 창 절단이 잡지 못하는 자리는 그 자리를
    지나가던 walker가 실은 없어도 되는 경우인데, 그건 기하에 따라 달라 돌려보는 편이
    확실하다.
    """
    for (word, L1, L2), before, after in (
            (("NESEN", 2, 2), 18, 14), (("NESEN", 4, 4), 30, 24),
            (("N2ESEN2", 4, 4), 53, 47)):
        code = build_directional_code(word, L1, L2)
        layout, _ = route_window_shortening(code, word)
        assert sum(1 for q in layout.values() if q.role == ROUTING) == before
        tested, _ = fully_pruned(code, word)
        assert sum(1 for q in tested.values() if q.role == ROUTING) == after


@pytest.mark.parametrize("word,L1,L2", LAYOUTS)
def test_the_tested_layout_still_measures_the_stabilisers(word, L1, L2):
    """더 줄인 배치로도 두 라운드가 각자 자기 행을 그대로 먹는다.

    채택 기준이 이것이므로 사실상 함수를 되읽는 것이지만, 반환된 배치가 마지막에
    되돌려진 상태로 새어나오는 것을 잡는다.
    """
    code = build_directional_code(word, L1, L2)
    two_rounds(code, word, *fully_pruned(code, word))


# 논문 Figure 7 (after)의 [[60,4,5]] 배치. arXiv HTML의 routing_optimisation.svg를
# 파싱해 우리 좌표로 옮긴 것 -- routing 30 = Table 2의 146 - 60 - 56. check 자리는
# ``route_window_shortening``의 shifts와 같고, 위쪽 data 여섯이 한 칸 S로 옮겨져 있다.
PAPER_FIGURE_7_ROUTING = [
    (-2, 8), (-1, -1), (-1, 0), (-1, 2), (-1, 4), (-1, 6), (-1, 8), (-1, 9),
    (1, -2), (1, 7), (3, -2), (3, 7), (5, -2), (5, 7), (7, -2), (7, 7),
    (8, -2), (8, -1), (8, 0), (8, 2), (8, 4), (8, 6), (8, 8), (9, -2), (9, 7),
    (10, -1), (10, 0), (10, 2), (10, 4), (10, 6)]
PAPER_FIGURE_7_DATA_SHIFTS = {(1, 10): (1, 9), (3, 10): (3, 9), (5, 10): (5, 9),
                              (7, 10): (7, 9), (9, 10): (9, 9), (10, 9): (10, 8)}


def test_the_paper_figure_7_layout_walks():
    """논문이 실제로 쓴 최적화 배치가 통과해야 한다.

    이 배치는 bottom X check가 창 안에서 두 번 기다린다 -- 층 3은 갈 자리 ``(1,-3)``
    이 없어서, 층 5는 앞이 다른 check라서 -- 그리고 층 6에 마지막 data를 만난다.
    빈 자리에서 측정하거나 check-check를 에러로 내면 라운드 0에서 ``check_x 0``이
    support ``[0, 40]`` 중 ``[0]``만 먹었다고 거절된다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    layout = {site: qubit for site, qubit in walk_layout(code, "N2ESEN2").items()
              if qubit.role != ROUTING}
    for home, seat in PAPER_FIGURE_7_DATA_SHIFTS.items():
        layout[seat] = Qubit(DATA, layout.pop(home).index, seat)
    _, shifts = route_window_shortening(code, "N2ESEN2")
    layout = {site: qubit for site, qubit in layout.items()
              if (qubit.role, qubit.index) not in shifts}   # 옮긴 check는 태어난다
    for serial, site in enumerate(PAPER_FIGURE_7_ROUTING):
        assert site not in layout
        layout[site] = Qubit(ROUTING, serial, site)
    assert sum(1 for q in layout.values() if q.role == ROUTING) == 30

    steps = parse_directional_word("N2ESEN2")
    for round_index in range(ROUNDS_TESTED):
        walk_round(code, layout,
                   steps if round_index % 2 == 0 else reverse_steps(steps),
                   births_for(code, "N2ESEN2", round_index, layout, shifts))


def test_a_detour_is_the_walk_between_two_data_meetings():
    """우회로 = 창 안에서 data를 안 만나는 연속 층에 check가 밟는 자리들.

    4x4 ``N2ESEN2``에서 우회로는 bottom X 넷과 top X 넷에만 있다. bottom X는 층 2에
    data를 먹고 S, E, N으로 돌아 층 6에 마지막 data를 먹으므로 그 세 층에 밟는 자리
    ``A + S_4, S_5, S_6``이 우회로다. 창 밖의 죽은 구간은 우회로가 아니고(그건 창
    절단 몫), 밟는 자리는 떠나는 자리 ``S_m``이 아니라 도착하는 자리 ``S_{m+1}``이다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    found = detours(code, "N2ESEN2")
    assert found[(CHECK_X, 0)] == [[(1, -3), (2, -3), (2, -2)]]
    assert found[(CHECK_X, 6)] == [[(0, 10), (1, 10), (1, 9)]]
    assert found[(CHECK_X, 10)] == []                  # 벌크
    assert sum(1 for runs in found.values() if runs) == 8


def test_group_deletions_reach_the_paper():
    """묶음 greedy(우회로 + 가장자리 줄)로 논문 Figure 7의 30 이하에 닿는다 -- 실측 29.

    단일 삭제로는 47에서 하나도 못 뺀다: 지운 자리를 밟던 check가 한 층만 기다려
    다른 walker와 겹친다. 우회로를 통째로(아래 줄 8개), 위 data를 한 칸 내리며 그 줄
    routing을 같이(4+4), 왼쪽 가장자리 줄의 절반(``(-2,5),(-2,7)``)을 같이 지워야
    전부가 기다리며 맞아떨어진다. 논문보다 하나 적은 것은 오른쪽 경계 data를 두 칸
    내리는 대신 논문이 routing을 남긴 자리 차이다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    layout, _ = optimise_routing(code, "N2ESEN2")
    assert sum(1 for q in layout.values() if q.role == ROUTING) == 29


def test_an_edge_line_goes_together_or_not_at_all():
    """가장 바깥 줄의 routing은 통째로 지워야 산다 -- 하나씩은 전부 실패한다.

    4x4 ``N2ESEN2``의 창 절단 배치에서 맨 아래 줄 ``y=-3``, bottom X 넷 아래의 routing
    8개(``x = 1..8``)는 넷의 우회로가 서로 떠받치는 자리다. 하나만 빼면 그 check만 한 층
    기다려 이웃과 겹치고(``two gates in one layer``), 여덟을 같이 빼면 넷이 다 기다리며
    맞는다. 줄 끝의 ``(-1,-3)``은 혼자 빠지는 꼬리 자리라 뺀다. 논문의 "iteratively
    deleting routing qubits near the edges"를 줄 단위로 읽는 근거다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    layout, shifts = route_window_shortening(code, "N2ESEN2")
    bottom = [site for site, qubit in layout.items()
              if qubit.role == ROUTING and site[1] == -3 and site[0] >= 1]
    assert len(bottom) == 8

    def survives(trial) -> bool:
        steps = parse_directional_word("N2ESEN2")
        qubit_at = dict(trial)
        try:
            for round_index in range(ROUNDS_TESTED):
                walk_round(code, qubit_at,
                           steps if round_index % 2 == 0 else reverse_steps(steps),
                           births_for(code, "N2ESEN2", round_index, qubit_at, shifts))
            return True
        except ValueError:
            return False

    assert survives({s: q for s, q in layout.items() if s not in bottom})
    for site in bottom:
        assert not survives({s: q for s, q in layout.items() if s != site})


def test_nothing_more_can_be_removed():
    """고정점까지 돈다 -- 남은 routing은 하나도 더 뺄 수 없다.

    한 번만 훑으면 안 된다: 하나를 빼면 다른 자리가 뺄 수 있게 되고, 출발점을 옮기면
    그 앞 구간이 다시 후보가 된다.
    """
    code = build_directional_code("NESEN", 2, 2)
    tested, shifts = fully_pruned(code, "NESEN")
    for site in [s for s, q in tested.items() if q.role == ROUTING]:
        trial = {s: q for s, q in tested.items() if s != site}
        with pytest.raises(ValueError):
            two_rounds(code, "NESEN", trial, shifts)


def test_the_single_user_condition_only_gates_the_second_substep():
    """원문의 순서: 먼저 check를 옮기고, **그 다음** 단수 조건으로 흡수한다.

    "the check start position can be shifted and the corresponding terminal routing
    sites can be removed, provided that no check-start collision or data-check
    overlap is introduced. **We then apply the same principle** to terminal routing
    sites **used only by a single** data or check qubit q."

    조건을 1a에도 걸면 check가 못 움직여 훨씬 덜 빠진다 -- ``N2E2SESE2N2`` 17x6에서
    248 대신 282였다. 여기서는 그 순서의 결과를 숫자로 고정한다: data 이동은 1b에서만
    일어나므로 창이 늦게 열리는 data 중 일부만 옮겨진다.
    """
    code = build_directional_code("N2ESEN2", 4, 4)
    layout, shifts = route_window_shortening(code, "N2ESEN2")
    home = {col: hardware_site(edge)
            for col, edge in enumerate(code.qubits)}
    moved_data = [site for site, qubit in layout.items()
                  if qubit.role == DATA and site != home[qubit.index]]
    assert shifts                              # 1a에서 check가 옮겨졌다
    assert moved_data                          # 1b에서 data도 옮겨졌다
    assert len(moved_data) < code.n            # 전부는 아니다
    two_rounds(code, "N2ESEN2", layout, shifts)


def test_a_missing_check_is_caught():
    """check가 배치에서 빠지면 아무 열도 못 먹어 대조에서 걸린다.

    원문의 "no **check-start collision**" 조건을 우리는 사전에 안 보고 검증으로
    대신한다. 두 check가 같은 자리를 쓰면 배치 dict에서 한쪽이 덮여 사라지는데,
    그 결과가 여기서 잡히는지를 고정한다. 빠진 check 자리를 뒤따르던 것이 밟아 한
    층에 gate가 겹치거나, 아니면 대조에서 걸린다.
    """
    code = build_directional_code("NESEN", 2, 2)
    layout = walk_layout(code, "NESEN")
    x_starts, _ = check_starts(code, "NESEN")
    del layout[x_starts[1]]
    with pytest.raises(ValueError, match="support|two gates"):
        walk_round(code, layout, parse_directional_word("NESEN"))


def test_a_check_on_a_data_site_is_caught():
    """check를 data 자리에 놓으면 그 data가 덮여 흐름이 깨진다.

    원문의 "no **data-check overlap**" 조건 쪽이다. 덮인 자리에서 걷는 것끼리 겹쳐
    ``two gates in one layer``로 걸린다 (check-check는 기다리므로 에러가 아니다).
    """
    code = build_directional_code("NESEN", 2, 2)
    layout = walk_layout(code, "NESEN")
    site = next(s for s, qubit in sorted(layout.items())
                if qubit.role == DATA)
    layout[site] = Qubit(CHECK_X, 0, site)
    with pytest.raises(ValueError, match="two gates|support"):
        walk_round(code, layout, parse_directional_word("NESEN"))


@pytest.mark.parametrize("word,L1,L2", PRUNED)
def test_moved_data_lands_on_a_vacated_site(word, L1, L2):
    """"a data-start shift onto a former **routing** coordinate".

    옮긴 data가 앉는 자리는 무손실 배치에서 routing이었던 곳이다 -- 단수 조건이 그
    자리를 그 열만 쓰게 보장하므로 다른 check의 출발점일 수 없다. 예외는 그 자리가
    **자기도 옮겨 간 다른 열의 제자리**인 경우다 (``N2E2SESE2N2`` 17x6에서 6건 중 1건).

    그리고 어느 경우든 열마다 자리가 하나씩이어야 한다 -- 두 열이 겹치면 한쪽이 덮여
    사라진다.
    """
    code = build_directional_code(word, L1, L2)
    plain = walk_layout(code, word)
    layout, _ = route_window_shortening(code, word)
    home_of = {col: hardware_site(edge)
               for col, edge in enumerate(code.qubits)}
    seats = {qubit.index: site for site, qubit in layout.items()
             if qubit.role == DATA}
    assert len(seats) == code.n                # 열마다 자리 하나
    moved = {col for col, site in seats.items() if site != home_of[col]}
    for col in moved:
        site = seats[col]
        if plain[site].role == ROUTING:
            continue
        other = plain[site].index              # 다른 열의 제자리였다면
        assert plain[site].role == DATA and other in moved


def occupancy(layout) -> dict:
    """자리별 (역할, index). routing끼리는 ``|0>``이라 구별되지 않는다."""
    return {site: (qubit.role, qubit.index if qubit.role != ROUTING else None)
            for site, qubit in layout.items()}


@pytest.mark.parametrize("word,L1,L2", [("NESEN", 2, 2), ("N2ESEN2", 4, 4)])
def test_the_layout_settles_into_a_two_round_cycle(word, L1, L2):
    """라운드 교대가 배치를 주기 2로 되돌린다.

    무손실 배치는 두 라운드마다 **초기 상태로** 복원된다 ("the physical layout is
    restored"). 줄인 배치는 첫 라운드에 태어날 자리들이 채워지며 자리가 한 번 늘고,
    그 뒤로는 라운드 2 상태에서 주기 2로 돈다 -- 초기 상태가 과도기다.

    ``survives``가 두 라운드만 보는 근거가 이것이다. 과도기(정방향)와 역방향을 한 번씩
    거치면 그 뒤는 반복이므로, 배치 규칙을 건드려 3라운드 이후에만 깨지는 회귀가 있으면
    여기서 드러난다.
    """
    code = build_directional_code(word, L1, L2)
    steps = parse_directional_word(word)
    backward = reverse_steps(steps)

    def marks_over(layout, shifts):
        qubit_at = dict(layout)
        marks = [occupancy(qubit_at)]
        for round_index in range(5):
            walk_round(code, qubit_at,
                       steps if round_index % 2 == 0 else backward,
                       births_for(code, word, round_index, qubit_at, shifts))
            marks.append(occupancy(qubit_at))
        return marks

    plain = marks_over(walk_layout(code, word), {})
    assert plain[2] == plain[0] and plain[4] == plain[0]

    pruned = marks_over(*fully_pruned(code, word))
    assert pruned[0] != pruned[2]              # 초기는 과도기
    assert pruned[4] == pruned[2]              # 그 뒤로는 주기 2
    assert len(pruned[0]) < len(pruned[2]) == len(pruned[4])


def test_more_rounds_stay_silent():
    """줄인 배치가 라운드 수와 무관하게 조용하다.

    ``rounds=3``만 보면 주기가 어긋나 5라운드에서만 우는 회귀를 놓친다. 홀수·짝수를
    섞어 걸어둔다.
    """
    code = build_directional_code("NESEN", 2, 2)
    layout, shifts = fully_pruned(code, "NESEN")
    for rounds in (1, 2, 3, 5, 8):
        circuit = walk_memory_z_base(code, "NESEN", rounds, layout, shifts)
        detections, observables = circuit.compile_detector_sampler().sample(
            32, separate_observables=True)
        assert not detections.any() and not observables.any(), rounds
        assert (circuit.detector_error_model().num_detectors
                == circuit.num_detectors), rounds


def test_the_tested_layout_shrinks_the_circuit_too():
    """pruning은 회로까지 줄인다 -- 실측 54 -> 48 qubit.

    detector와 observable은 그대로여야 한다 -- 그게 채택 기준이다.
    """
    code = build_directional_code("NESEN", 2, 2)
    plain = walk_memory_z_base(code, "NESEN", 3, walk_layout(code, "NESEN"))
    tested = walk_memory_z_base(code, "NESEN", 3, *fully_pruned(code, "NESEN"))
    assert (plain.num_qubits, tested.num_qubits) == (54, 48)
    assert tested.num_detectors == plain.num_detectors
    assert tested.num_observables == plain.num_observables
    detections, observables = tested.compile_detector_sampler().sample(
        64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()


# --- 스케줄 ----------------------------------------------------------------

def gate_sites(moment) -> set:
    """그 moment의 gate가 건드리는 자리."""
    _, gates, _ = moment
    return {site for _, site_a, site_b in gates for site in (site_a, site_b)}


def test_rounds_must_be_positive():
    code = build_directional_code("NESEN", 2, 2)
    with pytest.raises(ValueError, match="rounds"):
        walk_schedule(code, "NESEN", rounds=0)


def test_a_round_is_w_plus_two_moments_deep():
    """"one syndrome-extraction round has depth w+2, including check-qubit
    preparation and measurement."

    양 끝 moment에는 gate가 없다 -- 층 ``i``의 gate가 moment ``i+1``에 들어가므로
    gate는 moment 1..w에만 있다. 자는 check의 ``SWAP``을 빼지 않으면 그 자리가
    묶여서 reset이 자기 moment를 하나 더 차지하고 깊이가 늘어난다.
    """
    code = build_directional_code("NESEN", 2, 2)
    schedule, _, _ = walk_schedule(code, "NESEN", rounds=2)
    for moments in schedule:
        assert len(moments) == 5 + 2
        assert gate_sites(moments[0]) == set()
        assert gate_sites(moments[-1]) == set()


def test_every_check_is_reset_and_measured_once_per_round():
    """라운드마다 모든 check가 정확히 한 번 켜지고 한 번 읽힌다.

    빠지면 그 stabilizer의 결과가 없고, 겹치면 detector가 엉뚱한 짝을 비교한다.
    """
    code = build_directional_code("NESEN", 2, 2)
    checks = code.HX.shape[0] + code.HZ.shape[0]
    schedule, _, _ = walk_schedule(code, "NESEN", rounds=3)
    for moments in schedule:
        resets = [key for reset, _, _ in moments for key, _ in reset]
        measures = [key for _, _, measure in moments for key, _ in measure]
        assert len(resets) == len(set(resets)) == checks == 16
        assert len(measures) == len(set(measures)) == checks


def test_a_reset_site_is_idle_that_moment_and_used_the_next():
    """"qubits whose first interaction occurs only in a later layer are reset only
    immediately before use."

    켠 자리는 그 moment에 gate가 없어야 하고(그래야 reset이 gate와 나란히 들어간다)
    바로 다음 moment에 쓰여야 한다(그래야 일찍 켜서 오류를 줍지 않는다).
    """
    code = build_directional_code("NESEN", 2, 2)
    schedule, _, _ = walk_schedule(code, "NESEN", rounds=2)
    for moments in schedule:
        for moment, (reset, _, _) in enumerate(moments):
            for _, site in reset:
                assert site not in gate_sites(moments[moment])
                assert site in gate_sites(moments[moment + 1])


def test_a_measured_site_is_used_the_moment_before_and_idle_after():
    """읽는 자리는 직전 moment까지 일하고 그 moment에는 비어 있다."""
    code = build_directional_code("NESEN", 2, 2)
    schedule, _, _ = walk_schedule(code, "NESEN", rounds=2)
    for moments in schedule:
        for moment, (_, _, measure) in enumerate(moments):
            for _, site in measure:
                assert site in gate_sites(moments[moment - 1])
                assert site not in gate_sites(moments[moment])


def test_an_even_number_of_rounds_returns_the_data_home():
    """word 다음 역 word면 data가 제자리로 온다. 홀수로 끝나면 안 온다 --
    마지막 readout을 그 시점에 그 열을 들고 있는 qubit에 걸어야 한다."""
    code = build_directional_code("NESEN", 2, 2)
    home = {col: hardware_site(edge) for col, edge in enumerate(code.qubits)}
    _, _, after_two = walk_schedule(code, "NESEN", rounds=2)
    _, _, after_one = walk_schedule(code, "NESEN", rounds=1)
    assert after_two == home
    assert after_one != home


# --- memory-Z 회로 ----------------------------------------------------------

def figure4_circuit(rounds):
    code = build_directional_code("NESEN", 2, 2)
    return code, walk_memory_z_base(code, "NESEN", rounds)


def test_the_noiseless_circuit_is_silent():
    """detector도 observable도 안 울린다.

    스케줄과 detector 정의가 서로 맞는다는 뜻이고, 동시에 창 기반 reset/측정이 재는
    stabilizer를 바꾸지 않았다는 확인이다 -- 창을 잘못 잡아 아직 안 켠 check가 data를
    먹거나 이미 읽은 check가 더 먹으면 여기서 운다.
    """
    _, circuit = figure4_circuit(rounds=3)
    detections, observables = circuit.compile_detector_sampler().sample(
        64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()


def test_the_detectors_are_deterministic():
    """무잡음 회로의 detector가 결정적이지 않으면 stim이 DEM을 못 만들고 터진다."""
    _, circuit = figure4_circuit(rounds=3)
    assert circuit.detector_error_model().num_detectors == circuit.num_detectors


def test_the_detector_and_observable_counts():
    """Z는 라운드마다 하나씩에 마지막 재구성까지 ``(rounds+1)·mz``, X는 두 번째
    라운드부터 ``(rounds-1)·mx``."""
    code, circuit = figure4_circuit(rounds=3)
    mx, mz = code.HX.shape[0], code.HZ.shape[0]
    assert circuit.num_detectors == 4 * mz + 2 * mx
    assert circuit.num_observables == code.k == 2


def test_an_odd_number_of_rounds_still_reads_the_data():
    """홀수 라운드로 끝나면 data가 제자리로 안 온다. 마지막 ``MR``을 그 시점에 그 열을
    들고 있는 물리 qubit에 걸어야 하고, 아니면 여기서 운다."""
    _, circuit = figure4_circuit(rounds=1)
    assert not circuit.compile_detector_sampler().sample(64).any()


def test_the_round_is_w_plus_two_moments_deep():
    """"one syndrome-extraction round has depth w+2, including check-qubit
    preparation and measurement." """
    _, circuit = figure4_circuit(rounds=1)
    ticks = sum(1 for instruction in circuit.flattened()
                if instruction.name == "TICK")
    assert ticks == 5 + 2


def test_nothing_is_reset_long_before_it_is_used():
    """"qubits whose first interaction occurs only in a later layer are reset only
    immediately before use."

    data를 빼면(그건 실험의 초기 상태라 맨 앞이다) 어떤 qubit도 자기 첫 2q gate보다
    한 moment 넘게 앞서 켜지지 않아야 한다.
    """
    _, circuit = figure4_circuit(rounds=2)
    first_reset, first_gate, data, moment = {}, {}, [], 0
    for instruction in circuit.flattened():
        if instruction.name == "TICK":
            moment += 1
            continue
        targets = [target.value for target in instruction.targets_copy()]
        if instruction.name in ("R", "RX"):
            for qubit in targets:
                first_reset.setdefault(qubit, moment)
        elif instruction.name in ("CXSWAP", "SWAP"):
            for qubit in targets:
                first_gate.setdefault(qubit, moment)
        elif instruction.name == "MR":
            data = targets                     # 마지막 MR이 data readout이다
    for qubit, when in first_reset.items():
        if qubit in data or qubit not in first_gate:
            continue
        assert first_gate[qubit] - when <= 1


def test_every_two_qubit_gate_in_the_circuit_is_nearest_neighbour():
    """논문 제목의 주장이 최종 산출물에서도 성립하는지 회로를 직접 파싱해서 본다.
    좌표가 하드웨어 격자의 절반이라 이웃은 0.5만큼 떨어져 있다."""
    _, circuit = figure4_circuit(rounds=2)
    coords = circuit.get_final_qubit_coordinates()
    for instruction in circuit.flattened():
        if instruction.name not in ("CXSWAP", "SWAP"):
            continue
        targets = [target.value for target in instruction.targets_copy()]
        for first, second in zip(targets[::2], targets[1::2]):
            (first_x, first_y) = coords[first]
            (second_x, second_y) = coords[second]
            assert abs(first_x - second_x) + abs(first_y - second_y) == 0.5


def test_routing_that_never_gates_is_left_out():
    """gate에 한 번도 안 나오는 바깥 테두리 routing은 회로에 넣지 않는다.
    detector에 영향이 없고, 넣으면 qubit 수만 부풀린다."""
    code, circuit = figure4_circuit(rounds=2)
    assert circuit.num_qubits < len(walk_layout(code, "NESEN"))


@pytest.mark.parametrize("word,L1,L2", [("NESEN", 2, 2), ("N2ESEN2", 4, 4)])
def test_the_pruned_layout_gives_the_same_circuit(word, L1, L2):
    """줄인 배치로도 재는 것이 같다 -- qubit만 줄어든다.

    "The optimisation preserves the measured stabiliser supports, detectors and
    logical observables." 그래서 detector와 observable 수는 그대로여야 하고, qubit은
    줄기만 해야 한다 (실측: `NESEN` 2x2에서 54 -> 51, `N2ESEN2` 4x4에서 172 -> 164).
    """
    code = build_directional_code(word, L1, L2)
    plain = walk_memory_z_base(code, word, 3, walk_layout(code, word))
    pruned = walk_memory_z_base(code, word, 3, *route_window_shortening(code, word))
    assert (pruned.num_detectors, pruned.num_observables) \
        == (plain.num_detectors, plain.num_observables)
    assert pruned.num_qubits < plain.num_qubits
    detections, observables = pruned.compile_detector_sampler().sample(
        64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()


@pytest.mark.parametrize("pruned", [False, True])
def test_a_check_is_reborn_where_it_was_read(pruned):
    """라운드마다 읽은 자리가 다음 라운드에 켜는 자리다.

    창이 잘린 check는 매 라운드 다시 태어나는데, 죽은 자리와 태어날 자리가 두 곳을
    왕복한다: 짝수 라운드는 ``A + S_first``, 홀수 라운드는 ``A + S_{last+1}``. 창을
    다 채우는 check는 두 자리가 같아 저절로 성립한다. 어긋나면 이미 읽힌 자리를 두고
    엉뚱한 곳에 reset을 거는 것이다.
    """
    code = build_directional_code("NESEN", 2, 2)
    layout, shifts = (route_window_shortening(code, "NESEN") if pruned
                      else (walk_layout(code, "NESEN"), {}))
    schedule, _, _ = walk_schedule(code, "NESEN", 3, layout, shifts)
    for earlier, later in zip(schedule, schedule[1:]):
        read = {key: site for _, _, measure in earlier for key, site in measure}
        born = {key: site for reset, _, _ in later for key, site in reset}
        assert read == born


def test_every_check_is_reset_and_measured_once_per_round_when_pruned():
    """줄인 배치에서도 라운드마다 모든 check가 한 번 켜지고 한 번 읽힌다."""
    code = build_directional_code("NESEN", 2, 2)
    layout, shifts = route_window_shortening(code, "NESEN")
    schedule, _, _ = walk_schedule(code, "NESEN", 3, layout, shifts)
    for moments in schedule:
        resets = [key for reset, _, _ in moments for key, _ in reset]
        measures = [key for _, _, measure in moments for key, _ in measure]
        assert len(resets) == len(set(resets)) == 16
        assert len(measures) == len(set(measures)) == 16


def test_noise_reaches_the_detectors():
    """SI1000을 walk 회로에 주입할 수 있고, detector가 실제로 운다.

    ``noise_model``의 2q Clifford 목록에 ``CXSWAP``/``SWAP``이 없으면 여기서
    ``NotImplementedError``로 터진다 -- 잡음 축 전체가 그 한 줄에 걸려 있다.
    """
    code = build_directional_code("NESEN", 2, 2)
    noisy = NoiseModel.SI1000(0.01).noisy_circuit(
        walk_memory_z_base(code, "NESEN", rounds=3))
    assert noisy.compile_detector_sampler(seed=0).sample(64).any()


@pytest.mark.slow
def test_the_walk_schedule_does_not_halve_the_circuit_distance():
    """스케줄이 나쁘면 hook 오류로 회로 거리가 코드 거리 아래로 떨어진다. 그런데도
    무잡음 회로는 조용하므로, 이건 따로 봐야만 드러난다.

    Figure 4의 코드는 memory-Z가 막는 쪽(X 오류)의 거리가 ``dx=3``이라, 가장 가벼운
    검출 불가 logical 오류의 무게가 3이어야 한다 -- 반토막(2)이 아니다.
    """
    code = build_directional_code("NESEN", 2, 2)
    noisy = NoiseModel.SI1000(0.001).noisy_circuit(
        walk_memory_z_base(code, "NESEN", rounds=3))
    errors = noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=True,
        canonicalize_circuit_errors=True)
    assert len(errors) == 3
