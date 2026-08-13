"""Nearest-neighbour syndrome extraction: 걸음 한 층과 check 출발점."""
import itertools

import numpy as np
import pytest

from qec_tile.directional import (DIRECTIONS, build_directional_code,
                                  hardware_site, parse_directional_word,
                                  satisfies_parity_condition, walk_edges)
from qec_tile.noise_model import NoiseModel
from qec_tile.walk import (CHECK_X, CHECK_Z, DATA, FILLER, ROUTING, Qubit,
                           check_starts,
                           flow_step, walk_crossings, walk_layout,
                           walk_memory_z_base, walk_round)


def lattice(step, xs=range(4), ys=range(4)) -> dict:
    """하드웨어 격자 조각을 규칙대로 채운다: 짝수합 site가 check, 홀수합이 data.

    ``step``으로 걸어나갈 자리가 없는 walker는 아예 놓지 않는다 -- ``walk_layout``이
    질 의무(궤적이 통째로 들어오는 자리에만 걷는 qubit을 둔다)의 축소판이다.
    """
    qubit_at = {}
    for x in xs:
        for y in ys:
            if x % 2 == 0 and y % 2 == 0:
                role = CHECK_X
            elif x % 2 and y % 2:
                role = CHECK_Z
            else:
                role = DATA
            qubit_at[(x, y)] = Qubit(role, len(qubit_at), (x, y))
    return {site: qubit for site, qubit in qubit_at.items()
            if qubit.role == DATA
            or (site[0] + step[0], site[1] + step[1]) in qubit_at}


def test_x_check_crosses_the_edge_it_walks_over():
    """vertex ``(0,0)``의 X check가 N 한 걸음에 ``(0,1)``의 data와 CXSWAP한다.

    ``(0,1)``은 코드 격자의 ``V(0,0)`` -- 첫 글자가 건너는 edge다. control 순서가
    뒤집히면 ``|+>`` ancilla가 아니라 data가 control이 되어 X stabilizer가 아닌 것을
    재게 된다.
    """
    qubit_at = lattice(DIRECTIONS["N"])
    gates = flow_step(qubit_at, DIRECTIONS["N"])
    assert ("CXSWAP", (0, 0), (0, 1)) in gates
    assert qubit_at[(0, 1)].home == (0, 0)      # check가 앞으로 갔고
    assert qubit_at[(0, 0)].home == (0, 1)      # data가 뒤로 밀렸다


def test_z_check_walks_the_same_way_but_crosses_the_other_orientation():
    """같은 층에서 face 중심 ``(1,1)``의 Z check도 ``+N``으로 간다.

    건너는 ``(1,2)``는 H edge라, 방향이 같은데도 X와 반대 orientation을 먹는다 --
    (T2)의 H<->V 교환이 좌표 parity에서 공짜로 나온다. control은 data 쪽이다.
    """
    gates = flow_step(lattice(DIRECTIONS["N"]), DIRECTIONS["N"])
    assert ("CXSWAP", (1, 2), (1, 1)) in gates


@pytest.mark.parametrize("letter", ["N", "E", "S", "W"])
def test_every_letter_moves_the_walking_qubits_one_step(letter):
    """N/S/E/W 네 걸음의 계약. 음의 방향으로도 나갈 자리를 주려고 격자를 넓게 잡는다."""
    step = DIRECTIONS[letter]
    qubit_at = lattice(step, range(-2, 3), range(-2, 3))
    target = step                               # ``(0,0)``의 X check가 갈 자리
    flow_step(qubit_at, step)
    assert qubit_at[target].home == (0, 0)
    assert qubit_at[(0, 0)].home == target


def test_check_swaps_through_a_routing_qubit_instead_of_entangling():
    """X check가 routing(``|0>``)을 만나면 CXSWAP이 아니라 SWAP이어야 한다.

    CXSWAP이면 CX 부분이 ``|+>`` ancilla와 ``|0>``을 얽어버려 뒤이은 X 측정이
    무작위가 된다 -- 잡음이 없어도 detector가 울리는 실패 모드다.
    """
    qubit_at = lattice(DIRECTIONS["N"])
    qubit_at[(0, 1)] = Qubit(FILLER, 0, (0, 1))
    gates = flow_step(qubit_at, DIRECTIONS["N"])
    assert ("SWAP", (0, 0), (0, 1)) in gates


def test_routing_in_a_check_seat_pushes_data_along():
    """check 자리(짝수합)를 메운 routing도 같이 걸으며 data를 밀어준다.

    빠뜨리면 check 뒤로 data가 흘러오지 않아 두 번째 글자부터 엉뚱한 qubit을 먹는다.
    """
    qubit_at = lattice(DIRECTIONS["N"])
    qubit_at[(2, 2)] = Qubit(ROUTING, 0, (2, 2))
    gates = flow_step(qubit_at, DIRECTIONS["N"])
    assert ("SWAP", (2, 2), (2, 3)) in gates
    assert qubit_at[(2, 3)].role == ROUTING


def test_a_layer_is_a_perfect_matching():
    """한 층에서 모든 site가 정확히 한 번씩 gate에 들어간다.

    flow가 성립하는 근거인 sublattice 논증을 실측으로 고정한다. 빠지는 것은 밀어줄
    이가 격자 밖인 홀수합 site뿐이다 -- 6x6에서 아래 끝 ``(1,0),(3,0),(5,0)``.
    (맨 윗줄 walker는 헬퍼가 애초에 놓지 않는다.)
    """
    qubit_at = lattice(DIRECTIONS["N"], range(6), range(6))
    gates = flow_step(qubit_at, DIRECTIONS["N"])
    touched = [site for _, a, b in gates for site in (a, b)]
    assert len(touched) == len(set(touched))
    assert set(qubit_at) - set(touched) == {(1, 0), (3, 0), (5, 0)}


def test_a_step_and_its_reverse_restore_the_layout():
    """N 다음 S면 배치가 원래대로 -- 라운드 교대(D 다음 D^-1)로 layout이 복원된다는
    논문 문장의 최소 단위다. 여기서 보는 것은 배치뿐이고, CX가 두 번 걸리는 것은
    라운드마다 stabilizer를 다시 재기 때문이라 정상이다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(DATA, 0, (0, 1))}
    before = dict(qubit_at)
    flow_step(qubit_at, DIRECTIONS["N"])
    flow_step(qubit_at, DIRECTIONS["S"])
    assert qubit_at == before


def test_a_shuttle_passing_a_filler_emits_no_gate():
    """Algorithm 1의 16-19행: ``R``은 ``D``를 만날 때만 gate를 낸다.

    둘 다 ``|0>``이라 교환이 물리적으로 항등이기 때문이다. 그래도 자리는 바꾼다 --
    셔틀이 한 층 쉬면 그만 뒤처져 다음 층에 남의 자리를 밟는다.
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0)),    # 셔틀
                (0, 1): Qubit(FILLER, 1, (0, 1))}     # 디딤돌
    assert flow_step(qubit_at, DIRECTIONS["N"]) == []
    assert qubit_at[(0, 1)].role == ROUTING           # 자리는 바뀐다


def test_a_check_that_cannot_step_is_rejected():
    """check가 갈 자리가 없으면 stabilizer를 덜 재게 된다 -- 배치가 모자란 것이다."""
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0))}
    with pytest.raises(ValueError, match="cannot step"):
        flow_step(qubit_at, DIRECTIONS["N"])


def test_a_shuttle_with_nowhere_to_go_just_waits():
    """셔틀은 갈 자리가 없으면 그냥 선다 -- Algorithm 1이 ``end if``로 비워둔 분기다.

    ``walk_layout``이 만든 배치에서는 일어나지 않는다. 셔틀의 궤적을 통째로 깔아주기
    때문이고, 논문 코드 네 개로 두 라운드를 돌려 이 분기가 한 번도 안 타는 것을
    확인했다. check는 반대로 못 가면 오류다 -- stabilizer를 덜 재게 되므로.
    """
    qubit_at = {(0, 0): Qubit(ROUTING, 0, (0, 0))}
    assert flow_step(qubit_at, DIRECTIONS["N"]) == []
    assert qubit_at == {(0, 0): Qubit(ROUTING, 0, (0, 0))}


def test_a_check_facing_a_check_is_rejected():
    """check가 check를 미는 일은 parity상 불가능하다 -- 출발점 기하가 깨졌다는 뜻이다."""
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(CHECK_Z, 0, (1, 1)),   # 있을 수 없는 자리
                (0, 2): Qubit(DATA, 0, (1, 2))}      # 그 check가 갈 자리
    with pytest.raises(ValueError, match="two checks"):
        flow_step(qubit_at, DIRECTIONS["N"])


def test_the_same_site_cannot_take_two_gates():
    """멈췄던 walker가 나중에 다시 걸으면 한 site가 두 gate에 들어갈 수 있다.

    stim의 moment 규칙 위반이라 회로가 되기 전에 잡아야 한다. 여기서는 check가
    ``(0,1)``의 routing을 지나가는 동시에 그 routing이 ``(0,2)``의 data를 민다.
    """
    qubit_at = {(0, 0): Qubit(CHECK_X, 0, (0, 0)),
                (0, 1): Qubit(ROUTING, 0, (2, 0)),   # 짝수합 home -> 걷는다
                (0, 2): Qubit(DATA, 0, (1, 2))}
    with pytest.raises(ValueError, match="two gates"):
        flow_step(qubit_at, DIRECTIONS["N"])


# --- check 출발점 ----------------------------------------------------------

def assert_the_walk_reaches_the_support(code, word):
    """모든 check에 대해 ``{출발점 + 건너는 edge}`` 중 data인 것 == 그 행이 짚는 열.

    출발점, 걸음 부호, ``code.B``, 절단과 pruning이 동시에 맞아야만 성립한다. 오른쪽은
    ``tile.py``의 조립에서, 왼쪽은 흐름의 기하에서 나오므로 서로 독립이다.
    """
    crossings = walk_crossings(parse_directional_word(word))
    col_of = {hardware_site(edge): col for col, edge in enumerate(code.qubits)}
    x_starts, z_starts = check_starts(code, word)
    for checks, starts in ((code.HX, x_starts), (code.HZ, z_starts)):
        for row, (start_x, start_y) in zip(checks, starts):
            reached = {col_of[site]
                       for site in ((start_x + cx, start_y + cy)
                                    for cx, cy in crossings)
                       if site in col_of}
            assert reached == set(np.flatnonzero(row))


def test_x_check_starts_at_its_anchor_vertex():
    """``N2ESEN2``는 walk가 원점에 붙어 있어(``min = 0``) 보정이 없다."""
    code = build_directional_code("N2ESEN2", 4, 4)
    x_starts, _ = check_starts(code, "N2ESEN2")
    assert x_starts == [(2 * i, 2 * j) for i, j in code.x_anchors]


def test_starts_sit_on_the_right_sublattice():
    """X는 vertex ``(even, even)``, Z는 face 중심 ``(odd, odd)``에서 출발한다.
    아니면 애초에 check가 앉을 자리가 아니고, 걸을 때마다 data와 어긋난다."""
    code = build_directional_code("NESEN", 4, 4)
    x_starts, z_starts = check_starts(code, "NESEN")
    assert all(x % 2 == 0 and y % 2 == 0 for x, y in x_starts)
    assert all(x % 2 and y % 2 for x, y in z_starts)


def test_the_walk_reaches_exactly_the_check_support():
    """핵심 계약. ``NESEN``은 논문 Figure 5의 word이고 4x4에서 [[50,2,...]]다."""
    assert_the_walk_reaches_the_support(build_directional_code("NESEN", 4, 4),
                                        "NESEN")


def test_a_larger_box_moves_the_z_start_but_not_the_support():
    """``B``는 자유 파라미터라 더 큰 box로도 같은 코드가 나온다. Z 출발점은 ``B``와
    함께 움직이지만 닿는 support는 그대로다 -- 최소 box를 쓰고 ``code.B``를 안 보면
    여기서 깨진다."""
    assert_the_walk_reaches_the_support(
        build_directional_code("NESEN", 4, 4, 4), "NESEN")


def test_a_word_whose_z_tile_is_not_reachable_is_rejected():
    """``NESE``는 Definition 1의 parity 조건을 어긴다. 코드 자체는 지어지지만 Z check가
    같은 word로는 자기 tile에 닿지 못하므로, 회로를 지으려 할 때 걸려야 한다."""
    code = build_directional_code("NESE", 4, 4)
    with pytest.raises(ValueError, match="parity"):
        check_starts(code, "NESE")


def test_the_parity_condition_is_what_the_z_walk_needs():
    """Definition 1의 parity 조건 <=> walk의 edge 집합이 ``S_w``를 중심으로 점대칭.

    앞은 논문이 인쇄한 조건이고 뒤는 Z 출발점 공식이 요구하는 것인데, 갈라지는 word가
    생기면 ``check_starts``의 전제가 무너진다. 길이 6까지 전수 -- 8까지 돌려도 한쪽만
    만족하는 string은 없었다 (340개가 둘 다, 11176개가 둘 다 아님).
    """
    for length in range(2, 7):
        for letters in itertools.product("NESW", repeat=length):
            steps = [DIRECTIONS[letter] for letter in letters]
            edges = walk_edges(steps)
            if len(set(edges)) != len(edges):     # string이 아니다
                continue
            crossings = walk_crossings(steps)
            end_x = sum(step_x for step_x, _ in steps)
            end_y = sum(step_y for _, step_y in steps)
            symmetric = sorted((2 * end_x - cx, 2 * end_y - cy)
                               for cx, cy in crossings) == sorted(crossings)
            assert satisfies_parity_condition(edges) == symmetric


# --- 배치 ------------------------------------------------------------------

def small_layout():
    """``NESEN`` 4x4 -- 논문 Figure 5의 word로 지은 가장 작은 코드 (n=50, k=2)."""
    code = build_directional_code("NESEN", 4, 4)
    return code, walk_layout(code, "NESEN")


def test_data_qubits_sit_on_their_edge_midpoints():
    """모든 열이 정확히 한 번, 자기 edge 중점에 data로 앉는다."""
    code, qubit_at = small_layout()
    seats = {qubit.index: site for site, qubit in qubit_at.items()
             if qubit.role == DATA}
    assert len(seats) == code.n
    assert all(seats[col] == hardware_site(edge)
               for col, edge in enumerate(code.qubits))


def test_checks_sit_at_their_starts():
    """역할과 index가 ``HX``/``HZ`` 행과 정렬된다 -- 어긋나면 detector가 엉뚱한 행을
    비교하게 되고, 그건 무잡음 회로에서도 안 걸린다."""
    code, qubit_at = small_layout()
    x_starts, z_starts = check_starts(code, "NESEN")
    for row, site in enumerate(x_starts):
        assert qubit_at[site] == Qubit(CHECK_X, row, site)
    for row, site in enumerate(z_starts):
        assert qubit_at[site] == Qubit(CHECK_Z, row, site)


def test_roles_match_the_sublattice():
    """짝수합 자리에 data가 없고 홀수합 자리에 check가 없다.

    셔틀과 디딤돌을 가르는 것이 이 parity라, 어긋나면 흐름이 성립하지 않는다.
    """
    _, qubit_at = small_layout()
    for site, qubit in qubit_at.items():
        if qubit.role == DATA:
            assert sum(site) % 2 == 1
        elif qubit.role in (CHECK_X, CHECK_Z):
            assert sum(site) % 2 == 0


def test_every_check_is_present():
    """하나라도 빠지면 그 stabilizer를 아예 안 재는 회로가 된다."""
    code, qubit_at = small_layout()
    roles = [qubit.role for qubit in qubit_at.values()]
    assert roles.count(CHECK_X) == code.HX.shape[0]
    assert roles.count(CHECK_Z) == code.HZ.shape[0]


def test_a_full_round_runs_and_no_check_ever_stalls():
    """배치에 ``flow_step``을 ``w``번. 멈춘 check가 있으면 그 다음 글자부터 엉뚱한
    qubit을 먹는데, 끝난 자리가 정확히 ``출발점 + S_w``면 한 번도 안 멈춘 것이다."""
    code, qubit_at = small_layout()
    steps = parse_directional_word("NESEN")
    for layer, step in enumerate(steps):
        flow_step(qubit_at, step, layer)
    end_x = sum(step_x for step_x, _ in steps)
    end_y = sum(step_y for _, step_y in steps)
    checks = [(site, qubit) for site, qubit in qubit_at.items()
              if qubit.role in (CHECK_X, CHECK_Z)]
    assert len(checks) == code.HX.shape[0] + code.HZ.shape[0]
    for site, qubit in checks:
        assert site == (qubit.home[0] + end_x, qubit.home[1] + end_y)


def test_every_data_qubit_has_someone_to_push_it():
    """data는 스스로 못 움직인다. 층 ``k``마다 ``v - h_k`` 자리의 walker가 밀어줘야
    제때 check 앞에 온다.

    이게 배치 규칙이 존재하는 이유다. stabilizer의 상자만 채우면 코드 뒤쪽 자리가
    비어서 경계 data가 배달되지 않고, check가 열 하나를 덜 먹는다 -- 2x2에서 제자리가
    ``(4,3)``인 열 12가 ``(3,1)``이 비어 층 2에서 멈췄다. 상자를 ``2B x 2B``로 키워도
    같은 자리가 빈다.
    """
    code, qubit_at = small_layout()
    crossings = walk_crossings(parse_directional_word("NESEN"))
    for site, qubit in list(qubit_at.items()):
        if qubit.role != DATA:
            continue
        for cross_x, cross_y in crossings:
            assert (site[0] - cross_x, site[1] - cross_y) in qubit_at


def test_routing_grows_with_the_boundary_not_the_area():
    """routing은 둘레만큼만 자란다 -- 논문 Appendix C의 ``n_r = O(sqrt n)``.

    실측하면 ``routing / sqrt(n)``이 2x2에서 12x12까지 10.6~11.2로 거의 상수다.
    상자를 채우는 방식은 이 성질을 잃는다.
    """
    ratios = []
    for size in (2, 4, 6):
        code = build_directional_code("NESEN", size, size)
        routing = sum(1 for qubit in walk_layout(code, "NESEN").values()
                      if qubit.role == ROUTING)
        ratios.append(routing / code.n ** 0.5)
    assert max(ratios) / min(ratios) < 1.3


# --- 한 라운드 --------------------------------------------------------------

def figure4_layout():
    """논문 Figure 4가 그리는 코드: ``NESEN`` 2x2 = ``[[18,2,dx=3,dz=2]]``.

    같은 word로 ``[[18,2]]``를 주는 배치가 1x3에도 있지만 그쪽은 ``dx=6``이라,
    caption의 ``dx=3, dz=2``가 2x2를 집어준다.
    """
    code = build_directional_code("NESEN", 2, 2)
    return code, walk_layout(code, "NESEN")


def test_the_round_measures_exactly_the_stabilisers():
    """``walk_round``가 안에서 ``HX``/``HZ`` 행과 대조한다.

    여기서는 다른 각도로 한 번 더 건다: CXSWAP 총수가 support 무게의 총합과 같아야
    한다. 어느 check가 남의 qubit을 먹었으면 안쪽 대조가, 아예 덜 먹었으면 이 개수가
    잡는다.
    """
    code, qubit_at = figure4_layout()
    layers, _ = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    cxswaps = sum(1 for layer in layers for name, _, _ in layer
                  if name == "CXSWAP")
    assert cxswaps == int(code.HX.sum() + code.HZ.sum())


def test_the_round_has_one_layer_per_letter():
    """한 글자가 한 층 -- 라운드 깊이가 ``w``이고 코드 크기와 무관하다는 주장."""
    code, qubit_at = figure4_layout()
    steps = parse_directional_word("NESEN")
    layers, _ = walk_round(code, qubit_at, steps)
    assert len(layers) == len(steps) == 5


def test_every_gate_is_nearest_neighbour():
    """논문 제목이 주장하는 것. 모든 2q gate가 격자에서 한 칸 떨어진 두 자리에 걸린다."""
    code, qubit_at = figure4_layout()
    layers, _ = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    for layer in layers:
        for _, (first_x, first_y), (second_x, second_y) in layer:
            assert abs(first_x - second_x) + abs(first_y - second_y) == 1


def test_the_window_is_where_the_check_actually_touches_data():
    """창 = data를 처음 만나는 층부터 마지막으로 만나는 층까지 (0부터 셈).

    회로는 여기에 맞춰 reset과 측정을 놓는다. Figure 4 코드에서 X check는 여섯 개가
    전부 다섯 층을 꽉 채우지만, Z check는 경계에서 tile이 잘려 창이 짧다 -- 4~5번째
    층에서만 일하는 것 둘, 1~2번째 층에서 끝나는 것 둘.
    """
    code, qubit_at = figure4_layout()
    _, windows = walk_round(code, qubit_at, parse_directional_word("NESEN"))
    assert len(windows) == code.HX.shape[0] + code.HZ.shape[0]
    assert all(windows[(CHECK_X, row)] == (0, 4)
               for row in range(code.HX.shape[0]))
    assert sorted(windows[(CHECK_Z, row)]
                  for row in range(code.HZ.shape[0])) == [
        (0, 1), (0, 1), (0, 3), (0, 3), (0, 4),
        (0, 4), (1, 4), (1, 4), (3, 4), (3, 4)]


def test_the_inverse_word_restores_the_layout():
    """"Consecutive rounds are alternated between the word D and the inverse word,
    so that the physical layout is restored without introducing long-range
    operations." 역 word는 순서를 뒤집고 방향을 뒤집은 것이다."""
    def fingerprint(layout):                     # routing끼리는 구별되지 않는다
        return {site: (DATA if qubit.role == DATA else
                       qubit.role if qubit.role in (CHECK_X, CHECK_Z) else "zero",
                       qubit.index if qubit.role in (CHECK_X, CHECK_Z, DATA)
                       else None)
                for site, qubit in layout.items()}

    code, qubit_at = figure4_layout()
    before = fingerprint(qubit_at)
    steps = parse_directional_word("NESEN")
    walk_round(code, qubit_at, steps)
    assert fingerprint(qubit_at) != before       # 한 라운드 뒤에는 옮겨져 있고
    walk_round(code, qubit_at, [(-x, -y) for x, y in reversed(steps)],
               phase=len(steps))
    assert fingerprint(qubit_at) == before       # 역 word가 되돌린다


def test_a_check_carrying_the_wrong_row_is_caught():
    """안쪽 대조에 이빨이 있는지. 0번 X check에 1번 행을 붙이면 먹는 열이 달라진다."""
    code, qubit_at = figure4_layout()
    x_starts, _ = check_starts(code, "NESEN")
    qubit_at[x_starts[0]] = Qubit(CHECK_X, 1, x_starts[0])
    with pytest.raises(ValueError, match="support"):
        walk_round(code, qubit_at, parse_directional_word("NESEN"))


def test_the_larger_layout_measures_its_stabilisers_too():
    """4x4(n=50)에서도 같은 계약. 경계 tile이 훨씬 많은 배치를 한 번 걸어둔다."""
    code = build_directional_code("NESEN", 4, 4)
    walk_round(code, walk_layout(code, "NESEN"),
               parse_directional_word("NESEN"))


# --- memory-Z 회로 ----------------------------------------------------------

def figure4_circuit(rounds):
    code = build_directional_code("NESEN", 2, 2)
    return code, walk_memory_z_base(code, "NESEN", rounds)


def test_the_noiseless_circuit_is_silent():
    """detector도 observable도 안 울린다.

    스케줄과 detector 정의가 서로 맞는다는 뜻이고, 동시에 창 기반 reset/측정이
    재는 stabilizer를 바꾸지 않았다는 확인이다 -- 창을 잘못 잡아 아직 안 켠 check가
    data를 먹거나 이미 읽은 check가 더 먹으면 여기서 울린다.
    """
    _, circuit = figure4_circuit(rounds=3)
    detections, observables = circuit.compile_detector_sampler().sample(
        64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()


def test_the_detector_and_observable_counts():
    """Z는 라운드마다 하나씩에 마지막 재구성까지 ``(rounds+1)·mz``, X는 두 번째
    라운드부터 ``(rounds-1)·mx``."""
    code, circuit = figure4_circuit(rounds=3)
    mx, mz = code.HX.shape[0], code.HZ.shape[0]
    assert circuit.num_detectors == 4 * mz + 2 * mx
    assert circuit.num_observables == code.k


def test_the_round_is_w_plus_two_moments_deep():
    """"one syndrome-extraction round has depth w+2, including check-qubit
    preparation and measurement."

    창이 늦게 열리는 check의 reset과 일찍 닫히는 check의 측정이 **gate layer와 같은
    moment**에 들어가야 이 숫자가 나온다. 자는 check의 SWAP을 내면 그 자리가 묶여서
    reset이 자기 moment를 하나 차지하고 깊이가 늘어난다.
    """
    _, circuit = figure4_circuit(rounds=1)
    ticks = sum(1 for instruction in circuit.flattened()
                if instruction.name == "TICK")
    assert ticks == 5 + 2


def test_nothing_is_reset_long_before_it_is_used():
    """"qubits whose first interaction occurs only in a later layer are reset only
    immediately before use."

    data를 빼면(그건 실험의 초기 상태라 맨 앞이다) 어떤 qubit도 자기 첫 2q gate보다
    한 moment 넘게 앞서 켜지지 않아야 한다. 일찍 켤수록 그 ``|0>``이 오류를 주울
    구간만 길어진다.
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


def test_the_detectors_are_deterministic():
    """무잡음 회로의 detector가 결정적이지 않으면 stim이 DEM을 못 만들고 터진다."""
    code, circuit = figure4_circuit(rounds=3)
    assert circuit.detector_error_model().num_detectors == circuit.num_detectors


def test_an_odd_number_of_rounds_still_reads_the_data():
    """홀수 라운드로 끝나면 data가 제자리로 안 돌아온다. 마지막 ``M``을 그 시점에 그
    열을 들고 있는 물리 qubit에 걸어야 하고, 아니면 여기서 울린다."""
    _, circuit = figure4_circuit(rounds=1)
    assert not circuit.compile_detector_sampler().sample(64).any()


def test_every_two_qubit_gate_in_the_circuit_is_nearest_neighbour():
    """논문 제목의 주장이 최종 산출물에서도 성립하는지, 회로를 직접 파싱해서 본다.
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


def test_rounds_must_be_positive():
    code = build_directional_code("NESEN", 2, 2)
    with pytest.raises(ValueError, match="rounds"):
        walk_memory_z_base(code, "NESEN", rounds=0)


def test_noise_reaches_the_detectors():
    """SI1000을 walk 회로에 주입할 수 있고, detector가 실제로 울린다.

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

    Figure 4의 코드는 memory-Z가 막는 쪽(X 오류)의 거리가 ``dx=3``이고, 실측으로
    가장 가벼운 검출 불가 logical 오류의 무게가 정확히 3이었다 -- 반토막(2)이 아니다.
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
