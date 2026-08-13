"""Directional tile code의 nearest-neighbour syndrome extraction.

Gu, Noszko, Steffan, Eberhardt, Roffe, Eisert & Koutsioumpas, "Nearest-neighbour
gates are all you need: High-rate quantum low-density parity-check codes on a
planar grid" (arXiv:2606.19482)의 Algorithm 1. word의 한 글자가 gate 한 층이라
라운드 깊이가 ``w``이고 (준비와 측정까지 ``w+2``) 코드 크기와 무관하다.

흐름
----
하드웨어 격자는 코드 격자를 두 배로 늘린 것이고, 좌표합의 홀짝이 역할을 가른다::

    (even, even)  vertex        좌표합 짝수   XX-check
    (odd,  odd )  face 중심     좌표합 짝수   ZZ-check
    (odd,  even)  H edge 중점   좌표합 홀수   data
    (even, odd )  V edge 중점   좌표합 홀수   data

check가 data를 찾아가는 것이 아니라, **좌표합이 짝수인 qubit이 전부 같은 방향으로
한 칸씩 흐르고 data가 그 자리로 밀려온다**. 부분합을 ``S_m = d_1 + ... + d_m``이라
하면 anchor ``A``의 check가 층 ``m``에서 만나는 data의 제자리는
``A + S_m + S_{m-1}``인데, 이것이 정확히 walk의 ``m``번째 edge의 하드웨어 좌표다.
그래서 한 글자에 한 층이면 충분하다.

물리 qubit은 site에 고정이고 (stim의 qubit 인덱스가 그것이다) 그 위를 상태가
흐르므로, gate는 site 쌍으로 나온다.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import stim

from .directional import (hardware_site, parse_directional_word,
                          tile_from_word, walk_edges)
from .tile import z_tile_from_x

Site = tuple[int, int]
Step = tuple[int, int]

# Algorithm 1의 D, C_X, C_Z, R. stim의 "R"(reset), "X"/"Z"(Pauli)와 섞이지 않게
# 풀어 쓴다 -- 이 모듈이 그 instruction들을 그대로 뱉는 곳이다.
DATA = "data"
CHECK_X = "check_x"
CHECK_Z = "check_z"
ROUTING = "routing"

# 검증에 돌리는 라운드 수. 무손실 배치는 두 라운드마다 정확히 복원되지만,
# 줄인 배치는 첫 라운드가 과도기라 정상 궤도가 초기 상태와 다르다. 두 라운드만
# 보면 라운드 2 이후를 한 번도 검증하지 않게 되므로 한 주기를 더 돈다.
ROUNDS_TESTED = 4


@dataclass(frozen=True)
class Qubit:
    """layout을 이루는 qubit 하나. site에 고정된 물리 qubit이 아니라 그 위를 흐른다."""

    role: str          # DATA | CHECK_X | CHECK_Z | ROUTING
    index: int         # DATA는 code.qubits 열, CHECK_*는 HX/HZ 행, ROUTING은 일련번호
    home: Site         # 라운드를 시작할 때 앉아 있던 site


def flow_step(qubit_at: dict[Site, Qubit], step: Step
              ) -> tuple[list[tuple[str, Site, Site]], list[tuple[Qubit, Site]]]:
    """step 한 층. ``(gate 목록, 그 층에 읽힌 check)``를 돌려주고 배치를 갱신한다.

    Algorithm 1의 ``for i = 1..w`` 내부. ``C_X``, ``C_Z``, ``R``이 전부 같은 방향으로
    한 칸 전진하여 그 자리에 있던 qubit과 교환되고, ``D``는 밀리기만 한다. 층 안의
    이동은 동시에 일어나므로 목적지를 다 정한 뒤에 한꺼번에 교환한다.

    만나는 역할이 gate를 정한다. stim의 ``CXSWAP a b``는 control ``a``의 CX 다음
    SWAP이라 첫 자리가 control이다 -- X는 check가, Z는 data가 control이다::

        C_X + D  ->  CXSWAP check data      R + D    ->  SWAP
        C_Z + D  ->  CXSWAP data check      R + 그외 ->  없음 (자리도 그대로)
        C_* + R  ->  SWAP                   C_* + C_* -> 오류

    ``R``은 ``D``를 만날 때만 움직인다 (16-20행). 이것이 겹침을 막는다: 두 check
    사이에 낀 ``R``은 앞이 check라 서 있고, 뒤의 check만 그 자리로 들어온다.

    갈 자리가 없다는 것은 walk가 격자 밖에서 잘렸다는 뜻이다. check는 거기서 할 일이
    끝났으므로 **그 자리에서 읽히고**, ``MR``/``MRX`` 뒤에는 ``|0>``이므로 routing으로
    남는다. routing은 읽을 것이 없으니 그냥 선다 -- Algorithm 1이 16-20행의 ``end if``로
    비워둔 분기다. 지우면 안 된다: 그 자리의 물리 qubit은 여전히 거기 있고, 지우면
    뒤따르던 check가 밟을 자리를 잃어 support를 다 먹기 전에 읽히게 된다.
    """
    gates: list[tuple[str, Site, Site]] = []
    moves: list[tuple[Site, Site]] = []
    measured: list[tuple[Qubit, Site]] = []
    taken: set[Site] = set()          # 한 층에 한 qubit은 gate 하나

    for site in sorted(site for site, qubit in qubit_at.items()
                       if qubit.role != DATA):
        qubit = qubit_at[site]
        target = (site[0] + step[0], site[1] + step[1])
        if target not in qubit_at:
            if qubit.role != ROUTING:
                measured.append((qubit, site))
            continue
        met = qubit_at[target].role
        if qubit.role == ROUTING:
            if met != DATA:
                continue             # R은 D를 만날 때만 움직인다 (16-20행)
            gates.append(("SWAP", site, target))
        elif met == DATA:
            gates.append(("CXSWAP", site, target) if qubit.role == CHECK_X
                         else ("CXSWAP", target, site))
        elif met == ROUTING:
            gates.append(("SWAP", site, target))
        else:
            raise ValueError(f"{site} -> {target}: two checks meet, so the "
                             f"walk geometry is broken")
        if site in taken or target in taken:
            raise ValueError(f"{site} -> {target}: two gates in one layer")
        taken.update((site, target))
        moves.append((site, target))

    for site, target in moves:
        qubit_at[site], qubit_at[target] = qubit_at[target], qubit_at[site]
    for qubit, site in measured:
        qubit_at[site] = Qubit(ROUTING, -1, site)   # -1: 측정 뒤 남은 |0>
    return gates, measured


def partial_sums(steps: list[Step]) -> list[Site]:
    """걸음의 부분합 ``S_0 .. S_w``. walker가 밟는 자리를 출발점 기준으로 준다.

    ``S_0 = (0,0)``도 궤적의 일부다 -- 출발점 자신이 walker가 차지하는 자리다.
    """
    trajectory = [(0, 0)]
    for (step_x, step_y) in steps:
        trajectory.append((trajectory[-1][0] + step_x, trajectory[-1][1] + step_y))
    return trajectory


def reverse_steps(steps: list[Step]) -> list[Step]:
    """역 word의 걸음 -- 순서와 부호를 둘 다 뒤집는다.

    "Consecutive rounds are alternated between the word D and the inverse word,
    so that the physical layout is restored without introducing long-range
    operations." 하나만 뒤집으면 배치가 복원되지 않는다.
    """
    return [(-step_x, -step_y) for step_x, step_y in reversed(steps)]


def walk_crossings(steps: list[Step]) -> list[Site]:
    """walk가 건너는 edge들의 하드웨어 좌표, 출발점 기준. h_m = S_{m-1} + S_m.

    층 m에서 check가 만나는 data의 제자리다 (만나는 위치가 아니다). check는
    층 ``m``을 끝내면 ``start + S_m``에 있고, 제자리가 ``v``인 data는 매 층 한 칸씩
    밀려 층 ``m-1``까지 ``v - S_{m-1}``에 와 있다. 둘이 만나므로::

        v - S_{m-1} = start + S_m   =>   v = start + S_{m-1} + S_m

    ``S_{m-1} + S_m = 2*S_{m-1} + d_m``이라 이것은 vertex ``S_{m-1}``의 하드웨어
    좌표에 건너는 edge의 오프셋을 더한 것, 곧 ``hardware_site(walk_edges(steps)[m])``
    와 같다. 그래서 이 목록이 곧 check가 무엇을 어떤 순서로 먹는지다.
    """
    trajectory = partial_sums(steps)
    return [(before_x + after_x, before_y + after_y)
            for (before_x, before_y), (after_x, after_y)
            in zip(trajectory, trajectory[1:])]


def check_starts(code, word: str) -> tuple[list[Site], list[Site]]:
    """각 check가 라운드를 시작하는 site, ``HX``/``HZ`` 행 순서로.

    tile code가 tile을 anchor에 찍는 것과 같이 **anchor 상대 오프셋**으로 정한다.
    ``2*anchor`` 기준으로 재면 X support의 오프셋이 ``h_m - 2*min``이고, (T2)의 180도
    회전은 하드웨어에서 ``u -> C - u`` 한 줄이다 (``C = (2B-1, 2B-1)``, box의 반대쪽
    모서리). ``min``은 ``tile_from_word``가 tile을 box에 붙일 때 쓴 오프셋이다::

        X:  x_rel = -2*min                  walk의 원점
        Z:  z_rel = C - (x_rel + 2*S_w)     X가 도착하는 자리를 회전시킨 자리

    회전이 walk의 방향을 보존하므로, 같은 word를 같은 방향으로 걸어 회전된 tile을
    덮으려면 시작과 끝이 맞바뀐다.

    ``z_rel``이 맞는지는 Definition 1의 parity 조건(대리조건)이 아니라 필요한 것
    자체를 잰다: ``{z_rel + h_m}``이 ``z_tile_from_x``가 주는 site 집합과 같은지 본다.
    오른쪽은 ``tile.py``가 ``HZ``를 조립할 때 쓰는 그 값이라 위 대수와 독립이고, 더 긴
    word에서 두 조건이 갈라질 때 틀린 출발점이 조용히 통과하지 않는다.
    """
    steps = parse_directional_word(word)
    crossings = walk_crossings(steps)
    edges = walk_edges(steps)
    min_x = min(x for _, x, _ in edges)
    min_y = min(y for _, _, y in edges)
    (end_x, end_y) = partial_sums(steps)[-1]

    corner = 2 * code.B - 1
    x_rel = (-2 * min_x, -2 * min_y)
    z_rel = (corner - x_rel[0] - 2 * end_x,
             corner - x_rel[1] - 2 * end_y)

    x_h, x_v, _ = tile_from_word(word, code.B)
    z_h, z_v = z_tile_from_x(x_h, x_v, code.B)
    z_sites = sorted([hardware_site(("H", x, y)) for x, y in z_h]
                     + [hardware_site(("V", x, y)) for x, y in z_v])
    if sorted((z_rel[0] + cross_x, z_rel[1] + cross_y)
              for cross_x, cross_y in crossings) != z_sites:
        raise ValueError(f"{word!r}: walking the same word does not reach the "
                         f"Z-tile, so Definition 1's parity condition fails")

    x_starts = [(2 * i + x_rel[0], 2 * j + x_rel[1])
                for i, j in code.x_anchors]
    z_starts = [(2 * i + z_rel[0], 2 * j + z_rel[1])
                for i, j in code.z_anchors]
    return x_starts, z_starts


def walk_layout(code, word: str) -> dict[Site, Qubit]:
    """한 라운드를 걸을 수 있는 초기 배치.

    data는 자기 edge 중점에, check는 ``check_starts``가 준 자리에 앉는다. routing은
    **가야 할 경로에 빈 칸이 있는 자리**에만 깐다 -- 논문의 "routing qubits are
    inserted only where required to keep the ordered walk nearest-neighbour"다::

        check의 경로:  A + S_m        (word의 부분합)
        data의 경로:   v - S_m        (역 word의 부분합. data는 매 층 -d_m씩 밀린다)

    두 경로의 빈 칸이 routing이고, 그 routing이 걷는지 밀리는지는 따로 정하지 않는다.
    ``flow_step``에서 ``R``은 앞에 ``D``가 있을 때만 움직이므로, 밀려야 할 자리의
    ``R``은 앞이 check 편이라 저절로 서 있고 지나가는 check가 밀어준다.

    routing은 ``O(sqrt(n))``으로 자란다. ``NESEN``에서 재보면 ``routing/sqrt(n)``이
    2x2부터 8x8까지 6.60에서 6.91로 거의 상수다 (논문 Appendix C의 ``n_r = O(sqrt n)``).
    """
    steps = parse_directional_word(word)
    x_starts, z_starts = check_starts(code, word)

    qubit_at: dict[Site, Qubit] = {}
    data_sites = []
    for col, edge in enumerate(code.qubits):
        site = hardware_site(edge)
        data_sites.append(site)
        qubit_at[site] = Qubit(DATA, col, site)
    for row, site in enumerate(x_starts):
        qubit_at[site] = Qubit(CHECK_X, row, site)
    for row, site in enumerate(z_starts):
        qubit_at[site] = Qubit(CHECK_Z, row, site)

    path = {(start_x + offset_x, start_y + offset_y)
            for start_x, start_y in x_starts + z_starts
            for offset_x, offset_y in partial_sums(steps)}
    path |= {(start_x + offset_x, start_y + offset_y)
             for start_x, start_y in data_sites
             for offset_x, offset_y in partial_sums(reverse_steps(steps))}
    for serial, site in enumerate(sorted(path - set(qubit_at))):
        qubit_at[site] = Qubit(ROUTING, serial, site)
    return qubit_at


def route_windows(code, word: str) -> dict[tuple[str, int], tuple[int, int]]:
    """각 check가 data를 처음/마지막으로 만나는 층, ``(role, index)``별로.

    배치를 굴리지 않고 ``h_m``으로 바로 낸다 -- ``start + h[m]``이 data 제자리인 층을
    모으면 그 처음과 끝이다. ``walk_round``가 실제로 돌려서 주는 창과 같아야 하고,
    한쪽은 산술이고 한쪽은 흐름이라 서로 독립이다.

    support가 빈 check는 ``drop_empty_checks``가 이미 걷어냈으므로 항상 하나는 만난다.
    """
    crossings = walk_crossings(parse_directional_word(word))
    data_sites = {hardware_site(edge) for edge in code.qubits}
    x_starts, z_starts = check_starts(code, word)
    windows = {}
    for role, starts in ((CHECK_X, x_starts), (CHECK_Z, z_starts)):
        for index, (start_x, start_y) in enumerate(starts):
            met = [layer for layer, (cross_x, cross_y) in enumerate(crossings)
                   if (start_x + cross_x, start_y + cross_y) in data_sites]
            windows[(role, index)] = (met[0], met[-1])
    return windows


def shorten_route_windows(code, word: str):
    """죽은 앞뒤 구간을 잘라 줄인 배치 -- 논문 Appendix C의 route window shortening.

    ``(layout, shifts)``. ``shifts``는 ``{(role, index): j}``로, 배치에서 빼고
    ``A + S_j``에서 태어나게 한 check들이다. data의 출발점 이동은 배치 자체에 들어
    있다 (그 열이 ``v - S_j``에 앉는다).

    원문이 두 하위 단계로 적혀 있고 순서가 중요하다.

    **1a** -- "Any prefix or suffix outside this active window is not needed to
    measure the stabiliser support, so the check start position can be shifted and
    the corresponding terminal routing sites can be removed, provided that no
    check-start collision or data-check overlap is introduced." check만 옮기고,
    조건은 충돌·중첩 금지뿐이다 (우리는 그걸 검증으로 확인한다).

    **1b** -- "We then apply the same principle to terminal routing sites **used only
    by a single** data or check qubit q ... it is absorbed into the trajectory of q
    by moving the appropriate start position to that site." 남은 단말 자리를 흡수하고,
    여기에만 단수 조건이 붙는다. data는 "a data-start shift onto a former routing
    coordinate"로 구현된다.

    단수 조건을 1a에도 걸면 check가 못 움직여 훨씬 덜 빠진다 (``N2E2SESE2N2`` 17x6에서
    248 대신 282). 순서를 지키는 것이 조건 자체보다 중요하다.

    1a와 1b를 각각 고정점까지 돌린 뒤 **다시 1a로 돌아온다** -- 1b의 이동이 자리를
    비워주면 1a가 더 옮길 수 있다. 원문 Example 4의 "combining the two optimisation
    procedures and **repeating them** ... reduced iteratively"가 이것이다.

    뒤 절단은 창 계산에서 저절로 된다: check 경로를 ``last+1``까지만 깔면 그 다음 층에
    갈 자리가 없어 ``flow_step``이 그 자리에서 읽는다.

    출발점을 한 칸 옮길 때마다 **배치를 규칙으로 다시 짓고** 검증한다. 다시 짓는 것이
    핵심이다 -- 옮기면 필요 없어진 앞 구간이 통째로 빠지는데, 자리를 하나씩 지우며
    검증하는 방식으로는 그 조합을 못 넘는다 (중간을 하나만 빼면 갈 데 없는 walker가
    서고 다음 층에 겹친다).
    """
    steps = parse_directional_word(word)
    backward = reverse_steps(steps)
    forward = partial_sums(steps)
    depth = len(steps)
    crossings = walk_crossings(steps)
    windows = route_windows(code, word)
    x_starts, z_starts = check_starts(code, word)
    start_of = {(role, index): site
                for role, starts in ((CHECK_X, x_starts), (CHECK_Z, z_starts))
                for index, site in enumerate(starts)}
    home_of = {col: hardware_site(edge)
               for col, edge in enumerate(code.qubits)}
    check_sites = set(start_of.values())
    opens = {}                                 # data 열 -> 처음 먹히는 층
    for col, (home_x, home_y) in home_of.items():
        met = [layer for layer, (cross_x, cross_y) in enumerate(crossings)
               if (home_x - cross_x, home_y - cross_y) in check_sites]
        opens[col] = min(met) if met else 0

    shift = {key: 0 for key in windows}        # check 출발 층
    dshift = {col: 0 for col in home_of}       # data 출발 층

    def owners() -> dict:
        """자리 -> 그 자리를 궤적에 갖는 주체들. 단수 조건이 이걸로 판단한다."""
        who: dict = {}
        for key, (_, last) in windows.items():
            start_x, start_y = start_of[key]
            for m in range(shift[key], last + 2):
                site = (start_x + forward[m][0], start_y + forward[m][1])
                who.setdefault(site, set()).add(("check", key))
        for col, (home_x, home_y) in home_of.items():
            for m in range(dshift[col], depth + 1):
                site = (home_x - forward[m][0], home_y - forward[m][1])
                who.setdefault(site, set()).add(("data", col))
        return who

    def build() -> dict[Site, Qubit]:
        """지금 출발 층들로 배치를 규칙대로 다시 짓는다."""
        qubit_at = {}
        for col, (home_x, home_y) in home_of.items():
            seat = (home_x - forward[dshift[col]][0],
                    home_y - forward[dshift[col]][1])
            qubit_at[seat] = Qubit(DATA, col, seat)
        for key, site in start_of.items():
            if shift[key] == 0:
                qubit_at[site] = Qubit(key[0], key[1], site)
        born = {(start_of[key][0] + forward[shift[key]][0],
                 start_of[key][1] + forward[shift[key]][1])
                for key in windows if shift[key] > 0}
        for serial, site in enumerate(sorted(set(owners()) - set(qubit_at)
                                             - born)):
            qubit_at[site] = Qubit(ROUTING, serial, site)
        return qubit_at

    def survives() -> bool:
        moved = {key: j for key, j in shift.items() if j}
        try:
            qubit_at = build()
            for round_index in range(ROUNDS_TESTED):
                walk_round(code, qubit_at,
                           steps if round_index % 2 == 0 else backward,
                           births_for(code, word, round_index, qubit_at, moved))
            return True
        except ValueError:
            return False

    def advance_checks(gated: bool) -> int:
        moved, who = 0, owners()
        for key, (first, _) in windows.items():
            while shift[key] < first:          # 가까운 쪽부터 한 칸씩
                start_x, start_y = start_of[key]
                target = (start_x + forward[shift[key] + 1][0],
                          start_y + forward[shift[key] + 1][1])
                if gated and who.get(target) != {("check", key)}:
                    break
                shift[key] += 1
                if survives():
                    moved += 1
                    who = owners()
                else:
                    shift[key] -= 1
                    break
        return moved

    def advance_data(gated: bool) -> int:
        moved, who = 0, owners()
        for col, (home_x, home_y) in home_of.items():
            while dshift[col] < opens[col]:
                # 한 칸 막히면 포기하지 않고 더 멀리 시도한다 -- 첫 칸이 check 편
                # 자리라 막히고 두 칸째가 통하는 경우가 있다 (위쪽 경계 data).
                here = dshift[col]
                for target in range(here + 1, opens[col] + 1):
                    seat = (home_x - forward[target][0],
                            home_y - forward[target][1])
                    if gated and who.get(seat) != {("data", col)}:
                        continue
                    dshift[col] = target
                    if survives():
                        moved += 1
                        who = owners()
                        break
                    dshift[col] = here
                if dshift[col] == here:
                    break
        return moved

    if not survives():
        raise ValueError(f"{word!r}: the standard layout does not even walk")
    while True:
        moved = 0
        while advance_checks(gated=False):     # 1a
            moved += 1
        while advance_checks(gated=True) + advance_data(gated=True):   # 1b
            moved += 1
        if not moved:                          # 1b가 1a를 풀어줄 수 있어 되돌아온다
            break
    return build(), {key: j for key, j in shift.items() if j}

def walk_round(code, qubit_at: dict[Site, Qubit], steps: list[Step],
               births=()):
    """한 라운드. ``(층별 gate, check별 창)``을 돌려주고 ``qubit_at``을 갱신한다.

    돌면서 각 check가 ``CXSWAP``한 data 열을 모아 ``HX``/``HZ``의 그 행과 대조한다.
    왼쪽은 상태 추적에서, 오른쪽은 tile 조립에서 나오므로 서로 독립이고, 흐름의
    기하가 한 군데라도 어긋나면 여기서 드러난다. 조용히 넘어가면 무잡음 회로도
    멀쩡해 보이는 채로 엉뚱한 stabilizer를 재게 된다.

    창은 ``(role, index) -> (첫 층, 마지막 층)``, 0부터 센다. 그 check가 data를
    처음 만나는 층부터 마지막으로 만나는 층까지이고, **창 밖에서는 정의상 data를
    만나지 않는다** -- 그래서 회로가 reset을 창 시작 직전에, 측정을 창 끝 직후에
    놓아도 재는 stabilizer가 바뀌지 않는다 (Figure 4의 boundary scheduling).

    ``births``는 ``shorten_route_windows``이 배치에서 빼고 창이 열릴 때 태어나게 한 check들,
    ``(role, index, home, seat, first, last)``. 층 ``first``가 시작될 때 그 자리에
    check가 생긴다.

    역 word의 걸음을 넘기면 되돌아오는 라운드가 된다 -- 같은 집합을 반대 순서로
    먹으므로 대조는 그대로 성립한다.
    """
    collected: dict[tuple[str, int], set[int]] = {}
    windows: dict[tuple[str, int], tuple[int, int]] = {}
    layers = []
    for layer, step in enumerate(steps):
        for role, index, home, seat, opens, _ in births:
            # 창이 열릴 때 그 자리에 check가 생긴다. 그전까지 비어 있어야 한다.
            if opens == layer:
                occupant = qubit_at.get(seat)
                if occupant is not None and occupant.role != ROUTING:
                    raise ValueError(f"{seat}: a check is born onto a "
                                     f"{occupant.role}, not a |0>")
                qubit_at[seat] = Qubit(role, index, home)
        gates, _ = flow_step(qubit_at, step)
        for name, first, second in gates:
            if name != "CXSWAP":
                continue
            # 방금 교환됐으니 둘 다 그 자리에 있다. data가 아닌 쪽이 check다.
            one, other = qubit_at[first], qubit_at[second]
            check, data = (one, other) if one.role != DATA else (other, one)
            key = (check.role, check.index)
            collected.setdefault(key, set()).add(data.index)
            windows[key] = (windows.get(key, (layer, layer))[0], layer)
        layers.append(gates)

    for checks, role in ((code.HX, CHECK_X), (code.HZ, CHECK_Z)):
        for index, row in enumerate(checks):
            support = set(np.flatnonzero(row))
            if collected.get((role, index), set()) != support:
                raise ValueError(
                    f"{role} {index} met columns "
                    f"{sorted(collected.get((role, index), set()))}, but its "
                    f"support is {sorted(support)}")
    return layers, windows


def dormant(layer: int, window: tuple[int, int]) -> bool:
    """그 층에서 check가 자고 있는가 -- 아직 안 켰거나, 이미 읽혔거나.

    자는 동안에는 gate를 내지 않는다. 켜기 전에는 ``|0>``이라 routing과의 SWAP이
    항등이고, 읽힌 뒤에는 그 자리가 아무와도 상호작용하지 않는다.

    이게 깊이를 정한다: 자는 qubit은 그 moment에 비어 있으므로 reset과 측정이 gate
    층과 같은 moment에 들어가고, 라운드가 ``w+2`` moment로 끝난다.
    """
    first, last = window
    return layer < first or layer > last


def round_births(code, word: str, round_index: int, shifts=()):
    """그 라운드에 태어나야 하는 check 후보들.

    ``shorten_route_windows``이 줄인 배치에서는 check가 창 밖을 걷지 않는다. 앞은 배치에서
    빠져 있고, 뒤는 갈 자리가 없어 그 층에서 읽히고 ``|0>``으로 남는다. 그래서 매
    라운드 그 자리에서 다시 태어나야 한다.

    라운드가 word와 역 word를 번갈아 쓰므로 창도 뒤집힌다: 짝수 라운드의 창이
    ``(first, last)``면 홀수 라운드는 ``(w-1-last, w-1-first)``이고, 태어나는 자리는
    ``A + S_{last+1}`` -- 짝수 라운드에서 읽힌 바로 그 자리다. 반대로 홀수 라운드에서
    읽히는 자리가 ``A + S_first``라 짝수 라운드의 탄생 자리로 돌아온다. check가 두
    자리 사이를 왕복한다.

    후보 중 실제로 태어나는 것은 그 라운드 시작 시점에 배치에 없는 것뿐이라, 줄이지
    않은 배치에서는 하나도 태어나지 않는다.

    ``shifts``는 ``{(role, index): j}``로, 그 check가 ``A + S_j``에서 출발한다는 뜻이다.
    적혀 있지 않으면 안 옮긴 것(``j = 0``)이라 배치에 그대로 앉아 있고 태어나지 않는다.
    역 word 라운드는 ``j``와 무관하다 -- check는 여전히 ``A + S_{last+1}``에서 죽고
    거기서 태어나며, 되돌아와 ``A + S_j``에서 잘려 제자리로 온다.
    """
    steps = parse_directional_word(word)
    forward = partial_sums(steps)
    depth = len(steps)
    windows = route_windows(code, word)
    x_starts, z_starts = check_starts(code, word)
    start_of = {(role, index): site
                for role, starts in ((CHECK_X, x_starts), (CHECK_Z, z_starts))
                for index, site in enumerate(starts)}
    shifts = dict(shifts)

    births = []
    for key, (first, last) in sorted(windows.items()):
        if round_index % 2 == 0:
            born_at = shifts.get(key, 0)
            if born_at == 0:
                continue
            opens = born_at
        else:
            if last == depth - 1:
                continue
            born_at = last + 1
            opens = depth - 1 - last
        start_x, start_y = start_of[key]
        seat = (start_x + forward[born_at][0], start_y + forward[born_at][1])
        births.append((key[0], key[1], start_of[key], seat, opens, born_at))
    return births


def births_for(code, word: str, round_index: int, layout, shifts=()) -> list:
    """그 라운드에 실제로 태어날 check들 -- 후보 중 배치에 없는 것.

    줄이지 않은 배치에서는 모든 check가 자리에 있으므로 빈 목록이다.
    """
    present = {(qubit.role, qubit.index) for qubit in layout.values()
               if qubit.role in (CHECK_X, CHECK_Z)}
    return [birth for birth in round_births(code, word, round_index, shifts)
            if (birth[0], birth[1]) not in present]


def trace_prune(code, word: str, layout, shifts=(),
                     thorough: bool = False):
    """가장자리 routing을 하나씩 지워보고 살아남으면 채택 -- Appendix C의 trace pruning.

    "After each proposed removal, we try to greedily remove more routing qubits, by
    iteratively deleting routing qubits near the edges of the layout ... We then run
    the directional syndrome extraction normally for two rounds (forwards and
    backwards) and accept the move only if every stabiliser still interacts with
    exactly the data qubits specified by the original parity-check matrix."

    ``shorten_route_windows``의 창 절단은 최소가 아니다. 그 규칙이 남기는 자리 중에는 지나가던
    walker가 실은 없어도 되는 것이 있는데, 기하에 따라 달라 정적 규칙으로 잡기 어렵다.
    돌려보는 편이 확실하다. 원문의 직관: "even though in the original layout some
    routing qubits might appear necessary, through the displacement of early
    directional word steps, their neighbour routing qubits can fill the gap instead".

    **삭제만** 한다. 출발점 이동은 ``shorten_route_windows``이 고정점까지 하므로 여기서 다시
    시도해도 하나도 안 움직인다 (논문 코드 다섯 개에서 추가 이동 0회, 결과도 삭제만과
    동일). 원문의 단계 구분도 그렇다 -- 이동은 route window shortening 소속이다.

    배치 중심에서 먼 순으로("near the edges") 시도하고, 하나도 못 지우는 패스가 나올
    때까지 반복한다. 하나를 지우면 다른 자리가 지울 수 있게 되기도 한다.

    ``thorough``면 시도 순서를 바꿔 후보를 셋 만들고 물리 qubit이 가장 적은 것을 고른다
    (논문의 "several candidate circuits ... with different update orders"). 논문 코드
    다섯 개에서는 셋이 같은 값으로 수렴해 얻는 것이 없고 세 배 느리기만 하다.
    """
    steps = parse_directional_word(word)
    backward = reverse_steps(steps)

    def survives(trial, moved) -> bool:
        try:
            qubit_at = dict(trial)
            for round_index in range(ROUNDS_TESTED):
                walk_round(code, qubit_at,
                           steps if round_index % 2 == 0 else backward,
                           births_for(code, word, round_index, qubit_at, moved))
            return True
        except ValueError:
            return False

    reference = walk_memory_z_base(code, word, 2, walk_layout(code, word))

    def acceptable(candidate) -> bool:
        """최종 후보의 채택 기준 -- 원문 그대로.

        "The final candidate is accepted only if the measured stabiliser supports,
        detectors, and logical observables are unchanged, and if the detector error
        model remains deterministic."

        support는 삭제마다 ``walk_round``가 보므로 여기서는 나머지 셋을 본다: 줄이지
        않은 회로와 detector·observable 수가 같은지, 그리고 DEM이 결정적인지. support가
        멀쩡해도 창을 잘못 잡아 reset이나 측정이 엉뚱한 moment에 놓이면 detector가
        비결정적이 될 수 있고, 그러면 stim이 DEM을 못 만든다.
        """
        try:
            circuit = walk_memory_z_base(code, word, 2, *candidate)
            return ((circuit.num_detectors, circuit.num_observables)
                    == (reference.num_detectors, reference.num_observables)
                    and circuit.detector_error_model().num_detectors
                    == circuit.num_detectors)
        except ValueError:
            return False

    start_layout, start_shifts = dict(layout), dict(shifts)
    data_sites = [site for site, qubit in start_layout.items()
                  if qubit.role == DATA]
    center_x = sum(x for x, _ in data_sites) / len(data_sites)
    center_y = sum(y for _, y in data_sites) / len(data_sites)
    far = lambda site: -((site[0] - center_x) ** 2 + (site[1] - center_y) ** 2)
    orders = (far,                              # 가장자리부터
              lambda site: -far(site),          # 안쪽부터
              lambda site: (site[0], site[1]))  # 열 순서
    if not thorough:
        orders = orders[:1]

    def run(order):
        """한 후보. 하나도 못 지우는 패스가 나올 때까지 반복한다."""
        layout, shifts = dict(start_layout), dict(start_shifts)
        while True:
            changed = 0
            for site in sorted((site for site, qubit in layout.items()
                                if qubit.role == ROUTING), key=order):
                kept = layout.pop(site)
                if survives(layout, shifts):
                    changed += 1
                else:
                    layout[site] = kept
            if not changed:
                return layout, shifts

    best = None
    for order in orders:
        candidate = run(order)
        if not acceptable(candidate):
            continue
        size = len(candidate[0]) + len(candidate[1])
        if best is None or size < best[0]:
            best = (size, candidate)
    if best is None:                           # 하나도 통과 못 하면 안 줄인다
        return start_layout, start_shifts
    return best[1]

def walk_schedule(code, word: str, rounds: int, layout=None, shifts=()):
    """``rounds`` 번을 미리 돌려 언제 어디서 무엇을 할지 확정한다.

    ``(schedule, initial, final_data)``. ``schedule[r]``은 moment 리스트이고 moment
    하나가 ``(reset, gates, measure)``다. 층 ``i``의 gate는 moment ``i+1``에, 창이
    ``i``에서 열리는 check의 reset은 moment ``i``에, 닫히는 check의 측정은 moment
    ``i+2``에 들어간다 -- 그래서 라운드가 ``w+2`` moment다.

    라운드는 word와 역 word를 번갈아 쓴다. check의 자리는 gate 목록만으로 따라갈 수
    있다: check는 매 층 반드시 gate 하나에 참여하므로(목적지가 ``D``면 ``CXSWAP``,
    ``R``이면 ``SWAP``) 그 gate의 반대쪽 끝이 다음 자리다. 자는 check는 gate를 안
    내지만 자리는 그대로 따라간다 -- 물리적으로는 그 ``|0>``이 제자리에 있고 모델만
    옮기는 셈인데, 자는 것들끼리는 구별되지 않으므로 관측 가능한 차이가 없다.

    ``final_data``는 다 끝난 뒤 각 data 열이 앉아 있는 자리다. 라운드 수가 홀수면
    제자리가 아니라, 마지막 readout을 여기에 걸어야 한다.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    steps = parse_directional_word(word)
    backward = reverse_steps(steps)
    layout = dict(layout) if layout else walk_layout(code, word)
    initial = dict(layout)

    schedule = []
    for round_index in range(rounds):
        position = {(qubit.role, qubit.index): site
                    for site, qubit in layout.items()
                    if qubit.role in (CHECK_X, CHECK_Z)}
        births = births_for(code, word, round_index, layout, shifts)
        layers, windows = walk_round(
            code, layout, steps if round_index % 2 == 0 else backward, births)
        moments = [([], [], []) for _ in range(len(layers) + 2)]
        for layer, gates in enumerate(layers):
            for role, index, _, seat, opens, _ in births:
                if opens == layer:        # 태어난 자리에서 reset이 걸린다
                    position[(role, index)] = seat
            # 자는 check의 SWAP은 내지 않는다. 그래야 그 자리가 이 moment에 비어서
            # reset과 측정이 gate와 나란히 들어간다.
            asleep = {position[key] for key, window in windows.items()
                      if key in position and dormant(layer, window)}
            moments[layer + 1][1].extend(
                gate for gate in gates
                if not (gate[0] == "SWAP"
                        and (gate[1] in asleep or gate[2] in asleep)))
            for key, (first, _) in windows.items():
                if first == layer:
                    moments[layer][0].append((key, position[key]))

            partner = {}
            for _, site_a, site_b in gates:
                partner[site_a] = site_b
                partner[site_b] = site_a
            position = {key: partner.get(site, site)
                        for key, site in position.items()}

            for key, (_, last) in windows.items():
                if last == layer:
                    moments[layer + 2][2].append((key, position[key]))
        for reset, _, measure in moments:
            reset.sort()
            measure.sort()
        schedule.append(moments)

    final_data = {qubit.index: site for site, qubit in layout.items()
                  if qubit.role == DATA}
    return schedule, initial, final_data


def walk_memory_z_base(code, word: str, rounds: int, layout=None,
                       shifts=()) -> stim.Circuit:
    """walk 스케줄로 짓는 잡음 없는 memory-Z 회로.

    ``circuit.py``의 ``memory_z_base``와 계약이 같다: moment를 TICK으로 갈라
    ``NoiseModel``이 gate 잡음과 idle 잡음을 후처리로 주입할 수 있게 한다. 다른 것은
    스케줄뿐이다 -- 장거리 CX 대신 이웃 간 CXSWAP/SWAP으로 걷고, reset과 측정이
    라운드 양끝이 아니라 각 check의 창 끝에 붙는다.

    그래서 측정 시점이 check마다 달라 detector 오프셋을 산술로 계산할 수 없다.
    ``(라운드, check) -> 몇 번째 측정``을 사전으로 들고 다닌다.

    gate에 한 번도 나오지 않는 routing은 회로에서 뺀다. ``|0>``인 채로 아무와도 닿지
    않아 detector에 영향이 없고, 넣으면 qubit 수만 부푼다.
    """
    schedule, initial, final_data = walk_schedule(code, word, rounds, layout,
                                                  shifts)
    _, LZ = code.logicals()

    used = {site for moments in schedule
            for reset, gates, measure in moments
            for site in ([site for _, site in reset]
                         + [site for _, site in measure]
                         + [end for _, site_a, site_b in gates
                            for end in (site_a, site_b)])}
    used |= {site for site, qubit in initial.items() if qubit.role == DATA}
    used |= set(final_data.values())
    index_of = {site: index for index, site in enumerate(sorted(used))}

    circuit = stim.Circuit()
    # 좌표는 하드웨어 격자의 절반이라, data가 (x+0.5, y)에 오는 기존 그림 관례와 맞는다.
    # y를 뒤집어 넣는다: stim의 그림은 y가 클수록 화면 아래라, 그대로 두면 N(북)이
    # 아래로 그려지고 한 셀 안에서 Z check가 X check의 오른쪽 아래에 놓인다.
    for site, index in sorted(index_of.items(), key=lambda pair: pair[1]):
        circuit.append("QUBIT_COORDS", [index], (site[0] / 2, -site[1] / 2))

    # routing은 맨 앞이 아니라 첫 gate 직전 moment에 켠다. 그 자리에 gate가 처음
    # 걸리는 moment가 k라면 그전까지 아무것도 안 닿았으므로 처음의 |0>이 그대로 있고,
    # k-1에 reset해도 지우는 것이 없다. 일찍 켤수록 그 |0>이 오류를 주울 구간만 길어진다.
    first_gate: dict = {}
    for round_index, moments in enumerate(schedule):
        for moment, (_, gates, _) in enumerate(moments):
            for _, site_a, site_b in gates:
                first_gate.setdefault(site_a, (round_index, moment))
                first_gate.setdefault(site_b, (round_index, moment))
    wake: dict = defaultdict(list)
    for site, (round_index, moment) in first_gate.items():
        # 태어나는 자리는 배치에 없다. 그건 스케줄이 reset을 따로 놓는다.
        occupant = initial.get(site)
        if occupant is not None and occupant.role == ROUTING:
            wake[(round_index, moment - 1)].append(index_of[site])

    total = 0                                  # 지금까지의 측정 수
    record: dict = {}                          # (라운드, check) -> 몇 번째 측정
    for round_index, moments in enumerate(schedule):
        for moment, (reset, gates, measure) in enumerate(moments):
            if round_index == 0 and moment == 0:   # data는 |0>에서 시작한다
                circuit.append("R", sorted(
                    index_of[site] for site, qubit in initial.items()
                    if site in index_of and qubit.role == DATA))
            if wake[(round_index, moment)]:
                circuit.append("R", sorted(wake[(round_index, moment)]))
            for role, instruction in ((CHECK_X, "RX"), (CHECK_Z, "R")):
                targets = sorted(index_of[site] for key, site in reset
                                 if key[0] == role)
                if targets:
                    circuit.append(instruction, targets)
            for name in ("CXSWAP", "SWAP"):
                targets = [index_of[site] for gate, site_a, site_b in gates
                           if gate == name for site in (site_a, site_b)]
                if targets:
                    circuit.append(name, targets)
            for role, instruction in ((CHECK_X, "MRX"), (CHECK_Z, "MR")):
                batch = [(key, site) for key, site in measure if key[0] == role]
                if batch:
                    circuit.append(instruction,
                                   [index_of[site] for _, site in batch])
                    for key, _ in batch:
                        record[(round_index, key)] = total
                        total += 1
            circuit.append("TICK")

        # data가 |0>에서 시작하므로 Z 결과는 첫 라운드에서 결정적이고, X 결과는 두
        # 번째 라운드부터만 쓴다.
        for role, checks in ((CHECK_Z, code.HZ), (CHECK_X, code.HX)):
            for row in range(checks.shape[0]):
                if round_index == 0 and role == CHECK_X:
                    continue
                now = stim.target_rec(record[(round_index, (role, row))] - total)
                if round_index == 0:
                    circuit.append("DETECTOR", [now])
                else:
                    circuit.append("DETECTOR", [now, stim.target_rec(
                        record[(round_index - 1, (role, row))] - total)])

    # data의 마지막 transversal Z readout이 각 Z check를 재구성한다. 라운드 수가
    # 홀수면 data가 제자리가 아니므로, 그 시점에 그 열을 들고 있는 qubit을 읽는다.
    circuit.append("MR", [index_of[final_data[col]] for col in range(code.n)])
    data_record = {}
    for col in range(code.n):
        data_record[col] = total
        total += 1
    for row in range(code.HZ.shape[0]):
        targets = [stim.target_rec(data_record[col] - total)
                   for col in np.flatnonzero(code.HZ[row])]
        targets.append(stim.target_rec(
            record[(rounds - 1, (CHECK_Z, row))] - total))
        circuit.append("DETECTOR", targets)
    for index, logical in enumerate(LZ):
        circuit.append("OBSERVABLE_INCLUDE",
                       [stim.target_rec(data_record[col] - total)
                        for col in np.flatnonzero(logical)], index)
    return circuit
