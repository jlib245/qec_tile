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

X와 Z가 같은 방향으로 흐르는데도 Z-tile이 X-tile의 180도 회전이 되는 것은 논문의
word가 palindrome이기 때문이다: 회전이 평행이동과 같아지고, 그 평행이동이
``C - 2S_w`` (``C = (2B-1, 2B-1)``)라는 face 중심이다. 두 흐름은 좌표합 parity가
``(odd, odd)``만큼 어긋나 있어 서로의 자리를 노리는 일이 없다.

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

# 논문 Algorithm 1의 D, C_X, C_Z, R. stim의 "R"(reset), "X"/"Z"(Pauli)와 섞이지
# 않게 풀어 쓴다 -- 이 모듈이 그 instruction들을 그대로 뱉는 곳이다.
DATA = "data"
CHECK_X = "check_x"
CHECK_Z = "check_z"
ROUTING = "routing"   # 경계 data를 실어나르는 셔틀. 걷는다.
FILLER = "filler"     # check가 밟고 지나갈 디딤돌. 밀리기만 한다.


@dataclass(frozen=True)
class Qubit:
    """layout을 이루는 qubit 하나. site에 고정된 물리 qubit이 아니라 그 위를 흐른다."""

    role: str          # DATA | CHECK_X | CHECK_Z | ROUTING
    index: int         # DATA는 code.qubits 열, CHECK_*는 HX/HZ 행, ROUTING은 일련번호
    home: Site         # 라운드를 시작할 때 앉아 있던 site

def flow_step(qubit_at: dict[Site, Qubit], step: tuple[int, int],
              layer: int | None = None) -> list[tuple[str, Site, Site]]:
    """
    step 한 층. gate 목록을 돌려주고 qubit_at을 제자리에서 갱신한다.

    Algorithm 1의 for i = 1..w 내부 cycle.
    walking qubit은 전부 같은 방향으로 한 칸 전진하여 그 자리에 있던 qubit과 교환된다.
    층 안의 이동은 동시에 일어나므로 목표를 다 정한 뒤에 한꺼번에 교환한다.

    stim의 CXSWAP a b는 control a의 CX 다음 SWAP이라, 첫 자리가 control이다:
    X check는 자기가, Z check는 data가 control이다. routing이 끼면 SWAP이어야 한다.
    |+> check와 |0> routing에 CX를 걸면 둘이 얽혀 syndrome이 무작위가 된다.

    목적지가 없을 때 무엇을 하느냐는 누가 못 가느냐에 달렸다. check가 못 가면 stabilizer를
    덜 재는 것이라 배치가 모자란 것이고, routing이 못 가면 그냥 선다 -- Algorithm 1이
    end if로 비워둔 분기다. 유한한 격자에서는 이걸 허용하지 않으면 배치가 존재할 수 없다:
    경계에서 walker를 지우면 그 빈자리 때문에 안쪽 walker도 지워야 하고, 그 침식이 한
    궤적 폭씩 안으로 전파되며 격자를 통째로 먹는다.
    """
    checks, routing = [], []            # swap control 주체
    for site, qubit in qubit_at.items(): # qubit_at : 해당 자리에 있는 qubit
        if qubit.role in (CHECK_X, CHECK_Z):
            checks.append((site, qubit.role))
        elif qubit.role == ROUTING:
            routing.append(site)
    checks.sort()
    routing.sort()

    gates: list[tuple[str, Site, Site]] = []
    moves: list[tuple[Site, Site]] = [] # moves : site pair for each qubit swap
    taken: set[Site] = set()            # 한 moment에 한 qubit은 gate 하나만

    def claim(site, target):
        if site in taken or target in taken:
            raise ValueError(f"{site} -> {target}: two gates in one layer")
        taken.update((site, target))
        moves.append((site, target))

    for site, role in checks:                          # 5-15행
        target = (site[0] + step[0], site[1] + step[1])  # step : displacement
        if target not in qubit_at:
            raise ValueError(f"{site} -> {target}: a check cannot step onto an empty site; widen the layout")
        met = qubit_at[target].role
        if met == DATA:
            if role == CHECK_X :
                gates.append(("CXSWAP", site, target))
            else:
                gates.append(("CXSWAP", target, site))  # X는 check가, Z는 data가 control
        elif met in (ROUTING, FILLER):
            gates.append(("SWAP", site, target))
        else:
            raise ValueError(f"{site} -> {target}: two checks meet, so the walk geometry is broken")
        claim(site, target)

    for site in routing:                               # 16-20행
        target = (site[0] + step[0], site[1] + step[1])
        if target not in qubit_at:
            continue
        met = qubit_at[target].role
        if met == DATA:
            gates.append(("SWAP", site, target))       # 16-19행
        # 디딤돌을 지나갈 때는 |0> 둘의 교환이라 gate가 없다. 그래도 자리는 바꾼다 --
        # 셔틀이 한 층 쉬면 그만 뒤처져 다음 층에 남의 자리를 밟는다.
        claim(site, target)

    for site, target in moves:
        qubit_at[site], qubit_at[target] = qubit_at[target], qubit_at[site]
    return gates


def walk_crossings(steps: list[tuple[int, int]]) -> list[Site]:
    """walk가 건너는 edge들의 하드웨어 좌표, 출발점 기준.

    ``m``번째 걸음이 건너는 edge의 중점은 ``S_{m-1} + S_m``이다. 흐름이 층 ``m``에서
    check 앞으로 데려다주는 data의 제자리가 정확히 이것이라, 이 목록이 곧 check가
    무엇을 어떤 순서로 먹는지다.
    """
    partial = [(0, 0)]
    for (step_x, step_y) in steps:
        partial.append((partial[-1][0] + step_x, partial[-1][1] + step_y))
    return [(before_x + after_x, before_y + after_y)
            for (before_x, before_y), (after_x, after_y)
            in zip(partial, partial[1:])]


def check_starts(code, word: str) -> tuple[list[Site], list[Site]]:
    """각 check가 라운드를 시작하는 site, ``HX``/``HZ`` 행 순서로.

    X check는 자기 anchor vertex에서 출발한다. ``tile_from_word``가 tile을 box에
    붙이려고 ``(min_x, min_y)``만큼 옮겨놓았으므로 그만큼 되돌린다::

        A = 2 * (anchor - (min_x, min_y))

    Z check는 같은 word를 같은 방향으로 걷고도 180도 회전한 tile을 먹어야 해서 anchor
    box의 반대쪽 모서리에서 출발한다. ``C = (2B-1, 2B-1)``, ``M = 2(min_x, min_y)``,
    ``S_w`` = word의 총 변위일 때::

        F = 2 * anchor + C + M - 2 * S_w

    흐름이 주는 것은 ``{F + h_m}``, ``z_tile_from_x``가 요구하는 것은
    ``{C + M - h_m}``이라, 둘이 같으려면 walk의 edge 집합이 ``S_w``를 중심으로
    점대칭이어야 한다. 길이 8까지 전수로는 이것이 Definition 1의 parity 조건과
    동치다. 그래도 동치인 대리조건이 아니라 필요한 것 자체를 재는 이유는, 더 긴
    word에서 둘이 갈라질 때 틀린 출발점이 조용히 통과하면 안 되기 때문이다.
    """
    steps = parse_directional_word(word)
    edges = walk_edges(steps)
    min_x = min(x for _, x, _ in edges)
    min_y = min(y for _, _, y in edges)
    crossings = walk_crossings(steps)
    end_x = sum(step_x for step_x, _ in steps)
    end_y = sum(step_y for _, step_y in steps)

    # Z 출발점의 anchor 상대 오프셋. C는 box의 반대쪽 모서리다.
    shift = (2 * code.B - 1 + 2 * min_x - 2 * end_x,
             2 * code.B - 1 + 2 * min_y - 2 * end_y)

    x_h, x_v, _ = tile_from_word(word, code.B)
    z_h, z_v = z_tile_from_x(x_h, x_v, code.B)
    z_sites = sorted([hardware_site(("H", x, y)) for x, y in z_h]
                     + [hardware_site(("V", x, y)) for x, y in z_v])
    if sorted((shift[0] + cx, shift[1] + cy)
              for cx, cy in crossings) != z_sites:
        raise ValueError(
            f"{word!r}: walking the same word does not reach the Z-tile, so "
            f"Definition 1's parity condition fails")

    x_starts = [(2 * (i - min_x), 2 * (j - min_y)) for i, j in code.x_anchors]
    z_starts = [(2 * i + shift[0], 2 * j + shift[1]) for i, j in code.z_anchors]
    return x_starts, z_starts


def walk_layout(code, word: str) -> dict[Site, Qubit]:
    """한 라운드를 걸을 수 있는 초기 배치.

    data는 자기 edge 중점에, check는 ``check_starts``가 준 자리에 앉는다. routing은
    **실제로 걸어야 하는 경로의 빈 칸에만** 깐다 -- 논문의 "routing qubits are inserted
    only where required to keep the ordered walk nearest-neighbour"가 이것이다::

        걷는 자리 = check 출발점 ∪ { v - h_k : v는 data 제자리, k = 1..w }
        배치      = data 제자리  ∪ { 걷는 자리 + S_m : m = 0..w }

    뒷항이 덜 뻔하다. data는 스스로 못 움직이고 누가 자기 자리로 걸어 들어와야 한 칸
    밀리므로, 층 ``k``마다 다른 walker가 밀어줘야 한다. 그 walker의 출발점이
    ``v - h_k``다. bulk에서는 이웃 check들이 공짜로 해주지만 경계에서는 코드 바깥에
    walker가 서 있어야 하고, 그 자리들이 routing이다. stabilizer의 상자를 채우는
    방식으로는 여기에 닿지 않는다 -- 상자를 2B x 2B로 키워도 뒤쪽 자리가 빈다.

    routing은 ``O(sqrt(n))``으로 자란다. ``NESEN``에서 재보면 ``routing/sqrt(n)``이
    2x2부터 12x12까지 10.6에서 11.2로 거의 상수다 (논문 Appendix C의 ``n_r = O(sqrt n)``).

    routing이 하는 일은 둘이고, 걷기가 필요한 건 하나뿐이다. 대부분은 check가 밟고
    지나갈 자리를 채우는 디딤돌이라 밀려나기만 하면 된다 (경계 check의 절단도 여기서
    나온다 -- check가 CXSWAP이 아니라 SWAP으로 지나가므로). 걷는 것은 위의 seat,
    곧 밀어줄 data가 있는 자리뿐이고 그것만 ``pushes=True``다.
    """
    steps = parse_directional_word(word)
    x_starts, z_starts = check_starts(code, word)
    crossings = walk_crossings(steps)

    trajectory = [(0, 0)]                  # S_0 .. S_w, walker가 밟는 자리
    for (step_x, step_y) in steps:
        trajectory.append((trajectory[-1][0] + step_x,
                           trajectory[-1][1] + step_y))

    qubit_at: dict[Site, Qubit] = {}
    for col, edge in enumerate(code.qubits):
        site = hardware_site(edge)
        qubit_at[site] = Qubit(DATA, col, site)
    for row, site in enumerate(x_starts):
        qubit_at[site] = Qubit(CHECK_X, row, site)
    for row, site in enumerate(z_starts):
        qubit_at[site] = Qubit(CHECK_Z, row, site)

    data_sites = {site for site, qubit in qubit_at.items() if qubit.role == DATA}
    check_seats = set(x_starts) | set(z_starts)

    def last_meeting(seat):
        """그 자리에서 출발한 walker가 마지막으로 data를 만나는 layer (없으면 -1)."""
        met = [layer for layer, (cross_x, cross_y) in enumerate(crossings)
               if (seat[0] + cross_x, seat[1] + cross_y) in data_sites]
        return max(met) if met else -1

    seats = set(check_seats)
    seats |= {(site[0] - cross_x, site[1] - cross_y)   # data를 밀어줄 walker
              for site in data_sites for cross_x, cross_y in crossings}
    empty = {(seat[0] + offset_x, seat[1] + offset_y)
             for seat in seats for offset_x, offset_y in trajectory}
    empty -= set(qubit_at)

    for routing, site in enumerate(sorted(empty)):
        # 밀어줄 data가 있는 자리(seat)만 셔틀이고 나머지는 디딤돌이다. 자리의
        # 홀짝이 아니라 할 일이 있느냐로 가른다.
        qubit_at[site] = Qubit(ROUTING if site in seats else FILLER,
                               routing, site)
    return qubit_at


def prune_routing(code, word: str, layout: dict[Site, Qubit] | None = None,
                  shift: bool = True):
    """지우고 검증하며 배치를 줄인다 -- 논문 Appendix C의 "generating and testing".

    후보 둘을 번갈아 시도하고, 시도마다 두 라운드(word와 역 word)를 돌려 각 check가
    여전히 자기 ``HX``/``HZ`` 행을 먹으면 채택한다. 논문의 채택 기준과 같다:
    "The optimisation preserves the measured stabiliser supports, detectors and
    logical observables".

    * **옮기기** (route window shortening) -- 창이 ``first``층에 열리는 check를
      ``출발점 + S_first``로 옮기고 그때부터 걷게 한다. ``first``층 이후의 자취가
      원래와 같으므로 support는 그대로이고, 죽은 앞 구간이 사라져 그 자리의 routing이
      지울 수 있는 후보가 된다.
    * **지우기** -- routing site 하나를 빼고 돌려본다. 바깥쪽부터 시도한다.

    정적 규칙으로 강제하지 않는 이유는, 옮기거나 지운 자리를 지나가던 walker가
    있으면 흐름이 깨지는데 그게 기하에 따라 다르기 때문이다. 돌려보는 편이 확실하다.
    """
    steps = parse_directional_word(word)
    backward = [(-step_x, -step_y) for step_x, step_y in reversed(steps)]
    crossings = walk_crossings(steps)
    trajectory = [(0, 0)]
    for (step_x, step_y) in steps:
        trajectory.append((trajectory[-1][0] + step_x,
                           trajectory[-1][1] + step_y))
    if layout is None:
        layout = walk_layout(code, word)
    data_sites = {site for site, qubit in layout.items() if qubit.role == DATA}

    def survives():
        try:
            qubit_at = dict(layout)
            walk_round(code, qubit_at, steps)
            walk_round(code, qubit_at, backward)
            return True
        except ValueError:
            return False

    def window_at(start, crossings_of_round):
        """그 자리에서 출발한 check의 (first, last). data를 하나도 안 만나면 None."""
        met = [layer for layer, (cross_x, cross_y) in enumerate(crossings_of_round)
               if (start[0] + cross_x, start[1] + cross_y) in data_sites]
        return (met[0], met[-1]) if met else None

    back_crossings = walk_crossings(backward)
    back_trajectory = [(0, 0)]
    for (step_x, step_y) in backward:
        back_trajectory.append((back_trajectory[-1][0] + step_x,
                                back_trajectory[-1][1] + step_y))
    end = trajectory[-1]

    def births_for(reborn):
        """옮긴 check들의 라운드별 탄생 기록. 역 word 라운드는 끝 자리에서 다시 센다."""
        forward, back = [], []
        for role, index, start in reborn:
            window = window_at(start, crossings)
            if window:
                first, last = window
                forward.append((role, index, start,
                                (start[0] + trajectory[first][0],
                                 start[1] + trajectory[first][1]), first, last))
            # 역 라운드에서는 data가 이미 -S_w만큼 밀려 있다. 창도 그 자리 기준이다.
            far = (start[0] + end[0], start[1] + end[1])
            window = window_at((far[0] + end[0], far[1] + end[1]), back_crossings)
            if window:
                first, last = window
                back.append((role, index, start,
                             (far[0] + back_trajectory[first][0],
                              far[1] + back_trajectory[first][1]), first, last))
        return forward, back

    def survives(reborn=()):
        forward, back = births_for(reborn)
        try:
            qubit_at = dict(layout)
            walk_round(code, qubit_at, steps, forward)
            walk_round(code, qubit_at, backward, back, phase=len(steps))
            return True
        except ValueError:
            return False

    center_x = sum(x for x, _ in data_sites) / len(data_sites)
    center_y = sum(y for _, y in data_sites) / len(data_sites)
    reborn = []                                # 배치에서 빼고 탄생으로 돌린 check
    while True:
        changed = 0
        for site in ([s for s, q in layout.items()
                      if q.role in (CHECK_X, CHECK_Z)] if shift else []):
            window = window_at(site, crossings)
            if not window or window[0] == 0:
                continue
            check = layout[site]
            layout[site] = Qubit(ROUTING, -1, site)
            reborn.append((check.role, check.index, site))
            if survives(reborn):
                changed += 1
            else:
                layout[site] = check
                reborn.pop()
        for site in sorted((s for s, q in layout.items() if q.role == ROUTING),
                           key=lambda s: -((s[0] - center_x) ** 2
                                           + (s[1] - center_y) ** 2)):
            kept = layout.pop(site)
            if survives(reborn):
                changed += 1
            else:
                layout[site] = kept
        if not changed:
            return layout, reborn


def walk_round(code, qubit_at: dict[Site, Qubit],
               steps: list[tuple[int, int]], births=(), phase: int = 0):
    """한 라운드. ``(층별 gate, check별 활성 구간)``을 돌려주고 ``qubit_at``을 갱신한다.

    돌면서 각 check가 CXSWAP한 data 열을 모아 ``HX``/``HZ``의 그 행과 대조한다.
    왼쪽은 상태 추적에서, 오른쪽은 tile 조립에서 나오므로 서로 독립이고, 흐름의
    기하가 한 군데라도 어긋나면 여기서 드러난다. 조용히 넘어가면 무잡음 회로도
    멀쩡해 보이는 채로 엉뚱한 stabilizer를 재게 된다.

    활성 구간(창)은 ``(role, index) -> (첫 층, 마지막 층)``, 0부터 센다. 그 check가
    data를 처음 만나는 층부터 마지막으로 만나는 층까지이고, **창 밖에서는 정의상
    data를 만나지 않는다** -- 그래서 회로가 reset을 창 시작 직전에, 측정을 창 끝
    직후에 놓아도 재는 stabilizer가 바뀌지 않는다 (Figure 4의 "boundary scheduling
    optimisations"). 경계 check는 tile이 잘려 창이 짧다.

    역 word를 넘기면 되돌아오는 라운드가 된다 -- 같은 집합을 반대 순서로 먹으므로
    대조는 그대로 성립한다.
    """
    collected: dict[tuple[str, int], set[int]] = defaultdict(set)
    windows: dict[tuple[str, int], tuple[int, int]] = {}
    layers = []
    for layer, step in enumerate(steps):
        for role, index, home, site, first, last in births:
            # 창이 열릴 때 그 자리의 |0>이 check가 된다. 그전에는 check라는 것이
            # 없으므로, 지나가던 walker가 그 자리를 밟아도 그냥 SWAP이다.
            if first == layer:
                if site not in qubit_at:
                    raise ValueError(f"{site}: a check is born where the layout "
                                     f"has no qubit")
                qubit_at[site] = Qubit(role, index, home)
        gates = flow_step(qubit_at, step, phase + layer)
        for name, first, second in gates:
            if name != "CXSWAP":
                continue
            # 방금 교환됐으니 둘 다 그 자리에 있다. data가 아닌 쪽이 check다.
            one, other = qubit_at[first], qubit_at[second]
            check, data = (one, other) if one.role != DATA else (other, one)
            collected[(check.role, check.index)].add(data.index)
            opened = windows.get((check.role, check.index), (layer, layer))[0]
            windows[(check.role, check.index)] = (opened, layer)
        for role, index, home, site, first, last in births:
            if last == layer:
                for spot, qubit in list(qubit_at.items()):
                    if qubit.role == role and qubit.index == index:
                        qubit_at[spot] = Qubit(ROUTING, -1, spot)
        layers.append(gates)

    for checks, role in ((code.HX, CHECK_X), (code.HZ, CHECK_Z)):
        for index, row in enumerate(checks):
            support = set(np.flatnonzero(row))
            if collected[(role, index)] != support:
                raise ValueError(
                    f"{role} {index} met columns "
                    f"{sorted(collected[(role, index)])}, but its support is "
                    f"{sorted(support)}")
    return layers, windows


def dormant(layer: int, window: tuple[int, int]) -> bool:
    """그 layer에서 check가 자고 있는가 -- 아직 안 켰거나, 이미 읽혔거나.

    자는 동안에는 gate를 내지 않는다. 켜기 전에는 |0>이라 routing과의 SWAP이 항등이고,
    읽힌 뒤에는 그 자리가 아무와도 상호작용하지 않아 무엇이 남아 있든 상관없다
    (Figure 4가 측정된 자리를 이후 패널에서 빼는 것이 이것이다).

    이게 깊이를 정한다: 자는 qubit은 그 moment에 비어 있으므로 reset과 측정이 gate
    layer와 같은 moment에 들어가고, 라운드가 w+2 moment로 끝난다.
    """
    first, last = window
    return layer < first or layer > last


def check_births(word: str, reborn, data_sites, rounds: int):
    """옮긴 check들의 라운드별 탄생 기록 ``(role, index, home, site, first, last)``.

    ``prune_routing``이 배치에서 빼고 탄생으로 돌린 check들이다. 창은 그 라운드가
    시작될 때의 data 자리에 대고 센다 -- 홀수 라운드에는 data가 이미 ``-S_w``만큼
    밀려 있으므로 그만큼 보정한다.
    """
    steps = parse_directional_word(word)
    backward = [(-step_x, -step_y) for step_x, step_y in reversed(steps)]
    end = (sum(step_x for step_x, _ in steps), sum(step_y for _, step_y in steps))
    per_round = []
    for round_index in range(rounds):
        these = steps if round_index % 2 == 0 else backward
        crossings = walk_crossings(these)
        trajectory = [(0, 0)]
        for (step_x, step_y) in these:
            trajectory.append((trajectory[-1][0] + step_x,
                               trajectory[-1][1] + step_y))
        births = []
        for role, index, start in reborn:
            base = start if round_index % 2 == 0 else (start[0] + end[0],
                                                       start[1] + end[1])
            probe = base if round_index % 2 == 0 else (base[0] + end[0],
                                                       base[1] + end[1])
            met = [layer for layer, (cross_x, cross_y) in enumerate(crossings)
                   if (probe[0] + cross_x, probe[1] + cross_y) in data_sites]
            if not met:
                continue
            births.append((role, index, start,
                           (base[0] + trajectory[met[0]][0],
                            base[1] + trajectory[met[0]][1]), met[0], met[-1]))
        per_round.append(births)
    return per_round


def walk_schedule(code, word: str, rounds: int, layout=None, reborn=()):
    """
    rounds 번의 라운드를 미리 돌려, 언제 어디서 무엇을 할지 확정한다.

    return value : (schedule, layout, final_data).
    schedule은 라운드마다 moment list이고 moment 하나가 (reset, gates, measure).
    layer i의 gate는 moment i+1에, 창이 i에서 열리는 check의 reset은 moment i에,
    닫히는 check의 측정은 moment i+2에 들어간다 -- 그래서 라운드가 w+2 moment다.
    final_data는 다 끝난 뒤 각 data 열이 앉아 있는 자리다 (라운드 수가 홀수면 제자리가 아니다).

    라운드는 word와 reverse word를 번갈아 쓴다. check의 자리는 gate 목록만으로 따라갈 수 있다
    check는 매 layer 반드시 gate 하나에 참여하므로, 그 gate의 반대쪽 끝이 다음 자리다.
    자는 check는 gate를 안 내지만 자리는 그대로 따라간다 -- 물리적으로는 그 |0>이 제자리에
    있고 모델만 옮기는 셈인데, 자는 것들끼리는 구별되지 않으므로 관측 가능한 차이가 없다.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    steps = parse_directional_word(word)
    backward = [(-step_x, -step_y) for step_x, step_y in reversed(steps)]

    layout = dict(layout) if layout else walk_layout(code, word)
    initial = dict(layout)
    position = {(qubit.role, qubit.index): site
                for site, qubit in layout.items()
                if qubit.role in (CHECK_X, CHECK_Z)}

    data_sites = {site for site, qubit in initial.items() if qubit.role == DATA}
    per_round = check_births(word, reborn, data_sites, rounds)

    schedule = []
    for round_index in range(rounds):
        births = per_round[round_index]
        layers, windows = walk_round(
            code, layout, steps if round_index % 2 == 0 else backward, births,
            phase=round_index * len(steps))
        moments = [([], [], []) for _ in range(len(layers) + 2)]
        for layer, gates in enumerate(layers):
            for role, index, _, site, first, _ in births:
                if first == layer:        # 태어난 자리에서 reset이 걸린다
                    position[(role, index)] = site
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
            # 일이 끝난 check는 gate가 없다. 그런 것은 제자리에 둔다.
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


def walk_memory_z_base(code, word: str, rounds: int,
                       layout=None, reborn=()) -> stim.Circuit:
    """walk 스케줄로 짓는 잡음 없는 memory-Z 회로.

    ``circuit.py``의 ``memory_z_base``와 계약이 같다: moment를 TICK으로 갈라
    ``NoiseModel``이 gate 잡음과 idle 잡음을 후처리로 주입할 수 있게 한다. 다른 것은
    스케줄뿐이다 -- 장거리 CX 대신 이웃 간 CXSWAP/SWAP으로 걷고, reset과 측정이
    라운드 양끝이 아니라 각 check의 활성 구간 끝에 붙는다 (Figure 4의 "boundary
    scheduling optimisations").

    그래서 측정 시점이 check마다 달라 detector 오프셋을 산술로 계산할 수 없다.
    ``(라운드, check) -> 몇 번째 측정``을 사전으로 들고 다닌다.

    gate에 한 번도 나오지 않는 routing은 회로에서 뺀다. ``|0>``인 채로 아무와도 닿지
    않아 detector에 영향이 없고, 넣으면 qubit 수만 부푼다.
    """
    schedule, initial, final_data = walk_schedule(code, word, rounds,
                                                  layout, reborn)
    _, LZ = code.logicals()

    used = {site for round_layers in schedule
            for reset, gates, measure in round_layers
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
    # k-1에 reset해도 지우는 것이 없다. gate가 있는 moment는 1 이상이라 k-1은 같은
    # 라운드 안이다. 일찍 켤수록 그 |0>이 오류를 주울 구간만 길어진다.
    first_gate: dict = {}
    for round_index, moments in enumerate(schedule):
        for moment, (_, gates, _) in enumerate(moments):
            for _, site_a, site_b in gates:
                first_gate.setdefault(site_a, (round_index, moment))
                first_gate.setdefault(site_b, (round_index, moment))
    wake: dict = defaultdict(list)
    for site, (round_index, moment) in first_gate.items():
        if initial[site].role == ROUTING:
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
