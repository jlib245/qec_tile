"""Directional tile code — tile을 compass walk로 그리는 tile code.

구성은 Gu, Noszko, Steffan, Eberhardt, Roffe, Eisert & Koutsioumpas,
"Nearest-neighbour gates are all you need: High-rate quantum low-density parity-check
codes on a planar grid" (arXiv:2606.19482)를 따른다.

Definition 1: "A pair of XX- and ZZ-tiles is called directional if the tiles
satisfy the mutual condition, one tile forms an ordered connected string
labelled by a directional word D = d_1 d_2 ... d_w, and the other tile
contains the same string on the dual lattice.  In addition, every displacement
vector of the ordered connected string with odd vertical displacement must
occur with even multiplicity."

mutual condition -- "for each horizontal, respectively vertical, edge of the
XX-tile with coordinate (a,b), the ZZ-tile contains the vertical, respectively
horizontal, edge with coordinate (B-1-a, B-1-b)" -- 은 원래 tile code 논문의
(T2)다. 그래서 ``tile.z_tile_from_x``가 이미 구현하고 있고, directional code는
X-tile에 제약을 건 tile code다.

walk
----
string은 *edge*의 string이므로 한 글자가 한 edge, 곧 data qubit 하나다: weight w인
word가 weight-w stabilizer를 준다. check qubit이 코드 격자의 vertex를 걸으며 자기가
건너는 edge를 먹는다::

    (x, y) --N--> V(x,   y  ), 도착 (x,   y+1)
    (x, y) --S--> V(x,   y-1), 도착 (x,   y-1)
    (x, y) --E--> H(x,   y  ), 도착 (x+1, y  )
    (x, y) --W--> H(x-1, y  ), 도착 (x-1, y  )

하드웨어의 정사각 격자는 이 격자를 두 배로 늘린 것이라, 모든 정수 site ``(X, Y)``는
넷 중 하나다::

    (even, even)  vertex       -- XX-check anchor
    (odd,  even)  H((X-1)/2, Y/2)   data qubit
    (even, odd )  V(X/2, (Y-1)/2)   data qubit
    (odd,  odd )  face 중심     -- dual 격자 위의 ZZ-check anchor

string의 연속한 edge는 하드웨어에서 두 걸음 떨어져 있고 그 사이에 vertex나 face
중심이 있다. 그 중간 site가 check가 CXSWAP 사이에 SWAP으로 지나가는 routing
qubit이다. XX-check는 vertex에서, ZZ-check는 face 중심에서 시작하므로 두 walk는
절대 충돌하지 않는다.
"""
from __future__ import annotations

import itertools
from collections import Counter

from .tile import TileCode, build_tile_code

# 하드웨어 격자 위의 compass 걸음: N은 +y, E는 +x.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (0, 1), "E": (1, 0), "S": (0, -1), "W": (-1, 0)}

Edge = tuple[str, int, int]


def parse_directional_word(word: str) -> list[tuple[int, int]]:
    """directional word를 단위 걸음으로 펼친다.

    word : 논문의 지수 형태 "N^2 E^2 S E S E^2 N^2" -> 상수 형태 "N2E2SESE2N2"``.
    """
    steps: list[tuple[int, int]] = []
    position = 0
    compact = "".join(word.split())
    while position < len(compact):
        letter = compact[position]
        if letter not in DIRECTIONS:
            raise ValueError(
                f"{word!r}: expected one of {sorted(DIRECTIONS)} at index "
                f"{position}, got {letter!r}")
        position += 1
        start = position
        while position < len(compact) and compact[position].isdigit():
            position += 1
        repeat = int(compact[start:position]) if position > start else 1
        if repeat < 1:
            raise ValueError(f"{word!r}: repeat count must be >= 1")
        steps.extend([DIRECTIONS[letter]] * repeat)
    if not steps:
        raise ValueError("directional word is empty")
    return steps


def walk_edges(steps: list[tuple[int, int]]) -> list[Edge]:
    """격자의 vertex를 걸으며 각 걸음이 건너는 edge의 이름을 붙인다.

    한 글자가 한 edge이므로 결과는 걸음마다 data qubit 하나이고, walk 순서가
    syndrome 추출이 방문하는 순서다.

    좌표는 anchor 기준이라 음수일 수 있다. ``B x B`` box로 정규화하는 것은 호출자
    몫이다. 음의 방향으로 걸으면 vertex *뒤쪽* edge를 건넌다 -- ``(0, 0)``에서
    ``W``로 가면 ``H(0, 0)``이 아니라 ``H(-1, 0)``이다.
    """
    edges: list[Edge] = []
    x, y = 0, 0
    for (step_x, step_y) in steps:
        if step_x:
            edges.append(("H", x if step_x > 0 else x - 1, y))
        else:
            edges.append(("V", x, y if step_y > 0 else y - 1))
        x, y = x + step_x, y + step_y
    return edges


def hardware_site(edge: Edge) -> tuple[int, int]:
    """edge의 중점, 격자를 두 배로 늘린 하드웨어 격자 위에서.

    수평 edge는 하드웨어 y가 짝수, 수직 edge는 홀수에 앉는다. 그래서 변위의 수직
    성분이 홀수인 것은 정확히 H edge와 V edge를 이을 때다.
    """
    orient, x, y = edge
    return (2 * x + 1, 2 * y) if orient == "H" else (2 * x, 2 * y + 1)


def displacement_vectors(edges: list[Edge]) -> Counter[tuple[int, int]]:
    """string의 두 edge 사이 벡터가 각각 몇 번 나오는지.

    walk의 순서쌍 ``i < j`` 하나하나가 ``site_j - site_i``를 낸다. Figure 5가 "for
    each starting point"라고 그리는 것이다.
    """
    sites = [hardware_site(edge) for edge in edges]
    return Counter((later_x - earlier_x, later_y - earlier_y)
                   for (earlier_x, earlier_y), (later_x, later_y)
                   in itertools.combinations(sites, 2))


def satisfies_parity_condition(edges: list[Edge]) -> bool:
    """Definition 1: "every displacement vector of the ordered connected string
    with odd vertical displacement must occur an even number of times".

    mutual condition만으로 stabilizer는 이미 교환한다. 이건 syndrome 추출 회로를
    결정적으로 만드는 추가 조건이라, tile을 집합으로서가 아니라 walk를 스케줄로서
    제약한다.
    """
    return all(count % 2 == 0
               for vector, count in displacement_vectors(edges).items()
               if vector[1] % 2)


def tile_from_word(word: str,
                   B: int | None = None) -> tuple[list[tuple[int, int]],
                                                  list[tuple[int, int]], int]:
    """``word``가 그리는 X-tile ``(x_h, x_v, B)``, assembler에 바로 넣을 수 있게.

    ``build_tile_code``가 요구하는 대로 오프셋을 옮겨 tile이 원점에 붙게 한다.
    ``B``의 기본값은 walk를 담는 가장 작은 정사각 box다. 논문은 배치를
    ``(M+B-1) x (N+B-1)`` 격자로 고정하지만 자기 코드가 어떤 ``B``를 쓰는지는
    밝히지 않으므로, 덮어쓸 수 있게 남긴다.
    """
    edges = walk_edges(parse_directional_word(word))
    if len(set(edges)) != len(edges):
        raise ValueError(f"{word!r}: the walk crosses an edge twice, so it is "
                         f"not an ordered connected string")

    min_x = min(x for _, x, _ in edges)
    min_y = min(y for _, _, y in edges)
    span = max(max(x for _, x, _ in edges) - min_x,
               max(y for _, _, y in edges) - min_y) + 1
    if B is None:
        B = span
    elif B < span:
        raise ValueError(f"{word!r} spans {span}, does not fit a {B}x{B} box")

    x_h = sorted((x - min_x, y - min_y) for orient, x, y in edges
                 if orient == "H")
    x_v = sorted((x - min_x, y - min_y) for orient, x, y in edges
                 if orient == "V")
    return x_h, x_v, B


def build_directional_code(word: str, M: int, N: int,
                           B: int | None = None) -> TileCode:
    """``word``의 directional tile code를 논문의 M x N anchor 격자 위에 짓는다.

    조립은 원래 tile code의 조립 그대로다: 논문은 "(M+B-1) x (N+B-1) rectangular
    grid"에 "the vertices of an M x N subgrid as anchors"로 테셀레이션하는데, 이는
    ``build_tile_code``가 이미 하는 배치이고, mutual condition은 (T2)이므로
    ``z_tile_from_x``가 이미 강제한다.
    """
    x_h, x_v, B = tile_from_word(word, B)
    return build_tile_code(x_h, x_v, B, M, N)


# arXiv:2606.19482의 Table 2, (word, M, N, n, k, d) 형태. 논문은 word와 [[n,k,d]]는
# 인쇄하지만 B도 M x N anchor 격자도 밝히지 않아, 둘 다 탐색으로 복원했다. (n,k)만으로는
# 배치가 정해지지 않았다 -- 세 행에서 같은 (n,k)를 주는 두 번째 배치가 있었고, 거리가
# 그것을 갈랐다 (버려진 쪽은 d = 3, 2, 6으로 7, 15, 11과 크게 어긋났다). B는 walk를 담는
# 최소 box이고, 비용이 없다: 코드가 B에 의존하지 않는다.
PAPER_CODES: list[tuple[str, int, int, int, int, int]] = [
    ("N2ESEN2",       4, 4,  60,  4,  5),
    ("N2ESEN2",       8, 8, 180,  4,  9),
    ("N2E2SE2N2",    11, 6, 217, 10,  7),
    ("N2E2SE2N2",    15, 8, 351, 10,  9),
    ("N2E2SESE2N2",  12, 4, 182, 14, 10),
    ("N2E2SESE2N2",  17, 6, 323, 14, 15),
    ("N2E2SE3SE2N2", 16, 4, 248, 20, 11),
]
