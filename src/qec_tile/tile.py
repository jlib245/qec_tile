"""Tile code — 경계가 있는 평면 격자 위에서 O(1)-local인 CSS 코드.

구성은 Steffan, Choe, Breuckmann, Fernandes Pereira & Eberhardt,
"Tile Codes: High-Efficiency Quantum Codes on a Lattice with Boundary"
(arXiv:2504.09171)를 따른다.

기하
----
qubit은 정사각 격자의 edge에 앉는다. vertex는 정수 ``(x, y)``에 있다::

    H(x, y):  (x, y) -- (x+1, y)      수평 edge
    V(x, y):  (x, y) -- (x, y+1)      수직 edge

``B x B`` box는 B x B *cell*이다. 쓸 수 있는 edge는 맨 위 행과 맨 오른쪽 열에
걸리지 않는 것들, 즉 ``x, y in [0, B)``인 ``H(x, y)``와 ``V(x, y)`` — box당
``2*B**2``개의 후보.

tile은 그중 한 부분집합이다. 조건 (T2)가 X-tile로부터 Z-tile을 결정하고
(180도 회전 + H<->V 교환), 이것이 모든 상대 overlap을 짝수로 만든다::

    Z_V = {(B-1-x, B-1-y) for (x, y) in X_H}
    Z_H = {(B-1-x, B-1-y) for (x, y) in X_V}

배치
----
anchor는 자기 box의 왼쪽 아래 꼭짓점에 앉는다. bulk 블록이 ``L1 x L2``일 때 각
타입은 사각형 하나를 훑고, box가 걸쳐 나가는 축으로 ``B-1``만큼 더 나간다::

    X anchor:  i in [0, L1),                   j in [-(B-1), L2+B-1)
    Z anchor:  i in [-(B-1), L1+B-1),          j in [0, L2)

논문 그림은 ``[0,L1) x [0,L2)`` 안쪽을 검정, 걸쳐 나간 것을 빨강(X)과 파랑(Z)으로
칠한다. 위 anchor 집합은 논문의 (회전하지 않은) 정사각 배치이고, qubit 집합은
box의 합집합을 그대로 계산하므로 다른 bulk 배치로 바꾸려면 이 세 리스트만 고치면
된다.

qubit은 bulk anchor의 box만 합친 것이라 bulk tile은 절대 잘리지 않고, 경계 tile만
격자 밖으로 걸쳐 잘린다. 걸쳐 나간 X tile이 잃는 것은 언제나 ``y`` 범위 밖이고 Z
tile이 잃는 것은 ``x`` 범위 밖이므로, 잘린 qubit이 반대 타입의 tile에 들어 있는
일은 없다 — (T2)의 overlap 짝수성은 절단을 견딘다.

따라서 ``n = 2*(L1+B-1)*(L2+B-1)``이고, 모든 check가 독립이므로 배치 크기와
무관하게 ``k = 2*(B-1)**2``다.

논문은 pruning으로 끝맺는다: X-stabilizer가 (또는 Z-stabilizer가) 하나도
작용하지 않는 qubit을 버리고, 그로 인해 비어버린 stabilizer도 버린다. 논문 자신의
tile에서는 버려지는 것이 없고, box를 다 채우지 않는 tile에서 의미가 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import nullspace2, quotient_basis, rank2

Edge = tuple[str, int, int]


def z_tile_from_x(x_h, x_v, B: int) -> tuple[list, list]:
    """(T2): X-tile이 결정하는 Z-tile."""
    z_v = [(B - 1 - x, B - 1 - y) for (x, y) in x_h]
    z_h = [(B - 1 - x, B - 1 - y) for (x, y) in x_v]
    return z_h, z_v


@dataclass
class TileCode:
    HX: np.ndarray                 # (mx, n) uint8
    HZ: np.ndarray                 # (mz, n) uint8
    qubits: list[Edge]             # 열 인덱스 -> ('H'|'V', x, y)
    x_anchors: list[tuple[int, int]]   # HX의 행 인덱스 -> anchor
    z_anchors: list[tuple[int, int]]
    B: int
    L1: int
    L2: int

    @property
    def n(self) -> int:
        return len(self.qubits)

    @property
    def k(self) -> int:
        return self.n - rank2(self.HX) - rank2(self.HZ)

    def logicals(self) -> tuple[np.ndarray, np.ndarray]:
        """``(LX, LZ)``, 각각 GF(2) 위 ``(k, n)``.

        X 타입 logical은 ker(H_Z)를 X-stabilizer로 나눈 것이고, 반대도 같다.
        X 타입 residual 오류 ``r``(이미 Z-check syndrome과 맞는 것)이 logical
        실패인 것은 ``LZ @ r != 0``일 때다.
        """
        LX = quotient_basis(self.HX, nullspace2(self.HZ))
        LZ = quotient_basis(self.HZ, nullspace2(self.HX))
        return LX, LZ

    def __str__(self) -> str:
        return (f"TileCode(B={self.B}, layout={self.L1}x{self.L2}, "
                f"n={self.n}, k={self.k}, "
                f"w={int(self.HX.sum(1).max())}/{int(self.HZ.sum(1).max())})")


def build_tile_code(x_h, x_v, B: int, L1: int, L2: int) -> TileCode:
    """X-tile ``x_h``(수평) + ``x_v``(수직)로 tile code를 짓는다.

    좌표는 ``[0, B)^2`` 안의 box 상대 오프셋이다.
    """
    for (x, y) in list(x_h) + list(x_v):
        if not (0 <= x < B and 0 <= y < B):
            raise ValueError(f"offset ({x},{y}) outside the {B}x{B} box")

    z_h, z_v = z_tile_from_x(x_h, x_v, B)
    x_tile = [("H", x, y) for x, y in x_h] + [("V", x, y) for x, y in x_v]
    z_tile = [("H", x, y) for x, y in z_h] + [("V", x, y) for x, y in z_v]

    # 타입마다 한 번씩 훑고, j가 가장 안쪽이다. qubit 열이 x-major라서 j를 한 칸
    # 옮기면 check의 support가 한 열, i를 한 칸 옮기면 격자 열 하나만큼 밀린다.
    # 이렇게 훑으면 연속한 check가 격자를 건너뛰지 않고 겹친다. bulk와 경계
    # anchor를 따로 돌지 않는다 -- X anchor는 j로, Z anchor는 i로 격자보다
    # B-1만큼 더 나가면 되므로 각 집합이 사각형 하나다.
    x_anchors_all = [(i, j) for i in range(L1)
                     for j in range(-(B - 1), L2 + B - 1)]
    z_anchors_all = [(i, j) for i in range(-(B - 1), L1 + B - 1)
                     for j in range(L2)]
    bulk = [(i, j) for i in range(L1) for j in range(L2)]   # qubit은 여기서 나온다

    # qubit은 bulk anchor의 B x B box를 합친 것이다. 정사각 배치에서 그 합집합은
    # 사각형 [0, L1+B-1) x [0, L2+B-1)이지만, 곧이곧대로 합쳐두면 나머지 구성이
    # 배치에 무관해진다.
    qubits = sorted({(orient, i + dx, j + dy)
                     for orient in "HV"
                     for (i, j) in bulk
                     for dx in range(B)
                     for dy in range(B)})
    col_of = {qubit: col for col, qubit in enumerate(qubits)}

    def assemble(tile, anchors):
        """anchor마다 check 하나: tile을 찍고, qubit이 없는 자리는 잘라낸다."""
        checks, kept_anchors = [], []
        for (anchor_x, anchor_y) in anchors:
            check = np.zeros(len(qubits), dtype=np.uint8)
            for (orient, dx, dy) in tile:
                qubit = (orient, anchor_x + dx, anchor_y + dy)
                if qubit in col_of:          # 있는 qubit에 맞춰 절단
                    check[col_of[qubit]] ^= 1
            if check.any():                  # tile이 통째로 격자 밖
                checks.append(check)
                kept_anchors.append((anchor_x, anchor_y))
        return (np.array(checks, dtype=np.uint8) if checks
                else np.zeros((0, len(qubits)), dtype=np.uint8)), kept_anchors

    HX, x_anchors = assemble(x_tile, x_anchors_all)
    HZ, z_anchors = assemble(z_tile, z_anchors_all)

    # 논문의 Pruning 파트: X-check가 (또는 Z-check가) 하나도 건드리지 않는 qubit은 syndrome을 남기지 않으니 버린다. 그러면 비게 된 check도 버린다. 
    covered = (HX.sum(0) > 0) & (HZ.sum(0) > 0)
    if not covered.all():
        qubits = [qubit for qubit, is_covered in zip(qubits, covered)
                  if is_covered]
        HX, HZ = HX[:, covered], HZ[:, covered]
        HX, x_anchors = drop_empty_checks(HX, x_anchors)
        HZ, z_anchors = drop_empty_checks(HZ, z_anchors)

    if ((HX @ HZ.T) % 2).any():
        raise ValueError("stabilizers do not commute — X-tile violates (T2)")
    return TileCode(HX, HZ, qubits, x_anchors, z_anchors, B, L1, L2)


def drop_empty_checks(checks: np.ndarray,
                      anchors: list) -> tuple[np.ndarray, list]:
    """support가 남은 ``checks``의 행들과, 짝이 되는 anchor."""
    nonempty = checks.any(axis=1)
    return checks[nonempty], [anchor for anchor, is_kept
                              in zip(anchors, nonempty) if is_kept]


# 논문의 tile들, (X_H, X_V) 형태. Table 1의 3행과 4행은 tile을 공유하고 배치만
# 다르므로, 이름이 가리키는 것은 tile이지 코드가 아니다.
TILES: dict[str, tuple[list, list]] = {
    # Table 1
    "b3w6": ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    "b3w8": ([(0, 0), (0, 1), (0, 2), (2, 0)],
             [(0, 0), (0, 2), (1, 1), (2, 2)]),
    "b4w8": ([(0, 0), (0, 3), (2, 2), (3, 0)],
             [(0, 1), (1, 0), (1, 1), (3, 3)]),
    "b4w10": ([(0, 0), (1, 0), (2, 1), (2, 3), (3, 0)],
              [(0, 3), (1, 0), (3, 1), (3, 2), (3, 3)]),
}

# Table 2: 10x10에서 [[288,8,12]]를 주는, 그림에 실린 weight-6 B=3 X-tile 여덟 개.
# 논문이 세는 16개는 이것들과 그 X<->Z 교환이다.
TABLE2: list[tuple[list, list]] = [
    ([(0, 0), (0, 1), (2, 2)], [(0, 2), (1, 0), (2, 0)]),
    ([(0, 0), (0, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    ([(0, 0), (1, 0), (2, 2)], [(0, 1), (0, 2), (2, 0)]),
    ([(0, 0), (1, 0), (2, 2)], [(0, 2), (2, 0), (2, 1)]),
    ([(0, 0), (1, 2), (2, 2)], [(0, 1), (0, 2), (2, 0)]),
    ([(0, 0), (1, 2), (2, 2)], [(0, 2), (2, 0), (2, 1)]),
    ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 0), (2, 0)]),
    ([(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
]


def paper_code(name: str, L1: int, L2: int) -> TileCode:
    """편의 함수: 논문의 tile 하나를 주어진 배치 크기로 짓는다."""
    x_h, x_v = TILES[name]
    B = 3 if name.startswith("b3") else 4
    return build_tile_code(x_h, x_v, B, L1, L2)
