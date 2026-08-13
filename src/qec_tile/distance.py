"""부호 거리 — 가장 가벼운 logical 연산자.

    d = min { wt(e) : H_detect @ e = 0,  L_pair @ e != 0 }

첫 조건은 오류가 syndrome을 남기지 않는다는 것, 둘째는 그것이 단지 stabilizer인
것은 아니라는 뜻이다. CSS 코드는 이런 문제 두 개로 쪼개진다: X 타입 오류는
``HZ``가 보고 ``LZ``와 짝지어지며, Z 타입은 ``HX``/``LX``가 맡는다. 부호 거리는
둘 중 작은 쪽이다.

최소값 찾기는 NP-hard(coset leader 문제)이므로, 규모를 감당하면서 정확하기도 한
루틴은 없다:

``distance_bruteforce``   정확하지만 C(n, w)를 다 훑는다 — 작은 코드 전용.
``distance_upper_bound``  무작위. 항상 실제 logical이라 d 아래로 내려가지 않는다.
``distance_ilp``          정확하다. 정수 계획으로 풀어 더 멀리 가지만, 어디까지
                          가는지는 solver의 성질이지 보장이 아니다.

규모를 감당하는 둘은 짝이 된다: 거리를 싸게 위에서 누르고, ILP가 확정하게 한다.
b3w6 10x10 ([[288, 8, 12]])에서 실측 — 무작위 탐색은 1초쯤에 12에 닿고, ILP는
weight 11 이하가 없음을 ~32분에 증명한다 (2*k = 16개 부분문제, 각 ~2분, 그
bound로 상한을 걸었을 때).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .gf2 import nullspace2, rref2


def _sectors(code, sector: str):
    """요청한 sector마다 (name, H_detect, L_pair, same_type_generators)."""
    LX, LZ = code.logicals()
    both = {
        # X 타입 오류는 Z-check가 검출하고 LZ와 짝지어진다. 검출되지 않는
        # 것들은 X 타입 생성원에 LX를 더한 것이 span한다.
        "x": ("x", code.HZ, LZ, np.vstack([code.HX, LX])),
        "z": ("z", code.HX, LX, np.vstack([code.HZ, LZ])),
    }
    if sector is None:
        return list(both.values())
    if sector not in both:
        raise ValueError(f"sector must be 'x', 'z' or None, not {sector!r}")
    return [both[sector]]


def distance_bruteforce(code, max_weight: int | None = None,
                        sector: str | None = None) -> int | None:
    """weight 1, 2, 3, ...의 오류를 훑어 얻는 정확한 거리.

    ``max_weight`` 이하에서 아무것도 못 찾으면 ``None``. 비용에 유의 —
    C(288, 12)가 1e20쯤이라 작은 배치에만 쓴다.
    """
    best = None
    for _, H_detect, L_pair, _ in _sectors(code, sector):
        n = H_detect.shape[1]
        limit = n if max_weight is None else min(max_weight, n)
        if best is not None:
            limit = min(limit, best - 1)        # 더 나은 것만 찾는다
        found = _sector_bruteforce(H_detect, L_pair, limit)
        if found is not None:
            best = found
    return best


def _sector_bruteforce(H_detect, L_pair, limit: int) -> int | None:
    n = H_detect.shape[1]
    error = np.zeros(n, dtype=np.uint8)
    for weight in range(1, limit + 1):
        for support in itertools.combinations(range(n), weight):
            error[:] = 0
            error[list(support)] = 1
            if (not ((H_detect @ error) % 2).any()
                    and ((L_pair @ error) % 2).any()):
                return weight
    return None


def distance_upper_bound(code, trials: int = 200, seed: int | None = None,
                         sector: str | None = None) -> int:
    """거리의 무작위 상한.

    검출되지 않는 오류들은 ``[H_same; L_same]``의 행공간이다. 열을 무작위로 뒤섞은
    채 그 생성 행렬을 기약 행 사다리꼴로 만들면 행들의 weight가 낮아진다.
    stabilizer가 아닌 행은 logical 연산자이고, 그 weight가 거리의 상한이 된다.
    순열을 새로 뽑아 반복하면 상한이 내려간다. 고전적인 information-set 탐색이며,
    실제 logical 연산자의 weight만 보고하므로 답이 참 거리보다 낮아지는 일은 없다.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _, _, L_pair, generators in _sectors(code, sector):
        found = _sector_upper_bound(generators, L_pair, trials, rng)
        if found is not None and (best is None or found < best):
            best = found
    if best is None:                            # k = 0: logical이 아예 없다
        raise ValueError("code has no logical operators")
    return best


def classical_distance_upper_bound(H_detect: np.ndarray, L_pair: np.ndarray,
                                   trials: int = 200,
                                   seed: int | None = None) -> int:
    """``H_detect @ e = 0``이고 ``L_pair @ e != 0``인 가장 가벼운 ``e``의 무작위 상한.

    ``distance_upper_bound``의 고전판 — CSS sector 대신 행렬 짝을 직접 받는다.
    검출되지 않는 벡터가 ``nullspace2(H_detect)``의 행공간이라는 것만 다르고,
    information-set 탐색은 같은 ``_sector_upper_bound``다.
    """
    rng = np.random.default_rng(seed)
    found = _sector_upper_bound(nullspace2(H_detect), L_pair, trials, rng)
    if found is None:
        raise ValueError("no vector pairs nontrivially with L_pair")
    return found


def distance_ilp(code, upper_bound: int | None = None,
                 sector: str | None = None) -> int | None:
    """정수 계획으로 얻는 정확한 거리. scipy를 통해 HiGHS로 푼다.

    ``L_pair @ e != 0``은 논리합이라 ILP로는 적을 수 없다. 그래서 logical 하나당
    문제 하나로 쪼갠다: ``i``번째 행이 반교환하도록 강제하고 weight를 최소화한다.
    logical 연산자는 최소 한 행과 반교환하므로, 그 실행들 중 최선이 거리다.

    ``upper_bound``(``distance_upper_bound``에서 얻은 것)를 주면 weight에 상한을
    걸어 탐색을 잘라낸다. 그 상한에서도 실행 불가능하면 ``None``이 돌아오는데,
    그것 자체가 ``d > upper_bound``의 증명이다.
    """
    best = None
    for _, H_detect, L_pair, _ in _sectors(code, sector):
        for i in range(L_pair.shape[0]):
            cap = upper_bound
            if best is not None:                # 더 나은 것만 찾는다
                cap = best - 1 if cap is None else min(cap, best - 1)
            if cap is not None and cap < 1:     # 더 가벼운 것은 있을 수 없다
                break
            weight = _solve_one_logical(H_detect, L_pair[i], cap)
            if weight is not None and (best is None or weight < best):
                best = weight
    return best


def _solve_one_logical(H_detect, logical, cap: int | None) -> int | None:
    """``H_detect @ e = 0``이고 ``logical . e = 1``인(mod 2) 가장 가벼운 ``e``."""
    m, n = H_detect.shape
    # 변수: e (n개 이진) | check마다 slack (정수) | slack 하나 더.
    # mod-2 등식을 정수 등식으로: sum == 2 * slack (홀수면 + 1).
    n_var = n + m + 1
    rows, lower, upper = [], [], []

    def add(row, value, at_most=None):
        rows.append(row)
        lower.append(value)
        upper.append(value if at_most is None else at_most)

    for j in range(m):                          # H_detect[j] . e == 2 * slack
        row = np.zeros(n_var)
        row[:n] = H_detect[j]
        row[n + j] = -2
        add(row, 0)

    row = np.zeros(n_var)                       # logical . e == 2 * slack + 1
    row[:n] = logical
    row[-1] = -2
    add(row, 1)

    if cap is not None:                         # 1 <= sum(e) <= cap
        row = np.zeros(n_var)
        row[:n] = 1
        add(row, 1, cap)

    cost = np.zeros(n_var)
    cost[:n] = 1
    result = milp(c=cost,
                  constraints=LinearConstraint(np.array(rows), lower, upper),
                  integrality=np.ones(n_var),
                  bounds=Bounds(np.zeros(n_var),
                                np.concatenate([np.ones(n),
                                                np.full(m + 1, np.inf)])))
    if result.status == 0:
        return int(round(result.fun))
    if result.status == 2:                      # 실행 불가능: cap 안에는 없다
        return None
    raise RuntimeError(f"MILP solver did not finish: {result.message}")


def _sector_upper_bound(generators, L_pair, trials: int, rng) -> int | None:
    n = generators.shape[1]
    best = None
    for _ in range(trials):
        order = rng.permutation(n)
        reduced, _ = rref2(generators[:, order])
        rows = np.zeros_like(reduced)
        rows[:, order] = reduced                # 순열을 되돌린다
        nontrivial = ((L_pair @ rows.T) % 2).any(axis=0)
        for row in rows[nontrivial]:
            weight = int(row.sum())
            if best is None or weight < best:
                best = weight
    return best
