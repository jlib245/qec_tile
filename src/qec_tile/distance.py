"""Code distance — the lightest logical operator.

    d = min { wt(e) : H_detect @ e = 0,  L_pair @ e != 0 }

The first condition says the error leaves no syndrome, the second that it is
not merely a stabilizer.  A CSS code splits into two such problems: an X-type
error is seen by ``HZ`` and paired with ``LZ``, a Z-type one by ``HX``/``LX``,
and the code distance is the smaller of the two.

Finding the minimum is NP-hard (it is the coset leader problem), so there is no
one routine that both scales and stays exact:

``distance_bruteforce``   exact, but enumerates C(n, w) — small codes only.
``distance_upper_bound``  randomised, always a valid logical, never below d.
``distance_ilp``          exact, as an integer program; scales further, but
                          how far is a property of the solver, not a promise.

The two scalable ones pair up: bound the distance cheaply, then let the ILP
confirm it.  Measured on b3w6 at 10x10 ([[288, 8, 12]]), the randomised search
reaches 12 in about a second, and the ILP proves nothing of weight 11 or less
exists in ~32 min (2*k = 16 subproblems, ~2 min each, capped at that bound).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .gf2 import rref2


def _sectors(code, sector: str):
    """(name, H_detect, L_pair, same_type_generators) per requested sector."""
    LX, LZ = code.logicals()
    both = {
        # X-type errors are detected by the Z-checks and paired with LZ; the
        # undetectable ones are spanned by the X-type generators plus LX.
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
    """Exact distance by enumerating errors of weight 1, 2, 3, ...

    Returns ``None`` if nothing is found at or below ``max_weight``.  Note the
    cost: C(288, 12) is about 1e20, so this is for small layouts only.
    """
    best = None
    for _, H_detect, L_pair, _ in _sectors(code, sector):
        n = H_detect.shape[1]
        limit = n if max_weight is None else min(max_weight, n)
        if best is not None:
            limit = min(limit, best - 1)        # only look for something better
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
    """Randomised upper bound on the distance.

    The undetectable errors are the row space of ``[H_same; L_same]``.  Putting
    that generator matrix in reduced row echelon form under a random column
    permutation makes its rows low weight; any row that is not a stabilizer is
    a logical operator, and its weight bounds the distance.  Repeating with
    fresh permutations lowers the bound.  This is the classic information-set
    search, and it only ever reports the weight of a real logical operator, so
    the answer can never fall below the true distance.
    """
    rng = np.random.default_rng(seed)
    best = None
    for _, _, L_pair, generators in _sectors(code, sector):
        found = _sector_upper_bound(generators, L_pair, trials, rng)
        if found is not None and (best is None or found < best):
            best = found
    if best is None:                            # k = 0: no logicals at all
        raise ValueError("code has no logical operators")
    return best


def distance_ilp(code, upper_bound: int | None = None,
                 sector: str | None = None) -> int | None:
    """Exact distance as an integer program, solved with HiGHS via scipy.

    ``L_pair @ e != 0`` is a disjunction, which an ILP cannot state, so it is
    split into one problem per logical: force row ``i`` to anticommute and
    minimise the weight.  A logical operator anticommutes with at least one
    row, so the best of those runs is the distance.

    Passing ``upper_bound`` (from ``distance_upper_bound``) caps the weight and
    prunes the search.  ``None`` comes back when even that cap is infeasible,
    which is itself the proof that ``d > upper_bound``.
    """
    best = None
    for _, H_detect, L_pair, _ in _sectors(code, sector):
        for i in range(L_pair.shape[0]):
            cap = upper_bound
            if best is not None:                # only ever look for better
                cap = best - 1 if cap is None else min(cap, best - 1)
            if cap is not None and cap < 1:     # nothing lighter can exist
                break
            weight = _solve_one_logical(H_detect, L_pair[i], cap)
            if weight is not None and (best is None or weight < best):
                best = weight
    return best


def _solve_one_logical(H_detect, logical, cap: int | None) -> int | None:
    """Lightest ``e`` with ``H_detect @ e = 0`` and ``logical . e = 1``, mod 2."""
    m, n = H_detect.shape
    # Variables: e (n binary) | slack per check (integer) | one more slack.
    # Mod-2 equalities become integer ones: sum == 2 * slack (+ 1 when odd).
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
    if result.status == 2:                      # infeasible: none within cap
        return None
    raise RuntimeError(f"MILP solver did not finish: {result.message}")


def _sector_upper_bound(generators, L_pair, trials: int, rng) -> int | None:
    n = generators.shape[1]
    best = None
    for _ in range(trials):
        order = rng.permutation(n)
        reduced, _ = rref2(generators[:, order])
        rows = np.zeros_like(reduced)
        rows[:, order] = reduced                # undo the permutation
        nontrivial = ((L_pair @ rows.T) % 2).any(axis=0)
        for row in rows[nontrivial]:
            weight = int(row.sum())
            if best is None or weight < best:
                best = weight
    return best
