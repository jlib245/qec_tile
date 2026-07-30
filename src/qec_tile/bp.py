"""Belief propagation over GF(2), with the iteration trace exposed.

``ldpc``'s ``BpDecoder`` hands back only the final state, so there is no way to
watch beliefs move -- which is what a decoder viewer, or any study of why BP
stalls on a degenerate code, needs.  This is the same decoder with the loop
turned inside out: ``bp_trace`` yields after every iteration.

Conventions follow ldpc's so the two agree bit for bit (tests/test_bp.py pins
that): the log-likelihood ratio is ``log((1-p)/p)``, positive meaning "no error
on this qubit", and a hard decision is ``llr < 0``.  Only the check-node update
differs between the two schedules, and ldpc's names for them are kept:

``product_sum``   the sum-product algorithm, BP proper -- exact on a tree::

    m_{c->v} = 2 atanh( (-1)^{s_c} prod_{v' != v} tanh(m_{v'->c} / 2) )

``minimum_sum``   its max-log approximation, cheap enough for hardware.  It
overestimates the check messages, which the normalizing factor ``alpha``
(ldpc's ``ms_scaling_factor``) takes back out::

    m_{c->v} = alpha * (-1)^{s_c} * prod_{v' != v} sign(m_{v'->c})
                     * min_{v' != v} |m_{v'->c}|

References
----------
R. G. Gallager, "Low-Density Parity-Check Codes", MIT Press (1963)
-- the sum-product algorithm.

J. Chen and M. Fossorier, "Near optimum universal belief propagation based
decoding of low-density parity check codes", IEEE Trans. Commun. 50 (2002)
-- normalized min-sum and the scaling factor alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BpIteration:
    """One sweep of the schedule.

    ``llr`` is the posterior after this iteration, ``hard`` its sign as a 0/1
    vector, and ``converged`` says whether ``hard`` already explains the
    syndrome — in which case the trace ends here, as ldpc's own loop does.

    ``to_check`` and ``to_bit`` are the two halves of the sweep, one entry per
    Tanner edge in ``tanner_edges`` order: what the v-nodes sent and what the
    c-nodes answered.  Together they are the message passing itself, which the
    posterior only summarises.
    """
    iteration: int
    llr: np.ndarray
    hard: np.ndarray
    converged: bool
    to_check: np.ndarray
    to_bit: np.ndarray


def tanner_edges(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)``: the check and qubit each Tanner edge joins.

    This is the order every message array uses.  ``np.nonzero`` walks
    row-major, so a check's edges are contiguous, which is what the per-check
    reductions in ``bp_trace`` rely on.
    """
    return np.nonzero(np.ascontiguousarray(H, dtype=np.uint8))


def _prior_llr(channel, n: int) -> np.ndarray:
    """Channel probabilities -> prior LLRs; a scalar means one uniform rate."""
    probability = np.broadcast_to(np.asarray(channel, dtype=float), (n,))
    if not ((0 < probability) & (probability < 1)).all():
        raise ValueError("channel probabilities must lie strictly in (0, 1)")
    return np.log((1 - probability) / probability)


METHODS = ("minimum_sum", "product_sum")


def _phi(magnitude: np.ndarray) -> np.ndarray:
    """Gallager's ``-log tanh(x/2)``, which is its own inverse.

    Storing ``-log`` of the bias rather than the bias itself is what keeps
    strong beliefs: ``1 - tanh(x/2) ~ 2 exp(-x)`` slips under double eps around
    ``x = 37``, so a product of tanh cannot tell 37 from 100, while its log is
    an ordinary small number.  Both ends need the right identity, since
    ``1 + u`` and ``1 - u`` (with ``u = exp(-x)``) each lose everything at one
    extreme::

        x >= 1:  log1p(u) - log1p(-u)                       u is tiny
        x <  1:  log(2 + expm1(-x)) - log(-expm1(-x))       1 - u is about x

    ``phi(0)`` is ``+inf`` and ``phi(inf)`` is 0, which is the right behaviour:
    a check that hears "no idea" from one qubit has nothing to tell the others.
    """
    magnitude = np.abs(magnitude)
    small = magnitude < 1.0
    out = np.empty_like(magnitude, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = -np.expm1(-magnitude)              # 1 - exp(-x), exact for small x
        out[small] = (np.log(2.0 + np.expm1(-magnitude[small]))
                      - np.log(gap[small]))
        decay = np.exp(-magnitude[~small])
        out[~small] = np.log1p(decay) - np.log1p(-decay)
    return out


def bp_trace(H: np.ndarray, syndrome: np.ndarray, channel,
             method: str = "minimum_sum", max_iter: int = 50,
             ms_scaling_factor: float = 1.0):
    """Flooding BP, yielding a ``BpIteration`` after every sweep.

    ``channel`` is a scalar rate or one probability per column, matching
    ``decode.make_decoder``.  ``method`` picks the check-node update, using
    ldpc's names.  The generator stops early on convergence.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; have {list(METHODS)}")
    H = np.ascontiguousarray(H, dtype=np.uint8)
    n_checks, n = H.shape
    syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(n_checks)
    prior = _prior_llr(channel, n)

    rows, cols = tanner_edges(H)
    if rows.size == 0:                       # no checks: nothing to propagate
        empty = np.zeros(0)
        yield BpIteration(1, prior.copy(), np.zeros(n, dtype=np.uint8),
                          not syndrome.any(), empty, empty)
        return

    degree = np.bincount(rows, minlength=n_checks)
    if (degree == 0).any():
        raise ValueError("H has an all-zero check; drop it before decoding")
    row_starts = np.searchsorted(rows, np.arange(n_checks))
    # A degree-1 check has no "other" edge, so the min over the rest is +inf:
    # the check alone fixes its qubit.  Clip the index to stay in bounds; the
    # value is discarded wherever degree == 1.
    second_slot = np.where(degree > 1, np.minimum(row_starts + 1, rows.size - 1),
                           row_starts)
    # Position of each edge within its check, for the prefix/suffix sums
    # product_sum needs.
    slot = np.arange(rows.size) - row_starts[rows]
    widest = int(degree.max())
    check_flip = np.where(syndrome[rows] == 1, -1.0, 1.0)
    message_to_check = prior[cols].copy()

    for iteration in range(1, max_iter + 1):
        # Both updates need the parity of the signs of the *other* edges.
        negatives = np.bincount(rows, weights=(message_to_check < 0),
                                minlength=n_checks).astype(np.int64)
        parity = (negatives[rows] - (message_to_check < 0)) % 2
        sign = np.where(parity == 1, -1.0, 1.0)

        if method == "minimum_sum":
            magnitude = np.abs(message_to_check)
            # "Smallest magnitude among the *other* edges of this check": sort
            # each check's edges by magnitude and keep the best two, then hand
            # every edge the winner unless it is the winner itself.
            order = np.lexsort((magnitude, rows))
            smallest = order[row_starts]
            runner_up = np.where(degree > 1, magnitude[order[second_slot]],
                                 np.inf)
            is_smallest = np.zeros(rows.size, dtype=bool)
            is_smallest[smallest] = True
            other = np.where(is_smallest, runner_up[rows],
                             magnitude[smallest][rows])
            message_to_bit = ms_scaling_factor * sign * check_flip * other
        else:                                # product_sum
            # phi turns the product over "the others" into a sum.  Taking that
            # sum for the row and subtracting the edge's own term would look
            # cheaper, but it cancels catastrophically -- one dominant term
            # leaves 0.405 - 0.405 = 0 and phi(0) is infinite.  Summing what
            # lies before and after each edge instead keeps it to additions,
            # and infinities (a zero message) propagate correctly.
            transformed = _phi(message_to_check)
            grid = np.zeros((n_checks, widest))
            grid[rows, slot] = transformed
            before = np.zeros_like(grid)
            before[:, 1:] = np.cumsum(grid[:, :-1], axis=1)
            after = np.zeros_like(grid)
            after[:, :-1] = np.cumsum(grid[:, :0:-1], axis=1)[:, ::-1]
            others = (before + after)[rows, slot]
            message_to_bit = sign * check_flip * _phi(others)

        llr = prior + np.bincount(cols, weights=message_to_bit, minlength=n)
        hard = (llr < 0).astype(np.uint8)
        converged = np.array_equal((H @ hard) % 2, syndrome)
        yield BpIteration(iteration, llr, hard, bool(converged),
                          message_to_check.copy(), message_to_bit)
        if converged:
            return

        message_to_check = llr[cols] - message_to_bit


def bp(H: np.ndarray, syndrome: np.ndarray, channel,
       method: str = "minimum_sum", max_iter: int = 50,
       ms_scaling_factor: float = 1.0):
    """``(hard, llr, converged_at)``; ``converged_at`` is None if BP stalled."""
    last = None
    for last in bp_trace(H, syndrome, channel, method, max_iter,
                         ms_scaling_factor):
        pass
    return last.hard, last.llr, last.iteration if last.converged else None
