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
    """
    iteration: int
    llr: np.ndarray
    hard: np.ndarray
    converged: bool


def _prior_llr(channel, n: int) -> np.ndarray:
    """Channel probabilities -> prior LLRs; a scalar means one uniform rate."""
    probability = np.broadcast_to(np.asarray(channel, dtype=float), (n,))
    if not ((0 < probability) & (probability < 1)).all():
        raise ValueError("channel probabilities must lie strictly in (0, 1)")
    return np.log((1 - probability) / probability)


METHODS = ("minimum_sum", "product_sum")

# tanh saturates at +-1 in double precision, and the "all the others" product
# is taken by dividing the row's product by the edge's own factor, so a factor
# of exactly +-1 would divide by zero.  Clip just inside.
_TANH_LIMIT = 1 - 1e-12


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

    # One entry per Tanner-graph edge.  np.nonzero walks row-major, so the
    # edges of a check are already contiguous, which is what the per-check
    # reductions below rely on.
    rows, cols = np.nonzero(H)
    if rows.size == 0:                       # no checks: nothing to propagate
        hard = np.zeros(n, dtype=np.uint8)
        yield BpIteration(1, prior.copy(), hard, not syndrome.any())
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
            # Divide the row's product of tanh by the edge's own factor, done
            # as a subtraction of logs so the row reduction stays a bincount.
            factor = np.clip(np.tanh(message_to_check / 2),
                             -_TANH_LIMIT, _TANH_LIMIT)
            log_magnitude = np.log(np.abs(factor))
            row_log = np.bincount(rows, weights=log_magnitude,
                                  minlength=n_checks)
            other = np.exp(row_log[rows] - log_magnitude) * sign
            message_to_bit = 2 * np.arctanh(
                np.clip(other, -_TANH_LIMIT, _TANH_LIMIT)) * check_flip

        llr = prior + np.bincount(cols, weights=message_to_bit, minlength=n)
        hard = (llr < 0).astype(np.uint8)
        converged = np.array_equal((H @ hard) % 2, syndrome)
        yield BpIteration(iteration, llr, hard, bool(converged))
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
