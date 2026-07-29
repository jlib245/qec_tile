"""Belief propagation — cross-checked against ldpc, iteration by iteration.

The point of a local BP is the trace: ldpc hands back only the final state, so
there is no way to watch beliefs move.  These pin the trace to ldpc's numbers
for both schedules it names, ``minimum_sum`` and ``product_sum``, so the two
stay the same decoder.
"""
import numpy as np
import pytest
from ldpc import BpDecoder

from qec_tile.bp import METHODS, bp, bp_trace
from qec_tile.tile import paper_code

P = 0.05


def syndrome_of(code, qubits):
    error = np.zeros(code.n, dtype=np.uint8)
    error[qubits] = 1
    return ((code.HZ @ error) % 2).astype(np.uint8)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,qubits", [("b3w6", [4, 20]), ("b4w8", [3])])
def test_llr_matches_ldpc_each_iteration(name, qubits, method):
    """Same priors, same schedule, same numbers — to floating point.

    ldpc freezes its state once BP converges, so comparing up to our own stop
    point is the whole overlap.  Which iteration each side calls "converged"
    can differ by one: ldpc takes its hard decision mid-iteration, so its
    ``iter`` counts one later than sign(posterior) does.
    """
    code = paper_code(name, 3, 3)
    syndrome = syndrome_of(code, qubits)
    for step in bp_trace(code.HZ, syndrome, P, method=method, max_iter=6):
        reference = BpDecoder(code.HZ.astype(np.uint8), error_rate=P,
                              bp_method=method, max_iter=step.iteration,
                              ms_scaling_factor=1.0)
        reference.decode(syndrome)
        assert np.allclose(step.llr, np.asarray(reference.log_prob_ratios)), \
            (method, step.iteration)


@pytest.mark.parametrize("method", METHODS)
def test_zero_syndrome_never_flags_a_qubit(method):
    """With no checks lit every message is positive, so belief only grows.

    A sign slip in the check-node update (the syndrome flip, or the product of
    signs) shows up here as a bit driven negative out of nothing.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = np.zeros(code.HZ.shape[0], dtype=np.uint8)
    prior = np.log((1 - P) / P)
    steps = list(bp_trace(code.HZ, syndrome, P, method=method, max_iter=4))
    assert steps[0].converged                      # the zero error solves it
    assert len(steps) == 1
    assert steps[0].hard.sum() == 0
    assert (steps[0].llr >= prior - 1e-9).all()


@pytest.mark.parametrize("method", METHODS)
def test_trace_stops_at_convergence(method):
    """Once H @ hard == syndrome there is nothing left to propagate."""
    code = paper_code("b4w8", 3, 3)
    syndrome = syndrome_of(code, [3])
    steps = list(bp_trace(code.HZ, syndrome, P, method=method, max_iter=50))
    assert steps[-1].converged
    assert not any(step.converged for step in steps[:-1])
    hard, _llr, converged_at = bp(code.HZ, syndrome, P, method=method,
                                  max_iter=50)
    assert converged_at == len(steps)
    assert np.array_equal((code.HZ @ hard) % 2, syndrome)


@pytest.mark.parametrize("method", METHODS)
def test_channel_accepts_a_vector(method):
    """A per-column prior must reduce to the scalar one when it is flat.

    decode.py already takes both shapes (a space-time channel mixes data and
    measurement rates), so BP has to as well.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = syndrome_of(code, [4, 20])
    flat = list(bp_trace(code.HZ, syndrome, np.full(code.n, P), method=method,
                         max_iter=4))
    scalar = list(bp_trace(code.HZ, syndrome, P, method=method, max_iter=4))
    assert len(flat) == len(scalar)
    for from_vector, from_scalar in zip(flat, scalar):
        assert np.allclose(from_vector.llr, from_scalar.llr)


def test_unknown_method_is_rejected():
    code = paper_code("b3w6", 3, 3)
    with pytest.raises(ValueError):
        list(bp_trace(code.HZ, syndrome_of(code, [4]), P, method="magic"))


def test_the_approximation_costs_a_shot():
    """Min-sum is an approximation, and here is what it costs.

    On b4w10 L=3 with errors on qubits 5 and 17, sum-product converges at
    iteration 3 while min-sum is still oscillating after 50 — measured, not
    assumed.  The beliefs differ from the very first sweep, which also pins
    that the two methods are not the same code path.
    """
    code = paper_code("b4w10", 3, 3)
    syndrome = syndrome_of(code, [5, 17])
    assert bp(code.HZ, syndrome, P, method="product_sum")[2] == 3
    assert bp(code.HZ, syndrome, P, method="minimum_sum")[2] is None

    beliefs = [next(iter(bp_trace(code.HZ, syndrome, P, method=method))).llr
               for method in METHODS]
    assert not np.allclose(*beliefs)
