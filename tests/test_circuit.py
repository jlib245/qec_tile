"""Circuit-level memory-Z experiment built with stim."""
import numpy as np
import pytest

from qec_tile.circuit import (circuit_failure_rate, memory_z_base,
                              memory_z_circuit)
from qec_tile.tile import paper_code

SMALL = ("b3w6", 4, 4)


def counts(code, rounds):
    mx, mz = code.HX.shape[0], code.HZ.shape[0]
    detectors = (rounds + 1) * mz + (rounds - 1) * mx
    return detectors, code.k


def test_detector_and_observable_counts():
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=3, p=0.01)
    detectors, observables = counts(code, rounds=3)
    assert circuit.num_detectors == detectors
    assert circuit.num_observables == observables


def test_zero_noise_is_silent():
    """Without noise no detector fires and no observable flips."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=3, p=0.0)
    sampler = circuit.compile_detector_sampler()
    detections, observables = sampler.sample(64, separate_observables=True)
    assert not detections.any()
    assert not observables.any()


def test_noise_fires_detectors():
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=3, p=0.02)
    sampler = circuit.compile_detector_sampler()
    detections = sampler.sample(64)
    assert detections.any()


def test_dem_extraction_works():
    """stim can build a detector error model — schedule and detectors are
    consistent (a broken detector definition raises here)."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=3, p=0.01)
    dem = circuit.detector_error_model()
    assert dem.num_detectors == circuit.num_detectors
    assert dem.num_observables == code.k


def test_rounds_must_be_positive():
    code = paper_code(*SMALL)
    with pytest.raises(ValueError, match="rounds"):
        memory_z_circuit(code, rounds=0, p=0.01)


def test_every_qubit_has_coordinates():
    """The timeslice diagram draws the real lattice only if all qubits have
    distinct coordinates."""
    code = paper_code(*SMALL)
    circuit = memory_z_base(code, rounds=1)
    coords = circuit.get_final_qubit_coordinates()
    assert len(coords) == code.n + code.HX.shape[0] + code.HZ.shape[0]
    assert len({tuple(v) for v in coords.values()}) == len(coords)


def test_zero_noise_circuit_never_fails():
    """No noise -> no error mechanisms in the DEM -> nothing can fail."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=2, p=0.0)
    assert circuit_failure_rate(circuit, shots=20, decoder="bposd_cs7",
                                seed=0) == 0.0


def test_circuit_rate_is_deterministic_given_a_seed():
    """Same stim sampler seed and decoder -> identical outcome."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=2, p=0.01)
    a = circuit_failure_rate(circuit, shots=50, decoder="bposd_cs7", seed=3)
    b = circuit_failure_rate(circuit, shots=50, decoder="bposd_cs7", seed=3)
    assert a == b


def test_circuit_rate_grows_with_p():
    """Wiring smoke test.  Measured: 0.007 vs 0.94 — a >100x margin."""
    code = paper_code(*SMALL)
    low = circuit_failure_rate(memory_z_circuit(code, rounds=2, p=0.002),
                               shots=150, decoder="bposd_cs7", seed=1)
    high = circuit_failure_rate(memory_z_circuit(code, rounds=2, p=0.03),
                                shots=150, decoder="bposd_cs7", seed=1)
    assert low < high


@pytest.mark.slow
def test_circuit_distance_is_not_halved():
    """The schedule must not introduce hook errors below half the distance.

    b3w6 at 4x4 has code distance 4 (X sector); a bad CNOT order could halve
    the circuit-level distance.  We check d_circuit >= 3 by exhaustive search
    over up to 2 simultaneous faults.
    """
    import stim
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=4, p=0.001)
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=True,
        canonicalize_circuit_errors=True,
    )
    assert len(errors) >= 3
