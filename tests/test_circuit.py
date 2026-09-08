"""stim으로 지은 circuit-level memory-Z 실험."""
import numpy as np
import pytest

from qec_tile.circuit import (circuit_failure_counts, circuit_failure_rate,
                              memory_z_base, memory_z_circuit)
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
    """잡음이 없으면 detector도 울리지 않고 observable도 뒤집히지 않는다."""
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
    """stim이 detector error model을 만들 수 있다 — 스케줄과 detector가 서로
    맞는다는 뜻이다 (detector 정의가 깨졌으면 여기서 예외가 난다)."""
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
    """모든 qubit이 서로 다른 좌표를 가질 때만 timeslice 그림이 실제 격자를
    그린다."""
    code = paper_code(*SMALL)
    circuit = memory_z_base(code, rounds=1)
    coords = circuit.get_final_qubit_coordinates()
    assert len(coords) == code.n + code.HX.shape[0] + code.HZ.shape[0]
    assert len({tuple(v) for v in coords.values()}) == len(coords)


def test_zero_noise_circuit_never_fails():
    """잡음 없음 -> DEM에 error mechanism 없음 -> 실패할 수 없음."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=2, p=0.0)
    assert circuit_failure_rate(circuit, shots=20, decoder="bposd_cs7",
                                seed=0) == 0.0


def test_failure_counts_count_every_flipped_observable():
    """블록 실패 외에 뒤집힌 observable 수를 직접 센다 -- 독립 가정으로 역산하지 않는다.

    shot 하나가 observable 셋을 뒤집으면 ``logical_flips``에 3이 더해지므로
    ``block_fails <= logical_flips <= k * shots``이고, 같은 seed의
    ``circuit_failure_rate``와 블록 수가 일치한다. 잡음이 없으면 전부 0이다.
    """
    code = paper_code(*SMALL)
    noisy = memory_z_circuit(code, rounds=2, p=0.03)
    counts = circuit_failure_counts(noisy, shots=60, decoder="bposd_cs7",
                                    seed=5)
    assert counts.shots == 60
    assert counts.k == noisy.num_observables == code.k
    assert counts.block_fails <= counts.logical_flips <= counts.k * 60
    assert counts.block_fails > 0                     # p=0.03이면 몇 개는 실패한다
    assert counts.block_fails == round(60 * circuit_failure_rate(
        noisy, shots=60, decoder="bposd_cs7", seed=5))
    clean = circuit_failure_counts(memory_z_circuit(code, rounds=2, p=0.0),
                                   shots=20, decoder="bposd_cs7", seed=0)
    assert (clean.block_fails, clean.logical_flips) == (0, 0)


def test_circuit_rate_is_deterministic_given_a_seed():
    """stim sampler 시드와 디코더가 같으면 결과가 동일하다."""
    code = paper_code(*SMALL)
    circuit = memory_z_circuit(code, rounds=2, p=0.01)
    a = circuit_failure_rate(circuit, shots=50, decoder="bposd_cs7", seed=3)
    b = circuit_failure_rate(circuit, shots=50, decoder="bposd_cs7", seed=3)
    assert a == b


def test_circuit_rate_grows_with_p():
    """배선 확인용 smoke test. 실측: 0.007 대 0.94 — 100배 넘는 여유."""
    code = paper_code(*SMALL)
    low = circuit_failure_rate(memory_z_circuit(code, rounds=2, p=0.002),
                               shots=150, decoder="bposd_cs7", seed=1)
    high = circuit_failure_rate(memory_z_circuit(code, rounds=2, p=0.03),
                                shots=150, decoder="bposd_cs7", seed=1)
    assert low < high


@pytest.mark.slow
def test_circuit_distance_is_not_halved():
    """스케줄이 hook 오류를 만들어 거리를 절반 아래로 떨어뜨리면 안 된다.

    4x4의 b3w6은 부호 거리가 4다(X sector). CNOT 순서가 잘못되면 circuit-level
    거리가 반토막날 수 있다. 동시 결함 2개까지 전수 탐색해 d_circuit >= 3을
    확인한다.
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
