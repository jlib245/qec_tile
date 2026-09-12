"""시드로 재현되는 병렬 shot 수집.

sinter는 ``compile_detector_sampler()``에 시드를 넘길 자리가 없어서 실행마다 다른
shot을 뽑는다. 이 모듈은 그것을 대체하므로, 재현성과 worker 수 독립성이 전부다.
"""
import stim

from qec_tile.parallel_sampling import _chunk_seed, parallel_failure_counts


def small_circuit():
    """초 단위로 끝나는 회로. d=3, 3라운드."""
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=3, rounds=3,
                                  after_clifford_depolarization=0.02)


def test_counts_are_reproducible():
    """같은 인자면 카운트가 완전히 같다 — sinter가 못 하는 것."""
    circuit = small_circuit()
    first = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                    workers=2, chunk=50)
    second = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                     workers=2, chunk=50)
    assert first == second


def test_the_worker_count_does_not_change_the_data():
    """worker 수는 결과에 들어가지 않는다.

    조각을 worker 수로 나누면 코어 수가 데이터를 바꿔 머신 간 재현이 깨진다.
    고정 ``chunk``를 쓰는 설계가 여기에 걸려 있다.
    """
    circuit = small_circuit()
    few = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                  workers=2, chunk=50)
    many = parallel_failure_counts(circuit, 200, "bposd_0", seed=3,
                                   workers=4, chunk=50)
    assert few == many


def test_chunk_seeds_are_distinct():
    """조각마다, 그리고 시드마다 다른 샘플러 시드.

    모든 조각이 같은 시드를 쓰면 같은 shot을 조각 수만큼 반복해 통계가 망가진다 --
    에러 없이 조용히 틀리는 종류다. 카운트 비교로 테스트하면 우연히 일치해 flaky해
    지므로 시드 유도 자체를 본다.
    """
    assert _chunk_seed(3, 0) != _chunk_seed(4, 0)
    assert _chunk_seed(3, 0) != _chunk_seed(3, 1)


def test_max_errors_stops_at_a_chunk_boundary():
    """조각을 인덱스 순서대로 소비하며 멈추므로 멈추는 지점이 결정적이다.

    완료 순서대로 멈추면 총 shot이 실행마다 달라진다 (sinter에서 26,403과 33,708로
    갈린 원인).
    """
    circuit = small_circuit()
    first = parallel_failure_counts(circuit, 2000, "bposd_0", seed=5,
                                    workers=4, chunk=50, max_errors=3)
    second = parallel_failure_counts(circuit, 2000, "bposd_0", seed=5,
                                     workers=2, chunk=50, max_errors=3)
    assert first == second
    assert first.shots % 50 == 0 or first.shots == 2000
    assert first.block_fails >= 3
