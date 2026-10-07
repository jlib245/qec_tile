"""Coset MRF — logical class 확률을 partition function으로 (arXiv:2608.25545)."""
import numpy as np
import pytest

from scipy.special import logsumexp

from qec_tile.coset import (candidate_labels, class_representatives,
                            coset_energy, exact_log_partition,
                            exact_ml_decode, syndrome_solution)
from qec_tile.decode import make_decoder
from qec_tile.gf2 import nullspace2, rank2
from qec_tile.tile import paper_code

# n=32, k=8, X-check 12개: class당 2^12 구성을 전수 합산할 수 있는 크기.
TINY = ("b3w6", 2, 2)


def test_syndrome_solution_satisfies_syndrome():
    code = paper_code(*TINY)
    rng = np.random.default_rng(0)
    for _ in range(20):
        e = (rng.random(code.n) < 0.1).astype(np.uint8)
        syndrome = (code.HZ @ e) % 2
        solution = syndrome_solution(code.HZ, syndrome)
        assert solution.dtype == np.uint8 and solution.shape == (code.n,)
        assert np.array_equal((code.HZ @ solution) % 2, syndrome)


def test_syndrome_solution_zero_syndrome():
    code = paper_code(*TINY)
    zero = np.zeros(code.HZ.shape[0], dtype=np.uint8)
    assert not syndrome_solution(code.HZ, zero).any()


def test_syndrome_solution_inconsistent_raises():
    """열공간 밖의 syndrome에 조용히 틀린 벡터를 돌려주면 안 된다."""
    HZ = np.array([[1, 1, 0],
                   [1, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError):
        syndrome_solution(HZ, np.array([1, 0], dtype=np.uint8))


def test_candidate_labels_all():
    labels = candidate_labels(8)
    assert labels.shape == (256, 8) and labels.dtype == np.uint8
    assert not labels[0].any()
    assert len({row.tobytes() for row in labels}) == 256


def test_candidate_labels_shift2():
    labels = candidate_labels(8, max_shift=2)
    assert labels.shape == (1 + 8 + 28, 8)           # 1 + k + C(k,2)
    assert not labels[0].any()
    assert (labels.sum(1) <= 2).all()
    assert len({row.tobytes() for row in labels}) == len(labels)


def test_class_representatives_are_distinct_classes():
    """대표원은 모두 같은 syndrome을 갖되 서로 다른 logical class에 있어야 한다."""
    code = paper_code(*TINY)
    LX, LZ = code.logicals()
    rng = np.random.default_rng(3)
    e = (rng.random(code.n) < 0.1).astype(np.uint8)
    syndrome = (code.HZ @ e) % 2
    reps = class_representatives(syndrome_solution(code.HZ, syndrome), LX,
                                 candidate_labels(code.k))
    assert reps.shape == (2 ** code.k, code.n) and reps.dtype == np.uint8
    assert np.array_equal((reps @ code.HZ.T) % 2,
                          np.tile(syndrome, (len(reps), 1)))
    # LX·LZ 짝이 비퇴화이므로 LZ r_λ 는 λ 의 전단사다 -- 전부 달라야 한다
    classes = (reps @ LZ.T) % 2
    assert len({row.tobytes() for row in classes}) == 2 ** code.k


def test_coset_energy_matches_direct():
    code = paper_code(*TINY)
    rng = np.random.default_rng(4)
    r = (rng.random(code.n) < 0.2).astype(np.uint8)
    u = (rng.random((10, code.HX.shape[0])) < 0.5).astype(np.uint8)
    energies = coset_energy(code.HX, r, u)
    assert energies.shape == (10,)
    for row, energy in zip(u, energies):
        assert energy == int((r ^ ((row @ code.HX) % 2)).sum())


def test_exact_partition_by_hand():
    """coset {1000, 0100, 1011, 0111} 이므로 Z = 2w + 2w^3."""
    HX = np.array([[1, 1, 0, 0],
                   [0, 0, 1, 1]], dtype=np.uint8)
    r = np.array([1, 0, 0, 0], dtype=np.uint8)
    w = 0.1
    assert exact_log_partition(HX, r, w) == pytest.approx(
        np.log(2 * w + 2 * w ** 3), abs=1e-12)


def test_exact_partition_sums_over_syndrome_solutions():
    """Σ_λ Z_λ 는 syndrome을 만족하는 모든 오류의 확률 합이어야 한다."""
    code = paper_code(*TINY)
    assert rank2(code.HX) == code.HX.shape[0]     # u ↔ stabilizer 일대일 전제
    LX, _ = code.logicals()
    w = 0.05
    rng = np.random.default_rng(5)
    e = (rng.random(code.n) < 0.1).astype(np.uint8)
    syndrome = (code.HZ @ e) % 2
    e0 = syndrome_solution(code.HZ, syndrome)

    reps = class_representatives(e0, LX, candidate_labels(code.k))
    by_classes = logsumexp(exact_log_partition(code.HX, reps, w))

    kernel = nullspace2(code.HZ)                  # e0 ⊕ ker(HZ) 전수 열거
    solutions = e0 ^ ((candidate_labels(len(kernel)) @ kernel) % 2)
    direct = logsumexp(solutions.sum(1) * np.log(w))
    assert by_classes == pytest.approx(direct, abs=1e-9)


def test_exact_ml_decode_returns_best_class():
    code = paper_code(*TINY)
    LX, LZ = code.logicals()
    p = 0.05
    w = p / (1 - p)
    labels = candidate_labels(code.k)
    rng = np.random.default_rng(6)
    for _ in range(20):
        e = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = (code.HZ @ e) % 2
        e0 = syndrome_solution(code.HZ, syndrome)
        correction, log_Z = exact_ml_decode(code.HX, LX, e0, w, labels)
        assert np.array_equal((code.HZ @ correction) % 2, syndrome)
        assert log_Z.shape == (2 ** code.k,)
        best = class_representatives(e0, LX, labels[np.argmax(log_Z)])
        assert np.array_equal((LZ @ correction) % 2, (LZ @ best[0]) % 2)


def test_restricted_candidates_reach_the_optimum():
    """논문 8.5절: BP+OSD 해 주변 logical 두 개 이내에 전역 최적이 있다.

    d=2 코드라 정확히 동률인 class가 흔하므로 라벨이 아니라 log Z 값을 비교한다.
    """
    code = paper_code(*TINY)
    LX, _ = code.logicals()
    p = 0.05
    w = p / (1 - p)
    every = candidate_labels(code.k)
    nearby = candidate_labels(code.k, max_shift=2)
    bposd = make_decoder(code.HZ, p)
    rng = np.random.default_rng(7)
    misses = 0
    for _ in range(200):
        e = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        base = bposd.decode(syndrome).astype(np.uint8)
        _, log_Z_all = exact_ml_decode(code.HX, LX, base, w, every)
        _, log_Z_near = exact_ml_decode(code.HX, LX, base, w, nearby)
        if log_Z_all.max() - log_Z_near.max() > 1e-9:
            misses += 1
    assert misses == 0
