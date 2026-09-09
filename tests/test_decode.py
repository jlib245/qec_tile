"""BP+OSD로 하는 code-capacity 디코딩."""
import numpy as np
import pytest
from ldpc import BpDecoder

from qec_tile.decode import (DECODERS, VibeLsdDecoder, code_capacity_counts,
                             code_capacity_block_rate, make_decoder,
                             per_logical_rate, sample_residuals)
from qec_tile.tile import paper_code

SMALL = ("b3w6", 5, 5)


def test_zero_noise_never_fails():
    rate = code_capacity_block_rate(paper_code(*SMALL), p=0.0, shots=200,
                              decoder="bposd_cs7", seed=0)
    assert rate == 0.0


def test_correction_always_matches_syndrome():
    """BP+OSD가 무엇을 돌려주든, residual은 Z-syndrome이 0이어야 한다."""
    code = paper_code(*SMALL)
    for residual in sample_residuals(code, p=0.08, shots=100,
                                     decoder="bposd_cs7", seed=1):
        assert not ((code.HZ @ residual) % 2).any()


def test_single_errors_are_always_corrected():
    """d >= 3이므로 weight 1인 오류는 유일하게 디코딩된다."""
    code = paper_code(*SMALL)
    decoder = make_decoder(code.HZ, 0.05)
    _, LZ = code.logicals()
    for i in range(code.n):
        e = np.zeros(code.n, dtype=np.uint8)
        e[i] = 1
        ehat = decoder.decode(((code.HZ @ e) % 2).astype(np.uint8))
        residual = (e ^ ehat) % 2
        assert not ((LZ @ residual) % 2).any()


def test_rate_grows_with_p():
    code = paper_code(*SMALL)
    low = code_capacity_block_rate(code, p=0.02, shots=800,
                                   decoder="bposd_cs7", seed=2)
    high = code_capacity_block_rate(code, p=0.12, shots=800,
                                    decoder="bposd_cs7", seed=2)
    assert low < high


def test_seed_is_deterministic():
    code = paper_code(*SMALL)
    a = code_capacity_block_rate(code, p=0.08, shots=300,
                                 decoder="bposd_cs7", seed=7)
    b = code_capacity_block_rate(code, p=0.08, shots=300,
                                 decoder="bposd_cs7", seed=7)
    assert a == b


def test_rate_is_a_fraction():
    rate = code_capacity_block_rate(paper_code(*SMALL), p=0.1, shots=500,
                                    decoder="bposd_cs7", seed=3)
    assert 0.0 <= rate <= 1.0


def test_decoder_is_required():
    """기본값 없음: 어떤 디코더가 만든 결과인지 실행이 밝혀야 한다."""
    with pytest.raises(TypeError):
        code_capacity_block_rate(paper_code(*SMALL), p=0.05, shots=10)


def test_all_decoders_are_registered():
    assert set(DECODERS) == {"bposd_cs7", "bposd_0", "bplsd_0", "bplsd_cs7",
                             "vibelsd_32", "vibelsd_200"}


# --- VibeLSD -----------------------------------------------------------------

def test_vibelsd_holds_one_bp_decoder():
    """앙상블이 커져도 ``BpDecoder``는 하나다 -- 멤버 차이는 ``orders``에만 있다.

    멤버마다 객체를 두면 각자 sparse 행렬 사본을 들어 worker 메모리가 앙상블 배로
    커진다. 5x3 walk DEM(270 x 9632, nnz 61337)에서 실측 4.20 MB/개라 200개면
    839 MB, sinter worker 24개면 19.7 GB -- 실제로 ``MemoryError: bad allocation``
    이 났다.
    """
    code = paper_code(*SMALL)
    decoder = VibeLsdDecoder(code.HZ, 0.05, ensemble=8, seed=0)
    assert decoder.orders.shape == (8, code.n)
    assert not hasattr(decoder, "members")


def test_vibelsd_members_use_distinct_serial_schedules():
    """앙상블 멤버는 error mechanism의 순열이 서로 다른 직렬 스케줄이다.

    순열을 안 넘기면 전부 같은 스케줄이라 앙상블이 BP 하나와 같아진다. 시드가
    같으면 순열이 재현되어 결과가 결정적이다.
    """
    code = paper_code(*SMALL)
    decoder = VibeLsdDecoder(code.HZ, 0.05, ensemble=8, seed=0)
    orders = [tuple(order) for order in decoder.orders]
    assert len(orders) == 8
    assert len(set(orders)) == 8
    assert all(sorted(order) == list(range(code.n)) for order in orders)
    again = VibeLsdDecoder(code.HZ, 0.05, ensemble=8, seed=0)
    assert [tuple(order) for order in again.orders] == orders


def test_vibelsd_shared_bp_matches_separate_decoders():
    """순열을 갈아끼운 ``BpDecoder`` 하나 == 순열마다 만든 ``BpDecoder`` 여러 개.

    옵션 1(앙상블이 객체 하나를 공유)의 전제를 못박는다. 깨질 수 있는 곳이 둘이고
    둘 다 조용히 틀린다:
      - ``decode``가 앞 호출의 메시지를 물려받으면 답이 호출 순서에 오염된다
      - ``serial_schedule_order`` setter가 내부 구조를 다시 안 잡으면 옛 순서로 돈다

    4x2 walk DEM(170 x 5568, nnz 31534)에서 순열 16개 x syndrome 60개를 실측:
    불일치 0/60, 답이 갈린 syndrome 60/60(동치성이 공허하지 않다), 순열을 3 -> 7 ->
    3으로 되돌렸을 때 ``llr_max_diff`` 0.00e+00 (warm start 없음).

    ldpc의 동작을 고정하는 테스트라 구현 전에도 통과한다.
    """
    code = paper_code(*SMALL)
    orders = [list(np.random.default_rng(k).permutation(code.n))
              for k in range(4)]
    kwargs = dict(error_channel=[0.08] * code.n, bp_method="minimum_sum",
                  max_iter=15, schedule="serial")
    separate = [BpDecoder(code.HZ, serial_schedule_order=order, **kwargs)
                for order in orders]
    shared = BpDecoder(code.HZ, serial_schedule_order=orders[0], **kwargs)

    rng = np.random.default_rng(0)
    for _ in range(20):
        e = (rng.random(code.n) < 0.08).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        for order, member in zip(orders, separate):
            expected = member.decode(syndrome).copy()
            converged = member.converge
            shared.serial_schedule_order = order
            assert np.array_equal(shared.decode(syndrome), expected)
            assert shared.converge == converged


def test_vibelsd_budget_matches_running_every_member_fully():
    """예산 깎기 == L개를 full depth로 돌려 iter 최소 M개를 고른 것.

    예산이 너무 세게 조여 나중에 더 빨리 수렴할 멤버가 잘리면 갈린다 -- 에러 없이
    후보만 나빠지는 실패다. 근거: 이미 M개가 c <= t에 수렴해 있으면 t 안에 수렴
    못 하는 멤버는 c > t라 상위 M개에 못 든다.
    """
    code = paper_code(*SMALL)
    decoder = VibeLsdDecoder(code.HZ, 0.08, ensemble=16, converged=3, seed=2)
    rng = np.random.default_rng(3)
    for _ in range(30):
        e = (rng.random(code.n) < 0.08).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)

        naive = []                       # 예산 없이 전부 끝까지
        for order in decoder.orders:
            decoder.bp.serial_schedule_order = order
            decoder.bp.max_iter = decoder.max_iter
            candidate = decoder.bp.decode(syndrome)
            if decoder.bp.converge:
                naive.append((decoder.bp.iter,
                              float(decoder.log_weight @ candidate),
                              candidate.copy()))
        naive.sort(key=lambda c: c[0])
        del naive[decoder.converged:]

        got = decoder.decode(syndrome)
        if naive:
            assert np.array_equal(got, min(naive, key=lambda c: c[1])[2])
        assert (((code.HZ @ got) % 2) == syndrome).all()


def test_vibelsd_correction_matches_the_syndrome():
    """무엇을 돌려주든 ``H @ e_hat == s`` -- 수렴한 후보든 LSD fallback이든."""
    code = paper_code(*SMALL)
    decoder = VibeLsdDecoder(code.HZ, 0.08, ensemble=8, seed=1)
    rng = np.random.default_rng(1)
    for _ in range(100):
        e = (rng.random(code.n) < 0.08).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        e_hat = decoder.decode(syndrome)
        assert (((code.HZ @ e_hat) % 2) == syndrome).all()


def test_vibelsd_corrects_single_errors():
    """d >= 3이므로 weight 1인 오류는 유일하게 디코딩된다."""
    code = paper_code(*SMALL)
    decoder = VibeLsdDecoder(code.HZ, 0.05, ensemble=8, seed=0)
    _, LZ = code.logicals()
    for i in range(code.n):
        e = np.zeros(code.n, dtype=np.uint8)
        e[i] = 1
        ehat = decoder.decode(((code.HZ @ e) % 2).astype(np.uint8))
        assert not ((LZ @ ((e ^ ehat) % 2)) % 2).any()


def test_unknown_decoder_is_rejected():
    with pytest.raises(ValueError, match="unknown decoder"):
        code_capacity_block_rate(paper_code(*SMALL), p=0.05, shots=10,
                                 decoder="nn")


def test_decoder_prior_matches_the_sampling_rate():
    """BP는 참 오류율을 prior로 받아야 한다. make_decoder가 둘을 묶는다."""
    code = paper_code(*SMALL)
    decoder = make_decoder(code.HZ, 0.07)
    assert np.allclose(decoder.error_channel, 0.07)


def test_make_decoder_accepts_a_channel_vector():
    """열별 prior — p와 q가 섞인 시공간 행렬용."""
    code = paper_code(*SMALL)
    channel = np.full(code.n, 0.03)
    channel[0] = 0.11
    decoder = make_decoder(code.HZ, channel)
    assert np.allclose(decoder.error_channel, channel)


def test_a_wrong_prior_decodes_worse():
    """BP에 틀린 오류율을 주면 성능이 나빠진다 — prior를 p에 묶는 근거."""
    code = paper_code(*SMALL)
    _, LZ = code.logicals()
    p = 0.1

    def failures(prior):
        rng = np.random.default_rng(0)
        decoder = make_decoder(code.HZ, prior)
        count = 0
        for _ in range(1500):
            e = (rng.random(code.n) < p).astype(np.uint8)
            e_hat = decoder.decode(((code.HZ @ e) % 2).astype(np.uint8))
            count += bool(((LZ @ ((e ^ e_hat) % 2)) % 2).any())
        return count

    # 여기서 0.5는 일부러 틀리게 준 prior이고, 라이브러리 기본값이 아니다.
    assert failures(p) < failures(0.5)     # 맞는 prior가 크게 어긋난 것보다 낫다


def test_a_single_logical_is_unchanged():
    """k=1이면 항등이어야 한다 — 지수에서 k와 1/k를 뒤바꾼 것을 잡는다."""
    assert per_logical_rate(0.037, 1) == pytest.approx(0.037)


@pytest.mark.parametrize("k", [1, 2, 8, 12])
def test_round_trip_back_to_the_block_rate(k):
    """1-(1-single)^k는 변환이 출발한 블록 비율로 돌아온다."""
    block = 0.2
    single = per_logical_rate(block, k)
    assert 1 - (1 - single) ** k == pytest.approx(block)


def test_small_rates_divide_by_k():
    """1차 근사로 변환은 block/k다. k=8에서 1e-4 -> 1.25e-5.

    k로 나누는 대신 곱하는 변환을 잡는다.
    """
    assert per_logical_rate(1e-4, 8) == pytest.approx(1.25e-5, rel=1e-2)


@pytest.mark.parametrize("block", [0.0, 1.0])
def test_the_boundaries_are_fixed(block):
    """절대 안 실패하면 0, 항상 실패하면 1이고, k와 무관하다.

    log 기반 구현은 양 끝에서 터진다. 여기서 그것을 못박는다.
    """
    assert per_logical_rate(block, 8) == block


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_at_least_one(k):
    """k=0은 0으로 나누고 k<0은 코드가 아니다 — 둘 다 거부한다."""
    with pytest.raises(ValueError, match="k"):
        per_logical_rate(0.1, k)


@pytest.mark.parametrize("block", [-0.01, 1.5])
def test_the_block_rate_must_be_a_probability(block):
    with pytest.raises(ValueError, match="block_rate"):
        per_logical_rate(block, 8)


# --- logical qubit당 계수 ---------------------------------------------------
# b3w6 5x5에서 p=0.1이면 실패한 shot이 충분히 남아 아래 부등식들이 공허하지 않다.
# test_counts_are_not_vacuous가 그것을 못박는다.
COUNT_ARGS = dict(p=0.1, shots=400, decoder="bposd_cs7", seed=11)


@pytest.fixture
def counts():
    return code_capacity_counts(paper_code(*SMALL), **COUNT_ARGS)


def test_counts_agree_with_the_rate(counts):
    """리팩터가 이미 기록된 벤치마크 숫자를 움직여서는 안 된다."""
    rate = code_capacity_block_rate(paper_code(*SMALL), **COUNT_ARGS)
    assert counts.block_fails / counts.shots == rate


def test_counts_are_not_vacuous(counts):
    """아래 부등식들의 방패 — 실패가 0이면 전부 자명하게 성립한다."""
    assert counts.block_fails > 0


def test_flips_are_never_fewer_than_failed_shots(counts):
    """실패한 shot은 최소 하나를 뒤집는다 — .sum() 대신 .any()를 쓴 것을 잡는다."""
    assert counts.logical_flips >= counts.block_fails


def test_flips_cannot_exceed_k_per_shot(counts):
    """shot당 최대 k개가 뒤집힌다. 그보다 많으면 집계가 샜다."""
    assert counts.logical_flips <= counts.block_fails * counts.k


def test_per_logical_is_at_most_the_block_rate(counts):
    """위 두 경계에서 따라온다 — k로 나누는 것을 빼먹은 것을 잡는다."""
    per_logical = counts.logical_flips / (counts.shots * counts.k)
    assert per_logical <= counts.block_fails / counts.shots


def test_zero_noise_flips_nothing():
    counts = code_capacity_counts(paper_code(*SMALL), p=0.0, shots=50,
                                  decoder="bposd_cs7", seed=0)
    assert (counts.block_fails, counts.logical_flips) == (0, 0)


def test_k_is_the_number_of_observables(counts):
    """counts.k는 코드의 k이므로 어느 쪽으로 나눠도 같은 비율이 나온다."""
    code = paper_code(*SMALL)
    _, LZ = code.logicals()
    assert counts.k == LZ.shape[0] == code.k
