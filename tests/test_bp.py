"""Belief propagation — ldpc와 반복마다 교차 검증.

로컬 BP를 따로 두는 이유는 trace다: ldpc는 최종 상태만 돌려주므로 belief가 움직이는
것을 볼 방법이 없다. 여기서는 ldpc가 이름 붙인 두 스케줄 ``minimum_sum``과
``product_sum``에 대해 trace를 ldpc의 숫자에 못박아, 둘이 같은 디코더로 남게 한다.
"""
import numpy as np
import pytest
from ldpc import BpDecoder

from qec_tile.bp import METHODS, bp, bp_trace, tanner_edges
from qec_tile.tile import paper_code

P = 0.05


def syndrome_of(code, qubits):
    error = np.zeros(code.n, dtype=np.uint8)
    error[qubits] = 1
    return ((code.HZ @ error) % 2).astype(np.uint8)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,qubits", [("b3w6", [4, 20]), ("b4w8", [3])])
def test_llr_matches_ldpc_each_iteration(name, qubits, method):
    """prior가 같고 스케줄이 같으면 숫자도 같다 — 부동소수점 수준으로.

    ldpc는 BP가 수렴하면 상태를 얼려버리므로, 우리 쪽 정지 지점까지만 비교하는 것이
    겹치는 전부다. 어느 반복을 "수렴"이라 부르는지는 한 번 차이날 수 있다: ldpc는
    반복 도중에 hard decision을 하므로 그 ``iter``가 sign(posterior)보다 하나 늦게
    센다.
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
    """켜진 check가 없으면 모든 메시지가 양수이므로 belief는 커지기만 한다.

    check-node 갱신의 부호 실수(syndrome 뒤집기, 또는 부호의 곱)는 여기서 아무 근거
    없이 음수로 끌려가는 비트로 드러난다.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = np.zeros(code.HZ.shape[0], dtype=np.uint8)
    prior = np.log((1 - P) / P)
    steps = list(bp_trace(code.HZ, syndrome, P, method=method, max_iter=4))
    assert steps[0].converged                      # 오류가 0인 것이 답이다
    assert len(steps) == 1
    assert steps[0].hard.sum() == 0
    assert (steps[0].llr >= prior - 1e-9).all()


@pytest.mark.parametrize("method", METHODS)
def test_trace_stops_at_convergence(method):
    """H @ hard == syndrome이 되면 더 전달할 것이 없다."""
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
    """열별 prior가 평평하면 스칼라 prior로 환원되어야 한다.

    decode.py는 이미 두 모양을 다 받는다(시공간 채널은 data 오류율과 측정 오류율을
    섞는다). 그러니 BP도 받아야 한다.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = syndrome_of(code, [4, 20])
    flat = list(bp_trace(code.HZ, syndrome, np.full(code.n, P), method=method,
                         max_iter=4))
    scalar = list(bp_trace(code.HZ, syndrome, P, method=method, max_iter=4))
    assert len(flat) == len(scalar)
    for from_vector, from_scalar in zip(flat, scalar):
        assert np.allclose(from_vector.llr, from_scalar.llr)


@pytest.mark.parametrize("method", METHODS)
def test_messages_have_one_entry_per_edge(method):
    """메시지는 Tanner edge 위에 살고, 순서는 tanner_edges가 보고하는 그대로다."""
    code = paper_code("b3w6", 3, 3)
    rows, cols = tanner_edges(code.HZ)
    assert rows.size == int(code.HZ.sum())
    for step in bp_trace(code.HZ, syndrome_of(code, [4, 20]), P,
                         method=method, max_iter=3):
        assert step.to_check.shape == rows.shape
        assert step.to_bit.shape == rows.shape


@pytest.mark.parametrize("method", METHODS)
def test_first_sweep_sends_the_prior(method):
    """v-node는 아직 들은 게 없으므로 첫 메시지가 자기 prior다."""
    code = paper_code("b3w6", 3, 3)
    _rows, cols = tanner_edges(code.HZ)
    first = next(iter(bp_trace(code.HZ, syndrome_of(code, [4, 20]), P,
                               method=method, max_iter=3)))
    assert np.allclose(first.to_check, np.log((1 - P) / P))
    assert first.to_check.shape == cols.shape


def test_check_message_is_the_min_over_the_others():
    """min-sum 규칙을 여기서 느리고 뻔한 방식으로 다시 계산한다.

    그 edge 자신의 메시지를 제외하는 것이 핵심이다: c-node는 v-node가 방금 자기에게
    한 말을 그 v-node에게 되돌려주면 안 된다.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = syndrome_of(code, [4, 20])
    rows, _cols = tanner_edges(code.HZ)
    step = list(bp_trace(code.HZ, syndrome, P, method="minimum_sum",
                         max_iter=3))[-1]
    for edge in range(rows.size):
        others = [other for other in range(rows.size)
                  if rows[other] == rows[edge] and other != edge]
        magnitude = min(abs(step.to_check[other]) for other in others)
        sign = np.prod([np.sign(step.to_check[other]) for other in others])
        flip = -1 if syndrome[rows[edge]] else 1
        assert np.isclose(step.to_bit[edge], flip * sign * magnitude), edge


@pytest.mark.parametrize("method", METHODS)
def test_posterior_is_the_prior_plus_what_came_in(method):
    code = paper_code("b3w6", 3, 3)
    _rows, cols = tanner_edges(code.HZ)
    for step in bp_trace(code.HZ, syndrome_of(code, [4, 20]), P,
                         method=method, max_iter=3):
        incoming = np.bincount(cols, weights=step.to_bit, minlength=code.n)
        assert np.allclose(step.llr, np.log((1 - P) / P) + incoming)


@pytest.mark.parametrize("method", METHODS)
def test_next_sweep_subtracts_the_message_it_answered(method):
    """다음 sweep의 v -> c는 posterior에서 그 edge 자신의 답을 뺀 것이고, 이것이
    belief가 두 번 세어지지 않게 한다."""
    code = paper_code("b3w6", 3, 3)
    _rows, cols = tanner_edges(code.HZ)
    steps = list(bp_trace(code.HZ, syndrome_of(code, [4, 20]), P,
                          method=method, max_iter=4))
    for earlier, later in zip(steps, steps[1:]):
        assert np.allclose(later.to_check, earlier.llr[cols] - earlier.to_bit)


def test_strong_beliefs_do_not_saturate():
    """tanh의 곱은 심하게 잘린다: tanh(m/2)는 double 정밀도에서 m = 37 근처에 1.0에
    닿는다(그 지점에서 1 - tanh(m/2) ~ 2 exp(-m)이 eps 밑으로 떨어진다). 그다음부터는
    더 강한 belief가 전부 2 atanh(1 - 1e-12) = 26.71로 되돌아온다. Gallager의
    involution phi(x) = -log tanh(x/2)는 그 bias의 -log를 담으므로 부동소수점에
    여유가 있다. p = 1e-18에서 prior는 41.45이고 weight-6 check가 39.84로 답한다.
    """
    code = paper_code("b3w6", 3, 3)
    syndrome = np.zeros(code.HZ.shape[0], dtype=np.uint8)
    step = next(iter(bp_trace(code.HZ, syndrome, 1e-18, method="product_sum",
                              max_iter=1)))
    assert np.abs(step.to_bit).min() > 35


@pytest.mark.parametrize("method", METHODS)
def test_messages_stay_finite(method):
    """phi(0)은 무한이고, 무한들의 행 합에서 그중 하나를 빼면 nan이다 -- 제외 연산이
    0 메시지를 견뎌야 하고 행을 오염시켜서는 안 된다."""
    code = paper_code("b3w6", 3, 3)
    syndrome = syndrome_of(code, [4, 20])
    for step in bp_trace(code.HZ, syndrome, 1e-12, method=method, max_iter=20):
        assert np.isfinite(step.to_bit).all(), step.iteration
        assert np.isfinite(step.llr).all(), step.iteration


def test_unknown_method_is_rejected():
    code = paper_code("b3w6", 3, 3)
    with pytest.raises(ValueError):
        list(bp_trace(code.HZ, syndrome_of(code, [4]), P, method="magic"))


def test_the_approximation_costs_a_shot():
    """min-sum은 근사이고, 그 대가가 여기 있다.

    b4w10 L=3에서 qubit 5와 17에 오류를 두면 sum-product는 반복 3에서 수렴하는데
    min-sum은 50번을 넘겨도 진동한다 — 가정이 아니라 실측이다. belief는 첫 sweep부터
    다르고, 이것이 두 방법이 같은 코드 경로가 아니라는 것도 못박는다.
    """
    code = paper_code("b4w10", 3, 3)
    syndrome = syndrome_of(code, [5, 17])
    assert bp(code.HZ, syndrome, P, method="product_sum")[2] == 3
    assert bp(code.HZ, syndrome, P, method="minimum_sum")[2] is None

    beliefs = [next(iter(bp_trace(code.HZ, syndrome, P, method=method))).llr
               for method in METHODS]
    assert not np.allclose(*beliefs)
