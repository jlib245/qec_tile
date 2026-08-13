"""GF(2) 위의 belief propagation, 반복 과정을 밖으로 드러낸다.

``ldpc``의 ``BpDecoder``는 최종 상태만 돌려주므로 belief가 움직이는 것을 볼 방법이
없다 -- 디코더 뷰어나, degenerate 코드에서 BP가 왜 멈추는지 보려면 그게 필요하다.
같은 디코더의 루프를 뒤집어 놓은 것이다: ``bp_trace``는 반복마다 yield한다.

관례는 ldpc를 따라 둘이 비트 단위로 일치한다 (tests/test_bp.py가 이를 고정한다):
log-likelihood ratio는 ``log((1-p)/p)``이고 양수면 "이 qubit에 오류 없음",
hard decision은 ``llr < 0``이다. 두 스케줄이 다른 곳은 check-node 갱신뿐이고,
ldpc가 쓰는 이름을 그대로 쓴다:

``product_sum``   sum-product 알고리즘, 본래의 BP -- 트리에서 정확하다::

    m_{c->v} = 2 atanh( (-1)^{s_c} prod_{v' != v} tanh(m_{v'->c} / 2) )

``minimum_sum``   그것의 max-log 근사, 하드웨어에 얹을 만큼 싸다. check 메시지를
과대평가하는데, 정규화 인자 ``alpha``(ldpc의 ``ms_scaling_factor``)가 그만큼을 다시
덜어낸다::

    m_{c->v} = alpha * (-1)^{s_c} * prod_{v' != v} sign(m_{v'->c})
                     * min_{v' != v} |m_{v'->c}|

참고문헌
--------
R. G. Gallager, "Low-Density Parity-Check Codes", MIT Press (1963)
-- sum-product 알고리즘.

J. Chen and M. Fossorier, "Near optimum universal belief propagation based
decoding of low-density parity check codes", IEEE Trans. Commun. 50 (2002)
-- normalized min-sum과 스케일링 인자 alpha.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BpIteration:
    iteration: int
    llr: np.ndarray
    hard: np.ndarray
    converged: bool
    to_check: np.ndarray
    to_bit: np.ndarray


def tanner_edges(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)``: 각 Tanner edge가 잇는 check와 qubit.

    모든 메시지 배열이 쓰는 순서. ``np.nonzero``는 row-major로 훑으므로 한
    check의 edge들이 연속하고, ``bp_trace``의 check별 축약이 그것에 기댄다.
    """
    return np.nonzero(np.ascontiguousarray(H, dtype=np.uint8))


def _prior_llr(channel, n: int) -> np.ndarray:
    """채널 확률 -> prior LLR. 스칼라면 균일한 단일 오류율."""
    probability = np.broadcast_to(np.asarray(channel, dtype=float), (n,))
    if not ((0 < probability) & (probability < 1)).all():
        raise ValueError("channel probabilities must lie strictly in (0, 1)")
    return np.log((1 - probability) / probability)


METHODS = ("minimum_sum", "product_sum")


def _phi(magnitude: np.ndarray) -> np.ndarray:
    """Gallager의 ``-log tanh(x/2)``. involution.

    tanh의 곱은 37과 100을 구별하지 못하지만, 그 log는 평범한 작은 수다. 양 끝에서
    각각 맞는 항등식이 필요하다 — ``1 + u``와 ``1 - u``(``u = exp(-x)``)가 한쪽
    극단에서 각각 모든 것을 잃기 때문이다::
        x >= 1:  log1p(u) - log1p(-u)                       u가 아주 작다
        x <  1:  log(2 + expm1(-x)) - log(-expm1(-x))       1 - u가 x쯤이다
    """
    magnitude = np.abs(magnitude)
    small = magnitude < 1.0
    out = np.empty_like(magnitude, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = -np.expm1(-magnitude)              # 1 - exp(-x), 작은 x에서 정확
        out[small] = (np.log(2.0 + np.expm1(-magnitude[small]))
                      - np.log(gap[small]))
        decay = np.exp(-magnitude[~small])
        out[~small] = np.log1p(decay) - np.log1p(-decay)
    return out


def bp_trace(H: np.ndarray, syndrome: np.ndarray, channel,
             method: str = "minimum_sum", max_iter: int = 50,
             ms_scaling_factor: float = 1.0):
    """flooding BP. sweep마다 ``BpIteration``을 yield한다.

    ``channel``은 스칼라 오류율이거나 열마다 하나씩인 확률이고,
    ``decode.make_decoder``와 맞춘다. ``method``는 check-node 갱신을 고르며 ldpc의
    이름을 쓴다. 수렴하면 생성기가 일찍 멈춘다.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; have {list(METHODS)}")
    H = np.ascontiguousarray(H, dtype=np.uint8)
    n_checks, n = H.shape
    syndrome = np.asarray(syndrome, dtype=np.uint8).reshape(n_checks)
    prior = _prior_llr(channel, n)

    rows, cols = tanner_edges(H)
    if rows.size == 0:                       # check가 없다: 전달할 것이 없다
        empty = np.zeros(0)
        yield BpIteration(1, prior.copy(), np.zeros(n, dtype=np.uint8),
                          not syndrome.any(), empty, empty)
        return

    degree = np.bincount(rows, minlength=n_checks)
    if (degree == 0).any():
        raise ValueError("H has an all-zero check; drop it before decoding")
    row_starts = np.searchsorted(rows, np.arange(n_checks))
    # degree가 1인 check에는 "다른" edge가 없으므로 나머지에 대한 min은 +inf다:
    # 그 check 혼자 자기 qubit을 확정한다. 인덱스는 범위 안에 있도록 clip하고,
    # degree == 1인 자리에서는 값을 버린다.
    second_slot = np.where(degree > 1, np.minimum(row_starts + 1, rows.size - 1),
                           row_starts)
    # 각 edge가 자기 check 안에서 몇 번째인지. product_sum이 필요한 prefix/suffix
    # 합에 쓴다.
    slot = np.arange(rows.size) - row_starts[rows]
    widest = int(degree.max())
    check_flip = np.where(syndrome[rows] == 1, -1.0, 1.0)
    message_to_check = prior[cols].copy()

    for iteration in range(1, max_iter + 1):
        # 두 갱신 모두 *다른* edge들의 부호 parity가 필요하다.
        negatives = np.bincount(rows, weights=(message_to_check < 0),
                                minlength=n_checks).astype(np.int64)
        parity = (negatives[rows] - (message_to_check < 0)) % 2
        sign = np.where(parity == 1, -1.0, 1.0)

        if method == "minimum_sum":
            magnitude = np.abs(message_to_check)
            # "이 check의 다른 edge들 중 가장 작은 크기
            order = np.lexsort((magnitude, rows))
            smallest = order[row_starts]
            runner_up = np.where(degree > 1, magnitude[order[second_slot]],
                                 np.inf)
            is_smallest = np.zeros(rows.size, dtype=bool)
            is_smallest[smallest] = True
            other = np.where(is_smallest, runner_up[rows],
                             magnitude[smallest][rows])
            message_to_bit = ms_scaling_factor * sign * check_flip * other
        else:  # product_sum
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
    """``(hard, llr, converged_at)``. BP가 멈췄으면 ``converged_at``은 None."""
    last = None
    for last in bp_trace(H, syndrome, channel, method, max_iter,
                         ms_scaling_factor):
        pass
    return last.hard, last.llr, last.iteration if last.converged else None
