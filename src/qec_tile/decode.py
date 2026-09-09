"""BP+OSD로 하는 tile code의 code-capacity 디코딩.

잡음 모델은 가장 단순한 것이다: 각 qubit이 확률 ``p``로 독립적인 X 오류를 받고,
Z-check는 완벽하게 읽힌다. 한 shot은

    e ~ Bernoulli(p)^n            X 오류 패턴
    s = HZ @ e   (mod 2)          그 syndrome
    e_hat = decode(s)             BP, 멈추면 OSD로 넘어간다
    r = e ^ e_hat                 residual (구성상 HZ @ r = 0)
    실패  <=>  LZ @ r != 0        r이 단지 stabilizer가 아니라 logical이다

X sector만 시뮬레이션한다. Z sector는 HX/LX로 바꾸면 똑같다. 실패 판정은
tests/test_logicals.py가 고정해둔 바로 그것이다.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from ldpc import BpDecoder, BpLsdDecoder, BpOsdDecoder
from ldpc.lsd_decoder import LsdDecoder


def _prior(channel):
    """스칼라 오류율이면 error_rate, 열별 벡터면 error_channel."""
    channel = np.asarray(channel, dtype=float)
    return (dict(error_rate=float(channel)) if channel.ndim == 0
            else dict(error_channel=list(channel)))


def make_decoder(H: np.ndarray, channel, *, bp_method: str = "minimum_sum",
                 max_iter: int = 50, ms_scaling_factor: float = 1.0,
                 osd_method: str = "osd_cs",
                 osd_order: int = 7) -> BpOsdDecoder:
    """parity check ``H``에 prior ``channel``을 쓰는 BP+OSD 디코더.

    ``channel``은 BP의 prior이면서 샘플링 오류율이기도 하다: BP는 posterior를
    구하려면 참 오류율이 필요하고, OSD는 정확히 그 posterior로 비트를 정렬한다.
    스칼라면 균일한 단일 오류율이고, 벡터면 열별 prior다 (시공간 행렬은 data
    오류율과 측정 오류율을 섞는다).

    BP/OSD 손잡이를 밖으로 낸 것은 registry가 디코더마다 기준 설정을 못박을 수
    있게 하려는 것이다. ldpc 자신의 기본값은 약하다 (예: max_iter=3).
    """
    return BpOsdDecoder(
        H.astype(np.uint8),
        **_prior(channel),
        bp_method=bp_method,
        max_iter=max_iter,
        ms_scaling_factor=ms_scaling_factor,
        osd_method=osd_method,
        osd_order=osd_order,
    )


def make_lsd_decoder(H: np.ndarray, channel, *, bp_method: str = "minimum_sum",
                     max_iter: int = 30, ms_scaling_factor: float = 0.625,
                     lsd_method: str = "lsd_0",
                     lsd_order: int = 0) -> BpLsdDecoder:
    """BP+LSD 디코더 (Hillmann et al., arXiv:2406.18655).

    기본값은 그 논문의 설정이다: min-sum, 30회 반복, 스케일링 alpha=0.625,
    LSD order 0. ldpc 자신의 기본값은 훨씬 약해서(max_iter=3, product_sum) 여기서
    못박아둔다.
    """
    return BpLsdDecoder(
        H.astype(np.uint8),
        **_prior(channel),
        bp_method=bp_method,
        max_iter=max_iter,
        ms_scaling_factor=ms_scaling_factor,
        lsd_method=lsd_method,
        lsd_order=lsd_order,
    )


# 디코더 registry: name -> (H, channel) -> decoder. 각각 기준 설정에 못박아두어
# CSV/파일명의 디코더 축이 스스로를 설명하게 한다.
class VibeLsdDecoder:
    """VibeLSD (Koutsioumpas, Noszko, Sayginel, Webster & Roffe, arXiv:2508.15743).

    error mechanism을 무작위 순열한 직렬 min-sum BP ``ensemble``개를 차례로 돌려,
    ``converged``개가 수렴하면 멈추고 수렴한 후보 중 prior 우도가 가장 높은(가장
    가벼운) 것을 고른다. 하나도 수렴하지 않으면 정규화한 LLR의 평균
    ``(1/L) Σ LLR_i / ‖LLR_i‖``을 LSD에 넘긴다.

    직렬 스케줄은 앞 check가 방금 갱신한 메시지를 뒤 check가 바로 쓰므로 순서마다
    다른 고정점에 닿는다. 조밀한 DEM(hyperedge 차수 10 이상)에서 병렬 BP 하나가
    못 푸는 shot을 여러 순서가 나눠 푼다 -- 우리 walk 회로에서 BP+OSD-CS7 대비
    8배 이상 낮았다.

    ``decode``는 ``BpOsdDecoder``와 같은 계약(오류 벡터)이라 registry에 그대로 든다.
    """

    def __init__(self, H, channel, *, ensemble: int = 32, converged: int = 5,
                 max_iter: int = 20, ms_scaling_factor: float = 1.0,
                 lsd_order: int = 0, seed: int = 0):
        H = np.asarray(H, dtype=np.uint8)
        channel = np.asarray(channel, dtype=float)
        if channel.ndim == 0:
            channel = np.full(H.shape[1], float(channel))
        rng = np.random.default_rng(seed)
        # 멤버를 가르는 것은 순열뿐이라 BpDecoder는 하나로 족하다. 멤버마다 객체를
        # 두면 각자 sparse 사본을 들어 worker 메모리가 앙상블 배로 커진다.
        self.orders = np.array([rng.permutation(H.shape[1])
                                for _ in range(ensemble)], dtype=np.int32)
        self.bp = BpDecoder(H, error_channel=list(channel),
                            bp_method="minimum_sum", max_iter=max_iter,
                            ms_scaling_factor=ms_scaling_factor,
                            schedule="serial",
                            # 생성자는 list만 받는다 (setter는 ndarray도 받는다)
                            serial_schedule_order=self.orders[0].tolist())
        self.lsd = LsdDecoder(H, lsd_order=lsd_order)
        self.max_iter = max_iter
        self.converged = converged
        self.log_weight = np.log((1 - channel) / channel)   # 열별 -log 우도

    def decode(self, syndrome):
        """논문 III.4.2. 후보는 수렴 iteration이 가장 작은 ``converged``개다.

        논문은 L개를 동시에 돌려 M번째가 수렴하면 나머지를 끊는다. 순차로 돌되
        예산을 그 M번째 값으로 깎으면 같은 집합이 나온다 -- 예산 안에 수렴 못 하는
        멤버는 어차피 상위 M개가 아니다. break가 없는 것은 뒤쪽에 더 빨리 수렴하는
        멤버가 있을 수 있어서다.
        """
        budget = self.max_iter
        found, llrs = [], []             # found: (iter, weight, correction)
        for order in self.orders:
            self.bp.serial_schedule_order = order   # 둘 다 decode보다 먼저다
            self.bp.max_iter = budget
            candidate = self.bp.decode(syndrome)
            llr = np.asarray(self.bp.log_prob_ratios, dtype=float)
            norm = np.linalg.norm(llr)
            if norm > 0:
                llrs.append(llr / norm)
            if self.bp.converge:
                found.append((self.bp.iter,
                              float(self.log_weight @ candidate),
                              candidate.copy()))
                found.sort(key=lambda c: c[0])
                del found[self.converged:]
                if len(found) == self.converged:
                    budget = found[-1][0]           # M번째로 빠른 수렴
        if found:
            return min(found, key=lambda c: c[1])[2]    # prior 우도 최소
        # 수렴이 0개라 예산이 한 번도 안 줄었다 -- L개 전부가 full max_iter로 돌았고
        # 논문 step 5의 (1/L) Σ LLR_i / ‖LLR_i‖가 그대로 성립한다.
        return self.lsd.decode(syndrome, np.mean(llrs, axis=0))


DECODERS = {
    # VibeLSD: vibe 논문 기본(앙상블 32, 20회)과 directional 논문이 쓴 설정(200, 15회).
    "vibelsd_32": lambda H, ch: VibeLsdDecoder(H, ch, ensemble=32, max_iter=20),
    "vibelsd_200": lambda H, ch: VibeLsdDecoder(H, ch, ensemble=200,
                                                max_iter=15),
    # BB code 관례 (Bravyi et al.): OSD combination sweep, order 7.
    "bposd_cs7": lambda H, ch: make_decoder(H, ch, osd_method="osd_cs",
                                            osd_order=7),
    # 가장 싼 OSD: order-0 소거만, combination sweep 없음.
    "bposd_0": lambda H, ch: make_decoder(H, ch, osd_method="osd_0",
                                          osd_order=0),
    # LSD 논문 설정 (Hillmann et al.): order 0, min-sum, 30회, a=0.625.
    "bplsd_0": lambda H, ch: make_lsd_decoder(H, ch),
    # BB 스타일의 고차 sweep을 붙인 LSD, 비교용.
    "bplsd_cs7": lambda H, ch: make_lsd_decoder(H, ch, lsd_method="lsd_cs",
                                                lsd_order=7),
}


def sample_residuals(code, p: float, shots: int, decoder: str,
                     seed: int | None = None):
    """디코딩된 shot마다 residual ``e ^ e_hat``을 yield한다 (X sector)."""
    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    rng = np.random.default_rng(seed)
    decoder_obj = build(code.HZ, p)
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        e_hat = decoder_obj.decode(syndrome)
        yield (e ^ e_hat) % 2


class FailureCounts(NamedTuple):
    """디코딩 실행에서 나온 날 집계. 비율은 파생시키고 저장하지 않는다.

    ``logical_flips``는 shot이 아니라 observable을 센다: k개 중 셋을 뒤집은 shot은
    3을 더한다. 그래서 logical qubit당 비율(``logical_flips / (shots * k)``)이
    실측이 되고, ``per_logical_rate``의 독립성 가정 변환과 구별된다.
    """
    shots: int
    block_fails: int        # observable이 하나 이상 뒤집힌 shot 수
    logical_flips: int      # 뒤집힌 observable 수, 모든 shot에 걸쳐 합산
    k: int                  # shot당 observable 수 = L.shape[0]


def failure_counts(H: np.ndarray, L: np.ndarray, channel, shots: int,
                   decoder: str, seed: int | None = None) -> FailureCounts:
    """``shots``개를 디코딩해 두 종류의 실패를 집계한다.

    완전히 일반적이다: 열별 확률 ``channel``로 ``x``를 뽑고, syndrome ``H @ x``를
    디코딩한 뒤 ``L @ (x ^ x_hat)``을 본다. code capacity와 시공간
    (phenomenological) 행렬이 모두 이 모양에 들어맞는다.
    """
    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    rng = np.random.default_rng(seed)
    decoder_obj = build(H, channel)
    channel = np.asarray(channel, dtype=float)
    block_fails = 0
    logical_flips = 0
    for _ in range(shots):
        x = (rng.random(H.shape[1]) < channel).astype(np.uint8)
        x_hat = decoder_obj.decode(((H @ x) % 2).astype(np.uint8))
        flips = (L @ ((x ^ x_hat) % 2)) % 2
        block_fails += bool(flips.any())
        logical_flips += int(flips.sum())
    return FailureCounts(shots, block_fails, logical_flips, L.shape[0])


def block_failure_rate(H: np.ndarray, L: np.ndarray, channel, shots: int,
                       decoder: str, seed: int | None = None) -> float:
    """residual이 observable을 뒤집는 shot의 비율."""
    counts = failure_counts(H, L, channel, shots, decoder, seed)
    return counts.block_fails / counts.shots


def code_capacity_counts(code, p: float, shots: int, decoder: str,
                         seed: int | None = None) -> FailureCounts:
    """오류율 ``p``에서 logical X 실패의 블록별/observable별 집계."""
    _, LZ = code.logicals()
    return failure_counts(code.HZ, LZ, np.full(code.n, p), shots, decoder,
                          seed)


def code_capacity_block_rate(code, p: float, shots: int, decoder: str,
                             seed: int | None = None) -> float:
    """오류율 ``p``에서 logical X 실패로 끝나는 shot의 비율."""
    counts = code_capacity_counts(code, p, shots, decoder, seed)
    return counts.block_fails / counts.shots


def per_logical_rate(block_rate: float, k: int) -> float:
    """블록 실패율 -> logical qubit 하나의 실패율.

    ``block_failure_rate``는 k개 observable 중 *하나라도* 뒤집히면 그 shot을 실패로
    치므로, 물리적 성능이 같아도 k=8 코드가 k=1 surface code에 비해 여덟 배로
    불리해진다. 그것을 되돌리려면 k개 logical이 독립적으로 실패한다고 가정해야
    한다::

        block = 1 - (1 - single)**k    =>    single = 1 - (1 - block)**(1/k)

    정확하지는 않다 -- weight가 낮은 logical 연산자 하나가 observable 둘을 한꺼번에
    뒤집을 수 있다 -- 그래도 ``block``이 작을 때 쓰는 표준 변환이고, 1차 근사로는
    그냥 ``block / k``다.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0.0 <= block_rate <= 1.0:
        raise ValueError(f"block_rate must be in [0, 1], got {block_rate}")
    return 1.0 - (1.0 - block_rate) ** (1.0 / k)
