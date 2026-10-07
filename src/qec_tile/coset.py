"""Coset MRF — logical class의 확률을 제약 없는 partition function으로.

Krishnamoorthy et al., "Certified decoding of quantum LDPC codes"
(arXiv:2608.25545), Proposition 1을 따른다. X sector만 다루며 (decode.py와
같은 설정), 오류 ``e``는 확률 ``w^|e|``, ``w = p/(1-p)``로 난다.

syndrome ``s = HZ e``를 만족하는 오류들은 X-stabilizer(HX의 행공간)를 법으로
``2^k``개의 logical class로 갈라진다. class ``λ``의 대표원을 ``r = e_λ``라 하면
그 class의 원소는 ``r ⊕ uᵀHX`` (``u ∈ F2^{m_X}``)이고, class 확률은

    Z_λ = Σ_u w^{E(u)},    E(u) = |r ⊕ uᵀHX|

이다. ``u``에 대한 합에는 제약이 없다 -- 이것이 논문의 요점이다. syndrome
posterior의 hard constraint ``HZ e = s``가 ``u``로 좌표를 바꾸면 사라지므로
어떤 샘플러나 message passing도 그대로 돌릴 수 있다.

degenerate ML 디코더는 ``argmax_λ Z_λ``다. 최소 무게 하나만 보는 BP+OSD와
달리 class 안의 모든 오류를 더한다.

논문 표기 대응: ``r`` = ``e_λ``, ``u`` = X-check 변수, ``N(v)`` = qubit ``v``에
걸린 X-check 집합 (HX의 ``v``열의 support).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from qec_tile.gf2 import rref2


def syndrome_solution(HZ: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
    """``HZ @ e = syndrome`` (mod 2)인 해 하나. pivot 열에만 support가 있다.

    첨가 행렬 ``[HZ | s]``를 줄여 free 열을 전부 0으로 둔 해다 -- nullspace2가
    pivot 열을 RREF에서 베끼는 것과 같은 원리.
    """
    n = HZ.shape[1]
    augmented = np.hstack([np.asarray(HZ, dtype=np.uint8),
                           np.asarray(syndrome, dtype=np.uint8).reshape(-1, 1)])
    rref, pivot_cols = rref2(augmented)
    if pivot_cols and pivot_cols[-1] == n:      # 첨가 열이 pivot이면 해가 없다
        raise ValueError("syndrome이 HZ의 열공간 밖에 있다")
    solution = np.zeros(n, dtype=np.uint8)
    solution[pivot_cols] = rref[:, n]
    return solution


def candidate_labels(k: int, max_shift: int | None = None) -> np.ndarray:
    """class 라벨 ``λ ∈ F2^k`` 목록, 첫 행은 0 (기준 class 자신).

    ``max_shift``가 없으면 ``2^k`` 전부. 있으면 무게 ``≤ max_shift``만 -- 논문이
    BP+OSD 해 주변 logical 한두 개 뒤집기로 후보를 줄인 것 (``max_shift=2``면
    ``1 + k + C(k,2)``개).
    """
    if max_shift is None:
        return ((np.arange(2 ** k)[:, None] >> np.arange(k)) & 1).astype(np.uint8)
    rows = [np.zeros(k, dtype=np.uint8)]
    for weight in range(1, max_shift + 1):
        for support in itertools.combinations(range(k), weight):
            label = np.zeros(k, dtype=np.uint8)
            label[list(support)] = 1
            rows.append(label)
    return np.array(rows, dtype=np.uint8)


def class_representatives(base: np.ndarray, LX: np.ndarray,
                          labels: np.ndarray) -> np.ndarray:
    """``r_λ = base ⊕ λᵀLX``, 라벨마다 한 행. 논문의 ``e_λ``."""
    base = np.asarray(base, dtype=np.uint8)
    return (base[None, :] ^ ((labels @ LX) % 2)).astype(np.uint8)


def coset_energy(HX: np.ndarray, r: np.ndarray, u: np.ndarray) -> np.ndarray:
    """``E(u) = |r ⊕ uᵀHX|``. ``u``가 ``(batch, m_X)``면 ``(batch,)``."""
    coset_element = ((np.asarray(u, dtype=np.uint8) @ HX) % 2) ^ r
    return coset_element.sum(axis=-1)


def exact_log_partition(HX: np.ndarray, reps: np.ndarray, w: float) -> np.ndarray:
    """대표원마다 ``log Z_λ``, ``2^{m_X}`` 전수 합산. AIS/Bethe의 정답지.

    ``w^E``는 p가 작으면 underflow하므로 로그 영역에서 더한다. parity 표
    ``(u @ HX) % 2``는 대표원과 무관하므로 한 번만 만든다.

    HX에 rank 결손이 있으면 모든 class가 ``2^{m_X - rank}``배 균일하게
    과대집계된다 (argmax는 그대로). tile code의 HX는 full row rank다.
    """
    m_X = HX.shape[0]
    if m_X > 22:
        raise ValueError(f"m_X={m_X}: 2^m_X 전수 합산 범위를 넘는다")
    parity = (candidate_labels(m_X) @ HX) % 2                 # (2^m_X, n)
    reps = np.atleast_2d(np.asarray(reps, dtype=np.uint8))
    log_w = np.log(w)
    log_Z = np.array([logsumexp((parity ^ r).sum(1) * log_w) for r in reps])
    return log_Z[0] if log_Z.size == 1 else log_Z


def exact_ml_decode(HX: np.ndarray, LX: np.ndarray, base: np.ndarray, w: float,
                    labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """후보 class 중 ``log Z_λ`` 최대인 class의 대표원과 ``log Z`` 벡터.

    대표원은 그대로 정정 벡터다. ``base``는 후보 집합의 중심이다 -- 제한 집합
    (``candidate_labels(k, max_shift)``)을 쓸 때는 논문처럼 BP+OSD 해를 넣어야
    하고, 전체 ``2^k``를 쓸 때는 어느 해든 상관없다.
    """
    reps = class_representatives(base, LX, labels)
    log_Z = np.atleast_1d(exact_log_partition(HX, reps, w))
    return reps[int(np.argmax(log_Z))], log_Z
