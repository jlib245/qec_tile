"""Logical 연산자 — 검출되지 않지만 stabilizer는 아니다."""
import numpy as np
import pytest

from qec_tile.gf2 import rank2
from qec_tile.tile import paper_code

CASES = [("b3w6", 6, 6), ("b3w6", 8, 5), ("b3w8", 6, 6), ("b4w8", 6, 6)]


@pytest.mark.parametrize("name,L1,L2", CASES)
def test_logicals(name, L1, L2):
    c = paper_code(name, L1, L2)
    LX, LZ = c.logicals()
    k = c.k
    assert LX.shape == (k, c.n) and LZ.shape == (k, c.n)
    # 반대 타입 stabilizer와 교환한다 — syndrome을 남기지 않는다
    assert not ((LX @ c.HZ.T) % 2).any()
    assert not ((LZ @ c.HX.T) % 2).any()
    # 그러면서 자신은 stabilizer가 아니다 — 상태를 실제로 바꾼다
    assert rank2(np.vstack([c.HX, LX])) == rank2(c.HX) + k
    assert rank2(np.vstack([c.HZ, LZ])) == rank2(c.HZ) + k
    # symplectic 짝지음이 비퇴화다
    assert rank2((LX @ LZ.T) % 2) == k


@pytest.mark.parametrize("name,L1,L2", CASES)
def test_stabilizers_are_not_logical_failures(name, L1, L2):
    """residual이 stabilizer와 같으면 성공으로 읽혀야 한다."""
    _, LZ = (c := paper_code(name, L1, L2)).logicals()
    rng = np.random.default_rng(3)
    combo = (rng.random((20, c.HX.shape[0])) < 0.5).astype(np.uint8)
    stabilizers = (combo @ c.HX) % 2
    assert not ((stabilizers @ LZ.T) % 2).any()


@pytest.mark.parametrize("name,L1,L2", CASES)
def test_logicals_are_logical_failures(name, L1, L2):
    """residual이 logical과 같으면 실패로 읽혀야 한다."""
    c = paper_code(name, L1, L2)
    LX, LZ = c.logicals()
    assert ((LX @ LZ.T) % 2).any(axis=1).all()
