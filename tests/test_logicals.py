"""Logical operators — undetectable, yet not stabilizers."""
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
    # commute with the opposite stabilizer type — they leave no syndrome
    assert not ((LX @ c.HZ.T) % 2).any()
    assert not ((LZ @ c.HX.T) % 2).any()
    # yet are not stabilizers themselves — they do change the state
    assert rank2(np.vstack([c.HX, LX])) == rank2(c.HX) + k
    assert rank2(np.vstack([c.HZ, LZ])) == rank2(c.HZ) + k
    # the symplectic pairing is nondegenerate
    assert rank2((LX @ LZ.T) % 2) == k


@pytest.mark.parametrize("name,L1,L2", CASES)
def test_stabilizers_are_not_logical_failures(name, L1, L2):
    """A residual equal to a stabilizer must read as success."""
    _, LZ = (c := paper_code(name, L1, L2)).logicals()
    rng = np.random.default_rng(3)
    combo = (rng.random((20, c.HX.shape[0])) < 0.5).astype(np.uint8)
    stabilizers = (combo @ c.HX) % 2
    assert not ((stabilizers @ LZ.T) % 2).any()


@pytest.mark.parametrize("name,L1,L2", CASES)
def test_logicals_are_logical_failures(name, L1, L2):
    """A residual equal to a logical must read as failure."""
    c = paper_code(name, L1, L2)
    LX, LZ = c.logicals()
    assert ((LX @ LZ.T) % 2).any(axis=1).all()
