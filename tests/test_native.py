"""nanobind 확장이 빌드되고 import된다 — 툴체인 점검."""
from qec_tile import add, parity


def test_add():
    assert add(2, 3) == 5


def test_parity():
    assert parity([1, 0, 1, 1]) == 1
    assert parity([1, 1, 0, 0]) == 0
    assert parity([]) == 0
