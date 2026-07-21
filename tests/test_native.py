"""The nanobind extension builds and imports — a toolchain check."""
from qec_tile import add, parity


def test_add():
    assert add(2, 3) == 5


def test_parity():
    assert parity([1, 0, 1, 1]) == 1
    assert parity([1, 1, 0, 0]) == 0
    assert parity([]) == 0
