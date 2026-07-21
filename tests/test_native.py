"""C++ 확장이 빌드되어 import 되는지 — 툴체인 검증용."""
from qec_tile import add, parity


def test_add():
    assert add(2, 3) == 5


def test_parity():
    assert parity([1, 0, 1, 1]) == 1   # 1이 셋 -> 홀수
    assert parity([1, 1, 0, 0]) == 0   # 1이 둘 -> 짝수
    assert parity([]) == 0
