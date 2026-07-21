"""타일 기하 — 조건 (T2)가 만들어내는 대칭성."""
import numpy as np
import pytest

from qec_tile.tile import dual_tile

# (B, X_H, X_V) — 논문 b3w6 타일과 b4w8 타일
TILES = [
    (3, [(0, 0), (2, 1), (2, 2)], [(0, 2), (1, 2), (2, 0)]),
    (4, [(0, 0), (0, 3), (2, 2), (3, 0)], [(0, 1), (1, 0), (1, 1), (3, 3)]),
]


def as_edges(t_h, t_v):
    """(H목록, V목록) -> {('H'|'V', x, y)} 집합."""
    return {("H", x, y) for x, y in t_h} | {("V", x, y) for x, y in t_v}


def overlap(a: set, b: set, dx: int, dy: int) -> int:
    """``b``를 (dx, dy)만큼 옮겼을 때 ``a``와 겹치는 큐빗 수."""
    return len(a & {(o, x + dx, y + dy) for (o, x, y) in b})


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_is_an_involution(B, x_h, x_v):
    """Z-타일의 Z-타일은 다시 X-타일 — 180도를 두 번 돌면 제자리."""
    z_h, z_v = dual_tile(x_h, x_v, B)
    back_h, back_v = dual_tile(z_h, z_v, B)
    assert set(back_h) == set(x_h) and set(back_v) == set(x_v)


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_stays_inside_the_box(B, x_h, x_v):
    z_h, z_v = dual_tile(x_h, x_v, B)
    assert all(0 <= x < B and 0 <= y < B for x, y in z_h + z_v)


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_dual_preserves_weight(B, x_h, x_v):
    """H와 V가 맞바뀌므로 각각의 개수도 맞바뀐다 — 전체 무게는 보존."""
    z_h, z_v = dual_tile(x_h, x_v, B)
    assert (len(z_h), len(z_v)) == (len(x_v), len(x_h))


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_every_relative_overlap_is_even(B, x_h, x_v):
    """어떤 상대 위치에서도 X-타일과 Z-타일은 짝수 개만 겹친다.

    이게 (T2)의 존재 이유다. 두 안정자가 겹치는 큐빗 수가 짝수여야
    X와 Z가 교환하고, 그래야 CSS 코드가 성립한다.
    """
    X = as_edges(x_h, x_v)
    Z = as_edges(*dual_tile(x_h, x_v, B))
    for dx in range(-B, B + 1):
        for dy in range(-B, B + 1):
            assert overlap(X, Z, dx, dy) % 2 == 0, f"shift ({dx},{dy})"


@pytest.mark.parametrize("B,x_h,x_v", TILES)
def test_h_and_v_overlaps_are_equal(B, x_h, x_v):
    """짝수인 이유: H쪽 겹침 수와 V쪽 겹침 수가 항상 같아서 (합 = 2배)."""
    z_h, z_v = dual_tile(x_h, x_v, B)
    XH, XV = as_edges(x_h, []), as_edges([], x_v)
    ZH, ZV = as_edges(z_h, []), as_edges([], z_v)
    for dx in range(-B, B + 1):
        for dy in range(-B, B + 1):
            assert overlap(XH, ZH, dx, dy) == overlap(XV, ZV, dx, dy)


def test_even_overlap_holds_for_random_tiles():
    """앞의 두 타일이 운이 좋았던 게 아니라, (T2)면 무조건 성립한다."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        B = int(rng.integers(2, 6))
        cells = [(int(x), int(y)) for x in range(B) for y in range(B)]
        pick = lambda: [cells[i] for i in
                        rng.choice(len(cells), size=int(rng.integers(1, B * B)),
                                   replace=False)]
        x_h, x_v = pick(), pick()
        X = as_edges(x_h, x_v)
        Z = as_edges(*dual_tile(x_h, x_v, B))
        for dx in range(-B, B + 1):
            for dy in range(-B, B + 1):
                assert overlap(X, Z, dx, dy) % 2 == 0
