"""routing 최적화 전후를 나란히 그린다 -> SVG (cairosvg가 있으면 PNG도).

논문(arXiv:2606.19482) Figure 7과 같은 형식이다:

    빨간 x   지워지는 routing site
    파란 선  옮겨지는 출발점 (원래 자리 -> 옮긴 자리). check와 data 모두

caption의 "empty circles denote routing sites, red crosses mark routing sites that
are removed, and blue arrows indicate shifts of data or check start positions".

``shorten_route_windows``(창 절단: check 이동 -> 단수 조건 흡수)과 ``trace_prune``
(후보 이동 탐색)을 다 거친 결과를 그린다.

사용법:
    python scripts/prune_diagram.py --word N2ESEN2 --size 4x4
"""
from __future__ import annotations

import argparse
import os

from qec_tile.directional import build_directional_code, hardware_site
from qec_tile.walk2 import (CHECK_X, CHECK_Z, DATA, ROUTING, Qubit,
                            trace_prune, shorten_route_windows, round_births,
                            walk_layout)

STYLE = {
    CHECK_X: ("#f2c14e", "#8a6d1f", 11.0),   # 노랑
    CHECK_Z: ("#5aa469", "#2f6b3c", 11.0),   # 초록
    DATA:    ("#9e9e9e", "#5f5f5f", 8.0),    # 회색
    ROUTING: ("#ffffff", "#bdbdbd", 5.0),    # 흰색
}


def panel(qubit_at, removed, shifts, title, bounds, scale) -> str:
    """배치 한 장. ``removed``는 빨간 x, ``shifts``는 파란 선으로 얹는다."""
    left, right, bottom, top = bounds

    def place(site):
        return ((site[0] - left) * scale, (top - site[1]) * scale)

    parts = [f'<text x="{(right - left) * scale / 2}" y="-14" '
             f'text-anchor="middle" font-size="17" font-family="sans-serif">'
             f'{title}</text>']
    for site, qubit in sorted(qubit_at.items()):
        fill, stroke, radius = STYLE[qubit.role]
        x, y = place(site)
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1.3"/>')
    for site in sorted(removed):
        x, y = place(site)
        parts.append(f'<path d="M{x - 6},{y - 6} L{x + 6},{y + 6} '
                     f'M{x - 6},{y + 6} L{x + 6},{y - 6}" '
                     f'stroke="#d64545" stroke-width="2.4"/>')
    for start, seat in shifts:
        (x1, y1), (x2, y2) = place(start), place(seat)
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                     f'stroke="#3b6fb6" stroke-width="2.4"/>')
        parts.append(f'<circle cx="{x2}" cy="{y2}" r="3.5" fill="#3b6fb6"/>')
    return "\n".join(parts)


def figure(code, word: str, scale: float = 24.0) -> str:
    """최적화 전(지워질 것 표시)과 후를 두 패널로."""
    before = walk_layout(code, word)
    after, starts = trace_prune(code, word,
                                     *shorten_route_windows(code, word))
    # 빨간 x는 지워진 routing만. 옮긴 것의 원래 자리는 파란 선이 말해준다.
    removed = {site for site in set(before) - set(after)
               if before[site].role == ROUTING}

    shifts, moved = [], dict(after)
    for role, index, home, seat, _, _ in round_births(code, word, 0, starts):
        shifts.append((home, seat))
        moved[seat] = Qubit(role, index, home)       # 옮긴 자리에 그려준다
    for site, qubit in after.items():                # data 출발점 이동
        if qubit.role == DATA and site != hardware_site(code.qubits[qubit.index]):
            shifts.append((hardware_site(code.qubits[qubit.index]), site))

    sites = list(before)
    bounds = (min(x for x, _ in sites), max(x for x, _ in sites),
              min(y for _, y in sites), max(y for _, y in sites))
    routing_before = sum(1 for q in before.values() if q.role == ROUTING)
    routing_after = sum(1 for q in after.values() if q.role == ROUTING)
    panels = [panel(before, removed, shifts,
                    f"before  (site {len(before)}, routing {routing_before})",
                    bounds, scale),
              panel(moved, set(), [],
                    f"after  (site {len(after)}, routing {routing_after})",
                    bounds, scale)]
    width = (bounds[1] - bounds[0]) * scale + 2 * scale
    height = (bounds[3] - bounds[2]) * scale + 2.5 * scale
    body = "\n".join(
        f'<g transform="translate({index * width + scale}, {scale + 20})">'
        f'\n{one}\n</g>' for index, one in enumerate(panels))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * 2}" '
            f'height="{height + 30}">'
            f'\n<rect width="100%" height="100%" fill="white"/>\n{body}\n</svg>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="N2ESEN2")
    ap.add_argument("--size", default="4x4", help="directional MxN")
    ap.add_argument("--scale", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1700, help="PNG 가로 픽셀")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    M, N = (int(part) for part in args.size.lower().split("x"))
    code = build_directional_code(args.word, M, N)
    print(code)
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, f"{args.word}_{M}x{N}_prune")
    svg = figure(code, args.word, args.scale)
    with open(f"{stem}.svg", "w") as f:
        f.write(svg)
    print(f"wrote {stem}.svg")
    try:
        import cairosvg
    except ImportError:
        return
    cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{stem}.png",
                     output_width=args.width)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
