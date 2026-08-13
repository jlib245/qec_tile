"""stim의 timeslice 그림에 역할별 색을 입힌다 -> SVG (cairosvg가 있으면 PNG도).

stim은 qubit 점마다 ``id="qubit_dot:인덱스:x_y:프레임"``을 달아준다. 그 좌표가
하드웨어 좌표의 절반이므로 두 배 하면 배치의 자리가 나오고, 거기서 역할을 찾아
점의 색과 크기만 바꾼다. gate와 프레임 배치는 stim이 그린 것을 그대로 쓴다.

프레임 하나가 한 moment다 -- 라운드가 ``w+2``개이고, 그중 ``w``개가 directional
word의 글자다.

사용법:
    python scripts/walk_diagram.py --word NESEN --size 2x2
"""
from __future__ import annotations

import argparse
import os
import re

from qec_tile.directional import build_directional_code, parse_directional_word
from qec_tile.walk2 import (CHECK_X, CHECK_Z, DATA, ROUTING, flow_step,
                            walk_layout, walk_memory_z_base)

# 역할 -> (채움, 테두리, 반지름). 반지름은 stim의 점(r=2)보다 크게 잡는다 -- 프레임
# 안에서 한 칸이 64px이라 14 정도까지는 서로 안 겹친다. 노랑은 흰 바탕에서 윤곽이
# 안 보이므로 테두리를 꼭 준다.
STYLE = {
    CHECK_X: ("#f2c14e", "#8a6d1f", 15.0),   # 노랑
    CHECK_Z: ("#5aa469", "#2f6b3c", 15.0),   # 초록
    DATA:    ("#9e9e9e", "#5f5f5f", 15.0),   # 회색
    ROUTING: ("#ffffff", "#bdbdbd", 15.0),   # 흰색
}
DOT = re.compile(r'<circle id="qubit_dot:\d+:'
                 r'(?P<x>-?[\d.]+)_(?P<y>-?[\d.]+):(?P<frame>\d+)"'
                 r' cx="(?P<cx>-?[\d.]+)" cy="(?P<cy>-?[\d.]+)"[^/]*/>')


def occupancy_by_frame(code, word: str, frames: int, layout) -> list[dict]:
    """프레임마다의 배치. 물리 qubit은 자리에 고정이고 상태가 흐르므로, 프레임이
    넘어갈 때마다 그 자리에 있는 qubit이 바뀐다.

    프레임 ``i``는 moment ``i``이고, 그 moment의 gate가 걸리기 **전**의 배치를
    보여준다. moment ``i``(1..w)가 층 ``i-1``의 gate를 담으므로 층을 하나씩 굴린다.
    """
    steps = parse_directional_word(word)
    qubit_at = dict(layout)
    snapshots = []
    for frame in range(frames):
        snapshots.append(dict(qubit_at))
        if 1 <= frame <= len(steps):
            flow_step(qubit_at, steps[frame - 1])
    return snapshots


def colour_qubits(svg: str, snapshots: list[dict]) -> str:
    """timeslice SVG의 qubit 점을 그 프레임의 역할별 색으로 바꾼다."""
    def recolour(match):
        # 회로가 y를 뒤집어 넣었으므로(그림에서 N이 위로 가게) 되돌린다.
        site = (round(2 * float(match["x"])), round(-2 * float(match["y"])))
        frame = min(int(match["frame"]), len(snapshots) - 1)
        qubit = snapshots[frame].get(site)
        if qubit is None:                      # 배치 밖 -- stim 것 그대로 둔다
            return match[0]
        fill, stroke, radius = STYLE[qubit.role]
        return (f'<circle cx="{match["cx"]}" cy="{match["cy"]}" r="{radius}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    return DOT.sub(recolour, svg)


def layout_panel(layout, scale: float = 128.0):
    """걷기 전 배치 한 장. 각 X/Z check, data, routing이 어디서 출발하는지 보여준다.

    stim 그림에는 이 장면이 없다 -- Tick 0은 이미 첫 moment(reset)이라 라벨에 가린다.
    """
    left = min(x for x, _ in layout)
    right = max(x for x, _ in layout)
    bottom = min(y for _, y in layout)
    top = max(y for _, y in layout)
    parts = []
    for site, qubit in sorted(layout.items()):
        fill, stroke, radius = STYLE[qubit.role]
        x = (site[0] - left) * scale / 2
        y = (top - site[1]) * scale / 2
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1.5"/>')
    width = (right - left) * scale / 2 + scale
    height = (top - bottom) * scale / 2 + scale
    body = (f'<g transform="translate({scale / 2}, {scale / 2})">'
            f'\n' + "\n".join(parts) + '\n</g>'
            f'\n<text x="{width / 2}" y="{height - 8}" text-anchor="middle" '
            f'font-size="22" font-family="sans-serif">start</text>')
    return body, width, height + 20


def legend() -> str:
    """색이 무엇을 뜻하는지 한 줄. SVG 맨 위에 얹는다."""
    parts, x = [], 20.0
    for role, name in ((CHECK_X, "X check"), (CHECK_Z, "Z check"),
                       (DATA, "data"), (ROUTING, "routing")):
        fill, stroke, radius = STYLE[role]
        parts.append(f'<circle cx="{x}" cy="0" r="{radius}" fill="{fill}" '
                     f'stroke="{stroke}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 20}" y="7" font-size="20" '
                     f'font-family="sans-serif">{name}</text>')
        x += 20 + 12 * len(name) + 40
    return f'<g transform="translate(20, 28)">\n' + "\n".join(parts) + "\n</g>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="NESEN")
    ap.add_argument("--size", default="2x2", help="directional MxN")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--width", type=int, default=2400, help="PNG 가로 픽셀")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    M, N = (int(part) for part in args.size.lower().split("x"))
    code = build_directional_code(args.word, M, N)
    print(code)
    layout = walk_layout(code, args.word)
    print(f"routing {sum(1 for q in layout.values() if q.role == ROUTING)}")
    circuit = walk_memory_z_base(code, args.word, rounds=args.rounds,
                                 layout=layout)
    raw = str(circuit.diagram("timeslice-svg"))
    frames = max(int(frame) for frame in re.findall(r'qubit_dot:\d+:[^:]+:(\d+)"',
                                                    raw)) + 1
    svg = colour_qubits(raw, occupancy_by_frame(code, args.word, frames,
                                                layout))
    # stim SVG에는 배경이 없다. 투명인 채로 PNG로 만들면 검정으로 채워져 글자가 묻힌다.
    svg = svg.replace(">", '>\n<rect width="100%" height="100%" fill="white"/>', 1)
    svg = svg.replace("</svg>", legend() + "\n</svg>")

    # 걷기 전 배치를 맨 위에 한 장 얹는다.
    panel, panel_width, panel_height = layout_panel(layout)
    box = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    inner_width, inner_height = float(box[1]), float(box[2])
    inner = svg.replace("<svg ", f'<svg y="{panel_height}" '
                        f'width="{inner_width}" height="{inner_height}" ', 1)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'width="{max(panel_width, inner_width)}" '
           f'height="{panel_height + inner_height}">'
           f'\n<rect width="100%" height="100%" fill="white"/>\n{panel}\n{inner}\n</svg>')

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(
        args.out_dir, f"{args.word}_{M}x{N}_rounds{args.rounds}_walk")
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
