"""tile code의 memory-Z 회로를 stim 그림으로 렌더링 -> PNG (또는 --svg로 SVG).

잡음 없는 기본 회로를 그린다 (잡음 연산이 들어가면 구조가 묻힌다). memory_z_base의
QUBIT_COORDS가 모든 것을 실제 격자에 놓으므로, timeslice 프레임이 실제 tile 기하와
CNOT 스케줄을 보여준다.

``--word``를 주면 directional tile code의 walk 회로를 그린다. timeslice 프레임 하나가
directional word의 한 글자이므로, 논문(arXiv:2606.19482) Figure 4와 층별로 대조할 수
있다 -- ``--word NESEN --size 2x2``가 그림의 그 코드다.

사용법:
    python scripts/diagram.py
    python scripts/diagram.py --tile b4w8 --L 3 --types timeslice,detslice
    python scripts/diagram.py --word NESEN --size 2x2 --types timeslice
"""
from __future__ import annotations

import argparse
import os

import cairosvg

from qec_tile.circuit import memory_z_base
from qec_tile.directional import build_directional_code
from qec_tile.tile import TILES, paper_code
from qec_tile.walk2 import walk_memory_z_base

# stim의 그림 식별자, 우리 짧은 이름으로 키를 잡았다.
DIAGRAMS = {
    "timeslice": "timeslice-svg",       # TICK moment마다 프레임 하나
    "timeline": "timeline-svg",         # qubit x 시간
    "detslice": "detslice-with-ops-svg",  # 어떤 측정이 detector로 들어가는지
}


def render(circuit, diagram_type: str, out_stem: str,
           as_svg: bool = False) -> None:
    """``{out_stem}.png``를 쓴다. ``as_svg``면 원본 SVG를 그대로 쓴다."""
    svg = str(circuit.diagram(DIAGRAMS[diagram_type]))
    if as_svg:
        path = f"{out_stem}.svg"
        with open(path, "w") as f:
            f.write(svg)
    else:
        path = f"{out_stem}.png"
        cairosvg.svg2png(bytestring=svg.encode(), write_to=path,
                         output_width=2400)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="b3w6", choices=sorted(TILES))
    ap.add_argument("--L", type=int, default=2,
                    help="layout size; keep small or the frames get crowded")
    ap.add_argument("--rounds", type=int, default=1,
                    help="frames grow linearly with rounds; 1 is readable")
    ap.add_argument("--types", default="timeslice,timeline,detslice",
                    type=lambda spec: spec.split(","))
    ap.add_argument("--svg", action="store_true",
                    help="write SVG instead of PNG")
    ap.add_argument("--word", default=None,
                    help="directional word (e.g. NESEN); draws the walk circuit")
    ap.add_argument("--size", default=None,
                    help="directional MxN, e.g. 2x2 (required with --word)")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    unknown = set(args.types) - set(DIAGRAMS)
    if unknown:
        ap.error(f"unknown diagram types {sorted(unknown)}; "
                 f"have {sorted(DIAGRAMS)}")
    if args.word and not args.size:
        ap.error("--word requires --size")

    os.makedirs(args.out_dir, exist_ok=True)
    if args.word:
        M, N = (int(part) for part in args.size.lower().split("x"))
        code = build_directional_code(args.word, M, N)
        circuit = walk_memory_z_base(code, args.word, rounds=args.rounds)
        label = f"{args.word}_{M}x{N}"
    else:
        code = paper_code(args.tile, args.L, args.L)
        circuit = memory_z_base(code, rounds=args.rounds)
        label = f"{args.tile}_L{args.L}"
    print(code)
    for diagram_type in args.types:
        out_stem = os.path.join(
            args.out_dir, f"{label}_rounds{args.rounds}_{diagram_type}")
        render(circuit, diagram_type, out_stem, as_svg=args.svg)


if __name__ == "__main__":
    main()
