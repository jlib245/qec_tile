"""Render stim diagrams of a tile code's memory-Z circuit -> PNG (or SVG via --svg).

Draws the noise-free base circuit (noise operations would bury the structure).
QUBIT_COORDS in memory_z_base place everything on the real lattice, so the
timeslice frames show the actual tile geometry and the CNOT schedule.

Usage:
    python scripts/diagram.py
    python scripts/diagram.py --tile b4w8 --L 3 --types timeslice,detslice
"""
from __future__ import annotations

import argparse
import os

import cairosvg

from qec_tile.circuit import memory_z_base
from qec_tile.tile import TILES, paper_code

# stim's diagram identifiers, keyed by our short names.
DIAGRAMS = {
    "timeslice": "timeslice-svg",       # one frame per TICK moment
    "timeline": "timeline-svg",         # qubits x time
    "detslice": "detslice-with-ops-svg",  # which measurements feed a detector
}


def render(circuit, diagram_type: str, out_stem: str,
           as_svg: bool = False) -> None:
    """Write ``{out_stem}.png``, or the raw SVG instead when ``as_svg``."""
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
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()

    unknown = set(args.types) - set(DIAGRAMS)
    if unknown:
        ap.error(f"unknown diagram types {sorted(unknown)}; "
                 f"have {sorted(DIAGRAMS)}")

    os.makedirs(args.out_dir, exist_ok=True)
    code = paper_code(args.tile, args.L, args.L)
    circuit = memory_z_base(code, rounds=args.rounds)
    for diagram_type in args.types:
        out_stem = os.path.join(
            args.out_dir,
            f"{args.tile}_L{args.L}_rounds{args.rounds}_{diagram_type}")
        render(circuit, diagram_type, out_stem, as_svg=args.svg)


if __name__ == "__main__":
    main()
