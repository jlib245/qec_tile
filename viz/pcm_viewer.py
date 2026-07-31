"""Export an interactive parity-check-matrix viewer -> one self-contained HTML.

Codes are built here, in Python, by the repository's own constructors, and
baked into the page as JSON; the browser only draws.  That keeps the matrix on
screen identical to the one the benchmarks decode, at the cost of a fixed
parameter list -- rerun with different --layouts to widen it.

Surface and bivariate-bicycle codes come from qec_pem.py next door, which is a
standalone reference implementation, not part of the qec_tile package.

Usage:
    python viz/pcm_viewer.py
    python viz/pcm_viewer.py --layouts 4,8 --no-directional
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from qec_pem import (BB_CATALOG, bb_code, rotated_surface_code, toric_code,
                     unrotated_surface_code)

from qec_tile.directional import PAPER_CODES, build_directional_code
from qec_tile.gf2 import rank2
from qec_tile.tile import TILES, paper_code

SURFACE_DISTANCES = (3, 5, 7)      # rotated needs odd d; keep the others aligned
TORIC_SIZES = (3, 4, 5)
TEMPLATE = Path(__file__).with_name("pcm_viewer_template.html")
# Generated pages belong with the other outputs, not next to their source.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def matrix_payload(label: str, HX, HZ, note: str = "", dividers=(),
                   block=None) -> dict:
    """One code as a JSON-ready dict: sparse rows plus the stat-card numbers.

    Rows travel as ascending column indices -- a weight-6 check is 6 numbers
    instead of n.  ``dividers`` are qubit-sector boundaries (HGP and BB codes
    split into two blocks); the viewer draws a dashed line at each.

    ``block`` is where the index arithmetic folds -- ``{"rows_x", "rows_z",
    "cols"}`` -- so the viewer can rule the grid on the code's own period
    rather than an arbitrary five.  Rows count checks and columns count
    qubits, which is why the two are given apart, and ``cols`` carries one
    width per sector.  None for a code built geometrically, which has no such
    period.
    """
    HX = np.asarray(HX, dtype=np.uint8)
    HZ = np.asarray(HZ, dtype=np.uint8)
    n = int(HX.shape[1])
    # int64 first: a uint8 matmul wraps at 256 and can fake a zero overlap.
    overlap = (HX.astype(np.int64) @ HZ.astype(np.int64).T) % 2
    return dict(
        label=label,
        note=note,
        n=n,
        k=int(n - rank2(HX) - rank2(HZ)),
        mx=int(HX.shape[0]),
        mz=int(HZ.shape[0]),
        wx=int(HX.sum(1).max()) if HX.size else 0,
        wz=int(HZ.sum(1).max()) if HZ.size else 0,
        css=not bool(overlap.any()),
        dividers=[int(d) for d in dividers],
        block=block,
        X=[np.flatnonzero(row).tolist() for row in HX],
        Z=[np.flatnonzero(row).tolist() for row in HZ],
    )


def build_catalog(layouts=(4, 6, 8, 10), directional: bool = True,
                  extras: bool = True) -> list[dict]:
    """Every code the viewer offers, in select order.

    ``group`` names the family the select lists; ``param`` is what varies
    inside it, which the slider steps through.
    """
    entries: list[dict] = []

    def add(group, param, HX, HZ, note="", dividers=(), block=None):
        entries.append(dict(matrix_payload(f"{group}  {param}", HX, HZ,
                                           note, dividers, block),
                            group=group, param=param))

    for name in sorted(TILES):
        B = 3 if name.startswith("b3") else 4
        for L in layouts:
            code = paper_code(name, L, L)
            # Anchors fold every L2 rows, qubits every L2+B-1 columns: the
            # qubit lattice is B-1 wider than the anchor grid.  The H and V
            # sectors have the same shape, so both widths are the same.
            add(f"tile {name}", f"L={L}", code.HX, code.HZ,
                note=f"Tile {name} on {L}x{L} bulk anchors. "
                     f"Short rows are boundary tiles cut by the lattice edge.",
                dividers=(code.n // 2,),
                block=dict(rows_x=L, rows_z=L, cols=[L + B - 1, L + B - 1]))

    if directional:
        for word, M, N, n, k, d in PAPER_CODES:
            code = build_directional_code(word, M, N)
            add("directional", f"{word}, {M}x{N}", code.HX, code.HZ,
                note=f"Compass walk {word} on an {M}x{N} anchor grid. "
                     f"Paper reports [[{n},{k},{d}]].",
                dividers=(code.n // 2,),
                block=dict(rows_x=code.L2, rows_z=code.L2,
                           cols=[code.L2 + code.B - 1] * 2))

    if extras:
        for d in SURFACE_DISTANCES:
            add("rotated surface", f"d={d}", *rotated_surface_code(d),
                note=f"[[{d * d},1,{d}]]. Geometric, no product block "
                     f"structure.")
        for d in SURFACE_DISTANCES:
            # HGP of two length-d repetition codes: n1 = n2 = d, m1 = m2 = d-1.
            # H_X rows are indexed (a, j) and H_Z rows (i, b), so the two fold
            # at different heights.
            add("unrotated surface", f"d={d}", *unrotated_surface_code(d),
                note="Hypergraph product of two repetition codes; the dashed "
                     "line splits the two qubit sectors.",
                dividers=(d * d,),
                block=dict(rows_x=d, rows_z=d - 1, cols=[d, d - 1]))
        for L in TORIC_SIZES:
            add("toric", f"L={L}", *toric_code(L),
                note="Same blocks as unrotated, plus the wrap-around entries.",
                dividers=(L * L,),
                block=dict(rows_x=L, rows_z=L, cols=[L, L]))
        for name, params in BB_CATALOG.items():
            add("bivariate bicycle", name,
                *bb_code(params["l"], params["m"], params["A"], params["B"]),
                note=f"l={params['l']}, m={params['m']}. "
                     f"H_X = [A|B], H_Z = [B^T|A^T].",
                dividers=(params["l"] * params["m"],),
                # idx = i*m + j, so a block is one m x m circulant and the
                # blocks sit in an l x l grid.
                block=dict(rows_x=params["m"], rows_z=params["m"],
                           cols=[params["m"], params["m"]]))

    return entries


def render(entries: list[dict], template_path=TEMPLATE) -> str:
    """The template with the catalog baked into its one placeholder."""
    payload = json.dumps(entries, separators=(",", ":"))
    # HTML ends a script at the first </script>, string literal or not.
    payload = payload.replace("</", "<\\/")
    template = Path(template_path).read_text()
    if "__PAYLOAD__" not in template:
        raise ValueError(f"{template_path}: no __PAYLOAD__ placeholder")
    return template.replace("__PAYLOAD__", payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_DIR / "pcm_viewer.html"))
    ap.add_argument("--layouts", default=(4, 6, 8, 10),
                    type=lambda spec: tuple(int(L) for L in spec.split(",")),
                    help="bulk sizes L to bake for each tile, e.g. 4,8")
    ap.add_argument("--no-directional", action="store_false",
                    dest="directional")
    ap.add_argument("--no-extras", action="store_false", dest="extras",
                    help="drop the surface and BB codes taken from qec_pem.py")
    args = ap.parse_args()

    entries = build_catalog(args.layouts, args.directional, args.extras)
    html = render(entries)
    Path(args.out).write_text(html)
    print(f"wrote {args.out}  "
          f"({len(entries)} codes, {len(html) / 1024:.0f} KiB, "
          f"largest n={max(entry['n'] for entry in entries)})")


if __name__ == "__main__":
    main()
