"""Plot logical error rate vs p from benchmark CSV(s) -> PNG.

Reads the CSVs written by benchmark.py and draws one curve per (decoder, L), with
binomial error bars sqrt(r(1-r)/shots).  Pass several CSVs to overlay them --
e.g. a bposd file and an nn file to compare decoders on one axis.

The output PNG is named after the input by default and written next to the
data (in data/, gitignored): one CSV -> its own stem, several -> their common
prefix plus "_compare".  Commit a figure explicitly with `git add -f` if a
README ever needs one.

Usage:
    python scripts/plot.py data/b3w6_bposd_shots100000_seed0.csv
    python scripts/plot.py data/b3w6_*.csv          # -> data/b3w6_compare.png
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # headless: write a file, no display
import matplotlib.pyplot as plt            # noqa: E402


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def default_out(csv_paths: list[str]) -> str:
    """data/<stem>.png for one input, <common-prefix>_compare.png for many."""
    stems = [Path(p).stem for p in csv_paths]
    if len(stems) == 1:
        name = stems[0]
    else:
        name = (os.path.commonprefix(stems).rstrip("_") or "plot") + "_compare"
    return f"data/{name}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = default_out(args.csv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # One curve per (decoder, noise, L); carry n along for the legend label.
    # Old CSVs predate the noise column and were all code capacity.
    curves: dict[tuple, list] = defaultdict(list)
    for row in load(args.csv):
        key = (row["decoder"], row.get("noise", "capacity"),
               int(row["L"]), int(row["n"]), int(row["k"]))
        curves[key].append((float(row["p"]), int(row["fails"]),
                            int(row["shots"])))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (decoder, noise, L, n, k), points in sorted(curves.items()):
        points.sort()
        ps = [p for p, _, _ in points]
        rates = [fails / shots for _, fails, shots in points]
        errs = [math.sqrt(max(r * (1 - r), 1e-12) / shots)
                for r, (_, _, shots) in zip(rates, points)]
        ax.errorbar(ps, rates, yerr=errs, marker="o", capsize=2,
                    label=f"[[{n},{k}]] L={L} ({decoder}, {noise})")

    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel("logical error rate")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}  ({len(curves)} curves)")


if __name__ == "__main__":
    main()
