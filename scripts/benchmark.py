"""Sweep logical error rate over layout size and p -> CSV.

Calculation only; plotting reads the CSV separately.  One file per
(tile, decoder, noise, shots, seed), auto-named as
data/{tile}_{decoder}_{noise}_shots{shots}_seed{seed}.csv.  Rows are appended
and flushed one at a time, and a re-run skips (L, p) rows already in that
file, so a sweep can be extended or resumed without redoing work.

Noise models (--noise, explicit on purpose):
    capacity   i.i.d. X errors, perfect syndrome measurement
    pheno      phenomenological: measurement bits flip too (q, default q=p),
               --rounds rounds of measurement (default: rounds = L)

Overriding --q or --rounds changes the numbers without changing the default
filename; pass --out explicitly in that case.

Usage:
    python scripts/benchmark.py --decoder bposd --noise capacity
    python scripts/benchmark.py --decoder bposd --noise pheno --Ls 4,6,8
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from qec_tile.decode import DECODERS, failure_rate, logical_error_rate
from qec_tile.pheno import spacetime_channel, spacetime_matrices
from qec_tile.tile import paper_code

FIELDS = ["tile", "decoder", "noise", "rounds", "L", "n", "k", "p", "q",
          "seed", "shots", "fails", "rate", "sec"]


def parse_floats(spec: str) -> list[float]:
    """"a:b:n" -> n points from a to b inclusive; else a comma list."""
    if ":" in spec:
        lo, hi, count = spec.split(":")
        lo, hi, count = float(lo), float(hi), int(count)
        step = (hi - lo) / (count - 1) if count > 1 else 0.0
        return [round(lo + i * step, 10) for i in range(count)]
    return [float(x) for x in spec.split(",")]


def parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",")]


def already_done(path: str) -> set[tuple]:
    """Keys (L, p) already present — the filename fixes the rest."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(int(row["L"]), float(row["p"]))
                for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, choices=sorted(DECODERS))
    ap.add_argument("--noise", required=True, choices=["capacity", "pheno"])
    ap.add_argument("--tile", default="b3w6")
    ap.add_argument("--Ls", default="4,6,8,10", type=parse_ints)
    ap.add_argument("--ps", default="0.04:0.10:7", type=parse_floats)
    ap.add_argument("--rounds", type=int, default=None,
                    help="pheno only; default: rounds = L")
    ap.add_argument("--q", type=float, default=None,
                    help="pheno only; default: q = p")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = (f"data/{args.tile}_{args.decoder}_{args.noise}"
                    f"_shots{args.shots}_seed{args.seed}.csv")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = already_done(args.out)
    is_new = not os.path.exists(args.out)

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        for L in args.Ls:
            code = paper_code(args.tile, L, L)
            if args.noise == "pheno":
                rounds = args.rounds or L
                H, L_obs = spacetime_matrices(code, rounds)   # p-independent
            for p in args.ps:
                if (L, p) in done:
                    print(f"skip  L={L} p={p}")
                    continue
                start = time.time()
                if args.noise == "capacity":
                    rounds, q = 1, 0.0
                    rate = logical_error_rate(code, p, args.shots,
                                              args.decoder, seed=args.seed)
                else:
                    q = p if args.q is None else args.q
                    channel = spacetime_channel(code, rounds, p, q)
                    rate = failure_rate(H, L_obs, channel, args.shots,
                                        args.decoder, seed=args.seed)
                row = dict(tile=args.tile, decoder=args.decoder,
                           noise=args.noise, rounds=rounds, L=L, n=code.n,
                           k=code.k, p=p, q=q, seed=args.seed,
                           shots=args.shots, fails=round(rate * args.shots),
                           rate=rate, sec=round(time.time() - start, 1))
                writer.writerow(row)
                f.flush()
                print(f"done  L={L} p={p} rate={rate:.4f} ({row['sec']}s)")


if __name__ == "__main__":
    main()
