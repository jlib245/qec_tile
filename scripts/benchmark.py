"""Sweep logical error rate over layout size and p -> CSV.

Calculation only; plotting reads the CSV separately.  One file per
(tile, decoder, noise, shots, seed), auto-named as
data/{tile}_{decoder}_{noise}_shots{shots}_seed{seed}.csv.  Rows are appended
and flushed one at a time, and a re-run skips (L, p) rows already in that
file, so a sweep can be extended or resumed without redoing work.

Noise models (--noise, explicit on purpose):
    capacity   i.i.d. X errors, perfect syndrome measurement
    pheno      phenomenological: measurement bits flip too (--meas-error,
               default = p), --rounds rounds of measurement (default: L)
    circuit    memory-Z with uniform noise: every gate, measurement and
               reset fails with p, no idle noise
    si1000     memory-Z under SI1000 (Gidney 2021): superconducting-inspired,
               2q gates p, measure 5p, idle p/10, measure-idle 2p

Overriding --meas-error or --rounds changes the numbers without changing the
default filename; pass --out explicitly in that case.

Usage:
    python scripts/benchmark.py --decoder bposd --noise capacity
    python scripts/benchmark.py --decoder bposd --noise pheno --Ls 4,6,8
"""
from __future__ import annotations

import argparse
import csv
import os
import time

from qec_tile.circuit import (circuit_failure_rate, memory_z_base,
                              memory_z_circuit)
from qec_tile.decode import DECODERS, failure_rate, logical_error_rate
from qec_tile.noise_model import NoiseModel
from qec_tile.pheno import spacetime_channel, spacetime_matrices
from qec_tile import config
from qec_tile.sinter_sampling import collect
from qec_tile.tile import paper_code
from qec_tile.directional import build_directional_code


def iter_codes(args):
    """Yield (label, code, rounds) for each size to sweep.

    Tile mode walks --Ls (label is str(L)); directional mode walks --sizes
    (label is "MxN").  Labels are strings so the CSV 'L' column and the resume
    key stay one type across both.
    """
    if args.word:                                    # directional
        for M, N in args.sizes:
            code = build_directional_code(args.word, M, N)
            yield f"{M}x{N}", code, (args.rounds or max(M, N))
    else:                                            # original tile
        for L in args.Ls:
            code = paper_code(args.tile, L, L)
            yield str(L), code, (args.rounds or L)

FIELDS = ["tile", "decoder", "noise", "rounds", "L", "n", "k", "p",
          "meas_error", "seed", "shots", "fails", "rate", "sec"]


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


def parse_sizes(spec: str) -> list[tuple[int, int]]:
    """"4x4,8x8" -> [(4, 4), (8, 8)]."""
    sizes = []
    for token in spec.split(","):
        m, n = token.lower().split("x")
        sizes.append((int(m), int(n)))
    return sizes


def parallel_sweep(args, writer, csv_file, done) -> None:
    """One sinter.collect over every missing (L, p) point.

    Unlike the serial loop, seed and sec are left blank (worker scheduling is
    nondeterministic and per-point timing undefined) and shots records what
    actually ran (--max-errors can stop a point early).
    """
    circuits, codes = {}, {}
    for label, code, rounds in iter_codes(args):
        codes[label] = (code, rounds)
        for p in args.ps:
            if (label, p) in done:
                print(f"skip  L={label} p={p}")
                continue
            if args.noise == "circuit":
                circuits[(label, p)] = memory_z_circuit(code, rounds, p)
            else:                              # si1000
                circuits[(label, p)] = NoiseModel.SI1000(p).noisy_circuit(
                    memory_z_base(code, rounds))
    if not circuits:
        return
    stats = collect(circuits, args.decoder, max_shots=args.shots,
                    max_errors=args.max_errors, workers=args.workers)
    for (label, p), (shots, fails) in sorted(stats.items()):
        code, rounds = codes[label]
        writer.writerow(dict(
            tile=(args.word or args.tile), decoder=args.decoder,
            noise=args.noise, rounds=rounds, L=label, n=code.n, k=code.k,
            p=p, meas_error="", seed="", shots=shots, fails=fails,
            rate=fails / shots, sec=""))
        csv_file.flush()
        print(f"done  L={label} p={p} rate={fails / shots:.4f} ({shots} shots)")


def already_done(path: str) -> set[tuple]:
    """Keys (label, p) already present — the filename fixes the rest.

    The L column is a size label: str(L) for tiles, "MxN" for directional.
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(row["L"], float(row["p"]))
                for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True, choices=sorted(DECODERS))
    ap.add_argument("--noise", required=True,
                    choices=["capacity", "pheno", "circuit", "si1000"])
    ap.add_argument("--tile", default="b3w6")
    ap.add_argument("--Ls", default="4,6,8,10", type=parse_ints)
    ap.add_argument("--word", default=None,
                    help="directional compass word (e.g. N2ESEN2); "
                         "switches to directional mode")
    ap.add_argument("--sizes", type=parse_sizes,
                    help="directional MxN sizes, e.g. 4x4,8x8")
    ap.add_argument("--ps", default="0.04:0.10:7", type=parse_floats)
    ap.add_argument("--rounds", type=int, default=None,
                    help="pheno/circuit; default: rounds = L")
    ap.add_argument("--meas-error", type=float, default=None,
                    help="pheno only: syndrome-bit flip probability; "
                         "default: same as p")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=None,
                    help="serial runs only; default 0")
    ap.add_argument("--workers", type=int, nargs="?", const=-1, default=None,
                    help="circuit/si1000: parallel collection via sinter; "
                         "bare --workers uses QEC_TILE_WORKERS from .env")
    ap.add_argument("--max-errors", type=int, default=None,
                    help="stop a point after this many errors (needs --workers)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.word and not args.sizes:
        ap.error("--word requires --sizes")
    if args.sizes and not args.word:
        ap.error("--sizes requires --word")
    if args.word and args.noise not in ("capacity", "pheno"):
        ap.error("directional (--word) supports only capacity/pheno noise "
                 "(no CXSWAP circuit yet)")
    if args.noise != "pheno" and args.meas_error is not None:
        ap.error("--meas-error only applies to --noise pheno")
    if args.workers is not None and args.noise not in ("circuit", "si1000"):
        ap.error("--workers only applies to --noise circuit/si1000")
    if args.workers == -1:                     # bare --workers: the allocation
        args.workers = config.workers()
    if args.workers is not None and args.workers <= 0:
        ap.error("--workers must be a positive integer")
    if args.max_errors is not None and args.workers is None:
        ap.error("--max-errors requires --workers")
    if args.workers is not None and args.seed is not None:
        ap.error("--seed has no effect with --workers "
                 "(sinter's scheduling is nondeterministic)")
    if args.seed is None:
        args.seed = 0                          # serial default

    if args.out is None:
        stem = args.word if args.word else args.tile
        args.out = (f"data/{stem}_{args.decoder}_{args.noise}"
                    f"_shots{args.shots}_seed{args.seed}.csv")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = already_done(args.out)
    is_new = (not os.path.exists(args.out)
              or os.path.getsize(args.out) == 0)   # a killed run's leftover

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        if args.workers is not None:
            parallel_sweep(args, writer, f, done)
            return
        for label, code, rounds in iter_codes(args):
            if args.noise == "capacity":
                rounds = 1
            if args.noise == "pheno":
                H, L_obs = spacetime_matrices(code, rounds)   # p-independent
            for p in args.ps:
                if (label, p) in done:
                    print(f"skip  L={label} p={p}")
                    continue
                start = time.time()
                if args.noise == "capacity":
                    meas_error = 0.0
                    rate = logical_error_rate(code, p, args.shots,
                                              args.decoder, seed=args.seed)
                elif args.noise == "pheno":
                    meas_error = (p if args.meas_error is None
                                  else args.meas_error)
                    channel = spacetime_channel(code, rounds, p, meas_error)
                    rate = failure_rate(H, L_obs, channel, args.shots,
                                        args.decoder, seed=args.seed)
                else:                          # circuit noise: p is baked in
                    meas_error = ""
                    if args.noise == "circuit":
                        circuit = memory_z_circuit(code, rounds, p)
                    else:                      # si1000
                        circuit = NoiseModel.SI1000(p).noisy_circuit(
                            memory_z_base(code, rounds))
                    rate = circuit_failure_rate(circuit, args.shots,
                                                args.decoder, seed=args.seed)
                row = dict(tile=(args.word or args.tile), decoder=args.decoder,
                           noise=args.noise, rounds=rounds, L=label, n=code.n,
                           k=code.k, p=p, meas_error=meas_error,
                           seed=args.seed, shots=args.shots,
                           fails=round(rate * args.shots),
                           rate=rate, sec=round(time.time() - start, 1))
                writer.writerow(row)
                f.flush()
                print(f"done  L={label} p={p} rate={rate:.4f} ({row['sec']}s)")


if __name__ == "__main__":
    main()
