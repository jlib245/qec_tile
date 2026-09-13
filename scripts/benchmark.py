"""배치 크기와 p에 걸쳐 logical 오류율을 훑는다 -> CSV.

계산만 한다. 그림은 CSV를 따로 읽어 그린다. (tile, decoder, noise, shots, seed)마다
파일 하나이고, 이름은 data/{tile}_{decoder}_{noise}_shots{shots}_seed{seed}.csv로
자동으로 붙는다. 행은 하나씩 덧붙이고 그때마다 flush하며, 다시 돌리면 그 파일에 이미
있는 (L, p) 행은 건너뛴다. 그래서 훑기를 이어 붙이거나 중단된 곳에서 재개할 수 있다.

잡음 모델 (--noise, 일부러 명시하게 했다):
    capacity   i.i.d. X 오류, syndrome 측정은 완벽
    pheno      phenomenological: 측정 비트도 뒤집힌다 (--meas-error, 기본값 = p),
               --rounds 만큼의 측정 라운드 (기본값: L)
    circuit    균일 잡음의 memory-Z: 모든 gate, 측정, reset이 p로 실패하고 idle
               잡음은 없다
    si1000     SI1000 (Gidney 2021) 아래의 memory-Z: 초전도 기반으로 2q gate p,
               측정 5p, idle p/10, 측정-idle 2p
    uniform    directional 논문 Table 7의 균일 잡음: gate, 측정, reset, idle 전부 p

directional 모드는 --routing으로 배치를 고른다: plain은 walk_layout 그대로,
optimised는 optimise_routing이 줄인 배치 (파일명 stem에 _opt가 붙는다).

--meas-error나 --rounds를 덮어쓰면 기본 파일명은 그대로인데 숫자가 달라진다. 그럴
때는 --out을 명시적으로 넘긴다.

``rate`` 열은 블록 비율이다 -- 코드의 k개 observable 중 하나라도 뒤집히면 그 shot이
실패다. ``flips``는 shot에 걸쳐 뒤집힌 observable 수의 합이라 ``flips/(shots*k)``가
per-logical 실측이고, plot.py --per-logical이 그것을 쓴다 (없는 옛 CSV는 독립 가정
식으로 k를 되나눈다). 여기서는 k로 나누지 않는다.

사용법:
    python scripts/benchmark.py --decoder bposd --noise capacity
    python scripts/benchmark.py --decoder bposd --noise pheno --Ls 4,6,8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from qec_tile.circuit import (circuit_failure_counts, memory_z_base,
                              memory_z_circuit)
from qec_tile.decode import (DECODERS, code_capacity_counts, failure_counts)
from qec_tile.noise_model import NoiseModel
from qec_tile.pheno import spacetime_channel, spacetime_matrices
from qec_tile import config
from qec_tile.parallel_sampling import parallel_failure_counts
from qec_tile.sinter_sampling import SINTER_DECODERS, collect
from qec_tile.tile import paper_code
from qec_tile.directional import build_directional_code
from qec_tile.walk2 import optimise_routing, walk_memory_z_base


def iter_codes(args):
    """훑을 크기마다 (label, code, rounds, layout, shifts)를 yield한다.

    tile 모드는 --Ls를 걷고(label은 str(L)), directional 모드는 --sizes를 걷는다
    (label은 "MxN"). label을 문자열로 둔 것은 CSV의 'L' 열과 재개 키가 두 모드에서 한
    가지 타입으로 남게 하기 위해서다. ``layout``/``shifts``는 directional의 줄인
    배치이고, 아니면 ``(None, ())`` -- ``walk_memory_z_base``의 기본값이다.
    """
    if args.word:                                    # directional
        for M, N in args.sizes:
            code = build_directional_code(args.word, M, N)
            layout, shifts = ((None, ()) if args.routing == "plain"
                              else optimise_routing(code, args.word))
            yield f"{M}x{N}", code, (args.rounds or max(M, N)), layout, shifts
    else:                                            # 원래 tile
        for L in args.Ls:
            code = paper_code(args.tile, L, L)
            yield str(L), code, (args.rounds or L), None, ()


def uniform_noise(p: float) -> NoiseModel:
    """directional 논문 Table 7: 2q gate, 1q, 측정, reset, idle 전부 p.

    ``measure_reset_idle``은 0으로 둔다 -- walk 회로는 측정/reset이 gate와 같은
    moment에 들어가서 둘 다 켜면 그 moment의 idle qubit이 두 번 맞는다. ``idle``이
    moment마다 안 쓰인 qubit에 한 번 건다 (gate 없이 R/M만 있는 라운드 양끝 moment는
    빠진다).
    """
    return NoiseModel(idle=p, measure_reset_idle=0.0,
                      noisy_gates={"CX": p, "CXSWAP": p, "SWAP": p, "R": p,
                                   "RX": p, "M": p, "MR": p, "MRX": p})

FIELDS = ["tile", "decoder", "noise", "rounds", "L", "n", "k", "p",
          "meas_error", "seed", "shots", "fails", "flips", "rate", "sec"]


def parse_floats(spec: str) -> list[float]:
    """"a:b:n" -> a부터 b까지 양끝 포함 n개 점. 아니면 콤마 목록."""
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
    """빠진 (L, p) 점 전부에 대해 sinter.collect를 한 번.

    serial 루프와 달리 seed와 sec은 비워둔다 (worker 스케줄링이 비결정적이고 점별
    시간은 정의되지 않는다). shots에는 실제로 돌아간 수를 적는다 (--max-errors가 점을
    일찍 멈출 수 있다).
    """
    circuits, codes = {}, {}
    for label, code, rounds, layout, shifts in iter_codes(args):
        codes[label] = (code, rounds)
        for p in args.ps:
            if (label, p) in done:
                print(f"skip  L={label} p={p}")
                continue
            base = (walk_memory_z_base(code, args.word, rounds, layout, shifts)
                    if args.word else memory_z_base(code, rounds))
            if args.noise == "circuit":        # 균일 p: 2q gate 이름이 회로마다 다르다
                model = NoiseModel(
                    idle=0.0, measure_reset_idle=0.0,
                    noisy_gates={"CX": p, "CXSWAP": p, "SWAP": p, "R": p,
                                 "RX": p, "M": p, "MR": p, "MRX": p})
                circuits[(label, p)] = model.noisy_circuit(base)
            elif args.noise == "uniform":
                circuits[(label, p)] = uniform_noise(p).noisy_circuit(base)
            else:                              # si1000
                circuits[(label, p)] = NoiseModel.SI1000(p).noisy_circuit(base)
    if not circuits:
        return
    if not args.reproducible:
        for key, counts in sorted(collect(
                circuits, args.decoder, max_shots=args.shots,
                max_errors=args.max_errors, workers=args.workers,
                max_iter=args.max_iter).items()):
            write_point(writer, csv_file, args, codes, key, counts)
        return

    # 재현 경로: 점마다 돌고 바로 쓴다. 모아뒀다가 쓰면 중간에 죽을 때 끝난 점까지
    # 잃는다 (상태 파일에서는 이미 지워져 이어받지도 못한다).
    header = dict(decoder=args.decoder, seed=args.seed, chunk=args.chunk,
                  shots=args.shots, max_errors=args.max_errors,
                  max_iter=args.max_iter)
    state_path = _state_path(args.out)
    points = load_state(state_path, header)
    for key, circuit in sorted(circuits.items()):
        tag = f"{key[0]}|{key[1]}"
        at = points.get(tag, {})
        last = [0.0, at.get("chunks_done", 0)]   # [마지막 저장 시각, 조각 수]

        # 기본 인자로 묶는 이유: 파이썬은 루프 변수를 늦게 묶어서, 안 묶으면 모든
        # 점이 마지막 tag를 쓴다. 점이 하나면 안 드러나는 종류의 버그다.
        def note(done, shots, fails, flips, tag=tag, last=last,
                 gap=args.workers):
            now = time.monotonic()
            if now - last[0] < STATE_INTERVAL and done - last[1] < gap:
                return
            last[:] = [now, done]
            points[tag] = dict(chunks_done=done, shots=shots,
                               fails=fails, flips=flips)
            save_state(state_path, header, points)

        if at:
            print(f"resume L={key[0]} p={key[1]} "
                  f"({at['shots']:,} shots already)", file=sys.stderr)
        counts = parallel_failure_counts(
            circuit, args.shots, args.decoder, seed=args.seed,
            workers=args.workers, chunk=args.chunk,
            max_errors=args.max_errors, max_iter=args.max_iter,
            start_chunk=at.get("chunks_done", 0),
            start_counts=(at.get("shots", 0), at.get("fails", 0),
                          at.get("flips", 0)),
            checkpoint=note)
        write_point(writer, csv_file, args, codes, key, counts)
        points.pop(tag, None)            # 이 점은 CSV에 들어갔다
        save_state(state_path, header, points)


def write_point(writer, csv_file, args, codes, key, counts) -> None:
    """점 하나를 CSV에 적는다. 두 수집 경로가 공유한다."""
    label, p = key
    code, rounds = codes[label]
    rate = counts.block_fails / counts.shots
    writer.writerow(dict(
        tile=(args.word or args.tile), decoder=args.decoder,
        noise=args.noise, rounds=rounds, L=label, n=code.n, k=code.k,
        p=p, meas_error="",
        seed=(args.seed if args.reproducible else ""),
        shots=counts.shots,
        fails=counts.block_fails, flips=counts.logical_flips,
        rate=rate, sec=""))
    csv_file.flush()
    print(f"done  L={label} p={p} rate={rate:.4f} ({counts.shots} shots)")


# 상태 파일 저장 조건: 이 시간이 지나거나 worker 한 바퀴만큼 조각이 끝나면, 둘 중
# 먼저 오는 쪽에 쓴다. 시간만 쓰면 빠른 디코더에서 그 사이에 조각 수십 개가 끝나
# 날아가고, 조각 수만 쓰면 느린 디코더에서 저장이 너무 드물어진다. 파일은 1KB
# 미만이라 쓰기 자체는 싸다.
STATE_INTERVAL = 30.0


def _state_path(out: str) -> str:
    """CSV 옆의 상태 파일. 완주하면 지워지므로 존재 자체가 "중단됨"을 뜻한다."""
    return out + ".state.json"


def load_state(path: str, header: dict) -> dict:
    """이어받을 점들. 머리말이 다르면 버린다.

    chunk나 seed가 다르면 조각 경계와 shot 집합이 어긋나므로, 조용히 이어받으면
    서로 다른 실행의 데이터가 한 행에 섞인다. 반대로 말없이 처음부터 시작하면
    사용자가 왜 이어받지 않는지 모른다. 그래서 버리되 이유를 찍는다.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        state = json.load(f)
    stale = {k: (state.get(k), v) for k, v in header.items()
             if state.get(k) != v}
    if stale:
        print(f"이전 상태를 버립니다 ({path}): "
              + ", ".join(f"{k} {was!r} -> {now!r}"
                          for k, (was, now) in stale.items()),
              file=sys.stderr)
        return {}
    return state.get("points", {})


def save_state(path: str, header: dict, points: dict) -> None:
    """원자적으로 갈아끼운다. 쓰는 도중 죽어도 기존 상태가 남는다.

    같은 파일에 직접 쓰다 죽으면 JSON이 잘려 다음 실행의 ``json.load``가 터지고,
    그러면 이어받기가 아니라 전체를 다시 돌게 된다.
    """
    if not points:                       # 다 끝났다 -- 흔적을 남기지 않는다
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({**header, "points": points}, f)
    os.replace(tmp, path)


def already_done(path: str) -> set[tuple]:
    """이미 있는 (label, p) 키 — 나머지는 파일명이 고정한다.

    L 열은 크기 라벨이다: tile은 str(L), directional은 "MxN".
    """
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(row["L"], float(row["p"]))
                for row in csv.DictReader(f)}


# directional 논문(arXiv:2606.19482)이 "15 min-sum iterations per round"로 적은
# 설정을 쓰는 디코더들. vibelsd_32는 VibeLSD 논문(20회, round와 무관)을 따르므로
# 여기 없다.
PER_ROUND_ITER = {"vibelsd_200", "vibecoset_200",
                  "vibelsd_200_m20", "vibecoset_200_m20",
                  "vibelsd_200_m30", "vibecoset_200_m30",
                  "vibelsd_200_m50", "vibecoset_200_m50"}


def sweep_rounds(args) -> set[int]:
    """이 sweep이 쓸 round 수들. iter_codes의 기본값 규칙과 같아야 한다."""
    if args.rounds:
        return {args.rounds}
    if args.word:
        return {max(M, N) for M, N in args.sizes}
    return set(args.Ls)


def resolve_max_iter(decoder: str, explicit: int | None,
                     rounds: set[int], workers: int | None) -> int | None:
    """--max-iter 해석. None이면 디코더 자신의 기본값을 쓴다.

    directional 논문은 "15 min-sum iterations per round"로 적었다 -- rounds에
    비례한다. 해당하는 디코더는 ``PER_ROUND_ITER``에 있고, VibeLSD 논문(앙상블 32,
    20회)을 따르는 vibelsd_32는 거기 없으므로 건드리지 않는다.

    sinter는 모든 task에 디코더 하나를 쓰므로 값이 하나여야 한다. 크기마다
    round가 다르면 정할 수 없어 거부한다.

    단일 프로세스 경로(``workers`` 없음)는 max_iter를 아직 안 쓴다. 거기서 값을
    정하면 파일명에만 ``_iter60``이 붙고 실제로는 디코더 기본값으로 돌아
    CSV가 거짓말을 한다.
    """
    if workers is None:
        return None
    if explicit is not None:
        return explicit
    if decoder not in PER_ROUND_ITER:
        return None
    if len(rounds) != 1:
        raise ValueError(f"--rounds가 크기마다 다르다 ({sorted(rounds)}). "
                         f"--rounds나 --max-iter를 명시하라")
    return 15 * next(iter(rounds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", required=True,
                    choices=sorted(set(DECODERS) | set(SINTER_DECODERS)))
    ap.add_argument("--noise", required=True,
                    choices=["capacity", "pheno", "circuit", "si1000",
                             "uniform"])
    ap.add_argument("--routing", default="plain",
                    choices=["plain", "optimised"],
                    help="directional only: walk_layout as is, or the layout "
                         "optimise_routing shrinks (_opt in the file name)")
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
    ap.add_argument("--max-iter", type=int, default=None,
                    help="BP iterations; vibelsd_200 defaults to 15 per round")
    route = ap.add_mutually_exclusive_group()
    route.add_argument("--sinter", action="store_true",
                       help="collect through sinter (cannot be seeded)")
    route.add_argument("--parallel", action="store_true",
                       help="seeded multiprocessing pool (the default)")
    ap.add_argument("--chunk", type=int, default=100,
                    help="shots per chunk in the reproducible path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.word and not args.sizes:
        ap.error("--word requires --sizes")
    if args.sizes and not args.word:
        ap.error("--sizes requires --word")
    if args.routing != "plain" and not args.word:
        ap.error("--routing applies to directional mode only")
    if args.noise != "pheno" and args.meas_error is not None:
        ap.error("--meas-error only applies to --noise pheno")
    if args.workers is not None and args.noise not in ("circuit", "si1000",
                                                       "uniform"):
        ap.error("--workers only applies to --noise circuit/si1000/uniform")
    if args.workers == -1:                     # 맨 --workers: 할당량을 쓴다
        args.workers = config.workers()
    if args.workers is not None and args.workers <= 0:
        ap.error("--workers must be a positive integer")
    if args.max_errors is not None and args.workers is None:
        ap.error("--max-errors requires --workers")
    if args.decoder not in DECODERS and args.workers is None:
        ap.error(f"--decoder {args.decoder} is sinter-only (needs --workers)")
    if args.max_iter is not None and args.workers is None:
        ap.error("--max-iter is only wired through the sinter path "
                 "(needs --workers)")
    try:
        args.max_iter = resolve_max_iter(args.decoder, args.max_iter,
                                         sweep_rounds(args), args.workers)
    except ValueError as exc:
        ap.error(str(exc))
    # 수집기는 --sinter/--parallel이 고른다. 둘 다 없으면 재현되는 쪽이 기본이다.
    if (args.sinter or args.parallel) and args.workers is None:
        ap.error("--sinter/--parallel need --workers")
    args.reproducible = args.workers is not None and not args.sinter
    if args.sinter and args.seed is not None:
        ap.error("sinter cannot be seeded; drop --seed or use --parallel")
    if args.seed is None:
        args.seed = 42 if args.reproducible else 0

    if args.out is None:
        stem = args.word if args.word else args.tile
        if args.routing == "optimised":
            stem += "_opt"
        suffix = "" if args.max_iter is None else f"_iter{args.max_iter}"
        marker = "_rep" if args.reproducible else ""   # sinter 결과와 섞이지 않게
        args.out = (f"data/{stem}_{args.decoder}_{args.noise}"
                    f"_shots{args.shots}{suffix}{marker}_seed{args.seed}.csv")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = already_done(args.out)
    is_new = (not os.path.exists(args.out)
              or os.path.getsize(args.out) == 0)   # 죽은 실행이 남긴 껍데기

    with open(args.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        if args.workers is not None:
            parallel_sweep(args, writer, f, done)
            return
        for label, code, rounds, layout, shifts in iter_codes(args):
            if args.noise == "capacity":
                rounds = 1
            if args.noise == "pheno":
                H, L_obs = spacetime_matrices(code, rounds)   # p에 무관
            for p in args.ps:
                if (label, p) in done:
                    print(f"skip  L={label} p={p}")
                    continue
                start = time.time()
                if args.noise == "capacity":
                    meas_error = 0.0
                    counts = code_capacity_counts(code, p, args.shots,
                                                  args.decoder, seed=args.seed)
                elif args.noise == "pheno":
                    meas_error = (p if args.meas_error is None
                                  else args.meas_error)
                    channel = spacetime_channel(code, rounds, p, meas_error)
                    counts = failure_counts(H, L_obs, channel, args.shots,
                                            args.decoder, seed=args.seed)
                else:                          # 회로 잡음: p가 이미 박혀 있다
                    meas_error = ""
                    base = (walk_memory_z_base(code, args.word, rounds,
                                               layout, shifts)
                            if args.word else memory_z_base(code, rounds))
                    if args.noise == "circuit":
                        circuit = NoiseModel(
                            idle=0.0, measure_reset_idle=0.0,
                            noisy_gates={"CX": p, "CXSWAP": p, "SWAP": p,
                                         "R": p, "RX": p, "M": p, "MR": p,
                                         "MRX": p}).noisy_circuit(base)
                    elif args.noise == "uniform":
                        circuit = uniform_noise(p).noisy_circuit(base)
                    else:                      # si1000
                        circuit = NoiseModel.SI1000(p).noisy_circuit(base)
                    counts = circuit_failure_counts(circuit, args.shots,
                                                    args.decoder,
                                                    seed=args.seed)
                rate = counts.block_fails / counts.shots
                row = dict(tile=(args.word or args.tile), decoder=args.decoder,
                           noise=args.noise, rounds=rounds, L=label, n=code.n,
                           k=code.k, p=p, meas_error=meas_error,
                           seed=args.seed, shots=counts.shots,
                           fails=counts.block_fails, flips=counts.logical_flips,
                           rate=rate, sec=round(time.time() - start, 1))
                writer.writerow(row)
                f.flush()
                print(f"done  L={label} p={p} rate={rate:.4f} ({row['sec']}s)")


if __name__ == "__main__":
    main()
