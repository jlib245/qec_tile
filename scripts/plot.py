"""벤치마크 CSV에서 p 대비 logical 오류율 -> PNG.

benchmark.py가 쓴 CSV를 읽고 (decoder, L)마다 곡선 하나를 그린다. 오차 막대는 이항
sqrt(r(1-r)/shots)다. CSV를 여러 개 주면 겹쳐 그린다 -- 예를 들어 bposd 파일과 nn
파일을 한 축에서 비교할 때.

CSV의 ``rate``는 *블록* 비율이다: k개 observable 중 하나라도 뒤집히면 그 shot이
실패다. 물리적 성능이 같아도 k=8 tile code가 k=1 surface code에 비해 여덟 배로
불리해지므로, --per-logical이 logical qubit 하나의 비율로 바꾼다. ``flips`` 열
(뒤집힌 observable 수의 합)이 있으면 ``flips/(shots*k)``로 **직접 센** 값을 쓰고,
없는 옛 CSV는 k개가 독립 실패한다는 가정의 ``1-(1-rate)^(1/k)``로 되나눈다
(qec_tile.decode.per_logical_rate 참고).

--per-round는 여기에 ``rounds`` 열로 라운드당 비율 ``1-(1-q)^(1/rounds)``를 얹는다.
observable은 실험 끝에 한 번만 읽히므로 어느 라운드에서 뒤집혔는지는 관측할 수
없고, 라운드마다 독립·동일하게 실패한다는 가정이 남는다. directional 논문(Gu et al.,
arXiv:2606.19482)의 Table 6 "per-logical per-round logical error rate"가 이 단위인데
정의식은 밝히지 않았고, 이 독립 환산이 분야의 표준 관례다 (예: Bravyi et al.,
Nature 627 (2024)의 per-cycle 정의). 두 변환 모두 비율이 작을 때는 그냥 나눗셈과 같다.

출력 PNG는 기본적으로 입력 이름을 따르고 데이터 옆(data/, gitignore됨)에 쓴다:
CSV 하나면 그 stem, 여러 개면 공통 접두사에 "_compare"를 붙인다. README에 그림이
필요해지면 `git add -f`로 명시적으로 커밋한다.

사용법:
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

from qec_tile.decode import per_logical_rate

matplotlib.use("Agg")                      # headless: 화면 없이 파일로 쓴다
import matplotlib.pyplot as plt            # noqa: E402


def load(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def default_out(csv_paths: list[str]) -> str:
    """입력이 하나면 data/<stem>.png, 여러 개면 <공통접두사>_compare.png."""
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
    ap.add_argument("--per-logical", action="store_true",
                    help="rate per logical qubit: measured from the flips "
                         "column when present, else the block rate divided "
                         "out over k assuming independence")
    ap.add_argument("--per-round", action="store_true",
                    help="also divide out the CSV's rounds column, giving the "
                         "per-round rate the directional paper's Table 6 "
                         "reports (assumes rounds fail independently)")
    args = ap.parse_args()

    if args.out is None:
        args.out = default_out(args.csv)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # (decoder, noise, L)마다 곡선 하나. legend 라벨용으로 n을 함께 들고 간다.
    # L은 크기 라벨이다: tile은 "4", directional code는 "11x6".
    # 옛 CSV에는 noise 열이 없고 전부 code capacity였다.
    curves: dict[tuple, list] = defaultdict(list)
    for row in load(args.csv):
        key = (row["decoder"], row.get("noise", "capacity"),
               row["L"], int(row["n"]), int(row["k"]))
        flips = row.get("flips", "")
        curves[key].append((float(row["p"]), int(row["fails"]),
                            int(row["shots"]),
                            int(flips) if flips else None,
                            int(row.get("rounds") or 1)))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (decoder, noise, L, n, k), points in sorted(
            curves.items(), key=lambda kv: (kv[0][3], kv[0][2])):
        points.sort()
        ps, rates, errs = [], [], []
        for p, fails, shots, flips, rounds in points:
            if args.per_logical and flips is not None:
                # 직접 센 값: observable 하나당 뒤집힘 비율, 이항 오차.
                trials, hits = shots * k, flips
            else:
                trials, hits = shots, fails
            rate = hits / trials
            err = math.sqrt(max(rate * (1 - rate), 1e-12) / trials)
            # 남은 환산은 독립 가정: 1-(1-r)^(1/m). Delta method로 오차도 같이.
            m = ((k if args.per_logical and flips is None else 1)
                 * (rounds if args.per_round else 1))
            if m > 1:
                err *= max(1 - rate, 1e-12) ** (1 / m - 1) / m
                rate = per_logical_rate(rate, m)
            ps.append(p)
            rates.append(rate)
            errs.append(err)
        ax.errorbar(ps, rates, yerr=errs, marker="o", capsize=2,
                    label=f"[[{n},{k}]] L={L} ({decoder}, {noise})")

    unit = " per logical qubit" if args.per_logical else ""
    unit += " per round" if args.per_round else ""
    ax.set_xlabel("physical error rate  p")
    ax.set_ylabel(f"logical error rate{unit}" if unit
                  else "block error rate")
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
