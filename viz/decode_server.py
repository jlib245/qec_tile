"""Serve an interactive error -> syndrome -> decoding view of a CSS code.

The browser draws the layout and collects clicks; every number it shows comes
from this process, decoded by the same BP+OSD the benchmarks use, so what the
page reports is what qec_tile would report.

The X sector is the one on screen, following decode.py:

    e                 an X-error pattern (the qubits you click)
    s = HZ @ e        the lit checks
    e_hat = decode(s) the decoder's guess
    r = e ^ e_hat     the residual
    failure  <=>  LZ @ r != 0

Tile and directional codes come from qec_tile; surface, toric and bivariate
bicycle codes from qec_pem.py next door, which carries no coordinates, so the
layouts for those are built here (see the _view_* builders).

Run it (fastapi and uvicorn are in the dev dependency group):

    uv run python viz/decode_server.py
"""
# No `from __future__ import annotations` here: it stringifies annotations, and
# FastAPI resolves an endpoint's hints against module globals, where the request
# models defined inside create_app() do not exist -- the body silently becomes a
# query parameter and every POST answers 422.
import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from qec_pem import (BB_CATALOG, bb_code, hypergraph_product, repetition_H,
                     rotated_surface_code)

from qec_tile.bp import METHODS, bp_trace as run_bp, tanner_edges
from qec_tile.directional import build_directional_code
from qec_tile.gf2 import nullspace2, quotient_basis, rank2
from qec_tile.tile import TILES, paper_code

PAGE = Path(__file__).with_name("decode_viewer.html")

# How a qubit is drawn: an edge of the lattice, or a plain site.
EDGE_H, EDGE_V, SITE = "H", "V", "o"


@dataclass
class CodeView:
    """A CSS code plus a drawable layout — all the page needs about it.

    ``points[q]`` is ``(shape, x, y)``: where qubit ``q`` sits, and whether to
    draw it as a horizontal edge, a vertical edge or a dot.  Checks are not
    placed here; ``geometry`` puts each one at the centroid of its support,
    which lands a truncated boundary check on the qubits it actually touches.
    """
    label: str
    HX: np.ndarray
    HZ: np.ndarray
    points: list[tuple[str, float, float]]
    LX: np.ndarray
    LZ: np.ndarray
    # Where to draw the checks, when the construction knows better than the
    # centroid of the support does.  None means "work it out from the support".
    x_points: list[tuple[float, float]] | None = None
    z_points: list[tuple[float, float]] | None = None
    # (period_x, period_y) for a layout that wraps, so the page can draw the
    # neighbouring copies instead of long lines across the picture.
    period: tuple[float, float] | None = None

    @property
    def n(self) -> int:
        return int(self.HX.shape[1])

    @property
    def k(self) -> int:
        return self.n - rank2(self.HX) - rank2(self.HZ)


def _view(label: str, HX, HZ, points, x_points=None, z_points=None,
          period=None) -> CodeView:
    """Attach logicals to a code that only came with its two check matrices."""
    HX = np.asarray(HX, dtype=np.uint8)
    HZ = np.asarray(HZ, dtype=np.uint8)
    if len(points) != HX.shape[1]:
        raise ValueError(f"{label}: {len(points)} points for "
                         f"{HX.shape[1]} qubits")
    for name, given, count in (("x_points", x_points, HX.shape[0]),
                               ("z_points", z_points, HZ.shape[0])):
        if given is not None and len(given) != count:
            raise ValueError(f"{label}: {len(given)} {name} for {count} checks")
    return CodeView(label, HX, HZ, list(points),
                    quotient_basis(HX, nullspace2(HZ)),
                    quotient_basis(HZ, nullspace2(HX)),
                    x_points=x_points, z_points=z_points, period=period)


def _view_tile(label: str, code) -> CodeView:
    """Tile and directional codes: qubits are edges, so use edge midpoints.

    Checks go at their anchor -- the lower-left corner of their B x B box --
    not at the centroid of their support: a tile truncated by the boundary
    keeps only its inner qubits, so the centroid drifts into the bulk and the
    check ends up drawn on top of the lattice it actually hangs off.  Anchors
    are lattice vertices, and every qubit sits at a half-integer in one axis,
    so a check never lands on one.
    """
    points = [(EDGE_H, x + 0.5, float(y)) if orient == "H"
              else (EDGE_V, float(x), y + 0.5)
              for orient, x, y in code.qubits]
    return CodeView(label, code.HX, code.HZ, points, *code.logicals(),
                    x_points=[(float(x), float(y)) for x, y in code.x_anchors],
                    z_points=[(float(x), float(y)) for x, y in code.z_anchors])


def _view_rotated_surface(label: str, d: int) -> CodeView:
    """Qubit ``i*d + j`` sits at column j, row i of the d x d block."""
    HX, HZ = rotated_surface_code(d)
    points = [(SITE, float(j), float(d - 1 - i))
              for i in range(d) for j in range(d)]
    return _view(label, HX, HZ, points)


def _view_hgp(label: str, H1, H2, period=None) -> CodeView:
    """Hypergraph product layout: sector A on vertices, sector B on faces.

    ``hypergraph_product`` stacks ``[H1 (x) I_n2 | I_m1 (x) H2^T]``, so the
    first ``n1*n2`` columns are indexed ``(i, j)`` over the two code lengths
    and the rest ``(a, b)`` over the two check counts.  Offsetting the second
    sector by half a cell is what makes the planar picture readable.
    """
    HX, HZ = hypergraph_product(H1, H2)
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    points = [(SITE, float(j), float(i)) for i in range(n1) for j in range(n2)]
    points += [(SITE, b + 0.5, a + 0.5) for a in range(m1) for b in range(m2)]
    # Checks are indexed the same way and sit on the edges between the sites
    # they join.  The centroid of the support would not do for the cyclic case:
    # a check joining row 0 to row L-1 wraps, and its average lands in the
    # middle of the lattice, nowhere near either.
    x_points = [(float(j), a + 0.5) for a in range(m1) for j in range(n2)]
    z_points = [(b + 0.5, float(i)) for i in range(n1) for b in range(m2)]
    return _view(label, HX, HZ, points, x_points, z_points, period)


def _view_bb(label: str, l: int, m: int, A_terms, B_terms) -> CodeView:
    """Bivariate bicycle: two sectors on an l x m torus, index ``i*m + j``.

    There is no planar embedding — the lattice wraps in both directions, so a
    check's support can straddle opposite edges of the picture.  That is the
    code, not a drawing bug.
    """
    HX, HZ = bb_code(l, m, A_terms, B_terms)
    # Four sublattices per cell, as the paper draws them: the two qubit sectors
    # on opposite corners and the two check types on the other two.  Support
    # centroids are useless here -- every check wraps around the torus.
    left = [(SITE, float(j), float(i)) for i in range(l) for j in range(m)]
    right = [(SITE, j + 0.5, i + 0.5) for i in range(l) for j in range(m)]
    x_points = [(j + 0.5, float(i)) for i in range(l) for j in range(m)]
    z_points = [(float(j), i + 0.5) for i in range(l) for j in range(m)]
    return _view(label, HX, HZ, left + right, x_points, z_points,
                 period=(float(m), float(l)))


# Small enough that BP+OSD answers within a click and the layout still reads
# at screen size.  id -> (label, builder); the label is static so listing the
# catalog costs nothing -- building a code means solving for its logicals.
CATALOG: dict[str, tuple[str, callable]] = {
    **{f"tile:{name}:{L}": (f"tile {name}, L={L}",
                            lambda name=name, L=L:
                            _view_tile(f"tile {name}, L={L}",
                                       paper_code(name, L, L)))
       for name in sorted(TILES) for L in (3, 4, 5)},
    **{f"dir:{word}:{M}x{N}": (f"directional {word}, {M}x{N}",
                               lambda word=word, M=M, N=N:
                               _view_tile(f"directional {word}, {M}x{N}",
                                          build_directional_code(word, M, N)))
       for word, M, N in (("N2ESEN2", 4, 4), ("N2E2SE2N2", 5, 4),
                          ("N2E2SESE2N2", 5, 4))},
    **{f"rotated:{d}": (f"rotated surface, d={d}",
                        lambda d=d:
                        _view_rotated_surface(f"rotated surface, d={d}", d))
       for d in (3, 5, 7)},
    **{f"unrotated:{d}": (f"unrotated surface, d={d}",
                          lambda d=d:
                          _view_hgp(f"unrotated surface, d={d}",
                                    repetition_H(d), repetition_H(d)))
       for d in (3, 4, 5)},
    **{f"toric:{L}": (f"toric, L={L}",
                      lambda L=L:
                      _view_hgp(f"toric, L={L}",
                                repetition_H(L, cyclic=True),
                                repetition_H(L, cyclic=True),
                                period=(float(L), float(L))))
       for L in (3, 4, 5)},
    # The larger BB codes decode fine but the torus picture stops being useful.
    **{f"bb:{name}": (f"bivariate bicycle {name}",
                      lambda name=name:
                      _view_bb(f"bivariate bicycle {name}",
                               BB_CATALOG[name]["l"], BB_CATALOG[name]["m"],
                               BB_CATALOG[name]["A"], BB_CATALOG[name]["B"]))
       for name in ("[[72,12,6]]", "[[144,12,12]]")},
}


@lru_cache(maxsize=None)
def get_code(code_id: str) -> CodeView:
    if code_id not in CATALOG:
        raise KeyError(f"unknown code {code_id!r}")
    return CATALOG[code_id][1]()


@lru_cache(maxsize=None)
def _check_rank(code_id: str) -> int:
    """rank(HZ) — the yardstick for whether a hand-built syndrome is real."""
    return rank2(get_code(code_id).HZ)


def catalog() -> list[dict]:
    """What the code picker offers; n and k arrive with the geometry."""
    return [dict(id=code_id, label=label)
            for code_id, (label, _build) in CATALOG.items()]


def geometry(code_id: str) -> dict:
    """Everything the page needs to draw one code, once."""
    view = get_code(code_id)

    def centroids(checks, given):
        if given is not None:
            return [[float(x), float(y)] for x, y in given]
        out = []
        for row in checks:
            support = np.flatnonzero(row)
            xs = [view.points[col][1] for col in support]
            ys = [view.points[col][2] for col in support]
            out.append([float(np.mean(xs)), float(np.mean(ys))])
        return out

    return dict(
        id=code_id, label=view.label, n=view.n, k=view.k,
        period=list(view.period) if view.period else None,
        qubits=[dict(shape=shape, x=x, y=y) for shape, x, y in view.points],
        x_checks=[np.flatnonzero(row).tolist() for row in view.HX],
        z_checks=[np.flatnonzero(row).tolist() for row in view.HZ],
        x_centres=centroids(view.HX, view.x_points),
        z_centres=centroids(view.HZ, view.z_points),
        # The Tanner edges of HZ, in the order every message array uses.
        edges=[[int(check), int(qubit)]
               for check, qubit in zip(*tanner_edges(view.HZ))],
        logicals=[np.flatnonzero(row).tolist() for row in view.LZ],
    )


def _bit_vector(indices: list[int], length: int, what: str) -> np.ndarray:
    """Clicked indices -> a 0/1 vector; clicking the same one twice cancels."""
    vector = np.zeros(length, dtype=np.uint8)
    for index in indices:
        if not 0 <= index < length:
            raise ValueError(f"{what} {index} outside [0, {length})")
        vector[index] ^= 1
    return vector


def syndrome_of(code_id: str, error: list[int]) -> list[int]:
    """The checks an X-error pattern lights, for callers that start from one."""
    view = get_code(code_id)
    e = _bit_vector(error, view.n, "qubit")
    return np.flatnonzero((view.HZ @ e) % 2).tolist()


def bp_trace(code_id: str, syndrome: list[int], p: float, max_iter: int = 50,
             ms_scaling_factor: float = 1.0, error: list[int] | None = None,
             method: str = "minimum_sum") -> dict:
    """Run BP on a syndrome and report its belief after every iteration.

    The syndrome is the input because it is all a decoder ever sees.  ``error``
    is optional and only says what really happened: with it the residual and
    the logical verdict are meaningful, without it there is nothing to compare
    a correction against and those fields stay None.

    There is no OSD here.  When BP stalls the answer is not a worse correction,
    it is *no* correction: ``H @ correction != syndrome``, so the outcome is
    "stalled" rather than a logical failure.
    """
    view = get_code(code_id)
    s = _bit_vector(syndrome, view.HZ.shape[0], "check")

    # A syndrome invented by clicking checks need not be in the image of HZ.
    # Toric and BB codes have dependent checks, so unreachable combinations
    # exist and no decoder can explain them -- say so instead of blaming BP.
    # Reachable means HZ x = s has a solution, so s joins as a *column*:
    # appending it must not raise the rank.
    reachable = bool(rank2(np.hstack([view.HZ, s.reshape(-1, 1)]))
                     == _check_rank(code_id))

    trace = [dict(iteration=step.iteration,
                  converge=step.converged,
                  hard=np.flatnonzero(step.hard).tolist(),
                  # 3 decimals: the full arrays are 300 KB on the larger codes
                  llr=[round(float(value), 3) for value in step.llr])
             for step in run_bp(view.HZ, s, p, method=method,
                                max_iter=max_iter,
                                ms_scaling_factor=ms_scaling_factor)]

    correction = np.zeros(view.n, dtype=np.uint8)
    if trace:
        correction[trace[-1]["hard"]] = 1
    valid = bool(trace and trace[-1]["converge"])

    residual = flipped = None
    outcome = "corrected" if valid else ("stalled" if reachable
                                         else "unreachable")
    if error is not None:
        e = _bit_vector(error, view.n, "qubit")
        if not np.array_equal((view.HZ @ e) % 2, s):
            raise ValueError("error does not produce this syndrome")
        if valid:
            residual_vector = (e ^ correction) % 2
            residual = np.flatnonzero(residual_vector).tolist()
            flipped = [int(i) for i in
                       (np.flatnonzero((view.LZ @ residual_vector) % 2)
                        if view.LZ.size else [])]
            if flipped:
                outcome = "logical_error"

    return dict(
        max_iter=int(max_iter),
        method=method,
        converged_at=trace[-1]["iteration"] if valid else None,
        reachable=reachable,
        syndrome=np.flatnonzero(s).tolist(),
        correction=np.flatnonzero(correction).tolist(),
        error=None if error is None else np.flatnonzero(
            _bit_vector(error, view.n, "qubit")).tolist(),
        residual=residual,
        logicals_flipped=flipped,
        outcome=outcome,
        trace=trace,
    )


def bp_messages(code_id: str, syndrome: list[int], p: float, iteration: int,
                count: int = 1, max_iter: int = 50,
                ms_scaling_factor: float = 1.0,
                method: str = "minimum_sum") -> dict:
    """The two halves of each sweep: what v-nodes sent, what c-nodes answered.

    A window of ``count`` sweeps starting at ``iteration``, because the full
    trace would carry two numbers per Tanner edge per sweep -- 86k of them on
    the larger bicycle code -- and the unrolled picture only ever shows a few.
    Sweeps past the end of the run are simply absent from the result.
    """
    view = get_code(code_id)
    s = _bit_vector(syndrome, view.HZ.shape[0], "check")
    if iteration < 1:
        raise ValueError(f"iteration {iteration} is before the first sweep")
    if count < 1:
        raise ValueError(f"count {count} asks for no sweeps at all")

    wanted = range(iteration, iteration + count)
    steps = [dict(iteration=step.iteration,
                  to_check=[round(float(value), 3) for value in step.to_check],
                  to_bit=[round(float(value), 3) for value in step.to_bit])
             for step in run_bp(view.HZ, s, p, method=method,
                                max_iter=min(max(wanted), max_iter),
                                ms_scaling_factor=ms_scaling_factor)
             if step.iteration in wanted]
    return dict(steps=steps)


def random_error(code_id: str, p: float, seed: int | None = None) -> list[int]:
    """One Bernoulli(p) shot, the same draw sample_residuals makes."""
    rng = np.random.default_rng(seed)
    return np.flatnonzero(rng.random(get_code(code_id).n) < p).tolist()


def create_app():
    """Built here, not at import, so the module loads without fastapi."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    class DecodeRequest(BaseModel):
        id: str
        syndrome: list[int] = []
        error: list[int] | None = None
        p: float = 0.05
        max_iter: int = 50
        ms_scaling_factor: float = 1.0
        method: str = "minimum_sum"

    class MessageRequest(DecodeRequest):
        iteration: int = 1
        count: int = 1

    class SampleRequest(BaseModel):
        id: str
        p: float = 0.05
        seed: int | None = None

    app = FastAPI(title="CSS code decoding viewer")

    @app.get("/")
    def page():
        # The page is edited while the server runs; a cached copy against a
        # newer API is the confusing kind of broken.
        return FileResponse(PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/api/codes")
    def api_codes():
        return dict(codes=catalog(), methods=list(METHODS))

    @app.get("/api/geometry/{code_id}")
    def api_geometry(code_id: str):
        try:
            return geometry(code_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/bp_trace")
    def api_bp_trace(request: DecodeRequest):
        try:
            return bp_trace(request.id, request.syndrome, request.p,
                            request.max_iter, request.ms_scaling_factor,
                            request.error, request.method)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/bp_messages")
    def api_bp_messages(request: MessageRequest):
        try:
            return bp_messages(request.id, request.syndrome, request.p,
                               request.iteration, request.count,
                               request.max_iter, request.ms_scaling_factor,
                               request.method)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/sample")
    def api_sample(request: SampleRequest):
        try:
            error = random_error(request.id, request.p, request.seed)
            return dict(error=error, syndrome=syndrome_of(request.id, error))
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    return app


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="keep it on loopback and forward the port over ssh")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
