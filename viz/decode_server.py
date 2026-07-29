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

from qec_tile.decode import DECODERS
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

    @property
    def n(self) -> int:
        return int(self.HX.shape[1])

    @property
    def k(self) -> int:
        return self.n - rank2(self.HX) - rank2(self.HZ)


def _view(label: str, HX, HZ, points) -> CodeView:
    """Attach logicals to a code that only came with its two check matrices."""
    HX = np.asarray(HX, dtype=np.uint8)
    HZ = np.asarray(HZ, dtype=np.uint8)
    if len(points) != HX.shape[1]:
        raise ValueError(f"{label}: {len(points)} points for "
                         f"{HX.shape[1]} qubits")
    return CodeView(label, HX, HZ, list(points),
                    quotient_basis(HX, nullspace2(HZ)),
                    quotient_basis(HZ, nullspace2(HX)))


def _view_tile(label: str, code) -> CodeView:
    """Tile and directional codes: qubits are edges, so use edge midpoints."""
    points = [(EDGE_H, x + 0.5, float(y)) if orient == "H"
              else (EDGE_V, float(x), y + 0.5)
              for orient, x, y in code.qubits]
    return CodeView(label, code.HX, code.HZ, points, *code.logicals())


def _view_rotated_surface(label: str, d: int) -> CodeView:
    """Qubit ``i*d + j`` sits at column j, row i of the d x d block."""
    HX, HZ = rotated_surface_code(d)
    points = [(SITE, float(j), float(d - 1 - i))
              for i in range(d) for j in range(d)]
    return _view(label, HX, HZ, points)


def _view_hgp(label: str, H1, H2) -> CodeView:
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
    return _view(label, HX, HZ, points)


def _view_bb(label: str, l: int, m: int, A_terms, B_terms) -> CodeView:
    """Bivariate bicycle: two sectors on an l x m torus, index ``i*m + j``.

    There is no planar embedding — the lattice wraps in both directions, so a
    check's support can straddle opposite edges of the picture.  That is the
    code, not a drawing bug.
    """
    HX, HZ = bb_code(l, m, A_terms, B_terms)
    left = [(SITE, float(j), float(i)) for i in range(l) for j in range(m)]
    right = [(SITE, j + 0.5, i + 0.5) for i in range(l) for j in range(m)]
    return _view(label, HX, HZ, left + right)


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
                                repetition_H(L, cyclic=True)))
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
def get_decoder(code_id: str, p: float, decoder: str):
    """Decoders are expensive to build and cheap to reuse; keep them warm."""
    if decoder not in DECODERS:
        raise KeyError(f"unknown decoder {decoder!r}")
    return DECODERS[decoder](get_code(code_id).HZ, p)


def catalog() -> list[dict]:
    """What the code picker offers; n and k arrive with the geometry."""
    return [dict(id=code_id, label=label)
            for code_id, (label, _build) in CATALOG.items()]


def geometry(code_id: str) -> dict:
    """Everything the page needs to draw one code, once."""
    view = get_code(code_id)

    def centroids(checks):
        out = []
        for row in checks:
            support = np.flatnonzero(row)
            xs = [view.points[col][1] for col in support]
            ys = [view.points[col][2] for col in support]
            out.append([float(np.mean(xs)), float(np.mean(ys))])
        return out

    return dict(
        id=code_id, label=view.label, n=view.n, k=view.k,
        qubits=[dict(shape=shape, x=x, y=y) for shape, x, y in view.points],
        x_checks=[np.flatnonzero(row).tolist() for row in view.HX],
        z_checks=[np.flatnonzero(row).tolist() for row in view.HZ],
        x_centres=centroids(view.HX),
        z_centres=centroids(view.HZ),
        logicals=[np.flatnonzero(row).tolist() for row in view.LZ],
    )


def decode_shot(code_id: str, error: list[int], p: float,
                decoder: str = "bposd_cs7") -> dict:
    """Decode one X-error pattern and report every stage of the shot."""
    view = get_code(code_id)
    e = np.zeros(view.n, dtype=np.uint8)
    for qubit in error:
        if not 0 <= qubit < view.n:
            raise ValueError(f"qubit {qubit} outside [0, {view.n})")
        e[qubit] ^= 1

    syndrome = ((view.HZ @ e) % 2).astype(np.uint8)
    e_hat = get_decoder(code_id, p, decoder).decode(syndrome).astype(np.uint8)
    residual = (e ^ e_hat) % 2
    flipped = (np.flatnonzero((view.LZ @ residual) % 2) if view.LZ.size
               else np.array([], dtype=int))

    return dict(
        error=np.flatnonzero(e).tolist(),
        syndrome=np.flatnonzero(syndrome).tolist(),
        correction=np.flatnonzero(e_hat).tolist(),
        residual=np.flatnonzero(residual).tolist(),
        logicals_flipped=[int(i) for i in flipped],
        failure=bool(flipped.size),
    )


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
        error: list[int] = []
        p: float = 0.05
        decoder: str = "bposd_cs7"

    class SampleRequest(BaseModel):
        id: str
        p: float = 0.05
        seed: int | None = None

    app = FastAPI(title="CSS code decoding viewer")

    @app.get("/")
    def page():
        return FileResponse(PAGE)

    @app.get("/api/codes")
    def api_codes():
        return dict(codes=catalog(), decoders=sorted(DECODERS))

    @app.get("/api/geometry/{code_id}")
    def api_geometry(code_id: str):
        try:
            return geometry(code_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc))

    @app.post("/api/decode")
    def api_decode(request: DecodeRequest):
        try:
            return decode_shot(request.id, request.error, request.p,
                               request.decoder)
        except KeyError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @app.post("/api/sample")
    def api_sample(request: SampleRequest):
        try:
            return dict(error=random_error(request.id, request.p, request.seed))
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
