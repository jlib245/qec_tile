"""Serve an interactive error -> syndrome -> decoding view of a tile code.

The browser draws the lattice and collects clicks; every number it shows comes
from this process, decoded by the same BP+OSD the benchmarks use, so what the
page reports is what qec_tile would report.

The X sector is the one on screen, following decode.py:

    e                 an X-error pattern (the qubits you click)
    s = HZ @ e        the lit checks
    e_hat = decode(s) the decoder's guess
    r = e ^ e_hat     the residual
    failure  <=>  LZ @ r != 0

Run it (fastapi is not a project dependency -- keep it out of pyproject):

    uv run --with fastapi --with uvicorn python data/decode_server.py
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from qec_tile.decode import DECODERS
from qec_tile.directional import build_directional_code
from qec_tile.tile import TILES, paper_code

PAGE = Path(__file__).with_name("decode_viewer.html")

# Small enough that BP+OSD answers within a click, and that the lattice still
# reads at screen size.  id -> (label, builder).
CATALOG: dict[str, tuple[str, callable]] = {
    **{f"tile:{name}:{L}": (f"tile {name}, L={L}",
                            (lambda name=name, L=L: paper_code(name, L, L)))
       for name in sorted(TILES) for L in (3, 4, 5)},
    **{f"dir:{word}:{M}x{N}": (f"directional {word}, {M}x{N}",
                               (lambda word=word, M=M, N=N:
                                build_directional_code(word, M, N)))
       for word, M, N in (("N2ESEN2", 4, 4), ("N2E2SE2N2", 5, 4),
                          ("N2E2SESE2N2", 5, 4))},
}


@lru_cache(maxsize=None)
def get_code(code_id: str):
    """The TileCode plus its Z-logicals, built once per id."""
    if code_id not in CATALOG:
        raise KeyError(f"unknown code {code_id!r}")
    code = CATALOG[code_id][1]()
    _LX, LZ = code.logicals()
    return code, LZ


@lru_cache(maxsize=None)
def get_decoder(code_id: str, p: float, decoder: str):
    """Decoders are expensive to build and cheap to reuse; keep them warm."""
    if decoder not in DECODERS:
        raise KeyError(f"unknown decoder {decoder!r}")
    code, _LZ = get_code(code_id)
    return DECODERS[decoder](code.HZ, p)


def catalog() -> list[dict]:
    """What the code picker offers."""
    out = []
    for code_id, (label, _build) in CATALOG.items():
        code, _LZ = get_code(code_id)
        out.append(dict(id=code_id, label=label, n=code.n, k=code.k,
                        B=code.B, L1=code.L1, L2=code.L2))
    return out


def geometry(code_id: str) -> dict:
    """Everything the page needs to draw the lattice once.

    Qubits are edges, so each gets the midpoint of the edge it sits on; checks
    get the centroid of their support, which keeps a truncated boundary check
    on top of the qubits it actually touches.
    """
    code, LZ = get_code(code_id)
    midpoints = [(x + 0.5, float(y)) if orient == "H" else (float(x), y + 0.5)
                 for orient, x, y in code.qubits]

    def centroids(checks):
        out = []
        for row in checks:
            support = np.flatnonzero(row)
            points = np.array([midpoints[col] for col in support])
            out.append([float(points[:, 0].mean()), float(points[:, 1].mean())])
        return out

    return dict(
        id=code_id, n=code.n, k=code.k, B=code.B, L1=code.L1, L2=code.L2,
        qubits=[dict(orient=orient, x=int(x), y=int(y),
                     mx=float(midpoint[0]), my=float(midpoint[1]))
                for (orient, x, y), midpoint in zip(code.qubits, midpoints)],
        x_checks=[np.flatnonzero(row).tolist() for row in code.HX],
        z_checks=[np.flatnonzero(row).tolist() for row in code.HZ],
        x_centres=centroids(code.HX),
        z_centres=centroids(code.HZ),
        logicals=[np.flatnonzero(row).tolist() for row in LZ],
    )


def decode_shot(code_id: str, error: list[int], p: float,
                decoder: str = "bposd_cs7") -> dict:
    """Decode one X-error pattern and report every stage of the shot."""
    code, LZ = get_code(code_id)
    e = np.zeros(code.n, dtype=np.uint8)
    for qubit in error:
        if not 0 <= qubit < code.n:
            raise ValueError(f"qubit {qubit} outside [0, {code.n})")
        e[qubit] ^= 1

    syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
    e_hat = get_decoder(code_id, p, decoder).decode(syndrome).astype(np.uint8)
    residual = (e ^ e_hat) % 2
    flipped = np.flatnonzero((LZ @ residual) % 2) if LZ.size else np.array([])

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
    code, _LZ = get_code(code_id)
    rng = np.random.default_rng(seed)
    return np.flatnonzero(rng.random(code.n) < p).tolist()


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

    app = FastAPI(title="tile code decoding viewer")

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
