"""Allocation settings from ``.env`` (see ``.env.example``).

This is a shared host and the CPU/GPU quota is administrative — there is no
cgroup limit, so ``os.cpu_count()`` cannot see it.  The allocation therefore
has to be written down, and deliberately has **no defaults**: nothing here
invents a worker count, values are read lazily and raise when missing.  Real
environment variables take priority over ``.env``.

Import this before numpy when thread caps matter: BLAS reads the ``*_THREADS``
variables only once, when the library loads.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ENV_FILE: str | None = find_dotenv(usecwd=True) or None
if ENV_FILE is None:                       # fall back to the repo root
    candidate = Path(__file__).resolve().parents[2] / ".env"
    ENV_FILE = str(candidate) if candidate.is_file() else None
if ENV_FILE:
    load_dotenv(ENV_FILE, override=False)


def _require(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        raise RuntimeError(
            f"{name} is not set — copy .env.example to .env and fill in "
            f"your allocation")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def workers() -> int:
    """Sinter worker processes this project may run."""
    return _require("QEC_TILE_WORKERS")


def gpu() -> int:
    """Physical GPU index this project may use."""
    return _require("QEC_TILE_GPU")


def threads() -> int:
    """BLAS/OpenMP threads per process; the budget is workers() x threads()."""
    return _require("QEC_TILE_THREADS")


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def apply_thread_limits() -> None:
    """Publish threads() to the BLAS/OpenMP vars that are still unset.

    A no-op when QEC_TILE_THREADS is not configured: the cap protects the
    shared host once .env exists, but never blocks importing the package.
    """
    raw = os.environ.get("QEC_TILE_THREADS")
    if raw is None or not raw.strip():
        return
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(threads()))


apply_thread_limits()
