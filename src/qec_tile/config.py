"""``.env``에서 읽는 자원 할당 설정 (``.env.example`` 참고).

공유 호스트의 할당량은 행정적이라 cgroup에 없다 — ``os.cpu_count()``로는
보이지 않으니 직접 적어둔다. 기본값은 일부러 없다: 없는 값은 예외로 터진다.

thread cap이 필요하면 numpy보다 먼저 import한다. BLAS는 ``*_THREADS``를
로드될 때 한 번만 읽는다.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ENV_FILE: str | None = find_dotenv(usecwd=True) or None
if ENV_FILE is None:                       # 저장소 루트로 대체
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
    """이 프로젝트가 띄워도 되는 sinter worker 프로세스 수."""
    return _require("QEC_TILE_WORKERS")


def gpu() -> int:
    """이 프로젝트가 써도 되는 물리 GPU 인덱스."""
    return _require("QEC_TILE_GPU")


def threads() -> int:
    """프로세스당 BLAS/OpenMP thread 수. 예산은 workers() x threads()."""
    return _require("QEC_TILE_THREADS")


_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def apply_thread_limits() -> None:
    """아직 설정되지 않은 BLAS/OpenMP 변수에 threads()를 심는다.

    QEC_TILE_THREADS가 없으면 아무 일도 하지 않는다: cap은 공유 호스트를
    보호하는 장치일 뿐, import를 막지는 않는다.
    """
    raw = os.environ.get("QEC_TILE_THREADS")
    if raw is None or not raw.strip():
        return
    for var in _THREAD_VARS:
        os.environ.setdefault(var, str(threads()))


apply_thread_limits()
