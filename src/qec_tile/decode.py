"""Code-capacity decoding of tile codes with BP+OSD.

The noise model is the simplest one: each qubit takes an independent X error
with probability ``p`` and the Z-checks are read out perfectly.  One shot is

    e ~ Bernoulli(p)^n            an X-error pattern
    s = HZ @ e   (mod 2)          its syndrome
    e_hat = decode(s)             BP, falling back to OSD when it stalls
    r = e ^ e_hat                 the residual (HZ @ r = 0 by construction)
    failure  <=>  LZ @ r != 0     r is a logical, not just a stabilizer

Only the X sector is simulated; the Z sector is identical with HX/LX.  The
failure test is exactly the one pinned down in tests/test_logicals.py.
"""
from __future__ import annotations

import numpy as np
from ldpc import BpOsdDecoder


def make_decoder(H: np.ndarray, p: float, *, max_iter: int = 50,
                 osd_order: int = 7) -> BpOsdDecoder:
    """A BP+OSD decoder for parity checks ``H`` at physical error rate ``p``.

    ``p`` is the BP prior as well as the sampling rate: BP needs the true rate
    for its posteriors, and OSD ranks bits by exactly those posteriors.
    """
    return BpOsdDecoder(
        H.astype(np.uint8),
        error_rate=p,
        bp_method="minimum_sum",   # ldpc's default; scaling left at its default
        max_iter=max_iter,
        osd_method="osd_cs",       # combination sweep
        osd_order=osd_order,
    )


def sample_residuals(code, p: float, shots: int, seed: int | None = None):
    """Yield the residual ``e ^ e_hat`` for each decoded shot (X sector)."""
    rng = np.random.default_rng(seed)
    decoder = make_decoder(code.HZ, p)
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        e_hat = decoder.decode(syndrome)
        yield (e ^ e_hat) % 2


def logical_error_rate(code, p: float, shots: int,
                       seed: int | None = None) -> float:
    """Fraction of shots that end in a logical X failure at rate ``p``."""
    _, LZ = code.logicals()
    failures = sum(bool(((LZ @ residual) % 2).any())
                   for residual in sample_residuals(code, p, shots, seed))
    return failures / shots
