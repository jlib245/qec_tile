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


def make_decoder(H: np.ndarray, channel, *, max_iter: int = 50,
                 osd_order: int = 7) -> BpOsdDecoder:
    """A BP+OSD decoder for parity checks ``H`` with prior ``channel``.

    ``channel`` is the BP prior as well as the sampling rate: BP needs the
    true rates for its posteriors, and OSD ranks bits by exactly those
    posteriors.  A scalar means one uniform rate; a vector gives a per-column
    prior (space-time matrices mix data and measurement rates).
    """
    channel = np.asarray(channel, dtype=float)
    prior = (dict(error_rate=float(channel)) if channel.ndim == 0
             else dict(error_channel=list(channel)))
    return BpOsdDecoder(
        H.astype(np.uint8),
        **prior,
        bp_method="minimum_sum",   # ldpc's default; scaling left at its default
        max_iter=max_iter,
        osd_method="osd_cs",       # combination sweep
        osd_order=osd_order,
    )


# Decoder registry.  A future NN decoder registers its own builder here; there
# is deliberately no default, so every run records which decoder produced it.
DECODERS = {"bposd": make_decoder}


def sample_residuals(code, p: float, shots: int, decoder: str,
                     seed: int | None = None):
    """Yield the residual ``e ^ e_hat`` for each decoded shot (X sector)."""
    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    rng = np.random.default_rng(seed)
    decoder_obj = build(code.HZ, p)
    for _ in range(shots):
        e = (rng.random(code.n) < p).astype(np.uint8)
        syndrome = ((code.HZ @ e) % 2).astype(np.uint8)
        e_hat = decoder_obj.decode(syndrome)
        yield (e ^ e_hat) % 2


def failure_rate(H: np.ndarray, L: np.ndarray, channel, shots: int,
                 decoder: str, seed: int | None = None) -> float:
    """Fraction of decoded shots whose residual flips an observable.

    Fully generic: sample ``x`` with per-column probabilities ``channel``,
    decode the syndrome ``H @ x``, and call the shot a failure iff the
    residual ``x ^ x_hat`` has ``L @ residual != 0``.  Code capacity and the
    space-time (phenomenological) matrices both fit this shape.
    """
    build = DECODERS.get(decoder)
    if build is None:
        raise ValueError(
            f"unknown decoder {decoder!r}; have {sorted(DECODERS)}")
    rng = np.random.default_rng(seed)
    decoder_obj = build(H, channel)
    channel = np.asarray(channel, dtype=float)
    failures = 0
    for _ in range(shots):
        x = (rng.random(H.shape[1]) < channel).astype(np.uint8)
        x_hat = decoder_obj.decode(((H @ x) % 2).astype(np.uint8))
        failures += bool(((L @ ((x ^ x_hat) % 2)) % 2).any())
    return failures / shots


def logical_error_rate(code, p: float, shots: int, decoder: str,
                       seed: int | None = None) -> float:
    """Fraction of shots that end in a logical X failure at rate ``p``."""
    _, LZ = code.logicals()
    return failure_rate(code.HZ, LZ, np.full(code.n, p), shots, decoder, seed)
