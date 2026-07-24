"""Code-capacity decoding with BP+OSD."""
import numpy as np
import pytest

from qec_tile.decode import (DECODERS, logical_error_rate, make_decoder,
                             sample_residuals)
from qec_tile.tile import paper_code

SMALL = ("b3w6", 5, 5)


def test_zero_noise_never_fails():
    rate = logical_error_rate(paper_code(*SMALL), p=0.0, shots=200,
                              decoder="bposd_cs7", seed=0)
    assert rate == 0.0


def test_correction_always_matches_syndrome():
    """Whatever BP+OSD returns, the residual must carry a zero Z-syndrome."""
    code = paper_code(*SMALL)
    for residual in sample_residuals(code, p=0.08, shots=100,
                                     decoder="bposd_cs7", seed=1):
        assert not ((code.HZ @ residual) % 2).any()


def test_single_errors_are_always_corrected():
    """d >= 3, so any weight-1 error is uniquely decodable."""
    code = paper_code(*SMALL)
    decoder = make_decoder(code.HZ, 0.05)
    _, LZ = code.logicals()
    for i in range(code.n):
        e = np.zeros(code.n, dtype=np.uint8)
        e[i] = 1
        ehat = decoder.decode(((code.HZ @ e) % 2).astype(np.uint8))
        residual = (e ^ ehat) % 2
        assert not ((LZ @ residual) % 2).any()


def test_rate_grows_with_p():
    code = paper_code(*SMALL)
    low = logical_error_rate(code, p=0.02, shots=800, decoder="bposd_cs7", seed=2)
    high = logical_error_rate(code, p=0.12, shots=800, decoder="bposd_cs7", seed=2)
    assert low < high


def test_seed_is_deterministic():
    code = paper_code(*SMALL)
    a = logical_error_rate(code, p=0.08, shots=300, decoder="bposd_cs7", seed=7)
    b = logical_error_rate(code, p=0.08, shots=300, decoder="bposd_cs7", seed=7)
    assert a == b


def test_rate_is_a_fraction():
    rate = logical_error_rate(paper_code(*SMALL), p=0.1, shots=500,
                              decoder="bposd_cs7", seed=3)
    assert 0.0 <= rate <= 1.0


def test_decoder_is_required():
    """No default: the run must say which decoder produced it."""
    with pytest.raises(TypeError):
        logical_error_rate(paper_code(*SMALL), p=0.05, shots=10)


def test_all_decoders_are_registered():
    assert set(DECODERS) == {"bposd_cs7", "bposd_0", "bplsd_0", "bplsd_cs7"}


def test_unknown_decoder_is_rejected():
    with pytest.raises(ValueError, match="unknown decoder"):
        logical_error_rate(paper_code(*SMALL), p=0.05, shots=10, decoder="nn")


def test_decoder_prior_matches_the_sampling_rate():
    """BP needs the true rate as its prior; make_decoder ties them together."""
    code = paper_code(*SMALL)
    decoder = make_decoder(code.HZ, 0.07)
    assert np.allclose(decoder.error_channel, 0.07)


def test_make_decoder_accepts_a_channel_vector():
    """Per-column priors, for space-time matrices with mixed p and q."""
    code = paper_code(*SMALL)
    channel = np.full(code.n, 0.03)
    channel[0] = 0.11
    decoder = make_decoder(code.HZ, channel)
    assert np.allclose(decoder.error_channel, channel)


def test_a_wrong_prior_decodes_worse():
    """Feeding BP the wrong rate hurts — justifies binding prior to p."""
    code = paper_code(*SMALL)
    _, LZ = code.logicals()
    p = 0.1

    def failures(prior):
        rng = np.random.default_rng(0)
        decoder = make_decoder(code.HZ, prior)
        count = 0
        for _ in range(1500):
            e = (rng.random(code.n) < p).astype(np.uint8)
            e_hat = decoder.decode(((code.HZ @ e) % 2).astype(np.uint8))
            count += bool(((LZ @ ((e ^ e_hat) % 2)) % 2).any())
        return count

    # 0.5 is a deliberately wrong prior here, not a library default.
    assert failures(p) < failures(0.5)     # matched prior beats a way-off one
