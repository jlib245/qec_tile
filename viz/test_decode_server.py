"""Decoding viewer backend — one shot's stages, geometry and sampling.

The HTTP layer is a thin wrapper (create_app) and needs fastapi installed, so
these exercise the functions behind it instead.
"""
import numpy as np
import pytest

import decode_server
from decode_server import decode_shot, geometry, get_code, random_error

CODE_ID = "tile:b3w6:3"


@pytest.fixture
def code():
    return get_code(CODE_ID)[0]


def test_no_error_leaves_everything_empty():
    shot = decode_shot(CODE_ID, [], p=0.05)
    assert shot["syndrome"] == shot["correction"] == shot["residual"] == []
    assert shot["failure"] is False


def test_single_error_lights_exactly_its_z_checks(code):
    """The syndrome is the column of HZ, which is what the page draws lit."""
    qubit = 17
    shot = decode_shot(CODE_ID, [qubit], p=0.05)
    assert shot["syndrome"] == np.flatnonzero(code.HZ[:, qubit]).tolist()


def test_single_error_is_corrected(code):
    """d >= 3 for this layout, so weight 1 must come back clean."""
    for qubit in range(code.n):
        shot = decode_shot(CODE_ID, [qubit], p=0.05)
        assert shot["failure"] is False, qubit
        assert shot["residual"] == [], qubit


def test_stabilizer_error_is_silent_and_harmless(code):
    """An X-stabilizer commutes with every Z-check: no syndrome, no failure.

    The residual is then the stabilizer itself — the case the page reports as
    "corrected, residual is a stabilizer".
    """
    stabilizer = np.flatnonzero(code.HX[0]).tolist()
    shot = decode_shot(CODE_ID, stabilizer, p=0.05)
    assert shot["syndrome"] == []
    assert shot["correction"] == []
    assert shot["residual"] == stabilizer
    assert shot["failure"] is False


def test_logical_error_is_a_failure(code):
    """An X-logical is also syndrome-free, but flips an L_Z — the failure case."""
    LX, _LZ = code.logicals()
    logical = np.flatnonzero(LX[0]).tolist()
    shot = decode_shot(CODE_ID, logical, p=0.05)
    assert shot["syndrome"] == []
    assert shot["failure"] is True
    assert shot["logicals_flipped"]


def test_repeated_qubit_cancels():
    """X errors are mod 2: clicking the same qubit twice is no error at all."""
    assert decode_shot(CODE_ID, [5, 5], p=0.05)["error"] == []


def test_qubit_outside_the_code_is_rejected(code):
    with pytest.raises(ValueError):
        decode_shot(CODE_ID, [code.n], p=0.05)


def test_unknown_code_and_decoder_are_rejected():
    with pytest.raises(KeyError):
        decode_shot("tile:nope:3", [], p=0.05)
    with pytest.raises(KeyError):
        decode_shot(CODE_ID, [], p=0.05, decoder="nope")


def test_geometry_places_every_qubit_on_its_edge(code):
    """H(x,y) spans (x,y)-(x+1,y), so its midpoint is (x+0.5, y); V is the flip."""
    geo = geometry(CODE_ID)
    assert len(geo["qubits"]) == code.n
    for qubit, (orient, x, y) in zip(geo["qubits"], code.qubits):
        expected = (x + 0.5, y) if orient == "H" else (x, y + 0.5)
        assert (qubit["mx"], qubit["my"]) == expected


def test_geometry_check_rows_match_the_matrices(code):
    geo = geometry(CODE_ID)
    assert geo["z_checks"] == [np.flatnonzero(row).tolist() for row in code.HZ]
    assert geo["x_checks"] == [np.flatnonzero(row).tolist() for row in code.HX]
    assert len(geo["z_centres"]) == code.HZ.shape[0]


def test_random_error_is_reproducible_and_respects_p(code):
    assert random_error(CODE_ID, 0.1, seed=3) == random_error(CODE_ID, 0.1, seed=3)
    assert random_error(CODE_ID, 0.0, seed=0) == []
    assert random_error(CODE_ID, 1.0, seed=0) == list(range(code.n))


def test_catalog_ids_all_build():
    for entry in decode_server.catalog():
        assert entry["n"] > 0 and entry["k"] > 0
