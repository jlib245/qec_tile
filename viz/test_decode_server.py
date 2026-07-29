"""Decoding viewer backend — one shot's stages, layouts and sampling.

The HTTP layer is a thin wrapper (create_app) and needs fastapi installed, so
these exercise the functions behind it instead.  Codes are kept small on
purpose: solving for logicals is the expensive part of building one.
"""
import numpy as np
import pytest

from decode_server import (CATALOG, catalog, decode_shot, geometry, get_code,
                           random_error)

from qec_tile.gf2 import rank2

CODE_ID = "tile:b3w6:3"


@pytest.fixture
def view():
    return get_code(CODE_ID)


def test_no_error_leaves_everything_empty():
    shot = decode_shot(CODE_ID, [], p=0.05)
    assert shot["syndrome"] == shot["correction"] == shot["residual"] == []
    assert shot["failure"] is False


def test_single_error_lights_exactly_its_z_checks(view):
    """The syndrome is the column of HZ, which is what the page draws lit."""
    qubit = 17
    shot = decode_shot(CODE_ID, [qubit], p=0.05)
    assert shot["syndrome"] == np.flatnonzero(view.HZ[:, qubit]).tolist()


@pytest.mark.parametrize("code_id", [CODE_ID, "rotated:3", "toric:3",
                                     "bb:[[72,12,6]]"])
def test_single_error_never_fails(code_id):
    """d >= 3 everywhere here, so weight 1 is correctable — but degenerately.

    The decoder may answer with anything differing from e by a stabilizer, and
    on the tile layout it does for 5 of the 50 qubits (28 -> 4,28,39), because
    tiles truncated at the boundary leave HX rows of weight 2.  So the contract
    is "no logical error", checked against the stabilizer group rather than
    against decode_shot's own failure flag.
    """
    view = get_code(code_id)
    stabilizer_rank = rank2(view.HX)
    for qubit in range(view.n):
        shot = decode_shot(code_id, [qubit], p=0.05)
        assert shot["failure"] is False, (code_id, qubit)
        if shot["residual"]:
            residual = np.zeros(view.n, dtype=np.uint8)
            residual[shot["residual"]] = 1
            assert rank2(np.vstack([view.HX, residual])) == stabilizer_rank, \
                (code_id, qubit)


def test_stabilizer_error_is_silent_and_harmless(view):
    """An X-stabilizer commutes with every Z-check: no syndrome, no failure.

    The residual is then the stabilizer itself — the case the page reports as
    "corrected, residual is a stabilizer".
    """
    stabilizer = np.flatnonzero(view.HX[0]).tolist()
    shot = decode_shot(CODE_ID, stabilizer, p=0.05)
    assert shot["syndrome"] == []
    assert shot["correction"] == []
    assert shot["residual"] == stabilizer
    assert shot["failure"] is False


@pytest.mark.parametrize("code_id", [CODE_ID, "rotated:3", "toric:3"])
def test_logical_error_is_a_failure(code_id):
    """An X-logical is also syndrome-free, but flips an L_Z — the failure case."""
    view = get_code(code_id)
    logical = np.flatnonzero(view.LX[0]).tolist()
    shot = decode_shot(code_id, logical, p=0.05)
    assert shot["syndrome"] == []
    assert shot["failure"] is True
    assert shot["logicals_flipped"]


def test_repeated_qubit_cancels():
    """X errors are mod 2: clicking the same qubit twice is no error at all."""
    assert decode_shot(CODE_ID, [5, 5], p=0.05)["error"] == []


def test_qubit_outside_the_code_is_rejected(view):
    with pytest.raises(ValueError):
        decode_shot(CODE_ID, [view.n], p=0.05)


def test_unknown_code_and_decoder_are_rejected():
    with pytest.raises(KeyError):
        decode_shot("tile:nope:3", [], p=0.05)
    with pytest.raises(KeyError):
        decode_shot(CODE_ID, [], p=0.05, decoder="nope")


def test_catalog_lists_without_building():
    """Listing must stay cheap — the picker is fetched before any code loads."""
    entries = catalog()
    assert len(entries) == len(CATALOG)
    assert all(set(entry) == {"id", "label"} for entry in entries)


def test_tile_qubits_sit_on_their_edges():
    """H(x,y) spans (x,y)-(x+1,y), so its midpoint is (x+0.5, y); V is the flip."""
    view = get_code(CODE_ID)
    geo = geometry(CODE_ID)
    assert len(geo["qubits"]) == view.n
    for drawn, (orient, x, y) in zip(geo["qubits"], get_code(CODE_ID).points):
        assert (drawn["shape"], drawn["x"], drawn["y"]) == (orient, x, y)


@pytest.mark.parametrize("code_id", ["rotated:3", "unrotated:3", "toric:3",
                                     "bb:[[72,12,6]]"])
def test_layout_places_every_qubit_exactly_once(code_id):
    """A layout that reuses a point would stack qubits and make clicks pick
    whichever came first, so the positions have to be distinct."""
    geo = geometry(code_id)
    view = get_code(code_id)
    assert len(geo["qubits"]) == view.n
    points = {(qubit["x"], qubit["y"]) for qubit in geo["qubits"]}
    assert len(points) == view.n


@pytest.mark.parametrize("code_id", [CODE_ID, "rotated:3", "toric:3",
                                     "bb:[[72,12,6]]"])
def test_geometry_check_rows_match_the_matrices(code_id):
    view = get_code(code_id)
    geo = geometry(code_id)
    assert geo["z_checks"] == [np.flatnonzero(row).tolist() for row in view.HZ]
    assert geo["x_checks"] == [np.flatnonzero(row).tolist() for row in view.HX]
    assert len(geo["z_centres"]) == view.HZ.shape[0]


def test_random_error_is_reproducible_and_respects_p(view):
    assert random_error(CODE_ID, 0.1, seed=3) == random_error(CODE_ID, 0.1, seed=3)
    assert random_error(CODE_ID, 0.0, seed=0) == []
    assert random_error(CODE_ID, 1.0, seed=0) == list(range(view.n))
