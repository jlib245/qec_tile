"""Decoding viewer backend — BP on a syndrome, layouts, and sampling.

The HTTP layer is a thin wrapper (create_app) and needs fastapi installed, so
these exercise the functions behind it instead.  Codes are kept small on
purpose: solving for logicals is the expensive part of building one.

There is no OSD in this viewer, so "BP stalled" is a third outcome next to
"corrected" and "logical_error", and the tests say so rather than pretending
weight-1 errors always come back.
"""
import numpy as np
import pytest

from decode_server import (CATALOG, bp_messages, bp_trace, catalog, geometry,
                           get_code, random_error, syndrome_of)

from qec_tile.gf2 import nullspace2

CODE_ID = "tile:b3w6:3"
FAMILIES = [CODE_ID, "rotated:3", "toric:3", "bb:[[72,12,6]]"]
# The codes that lose something: the rotated ones are cut to a diamond, and a
# directional walk shorter than its box gets pruned.  Paper tiles lose nothing.
GHOST_CODES = ["rotated:3", "rotated:5", "rotated:7", "dir:N2ESEN2:4x4",
               "dir:N2E2SE2N2:5x4", "dir:N2E2SESE2N2:5x4"]


@pytest.fixture
def view():
    return get_code(CODE_ID)


def test_zero_syndrome_converges_immediately():
    """Nothing lit means the zero error explains it — BP should say so at once."""
    result = bp_trace(CODE_ID, [], p=0.05)
    assert result["outcome"] == "corrected"
    assert result["correction"] == []
    assert result["converged_at"] == 1
    assert len(result["trace"]) == 1


@pytest.mark.parametrize("code_id", FAMILIES)
def test_single_error_never_causes_a_logical_error(code_id):
    """BP alone does stall on weight-1 errors (6 of 50 qubits on b3w6 L=3,
    4 of 9 on rotated d=3), but when it does converge it never lands on a
    logical.  That, not "always corrected", is what BP without OSD promises.
    """
    view = get_code(code_id)
    stalled = 0
    for qubit in range(view.n):
        result = bp_trace(code_id, syndrome_of(code_id, [qubit]), p=0.05,
                          error=[qubit])
        assert result["outcome"] in ("corrected", "stalled"), (code_id, qubit)
        stalled += result["outcome"] == "stalled"
    assert stalled < view.n                     # not a wholesale failure


def test_a_stalled_shot_has_no_valid_correction():
    """Stalling is not a worse answer, it is no answer: H @ correction != s."""
    view = get_code(CODE_ID)
    for qubit in range(view.n):
        result = bp_trace(CODE_ID, syndrome_of(CODE_ID, [qubit]), p=0.05)
        if result["outcome"] != "stalled":
            continue
        correction = np.zeros(view.n, dtype=np.uint8)
        correction[result["correction"]] = 1
        assert not np.array_equal((view.HZ @ correction) % 2,
                                  np.isin(np.arange(view.HZ.shape[0]),
                                          result["syndrome"]).astype(np.uint8))
        assert result["converged_at"] is None
        assert result["residual"] is None       # nothing to compare against
        return
    pytest.fail("no stalling weight-1 error on this code any more")


def test_error_derived_syndromes_are_always_reachable():
    result = bp_trace(CODE_ID, syndrome_of(CODE_ID, [4, 20]), p=0.05)
    assert result["reachable"]


def test_an_unreachable_syndrome_is_reported_as_such():
    """Toric checks are dependent, so some lit patterns cannot come from any
    error at all.  The oracle here is the left nullspace -- a combination
    ``u`` of checks with ``u @ HZ = 0`` -- which is independent of the rank
    comparison the server uses: if ``u @ s`` is odd, no error explains ``s``.
    """
    view = get_code("toric:3")
    dependencies = nullspace2(view.HZ.T)
    assert dependencies.size, "toric checks are supposed to be dependent"
    check = int(np.flatnonzero(dependencies[0])[0])
    result = bp_trace("toric:3", [check], p=0.05)
    assert not result["reachable"]
    assert result["outcome"] == "unreachable"


def test_error_must_match_the_syndrome():
    """Passing both is a claim about one shot; a mismatch is a caller bug."""
    with pytest.raises(ValueError):
        bp_trace(CODE_ID, syndrome_of(CODE_ID, [4]), p=0.05, error=[20])


def test_the_method_reaches_the_decoder():
    """Sum-product solves a shot min-sum stalls on (tests/test_bp.py measures
    it), so the two must not be reporting the same run."""
    syndrome = syndrome_of("tile:b4w10:3", [5, 17])
    exact = bp_trace("tile:b4w10:3", syndrome, p=0.05, method="product_sum")
    approximate = bp_trace("tile:b4w10:3", syndrome, p=0.05,
                           method="minimum_sum")
    assert exact["outcome"] == "corrected" and exact["converged_at"] == 3
    assert approximate["outcome"] == "stalled"


def test_messages_line_up_with_the_edges():
    """One value per edge per direction, in the order geometry reports them."""
    view = get_code(CODE_ID)
    geo = geometry(CODE_ID)
    assert len(geo["edges"]) == int(view.HZ.sum())
    for check, qubit in geo["edges"]:
        assert view.HZ[check, qubit] == 1

    window = bp_messages(CODE_ID, syndrome_of(CODE_ID, [4, 20]), p=0.05,
                         iteration=2, count=3)
    assert [step["iteration"] for step in window["steps"]] == [2, 3, 4]
    for step in window["steps"]:
        assert len(step["to_check"]) == len(geo["edges"])
        assert len(step["to_bit"]) == len(geo["edges"])


def test_messages_stop_where_bp_stopped():
    """Sweeps that never happened are absent, not fabricated — the page's
    window runs one iteration past the slider and must survive the end."""
    converging = "tile:b4w8:3"               # a single error it solves at once
    syndrome = syndrome_of(converging, [3])
    trace = bp_trace(converging, syndrome, p=0.05)
    assert trace["converged_at"] == 1, "this shot is supposed to converge at 1"
    assert bp_messages(converging, syndrome, p=0.05, iteration=40)["steps"] == []
    window = bp_messages(converging, syndrome, p=0.05, iteration=1, count=2)
    assert [step["iteration"] for step in window["steps"]] == [1]


def test_max_iter_bounds_the_trace():
    result = bp_trace(CODE_ID, syndrome_of(CODE_ID, [4, 20]), p=0.05,
                      max_iter=3)
    assert result["max_iter"] == 3
    assert len(result["trace"]) <= 3


def test_trace_llr_covers_every_qubit(view):
    result = bp_trace(CODE_ID, syndrome_of(CODE_ID, [4, 20]), p=0.05)
    assert all(len(step["llr"]) == view.n for step in result["trace"])


def test_repeated_check_cancels():
    """Clicking the same check twice clears it, as on the page."""
    assert bp_trace(CODE_ID, [2, 2], p=0.05)["syndrome"] == []


def test_index_outside_the_code_is_rejected(view):
    with pytest.raises(ValueError):
        bp_trace(CODE_ID, [view.HZ.shape[0]], p=0.05)


def test_unknown_code_is_rejected():
    with pytest.raises(KeyError):
        bp_trace("tile:nope:3", [], p=0.05)


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
    for drawn, (shape, x, y) in zip(geo["qubits"], view.points):
        assert (drawn["shape"], drawn["x"], drawn["y"]) == (shape, x, y)


def test_boundary_checks_hang_off_the_lattice():
    """A truncated tile keeps its box, so its check sits outside the qubits.

    Placing it at the centroid of the surviving support instead would drag it
    into the bulk — check (-2,0) on b3w6 L=3 belongs at x = -1 but its support
    averages to x = 0.25, on top of the lattice it is supposed to border.
    """
    geo = geometry(CODE_ID)
    left_edge = min(qubit["x"] for qubit in geo["qubits"])
    assert min(centre[0] for centre in geo["z_centres"]) < left_edge


@pytest.mark.parametrize("code_id", ["toric:3", "unrotated:3"])
def test_hgp_checks_sit_on_the_edge_they_join(code_id):
    """Each HGP check goes at its own index, not at the mean of its support.

    On the cyclic (toric) code a check joins row 0 to row L-1, so the mean
    lands in the middle of the lattice, a full lattice step from either qubit.
    Here every check must be half a step from one of the qubits it acts on.
    """
    geo = geometry(code_id)
    qubits = np.array([[qubit["x"], qubit["y"]] for qubit in geo["qubits"]])
    for centre, support in zip(geo["z_centres"], geo["z_checks"]):
        nearest = np.hypot(*(qubits[support] - np.array(centre)).T).min()
        assert nearest <= 0.75, (centre, support)


@pytest.mark.parametrize("code_id,period", [("toric:4", [4.0, 4.0]),
                                            ("bb:[[72,12,6]]", [6.0, 6.0]),
                                            ("unrotated:4", None),
                                            (CODE_ID, None)])
def test_only_wrapping_layouts_report_a_period(code_id, period):
    """The page draws ghost copies at this offset, so a period on a layout
    with boundaries would show neighbours that do not exist."""
    assert geometry(code_id)["period"] == period


def test_bb_uses_four_sublattices():
    """Two qubit sectors and two check types, one per corner of the cell.

    Putting the checks on the qubits' own sublattices would hide them under
    the data qubits — the L sector already owns the vertices and the R sector
    the face centres.
    """
    geo = geometry("bb:[[72,12,6]]")
    l, m = 6, 6
    assert {tuple(centre) for centre in geo["x_centres"]} == \
        {(j + 0.5, float(i)) for i in range(l) for j in range(m)}
    assert {tuple(centre) for centre in geo["z_centres"]} == \
        {(float(j), i + 0.5) for i in range(l) for j in range(m)}


def test_check_points_never_land_on_a_qubit():
    """Overlapping nodes make a click ambiguous, on any of these layouts."""
    for code_id in FAMILIES:
        geo = geometry(code_id)
        qubits = {(qubit["x"], qubit["y"]) for qubit in geo["qubits"]}
        assert not qubits & {tuple(centre) for centre in geo["z_centres"]}, code_id


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


@pytest.mark.parametrize("code_id", FAMILIES)
def test_geometry_check_rows_match_the_matrices(code_id):
    view = get_code(code_id)
    geo = geometry(code_id)
    assert geo["z_checks"] == [np.flatnonzero(row).tolist() for row in view.HZ]
    assert geo["x_checks"] == [np.flatnonzero(row).tolist() for row in view.HX]
    assert len(geo["z_centres"]) == view.HZ.shape[0]


def slot_entries(geo, name):
    """The block as the viewer draws it: real 1s and cut edges, by slot."""
    row_slots = geo["slots"][f"{name}_rows"]
    real = {(row_slots[row], geo["slots"]["columns"][column])
            for row, support in enumerate(geo[f"{name}_checks"])
            for column in support}
    return real, {tuple(edge) for edge in geo[f"ghost_edges_{name}"]}


@pytest.mark.parametrize("code_id", GHOST_CODES)
def test_a_cut_edge_never_lands_on_a_real_one(code_id):
    """Both ends have to be gone for a connection to count as cut.  Reading it
    as "the qubit is off the lattice" instead would draw a ghost cell on top of
    an entry the code actually has, on every truncated boundary check."""
    geo = geometry(code_id)
    for name in "xz":
        real, cut = slot_entries(geo, name)
        assert not real & cut, (code_id, name)


@pytest.mark.parametrize("code_id", ["dir:N2ESEN2:4x4", "dir:N2E2SE2N2:5x4",
                                     "dir:N2E2SESE2N2:5x4"])
def test_slot_space_blocks_are_the_layout_before_pruning(code_id):
    """The slot matrix is the layout the construction started from, so its
    period is the paper's arithmetic and its sector split counts slots, not
    surviving qubits.  Drawn at the code's own boundary the split lands
    mid-sector and the right-hand block gets ruled at the left-hand width.
    """
    geo = geometry(code_id)
    slots, block = geo["slots"], geo["slots"]["block"]
    divider, = slots["dividers"]
    # H edges come first and both sectors are the same rectangle of lattice
    # columns -- pruning is what makes the two uneven in the code's own matrix.
    assert slots["column_count"] == 2 * divider
    assert block["cols"] == [block["cols"][0]] * 2
    assert divider % block["cols"][0] == 0
    assert slots["x_row_count"] % block["rows_x"] == 0
    assert slots["z_row_count"] % block["rows_z"] == 0
    # And the folded split is the code's own: the same qubits, gaps closed up.
    assert sum(1 for slot in slots["columns"] if slot < divider) == \
        geo["dividers"][0]


@pytest.mark.parametrize("code_id", ["rotated:5", "dir:N2ESEN2:4x4"])
def test_pruned_rows_and_columns_carry_their_cut_edges(code_id):
    """A row pruning took is the connections it had, not a blank line.

    With the pruned rows and columns back in their slots the matrix is the one
    the construction laid out, so every row and column of it -- surviving or
    not -- has something in it; a ghost row with nothing drawn would say the
    check reached for no qubits.

    Not every code can be asked this: the anchor rectangle is sized by the box
    B, so a walk shorter than its box leaves anchors whose stamp never reaches
    the lattice, and those rows stay blank on purpose -- N2E2SE2N2 has ten of
    them.  These two span their box.
    """
    geo = geometry(code_id)
    slots = geo["slots"]
    columns = set()
    for name in "xz":
        real, cut = slot_entries(geo, name)
        drawn = real | cut
        assert {row for row, _column in drawn} == \
            set(range(slots[f"{name}_row_count"])), code_id
        columns |= {column for _row, column in drawn}
    # A tile's X and Z offsets differ, so an edge no X-check reaches leaves its
    # column empty in H_X however little was pruned; only across both blocks
    # does every qubit of the lattice have to connect to something.
    assert columns == set(range(slots["column_count"])), code_id


@pytest.mark.parametrize("d", [3, 5])
def test_the_pruned_rotated_matrix_is_the_unrotated_one(d):
    """The diamond is cut out of the square, so putting the cut rows, columns
    and edges back gives the square's own matrix -- entry for entry, at the
    same row and column, not merely a matrix of the same shape.  That only
    holds if the slots are the unrotated code's indices; any other order (a
    sort of the coordinates, say) leaves the entries scattered.
    """
    rotated = geometry(f"rotated:{d}")
    square = geometry(f"unrotated:{d}")
    slots = rotated["slots"]
    for name in "xz":
        drawn = {(slots[f"{name}_rows"][row], slots["columns"][column])
                 for row, support in enumerate(rotated[f"{name}_checks"])
                 for column in support}
        drawn |= {tuple(edge) for edge in rotated[f"ghost_edges_{name}"]}
        assert drawn == {(row, column)
                         for row, support in enumerate(square[f"{name}_checks"])
                         for column in support}, name


def test_random_error_is_reproducible_and_respects_p(view):
    assert random_error(CODE_ID, 0.1, seed=3) == random_error(CODE_ID, 0.1, seed=3)
    assert random_error(CODE_ID, 0.0, seed=0) == []
    assert random_error(CODE_ID, 1.0, seed=0) == list(range(view.n))
