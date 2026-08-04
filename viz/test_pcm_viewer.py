"""Parity-check-matrix viewer export — payload shape and self-contained HTML."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

import pcm_viewer
from qec_tile.directional import PAPER_CODES
from qec_tile.tile import TILES, paper_code


@pytest.fixture
def code():
    """[[50,8]] — the smallest paper tile, cheap enough to rebuild per test."""
    return paper_code("b3w6", 3, 3)


@pytest.fixture
def payload(code):
    return pcm_viewer.matrix_payload("b3w6 L=3", code.HX, code.HZ)


def test_payload_rows_are_the_matrix_support(code, payload):
    """Rows go over the wire as ascending column indices, not 0/1 vectors."""
    assert payload["X"] == [np.flatnonzero(row).tolist() for row in code.HX]
    assert payload["Z"] == [np.flatnonzero(row).tolist() for row in code.HZ]


def test_payload_counts_match_the_code(code, payload):
    """The numbers behind the viewer's stat cards."""
    assert payload["n"] == code.n == 2 * (3 + 3 - 1) * (3 + 3 - 1)
    assert payload["k"] == code.k
    assert payload["mx"] == code.HX.shape[0]
    assert payload["mz"] == code.HZ.shape[0]
    assert payload["wx"] == int(code.HX.sum(1).max())
    assert payload["wz"] == int(code.HZ.sum(1).max())


def test_payload_is_json_serialisable(payload):
    """numpy scalars survive every assert above but blow up at json.dumps."""
    json.dumps(payload)


def test_catalog_covers_every_paper_directional_code():
    """Every row of Table 2, built with M and N the right way round."""
    entries = pcm_viewer.build_catalog(layouts=(4,), extras=False)
    by_param = {entry["param"]: entry
                for entry in entries if entry["group"] == "directional"}
    for word, M, N, n, k, _d in PAPER_CODES:
        entry = by_param[f"{word}, {M}x{N}"]
        assert (entry["n"], entry["k"]) == (n, k)


def test_catalog_tile_entries_use_square_layouts():
    entries = pcm_viewer.build_catalog(layouts=(4,), directional=False,
                                       extras=False)
    by_group = {entry["group"]: entry for entry in entries}
    assert set(by_group) == {f"tile {name}" for name in TILES}
    for name in TILES:
        entry = by_group[f"tile {name}"]
        B = int(name[1])                       # "b3w6" -> 3
        assert entry["param"] == "L=4"
        assert entry["n"] == 2 * (4 + B - 1) ** 2


def test_every_catalog_entry_commutes():
    """The only check qec_pem.py's surface and BB constructors ever get."""
    for entry in pcm_viewer.build_catalog(layouts=(4,)):
        assert entry["css"], entry["label"]


def test_render_substitutes_the_payload():
    """A missed placeholder still writes a file — it just opens up blank."""
    html = pcm_viewer.render(pcm_viewer.build_catalog(layouts=(4,),
                                                      extras=False))
    assert "__PAYLOAD__" not in html
    assert "tile b3w6" in html


def test_rendered_page_is_self_contained():
    html = pcm_viewer.render(pcm_viewer.build_catalog(layouts=(4,),
                                                      directional=False,
                                                      extras=False))
    assert not re.search(r'(?:src|href)\s*=\s*["\']https?:', html)


def test_payload_cannot_close_the_script_tag():
    """HTML ends a script at the first </script>, JSON string or not."""
    entry = dict(pcm_viewer.matrix_payload("</script><b>x",
                                           np.eye(2, dtype=np.uint8),
                                           np.zeros((0, 2), dtype=np.uint8)),
                 group="g", param="p")
    html = pcm_viewer.render([entry])
    template = Path(pcm_viewer.TEMPLATE).read_text()
    assert html.count("</script>") == template.count("</script>")


def test_payload_carries_the_block_shape(code):
    """Absent by default: a code with no repeating block must not get lines."""
    plain = pcm_viewer.matrix_payload("x", code.HX, code.HZ)
    assert plain["block"] is None
    shaped = pcm_viewer.matrix_payload("x", code.HX, code.HZ,
                                       block=dict(rows_x=3, rows_z=3,
                                                  cols=[5, 5]))
    assert shaped["block"] == dict(rows_x=3, rows_z=3, cols=[5, 5])


def test_catalog_block_shapes_match_the_construction():
    """Where the index arithmetic folds, and it is not the same on both axes.

    Rows count checks and columns count qubits, so a tile's block is
    L2 x (L2+B-1) -- the qubit lattice is B-1 wider than the anchor grid.  An
    HGP splits differently again: H_X rows are indexed (a, j) and H_Z rows
    (i, b), so the two matrices fold at different heights -- d=5 is the case
    where all four numbers are not the same, so an axis mix-up shows.
    """
    entries = {(entry["group"], entry["param"]): entry
               for entry in pcm_viewer.build_catalog(layouts=(4,))}

    # X anchors sweep j over [-(B-1), L2+B-1) and Z anchors over [0, L2), so
    # the two matrices fold at different heights: 4+2*2 = 8 against 4.
    tile = entries[("tile b3w6", "L=4")]
    assert tile["block"] == dict(rows_x=8, rows_z=4, cols=[6, 6])
    assert tile["dividers"] == [tile["n"] // 2]

    bb = entries[("bivariate bicycle", "[[72,12,6]]")]
    assert bb["block"] == dict(rows_x=6, rows_z=6, cols=[6, 6])

    unrotated = entries[("unrotated surface", "d=5")]
    assert unrotated["block"] == dict(rows_x=5, rows_z=4, cols=[5, 4])

    # A lattice row of the rotated code: (d-1)/2 faces plus one boundary check.
    assert entries[("rotated surface", "d=5")]["block"] == \
        dict(rows_x=3, rows_z=3, cols=[5])


def test_rotated_surface_checks_are_one_sweep():
    """Each lattice row carries its faces and its own boundary check.

    The faces of a rotated code sit on a checkerboard, so within a row the
    support steps two columns at a time; the weight-2 boundary check belongs to
    that same row and is emitted in its j order rather than appended at the
    end.  That makes every row block the same height -- (d-1)/2 faces plus one
    -- which is what lets the viewer rule a grid at all.
    """
    import numpy as np
    from qec_pem import rotated_surface_code

    d = 7
    HX, _HZ = rotated_surface_code(d)
    group = (d - 1) // 2 + 1
    assert HX.shape[0] == (d - 1) * group

    for start in range(0, HX.shape[0], group):
        rows = HX[start:start + group]
        firsts = [int(np.flatnonzero(row)[0]) for row in rows]
        assert firsts == sorted(firsts), (start, firsts)
        assert sorted(int(row.sum()) for row in rows) == [2] + [4] * (group - 1)


def test_catalog_labels_are_unique():
    """Colliding (group, param) pairs would stack up in the viewer's select."""
    keys = [(entry["group"], entry["param"])
            for entry in pcm_viewer.build_catalog(layouts=(4, 6))]
    assert len(set(keys)) == len(keys)
