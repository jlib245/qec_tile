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


def test_catalog_labels_are_unique():
    """Colliding (group, param) pairs would stack up in the viewer's select."""
    keys = [(entry["group"], entry["param"])
            for entry in pcm_viewer.build_catalog(layouts=(4, 6))]
    assert len(set(keys)) == len(keys)
