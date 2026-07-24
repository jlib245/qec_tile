"""Parallel shot collection through sinter."""
import pytest

from qec_tile.circuit import memory_z_base, memory_z_circuit
from qec_tile.sinter_sampling import collect
from qec_tile.tile import paper_code

SMALL = ("b3w6", 2, 2)


def test_collect_returns_stats_for_every_circuit():
    code = paper_code(*SMALL)
    circuits = {("L2", 0.02): memory_z_circuit(code, 2, 0.02),
                ("L2", 0.05): memory_z_circuit(code, 2, 0.05)}
    stats = collect(circuits, decoder="bposd", max_shots=50, workers=2)
    assert set(stats) == set(circuits)
    for shots, errors in stats.values():
        assert 0 < shots <= 50
        assert 0 <= errors <= shots


def test_zero_noise_gives_zero_errors():
    code = paper_code(*SMALL)
    stats = collect({"clean": memory_z_base(code, 2)}, decoder="bposd",
                    max_shots=30, workers=2)
    assert stats["clean"][1] == 0


def test_unknown_decoder_is_rejected():
    with pytest.raises(ValueError, match="unknown decoder"):
        collect({}, decoder="nn", max_shots=10)


def test_max_errors_stops_early():
    """p=0.05 at rounds=2 fails ~30% of shots, so 5 errors arrive fast."""
    code = paper_code(*SMALL)
    noisy = memory_z_circuit(code, 2, 0.05)
    stats = collect({"x": noisy}, decoder="bposd",
                    max_shots=10_000, max_errors=5, workers=2)
    shots, errors = stats["x"]
    assert errors >= 5
    assert shots < 10_000
