"""sinter를 통한 병렬 shot 수집."""
import pytest

from qec_tile.circuit import memory_z_base, memory_z_circuit
from qec_tile.sinter_sampling import collect
from qec_tile.tile import paper_code

SMALL = ("b3w6", 2, 2)


def test_collect_returns_stats_for_every_circuit():
    code = paper_code(*SMALL)
    circuits = {("L2", 0.02): memory_z_circuit(code, 2, 0.02),
                ("L2", 0.05): memory_z_circuit(code, 2, 0.05)}
    stats = collect(circuits, decoder="bposd_cs7", max_shots=50, workers=2)
    assert set(stats) == set(circuits)
    for counts in stats.values():
        assert 0 < counts.shots <= 50
        assert 0 <= counts.block_fails <= counts.shots


def test_zero_noise_gives_zero_errors():
    code = paper_code(*SMALL)
    stats = collect({"clean": memory_z_base(code, 2)}, decoder="bposd_cs7",
                    max_shots=30, workers=2)
    assert (stats["clean"].block_fails, stats["clean"].logical_flips) == (0, 0)
    assert stats["clean"].k == code.k


def test_collect_counts_logical_flips():
    """sinter의 observable 조합 카운트에서 뒤집힌 observable 수를 직접 센다."""
    code = paper_code(*SMALL)
    counts = collect({"x": memory_z_circuit(code, 2, 0.05)},
                     decoder="bposd_cs7", max_shots=100, workers=2)["x"]
    assert counts.k == code.k
    assert 0 < counts.block_fails <= counts.logical_flips <= counts.k * counts.shots


def test_unknown_decoder_is_rejected():
    with pytest.raises(ValueError, match="unknown decoder"):
        collect({}, decoder="nn", max_shots=10)


def test_registry_mirrors_serial_except_lsd_cs():
    """SinterLsdDecoder에는 lsd_method 손잡이가 없어서 bplsd_cs7은 serial 전용."""
    from qec_tile.decode import DECODERS
    from qec_tile.sinter_sampling import SINTER_DECODERS
    assert set(SINTER_DECODERS) == set(DECODERS) - {"bplsd_cs7"}


def test_vibelsd_collects_through_sinter():
    """VibeLSD가 worker로 pickle되어 ``decode_via_files`` 계약대로 돈다."""
    code = paper_code(*SMALL)
    stats = collect({"x": memory_z_circuit(code, 2, 0.02)},
                    decoder="vibelsd_32", max_shots=40, workers=2)
    assert 0 < stats["x"].shots <= 40
    assert 0 <= stats["x"].block_fails <= stats["x"].shots


def test_max_errors_stops_early():
    """rounds=2에서 p=0.05는 shot의 ~30%를 실패시키므로 오류 5개가 빨리 모인다."""
    code = paper_code(*SMALL)
    noisy = memory_z_circuit(code, 2, 0.05)
    stats = collect({"x": noisy}, decoder="bposd_cs7",
                    max_shots=10_000, max_errors=5, workers=2)
    assert stats["x"].block_fails >= 5
    assert stats["x"].shots < 10_000
