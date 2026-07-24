"""Allocation settings from .env — no defaults, loud when missing."""
import os

import pytest

from qec_tile import config


def test_require_parses_a_set_value(monkeypatch):
    monkeypatch.setenv("QEC_TILE_WORKERS", "12")
    assert config.workers() == 12


def test_missing_value_raises_with_guidance(monkeypatch):
    monkeypatch.delenv("QEC_TILE_WORKERS", raising=False)
    with pytest.raises(RuntimeError, match="QEC_TILE_WORKERS.*\\.env"):
        config.workers()


def test_blank_value_counts_as_missing(monkeypatch):
    monkeypatch.setenv("QEC_TILE_GPU", "   ")
    with pytest.raises(RuntimeError, match="QEC_TILE_GPU"):
        config.gpu()


def test_garbage_value_raises_by_name(monkeypatch):
    monkeypatch.setenv("QEC_TILE_THREADS", "one")
    with pytest.raises(ValueError, match="QEC_TILE_THREADS"):
        config.threads()


def test_thread_limits_respect_existing(monkeypatch):
    monkeypatch.setenv("QEC_TILE_THREADS", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    config.apply_thread_limits()
    assert os.environ["OMP_NUM_THREADS"] == "5"


def test_thread_limits_fill_unset(monkeypatch):
    monkeypatch.setenv("QEC_TILE_THREADS", "2")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    config.apply_thread_limits()
    assert os.environ["OMP_NUM_THREADS"] == "2"


def test_thread_limits_do_nothing_when_unconfigured(monkeypatch):
    monkeypatch.delenv("QEC_TILE_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    config.apply_thread_limits()
    assert "OMP_NUM_THREADS" not in os.environ