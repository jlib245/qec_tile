"""benchmark.py의 CLI 해석 로직."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from benchmark import resolve_max_iter


def test_max_iter_stays_unset_without_workers():
    """단일 프로세스 경로는 max_iter를 안 쓴다 — 값을 정하면 파일명이 거짓말한다.

    ``_iter60``이 붙은 CSV가 실제로는 디코더 기본값 15회로 디코딩되어 나온다.
    """
    assert resolve_max_iter("vibelsd_200", None, {4}, None) is None


def test_max_iter_follows_the_round_count_with_workers():
    """sinter 경로에서는 directional 논문대로 15 x rounds."""
    assert resolve_max_iter("vibelsd_200", None, {4}, 24) == 60
