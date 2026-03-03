from __future__ import annotations

from datetime import datetime

from value_finder.features import chelt_form_index


def test_chelt_smoothing_small_sample_shrinks_to_prior() -> None:
    ref = datetime(2026, 3, 3)
    runs = [{"date": "2026-01-01", "finish_position": 1}]
    score, n, _ = chelt_form_index(runs, ref, recency_years=3, alpha=2.0, beta=3.0)
    assert n >= 1
    assert 0.45 <= score <= 0.7


def test_chelt_smoothing_larger_sample_moves_away_from_prior() -> None:
    ref = datetime(2026, 3, 3)
    runs = [
        {"date": "2025-12-01", "finish_position": 1},
        {"date": "2025-11-01", "finish_position": 2},
        {"date": "2025-10-01", "finish_position": 3},
        {"date": "2025-09-01", "finish_position": 1},
        {"date": "2025-08-01", "finish_position": 2},
    ]
    score, n, _ = chelt_form_index(runs, ref, recency_years=3, alpha=2.0, beta=3.0)
    assert n >= 3
    assert score > 0.6


def test_chelt_recency_weighting() -> None:
    ref = datetime(2026, 3, 3)
    recent = [{"date": "2025-12-01", "finish_position": 1}]
    old = [{"date": "2023-03-03", "finish_position": 1}]
    s_recent, _, _ = chelt_form_index(recent, ref, recency_years=3, alpha=2.0, beta=3.0)
    s_old, _, _ = chelt_form_index(old, ref, recency_years=3, alpha=2.0, beta=3.0)
    assert s_recent > s_old
