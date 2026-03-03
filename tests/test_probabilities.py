from __future__ import annotations

from value_finder.probabilities import normalize_implied_probabilities, place_probabilities, softmax_probabilities


def test_implied_probability_normalization_sums_to_one() -> None:
    odds = {"a": 2.0, "b": 4.0, "c": 8.0}
    p = normalize_implied_probabilities(odds)
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_softmax_probabilities_sum_to_one() -> None:
    scores = {"a": 0.2, "b": 0.5, "c": 0.8}
    p = softmax_probabilities(scores, temperature=0.2)
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_place_probability_scaling_bounded() -> None:
    scores = {"a": 0.1, "b": 0.2, "c": 0.9, "d": 0.7}
    p = place_probabilities(scores, temperature=0.3, k_places=3)
    assert all(0.0 <= v <= 1.0 for v in p.values())
    assert sum(p.values()) <= 3.0 + 1e-9
