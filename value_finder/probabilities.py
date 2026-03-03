from __future__ import annotations

import math
from typing import Iterable


def normalize_implied_probabilities(odds: dict[str, float]) -> dict[str, float]:
    inv = {k: (1.0 / v) for k, v in odds.items() if v and v > 1.0}
    total = sum(inv.values())
    if total <= 0:
        return {k: 0.0 for k in odds}
    return {k: inv.get(k, 0.0) / total for k in odds}


def _softmax(values: list[float], temperature: float) -> list[float]:
    t = max(temperature, 1e-6)
    scaled = [v / t for v in values]
    m = max(scaled) if scaled else 0.0
    exps = [math.exp(v - m) for v in scaled]
    denom = sum(exps) or 1.0
    return [e / denom for e in exps]


def softmax_probabilities(
    scores: dict[str, float], temperature: float, max_prob_cap: float | None = None
) -> dict[str, float]:
    ids = list(scores.keys())
    probs = _softmax([scores[i] for i in ids], temperature)
    if max_prob_cap is not None and max_prob_cap > 0:
        probs = [min(p, max_prob_cap) for p in probs]
        total = sum(probs) or 1.0
        probs = [p / total for p in probs]
    return {rid: p for rid, p in zip(ids, probs)}


def place_probabilities(scores: dict[str, float], temperature: float, k_places: int) -> dict[str, float]:
    ids = list(scores.keys())
    p_raw = _softmax([scores[i] for i in ids], temperature)
    scale = float(max(k_places, 0)) / max(sum(p_raw), 1e-12)
    p_scaled = [min(1.0, max(0.0, p * scale)) for p in p_raw]
    return {rid: p for rid, p in zip(ids, p_scaled)}


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
