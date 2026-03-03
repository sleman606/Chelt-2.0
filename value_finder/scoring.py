from __future__ import annotations

from typing import Any

from value_finder.features import CategoryScore


def compute_runner_score(categories: dict[str, CategoryScore]) -> tuple[float, float, dict[str, Any]]:
    total_weight = sum(c.weight for c in categories.values()) or 1.0
    weighted_sum = sum(c.contribution for c in categories.values())
    score = weighted_sum / total_weight
    score = max(0.0, min(1.0, score))
    confidence = sum(c.confidence for c in categories.values()) / max(len(categories), 1)
    breakdown = {k: v.to_dict() for k, v in categories.items()}
    return score, confidence, breakdown
