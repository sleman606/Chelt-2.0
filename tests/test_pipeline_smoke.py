from __future__ import annotations

from value_finder.cli import build_day_output
from value_finder.config import AppConfig
from value_finder.storage import Storage


class DummyCache:
    def __init__(self) -> None:
        self.data = {}

    def get(self, key: str, ttl_seconds: int | None = None):
        return self.data.get(key)

    def set(self, key: str, value):
        self.data[key] = value


class DummyClient:
    def get_racecards(self, date: str, country: str = "UK"):
        return [{"race_id": "r1"}]

    def get_race(self, race_id: str):
        return {
            "race_id": "r1",
            "track": "Cheltenham",
            "off_time": "13:20",
            "going": "Good",
            "runners": [
                {"runner_id": "h1", "horse": "Horse A", "horse_id": "h1", "weight_lbs": 160},
                {"runner_id": "h2", "horse": "Horse B", "horse_id": "h2", "weight_lbs": 156},
                {"runner_id": "h3", "horse": "Horse C", "horse_id": "h3", "weight_lbs": 158},
                {"runner_id": "h4", "horse": "Horse D", "horse_id": "h4", "weight_lbs": 155},
                {"runner_id": "h5", "horse": "Horse E", "horse_id": "h5", "weight_lbs": 154},
            ],
        }

    def get_runner_history(self, horse_id: str, lookback_days: int = 1825):
        return [
            {"date": "2025-11-01", "finish_position": 1, "track": "Cheltenham", "going": "Good", "distance": 4000},
            {"date": "2025-09-01", "finish_position": 4, "track": "Aintree", "going": "Soft", "distance": 3900},
        ]

    def get_odds(self, race_id: str):
        return {
            "h1": {"odds_decimal": 4.5},
            "h2": {"odds_decimal": 6.0},
            "h3": {"odds_decimal": 8.0},
            "h4": {"odds_decimal": 7.0},
            "h5": {"odds_decimal": 10.0},
        }


def test_pipeline_smoke(tmp_path):
    cfg = {
        "api": {"country": "UK"},
        "features": {
            "recent_form": {"n_runs": 5, "half_life_days": 90},
            "distance": {"band_pct": 0.1},
            "going": {"bucket_map": {"good": ["Good"]}},
            "priors": {k: 0.5 for k in ["recent_form", "course_distance", "going", "handicap_ratings", "jockey", "trainer", "cheltenham_form"]},
            "confidence_saturation": {k: 5 for k in ["recent_form", "course_distance", "going", "handicap_ratings", "jockey", "trainer", "cheltenham_form"]},
            "cheltenham": {"recency_years": 3, "shrinkage_alpha": 2.0, "shrinkage_beta": 3.0},
        },
        "weights": {k: 1.0 for k in ["recent_form", "course_distance", "going", "handicap_ratings", "jockey", "trainer", "cheltenham_form"]},
        "probability": {"softmax_temperature_win": 0.2, "softmax_temperature_place": 0.25, "max_prob_cap": 0.7},
        "filters": {"min_ev": 0.01, "min_edge": 0.0, "min_confidence": 0.1, "min_field_size": 5, "top_n_value": 10, "safest_min_win_prob": 0.1},
        "place_rules": {"fallback_by_field_size": {"5-7": 2, "8-11": 3}},
        "storage": {"backend": "sqlite", "sqlite_path": str(tmp_path / "test.sqlite3")},
        "calibration": {"artifact_path": str(tmp_path / "cal.json")},
        "reporting": {"output_dir": str(tmp_path), "json_name": "r.json", "markdown_name": "r.md"},
    }
    app_cfg = AppConfig(raw=cfg)
    storage = Storage(cfg)
    try:
        out = build_day_output("2026-03-03", app_cfg, DummyClient(), DummyCache(), storage)
        assert out["date"] == "2026-03-03"
        assert "races" in out and len(out["races"]) == 1
        assert "top_value_bets" in out
    finally:
        storage.close()
