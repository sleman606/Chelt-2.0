from __future__ import annotations

from value_finder.value import RaceEvaluation, RunnerEvaluation, select_bets


def test_ev_and_edge_values() -> None:
    p_model = 0.25
    odds = 6.0
    p_mkt = 0.18
    ev = p_model * odds - 1
    edge = p_model - p_mkt
    assert abs(ev - 0.5) < 1e-9
    assert abs(edge - 0.07) < 1e-9


def test_guardrails_filtering() -> None:
    race = RaceEvaluation(
        race_id="r1",
        track="Cheltenham",
        off_time="13:20",
        going="Good",
        field_size=8,
        places_k=3,
        runners=[
            RunnerEvaluation(
                runner_id="x",
                horse="Value Horse",
                odds_decimal=5.0,
                p_mkt_norm=0.18,
                p_model_win=0.27,
                p_model_place=0.45,
                edge=0.09,
                ev_win=0.35,
                confidence_overall=0.7,
                score=0.66,
                score_breakdown={},
                notes=[],
            ),
            RunnerEvaluation(
                runner_id="y",
                horse="No Value",
                odds_decimal=2.5,
                p_mkt_norm=0.42,
                p_model_win=0.39,
                p_model_place=0.58,
                edge=-0.03,
                ev_win=-0.025,
                confidence_overall=0.8,
                score=0.72,
                score_breakdown={},
                notes=[],
            ),
        ],
    )
    cfg = {
        "filters": {
            "min_ev": 0.05,
            "min_edge": 0.03,
            "min_confidence": 0.6,
            "min_field_size": 5,
            "top_n_value": 10,
            "safest_min_win_prob": 0.2,
        }
    }
    top, safest, nap = select_bets([race], cfg)
    assert len(top) == 1
    assert top[0]["horse"] == "Value Horse"
    assert nap and nap["horse"] == "Value Horse"
    assert len(safest) == 1
