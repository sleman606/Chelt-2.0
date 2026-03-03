from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable

from value_finder.features import build_category_scores
from value_finder.probabilities import mean, normalize_implied_probabilities, place_probabilities, softmax_probabilities
from value_finder.scoring import compute_runner_score


@dataclass
class RunnerEvaluation:
    runner_id: str
    horse: str
    odds_decimal: float | None
    p_mkt_norm: float
    p_model_win: float
    p_model_place: float
    edge: float
    ev_win: float
    confidence_overall: float
    score: float
    score_breakdown: dict[str, Any]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RaceEvaluation:
    race_id: str
    track: str
    off_time: str
    going: str
    field_size: int
    places_k: int
    runners: list[RunnerEvaluation]

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["runners"] = [r.to_dict() for r in self.runners]
        return obj


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _parse_runners(race: dict[str, Any]) -> list[dict[str, Any]]:
    rows = race.get("runners") or race.get("entries") or []
    return [r for r in rows if isinstance(r, dict)]


def place_k_for_race(field_size: int, race: dict[str, Any], place_cfg: dict[str, Any]) -> int:
    provided = race.get("places") or race.get("each_way_places")
    if provided:
        try:
            return max(int(provided), 1)
        except Exception:
            pass

    table = place_cfg.get("fallback_by_field_size", {})
    for k, v in table.items():
        lo, hi = k.split("-")
        if int(lo) <= field_size <= int(hi):
            return int(v)
    return 3


def evaluate_race(
    date: str,
    race: dict[str, Any],
    odds_map: dict[str, dict[str, Any]],
    histories: dict[str, list[dict[str, Any]]],
    cfg: dict[str, Any],
    calibrator: Callable[[float], float] | None = None,
) -> RaceEvaluation:
    runners = _parse_runners(race)
    weights = [_safe_float(r.get("weight_lbs") or r.get("weight")) for r in runners]
    field_mean_weight = mean([w for w in weights if w is not None])

    score_map: dict[str, float] = {}
    conf_map: dict[str, float] = {}
    breakdown_map: dict[str, dict[str, Any]] = {}
    notes_map: dict[str, list[str]] = {}
    odds_simple: dict[str, float] = {}

    for runner in runners:
        rid = str(runner.get("runner_id") or runner.get("id") or runner.get("horse_id") or "")
        if not rid:
            continue
        history = histories.get(rid, [])
        cat = build_category_scores(
            runner=runner,
            race=race,
            history=history,
            cfg=cfg,
            field_ctx={"mean_weight": field_mean_weight},
            run_date=date,
        )
        score, conf, breakdown = compute_runner_score(cat)
        score_map[rid] = score
        conf_map[rid] = conf
        breakdown_map[rid] = breakdown
        notes = []
        for c in cat.values():
            notes.extend(c.notes)
        notes_map[rid] = sorted(set(notes))

        odd = _safe_float((odds_map.get(rid) or {}).get("odds_decimal") or (odds_map.get(rid) or {}).get("best_odds") or (odds_map.get(rid) or {}).get("decimal"))
        if odd and odd > 1.0:
            odds_simple[rid] = odd

    prob_cfg = cfg.get("probability", {})
    p_win = softmax_probabilities(
        scores=score_map,
        temperature=float(prob_cfg.get("softmax_temperature_win", 0.16)),
        max_prob_cap=_safe_float(prob_cfg.get("max_prob_cap")),
    )

    if calibrator:
        p_win = {k: max(0.0, min(1.0, calibrator(v))) for k, v in p_win.items()}
        denom = sum(p_win.values()) or 1.0
        p_win = {k: v / denom for k, v in p_win.items()}

    places_k = place_k_for_race(len(score_map), race, cfg.get("place_rules", {}))
    p_place = place_probabilities(
        scores=score_map,
        temperature=float(prob_cfg.get("softmax_temperature_place", 0.22)),
        k_places=places_k,
    )

    p_mkt = normalize_implied_probabilities(odds_simple)

    evals: list[RunnerEvaluation] = []
    id_to_runner = {str(r.get("runner_id") or r.get("id") or r.get("horse_id") or ""): r for r in runners}
    for rid in sorted(score_map.keys()):
        raw_runner = id_to_runner[rid]
        horse = str(raw_runner.get("horse") or raw_runner.get("horse_name") or raw_runner.get("name") or rid)
        odds = odds_simple.get(rid)
        p_model = p_win.get(rid, 0.0)
        p_market = p_mkt.get(rid, 0.0)
        edge = p_model - p_market
        ev = (p_model * odds) - 1.0 if odds else -1.0
        notes = list(notes_map.get(rid, []))
        if odds is None:
            notes.append("Missing odds")

        evals.append(
            RunnerEvaluation(
                runner_id=rid,
                horse=horse,
                odds_decimal=odds,
                p_mkt_norm=p_market,
                p_model_win=p_model,
                p_model_place=p_place.get(rid, 0.0),
                edge=edge,
                ev_win=ev,
                confidence_overall=conf_map.get(rid, 0.0),
                score=score_map.get(rid, 0.0),
                score_breakdown=breakdown_map.get(rid, {}),
                notes=sorted(set(notes)),
            )
        )

    return RaceEvaluation(
        race_id=str(race.get("race_id") or race.get("id") or ""),
        track=str(race.get("track") or race.get("course") or ""),
        off_time=str(race.get("off_time") or race.get("time") or ""),
        going=str(race.get("going") or ""),
        field_size=len(evals),
        places_k=places_k,
        runners=evals,
    )


def select_bets(races: list[RaceEvaluation], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    f = cfg.get("filters", {})
    min_ev = float(f.get("min_ev", 0.05))
    min_edge = float(f.get("min_edge", 0.03))
    min_conf = float(f.get("min_confidence", 0.6))
    min_field = int(f.get("min_field_size", 5))
    safest_min_win = float(f.get("safest_min_win_prob", 0.2))
    top_n = int(f.get("top_n_value", 10))

    all_rows: list[dict[str, Any]] = []
    for race in races:
        if race.field_size < min_field:
            continue
        for r in race.runners:
            row = {
                "race_id": race.race_id,
                "track": race.track,
                "off_time": race.off_time,
                "runner_id": r.runner_id,
                "horse": r.horse,
                "odds_decimal": r.odds_decimal,
                "p_model_win": r.p_model_win,
                "p_mkt_norm": r.p_mkt_norm,
                "edge": r.edge,
                "ev_win": r.ev_win,
                "confidence_overall": r.confidence_overall,
                "notes": r.notes,
            }
            if (
                r.odds_decimal
                and r.ev_win > min_ev
                and r.edge > min_edge
                and r.confidence_overall > min_conf
            ):
                all_rows.append(row)

    all_rows.sort(key=lambda x: (x["ev_win"], x["edge"], x["p_model_win"]), reverse=True)
    top_value = all_rows[:top_n]

    safest = [r for r in all_rows if r["p_model_win"] >= safest_min_win]
    safest.sort(key=lambda x: (x["p_model_win"], x["ev_win"]), reverse=True)

    nap = top_value[0] if top_value else None
    return top_value, safest[:top_n], nap


def day_output(
    date: str,
    races: list[RaceEvaluation],
    cfg: dict[str, Any],
    calibrated_status: str,
) -> dict[str, Any]:
    top_value, safest, nap = select_bets(races, cfg)
    return {
        "date": date,
        "calibrated_status": calibrated_status,
        "nap_by_value": nap,
        "top_value_bets": top_value,
        "safest_overlays": safest,
        "races": [r.to_dict() for r in races],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
