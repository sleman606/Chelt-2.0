from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any


@dataclass
class CategoryScore:
    raw: float
    confidence: float
    adjusted: float
    weight: float
    contribution: float
    samples: int
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _conf_from_n(n: int, saturation: float) -> float:
    sat = max(saturation, 1e-6)
    return _clip01(n / (n + sat))


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _finish_score(pos: Any) -> float:
    try:
        p = int(pos)
    except Exception:
        return 0.3
    if p <= 1:
        return 1.0
    if p <= 3:
        return 0.75
    if p <= 5:
        return 0.45
    return 0.15


def _safe_num(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _going_bucket(going: str, bucket_map: dict[str, list[str]]) -> str:
    g = (going or "").strip().lower()
    for bucket, aliases in bucket_map.items():
        if g == bucket.lower() or any(g == a.lower() for a in aliases):
            return bucket
    return "unknown"


def chelt_form_index(
    chelt_runs: list[dict[str, Any]],
    reference_date: datetime,
    recency_years: int,
    alpha: float,
    beta: float,
) -> tuple[float, int, list[str]]:
    notes: list[str] = []
    prior = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
    if not chelt_runs:
        return _clip01(prior), 0, ["No Cheltenham history"]

    cutoff_days = recency_years * 365
    weighted_success = 0.0
    weighted_total = 0.0
    for run in chelt_runs:
        run_dt = _parse_date(run.get("date"))
        if not run_dt:
            notes.append("Cheltenham run missing date")
            continue
        age_days = max((reference_date - run_dt.replace(tzinfo=None)).days, 0)
        if age_days > cutoff_days:
            continue
        weight = max(0.15, 1.0 - (age_days / cutoff_days))
        pos = run.get("finish_position")
        success = 1.0 if _safe_num(pos, 99) <= 3 else 0.0
        weighted_success += weight * success
        weighted_total += weight

    if weighted_total <= 0:
        return _clip01(prior), 0, ["No recent Cheltenham runs"]

    smoothed = (weighted_success + alpha) / (weighted_total + alpha + beta)
    return _clip01(smoothed), int(round(weighted_total)), notes


def build_category_scores(
    runner: dict[str, Any],
    race: dict[str, Any],
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
    field_ctx: dict[str, Any],
    run_date: str,
) -> dict[str, CategoryScore]:
    feature_cfg = cfg.get("features", {})
    priors = feature_cfg.get("priors", {})
    sat = feature_cfg.get("confidence_saturation", {})
    weights = cfg.get("weights", {})

    race_going = str(race.get("going") or "")
    race_course = str(race.get("track") or race.get("course") or "")
    race_distance = _safe_num(race.get("distance_yards") or race.get("distance") or 0.0)
    going_map = feature_cfg.get("going", {}).get("bucket_map", {})
    target_bucket = _going_bucket(race_going, going_map)

    today = _parse_date(run_date) or datetime.now(timezone.utc).replace(tzinfo=None)

    # Recent form
    n_runs = int(feature_cfg.get("recent_form", {}).get("n_runs", 8))
    half_life = float(feature_cfg.get("recent_form", {}).get("half_life_days", 90))
    recent = history[:n_runs]
    wf = 0.0
    wt = 0.0
    recent_notes: list[str] = []
    for run in recent:
        dt = _parse_date(run.get("date"))
        age_days = max((today - dt.replace(tzinfo=None)).days, 0) if dt else 999
        decay = 0.5 ** (age_days / max(half_life, 1.0))
        wf += decay * _finish_score(run.get("finish_position"))
        wt += decay
        if not dt:
            recent_notes.append("History row missing date")
    recent_raw = wf / wt if wt > 0 else 0.5
    recent_n = len(recent)
    recent_conf = _conf_from_n(recent_n, float(sat.get("recent_form", 6)))

    # Course + distance
    band_pct = float(feature_cfg.get("distance", {}).get("band_pct", 0.10))
    cd_matches = []
    for run in history:
        same_course = str(run.get("track") or run.get("course") or "").lower() == race_course.lower()
        d = _safe_num(run.get("distance_yards") or run.get("distance") or 0.0)
        dist_ok = race_distance > 0 and abs(d - race_distance) <= (race_distance * band_pct)
        if same_course or dist_ok:
            cd_matches.append(run)
    cd_scores = [_finish_score(r.get("finish_position")) for r in cd_matches]
    cd_raw = sum(cd_scores) / len(cd_scores) if cd_scores else 0.5
    cd_n = len(cd_matches)
    cd_conf = _conf_from_n(cd_n, float(sat.get("course_distance", 4)))
    cd_notes = ["First run at course/distance band"] if cd_n == 0 else []

    # Going
    going_matches = []
    for run in history:
        b = _going_bucket(str(run.get("going") or ""), going_map)
        if b == target_bucket and b != "unknown":
            going_matches.append(run)
    g_scores = [_finish_score(r.get("finish_position")) for r in going_matches]
    going_raw = sum(g_scores) / len(g_scores) if g_scores else 0.5
    going_n = len(going_matches)
    going_conf = _conf_from_n(going_n, float(sat.get("going", 4)))
    going_notes = []
    if target_bucket == "unknown":
        going_notes.append("Going unknown/unmapped")
    if going_n == 0:
        going_notes.append("No prior runs in going bucket")

    # Handicap/ratings
    today_or = _safe_num(runner.get("or") or runner.get("official_rating") or 0.0)
    win_marks = [_safe_num(h.get("or") or h.get("official_rating") or 0.0) for h in history if _safe_num(h.get("finish_position"), 99) == 1]
    if today_or > 0 and win_marks:
        best = min(win_marks)
        diff = max(min((best - today_or) / 10.0, 1.0), -1.0)
        hr_raw = _clip01(0.5 + 0.5 * diff)
        hr_n = len(win_marks)
        hr_notes: list[str] = []
    else:
        w = _safe_num(runner.get("weight_lbs") or runner.get("weight") or 0.0)
        field_mean = _safe_num(field_ctx.get("mean_weight") or 0.0)
        if w > 0 and field_mean > 0:
            diff = (field_mean - w) / max(field_mean, 1.0)
            hr_raw = _clip01(0.5 + diff)
            hr_notes = ["Fallback handicap proxy used (weight vs field mean)"]
        else:
            hr_raw = 0.5
            hr_notes = ["No OR/weight data"]
        hr_n = 0
    hr_conf = _conf_from_n(hr_n, float(sat.get("handicap_ratings", 6)))

    # Jockey/trainer impact
    jockey = str(runner.get("jockey") or "")
    trainer = str(runner.get("trainer") or "")
    jockey_runs = [h for h in history if str(h.get("jockey") or "") == jockey and jockey]
    trainer_runs = [h for h in history if str(h.get("trainer") or "") == trainer and trainer]

    def strike(rs: list[dict[str, Any]]) -> float:
        if not rs:
            return 0.5
        wins = sum(1 for r in rs if _safe_num(r.get("finish_position"), 99) == 1)
        return _clip01(wins / len(rs))

    jockey_raw = strike(jockey_runs)
    trainer_raw = strike(trainer_runs)
    jockey_n = len(jockey_runs)
    trainer_n = len(trainer_runs)
    jockey_conf = _conf_from_n(jockey_n, float(sat.get("jockey", 10)))
    trainer_conf = _conf_from_n(trainer_n, float(sat.get("trainer", 12)))
    jockey_notes = ["Sparse jockey sample"] if jockey_n < 3 else []
    trainer_notes = ["Sparse trainer sample"] if trainer_n < 3 else []

    # Cheltenham form
    chelt_runs = [h for h in history if "cheltenham" in str(h.get("track") or h.get("course") or "").lower()]
    chelt_cfg = feature_cfg.get("cheltenham", {})
    chelt_raw, chelt_weighted_n, chelt_notes = chelt_form_index(
        chelt_runs=chelt_runs,
        reference_date=today,
        recency_years=int(chelt_cfg.get("recency_years", 3)),
        alpha=float(chelt_cfg.get("shrinkage_alpha", 2.0)),
        beta=float(chelt_cfg.get("shrinkage_beta", 3.0)),
    )
    chelt_conf = _conf_from_n(chelt_weighted_n, float(sat.get("cheltenham_form", 4)))

    values: dict[str, tuple[float, float, int, list[str]]] = {
        "recent_form": (recent_raw, recent_conf, recent_n, recent_notes),
        "course_distance": (cd_raw, cd_conf, cd_n, cd_notes),
        "going": (going_raw, going_conf, going_n, going_notes),
        "handicap_ratings": (hr_raw, hr_conf, hr_n, hr_notes),
        "jockey": (jockey_raw, jockey_conf, jockey_n, jockey_notes),
        "trainer": (trainer_raw, trainer_conf, trainer_n, trainer_notes),
        "cheltenham_form": (chelt_raw, chelt_conf, chelt_weighted_n, chelt_notes),
    }

    out: dict[str, CategoryScore] = {}
    for name, (raw, conf, samples, notes) in values.items():
        prior = float(priors.get(name, 0.5))
        adjusted = _clip01(raw * conf + prior * (1.0 - conf))
        weight = float(weights.get(name, 1.0))
        contribution = adjusted * weight
        out[name] = CategoryScore(
            raw=_clip01(raw),
            confidence=_clip01(conf),
            adjusted=adjusted,
            weight=weight,
            contribution=contribution,
            samples=int(samples),
            notes=notes,
        )
    return out
