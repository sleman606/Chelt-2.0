from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from value_finder.api_client import RacingAPIClient
from value_finder.cache import FileCache
from value_finder.config import AppConfig, load_config
from value_finder.report import write_json_report, write_markdown_report
from value_finder.storage import Storage
from value_finder.value import day_output, evaluate_race


@dataclass
class CalibrationModel:
    method: str
    transform: Callable[[float], float]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def load_calibration_model(cfg: AppConfig) -> CalibrationModel | None:
    path = cfg.calibration.get("artifact_path")
    if not path or not Path(path).exists():
        return None

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    method = payload.get("method")
    if method == "platt":
        coef = float(payload.get("coef", 0.0))
        intercept = float(payload.get("intercept", 0.0))

        def transform(p: float) -> float:
            p = min(max(p, 1e-6), 1 - 1e-6)
            logit = math.log(p / (1 - p))
            return _sigmoid(coef * logit + intercept)

        return CalibrationModel(method=method, transform=transform)

    if method == "isotonic":
        x = payload.get("x", [])
        y = payload.get("y", [])
        model = IsotonicRegression(out_of_bounds="clip")
        model.X_thresholds_ = x
        model.y_thresholds_ = y

        def transform(p: float) -> float:
            return float(model.predict([p])[0])

        return CalibrationModel(method=method, transform=transform)

    return None


def _extract_race_id(race: dict[str, Any]) -> str:
    return str(race.get("race_id") or race.get("id") or "")


def _extract_runner_id(runner: dict[str, Any]) -> str:
    return str(runner.get("runner_id") or runner.get("id") or runner.get("horse_id") or "")


def _extract_horse_id(runner: dict[str, Any]) -> str:
    return str(runner.get("horse_id") or runner.get("id") or runner.get("runner_id") or "")


def build_day_output(
    date: str,
    cfg: AppConfig,
    client: RacingAPIClient,
    cache: FileCache,
    storage: Storage,
) -> dict[str, Any]:
    model = load_calibration_model(cfg)
    calibrator = model.transform if model else None

    racecards = client.get_racecards(date, country=cfg.api.get("country", "UK"))
    races_eval = []

    for race_row in racecards:
        race_id = _extract_race_id(race_row)
        if not race_id:
            continue

        race_detail = client.get_race(race_id)
        race = race_detail if race_detail else race_row
        odds = client.get_odds(race_id)

        histories: dict[str, list[dict[str, Any]]] = {}
        for runner in race.get("runners", race.get("entries", [])):
            rid = _extract_runner_id(runner)
            horse_id = _extract_horse_id(runner)
            if not rid:
                continue
            cache_key = f"history:{horse_id}:1825"
            cached = cache.get(cache_key, ttl_seconds=3600)
            if cached is None:
                try:
                    hist = client.get_runner_history(horse_id, lookback_days=1825) if horse_id else []
                except Exception:
                    hist = []
                cache.set(cache_key, hist)
                histories[rid] = hist
            else:
                histories[rid] = cached

        race_eval = evaluate_race(
            date=date,
            race=race,
            odds_map=odds,
            histories=histories,
            cfg=cfg.raw,
            calibrator=calibrator,
        )
        races_eval.append(race_eval)

    status = "calibrated" if model else "uncalibrated"
    output = day_output(date=date, races=races_eval, cfg=cfg.raw, calibrated_status=status)
    storage.save_day_snapshot(output)
    return output


def run_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    client = RacingAPIClient(cfg.api)
    cache = FileCache()
    storage = Storage(cfg.raw)

    try:
        if args.refresh_minutes:
            while True:
                out = build_day_output(args.date, cfg, client, cache, storage)
                out_dir = Path(cfg.reporting.get("output_dir", "outputs"))
                write_json_report(out, str(out_dir / cfg.reporting.get("json_name", "value_report.json")))
                write_markdown_report(out, str(out_dir / cfg.reporting.get("markdown_name", "value_report.md")))

                now = datetime.utcnow()
                if args.date != now.strftime("%Y-%m-%d"):
                    break
                time.sleep(int(args.refresh_minutes) * 60)
        else:
            out = build_day_output(args.date, cfg, client, cache, storage)
            out_dir = Path(cfg.reporting.get("output_dir", "outputs"))
            write_json_report(out, str(out_dir / cfg.reporting.get("json_name", "value_report.json")))
            write_markdown_report(out, str(out_dir / cfg.reporting.get("markdown_name", "value_report.md")))
    finally:
        storage.close()


def _fit_platt(x_train: list[float], y_train: list[int]) -> dict[str, Any]:
    X = [[math.log(min(max(p, 1e-6), 1 - 1e-6) / (1 - min(max(p, 1e-6), 1 - 1e-6)))] for p in x_train]
    model = LogisticRegression()
    model.fit(X, y_train)
    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return {"method": "platt", "coef": coef, "intercept": intercept}


def _apply_platt(payload: dict[str, Any], x: list[float]) -> list[float]:
    coef = float(payload["coef"])
    intercept = float(payload["intercept"])
    out = []
    for p in x:
        p = min(max(p, 1e-6), 1 - 1e-6)
        logit = math.log(p / (1 - p))
        out.append(_sigmoid(coef * logit + intercept))
    return out


def _fit_isotonic(x_train: list[float], y_train: list[int]) -> dict[str, Any]:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x_train, y_train)
    return {
        "method": "isotonic",
        "x": model.X_thresholds_.tolist(),
        "y": model.y_thresholds_.tolist(),
    }


def _apply_isotonic(payload: dict[str, Any], x: list[float]) -> list[float]:
    model = IsotonicRegression(out_of_bounds="clip")
    model.X_thresholds_ = payload["x"]
    model.y_thresholds_ = payload["y"]
    return [float(v) for v in model.predict(x)]


def _log_loss(y_true: list[int], y_prob: list[float]) -> float:
    eps = 1e-12
    vals = []
    for y, p in zip(y_true, y_prob):
        p = min(max(p, eps), 1 - eps)
        vals.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return sum(vals) / len(vals) if vals else 0.0


def calibrate_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    storage = Storage(cfg.raw)
    try:
        rows = storage.fetch_calibration_dataset(args.start_date, args.end_date)
        if len(rows) < int(cfg.calibration.get("min_samples", 100)):
            raise RuntimeError(f"Not enough samples for calibration: {len(rows)}")

        probs = [float(r["p_model_win"]) for r in rows]
        labels = [int(r["did_win"]) for r in rows]

        split_idx = int((1 - float(cfg.calibration.get("holdout_ratio", 0.2))) * len(rows))
        x_train, y_train = probs[:split_idx], labels[:split_idx]
        x_hold, y_hold = probs[split_idx:], labels[split_idx:]

        method = args.method or cfg.calibration.get("default_method", "platt")
        candidates: dict[str, dict[str, Any]] = {}
        if method in ("platt", "auto"):
            p = _fit_platt(x_train, y_train)
            candidates["platt"] = p
        if method in ("isotonic", "auto"):
            i = _fit_isotonic(x_train, y_train)
            candidates["isotonic"] = i

        scored = []
        for name, payload in candidates.items():
            if name == "platt":
                ph = _apply_platt(payload, x_hold)
            else:
                ph = _apply_isotonic(payload, x_hold)
            scored.append((name, payload, _log_loss(y_hold, ph)))

        best_name, best_payload, best_loss = sorted(scored, key=lambda x: x[2])[0]
        artifact_path = Path(cfg.calibration.get("artifact_path", "artifacts/calibration_win.json"))
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

        metrics = storage.calibration_metrics(args.start_date, args.end_date)
        metrics["selected_method"] = best_name
        metrics["holdout_log_loss"] = best_loss
        print(json.dumps(metrics, indent=2))
    finally:
        storage.close()


def ingest_results_command(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    client = RacingAPIClient(cfg.api)
    storage = Storage(cfg.raw)
    try:
        rows = client.get_results(args.date, country=cfg.api.get("country", "UK"))
        storage.upsert_results(rows, args.date)
        metrics = storage.calibration_metrics(args.date, args.date)
        out_dir = Path(cfg.reporting.get("output_dir", "outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        storage.export_bins_csv(metrics.get("calibration_bins", []), str(out_dir / "calibration_bins.csv"))
        print(json.dumps(metrics, indent=2))
    finally:
        storage.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="value_finder")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run")
    run_p.add_argument("--date", required=True)
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--refresh-minutes", type=int, default=0)

    cal_p = sub.add_parser("calibrate")
    cal_p.add_argument("--from", dest="start_date", required=True)
    cal_p.add_argument("--to", dest="end_date", required=True)
    cal_p.add_argument("--config", required=True)
    cal_p.add_argument("--method", choices=["platt", "isotonic", "auto"], default=None)

    ingest_p = sub.add_parser("ingest-results")
    ingest_p.add_argument("--date", required=True)
    ingest_p.add_argument("--config", required=True)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        run_command(args)
    elif args.command == "calibrate":
        calibrate_command(args)
    elif args.command == "ingest-results":
        ingest_results_command(args)


if __name__ == "__main__":
    main()
