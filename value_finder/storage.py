from __future__ import annotations

import csv
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable


def _log_loss(y_true: list[int], y_prob: list[float]) -> float:
    eps = 1e-12
    terms = []
    for y, p in zip(y_true, y_prob):
        p = min(max(p, eps), 1 - eps)
        terms.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    return sum(terms) / len(terms) if terms else 0.0


def _brier(y_true: list[int], y_prob: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum((y - p) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


class Storage:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.backend = cfg.get("storage", {}).get("backend", "sqlite")
        self.sqlite_path = cfg.get("storage", {}).get("sqlite_path", "artifacts/value_finder.sqlite3")
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS runner_snapshots (
              date TEXT,
              race_id TEXT,
              runner_id TEXT,
              horse TEXT,
              track TEXT,
              off_time TEXT,
              odds_decimal REAL,
              p_mkt_norm REAL,
              p_model_win REAL,
              p_model_place REAL,
              edge REAL,
              ev_win REAL,
              confidence_overall REAL,
              score_total REAL,
              calibrated_flag INTEGER,
              feature_json TEXT,
              notes_json TEXT,
              created_at TEXT,
              PRIMARY KEY(date, race_id, runner_id)
            );

            CREATE TABLE IF NOT EXISTS race_meta (
              race_id TEXT,
              date TEXT,
              going TEXT,
              field_size INTEGER,
              places_k INTEGER,
              raw_payload_json TEXT,
              PRIMARY KEY(date, race_id)
            );

            CREATE TABLE IF NOT EXISTS results (
              date TEXT,
              race_id TEXT,
              runner_id TEXT,
              finish_position INTEGER,
              did_win INTEGER,
              did_place INTEGER,
              ingested_at TEXT,
              PRIMARY KEY(date, race_id, runner_id)
            );

            CREATE TABLE IF NOT EXISTS calibration_runs (
              run_id TEXT PRIMARY KEY,
              method TEXT,
              train_from TEXT,
              train_to TEXT,
              holdout_from TEXT,
              holdout_to TEXT,
              metrics_json TEXT,
              artifact_path TEXT
            );
            """
        )
        self.conn.commit()

    def save_day_snapshot(self, day_output: dict[str, Any]) -> None:
        cur = self.conn.cursor()
        date = day_output.get("date")
        calibrated = 1 if day_output.get("calibrated_status") == "calibrated" else 0

        for race in day_output.get("races", []):
            cur.execute(
                """
                INSERT OR REPLACE INTO race_meta
                (race_id, date, going, field_size, places_k, raw_payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    race.get("race_id"),
                    date,
                    race.get("going"),
                    race.get("field_size"),
                    race.get("places_k"),
                    json.dumps(race, ensure_ascii=True),
                ),
            )

            for r in race.get("runners", []):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO runner_snapshots
                    (date, race_id, runner_id, horse, track, off_time, odds_decimal, p_mkt_norm,
                     p_model_win, p_model_place, edge, ev_win, confidence_overall, score_total,
                     calibrated_flag, feature_json, notes_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        date,
                        race.get("race_id"),
                        r.get("runner_id"),
                        r.get("horse"),
                        race.get("track"),
                        race.get("off_time"),
                        r.get("odds_decimal"),
                        r.get("p_mkt_norm"),
                        r.get("p_model_win"),
                        r.get("p_model_place"),
                        r.get("edge"),
                        r.get("ev_win"),
                        r.get("confidence_overall"),
                        r.get("score"),
                        calibrated,
                        json.dumps(r.get("score_breakdown", {}), ensure_ascii=True),
                        json.dumps(r.get("notes", []), ensure_ascii=True),
                    ),
                )
        self.conn.commit()

    def upsert_results(self, rows: Iterable[dict[str, Any]], date: str) -> None:
        cur = self.conn.cursor()
        for row in rows:
            race_id = str(row.get("race_id") or row.get("id") or "")
            for rr in row.get("runners", row.get("results", [])):
                rid = str(rr.get("runner_id") or rr.get("id") or rr.get("horse_id") or "")
                pos = rr.get("finish_position")
                try:
                    pos_i = int(pos)
                except Exception:
                    pos_i = None
                did_win = 1 if pos_i == 1 else 0
                did_place = 1 if pos_i is not None and pos_i <= 3 else 0
                cur.execute(
                    """
                    INSERT OR REPLACE INTO results
                    (date, race_id, runner_id, finish_position, did_win, did_place, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (date, race_id, rid, pos_i, did_win, did_place),
                )
        self.conn.commit()

    def fetch_calibration_dataset(self, start_date: str, end_date: str) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT s.date, s.race_id, s.runner_id, s.p_model_win, s.odds_decimal, r.did_win
            FROM runner_snapshots s
            JOIN results r
              ON s.date = r.date AND s.race_id = r.race_id AND s.runner_id = r.runner_id
            WHERE s.date >= ? AND s.date <= ?
            ORDER BY s.date, s.race_id, s.runner_id
            """,
            (start_date, end_date),
        )
        return cur.fetchall()

    def calibration_metrics(self, start_date: str, end_date: str) -> dict[str, Any]:
        rows = self.fetch_calibration_dataset(start_date, end_date)
        y = [int(r["did_win"]) for r in rows]
        p = [float(r["p_model_win"]) for r in rows]

        stakes = [1.0] * len(rows)
        returns = []
        for r, prob, stake in zip(rows, p, stakes):
            odds = float(r["odds_decimal"] or 0.0)
            win = int(r["did_win"])
            ret = stake * (odds - 1.0) if win else -stake
            returns.append(ret)

        total_staked = sum(stakes) or 1.0
        roi_flat = sum(returns) / total_staked

        kelly_returns = []
        for r, prob in zip(rows, p):
            odds = float(r["odds_decimal"] or 0.0)
            b = max(odds - 1.0, 1e-6)
            q = 1.0 - prob
            f_star = max((b * prob - q) / b, 0.0)
            stake = min(f_star, 0.25)
            win = int(r["did_win"])
            ret = stake * b if win else -stake
            kelly_returns.append(ret)

        bins = []
        for i in range(10):
            lo = i / 10
            hi = (i + 1) / 10
            idx = [j for j, val in enumerate(p) if lo <= val < hi or (i == 9 and val == 1.0)]
            if not idx:
                continue
            avg_p = sum(p[j] for j in idx) / len(idx)
            avg_y = sum(y[j] for j in idx) / len(idx)
            bins.append({"bin": f"{lo:.1f}-{hi:.1f}", "avg_p": avg_p, "avg_y": avg_y, "n": len(idx)})

        return {
            "n": len(rows),
            "roi_flat": roi_flat,
            "roi_kelly": sum(kelly_returns),
            "brier": _brier(y, p),
            "log_loss": _log_loss(y, p),
            "calibration_bins": bins,
        }

    def export_bins_csv(self, bins: list[dict[str, Any]], path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["bin", "avg_p", "avg_y", "n"])
            writer.writeheader()
            writer.writerows(bins)

    def close(self) -> None:
        self.conn.close()
