from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(output: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")


def _fmt_pct(x: float) -> str:
    return f"{100*x:.1f}%"


def build_markdown_report(output: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Value Finder Report ({output.get('date','')})")
    lines.append("")
    lines.append(f"Calibration: **{output.get('calibrated_status','uncalibrated')}**")
    lines.append("")

    lines.append("## Top Value Bets")
    lines.append("| Horse | Race | Odds | EV | Edge | Model Win | Market Win |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for row in output.get("top_value_bets", [])[:10]:
        lines.append(
            f"| {row['horse']} | {row['track']} {row['off_time']} | {row['odds_decimal']:.2f} | {row['ev_win']:.3f} | {row['edge']:.3f} | {_fmt_pct(row['p_model_win'])} | {_fmt_pct(row['p_mkt_norm'])} |"
        )
    lines.append("")

    lines.append("## Safest Overlays")
    lines.append("| Horse | Race | EV | Model Win | Edge |")
    lines.append("|---|---|---:|---:|---:|")
    for row in output.get("safest_overlays", [])[:10]:
        lines.append(
            f"| {row['horse']} | {row['track']} {row['off_time']} | {row['ev_win']:.3f} | {_fmt_pct(row['p_model_win'])} | {row['edge']:.3f} |"
        )
    lines.append("")

    lines.append("## NAP by Value")
    nap = output.get("nap_by_value")
    if nap:
        lines.append(
            f"**{nap['horse']}** ({nap['track']} {nap['off_time']}) - Odds {nap['odds_decimal']:.2f}, EV {nap['ev_win']:.3f}, Edge {nap['edge']:.3f}"
        )
    else:
        lines.append("No qualifying NAP under current guardrails.")
    lines.append("")

    lines.append("## Races")
    for race in output.get("races", []):
        lines.append(f"### {race.get('track','')} {race.get('off_time','')} ({race.get('race_id','')})")
        lines.append(
            f"Going: {race.get('going','')} | Field: {race.get('field_size',0)} | Places K: {race.get('places_k',0)}"
        )
        lines.append("")
        lines.append("| Horse | Odds | p_model_win | p_mkt | Edge | EV | Conf |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in race.get("runners", []):
            odds = r.get("odds_decimal")
            odds_s = f"{odds:.2f}" if isinstance(odds, (int, float)) else "-"
            lines.append(
                f"| {r.get('horse','')} | {odds_s} | {_fmt_pct(float(r.get('p_model_win',0)))} | {_fmt_pct(float(r.get('p_mkt_norm',0)))} | {float(r.get('edge',0)):.3f} | {float(r.get('ev_win',0)):.3f} | {_fmt_pct(float(r.get('confidence_overall',0)))} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_markdown_report(output: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(build_markdown_report(output), encoding="utf-8")
