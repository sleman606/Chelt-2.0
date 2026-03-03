from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class AppConfig:
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def api(self) -> dict[str, Any]:
        return self.raw.get("api", {})

    @property
    def features(self) -> dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def weights(self) -> dict[str, float]:
        return self.raw.get("weights", {})

    @property
    def probability(self) -> dict[str, Any]:
        return self.raw.get("probability", {})

    @property
    def filters(self) -> dict[str, Any]:
        return self.raw.get("filters", {})

    @property
    def place_rules(self) -> dict[str, Any]:
        return self.raw.get("place_rules", {})

    @property
    def storage(self) -> dict[str, Any]:
        return self.raw.get("storage", {})

    @property
    def calibration(self) -> dict[str, Any]:
        return self.raw.get("calibration", {})

    @property
    def reporting(self) -> dict[str, Any]:
        return self.raw.get("reporting", {})

    @property
    def refresh(self) -> dict[str, Any]:
        return self.raw.get("refresh", {})


def load_config(path: str) -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    default_path = cfg_path.parent / "config.example.yaml"
    default_data: dict[str, Any] = {}
    if default_path.exists() and default_path != cfg_path:
        default_data = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}

    user_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    merged = _deep_merge(default_data, user_data)

    storage_path = merged.get("storage", {}).get("sqlite_path")
    if storage_path:
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
    output_dir = merged.get("reporting", {}).get("output_dir")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    artifact_path = merged.get("calibration", {}).get("artifact_path")
    if artifact_path:
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)

    return AppConfig(raw=merged)
