from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx


class RacingAPIClient:
    def __init__(self, api_cfg: dict[str, Any]) -> None:
        self.base_url = api_cfg.get("base_url", "").rstrip("/")
        self.country = api_cfg.get("country", "UK")
        self.timeout_seconds = float(api_cfg.get("timeout_seconds", 20))
        self.retries = int(api_cfg.get("retries", 3))
        self.backoff_seconds = float(api_cfg.get("backoff_seconds", 0.6))
        self.auth_mode = str(api_cfg.get("auth_mode", "basic")).lower()
        self.auth_header = api_cfg.get("auth_header", "X-API-Key")
        self.endpoints = api_cfg.get("endpoints", {})
        self.api_key = os.getenv("RACING_API_KEY", "")
        self.api_username = os.getenv("RACING_API_USERNAME", "")
        self.api_password = os.getenv("RACING_API_PASSWORD", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.auth_mode == "apikey" and self.api_key:
            headers[self.auth_header] = self.api_key
        return headers

    def _auth(self) -> httpx.BasicAuth | None:
        if self.auth_mode == "basic" and self.api_username and self.api_password:
            return httpx.BasicAuth(self.api_username, self.api_password)
        return None

    def _url(self, endpoint_key: str, **kwargs: Any) -> str:
        template = self.endpoints.get(endpoint_key, "")
        return f"{self.base_url}{template.format(**kwargs)}"

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.get(
                        url,
                        headers=self._headers(),
                        params=params,
                        auth=self._auth(),
                    )
                    response.raise_for_status()
                    return response.json()
            except Exception as exc:  # fail-soft by caller
                last_error = exc
                if attempt >= self.retries:
                    break
                sleep_s = self.backoff_seconds * (2**attempt) + random.uniform(0, 0.2)
                time.sleep(sleep_s)
        raise RuntimeError(f"API request failed: {url}; error={last_error}")

    def get_racecards(self, date: str, country: str = "UK") -> list[dict[str, Any]]:
        url = self._url("racecards")
        payload = self._request_json(url, params={"date": date, "country": country})
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("races") or payload.get("data") or []
        return []

    def get_race(self, race_id: str) -> dict[str, Any]:
        url = self._url("race", race_id=race_id)
        payload = self._request_json(url)
        return payload if isinstance(payload, dict) else {}

    def get_runner_history(self, horse_id: str, lookback_days: int = 1825) -> list[dict[str, Any]]:
        url = self._url("runner_history", horse_id=horse_id)
        payload = self._request_json(url, params={"lookback_days": lookback_days})
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("history") or payload.get("runs") or payload.get("data") or []
        return []

    def get_odds(self, race_id: str) -> dict[str, dict[str, Any]]:
        url = self._url("odds", race_id=race_id)
        payload = self._request_json(url)
        if isinstance(payload, dict):
            rows = payload.get("odds") or payload.get("runners") or payload.get("data") or []
        else:
            rows = payload
        result: dict[str, dict[str, Any]] = {}
        if not isinstance(rows, list):
            return result
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("runner_id") or row.get("id") or row.get("horse_id") or "")
            if not rid:
                continue
            result[rid] = row
        return result

    def get_results(self, date: str, country: str = "UK") -> list[dict[str, Any]]:
        url = self._url("results")
        payload = self._request_json(url, params={"date": date, "country": country})
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("results") or payload.get("data") or []
        return []
