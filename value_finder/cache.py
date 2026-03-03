from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class FileCache:
    def __init__(self, root: str = ".cache/value_finder") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def get(self, key: str, ttl_seconds: int | None = None) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if ttl_seconds is not None:
            age = time.time() - path.stat().st_mtime
            if age > ttl_seconds:
                return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")
