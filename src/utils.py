"""Shared helpers: config loading, project paths, logging, JSON caches and checkpoints.

Every pipeline stage imports from here so that paths, logging format and
resumability behave the same way everywhere.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import yaml

try:  # Use the OS certificate store (fixes TLS errors behind antivirus/corporate proxies).
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a path from config['paths'] to an absolute path."""
    return PROJECT_ROOT / load_config()["paths"][key]


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to stdout and logs/<name>.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    logs_dir = path("logs_dir")
    logs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class JsonCache:
    """Tiny persistent key->value cache (one JSON file).

    Used for geocoding / web lookups so that an interrupted run never repeats
    a completed API call. Writes atomically after every `set`.
    """

    def __init__(self, file: Path):
        self.file = Path(file)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        # OneDrive / antivirus can briefly lock the target file; retry instead of crashing.
        for attempt in range(10):
            try:
                os.replace(tmp, self.file)
                return
            except PermissionError:
                time.sleep(0.5 * (attempt + 1))
        raise PermissionError(f"could not update cache {self.file} (file locked)")


class RateLimiter:
    """Enforce a minimum interval between calls to an external API."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval_s:
            time.sleep(self.min_interval_s - dt)
        self._last = time.monotonic()
