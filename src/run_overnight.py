"""Unattended end-to-end run: locate -> geocode -> imagery -> review pages -> estimates.

Runs every stage in order as separate processes, keeps going if one stage
fails, retries the long network stages, and writes a human-readable progress
file (logs/overnight_status.md) after every step.

While it runs it asks Windows not to go to sleep (SetThreadExecutionState -
the same request video players make). This is not a settings change: it ends
automatically when this process exits. Closing a laptop lid can still force
sleep; if the machine sleeps anyway, rerun this script - every stage is
resumable and skips finished work.

Usage (detached, survives closing the terminal):
  Start-Process -WindowStyle Hidden python src/run_overnight.py
"""
from __future__ import annotations

import ctypes
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "logs" / "overnight_status.md"
PY = sys.executable
LINES: list[str] = []


def keep_awake(on: bool) -> None:
    if sys.platform != "win32":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def note(msg: str) -> None:
    LINES.append(f"- {dt.datetime.now():%Y-%m-%d %H:%M} {msg}")
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text("# Overnight run status\n\n" + "\n".join(LINES) + "\n", encoding="utf-8")


def run(name: str, args: list[str], tries: int = 1, log: str | None = None) -> bool:
    for attempt in range(1, tries + 1):
        note(f"START {name} (attempt {attempt}/{tries})")
        out = open(ROOT / "logs" / (log or f"overnight_{name}.out"), "a", encoding="utf-8")
        rc = subprocess.call([PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT,
                             env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        out.close()
        if rc == 0:
            note(f"DONE  {name}")
            return True
        note(f"FAIL  {name} (exit {rc})")
        time.sleep(60 * attempt)
    return False


def summary() -> None:
    import pandas as pd
    try:
        loc = pd.read_csv(ROOT / "outputs" / "facility_locations.csv", dtype=str, keep_default_na=False)
        man = pd.read_csv(ROOT / "data" / "imagery" / "manifest.csv", dtype=str, keep_default_na=False)
        ok = man[(man["status"] == "ok") & (man["image_available"] == "True")]
        note(f"SUMMARY located (clip_ok): {(loc['clip_ok'] == 'True').sum()} | address confidence: "
             f"{loc['address_confidence'].value_counts().to_dict()} | facilities with imagery: "
             f"{ok['facility_id'].nunique()} | images: {len(ok)}")
    except Exception as e:  # noqa: BLE001
        note(f"summary unavailable: {e}")


def main() -> None:
    keep_awake(True)
    note("overnight run started")
    try:
        run("company_locations", ["src/company_locations.py"], tries=3)
        run("merge", ["src/find_addresses.py", "--merge-only"])
        run("geocode", ["src/geocode.py"], tries=2)
        run("master_1", ["src/build_master.py"])
        summary()
        # Imagery: several passes; each pass skips finished images and retries failed ones.
        for p in range(1, 4):
            run(f"imagery_pass{p}", ["src/export_yearly_imagery.py", "--subset", "all", "--workers", "6"],
                log="overnight_imagery.out")
        run("review_pages", ["src/review_facilities.py"])
        run("estimates", ["src/estimate_build_year.py"])
        run("validate", ["src/validate.py"])
        run("master_2", ["src/build_master.py"])
        summary()
        note("overnight run finished")
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
