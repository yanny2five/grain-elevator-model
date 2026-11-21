"""Phase 9 - compare imagery-based estimates with independently known dates.

Input : data/validation/known_dates.csv  (documented events with sources)
        outputs/estimated_build_years.csv
Output: outputs/validation_report.csv

How events are compared
-----------------------
construction_completed / founded_built / construction_expected_complete
    -> should fall inside (or within 1 year of) the estimated build window.
state_license_effective / ownership_change
    -> an UPPER BOUND on existence: the structure should be visible on or
       before that year (first_visible <= event year + 1 tolerates late imagery).
       A licence date is never treated as a construction date.

Each comparison is CONSISTENT, INCONSISTENT, or NOT_COMPARABLE (no estimate yet,
or the facility was present at the start of the record).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("validate")
BUILD_EVENTS = {"construction_completed", "founded_built", "construction_expected_complete"}
BOUND_EVENTS = {"state_license_effective", "ownership_change"}


def window(s: str) -> tuple[int | None, int | None]:
    s = str(s).strip()
    if m := re.fullmatch(r"<=\s*(\d{4})", s):
        return None, int(m.group(1))
    if m := re.fullmatch(r"(\d{4})-(\d{4})", s):
        return int(m.group(1)), int(m.group(2))
    if re.fullmatch(r"\d{4}", s):
        return int(s), int(s)
    return None, None


def main() -> None:
    kd = pd.read_csv(PROJECT_ROOT / "data" / "validation" / "known_dates.csv", dtype=str, keep_default_na=False)
    est = pd.read_csv(path("outputs_dir") / "estimated_build_years.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    rows = []
    for _, k in kd.iterrows():
        r = {**k.to_dict(), "estimated_build_window": "", "estimated_first_visible_year": "", "result": "NOT_COMPARABLE", "detail": ""}
        if k["facility_id"] not in est.index or not est.loc[k["facility_id"], "estimated_first_visible_year"]:
            r["detail"] = "no imagery estimate yet"
            rows.append(r)
            continue
        e = est.loc[k["facility_id"]]
        lo, hi = window(e["estimated_build_window"])
        fv = int(float(e["estimated_first_visible_year"]))
        y = int(k["year"])
        r.update(estimated_build_window=e["estimated_build_window"], estimated_first_visible_year=fv,
                 build_year_confidence=e["build_year_confidence"])
        if k["event"] in BUILD_EVENTS:
            if lo is None:
                r["result"] = "CONSISTENT" if y >= (hi or fv) - 1 else "NOT_COMPARABLE"
                r["detail"] = "facility present at start of record; event can only be checked as an upper bound"
            else:
                ok = lo - 1 <= y <= hi + 1
                r["result"] = "CONSISTENT" if ok else "INCONSISTENT"
                r["detail"] = f"event {y} vs window {lo}-{hi} (+/-1 yr tolerance); error = {0 if lo <= y <= hi else min(abs(y - lo), abs(y - hi))} yr"
        elif k["event"] in BOUND_EVENTS:
            if lo is None:
                # Present in the first usable image: "built on or before <hi>" cannot contradict
                # "existed by <y>" - the imagery simply starts too late to test it.
                r["result"] = "CONSISTENT"
                r["detail"] = f"present in first image ({hi}); record says it existed by {y} - imagery starts too late to test further"
            else:
                ok = fv <= y + 1
                r["result"] = "CONSISTENT" if ok else "INCONSISTENT"
                r["detail"] = f"first visible {fv} should be <= {y} (upper bound on existence)"
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(path("outputs_dir") / "validation_report.csv", index=False)
    log.info("validation: %s", out["result"].value_counts().to_dict())


if __name__ == "__main__":
    main()
