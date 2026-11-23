"""Phase 10 - final master dataset: USDA fields + location + imagery estimates.

The USDA/WCMD columns are copied unchanged from the Phase 1 master table
(which itself keeps every raw value in raw_* columns). Nothing here edits them.

Output: outputs/master_facility_dataset.csv  (+ .xlsx if openpyxl is installed)

Usage:  python src/build_master.py [--all-records]
        default = USWA records only; --all-records includes state/unlicensed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, path  # noqa: E402

log = get_logger("build_master")

USDA = ["facility_id", "state", "county", "city", "ccc_code", "ccc_status", "license_number", "license_type",
        "license_status", "business_entity", "warehouse_name", "functional_unit", "warehouse_capacity",
        "is_uswa", "pdf_page", "data_quality_flags"]
LOCATION = ["street_address", "city_verified", "state_verified", "zip_code", "latitude", "longitude",
            "address_source", "address_confidence", "geocode_confidence", "coord_source", "clip_ok"]
ESTIMATE = ["earliest_imagery_year", "last_year_absent", "first_year_present", "estimated_construction_start",
            "estimated_build_window", "estimated_first_visible_year", "build_year_confidence",
            "latest_imagery_year", "possible_last_year_present", "possible_disappearance_year"]


def manual_review_status(r: pd.Series) -> str:
    if r.get("address_confidence", "") in ("", "UNRESOLVED", "NEEDS_REVIEW", "LOW", "ERROR"):
        return "LOCATION_PENDING"
    if r.get("clip_ok", "") != "True":
        return "LOCATION_PENDING"
    if r.get("n_images", 0) == 0:
        return "IMAGERY_PENDING"
    if r.get("build_year_confidence", "") in ("", "NOT_REVIEWED"):
        return "REVIEW_PENDING"
    return "REVIEWED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-records", action="store_true")
    a = ap.parse_args()
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)
    if not a.all_records:
        master = master[master["is_uswa"] == "True"]
    df = master[USDA].copy()

    f = path("outputs_dir") / "facility_locations.csv"
    if f.exists():
        loc = pd.read_csv(f, dtype=str, keep_default_na=False)
        df = df.merge(loc[["facility_id"] + [c for c in LOCATION if c in loc.columns]], on="facility_id", how="left")
    f = path("imagery_dir") / "manifest.csv"
    if f.exists():
        man = pd.read_csv(f, dtype=str, keep_default_na=False)
        man = man[(man["status"] == "ok") & (man["image_available"] == "True")]
        agg = man.groupby("facility_id").agg(n_images=("year", "size"),
                                             imagery_source=("sensor", lambda s: ";".join(sorted(set(s)))))
        df = df.merge(agg, on="facility_id", how="left")
    f = path("outputs_dir") / "estimated_build_years.csv"
    if f.exists():
        est = pd.read_csv(f, dtype=str, keep_default_na=False)
        keep = ["facility_id"] + [c for c in ESTIMATE + ["flags", "review_notes"] if c in est.columns]
        df = df.merge(est[keep], on="facility_id", how="left")

    # Phase 8 machine suggestions - separate columns, never mixed into the reviewed estimates.
    f = path("outputs_dir") / "ml_first_present.csv"
    if f.exists():
        ml = pd.read_csv(f, dtype=str, keep_default_na=False)
        ml = ml[["facility_id", "ml_first_present", "ml_window", "ml_confidence", "ml_flags"]].rename(
            columns={"ml_first_present": "ml_suggested_first_present", "ml_window": "ml_suggested_window",
                     "ml_confidence": "ml_suggestion_confidence"})
        df = df.merge(ml, on="facility_id", how="left")

    for c in LOCATION + ESTIMATE + ["imagery_source", "flags", "review_notes"]:
        if c not in df.columns:
            df[c] = ""
    df["n_images"] = (pd.to_numeric(df["n_images"], errors="coerce").fillna(0).astype(int)
                      if "n_images" in df.columns else 0)
    df = df.fillna("")
    df["manual_review_status"] = df.apply(manual_review_status, axis=1)
    df["research_notes"] = (df["flags"].astype(str) + " " + df["review_notes"].astype(str)).str.strip()
    df = df.rename(columns={"build_year_confidence": "build_year_confidence"})
    df = df.drop(columns=["flags", "review_notes"])

    out = path("outputs_dir") / "master_facility_dataset.csv"
    df.to_csv(out, index=False)
    try:
        df.to_excel(out.with_suffix(".xlsx"), index=False)
    except ImportError:
        pass
    log.info("wrote %s (%d rows); status: %s", out, len(df), df["manual_review_status"].value_counts().to_dict())


if __name__ == "__main__":
    main()
