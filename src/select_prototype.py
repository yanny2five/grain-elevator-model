"""Select the ~10 prototype facilities used to validate the whole pipeline.

The list is hand-curated (not random) so that the prototype covers the cases
the method must handle. The rationale for each pick is stored in the output
so the choice is documented and reproducible.

Usage:  python src/select_prototype.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, path  # noqa: E402

log = get_logger("select_prototype")

PROTOTYPE = {
    "F04448": "Large export terminal (EGT Longview, WA) with a publicly reported opening in 2012 - "
              "best candidate for validating imagery estimate vs. a known date.",
    "F02951": "Shuttle-loader country elevator (EGT Montana, Chester MT) from the same company; "
              "likely 2010s build, tests a mid-size rural site.",
    "F03205": "License PENDING (Walsh Grain Terminal, Park River ND) - probably a very recent build; "
              "tests Sentinel-2/NAIP era and that non-Issued licenses are kept.",
    "F03212": "License PENDING (Dakota Midland Grain, Surrey ND) - second recent-build test case.",
    "F01893": "Large shuttle elevator (WindRiver Garden City, KS) on the High Plains - clear "
              "background, tests construction-window detection.",
    "F04273": "License PENDING CANCELLATION (Hansen Metro Elevation, Houston port) - tests the "
              "disappearance/closure logic and dense industrial surroundings.",
    "F00003": "Oilseed plant + elevator (Cargill Guntersville, AL) with a prior HIGH-confidence address; "
              "likely pre-1984 - tests the 'existed before earliest imagery' case.",
    "F00026": "Delta elevator (Cain Agra, Lake Village AR) with a prior address from a generic "
              "source - tests address re-verification.",
    "F00246": "Small country elevator (Nodaway, IA), 786k bu - tests whether small sites are "
              "visible at Landsat resolution.",
    "F00941": "River soy plant (Bunge Chevron Ag Renewables, Cairo IL) - river terminal setting with "
              "neighbouring industrial structures; a recent ownership/name change.",
}


def main() -> None:
    df = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)
    missing = set(PROTOTYPE) - set(df["facility_id"])
    if missing:
        raise SystemExit(f"Prototype ids not in master table: {missing}")
    sel = df[df["facility_id"].isin(PROTOTYPE)].copy()
    sel.insert(1, "prototype_rationale", sel["facility_id"].map(PROTOTYPE))
    out = path("processed_dir") / "prototype_facilities.csv"
    sel.to_csv(out, index=False, encoding="utf-8")
    log.info("Wrote %d prototype facilities to %s", len(sel), out)


if __name__ == "__main__":
    main()
