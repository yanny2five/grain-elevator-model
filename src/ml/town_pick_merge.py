"""Merge a person's picks from outputs/review/town_pick_<ST>.html into the location evidence.

Input : data/reviews/town_picks/town_picks_<ST>.csv  (downloaded from the review page)
        columns: town, facility_ids, pick (A-D / none / unsure), lat, lon, p
Effect: for every town where the reviewer picked a candidate, one evidence row per
        facility listed for that town is appended to data/addresses/address_evidence.csv
        (MEDIUM, "picked by a person from AI candidates").
        When several facilities share one town, all get the picked point and the
        note says so - they may be different elevators in the same town, which a
        person should check.
        "none"/"unsure" rows are only reported.
Then run:  python src/find_addresses.py --merge-only && python src/geocode.py
           python src/export_yearly_imagery.py --ids <new ids> --sensors naip
Usage:     python src/ml/town_pick_merge.py data/reviews/town_picks/town_picks_SD.csv --reviewer "your name"
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import get_logger, path  # noqa: E402

log = get_logger("town_pick_merge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("picks")
    ap.add_argument("--reviewer", required=True)
    a = ap.parse_args()
    picks = pd.read_csv(a.picks, dtype=str, keep_default_na=False)
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    evf = path("addresses_dir") / "address_evidence.csv"
    ev = pd.read_csv(evf, dtype=str, keep_default_na=False)
    today = dt.date.today().isoformat()
    rows, new_ids = [], []
    for r in picks.itertuples():
        if r.pick not in "ABCD" or not r.lat:
            continue
        fids = r.facility_ids.split(";")
        for fid in fids:
            m = master.loc[fid]
            rows.append({"facility_id": fid, "street_address": "", "city_verified": m["city"], "state_verified": m["state"],
                         "zip_code": "", "latitude": r.lat, "longitude": r.lon,
                         "coord_source": f"picked by a person from AI candidates (candidate {r.pick}, p={r.p}) "
                                         f"in a road-buffer search of the WCMD town",
                         "address_source": "imagery (town_candidates.py)", "source_type": "imagery + human pick",
                         "address_confidence": "MEDIUM",
                         "address_notes": "Operator not verified from a record. " + (
                             f"{len(fids)} facilities share this town and got the same point - check whether they "
                             "are separate elevators." if len(fids) > 1 else ""),
                         "verified_by": a.reviewer, "verified_date": today})
            new_ids.append(fid)
    ev = pd.concat([ev, pd.DataFrame(rows)], ignore_index=True)
    ev.to_csv(evf, index=False)
    log.info("picks: %s", picks["pick"].value_counts().to_dict())
    log.info("added %d evidence rows. Next: re-merge, geocode, then export imagery for: %s", len(rows), " ".join(new_ids))


if __name__ == "__main__":
    main()
