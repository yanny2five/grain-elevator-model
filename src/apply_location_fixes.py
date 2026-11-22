"""Apply the human-reviewed pin fixes (data/addresses/location_fixes_2026-11-22.csv).

- action == "move": new coordinates go into data/addresses/address_evidence.csv (the source geocode.py
  reads first) and into outputs/facility_locations.csv, with geocode_confidence STRUCTURE.
- action == "needs_relocating": facility_locations gets needs_manual_review=True and a note.
- HOLD: fixes not to apply (none now; photo alignment checked in outputs/review/fix_alignment_check.csv).
Backups are written next to each file with the date in the name.

Usage: python src/apply_location_fixes.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT  # noqa: E402

DATE = "2026-11-22"
HOLD: dict[str, str] = {}  # F00907 photo was shifted 47 px; fixed by giving it F00902's target (same elevator)
SRC = "Anay's hand review of the location check, 2026-11-21/22 (pin placed from his note; Claude converted it)"


def main() -> None:
    fx = pd.read_csv(PROJECT_ROOT / f"data/addresses/location_fixes_{DATE}.csv", dtype=str, keep_default_na=False)
    ev_f = PROJECT_ROOT / "data/addresses/address_evidence.csv"
    loc_f = PROJECT_ROOT / "outputs/facility_locations.csv"
    for f in (ev_f, loc_f):
        b = f.with_name(f"{f.stem}_before_{DATE}{f.suffix}")
        if not b.exists():  # keep the first backup if this is re-run
            shutil.copy(f, b)
    ev = pd.read_csv(ev_f, dtype=str, keep_default_na=False)
    loc = pd.read_csv(loc_f, dtype=str, keep_default_na=False)
    moved = held = reloc = 0
    for r in fx.itertuples():
        if r.facility_id in HOLD:
            held += 1
            continue
        if r.action == "move":
            e = ev.facility_id == r.facility_id
            if not e.any():  # coordinates came from elsewhere: add an evidence row from the locations file
                src = loc[loc.facility_id == r.facility_id].iloc[0]
                new = {c: src.get(c, "") for c in ev.columns}
                new["source_type"] = "imagery (hand review)"
                ev.loc[len(ev)] = new
                e = ev.facility_id == r.facility_id
            ev.loc[e, ["latitude", "longitude"]] = [r.new_lat, r.new_lon]
            ev.loc[e, "coord_source"] = f"{SRC}: {r.note}"
            tag = f" | Pin moved on {DATE} after hand review."
            if tag not in ev.loc[e, "address_notes"].iloc[0]:
                ev.loc[e, "address_notes"] = ev.loc[e, "address_notes"] + tag
            ev.loc[e, "verified_by"] = "Anay Kaundinya (hand review on NAIP)"
            ev.loc[e, "verified_date"] = DATE
            m = loc.facility_id == r.facility_id
            loc.loc[m, ["latitude", "longitude"]] = [r.new_lat, r.new_lon]
            loc.loc[m, ["coord_source", "geocode_source"]] = f"{SRC}: {r.note}"
            loc.loc[m, ["geocode_confidence", "clip_ok", "needs_manual_review"]] = ["STRUCTURE", "True", "False"]
            moved += 1
        elif r.action == "needs_relocating":
            m = loc.facility_id == r.facility_id
            loc.loc[m, "needs_manual_review"] = "True"
            tag = f"{DATE} hand review: no elevator at pin; needs relocating."
            if tag not in loc.loc[m, "auto_note"].iloc[0]:
                loc.loc[m, "auto_note"] = (loc.loc[m, "auto_note"] + " | " + tag).str.strip(" |")
            reloc += 1
    ev.to_csv(ev_f, index=False)
    loc.to_csv(loc_f, index=False)
    print(f"moved {moved}, needs relocating {reloc}, held {held}")


if __name__ == "__main__":
    main()
