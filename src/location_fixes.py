"""Turn Anay's location-check notes (2026-11-21/22) into new pin coordinates.

Inputs
  data/reviews/location_check_review.csv   agree/disagree/unsure per facility (column `final`)
  data/reviews/location_check_notes.csv    free-text notes on where the elevator is
  data/addresses/location_check_2026-*.csv Claude's Oct check (old point, moved point)
  data/ml/candidates.csv                   AI candidates A/B/C (lat, lon) drawn on the photos

Geometry: each review photo is 800 px for 1.2 km (1.5 m per px, grid = 100 m), centred on the OLD point.
Bug (2026-11-20 to 11-22): the page drew the green cross at 60% of the true offset, so notes that say
"relative to the green point" refer to the DRAWN green, which this script reconstructs.

SPEC below is my (Claude's) reading of each note, by hand:
  ("cand", L, dx_m, dy_m)  candidate L plus an offset in metres (east +, north +); 1 grid square = 100 m
  ("px", x, y)             a spot on the 800 px photo
  ("red", dx_m, dy_m)      old point plus offset
  ("green", dx_m, dy_m)    DRAWN green point plus offset
  ("keep",)                current pin is fine (no change)
  ("none",)                no elevator found: needs relocating
  ("same", fid)            same site as another facility (use its target)

Output: data/addresses/location_fixes_2026-11-22.csv
Usage:  python src/location_fixes.py
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT  # noqa: E402

M_PER_PX, DRAWN = 1.5, 0.6
A, B, C = "A", "B", "C"
SPEC = {
    "F02307": ("cand", C, 0, 0), "F00038": ("px", 235, 510), "F02768": ("cand", B, 0, 0),
    "F00453": ("cand", A, 0, 0), "F00662": ("cand", A, 0, 0), "F00738": ("cand", B, 0, 0),
    "F00358": ("cand", C, 0, 0), "F00843": ("cand", A, 0, 0), "F00599": ("px", 180, 595),
    "F00902": ("cand", A, 0, 0), "F00907": ("same", "F00902"), "F01610": ("px", 330, 510),
    "F01139": ("cand", A, 0, 0), "F01144": ("cand", A, -100, 0), "F01668": ("cand", C, 0, 100),
    "F01641": ("cand", A, 0, 0), "F01658": ("cand", A, 0, -100), "F01694": ("cand", A, 0, 0),
    "F01663": ("cand", B, 0, 0), "F02295": ("px", 520, 225), "F02327": ("cand", C, 0, 100),
    "F02130": ("none",), "F02037": ("cand", A, 0, 0), "F02058": ("cand", A, 0, 0),
    "F02060": ("cand", A, 0, 0), "F02303": ("cand", B, 100, 0), "F02073": ("cand", C, 100, 100),
    "F02639": ("cand", A, 0, 0), "F02664": ("cand", A, 0, 0), "F02630": ("px", 300, 385),
    "F02888": ("cand", A, 0, 0), "F02924": ("cand", A, 0, 0), "F03110": ("cand", C, 0, 0),
    "F03042": ("cand", B, 0, 100), "F03121": ("cand", A, 0, 0), "F03019": ("px", 75, 495),
    "F03051": ("cand", A, 0, 100), "F03145": ("none",), "F03067": ("cand", A, 0, 0),
    "F03068": ("cand", A, 0, 0), "F03241": ("cand", A, 0, 100), "F03389": ("cand", A, 100, 0),
    "F03486": ("none",), "F03322": ("px", 295, 388), "F03488": ("cand", A, 0, 0),
    "F03705": ("cand", B, 0, -100), "F03753": ("px", 355, 440), "F03930": ("cand", A, 0, 0),
    "F03972": ("keep",), "F04039": ("px", 565, 375), "F04148": ("keep",),
    "F04085": ("cand", A, 0, 100), "F04120": ("cand", A, 0, 0), "F04186": ("cand", A, 0, 0),
    "F04732": ("px", 155, 160), "F00169": ("red", -200, 0), "F00413": ("cand", A, 0, 0),
    "F00684": ("green", 200, 0), "F00345": ("cand", A, 0, 0), "F00782": ("keep",),
    "F00510": ("red", 0, -100), "F01152": ("cand", A, 0, 0), "F01970": ("cand", A, 0, 0),
    "F03562": ("cand", A, 0, 0), "F03352": ("px", 445, 40), "F03386": ("cand", A, 0, 0),
    "F03226": ("keep",), "F03239": ("cand", A, 0, 0), "F03495": ("cand", A, 0, 0),
    "F01958": ("keep",), "F01836": ("px", 530, 392), "F03941": ("px", 430, 495),
    "F03960": ("px", 30, 355), "F03850": ("px", 548, 185), "F03892": ("px", 540, 275),
    "F03791": ("cand", A, -100, 0), "F03792": ("cand", A, 100, 100), "F03793": ("cand", C, -100, 0),
    "F03879": ("px", 465, 620), "F03920": ("cand", B, 0, 0), "F03903": ("px", 565, 455),
    "F03867": ("keep",), "F03881": ("cand", A, 0, 0), "F03971": ("cand", A, -100, 0),
    "F04309": ("px", 265, 635), "F04335": ("red", 0, 0), "F04336": ("cand", C, -100, 0),
    "F04396": ("keep",), "F04572": ("px", 490, 445), "F04651": ("cand", A, 0, 0),
    "F04647": ("cand", A, 0, 0), "F01928": ("px", 190, 505), "F03230": ("cand", A, 0, 0),
    "F00576": ("cand", A, 0, 0), "F00920": ("cand", C, 0, 100), "F01386": ("px", 90, 475),
    "F02540": ("cand", B, 0, 0), "F03485": ("px", 480, 630), "F03813": ("cand", A, 0, -100),
    "F03825": ("cand", B, 100, 0), "F03973": ("cand", B, 0, 0), "F03876": ("px", 340, 510),
    "F04333": ("keep",),
    # recheck on the fixed page (2026-11-22)
    "F00672": ("cand", B, 0, 0), "F02062": ("px", 392, 475), "F03861": ("cand", A, 0, 80),
    "F01929": ("same", "F01928"), "F03785": ("cand", C, 0, 100), "F04306": ("none",),
}


def to_m(lat, lon, lat0, lon0):
    return ((lon - lon0) * 111_320 * math.cos(math.radians(lat0)), (lat - lat0) * 111_320)


def to_ll(dx, dy, lat0, lon0):
    return lat0 + dy / 111_320, lon0 + dx / (111_320 * math.cos(math.radians(lat0)))


def main() -> None:
    chk = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in
                     sorted(glob.glob(str(PROJECT_ROOT / "data/addresses/location_check_2026-*.csv")))])
    chk = chk.drop_duplicates("facility_id", keep="last").set_index("facility_id")
    cand = pd.read_csv(PROJECT_ROOT / "data/ml/candidates.csv", dtype={"facility_id": str})
    rv = pd.read_csv(PROJECT_ROOT / "data/reviews/location_check_review.csv", dtype=str,
                     keep_default_na=False).set_index("facility_id")
    notes = pd.read_csv(PROJECT_ROOT / "data/reviews/location_check_notes.csv", dtype=str,
                        keep_default_na=False).drop_duplicates("facility_id", keep="last").set_index("facility_id")
    rows, done = [], {}
    for fid, s in SPEC.items():
        r = chk.loc[fid]
        lat0, lon0 = float(r.old_lat), float(r.old_lon)
        moved = r.verdict == "move" and r.new_lat != ""
        cur = (float(r.new_lat), float(r.new_lon)) if moved else (lat0, lon0)
        cx, cy = to_m(*cur, lat0, lon0)
        kind = s[0]
        if kind == "none":
            rows.append(dict(facility_id=fid, action="needs_relocating", new_lat="", new_lon="", moved_m=""))
            continue
        if kind == "keep":
            tx, ty = cx, cy
        elif kind == "cand":
            q = cand[(cand.facility_id == fid) & (cand.cand == s[1])]
            if q.empty:
                raise SystemExit(f"{fid}: no candidate {s[1]}")
            ax, ay = to_m(q.lat.iloc[0], q.lon.iloc[0], lat0, lon0)
            tx, ty = ax + s[2], ay + s[3]
        elif kind == "px":
            tx, ty = (s[1] - 400) * M_PER_PX, (400 - s[2]) * M_PER_PX
        elif kind == "red":
            tx, ty = s[1], s[2]
        elif kind == "green":
            tx, ty = cx * DRAWN + s[1], cy * DRAWN + s[2]
        elif kind == "same":  # same site as another facility: reuse its target
            tx, ty = to_m(*done[s[1]], lat0, lon0)
        gap = math.hypot(tx - cx, ty - cy)
        lat, lon = to_ll(tx, ty, lat0, lon0)
        done[fid] = (lat, lon)
        if kind == "keep" or gap <= 100:
            action = "keep (pin already within 100 m)"
            if moved and kind != "keep" and gap <= 100 and rv.loc[fid, "final"] == "disagree":
                action = "keep (pin was right; green drawn wrong on page)"
            lat, lon = cur
        else:
            action = "move"
        rows.append(dict(facility_id=fid, action=action, new_lat=round(lat, 6), new_lon=round(lon, 6),
                         moved_m=round(gap)))
    out = pd.DataFrame(rows)
    out.insert(1, "city", out.facility_id.map(chk.city))
    out.insert(2, "state", out.facility_id.map(chk.state))
    out.insert(3, "claude_verdict", out.facility_id.map(chk.verdict))
    out.insert(4, "review", out.facility_id.map(rv["final"]))
    out["note"] = out.facility_id.map(notes.note)
    out["source"] = "Anay's location review 2026-11-21/22 notes, read into coordinates by Claude (src/location_fixes.py)"
    f = PROJECT_ROOT / "data/addresses/location_fixes_2026-11-22.csv"
    out.to_csv(f, index=False)
    print(out.action.value_counts().to_string())
    print("wrote", f)


if __name__ == "__main__":
    main()
