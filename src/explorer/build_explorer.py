"""Package a curated set of facilities for the time-lapse explorer page.

For each selected facility: one JPEG sprite strip with every NAIP year (same
ground square, centre 800 m), plus a JSON record with location provenance,
per-year detector scores, human labels and both build-year estimates.

Output: outputs/explorer/data.json, outputs/explorer/sprites/<id>.jpg
Usage:  python src/explorer/build_explorer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, path  # noqa: E402

OUT = PROJECT_ROOT / "outputs" / "explorer"
TILE = 300           # px per year in the sprite
VIEW_M = 800         # ground width shown (centre crop of the 1.2 km square)


def main() -> None:
    (OUT / "sprites").mkdir(parents=True, exist_ok=True)
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    ml = pd.read_csv(path("outputs_dir") / "ml_first_present.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    by = pd.read_csv(path("outputs_dir") / "ml_presence_by_year.csv", dtype=str)
    est = pd.read_csv(path("outputs_dir") / "estimated_build_years.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    labels = pd.read_csv(path("outputs_dir") / "review_year_labels.csv", dtype=str, keep_default_na=False)
    labels = labels[labels["usable"] == "True"]
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["sensor"] == "naip") & (man["image_available"] == "True") & (man["status"] == "ok")]
    man = man.drop_duplicates(["facility_id", "year"], keep="last")
    known = pd.read_csv(PROJECT_ROOT / "data" / "validation" / "known_dates.csv", dtype=str, keep_default_na=False)

    # Selection: labelled facilities, the ML review queue, then a spread of other outcomes.
    labelled = sorted(set(labels["facility_id"]))
    struct = ml[(ml["chip_mode"] == "centre_300m")]
    queue = list(struct[(struct["ml_confidence"].isin(["HIGH", "MEDIUM"])) & struct["ml_last_absent"].ne("")].index)
    start = list(struct[struct["ml_window"].str.startswith("<=") & struct["ml_confidence"].eq("HIGH")].index)
    hard = list(struct[struct["ml_confidence"].eq("LOW")].index)
    pick = list(dict.fromkeys(labelled + queue + start[:10] + hard[:6]))

    recs = []
    for fid in pick:
        m = man[man["facility_id"] == fid].assign(y=lambda d: d["year"].astype(int)).sort_values("y")
        if len(m) < 5 or fid not in loc.index:
            continue
        tiles = []
        for _, r in m.iterrows():
            im = Image.open(PROJECT_ROOT / r["image_path"]).convert("RGB")
            w, h = im.size
            f = VIEW_M / (2 * float(r["roi_half_width_m"]))
            cw, ch = w * f, h * f
            tiles.append(im.crop((int((w - cw) / 2), int((h - ch) / 2), int((w + cw) / 2), int((h + ch) / 2)))
                         .resize((TILE, TILE), Image.LANCZOS))
        sprite = Image.new("RGB", (TILE * len(tiles), TILE))
        for i, t in enumerate(tiles):
            sprite.paste(t, (i * TILE, 0))
        sprite.save(OUT / "sprites" / f"{fid}.jpg", quality=78, optimize=True)
        p = by[by["facility_id"] == fid].set_index("year")["p_elevator"]
        lab = labels[labels["facility_id"] == fid].set_index("year")["label"]
        L, M, E = loc.loc[fid], master.loc[fid], ml.loc[fid] if fid in ml.index else None
        e = est.loc[fid] if fid in est.index else None
        kd = known[known["facility_id"] == fid]
        recs.append({
            "id": fid, "name": M["warehouse_name"], "entity": M["business_entity"],
            "city": L["city_verified"] or M["city"], "state": M["state"], "capacity": M["warehouse_capacity"],
            "license": M["license_status"] or "(blank)",
            "lat": round(float(L["latitude"]), 5), "lon": round(float(L["longitude"]), 5),
            "address": L["street_address"], "addrConf": L["address_confidence"], "coordSource": L["coord_source"],
            "addrSource": L["address_source"].split(" ; ")[0],
            "years": [int(y) for y in m["y"]],
            "p": [float(p.get(str(y), "nan")) for y in m["y"]],
            "labels": {str(k): v for k, v in lab.items()},
            "ml": None if E is None else {"window": E["ml_window"], "first": E["ml_first_present"],
                                          "conf": E["ml_confidence"], "flags": E["ml_flags"]},
            "human": None if e is None or e["build_year_confidence"] in ("", "NOT_REVIEWED") else
                     {"window": e["estimated_build_window"], "firstVisible": e["estimated_first_visible_year"],
                      "conf": e["build_year_confidence"], "by": e.get("reviewed_by", "")},
            "known": [{"event": k.event.replace("_", " "), "year": k.year, "url": k.source_url} for k in kd.itertuples()],
            "group": "labelled" if fid in labelled else ("ml_change" if fid in queue else
                                                          ("ml_start" if fid in start else "uncertain")),
        })
    (OUT / "data.json").write_text(json.dumps({"tile": TILE, "viewM": VIEW_M, "facilities": recs}), encoding="utf-8")
    size = sum(f.stat().st_size for f in (OUT / "sprites").glob("*.jpg"))
    print(f"{len(recs)} facilities, sprites {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
