"""Road-buffer test - how close are known grain elevators to a road?

Question (from the advisor meeting): do ~95% of elevators sit within ~100 m of a
road? If so, a road buffer is a safe search area for locating the rest.

Sample (only locations we trust):
  CHECKED    facilities whose location was checked by eye on NAIP
             (the explorer set + hand-verified evidence rows)
  STRUCTURE  facilities snapped onto mapped silos (geocode_confidence STRUCTURE)
             not already in CHECKED - ~1 in 5 of these may still be off.

Roads: US Census TIGER/Line 2016 roads in Earth Engine (TIGER/2016/Roads).
  PUBLIC = MTFCC S1100 primary, S1200 secondary, S1400 local streets
  ALL    = every TIGER road class (adds service drives, private roads, trails)
Distance is from the facility point (the centre of the elevator complex) to the
nearest road line, so a large complex can sit farther from a road than its gate.

Output: outputs/road_distance_test.csv and a summary in the log.
Usage:  python src/road_distance_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("road_distance_test")
PUBLIC = ["S1100", "S1200", "S1400"]
SEARCH_M = 2000          # look for roads within this radius; farther = ">2000"
THRESHOLDS = [25, 50, 100, 200, 500]


def sample() -> pd.DataFrame:
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    loc = loc[loc["clip_ok"] == "True"]
    explorer = json.load(open(PROJECT_ROOT / "outputs" / "explorer" / "data.json"))
    checked = {f["id"] for f in explorer["facilities"]}
    ev = pd.read_csv(path("addresses_dir") / "address_evidence.csv", dtype=str, keep_default_na=False)
    checked |= set(ev["facility_id"])
    loc["group"] = ""
    loc.loc[loc["facility_id"].isin(checked), "group"] = "CHECKED"
    loc.loc[(loc["group"] == "") & (loc["geocode_confidence"] == "STRUCTURE"), "group"] = "STRUCTURE"
    return loc[loc["group"] != ""][["facility_id", "state", "city", "warehouse_name", "latitude", "longitude", "group"]]


def nearest(rows: pd.DataFrame, roads: ee.FeatureCollection) -> list[float | None]:
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r.longitude), float(r.latitude)]), {"i": k})
                                for k, r in enumerate(rows.itertuples())])

    def dist(f):
        g = f.geometry()
        near = roads.filterBounds(g.buffer(SEARCH_M))
        d = near.map(lambda r: r.set("d", r.geometry().distance(g, 1)))
        return f.set("d", ee.Algorithms.If(near.size().gt(0), d.aggregate_min("d"), -1))

    out = pts.map(dist).aggregate_array("d").getInfo()
    return [None if v is None or v < 0 else round(v, 1) for v in out]


def main() -> None:
    gee.init()
    df = sample().reset_index(drop=True)
    log.info("sample: %s", df["group"].value_counts().to_dict())
    roads_all = ee.FeatureCollection("TIGER/2016/Roads")
    roads_pub = roads_all.filter(ee.Filter.inList("mtfcc", PUBLIC))
    d_pub, d_all = [], []
    for s in range(0, len(df), 25):
        chunk = df.iloc[s:s + 25]
        d_pub += nearest(chunk, roads_pub)
        d_all += nearest(chunk, roads_all)
        log.info("done %d/%d", min(s + 25, len(df)), len(df))
    df["dist_public_road_m"] = d_pub
    df["dist_any_road_m"] = d_all
    df.to_csv(path("outputs_dir") / "road_distance_test.csv", index=False)

    for col in ["dist_public_road_m", "dist_any_road_m"]:
        for grp, g in [("ALL", df), *df.groupby("group")]:
            v = g[col].fillna(1e9)
            share = {f"<={t}m": f"{(v <= t).mean():.0%}" for t in THRESHOLDS}
            log.info("%s %s n=%d median=%.0f m %s", col, grp, len(g), v.median(), share)


if __name__ == "__main__":
    main()
