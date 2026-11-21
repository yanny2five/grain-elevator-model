"""Does a road buffer improve the town search?

Advisor suggestion: only search near real roads. road_distance_test.py showed
98% of trusted elevators lie within 250 m of a public road. Here we check
whether dropping scan chips far from roads cuts false sites in the 20 held-out
towns without losing the real elevator.

Setup (same as tune_search.py): 4 km x 4 km NAIP scan per town, 300 m chips
every 200 m, best search model (data/ml/model_search.joblib). A chip is kept
when its centre is within BUFFER m of a TIGER 2016 road (public classes, or all
classes). "Found" = a candidate site within 400 m of the known elevator.

Two towns were relocated on 2026-10-11 after the scans were made: their scan
images are centred on the OLD point, and the truth is the NEW point.

Output: outputs/road_buffer_search_test.csv
Usage:  python src/ml/road_buffer_search_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from find_addresses import haversine_m  # noqa: E402
from hard_negatives import chips_with_centres  # noqa: E402
from town_scan import OUT  # noqa: E402
from tune_search import sites  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("road_buffer_search_test")
ML = PROJECT_ROOT / "data" / "ml"
SCAN_CENTRE_BEFORE_MOVE = {"F03546": (40.30424, -97.68849), "F03840": (36.01564, -99.29208)}
BUFFERS = [None, 100, 150, 250]
PUBLIC = ["S1100", "S1200", "S1400"]


def road_distances(centres, lat0, lon0, roads) -> list[float]:
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([lo, la]), {"i": i}) for i, (la, lo) in enumerate(centres)])
    near = roads.filterBounds(ee.Geometry.Point([lon0, lat0]).buffer(2600))

    def dist(f):
        g = f.geometry()
        d = near.filterBounds(g.buffer(600)).map(lambda r: r.set("d", r.geometry().distance(g, 1)))
        return f.set("d", ee.Algorithms.If(d.size().gt(0), d.aggregate_min("d"), 9999))

    return pts.map(dist).aggregate_array("d").getInfo()


def main() -> None:
    gee.init()
    model = joblib.load(ML / "model_search.joblib")
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    test_ids = list(pd.read_csv(path("outputs_dir") / "ml_search_eval.csv", dtype=str)["facility_id"])
    roads = {"public": ee.FeatureCollection("TIGER/2016/Roads").filter(ee.Filter.inList("mtfcc", PUBLIC)),
             "all": ee.FeatureCollection("TIGER/2016/Roads")}

    towns = []
    for fid in test_ids:
        f = OUT / f"hn_{fid}.jpg"
        emb = ML / f"emb_town_{fid}.npz"
        if not (f.exists() and emb.exists()):
            continue
        true = (float(loc.loc[fid, "latitude"]), float(loc.loc[fid, "longitude"]))
        lat0, lon0 = SCAN_CENTRE_BEFORE_MOVE.get(fid, true)
        cells, _ = chips_with_centres(Image.open(f), lat0, lon0)
        centres = [(a, b) for _, a, b in cells]
        p = model.predict_proba(np.load(emb)["X"])[:, 1]
        d = {k: np.array(road_distances(centres, lat0, lon0, r)) for k, r in roads.items()}
        towns.append({"fid": fid, "true": true, "centres": centres, "p": p, "d": d})
        log.info("%s: %d chips, %.0f%% within 250 m of a public road", fid, len(p), 100 * (d["public"] <= 250).mean())

    rows = []
    for kind in ["public", "all"]:
        for buf in BUFFERS:
            for rule, th in [("top1", 0.3), ("threshold", 0.9), ("threshold", 0.7)]:
                found = topok = false = 0
                kept = []
                for t in towns:
                    keep = np.ones(len(t["p"]), bool) if buf is None else t["d"][kind] <= buf
                    kept.append(keep.mean())
                    p = np.where(keep, t["p"], 0.0)
                    s = sites(p, t["centres"], th, top1=(rule == "top1"))
                    hit = [x for x in s if haversine_m(x[0], x[1], *t["true"]) < 400]
                    found += bool(hit)
                    topok += bool(s) and haversine_m(s[0][0], s[0][1], *t["true"]) < 400
                    false += len(s) - len(hit)
                n = len(towns)
                rows.append({"roads": kind, "buffer_m": buf or "none", "rule": rule, "threshold": th, "towns": n,
                             "area_kept": f"{np.mean(kept):.0%}", "found": found, "top_correct": topok,
                             "false_sites_per_town": round(false / n, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(path("outputs_dir") / "road_buffer_search_test.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
