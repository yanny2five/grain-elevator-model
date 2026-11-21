"""Hard-negative mining so the detector can SEARCH a town, not just verify a known site.

Problem (see README, town_scan pilot): scoring ~360 chips per town with a ~1%
false-positive rate gives several false "elevators" per town.

Fix: scan towns where the elevator location is already known (HIGH confidence,
coordinates on the structures), and
  * add every chip > 500 m from any known elevator / mapped silo cluster as a
    NEGATIVE - prioritising the ones the current model wrongly scores high;
  * add the chip centred on the known elevator, cut from the same scan image,
    as a POSITIVE (same JPEG/scan domain as search time).
Retrain, then evaluate search on held-out towns: is the top-scoring site within
400 m of the known elevator, and how many other sites are flagged?

Usage: python src/ml/hard_negatives.py --towns 90 --holdout 20
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from build_dataset import CHIP_M, SIZE  # noqa: E402
from find_addresses import ACFG, CachedHTTP, haversine_m, osm_candidates  # noqa: E402
from town_scan import HALF_M, OUT, STRIDE_M, latest_naip  # noqa: E402
from train_eval import Embedder, make_model  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("hard_negatives")
ML = PROJECT_ROOT / "data" / "ml"
HN = ML / "hard_neg"
random.seed(7)


def chips_with_centres(img: Image.Image, lat0: float, lon0: float):
    w, h = img.size
    mpp = 2 * HALF_M / w
    side, step = CHIP_M / mpp, STRIDE_M / mpp
    out, y = [], 0.0
    while y + side <= h:
        x = 0.0
        while x + side <= w:
            cx, cy = x + side / 2, y + side / 2
            lat = lat0 - (cy - h / 2) * mpp / 111_320
            lon = lon0 + (cx - w / 2) * mpp / (111_320 * math.cos(math.radians(lat0)))
            out.append(((int(x), int(y), int(x + side), int(y + side)), lat, lon))
            x += step
        y += step
    return out, mpp


def crop_at(img, lat0, lon0, lat, lon, mpp):
    w, h = img.size
    side = CHIP_M / mpp
    cx = w / 2 + (lon - lon0) * 111_320 * math.cos(math.radians(lat0)) / mpp
    cy = h / 2 - (lat - lat0) * 111_320 / mpp
    return img.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2)))


def sites_from(probs, centres, p_min=0.9, merge_m=350):
    hits = sorted([(c, p) for c, p in zip(centres, probs) if p >= p_min], key=lambda t: -t[1])
    sites = []
    for (la, lo), p in hits:
        for s in sites:
            if haversine_m(la, lo, s[0], s[1]) < merge_m:
                break
        else:
            sites.append((la, lo, p))
    return sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", type=int, default=90)
    ap.add_argument("--holdout", type=int, default=20)
    a = ap.parse_args()
    HN.mkdir(parents=True, exist_ok=True)
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    known = loc[(loc["clip_ok"] == "True") & (loc["geocode_confidence"] == "STRUCTURE")].copy()
    known["lat"], known["lon"] = known["latitude"].astype(float), known["longitude"].astype(float)
    held = set(pd.concat([pd.read_csv(f, dtype=str) for f in (PROJECT_ROOT / "data" / "reviews").glob("*.csv")])
               ["facility_id"])
    anchors = known[~known["facility_id"].isin(held)].drop_duplicates(["state", "city"]).sample(frac=1, random_state=7)
    anchors = anchors.head(a.towns + a.holdout)
    test_towns, train_towns = anchors.head(a.holdout), anchors.iloc[a.holdout:]
    model, emb = joblib.load(ML / "model.joblib"), Embedder()
    http = CachedHTTP("overpass", ACFG["overpass_min_interval_s"])
    gee.init()

    def scan_town(r):
        key = f"{r['facility_id']}"
        f = OUT / f"hn_{key}.jpg"
        if f.exists():
            img = Image.open(f).convert("RGB")
        else:
            img, _ = latest_naip(r["lat"], r["lon"])
            if img is None:
                return None
            img.save(f, quality=85)
        return img

    rows = []
    for i, (_, r) in enumerate(train_towns.iterrows(), 1):
        try:
            img = scan_town(r)
            if img is None:
                continue
            cells, mpp = chips_with_centres(img, r["lat"], r["lon"])
            osm = osm_candidates(http, r["lat"], r["lon"], HALF_M + 500, [])
            avoid = [(c["lat"], c["lon"]) for c in osm if c["kind"] == "silo_cluster"]
            avoid += list(zip(known["lat"], known["lon"]))
            far = [(b, la, lo) for b, la, lo in cells
                   if all(haversine_m(la, lo, x, y) > 500 for x, y in avoid
                          if abs(la - x) < 0.05 and abs(lo - y) < 0.07)]
            ims = [img.crop(b).resize((SIZE, SIZE), Image.BILINEAR) for b, _, _ in far]
            p = model.predict_proba(emb(ims))[:, 1] if ims else np.array([])
            # Keep all confident false positives + a random sample of the rest.
            order = list(np.argsort(-p))
            keep = [k for k in order if p[k] >= 0.3] + random.sample(order, min(25, len(order)))
            for k in dict.fromkeys(keep):
                out = HN / f"neg_{r['facility_id']}_{k}.png"
                ims[k].save(out)
                rows.append({"path": out.relative_to(PROJECT_ROOT).as_posix(), "label": 0,
                             "facility_id": r["facility_id"], "year": "", "kind": "scan_negative",
                             "lat": far[k][1], "lon": far[k][2], "p_before": round(float(p[k]), 3)})
            pos = crop_at(img, r["lat"], r["lon"], r["lat"], r["lon"], mpp).resize((SIZE, SIZE), Image.BILINEAR)
            out = HN / f"pos_{r['facility_id']}.png"
            pos.save(out)
            rows.append({"path": out.relative_to(PROJECT_ROOT).as_posix(), "label": 1,
                         "facility_id": r["facility_id"], "year": "", "kind": "scan_positive",
                         "lat": r["lat"], "lon": r["lon"], "p_before": ""})
            log.info("[%d/%d] %s, %s: %d far chips, %d false positives >=0.5 kept",
                     i, len(train_towns), r["city"], r["state"], len(far), int((p >= 0.5).sum()))
        except Exception as e:  # noqa: BLE001
            log.error("%s failed: %s", r["facility_id"], e)
        pd.DataFrame(rows).to_csv(ML / "hard_negatives.csv", index=False)

    # Retrain on original chips + mined chips.
    base = pd.read_csv(ML / "chips.csv")
    allc = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    X = emb([Image.open(PROJECT_ROOT / p).convert("RGB") for p in allc["path"]])
    y = allc["label"].to_numpy()
    model2 = make_model().fit(X, y)
    joblib.dump(model2, ML / "model_v2.joblib")
    allc.to_csv(ML / "chips_v2.csv", index=False)
    log.info("retrained v2 on %d chips (%d mined)", len(allc), len(rows))

    # Search evaluation on held-out towns: old vs new model.
    res = []
    for _, r in test_towns.iterrows():
        img = scan_town(r)
        if img is None:
            continue
        cells, _ = chips_with_centres(img, r["lat"], r["lon"])
        E = emb([img.crop(b).resize((SIZE, SIZE), Image.BILINEAR) for b, _, _ in cells])
        cent = [(la, lo) for _, la, lo in cells]
        row = {"facility_id": r["facility_id"], "town": f"{r['city']}, {r['state']}"}
        for name, m in [("v1", model), ("v2", model2)]:
            sites = sites_from(m.predict_proba(E)[:, 1], cent)
            hit = [s for s in sites if haversine_m(s[0], s[1], r["lat"], r["lon"]) < 400]
            top_ok = bool(sites) and haversine_m(sites[0][0], sites[0][1], r["lat"], r["lon"]) < 400
            row.update({f"{name}_sites": len(sites), f"{name}_found": bool(hit), f"{name}_top_is_known": top_ok,
                        f"{name}_false_sites": len(sites) - len(hit)})
        res.append(row)
    ev = pd.DataFrame(res)
    ev.to_csv(path("outputs_dir") / "ml_search_eval.csv", index=False)
    summ = {n: {"towns": len(ev), "found_known_elevator": int(ev[f"{n}_found"].sum()),
                "top_site_is_known": int(ev[f"{n}_top_is_known"].sum()),
                "mean_false_sites_per_town": round(float(ev[f"{n}_false_sites"].mean()), 2)} for n in ["v1", "v2"]}
    (path("outputs_dir") / "ml_search_eval.json").write_text(json.dumps(summ, indent=1))
    log.info("search eval: %s", summ)


if __name__ == "__main__":
    main()
