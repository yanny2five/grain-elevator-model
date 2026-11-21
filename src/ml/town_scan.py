"""Locate unresolved facilities by scanning the WCMD town with the elevator detector.

For facilities no company/registry source covers: download one recent NAIP
image (~4 km square) around the Census centroid of the WCMD town, slide 300 m
chips over it (200 m stride), score each chip with the Phase 8 classifier, and
merge high-scoring chips into candidate sites. A candidate is kept only if it
also looks commercial: a railway within 250 m or >= 3 mapped silos nearby
(farm bins also look like "elevator" to the classifier - see Black Creek WI).

Decision per town (conservative):
  exactly one commercial candidate  -> MEDIUM "elevator found in imagery in the
                                       WCMD town; operator not verified"
  several                           -> NEEDS_REVIEW, all candidates listed
  none                              -> UNRESOLVED
Results are written as candidates; facility evidence is written to
data/addresses/imagery_scan_evidence.csv, which the merge treats like
automated evidence (below company and verified evidence).

Usage: python src/ml/town_scan.py --entity "CoMark Equity Alliance, LLC"
       python src/ml/town_scan.py --unresolved          # every facility without a usable location
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from build_dataset import CHIP_M, SIZE  # noqa: E402
from clean_dataset import _norm_place  # noqa: E402
from find_addresses import ACFG, EVIDENCE_COLUMNS, CachedHTTP, haversine_m, load_places, osm_candidates, rail_within  # noqa: E402
from train_eval import Embedder  # noqa: E402
from utils import JsonCache, PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("town_scan")
HALF_M = 2000         # scan a 4 km x 4 km square around the town centroid
STRIDE_M = 200
P_MIN = 0.9           # chip probability to count as elevator-like
MERGE_M = 350         # elevator-like chips closer than this form one candidate site
SCALE = 2             # NAIP metres per pixel for the scan
OUT = PROJECT_ROOT / "data" / "ml" / "town_scans"


def latest_naip(lat: float, lon: float) -> tuple[Image.Image | None, int | None]:
    region = gee.roi(lat, lon, HALF_M)
    for year in range(2025, 2014, -1):
        b = gee.build("naip", year, region)
        if gee.cloud_quality("naip", b["n_scenes"], b["clear_fraction"]) != "GOOD":
            continue
        px = int(2 * HALF_M / SCALE)
        url = b["image"].getThumbURL({**gee.VIS["naip"], "region": region, "dimensions": f"{px}x{px}",
                                      "crs": "EPSG:3857", "format": "jpg"})
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB"), year
    return None, None


def scan(img: Image.Image, lat0: float, lon0: float, model, emb) -> list[dict]:
    w, h = img.size
    mpp = 2 * HALF_M / w
    side, step = CHIP_M / mpp, STRIDE_M / mpp
    chips, centres = [], []
    y = 0.0
    while y + side <= h:
        x = 0.0
        while x + side <= w:
            chips.append(img.crop((int(x), int(y), int(x + side), int(y + side))).resize((SIZE, SIZE), Image.BILINEAR))
            cx, cy = x + side / 2, y + side / 2
            dlat = -(cy - h / 2) * mpp / 111_320
            dlon = (cx - w / 2) * mpp / (111_320 * math.cos(math.radians(lat0)))
            centres.append((lat0 + dlat, lon0 + dlon))
            x += step
        y += step
    p = model.predict_proba(emb(chips))[:, 1]
    hits = [(c, float(pp)) for c, pp in zip(centres, p) if pp >= P_MIN]
    # Merge neighbouring hits into candidate sites (probability-weighted centroid).
    sites: list[dict] = []
    for (la, lo), pp in sorted(hits, key=lambda t: -t[1]):
        for s in sites:
            if haversine_m(la, lo, s["lat"], s["lon"]) < MERGE_M:
                s["members"].append((la, lo, pp))
                break
        else:
            sites.append({"lat": la, "lon": lo, "members": [(la, lo, pp)]})
    for s in sites:
        ws = sum(m[2] for m in s["members"])
        s["lat"] = sum(m[0] * m[2] for m in s["members"]) / ws
        s["lon"] = sum(m[1] * m[2] for m in s["members"]) / ws
        s["n_chips"], s["p_max"] = len(s["members"]), max(m[2] for m in s["members"])
        del s["members"]
    return sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity")
    ap.add_argument("--unresolved", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    located = set(loc.loc[loc["clip_ok"] == "True", "facility_id"])
    fac = master[(master["is_uswa"] == "True") & ~master["facility_id"].isin(located)]
    if a.entity:
        fac = fac[fac["business_entity"] == a.entity]
    if a.limit:
        fac = fac.head(a.limit)
    places = load_places()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = JsonCache(OUT / "scan_cache.json")
    http = CachedHTTP("overpass", ACFG["overpass_min_interval_s"])
    model, emb = joblib.load(PROJECT_ROOT / "data" / "ml" / "model.joblib"), Embedder()
    gee.init()

    ev_rows, cand_rows = [], []
    towns = fac.groupby(["state", "city"])
    log.info("%d facilities in %d towns", len(fac), towns.ngroups)
    for i, ((st, city), g) in enumerate(towns, 1):
        key = f"{st}|{_norm_place(city)}"
        place = places.get((st, _norm_place(city))) if city else None
        if not place:
            for fid in g["facility_id"]:
                cand_rows.append({"facility_id": fid, "town": city, "state": st, "status": "UNRESOLVED",
                                  "note": "town not in Census gazetteer"})
            continue
        try:
            if key not in cache:
                img, year = latest_naip(place[0], place[1])
                if img is None:
                    cache.set(key, {"year": None, "sites": []})
                else:
                    img.save(OUT / f"{st}_{_norm_place(city).replace(' ', '_')}_{year}.jpg", quality=85)
                    sites = scan(img, place[0], place[1], model, emb)
                    # Commercial check: rail nearby or >= 3 mapped silos within 300 m.
                    osm = osm_candidates(http, place[0], place[1], HALF_M + 500, [])
                    clusters = [c for c in osm if c["kind"] == "silo_cluster"]
                    for s in sites:
                        silos = sum(c["n_silos"] for c in clusters
                                    if haversine_m(s["lat"], s["lon"], c["lat"], c["lon"]) < 300)
                        s["osm_silos"] = silos
                        s["rail"] = rail_within(http, s["lat"], s["lon"])
                        s["commercial"] = bool(s["rail"] or silos >= 3)
                    cache.set(key, {"year": year, "sites": sites})
            res = cache.get(key)
            com = [s for s in res["sites"] if s["commercial"]]
            com.sort(key=lambda s: (-s["n_chips"], -s["p_max"]))
            for fid in g["facility_id"]:
                for k, s in enumerate(res["sites"]):
                    cand_rows.append({"facility_id": fid, "town": city, "state": st, "rank": k + 1, **s,
                                      "naip_year": res["year"]})
                if len(com) == 1:
                    s = com[0]
                    note = (f"Elevator detected in NAIP {res['year']} scan of {city}, {st} (4 km square around the "
                            f"town centroid): one commercial-looking site ({s['n_chips']} chips, max p "
                            f"{s['p_max']:.2f}, rail={s['rail']}, mapped silos={s['osm_silos']}). Operator not "
                            f"verified - assumes the WCMD facility is the town's elevator.")
                    if len(g) > 1:
                        note += f" {len(g)} WCMD units of this entity in this town share the site."
                    ev_rows.append({"facility_id": fid, "street_address": "", "city_verified": city, "state_verified": st,
                                    "zip_code": "", "latitude": round(s["lat"], 6), "longitude": round(s["lon"], 6),
                                    "coord_source": "imagery scan: elevator-detector site centroid",
                                    "address_source": f"NAIP {res['year']} (Google Earth Engine) + src/ml/town_scan.py",
                                    "source_type": "imagery detection", "address_confidence": "MEDIUM",
                                    "address_notes": note, "verified_by": "town_scan.py (automated)",
                                    "verified_date": pd.Timestamp.today().strftime("%Y-%m-%d")})
        except Exception as e:  # noqa: BLE001
            log.error("%s, %s failed: %s", city, st, e)
        if i % 10 == 0:
            log.info("%d/%d towns", i, towns.ngroups)
            pd.DataFrame(ev_rows, columns=EVIDENCE_COLUMNS).to_csv(
                path("addresses_dir") / "imagery_scan_evidence.csv", index=False)
    pd.DataFrame(ev_rows, columns=EVIDENCE_COLUMNS).to_csv(path("addresses_dir") / "imagery_scan_evidence.csv",
                                                           index=False)
    pd.DataFrame(cand_rows).to_csv(path("addresses_dir") / "imagery_scan_candidates.csv", index=False)
    n_one = len({r["facility_id"] for r in ev_rows})
    log.info("done: %d facilities, %d located by a single commercial site", len(fac), n_one)


if __name__ == "__main__":
    main()
