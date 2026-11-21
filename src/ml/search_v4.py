"""Town-search detector v4: retrain with the many locations confirmed since v3, test on held-out towns.

Ground truth: facilities whose location Claude checked on imagery and judged on an
elevator ('ok' or 'move' in data/addresses/location_check_2026-*.csv), using the
current (corrected) coordinates. Every known elevator in the same town counts as a hit.

Towns are split by town (never by facility): ~2/3 train, ~1/3 test, fixed seed.
For each town: one recent NAIP 4 km scan around the Census town centroid (cached
in data/ml/town_scans/sv4_*.jpg), 300 m chips every 200 m, road buffer as in
town_candidates.py.

Training set for v4 = v3 data (chips_v2 + augmented scan positives, as tune_search.py)
  + positives: train-town chips whose centre is within 150 m of a known elevator
  + hard negatives: train-town chips > 600 m from every known elevator and mapped
    location that v3 scores >= 0.5
Test: top-4 shortlist per test town (road buffer, MERGE_M apart). Hit = a candidate within
400 m of a known elevator. Reports v3 vs v4: found-in-top-4, top-1 correct.

Outputs: data/ml/model_search_v4.joblib (only if better on top-4), outputs/ml_search_v4.md
Usage:   python src/ml/search_v4.py [--towns 150]
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from build_dataset import SIZE  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from clean_dataset import _norm_place  # noqa: E402
from find_addresses import haversine_m, load_places  # noqa: E402
from town_candidates import MERGE_M, TOP_N, chips, road_mask, slug  # noqa: E402
from town_scan import latest_naip  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("ml_search_v4")
ML = PROJECT_ROOT / "data" / "ml"
SCANS = ML / "town_scans"


def known_sites() -> pd.DataFrame:
    chk = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False)
                     for f in glob.glob(str(PROJECT_ROOT / "data" / "addresses" / "location_check_2026-*.csv"))])
    chk = chk.drop_duplicates("facility_id", keep="last")
    good = set(chk.loc[chk["verdict"].isin(["ok", "move"]), "facility_id"])
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    loc = loc[(loc["clip_ok"] == "True") & loc["facility_id"].isin(good)].copy()
    loc["lat"], loc["lon"] = loc["latitude"].astype(float), loc["longitude"].astype(float)
    return loc


def shortlist(p, cs, k=TOP_N):
    p_eff = np.where([c["road"] for c in cs], p, 0.0)
    out = []
    for i in np.argsort(-p_eff):
        if all(haversine_m(cs[i]["lat"], cs[i]["lon"], q["lat"], q["lon"]) >= MERGE_M for q in out):
            out.append({**cs[i], "p": float(p[i])})
        if len(out) == k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--towns", type=int, default=150)
    a = ap.parse_args()
    sites = known_sites()
    places = load_places()
    towns = []
    for (st, city), g in sites.groupby(["state", "city"]):
        pl = places.get((st, _norm_place(city)))
        if not pl:
            continue
        pts = list(zip(g["lat"], g["lon"]))
        # only towns whose known elevator is inside the 4 km scan square
        if min(haversine_m(pl[0], pl[1], la, lo) for la, lo in pts) > 1700:
            continue
        towns.append({"st": st, "city": city, "lat": pl[0], "lon": pl[1], "pts": pts})
    random.Random(20261024).shuffle(towns)
    towns = towns[:a.towns]
    log.info("%d towns with confirmed elevators inside the scan square", len(towns))
    all_pts = list(zip(sites["lat"], sites["lon"]))

    gee.init()
    emb = GpuEmbedder()
    v3 = joblib.load(ML / "model_search.joblib")
    for i, t in enumerate(towns, 1):
        f = SCANS / f"sv4_{t['st']}_{slug(t['city'])}.jpg"
        fe = f.with_suffix(".npz")
        try:
            _scan_one(t, f, fe, emb)
        except Exception as e:  # noqa: BLE001 - a transient Earth Engine error skips one town
            log.warning("%s %s skipped: %s", t["st"], t["city"], str(e)[:120])
            continue
        if not fe.exists():
            continue
        z = np.load(fe)
        t["X"] = z["X"]
        t["cs"] = [{"lat": la, "lon": lo, "road": bool(r)} for la, lo, r in zip(z["lat"], z["lon"], z["road"])]
        if i % 10 == 0:
            log.info("%d/%d towns scanned", i, len(towns))
    towns = [t for t in towns if "X" in t]
    _rest(towns, all_pts, v3)


def _scan_one(t, f, fe, emb):
    if not fe.exists():
        if f.exists():
            img = Image.open(f).convert("RGB")
        else:
            img, _ = latest_naip(t["lat"], t["lon"])
            if img is None:
                return
            img.save(f, quality=90)
        mask = road_mask(t["lat"], t["lon"], 400)
        cs, _ = chips(img, mask, t["lat"], t["lon"])
        X = emb([img.crop(c["box"]).resize((SIZE, SIZE), Image.BILINEAR) for c in cs])
        np.savez_compressed(fe, X=X, lat=[c["lat"] for c in cs], lon=[c["lon"] for c in cs],
                                road=[c["road"] for c in cs])


def _rest(towns, all_pts, v3):
    n_test = len(towns) // 3
    test, train = towns[:n_test], towns[n_test:]

    # --- v3 training data (as tune_search.py) + new positives / hard negatives from train towns
    base = pd.read_csv(ML / "chips_v2.csv")
    Xb = np.load(ML / "emb_chips_v2.npz")["X"]
    Xa = np.load(ML / "emb_scan_pos_aug.npz")["X"]
    Xs, ys = [Xb, Xa], [base["label"].to_numpy(), np.ones(len(Xa), int)]
    n_pos = n_neg = 0
    for t in train:
        d_known = np.array([min(haversine_m(c["lat"], c["lon"], la, lo) for la, lo in t["pts"]) for c in t["cs"]])
        pos = d_known <= 150
        p3 = v3.predict_proba(t["X"])[:, 1]
        far = np.array([min(haversine_m(c["lat"], c["lon"], la, lo) for la, lo in all_pts
                            if abs(la - c["lat"]) < 0.05) if any(abs(la - c["lat"]) < 0.05 for la, _ in all_pts) else 9e9
                        for c in t["cs"]])
        neg = (far > 600) & (p3 >= 0.5)
        Xs += [t["X"][pos], t["X"][neg]]
        ys += [np.ones(pos.sum(), int), np.zeros(neg.sum(), int)]
        n_pos += pos.sum()
        n_neg += neg.sum()
    log.info("added %d positives and %d hard negatives from %d train towns", n_pos, n_neg, len(train))
    X, y = np.vstack(Xs), np.concatenate(ys)

    def evaluate(model):
        found = top1 = 0
        for t in test:
            p = model.predict_proba(t["X"])[:, 1]
            sl = shortlist(p, t["cs"])
            hits = [min(haversine_m(c["lat"], c["lon"], la, lo) for la, lo in t["pts"]) < 400 for c in sl]
            found += any(hits)
            top1 += bool(hits) and hits[0]
        return found, top1

    res = [("v3 (current)", *evaluate(v3))]
    best = None
    for C in [0.01, 0.05]:
        for wpos in [1, 3]:
            w = np.where(y == 1, wpos, 1.0)
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            m.fit(X, y, logisticregression__sample_weight=w)
            f4, t4 = evaluate(m)
            res.append((f"v4 C={C} pos_weight={wpos}", f4, t4))
            if best is None or (f4, t4) > best[1:]:
                best = (m, f4, t4)
    n = len(test)
    lines = ["# Town search v4 vs v3 (held-out towns)", "",
             f"Test towns: {n} (towns with a Claude-confirmed elevator inside the 4 km scan; split by town).",
             f"Training additions: {n_pos} positive chips and {n_neg} hard negatives from {len(train)} other towns.", "",
             "| Model | Elevator in top-4 shortlist | Top pick correct |", "|---|---|---|"]
    lines += [f"| {name} | {f}/{n} ({f / n:.0%}) | {t1}/{n} ({t1 / n:.0%}) |" for name, f, t1 in res]
    if best and (best[1], best[2]) > (res[0][1], res[0][2]):
        joblib.dump(best[0], ML / "model_search_v4.joblib")
        lines += ["", "Best v4 saved as data/ml/model_search_v4.joblib."]
    else:
        lines += ["", "v4 did not beat v3; v3 kept."]
    (PROJECT_ROOT / "outputs" / "ml_search_v4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
