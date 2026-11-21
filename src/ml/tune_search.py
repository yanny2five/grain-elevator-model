"""Rebalance and tune the town-search detector (follow-up to hard_negatives.py).

hard_negatives.py v2 cut false sites from 2.75 to 0.25 per town but found only
6/20 known elevators: ~40 mined negatives per town vs 1 positive made it say
"no" almost everywhere. Here:
  1. add ~10 positives per training town: crops around the known elevator in the
     same scan image, shifted by up to 60 m, plus mirror images;
  2. cache embeddings for every training chip and every held-out town chip;
  3. train a few model settings and sweep the decision rule:
       threshold rule  - every chip >= t becomes a candidate site
       top-1 rule      - only the town's best chip, if >= t
     scored on the 20 held-out towns: found (a site within 400 m of the known
     elevator), top site correct, false sites per town.
Outputs: outputs/ml_search_tuning.csv, data/ml/model_search.joblib (best setting)
Usage: python src/ml/tune_search.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SIZE  # noqa: E402
from find_addresses import haversine_m  # noqa: E402
from hard_negatives import chips_with_centres, crop_at  # noqa: E402
from town_scan import OUT  # noqa: E402
from train_eval import Embedder  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("tune_search")
ML = PROJECT_ROOT / "data" / "ml"
JITTER_M = [-60, 0, 60]


def embed_cached(emb, name: str, make_images):
    f = ML / f"emb_{name}.npz"
    if f.exists():
        return np.load(f)["X"]
    X = emb(make_images())
    np.savez_compressed(f, X=X)
    return X


def sites(probs, centres, t, merge_m=350, top1=False):
    order = np.argsort(-probs)
    if top1:
        k = order[0]
        return [(centres[k][0], centres[k][1], probs[k])] if probs[k] >= t else []
    out = []
    for k in order:
        if probs[k] < t:
            break
        la, lo = centres[k]
        if all(haversine_m(la, lo, s[0], s[1]) >= merge_m for s in out):
            out.append((la, lo, probs[k]))
    return out


def main() -> None:
    emb = Embedder()
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    hn = pd.read_csv(ML / "hard_negatives.csv")
    train_ids = sorted(hn.loc[hn["kind"] == "scan_positive", "facility_id"])
    test_ids = list(pd.read_csv(path("outputs_dir") / "ml_search_eval.csv", dtype=str)["facility_id"])

    # --- training set: original chips + mined negatives + augmented scan positives
    base = pd.read_csv(ML / "chips_v2.csv")
    Xb = embed_cached(emb, "chips_v2", lambda: [Image.open(PROJECT_ROOT / p).convert("RGB") for p in base["path"]])
    aug = []
    for fid in train_ids:
        f = OUT / f"hn_{fid}.jpg"
        if not f.exists():
            continue
        img = Image.open(f).convert("RGB")
        lat, lon = float(loc.loc[fid, "latitude"]), float(loc.loc[fid, "longitude"])
        mpp = 4000 / img.size[0]
        for dy in JITTER_M:
            for dx in JITTER_M:
                c = crop_at(img, lat, lon, lat + dy / 111_320,
                            lon + dx / (111_320 * np.cos(np.radians(lat))), mpp).resize((SIZE, SIZE), Image.BILINEAR)
                aug.append(c)
                if dx == 0 and dy == 0:
                    aug += [ImageOps.mirror(c), ImageOps.flip(c), c.rotate(90)]
    Xa = embed_cached(emb, "scan_pos_aug", lambda: aug)
    X = np.vstack([Xb, Xa])
    y = np.concatenate([base["label"].to_numpy(), np.ones(len(Xa), int)])
    log.info("training chips: %d (+%d augmented scan positives)", len(X), len(Xa))

    # --- held-out towns: embed every scan chip once
    towns = []
    for fid in test_ids:
        f = OUT / f"hn_{fid}.jpg"
        if not f.exists():
            continue
        img = Image.open(f).convert("RGB")
        lat, lon = float(loc.loc[fid, "latitude"]), float(loc.loc[fid, "longitude"])
        cells, _ = chips_with_centres(img, lat, lon)
        E = embed_cached(emb, f"town_{fid}", lambda: [img.crop(b).resize((SIZE, SIZE), Image.BILINEAR) for b, _, _ in cells])
        towns.append({"fid": fid, "lat": lat, "lon": lon, "E": E, "centres": [(a, b) for _, a, b in cells]})

    rows, best = [], None
    for C in [0.01, 0.05, 0.2]:
        for pos_weight in [1, 3, 10]:
            w = np.where(y == 1, pos_weight * (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
            model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            model.fit(X, y, logisticregression__sample_weight=w)
            P = [model.predict_proba(t["E"])[:, 1] for t in towns]
            for rule in ["threshold", "top1"]:
                for th in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
                    found = topok = false = 0
                    for t, p in zip(towns, P):
                        s = sites(p, t["centres"], th, top1=(rule == "top1"))
                        hit = [x for x in s if haversine_m(x[0], x[1], t["lat"], t["lon"]) < 400]
                        found += bool(hit)
                        topok += bool(s) and haversine_m(s[0][0], s[0][1], t["lat"], t["lon"]) < 400
                        false += len(s) - len(hit)
                    n = len(towns)
                    r = {"C": C, "pos_weight": pos_weight, "rule": rule, "threshold": th, "towns": n,
                         "found": found, "top_correct": topok, "false_sites_per_town": round(false / n, 2)}
                    # score: correct top picks, penalising false sites
                    r["score"] = round(topok / n - 0.25 * false / n, 3)
                    rows.append(r)
                    if best is None or r["score"] > best[0]["score"]:
                        best = (r, model)
    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    res.to_csv(path("outputs_dir") / "ml_search_tuning.csv", index=False)
    joblib.dump(best[1], ML / "model_search.joblib")
    (path("outputs_dir") / "ml_search_best.json").write_text(json.dumps(best[0], indent=1))
    log.info("best: %s", best[0])
    print(res.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
