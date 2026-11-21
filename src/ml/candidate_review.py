"""'AI suggests, person picks' - candidate spots for checking located facilities.

For each facility, slide 300 m chips (every STRIDE_M) over its latest NAIP image
(1.2 km square around the recorded point), score them with the town-search
model, and keep the top N local maxima (at least MERGE_M apart) as candidates
A, B, C. A reviewer then picks the candidate on the elevator, keeps the
recorded point, or rejects all.

Outputs
  data/ml/candidates.csv                 facility_id, cand, lat, lon, p, dist_m
  outputs/review/candidates/<id>.jpg     image with a 100 m grid, the recorded
                                         point (red cross) and candidates A/B/C
Usage:  python src/ml/candidate_review.py [--ids F00001 ...]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SIZE  # noqa: E402
from find_addresses import haversine_m  # noqa: E402
from train_eval import Embedder  # noqa: E402
from utils import PROJECT_ROOT, get_logger, load_config, path  # noqa: E402

log = get_logger("candidate_review")
ML = PROJECT_ROOT / "data" / "ml"
OUT = PROJECT_ROOT / "outputs" / "review" / "candidates"
CHIP_M, STRIDE_M, MERGE_M, TOP_N = 300, 100, 200, 3
LETTERS = "ABC"


class GpuEmbedder(Embedder):
    """Same frozen ResNet-50, run on the GPU when available (inference only, light load)."""

    def __init__(self):
        super().__init__()
        import torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.dev)

    def __call__(self, images, batch: int = 32):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + batch]]).to(self.dev)
                out.append(self.model(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def latest_naip(fid: str) -> Path | None:
    fs = sorted((PROJECT_ROOT / "data" / "imagery" / fid / "naip").glob("*.png"))
    return fs[-1] if fs else None


def chips(img: Image.Image, half_m: float):
    w = img.size[0]
    mpp = 2 * half_m / w
    side, step = CHIP_M / mpp, STRIDE_M / mpp
    out, y = [], 0.0
    while y + side <= w + 1e-6:
        x = 0.0
        while x + side <= w + 1e-6:
            out.append((x + side / 2, y + side / 2, img.crop((int(x), int(y), int(x + side), int(y + side)))))
            x += step
        y += step
    return out, mpp


def to_latlon(px, py, w, mpp, lat0, lon0):
    dx, dy = (px - w / 2) * mpp, (w / 2 - py) * mpp
    return lat0 + dy / 111_320, lon0 + dx / (111_320 * math.cos(math.radians(lat0)))


def draw(img, cands, mpp, fid, title):
    S = 800
    im = img.convert("RGB").resize((S, S))
    k = S / img.size[0]
    dr = ImageDraw.Draw(im)
    for m in range(-600, 601, 100):
        p = S / 2 + m / mpp * k
        dr.line([(p, 0), (p, S)], fill=(255, 255, 255), width=1)
        dr.line([(0, p), (S, p)], fill=(255, 255, 255), width=1)
    c = S / 2
    for a, b in [((c - 22, c), (c - 7, c)), ((c + 7, c), (c + 22, c)), ((c, c - 22), (c, c - 7)), ((c, c + 7), (c, c + 22))]:
        dr.line([a, b], fill="red", width=3)
    for L, (px, py, p) in zip(LETTERS, cands):
        x, y = px * k, py * k
        dr.ellipse([x - 14, y - 14, x + 14, y + 14], outline="yellow", width=3)
        dr.rectangle([x + 12, y - 22, x + 52, y - 4], fill="black")
        dr.text((x + 15, y - 20), f"{L} {p:.2f}", fill="yellow")
    dr.rectangle([0, 0, S, 18], fill="black")
    dr.text((4, 3), f"{fid} {title}", fill="white")
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    half_m = float(load_config()["earth_engine"]["roi_half_width_m"])
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    loc = loc[loc["clip_ok"] == "True"]
    if args.ids:
        loc = loc[loc["facility_id"].isin(args.ids)]
    model = joblib.load(ML / "model_search_v4.joblib") if (ML / "model_search_v4.joblib").exists() else joblib.load(ML / "model_search.joblib")
    emb = GpuEmbedder()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, r in enumerate(loc.itertuples(), 1):
        f = latest_naip(r.facility_id)
        if f is None:
            continue
        img = Image.open(f).convert("RGB")
        cs, mpp = chips(img, half_m)
        X = emb([c.resize((SIZE, SIZE), Image.BILINEAR) for _, _, c in cs])
        p = model.predict_proba(X)[:, 1]
        lat0, lon0 = float(r.latitude), float(r.longitude)
        picked = []
        for i in np.argsort(-p):
            px, py, _ = cs[i]
            if all(math.hypot(px - a, py - b) * mpp >= MERGE_M for a, b, _ in picked):
                picked.append((px, py, float(p[i])))
            if len(picked) == TOP_N:
                break
        w = img.size[0]
        for L, (px, py, pp) in zip(LETTERS, picked):
            la, lo = to_latlon(px, py, w, mpp, lat0, lon0)
            rows.append({"facility_id": r.facility_id, "cand": L, "lat": round(la, 6), "lon": round(lo, 6),
                         "p": round(pp, 3), "dist_m": round(haversine_m(lat0, lon0, la, lo))})
        draw(img, picked, mpp, r.facility_id, f"{r.city} {r.state} {f.stem} | {r.warehouse_name}"[:90]).save(
            OUT / f"{r.facility_id}.jpg", quality=85)
        if n % 25 == 0:
            log.info("%d/%d", n, len(loc))
            pd.DataFrame(rows).to_csv(ML / "candidates.csv", index=False)
    pd.DataFrame(rows).to_csv(ML / "candidates.csv", index=False)
    log.info("done: %d facilities", len(loc))


if __name__ == "__main__":
    main()
