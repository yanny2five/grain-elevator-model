"""Where is the elevator in each date-check photo? (2026-11-14)

For every site on the date check page, scan the two newest NAIP photos with the
town-search detector (frozen ResNet-50 + model_search_v4.joblib) on 300 m chips
every 50 m, average the two score maps, and keep the score-weighted centre of the
best chips. The page draws that spot on every year, so the reviewer looks there
instead of searching the whole photo.

The scan covers the central 1 km of each 2 km photo (chip centres), so a spot can
fall outside the 600 m "wide" view; the page then says so (often a location problem).

Output: outputs/review/date_check_spots.json  {fid: {"x": .., "y": .., "p": ..}}
        x, y are fractions of the full photo (0.5, 0.5 = the mapped location).
Usage:  python src/ml/date_check_spots.py   (then python src/ml/date_check_page.py)
"""
import json
import os
import time
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SIZE  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/review/date_check_spots.json"
PHOTO_M = 2000          # each saved NAIP photo covers 2 km (the 15% centre crop is 300 m)
CHIP_M, STEP_M = 300, 100
SCAN_HALF_M = 500       # chip centres within 500 m of the mapped point
N_YEARS = 2
LIMIT = int(os.environ.get("SPOTS_LIMIT", "0"))


def grid(w):
    mpp = PHOTO_M / w
    side, step = CHIP_M / mpp, STEP_M / mpp
    lo, hi = w / 2 - SCAN_HALF_M / mpp, w / 2 + SCAN_HALF_M / mpp
    cs = np.arange(lo, hi + 1e-6, step)
    return [(cx, cy, side) for cy in cs for cx in cs]


def main():
    q = pd.read_csv(ROOT / "data/reviews/dating_review_queue.csv", dtype={"facility_id": str})
    by = pd.read_csv(ROOT / "outputs/ml_presence_by_year.csv", dtype={"facility_id": str})
    page = (ROOT / "outputs/review/date_check.html").read_text(encoding="utf-8")
    fids = sorted(set(q.facility_id) | set(by.facility_id))
    fids = [f for f in fids if f'"fid": "{f}"' in page or f'"fid":"{f}"' in page]
    done = json.loads(OUT.read_text()) if OUT.exists() else {}
    emb = GpuEmbedder()
    model = joblib.load(ROOT / "data/ml/model_search_v4.joblib")
    print(len(fids), "sites;", len(done), "already done", time.strftime("%H:%M:%S"), flush=True)
    for i, fid in enumerate(fids, 1):
        if fid in done:
            continue
        files = sorted((ROOT / "data/imagery" / fid / "naip").glob("*.png"))
        maps = []
        for f in files[::-1]:
            im = Image.open(f).convert("RGB")
            a = np.asarray(im)
            if (a.max(axis=2) < 10).mean() > 0.5:   # blank / no-data photo
                continue
            g = grid(im.size[0])
            crops = [im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2)))
                     .resize((SIZE, SIZE), Image.BILINEAR) for cx, cy, s in g]
            maps.append(model.predict_proba(emb(crops))[:, 1])
            w = im.size[0]
            if len(maps) == N_YEARS:
                break
        if not maps:
            continue
        p = np.mean(maps, axis=0)
        top = p >= max(p.max() * 0.85, p.max() - 0.1)
        wts = p[top]
        cx = float(np.average([g[k][0] for k in np.flatnonzero(top)], weights=wts)) / w
        cy = float(np.average([g[k][1] for k in np.flatnonzero(top)], weights=wts)) / w
        done[fid] = {"x": round(cx, 4), "y": round(cy, 4), "p": round(float(p.max()), 2)}
        if i % 10 == 0:
            OUT.write_text(json.dumps(done))
            print(i, "/", len(fids), time.strftime("%H:%M:%S"), flush=True)
        if LIMIT and i >= LIMIT:
            break
    OUT.write_text(json.dumps(done))
    print("wrote", OUT, len(done), "sites")


if __name__ == "__main__":
    main()
