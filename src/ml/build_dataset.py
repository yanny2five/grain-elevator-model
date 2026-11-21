"""Phase 8 (proof of concept) - build a weakly labelled chip dataset.

Question: can imagery automatically tell "grain elevator present" from "absent"?

Training labels come for free, without hand labelling:
  POSITIVE  NAIP chips (300 m x 300 m) centred on facilities whose coordinates
            are confirmed on the structures (address HIGH + geocode STRUCTURE),
            from recent years (>= 2015) when the elevator certainly exists.
            Label noise: low - these sites were matched to company/registry
            records AND to mapped silos.
  NEGATIVE  NAIP chips of the same size and year taken 2.5-4 km away from each
            positive facility in a random direction (fields, farmsteads, towns).
            Label noise: a chip could occasionally contain another elevator.

The 8 prototype facilities with year-by-year human/preliminary labels are
EXCLUDED from training; they are the temporal test set.

Output: data/ml/chips/*.png (224x224) and data/ml/chips.csv
Usage:  python src/ml/build_dataset.py
"""
from __future__ import annotations

import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import earth_engine as gee  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("ml_build_dataset")
ML = PROJECT_ROOT / "data" / "ml"
CHIPS = ML / "chips"
CHIP_M = 300          # chip side on the ground
SIZE = 224            # network input size
RECENT_YEAR = 2015
NEG_PER_FAC = 3
random.seed(20261003)


def labelled_ids() -> set[str]:
    files = (PROJECT_ROOT / "data" / "reviews").glob("*.csv")
    return set(pd.concat([pd.read_csv(f, dtype=str) for f in files])["facility_id"])


def center_crop(img_path: Path, half_width_m: float) -> Image.Image:
    """Central CHIP_M x CHIP_M of a facility image covering 2*half_width_m."""
    im = Image.open(img_path).convert("RGB")
    frac = CHIP_M / (2 * half_width_m)
    w, h = im.size
    cw, ch = w * frac, h * frac
    box = (int((w - cw) / 2), int((h - ch) / 2), int((w + cw) / 2), int((h + ch) / 2))
    return im.crop(box).resize((SIZE, SIZE), Image.BILINEAR)


def offset_point(lat: float, lon: float, dist_m: float, bearing_deg: float) -> tuple[float, float]:
    b = math.radians(bearing_deg)
    dlat = dist_m * math.cos(b) / 111_320
    dlon = dist_m * math.sin(b) / (111_320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


def fetch_negative(task: dict) -> dict | None:
    try:
        region = gee.roi(task["lat"], task["lon"], CHIP_M / 2)
        b = gee.build("naip", task["year"], region)
        if gee.cloud_quality("naip", b["n_scenes"], b["clear_fraction"]) != "GOOD":
            return None
        px = int(round(CHIP_M / b["scale"]))
        url = gee.thumb_url(b["image"], "naip", region, px)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out = CHIPS / f"neg_{task['facility_id']}_{task['k']}_{task['year']}.png"
        out.write_bytes(r.content)
        Image.open(out).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR).save(out)
        return {"path": out.relative_to(PROJECT_ROOT).as_posix(), "label": 0, "facility_id": task["facility_id"],
                "year": task["year"], "kind": "offset_negative", "lat": task["lat"], "lon": task["lon"]}
    except Exception as e:  # noqa: BLE001
        log.warning("negative %s/%s failed: %s", task["facility_id"], task["k"], str(e)[:120])
        return None


def main() -> None:
    CHIPS.mkdir(parents=True, exist_ok=True)
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    pos_fac = loc[(loc["address_confidence"] == "HIGH") & (loc["geocode_confidence"] == "STRUCTURE")]
    held_out = labelled_ids()
    pos_fac = pos_fac[~pos_fac["facility_id"].isin(held_out)]
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["sensor"] == "naip") & (man["image_available"] == "True") & (man["status"] == "ok")]
    man = man.drop_duplicates(["facility_id", "year"], keep="last")
    log.info("positive facilities: %d (held out for temporal test: %d)", len(pos_fac), len(held_out))

    rows, neg_tasks = [], []
    for _, f in pos_fac.iterrows():
        m = man[man["facility_id"] == f["facility_id"]].copy()
        m["year"] = m["year"].astype(int)
        recent = m[m["year"] >= RECENT_YEAR]
        for _, r in recent.iterrows():
            out = CHIPS / f"pos_{f['facility_id']}_{r['year']}.png"
            if not out.exists():
                center_crop(PROJECT_ROOT / r["image_path"], float(r["roi_half_width_m"])).save(out)
            rows.append({"path": out.relative_to(PROJECT_ROOT).as_posix(), "label": 1,
                         "facility_id": f["facility_id"], "year": int(r["year"]), "kind": "facility_recent",
                         "lat": f["latitude"], "lon": f["longitude"]})
        # Hard negatives: the four corners of the newest facility image (centred 450 m
        # from the elevator, covering 300-600 m away) - town blocks, rail yards, other
        # industry - so the model must learn "elevator", not just "built-up vs field".
        if len(m):
            newest = m.sort_values("year").iloc[-1]
            im = Image.open(PROJECT_ROOT / newest["image_path"]).convert("RGB")
            w, h = im.size
            frac = CHIP_M / (2 * float(newest["roi_half_width_m"]))
            cw, ch = int(w * frac), int(h * frac)
            for k, (x0, y0) in enumerate([(0, 0), (w - cw, 0), (0, h - ch), (w - cw, h - ch)]):
                out = CHIPS / f"ctx_{f['facility_id']}_{k}_{newest['year']}.png"
                if not out.exists():
                    im.crop((x0, y0, x0 + cw, y0 + ch)).resize((SIZE, SIZE), Image.BILINEAR).save(out)
                rows.append({"path": out.relative_to(PROJECT_ROOT).as_posix(), "label": 0,
                             "facility_id": f["facility_id"], "year": int(newest["year"]),
                             "kind": "context_negative", "lat": f["latitude"], "lon": f["longitude"]})
        # Negatives: same years as the newest and an older NAIP flight, random direction 2.5-4 km away.
        years = sorted(m["year"])
        pick = {years[-1], years[len(years) // 2]} if years else set()
        for k in range(NEG_PER_FAC):
            lat, lon = offset_point(float(f["latitude"]), float(f["longitude"]),
                                    random.uniform(2500, 4000), random.uniform(0, 360))
            for y in pick:
                neg_tasks.append({"facility_id": f["facility_id"], "k": k, "year": y, "lat": lat, "lon": lon})

    gee.init()
    todo = [t for t in neg_tasks
            if not (CHIPS / f"neg_{t['facility_id']}_{t['k']}_{t['year']}.png").exists()]
    log.info("positives: %d chips; negatives to fetch: %d (of %d)", len(rows), len(todo), len(neg_tasks))
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(fetch_negative, todo):
            if res:
                rows.append(res)
    # Include negatives fetched in an earlier run.
    have = {r["path"] for r in rows}
    for t in neg_tasks:
        p = CHIPS / f"neg_{t['facility_id']}_{t['k']}_{t['year']}.png"
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        if p.exists() and rel not in have:
            rows.append({"path": rel, "label": 0, "facility_id": t["facility_id"], "year": t["year"],
                         "kind": "offset_negative", "lat": t["lat"], "lon": t["lon"]})
    df = pd.DataFrame(rows)
    df.to_csv(ML / "chips.csv", index=False)
    log.info("dataset: %s", df.groupby("label").size().to_dict())


if __name__ == "__main__":
    main()
