"""Phase 5 - export one image per facility x year x sensor, resumable.

Output layout
-------------
data/imagery/<facility_id>/<sensor>/<year>.png    native resolution, same ROI every year
data/imagery/manifest.csv                          one row per facility x sensor x year

Manifest columns: facility_id, sensor, year, image_available, n_scenes,
clear_fraction, cloud_quality, satellite_source, first_date, last_date,
scale_m, image_path, roi_lat, roi_lon, roi_half_width_m, status, error.

A year with no scenes (or no clear pixels) is written with
image_available=False and no file - never borrowed from another year.

outputs/imagery_index.csv picks the representative image per facility-year:
NAIP (sharpest) > Sentinel-2 > Landsat, skipping cloud_quality NONE/POOR when
a better one exists.

Images are fetched with getThumbURL (small ROI, direct download) instead of
Drive export tasks: ~1-3 s per image, no task queue, easy to resume.

Usage:
  python src/export_yearly_imagery.py --subset prototype
  python src/export_yearly_imagery.py --ids F04448 --sensors naip landsat
  python src/export_yearly_imagery.py --subset all --workers 8     # full run
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from utils import get_logger, load_config, path  # noqa: E402

log = get_logger("export_yearly_imagery")
CFG = load_config()["earth_engine"]
FIELDS = ["facility_id", "sensor", "year", "image_available", "n_scenes", "clear_fraction", "cloud_quality",
          "satellite_source", "first_date", "last_date", "scale_m", "image_path", "roi_lat", "roi_lon",
          "roi_half_width_m", "status", "error"]
SENSORS = ["naip", "sentinel2", "landsat"]
_lock = threading.Lock()


def manifest_path() -> Path:
    return path("imagery_dir") / "manifest.csv"


def load_done(current: dict[str, tuple[float, float]] | None = None) -> set[tuple[str, str, int]]:
    """Images already fetched successfully *at the facility's current coordinates*.

    If a facility's location was corrected later (e.g. a better snap), its old
    images cover the wrong square; they no longer count as done and are refetched
    (the newest manifest row wins downstream).
    """
    f = manifest_path()
    if not f.exists():
        return set()
    m = pd.read_csv(f, dtype=str, keep_default_na=False)
    m = m[m["status"] == "ok"].drop_duplicates(["facility_id", "sensor", "year"], keep="last")
    done = set()
    for r in m.itertuples():
        if current and r.facility_id in current and r.roi_lat:
            lat, lon = current[r.facility_id]
            moved_m = ((float(r.roi_lat) - lat) * 111_320) ** 2 + \
                      ((float(r.roi_lon) - lon) * 111_320 * 0.75) ** 2
            if moved_m > 50 ** 2:
                continue
        if r.roi_half_width_m and float(r.roi_half_width_m) != float(CFG["roi_half_width_m"]):
            continue  # ROI size changed since this image was fetched -> refetch
        done.add((r.facility_id, r.sensor, int(r.year)))
    return done


def append(row: dict) -> None:
    f = manifest_path()
    with _lock:
        new = not f.exists()
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def one(fid: str, lat: float, lon: float, sensor: str, year: int) -> dict:
    half = CFG["roi_half_width_m"]
    region = gee.roi(lat, lon, half)
    row = {"facility_id": fid, "sensor": sensor, "year": year, "roi_lat": lat, "roi_lon": lon,
           "roi_half_width_m": half, "status": "ok", "error": ""}
    try:
        b = gee.build(sensor, year, region)
        q = gee.cloud_quality(sensor, b["n_scenes"], b["clear_fraction"])
        row.update(n_scenes=b["n_scenes"], clear_fraction=round(b["clear_fraction"] or 0, 3), cloud_quality=q,
                   satellite_source=";".join(b["sources"]), first_date=b["first"], last_date=b["last"],
                   scale_m=b["scale"], image_available=q != "NONE", image_path="")
        if q != "NONE":
            px = int(round(2 * half / b["scale"]))
            url = gee.thumb_url(b["image"], sensor, region, px)
            out = path("imagery_dir") / fid / sensor / f"{year}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            out.write_bytes(r.content)
            row["image_path"] = out.relative_to(path("imagery_dir").parent.parent).as_posix()
    except Exception as e:  # noqa: BLE001 - record and move on
        row.update(status="error", error=str(e)[:300], image_available=False)
    append(row)
    return row


def build_index() -> pd.DataFrame:
    m = pd.read_csv(manifest_path(), dtype=str, keep_default_na=False)
    m = m[m["status"] == "ok"].drop_duplicates(["facility_id", "sensor", "year"], keep="last")
    rank_s = {"naip": 0, "sentinel2": 1, "landsat": 2}
    rank_q = {"GOOD": 0, "FAIR": 1, "POOR": 2, "NONE": 9}
    m["_q"] = m["cloud_quality"].map(rank_q)
    m["_s"] = m["sensor"].map(rank_s)
    best = (m[m["image_available"] == "True"].assign(_qq=lambda d: (d["_q"] >= 2).astype(int))
            .sort_values(["facility_id", "year", "_qq", "_s", "_q"])
            .drop_duplicates(["facility_id", "year"]))
    years = m[["facility_id", "year"]].drop_duplicates()
    idx = years.merge(best[["facility_id", "year", "sensor", "cloud_quality", "image_path"]],
                      on=["facility_id", "year"], how="left").fillna("")
    idx["image_available"] = idx["image_path"].ne("")
    idx = idx.rename(columns={"sensor": "satellite_source"}).sort_values(["facility_id", "year"])
    idx.to_csv(path("outputs_dir") / "imagery_index.csv", index=False)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="prototype", choices=["prototype", "all"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--sensors", nargs="*", default=SENSORS, choices=SENSORS)
    ap.add_argument("--last-year", type=int, default=dt.date.today().year)
    ap.add_argument("--workers", type=int, default=6)  # failed images are retried automatically on rerun
    a = ap.parse_args()

    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    if a.ids:
        loc = loc[loc["facility_id"].isin(a.ids)]
    elif a.subset == "prototype":
        proto = pd.read_csv(path("processed_dir") / "prototype_facilities.csv", dtype=str)
        loc = loc[loc["facility_id"].isin(proto["facility_id"])]
    skipped = loc[loc["clip_ok"] != "True"]
    for fid in skipped["facility_id"]:
        log.warning("%s skipped: location not clip_ok (resolve address first)", fid)
    loc = loc[loc["clip_ok"] == "True"]

    gee.init()
    done = load_done({r.facility_id: (float(r.latitude), float(r.longitude)) for r in loc.itertuples()})
    tasks = [(r.facility_id, float(r.latitude), float(r.longitude), s, y)
             for r in loc.itertuples() for s in a.sensors for y in gee.sensor_years(s, a.last_year)
             if (r.facility_id, s, y) not in done]
    log.info("%d facilities, %d images to fetch (%d already done)", len(loc), len(tasks), len(done))

    n_err = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one, *t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r["status"] != "ok":
                n_err += 1
                log.error("%s %s %s: %s", r["facility_id"], r["sensor"], r["year"], r["error"][:150])
            if i % 25 == 0 or i == len(futs):
                log.info("progress %d/%d (errors %d)", i, len(futs), n_err)
    if manifest_path().exists():
        idx = build_index()
        log.info("index: %d facility-years, %d with an image", len(idx), idx["image_available"].sum())


if __name__ == "__main__":
    main()
