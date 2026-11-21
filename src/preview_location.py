"""Location check: save a CURRENT aerial snapshot around candidate coordinates.

This tool only confirms *where* a facility is (e.g. which of several silo
clusters is the shuttle terminal). It is not part of the historical analysis;
that uses Google Earth Engine (earth_engine.py).

Imagery: Esri World Imagery export service (current mosaic, no API key).
Use for research verification only; cite "Esri World Imagery" if shown.

Usage:
  python src/preview_location.py --lat 48.40 --lon -97.75 --half-width-m 4000 --out tmp.png
  python src/preview_location.py --facility F04448            # uses facility_addresses.csv
  python src/preview_location.py --candidates F03205          # one tile per candidate
"""
from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, path  # noqa: E402

log = get_logger("preview_location")
EXPORT = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"


def snapshot(lat: float, lon: float, half_m: float, out: Path, px: int = 900,
             marks: list[tuple[float, float, str]] | None = None) -> Path:
    dlat = half_m / 111_320
    dlon = half_m / (111_320 * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    r = requests.get(EXPORT, params={"bbox": ",".join(f"{v:.6f}" for v in bbox), "bboxSR": 4326,
                                     "imageSR": 4326, "size": f"{px},{px}", "format": "jpg", "f": "image"},
                     headers={"User-Agent": load_config()["geocode"]["user_agent"]}, timeout=120)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    d = ImageDraw.Draw(img)
    for mlat, mlon, label in (marks or [(lat, lon, "")]):
        x = (mlon - bbox[0]) / (bbox[2] - bbox[0]) * px
        y = (bbox[3] - mlat) / (bbox[3] - bbox[1]) * px
        d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(255, 0, 0), width=3)
        if label:
            d.text((x + 12, y - 8), label, fill=(255, 255, 0))
    bar_px = px * 500 / (2 * half_m)  # 500 m scale bar
    d.rectangle([20, px - 30, 20 + bar_px, px - 24], fill=(255, 255, 255))
    d.text((20, px - 48), "500 m", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=90)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--facility")
    ap.add_argument("--candidates", help="facility_id: mark every candidate on one overview")
    ap.add_argument("--half-width-m", type=float, default=600)
    ap.add_argument("--out")
    a = ap.parse_args()
    pdir = path("addresses_dir") / "previews"
    if a.candidates:
        c = pd.read_csv(path("addresses_dir") / "location_candidates.csv", dtype=str, keep_default_na=False)
        c = c[c["facility_id"] == a.candidates].reset_index(drop=True)
        lats, lons = c["lat"].astype(float), c["lon"].astype(float)
        marks = [(la, lo, str(i)) for i, (la, lo) in enumerate(zip(lats, lons))]
        out = Path(a.out) if a.out else pdir / f"{a.candidates}_candidates.jpg"
        half = max(a.half_width_m, 1.15 * max(
            (lats.max() - lats.min()) * 111_320 / 2,
            (lons.max() - lons.min()) * 111_320 * math.cos(math.radians(lats.mean())) / 2, 500))
        print(snapshot(lats.mean(), lons.mean(), half, out, marks=marks))
        print(c[["source", "kind", "name", "lat", "lon", "n_silos"]].to_string())
        return
    if a.facility:
        f = pd.read_csv(path("addresses_dir") / "facility_addresses.csv", dtype=str, keep_default_na=False)
        row = f[f["facility_id"] == a.facility].iloc[0]
        a.lat, a.lon = float(row["latitude"]), float(row["longitude"])
    out = Path(a.out) if a.out else pdir / f"{a.facility or f'{a.lat:.4f}_{a.lon:.4f}'}.jpg"
    print(snapshot(a.lat, a.lon, a.half_width_m, out))


if __name__ == "__main__":
    main()
