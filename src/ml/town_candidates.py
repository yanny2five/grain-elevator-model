"""'AI suggests, person picks' for UNLOCATED facilities (road-buffer search).

For every town with unlocated USWA facilities in the chosen state(s):
  1. one recent NAIP image, 4 km x 4 km around the Census centroid of the town
     (same as town_scan.py);
  2. a road mask for the same square: Census TIGER 2016 public roads (S1100,
     S1200, S1400), buffered by ROAD_BUFFER_M (250 m keeps ~98% of known
     elevators - road_distance_test.py);
  3. 300 m chips every 200 m, scored by the town-search model
     (data/ml/model_search_v4.joblib, else v3 model_search.joblib); chips off the road buffer are dropped;
  4. the best TOP_N spots at least MERGE_M apart become candidates A-D,
     whatever their score - this is a shortlist for a person, not a decision.
Nothing is written to the location evidence here: a person picks the right
candidate on outputs/review/town_pick_<STATE>.html and the picks are merged
later (town_pick_merge.py).

Outputs
  data/ml/town_candidates.csv                     one row per town x candidate
  outputs/review/town_candidates/<ST>/<town>.jpg  overview + candidate close-ups
  outputs/review/town_pick_<ST>.html              review page
Usage:  python src/ml/town_candidates.py --state SD [--limit 5]
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import sys
from pathlib import Path

import ee
import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import earth_engine as gee  # noqa: E402
from build_dataset import CHIP_M, SIZE  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from clean_dataset import _norm_place  # noqa: E402
from find_addresses import haversine_m, load_places  # noqa: E402
from town_resolve import _squeeze_index, resolve  # noqa: E402
from town_scan import HALF_M, STRIDE_M, latest_naip  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("town_candidates")
ML = PROJECT_ROOT / "data" / "ml"
IMG_DIR = ML / "town_scans"
REVIEW = PROJECT_ROOT / "outputs" / "review"
ROAD_BUFFER_M, MERGE_M, TOP_N = 500, 350, 4  # buffer 250 -> 500 m on 2026-10-26 (advisor; 100% of trusted elevators within 500 m)
PUBLIC = ["S1100", "S1200", "S1400"]
LETTERS = "ABCD"


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def road_mask(lat: float, lon: float, px: int) -> np.ndarray:
    """Boolean px x px mask: True within ROAD_BUFFER_M of a public road (same square/projection as the NAIP scan)."""
    region = gee.roi(lat, lon, HALF_M)
    roads = ee.FeatureCollection("TIGER/2016/Roads").filter(ee.Filter.inList("mtfcc", PUBLIC)).filterBounds(
        region.buffer(ROAD_BUFFER_M + 50))
    near = roads.distance(ROAD_BUFFER_M + 50).lte(ROAD_BUFFER_M).unmask(0).rename("m")
    url = near.visualize(min=0, max=1, palette=["000000", "ffffff"]).getThumbURL(
        {"region": region, "dimensions": f"{px}x{px}", "crs": "EPSG:3857", "format": "png"})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("L")) > 127


def chips(img: Image.Image, mask: np.ndarray, lat0: float, lon0: float):
    w = img.size[0]
    mpp = 2 * HALF_M / w
    side, step = CHIP_M / mpp, STRIDE_M / mpp
    mk = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, w))) > 127
    out, y = [], 0.0
    while y + side <= w:
        x = 0.0
        while x + side <= w:
            cx, cy = x + side / 2, y + side / 2
            la = lat0 - (cy - w / 2) * mpp / 111_320
            lo = lon0 + (cx - w / 2) * mpp / (111_320 * math.cos(math.radians(lat0)))
            out.append({"box": (int(x), int(y), int(x + side), int(y + side)), "px": (cx, cy), "lat": la, "lon": lo,
                        "road": bool(mk[int(cy), int(cx)])})
            x += step
        y += step
    return out, mpp


def sheet(img: Image.Image, mask_img: np.ndarray, cands: list[dict], mpp: float, title: str) -> Image.Image:
    O = 900
    ov = img.resize((O, O))
    k = O / img.size[0]
    m = Image.fromarray((~np.asarray(Image.fromarray(mask_img.astype(np.uint8) * 255).resize((O, O))).astype(bool)
                         ).astype(np.uint8) * 90)
    shade = Image.new("RGB", (O, O), (0, 0, 0))
    ov = Image.composite(shade, ov, m)  # darken outside the road buffer
    d = ImageDraw.Draw(ov)
    for L, c in zip(LETTERS, cands):
        x, y = c["px"][0] * k, c["px"][1] * k
        d.ellipse([x - 16, y - 16, x + 16, y + 16], outline="yellow", width=3)
        d.rectangle([x + 14, y - 26, x + 34, y - 6], fill="black")
        d.text((x + 19, y - 23), L, fill="yellow")
    d.rectangle([0, 0, O, 18], fill="black")
    d.text((4, 3), title[:130], fill="white")
    Z = 450
    out = Image.new("RGB", (O + 2 * Z, O), "white")
    out.paste(ov, (0, 0))
    half = 300 / mpp  # 600 m close-up
    for j, (L, c) in enumerate(zip(LETTERS, cands)):
        cx, cy = c["px"]
        z = img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half))).resize((Z, Z))
        dz = ImageDraw.Draw(z)
        dz.rectangle([0, 0, Z, 20], fill="black")
        dz.text((4, 4), f"{L}  p={c['p']:.2f}  {'' if c['road'] else '(off road buffer)'}", fill="yellow")
        dz.line([(Z / 2 - 10, Z / 2), (Z / 2 + 10, Z / 2)], fill="red", width=2)
        dz.line([(Z / 2, Z / 2 - 10), (Z / 2, Z / 2 + 10)], fill="red", width=2)
        out.paste(z, (O + (j % 2) * Z, (j // 2) * Z))
    return out


def write_page(st: str, rows: pd.DataFrame) -> Path:
    towns = []
    for (town, fids), g in rows.groupby(["town", "facility_ids"], sort=True):
        towns.append({"town": town, "fids": fids, "names": g["names"].iloc[0], "img": g["image"].iloc[0],
                      "cands": [{"c": r.cand, "lat": r.lat, "lon": r.lon, "p": r.p} for r in g.itertuples()]})
    data = json.dumps(towns)
    page = REVIEW / f"town_pick_{st}.html"
    page.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Pick the elevator - {st}</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
.t{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 18px}}
.t img{{max-width:100%;border-radius:4px}} label{{margin-right:14px;font-size:15px}}
.bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ddd;z-index:2}}
button{{font-size:15px;padding:6px 12px}} .done{{border-color:#2a7}}</style></head><body>
<div class="bar"><b>Pick the elevator - {st}</b> &nbsp; <span id="n"></span> &nbsp;
<button onclick="dl()">Download picks (CSV)</button> &nbsp; Save the file into
<code>data/reviews/town_picks/</code>. Yellow circles = AI suggestions; darker areas are &gt;{ROAD_BUFFER_M} m from a road.</div>
<div id="list"></div>
<script>
const T={data}; const KEY="town_pick_{st}_byname";
let P={{}}; try{{P=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(P))}}catch(e){{}}; cnt()}}
function cnt(){{document.getElementById("n").textContent=T.filter(t=>P[t.town]).length+" / "+T.length+" towns picked"}}
const L=document.getElementById("list");
T.forEach((t,i)=>{{const d=document.createElement("div");d.className="t"+(P[t.town]?" done":"");
 let h=`<div><b>${{t.town}}</b> - ${{t.names}} <small>(${{t.fids}})</small></div><img loading="lazy" src="${{t.img}}"><div>`;
 for(const o of [...t.cands.map(c=>c.c),"none","unsure"]) h+=`<label><input type="radio" name="r${{i}}" value="${{o}}" ${{P[t.town]==o?"checked":""}}> ${{o}}</label>`;
 d.innerHTML=h+"</div>"; d.querySelectorAll("input").forEach(x=>x.onchange=()=>{{P[t.town]=x.value;d.classList.add("done");save()}}); L.appendChild(d)}});
function dl(){{let s="town,facility_ids,pick,lat,lon,p\\n";
 T.forEach((t,i)=>{{if(!P[t.town])return;const c=t.cands.find(c=>c.c==P[t.town]);
  s+=`"${{t.town}}","${{t.fids}}",${{P[t.town]}},${{c?c.lat:""}},${{c?c.lon:""}},${{c?c.p:""}}\\n`}});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{{type:"text/csv"}}));
 a.download="town_picks_{st}.csv";a.click()}}
cnt();
</script></body></html>""", encoding="utf-8")
    return page


def scan_centre(st, key, label, fids, names, lat0, lon0, model, emb) -> list[dict]:
    """Search one 4 km square around (lat0, lon0); return candidate rows (or one status row)."""
    base = {"state": st, "town": label, "facility_ids": fids, "names": names}
    try:
        f = IMG_DIR / f"tc_{st}_{slug(key)}.jpg"
        if f.exists():
            img = Image.open(f).convert("RGB")
            year = int(f.with_suffix(".txt").read_text()) if f.with_suffix(".txt").exists() else None
        else:
            img, year = latest_naip(lat0, lon0)
            if img is None:
                return [{**base, "cand": "", "status": "no good NAIP"}]
            img.save(f, quality=90)
            f.with_suffix(".txt").write_text(str(year))
        mask = road_mask(lat0, lon0, 400)
        cs, mpp = chips(img, mask, lat0, lon0)
        p = model.predict_proba(emb([img.crop(c["box"]).resize((SIZE, SIZE), Image.BILINEAR) for c in cs]))[:, 1]
        p_eff = np.where([c["road"] for c in cs], p, 0.0)
        picked = []
        for k in np.argsort(-p_eff):
            if all(haversine_m(cs[k]["lat"], cs[k]["lon"], q["lat"], q["lon"]) >= MERGE_M for q in picked):
                picked.append({**cs[k], "p": float(p[k])})
            if len(picked) == TOP_N:
                break
        sd = REVIEW / "town_candidates" / st
        sd.mkdir(parents=True, exist_ok=True)
        imgf = sd / f"{slug(key)}.jpg"
        sheet(img, mask, picked, mpp, f"{label}, {st}  NAIP {year}  | {names}").save(imgf, quality=82)
        return [{**base, "cand": L, "lat": round(c["lat"], 6), "lon": round(c["lon"], 6), "p": round(c["p"], 3),
                 "dist_from_centroid_m": round(haversine_m(lat0, lon0, c["lat"], c["lon"])),
                 "naip_year": year, "image": f"town_candidates/{st}/{imgf.name}", "status": "candidates"}
                for L, c in zip(LETTERS, picked)]
    except Exception as e:  # noqa: BLE001
        log.error("%s %s failed: %s", st, label, e)
        return [{**base, "cand": "", "status": f"error: {e}"[:200]}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", nargs="+", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-missing", action="store_true",
                    help="only towns whose earlier status was not 'candidates' (name not found, errors)")
    a = ap.parse_args()
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    located = set(loc.loc[loc["clip_ok"] == "True", "facility_id"])
    fac = master[(master["is_uswa"] == "True") & ~master["facility_id"].isin(located) & master["state"].isin(a.state)]
    places = load_places()
    model = joblib.load(ML / "model_search_v4.joblib") if (ML / "model_search_v4.joblib").exists() else joblib.load(ML / "model_search.joblib")
    emb = GpuEmbedder()
    gee.init()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = ML / "town_candidates.csv"
    old = pd.read_csv(out_csv, dtype=str, keep_default_na=False) if out_csv.exists() else pd.DataFrame()
    rows = []
    towns = list(fac.groupby(["state", "city"]))
    if a.retry_missing and len(old):
        ok = old[old["status"] == "candidates"]
        done = set(zip(ok["state"], ok["town"].str.split(" -> ").str[0]))
        towns = [t for t in towns if (t[0][0], t[0][1]) not in done]
    squeezed = _squeeze_index(places)
    if a.limit:
        towns = towns[:a.limit]
    log.info("%d unlocated facilities in %d towns", len(fac), len(towns))
    for i, ((st, city), g) in enumerate(towns, 1):
        fids = ";".join(g["facility_id"])
        names = "; ".join(sorted(set(g["warehouse_name"])))[:120]
        county = g["county"].iloc[0] if "county" in g else ""
        # "HURON (EAST)", "Aberdeen (West)": the parenthesis names a site within the town
        place = (places.get((st, _norm_place(city))) or places.get((st, _norm_place(re.sub(r"\s*\(.*\)", "", city))))
                 if city else None)
        centres = ([{"name": city, "lat": place[0], "lon": place[1], "how": "Census place"}] if place
                   else resolve(st, city, county, places, squeezed))
        if not centres:
            rows.append({"state": st, "town": city, "facility_ids": fids, "names": names, "cand": "",
                         "status": "town not found (Census places, OpenStreetMap)"})
            continue
        for c0 in centres:
            # a damaged name can give several possible towns; each is searched and shown separately
            label = city if c0["how"] == "Census place" else f"{city} -> {c0['name']} ({c0['how']})"
            key = city if c0["how"] == "Census place" else f"{city}__{c0['name']}"
            rows += scan_centre(st, key, label, fids, names, float(c0["lat"]), float(c0["lon"]), model, emb)
        if i % 10 == 0:
            log.info("%d/%d towns", i, len(towns))
    new = pd.DataFrame(rows)
    if len(old) and a.retry_missing:  # replace only the towns retried (matched by the original WCMD town name)
        retried = set(zip(new["state"], new["town"].str.split(" -> ").str[0]))
        old = old[[(s_, t.split(" -> ")[0]) not in retried for s_, t in zip(old["state"], old["town"])]]
    elif len(old):
        old = old[~old["state"].isin(a.state)]
    allr = pd.concat([old, new], ignore_index=True)
    # drop shortlists whose facilities have all been located since (e.g. by a new company location list)
    allr = allr[[not set(str(f).split(";")) <= located for f in allr["facility_ids"]]]
    allr.to_csv(out_csv, index=False)
    for st in a.state:
        r = allr[(allr["state"] == st) & (allr["status"] == "candidates")]
        if len(r):
            log.info("review page: %s", write_page(st, r))
    log.info("status: %s", new.drop_duplicates(["state", "town"])["status"].str[:30].value_counts().to_dict())


if __name__ == "__main__":
    main()
