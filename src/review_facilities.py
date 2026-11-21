"""Phase 6 - human review: chronological image strips + per-image labels.

Builds one self-contained HTML page per facility (outputs/review/<id>.html)
plus an index page. Each page shows NAIP, Sentinel-2 and Landsat images in
chronological rows, all covering the same ground square, with a crosshair on
the facility point. The reviewer labels each image:

    A  absent        - no elevator/bins on site
    C  construction  - site work, foundations, partial structures
    P  present       - elevator/bins clearly standing
    R  removed       - structures gone after having been present
    ?  unclear       - resolution, cloud, or ambiguity prevents a call

Labels autosave in the browser (localStorage) and are exported with the
"Download labels CSV" button. Save that file into data/reviews/ (any name);
estimate_build_year.py reads every CSV there.

Usage:  python src/review_facilities.py [--ids F04448 ...]
        then open outputs/review/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("review_facilities")
SENSOR_LABEL = {"naip": "NAIP aerial (~1 m)", "sentinel2": "Sentinel-2 (10 m)", "landsat": "Landsat (30 m)"}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Review __FID__</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#ddd;--A:#3b6fd1;--C:#d18b1f;--P:#2e9b55;--R:#b03a3a;--U:#888}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--fg:#eee;--muted:#a3a3a0;--card:#232326;--line:#3a3a3e}}
body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}
header{padding:16px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:18px;margin:0 0 4px}.meta{color:var(--muted);font-size:13px}
main{padding:12px 20px 60px}h2{font-size:15px;margin:22px 0 8px}
.strip{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}
.card{flex:0 0 auto;width:210px;background:var(--card);border:2px solid var(--line);border-radius:8px;padding:6px}
.card.l-A{border-color:var(--A)}.card.l-C{border-color:var(--C)}.card.l-P{border-color:var(--P)}.card.l-R{border-color:var(--R)}.card.l-U{border-color:var(--U)}
.imgwrap{position:relative;width:196px;height:196px;background:#000;cursor:zoom-in}
.imgwrap img{width:100%;height:100%;image-rendering:pixelated;display:block}
.imgwrap:after{content:"";position:absolute;left:50%;top:50%;width:22px;height:22px;margin:-11px;border:2px solid #f33;border-radius:50%}
.none{display:flex;align-items:center;justify-content:center;color:#999;height:196px;background:repeating-linear-gradient(45deg,#0000,#0000 6px,#8882 6px,#8882 12px)}
.yr{display:flex;justify-content:space-between;font-weight:600;margin:4px 2px}.q{font-weight:400;color:var(--muted);font-size:12px}
.btns{display:flex;gap:3px}.btns button{flex:1;padding:4px 0;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:4px;cursor:pointer;font-weight:600}
.btns button.on[data-l=A]{background:var(--A);color:#fff}.btns button.on[data-l=C]{background:var(--C);color:#fff}
.btns button.on[data-l=P]{background:var(--P);color:#fff}.btns button.on[data-l=R]{background:var(--R);color:#fff}.btns button.on[data-l=U]{background:var(--U);color:#fff}
.bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:8px}
.bar button{padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
textarea{width:100%;max-width:900px;min-height:50px;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
.legend span{display:inline-block;margin-right:12px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}
#zoom{position:fixed;inset:0;background:#000c;display:none;align-items:center;justify-content:center;z-index:9}
#zoom img{max-width:92vw;max-height:92vh;image-rendering:pixelated}#zoom.on{display:flex}
</style></head><body>
<header><h1>__FID__ &middot; __NAME__</h1>
<div class="meta">__META__</div>
<div class="bar legend"><span><i class="dot" style="background:var(--A)"></i>A absent</span><span><i class="dot" style="background:var(--C)"></i>C construction</span>
<span><i class="dot" style="background:var(--P)"></i>P present</span><span><i class="dot" style="background:var(--R)"></i>R removed</span><span><i class="dot" style="background:var(--U)"></i>? unclear</span>
<button id="dl">Download labels CSV</button><span id="count" class="meta"></span></div>
<div class="meta" id="pre"></div></header>
<main>
<p class="meta">Every image shows the same __SIZE__ m square; the red circle marks the facility point. Click an image to enlarge.
Label what you can see. Use "?" when resolution or cloud prevents a call.</p>
__ROWS__
<h2>Notes</h2><textarea id="notes" placeholder="e.g. bins visible 2006 NAIP; Landsat cannot resolve before 2003; neighbouring elevator to the east"></textarea>
<div class="bar"><label>Reviewer <input id="reviewer" size="18"></label></div>
</main><div id="zoom"><img alt=""></div>
<script>
const FID="__FID__", KEY="review:"+FID;
// Labels already saved in data/reviews/ (e.g. a preliminary pass) are the
// starting point; anything the reviewer changes in this browser takes priority.
const PRELOAD=__PRELOAD__;
let st={labels:Object.assign({},PRELOAD.labels),notes:PRELOAD.notes||"",reviewer:"",from:PRELOAD.reviewer||""};
try{const saved=JSON.parse(localStorage.getItem(KEY)||"null");if(saved){st=Object.assign(st,saved,{labels:Object.assign({},PRELOAD.labels,saved.labels)})}}catch(e){}
try{st.reviewer=st.reviewer||localStorage.getItem("reviewer")||""}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st));localStorage.setItem("reviewer",st.reviewer)}catch(e){};count()};
function paint(card){const k=card.dataset.k,l=st.labels[k]||"";card.className="card"+(l?" l-"+l:"");
 card.querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.l===l))}
document.querySelectorAll(".card").forEach(card=>{paint(card);card.querySelectorAll("button").forEach(b=>b.onclick=()=>{
 const k=card.dataset.k;st.labels[k]=st.labels[k]===b.dataset.l?"":b.dataset.l;paint(card);save()})});
document.querySelectorAll(".imgwrap").forEach(w=>w.onclick=()=>{const z=document.getElementById("zoom");z.querySelector("img").src=w.querySelector("img").src;z.classList.add("on")});
document.getElementById("zoom").onclick=e=>e.currentTarget.classList.remove("on");
const notes=document.getElementById("notes"),rev=document.getElementById("reviewer");
notes.value=st.notes||"";rev.value=st.reviewer||"";
notes.oninput=()=>{st.notes=notes.value;save()};rev.oninput=()=>{st.reviewer=rev.value;save()};
function count(){const n=Object.values(st.labels).filter(Boolean).length;document.getElementById("count").textContent=n+" images labelled"}
count();
if(PRELOAD.reviewer){document.getElementById("pre").textContent="Pre-filled from "+PRELOAD.reviewer+" - check each label; click to change. Your changes save automatically.";}
const NAME={A:"absent",C:"construction",P:"present",R:"removed",U:"unclear"};
document.getElementById("dl").onclick=()=>{
 const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
 const rows=[["facility_id","sensor","year","label","reviewer","review_date","facility_notes"].join(",")];
 const today=new Date().toISOString().slice(0,10);
 Object.entries(st.labels).filter(([k,v])=>v).sort().forEach(([k,v])=>{const [s,y]=k.split("|");
  rows.push([FID,s,y,NAME[v],q(st.reviewer),today,q(st.notes)].join(","))});
 const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")],{type:"text/csv"}));
 a.download="review_"+FID+".csv";a.click()};
</script></body></html>"""


def load_ml() -> dict:
    """{(facility_id, year): P(elevator)} from the Phase 8 classifier, if it has been run."""
    f = path("outputs_dir") / "ml_presence_by_year.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f, dtype=str)
    return {(a, b): float(c) for a, b, c in zip(d["facility_id"], d["year"], d["p_elevator"])}


ML = load_ml()


def card(sensor: str, r: pd.Series, page_dir: Path) -> str:
    k = f"{sensor}|{r['year']}"
    if r["image_available"] == "True" and r["image_path"]:
        rel = os.path.relpath(PROJECT_ROOT / r["image_path"], page_dir).replace("\\", "/")
        body = f'<div class="imgwrap"><img loading="lazy" src="{html.escape(rel)}" alt="{sensor} {r["year"]}"></div>'
        q = f'{r["cloud_quality"]} &middot; {r["n_scenes"]} sc'
        ml = ML.get((r["facility_id"], str(r["year"]))) if sensor == "naip" else None
        if ml is not None:
            q += f' &middot; <span title="classifier P(elevator) - a suggestion, not a label">ML {ml:.2f}</span>'
    else:
        body = '<div class="none">no usable image</div>'
        q = "none"
    return (f'<div class="card" data-k="{k}"><div class="yr">{r["year"]}<span class="q">{q}</span></div>{body}'
            '<div class="btns">' + "".join(f'<button data-l="{l}">{t}</button>' for l, t in
                                           [("A", "A"), ("C", "C"), ("P", "P"), ("R", "R"), ("U", "?")]) + "</div></div>")


CODE = {"absent": "A", "construction": "C", "present": "P", "removed": "R", "unclear": "U"}


def preload(fid: str) -> dict:
    """Existing labels for this facility from data/reviews/*.csv (latest file wins)."""
    labels, reviewers, notes = {}, [], ""
    for f in sorted((PROJECT_ROOT / "data" / "reviews").glob("*.csv"), key=lambda p: p.stat().st_mtime):
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        df = df[df["facility_id"] == fid]
        if df.empty:
            continue
        for r in df.itertuples():
            labels[f"{r.sensor}|{r.year}"] = CODE.get(r.label, "")
        reviewers.append(df["reviewer"].iloc[0])
        notes = df["facility_notes"].iloc[0] or notes
    return {"labels": labels, "reviewer": " + ".join(dict.fromkeys(reviewers)), "notes": notes}


def build_page(fid: str, man: pd.DataFrame, loc: pd.Series, out_dir: Path) -> Path:
    rows = []
    for sensor in ["naip", "sentinel2", "landsat"]:
        m = man[man["sensor"] == sensor].sort_values("year", key=lambda s: s.astype(int))
        if m.empty:
            continue
        rows.append(f'<h2>{SENSOR_LABEL[sensor]}</h2><div class="strip">'
                    + "".join(card(sensor, r, out_dir) for _, r in m.iterrows()) + "</div>")
    meta = " &middot; ".join(html.escape(x) for x in [
        f"{loc['business_entity']}", f"{loc['street_address'] or '(no street address)'}, {loc['city_verified'] or loc['city']}, {loc['state']}",
        f"{loc['latitude']}, {loc['longitude']}", f"address {loc['address_confidence']}", f"coords: {loc['coord_source'][:90]}"] if x)
    size = int(2 * float(man["roi_half_width_m"].iloc[0]))
    page = (PAGE.replace("__FID__", fid).replace("__PRELOAD__", json.dumps(preload(fid))).replace("__NAME__", html.escape(loc["warehouse_name"]))
            .replace("__META__", meta).replace("__ROWS__", "\n".join(rows)).replace("__SIZE__", f"{size:,}"))
    f = out_dir / f"{fid}.html"
    f.write_text(page, encoding="utf-8")
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[man["status"] == "ok"].drop_duplicates(["facility_id", "sensor", "year"], keep="last")
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    out_dir = path("outputs_dir") / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "reviews").mkdir(parents=True, exist_ok=True)
    ids = a.ids or sorted(man["facility_id"].unique())
    links = []
    for fid in ids:
        f = build_page(fid, man[man["facility_id"] == fid], loc.loc[fid], out_dir)
        n = (man[(man["facility_id"] == fid)]["image_available"] == "True").sum()
        links.append(f'<li><a href="{f.name}">{fid}</a> &middot; {html.escape(loc.loc[fid, "warehouse_name"])}, '
                     f'{html.escape(loc.loc[fid, "city"])} {loc.loc[fid, "state"]} &middot; {n} images</li>')
        log.info("wrote %s", f)
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Facility review</title>"
        "<body style='font:15px system-ui;margin:24px;max-width:900px'><h1>Facility imagery review</h1>"
        "<p>Label each facility, click <b>Download labels CSV</b>, and save the file into <code>data/reviews/</code>. "
        "Then run <code>python src/estimate_build_year.py</code>.</p><ul>" + "".join(links) + "</ul>",
        encoding="utf-8")


if __name__ == "__main__":
    main()
