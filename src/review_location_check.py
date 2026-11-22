"""Review page for Claude's 2026-10-17 location check - so a person can confirm or reject each decision.

For every facility in data/addresses/location_check_2026-*.csv (Claude's checks), draws on the candidate image (latest NAIP
before the move, 1.2 km square centred on the OLD point):
  red cross    = old point
  green cross  = new point (moves only)
  yellow A/B/C = AI candidates
and lists Claude's verdict. The reviewer clicks agree / disagree / unsure and
downloads a CSV (save to data/reviews/location_check_review.csv).

Output: outputs/review/location_check.html, outputs/review/location_check/<id>.jpg
Usage:  python src/review_location_check.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, load_config  # noqa: E402

REVIEW = PROJECT_ROOT / "outputs" / "review"
SRC = REVIEW / "candidates"
OUT = REVIEW / "location_check"
CAND_HALF_M = 600.0  # half width of outputs/review/candidates photos (1.2 km square, 100 m grid)
LABEL = {"ok": "point already on the elevator - kept", "move": "moved onto the elevator (green)",
         "no_elevator": "no elevator nearby - marked LOW (needs relocating)", "unclear": "unclear - left as it was"}


def main() -> None:
    # optional: python src/review_location_check.py recheck <ids.txt>  -> a separate page for just those ids
    recheck = len(sys.argv) > 2 and sys.argv[1] == "recheck"
    only = set(Path(sys.argv[2]).read_text().split()) if recheck else None
    # The candidate photos were made in Oct 2026 with a 1.2 km square (half width 600 m). The config was
    # widened to 1000 m on 2026-10-26, so reading it here drew the green cross at 60% of the true offset
    # (bug from 2026-11-20 to 2026-11-22). Use the photos' own scale.
    half_m = CAND_HALF_M
    files = sorted((PROJECT_ROOT / "data" / "addresses").glob("location_check_2026-*.csv"))
    chk = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files]).drop_duplicates("facility_id", keep="last")
    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for r in chk.itertuples():
        if only is not None and r.facility_id not in only:
            continue
        f = SRC / f"{r.facility_id}.jpg"
        if not f.exists():
            continue
        im = Image.open(f).convert("RGB")
        S = im.size[0]
        if r.verdict == "move" and r.new_lat:
            dy = (float(r.new_lat) - float(r.old_lat)) * 111_320
            dx = (float(r.new_lon) - float(r.old_lon)) * 111_320 * math.cos(math.radians(float(r.old_lat)))
            x, y = S / 2 + dx / (2 * half_m) * S, S / 2 - dy / (2 * half_m) * S
            d = ImageDraw.Draw(im)
            for a, b in [((x - 22, y), (x - 7, y)), ((x + 7, y), (x + 22, y)), ((x, y - 22), (x, y - 7)), ((x, y + 7), (x, y + 22))]:
                d.line([a, b], fill=(0, 255, 90), width=4)
        im.save(OUT / f.name, quality=82)
        items.append({"id": r.facility_id, "name": f"{r.warehouse_name} - {r.city}, {r.state}", "verdict": r.verdict,
                      "label": LABEL[r.verdict] + (" (tentative)" if r.tentative == "True" else ""), "note": r.note,
                      "img": f"location_check/{f.name}"})
    data = json.dumps(items)
    # answers already saved, so any browser starts with them (2026-11-20)
    prior = {}
    rf = PROJECT_ROOT / "data" / "reviews" / "location_check_review.csv"
    if rf.exists() and not recheck:
        rv = pd.read_csv(rf, dtype=str, keep_default_na=False)
        col = "final" if "final" in rv.columns else "review"
        prior = {r["facility_id"]: r[col] for _, r in rv.iterrows() if r[col]}
    prior_json = json.dumps(prior)
    page, key, title = ("location_recheck.html", "location_recheck", "Location recheck") if recheck \
        else ("location_check.html", "location_check_review", "Location check review")
    (REVIEW / page).write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#f6f6f4;color:#222}}
.t{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:0 0 18px;max-width:840px}}
.t img{{width:100%;max-width:800px;border-radius:4px}} label{{margin-right:16px}} .done{{border-color:#2a7}}
.bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ddd;z-index:2}}
.v{{font-weight:600}} .move{{color:#07813a}} .no_elevator{{color:#b3261e}} .unclear{{color:#8a6d00}}
select,button{{font-size:15px;padding:4px 8px}}</style></head><body>
<div class="bar"><b>{title}</b> - red = old point, green = Claude's new point, yellow = AI candidates.
Grid lines every 100 m. &nbsp; Show <select id="f"><option value="">all</option><option>move</option><option>ok</option>
<option>no_elevator</option><option>unclear</option></select> &nbsp; <span id="n"></span> &nbsp;
<label><input type="checkbox" id="hide" checked> hide done</label>
<button onclick="dl()">Download review (CSV)</button></div><div id="list"></div>
<script>
const T={data}; const KEY="{key}"; let P={{}}; try{{P=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
const PRIOR={prior_json}; for(const k in PRIOR){{if(!P[k])P[k]=PRIOR[k]}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(P))}}catch(e){{}};cnt()}}
function cnt(){{document.getElementById("n").textContent=Object.keys(P).length+" / "+T.length+" reviewed"}}
function draw(){{const f=document.getElementById("f").value, L=document.getElementById("list"); L.innerHTML="";
 const hide=document.getElementById("hide").checked; T.forEach(t=>{{ if(f&&t.verdict!=f) return; if(hide&&P[t.id]) return; const d=document.createElement("div"); d.className="t"+(P[t.id]?" done":"");
  let h=`<div><b>${{t.id}}</b> ${{t.name}}</div><div class="v ${{t.verdict}}">${{t.label}}</div><div><small>${{t.note}}</small></div><img loading="lazy" src="${{t.img}}"><div>`;
  for(const o of ["agree","disagree","unsure"]) h+=`<label><input type="radio" name="${{t.id}}" value="${{o}}" ${{P[t.id]==o?"checked":""}}> ${{o}}</label>`;
  d.innerHTML=h+"</div>"; d.querySelectorAll("input").forEach(x=>x.onchange=()=>{{P[t.id]=x.value;d.classList.add("done");save()}}); L.appendChild(d)}})}}
function dl(){{let s="facility_id,claude_verdict,review\\n"; T.forEach(t=>{{if(P[t.id]) s+=`${{t.id}},${{t.verdict}},${{P[t.id]}}\\n`}});
 const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([s],{{type:"text/csv"}})); a.download="{key}.csv"; a.click()}}
document.getElementById("f").onchange=draw; document.getElementById("hide").onchange=draw; save(); draw();
</script></body></html>""", encoding="utf-8")
    print(len(items), "items")


if __name__ == "__main__":
    main()
