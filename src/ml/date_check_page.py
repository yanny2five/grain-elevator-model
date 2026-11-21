"""Fast human check of ML dates (2026-11-07).

Builds outputs/review/date_check.html from the triage queue:
  * every site in data/reviews/dating_review_queue.csv with queue "1 review date"
    (precise location, at least one NAIP year scored between 0.2 and 0.8), and
  * a seeded random sample of 50 CLEAR precise-location sites (spot check of the
    rule itself).

Each card shows a centre crop (about 600 m) of every NAIP year with the model's
score, and the model's call. The reviewer clicks the first year the elevator is
visible ("present at start" = first photo), or Agree / Unsure / No elevator.
Answers are kept in the browser and exported with "Download CSV"; save the file
as data/reviews/date_check_review.csv.

Usage: python src/ml/date_check_page.py
Serve:  python -m http.server 8766 --bind 127.0.0.1 --directory <project>
Open:   http://127.0.0.1:8766/outputs/review/date_check.html
"""
import html
import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/review"
THUMBS = OUT / "date_check"
CROPS = {"close": 0.15, "wide": 0.3}  # centre 15% = about 300 m (what the model sees); 30% = about 600 m
THUMB_PX = 480
SPOT_N, SEED = 50, 7


def thumb(fid, year, mode="close"):
    src = ROOT / "data/imagery" / fid / "naip" / f"{year}.png"
    dst = THUMBS / f"{fid}_{year}_{mode}_{THUMB_PX}.jpg"
    if not src.exists():
        return None
    if not dst.exists():
        im = Image.open(src).convert("RGB")
        w, h = im.size
        c = int(min(w, h) * CROPS[mode] / 2)
        im = im.crop((w // 2 - c, h // 2 - c, w // 2 + c, h // 2 + c)).resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
        im.save(dst, quality=85)
    return f"date_check/{dst.name}"


def main():
    THUMBS.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(ROOT / "data/reviews/dating_review_queue.csv", dtype={"facility_id": str})
    review = q[q.queue.str.startswith("1")].facility_id.tolist()
    fp = pd.read_csv(ROOT / "outputs/ml_first_present.csv", dtype={"facility_id": str}).set_index("facility_id")
    by = pd.read_csv(ROOT / "outputs/ml_presence_by_year.csv", dtype={"facility_id": str})
    qf = ROOT / "outputs/naip_image_quality.csv"
    if qf.exists():  # skip blank (no-data) images, same rule as apply.py
        qd = pd.read_csv(qf, dtype={"facility_id": str})
        bl = set(zip(qd.loc[qd.dark_frac >= 0.9, "facility_id"], qd.loc[qd.dark_frac >= 0.9, "year"].astype(int)))
        by = by[[(f, int(y)) not in bl for f, y in zip(by.facility_id, by.year)]]
    amb =by.groupby("facility_id").p_elevator.apply(lambda s: int(((s > 0.2) & (s < 0.8)).sum()))
    clear = [f for f in fp.index if fp.loc[f, "chip_mode"] == "centre_300m" and amb.get(f, 0) == 0]
    spot = pd.Series(sorted(clear)).sample(min(SPOT_N, len(clear)), random_state=SEED).tolist()
    m = pd.read_csv(ROOT / "outputs/master_facility_dataset.csv", dtype=str, keep_default_na=False).set_index("facility_id")

    items = []
    for kind, fids in [("review", review), ("spot check", spot)]:
        for fid in fids:
            rows = by[by.facility_id == fid].sort_values("year")
            years = []
            for _, r in rows.iterrows():
                t = thumb(fid, int(r.year))
                if t:
                    years.append({"y": int(r.year), "p": round(float(r.p_elevator), 2), "img": t,
                                  "wide": thumb(fid, int(r.year), "wide"),
                                  "full": f"../../data/imagery/{fid}/naip/{int(r.year)}.png"})
            if not years:
                continue
            f = fp.loc[fid]
            start = pd.isna(f.ml_last_absent)
            nofirst = pd.isna(f.ml_first_present)
            call = ("present at start" if start else "no year found (never clearly present)" if nofirst
                    else f"first visible {int(f.ml_first_present)}")
            mm = m.loc[fid] if fid in m.index else {}
            items.append({"fid": fid, "kind": kind, "call": call,
                          "first": None if (start or nofirst) else int(f.ml_first_present),
                          "start": bool(start),
                          "name": str(mm.get("warehouse_name", "")) if len(mm) else "",
                          "place": f"{mm.get('city', '')}, {mm.get('state', '')}" if len(mm) else "",
                          "years": years})
        print(kind, len([i for i in items if i["kind"] == kind]))

    # where the detector thinks the elevator is (date_check_spots.py), as fractions of the full photo
    sf = OUT / "date_check_spots.json"
    spots = json.loads(sf.read_text()) if sf.exists() else {}
    for it in items:
        it["spot"] = spots.get(it["fid"])
    # answers already saved to CSV, so a new browser starts with them
    prior = {}
    rf = ROOT / "data/reviews/date_check_review.csv"
    if rf.exists():
        for _, r in pd.read_csv(rf, dtype=str, keep_default_na=False).iterrows():
            prior[r.facility_id] = {"first": int(r.human_first_visible) if r.human_first_visible else None,
                                    "label": r.human_label}
    data = json.dumps(items)
    page = (PAGE.replace("__DATA__", data).replace("__PRIOR__", json.dumps(prior))
            .replace("__CROPS__", json.dumps(CROPS)).replace("__VER__", str(int(rf.stat().st_mtime)) if rf.exists() else "0").replace("__N__", str(len(items))))
    (OUT / "date_check.html").write_text(page, encoding="utf-8")
    print("wrote", OUT / "date_check.html", len(items), "sites")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Date Check</title>
<style>
body{margin:0;background:#f6f4ee;color:#1f2a24;font:18px/1.45 system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:#1f2a24;color:#f6f4ee;padding:10px 16px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:2}
header b{color:#f2c744} button{font:inherit;cursor:pointer;border-radius:6px;border:1px solid #9aa79f;background:#fffdf8;padding:5px 10px}
main{max-width:1500px;margin:0 auto;padding:14px 16px 60px}
.card{background:#fffdf8;border:1px solid #d9d3c3;border-radius:10px;padding:12px 14px;margin:0 0 14px}
.card.done{border-color:#2f6b4c;background:#f1f7f3}
.top{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
.tag{font-size:12px;background:#eae6da;padding:2px 8px;border-radius:99px}
.call{font-weight:700;color:#2f6b4c}
.strip{display:flex;flex-wrap:wrap;gap:10px;padding:10px 0}
.yr{flex:none;width:calc(var(--sz) + 6px);text-align:center;cursor:pointer;border:3px solid transparent;border-radius:8px;padding:2px}
.yr img{width:var(--sz);height:var(--sz);display:block;border-radius:5px}
.ph{position:relative;width:var(--sz);height:var(--sz)}
.ring{position:absolute;border:3px solid #ff2d55;border-radius:50%;box-shadow:0 0 0 2px rgba(255,255,255,.8);pointer-events:none;transform:translate(-50%,-50%)}
.pin{position:absolute;left:50%;top:50%;width:18px;height:18px;transform:translate(-50%,-50%);pointer-events:none;
 background:linear-gradient(#00e5ff,#00e5ff) center/2px 100% no-repeat,linear-gradient(#00e5ff,#00e5ff) center/100% 2px no-repeat}
.off{font-size:15px;color:#8a3b12;font-weight:600}
.yr.model{border-color:#f2c744} .yr.pick{border-color:#2f6b4c;background:#e1edE5}
.p{font-size:17px} .yr div{font-size:18px;font-weight:600} .hi{color:#2f6b4c;font-weight:700} .lo{color:#8a3b12} .mid{color:#7a6400;font-weight:700}
.act{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.act button.on{background:#2f6b4c;color:#fff;border-color:#2f6b4c}
:root{--sz:300px} .help{color:#c9d3cc;font-size:15px} .act button{font-size:17px;padding:8px 14px}
</style></head><body>
<header><span><b>Date check</b> · <span id="cnt">0</span> / __N__ done</span>
<button onclick="dl()">Download CSV</button>
<button id="zb" onclick="zoom()">Zoom: close (300 m)</button>
<label>Photo size <input type="range" id="sz" min="150" max="480" step="10" value="300" oninput="sz(this.value)"></label>
<label><input type="checkbox" id="hide" onchange="keep('date_check_hide',this.checked);draw()"> hide done</label>
<label><input type="checkbox" id="newest" onchange="keep('date_check_newest',this.checked);draw()"> newest first</label>
<label><input type="checkbox" id="rings" onchange="keep('date_check_rings',this.checked);draw()"> show spot</label>
<span class="help">Red circle = where the model thinks the elevator is (same spot every year). Blue + = the mapped location. Yellow box = model’s first visible year. Click the first year YOU see the elevator, or a button. Click a photo with Ctrl to open it full size.</span></header>
<main id="m"></main>
<script>
const D=__DATA__; const PRIOR=__PRIOR__; const CROPS=__CROPS__; const KEY="date_check_review"; let P={}; try{P=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
try{if(localStorage.getItem("date_check_prior_ver")!=="__VER__"){for(const k in PRIOR)P[k]=PRIOR[k];localStorage.setItem("date_check_prior_ver","__VER__")}
 else{for(const k in PRIOR){if(!P[k])P[k]=PRIOR[k]}}}catch(e){for(const k in PRIOR){if(!P[k])P[k]=PRIOR[k]}}
function keep(k,v){try{localStorage.setItem(k,v?"1":"0")}catch(e){}}
function flag(k,d){try{const v=localStorage.getItem(k);return v===null?d:v==="1"}catch(e){return d}}
document.getElementById("newest").checked=flag("date_check_newest",true);
document.getElementById("hide").checked=flag("date_check_hide",true);
document.getElementById("rings").checked=flag("date_check_rings",true);
function ring(sp){if(!sp||!document.getElementById("rings").checked)return"";const f=CROPS[Z];
 const x=(sp.x-0.5)/f+0.5,y=(sp.y-0.5)/f+0.5; if(x<0||x>1||y<0||y>1)return"";
 const d=Math.max(14,Math.round(220/(2000*f)*100)); return '<div class="ring" style="left:'+(x*100)+'%;top:'+(y*100)+'%;width:'+d+'%;height:'+d+'%"></div>'}
function outside(sp){if(!sp)return"";const f=CROPS[Z];const x=(sp.x-0.5)/f+0.5,y=(sp.y-0.5)/f+0.5;
 if(x>=0&&x<=1&&y>=0&&y<=1)return""; return Z==="close"?' <span class="off">· model’s spot is outside this view: try Zoom wide</span>':' <span class="off">· model’s spot is outside this view (location may be off)</span>'}
function save(){try{localStorage.setItem(KEY,JSON.stringify(P))}catch(e){}; document.getElementById("cnt").textContent=Object.keys(P).length}
function set(fid,v){P[fid]=v;save();draw()}
function cls(p){return p>=0.8?"hi":p<=0.2?"lo":"mid"}
function draw(){const m=document.getElementById("m");m.innerHTML="";const hide=document.getElementById("hide").checked;
 D.forEach((it,i)=>{const a=P[it.fid]; if(hide&&a)return; const c=document.createElement("div"); c.className="card"+(a?" done":"");
  const modelYear=it.start?it.years[0].y:it.first;
  let s='<div class="top"><b>'+(i+1)+'. '+it.fid+'</b><span>'+esc(it.name)+' · '+esc(it.place)+'</span><span class="tag">'+it.kind+'</span><span>Model: <span class="call">'+it.call+'</span></span>'+(a?'<span>· You: <b>'+esc(a.label)+'</b></span>':'')+outside(it.spot)+'</div><div class="strip">';
  const order=it.years.map((y,j)=>j); if(document.getElementById("newest").checked)order.reverse();
  order.forEach(j=>{const y=it.years[j];const pk=a&&a.first===y.y; s+='<div class="yr'+(y.y===modelYear?' model':'')+(pk?' pick':'')+'" data-j="'+j+'"><div class="ph"><img loading="lazy" src="'+(Z==="wide"?y.wide:y.img)+'" alt="'+y.y+'">'+(document.getElementById("rings").checked?'<div class="pin"></div>':'')+ring(it.spot)+'</div><div>'+y.y+' <span class="p '+cls(y.p)+'">'+y.p.toFixed(2)+'</span></div></div>'});
  s+='</div><div class="act"><button data-a="agree">Agree with model</button><button data-a="start">Present at start</button><button data-a="none">No elevator here</button><button data-a="unsure">Unsure</button></div>';
  c.innerHTML=s;
  c.querySelectorAll(".yr").forEach(e=>e.onclick=ev=>{const y=it.years[+e.dataset.j]; if(ev.ctrlKey||ev.metaKey){window.open(y.full);return}
    set(it.fid,{first:y.y,label:(+e.dataset.j===0?"present at start ("+y.y+")":"first visible "+y.y)})});
  c.querySelectorAll("button[data-a]").forEach(b=>b.onclick=()=>{const k=b.dataset.a;
    const v=k==="agree"?{first:modelYear??null,label:"agree: "+it.call}:k==="start"?{first:it.years[0].y,label:"present at start ("+it.years[0].y+")"}:k==="none"?{first:null,label:"no elevator"}:{first:null,label:"unsure"};
    set(it.fid,v)});
  m.appendChild(c)})}
function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
function dl(){let s="facility_id,kind,model_call,human_first_visible,human_label\n";
 D.forEach(it=>{const a=P[it.fid]; if(!a)return; s+=[it.fid,it.kind,it.call,a.first??"",'"'+a.label.replace(/"/g,'""')+'"'].join(",")+"\n"});
 const e=document.createElement("a"); e.href=URL.createObjectURL(new Blob([s],{type:"text/csv"})); e.download="date_check_review.csv"; e.click()}
let Z="close"; try{Z=localStorage.getItem("date_check_zoom")||"close"}catch(e){}
function zoomLabel(){document.getElementById("zb").textContent=Z==="close"?"Zoom: close (300 m)":"Zoom: wide (600 m)"}
function zoom(){Z=Z==="close"?"wide":"close";try{localStorage.setItem("date_check_zoom",Z)}catch(e){};zoomLabel();draw()}
function sz(v){document.documentElement.style.setProperty("--sz",v+"px");try{localStorage.setItem("date_check_size",v)}catch(e){}}
try{const v=localStorage.getItem("date_check_size");if(v){sz(v);document.getElementById("sz").value=v}}catch(e){}
zoomLabel(); save(); draw();
</script></body></html>"""

if __name__ == "__main__":
    main()
