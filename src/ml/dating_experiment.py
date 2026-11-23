"""Improve ML dating - experiment on the checked facilities.

Problem found in the 2026-10-11 check: the detector gives some early NAIP years
(haze, false-colour, 2 m scans) a confidently low score although the elevator is
there, and step_fit then reports a false "built in year X". Only 5 of 15 "built
during the record" ML dates held up.

Test set: every facility with year-by-year review labels (data/reviews/*.csv);
present/construction = 1, absent = 0, unclear = skipped.

Methods, all scored LEAVE-ONE-FACILITY-OUT (the held-out facility's year images
and its original training chips are never used to train the model that scores it):
  M0  current model (original chips) + step_fit                      (baseline)
  M1  M0 + guard: if any year before the fitted step scores >= GUARD_P, the
      "absent" phase is contradicted -> report "present at start" instead
  M2  retrain with the other labelled facilities' year images added (teaches
      the model what old imagery of an existing elevator looks like) + step_fit
  M3  M2 + guard
  M4  M3 + a transition needs at least 2 absent-looking images before it

Facility-level outcome compared with the label-based estimate:
  start  = both say "present at start", or both give a first year within +-1
Outputs: outputs/ml_dating_experiment.csv (per facility), outputs/ml_dating_experiment.md
Usage:   python src/ml/dating_experiment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SIZE, center_crop, image_half_widths  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from train_eval import make_model, step_fit  # noqa: E402
from utils import PROJECT_ROOT, get_logger  # noqa: E402

log = get_logger("ml_dating_experiment")
ML = PROJECT_ROOT / "data" / "ml"
GUARD_P = 0.6
LAB = {"present": 1, "construction": 1, "absent": 0}


def label_table() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "reviews").glob("review_*.csv"))
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files])
    df = df[df["sensor"] == "naip"].copy()
    df["year"] = df["year"].astype(int)
    df["y"] = df["label"].map(LAB)
    return df.drop_duplicates(["facility_id", "year"], keep="last")


def truth(g: pd.DataFrame) -> dict:
    g = g.dropna(subset=["y"]).sort_values("year")
    if g.empty:
        return {"start": None, "first": None}
    if g["y"].iloc[0] == 1:
        return {"start": True, "first": int(g["year"].iloc[0])}
    pres = g[g["y"] == 1]
    return {"start": False, "first": int(pres["year"].iloc[0]) if len(pres) else None}


def predict(years, p, guard: bool, min_absent: int = 1, guard_p: float = GUARD_P) -> dict:
    s = step_fit(years, p)
    if not s["present_at_start"] and s["first_present"] is not None:
        k = years.index(s["first_present"])
        # one absent-looking image at the very start is not evidence of absence
        if k < min_absent or (guard and max(p[:k]) >= guard_p):
            return {"start": True, "first": years[0], "guarded": True}
    return {"start": s["present_at_start"], "first": s["first_present"] if not s["present_at_start"] else years[0],
            "guarded": False}


def agree(pred: dict, tru: dict) -> bool:
    if tru["start"] is None:
        return False
    if tru["start"] or pred["start"]:
        return bool(tru["start"]) == bool(pred["start"])
    return tru["first"] is not None and pred["first"] is not None and abs(pred["first"] - tru["first"]) <= 1


def main() -> None:
    half = image_half_widths()  # per image, not the config (old images are 600 m, new 1000 m)
    lab = label_table()
    fids = sorted(lab["facility_id"].unique())
    # year images of labelled facilities: centre 300 m chip of every NAIP year
    recs = []
    for fid in fids:
        for f in sorted((PROJECT_ROOT / "data" / "imagery" / fid / "naip").glob("*.png")):
            recs.append({"facility_id": fid, "year": int(f.stem), "path": f})
    yr = pd.DataFrame(recs).merge(lab[["facility_id", "year", "y"]], on=["facility_id", "year"], how="left")
    emb = GpuEmbedder()
    Xy = emb([center_crop(p, half[(f, int(y))]) for p, f, y in zip(yr["path"], yr["facility_id"], yr["year"])])
    log.info("labelled facilities %d, year images %d (%d with a usable label)", len(fids), len(yr), yr["y"].notna().sum())

    chips = pd.read_csv(ML / "chips.csv")
    Xc = np.load(ML / "embeddings.npz")["X"]
    yc = chips["label"].to_numpy()

    rows = []
    for fid in fids:
        keep_c = (chips["facility_id"] != fid).to_numpy()
        m0 = make_model().fit(Xc[keep_c], yc[keep_c])
        tr = (yr["facility_id"] != fid) & yr["y"].notna()
        X2 = np.vstack([Xc[keep_c], Xy[tr.to_numpy()]])
        y2 = np.concatenate([yc[keep_c], yr.loc[tr, "y"].astype(int).to_numpy()])
        m2 = make_model().fit(X2, y2)
        te = (yr["facility_id"] == fid).to_numpy()
        years = yr.loc[te, "year"].tolist()
        p0 = m0.predict_proba(Xy[te])[:, 1].tolist()
        p2 = m2.predict_proba(Xy[te])[:, 1].tolist()
        t = truth(lab[lab["facility_id"] == fid])
        ylab = yr.loc[te, "y"].to_numpy()
        ok = ~np.isnan(ylab.astype(float))
        r = {"facility_id": fid, "truth_start": t["start"], "truth_first": t["first"],
             "year_acc_m0": float(((np.array(p0) >= .5) == ylab)[ok].mean()) if ok.any() else None,
             "year_acc_m2": float(((np.array(p2) >= .5) == ylab)[ok].mean()) if ok.any() else None}
        r["years"], r["p0"], r["p2"] = years, [round(v, 3) for v in p0], [round(v, 3) for v in p2]
        for name, p, g, ma in [("M0", p0, False, 1), ("M1", p0, True, 1), ("M2", p2, False, 1), ("M3", p2, True, 1),
                               ("M4", p2, True, 2)]:
            pr = predict(years, p, g, ma)
            r[f"{name}_start"], r[f"{name}_first"], r[f"{name}_ok"] = pr["start"], pr["first"], agree(pr, t)
        rows.append(r)
        log.info("%s truth=%s/%s  M0=%s/%s  M3=%s/%s", fid, t["start"], t["first"], r["M0_start"], r["M0_first"],
                 r["M3_start"], r["M3_first"])
    res = pd.DataFrame(rows)
    res.to_csv(PROJECT_ROOT / "outputs" / "ml_dating_experiment.csv", index=False)

    n = len(res)
    trans = res[res["truth_start"] == False]  # noqa: E712
    start = res[res["truth_start"] == True]  # noqa: E712
    lines = ["# ML dating experiment (leave-one-facility-out)", "",
             f"Facilities with labels: {n} ({len(start)} present at start, {len(trans)} built during the record).", "",
             "| Method | Facilities agreeing | Present-at-start sites right | Built-during-record sites right |",
             "|---|---|---|---|"]
    names = {"M0": "M0 current model + step rule", "M1": "M1 + guard rule", "M2": "M2 retrained with year images",
             "M3": "M3 retrained + guard", "M4": "M4 retrained + guard + 2 absent images needed"}
    for k, label in names.items():
        lines.append(f"| {label} | {res[f'{k}_ok'].sum()}/{n} | {start[f'{k}_ok'].sum()}/{len(start)} | "
                     f"{trans[f'{k}_ok'].sum()}/{len(trans)} |")
    lines += ["", f"Per-year accuracy (p >= 0.5 vs label): M0 {res['year_acc_m0'].mean():.3f}, "
                  f"M2 {res['year_acc_m2'].mean():.3f} (mean over facilities)."]
    (PROJECT_ROOT / "outputs" / "ml_dating_experiment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
