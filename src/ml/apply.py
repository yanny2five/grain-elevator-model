"""Phase 8 - apply the presence classifier to every facility's NAIP series.

These are MACHINE SUGGESTIONS, kept separate from human review labels
(outputs/estimated_build_years.csv). They are meant to prioritise and pre-fill
human review, not to replace it.

Chip choice:
  geocode STRUCTURE  -> centre 300 m chip (coordinates are on the structures)
  otherwise          -> 3x3 grid of 300 m chips over the central 900 m, max P
                        (street-address geocodes can be a few hundred m off;
                        the grid may also pick up a neighbouring elevator - flagged)

Per facility a single absent->present step is fitted to the year series
(train_eval.robust_step_fit: step fit + guards against one bad early image). Outputs:
  outputs/ml_presence_by_year.csv   facility x NAIP year, P(elevator)
  outputs/ml_first_present.csv      step-fit result + flags per facility
Usage: python src/ml/apply.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import CHIP_M, SIZE  # noqa: E402
from train_eval import THRESH, Embedder, robust_step_fit  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("ml_apply")
ML = PROJECT_ROOT / "data" / "ml"


def chips_for(img_path: Path, half_width_m: float, grid: bool) -> list[Image.Image]:
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    mpp = 2 * half_width_m / w  # metres per pixel
    side = CHIP_M / mpp
    offsets = [-CHIP_M, 0, CHIP_M] if grid else [0]
    out = []
    for dy in offsets:
        for dx in offsets:
            cx, cy = w / 2 + dx / mpp, h / 2 + dy / mpp
            box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
            out.append(im.crop(box).resize((SIZE, SIZE), Image.BILINEAR))
    return out


def judge(years: list[int], probs: list[float]) -> dict:
    """Step fit + two quality measures:
    fit_cost_per_year: mean disagreement between the step and the scores (0 = perfectly clean series)
    gain_vs_no_change: how much better the best step is than the best no-change answer
                       (always present / never present) - only meaningful for transitions."""
    fit = robust_step_fit(years, probs)
    p = np.asarray(probs, float)
    no_change = min((1 - p).sum(), p.sum()) / len(p)
    fit["gain_vs_no_change"] = round(float(no_change - fit["fit_cost_per_year"]), 3)
    return fit


BLANK_DARK_FRAC = 0.9  # image_quality.py: centre almost all black = no imagery


def blank_images() -> set[tuple[str, int]]:
    q = path("outputs_dir") / "naip_image_quality.csv"
    if not q.exists():
        return set()
    d = pd.read_csv(q, dtype={"facility_id": str})
    d = d[d["dark_frac"] >= BLANK_DARK_FRAC]
    return {(f, int(y)) for f, y in zip(d["facility_id"], d["year"])}


def summarize(by_year: pd.DataFrame, train_ids: set[str]) -> pd.DataFrame:
    rows = []
    blank = blank_images()
    for fid, g in by_year.groupby("facility_id"):
        g = g.sort_values("year")
        n_all = len(g)
        g = g[[(fid, int(y)) not in blank for y in g["year"]]]
        n_blank = n_all - len(g)
        if g.empty:
            continue
        years, probs = [int(y) for y in g["year"]], list(g["p_elevator"].astype(float))
        fit = judge(years, probs)
        grid = g["chip_mode"].iloc[0] != "centre_300m"
        transition = fit["last_absent"] is not None and fit["first_present"] is not None
        flags = []
        if fit["fit_cost_per_year"] > 0.2:
            flags.append("noisy series - review")
        if transition and fit["gain_vs_no_change"] < 0.15:
            flags.append("weak transition - review")
        if fit["never_present"]:
            flags.append("no elevator detected in any year - check location")
        if fit["guarded"]:
            flags.append("early low scores contradicted by other years - treated as present at start; review")
        if fit["present_at_start"]:
            flags.append(f"present in first NAIP year ({years[0]}) - built on/before then")
        if grid:
            flags.append("address-level coordinates: grid search may include a neighbouring elevator")
        if fid in train_ids:
            flags.append("recent years used as training positives")
        if n_blank:
            flags.append(f"{n_blank} blank NAIP image(s) skipped")
        conf = ("LOW" if any("review" in f for f in flags) or grid else
                "MEDIUM" if fit["fit_cost_per_year"] > 0.1 else "HIGH")
        rows.append({"facility_id": fid, "naip_years": len(years), "first_naip_year": years[0],
                     "last_naip_year": years[-1], "ml_last_absent": fit["last_absent"],
                     "ml_first_present": fit["first_present"],
                     "ml_window": (f"{fit['last_absent'] + 1}-{fit['first_present']}" if transition else
                                   (f"<= {fit['first_present']}" if fit["present_at_start"] else "")),
                     "fit_cost_per_year": fit["fit_cost_per_year"], "gain_vs_no_change": fit["gain_vs_no_change"],
                     "ml_confidence": conf, "chip_mode": g["chip_mode"].iloc[0], "ml_flags": " | ".join(flags)})
    out = pd.DataFrame(rows)
    for c in ["ml_last_absent", "ml_first_present"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def main() -> None:
    train_ids = set(pd.read_csv(ML / "chips.csv", dtype=str)["facility_id"])
    if "--summarize-only" in sys.argv:
        by = pd.read_csv(path("outputs_dir") / "ml_presence_by_year.csv", dtype={"facility_id": str})
        s = summarize(by, train_ids)
        s.to_csv(path("outputs_dir") / "ml_first_present.csv", index=False)
        log.info("summary: %s", s["ml_confidence"].value_counts().to_dict())
        return
    # v2 = original chips + labelled year images (train_v2.py); fall back to v1
    model = joblib.load(ML / "model_v2.joblib") if (ML / "model_v2.joblib").exists() else joblib.load(ML / "model.joblib")
    emb = GpuEmbedder()
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False)
    loc = loc[loc["clip_ok"] == "True"].set_index("facility_id")
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["sensor"] == "naip") & (man["image_available"] == "True") & (man["status"] == "ok")]
    man = man.drop_duplicates(["facility_id", "year"], keep="last")
    by_year = []
    fids = sorted(set(man["facility_id"]) & set(loc.index))
    only = [a for a in sys.argv[1:] if a.startswith("F")]  # e.g. apply.py F00295 -> rescore just these
    if only:
        fids = [f for f in fids if f in only]
    for i, fid in enumerate(fids, 1):
        try:
            grid = loc.loc[fid, "geocode_confidence"] != "STRUCTURE"
            m = man[man["facility_id"] == fid].assign(year=lambda d: d["year"].astype(int)).sort_values("year")
            chips, owner = [], []
            for _, r in m.iterrows():
                cs = chips_for(PROJECT_ROOT / r["image_path"], float(r["roi_half_width_m"]), grid)
                chips += cs
                owner += [r["year"]] * len(cs)
            p = model.predict_proba(emb(chips))[:, 1]
            per_year = pd.DataFrame({"year": owner, "p": p}).groupby("year")["p"].agg(["max", "idxmax"])
            years = list(per_year.index)
            probs = list(per_year["max"])
            for y, pr in zip(years, probs):
                by_year.append({"facility_id": fid, "year": y, "p_elevator": round(float(pr), 3),
                                "chip_mode": "grid_max_900m" if grid else "centre_300m"})
        except Exception as e:  # noqa: BLE001
            log.error("%s failed: %s", fid, e)
        if i % 25 == 0:
            log.info("%d/%d facilities", i, len(fids))
    by = pd.DataFrame(by_year)
    if only:  # keep everyone else's earlier scores
        old = pd.read_csv(path("outputs_dir") / "ml_presence_by_year.csv", dtype={"facility_id": str})
        by = pd.concat([old[~old["facility_id"].isin(only)], by], ignore_index=True)
    by.to_csv(path("outputs_dir") / "ml_presence_by_year.csv", index=False)
    s = summarize(by, train_ids)
    s.to_csv(path("outputs_dir") / "ml_first_present.csv", index=False)
    log.info("done: %d facilities; %s", len(s), s["ml_confidence"].value_counts().to_dict())


if __name__ == "__main__":
    main()
