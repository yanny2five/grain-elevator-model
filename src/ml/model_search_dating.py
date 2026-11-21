"""Search model settings for dating (leave-one-facility-out on all labelled facilities).

Same protocol as dating_experiment.py (M4 rule: robust_step_fit), but varies the
model trained on chips + labelled year images:
  C          logistic-regression regularisation (default 0.05)
  w_year     weight of labelled year images relative to the weak-label chips
  flips      add mirrored/rotated copies of the training year images
Embeddings are computed once and cached in data/ml/emb_dating_*.npz.
Output: outputs/ml_model_search_dating.csv
Usage:  python src/ml/model_search_dating.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import center_crop  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from dating_experiment import agree, label_table, truth  # noqa: E402
from train_eval import robust_step_fit  # noqa: E402
from utils import PROJECT_ROOT, get_logger, load_config  # noqa: E402

log = get_logger("ml_model_search_dating")
ML = PROJECT_ROOT / "data" / "ml"


def main() -> None:
    half_m = float(load_config()["earth_engine"]["roi_half_width_m"])
    lab = label_table()
    fids = sorted(lab["facility_id"].unique())
    recs = [{"facility_id": f, "year": int(p.stem), "path": p}
            for f in fids for p in sorted((PROJECT_ROOT / "data" / "imagery" / f / "naip").glob("*.png"))]
    yr = pd.DataFrame(recs).merge(lab[["facility_id", "year", "y"]], on=["facility_id", "year"], how="left")
    cache = ML / f"emb_dating_{len(yr)}.npz"
    if cache.exists():
        z = np.load(cache)
        Xy, Xf = z["X"], z["F"]
    else:
        emb = GpuEmbedder()
        crops = [center_crop(p, half_m) for p in yr["path"]]
        Xy = emb(crops)
        Xf = emb([ImageOps.mirror(c) for c in crops])  # one augmented copy per image
        np.savez_compressed(cache, X=Xy, F=Xf)
    chips = pd.read_csv(ML / "chips.csv")
    Xc = np.load(ML / "embeddings.npz")["X"]
    yc = chips["label"].to_numpy()
    truths = {f: truth(lab[lab["facility_id"] == f]) for f in fids}
    rows = []
    for C in [0.01, 0.05, 0.2]:
        for w_year in [1, 3]:
            for flips in [False, True]:
                ok = start_ok = built_ok = 0
                for fid in fids:
                    kc = (chips["facility_id"] != fid).to_numpy()
                    tr = ((yr["facility_id"] != fid) & yr["y"].notna()).to_numpy()
                    Xs = [Xc[kc], Xy[tr]] + ([Xf[tr]] if flips else [])
                    ys = [yc[kc], yr.loc[tr, "y"].astype(int).to_numpy()] + (
                        [yr.loc[tr, "y"].astype(int).to_numpy()] if flips else [])
                    ws = [np.ones(kc.sum()), np.full(tr.sum(), w_year)] + ([np.full(tr.sum(), w_year)] if flips else [])
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
                    m.fit(np.vstack(Xs), np.concatenate(ys), logisticregression__sample_weight=np.concatenate(ws))
                    te = (yr["facility_id"] == fid).to_numpy()
                    years = yr.loc[te, "year"].tolist()
                    p = m.predict_proba(Xy[te])[:, 1].tolist()
                    fit = robust_step_fit(years, p)
                    pr = {"start": fit["present_at_start"], "first": fit["first_present"]}
                    a = agree(pr, truths[fid])
                    ok += a
                    if truths[fid]["start"]:
                        start_ok += a
                    else:
                        built_ok += a
                r = {"C": C, "w_year": w_year, "flips": flips, "agree": ok, "of": len(fids),
                     "start_ok": start_ok, "built_ok": built_ok}
                rows.append(r)
                log.info("%s", r)
    out = pd.DataFrame(rows).sort_values("agree", ascending=False)
    out.to_csv(PROJECT_ROOT / "outputs" / "ml_model_search_dating.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
