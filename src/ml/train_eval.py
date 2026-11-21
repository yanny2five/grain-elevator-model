"""Phase 8 (proof of concept) - train and evaluate an elevator-presence classifier.

Model: ImageNet-pretrained ResNet-50 as a fixed feature extractor (2048-d
embedding per chip) + L2-regularised logistic regression. With ~2,000 chips a
frozen backbone + linear head is the right size: fine-tuning would overfit, and
the embedding step is reusable for the similarity/change-detection ideas.

Evaluation
1. Chip classification, 5-fold cross-validation GROUPED BY FACILITY (chips of
   one facility never appear in both train and test): ROC AUC, accuracy.
2. Temporal test on the 8 held-out labelled facilities (never in training):
   score the centre chip of every NAIP year and compare with the year labels;
   derive a first-present year and compare with the human/preliminary estimate.

Outputs: data/ml/embeddings.npz, data/ml/model.joblib,
         outputs/ml_eval_report.md, outputs/ml_temporal_test.csv
Usage:   python src/ml/train_eval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import CHIP_M, SIZE, center_crop  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("ml_train_eval")
ML = PROJECT_ROOT / "data" / "ml"
THRESH = 0.5


class Embedder:
    def __init__(self):
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        m = torchvision.models.resnet50(weights=w)
        m.fc = torch.nn.Identity()
        self.model = m.eval()
        self.tf = w.transforms()
        torch.set_num_threads(max(1, (torch.get_num_threads())))

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            x = torch.stack([self.tf(im.convert("RGB")) for im in images[i:i + batch]])
            out.append(self.model(x).numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight="balanced"))


def step_fit(years: list[int], probs: list[float]) -> dict:
    """Fit one absent->present step to a year series of P(elevator).

    Cost of a step at index k = sum(p before k) + sum(1-p from k on); k=0 means
    present in every year ("present at start"), k=n means never present.
    Robust to isolated noisy years (haze, odd sun angle, terminal geometry).
    Returns the step year, its cost per year, and the margin to the runner-up
    (small margin = ambiguous series -> needs human review).
    """
    p = np.asarray(probs, float)
    n = len(p)
    costs = [p[:k].sum() + (1 - p[k:]).sum() for k in range(n + 1)]
    order = np.argsort(costs)
    k = int(order[0])
    margin = (costs[order[1]] - costs[order[0]]) / max(n, 1) if n > 1 else 0.0
    return {"first_present": None if k == n else int(years[k]),
            "last_absent": int(years[k - 1]) if 0 < k <= n else None,
            "present_at_start": k == 0, "never_present": k == n,
            "fit_cost_per_year": round(costs[k] / max(n, 1), 3), "margin": round(float(margin), 3)}


GUARD_P, MIN_ABSENT = 0.6, 2


def robust_step_fit(years: list[int], probs: list[float]) -> dict:
    """step_fit plus two guards against false "built in year X" calls (dating_experiment.py, 2026-10-17):
    - a transition needs at least MIN_ABSENT absent-looking images before it (one hazy first image is not
      evidence of absence);
    - if any year before the step scores >= GUARD_P, the "absent" phase is contradicted.
    Either case is reported as present at start, with guarded=True so it can be flagged for review."""
    fit = step_fit(years, probs)
    fit["guarded"] = False
    if fit["last_absent"] is not None and fit["first_present"] is not None:
        k = years.index(fit["first_present"])
        if k < MIN_ABSENT or max(probs[:k]) >= GUARD_P:
            cost = float((1 - np.asarray(probs, float)).sum()) / max(len(probs), 1)
            fit.update(first_present=int(years[0]), last_absent=None, present_at_start=True,
                       fit_cost_per_year=round(cost, 3), guarded=True)
    return fit


def first_present(years: list[int], probs: list[float]) -> int | None:
    return step_fit(years, probs)["first_present"]


def main() -> None:
    chips = pd.read_csv(ML / "chips.csv")
    emb = Embedder()
    cache = ML / "embeddings.npz"
    if cache.exists() and len(np.load(cache)["X"]) == len(chips):
        X = np.load(cache)["X"]
    else:
        log.info("embedding %d chips ...", len(chips))
        X = emb([Image.open(PROJECT_ROOT / p) for p in chips["path"]])
        np.savez_compressed(cache, X=X)
    y = chips["label"].to_numpy()
    groups = chips["facility_id"].to_numpy()

    # ---- 1. grouped cross-validation
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, oof)
    acc = accuracy_score(y, oof >= THRESH)
    by_kind = chips.assign(p=oof).groupby("kind").apply(
        lambda d: accuracy_score(d["label"], d["p"] >= THRESH), include_groups=False)
    log.info("grouped 5-fold CV: AUC %.3f, accuracy %.3f", auc, acc)

    model = make_model().fit(X, y)
    joblib.dump(model, ML / "model.joblib")

    # ---- 2. temporal test on held-out labelled facilities
    labels = pd.read_csv(path("outputs_dir") / "review_year_labels.csv", dtype=str)
    labels = labels[labels["usable"] == "True"]
    est = pd.read_csv(path("outputs_dir") / "estimated_build_years.csv", dtype=str, keep_default_na=False)
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["sensor"] == "naip") & (man["image_available"] == "True") & (man["status"] == "ok")]
    man = man.drop_duplicates(["facility_id", "year"], keep="last")
    rows, fac_rows = [], []
    for fid in sorted(labels["facility_id"].unique()):
        m = man[man["facility_id"] == fid].assign(year=lambda d: d["year"].astype(int)).sort_values("year")
        if m.empty:
            continue
        ims = [center_crop(PROJECT_ROOT / p, float(h)) for p, h in zip(m["image_path"], m["roi_half_width_m"])]
        probs = model.predict_proba(emb(ims))[:, 1]
        lab = labels[labels["facility_id"] == fid].set_index("year")["label"]
        for yr, p in zip(m["year"], probs):
            human = lab.get(str(yr), "")
            rows.append({"facility_id": fid, "year": yr, "p_elevator": round(float(p), 3),
                         "ml_call": "present" if p >= THRESH else "absent", "label": human})
        e = est[est["facility_id"] == fid].iloc[0]
        fp = first_present(list(m["year"]), list(probs))
        fac_rows.append({"facility_id": fid, "ml_first_present_naip": fp,
                         "label_first_present": e["first_year_present"],
                         "label_window": e["estimated_build_window"], "label_conf": e["build_year_confidence"],
                         "reviewed_by": e.get("reviewed_by", "")})
    t = pd.DataFrame(rows)
    t.to_csv(path("outputs_dir") / "ml_temporal_test.csv", index=False)
    f = pd.DataFrame(fac_rows)
    scored = t[t["label"].isin(["absent", "present"])]
    yr_acc = accuracy_score(scored["label"], scored["ml_call"]) if len(scored) else float("nan")

    rep = [
        "# Phase 8 proof of concept: elevator-presence classifier", "",
        "Model: frozen ImageNet ResNet-50 embeddings + logistic regression; NAIP chips 300 m x 300 m.", "",
        f"Training chips: {len(chips)} ({int(y.sum())} elevator, {int((1 - y).sum())} not) from "
        f"{chips['facility_id'].nunique()} facilities; the {f.shape[0]} labelled facilities are held out.", "",
        "## 1. Chip classification (5-fold CV grouped by facility)", "",
        f"- ROC AUC: **{auc:.3f}**", f"- Accuracy at p>=0.5: **{acc:.3f}**", "",
        "Accuracy by chip type:", "", by_kind.rename("accuracy").round(3).to_markdown(), "",
        "## 2. Temporal test on held-out labelled facilities", "",
        f"Year-level agreement with absent/present labels: **{yr_acc:.3f}** on {len(scored)} labelled NAIP years.", "",
        f.to_markdown(index=False), "",
        "ml_first_present_naip = first NAIP year from which the model calls the elevator present "
        "(allowing one later miss). NAIP starts in 2003, so 2003-2005 means 'present at start of NAIP'.",
    ]
    (path("outputs_dir") / "ml_eval_report.md").write_text("\n".join(rep), encoding="utf-8")
    log.info("temporal year accuracy %.3f; report written", yr_acc)
    print("\n".join(rep))


if __name__ == "__main__":
    main()
