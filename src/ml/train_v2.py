"""Train detector v2: original weak-label chips + every labelled year image.

Why: v1 was trained only on recent (>= 2015) imagery, so it often scored older
NAIP years (haze, false colour, 2 m scans) as "no elevator" even when one was
there. dating_experiment.py showed that adding labelled year images (all eras)
raises facility-level dating agreement from 20/37 (v1) to 25/37, and to 30/37
together with robust_step_fit (leave-one-facility-out).

Same architecture as v1: frozen ImageNet ResNet-50 embeddings + logistic regression.
v1 (data/ml/model.joblib) is kept; apply.py uses v2 when it exists.

Output: data/ml/model_v2.joblib, data/ml/model_v2_training.csv (what went in)
Usage:  python src/ml/train_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import center_crop  # noqa: E402
from candidate_review import GpuEmbedder  # noqa: E402
from dating_experiment import label_table  # noqa: E402
from train_eval import make_model  # noqa: E402
from utils import PROJECT_ROOT, get_logger, load_config  # noqa: E402

log = get_logger("ml_train_v2")
ML = PROJECT_ROOT / "data" / "ml"


def main() -> None:
    half_m = float(load_config()["earth_engine"]["roi_half_width_m"])
    lab = label_table().dropna(subset=["y"])
    lab["path"] = [PROJECT_ROOT / "data" / "imagery" / f / "naip" / f"{y}.png" for f, y in zip(lab["facility_id"], lab["year"])]
    lab = lab[[p.exists() for p in lab["path"]]]
    Xy = GpuEmbedder()([center_crop(p, half_m) for p in lab["path"]])

    chips = pd.read_csv(ML / "chips.csv")
    Xc = np.load(ML / "embeddings.npz")["X"]
    X = np.vstack([Xc, Xy])
    y = np.concatenate([chips["label"].to_numpy(), lab["y"].astype(int).to_numpy()])
    model = make_model().fit(X, y)
    joblib.dump(model, ML / "model_v2.joblib")
    pd.concat([chips.assign(source="v1_chips")[["facility_id", "label", "source"]],
               lab.assign(label=lab["y"].astype(int), source="labelled_year_image")[["facility_id", "year", "label", "source"]]]
              ).to_csv(ML / "model_v2_training.csv", index=False)
    log.info("v2 trained on %d chips + %d labelled year images (%d facilities); positives %d, negatives %d",
             len(Xc), len(Xy), lab["facility_id"].nunique(), int(y.sum()), int((y == 0).sum()))


if __name__ == "__main__":
    main()
