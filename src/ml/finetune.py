"""Fine-tune the whole ResNet-50 on the GPU (follow-up to tune_search.py).

The frozen-feature detector cannot locate elevators alone (best: top pick
correct in 9/20 held-out towns). Here every layer is trained on:
  original facility chips + 3.6k mined town negatives + jittered scan positives,
with augmentation (flips, 90-degree rotations, colour jitter, mild crops).
The 20 held-out test towns are excluded from training entirely.

GPU safety: a watchdog reads the GPU temperature (nvidia-smi) every 20 batches
and pauses training at >= 83 C until it cools to <= 75 C. Mixed precision,
batch size 32, and a hard wall-clock cap keep the load moderate. No power,
clock or driver settings are changed.

Evaluation (same as before, so results are comparable):
  1. town search on the 20 held-out towns (found / top correct / false sites)
  2. year-by-year test on the 8 labelled facilities (dating must not get worse)
Outputs: data/ml/model_finetuned.pt, outputs/ml_finetune_report.md
Usage:   python src/ml/finetune.py [--minutes 35]
"""
from __future__ import annotations

import argparse
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import SIZE, center_crop  # noqa: E402
from find_addresses import haversine_m  # noqa: E402
from hard_negatives import chips_with_centres, crop_at  # noqa: E402
from town_scan import OUT  # noqa: E402
from train_eval import step_fit  # noqa: E402
from tune_search import JITTER_M, sites  # noqa: E402
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("finetune")
ML = PROJECT_ROOT / "data" / "ml"
TEMP_PAUSE, TEMP_RESUME = 83, 75
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
random.seed(11)
torch.manual_seed(11)


# ---------------------------------------------------------------- GPU watchdog
def gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def cool_down_if_hot(stats: dict) -> None:
    t = gpu_temp()
    if t is None:
        return
    stats["max_temp"] = max(stats.get("max_temp", 0), t)
    if t >= TEMP_PAUSE:
        log.warning("GPU at %d C - pausing until <= %d C", t, TEMP_RESUME)
        stats["pauses"] = stats.get("pauses", 0) + 1
        while (t := gpu_temp()) is not None and t > TEMP_RESUME:
            time.sleep(10)
        log.info("GPU back to %s C - resuming", t)


# ---------------------------------------------------------------- data
class Chips(Dataset):
    def __init__(self, paths, labels, train: bool):
        self.paths, self.labels = list(paths), list(labels)
        aug = [T.RandomResizedCrop(SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(),
               T.RandomVerticalFlip(), T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
               T.ColorJitter(0.25, 0.25, 0.2, 0.02)]
        self.tf = T.Compose((aug if train else [T.Resize(SIZE)]) + [T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), torch.tensor(float(self.labels[i]))


def build_training_table(test_ids: set[str]) -> pd.DataFrame:
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    base = pd.read_csv(ML / "chips_v2.csv", dtype={"facility_id": str})
    base["path"] = [str(PROJECT_ROOT / p) for p in base["path"]]
    # Jittered positives from the town-scan images (same domain as search time), saved once.
    hn = pd.read_csv(ML / "hard_negatives.csv", dtype={"facility_id": str})
    rows = []
    for fid in sorted(hn.loc[hn["kind"] == "scan_positive", "facility_id"]):
        f = OUT / f"hn_{fid}.jpg"
        if not f.exists():
            continue
        lat, lon = float(loc.loc[fid, "latitude"]), float(loc.loc[fid, "longitude"])
        img = None
        for dy in JITTER_M:
            for dx in JITTER_M:
                out = ML / "hard_neg" / f"posj_{fid}_{dy}_{dx}.png"
                if not out.exists():
                    img = img or Image.open(f).convert("RGB")
                    crop_at(img, lat, lon, lat + dy / 111_320, lon + dx / (111_320 * math.cos(math.radians(lat))),
                            4000 / img.size[0]).resize((SIZE, SIZE), Image.BILINEAR).save(out)
                rows.append({"path": str(out), "label": 1, "facility_id": fid, "kind": "scan_positive_jitter"})
    df = pd.concat([base[["path", "label", "facility_id", "kind"]], pd.DataFrame(rows)], ignore_index=True)
    return df[~df["facility_id"].isin(test_ids)].reset_index(drop=True)


# ---------------------------------------------------------------- model
def make_model() -> nn.Module:
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, 1))
    return m


@torch.no_grad()
def predict(model, images: list[Image.Image], dev, batch=64) -> np.ndarray:
    tf = T.Compose([T.Resize(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])
    model.eval()
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([tf(im.convert("RGB")) for im in images[i:i + batch]]).to(dev)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(x).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=35)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s %s", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "")
    test_ids = set(pd.read_csv(path("outputs_dir") / "ml_search_eval.csv", dtype=str)["facility_id"])
    df = build_training_table(test_ids)
    groups = sorted(df["facility_id"].unique())
    random.shuffle(groups)
    val_g = set(groups[: max(1, len(groups) // 7)])
    tr, va = df[~df["facility_id"].isin(val_g)], df[df["facility_id"].isin(val_g)]
    log.info("train %d chips (%d pos), val %d chips (%d pos); test towns excluded: %d",
             len(tr), int(tr["label"].sum()), len(va), int(va["label"].sum()), len(test_ids))
    # Balanced sampling: each batch ~half elevators.
    w = np.where(tr["label"] == 1, 0.5 / tr["label"].sum(), 0.5 / (1 - tr["label"]).sum())
    sampler = torch.utils.data.WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
    dl = DataLoader(Chips(tr["path"], tr["label"], True), batch_size=32, sampler=sampler, num_workers=2)
    dv = DataLoader(Chips(va["path"], va["label"], False), batch_size=64, num_workers=2)

    model = make_model().to(dev)
    head = list(model.fc.parameters())
    body = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 5e-5}, {"params": head, "lr": 5e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    lossf = nn.BCEWithLogitsLoss()
    t0, stats, best = time.time(), {}, (1e9, None)
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for b, (x, y) in enumerate(dl):
            if b % 20 == 0:
                cool_down_if_hot(stats)
            x, y = x.to(dev, non_blocking=True), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = lossf(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            if time.time() - t0 > a.minutes * 60:
                break
        # validation loss (grouped by facility)
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for x, y in dv:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                    vl += lossf(model(x).squeeze(1), y).item() * len(y)
                n += len(y)
        vl /= max(n, 1)
        log.info("epoch %d: train loss %.4f, val loss %.4f, %.1f min, max GPU temp %s C, pauses %s",
                 ep, tot / len(tr), vl, (time.time() - t0) / 60, stats.get("max_temp"), stats.get("pauses", 0))
        if vl < best[0]:
            best = (vl, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save(best[1], ML / "model_finetuned.pt")
        if time.time() - t0 > a.minutes * 60:
            log.info("time cap reached")
            break
    model.load_state_dict(best[1])

    # ---- 1. town search on held-out towns
    loc = pd.read_csv(path("outputs_dir") / "facility_locations.csv", dtype=str, keep_default_na=False).set_index("facility_id")
    towns = []
    for fid in sorted(test_ids):
        f = OUT / f"hn_{fid}.jpg"
        if not f.exists():
            continue
        img = Image.open(f).convert("RGB")
        lat, lon = float(loc.loc[fid, "latitude"]), float(loc.loc[fid, "longitude"])
        cells, _ = chips_with_centres(img, lat, lon)
        cool_down_if_hot(stats)
        p = predict(model, [img.crop(bx).resize((SIZE, SIZE), Image.BILINEAR) for bx, _, _ in cells], dev)
        towns.append((lat, lon, p, [(la, lo) for _, la, lo in cells]))
    rows = []
    for rule in ["threshold", "top1"]:
        for th in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
            found = topok = false = 0
            for lat, lon, p, cen in towns:
                s = sites(p, cen, th, top1=(rule == "top1"))
                hit = [x for x in s if haversine_m(x[0], x[1], lat, lon) < 400]
                found += bool(hit)
                topok += bool(s) and haversine_m(s[0][0], s[0][1], lat, lon) < 400
                false += len(s) - len(hit)
            rows.append({"rule": rule, "threshold": th, "found": found, "top_correct": topok,
                         "false_sites_per_town": round(false / len(towns), 2)})
    search = pd.DataFrame(rows)

    # ---- 2. dating test on the 8 labelled facilities
    labels = pd.read_csv(path("outputs_dir") / "review_year_labels.csv", dtype=str)
    labels = labels[labels["usable"] == "True"]
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["sensor"] == "naip") & (man["image_available"] == "True") & (man["status"] == "ok")]
    man = man.drop_duplicates(["facility_id", "year"], keep="last")
    agree = total = 0
    fac_rows = []
    for fid in sorted(labels["facility_id"].unique()):
        m = man[man["facility_id"] == fid].assign(y=lambda d: d["year"].astype(int)).sort_values("y")
        ims = [center_crop(PROJECT_ROOT / p, float(h)) for p, h in zip(m["image_path"], m["roi_half_width_m"])]
        p = predict(model, ims, dev)
        lab = labels[labels["facility_id"] == fid].set_index("year")["label"]
        for yr, pp in zip(m["y"], p):
            l = lab.get(str(yr))
            if l in ("absent", "present"):
                total += 1
                agree += (pp >= 0.5) == (l == "present")
        fit = step_fit(list(m["y"]), list(p))
        fac_rows.append({"facility_id": fid, "finetuned_first_present": fit["first_present"],
                         "present_at_start": fit["present_at_start"]})
    rep = ["# Fine-tuned detector (whole ResNet-50 on GPU)", "",
           f"Training chips: {len(tr)} (+{len(va)} validation), test towns excluded. "
           f"Max GPU temperature {stats.get('max_temp')} C, cooling pauses {stats.get('pauses', 0)}.", "",
           "## Town search on the 20 held-out towns", "",
           "Previous best (frozen features): top pick correct 9/20 with 0.55 false sites/town; "
           "or elevator among candidates in 16/20 with 3.2 false sites/town.", "",
           search.to_markdown(index=False), "",
           "## Dating test on the 8 labelled facilities", "",
           f"Year-level agreement with absent/present labels: {agree}/{total} "
           f"(frozen-feature model: 0.928).", "", pd.DataFrame(fac_rows).to_markdown(index=False)]
    (path("outputs_dir") / "ml_finetune_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
