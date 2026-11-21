"""Review triage for ML dates (2026-11-07).

Question: can the model's own scores tell us which sites need a human date check?

Rule tested: a site is CLEAR when every NAIP year scored P(elevator) <= 0.2 or >= 0.8
(no "in-between" year). Otherwise it is AMBIGUOUS and goes to human review.

1. Test the rule on the 113 labelled sites, using the leave-one-site-out M2
   scores from outputs/ml_dating_experiment.csv (M4 decision = the rule in use).
2. Apply it to all located sites (outputs/ml_presence_by_year.csv) and write a
   review queue: data/reviews/dating_review_queue.csv (ambiguous sites first,
   then imprecise-location sites).
3. Sample sizes needed to estimate accuracy from a random review sample.

Writes outputs/review_triage.md.
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LO, HI = 0.2, 0.8


def n_for_margin(p, margin, z=1.96):
    return math.ceil(z * z * p * (1 - p) / margin ** 2)


def main():
    exp = pd.read_csv(ROOT / "outputs/ml_dating_experiment.csv")
    exp["amb"] = exp.p2.apply(lambda s: sum(LO < x < HI for x in json.loads(s)))
    exp["clear"] = exp.amb == 0
    t = exp.groupby("clear").M4_ok.agg(["count", "sum"])

    by = pd.read_csv(ROOT / "outputs/ml_presence_by_year.csv", dtype={"facility_id": str})
    qf = ROOT / "outputs/naip_image_quality.csv"
    if qf.exists():  # skip blank (no-data) images, same rule as apply.py
        qd = pd.read_csv(qf, dtype={"facility_id": str})
        bl = set(zip(qd.loc[qd.dark_frac >= 0.9, "facility_id"], qd.loc[qd.dark_frac >= 0.9, "year"].astype(int)))
        by = by[[(f, int(y)) not in bl for f, y in zip(by.facility_id, by.year)]]
    amb =by.groupby("facility_id").p_elevator.apply(lambda s: int(((s > LO) & (s < HI)).sum()))
    fp = pd.read_csv(ROOT / "outputs/ml_first_present.csv", dtype={"facility_id": str})
    fp["ambiguous_years"] = fp.facility_id.map(amb).fillna(0).astype(int)
    fp["clear"] = fp.ambiguous_years == 0
    fp["precise_location"] = fp.chip_mode == "centre_300m"
    fp["queue"] = "skip (clear)"
    fp.loc[~fp.clear & fp.precise_location, "queue"] = "1 review date (ambiguous scores)"
    fp.loc[~fp.precise_location, "queue"] = "2 fix location first (imprecise)"
    q = fp[fp.queue != "skip (clear)"].sort_values(["queue", "ambiguous_years"], ascending=[True, False])
    out = ROOT / "data/reviews/dating_review_queue.csv"
    q[["facility_id", "queue", "ambiguous_years", "ml_first_present", "ml_window", "ml_confidence", "ml_flags"]].to_csv(out, index=False)

    precise = fp[fp.precise_location]
    lines = ["# Review triage for ML dates", "",
             f"Rule: CLEAR = no NAIP year with {LO} < P(elevator) < {HI}.", "",
             "## 1. Test on the 113 labelled sites (leave-one-site-out scores, M4 rule)", "",
             "| Group | Sites | Date agrees with label |", "|---|---|---|"]
    for k, name in [(True, "Clear"), (False, "Ambiguous")]:
        c, s = int(t.loc[k, "count"]), int(t.loc[k, "sum"])
        lines.append(f"| {name} | {c} | {s} ({s / c:.0%}) |")
    lines += ["", f"Errors caught by reviewing only ambiguous sites: "
                  f"{int(t.loc[False, 'count'] - t.loc[False, 'sum'])} of {int((~exp.M4_ok).sum())}.", "",
              "Caveat: labels are mostly Claude's drafts (3 of 113 human-checked).", "",
              "## 2. Applied to all located sites", "",
              f"- Located sites scored: {len(fp)}",
              f"- Precise location (centre chip): {len(precise)}; clear {int(precise.clear.sum())}, "
              f"ambiguous {int((~precise.clear).sum())}",
              f"- Imprecise location (grid search): {int((~fp.precise_location).sum())} -> fix location first",
              f"- Review queue written: {out.relative_to(ROOT)} ({len(q)} rows)", "",
              "## 3. Random-sample sizes (95% confidence)", "",
              "| Expected accuracy | +/-10 points | +/-7 points | +/-5 points |", "|---|---|---|---|"]
    for p in [0.7, 0.85]:
        lines.append(f"| {p:.0%} | {n_for_margin(p, .10)} | {n_for_margin(p, .07)} | {n_for_margin(p, .05)} |")
    (ROOT / "outputs/review_triage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
