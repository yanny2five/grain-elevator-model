"""Summarise human answers from the date check page (2026-11-08).

Reads data/reviews/date_check_review.csv (exported from outputs/review/date_check.html)
and compares each answer with the model's call. Writes outputs/date_check_summary.md
and outputs/date_check_location_problems.csv (sites where the reviewer saw no elevator,
which are location problems, not dating errors).
Usage: python src/ml/date_check_summary.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def outcome(r):
    lab = str(r.human_label)
    if lab == "no elevator":
        return "no elevator at location"
    if lab == "unsure":
        return "unsure"
    if lab.startswith("agree"):
        return "agrees"
    human_start = lab.startswith("present at start")
    if r.model_call == "present at start":
        return "agrees" if human_start else "disagrees"
    if r.model_call.startswith("first visible"):
        return "agrees" if (not human_start and str(r.human_first_visible) == r.model_call.split()[-1]) else "disagrees"
    return "disagrees"


def main():
    r = pd.read_csv(ROOT / "data/reviews/date_check_review.csv", dtype=str, keep_default_na=False)
    r["outcome"] = r.apply(outcome, axis=1)
    lines = ["# Date check: human answers vs model (auto-generated)", "",
             f"Sites answered: {len(r)}", ""]
    for kind, g in r.groupby("kind"):
        c = g.outcome.value_counts()
        dated = int(c.get("agrees", 0) + c.get("disagrees", 0))
        lines.append(f"## {kind} ({len(g)} sites)")
        for k, v in c.items():
            lines.append(f"- {k}: {v}")
        if dated:
            lines.append(f"- date agreement where an elevator was seen: {int(c.get('agrees', 0))} of {dated}")
        lines.append("")
    lines.append("Sites marked 'no elevator' go to outputs/date_check_location_problems.csv for a location fix.")
    (ROOT / "outputs/date_check_summary.md").write_text("\n".join(lines), encoding="utf-8")
    r[r.outcome == "no elevator at location"].to_csv(ROOT / "outputs/date_check_location_problems.csv", index=False)
    print("\n".join(lines))
    print(r[["facility_id", "model_call", "human_label", "outcome"]].to_string(index=False))


if __name__ == "__main__":
    main()
