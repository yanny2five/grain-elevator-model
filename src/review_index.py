"""One start page for all person-review pages: outputs/review/index.html

Lists each 'pick the elevator' page with its town count and the location-check
page with its decision count, plus how to save and merge the downloads.
Picks already saved in the browser (localStorage) show as progress on each page itself.
Usage: python src/review_index.py   (then open http://127.0.0.1:8765/outputs/review/index.html)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT  # noqa: E402

REVIEW = PROJECT_ROOT / "outputs" / "review"


def main() -> None:
    tc = pd.read_csv(PROJECT_ROOT / "data" / "ml" / "town_candidates.csv", dtype=str, keep_default_na=False)
    tc = tc[tc["status"] == "candidates"].drop_duplicates(["state", "town"])
    counts = tc["state"].value_counts().sort_values(ascending=False)
    checks = sorted((PROJECT_ROOT / "data" / "addresses").glob("location_check_2026-*.csv"))
    n_check = pd.concat([pd.read_csv(f, dtype=str) for f in checks]).drop_duplicates("facility_id").shape[0] if checks else 0
    rows = "\n".join(
        f'<tr><td><a href="town_pick_{st}.html">{st}</a></td><td>{n}</td></tr>' for st, n in counts.items()
        if (REVIEW / f"town_pick_{st}.html").exists())
    (REVIEW / "index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Review pages</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:760px;color:#222;background:#f6f6f4}}
table{{border-collapse:collapse;background:#fff}} td,th{{border:1px solid #ddd;padding:6px 12px;text-align:left}}
code{{background:#eee;padding:1px 4px;border-radius:3px}} h2{{margin-top:28px}}</style></head><body>
<h1>Elevator project: review pages</h1>
<h2>1. Check Claude's location decisions ({n_check})</h2>
<p><a href="location_check.html">location_check.html</a>. Red is the old point and green is Claude's new point.
Click agree, disagree or unsure. Start with the "move" filter. Download the CSV and save it as
<code>data/reviews/location_check_review.csv</code>.</p>
<h2>1b. Check the model's uncertain dates</h2>
<p><a href="date_check.html">date_check.html</a>. Each row is one site, one photo per year. The yellow box is the
model's first visible year. Click the first year you see the elevator, or Agree / Present at start / No elevator /
Unsure. Download the CSV and save it as <code>data/reviews/date_check_review.csv</code>.</p>
<h2>2. Pick the elevator in unlocated towns ({counts.sum()} towns)</h2>
<p>For each town, click A to D if a yellow circle is on a grain elevator, otherwise click none or unsure. Press
<b>Download picks</b>, save the file into <code>data/reviews/town_picks/</code>, then run<br>
<code>python src/ml/town_pick_merge.py data/reviews/town_picks/town_picks_XX.csv --reviewer "your name"</code></p>
<table><tr><th>State</th><th>Towns</th></tr>
{rows}
</table>
<p>Your clicks are saved in this browser as you go, so you can stop and come back.</p>
</body></html>""", encoding="utf-8")
    print(f"index: {len(counts)} states, {counts.sum()} towns, {n_check} location checks")


if __name__ == "__main__":
    main()
