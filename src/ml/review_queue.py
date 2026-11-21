"""Build a short human-review queue of the ML's confident build-year suggestions.

outputs/review/ml_check.html lists facilities where the classifier found an
absent->present transition with HIGH/MEDIUM suggestion confidence, each with
the suggested window and a link to the facility's review page (which shows the
ML score on every NAIP image). Checking these gives a real accuracy figure for
the ML suggestions and new human labels for retraining.

Usage: python src/ml/review_queue.py
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, path  # noqa: E402


def main() -> None:
    s = pd.read_csv(path("outputs_dir") / "ml_first_present.csv", dtype=str, keep_default_na=False)
    m = pd.read_csv(path("outputs_dir") / "master_facility_dataset.csv", dtype=str, keep_default_na=False)
    m = m.set_index("facility_id")
    reviewed = set(pd.concat([pd.read_csv(f, dtype=str) for f in (PROJECT_ROOT / "data" / "reviews").glob("*.csv")])
                   ["facility_id"])
    q = s[(s["ml_confidence"].isin(["HIGH", "MEDIUM"])) & s["ml_last_absent"].ne("") & s["ml_first_present"].ne("")]
    q = q[~q["facility_id"].isin(reviewed)].sort_values(["ml_confidence", "ml_first_present"])
    rows = []
    for _, r in q.iterrows():
        f = m.loc[r["facility_id"]]
        rows.append(
            f"<tr><td><a href='{r['facility_id']}.html'>{r['facility_id']}</a></td>"
            f"<td>{html.escape(f['warehouse_name'])}</td><td>{html.escape(f['city'])}, {f['state']}</td>"
            f"<td><b>{r['ml_window']}</b></td><td>{r['ml_confidence']}</td>"
            f"<td>{r['fit_cost_per_year']}</td></tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>ML suggestions to check</title>
<body style="font:15px system-ui;margin:24px;max-width:1000px">
<h1>ML build-year suggestions to check ({len(rows)})</h1>
<p>For each facility, open its page and look at the NAIP row around the suggested window (each image shows
<i>ML p</i>, the model's probability that an elevator is in the centre). Label the NAIP years you can judge
(A absent / C construction / P present / ? unclear) - especially the years either side of the suggested change -
then <b>Download labels CSV</b> into <code>data/reviews/</code> (or tell Claude you are done).</p>
<p>The model counts construction as present, so its year matches the <i>first visible</i> year.</p>
<table border=1 cellpadding=6 style="border-collapse:collapse">
<tr><th>Facility</th><th>Name</th><th>Town</th><th>ML suggested window</th><th>ML confidence</th>
<th>series noise</th></tr>{''.join(rows)}</table></body>"""
    out = path("outputs_dir") / "review" / "ml_check.html"
    out.write_text(page, encoding="utf-8")
    print(out, len(rows))


if __name__ == "__main__":
    main()
