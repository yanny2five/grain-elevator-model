"""Phase 6-7 - turn review labels into first-appearance / disappearance estimates.

Input : data/reviews/*.csv (exported from the review pages; one row per labelled image)
        outputs/imagery_index.csv / data/imagery/manifest.csv (imagery coverage)
Output: outputs/estimated_build_years.csv

Terminology (deliberately conservative)
---------------------------------------
estimated_first_visible_year      first year any structure/site work is seen
                                  (first 'construction' or 'present' label)
first_year_present                first year the elevator is clearly standing
last_year_absent                  last year the site is seen empty before that
estimated_build_window            "<last_absent+1>-<first_present>", i.e. the
                                  years in which construction happened
estimated_construction_start      first 'construction' label, if any
These describe PHYSICAL APPEARANCE IN IMAGERY. They are not opening,
licensing or operating dates.

Combining sensors for one year
------------------------------
Each image is labelled separately. For a year, the label from the sharpest
sensor with a decisive call wins (NAIP > Sentinel-2 > Landsat); 'unclear'
never overrides a decisive label. If two decisive labels of the *same*
sensor rank conflict, the year is marked conflicting and excluded.

Confidence
----------
HIGH    window <= 2 years and both bounding years come from NAIP/Sentinel-2
MEDIUM  window <= 5 years, or a bound comes from Landsat only
LOW     wider window, inconsistent sequence (absent after present without
        'removed'), or only one side bounded
PRESENT_AT_START  present in the earliest usable image: built on or before that
                  year (left-censored) - not a build year

Disappearance (Phase 7) is reported only when at least two usable years after
presence are labelled 'removed' (or absent), and is phrased as
possible_disappearance_year. It does not mean closure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, get_logger, path  # noqa: E402

log = get_logger("estimate_build_year")
RANK = {"naip": 0, "sentinel2": 1, "landsat": 2}
DECISIVE = {"absent", "construction", "present", "removed"}


def load_reviews() -> pd.DataFrame:
    files = sorted((PROJECT_ROOT / "data" / "reviews").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=["facility_id", "sensor", "year", "label", "reviewer", "review_date", "facility_notes"])
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False).assign(_file=f.name) for f in files])
    df["year"] = df["year"].astype(int)
    # The latest review of an image wins (re-reviews overwrite earlier ones).
    return df.sort_values("review_date").drop_duplicates(["facility_id", "sensor", "year"], keep="last")


def year_labels(rv: pd.DataFrame) -> pd.DataFrame:
    """One label per facility-year, from the sharpest decisive sensor."""
    rv = rv.assign(_r=rv["sensor"].map(RANK), _d=rv["label"].isin(DECISIVE))
    out = []
    for (fid, y), g in rv.groupby(["facility_id", "year"]):
        dec = g[g["_d"]]
        if dec.empty:
            out.append((fid, y, "unclear", g["sensor"].iloc[0], False))
            continue
        top = dec[dec["_r"] == dec["_r"].min()]
        labs = set(top["label"])
        if len(labs) > 1:
            out.append((fid, y, "conflict", ";".join(sorted(top["sensor"])), False))
        else:
            out.append((fid, y, labs.pop(), top["sensor"].iloc[0], True))
    return pd.DataFrame(out, columns=["facility_id", "year", "label", "sensor", "usable"])


def estimate(fid: str, yl: pd.DataFrame, cov: pd.DataFrame) -> dict:
    yl = yl.sort_values("year")
    use = yl[yl["usable"]]
    res = {"facility_id": fid,
           "earliest_imagery_year": int(cov["year"].min()) if len(cov) else "",
           "latest_imagery_year": int(cov["year"].max()) if len(cov) else "",
           "n_years_labelled": int(len(use)), "flags": []}
    built = use[use["label"].isin(["construction", "present"])]
    if built.empty:
        res.update(build_year_confidence="NOT_OBSERVED" if len(use) else "NOT_REVIEWED")
        if len(use) and (use["label"] == "absent").all():
            res["flags"].append("no structure seen in any labelled year - check location")
        return finish(res)

    first_vis = int(built["year"].min())
    present = use[use["label"] == "present"]
    first_present = int(present["year"].min()) if len(present) else None
    constr = use[use["label"] == "construction"]
    before = use[(use["year"] < first_vis) & (use["label"] == "absent")]
    last_absent = int(before["year"].max()) if len(before) else None
    res.update(estimated_first_visible_year=first_vis, first_year_present=first_present or "",
               last_year_absent=last_absent or "",
               estimated_construction_start=int(constr["year"].min()) if len(constr) else "")

    # Sequence check: 'absent' after first appearance without a 'removed' in between.
    after = use[use["year"] > first_vis]
    rem = after[after["label"] == "removed"]
    first_removed = int(rem["year"].min()) if len(rem) else 9999
    odd = after[(after["label"] == "absent") & (after["year"] < first_removed)]
    if len(odd):
        res["flags"].append(f"absent after appearance in {','.join(map(str, odd['year']))} - re-review")

    end = first_present or first_vis
    if last_absent is None:
        res["estimated_build_window"] = f"<= {end}"
        res["build_year_confidence"] = "PRESENT_AT_START"
        res["flags"].append(f"present in earliest usable labelled year ({first_vis}); built on or before then")
    else:
        res["estimated_build_window"] = f"{last_absent + 1}-{end}" if end > last_absent + 1 else f"{end}"
        width = end - last_absent
        bound_sensors = {use.loc[use["year"] == last_absent, "sensor"].iloc[0],
                         use.loc[use["year"] == end, "sensor"].iloc[0]}
        if odd.empty and width <= 2 and bound_sensors <= {"naip", "sentinel2"}:
            conf = "HIGH"
        elif odd.empty and width <= 5:
            conf = "MEDIUM"
        else:
            conf = "LOW"
        res["build_year_confidence"] = conf
        res["window_years"] = width
        res["window_bound_sensors"] = ";".join(sorted(bound_sensors))

    # Phase 7: conservative disappearance.
    post = use[use["year"] > end]
    gone = post[post["label"].isin(["removed"])]
    if len(gone) >= 2 or (len(gone) == 1 and len(post[(post["year"] > int(gone["year"].min()))
                                                       & post["label"].isin(["removed", "absent"])]) >= 1):
        first_gone = int(gone["year"].min())
        pres_before = use[(use["year"] < first_gone) & use["label"].isin(["present", "construction"])]
        lp = int(pres_before["year"].max()) if len(pres_before) else ""
        res.update(possible_last_year_present=lp, first_year_absent_after_operation=first_gone,
                   possible_disappearance_year=f"{lp + 1}-{first_gone}" if lp != "" and first_gone > lp + 1 else first_gone)
        res["flags"].append("structures not visible after presence - physical removal only, NOT evidence of closure")
    return finish(res)


def finish(res: dict) -> dict:
    res["flags"] = " | ".join(res["flags"])
    return res


def main() -> None:
    rv = load_reviews()
    man = pd.read_csv(path("imagery_dir") / "manifest.csv", dtype=str, keep_default_na=False)
    man = man[(man["status"] == "ok") & (man["image_available"] == "True")]
    man["year"] = man["year"].astype(int)
    yl = year_labels(rv) if len(rv) else pd.DataFrame(columns=["facility_id", "year", "label", "sensor", "usable"])
    rows = []
    for fid in sorted(set(man["facility_id"]) | set(rv["facility_id"])):
        rows.append(estimate(fid, yl[yl["facility_id"] == fid], man[man["facility_id"] == fid]))
    cols = ["facility_id", "earliest_imagery_year", "last_year_absent", "estimated_construction_start",
            "estimated_first_visible_year", "first_year_present", "estimated_build_window", "window_years",
            "window_bound_sensors", "build_year_confidence", "latest_imagery_year", "possible_last_year_present",
            "first_year_absent_after_operation", "possible_disappearance_year", "n_years_labelled", "flags"]
    out = pd.DataFrame(rows).reindex(columns=cols)
    for c in ["earliest_imagery_year", "last_year_absent", "estimated_construction_start", "estimated_first_visible_year",
              "first_year_present", "window_years", "latest_imagery_year", "possible_last_year_present",
              "first_year_absent_after_operation", "n_years_labelled"]:
        out[c] = pd.to_numeric(out[c].replace("", None), errors="coerce").astype("Int64")
    nonblank = rv[rv["facility_notes"].astype(str).str.strip() != ""] if len(rv) else rv
    notes = nonblank.groupby("facility_id")["facility_notes"].last() if len(nonblank) else pd.Series(dtype=str)
    reviewers = rv.groupby("facility_id")["reviewer"].agg(lambda s: " + ".join(dict.fromkeys(s))) if len(rv) else pd.Series(dtype=str)
    out["review_notes"] = out["facility_id"].map(notes).fillna("")
    out["reviewed_by"] = out["facility_id"].map(reviewers).fillna("")
    out.to_csv(path("outputs_dir") / "estimated_build_years.csv", index=False)
    yl.to_csv(path("outputs_dir") / "review_year_labels.csv", index=False)
    log.info("build_year_confidence: %s", out["build_year_confidence"].value_counts().to_dict())


if __name__ == "__main__":
    main()
