"""Phase 1 - clean the raw WCMD export and build the facility master table.

The raw CSV was produced by extracting tables from the WCMD "Commodity
Warehouses" PDF (161 pages). That extraction left three kinds of damage that
this script detects but does NOT silently "fix":

1. Merged rows. ~190 rows contain two WCMD records squashed into one line,
   e.g. Functional Unit "6 7" and Warehouse Capacity "1,957,000 63,000".
   We split them into one record per functional unit. Fields that can be split
   unambiguously (unit, capacity, license number/type/status, CCC status,
   2-letter state) are split; free-text fields (county, city, entity, name)
   cannot be split reliably, so both parts keep the raw text and are flagged
   for a check against the source PDF page.
2. County/City misalignment. Many rows have a blank city or a county that is
   shifted relative to its city (e.g. JONESBORO listed under Conway County).
   We check every (state, city, county) triple against the Census 2020
   place-by-county file and store the gazetteer's suggestion separately.
3. Text artifacts such as "[8-0015]" fragments inside the city field.

Nothing is overwritten: the raw value of every original column is kept in a
`raw_*` column, and the raw file itself is never modified.

facility_id definition
----------------------
One facility_id per WCMD *record* (= one license/CCC entry x functional unit).
This is the finest unit the data supports and guarantees we never collapse
two separate physical places. The opposite error (two records that are in
fact the same physical site, e.g. one building listed under both a state and
a federal license) is handled later by `site_id`, assigned after geocoding by
clustering coordinates - not by guessing from names.

Usage:  python src/clean_dataset.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config, path, sha256_file  # noqa: E402

log = get_logger("clean_dataset")

RAW_COLUMNS = {
    "State": "state",
    "County": "county",
    "City": "city",
    "CCC Code": "ccc_code",
    "CCC Status": "ccc_status",
    "License Number": "license_number",
    "License Type": "license_type",
    "License Status": "license_status",
    "Business Entity": "business_entity",
    "Warehouse Name": "warehouse_name",
    "Functional Unit": "functional_unit",
    "Warehouse Capacity": "warehouse_capacity",
    "PDF Page": "pdf_page",
}
# Columns added by earlier manual work in the "working" file. Kept as prior_*.
PRIOR_COLUMNS = {
    "Search Query": "prior_search_query",
    "Street Address": "prior_street_address",
    "Latitude": "prior_latitude",
    "Longitude": "prior_longitude",
    "Address Confidence": "prior_address_confidence",
    "Address Source": "prior_address_source",
    "Research Notes": "prior_research_notes",
}

CCC_STATUS_RE = re.compile(r"Not Applicable|Approved|Authorized|Pending|Removed")
STATUS_FIX = {"C ancellatio n": "Cancellation"}


# ----------------------------------------------------------------------------
# Splitting helpers for merged rows
# ----------------------------------------------------------------------------
def split_license_numbers(raw: str, n: int) -> list[str] | None:
    """Split '3-9833] [3-10225', '[3-7221', '3-9862]', ' Unlicensed [] Unlicensed'.

    Brackets mark the boundary between two records in the PDF extraction.
    Returns None if the value has no bracket (= one license shared by all parts).
    """
    s = raw.strip()
    if "[" not in s and "]" not in s:
        return None
    s = s.replace("] [", "|").replace("[]", "|").replace("][", "|")
    if s.startswith("["):
        s = "|" + s[1:]
    if s.endswith("]"):
        s = s[:-1] + "|"
    parts = [p.strip(" []") for p in s.split("|")]
    return parts if len(parts) == n else None


def split_tokens(raw: str, n: int) -> list[str] | None:
    toks = raw.split()
    return toks if len(toks) == n else None


def split_ccc_status(raw: str, n: int) -> list[str] | None:
    toks = CCC_STATUS_RE.findall(raw)
    return toks if len(toks) == n else None


def normalize_status(s: str) -> str:
    s = s.strip()
    for bad, good in STATUS_FIX.items():
        s = s.replace(bad, good)
    return s


def explode_row(row: pd.Series) -> list[dict]:
    """Return one dict per WCMD record contained in a raw row."""
    units = row["functional_unit"].split()
    caps = row["warehouse_capacity"].split()
    n = len(units)
    base = {k: row[k].strip() for k in RAW_COLUMNS.values()}
    base.update({f"raw_{k}": row[k] for k in RAW_COLUMNS.values()})
    base.update({k: row[k] for k in PRIOR_COLUMNS.values()})
    base["raw_row_number"] = int(row["raw_row_number"])
    flags: list[str] = []

    if n <= 1:
        rec = dict(base, split_part=1, n_parts_in_raw_row=1, _flags=flags)
        rec["license_status"] = normalize_status(rec["license_status"])
        return [rec]

    flags.append("MERGED_ROW_SPLIT")
    if len(caps) != n:
        flags.append("CAPACITY_SPLIT_MISMATCH")
        caps = [row["warehouse_capacity"]] * n

    lic_nums = split_license_numbers(row["license_number"], n)
    lic_types = split_tokens(row["license_type"], n)
    lic_status = split_tokens(row["license_status"], n)
    ccc_status = split_ccc_status(row["ccc_status"], n)
    states = split_tokens(row["state"], n)
    if states and not all(len(s) == 2 for s in states):
        states = None

    # Two *different* records were merged if any per-record field differs.
    distinct = bool(
        lic_nums or (lic_types and len(set(lic_types)) > 1) or (ccc_status and len(set(ccc_status)) > 1)
        or (states and len(set(states)) > 1)
    )
    if distinct:
        flags.append("TEXT_FIELDS_FROM_TWO_RECORDS")  # county/city/entity/name hold text of both
    else:
        flags.append("SAME_ENTITY_UNITS_MERGED")  # likely same license, different units/cities

    out = []
    for i in range(n):
        rec = dict(base, split_part=i + 1, n_parts_in_raw_row=n, _flags=list(flags))
        rec["functional_unit"] = units[i]
        rec["warehouse_capacity"] = caps[i]
        if lic_nums:
            rec["license_number"] = lic_nums[i]
        if lic_types:
            rec["license_type"] = lic_types[i]
        elif len(row["license_type"].split()) > 1:
            rec["_flags"].append("LICENSE_TYPE_UNSPLIT")
        if lic_status:
            rec["license_status"] = lic_status[i]
        if ccc_status:
            rec["ccc_status"] = ccc_status[i]
        if states:
            rec["state"] = states[i]
        rec["license_status"] = normalize_status(rec["license_status"])
        if rec["license_status"] == "NA":
            rec["license_status"] = ""
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Census gazetteer checks for state / city / county consistency
# ----------------------------------------------------------------------------
def _norm_place(s: str) -> str:
    s = s.upper().strip()
    s = re.sub(r"\bST\.?\s", "SAINT ", s)
    s = re.sub(r"\bFT\.?\s", "FORT ", s)
    s = re.sub(r"\bMT\.?\s", "MOUNT ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_county(s: str) -> str:
    s = re.sub(r"\s+(County|Parish|Borough|Census Area|city)$", "", s.strip(), flags=re.I)
    return _norm_place(s)


def load_gazetteer() -> tuple[dict, set]:
    """Return {(state, place): set(counties)} and {(state, county)}."""
    pbc = pd.read_csv(path("place_by_county"), sep="|", dtype=str, encoding="latin-1")
    # Place names end with a lowercase legal type ("Ames city", "Alta CDP"); drop it.
    pbc["place"] = pbc["PLACENAME"].str.replace(r"(\s+[a-z][a-z\-]*)+(\s*\(.*\))?$", "", regex=True)
    pbc["place"] = pbc["place"].str.replace(r"\s+CDP$", "", regex=True).map(_norm_place)
    pbc["cty"] = pbc["COUNTYNAME"].map(_norm_county)
    place_counties: dict = {}
    for st, pl, ct in zip(pbc["STATE"], pbc["place"], pbc["cty"]):
        place_counties.setdefault((st, pl), set()).add(ct)
    cty = pd.read_csv(path("county_list"), sep="|", dtype=str, encoding="latin-1")
    counties = {(st, _norm_county(c)) for st, c in zip(cty["STATE"], cty["COUNTYNAME"])}
    return place_counties, counties


def gazetteer_checks(df: pd.DataFrame) -> pd.DataFrame:
    place_counties, counties = load_gazetteer()
    places_by_state: dict = {}
    for st, pl in place_counties:
        places_by_state.setdefault(st, set()).add(pl)

    city_ok, county_ok, match, sugg, hint = [], [], [], [], []
    for st, city, county, name in zip(df["state"], df["city"], df["county"], df["warehouse_name"]):
        c, k = _norm_place(city), _norm_county(county)
        cs = place_counties.get((st, c)) if c else None
        city_ok.append(bool(cs))
        county_ok.append((st, k) in counties if k else False)
        match.append(bool(cs and k in cs))
        sugg.append(";".join(sorted(cs)).title() if cs else "")
        # City hint from the warehouse name, e.g. "JONESBORO RICE WAREHOUSE".
        h = ""
        words = _norm_place(name).split()
        for L in (3, 2, 1):
            cand = " ".join(words[:L])
            if len(words) > L - 1 and cand in places_by_state.get(st, set()):
                h = cand.title()
                break
        hint.append(h)
    df["city_in_census_gazetteer"] = city_ok
    df["county_valid_for_state"] = county_ok
    df["city_county_consistent"] = match
    df["census_counties_for_city"] = sugg
    df["city_hint_from_warehouse_name"] = hint
    return df


# ----------------------------------------------------------------------------
def license_category(lt: str) -> str:
    lt = lt.strip()
    if lt == "Federal":
        return "USWA_FEDERAL"
    if lt == "State":
        return "STATE"
    if lt == "":
        return "UNLICENSED_OR_UNKNOWN"
    return "AMBIGUOUS"


def build_master() -> pd.DataFrame:
    raw_path = path("raw_csv")
    log.info("Raw file %s sha256=%s", raw_path.name, sha256_file(raw_path))
    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    log.info("Raw rows: %d, columns: %d", len(raw), raw.shape[1])
    raw = raw.rename(columns={**RAW_COLUMNS, **PRIOR_COLUMNS})
    raw["raw_row_number"] = range(1, len(raw) + 1)  # 1-based data row in the raw CSV

    recs = [r for _, row in raw.iterrows() for r in explode_row(row)]
    df = pd.DataFrame(recs)
    log.info("Records after splitting merged rows: %d", len(df))

    df["license_category"] = df["license_type"].map(license_category)
    df["is_uswa"] = df["license_category"].eq("USWA_FEDERAL")
    lic = df["license_number"].str.strip()
    df.loc[lic.str.lower().eq("unlicensed") & df["license_type"].eq(""), "license_category"] = "UNLICENSED"
    df["warehouse_capacity_num"] = pd.to_numeric(df["warehouse_capacity"].str.replace(",", ""), errors="coerce")

    df = gazetteer_checks(df)

    # ------------------------------------------------------------------ flags
    def add(mask, flag):
        for i in df.index[mask]:
            df.at[i, "_flags"].append(flag)

    # "Suspend..Hansen-Mueller" + entity "Co.": the PDF cell "Suspended" was
    # truncated and the first line of the entity name bled into it. Keep the
    # raw text; expose the status and the name fragment separately.
    bleed = df["license_status"].str.extract(r"^Suspend\.\.(.+)$")[0]
    df["entity_fragment_from_status"] = bleed.fillna("")
    add(bleed.notna(), "STATUS_ENTITY_TEXT_BLEED")
    df.loc[bleed.notna(), "license_status"] = "Suspended"
    df["license_status_not_issued"] = ~df["license_status"].isin(["Issued", ""])
    add(df["business_entity"].eq("") & df["warehouse_name"].eq(""), "NAME_MISSING")

    add(df["city"].eq(""), "CITY_MISSING")
    add(df["county"].eq(""), "COUNTY_MISSING")
    add(df["city"].str.contains(r"[\[\]()]|\.\.", regex=True), "CITY_TEXT_ARTIFACT")
    add(df["business_entity"].str.contains(r"\.\.\.$") | df["warehouse_name"].str.contains(r"\.\.\.$"), "NAME_TRUNCATED")
    add(df["city"].ne("") & ~df["city_in_census_gazetteer"], "CITY_NOT_IN_GAZETTEER")
    add(df["city_in_census_gazetteer"] & df["county"].ne("") & ~df["city_county_consistent"], "COUNTY_CITY_MISMATCH")
    add(df["license_category"].eq("AMBIGUOUS"), "LICENSE_TYPE_AMBIGUOUS")
    add(df["warehouse_capacity_num"].fillna(0).eq(0), "ZERO_OR_MISSING_CAPACITY")
    add(df["warehouse_name"].str.contains(r"[^\x00-\x7f]", regex=True), "NON_ASCII_CHARS")

    # Duplicate checks (flag, never drop).
    key_exact = ["state", "county", "city", "ccc_code", "license_number", "business_entity",
                 "warehouse_name", "functional_unit", "warehouse_capacity"]
    add(df.duplicated(key_exact, keep=False), "EXACT_DUPLICATE_RECORD")
    fed = df["is_uswa"] & df["license_number"].ne("")
    add(fed & df.duplicated(["license_number", "functional_unit"], keep=False), "DUP_LICENSE_UNIT")
    add(df.duplicated(["ccc_code", "functional_unit", "license_number"], keep=False), "DUP_CCC_UNIT")

    df["data_quality_flags"] = df["_flags"].map(lambda f: ";".join(dict.fromkeys(f)))
    hard = (r"MERGED_ROW_SPLIT|CITY_MISSING|CITY_TEXT_ARTIFACT|NAME_TRUNCATED|NAME_MISSING|"
            r"COUNTY_CITY_MISMATCH|DUP_|STATUS_ENTITY_TEXT_BLEED")
    df["needs_pdf_check"] = df["data_quality_flags"].str.contains(hard, regex=True)
    df = df.drop(columns="_flags")

    df.insert(0, "facility_id", [f"F{i:05d}" for i in range(1, len(df) + 1)])

    first = ["facility_id", "state", "county", "city", "ccc_code", "ccc_status", "license_number",
             "license_type", "license_status", "business_entity", "warehouse_name", "functional_unit",
             "warehouse_capacity", "warehouse_capacity_num", "license_category", "is_uswa",
             "license_status_not_issued", "pdf_page", "raw_row_number", "split_part", "n_parts_in_raw_row",
             "data_quality_flags", "needs_pdf_check", "city_in_census_gazetteer", "county_valid_for_state",
             "city_county_consistent", "census_counties_for_city", "city_hint_from_warehouse_name"]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def write_report(df: pd.DataFrame, n_raw: int, out: Path) -> None:
    u = df[df["is_uswa"]]
    flags = df["data_quality_flags"].str.split(";").explode().replace("", pd.NA).dropna().value_counts()
    uflags = u["data_quality_flags"].str.split(";").explode().replace("", pd.NA).dropna().value_counts()
    lines = [
        "# Phase 1 QC report", "",
        f"- Raw rows: {n_raw}",
        f"- WCMD records after splitting merged rows: {len(df)}",
        f"- Raw rows that held 2+ records: {(df['split_part'] == 2).sum()}",
        f"- USWA (federally licensed) records: {len(u)}",
        f"- USWA distinct license numbers: {u['license_number'].nunique()}",
        f"- USWA distinct CCC codes: {u['ccc_code'].nunique()}",
        f"- USWA records with license status other than Issued: {u['license_status_not_issued'].sum()} (kept)",
        f"- USWA records needing a PDF check: {u['needs_pdf_check'].sum()}", "",
        "## License category (all records)", "", df["license_category"].value_counts().to_markdown(), "",
        "## License status (USWA records)", "", u["license_status"].replace("", "(blank)").value_counts().to_markdown(), "",
        "## Data-quality flags (all records)", "", flags.to_markdown(), "",
        "## Data-quality flags (USWA records)", "", uflags.to_markdown(), "",
        "## USWA records by state", "", u["state"].value_counts().to_markdown(), "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    load_config()
    df = build_master()
    out_dir = path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "facilities_master.csv", index=False, encoding="utf-8")
    df[df["is_uswa"]].to_csv(out_dir / "uswa_facilities.csv", index=False, encoding="utf-8")
    df[df["needs_pdf_check"]].to_csv(out_dir / "records_needing_pdf_check.csv", index=False, encoding="utf-8")
    n_raw = int(df["raw_row_number"].max())
    write_report(df, n_raw, out_dir / "phase1_qc_report.md")
    log.info("USWA records: %d of %d", df["is_uswa"].sum(), len(df))
    log.info("Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
