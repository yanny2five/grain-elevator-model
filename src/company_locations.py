"""Phase 2b - match company-published location addresses to WCMD records.

Inputs
------
data/addresses/company_pages/*.tsv   one file per company/regional website,
    captured with src/browser/extract_addresses.js (sites block scripted
    downloads, so pages are opened in a browser). Header comments carry the
    company name, page URL and capture date - the provenance of every address.
    Columns: street, city, state, zip, label, heading, services

Matching (conservative)
-----------------------
A WCMD record matches a published location when
  1. company:  the TSV's company equals the record's business_entity, and
  2. state:    same state, and
  3. town:     the WCMD city (or a name inside parentheses, e.g. "VILAS (SPRINGFIELD)")
               equals the location name in the page heading ("New Haven - Fleming,
               Colorado" -> New Haven) or the postal city.
If several addresses match one record, grain addresses (services or label mention
grain/elevator/shuttle) are preferred; if more than one remains and they are not
within `same_site_m` of each other after geocoding -> NEEDS_REVIEW (listed, not chosen).

Coordinates: Census geocode of the street address, snapped to an OSM silo cluster
within `snap_m` when one exists (then the coordinates sit on the structures).

Confidence written to data/addresses/company_evidence.csv:
  HIGH    company + town match from the company's own site AND silo cluster corroborates
  MEDIUM  company + town match, coordinates = street-address geocode only
  NEEDS_REVIEW / UNRESOLVED otherwise (address could not be geocoded, or ambiguous)
Several WCMD functional units in one town can legitimately share one address
(one complex); this is noted, not treated as an error.

Usage:  python src/company_locations.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_dataset import _norm_place  # noqa: E402
from find_addresses import ACFG, EVIDENCE_COLUMNS, CachedHTTP, haversine_m, osm_candidates, snap_to_structures  # noqa: E402
from geocode import Geocoder  # noqa: E402
from utils import get_logger, path  # noqa: E402

log = get_logger("company_locations")
SNAP_M = 500
GRAIN_RX = re.compile(r"grain|elevator|shuttle|terminal", re.I)
STATE_NAMES = r"(Alabama|Arkansas|Colorado|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Michigan|Minnesota|" \
              r"Mississippi|Missouri|Montana|Nebraska|New Mexico|North Dakota|Ohio|Oklahoma|Oregon|South Dakota|" \
              r"Tennessee|Texas|Washington|Wisconsin|Wyoming)"


def load_pages() -> pd.DataFrame:
    frames = []
    for f in sorted((path("addresses_dir") / "company_pages").glob("*.tsv")):
        meta = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                k, _, v = line[1:].partition(":")
                meta[k.strip()] = v.strip()
        df = pd.read_csv(f, sep="\t", comment="#", dtype=str, keep_default_na=False)
        frames.append(df.assign(company=meta.get("company", ""), page_url=meta.get("url", ""),
                                captured=meta.get("captured", "")[:10], page=f.stem))
    # Files differ in optional columns (lat/lon); missing values must be "" not NaN.
    return pd.concat(frames, ignore_index=True).fillna("") if frames else pd.DataFrame()


def location_names(heading: str, city: str) -> set[str]:
    """'New Haven – Fleming, Colorado' -> {NEW HAVEN, FLEMING}; 'Dailey (Haxtun, Colorado)' -> {DAILEY, HAXTUN}."""
    h = re.sub(rf",?\s*{STATE_NAMES}\)?\s*$", "", heading)
    parts = re.split(r"\s+[–-]\s+|\s*\(\s*|\)|,| and ", h)
    names = {_norm_place(p) for p in parts if p.strip()}
    names.add(_norm_place(city))
    # "Hutchinson East", "Paris Mill", "Gladstone Main" -> also "Hutchinson", "Paris", "Gladstone"
    names |= {re.sub(r"\s+(EAST|WEST|NORTH|SOUTH|MILL|MAIN|TERMINAL|ELEVATOR)$", "", n) for n in names}
    return {n for n in names if n and n not in {"GRAIN", "OFFICE", "ADMINISTRATION", "ADDRESS", "IDLE", "CGB GRAIN",
                                                  "NE", "MO", "MAIN"}}


def wcmd_names(city: str) -> set[str]:
    """'VILAS (SPRINGFIELD)' -> {VILAS, SPRINGFIELD}; 'HURON (EAST)' -> {HURON}."""
    parts = re.split(r"[()\[\]/,]", city)
    out = {_norm_place(p) for p in parts if p.strip()}
    return {n for n in out if n and n not in {"EAST", "WEST", "NORTH", "SOUTH"}}


def save(rows: list[dict], review: list[dict]) -> None:
    pd.DataFrame(rows, columns=EVIDENCE_COLUMNS).to_csv(path("addresses_dir") / "company_evidence.csv", index=False)
    pd.DataFrame(review, columns=["facility_id", "status", "note"]).to_csv(
        path("addresses_dir") / "company_match_review.csv", index=False)


def main() -> None:
    pages = load_pages()
    if pages.empty:
        log.warning("no company pages found")
        return
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)
    uswa = master[master["is_uswa"] == "True"]
    pages["names"] = [location_names(h, c) for h, c in zip(pages["heading"], pages["city"])]
    pages["is_grain"] = (pages["services"] + " " + pages["label"]).str.contains(GRAIN_RX)

    geo = Geocoder()
    osm_http = CachedHTTP("overpass", ACFG["overpass_min_interval_s"])
    rows, review = [], []
    for company, pg in pages.groupby("company"):
        # A page may list several WCMD entity names for one company, separated by ';'.
        entities = {c.strip() for c in company.split(";") if c.strip()}
        fac = uswa[uswa["business_entity"].isin(entities)]
        for _, r in fac.iterrows():
            names = wcmd_names(r["city"])
            if not names:
                continue
            hits = pg[(pg["state"] == r["state"]) & pg["names"].map(lambda s: bool(s & names))]
            if hits.empty:
                continue
            # "Howell (Dodge City)": the name before the parenthesis is the site, the one inside is the
            # nearby town. When several branches match, keep those whose own heading names the site.
            primary = _norm_place(re.split(r"[(\[]", r["city"])[0])
            if len(hits) > 1 and primary:
                own = hits[hits["heading"].map(lambda h: primary in location_names(h, ""))]
                if not own.empty:
                    hits = own
            grain_listed = bool(hits["is_grain"].any())
            if grain_listed:
                hits = hits[hits["is_grain"]]
            located = []
            for _, h in hits.iterrows():
                addr = ", ".join(p for p in [h["street"], h["city"], f"{h['state']} {h['zip']}".strip()] if p)
                if h.get("lat") and h.get("lon"):
                    # The company publishes its own map coordinates - better than interpolating the street.
                    g = {"lat": float(h["lat"]), "lon": float(h["lon"]), "source": "company site coordinates",
                         "matched": h["heading"]}
                else:
                    g = geo.geocode(addr) if h["street"] else None
                located.append((h, g))
            ok = [(h, g) for h, g in located if g]
            if not ok:
                h = hits.iloc[0]
                review.append({"facility_id": r["facility_id"], "status": "UNRESOLVED",
                               "note": f"company address not geocodable: {h['street']}, {h['city']}"})
                continue
            h0, g0 = ok[0]
            spread = max(haversine_m(g0["lat"], g0["lon"], g["lat"], g["lon"]) for _, g in ok)
            if spread > ACFG["same_site_m"]:
                review.append({"facility_id": r["facility_id"], "status": "NEEDS_REVIEW",
                               "note": "several company grain addresses in town: "
                                       + " | ".join(f"{h['street']} ({h['heading']})" for h, _ in ok)})
                continue
            # Snap to mapped silos near the geocoded address, if any.
            try:
                osm = osm_candidates(osm_http, g0["lat"], g0["lon"], 1500, [])
            except Exception as e:  # noqa: BLE001 - Overpass outage must not stop the batch
                log.warning("%s: OSM lookup failed (%s); keeping unsnapped coordinates", r["facility_id"], e)
                osm = []
            clusters = [c for c in osm if c["kind"] == "silo_cluster"]
            try:
                c, d, why = snap_to_structures(osm_http, g0["lat"], g0["lon"], clusters, SNAP_M)
            except Exception as e:  # noqa: BLE001
                c, d, why = None, 0.0, f"silo check failed ({str(e)[:60]})"
            if c is not None:
                lat, lon, conf = c["lat"], c["lon"], "HIGH"
                cs = f"OSM silo cluster ({c['n_silos']} silos) {d:.0f} m from geocoded company address"
            else:
                lat, lon, conf = g0["lat"], g0["lon"], "MEDIUM"
                what = "point" if g0["source"] == "company site coordinates" else "street-address geocode"
                cs = f"{g0['source']} {what} ({g0['matched']}); structures not confirmed"
                if why:
                    cs += f"; {why}"
            if not grain_listed and conf != "HIGH":
                conf = "LOW"  # only an agronomy/energy address is published; elevator may be elsewhere in town
            n_units = int(((fac["state"] == r["state"]) & (fac["city"] == r["city"])).sum())
            note = (f"Company site lists '{h0['heading']}' ({h0['label']}; services: {h0['services'] or 'n/a'}). "
                    f"Matched on company + state + town '{r['city']}'.")
            if n_units > 1:
                note += f" {n_units} WCMD units of this company in this town share the match - may be one complex."
            if len(ok) > 1:
                note += f" {len(ok)} nearby company addresses (within {spread:.0f} m) - same site."
            rows.append({"facility_id": r["facility_id"], "street_address": h0["street"], "city_verified": h0["city"],
                         "state_verified": h0["state"], "zip_code": h0["zip"], "latitude": round(lat, 6),
                         "longitude": round(lon, 6), "coord_source": cs, "address_source": h0["page_url"],
                         "source_type": "company website (own published locations)", "address_confidence": conf,
                         "address_notes": note, "verified_by": "company_locations.py (automated)",
                         "verified_date": h0["captured"]})
        log.info("%s: %d WCMD records, %d matched, %d need review", company, len(fac),
                 sum(1 for x in rows if x["facility_id"] in set(fac["facility_id"])),
                 sum(1 for x in review if x["facility_id"] in set(fac["facility_id"])))
        save(rows, review)  # after every company, so an interrupted run keeps its results
    save(rows, review)
    conf = pd.Series([x["address_confidence"] for x in rows]).value_counts().to_dict()
    log.info("company evidence: %d records %s; %d need review", len(rows), conf, len(review))


if __name__ == "__main__":
    main()
