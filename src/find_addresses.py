"""Phase 2 - locate each facility: automated candidates + verified evidence.

Sources, in order of authority
------------------------------
1. Verified evidence  data/addresses/address_evidence.csv
   Rows a person checked against a cited source (company page, state permit,
   government registry). Always wins. The only route to HIGH for sites the
   automated sources cannot match.
2. EPA ECHO / FRS     (federal facility registry, free REST API, no key)
   Searched by WCMD city + state, restricted to grain-related NAICS codes
   (493130 farm-product warehousing, 424510 grain wholesale, 3112 grain/oilseed
   milling, 3111 feed). Matched by distinctive name words. Gives a street
   address and a registry point. Grain elevators with air/water permits are
   well covered; small country elevators often are not.
3. OpenStreetMap      (Overpass API)
   Mapped silos (man_made=silo, building=silo/grain_silo) are clustered into
   sites; named industrial features are matched by name words. Gives the
   position of the *structures*, which is what imagery clipping needs.

Decision rules (automated - conservative on purpose)
----------------------------------------------------
- One ECHO name match (or several within `same_site_m` of each other):
    * an OSM silo cluster within `corroborate_m`  -> HIGH, coordinates snapped
      to the silo cluster (the structures), address from ECHO;
    * no silo corroboration                       -> MEDIUM, ECHO point.
- Several ECHO name matches far apart             -> NEEDS_REVIEW.
- No ECHO match: OSM only, capped at MEDIUM (name match on a silo cluster),
  LOW (single unnamed cluster) or NEEDS_REVIEW (several clusters).
- Nothing at all                                  -> UNRESOLVED.
Never choose at random between plausible candidates.

All API responses are cached (data/cache/*.json), so reruns are free and an
interrupted full run resumes where it stopped. Errors on one facility are
logged and the batch continues.

Usage:
  python src/find_addresses.py --subset prototype
  python src/find_addresses.py --subset uswa
  python src/find_addresses.py --ids F00003 F04448
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_dataset import _norm_place  # noqa: E402
from utils import JsonCache, RateLimiter, get_logger, load_config, path  # noqa: E402

log = get_logger("find_addresses")
CFG = load_config()
ACFG = CFG["addresses"]
UA = CFG["geocode"]["user_agent"]

# Words that say nothing about *which* company/elevator a record is.
STOPWORDS = set("""
ELEVATOR ELEVATORS GRAIN GRAINS INC INCORPORATED LLC CO COMPANY CORP CORPORATION
COOPERATIVE COOP ASSOCIATION THE AND WAREHOUSE WAREHOUSES TERMINAL TERMINALS
FARMERS FARMER FARM SERVICE SERVICES PLANT PROCESSING USA NORTH AMERICA AGRI LLP
EQUITY EXCHANGE STORAGE MILL MILLING RIVER UNITED SOY OPERATIONS PORT HOLDINGS INTERNATIONAL
""".split())

EVIDENCE_COLUMNS = [
    "facility_id", "street_address", "city_verified", "state_verified", "zip_code",
    "latitude", "longitude", "coord_source", "address_source", "source_type",
    "address_confidence", "address_notes", "verified_by", "verified_date",
]
FINAL_COLUMNS = [
    "facility_id", "state", "city", "business_entity", "warehouse_name",
    "street_address", "city_verified", "state_verified", "zip_code", "latitude", "longitude",
    "address_source", "address_confidence", "address_notes", "coord_source", "location_basis",
    "needs_manual_review",
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp, dl = p2 - p1, math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def words(s: str) -> list[str]:
    return re.sub(r"[^A-Z0-9 ]", " ", s.upper()).split()


def distinctive_tokens(names: list[str], exclude: set[str]) -> list[str]:
    toks: list[str] = []
    for n in names:
        for t in words(n):
            if len(t) >= 3 and t not in STOPWORDS and t not in exclude and not t.isdigit() and t not in toks:
                toks.append(t)
    return toks


def token_hits(text: str, tokens: list[str]) -> list[str]:
    w = set(words(text))
    return [t for t in tokens if t in w]


def load_places() -> dict:
    """{(state, normalized place): (lat, lon, land_area_m2)} from the Census Gazetteer."""
    g = pd.read_csv(path("place_centroids"), sep="\t", dtype=str, encoding="latin-1")
    g.columns = [c.strip() for c in g.columns]
    g["place"] = g["NAME"].str.replace(r"(\s+[a-z][a-z\-]*)+(\s*\(.*\))?$", "", regex=True)
    g["place"] = g["place"].str.replace(r"\s+CDP$", "", regex=True).map(_norm_place)
    return {(s, p): (float(la), float(lo), float(a)) for s, p, la, lo, a
            in zip(g["USPS"], g["place"], g["INTPTLAT"], g["INTPTLONG"], g["ALAND"])}


def search_radius(land_m2: float) -> float:
    """Bigger towns need a bigger search area (Houston's port is 10+ km from its centroid)."""
    r = math.sqrt(land_m2) / 2 + 3000
    return max(ACFG["search_radius_m"], min(r, ACFG["max_search_radius_m"]))


class CachedHTTP:
    """GET/POST with a persistent cache, a rate limit and retries."""

    def __init__(self, name: str, min_interval_s: float):
        self.cache = JsonCache(path("cache_dir") / f"{name}_cache.json")
        self.rl = RateLimiter(min_interval_s)

    def fetch(self, url: str, params: dict | None = None, data: dict | None = None):
        key = hashlib.sha1(json.dumps([url, params, data], sort_keys=True).encode()).hexdigest()
        if key in self.cache:
            return self.cache.get(key)
        last = None
        tries = int(os.environ.get("WCMD_HTTP_TRIES", "4"))  # lower it to fail fast during an Overpass outage
        for attempt in range(tries):
            self.rl.wait()
            try:
                if data is not None:
                    r = requests.post(url, data=data, headers={"User-Agent": UA}, timeout=150)
                else:
                    r = requests.get(url, params=params, headers={"User-Agent": UA}, timeout=150)
                if r.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                out = r.json()
                self.cache.set(key, out)
                return out
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt + 1 == tries:
                    break
                wait = 15 * (attempt + 1)
                log.warning("%s attempt %d failed (%s); retry in %ds", url, attempt + 1, str(e)[:150], wait)
                time.sleep(wait)
        raise RuntimeError(f"request failed after retries: {last}")


# ----------------------------------------------------------------------------
# Source 2: EPA ECHO
# ----------------------------------------------------------------------------
ECHO_URL = "https://echodata.epa.gov/echo/echo_rest_services.get_facility_info"
ECHO_NAICS = "4931,4245,3112,3111"
ECHO_COLS = "1,2,3,4,5,6,7,16,17,18,95"  # name, street, city, state, zip, registry, county, naics, lat, lon, active


def echo_candidates(http: CachedHTTP, city: str, state: str, tokens: list[str]) -> list[dict]:
    res = http.fetch(ECHO_URL, params={"output": "JSON", "p_ct": city.upper(), "p_st": state,
                                       "p_ncs": ECHO_NAICS, "qcolumns": ECHO_COLS})
    out = []
    for f in (res.get("Results", {}).get("Facilities") or []):
        hits = token_hits(f.get("FacName", ""), tokens)
        if not hits or not f.get("FacLat") or not f.get("FacLong"):
            continue
        out.append({
            "source": "epa_echo", "name": f["FacName"], "hits": hits,
            "lat": float(f["FacLat"]), "lon": float(f["FacLong"]),
            "street": f.get("FacStreet") or "", "city": f.get("FacCity") or "",
            "state": f.get("FacState") or "", "zip": f.get("FacZip") or "",
            "county": f.get("FacCounty") or "", "naics": f.get("FacNAICSCodes") or "",
            "active": f.get("FacActiveFlag") or "",
            "url": f"https://echo.epa.gov/detailed-facility-report?fid={f['RegistryID']}",
        })
    return sorted(out, key=lambda c: -len(c["hits"]))


# ----------------------------------------------------------------------------
# Source 3: OpenStreetMap
# ----------------------------------------------------------------------------
def osm_candidates(http: CachedHTTP, lat: float, lon: float, R: float, tokens: list[str]) -> list[dict]:
    # A bounding box with plain tag filters is cheap for the public Overpass
    # server; name matching is done here in Python instead of server-side regex.
    dlat = R / 111_320
    dlon = R / (111_320 * math.cos(math.radians(lat)))
    bbox = f"{lat - dlat:.5f},{lon - dlon:.5f},{lat + dlat:.5f},{lon + dlon:.5f}"
    parts = ['nwr["man_made"="silo"];', 'nwr["building"~"^(silo|grain_silo)$"];',
             'nwr["landuse"="industrial"]["name"];', 'nwr["industrial"]["name"];',
             'nwr["man_made"="works"]["name"];', 'nwr["building"~"^(industrial|warehouse)$"]["name"];',
             'nwr["shop"="agrarian"];']
    q = f"[out:json][timeout:90][bbox:{bbox}];({''.join(parts)});out center tags;"
    try:
        els = http.fetch(ACFG["overpass_url"], data={"data": q})["elements"]
    except Exception:  # noqa: BLE001 - main server overloaded/rate-limited: try the mirror once
        els = http.fetch(ACFG["overpass_mirror_url"], data={"data": q})["elements"]

    silos, named = [], []
    for e in els:
        la = e.get("lat", e.get("center", {}).get("lat"))
        lo = e.get("lon", e.get("center", {}).get("lon"))
        if la is None or haversine_m(lat, lon, la, lo) > R:
            continue
        tags = e.get("tags", {})
        rec = {"osm": f"{e['type']}/{e['id']}", "lat": la, "lon": lo, "tags": tags}
        if tags.get("man_made") == "silo" or tags.get("building") in ("silo", "grain_silo"):
            silos.append(rec)
        hits = token_hits(f"{tags.get('name', '')} {tags.get('operator', '')}", tokens)
        if hits:
            named.append(dict(rec, hits=hits))

    clusters: list[dict] = []  # greedy clustering of silo objects into sites
    for s in silos:
        for c in clusters:
            if haversine_m(s["lat"], s["lon"], c["lat"], c["lon"]) < ACFG["silo_cluster_m"]:
                c["members"].append(s)
                c["lat"] = sum(m["lat"] for m in c["members"]) / len(c["members"])
                c["lon"] = sum(m["lon"] for m in c["members"]) / len(c["members"])
                break
        else:
            clusters.append({"lat": s["lat"], "lon": s["lon"], "members": [s]})

    out = []
    for c in clusters:
        nm = [n for n in named if haversine_m(n["lat"], n["lon"], c["lat"], c["lon"]) < ACFG["name_match_to_silo_m"]]
        out.append({"source": "osm", "kind": "silo_cluster", "lat": c["lat"], "lon": c["lon"],
                    "n_silos": len(c["members"]), "hits": sorted({h for n in nm for h in n["hits"]}),
                    "name": "; ".join(sorted({n["tags"].get("name", "") for n in nm} - {""}))[:200],
                    "osm_ids": ";".join(m["osm"] for m in c["members"][:20])})
    for n in named:
        if any(haversine_m(n["lat"], n["lon"], c["lat"], c["lon"]) < ACFG["name_match_to_silo_m"] for c in clusters):
            continue
        out.append({"source": "osm", "kind": "named_feature", "lat": n["lat"], "lon": n["lon"], "n_silos": 0,
                    "hits": n["hits"], "name": n["tags"].get("name", "")[:200], "osm_ids": n["osm"]})
    return out


def _norm_street(s: str) -> str:
    s = re.sub(r"\bHIWAY\b|\bHWY\b|\bHIGHWAY\b", "HWY", s.upper())
    return re.sub(r"[^A-Z0-9]", "", s)


def nearest(c: dict, pool: list[dict]) -> tuple[float, dict | None]:
    best = min(((haversine_m(c["lat"], c["lon"], p["lat"], p["lon"]), p) for p in pool),
               key=lambda t: t[0], default=(float("inf"), None))
    return best


def rail_within(http: CachedHTTP, lat: float, lon: float, radius_m: float = 250) -> bool:
    """Is a railway line within radius_m? Commercial elevators are almost always rail-served."""
    q = (f'[out:json][timeout:60];way(around:{int(radius_m)},{lat:.6f},{lon:.6f})'
         f'["railway"~"^(rail|spur|siding|disused|abandoned)$"];out ids 1;')
    try:
        els = http.fetch(ACFG["overpass_url"], data={"data": q})["elements"]
    except Exception:  # noqa: BLE001
        els = http.fetch(ACFG["overpass_mirror_url"], data={"data": q})["elements"]
    return bool(els)


def max_silo_diameter_m(http: CachedHTTP, osm_ids: str) -> float:
    """Largest footprint side (m) among the mapped silo ways of a cluster (nodes have no size -> 0)."""
    ways = [i.split("/")[1] for i in osm_ids.split(";") if i.startswith("way/")][:10]
    if not ways:
        return 0.0
    q = f"[out:json][timeout:60];way(id:{','.join(ways)});out geom;"
    try:
        els = http.fetch(ACFG["overpass_url"], data={"data": q})["elements"]
    except Exception:  # noqa: BLE001
        els = http.fetch(ACFG["overpass_mirror_url"], data={"data": q})["elements"]
    best = 0.0
    for e in els:
        g = e.get("geometry") or []
        if len(g) < 3:
            continue
        lats, lons = [p["lat"] for p in g], [p["lon"] for p in g]
        h = (max(lats) - min(lats)) * 111_320
        w = (max(lons) - min(lons)) * 111_320 * math.cos(math.radians(sum(lats) / len(lats)))
        best = max(best, min(h, w))  # min side of the bbox ~ diameter of a round bin
    return best


def snap_to_structures(http: CachedHTTP, lat: float, lon: float, clusters: list[dict],
                       max_m: float) -> tuple[dict | None, float, str]:
    """Pick the mapped silo cluster to snap a facility point to - or none.

    A cluster only qualifies if it looks commercial: >= 3 mapped silos, a
    railway within 250 m, or a silo >= 15 m across (farm bins are ~5-12 m; one
    or two large new commercial bins, e.g. Hamlin IA, are 20-30 m). Otherwise
    it is usually a farmstead
    (lesson from the ML spot check: Black Creek WI snapped onto farm bins).
    Among qualifying clusters, prefer more silos and shorter distance.
    Returns (cluster or None, distance_m, note).
    """
    near = [(haversine_m(lat, lon, c["lat"], c["lon"]), c) for c in clusters]
    near = [(d, c) for d, c in near if d <= max_m]
    if not near:
        return None, 0.0, ""
    good = []
    for d, c in near:
        if (c["n_silos"] >= 3 or rail_within(http, c["lat"], c["lon"])
                or max_silo_diameter_m(http, c.get("osm_ids", "")) >= 15):
            good.append((d, c))
    if not good:
        d, c = min(near, key=lambda t: t[0])
        return None, d, (f"nearest mapped silos ({c['n_silos']}, {d:.0f} m) look farm-scale "
                         f"(<3 silos, no rail within 250 m, no silo >= 15 m) - not snapped")
    d, c = max(good, key=lambda t: min(t[1]["n_silos"], 20) - t[0] / 100)
    return c, d, ""


def group_same_site(cands: list[dict], same_site_m: float, anchors: list[dict] | None = None) -> list[list[dict]]:
    """Group candidates that describe one site: close together, same street
    address, or snapping to the same OSM structure (`anchors`)."""
    def same(a, b):
        if haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) < same_site_m:
            return True
        if a.get("street") and _norm_street(a["street"]) == _norm_street(b.get("street", "")):
            return True
        if anchors:
            da, pa = nearest(a, anchors)
            db, pb = nearest(b, anchors)
            lim = ACFG["corroborate_m"]
            return pa is not None and pa is pb and da <= lim and db <= lim
        return False

    groups: list[list[dict]] = []
    for c in cands:
        for g in groups:
            if any(same(c, x) for x in g):
                g.append(c)
                break
        else:
            groups.append([c])
    return groups


# ----------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------
def decide(echo: list[dict], osm: list[dict], n_tokens: int) -> dict:
    clusters = [c for c in osm if c["kind"] == "silo_cluster"]
    named_osm = [c for c in osm if c["hits"]]
    same = ACFG["same_site_m"]

    if echo:
        best = max(len(c["hits"]) for c in echo)
        top_groups = group_same_site([c for c in echo if len(c["hits"]) == best], same, anchors=clusters)
        if len(top_groups) > 1:
            return {"conf": "NEEDS_REVIEW",
                    "note": f"{len(top_groups)} EPA registry sites match the name equally well ("
                            + " | ".join(f"{g[0]['name']}, {g[0]['street']}" for g in top_groups[:4]) + ")"}
        e = top_groups[0][0]
        res = {"street": e["street"].title(), "city_v": e["city"].title(), "state_v": e["state"],
               "zip": e["zip"], "source": e["url"]}
        # A strong name match shares >= 2 distinctive words, or all of them.
        strong = len(e["hits"]) >= 2 or len(e["hits"]) == n_tokens
        # Corroborate the registry point with the structures: mapped silos
        # (preferred, they *are* the elevator) or an OSM feature with the name.
        # Only commercial-looking clusters (>= 3 silos) count; 1-2 silos is often a farmstead.
        d_s, c_s = nearest(e, [c for c in clusters if c["n_silos"] >= 3])
        d_n, c_n = nearest(e, named_osm)
        lim = ACFG["corroborate_m"]
        if c_s is not None and d_s <= lim:
            lat, lon, cs = c_s["lat"], c_s["lon"], f"OSM silo cluster ({c_s['n_silos']} silos) {d_s:.0f} m from EPA registry point"
        elif c_n is not None and d_n <= lim:
            lat, lon, cs = c_n["lat"], c_n["lon"], f"OSM feature '{c_n['name']}' {d_n:.0f} m from EPA registry point"
        else:
            return {**res, "conf": "MEDIUM", "lat": e["lat"], "lon": e["lon"],
                    "coord_source": "EPA FRS registry point (structures not confirmed)",
                    "note": f"EPA registry match '{e['name']}' (words: {','.join(e['hits'])}); "
                            f"nothing mapped within {lim} m - confirm on imagery"}
        conf = "HIGH" if strong else "MEDIUM"
        why = "" if strong else " - weak name match, verify company/ownership"
        return {**res, "conf": conf, "lat": lat, "lon": lon, "coord_source": cs,
                "note": f"EPA registry match '{e['name']}' (words: {','.join(e['hits'])}); "
                        f"corroborated in OSM{why}"}

    named_silo = [c for c in clusters if c["hits"]]
    g = group_same_site(named_silo, ACFG["same_site_m"])
    if len(g) == 1:
        c = max(g[0], key=lambda x: x["n_silos"])
        return {"conf": "MEDIUM", "lat": c["lat"], "lon": c["lon"], "coord_source": f"OSM silo cluster {c['osm_ids'][:60]}",
                "note": f"OSM silos next to a feature named '{c['name']}'; no registry address", "source": "OpenStreetMap"}
    if len(g) > 1:
        return {"conf": "NEEDS_REVIEW", "note": f"{len(g)} separate OSM silo sites carry a matching name"}
    if len(clusters) == 1:
        c = clusters[0]
        return {"conf": "LOW", "lat": c["lat"], "lon": c["lon"], "coord_source": f"OSM silo cluster {c['osm_ids'][:60]}",
                "note": "only one mapped silo cluster in search area, no name match - may be a different elevator",
                "source": "OpenStreetMap"}
    named = [c for c in osm if c["hits"]]
    if len(named) == 1 and not clusters:
        c = named[0]
        return {"conf": "LOW", "lat": c["lat"], "lon": c["lon"], "coord_source": f"OSM feature {c['osm_ids']}",
                "note": f"OSM feature named '{c['name']}' but no mapped silos", "source": "OpenStreetMap"}
    if not osm:
        return {"conf": "UNRESOLVED", "note": "no EPA registry match and nothing relevant mapped in OSM"}
    return {"conf": "NEEDS_REVIEW",
            "note": f"{len(clusters)} unnamed OSM silo clusters / {len(named)} named features, none decisive"}


# ----------------------------------------------------------------------------
def load_evidence() -> pd.DataFrame:
    f = path("addresses_dir") / "address_evidence.csv"
    if not f.exists():
        pd.DataFrame(columns=EVIDENCE_COLUMNS).to_csv(f, index=False)
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def select(df: pd.DataFrame, subset: str, ids: list[str] | None) -> pd.DataFrame:
    if ids:
        return df[df["facility_id"].isin(ids)]
    if subset == "prototype":
        proto = pd.read_csv(path("processed_dir") / "prototype_facilities.csv", dtype=str)
        return df[df["facility_id"].isin(proto["facility_id"])]
    if subset == "uswa":
        return df[df["is_uswa"].eq("True")]
    return df


def upsert(new: pd.DataFrame, file: Path, done: set) -> pd.DataFrame:
    """Rows for facilities processed in this run replace older rows; others are kept."""
    new = new.astype(str).replace({"nan": "", "None": ""})
    if file.exists() and file.stat().st_size > 1:
        old = pd.read_csv(file, dtype=str, keep_default_na=False)
        new = pd.concat([old[~old["facility_id"].isin(done)], new], ignore_index=True)
    new = new.sort_values("facility_id", kind="stable")
    new.to_csv(file, index=False)
    return new


def locate(r: pd.Series, places: dict, echo_http: CachedHTTP, osm_http: CachedHTTP) -> tuple[dict, list[dict]]:
    # The name-derived hint (e.g. "JONESBORO RICE WAREHOUSE") is used only when the
    # city cell is blank; "Aurora Cooperative" at Sedan NE must not search Aurora.
    if r["city_in_census_gazetteer"] == "True":
        city = r["city"]
    elif r["city"] == "":
        city = r["city_hint_from_warehouse_name"]
    else:
        city = ""
    place = places.get((r["state"], _norm_place(city))) if city else None
    if not place:
        return {"auto_confidence": "UNRESOLVED", "auto_note": "WCMD city not found in Census gazetteer"}, []
    lat, lon, area = place
    R = search_radius(area)
    toks = distinctive_tokens([r["business_entity"], r["warehouse_name"], r["entity_fragment_from_status"]],
                              exclude=set(_norm_place(city).split()))
    echo = echo_candidates(echo_http, city, r["state"], toks) if toks else []
    osm = osm_candidates(osm_http, lat, lon, R, toks)
    d = decide(echo, osm, len(toks))
    cands = [{"facility_id": r["facility_id"], **c, "hits": ";".join(c["hits"]),
              "dist_from_city_m": round(haversine_m(lat, lon, c["lat"], c["lon"]))} for c in echo + osm]
    summary = {
        "anchor_city": city, "anchor_lat": lat, "anchor_lon": lon, "search_radius_m": round(R),
        "search_tokens": " ".join(toks), "n_echo": len(echo), "n_osm": len(osm),
        "auto_confidence": d["conf"], "auto_note": d.get("note", ""),
        "auto_lat": d.get("lat", ""), "auto_lon": d.get("lon", ""), "auto_coord_source": d.get("coord_source", ""),
        "auto_street": d.get("street", ""), "auto_city": d.get("city_v", ""), "auto_state": d.get("state_v", ""),
        "auto_zip": d.get("zip", ""), "auto_source": d.get("source", ""),
    }
    return summary, cands


def merge_final(master: pd.DataFrame, auto: pd.DataFrame) -> pd.DataFrame:
    """Layer the sources: verified evidence > company-site evidence > automated."""
    ev = load_evidence().assign(_basis="verified_evidence")
    cf = path("addresses_dir") / "company_evidence.csv"
    if cf.exists():
        comp = pd.read_csv(cf, dtype=str, keep_default_na=False).reindex(columns=EVIDENCE_COLUMNS).fillna("")
        ev = pd.concat([ev, comp.assign(_basis="company_site")], ignore_index=True)
    ev = ev.drop_duplicates("facility_id", keep="first")
    ids = set(auto["facility_id"]) | set(ev["facility_id"])
    fac = master[master["facility_id"].isin(ids)]
    m = fac[["facility_id", "state", "city", "business_entity", "warehouse_name"]].merge(auto, on="facility_id", how="left")
    m = m.merge(ev, on="facility_id", how="left").fillna("")
    for c in ["auto_street", "auto_city", "auto_state", "auto_zip", "auto_lat", "auto_lon", "auto_coord_source",
              "auto_source", "auto_confidence", "auto_note"]:
        if c not in m.columns:
            m[c] = ""
    has_ev = m["address_confidence"].ne("")

    def pick(ev_col, auto_col):
        # Verified evidence is taken as-is, blanks included: a reviewer leaving
        # coordinates empty (e.g. rejected candidate) must not be overridden.
        return m[ev_col].where(has_ev, m[auto_col])

    out = m[["facility_id", "state", "city", "business_entity", "warehouse_name"]].copy()
    out["street_address"] = pick("street_address", "auto_street")
    out["city_verified"] = pick("city_verified", "auto_city")
    out["state_verified"] = pick("state_verified", "auto_state")
    out["zip_code"] = pick("zip_code", "auto_zip")
    out["latitude"] = pick("latitude", "auto_lat")
    out["longitude"] = pick("longitude", "auto_lon")
    out["coord_source"] = pick("coord_source", "auto_coord_source")
    out["address_source"] = pick("address_source", "auto_source")
    out["address_confidence"] = m["address_confidence"].where(has_ev, m["auto_confidence"])
    out["address_notes"] = m["address_notes"].where(has_ev, m["auto_note"])
    out["location_basis"] = m["_basis"].where(has_ev, "automated")
    out["needs_manual_review"] = ~out["address_confidence"].isin(["HIGH", "MEDIUM"])
    out["auto_confidence"] = m["auto_confidence"]
    out["auto_note"] = m["auto_note"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="prototype", choices=["prototype", "uswa", "all"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--merge-only", action="store_true", help="only re-merge evidence; no API calls")
    args = ap.parse_args()

    adir = path("addresses_dir")
    adir.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(path("processed_dir") / "facilities_master.csv", dtype=str, keep_default_na=False)

    if not args.merge_only:
        fac = select(master, args.subset, args.ids)
        places = load_places()
        echo_http = CachedHTTP("echo", ACFG["echo_min_interval_s"])
        osm_http = CachedHTTP("overpass", ACFG["overpass_min_interval_s"])
        summaries, cand_rows = [], []
        for i, (_, r) in enumerate(fac.iterrows(), 1):
            fid = r["facility_id"]
            try:
                s, c = locate(r, places, echo_http, osm_http)
                cand_rows += c
                log.info("[%d/%d] %s %s, %s -> %s", i, len(fac), fid, r["city"], r["state"], s["auto_confidence"])
            except Exception as e:  # noqa: BLE001 - one bad facility must not stop the batch
                log.error("%s failed: %s", fid, e)
                s = {"auto_confidence": "ERROR", "auto_note": str(e)[:200]}
            summaries.append({"facility_id": fid, **s})
        done = set(fac["facility_id"])
        cand_cols = ["facility_id", "source", "kind", "name", "hits", "lat", "lon", "n_silos", "dist_from_city_m",
                     "street", "city", "state", "zip", "county", "naics", "active", "url", "osm_ids"]
        upsert(pd.DataFrame(cand_rows).reindex(columns=cand_cols), adir / "location_candidates.csv", done)
        upsert(pd.DataFrame(summaries), adir / "auto_locations.csv", done)

    auto = pd.read_csv(adir / "auto_locations.csv", dtype=str, keep_default_na=False)
    final = merge_final(master, auto)
    final.to_csv(adir / "facility_addresses.csv", index=False)
    out = path("outputs_dir")
    out.mkdir(exist_ok=True)
    final[final["needs_manual_review"]].to_csv(out / "unresolved_locations.csv", index=False)
    log.info("Address confidence: %s", final["address_confidence"].value_counts().to_dict())


if __name__ == "__main__":
    main()
