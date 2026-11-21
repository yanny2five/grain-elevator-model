"""Phase 3 - geocoding: street address -> coordinates, with explicit quality.

Order of preference for a facility's coordinates
------------------------------------------------
1. Coordinates already attached to the location evidence (verified structure
   position, EPA registry point snapped to mapped silos, ...). These are
   better than any address geocode because they point at the structures.
2. US Census Bureau geocoder (free, no key, public domain). Address-range
   interpolation along the street: usually within ~50-200 m in towns, much
   worse on rural roads.
3. Nominatim (OpenStreetMap), max 1 request/second per its usage policy.

geocode_confidence
------------------
STRUCTURE  coordinates on the facility structures (seen on imagery / mapped silos)
REGISTRY   official registry point (EPA FRS etc.); usually on site, not guaranteed
ADDRESS    interpolated street-address match (Census/Nominatim)
CITY       only a town centroid  -> NOT sufficient for image clipping
NONE       no coordinates

`clip_ok` is True only for STRUCTURE / REGISTRY / ADDRESS. Rows that are not
clip_ok are not sent to Earth Engine until a human resolves them.

Usage:
  python src/geocode.py                       # geocode everything in facility_addresses.csv
  python src/geocode.py --address "150 E Mill Rd, Longview, WA 98632"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import JsonCache, RateLimiter, get_logger, load_config, path  # noqa: E402

log = get_logger("geocode")
CFG = load_config()
GCFG = CFG["geocode"]

CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


class Geocoder:
    def __init__(self):
        self.cache = JsonCache(path("cache_dir") / "geocode_cache.json")
        self.census_rl = RateLimiter(GCFG["census_min_interval_s"])
        self.nom_rl = RateLimiter(GCFG["nominatim_min_interval_s"])

    def _get(self, url, params, rl):
        rl.wait()
        r = requests.get(url, params=params, headers={"User-Agent": GCFG["user_agent"]}, timeout=60)
        r.raise_for_status()
        return r.json()

    def census(self, address: str) -> dict | None:
        key = f"census|{address}"
        if key not in self.cache:
            js = self._get(CENSUS_URL, {"address": address, "benchmark": "Public_AR_Current", "format": "json"},
                           self.census_rl)
            m = js.get("result", {}).get("addressMatches", [])
            self.cache.set(key, [{"lat": x["coordinates"]["y"], "lon": x["coordinates"]["x"],
                                  "matched": x["matchedAddress"]} for x in m])
        hits = self.cache.get(key)
        if len(hits) == 1:
            return {**hits[0], "source": "US Census geocoder"}
        if len(hits) > 1:
            log.warning("Census returned %d matches for %r - not choosing", len(hits), address)
        return None

    def nominatim(self, address: str) -> dict | None:
        key = f"nominatim|{address}"
        if key not in self.cache:
            js = self._get(NOMINATIM_URL, {"q": address, "format": "jsonv2", "countrycodes": "us", "limit": 2},
                           self.nom_rl)
            self.cache.set(key, [{"lat": float(x["lat"]), "lon": float(x["lon"]), "matched": x["display_name"],
                                  "rank": x.get("place_rank")} for x in js])
        hits = self.cache.get(key)
        # place_rank >= 26 = street/house level; lower ranks are towns/regions.
        good = [h for h in hits if (h.get("rank") or 0) >= 26]
        if good:
            return {**good[0], "source": "Nominatim (OpenStreetMap)"}
        return None

    def geocode(self, address: str) -> dict | None:
        for fn in (self.census, self.nominatim):
            try:
                g = fn(address)
                if g:
                    return g
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed for %r: %s", fn.__name__, address, e)
        return None


def classify_existing(coord_source: str) -> str:
    s = coord_source.lower()
    if "imagery" in s or "silo cluster" in s or "structures" in s and "not confirmed" not in s:
        return "STRUCTURE"
    if "osm feature" in s:
        return "STRUCTURE"
    if "registry" in s or "frs" in s or "ecology" in s:
        return "REGISTRY"
    if "census" in s or "nominatim" in s:
        return "ADDRESS"
    if "city" in s or "centroid" in s:
        return "CITY"
    return "REGISTRY" if s else "NONE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--address")
    args = ap.parse_args()
    g = Geocoder()
    if args.address:
        print(g.geocode(args.address))
        return

    f = path("addresses_dir") / "facility_addresses.csv"
    df = pd.read_csv(f, dtype=str, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        out = {"facility_id": r["facility_id"]}
        if r["latitude"] and r["longitude"]:
            out.update(latitude=r["latitude"], longitude=r["longitude"], geocode_source=r["coord_source"],
                       geocode_confidence=classify_existing(r["coord_source"]))
        elif r["street_address"]:
            addr = f"{r['street_address']}, {r['city_verified'] or r['city']}, {r['state_verified'] or r['state']} {r['zip_code']}".strip()
            hit = g.geocode(addr)
            if hit:
                out.update(latitude=hit["lat"], longitude=hit["lon"], geocode_confidence="ADDRESS",
                           geocode_source=f"{hit['source']}: {hit['matched']}")
            else:
                out.update(latitude="", longitude="", geocode_confidence="NONE",
                           geocode_source=f"no unique geocoder match for '{addr}'")
        else:
            out.update(latitude="", longitude="", geocode_confidence="NONE", geocode_source="no address or coordinates")
        out["clip_ok"] = (out["geocode_confidence"] in ("STRUCTURE", "REGISTRY", "ADDRESS")
                          and r["address_confidence"] in ("HIGH", "MEDIUM"))
        rows.append(out)
        log.info("%s -> %s", r["facility_id"], out["geocode_confidence"])

    geo = pd.DataFrame(rows)
    merged = df.drop(columns=["latitude", "longitude"]).merge(geo, on="facility_id", how="left")
    merged.to_csv(path("outputs_dir") / "facility_locations.csv", index=False)
    log.info("geocode_confidence: %s", geo["geocode_confidence"].value_counts().to_dict())


if __name__ == "__main__":
    main()
