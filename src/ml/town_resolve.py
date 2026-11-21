"""Find the search centre(s) for a WCMD town name that is not a plain Census place.

WCMD city values come from PDF extraction and are sometimes damaged:
  - merged rows:       "Galva GALVA", "Colton DELL RAPIDS", "(NORTH)[2-6905] REDFIELD Tulare"
  - spacing/spelling:  "DESMET" (De Smet), "LUVERNE" (Lu Verne), "Le Land" (Leland), "Calcumet" (Calumet)
  - unincorporated:    "CRAVEN", "RITTER", "PORT NEAL" - real places that are not Census places

resolve() tries, in order:
  1. exact Census place match (after dropping "(...)" and "[...]" parts)
  2. the same, ignoring spaces              ("DESMET" -> DE SMET)
  3. every 1-3 word piece of the name that is a Census place - a merged row
     gives several towns, ALL are returned so a person sees both
  4. Nominatim (OpenStreetMap) settlement search, kept only if it is a
     city/town/village/hamlet/locality of that exact name in the right state (the WCMD
     county is only reported - it is unreliable after PDF extraction)
  5. one unique close spelling among the Census places of the state
Then any town written in parentheses is added ("Howell (Dodge City)"); at most 3 results.
Each result carries how it was found, so the review page can say so.
Never guesses beyond these rules: an empty name, or no match, returns [].
"""
from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clean_dataset import _norm_place  # noqa: E402
from find_addresses import CachedHTTP  # noqa: E402
from utils import load_config  # noqa: E402

STATES = {"AL": "Alabama", "AR": "Arkansas", "CA": "California", "CO": "Colorado", "GA": "Georgia", "IA": "Iowa",
          "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
          "MD": "Maryland", "MI": "Michigan", "MN": "Minnesota", "MO": "Missouri", "MS": "Mississippi",
          "MT": "Montana", "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska", "NM": "New Mexico",
          "NY": "New York", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
          "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
          "VA": "Virginia", "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia", "WY": "Wyoming"}
SETTLEMENT = {"city", "town", "village", "hamlet", "locality", "isolated_dwelling", "suburb", "neighbourhood"}
NOMINATIM = "https://nominatim.openstreetmap.org/search"
_http: CachedHTTP | None = None


def _clean(city: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)|\[.*?\]", " ", city)).strip()


def _squeeze_index(places: dict) -> dict:
    return {(st, p.replace(" ", "")): (st, p) for st, p in places}


# A single word cut out of "NEW HAVEN", "FORT SCOTT", ... is a different town ("HAVEN"), so it is not used.
NAME_PREFIX = {"NEW", "FORT", "PORT", "LAKE", "MOUNT", "MT", "ST", "SAINT", "EL", "LA", "LE", "DE", "GRAND", "GREAT"}


def _hit(places, key, how):
    la, lo, _ = places[key]
    return {"name": key[1].title(), "lat": la, "lon": lo, "how": how}


def resolve(st: str, city: str, county: str, places: dict, squeezed: dict | None = None) -> list[dict]:
    """Main name first; then any town written in parentheses ("Howell (Dodge City)", "PENCE (SCOTT CITY)"),
    which in WCMD usually names the nearby town of a rural site. At most 3 results."""
    squeezed = squeezed or _squeeze_index(places)
    out = _resolve_one(st, _clean(city), county, places, squeezed) if _clean(city) else []
    for p in re.findall(r"\(([^()\[\]\d]+)\)", city):
        n = _norm_place(p)
        key = (st, n) if (st, n) in places else squeezed.get((st, n.replace(" ", "")))
        if key and key[1].title() not in [o["name"] for o in out]:
            out.append(_hit(places, key, "town written in parentheses in the WCMD city"))
    return out[:3]


def _resolve_one(st: str, name: str, county: str, places: dict, squeezed: dict) -> list[dict]:
    global _http

    def hit(key, how):
        return _hit(places, key, how)

    n = _norm_place(name)
    if (st, n) in places:
        return [hit((st, n), "Census place")]
    if (st, n.replace(" ", "")) in squeezed:
        return [hit(squeezed[(st, n.replace(" ", ""))], "Census place (spacing differs)")]

    words = n.split()
    found = []
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            piece = " ".join(words[i:i + size])
            if size == 1 and i > 0 and words[i - 1] in NAME_PREFIX:
                continue
            key = places.get((st, piece)) and (st, piece) or squeezed.get((st, piece.replace(" ", "")))
            if key and key not in [f[0] for f in found] and not any(piece in f[1] for f in found):
                found.append((key, piece))
    if found:
        how = "merged name - one of several towns" if len(found) > 1 else "Census place found inside a damaged name"
        return [hit(k, how) for k, _ in found]

    if _http is None:
        _http = CachedHTTP("nominatim_town", float(load_config()["geocode"]["nominatim_min_interval_s"]))
    js = _http.fetch(NOMINATIM, params={"q": f"{name}, {STATES.get(st, st)}", "countrycodes": "us",
                                        "format": "jsonv2", "addressdetails": 1, "limit": 5})
    out = []
    for r in js or []:
        a = r.get("address", {})
        if r.get("addresstype") not in SETTLEMENT and r.get("type") not in SETTLEMENT:
            continue
        if a.get("state") != STATES.get(st) or _norm_place(r.get("name", "")) != n:
            continue
        # The WCMD county is NOT used to reject: PDF damage shifted it for many rows
        # (e.g. Calumet and Germantown IA listed in Muscatine, really O'Brien). It is only reported.
        cty = a.get("county", "")
        agree = "county matches WCMD" if county and _norm_place(county) in _norm_place(cty) else \
            f"WCMD county {county} differs" if county else "no WCMD county"
        out.append({"name": r.get("name", name), "lat": float(r["lat"]), "lon": float(r["lon"]),
                    "how": f"OpenStreetMap settlement ({r.get('addresstype') or r.get('type')}, {cty}; {agree})"})
    if out:
        if len(out) > 1:  # same name twice in the state: prefer the county match, else show all
            same = [o for o in out if "matches" in o["how"]]
            out = same[:1] or out[:3]
        return out

    # last resort: one clearly closest Census place spelling in the state ("CALCUMET" -> CALUMET)
    close = difflib.get_close_matches(n, [p for s, p in places if s == st], n=2, cutoff=0.85)
    if len(close) == 1:
        return [hit((st, close[0]), f"Census place with close spelling ({close[0].title()})")]
    return []
