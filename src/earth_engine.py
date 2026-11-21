"""Phase 4 - Google Earth Engine: one representative image per facility, year and sensor.

Sensor choice (and why)
-----------------------
A grain elevator footprint is ~20-80 m, so resolution decides what is visible.

NAIP  (USDA/NAIP/DOQQ)            2003-present, 0.6-1 m, every 2-3 years per state.
      Leaf-on aerial orthophotos. Bins, silos, rail loops are unmistakable.
      -> primary source for human review and for any ML labels.
Landsat 5/7/8/9 Collection 2 L2   1984-present, 30 m, many scenes per year.
      The only consistent record before 2003. Large terminals and big bin
      fields are visible as bright blocks; small country elevators mostly not.
      -> brackets construction for large sites pre-2003, fills NAIP gap years.
Sentinel-2 SR harmonized          2017-present, 10 m, every ~5 days.
      -> latest years and gap years between NAIP flights.

Same geographic extent every year: the region of interest is a fixed square
(roi_half_width_m around the facility point, config) reused for every sensor
and year.

Annual compositing / bad imagery: Landsat and Sentinel-2 scenes are cloud-
masked per pixel (Landsat QA_PIXEL bits; Sentinel-2 Cloud Score+) and reduced
to a median of the growing season (May-Sep, avoids snow). NAIP is a single
flight per year and is mosaicked. Every image carries its own quality numbers
(scenes used, clear-pixel fraction over the ROI); a year with no usable pixels
is reported as unavailable - never filled from another year.
"""
from __future__ import annotations

import ee

from utils import load_config

CFG = load_config()["earth_engine"]

SEASON = (5, 9)  # May..September

LANDSAT = {
    "LT05": "LANDSAT/LT05/C02/T1_L2",
    "LE07": "LANDSAT/LE07/C02/T1_L2",
    "LC08": "LANDSAT/LC08/C02/T1_L2",
    "LC09": "LANDSAT/LC09/C02/T1_L2",
}
S2 = "COPERNICUS/S2_SR_HARMONIZED"
S2_CS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
NAIP = "USDA/NAIP/DOQQ"

# Display stretch per sensor (reflectance for Landsat/S2, DN for NAIP).
VIS = {
    "landsat": {"bands": ["R", "G", "B"], "min": 0.0, "max": 0.3, "gamma": 1.2},
    "sentinel2": {"bands": ["R", "G", "B"], "min": 0.0, "max": 0.3, "gamma": 1.2},
    "naip": {"bands": ["R", "G", "B"], "min": 0, "max": 255},
}


def init() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    ee.Initialize(project=CFG["project"])


def roi(lat: float, lon: float, half_m: float | None = None) -> ee.Geometry:
    """Fixed square around the facility (same extent for every year/sensor)."""
    half = half_m or CFG["roi_half_width_m"]
    return ee.Geometry.Point([lon, lat]).buffer(half).bounds()


def _season(col: ee.ImageCollection, year: int) -> ee.ImageCollection:
    return col.filterDate(f"{year}-01-01", f"{year + 1}-01-01").filter(
        ee.Filter.calendarRange(SEASON[0], SEASON[1], "month"))


# ----------------------------------------------------------------------------
# Landsat
# ----------------------------------------------------------------------------
def _landsat_prep(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    # bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
    bad = (qa.bitwiseAnd(1 << 0).Or(qa.bitwiseAnd(1 << 1)).Or(qa.bitwiseAnd(1 << 2))
           .Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4)).Or(qa.bitwiseAnd(1 << 5))).neq(0)
    sat = img.select("QA_RADSAT").eq(0)
    spacecraft = ee.String(img.get("SPACECRAFT_ID"))
    is_oli = spacecraft.match("LANDSAT_[89]").length().gt(0)
    rgb_oli = img.select(["SR_B4", "SR_B3", "SR_B2"], ["R", "G", "B"])
    rgb_tm = img.select(["SR_B3", "SR_B2", "SR_B1"], ["R", "G", "B"])
    rgb = ee.Image(ee.Algorithms.If(is_oli, rgb_oli, rgb_tm))
    return rgb.multiply(0.0000275).add(-0.2).updateMask(bad.Not()).updateMask(sat)


def landsat_collection(year: int, region: ee.Geometry) -> tuple[ee.ImageCollection, list[str]]:
    sensors = []
    if year <= 2012:
        sensors.append("LT05")
    # Landsat 7 has scan-line gaps after May 2003; use it only where nothing else exists.
    if 1999 <= year <= 2002 or 2012 == year:
        sensors.append("LE07")
    if year >= 2013:
        sensors.append("LC08")
    if year >= 2022:
        sensors.append("LC09")
    col = ee.ImageCollection([])
    for s in sensors:
        col = col.merge(_season(ee.ImageCollection(LANDSAT[s]).filterBounds(region), year))
    return col, sensors


def landsat_image(year: int, region: ee.Geometry):
    col, sensors = landsat_collection(year, region)
    img = col.map(_landsat_prep).median().clip(region)
    return img, col, [LANDSAT[s] for s in sensors], 30


# ----------------------------------------------------------------------------
# Sentinel-2
# ----------------------------------------------------------------------------
def sentinel2_image(year: int, region: ee.Geometry):
    col = _season(ee.ImageCollection(S2).filterBounds(region), year)
    cs = ee.ImageCollection(S2_CS)
    linked = col.linkCollection(cs, ["cs_cdf"])
    img = (linked.map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6))
                      .select(["B4", "B3", "B2"], ["R", "G", "B"]).divide(10000))
           .median().clip(region))
    return img, col, [S2, S2_CS], 10


# ----------------------------------------------------------------------------
# NAIP
# ----------------------------------------------------------------------------
def naip_image(year: int, region: ee.Geometry):
    col = ee.ImageCollection(NAIP).filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    # Some early NAIP years (2003-2005) include colour-infrared-only tiles
    # (bands N,R,G). Use true colour when available; otherwise render CIR as
    # standard false colour (NIR->red), which still shows structures clearly.
    has_b = ee.Filter.listContains("system:band_names", "B")
    rgb, cir = col.filter(has_b), col.filter(has_b.Not())
    img = ee.Image(ee.Algorithms.If(
        rgb.size().gt(0),
        rgb.select(["R", "G", "B"]).mosaic().set("variant", "true colour"),
        cir.select(["N", "R", "G"], ["R", "G", "B"]).mosaic().set("variant", "CIR false colour"),
    ))
    variant = img.get("variant")
    img = img.clip(region).set("variant", variant)
    # Output resolution is configurable: 1 m = full detail (~40 MB/facility),
    # 2 m = a quarter of the storage and bins/silos remain clearly visible.
    return img, col, [NAIP], CFG["export_scale_m"]["naip"]


BUILDERS = {"landsat": landsat_image, "sentinel2": sentinel2_image, "naip": naip_image}


def sensor_years(sensor: str, last_year: int) -> range:
    start = {"landsat": CFG["landsat_start_year"], "naip": CFG["naip_start_year"],
             "sentinel2": CFG["sentinel2_start_year"]}[sensor]
    return range(start, last_year + 1)


def build(sensor: str, year: int, region: ee.Geometry) -> dict:
    """Return the image plus quality info, computed in one server round trip."""
    img, col, sources, scale = BUILDERS[sensor](year, region)
    # Clear-pixel fraction over the ROI, at the sensor's native scale.
    valid = img.select("R").mask().gt(0).unmask(0)
    clear = valid.reduceRegion(ee.Reducer.mean(), region, scale=max(scale, 10), maxPixels=1e8).get("R")
    dates = col.aggregate_array("system:time_start")
    info = ee.Dictionary({
        "n_scenes": col.size(),
        "clear_fraction": ee.Algorithms.If(col.size().gt(0), clear, 0),
        "first": ee.Algorithms.If(col.size().gt(0), ee.Date(dates.reduce(ee.Reducer.min())).format("YYYY-MM-dd"), ""),
        "last": ee.Algorithms.If(col.size().gt(0), ee.Date(dates.reduce(ee.Reducer.max())).format("YYYY-MM-dd"), ""),
        "variant": ee.Algorithms.If(col.size().gt(0), img.get("variant"), ""),
    }).getInfo()
    if info.get("variant"):
        info["sources"] = [f"{s} ({info['variant']})" for s in sources]
        sources = info.pop("sources")
    return {"image": img, "sources": sources, "scale": scale, **info}


def cloud_quality(sensor: str, n_scenes: int, clear: float | None) -> str:
    """GOOD / FAIR / POOR / NONE - a simple, documented rule."""
    clear = clear or 0.0
    if n_scenes == 0 or clear < 0.05:
        return "NONE"
    if sensor == "naip":
        return "GOOD" if clear > 0.95 else "FAIR"  # partial NAIP = ROI on a tile edge gap
    if clear > 0.95 and n_scenes >= 3:
        return "GOOD"
    if clear > 0.6:
        return "FAIR"
    return "POOR"


def thumb_url(img: ee.Image, sensor: str, region: ee.Geometry, px: int) -> str:
    # Web Mercator is locally conformal, so the square ROI renders as a square
    # image (EPSG:4326 output would be squashed by cos(latitude)).
    return img.getThumbURL({**VIS[sensor], "region": region, "dimensions": f"{px}x{px}",
                            "crs": "EPSG:3857", "format": "png"})
