"""Build a Sentinel-2 dNBR burn severity map of the Caldor demo AOI, in MTBS dnbr6 codes.

MTBS cannot be reached from the build environment (mtbs.gov and edcintl.cr.usgs.gov are blocked at
the proxy), so this script makes an independent severity reference from the Sentinel-2 L2A COG
archive on AWS (the Element 84 ``sentinel-cogs`` bucket, read anonymously, no AWS credentials):

1. **Grid.** The Caldor demo AOI (``caldor.geojson`` feature 0) in EPSG:32610 at 20 m, on the
   Sentinel-2 20 m grid (edges on multiples of 20 m), so every tile window is read without
   resampling. Pixels outside the AOI polygon (pixel-centre rule) are nodata (255): the map does
   not cover them, so the wildfire app leaves them out instead of scoring them as background (0,
   which this map never uses).
2. **Tiles.** The MGRS 100 km squares of UTM zone 10 under the AOI and their neighbours, kept when
   the bucket lists them, reduced to the fewest whose footprints (``tileinfo_metadata.json``)
   cover the AOI. Zone 10 tiles reach past 120 W, so the zone 11 part of the AOI needs no
   reprojection.
3. **Scenes.** Pre-fire 2021-06-15 to 2021-08-10 (ignition was 2021-08-14); post-fire one year on,
   the MTBS extended assessment, 2022-06-15 to 2022-08-31 (before the Mosquito Fire of September
   2022). The bucket holds several products per tile and date (partial datastrips and the
   Collection-1 reprocessing): the most complete is used, and among products within one point of
   its nodata share, the newest processing baseline. A product is used when its STAC item's
   ``eo:cloud_cover`` is under ``--max-cloud``; per period the ``--max-dates`` dates with the most
   AOI coverage, then the least cloud, are kept, which caps the download.
4. **Reflectance.** ``B8A`` and ``B12`` DN times 1e-4. Processing baseline 04.00 and later adds a
   BOA offset of 1000 DN, which the STAC item is meant to describe (``s2:processing_baseline``,
   ``earthsearch:boa_offset_applied``, ``raster:bands`` offset). In this bucket that metadata is
   wrong both ways: N05.00 products flagged as applied with an offset of -0.1, and N04.00 ones
   flagged as not applied, both read about 0 DN over clear water (Lake Tahoe), the same as the
   N03.01 product of the same acquisition, so their values are already offset-free. The item
   therefore only says whether an offset can exist (baseline 04.00 or later), and the values
   decide whether it is still in them: a clear-pixel 1st percentile of ``B12`` at or above
   1000 DN (clear land is above 0.03 reflectance, so 1300 DN or more with the offset, a few hundred
   without). A product of an older baseline whose values look raw stops the script, and a product
   with fewer than 1000 clear pixels in the window is skipped, since its offset cannot be checked.
   Each product's metadata and the decision are recorded.
5. **NBR.** Clear pixels are SCL classes 4 (vegetation) and 5 (not vegetated);
   ``NBR = (B8A - B12) / (B8A + B12)``. One observation per date (the tiles of a date are
   mosaicked, first tile wins in their overlap, so no date counts twice), then per period the
   per-pixel median of clear NBR over the dates.
6. **dNBR** = NBR_pre - NBR_post, minus its median over unburned land (as MTBS offsets dNBR): AOI
   pixels outside the Hansen-derived Caldor outline (``caldor.geojson`` feature 1) buffered by
   2 km.
7. **Classes** on ``round(1000 * dNBR)`` (Key and Benson): below -100 increased greenness (5), -100
   to 99 unburned to low (1), 100 to 269 low (2), 270 to 659 moderate (3), 660 and above high (4);
   no clear observation in one period or both (dNBR undefined), non-processing (6).

Writes a uint8 COG with the tag ``TESSERA_APPS_SOURCE=sentinel2_dnbr`` (the wildfire app's
``--severity`` names it as the Sentinel-2 proxy, not MTBS) and tags recording the scenes, windows,
thresholds and offset; and a JSON next to it with per-class areas and the sanity checks: the share
of the Hansen outline in each class, and burned-looking clusters away from the Caldor scar. The COG
holds nothing that depends on when it was made (the creation time is in the JSON only), so a rerun
on the same products writes the same bytes, and the file's sha256, which the wildfire app's
arguments digest keys ``--severity`` on, identifies its content.

Usage:
    python scripts/make_caldor_dnbr.py [--out PATH] [--max-dates 8] [--max-cloud 20]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import ssl
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil
import shapely
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from tessera_apps import demo, export, validate
from tessera_apps.cache import cache_dir

logger = logging.getLogger("make_caldor_dnbr")

BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
PREFIX = "sentinel-s2-l2a-cogs"
#: UTM zone and CRS of the output grid.
ZONE = 10
EPSG = 32600 + ZONE
RES = 20.0
PIXEL_HA = RES * RES / 1e4
PRE = (dt.date(2021, 6, 15), dt.date(2021, 8, 10))
POST = (dt.date(2022, 6, 15), dt.date(2022, 8, 31))
#: SCL classes kept as clear: 4 vegetation, 5 not vegetated.
SCL_CLEAR = (4, 5)
DN_SCALE = 1e-4
#: BOA offset of processing baseline 04.00 and later, in reflectance.
BOA_OFFSET = -0.1
OFFSET_BASELINE = (4, 0)
#: Clear-pixel 1st percentile of B12 DN that separates offset-free values from raw ones.
RAW_OFFSET_P1_DN = 1000
#: Fewest clear pixels a product needs before its offset check counts.
OFFSET_CHECK_MIN_PIXELS = 1000
#: Products within this many points of the most complete product's nodata share count as complete.
NODATA_TIE_POINTS = 1.0
UNBURNED_BUFFER_M = 2000.0
#: Key and Benson classes on round(1000 * dNBR).
THRESHOLDS = {"increased_greenness_below": -100, "low_from": 100, "moderate_from": 270, "high_from": 660}
#: Value of the pixels outside the AOI, declared as the COG's nodata.
NODATA = 255
CLASS_NAMES = {
    0: "background (not used: pixels outside the AOI are nodata, 255)",
    1: "unburned to low",
    2: "low",
    3: "moderate",
    4: "high",
    5: "increased greenness",
    6: "non-processing (no clear observation in a period)",
}
#: MTBS-like colours for viewing.
COLOURS = {
    0: (0, 0, 0, 0),
    1: (0, 100, 0, 255),
    2: (127, 255, 212, 255),
    3: (255, 255, 0, 255),
    4: (255, 0, 0, 255),
    5: (127, 255, 0, 255),
    6: (255, 255, 255, 255),
    NODATA: (0, 0, 0, 0),
}
SOURCE_TAG = "sentinel2_dnbr"
#: Burned-looking clusters (moderate or high) away from the scar at least this large are reported.
CLUSTER_MIN_HA = 5.0
CLUSTERS_LISTED = 15
HTTP_TIMEOUT_S = 60.0
HTTP_TRIES = 3
_SCENE = re.compile(r"^(S2[ABCD])_(\d{2}[A-Z]{3})_(\d{8})_(\d+)_L2A$")
_S3 = "{http://s3.amazonaws.com/doc/2006-03-01/}"
_COLUMN_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def default_out() -> Path:
    """``<tessera_apps cache>/reference/s2_dnbr/caldor_s2_dnbr6.tif``."""
    return cache_dir() / "reference" / "s2_dnbr" / "caldor_s2_dnbr6.tif"


# --------------------------------------------------------------------------------------------
# HTTP and the bucket listing
# --------------------------------------------------------------------------------------------


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        bundle = os.environ.get(var)
        if bundle and Path(bundle).is_file():
            ctx.load_verify_locations(cafile=bundle)
    return ctx


_CTX = _ssl_context()


def http_get(url: str) -> bytes:
    """GET ``url`` through the environment's proxy, with a few retries.

    Raises:
        RuntimeError: Every try failed.
    """
    last: Exception | None = None
    for attempt in range(HTTP_TRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "tessera_apps make_caldor_dnbr"})
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S, context=_CTX) as response:
                return bytes(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Could not GET {url}: {last}")


def list_prefixes(prefix: str) -> list[str]:
    """Common prefixes (sub-folders) directly under ``prefix`` in the bucket (ListObjectsV2)."""
    out: list[str] = []
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            query["continuation-token"] = token
        root = ET.fromstring(http_get(f"{BUCKET}/?{urllib.parse.urlencode(query)}"))
        out += [el.text or "" for el in root.iter(f"{_S3}Prefix") if el.text and el.text != prefix]
        truncated = (root.findtext(f"{_S3}IsTruncated") or "false") == "true"
        token = root.findtext(f"{_S3}NextContinuationToken")
        if not truncated or not token:
            return out


# --------------------------------------------------------------------------------------------
# Grid and tiles
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Grid:
    """The output grid: EPSG:32610, 20 m, edges on multiples of 20 m."""

    transform: Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols)."""
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top)."""
        left, top = self.transform.c, self.transform.f
        return left, top - self.height * RES, left + self.width * RES, top


def to_utm(geom: BaseGeometry) -> BaseGeometry:
    """A lon/lat geometry in EPSG:32610, densified first so long edges follow the projection."""
    to = Transformer.from_crs(4326, EPSG, always_xy=True)
    return shapely.transform(shapely.segmentize(geom, 0.001), to.transform, interleaved=False)


def make_grid(aoi_utm: BaseGeometry) -> Grid:
    """The 20 m grid covering ``aoi_utm``, snapped outward to multiples of 20 m."""
    minx, miny, maxx, maxy = aoi_utm.bounds
    x0, x1 = math.floor(minx / RES) * RES, math.ceil(maxx / RES) * RES
    y0, y1 = math.floor(miny / RES) * RES, math.ceil(maxy / RES) * RES
    return Grid(from_origin(x0, y1, RES, RES), round((x1 - x0) / RES), round((y1 - y0) / RES))


def burn(geom: BaseGeometry, grid: Grid) -> np.ndarray:
    """Pixel-centre rasterization of a 32610 geometry on the grid."""
    if geom.is_empty:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize([geom], out_shape=grid.shape, transform=grid.transform, fill=0, default_value=1).astype(bool)


def mgrs_square(easting: float, northing: float, zone: int) -> str:
    """The MGRS 100 km square letters of a UTM coordinate (standard lettering, AA scheme)."""
    column = _COLUMN_SETS[(zone - 1) % 3][int(easting // 100_000) - 1]
    row = _ROW_LETTERS[(int(northing // 100_000) + (5 if zone % 2 == 0 else 0)) % 20]
    return column + row


def lat_band(lat: float) -> str:
    """The MGRS latitude band letter."""
    return _LAT_BANDS[min(int((lat + 80.0) // 8), len(_LAT_BANDS) - 1)]


def candidate_tiles(aoi_ll: BaseGeometry, aoi_utm: BaseGeometry) -> list[tuple[str, str]]:
    """(latitude band, square) of every 100 km square within 110 km of the AOI's corners."""
    minx, miny, maxx, maxy = aoi_utm.bounds
    bands = {lat_band(aoi_ll.bounds[1]), lat_band(aoi_ll.bounds[3])}
    out = set()
    for e in np.arange(minx - 110_000, maxx + 110_001, 50_000):
        for n in np.arange(miny - 110_000, maxy + 110_001, 50_000):
            if 100_000 <= e < 900_000:
                out |= {(b, mgrs_square(float(e), float(n), ZONE)) for b in bands}
    return sorted(out)


def tile_footprint(band: str, square: str) -> BaseGeometry | None:
    """A tile's footprint in 32610, from ``tileinfo_metadata.json`` of one of its 2021 scenes."""
    for month in (7, 6, 8):
        scenes = list_prefixes(f"{PREFIX}/{ZONE}/{band}/{square}/2021/{month}/")
        for folder in scenes[:1]:
            info = json.loads(http_get(f"{BUCKET}/{folder}tileinfo_metadata.json"))
            geom = info["tileGeometry"]
            if not str(geom["crs"]["properties"]["name"]).endswith(str(EPSG)):
                return None
            return shape({"type": geom["type"], "coordinates": geom["coordinates"]})
    return None


def choose_tiles(aoi_utm: BaseGeometry) -> list[tuple[str, BaseGeometry]]:
    """The fewest zone 10 tiles whose footprints cover the AOI (greedy by new area).

    Raises:
        RuntimeError: The listed tiles do not cover the AOI.
    """
    aoi_ll = shapely.transform(aoi_utm, Transformer.from_crs(EPSG, 4326, always_xy=True).transform, interleaved=False)
    listed: dict[str, set[str]] = {}
    footprints: dict[str, BaseGeometry] = {}
    for band, square in candidate_tiles(aoi_ll, aoi_utm):
        if band not in listed:
            listed[band] = {p.rstrip("/").rsplit("/", 1)[-1] for p in list_prefixes(f"{PREFIX}/{ZONE}/{band}/")}
        if square not in listed[band]:
            continue
        name = f"{ZONE}{band}{square}"
        fp = tile_footprint(band, square)
        if fp is not None and fp.intersects(aoi_utm):
            footprints[name] = fp
    chosen: list[tuple[str, BaseGeometry]] = []
    left = aoi_utm
    while left.area > 1.0 and footprints:
        name, fp = max(footprints.items(), key=lambda kv: kv[1].intersection(left).area)
        if fp.intersection(left).area <= 1.0:
            break
        chosen.append((name, fp))
        left = left.difference(fp)
        del footprints[name]
    if left.area > RES * RES:
        raise RuntimeError(f"The listed zone {ZONE} tiles leave {left.area / 1e4:.0f} ha of the AOI uncovered.")
    return chosen


# --------------------------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Product:
    """One L2A product of one tile and date, with what its STAC item says about it."""

    tile: str
    date: dt.date
    folder: str
    product_uri: str
    cloud: float
    nodata_pct: float
    baseline: tuple[int, int]
    offset_applied: bool
    aoi_cover: float
    footprint: BaseGeometry

    @property
    def name(self) -> str:
        """The scene folder name, for example ``S2B_10SGJ_20210720_3_L2A``."""
        return self.folder.rstrip("/").rsplit("/", 1)[-1]

    @property
    def offset_possible(self) -> bool:
        """The processing baseline adds a BOA offset (04.00 and later)."""
        return self.baseline >= OFFSET_BASELINE

    @property
    def item_offset(self) -> float:
        """The reflectance offset the item's flag implies (unreliable here, recorded for comparison)."""
        return BOA_OFFSET if self.offset_possible and not self.offset_applied else 0.0

    def url(self, band: str) -> str:
        """HTTPS URL of a band COG."""
        return f"{BUCKET}/{self.folder}{band}.tif"


def _baseline(text: str) -> tuple[int, int]:
    major, _, minor = str(text).partition(".")
    return int(major), int(minor or 0)


def list_products(tile: str, period: tuple[dt.date, dt.date], aoi_utm: BaseGeometry) -> list[Product]:
    """Every L2A product of ``tile`` dated inside ``period``, with its STAC item read."""
    zone, band, square = tile[:2], tile[2], tile[3:]
    start, end = period
    months = sorted({(d.year, d.month) for d in (start + dt.timedelta(days=i) for i in range((end - start).days + 1))})
    found: list[Product] = []
    to_utm_tf = Transformer.from_crs(4326, EPSG, always_xy=True)
    for year, month in months:
        for folder in list_prefixes(f"{PREFIX}/{zone}/{band}/{square}/{year}/{month}/"):
            name = folder.rstrip("/").rsplit("/", 1)[-1]
            match = _SCENE.match(name)
            if not match:
                continue
            day = dt.datetime.strptime(match.group(3), "%Y%m%d").date()
            if not start <= day <= end:
                continue
            item = json.loads(http_get(f"{BUCKET}/{folder}{name}.json"))
            props = item["properties"]
            fp = shapely.transform(shape(item["geometry"]), to_utm_tf.transform, interleaved=False)
            found.append(
                Product(
                    tile=tile,
                    date=day,
                    folder=folder,
                    product_uri=str(props.get("s2:product_uri", name)),
                    cloud=float(props["eo:cloud_cover"]),
                    nodata_pct=float(props.get("s2:nodata_pixel_percentage", 0.0)),
                    baseline=_baseline(props["s2:processing_baseline"]),
                    offset_applied=bool(props.get("earthsearch:boa_offset_applied", False)),
                    aoi_cover=float(fp.intersection(aoi_utm).area / aoi_utm.area),
                    footprint=fp,
                )
            )
    return found


def pick_products(products: Sequence[Product]) -> list[Product]:
    """One product per tile and date: the most complete, then the newest processing baseline."""
    groups: dict[tuple[str, dt.date], list[Product]] = {}
    for p in products:
        groups.setdefault((p.tile, p.date), []).append(p)
    out = []
    for group in groups.values():
        best = min(p.nodata_pct for p in group)
        complete = [p for p in group if p.nodata_pct <= best + NODATA_TIE_POINTS]
        out.append(max(complete, key=lambda p: (p.baseline, -p.nodata_pct, p.name)))
    return sorted(out, key=lambda p: (p.date, p.tile))


def select_dates(
    products: Sequence[Product], aoi_utm: BaseGeometry, *, max_cloud: float, max_dates: int
) -> dict[dt.date, list[Product]]:
    """The ``max_dates`` dates with the most AOI coverage by products under ``max_cloud``, then the least cloud."""
    by_date: dict[dt.date, list[Product]] = {}
    for p in products:
        if p.cloud < max_cloud:
            by_date.setdefault(p.date, []).append(p)
    scored = []
    for day, group in by_date.items():
        union = shapely.union_all([p.footprint for p in group]).intersection(aoi_utm)
        cover = float(union.area / aoi_utm.area)
        weights = [p.aoi_cover for p in group]
        cloud = float(np.average([p.cloud for p in group], weights=weights)) if sum(weights) > 0 else 100.0
        scored.append((round(cover, 2), -cloud, day))
    scored.sort(reverse=True)
    keep = sorted(day for cover, _, day in scored[:max_dates] if cover > 0)
    return {day: sorted(by_date[day], key=lambda p: -p.aoi_cover) for day in keep}


# --------------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class ProductRead:
    """One product's clear NBR on its part of the grid."""

    product: Product
    rows: slice
    cols: slice
    window: tuple[int, int, int, int]
    nbr: np.ndarray | None
    clear: int
    b12_p1_dn: float | None
    offset: float
    bytes: int
    note: str | None = None


def _window_bytes(ds: rasterio.io.DatasetReader, window: Window) -> int:
    """Payload bytes of the COG blocks a window read touches (from the TIFF block size tags)."""
    bh, bw = ds.block_shapes[0]
    total = 0
    for by in range(int(window.row_off) // bh, (int(window.row_off) + int(window.height) - 1) // bh + 1):
        for bx in range(int(window.col_off) // bw, (int(window.col_off) + int(window.width) - 1) // bw + 1):
            size = ds.get_tag_item(f"BLOCK_SIZE_{bx}_{by}", "TIFF", bidx=1)
            total += int(size) if size else 0
    return total


def read_product(product: Product, grid: Grid) -> ProductRead | None:
    """Read the grid window of one product's B8A, B12 and SCL and compute clear NBR.

    The BOA offset is taken from the values (section 4 of the module docstring): a product of
    processing baseline 04.00 or later whose clear-pixel 1st percentile of B12 is at or above
    :data:`RAW_OFFSET_P1_DN` still holds it, and 0.1 is subtracted from its reflectance. A product
    with fewer than :data:`OFFSET_CHECK_MIN_PIXELS` clear pixels is returned without NBR.

    Raises:
        RuntimeError: A band is not on the 20 m grid, or a product of a baseline before 04.00 has
            values that look offset.
    """
    left, _, _, top = grid.bounds
    arrays: dict[str, np.ndarray] = {}
    nbytes = 0
    window: Window | None = None
    origin = (0, 0)
    with rasterio.Env(**validate.gdal_options()):
        for band in ("B8A", "B12", "SCL"):
            with rasterio.open(f"/vsicurl/{product.url(band)}") as ds:
                t = ds.transform
                if (abs(t.a - RES), abs(t.e + RES)) != (0.0, 0.0) or ds.crs.to_epsg() != EPSG:
                    raise RuntimeError(f"{product.url(band)} is not on the EPSG:{EPSG} 20 m grid.")
                c0, r0 = (left - t.c) / RES, (t.f - top) / RES
                if c0 != round(c0) or r0 != round(r0):
                    raise RuntimeError(f"{product.url(band)} is not aligned with the 20 m grid.")
                col0, row0 = max(round(c0), 0), max(round(r0), 0)
                col1, row1 = min(round(c0) + grid.width, ds.width), min(round(r0) + grid.height, ds.height)
                if col1 <= col0 or row1 <= row0:
                    return None
                window = Window(col0, row0, col1 - col0, row1 - row0)
                arrays[band] = ds.read(1, window=window)
                nbytes += _window_bytes(ds, window)
                origin = (round(c0), round(r0))
    assert window is not None
    rows = slice(int(window.row_off) - origin[1], int(window.row_off) - origin[1] + int(window.height))
    cols = slice(int(window.col_off) - origin[0], int(window.col_off) - origin[0] + int(window.width))
    box_ = (int(window.col_off), int(window.row_off), int(window.width), int(window.height))
    dn8, dn12, scl = arrays["B8A"].astype(np.float32), arrays["B12"].astype(np.float32), arrays["SCL"]
    clear = np.isin(scl, SCL_CLEAR) & (dn8 > 0) & (dn12 > 0)
    n_clear = int(clear.sum())
    if n_clear < OFFSET_CHECK_MIN_PIXELS:
        skipped = f"skipped: {n_clear} clear pixels, too few to check the BOA offset"
        return ProductRead(product, rows, cols, box_, None, n_clear, None, 0.0, nbytes, skipped)
    p1 = float(np.percentile(dn12[clear], 1))
    raw = p1 >= RAW_OFFSET_P1_DN
    if raw and not product.offset_possible:
        raise RuntimeError(
            f"{product.name}: processing baseline {product.baseline} has no BOA offset, but its clear-pixel 1st "
            f"percentile of B12 is {p1:.0f} DN, which looks offset."
        )
    offset = BOA_OFFSET if raw else 0.0
    note: str | None = None
    if offset != product.item_offset:
        note = (
            f"item metadata (boa_offset_applied {product.offset_applied}) implies an offset of {product.item_offset}, "
            f"the values (B12 p1 {p1:.0f} DN) say {offset}"
        )
    b8 = dn8 * DN_SCALE + offset
    b12 = dn12 * DN_SCALE + offset
    den = b8 + b12
    ok = clear & (den > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = np.where(ok, (b8 - b12) / den, np.nan).astype(np.float32)
    return ProductRead(product, rows, cols, box_, nbr, int(ok.sum()), p1, offset, nbytes, note)


@dataclasses.dataclass
class Period:
    """A period's per-pixel median NBR and what went into it."""

    median: np.ndarray
    n_obs: np.ndarray
    records: list[dict[str, Any]]
    bytes: int
    seconds: float


def compose_period(dates: dict[dt.date, list[Product]], grid: Grid, *, workers: int) -> Period:
    """Read every product of the selected dates and take the per-pixel median of clear NBR over the dates."""
    t0 = time.perf_counter()
    order = sorted(dates)
    layers = np.full((len(order), *grid.shape), np.nan, dtype=np.float32)
    jobs = [(i, p) for i, day in enumerate(order) for p in dates[day]]
    records: list[dict[str, Any]] = []
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(read_product, p, grid): (i, p) for i, p in jobs}
        results: dict[tuple[int, str], ProductRead | None] = {}
        for future in concurrent.futures.as_completed(futures):
            i, p = futures[future]
            got = future.result()
            results[(i, p.name)] = got
            if got is not None:
                total += got.bytes
                logger.info(
                    "%s: %d clear pixels, B12 p1 %s DN, offset %s, %.1f MB%s",
                    p.name,
                    got.clear,
                    got.b12_p1_dn,
                    got.offset,
                    got.bytes / 1e6,
                    f" ({got.note})" if got.note else "",
                )
    # Fill each date's layer in a fixed order (most AOI coverage first), so the result is deterministic.
    for i, p in jobs:
        got = results[(i, p.name)]
        record = {
            "scene": p.name,
            "product": p.product_uri,
            "tile": p.tile,
            "date": p.date.isoformat(),
            "cloud_cover": p.cloud,
            "nodata_pixel_percentage": p.nodata_pct,
            "processing_baseline": ".".join(f"{v:02d}" for v in p.baseline),
            "boa_offset_applied": p.offset_applied,
            "item_reflectance_offset": p.item_offset,
            "reflectance_offset": None,
            "window": None,
            "clear_pixels": 0,
            "b12_p1_dn": None,
            "bytes": 0,
            "note": None,
        }
        if got is not None:
            record.update(
                window=list(got.window),
                clear_pixels=got.clear,
                b12_p1_dn=got.b12_p1_dn,
                reflectance_offset=got.offset if got.nbr is not None else None,
                bytes=got.bytes,
                note=got.note,
            )
        if got is not None and got.nbr is not None:
            layer = layers[i, got.rows, got.cols]
            fill = np.isnan(layer) & np.isfinite(got.nbr)
            layer[fill] = got.nbr[fill]
        records.append(record)
    n_obs = np.isfinite(layers).sum(axis=0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels give NaN, as they should
        median = np.nanmedian(layers, axis=0).astype(np.float32)
    del layers
    return Period(median=median, n_obs=n_obs, records=records, bytes=total, seconds=time.perf_counter() - t0)


# --------------------------------------------------------------------------------------------
# Classes, sanity checks and outputs
# --------------------------------------------------------------------------------------------


def classify(dnbr: np.ndarray, aoi: np.ndarray) -> np.ndarray:
    """MTBS dnbr6 codes from an offset dNBR (NaN where undefined) inside the AOI mask, :data:`NODATA` outside it."""
    d = np.rint(dnbr.astype(np.float64) * 1000.0)
    has = aoi & np.isfinite(d)
    out = np.full(dnbr.shape, NODATA, dtype=np.uint8)
    out[aoi] = 6
    with np.errstate(invalid="ignore"):
        out[has & (d < THRESHOLDS["increased_greenness_below"])] = 5
        out[has & (d >= THRESHOLDS["increased_greenness_below"]) & (d < THRESHOLDS["low_from"])] = 1
        out[has & (d >= THRESHOLDS["low_from"]) & (d < THRESHOLDS["moderate_from"])] = 2
        out[has & (d >= THRESHOLDS["moderate_from"]) & (d < THRESHOLDS["high_from"])] = 3
        out[has & (d >= THRESHOLDS["high_from"])] = 4
    return out


def class_table(classes: np.ndarray, where: np.ndarray) -> dict[str, dict[str, Any]]:
    """Pixels, hectares and share of each code among ``where`` pixels."""
    counts = np.bincount(classes[where].ravel(), minlength=7)
    total = int(counts.sum())
    return {
        str(c): {
            "name": CLASS_NAMES[c],
            "pixels": int(counts[c]),
            "hectares": round(float(counts[c]) * PIXEL_HA, 2),
            "share": float(counts[c] / total) if total else None,
        }
        for c in range(7)
    }


def clusters_away(classes: np.ndarray, away: np.ndarray, grid: Grid) -> dict[str, Any]:
    """Connected (8-neighbour) moderate or high clusters in ``away``, the largest first."""
    hot = away & ((classes == 3) | (classes == 4))
    labels, n = ndimage.label(hot, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return {"pixels": 0, "hectares": 0.0, "clusters": []}
    index = np.arange(1, n + 1)
    sizes = ndimage.sum(hot, labels, index)
    highs = ndimage.sum(classes == 4, labels, index)
    centres = ndimage.center_of_mass(hot, labels, index)
    to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
    big = [k for k in np.argsort(sizes)[::-1] if sizes[k] * PIXEL_HA >= CLUSTER_MIN_HA]
    listed = []
    for k in big[:CLUSTERS_LISTED]:
        r, c = centres[k]
        x, y = grid.transform * (c + 0.5, r + 0.5)
        lon, lat = to_ll.transform(x, y)
        listed.append(
            {
                "hectares": round(float(sizes[k]) * PIXEL_HA, 2),
                "high_share": float(highs[k] / sizes[k]),
                "lon": round(float(lon), 5),
                "lat": round(float(lat), 5),
            }
        )
    return {
        "pixels": int(hot.sum()),
        "hectares": round(float(hot.sum()) * PIXEL_HA, 2),
        "clusters_at_least_ha": CLUSTER_MIN_HA,
        "n_clusters_at_least": len(big),
        "hectares_in_clusters_at_least": round(float(sum(sizes[k] for k in big)) * PIXEL_HA, 2),
        "clusters": listed,
    }


def write_cog(path: Path, classes: np.ndarray, grid: Grid, tags: dict[str, str]) -> Path:
    """Write the classes as a uint8 COG with tags, a colour table and nodata :data:`NODATA`, atomically.

    The bytes depend only on the arguments, so pass no tag that changes from run to run.
    """
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "uint8",
        "nodata": NODATA,
        "crs": rasterio.crs.CRS.from_epsg(EPSG),
        "transform": grid.transform,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "DEFLATE",
    }
    with export.atomic_path(path) as out, tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        src = Path(tmp) / "classes.tif"
        with rasterio.open(src, "w", **profile) as ds:
            # The colour table sets the photometric tag, which GDAL cannot change once data is written.
            ds.write_colormap(1, COLOURS)
            ds.write(classes, 1)
            ds.update_tags(**tags)
            ds.set_band_description(1, "dnbr6")
        rasterio.shutil.copy(
            src, out, driver="COG", COMPRESS="DEFLATE", BLOCKSIZE="512", OVERVIEWS="AUTO", RESAMPLING="NEAREST"
        )
    return path


def cog_tags(
    *,
    offset: float,
    pre: Sequence[dict[str, Any]],
    post: Sequence[dict[str, Any]],
    tiles: Sequence[str],
    aoi_sha256: str,
) -> dict[str, str]:
    """The COG's ``TESSERA_APPS_*`` tags: source, codes, thresholds, offset, scenes, windows, grid inputs.

    Only what the map is made from goes in, never the time it was made, so the same products give
    the same tags (and the same COG bytes).

    Args:
        offset: The dNBR offset subtracted (median over unburned land).
        pre: The pre-fire period's scene records (:func:`compose_period`).
        post: The post-fire period's scene records.
        tiles: The MGRS tiles read.
        aoi_sha256: sha256 of the bundled ``caldor.geojson``.
    """
    windows = {r["scene"]: r["window"] for r in (*pre, *post) if r["window"] is not None}
    return {
        "TESSERA_APPS_SOURCE": SOURCE_TAG,
        "TESSERA_APPS_CODES": json.dumps(CLASS_NAMES),
        "TESSERA_APPS_NODATA": f"{NODATA}: outside the AOI, not mapped",
        "TESSERA_APPS_THRESHOLDS": json.dumps(
            {**THRESHOLDS, "scale": "round(1000 * dNBR)", "method": "Key and Benson"}
        ),
        "TESSERA_APPS_DNBR_OFFSET": f"{offset:.6f}",
        "TESSERA_APPS_OFFSET_REGION": (
            "median dNBR of AOI pixels outside caldor.geojson feature 1 (Hansen 2021 loss outline) buffered by "
            f"{UNBURNED_BUFFER_M:.0f} m"
        ),
        "TESSERA_APPS_PRE_PERIOD": f"{PRE[0]}/{PRE[1]}",
        "TESSERA_APPS_POST_PERIOD": f"{POST[0]}/{POST[1]}",
        "TESSERA_APPS_PRE_SCENES": json.dumps([r["scene"] for r in pre if r["window"] is not None]),
        "TESSERA_APPS_POST_SCENES": json.dumps([r["scene"] for r in post if r["window"] is not None]),
        "TESSERA_APPS_WINDOWS": json.dumps(windows),
        "TESSERA_APPS_TILES": ",".join(tiles),
        "TESSERA_APPS_SCL_CLEAR": ",".join(str(c) for c in SCL_CLEAR),
        "TESSERA_APPS_NBR": "median over dates of (B8A - B12) / (B8A + B12) on clear pixels, per period",
        "TESSERA_APPS_REFLECTANCE": (
            "DN * 1e-4, minus 0.1 when processing baseline >= 04.00 and the clear-pixel B12 1st percentile is at least "
            "1000 DN (the item's earthsearch:boa_offset_applied flag is wrong in this bucket)"
        ),
        "TESSERA_APPS_AOI": f"caldor.geojson feature 0 (sha256 {aoi_sha256})",
        "TESSERA_APPS_BUCKET": f"{BUCKET}/{PREFIX}",
        "TESSERA_APPS_SCRIPT": "scripts/make_caldor_dnbr.py",
    }


def _stats(values: np.ndarray) -> dict[str, float | int | None]:
    v = values[np.isfinite(values)]
    if not v.size:
        return {"pixels": 0, "median": None, "p05": None, "p95": None}
    p05, p50, p95 = np.percentile(v, [5, 50, 95])
    return {"pixels": int(v.size), "median": float(p50), "p05": float(p05), "p95": float(p95)}


def main() -> None:
    """Build the severity map and its JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="Output COG (default: the tessera_apps cache).")
    ap.add_argument("--max-dates", type=int, default=8, help="Dates kept per period.")
    ap.add_argument("--max-cloud", type=float, default=20.0, help="Largest eo:cloud_cover of a product used.")
    ap.add_argument("--workers", type=int, default=4, help="Products read at once.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out) if args.out is not None else default_out()
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    aoi_ll = demo.demo_aoi("caldor")
    outline_ll = demo.demo_feature("caldor", "hansen_2021_loss_outline")
    aoi_utm, outline_utm = to_utm(aoi_ll), to_utm(outline_ll)
    grid = make_grid(aoi_utm)
    aoi = burn(aoi_utm, grid)
    buffered = burn(outline_utm.buffer(UNBURNED_BUFFER_M), grid)
    inside_outline = aoi & burn(outline_utm, grid)
    unburned = aoi & ~buffered
    logger.info("Grid %d x %d at 20 m, %s; AOI %.0f ha.", grid.height, grid.width, grid.bounds, aoi.sum() * PIXEL_HA)

    tiles = choose_tiles(aoi_utm)
    logger.info("Tiles: %s", ", ".join(name for name, _ in tiles))
    periods: dict[str, Period] = {}
    selected: dict[str, dict[dt.date, list[Product]]] = {}
    for label, window in (("pre", PRE), ("post", POST)):
        products = pick_products([p for name, _ in tiles for p in list_products(name, window, aoi_utm)])
        dates = select_dates(products, aoi_utm, max_cloud=args.max_cloud, max_dates=args.max_dates)
        if not dates:
            raise RuntimeError(f"No {label}-fire product under {args.max_cloud} percent cloud in {window}.")
        selected[label] = dates
        logger.info(
            "%s-fire: %d products on %d dates listed, %d dates kept: %s",
            label,
            len(products),
            len({p.date for p in products}),
            len(dates),
            ", ".join(d.isoformat() for d in dates),
        )
        periods[label] = compose_period(dates, grid, workers=args.workers)
        logger.info("%s-fire: read %.1f MB in %.0f s.", label, periods[label].bytes / 1e6, periods[label].seconds)

    pre, post = periods["pre"], periods["post"]
    dnbr_raw = pre.median - post.median
    offset = float(np.nanmedian(dnbr_raw[unburned]))
    dnbr = dnbr_raw - np.float32(offset)
    classes = classify(dnbr, aoi)
    del dnbr_raw

    aoi_sha = hashlib.sha256(files("tessera_apps.data").joinpath("caldor.geojson").read_bytes()).hexdigest()
    tags = cog_tags(
        offset=offset, pre=pre.records, post=post.records, tiles=[name for name, _ in tiles], aoi_sha256=aoi_sha
    )
    write_cog(out, classes, grid, tags)

    seconds = time.perf_counter() - t0
    nbytes = pre.bytes + post.bytes
    sanity = {
        "class_areas_aoi": class_table(classes, aoi),
        "hansen_outline_by_class": class_table(classes, inside_outline),
        "offset_region_by_class": class_table(classes, unburned),
        "away_from_scar": clusters_away(classes, unburned, grid),
        "dnbr_x1000_inside_outline": _stats(dnbr[inside_outline] * 1000.0),
        "dnbr_x1000_offset_region": _stats(dnbr[unburned] * 1000.0),
        "observations_pre": {"min": int(pre.n_obs[aoi].min()), "median": float(np.median(pre.n_obs[aoi]))},
        "observations_post": {"min": int(post.n_obs[aoi].min()), "median": float(np.median(post.n_obs[aoi]))},
    }
    record = {
        "file": out.name,
        "sha256": export.file_sha256(out),
        "sha256_note": (
            "The COG holds no creation time, so this hash identifies its content: a rerun on the same products "
            "gives the same file."
        ),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "tags": tags,
        "grid": {"epsg": EPSG, "res_m": RES, "bounds": grid.bounds, "shape": grid.shape},
        "aoi_hectares": round(float(aoi.sum()) * PIXEL_HA, 2),
        "dnbr_offset": offset,
        "offset_region_pixels": int(np.isfinite(dnbr[unburned]).sum()),
        "tiles": [name for name, _ in tiles],
        "scenes": {"pre": pre.records, "post": post.records},
        "max_dates": args.max_dates,
        "max_cloud": args.max_cloud,
        "bytes_read": nbytes,
        "bytes_note": "Payload bytes of the COG blocks the windows touched, from the TIFF block size tags.",
        "seconds": round(seconds, 1),
        "sanity": sanity,
    }
    json_path = export.write_json(out.with_suffix(".json"), record)
    logger.info("Wrote %s and %s; read %.1f MB in %.0f s.", out, json_path, nbytes / 1e6, seconds)
    for c in range(7):
        row = sanity["class_areas_aoi"][str(c)]
        print(f"class {c} {row['name']:<52} {row['hectares']:>12,.0f} ha")
    print(f"dNBR offset {offset:+.4f}; Hansen outline by class:")
    for c in range(7):
        row = sanity["hansen_outline_by_class"][str(c)]
        share = row["share"]
        print(f"  {c} {row['name']:<52} {0.0 if share is None else share:7.1%}")
    away = sanity["away_from_scar"]
    n_big = away["n_clusters_at_least"]
    print(f"Moderate or high away from the scar: {away['hectares']:,.0f} ha, {n_big} clusters of 5 ha or more")


if __name__ == "__main__":
    main()
