"""Score app outputs against public reference datasets, and load those references on the AOI grid.

Scoring is pure numpy: :func:`binary_scores` and :func:`merge_scores` for a binary mask (ratios
are ``None`` when their denominator is zero, never NaN), :func:`confusion_matrix` and
:func:`per_class_f1` for labelled maps, and :func:`write_validation_json` for the
``<app>_validation.json`` file.

Every reference loader returns a bool array with shape ``plan.shape`` for one
:class:`~tessera_apps.io.ZonePlan` (apps call it once per zone). Pixels outside the plan's AOI
mask, or outside the reference's coverage, are False. Pixels are assigned by the pixel-centre
rule, the same rule :func:`~tessera_apps.io.plan_aoi` uses for the AOI mask.

Dataset loaders (survey section "Reference dataset endpoints"):

- **NIFC** wildland fire perimeters (US): :func:`nifc_perimeters`, ArcGIS REST.
- **EFFIS** burnt areas (Europe): :func:`effis_burnt_areas`, WFS.
- **Hansen GFC** (global forest loss): :func:`hansen_layer`, :func:`hansen_loss`,
  :func:`hansen_forest`, :func:`hansen_valid`, read as windows over ``/vsicurl/``.
- **Global Mangrove Watch**: :func:`gmw_extent`, :func:`gmw_coverage`, from Zenodo or a local copy.
- **Natural Earth** coastline: :func:`natural_earth_coastline`.
- **USFS FACTS** reforestation polygons: :func:`usfs_reforestation` (a demo site source, not a
  validation reference).

Downloads use the standard library (``urllib``), which honours ``HTTPS_PROXY``, and are cached
under :func:`cache_dir` ``/reference/<dataset>/``. Every download and cache file is written to a
temporary name in its destination directory and then renamed, so a failure never leaves a partial
file. Raster reads through GDAL get the proxy and CA bundle from the usual environment variables
inside a ``rasterio.Env`` (see :func:`gdal_options`); ``os.environ`` is never changed. When a host
cannot be reached, the error names the host and says how to supply the data by hand.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import logging
import math
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import rasterio
import rasterio.errors
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window as RasterWindow
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry

from tessera_apps import __version__
from tessera_apps.cache import cache_dir
from tessera_apps.export import args_digest, atomic_path, write_json
from tessera_apps.io import ZonePlan, _polygonal

if TYPE_CHECKING:
    import geopandas

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------

#: Hansen Global Forest Change version read by default (loss years 2001 to 2025, survey S13).
HANSEN_VERSION = "GFC-2025-v1.13"
#: Hansen tile URL. ``{tile}`` is the tile's top-left corner, for example ``40N_120W``. Point it at
#: local copies by passing ``url_template`` (a path with the same placeholders) to the loaders.
HANSEN_URL_TEMPLATE = (
    "https://storage.googleapis.com/earthenginepartners-hansen/{version}/Hansen_{version}_{layer}_{tile}.tif"
)
#: Single-band Hansen layers the loaders read. ``first`` and ``last`` are multi-band composites.
HANSEN_LAYERS = ("treecover2000", "gain", "lossyear", "datamask")
#: Hansen pixel size in degrees (about 30 m at the equator).
HANSEN_RES_DEG = 0.00025
#: Default canopy cover threshold (percent) for the Hansen forest proxy.
HANSEN_FOREST_THRESHOLD = 30
_HANSEN_MARGIN_DEG = 0.001

#: NIFC InterAgency Fire Perimeter History, all years (ArcGIS FeatureServer layer).
NIFC_HISTORY_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0"
)
#: EFFIS MapServer endpoint that serves WFS.
EFFIS_WFS_URL = "https://maps.effis.emergency.copernicus.eu/effis"
#: EFFIS burnt area layer (archive included, survey "Reference dataset endpoints").
EFFIS_LAYER = "effis.nrt.ba.poly"
#: USFS FACTS Silviculture Reforestation (all years), ArcGIS MapServer layer.
USFS_REFORESTATION_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_SilvicultureReforestation_01/MapServer/8"
#: Natural Earth coastline zip on the S3 mirror.
NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com/{scale}_physical/ne_{scale}_coastline.zip"
NATURAL_EARTH_SCALES = ("10m", "50m", "110m")

#: Zenodo REST endpoint for one record's metadata and file list.
ZENODO_RECORD_API = "https://zenodo.org/api/records/{record_id}"
#: Zenodo records of the Global Mangrove Watch versions this module reads.
GMW_RECORDS = {"v3": "6894273", "v4.1.12": "21346457"}
#: GMW v3 epochs. There is nothing after 2020 in v3 (survey S15).
GMW_V3_YEARS = (1996, 2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020)
#: GMW v4.1.12 is annual, one raster band per year: band = year - 1984.
GMW_V4_YEARS = tuple(range(1985, 2026))

_VECTOR_SUFFIXES = frozenset({".shp", ".gpkg", ".geojson", ".json", ".fgb"})
_RASTER_SUFFIXES = frozenset({".tif", ".tiff", ".kea", ".vrt"})
_USER_AGENT = f"tessera_apps/{__version__} (Python urllib)"
_TIMEOUT_S = 120.0
_MAX_PAGES = 1000
_READ_BLOCK = 1 << 20

_HINTS = {
    "NIFC": (
        "Download the perimeter by hand (for example from the NIFC open data site, "
        "https://data-nifc.opendata.arcgis.com) and pass it with --reference path.geojson "
        "(reference_from_file)."
    ),
    "EFFIS": (
        "Download the burnt area polygons by hand from EFFIS (https://forest-fire.emergency.copernicus.eu, "
        "Data and services) and pass them with --reference path.geojson (reference_from_file)."
    ),
    "Hansen GFC": (
        "Download the tiles by hand (https://storage.googleapis.com/earthenginepartners-hansen/{version}/"
        "download.html) and pass url_template='/your/dir/Hansen_{{version}}_{{layer}}_{{tile}}.tif', "
        "or pass a reference raster with --reference."
    ),
    "GMW": (
        "Download the GMW {version} files by hand from https://zenodo.org/records/{record} and pass the file "
        "or folder with source= (or the polygons with --reference path.geojson)."
    ),
    "Natural Earth": (
        "Download ne_{scale}_coastline.zip by hand from https://www.naturalearthdata.com and pass it with source=."
    ),
    "USFS FACTS": (
        "Export the reforestation polygons by hand from the USFS FACTS data "
        "(https://data.fs.usda.gov/geodata/edw/datasets.php) and pass them with --reference or --site."
    ),
}

# --------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------


class ReferenceDataError(RuntimeError):
    """A reference dataset could not be loaded, or did not have the form this module expects."""


class ReferenceUnavailableError(ReferenceDataError):
    """The host serving a reference dataset could not be reached (or refused the request)."""


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


@dataclasses.dataclass(frozen=True)
class BinaryScores:
    """Pixel counts and agreement ratios of a binary prediction against a reference.

    Ratios are ``None`` (JSON null) when their denominator is zero: precision with no predicted
    positive, recall with no reference positive, F1 and IoU with neither, accuracy with no pixel.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    iou: float | None
    accuracy: float | None
    n: int

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int, tn: int) -> BinaryScores:
        """Build the scores from the four confusion counts.

        Raises:
            ValueError: A count is negative.
        """
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        if min(tp, fp, fn, tn) < 0:
            raise ValueError(f"Counts must be non-negative, got tp={tp} fp={fp} fn={fn} tn={tn}.")
        n = tp + fp + fn + tn
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=_ratio(tp, tp + fp),
            recall=_ratio(tp, tp + fn),
            f1=_ratio(2 * tp, 2 * tp + fp + fn),
            iou=_ratio(tp, tp + fp + fn),
            accuracy=_ratio(tp + tn, n),
            n=n,
        )


def _as_bool(a: np.ndarray, name: str) -> np.ndarray:
    """``a`` as a bool array; integers and floats must hold only 0 and 1."""
    arr = np.asarray(a)
    if arr.dtype == np.bool_:
        return arr
    if arr.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a bool or 0/1 numeric array, got dtype {arr.dtype}.")
    if arr.size and not np.isin(arr, (0, 1)).all():
        raise ValueError(f"{name} must hold only 0 and 1 (or be bool); found other values (NaN or nodata?).")
    return arr.astype(bool)


def _check_shapes(**arrays: np.ndarray | None) -> tuple[int, ...]:
    shapes = {k: np.shape(v) for k, v in arrays.items() if v is not None}
    first = next(iter(shapes.values()))
    if any(s != first for s in shapes.values()):
        raise ValueError(f"Shapes differ: {shapes}.")
    return first


def binary_scores(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray | None = None) -> BinaryScores:
    """Confusion counts and ratios of a binary prediction against a binary reference.

    Args:
        pred: Predicted mask, bool or 0/1.
        ref: Reference mask, bool or 0/1, same shape.
        valid: Pixels to score (for example the AOI mask and the reference's coverage). All pixels
            when ``None``.

    Raises:
        ValueError: Shapes differ, or a numeric input holds values other than 0 and 1.
    """
    _check_shapes(pred=pred, ref=ref, valid=valid)
    p = _as_bool(pred, "pred")
    r = _as_bool(ref, "ref")
    if valid is not None:
        v = _as_bool(valid, "valid")
        p, r = p[v], r[v]
    tp = int(np.count_nonzero(p & r))
    fp = int(np.count_nonzero(p & ~r))
    fn = int(np.count_nonzero(~p & r))
    tn = int(p.size) - tp - fp - fn
    return BinaryScores.from_counts(tp, fp, fn, tn)


def merge_scores(parts: Sequence[BinaryScores]) -> BinaryScores:
    """Sum the counts of several scores (zones or blocks of one AOI) and recompute the ratios.

    The result equals :func:`binary_scores` on the concatenated pixels. An empty sequence gives
    zero counts and ``None`` ratios.
    """
    return BinaryScores.from_counts(
        sum(p.tp for p in parts), sum(p.fp for p in parts), sum(p.fn for p in parts), sum(p.tn for p in parts)
    )


def confusion_matrix(
    pred: np.ndarray, ref: np.ndarray, labels: Sequence[int], valid: np.ndarray | None = None
) -> np.ndarray:
    """Confusion matrix of labelled maps: rows are reference labels, columns predicted labels.

    Pixels whose reference or prediction is not in ``labels`` are left out, as in
    ``sklearn.metrics.confusion_matrix``.

    Args:
        pred: Predicted labels.
        ref: Reference labels, same shape.
        labels: The labels, in the order of the rows and columns. Must not repeat.
        valid: Pixels to count; all when ``None``.

    Returns:
        int64 array of shape ``(len(labels), len(labels))``.

    Raises:
        ValueError: Shapes differ, or ``labels`` is empty or repeats a label.
    """
    _check_shapes(pred=pred, ref=ref, valid=valid)
    labs = np.asarray(labels)
    if labs.ndim != 1 or labs.size == 0:
        raise ValueError("labels must be a non-empty sequence.")
    if np.unique(labs).size != labs.size:
        raise ValueError(f"labels repeat: {list(labels)}.")
    p = np.asarray(pred)
    r = np.asarray(ref)
    if valid is not None:
        v = _as_bool(valid, "valid")
        p, r = p[v], r[v]
    p, r = p.ravel(), r.ravel()
    order = np.argsort(labs, kind="stable")
    ordered = labs[order]
    k = labs.size

    def index(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos = np.clip(np.searchsorted(ordered, x), 0, k - 1)
        return order[pos], ordered[pos] == x

    pi, p_ok = index(p)
    ri, r_ok = index(r)
    keep = p_ok & r_ok
    flat = ri[keep].astype(np.int64) * k + pi[keep]
    return np.bincount(flat, minlength=k * k).reshape(k, k).astype(np.int64)


def per_class_f1(cm: np.ndarray, labels: Sequence[int]) -> dict[int, float | None]:
    """F1 of each class from a confusion matrix (rows reference, columns prediction).

    A class that is neither in the reference nor predicted has F1 ``None``.

    Raises:
        ValueError: ``cm`` is not square with one row per label.
    """
    m = np.asarray(cm)
    k = len(labels)
    if m.shape != (k, k):
        raise ValueError(f"cm has shape {m.shape}; expected ({k}, {k}) for {k} labels.")
    tp = np.diag(m)
    den = m.sum(axis=0) + m.sum(axis=1)
    return {int(lab): _ratio(2 * int(tp[i]), int(den[i])) for i, lab in enumerate(labels)}


def write_validation_json(
    path: Path, *, scores: Mapping[str, Any], reference: Mapping[str, Any], caveats: Sequence[str]
) -> Path:
    """Write ``<app>_validation.json`` atomically: ``{"scores", "reference", "caveats"}``.

    Args:
        path: Destination.
        scores: Named scores, for example ``{"burned": binary_scores(...)}``. :class:`BinaryScores`
            and numpy values are converted; ``None`` and NaN become null.
        reference: What was compared against: dataset, version, URL or file, year window.
        caveats: Plain sentences on what the comparison can and cannot show.

    Returns:
        ``path``.
    """
    notes = [caveats] if isinstance(caveats, str) else [str(c) for c in caveats]
    return write_json(Path(path), {"scores": dict(scores), "reference": dict(reference), "caveats": notes})


# --------------------------------------------------------------------------------------------
# Reference rasters on the AOI grid
# --------------------------------------------------------------------------------------------


def _plan_bounds(plan: ZonePlan) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) of the plan window in the zone CRS."""
    t = plan.transform
    h, w = plan.shape
    return t.c, t.f + h * t.e, t.c + w * t.a, t.f


def plan_lonlat_bounds(plan: ZonePlan, *, margin_deg: float = 0.0) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the plan's window in lon/lat, widened by ``margin_deg``."""
    west, south, east, north = plan.grid.to_lonlat().transform_bounds(*_plan_bounds(plan), densify_pts=21)
    return (
        max(west - margin_deg, -180.0),
        max(south - margin_deg, -90.0),
        min(east + margin_deg, 180.0),
        min(north + margin_deg, 90.0),
    )


def _lonlat_polygons(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    """Union of the polygonal parts of ``geometries`` (invalid ones repaired)."""
    parts = []
    for g in geometries:
        if g is None or g.is_empty:
            continue
        valid = g if g.is_valid else shapely.make_valid(g)
        poly = _polygonal(valid)
        if not poly.is_empty:
            parts.append(poly)
    if not parts:
        return shapely.Polygon()
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def reference_from_geometries(
    plan: ZonePlan, geometries: Iterable[BaseGeometry], *, densify_deg: float = 0.001
) -> np.ndarray:
    """Rasterize lon/lat polygons onto a zone plan's grid by the pixel-centre rule.

    Only polygonal parts count (lines and points are dropped, as in :func:`~tessera_apps.io.plan_aoi`).
    The polygons are clipped to the plan's window, densified to ``densify_deg`` so long edges follow
    the projection's curvature, projected to the zone CRS and burned where a pixel centre falls
    inside. Pixels outside ``plan.mask`` are False.

    Returns:
        bool array of shape ``plan.shape``.
    """
    out = np.zeros(plan.shape, dtype=bool)
    union = _lonlat_polygons(geometries)
    if union.is_empty:
        return out
    clip = box(*plan_lonlat_bounds(plan, margin_deg=0.01))
    piece = _polygonal(union.intersection(clip))
    if piece.is_empty:
        return out
    projected = _polygonal(
        shapely.transform(shapely.segmentize(piece, densify_deg), plan.grid.from_lonlat().transform, interleaved=False)
    )
    if projected.is_empty or projected.area == 0:
        return out
    burned = rasterize(
        [(projected, 1)], out_shape=plan.shape, transform=plan.transform, all_touched=False, fill=0, dtype="uint8"
    )
    return burned.astype(bool) & plan.mask


def _read_vector(
    path: str | Path,
    bbox: tuple[float, float, float, float] | None,
    **kwargs: Any,  # noqa: ANN401
) -> geopandas.GeoDataFrame:
    """Read a vector file in EPSG:4326, keeping only features that intersect a lon/lat bbox.

    A file that declares no CRS is read whole and taken as lon/lat only if its coordinates fit.
    """
    import geopandas as gpd
    import pyogrio

    if pyogrio.read_info(str(path), **kwargs).get("crs") is not None:
        if bbox is None:
            return gpd.read_file(path, **kwargs).to_crs("EPSG:4326")
        frame = gpd.read_file(path, bbox=gpd.GeoSeries([box(*bbox)], crs="EPSG:4326"), **kwargs)
        return frame.to_crs("EPSG:4326")
    frame = gpd.read_file(path, **kwargs)
    if len(frame):
        west, south, east, north = frame.total_bounds
        if not (-180.0 <= west <= east <= 180.0 and -90.0 <= south <= north <= 90.0):
            raise ReferenceDataError(
                f"{path} declares no CRS and its coordinates are not lon/lat; add a .prj or save it with a CRS."
            )
        logger.warning("%s declares no CRS; its coordinates look like lon/lat, so EPSG:4326 is assumed.", path)
    frame = frame.set_crs("EPSG:4326")
    if bbox is not None:
        frame = frame[frame.intersects(box(*bbox))]
    return frame


def reference_from_file(plan: ZonePlan, path: Path | str, *, layer: str | int | None = None) -> np.ndarray:
    """Rasterize a polygon file (GeoJSON, GeoPackage, shapefile, any CRS) onto a zone plan's grid.

    This is the ``--reference`` path. Features outside the plan's window are not read; the rest are
    reprojected to EPSG:4326 and rasterized by :func:`reference_from_geometries`.

    Args:
        plan: The zone plan.
        path: Vector file readable by GDAL. A file without a CRS is taken as lon/lat only when its
            coordinates fit lon/lat.
        layer: Layer of a multi-layer file (GeoPackage); the first layer when ``None``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ReferenceDataError: The file has no CRS and is not in lon/lat.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference file {p} does not exist.")
    kwargs: dict[str, Any] = {} if layer is None else {"layer": layer}
    frame = _read_vector(p, plan_lonlat_bounds(plan, margin_deg=0.01), **kwargs)
    return reference_from_geometries(plan, [g for g in frame.geometry if g is not None])


#: GDAL settings for ``/vsicurl/`` reads. Each one gives way to the same variable in the environment.
#: The timeouts turn a stalled connection into an error instead of a hang: 30 s to connect, 300 s for
#: one request, and a request that moves less than 1 byte/s for 60 s is dropped. GDAL retries HTTP
#: 429 and 5xx answers up to ``GDAL_HTTP_MAX_RETRY`` times, so the worst wait for one range is about
#: four times ``GDAL_HTTP_TIMEOUT``; a stall is not retried.
_GDAL_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF,.vrt",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_CONNECTTIMEOUT": "30",
    "GDAL_HTTP_TIMEOUT": "300",
    "GDAL_HTTP_LOW_SPEED_TIME": "60",
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1",
}


def gdal_options() -> dict[str, str]:
    """GDAL configuration for ``/vsicurl/`` reads, for use in ``rasterio.Env(**gdal_options())``.

    ``GDAL_HTTP_PROXY`` comes from ``HTTPS_PROXY`` (or ``https_proxy``) unless it is already set.
    ``CURL_CA_BUNDLE`` comes from ``CURL_CA_BUNDLE``, ``REQUESTS_CA_BUNDLE`` or ``SSL_CERT_FILE`` (first
    set) unless ``GDAL_CURL_CA_BUNDLE`` is set. Directory listing on open is disabled, so a tile read
    costs a header request plus the ranges it needs. Connect, request and low-speed timeouts are set
    so a stalled connection raises instead of blocking forever. Any of these settings already in the
    environment is left to GDAL, which reads it from there.
    """
    env = os.environ
    opts = {key: value for key, value in _GDAL_DEFAULTS.items() if not env.get(key)}
    if not env.get("GDAL_HTTP_PROXY"):
        proxy = env.get("HTTPS_PROXY") or env.get("https_proxy")
        if proxy:
            opts["GDAL_HTTP_PROXY"] = proxy
    if not env.get("GDAL_CURL_CA_BUNDLE"):
        bundle = env.get("CURL_CA_BUNDLE") or env.get("REQUESTS_CA_BUNDLE") or env.get("SSL_CERT_FILE")
        if bundle:
            opts["CURL_CA_BUNDLE"] = bundle
    return opts


def _gdal_path(src: str | Path) -> str:
    text = str(src)
    if text.startswith(("http://", "https://")):
        return f"/vsicurl/{text}"
    if text.startswith("file://"):
        return urllib.request.url2pathname(urllib.parse.urlsplit(text).path)
    return text


def _is_remote(src: str | Path) -> bool:
    return str(src).startswith(("http://", "https://"))


def _host(url: str) -> str:
    return urllib.parse.urlsplit(url).hostname or url


_RASTER_HINT = "Download the file and pass its local path (--reference path.tif)."


def _gdal_reason(exc: BaseException) -> str:
    """GDAL's message for a failed read, with the underlying GDAL error when rasterio chained one."""
    cause = exc.__cause__
    return f"{exc} ({cause})" if cause is not None and str(cause) not in str(exc) else str(exc)


def _remote_read_error(src: str, dataset: str, hint: str, exc: BaseException) -> ReferenceUnavailableError:
    return ReferenceUnavailableError(
        f"Could not read {src} from {_host(src)} for {dataset} ({_gdal_reason(exc)}). {hint}"
    )


def _open_raster(src: str | Path, *, dataset: str, hint: str) -> rasterio.io.DatasetReader:
    """Open a raster path or URL. Call inside ``rasterio.Env(**gdal_options())``.

    For an ``http(s)`` URL, a failed open and a dataset that came back without its georeferencing
    (no CRS and an identity transform, which is what GDAL returns when the request for the GeoTIFF
    tags fails after the header was read) both raise ReferenceUnavailableError naming the host.
    """
    try:
        ds = rasterio.open(_gdal_path(src))
    except rasterio.errors.RasterioIOError as exc:
        if _is_remote(src):
            raise _remote_read_error(str(src), dataset, hint, exc) from exc
        raise
    if _is_remote(src) and ds.crs is None and ds.transform.is_identity:
        ds.close()
        raise ReferenceUnavailableError(
            f"Could not read the georeferencing of {src} from {_host(str(src))} for {dataset}: GDAL opened it "
            f"with no CRS and no transform, so the request for the GeoTIFF tags failed or the file has none. {hint}"
        )
    return ds


def _pixel_window(
    ds: rasterio.io.DatasetReader, bounds: tuple[float, float, float, float], margin: float
) -> RasterWindow | None:
    """Pixel window of ``ds`` covering ``bounds`` (in its CRS) plus ``margin``, clipped to the raster.

    The four corners go through the inverse transform, so north-up, south-up (positive y pixel size)
    and rotated rasters all work. ``None`` when the window misses the raster.
    """
    left, bottom, right, top = bounds
    inverse = ~ds.transform
    corners = [inverse @ (x, y) for x in (left - margin, right + margin) for y in (bottom - margin, top + margin)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    col0 = max(math.floor(min(cols)), 0)
    row0 = max(math.floor(min(rows)), 0)
    col1 = min(math.ceil(max(cols)), ds.width)
    row1 = min(math.ceil(max(rows)), ds.height)
    if col1 <= col0 or row1 <= row0:
        return None
    return RasterWindow(col0, row0, col1 - col0, row1 - row0)


def _raster_on_plan(
    src: str | Path, plan: ZonePlan, *, band: int = 1, resampling: Resampling = Resampling.nearest
) -> tuple[np.ndarray, np.ndarray]:
    """Read the window of a raster around a zone plan and reproject it onto the plan grid.

    Returns:
        ``(values, covered)``: values in the raster's dtype (0 where not covered) and a bool mask of
        plan pixels that received data (inside the raster and not nodata or masked).
    """
    values, covered, _inside = _read_on_plan(src, plan, band=band, resampling=resampling)
    return values, covered


def _read_on_plan(
    src: str | Path,
    plan: ZonePlan,
    *,
    band: int = 1,
    resampling: Resampling = Resampling.nearest,
    dataset: str = "the reference raster",
    hint: str = _RASTER_HINT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`_raster_on_plan` plus the raster's footprint on the plan grid.

    Returns:
        ``(values, covered, inside)``: ``inside`` marks plan pixels whose centre falls inside the
        raster, whatever their value (nodata included).

    Raises:
        ReferenceUnavailableError: A remote raster could not be opened or read, at any point.
    """
    values: np.ndarray
    covered = np.zeros(plan.shape, dtype=np.uint8)
    inside = np.zeros(plan.shape, dtype=np.uint8)
    with rasterio.Env(**gdal_options()), _open_raster(src, dataset=dataset, hint=hint) as ds:
        if not 1 <= band <= ds.count:
            raise ValueError(f"{src} has {ds.count} band(s); band {band} does not exist.")
        if ds.crs is None:
            raise ReferenceDataError(f"{src} has no CRS; it cannot be placed on the AOI grid.")
        values = np.zeros(plan.shape, dtype=ds.dtypes[band - 1])
        bounds = transform_bounds(plan.grid.crs, ds.crs, *_plan_bounds(plan), densify_pts=21)
        window = _pixel_window(ds, bounds, 2 * max(abs(ds.res[0]), abs(ds.res[1])))
        if window is None:
            logger.info("%s does not overlap the %s plan window.", src, plan.zone)
            return values, covered.astype(bool), inside.astype(bool)
        try:
            data = ds.read(band, window=window, masked=True)
        except rasterio.errors.RasterioIOError as exc:
            if _is_remote(src):
                raise _remote_read_error(str(src), dataset, hint, exc) from exc
            raise
        src_transform = ds.window_transform(window)
        src_crs = ds.crs
        nodata = ds.nodata
    present = ~np.ma.getmaskarray(data)
    if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
        filled = data.filled(nodata)
        reproject(
            filled,
            values,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=nodata,
            dst_transform=plan.transform,
            dst_crs=plan.grid.crs,
            dst_nodata=nodata,
            resampling=resampling,
            tolerance=0.0,
        )
    else:
        reproject(
            data.filled(0),
            values,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=plan.transform,
            dst_crs=plan.grid.crs,
            resampling=resampling,
            tolerance=0.0,
        )
    for flags, dst in ((present.astype(np.uint8), covered), (np.ones(present.shape, dtype=np.uint8), inside)):
        reproject(
            flags,
            dst,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=plan.transform,
            dst_crs=plan.grid.crs,
            resampling=Resampling.nearest,
            tolerance=0.0,
        )
    is_covered = covered.astype(bool)
    values[~is_covered] = 0
    return values, is_covered, inside.astype(bool)


def reference_from_raster(
    plan: ZonePlan,
    src: str | Path,
    predicate: Callable[[np.ndarray], np.ndarray],
    *,
    band: int = 1,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """A reference mask from a raster in any CRS: ``predicate(values)`` on the plan grid.

    Only the window of ``src`` covering the plan (plus a two-pixel margin) is read. An ``http(s)``
    URL is read in place through ``/vsicurl/``.

    Args:
        plan: The zone plan.
        src: Raster path or URL.
        predicate: Maps the band's values (in the raster's dtype, on the plan grid) to a bool mask,
            for example ``lambda v: v == 1``.
        band: 1-based band index.
        resampling: Resampling from the raster's grid to the 10 m grid. Nearest suits categories.

    Returns:
        bool array of shape ``plan.shape``; False outside the raster, on its nodata pixels and
        outside ``plan.mask``.

    Raises:
        ValueError: The band does not exist, or ``predicate`` returns the wrong shape.
        ReferenceUnavailableError: A remote raster could not be read.
    """
    values, covered = _raster_on_plan(src, plan, band=band, resampling=resampling)
    result = np.asarray(predicate(values), dtype=bool)
    if result.shape != plan.shape:
        raise ValueError(f"predicate returned shape {result.shape}; expected {plan.shape}.")
    return result & covered & plan.mask


# --------------------------------------------------------------------------------------------
# Downloads and the cache
# --------------------------------------------------------------------------------------------


def reference_dir(dataset: str) -> Path:
    """Cache directory of one reference dataset: ``cache_dir() / "reference" / dataset``."""
    return cache_dir() / "reference" / dataset


def _cache_key(params: Mapping[str, Any]) -> str:
    """Short cache file key: the first 16 hex digits of :func:`tessera_apps.export.args_digest`."""
    return args_digest(params)[:16]


def _write_json_atomic(path: Path, obj: object) -> None:
    """Compact JSON for a cache entry, written through :func:`tessera_apps.export.atomic_path`."""
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def _ssl_context() -> ssl.SSLContext:
    """Default trust store (which honours ``SSL_CERT_FILE``) plus any bundle named by the usual variables."""
    ctx = ssl.create_default_context()
    for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        bundle = os.environ.get(var)
        if bundle and Path(bundle).is_file():
            ctx.load_verify_locations(cafile=bundle)
    return ctx


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:  # noqa: ANN401
    """Open a URL with the default handlers (``HTTPS_PROXY`` and ``NO_PROXY`` honoured). Tests replace this."""
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())


_NET_ERRORS: tuple[type[BaseException], ...] = (urllib.error.URLError, http.client.HTTPException, OSError)


def _reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return f"{type(exc).__name__}: {exc}"


def _unreachable(url: str, dataset: str, hint: str, exc: BaseException) -> ReferenceUnavailableError:
    return ReferenceUnavailableError(f"Could not reach {_host(url)} for {dataset} ({_reason(exc)}). {hint}")


def _fetch(
    url: str,
    *,
    dataset: str,
    hint: str,
    params: Mapping[str, object] | None = None,
    post: bool = False,
    timeout: float = _TIMEOUT_S,
) -> bytes:
    """GET (or form POST) ``url`` and return the body; network failures raise ReferenceUnavailableError."""
    data = None
    headers = {"User-Agent": _USER_AGENT}
    if params:
        encoded = urllib.parse.urlencode({k: str(v) for k, v in params.items()})
        if post:
            data = encoded.encode("ascii")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            url = f"{url}{'&' if '?' in url else '?'}{encoded}"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with _urlopen(request, timeout) as resp:
            return resp.read()
    except _NET_ERRORS as exc:
        raise _unreachable(url, dataset, hint, exc) from exc


def _exists(url: str, *, dataset: str, hint: str, timeout: float = _TIMEOUT_S) -> bool:
    """HEAD ``url``: True on success, False on HTTP 404; anything else raises ReferenceUnavailableError."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with _urlopen(request, timeout):
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise _unreachable(url, dataset, hint, exc) from exc
    except _NET_ERRORS as exc:
        raise _unreachable(url, dataset, hint, exc) from exc


def _download(url: str, dest: Path, *, dataset: str, hint: str, timeout: float = _TIMEOUT_S) -> Path:
    """Download ``url`` to ``dest`` once (atomic); an existing ``dest`` is reused as is."""
    if dest.exists():
        return dest
    logger.info("Downloading %s to %s", url, dest)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        resp = _urlopen(request, timeout)
    except _NET_ERRORS as exc:
        raise _unreachable(url, dataset, hint, exc) from exc
    with resp, atomic_path(dest) as tmp, tmp.open("wb") as fh:
        expected = resp.headers.get("Content-Length") if getattr(resp, "headers", None) is not None else None
        written = 0
        while True:
            try:
                block = resp.read(_READ_BLOCK)
            except (http.client.HTTPException, OSError) as exc:
                raise _unreachable(url, dataset, hint, exc) from exc
            if not block:
                break
            fh.write(block)
            written += len(block)
        if expected is not None and int(expected) != written:
            raise ReferenceUnavailableError(
                f"Download of {url} from {_host(url)} for {dataset} stopped after {written} of {expected} bytes. {hint}"
            )
    return dest


def _check_bbox(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """A lon/lat ``(west, south, east, north)`` with west < east and south < north, as floats."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (west, south, east, north); got {bbox!r}.")
    west, south, east, north = (float(v) for v in bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError(f"bbox {bbox!r} is not finite.")
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError(
            f"bbox {bbox!r} must be lon/lat (west, south, east, north) with west < east and south < north."
        )
    return west, south, east, north


def _geometries(collection: Mapping[str, Any]) -> list[BaseGeometry]:
    return [shape(f["geometry"]) for f in collection.get("features", []) if f.get("geometry")]


def _read_json_file(path: Path) -> Any:  # noqa: ANN401
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------------
# ArcGIS REST (NIFC, USFS FACTS)
# --------------------------------------------------------------------------------------------


def _layer_url(url: str) -> str:
    url = url.rstrip("/")
    return url[: -len("/query")] if url.endswith("/query") else url


def _arcgis_json(
    url: str, params: Mapping[str, object], *, dataset: str, hint: str, post: bool = False
) -> dict[str, Any]:
    body = _fetch(url, params=params, post=post, dataset=dataset, hint=hint)
    try:
        obj = json.loads(body)
    except ValueError as exc:
        raise ReferenceDataError(f"{dataset}: {_host(url)} did not return JSON (first bytes: {body[:120]!r}).") from exc
    if not isinstance(obj, dict):
        raise ReferenceDataError(f"{dataset}: {_host(url)} returned JSON that is not an object.")
    if "error" in obj:
        err = obj["error"] or {}
        details = "; ".join(str(d) for d in err.get("details") or [])
        raise ReferenceDataError(
            f"{dataset}: the ArcGIS service at {_host(url)} returned error {err.get('code')}: "
            f"{err.get('message')}{f' ({details})' if details else ''}."
        )
    return obj


def _find_field(info: Mapping[str, Any], candidates: Sequence[str]) -> dict[str, Any] | None:
    """The first field of a layer whose name matches one of ``candidates`` (case-insensitive)."""
    fields = {str(f.get("name", "")).upper(): f for f in info.get("fields") or []}
    for name in candidates:
        if name.upper() in fields:
            return dict(fields[name.upper()])
    return None


def _field_names(info: Mapping[str, Any]) -> list[str]:
    return [str(f.get("name")) for f in info.get("fields") or []]


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _is_text_field(field: Mapping[str, Any]) -> bool:
    return str(field.get("type", "")).endswith("String")


def _exceeded(collection: Mapping[str, Any]) -> bool:
    """ArcGIS paging flag: top-level ``exceededTransferLimit`` or, in GeoJSON, under ``properties``."""
    if collection.get("exceededTransferLimit"):
        return True
    props = collection.get("properties") or {}
    return bool(props.get("exceededTransferLimit"))


def _feature_id(feature: Mapping[str, Any], oid: str) -> object:
    if feature.get("id") is not None:
        return feature["id"]
    props = feature.get("properties") or {}
    return props.get(oid, props.get(oid.lower()))


def _arcgis_query(
    layer: str,
    info: Mapping[str, Any],
    *,
    where: str,
    bbox: tuple[float, float, float, float],
    dataset: str,
    hint: str,
) -> list[dict[str, Any]]:
    """Every feature matching ``where`` and intersecting ``bbox``, as GeoJSON features in EPSG:4326.

    Pages with ``resultOffset`` while the response sets ``exceededTransferLimit``. A layer that does
    not support pagination is read by object id instead: ids first, then batches of features.
    """
    query = f"{layer}/query"
    west, south, east, north = bbox
    spatial = {
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }
    out_params = {"outFields": "*", "returnGeometry": "true", "outSR": "4326", "f": "geojson"}
    page = max(1, min(int(info.get("maxRecordCount") or 1000), 2000))
    oid_field = info.get("objectIdField") or next(
        (f["name"] for f in info.get("fields") or [] if f.get("type") == "esriFieldTypeOID"), "OBJECTID"
    )
    paging = (info.get("advancedQueryCapabilities") or {}).get("supportsPagination", True)
    features: list[dict[str, Any]] = []
    if paging:
        seen: set[object] = set()
        offset = 0
        for _ in range(_MAX_PAGES):
            params = {
                "where": where,
                **spatial,
                **out_params,
                "orderByFields": oid_field,
                "resultOffset": offset,
                "resultRecordCount": page,
            }
            fc = _arcgis_json(query, params, dataset=dataset, hint=hint, post=True)
            batch = fc.get("features") or []
            ids = {_feature_id(f, oid_field) for f in batch} - {None}
            if batch and ids and ids <= seen:
                raise ReferenceDataError(
                    f"{dataset}: {_host(layer)} returned the same page twice at offset {offset}; paging is broken."
                )
            seen |= ids
            features.extend(batch)
            if not batch or not _exceeded(fc):
                return features
            offset += len(batch)
        raise ReferenceDataError(f"{dataset}: more than {_MAX_PAGES} pages from {_host(layer)}; narrow the query.")
    id_resp = _arcgis_json(
        query, {"where": where, **spatial, "returnIdsOnly": "true", "f": "json"}, dataset=dataset, hint=hint, post=True
    )
    ids_all = sorted(int(i) for i in id_resp.get("objectIds") or [])
    for start in range(0, len(ids_all), page):
        chunk = ids_all[start : start + page]
        params = {"objectIds": ",".join(str(i) for i in chunk), **out_params}
        fc = _arcgis_json(query, params, dataset=dataset, hint=hint, post=True)
        features.extend(fc.get("features") or [])
    return features


def nifc_perimeters(
    bbox: tuple[float, float, float, float],
    *,
    year: int,
    name: str | None = None,
    url: str = NIFC_HISTORY_URL,
) -> list[BaseGeometry]:
    """Wildland fire perimeters from the NIFC InterAgency Fire Perimeter History (US only).

    Queries the ArcGIS FeatureServer layer ``url`` for perimeters intersecting ``bbox`` in fire year
    ``year``, optionally only incidents whose name contains ``name`` (``UPPER(INCIDENT) LIKE``).
    The year and incident field names are read from the layer (``FIRE_YEAR_INT`` or ``FIRE_YEAR``,
    ``INCIDENT``), results are paged with ``resultOffset``, and the answer is cached under
    ``reference_dir("nifc")`` by query (url, bbox, year and name). An empty answer is not cached, so
    a later run asks again once the perimeter is published. NIFC perimeters are outer boundaries
    that include unburned islands, so pixel-level IoU against them has a ceiling well below 1
    (survey S10).

    Unverified against the live service until the network test runs: the build sandbox could not
    reach services3.arcgis.com. The request format follows the ArcGIS REST API documentation.

    Args:
        bbox: Lon/lat ``(west, south, east, north)``.
        year: Fire year.
        name: Part of the incident name, for example ``"Caldor"``. Case does not matter.
        url: Layer URL (a trailing ``/query`` is accepted).

    Returns:
        The perimeters in lon/lat; empty when none match.

    Raises:
        ReferenceUnavailableError: The service could not be reached.
        ReferenceDataError: The layer lacks a year or name field, or returned an error.
    """
    bbox = _check_bbox(bbox)
    layer = _layer_url(url)
    hint = _HINTS["NIFC"]
    key = _cache_key({"url": layer, "bbox": bbox, "year": int(year), "name": name})
    path = reference_dir("nifc") / f"perimeters_{int(year)}_{key}.geojson"
    if path.exists():
        return _geometries(_read_json_file(path))
    info = _arcgis_json(layer, {"f": "json"}, dataset="NIFC", hint=hint)
    year_field = _find_field(info, ("FIRE_YEAR_INT", "FIRE_YEAR", "FIREYEAR"))
    if year_field is None:
        raise ReferenceDataError(f"NIFC layer {layer} has no fire year field; its fields are {_field_names(info)}.")
    yname = year_field["name"]
    where = f"{yname}={_sql_text(str(int(year)))}" if _is_text_field(year_field) else f"{yname}={int(year)}"
    if name:
        name_field = _find_field(info, ("INCIDENT", "poly_IncidentName", "IncidentName", "FIRE_NAME"))
        if name_field is None:
            raise ReferenceDataError(f"NIFC layer {layer} has no incident name field; fields: {_field_names(info)}.")
        where += f" AND UPPER({name_field['name']}) LIKE {_sql_text('%' + name.strip().upper() + '%')}"
    features = _arcgis_query(layer, info, where=where, bbox=bbox, dataset="NIFC", hint=hint)
    logger.info("NIFC: %d perimeter feature(s) for %s in %d within %s.", len(features), name or "any fire", year, bbox)
    collection = {"type": "FeatureCollection", "features": features}
    if features:
        _write_json_atomic(path, collection)
    return _geometries(collection)


def usfs_reforestation(
    bbox: tuple[float, float, float, float],
    *,
    completed_from: int,
    completed_to: int,
    url: str = USFS_REFORESTATION_URL,
) -> geopandas.GeoDataFrame:
    """USFS FACTS reforestation activity polygons completed in a range of years (restore demo sites).

    Queries the FACTS Silviculture Reforestation MapServer layer for polygons intersecting ``bbox``
    with ``DATE_COMPLETED`` in the calendar years ``completed_from`` to ``completed_to`` (both
    included). The answer is cached under ``reference_dir("usfs_facts")``. This is a source of demo
    sites, not a validation reference: FACTS records that work was done, not that trees survived.

    Unverified against the live service until the network test runs: the build sandbox could not
    reach apps.fs.usda.gov.

    Returns:
        GeoDataFrame in EPSG:4326 with every attribute; date fields are pandas datetimes (UTC).

    Raises:
        ValueError: The year range is reversed.
        ReferenceUnavailableError: The service could not be reached.
        ReferenceDataError: The layer has no ``DATE_COMPLETED`` field, or returned an error.
    """
    import geopandas as gpd
    import pandas as pd

    bbox = _check_bbox(bbox)
    if completed_to < completed_from:
        raise ValueError(f"completed_to ({completed_to}) is before completed_from ({completed_from}).")
    layer = _layer_url(url)
    hint = _HINTS["USFS FACTS"]
    key = _cache_key({"url": layer, "bbox": bbox, "from": int(completed_from), "to": int(completed_to)})
    path = reference_dir("usfs_facts") / f"reforestation_{int(completed_from)}_{int(completed_to)}_{key}.geojson"
    if path.exists():
        collection = _read_json_file(path)
    else:
        info = _arcgis_json(layer, {"f": "json"}, dataset="USFS FACTS", hint=hint)
        field = _find_field(info, ("DATE_COMPLETED",))
        if field is None:
            raise ReferenceDataError(f"USFS layer {layer} has no DATE_COMPLETED field; fields: {_field_names(info)}.")
        fname = field["name"]
        lo, hi = f"{int(completed_from):04d}-01-01", f"{int(completed_to) + 1:04d}-01-01"
        ftype = str(field.get("type", ""))
        if ftype == "esriFieldTypeDate":
            where = f"{fname} >= DATE '{lo}' AND {fname} < DATE '{hi}'"
        elif _is_text_field(field):
            where = f"{fname} >= {_sql_text(lo)} AND {fname} < {_sql_text(hi)}"
        else:
            where = f"{fname} >= {int(completed_from)} AND {fname} <= {int(completed_to)}"
        features = _arcgis_query(layer, info, where=where, bbox=bbox, dataset="USFS FACTS", hint=hint)
        date_fields = [f["name"] for f in info.get("fields") or [] if f.get("type") == "esriFieldTypeDate"]
        collection = {"type": "FeatureCollection", "features": features, "tessera_apps:date_fields": date_fields}
        if features:
            _write_json_atomic(path, collection)
    features = collection.get("features") or []
    if not features:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    for col in collection.get("tessera_apps:date_fields") or []:
        if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
            frame[col] = pd.to_datetime(frame[col], unit="ms", utc=True)
    return frame


# --------------------------------------------------------------------------------------------
# WFS (EFFIS)
# --------------------------------------------------------------------------------------------


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _xml_int(text: str | None) -> int | None:
    """A non-negative integer from an XML attribute or element, else None (``unknown``, missing)."""
    if text is None or not text.strip().isdigit():
        return None
    return int(text.strip())


@dataclasses.dataclass(frozen=True)
class _WFSCapabilities:
    typename: str
    formats: list[str]
    count_default: int | None  # the server's per-response feature cap (WFS 2.0 CountDefault)


def _wfs_capabilities(xml: bytes, layer: str) -> _WFSCapabilities:
    """Read a WFS GetCapabilities document for one layer.

    Returns the full type name of ``layer`` (any namespace prefix), the GetFeature output formats and
    the ``CountDefault`` constraint, the server's per-response feature cap (MapServer sets it from
    ``wfs_maxfeatures``).
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ReferenceDataError(f"EFFIS: GetCapabilities did not return XML ({xml[:120]!r}).") from exc
    names: list[str] = []
    formats: list[str] = []
    caps: list[int] = []
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "FeatureType":
            for child in el:
                if _local(child.tag) == "Name" and child.text:
                    names.append(child.text.strip())
        elif tag == "Parameter" and el.get("name", "").lower() == "outputformat":
            formats.extend(v.text.strip() for v in el.iter() if _local(v.tag) == "Value" and v.text)
        elif tag == "ResultFormat":
            formats.extend(_local(c.tag) for c in el)
        elif tag == "Constraint" and el.get("name", "").lower() == "countdefault":
            values = [_xml_int(v.text) for v in el.iter() if _local(v.tag) in ("DefaultValue", "Value")]
            caps.extend(n for n in values if n)
    wanted = layer.lower()
    for n in names:
        if n.lower() == wanted or n.lower().endswith(":" + wanted):
            return _WFSCapabilities(n, list(dict.fromkeys(formats)), min(caps) if caps else None)
    near = [n for n in names if "ba" in n.lower()][:20] or names[:20]
    raise ReferenceDataError(f"EFFIS: the WFS does not list {layer}; it lists for example {near}.")


def _wfs_error(body: bytes) -> str | None:
    head = body[:2000].lstrip()
    if not head.startswith(b"<") or b"Exception" not in head:
        return None
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return body[:300].decode("utf-8", "replace")
    texts = [el.text.strip() for el in root.iter() if _local(el.tag) == "ExceptionText" and el.text]
    if _local(root.tag) in ("ExceptionReport", "ServiceExceptionReport") or texts:
        return "; ".join(texts) or (root.text or "").strip() or "unknown WFS exception"
    return None


def _parse_wfs_features(body: bytes, *, as_json: bool) -> list[dict[str, Any]]:
    """GeoJSON features from a GetFeature response in GeoJSON or GML (empty for an empty page)."""
    return _parse_wfs_page(body, as_json=as_json)[0]


_GML_MEMBERS = frozenset({"member", "featureMember", "featureMembers"})


def _parse_wfs_page(body: bytes, *, as_json: bool) -> tuple[list[dict[str, Any]], int | None]:
    """GeoJSON features of one GetFeature page, and ``numberMatched`` when the server states it.

    A GML collection with no member element (``numberReturned="0"``, or a self-closing root) is an
    empty page; GDAL cannot read it, so it is recognised here first.

    Raises:
        ReferenceDataError: The service returned an exception report, or a body that is not the
            requested format.
    """
    error = _wfs_error(body)
    if error is not None:
        raise ReferenceDataError(f"EFFIS: the WFS returned an exception: {error}")
    if as_json:
        try:
            obj = json.loads(body)
        except ValueError as exc:
            raise ReferenceDataError(f"EFFIS: GetFeature did not return JSON ({body[:120]!r}).") from exc
        if not isinstance(obj, dict):
            raise ReferenceDataError(f"EFFIS: GetFeature returned JSON that is not an object ({body[:120]!r}).")
        matched = obj.get("numberMatched")
        return list(obj.get("features") or []), matched if isinstance(matched, int) and matched >= 0 else None
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ReferenceDataError(f"EFFIS: GetFeature did not return GML ({body[:120]!r}).") from exc
    matched_gml = _xml_int(root.get("numberMatched"))
    if not any(_local(child.tag) in _GML_MEMBERS for child in root):
        return [], matched_gml
    import io

    import geopandas as gpd

    try:
        frame = gpd.read_file(io.BytesIO(body))
    except Exception as exc:  # GDAL and pyogrio raise many types on a body they cannot read
        raise ReferenceDataError(f"EFFIS: could not read the GML GetFeature response ({exc}).") from exc
    features = []
    for _, row in frame.iterrows():
        geom = row.geometry
        props = {}
        for col, value in row.items():
            if col == frame.geometry.name:
                continue
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif isinstance(value, float) and math.isnan(value):
                value = None
            elif isinstance(value, np.generic):
                value = value.item()
            props[str(col)] = value
        features.append({"type": "Feature", "geometry": None if geom is None else mapping(geom), "properties": props})
    return features, matched_gml


def _swap_xy(geom: BaseGeometry) -> BaseGeometry:
    return shapely.transform(geom, lambda xy: xy[:, ::-1])


def _lonlat_features(features: list[dict[str, Any]], bbox: tuple[float, float, float, float]) -> list[dict[str, Any]]:
    """Swap every feature's axes if the features fit ``bbox`` better as (lat, lon) than as (lon, lat).

    WFS 2.0 serves EPSG:4326 in latitude, longitude order, and servers differ in whether GeoJSON
    output follows it. The features were selected by ``bbox``, so the right order is the one that
    puts them inside it.
    """
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    if not geoms:
        return features
    area = box(*bbox)
    as_is = sum(1 for g in geoms if g.intersects(area))
    swapped = sum(1 for g in geoms if _swap_xy(g).intersects(area))
    if swapped <= as_is:
        return features
    logger.info("EFFIS: features came back in latitude, longitude order; swapping to lon, lat.")
    out = []
    for f in features:
        g = f.get("geometry")
        out.append({**f, "geometry": mapping(_swap_xy(shape(g))) if g else g})
    return out


def _year_of(value: object) -> int | None:
    """Calendar year of a date-like property: ISO or day-first text, a datetime, or epoch milliseconds."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    year = getattr(value, "year", None)
    if isinstance(year, int):
        return year
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(value):
            return None
        if abs(value) > 1e11:  # epoch milliseconds
            import datetime as dt

            return dt.datetime.fromtimestamp(value / 1000.0, tz=dt.UTC).year
        return int(value) if 1900 <= value <= 2100 else None
    text = str(value).strip()
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", text)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"(?:19|20)\d{6}.*", text):
        return int(text[:4])
    return None


def _property(feature: Mapping[str, Any], name: str) -> tuple[bool, object]:
    props = feature.get("properties") or {}
    for k, v in props.items():
        if str(k).upper() == name.upper():
            return True, v
    return False, None


def effis_burnt_areas(
    bbox: tuple[float, float, float, float],
    *,
    year: int,
    url: str = EFFIS_WFS_URL,
    layer: str = EFFIS_LAYER,
    page_size: int = 1000,
) -> list[BaseGeometry]:
    """EFFIS burnt area polygons (Europe, Middle East and North Africa) for one fire year.

    Reads the WFS capabilities to find the full type name of ``layer`` (for example
    ``ms:effis.nrt.ba.poly``), whether GeoJSON output is offered and the server's per-response cap
    (``CountDefault``), then pages through WFS 2.0 GetFeature (``COUNT`` and ``STARTINDEX``) with a
    lon/lat bbox filter, reading GeoJSON if offered and GML otherwise. A page shorter than ``COUNT``
    does not end the paging, because a server may cap responses below ``COUNT`` without saying so:
    paging stops at an empty page, or once ``numberMatched`` features have arrived when the server
    states it. Features are kept when their ``FIREDATE`` falls in ``year`` (the filter runs here, so
    the whole archive in the bbox is paged). The answer is cached under ``reference_dir("effis")``
    only after paging ended that way. EFFIS maps fires of roughly 30 ha and more from MODIS and
    Sentinel-2, so small fires are missing.

    Unverified against the live service until the network test runs: the build sandbox could not
    reach maps.effis.emergency.copernicus.eu. The request format follows the OGC WFS 2.0 standard.

    Raises:
        ReferenceUnavailableError: The service could not be reached.
        ReferenceDataError: The layer is not listed, the service returned an exception, it returned
            the same page twice (it ignores ``STARTINDEX``, so the answer would be incomplete), or
            the features have no ``FIREDATE``.
    """
    bbox = _check_bbox(bbox)
    if page_size < 1:
        raise ValueError("page_size must be at least 1.")
    hint = _HINTS["EFFIS"]
    key = _cache_key({"url": url, "layer": layer, "bbox": bbox, "year": int(year)})
    path = reference_dir("effis") / f"burnt_areas_{int(year)}_{key}.geojson"
    if path.exists():
        return _geometries(_read_json_file(path))
    caps = _wfs_capabilities(
        _fetch(
            url, params={"SERVICE": "WFS", "REQUEST": "GetCapabilities", "VERSION": "2.0.0"}, dataset="EFFIS", hint=hint
        ),
        layer,
    )
    json_format = next((f for f in caps.formats if "json" in f.lower()), None)
    count = page_size if caps.count_default is None else min(page_size, caps.count_default)
    west, south, east, north = bbox
    features: list[dict[str, Any]] = []
    previous: str | None = None
    for _ in range(_MAX_PAGES):
        params: dict[str, object] = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": caps.typename,
            "BBOX": f"{south},{west},{north},{east},urn:ogc:def:crs:EPSG::4326",
            "SRSNAME": "urn:ogc:def:crs:EPSG::4326",
            "COUNT": count,
            "STARTINDEX": len(features),
        }
        if json_format:
            params["OUTPUTFORMAT"] = json_format
        body = _fetch(url, params=params, dataset="EFFIS", hint=hint)
        batch, matched = _parse_wfs_page(body, as_json=json_format is not None)
        if not batch:
            break
        signature = json.dumps(batch, sort_keys=True, default=str)
        if signature == previous:
            raise ReferenceDataError(
                f"EFFIS: {_host(url)} returned the same page twice at STARTINDEX {len(features)}; it ignores "
                f"paging, so the answer would be incomplete. Narrow the bbox, or: {hint}"
            )
        previous = signature
        features.extend(batch)
        if matched is not None and len(features) >= matched:
            break
    else:
        raise ReferenceDataError(f"EFFIS: more than {_MAX_PAGES} pages; narrow the bbox.")
    features = _lonlat_features(features, bbox)
    if features and not any(_property(f, "FIREDATE")[0] for f in features):
        names = sorted((features[0].get("properties") or {}).keys())
        raise ReferenceDataError(f"EFFIS features have no FIREDATE property; they have {names}.")
    kept = [f for f in features if _year_of(_property(f, "FIREDATE")[1]) == int(year)]
    logger.info("EFFIS: %d of %d burnt area(s) in %s have a FIREDATE in %d.", len(kept), len(features), bbox, year)
    collection = {"type": "FeatureCollection", "features": kept}
    if features:
        _write_json_atomic(path, collection)
    return _geometries(collection)


# --------------------------------------------------------------------------------------------
# Hansen Global Forest Change
# --------------------------------------------------------------------------------------------


def hansen_tile(lon: float, lat: float) -> str:
    """Hansen tile holding a point, named by its top-left corner, for example ``"40N_120W"``.

    LAT = ceil(lat / 10) * 10 and LON = floor(lon / 10) * 10, so a point on a tile edge belongs to the
    tile north and east of it on longitude, and to the tile whose top edge it lies on for latitude.
    """
    if not (math.isfinite(lon) and math.isfinite(lat) and -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        raise ValueError(f"({lon}, {lat}) is not a lon/lat point.")
    top = math.ceil(lat / 10.0) * 10
    left = min(math.floor(lon / 10.0) * 10, 170)
    return _tile_name(top, left)


def _tile_name(top: int, left: int) -> str:
    return f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"


def hansen_tiles(bbox: tuple[float, float, float, float]) -> list[str]:
    """Hansen tiles a lon/lat bbox overlaps with positive area, north to south then west to east.

    A bbox edge lying exactly on a tile edge does not pull in the neighbouring tile. A bbox with zero
    width or height uses the point rule of :func:`hansen_tile` on that axis.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must be (west, south, east, north); got {bbox!r}.")
    west, south, east, north = (float(v) for v in bbox)
    if not (-180.0 <= west <= east <= 180.0 and -90.0 <= south <= north <= 90.0):
        raise ValueError(f"bbox {bbox!r} must be lon/lat (west, south, east, north).")
    if south < north:
        tops = range(math.floor(south / 10.0) * 10 + 10, math.ceil(north / 10.0) * 10 + 1, 10)
    else:
        tops = range(math.ceil(south / 10.0) * 10, math.ceil(south / 10.0) * 10 + 1, 10)
    if west < east:
        lefts = range(math.floor(west / 10.0) * 10, math.ceil(east / 10.0) * 10, 10)
    else:
        first = min(math.floor(west / 10.0) * 10, 170)
        lefts = range(first, first + 1, 10)
    return [_tile_name(t, min(lon, 170)) for t in sorted(tops, reverse=True) for lon in dict.fromkeys(lefts)]


def _hansen_last_year(version: str) -> int:
    m = re.match(r"GFC-(\d{4})-v", version)
    if not m:
        raise ValueError(f"Hansen version {version!r} is not like 'GFC-2025-v1.13'.")
    return int(m.group(1))


def _hansen_bounds(plan: ZonePlan) -> tuple[float, float, float, float]:
    """The plan window's lon/lat bounds plus a margin, snapped outward to the Hansen pixel grid."""
    west, south, east, north = plan_lonlat_bounds(plan, margin_deg=_HANSEN_MARGIN_DEG)
    r = HANSEN_RES_DEG
    return (
        round(math.floor(west / r + 1e-9) * r, 6),
        round(math.floor(south / r + 1e-9) * r, 6),
        round(math.ceil(east / r - 1e-9) * r, 6),
        round(math.ceil(north / r - 1e-9) * r, 6),
    )


def _paste_tile(ds: rasterio.io.DatasetReader, mosaic: np.ndarray, bounds: tuple[float, float, float, float]) -> int:
    """Copy the part of one Hansen tile inside ``bounds`` into the mosaic. Returns pixels copied."""
    west, south, east, north = bounds
    t = ds.transform
    r = HANSEN_RES_DEG
    if t.b != 0 or t.d != 0 or not math.isclose(t.a, r) or not math.isclose(-t.e, r):
        raise ReferenceDataError(f"Hansen tile {ds.name} has transform {tuple(t)[:6]}; expected {r} degree pixels.")
    b = ds.bounds
    iw, ie = max(west, b.left), min(east, b.right)
    is_, in_ = max(south, b.bottom), min(north, b.top)
    if iw >= ie or is_ >= in_:
        return 0
    col0, col1 = round((iw - t.c) / r), round((ie - t.c) / r)
    row0, row1 = round((t.f - in_) / r), round((t.f - is_) / r)
    if col1 <= col0 or row1 <= row0:
        return 0
    data = ds.read(1, window=RasterWindow(col0, row0, col1 - col0, row1 - row0))
    dc, dr = round((iw - west) / r), round((north - in_) / r)
    mosaic[dr : dr + data.shape[0], dc : dc + data.shape[1]] = data
    return int(data.size)


def _hansen_cache_file(layer: str, version: str, bounds: tuple[float, float, float, float], url_template: str) -> Path:
    """Build (once) and return the cached EPSG:4326 window of one Hansen layer."""
    key = _cache_key({"template": url_template, "version": version, "layer": layer, "bounds": bounds})
    path = reference_dir("hansen") / version / f"{layer}_{key}.tif"
    if path.exists():
        return path
    west, south, east, north = bounds
    width = round((east - west) / HANSEN_RES_DEG)
    height = round((north - south) / HANSEN_RES_DEG)
    mosaic = np.zeros((height, width), dtype=np.uint8)
    hint = _HINTS["Hansen GFC"].format(version=version)
    for tile in hansen_tiles(bounds):
        src = url_template.format(version=version, layer=layer, tile=tile)
        if _is_remote(src):
            if not _exists(src, dataset="Hansen GFC", hint=hint):
                logger.info("Hansen %s tile %s does not exist (ocean); read as 0.", layer, tile)
                continue
        elif not Path(_gdal_path(src)).exists():
            logger.info("Hansen %s tile %s not found at %s; read as 0.", layer, tile, src)
            continue
        with rasterio.Env(**gdal_options()), _open_raster(src, dataset="Hansen GFC", hint=hint) as ds:
            try:
                _paste_tile(ds, mosaic, bounds)
            except rasterio.errors.RasterioIOError as exc:
                if _is_remote(src):
                    raise _remote_read_error(src, "Hansen GFC", hint, exc) from exc
                raise
    with atomic_path(path) as tmp:
        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_origin(west, north, HANSEN_RES_DEG, HANSEN_RES_DEG),
            "compress": "deflate",
            "tiled": width >= 256 and height >= 256,
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(mosaic, 1)
            dst.update_tags(hansen_version=version, layer=layer, bounds=json.dumps(bounds))
    return path


def hansen_layer(
    plan: ZonePlan,
    layer: str,
    *,
    version: str = HANSEN_VERSION,
    resampling: Resampling = Resampling.nearest,
    url_template: str = HANSEN_URL_TEMPLATE,
) -> np.ndarray:
    """One raw Hansen GFC layer on a zone plan's grid (uint8, the whole window, not masked).

    The tiles the plan touches are read as windows through ``/vsicurl/`` (never downloaded whole),
    mosaicked in EPSG:4326, cached under ``reference_dir("hansen")`` per (version, layer, snapped
    bbox), then reprojected onto the 10 m grid. A missing tile (HTTP 404, open ocean) reads as 0.

    Args:
        plan: The zone plan.
        layer: ``treecover2000`` (percent canopy cover in 2000), ``lossyear`` (0 none, N means
            2000 + N), ``gain`` (2000 to 2012) or ``datamask`` (0 no data, 1 land, 2 water).
        version: GFC version, for example ``GFC-2025-v1.13``.
        resampling: From 30 m to 10 m; nearest keeps the codes intact.
        url_template: Tile URL or local path with ``{version}``, ``{layer}`` and ``{tile}``.

    Raises:
        ValueError: Unknown layer.
        ReferenceUnavailableError: A tile host could not be reached.
    """
    if layer not in HANSEN_LAYERS:
        raise ValueError(f"Hansen layer {layer!r} is not one of {HANSEN_LAYERS} (first and last are multi-band).")
    bounds = _hansen_bounds(plan)
    path = _hansen_cache_file(layer, version, bounds, url_template)
    values, _covered = _raster_on_plan(path, plan, band=1, resampling=resampling)
    return values


def _check_hansen_year(year: int, version: str, what: str) -> None:
    last = _hansen_last_year(version)
    if not 2000 <= year <= last:
        raise ValueError(f"{what} {year} is outside Hansen {version}, which covers 2000 to {last}.")


def hansen_loss(
    plan: ZonePlan,
    year_a: int,
    year_b: int,
    *,
    version: str = HANSEN_VERSION,
    url_template: str = HANSEN_URL_TEMPLATE,
) -> np.ndarray:
    """Hansen forest loss after ``year_a`` up to and including ``year_b`` (survey S13).

    True where ``lossyear`` is in ``(year_a - 2000, year_b - 2000]``, which matches a conversion
    between the ``year_a`` and ``year_b`` embeddings. Hansen is itself a 30 m model, so disagreement
    at patch edges is expected.

    Raises:
        ValueError: ``year_a >= year_b`` or either year is outside the version's range.
    """
    if year_a >= year_b:
        raise ValueError(f"year_a ({year_a}) must be before year_b ({year_b}).")
    _check_hansen_year(year_a, version, "year_a")
    _check_hansen_year(year_b, version, "year_b")
    loss = hansen_layer(plan, "lossyear", version=version, url_template=url_template)
    return (loss > year_a - 2000) & (loss <= year_b - 2000) & plan.mask


def hansen_valid(
    plan: ZonePlan, *, version: str = HANSEN_VERSION, url_template: str = HANSEN_URL_TEMPLATE
) -> np.ndarray:
    """Hansen coverage: ``datamask == 1`` (mapped land). Water (2), no data (0) and missing tiles are False."""
    mask = hansen_layer(plan, "datamask", version=version, url_template=url_template)
    return (mask == 1) & plan.mask


def hansen_forest(
    plan: ZonePlan,
    year: int,
    *,
    threshold: int = HANSEN_FOREST_THRESHOLD,
    version: str = HANSEN_VERSION,
    url_template: str = HANSEN_URL_TEMPLATE,
) -> np.ndarray:
    """A forest proxy for ``year``: canopy cover in 2000 at least ``threshold`` percent, not lost since.

    True where ``treecover2000 >= threshold`` and ``lossyear`` is not in 1 to ``year - 2000``. Tree
    cover is not forest: plantations, orchards and tall crops count, and forest that grew after 2000
    does not (Hansen ``gain`` covers 2000 to 2012 only and is ignored). The 30 percent default is the
    threshold most GFC studies use; there is no per-year forest layer in GFC (survey S13).

    Raises:
        ValueError: ``year`` is outside the version's range or ``threshold`` is outside 0 to 100.
    """
    _check_hansen_year(year, version, "year")
    if not 0 <= threshold <= 100:
        raise ValueError(f"threshold {threshold} must be a percentage, 0 to 100.")
    cover = hansen_layer(plan, "treecover2000", version=version, url_template=url_template)
    loss = hansen_layer(plan, "lossyear", version=version, url_template=url_template)
    lost = (loss >= 1) & (loss <= year - 2000)
    return (cover >= threshold) & ~lost & plan.mask


# --------------------------------------------------------------------------------------------
# Global Mangrove Watch
# --------------------------------------------------------------------------------------------


def _gmw_version(version: str, year: int) -> str:
    """Normalise and check a GMW version against the year; never fall back to another version."""
    v = version.strip().lower()
    if v not in GMW_RECORDS:
        raise ValueError(f"GMW version {version!r} is not supported; use one of {sorted(GMW_RECORDS)}.")
    if v == "v3" and year not in GMW_V3_YEARS:
        raise ValueError(
            f"GMW v3 has epochs 1996, 2007 to 2010 and 2015 to 2020 only, not {year}. GMW v4.1.12 is annual "
            f"for {GMW_V4_YEARS[0]} to {GMW_V4_YEARS[-1]}: pass version='v4.1.12' for both years of a comparison, "
            "since versions must not be mixed."
        )
    if v == "v4.1.12" and year not in GMW_V4_YEARS:
        raise ValueError(
            f"GMW v4.1.12 covers {GMW_V4_YEARS[0]} to {GMW_V4_YEARS[-1]}, not {year}; GMW v3 covers "
            f"{', '.join(str(y) for y in GMW_V3_YEARS)}."
        )
    return v


def _year_token(name: str, year: int) -> bool:
    return re.search(rf"(?<!\d){year}(?!\d)", name) is not None


def _zenodo_files(record_id: str, *, hint: str) -> list[tuple[str, str]]:
    """``(key, download url)`` for every file of a Zenodo record, from the REST API."""
    api = ZENODO_RECORD_API.format(record_id=record_id)
    body = _fetch(api, dataset="GMW", hint=hint)
    try:
        record = json.loads(body)
    except ValueError as exc:
        raise ReferenceDataError(f"GMW: Zenodo record {record_id} listing is not JSON ({body[:120]!r}).") from exc
    files = record.get("files") or []
    if isinstance(files, dict):
        entries = files.get("entries") or []
        files = list(entries.values()) if isinstance(entries, dict) else list(entries)
    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        links = f.get("links") or {}
        link = links.get("self") or links.get("content") or links.get("download")
        if key and link:
            out.append((str(key), str(link)))
    if not out:
        raise ReferenceDataError(f"GMW: Zenodo record {record_id} lists no downloadable files.")
    return out


def _cell_names(bbox: tuple[float, float, float, float]) -> set[str]:
    """Names of the 1 degree cells a bbox overlaps, in both top-left and bottom-left conventions."""
    west, south, east, north = bbox
    names = set()
    for lat0 in range(math.floor(south), math.ceil(north)):
        for lon0 in range(math.floor(west), math.ceil(east)):
            ew = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}"
            for lat in (lat0, lat0 + 1):
                names.add(f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{ew}")
    return names


_CELL_TOKEN = re.compile(r"([NS])(\d{2})([EW])(\d{3})", re.IGNORECASE)


_YEAR_TOKEN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_VERSION_TOKEN = re.compile(r"(?<![0-9A-Za-z])v(\d+(?:\.\d+)*)(?![0-9A-Za-z])", re.IGNORECASE)


def _name_years(label: str) -> set[int]:
    """The four-digit years (1900 to 2099) written in a file name or relative path."""
    return {int(y) for y in _YEAR_TOKEN.findall(label)}


def _other_version(label: str, version: str) -> str | None:
    """A version token in ``label`` that names another GMW version, else None.

    ``v4`` and ``v4.1`` fit ``v4.1.12`` (one is a prefix of the other); ``v3`` or ``v4.0`` do not.
    """
    want = tuple(int(p) for p in version.lstrip("v").split("."))
    for m in _VERSION_TOKEN.finditer(label):
        have = tuple(int(p) for p in m.group(1).split("."))
        n = min(len(have), len(want))
        if have[:n] != want[:n]:
            return m.group(0)
    return None


def _other_years(label: str, year: int) -> list[int] | None:
    """The years ``label`` is named for when none of them is ``year``; None when the name fits ``year``.

    A name with no year fits every year, and a name with two or more years fits the span they bound
    (``1985_2025``), so a file is left out only when its name says it holds other years.
    """
    years = _name_years(label)
    if not years or year in years or (len(years) > 1 and min(years) <= year <= max(years)):
        return None
    return sorted(years)


@dataclasses.dataclass(frozen=True)
class _GmwFile:
    """A GMW file to read, and the label its year and version are read from.

    The label is the file name, or the path relative to the source folder so per-year subfolders count.
    """

    path: Path
    label: str


def _pick_gmw_files(
    files: Sequence[tuple[str, str]], version: str, year: int, bbox: tuple[float, float, float, float]
) -> list[tuple[str, str]]:
    """The Zenodo files to download for a GMW version, year and lon/lat bbox.

    v3: the per-year vector file ``gmw_v3_{year}_vec*`` (a zip is preferred). v4.1.12: the raster
    tiles whose 1 degree cell name (``N25W082``, top-left or bottom-left corner; both are tried and
    the tile's real bounds decide) overlaps the bbox. Files named for another version are never
    picked, and neither are tiles named only for other years (``GMW_N25W082_2024_v4.1.12.tif`` when
    2018 is asked), so per-year tiles of one cell are not mixed.

    Raises:
        ReferenceDataError: No file fits, so the record's layout is not the one expected; or the
            tiles over the bbox are all named for other years.
    """
    keys = [k for k, _ in files]
    usable = [f for f in files if _other_version(Path(f[0]).name, version) is None]
    if version == "v3":
        pattern = re.compile(rf"^gmw_v3_{year}_vec", re.IGNORECASE)
        hits = [f for f in usable if pattern.match(Path(f[0]).name)]
        if not hits:
            same_year = [k for k in keys if _year_token(k, year)]
            raise ReferenceDataError(
                f"GMW v3: no file named gmw_v3_{year}_vec* in the record; files for {year}: {same_year or keys[:20]}."
            )
        hits.sort(key=lambda f: (not f[0].lower().endswith(".zip"), f[0]))
        return hits[:1]
    tiles: list[tuple[tuple[str, str], str]] = []
    for f in usable:
        m = _CELL_TOKEN.search(Path(f[0]).name)
        if m is not None and Path(f[0]).suffix.lower() in _RASTER_SUFFIXES:
            tiles.append((f, m.group(0).upper()))
    if not tiles:
        raise ReferenceDataError(
            f"GMW {version}: the record has no per-tile rasters for {version} that this loader recognises "
            f"(files: {keys[:20]})."
        )
    wanted = _cell_names(bbox)
    in_cells = [f for f, cell in tiles if cell in wanted]
    picked = [f for f in in_cells if _other_years(Path(f[0]).name, year) is None]
    if in_cells and not picked:
        raise ReferenceDataError(
            f"GMW {version}: the record's tiles over this area are all named for other years, none for {year} "
            f"({[k for k, _ in in_cells][:20]})."
        )
    return picked


def _local_gmw_files(source: Path, version: str, year: int) -> tuple[list[_GmwFile], bool]:
    """Files of a local GMW copy to read for one version and year, and whether ``source`` is one file.

    In a folder, files named for another version (``_v3`` beside v4.1.12 tiles) or only for other
    years (``_2024`` when 2018 is asked) are left out, so neither versions nor years mix. Year and
    version are read from the path relative to the folder.

    Raises:
        FileNotFoundError: ``source`` does not exist or holds no file for the version and year.
        ReferenceDataError: ``source`` is one file whose name says it is another version or year.
    """
    if not source.exists():
        raise FileNotFoundError(f"GMW source {source} does not exist.")
    if source.is_file():
        other = _other_version(source.name, version)
        if other is not None:
            raise ReferenceDataError(
                f"GMW source {source.name} is named for {other}, not GMW {version}; versions must not be mixed. "
                f"Pass a {version} file, or version= to match the file."
            )
        years = _other_years(source.name, year)
        if years is not None:
            raise ReferenceDataError(f"GMW source {source.name} is named for {years}, not {year}.")
        return [_GmwFile(source, source.name)], True
    found = [
        _GmwFile(p, p.relative_to(source).as_posix())
        for p in sorted(source.rglob("*"))
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in (_VECTOR_SUFFIXES | _RASTER_SUFFIXES | {".zip"})
    ]
    other_version = [f.label for f in found if _other_version(f.label, version) is not None]
    if other_version:
        logger.info(
            "GMW %s: left out %d file(s) named for another version: %s", version, len(other_version), other_version[:10]
        )
    found = [f for f in found if _other_version(f.label, version) is None]
    dated = [f for f in found if _year_token(f.label, year)]
    vectors = [f for f in dated if f.path.suffix.lower() in (_VECTOR_SUFFIXES | {".zip"})]
    if vectors:
        return vectors, False
    left_out = f" Left out as another version: {other_version[:10]}." if other_version else ""
    if version == "v3":
        if not dated:
            raise FileNotFoundError(
                f"No GMW v3 file for {year} in {source}; expected a name like gmw_v3_{year}_vec.shp.{left_out}"
            )
        return dated, False
    rasters = [f for f in found if f.path.suffix.lower() in _RASTER_SUFFIXES]
    other_year = [f.label for f in rasters if _other_years(f.label, year) is not None]
    kept = [f for f in rasters if _other_years(f.label, year) is None]
    if other_year:
        logger.info("GMW %s: left out %d raster(s) named for other years than %d.", version, len(other_year), year)
    if not kept:
        dropped = f" Left out as other years: {other_year[:10]}." if other_year else ""
        raise FileNotFoundError(f"No GMW {version} raster (.tif) for {year} in {source}.{dropped}{left_out}")
    return kept, False


def _vector_targets(path: Path, year: int) -> list[str]:
    """Readable vector dataset names in a path; for a zip, its vector members (by year when several)."""
    if path.suffix.lower() != ".zip":
        return [str(path)]
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if Path(n).suffix.lower() in _VECTOR_SUFFIXES]
    if not members:
        raise ReferenceDataError(f"{path} holds no vector file (.shp, .gpkg, .geojson).")
    if len(members) > 1:
        dated = [n for n in members if _year_token(Path(n).name, year)]
        members = dated or members
    return [f"/vsizip/{path}/{m}" for m in members]


def _gmw_band(file: _GmwFile, version: str, year: int, *, explicit: bool) -> tuple[int, bool]:
    """The band of a GMW raster that holds ``year``, and whether the raster has one band per year.

    A v4.1.12 raster with 41 bands is read at band ``year - 1984``. A single-band v4.1.12 raster is
    read only when its name is dated ``year`` alone, or when it was passed by itself as ``source``
    and its name has no year (then with a warning): any other single-band file could be another
    year's extent.

    Raises:
        ReferenceDataError: The band count does not fit, or a single-band file's year is unknown.
    """
    with rasterio.open(file.path) as ds:
        count = ds.count
    if version == "v4.1.12" and count == len(GMW_V4_YEARS):
        return year - 1984, True
    if count != 1:
        raise ReferenceDataError(
            f"{file.path} has {count} bands; expected 1, or {len(GMW_V4_YEARS)} (one per year) for v4.1.12."
        )
    if version == "v4.1.12":
        years = _name_years(file.label)
        if years == {year}:
            return 1, False
        if not years and explicit:
            logger.warning("%s has one band and no year in its name; it is read as GMW v4.1.12 %d.", file.path, year)
            return 1, False
        named = f"is named for {sorted(years)}" if years else "has no year in its name"
        raise ReferenceDataError(
            f"{file.label} has one band and {named}, so it cannot be told apart from the other years of GMW "
            f"v4.1.12 and is not read as {year}. Keep the 41-band tiles, or only files named for {year} (for "
            f"example GMW_N25W082_{year}_v4.1.12.tif), or pass the one {year} file itself as source=."
        )
    return 1, False


def _check_gmw_rasters(rasters: Sequence[tuple[_GmwFile, int, bool]], version: str) -> None:
    """Refuse GMW rasters that would mix products.

    That is 41-band and single-band files together, or two single-band files for one 1 degree cell.
    """
    multi = [f.label for f, _, per_year in rasters if per_year]
    single = [f for f, _, per_year in rasters if not per_year]
    if multi and single:
        raise ReferenceDataError(
            f"GMW {version}: the source mixes 41-band tiles ({multi[:5]}) with single-band files "
            f"({[f.label for f in single][:5]}); keep one kind so every pixel comes from one product."
        )
    cells: dict[str, str] = {}
    for f in single:
        m = _CELL_TOKEN.search(f.label)
        if m is None:
            continue
        cell = m.group(0).upper()
        if cell in cells:
            raise ReferenceDataError(
                f"GMW {version}: two single-band files for cell {cell} ({cells[cell]} and {f.label}); "
                "keep one per cell."
            )
        cells[cell] = f.label


def _gmw_read(
    plan: ZonePlan, year: int, version: str, files: Sequence[_GmwFile], *, explicit: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """(extent, coverage) on the plan grid from local GMW files.

    For rasters, extent is ``value == 1`` and coverage is the tile footprint: every plan pixel whose
    centre falls inside a tile, whatever its value. A nodata value (often 0 on a 0/1 mask) inside a
    tile therefore counts as covered and not mangrove, so coverage does not shrink to the mangrove
    pixels. For global vectors, coverage is the whole AOI.
    """
    extent = np.zeros(plan.shape, dtype=bool)
    coverage = np.zeros(plan.shape, dtype=bool)
    bbox = plan_lonlat_bounds(plan, margin_deg=0.01)
    geoms: list[BaseGeometry] = []
    vector_read = False
    rasters: list[tuple[_GmwFile, int, bool]] = []
    for file in files:
        if file.path.suffix.lower() in _RASTER_SUFFIXES:
            rasters.append((file, *_gmw_band(file, version, year, explicit=explicit)))
        else:
            vector_read = True
            for target in _vector_targets(file.path, year):
                frame = _read_vector(target, bbox)
                geoms.extend(g for g in frame.geometry if g is not None)
    _check_gmw_rasters(rasters, version)
    for file, band, _per_year in rasters:
        values, valid, inside = _read_on_plan(file.path, plan, band=band, dataset=f"GMW {version}")
        extent |= valid & (values == 1)
        coverage |= inside
    if vector_read:
        extent |= reference_from_geometries(plan, geoms)
        coverage |= plan.mask
    return extent & plan.mask, coverage & plan.mask


def _zenodo_listing(record_id: str, *, hint: str) -> list[tuple[str, str]]:
    """The file list of a Zenodo record, cached as JSON under ``reference_dir("gmw")``.

    Published Zenodo records do not change, so the listing is fetched once and a rerun needs no
    network for files already downloaded.
    """
    path = reference_dir("gmw") / f"zenodo_{record_id}_files.json"
    if path.exists():
        try:
            return [(str(k), str(u)) for k, u in _read_json_file(path)["files"]]
        except (ValueError, KeyError, TypeError):
            logger.warning("Ignoring the unreadable cached Zenodo listing %s.", path)
    files = _zenodo_files(record_id, hint=hint)
    _write_json_atomic(path, {"record": record_id, "files": files})
    return files


def _cached_gmw_v3(folder: Path, year: int) -> Path | None:
    """An already downloaded ``gmw_v3_{year}_vec*`` file in the cache (a zip preferred), if any."""
    if not folder.is_dir():
        return None
    pattern = re.compile(rf"^gmw_v3_{year}_vec", re.IGNORECASE)
    hits = sorted(
        (p for p in folder.iterdir() if p.is_file() and pattern.match(p.name)),
        key=lambda p: (not p.name.lower().endswith(".zip"), p.name),
    )
    return hits[0] if hits else None


def _gmw_paths(plan: ZonePlan, year: int, version: str, source: Path | str | None) -> tuple[list[_GmwFile], bool]:
    """The GMW files to read for a plan, and whether they are one file the user passed."""
    if source is not None:
        return _local_gmw_files(Path(source), version, year)
    record = GMW_RECORDS[version]
    hint = _HINTS["GMW"].format(version=version, record=record)
    folder = reference_dir("gmw") / version
    if version == "v3":
        cached = _cached_gmw_v3(folder, year)
        if cached is not None:
            return [_GmwFile(cached, cached.name)], False
    picked = _pick_gmw_files(_zenodo_listing(record, hint=hint), version, year, plan_lonlat_bounds(plan))
    if not picked:
        logger.info("GMW %s: no tile overlaps the %s plan; no mangrove recorded there.", version, plan.zone)
    paths = [_download(link, folder / Path(key).name, dataset="GMW", hint=hint) for key, link in picked]
    return [_GmwFile(p, p.name) for p in paths], False


def gmw_extent(plan: ZonePlan, year: int, *, version: str = "v3", source: Path | str | None = None) -> np.ndarray:
    """Global Mangrove Watch mangrove extent for one year on a zone plan's grid.

    With ``source`` (a file or folder you downloaded: shapefile, GeoPackage, GeoJSON, zip or
    GeoTIFF tiles) that copy is read. Otherwise the files are found through the Zenodo REST API
    (``files[].key`` and ``files[].links.self``) and cached, with the record's file list, under
    ``reference_dir("gmw")``, so a rerun needs no network:

    - ``v3`` (record 6894273): the per-year vector ``gmw_v3_{year}_vec*``. Epochs 1996, 2007 to 2010
      and 2015 to 2020 only.
    - ``v4.1.12`` (record 21346457): annual rasters 1985 to 2025, band = year - 1984, value 1 is
      mangrove, read from the tiles overlapping the plan.

    A year the version does not cover raises; the loader never switches version on its own, so both
    years of a comparison come from one version. Files named for another version or only for other
    years are never read, and a single-band v4.1.12 raster is read only when its name is dated for
    ``year`` (or it is the one file passed as ``source``), so years and versions do not mix.

    Raises:
        ValueError: Unknown version, or the version does not cover ``year``.
        FileNotFoundError: ``source`` does not exist or holds no file for the version and year.
        ReferenceDataError: ``source`` is named for another version or year, or its rasters would
            mix years or products.
        ReferenceUnavailableError: Zenodo could not be reached.
    """
    v = _gmw_version(version, year)
    files, explicit = _gmw_paths(plan, year, v, source)
    return _gmw_read(plan, year, v, files, explicit=explicit)[0]


def gmw_coverage(plan: ZonePlan, year: int, *, version: str = "v3", source: Path | str | None = None) -> np.ndarray:
    """Plan pixels the GMW data covers: the whole AOI for global vectors, the tile footprints for rasters.

    A raster's footprint does not depend on its values, so a tile that marks non-mangrove as nodata
    still covers those pixels. Score against :func:`gmw_extent` inside this mask. Takes the same
    arguments.
    """
    v = _gmw_version(version, year)
    files, explicit = _gmw_paths(plan, year, v, source)
    return _gmw_read(plan, year, v, files, explicit=explicit)[1]


# --------------------------------------------------------------------------------------------
# Natural Earth coastline
# --------------------------------------------------------------------------------------------


def natural_earth_coastline(
    bbox: tuple[float, float, float, float], *, scale: str = "10m", source: Path | str | None = None
) -> list[BaseGeometry]:
    """Natural Earth coastline lines clipped to a lon/lat bbox, one LineString per piece.

    Downloads ``ne_{scale}_coastline.zip`` once into ``reference_dir("natural_earth")`` unless
    ``source`` names a local copy. Approximate: 10m segments in the Everglades AOI have a median
    length of about 1.2 km, the same scale as a 1 km coastal buffer (survey N9), so "within 1 km of
    the coast" is good to a few hundred metres at best. Pass a bbox widened by the buffer distance so
    coastline just outside the AOI still counts.

    Raises:
        ValueError: Unknown scale or bad bbox.
        ReferenceUnavailableError: The Natural Earth mirror could not be reached.
    """
    bbox = _check_bbox(bbox)
    if scale not in NATURAL_EARTH_SCALES:
        raise ValueError(f"scale {scale!r} must be one of {NATURAL_EARTH_SCALES}.")
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Natural Earth source {path} does not exist.")
    else:
        url = NATURAL_EARTH_URL.format(scale=scale)
        path = _download(
            url,
            reference_dir("natural_earth") / f"ne_{scale}_coastline.zip",
            dataset="Natural Earth",
            hint=_HINTS["Natural Earth"].format(scale=scale),
        )
    frame = _read_vector(path, bbox)
    area = box(*bbox)
    lines: list[BaseGeometry] = []
    for g in frame.geometry:
        if g is None or g.is_empty:
            continue
        clipped = g.intersection(area)
        for part in shapely.get_parts(clipped):
            if part.geom_type == "LineString" and not part.is_empty:
                lines.append(part)
            elif part.geom_type in ("MultiLineString", "GeometryCollection"):
                lines.extend(p for p in shapely.get_parts(part) if p.geom_type == "LineString" and not p.is_empty)
    return lines
