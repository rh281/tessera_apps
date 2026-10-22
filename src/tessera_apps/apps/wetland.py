"""Wetland: given a few confirmed locations of a wetland type, where else is it, and how has its extent moved?

Brief section 4.3, as corrected by the pre-build survey (B5, B6, S2, S15, N9, N10):

- **Years.** ``year_a`` and ``year_b`` (``year_b > year_a``) are checked against ``years_complete``
  before any chunk is read, for every zone the AOI touches and every zone the seeds touch (survey
  S2): the seeds are read in both years, ``year_a`` to set the reference and ``year_b`` to report how
  the seeds themselves score there.
- **Seeds** (:func:`load_seeds`, :func:`place_seeds`). Points or polygons of the wetland type, in
  any CRS, inside or outside the AOI (a vector file, or a points CSV with a WKT column or lon/lat
  columns). A point samples the pixel it falls in; a polygon samples every
  pixel centre inside it (planned like an AOI, :func:`~tessera_apps.io.plan_aoi`). The pixels are
  read with :func:`~tessera_apps.io.read_points`, which reads each store chunk once however many
  seeds share it. Pixels with no data, pixels already sampled by another seed, and pixels whose
  ``year_a`` radar regime is not the AOI's majority regime (survey B5, probed on a sample of AOI
  blocks as the deforest app does) are dropped. Fewer than :data:`MIN_SEED_PIXELS` valid seed
  pixels stop the run before the AOI is streamed (brief section 6).
- **Spatial folds** (survey B6). Seed pixels are grouped into spatial blocks on a grid of
  ``fold_cell_m`` (default :data:`FOLD_CELL_M`, 2 km) cells in the zone's metres
  (:func:`spatial_blocks`): a point joins the cell it falls in, a polygon every cell its pixels fall
  in (so a polygon is never split), and seeds whose pixels touch (8-neighbours, or the same pixel)
  join one block, so a stand split into several features or polygonised pixel by pixel is one
  block, as are nearby stands in one cell. Blocks are assigned to ``folds`` folds at random
  (seeded, :func:`assign_folds`), so no block is in two folds.
- **Score** (:class:`SeedModel`). The reference vector is the mean of the dequantized seed vectors.
  ``cosine`` (the default, survey B6): score = 1 - cosine similarity to the reference, on the raw
  int8 embeddings (scales cancel). ``mahalanobis``: the Mahalanobis distance under a Ledoit-Wolf
  shrunk covariance of the seed vectors, allowed only with :data:`MIN_MAHALANOBIS_PIXELS` or more
  valid seed pixels (with 128 dimensions and fewer than about 129 pixels the sample covariance is
  singular, and an unshrunk inverse is meaningless). Lower is more seed-like.
- **Threshold from held-out seed scores** (survey B6, replacing the brief's in-sample percentile).
  For each fold the reference (and covariance) is fitted on the other folds and the fold's seed
  pixels are scored; the threshold is the ``threshold_pct`` percentile of the pooled held-out
  scores (:func:`choose_threshold`), so about that share of held-out seed pixels falls inside. The
  in-sample percentile is reported next to it. With fewer than two folds (one spatial block, or
  ``folds`` below 2) it falls back to the in-sample percentile with a warning. The final reference
  uses every seed.
- **Candidate extent**, both years, with the same reference and threshold: score <= threshold over
  AOI pixels valid in both years and in the seeds' radar regime in both years (survey B5: a pixel
  that switches regime, or is in the other regime, is not classified). A 3 x 3 opening and
  ``min_area_ha`` run on the mosaic of every zone (:func:`~tessera_apps.apps._common.mosaic_and_clean`).
  Gain is extent in ``year_b`` and not ``year_a``; loss the reverse.
- **Coastal fraction** (brief 4.3 step 5, survey N9): the share of each year's extent within
  ``coast_km`` of the shoreline, from ``shoreline`` (lines or polygon outlines, any CRS) or the
  Natural Earth 10 m coastline for the AOI bbox, which is approximate: its segments here have a
  median length of about 1 km, the same scale as the buffer. Distances are computed on the zone
  grid (:func:`shoreline_distance_m`: the lines rasterized, then
  ``scipy.ndimage.distance_transform_edt`` in metres). When Natural Earth cannot be loaded the run
  warns and completes without the fraction; an identical rerun retries it, and is redone once it
  loads (the coastline is not in the run digest, so the outputs would otherwise stay current).
- **Storm context** (survey N10). An AOI in South Florida gets a note: Irma (September 2017) caused
  about 10,760 ha of mangrove dieback in southwest Florida, and Ian (September 2022) hit the coast
  north of the demo AOI, so a 2020 to 2024 change mixes recovery and storm damage.
- **Validation** (survey S15, B6). Against ``reference`` (a vector file or GeoTIFF, undated, so both
  years are scored against it) or ``reference_source``: ``worldcover`` (default) scores each
  requested year that ESA WorldCover maps (2020 from v100, 2021 from v200) against
  ``reference_classes`` (default 95, mangroves; 90 is herbaceous wetland); ``gmw`` scores each
  requested year one Global Mangrove Watch version covers, never mixing versions. Scores are over
  classified AOI pixels with reference coverage (a ``reference`` raster covers its footprint less
  its nodata, a vector file the whole AOI; WorldCover its non-zero codes), excluding the seed pixels
  and every pixel within ``seed_buffer_m`` of one from both recall and precision (the seeds would
  flatter recall). The report gives recall (the brief's checkpoint metric), precision, IoU and F1
  per year, the recall with the seeds left in, and which WorldCover classes the false positives
  fall in. When 2020 and 2021 are both scored against WorldCover, a caveat says they come from two
  product versions (v100 and v200, different algorithms) and ``version_mix`` gives how far the two
  maps agree. A reference that cannot be loaded, or covers no AOI pixel, warns and the run
  completes (or stops before any chunk read under ``strict_validate``); an identical validating
  rerun then redoes only the validation, from the outputs on disk (as does a validating rerun of a
  run that did not validate).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
import shapely
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from tessera_apps import demo, export, io, metrics, validate
from tessera_apps.apps._common import (
    AppInputError,
    BandMosaic,
    LoadedAOI,
    RunContext,
    StrictValidationError,
    aoi_reference,
    finish_run,
    finish_validation_only,
    load_aoi,
    mosaic_and_clean,
    prepare_run,
    probe_radar_regime,
    regime_name,
    stride_for_preview,
    zone_values_from_raster,
)
from tessera_apps.export import GridArray, OutputPaths
from tessera_apps.io import FIRST_YEAR, HA_PER_PIXEL, LAST_YEAR, AOIPlan, Reader, ZonePlan

logger = logging.getLogger(__name__)

APP = "wetland"
#: Layers the AOI stream reads: the radar counts give each pixel's regime (survey B5).
LAYERS = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
#: Scores the app can run.
SCORES = ("cosine", "mahalanobis")
DEFAULT_SCORE = "cosine"
DEFAULT_THRESHOLD_PCT = 95.0
DEFAULT_FOLDS = 5
DEFAULT_MIN_AREA_HA = 0.1
DEFAULT_COAST_KM = 1.0
DEFAULT_SEED_BUFFER_M = 200.0
DEFAULT_SEED = 0
#: Fewest valid seed pixels a run accepts (brief section 6).
MIN_SEED_PIXELS = 5
#: Fewest valid seed pixels the shrinkage Mahalanobis score accepts (survey B6).
MIN_MAHALANOBIS_PIXELS = 500
#: Side of the grid cells (metres, in the zone's CRS) that block the seeds into spatial folds.
FOLD_CELL_M = 2000.0
#: Suffixes of the seed files read as CSV tables, whose coordinate columns are named by these
#: (GDAL CSV open options, matched in any case); a WKT column wins over them.
CSV_SUFFIXES = frozenset({".csv", ".tsv"})
CSV_X_NAMES = "lon,longitude,x"
CSV_Y_NAMES = "lat,latitude,y"
CSV_COORDINATE_OPTIONS = {"X_POSSIBLE_NAMES": CSV_X_NAMES, "Y_POSSIBLE_NAMES": CSV_Y_NAMES}
#: Most seed pixels a run samples; polygons covering more are refused before any chunk read.
MAX_SEED_PIXELS = 250_000
#: Largest seed polygon (km2) planned; bigger ones are refused before any chunk read.
MAX_SEED_POLYGON_KM2 = 25.0
#: Most store chunks the seeds may touch without lifting the size limit (each is about 8.7 MB a year).
MAX_SEED_CHUNKS = 256
#: AOI blocks probed for the majority radar regime before the seeds are filtered.
REGIME_PROBE_BLOCKS = 8
#: Pixels dequantized at once for the Mahalanobis score (2**16 x 128 float64 is 64 MB).
MAHALANOBIS_STRIP = 2**16
#: Reference sources for ``--reference-source``.
REFERENCE_SOURCES = ("worldcover", "gmw")
DEFAULT_REFERENCE_CLASSES = (validate.WORLDCOVER_MANGROVE,)
#: A lon/lat box over South Florida: an AOI touching it gets the storm note (survey N10).
SOUTH_FLORIDA = (-83.5, 24.0, -79.5, 27.5)
#: Columns of ``wetland_seeds.parquet``.
SEED_COLUMNS = (
    "lon",
    "lat",
    "feature",
    "kind",
    "group",
    "fold",
    "zone",
    "row",
    "col",
    "inside_aoi",
    "valid",
    "regime",
    "used",
    "drop_reason",
    "score_in_sample",
    "score_heldout",
    "valid_year_b",
    "regime_year_b",
    "score_year_b",
)
#: GDAL block cache for the output COG.
WRITE_CACHE_BYTES = 128 * 2**20
VALIDATION_SKIPPED = "Validation skipped, the reference could not be loaded"

# Peak memory model (survey S18), bytes per pixel of the AOI windows unless named otherwise: the
# stream keeps two float32 scores per zone and up to six bool reference masks; the write stage holds
# the five-band float32 mosaic, the cleaned masks and the scores; the coastal stage the rasterized
# lines, the distance transform's float64 distances and int32 feature transform on the padded window.
MEMORY_FIXED = 800_000_000
MEMORY_STREAM_PER_PIXEL = 16
MEMORY_WRITE_PER_PIXEL = 56
MEMORY_COAST_PER_PIXEL = 26
#: Each block-pixel-year term rose from 210 to 360 bytes when reads moved to geotessera (0.2.0): the
#: Rondonia and Everglades demos peaked at 1.92 GB against estimates of 1.51 and 1.58 GB at 210.
MEMORY_PER_BLOCK_PIXEL_YEAR = 360
MEMORY_MARGIN = 1.4

SCOPE_CAVEAT = (
    "Scored over AOI pixels classified in both years (valid, and in the seeds' radar regime in both years) where the "
    "reference has data, leaving out the seed pixels and every pixel within the seed buffer of one, from both recall "
    "and precision (survey B6: the seeds would flatter recall). recall_with_seeds gives the recall with them left in."
)
SAME_PRODUCT_CAVEAT = (
    "The seeds come from the same product as the reference ({product}), so the score is not independent of the "
    "seeds: the buffer leaves the seed stands themselves out, but the reference vector and the threshold were still "
    "learned from that product's own map. Seeds from an independent source (Global Mangrove Watch stands, field "
    "points) give an honest score."
)
WORLDCOVER_CAVEAT = (
    "ESA WorldCover is itself a model: a 10 m land cover map from Sentinel-1 and Sentinel-2, published for 2020 "
    "(v100) and 2021 (v200) only, with its own classification errors, so disagreement at stand edges and between "
    "wetland classes is expected."
)
FALSE_POSITIVE_CAVEAT = (
    "false_positive_classes gives the WorldCover class of every extent pixel outside the reference: for mangroves, "
    "herbaceous wetland (90) and tree cover (10) are the expected confusions, and permanent water (80) points at "
    "tidal edges."
)
GMW_CAVEAT = (
    "Global Mangrove Watch is itself a mapped product (from ALOS PALSAR and Landsat, about 25 to 30 m), so "
    "disagreement at 10 m stand edges is expected. One version scores every year; versions are never mixed."
)
FILE_CAVEAT = (
    "A user-supplied reference is undated: the extent of each year is scored against the same file, so the year_b "
    "score also measures the change since the file was made."
)
FILE_RASTER_COVERAGE_CAVEAT = (
    "Coverage: only AOI pixels inside the reference raster and not on its nodata are scored ({pixels:,} of {aoi:,} "
    "AOI pixels); its non-zero pixels are the wetland type, its zero pixels are not."
)
FILE_VECTOR_COVERAGE_CAVEAT = (
    "Coverage: a vector reference covers the whole AOI ({aoi:,} pixels): inside its polygons is the wetland type, "
    "everywhere else is not."
)
VERSION_MIX_CAVEAT = (
    "The two years are scored against two WorldCover product versions: 2020 against v100 and 2021 against v200, "
    "made with different algorithms (ESA warns that differences between the two maps come from the algorithm change "
    "as well as from real change). So a difference between the 2020 and 2021 scores, or between the two reference "
    "maps, is not by itself app error or change on the ground; version_mix gives how often the two maps agree on the "
    "reference classes over the pixels scored in both years."
)
OTHER_CLASSES_CAVEAT = (
    "The reference classes are {classes}, not mangroves (95) alone: this run measures a different wetland type from "
    "the mangrove default, and the seeds should be of that type too."
)
REDONE_CAVEAT = (
    "This validation was redone from the outputs on disk without reading the store ({reason}). For an AOI in one zone "
    "the scores equal a full run's; across zones a few pixels along the seam may differ, where the mosaic moved them."
)
#: Why a validation was redone, from the earlier run's summary (:func:`_redo_reason`).
REDO_REASONS = {
    "not_validated": "the run that made them did not validate",
    "reference_failed": "the run that made them could not load its reference",
    "missing": "their validation.json was missing or did not match their report",
}
COAST_NOTE = (
    "Natural Earth 10m coastline: approximate. Its segments around the AOI have a median length of about 1 km "
    "(survey N9), the same scale as the coastal buffer, so a pixel near the buffer's edge can fall either side; pass "
    "--shoreline with a finer shoreline (OSM land polygons, GSHHG full resolution) for a sharper fraction."
)


class WetlandError(io.TesseraError):
    """The data cannot answer the question as asked: too few valid seed pixels, or no comparable AOI pixel."""


# --------------------------------------------------------------------------------------------
# Seeds: loading and placing (no chunk read)
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SeedSet:
    """Seeds as loaded, in lon/lat.

    Attributes:
        polygons: ``(feature, geometry)`` for every polygonal seed feature; the feature index is the
            seed's spatial fold group.
        points: ``(feature, lon, lat)`` for every point seed (a MultiPoint gives one row per point).
        path: The file read, or None for a bundled seed set.
        sha256: Hex sha256 of the file's bytes.
        geometry_sha256: Hex sha256 of the loaded geometries in feature order (the run digest covers
            it, so an edit to a sidecar file that changes the seeds changes the digest).
        name: The seed set's name or the file's stem.
        sources: The ``source`` property of each feature that has one (for the same-product caveat).
        demo: The bundled seed set's name, else None.
        dropped_features: Features with no point or polygon (lines, empty geometries).
    """

    polygons: list[tuple[int, BaseGeometry]]
    points: list[tuple[int, float, float]]
    path: Path | None
    sha256: str
    geometry_sha256: str
    name: str
    sources: list[str]
    demo: str | None = None
    dropped_features: int = 0

    @property
    def source(self) -> str:
        """Where the seeds came from: the file path, or ``demo:<name>``."""
        if self.demo is not None:
            return f"demo:{self.demo}"
        return str(self.path) if self.path is not None else self.name

    def record(self) -> dict[str, Any]:
        """The seeds' identity for the report."""
        return {
            "source": self.source,
            **self.digest_record(),
            "polygon_features": len(self.polygons),
            "point_seeds": len(self.points),
            "features_without_point_or_polygon": self.dropped_features,
        }

    def digest_record(self) -> dict[str, Any]:
        """What the run digest covers: the file's name and bytes and the loaded geometry, not where it lies."""
        return {
            "name": self.name,
            "bundled": self.demo,
            "sha256": self.sha256,
            "geometry_sha256": self.geometry_sha256,
        }


def _geometry_digest(polygons: Sequence[tuple[int, BaseGeometry]], points: Sequence[tuple[int, float, float]]) -> str:
    h = hashlib.sha256()
    for feature, geom in polygons:
        g = shapely.transform(shapely.normalize(geom), lambda xy: np.round(xy, 9) + 0.0)
        h.update(f"polygon {feature}\0".encode())
        h.update(bytes(shapely.to_wkb(g, output_dimension=2, byte_order=1, include_srid=False)))
    for feature, lon, lat in points:
        h.update(f"point {feature} {lon:.9f} {lat:.9f}\0".encode())
    return h.hexdigest()


def _seed_parts(geom: BaseGeometry) -> tuple[list[BaseGeometry], list[tuple[float, float]]]:
    """The polygonal parts (repaired) and the points of one feature's geometry."""
    polys: list[BaseGeometry] = []
    points: list[tuple[float, float]] = []
    stack = [geom]
    while stack:
        g = stack.pop()
        if g is None or g.is_empty:
            continue
        kind = g.geom_type
        if kind == "Point":
            points.append((float(g.x), float(g.y)))  # type: ignore[attr-defined]
        elif kind in ("Polygon", "MultiPolygon"):
            fixed = io._polygonal(g if g.is_valid else shapely.make_valid(g))
            if not fixed.is_empty and fixed.area > 0:
                polys.append(fixed)
        elif kind in ("MultiPoint", "GeometryCollection"):
            stack.extend(reversed(list(shapely.get_parts(g))))
    return polys, points


def _bundled_seeds(name: str) -> tuple[Any, str, Path | None]:
    """A bundled seed set as a GeoDataFrame in EPSG:4326, with its sha256 and path (when on disk)."""
    import geopandas as gpd

    key = demo.seeds_name(name)
    resource = files("tessera_apps.data").joinpath(f"{key}.geojson")
    data = resource.read_bytes()
    on_disk = Path(str(resource))
    frame = gpd.GeoDataFrame.from_features(json.loads(data)["features"], crs="EPSG:4326")
    return frame, hashlib.sha256(data).hexdigest(), on_disk if on_disk.is_file() else None


def _has_geometry(frame: Any) -> bool:  # noqa: ANN401
    """Whether ``gpd.read_file`` gave a GeoDataFrame with an active geometry column (not a plain table)."""
    import geopandas as gpd

    if not isinstance(frame, gpd.GeoDataFrame):
        return False
    try:
        frame.geometry  # noqa: B018
    except AttributeError:
        return False
    return True


def load_seeds(spec: str | Path | SeedSet | None, *, aoi_demo: str | None = None) -> SeedSet:
    """Load wetland seeds from a vector file, a bundled seed set, or the AOI demo's default.

    A file (GeoJSON, GeoPackage, shapefile, anything GDAL reads) may be in any CRS; it is
    reprojected to EPSG:4326. A points CSV holds either a WKT column or coordinate columns named
    like :data:`CSV_X_NAMES` and :data:`CSV_Y_NAMES` (``lon,lat``, ``longitude,latitude`` or ``x,y``,
    any case), read as lon/lat when the file declares no CRS. Points (and MultiPoint parts) and
    polygons (repaired) count; lines are ignored. When the file has a ``role`` column and some
    feature has ``role == "seed"`` only those features count. ``"demo"`` (or None, the default)
    loads the seeds bundled for the AOI's demo (``everglades``); a seed set's name
    (``everglades_seeds``) or its demo's name loads it directly, unless a regular file of that name
    exists.

    Args:
        spec: Path, ``"demo"``, a bundled seed set name, an already loaded set (returned as is), or
            None for the AOI demo's seeds.
        aoi_demo: The AOI's demo name, when the AOI is a bundled demo.

    Raises:
        AppInputError: No seeds were given and the AOI has none bundled, the file does not exist or
            cannot be read, has no geometry (a table without WKT or coordinate columns), has no CRS
            and is not in lon/lat, or holds no point or polygon.
    """
    if isinstance(spec, SeedSet):
        return spec
    import geopandas as gpd

    bundled: str | None = None
    path: Path | None = None
    if spec is None or str(spec).strip().lower() == "demo":
        if spec is not None and Path(str(spec)).expanduser().is_file():
            path = Path(str(spec)).expanduser()
        elif aoi_demo is not None and aoi_demo in demo.DEMO_SEEDS:
            bundled = demo.DEMO_SEEDS[aoi_demo]
        else:
            what = "--seeds demo" if spec is not None else "No --seeds was given and"
            raise AppInputError(
                f"{what} the AOI has no bundled seeds (seeds are bundled for "
                f"{', '.join(demo.DEMO_SEEDS)}). Pass --seeds PATH: points or polygons of the wetland type, any CRS."
            )
    else:
        text = str(spec).strip()
        candidate = Path(text).expanduser()
        key = text.lower()
        if not candidate.is_file() and (key in demo.SEED_NAMES or key in demo.DEMO_SEEDS):
            bundled = demo.seeds_name(key)
        else:
            path = candidate
    if bundled is not None:
        frame, sha, on_disk = _bundled_seeds(bundled)
        name = bundled
    else:
        assert path is not None
        if not path.is_file():
            raise AppInputError(
                f"Seeds {str(path)!r} is neither an existing file nor a bundled seed set "
                f"({', '.join(demo.SEED_NAMES)}). Pass a GeoJSON, GeoPackage or shapefile of points or polygons."
            )
        # GDAL's CSV driver builds geometry from a WKT column only, unless told which columns hold
        # the coordinates of a points table.
        options = CSV_COORDINATE_OPTIONS if path.suffix.lower() in CSV_SUFFIXES else {}
        try:
            frame = gpd.read_file(path, **options)
        except Exception as exc:  # pyogrio raises its own DataSourceError and friends
            raise AppInputError(f"Could not read the seeds file {path}: {exc}") from exc
        if not _has_geometry(frame):
            raise AppInputError(
                f"The seeds file {path} has no geometry (GDAL read it as a plain table). Pass a GeoJSON, GeoPackage "
                "or shapefile of points or polygons, or a CSV with a WKT column or coordinate columns named "
                f"{CSV_X_NAMES} and {CSV_Y_NAMES} (lon/lat unless a CRS is given)."
            )
        if frame.crs is None:
            located = frame.geometry.notna() & ~frame.geometry.is_empty
            if located.any():
                west, south, east, north = frame.geometry[located].total_bounds
                if not (-180.0 <= west <= east <= 180.0 and -90.0 <= south <= north <= 90.0):
                    raise AppInputError(
                        f"{path} declares no CRS and its coordinates are not lon/lat; save it with a CRS (a .prj "
                        "file for a shapefile)."
                    )
            logger.warning("%s declares no CRS; its coordinates look like lon/lat, so EPSG:4326 is assumed.", path)
            frame = frame.set_crs("EPSG:4326")
        else:
            frame = frame.to_crs("EPSG:4326")
        sha, on_disk, name = export.file_sha256(path), path, path.stem
    if "role" in frame.columns and (frame["role"] == "seed").any():
        frame = frame[frame["role"] == "seed"]
    polygons: list[tuple[int, BaseGeometry]] = []
    points: list[tuple[int, float, float]] = []
    dropped = 0
    sources: list[str] = []
    for feature, (geom, src) in enumerate(
        zip(frame.geometry, frame["source"] if "source" in frame.columns else [None] * len(frame), strict=True)
    ):
        polys, pts = _seed_parts(geom)
        if not polys and not pts:
            dropped += 1
            continue
        if polys:
            polygons.append((feature, polys[0] if len(polys) == 1 else shapely.union_all(polys)))
        points.extend((feature, lon, lat) for lon, lat in pts)
        if isinstance(src, str) and src:
            sources.append(src)
    where = f"bundled seed set {name}" if bundled is not None else str(path)
    if not polygons and not points:
        raise AppInputError(
            f"The seeds in {where} hold no point or polygon ({len(frame)} features, {dropped} without one). Pass "
            "points or polygons of confirmed locations of the wetland type."
        )
    for _, lon, lat in points:
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise AppInputError(f"A seed point in {where} lies at ({lon}, {lat}), outside lon/lat.")
    return SeedSet(
        polygons=polygons,
        points=points,
        path=on_disk if bundled is not None else path,
        sha256=sha,
        geometry_sha256=_geometry_digest(polygons, points),
        name=name,
        sources=sources,
        demo=bundled,
        dropped_features=dropped,
    )


#: Columns of the seed placement table (:func:`place_seeds`).
PLACEMENT_COLUMNS = (
    "feature",
    "kind",
    "group",
    "lon",
    "lat",
    "zone",
    "row",
    "col",
    "centre_lon",
    "centre_lat",
    "duplicate",
)


def _centres(grid: io.ZoneGrid, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = grid.left + (cols + 0.5) * grid.pixel
    y = grid.top - (rows + 0.5) * grid.pixel
    lon, lat = grid.to_lonlat().transform(x, y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def place_seeds(reader: Reader, seeds: SeedSet, *, max_zones: int = 4, cell_m: float = FOLD_CELL_M) -> pd.DataFrame:
    """Every seed pixel on the store grid, from zone metadata only (no chunk is read).

    A point samples the pixel it falls in (its nominal zone, as :func:`~tessera_apps.io.read_points`
    reads it). A polygon samples every pixel centre inside it: each polygonal part is planned like
    an AOI (:func:`~tessera_apps.io.plan_aoi`, so a pixel on a zone seam is counted once); a part too
    small to hold a pixel centre samples the pixel under its representative point. A pixel already
    sampled by an earlier seed is marked ``duplicate``.

    Groups are the spatial blocks of :func:`spatial_blocks` (the fold unit, survey B6), on cells of
    ``cell_m`` metres of each zone's CRS, named ``cell <zone> <i> <j>`` after a block's first cell.

    Returns:
        One row per seed pixel with :data:`PLACEMENT_COLUMNS`: ``lon``, ``lat`` are where the pixel is
        read (a point, or a pixel centre), ``centre_lon``, ``centre_lat`` the pixel centre (NaN, and row
        and col -1, for a point outside its zone's grid).

    Raises:
        AppInputError: A seed polygon is over :data:`MAX_SEED_POLYGON_KM2`, touches more than
            ``max_zones`` zones, or is spread too thinly to plan.
    """
    out: dict[str, list[Any]] = {k: [] for k in PLACEMENT_COLUMNS if k not in ("group", "duplicate")}
    units: list[int] = []
    cell_x: list[int] = []
    cell_y: list[int] = []
    polygon_unit: dict[int, int] = {}

    def add(
        feature: int,
        kind: str,
        unit: int,
        zone: str,
        rows: np.ndarray,
        cols: np.ndarray,
        lon: Sequence[float] | np.ndarray,
        lat: Sequence[float] | np.ndarray,
        xy: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        grid = reader.grid(zone)
        inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        clon, clat = _centres(grid, rows, cols)
        if xy is None:
            xy = (grid.left + (cols + 0.5) * grid.pixel, grid.top - (rows + 0.5) * grid.pixel)
        n = rows.size
        out["feature"].extend([feature] * n)
        out["kind"].extend([kind] * n)
        out["lon"].extend(np.asarray(lon, dtype=np.float64).tolist())
        out["lat"].extend(np.asarray(lat, dtype=np.float64).tolist())
        out["zone"].extend([zone] * n)
        out["row"].extend(np.where(inside, rows, -1).tolist())
        out["col"].extend(np.where(inside, cols, -1).tolist())
        out["centre_lon"].extend(np.where(inside, clon, np.nan).tolist())
        out["centre_lat"].extend(np.where(inside, clat, np.nan).tolist())
        units.extend([unit] * n)
        cell_x.extend(np.floor(np.asarray(xy[0], dtype=np.float64) / cell_m).astype(np.int64).tolist())
        cell_y.extend(np.floor(np.asarray(xy[1], dtype=np.float64) / cell_m).astype(np.int64).tolist())

    def add_point(feature: int, kind: str, unit: int, lon: float, lat: float) -> None:
        zone = io._point_zone(lon, lat)
        grid = reader.grid(zone)
        x, y = grid.from_lonlat().transform(lon, lat)
        r, c = grid.rowcol(float(x), float(y))
        add(feature, kind, unit, zone, np.array([r]), np.array([c]), [lon], [lat], (np.array([x]), np.array([y])))

    for feature, geom in seeds.polygons:
        unit = polygon_unit.setdefault(feature, len(polygon_unit))
        for part in shapely.get_parts(geom):
            try:
                plan = io.plan_aoi(reader, part, max_zones=max_zones, max_area_km2=MAX_SEED_POLYGON_KM2)
            except io.AOIError as exc:
                # A sliver smaller than a pixel holds no pixel centre: sample the pixel under it.
                if part.area * 111_320.0**2 * math.cos(math.radians(part.centroid.y)) < 4 * io.PIXEL_M**2:
                    pt = part.representative_point()
                    add_point(feature, "polygon", unit, float(pt.x), float(pt.y))
                    continue
                raise AppInputError(f"Seed polygon {feature} cannot be used: {exc}") from exc
            for zp in plan.zones:
                rr, cc = np.nonzero(zp.mask)
                rows, cols = rr + zp.rows.start, cc + zp.cols.start
                plon, plat = _centres(zp.grid, rows, cols)
                add(feature, "polygon", unit, zp.zone, rows, cols, plon, plat)
            if len(out["row"]) > MAX_SEED_PIXELS:
                raise AppInputError(
                    f"The seed polygons cover more than {MAX_SEED_PIXELS:,} pixels; the app samples at most that "
                    "many. Pass smaller polygons, or points."
                )
    for i, (feature, lon, lat) in enumerate(seeds.points):
        add_point(feature, "point", len(polygon_unit) + i, lon, lat)
    table = pd.DataFrame({k: out[k] for k in out})
    table["row"] = table["row"].astype(np.int64)
    table["col"] = table["col"].astype(np.int64)
    on_grid = table["row"].to_numpy() >= 0
    keys = table["zone"].astype(str) + ":" + table["row"].astype(str) + ":" + table["col"].astype(str)
    table["duplicate"] = keys.duplicated().to_numpy() & on_grid
    if len(table) > MAX_SEED_PIXELS:
        raise AppInputError(
            f"The seeds cover {len(table):,} pixels; the app samples at most {MAX_SEED_PIXELS:,}. Pass smaller "
            "polygons, or points."
        )
    groups = spatial_blocks(
        np.asarray(units, dtype=np.int64),
        table["zone"].astype(str).to_numpy(),
        np.asarray(cell_x, dtype=np.int64),
        np.asarray(cell_y, dtype=np.int64),
        table["row"].to_numpy(),
        table["col"].to_numpy(),
    )
    table.insert(PLACEMENT_COLUMNS.index("group"), "group", groups)
    return table


#: Bits of a pixel key given to the column (zone grids are under 2**20 pixels wide).
_KEY_COL_BITS = 20


def spatial_blocks(
    units: np.ndarray,
    zones: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """The spatial block of every seed pixel: the fold unit of survey B6.

    A seed is a unit: one polygon feature (all its pixels) or one point. Two units are linked when
    a pixel of each falls in the same cell (``cell_x``, ``cell_y``: cell indices in the zone's
    metres), or when a pixel of one is the same pixel as, or an 8-neighbour of, a pixel of the other
    on their zone grid (``rows``, ``cols``; -1 off the grid). Blocks are the connected components:
    a polygon is never split, a stand split into touching features (or polygonised pixel by pixel)
    is one block even across a cell edge, and seeds in one cell share a block, as points always did.
    Links do not cross a zone seam except through one polygon's own pixels.

    Args:
        units: Unit id of every pixel (0 to the number of units - 1).
        zones: Zone of every pixel.
        cell_x: Cell column of every pixel.
        cell_y: Cell row of every pixel.
        rows: Zone grid row of every pixel, -1 when off the grid.
        cols: Zone grid column of every pixel, -1 when off the grid.

    Returns:
        str array of block names, ``cell <zone> <i> <j>`` after the block's first cell (sorted by
        zone, then ``i``, then ``j``).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = int(units.size)
    if n == 0:
        return np.zeros(0, dtype=object)
    u = np.asarray(units, dtype=np.int64)
    n_units = int(u.max()) + 1
    cells = pd.DataFrame({"zone": np.asarray(zones).astype(str), "i": cell_x, "j": cell_y})
    cell_code, _ = pd.factorize(cells["zone"] + " " + cells["i"].astype(str) + " " + cells["j"].astype(str))
    src = [u]
    dst = [n_units + cell_code.astype(np.int64)]
    on = np.asarray(rows) >= 0
    for zone in np.unique(cells["zone"].to_numpy()[on]):
        idx = np.flatnonzero(on & (cells["zone"].to_numpy() == zone))
        key = (np.asarray(rows)[idx].astype(np.int64) << _KEY_COL_BITS) + np.asarray(cols)[idx].astype(np.int64)
        uniq, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        rep = u[idx[first]]
        # The same pixel in two units.
        src.append(u[idx])
        dst.append(rep[inverse])
        # Neighbours: half the 8-neighbourhood suffices, links are undirected.
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            target = uniq + (dr << _KEY_COL_BITS) + dc
            pos = np.searchsorted(uniq, target)
            hit = pos < uniq.size
            hit[hit] = uniq[pos[hit]] == target[hit]
            src.append(rep[hit])
            dst.append(rep[pos[hit]])
    a = np.concatenate(src)
    b = np.concatenate(dst)
    size = n_units + int(cell_code.max()) + 1
    # Repeated links are summed by tocsr, so the weights must not wrap (every pixel links its cell).
    graph = coo_matrix((np.ones(a.size, dtype=np.int32), (a, b)), shape=(size, size)).tocsr()
    _, label = connected_components(graph, directed=False)
    block = label[u]
    order = cells.assign(block=block).sort_values(["zone", "i", "j"], kind="stable")
    first_cell = order.groupby("block", sort=False).head(1).set_index("block")
    names = {
        int(b): f"cell {zone} {int(i)} {int(j)}"
        for b, zone, i, j in zip(first_cell.index, first_cell["zone"], first_cell["i"], first_cell["j"], strict=True)
    }
    return np.array([names[int(b)] for b in block], dtype=object)


def seed_chunks(reader: Reader, table: pd.DataFrame) -> set[tuple[str, int, int]]:
    """The store chunks ``(zone, chunk_row, chunk_col)`` the seed pixels on the grid fall in."""
    keys: set[tuple[str, int, int]] = set()
    on = table[table["row"] >= 0]
    for zone, part in on.groupby("zone"):
        ch, cw = reader.chunk_shape(str(zone))
        for r, c in zip(part["row"].to_numpy() // ch, part["col"].to_numpy() // cw, strict=True):
            keys.add((str(zone), int(r), int(c)))
    return keys


# --------------------------------------------------------------------------------------------
# Folds, scores and the threshold
# --------------------------------------------------------------------------------------------


def assign_folds(groups: Sequence[str] | np.ndarray, n_folds: int, seed: int) -> tuple[np.ndarray, int]:
    """Spatial folds: the unique groups shuffled (seeded) and split into ``min(n_folds, groups)`` folds.

    Every pixel of a group (a spatial block, :func:`spatial_blocks`) lands in the same fold, so a
    held-out fold never shares a seed polygon, a grid cell or a touching seed with the seeds its
    reference was fitted on.

    Returns:
        ``(fold per pixel, number of folds)``. With fewer than two folds every pixel is fold 0.
    """
    g = np.asarray(groups).astype(str)
    unique = np.unique(g)
    k = min(int(n_folds), unique.size)
    if k < 2:
        return np.zeros(g.size, dtype=np.int64), max(k, 1) if g.size else 0
    order = np.random.default_rng(seed).permutation(unique)
    fold_of = {str(name): i for i, part in enumerate(np.array_split(order, k)) for name in part}
    return np.array([fold_of[str(x)] for x in g], dtype=np.int64), k


@dataclasses.dataclass(frozen=True, eq=False)
class SeedModel:
    """A fitted seed reference: the score of any vector, lower is more seed-like.

    Attributes:
        kind: ``cosine`` or ``mahalanobis``.
        mean: (D,) float64 mean of the dequantized seed vectors (the reference vector).
        precision: (D, D) float64 inverse of the Ledoit-Wolf shrunk covariance (mahalanobis only).
        shrinkage: The Ledoit-Wolf shrinkage (mahalanobis only).
        n: Seed pixels fitted.
    """

    kind: str
    mean: np.ndarray
    precision: np.ndarray | None
    shrinkage: float | None
    n: int

    def _mahalanobis(self, x: np.ndarray) -> np.ndarray:
        assert self.precision is not None
        d = np.asarray(x, dtype=np.float64) - self.mean
        d2 = np.einsum("ij,ij->i", d @ self.precision, d)
        return np.sqrt(np.maximum(d2, 0.0))

    def score_vectors(self, x: np.ndarray) -> np.ndarray:
        """Scores of dequantized (N, D) vectors as float64; NaN for a non-finite or zero row.

        cosine: 1 - cosine similarity to :attr:`mean`. mahalanobis: the distance
        ``sqrt((x - mean)^T precision (x - mean))``.
        """
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"score_vectors needs an (N, D) array, got shape {arr.shape}.")
        ok = np.isfinite(arr).all(axis=1) & np.any(arr != 0, axis=1)
        out = np.full(arr.shape[0], np.nan, dtype=np.float64)
        if not ok.any():
            return out
        if self.kind == "cosine":
            out[ok] = 1.0 - metrics.cosine_similarity(arr[ok], self.mean).astype(np.float64)
        else:
            out[ok] = self._mahalanobis(arr[ok])
        return out

    def score_window(self, win: io.Window, mask: np.ndarray | None = None) -> np.ndarray:
        """Scores of every pixel of a window as float32, NaN where the pixel is not valid.

        cosine works on the raw int8 embeddings (the per-pixel scale cancels, so it is exact);
        mahalanobis dequantizes :data:`MAHALANOBIS_STRIP` pixels at a time, only inside ``mask``
        when one is given (cosine ignores it: it is cheap on int8).
        """
        if self.kind == "cosine":
            return (np.float32(1.0) - metrics.cosine_to_vector(win, self.mean)).astype(np.float32)
        if win.embeddings is None:
            raise ValueError("score_window needs the embeddings layer.")
        valid = win.valid & (win.scales > 0)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        h, w = win.shape
        out = np.full(h * w, np.nan, dtype=np.float32)
        emb = win.embeddings.reshape(h * w, -1)
        scales = win.scales.reshape(-1)
        idx = np.flatnonzero(valid.reshape(-1))
        for start in range(0, idx.size, MAHALANOBIS_STRIP):
            sel = idx[start : start + MAHALANOBIS_STRIP]
            xs = emb[sel].astype(np.float64) * scales[sel, None].astype(np.float64)
            out[sel] = self._mahalanobis(xs).astype(np.float32)
        return out.reshape(h, w)


def fit_seed_model(x: np.ndarray, kind: str) -> SeedModel:
    """Fit the reference on dequantized seed vectors.

    cosine: the mean vector. mahalanobis: ``sklearn.covariance.LedoitWolf`` (its location is the
    mean, its precision the inverse of the shrunk covariance), never the plain sample covariance,
    whose inverse is meaningless below about 129 pixels in 128 dimensions (survey B6).

    Raises:
        ValueError: Unknown kind, or no vector (two for mahalanobis).
    """
    arr = np.asarray(x, dtype=np.float64)
    if kind not in SCORES:
        raise ValueError(f"score {kind!r} must be one of {SCORES}.")
    if arr.ndim != 2 or arr.shape[0] < (2 if kind == "mahalanobis" else 1):
        need = 2 if kind == "mahalanobis" else 1
        raise ValueError(f"fit_seed_model needs (N, D) vectors with N at least {need}, got shape {arr.shape}.")
    if kind == "cosine":
        return SeedModel(kind=kind, mean=arr.mean(axis=0), precision=None, shrinkage=None, n=arr.shape[0])
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf(store_precision=True, assume_centered=False).fit(arr)
    return SeedModel(
        kind=kind,
        mean=np.asarray(lw.location_, dtype=np.float64),
        precision=np.asarray(lw.precision_, dtype=np.float64),
        shrinkage=float(lw.shrinkage_),
        n=arr.shape[0],
    )


def heldout_scores(x: np.ndarray, folds: np.ndarray, kind: str) -> np.ndarray:
    """Each seed's score under a reference fitted on the other folds only (NaN with one fold).

    A fold whose training part is too small to fit (no other vector, or one for mahalanobis) is
    left NaN.
    """
    arr = np.asarray(x, dtype=np.float64)
    f = np.asarray(folds)
    out = np.full(arr.shape[0], np.nan, dtype=np.float64)
    need = 2 if kind == "mahalanobis" else 1
    for k in np.unique(f):
        test = f == k
        train = ~test
        if int(train.sum()) < need:
            continue
        out[test] = fit_seed_model(arr[train], kind).score_vectors(arr[test])
    return out


@dataclasses.dataclass(frozen=True)
class ThresholdChoice:
    """The extent threshold and how it was set.

    Attributes:
        threshold: Pixels scoring at or below it are candidate extent.
        method: ``heldout`` (the pooled held-out seed scores) or ``in_sample`` (the fallback).
        percentile: ``threshold_pct``.
        heldout_percentile: That percentile of the pooled held-out scores, or None.
        in_sample_percentile: That percentile of the seeds' scores under the final reference.
        folds: Folds used (0 or 1 means the fallback ran).
        heldout_pixels: Seed pixels with a held-out score.
        fallback_reason: Why the fallback ran, or None.
    """

    threshold: float
    method: str
    percentile: float
    heldout_percentile: float | None
    in_sample_percentile: float
    folds: int
    heldout_pixels: int
    fallback_reason: str | None

    def record(self) -> dict[str, Any]:
        """The choice for the report."""
        return {**dataclasses.asdict(self), "fallback_ran": self.method != "heldout"}


def choose_threshold(in_sample: np.ndarray, heldout: np.ndarray, percentile: float, n_folds: int) -> ThresholdChoice:
    """The ``percentile`` of the pooled held-out seed scores, or of the in-sample ones with fewer than two folds."""
    ins = np.asarray(in_sample, dtype=np.float64)
    held = np.asarray(heldout, dtype=np.float64)
    ins = ins[np.isfinite(ins)]
    held = held[np.isfinite(held)]
    in_pct = float(np.percentile(ins, percentile))
    if n_folds >= 2 and held.size:
        value = float(np.percentile(held, percentile))
        return ThresholdChoice(value, "heldout", percentile, value, in_pct, n_folds, int(held.size), None)
    reason = (
        f"fewer than two spatial folds ({n_folds}): the seeds form one spatial block, or --folds is below 2"
        if n_folds < 2
        else "no seed pixel could be scored held out"
    )
    return ThresholdChoice(in_pct, "in_sample", percentile, None, in_pct, n_folds, int(held.size), reason)


# --------------------------------------------------------------------------------------------
# Per-block scoring and the extent
# --------------------------------------------------------------------------------------------

BLOCK_COUNTS = (
    "valid_a",
    "valid_b",
    "valid_both",
    "regime_mismatch",
    "other_regime",
    "other_regime_a",
    "other_regime_b",
)


def _valid(win: io.Window) -> np.ndarray:
    return win.valid & (win.scales > 0)


def score_block(
    wa: io.Window, wb: io.Window, mask: np.ndarray, model: SeedModel, seed_radar: bool
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Both years' scores on one block, survey B5 applied.

    A year's score is set only on ``mask`` pixels valid that year whose radar regime is the seeds'
    regime, so a pixel that switches regime between the years, or is in the other regime, is never
    classified in both years.

    Returns:
        ``(score_a, score_b, counts)``: float32 (NaN where unset) and the counts in :data:`BLOCK_COUNTS`.
    """
    m = np.asarray(mask, dtype=bool)
    va = _valid(wa) & m
    vb = _valid(wb) & m
    ra, rb = wa.radar_present, wb.radar_present
    ok_a = va & (ra == seed_radar)
    ok_b = vb & (rb == seed_radar)
    nan = np.float32(np.nan)
    score_a = np.where(ok_a, model.score_window(wa, ok_a), nan).astype(np.float32)
    score_b = np.where(ok_b, model.score_window(wb, ok_b), nan).astype(np.float32)
    both = va & vb
    same = ra == rb
    counts = {
        "valid_a": int(va.sum()),
        "valid_b": int(vb.sum()),
        "valid_both": int(both.sum()),
        "regime_mismatch": int((both & ~same).sum()),
        "other_regime": int((both & same & (ra != seed_radar)).sum()),
        "other_regime_a": int((va & ~ok_a).sum()),
        "other_regime_b": int((vb & ~ok_b).sum()),
    }
    return score_a, score_b, counts


def candidate_extent(score: np.ndarray, threshold: float, classified: np.ndarray) -> np.ndarray:
    """Classified pixels scoring at or below the threshold."""
    s = np.asarray(score)
    with np.errstate(invalid="ignore"):
        return np.asarray(classified, dtype=bool) & np.isfinite(s) & (s <= threshold)


def change_map(extent_a: np.ndarray, extent_b: np.ndarray, classified: np.ndarray) -> np.ndarray:
    """+1 gain (extent in b only), -1 loss (in a only), 0 stable, NaN where not classified (float32)."""
    a = np.asarray(extent_a, dtype=bool)
    b = np.asarray(extent_b, dtype=bool)
    out = (b & ~a).astype(np.float32) - (a & ~b).astype(np.float32)
    return np.where(np.asarray(classified, dtype=bool), out, np.float32(np.nan)).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Shoreline and coastal fraction
# --------------------------------------------------------------------------------------------


def _line_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """The LineStrings of a geometry; polygon outlines count as lines, points are dropped."""
    out: list[BaseGeometry] = []
    stack = [geom]
    while stack:
        g = stack.pop()
        if g is None or g.is_empty:
            continue
        kind = g.geom_type
        if kind == "LineString" or kind == "LinearRing":
            out.append(shapely.LineString(g.coords))
        elif kind in ("Polygon", "MultiPolygon"):
            stack.append(g.boundary)
        elif kind in ("MultiLineString", "GeometryCollection"):
            stack.extend(shapely.get_parts(g))
    return out


def read_shoreline(path: Path | str, bbox: tuple[float, float, float, float] | None = None) -> list[BaseGeometry]:
    """Shoreline lines in lon/lat from a vector file in any CRS (polygon outlines count as shoreline).

    Args:
        path: The file.
        bbox: Read only the features that intersect this lon/lat box (the AOI widened by the coastal
            buffer); a file with none there gives no line, and the coastal fraction is then 0.

    Raises:
        AppInputError: The file cannot be read, or (without ``bbox``) holds no line or polygon.
    """
    p = Path(path)
    try:
        frame = validate._read_vector(p, bbox)
    except Exception as exc:  # pyogrio and validate raise their own errors
        raise AppInputError(f"Could not read the shoreline file {p}: {exc}") from exc
    lines = [part for g in frame.geometry for part in _line_parts(g)]
    if not lines and bbox is None:
        raise AppInputError(f"The shoreline file {p} holds no line or polygon outline.")
    return lines


def shoreline_distance_m(plan: ZonePlan, lines: Sequence[BaseGeometry], max_m: float) -> np.ndarray:
    """Distance in metres from every pixel centre of a zone plan's window to the nearest shoreline pixel.

    The lines (lon/lat) are projected to the zone CRS and rasterized with ``all_touched`` on the
    window padded by ``max_m``, so shoreline just outside the AOI counts; then
    ``scipy.ndimage.distance_transform_edt`` with the pixel size as sampling gives metres (pixel
    centre to the centre of the nearest pixel a line touches, good to about half a pixel).

    Returns:
        float32 array of ``plan.shape``; inf where no line lies within the padded window.
    """
    pixel = plan.grid.pixel
    pad = math.ceil(max_m / pixel) + 2
    h, w = plan.shape
    transform = plan.grid.window_transform(plan.rows.start - pad, plan.cols.start - pad)
    out = np.full((h, w), np.inf, dtype=np.float32)
    if not lines:
        return out
    left, top = transform.c, transform.f
    right, bottom = left + (w + 2 * pad) * pixel, top - (h + 2 * pad) * pixel
    area = box(left, bottom, right, top)
    to_zone = plan.grid.from_lonlat().transform
    shapes = []
    for line in lines:
        projected = shapely.transform(shapely.segmentize(line, 0.001), to_zone, interleaved=False)
        clipped = projected.intersection(area)
        if not clipped.is_empty:
            shapes.append((clipped, 1))
    if not shapes:
        return out
    burned = rasterize(
        shapes, out_shape=(h + 2 * pad, w + 2 * pad), transform=transform, all_touched=True, fill=0, dtype="uint8"
    )
    if not burned.any():
        return out
    dist = ndimage.distance_transform_edt(burned == 0, sampling=pixel)
    return np.asarray(dist[pad : pad + h, pad : pad + w], dtype=np.float32)


def coastal_fraction(extent: Mapping[str, np.ndarray], near: Mapping[str, np.ndarray]) -> float | None:
    """Share of extent pixels within the coastal buffer, pooled over zones; None with no extent."""
    n = sum(int(np.asarray(m, dtype=bool).sum()) for m in extent.values())
    if n == 0:
        return None
    k = sum(int((np.asarray(extent[z], dtype=bool) & np.asarray(near[z], dtype=bool)).sum()) for z in extent)
    return k / n


def _median_segment_m(lines: Sequence[BaseGeometry]) -> float | None:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    lengths: list[float] = []
    for line in lines:
        xy = np.asarray(line.coords)
        if len(xy) >= 2:
            _, _, d = geod.inv(xy[:-1, 0], xy[:-1, 1], xy[1:, 0], xy[1:, 1])
            lengths.extend(float(v) for v in np.atleast_1d(d))
    return float(np.median(lengths)) if lengths else None


def _lines_length_km(lines: Sequence[BaseGeometry]) -> float:
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    return sum(float(geod.geometry_length(g)) for g in lines) / 1000.0


# --------------------------------------------------------------------------------------------
# Seed exclusion for validation
# --------------------------------------------------------------------------------------------


def seed_exclusion(plan: ZonePlan, centres_lonlat: np.ndarray, buffer_m: float) -> np.ndarray:
    """AOI pixels within ``buffer_m`` of a seed pixel centre, on one zone plan's grid.

    Each seed pixel centre (lon/lat) is projected into this zone's CRS and marked on the window
    padded by the buffer, so a seed just across a zone seam or outside the AOI still excludes its
    neighbours; ``distance_transform_edt`` in metres then grows the marks. ``buffer_m`` 0 excludes
    the seed pixels alone.

    Returns:
        bool array of ``plan.shape``, False outside ``plan.mask``.
    """
    pixel = plan.grid.pixel
    pad = math.ceil(buffer_m / pixel) + 1
    h, w = plan.shape
    out = np.zeros((h, w), dtype=bool)
    pts = np.asarray(centres_lonlat, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.size == 0:
        return out
    x, y = plan.grid.from_lonlat().transform(pts[:, 0], pts[:, 1])
    left = plan.grid.left + (plan.cols.start - pad) * pixel
    top = plan.grid.top - (plan.rows.start - pad) * pixel
    rr = np.floor((top - np.asarray(y)) / pixel).astype(np.int64)
    cc = np.floor((np.asarray(x) - left) / pixel).astype(np.int64)
    hh, ww = h + 2 * pad, w + 2 * pad
    keep = (rr >= 0) & (rr < hh) & (cc >= 0) & (cc < ww)
    if not keep.any():
        return out
    marks = np.zeros((hh, ww), dtype=bool)
    marks[rr[keep], cc[keep]] = True
    if buffer_m > 0:
        near = ndimage.distance_transform_edt(~marks, sampling=pixel) <= buffer_m
    else:
        near = marks
    return near[pad : pad + h, pad : pad + w] & plan.mask


# --------------------------------------------------------------------------------------------
# Memory and the storm note
# --------------------------------------------------------------------------------------------


def memory_estimate(plan: AOIPlan, *, block: tuple[int, int], coast_km: float) -> int:
    """Peak memory of a run in bytes, from the plan's windows, the block shape and the coastal buffer (survey S18)."""
    window_px = sum(z.shape[0] * z.shape[1] for z in plan.zones)
    block_px = max((min(block[0], z.shape[0]) * min(block[1], z.shape[1]) for z in plan.zones), default=0)
    stream = MEMORY_STREAM_PER_PIXEL * window_px + MEMORY_PER_BLOCK_PIXEL_YEAR * block_px * 2
    write = MEMORY_WRITE_PER_PIXEL * window_px
    pad = 2 * (math.ceil(coast_km * 1000.0 / io.PIXEL_M) + 2)
    coast = MEMORY_STREAM_PER_PIXEL * window_px + MEMORY_COAST_PER_PIXEL * max(
        ((z.shape[0] + pad) * (z.shape[1] + pad) for z in plan.zones), default=0
    )
    return int(MEMORY_FIXED + MEMORY_MARGIN * max(stream, write, coast))


def storm_note(aoi: BaseGeometry, year_a: int, year_b: int) -> str | None:
    """The survey N10 storm context for an AOI in South Florida, else None."""
    if not aoi.intersects(box(*SOUTH_FLORIDA)):
        return None
    parts = [
        "South Florida storms (survey N10): Hurricane Irma (September 2017) caused about 10,760 ha of mangrove "
        "dieback in southwest Florida (Lagomasino et al. 2021), and Hurricane Ian (September 2022) hit the coast "
        "north of the Everglades demo AOI."
    ]
    if year_a >= 2018:
        parts.append(f"{year_a} is after Irma, so gain from {year_a} may be recovery from its dieback.")
    else:
        parts.append(f"The {year_a} to {year_b} window spans Irma, so loss may be its dieback.")
    spans_storm = year_a < 2018 <= year_b or year_a < 2022 <= year_b
    if year_a < 2022 <= year_b:
        parts.append(f"The window spans Ian, so loss by {year_b} may be its damage.")
    if spans_storm:
        parts.append(f"The {year_a} to {year_b} change therefore mixes recovery, storm damage and threshold noise.")
    else:
        parts.append(
            f"No hurricane named here falls inside {year_a} to {year_b}, so its change is recovery or drift plus "
            "threshold noise."
        )
    if year_b - year_a == 1:
        parts.append(
            "An adjacent-year window like this one shows how much change the method reports when little happened."
        )
    else:
        parts.append(
            "Compare it with an adjacent-year window (for example 2020 to 2021) to see how much change the method "
            "reports when little happened."
        )
    return " ".join(parts)


# --------------------------------------------------------------------------------------------
# The validation reference
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Reference:
    """The validation reference on every AOI zone plan, loaded before any chunk read (or why it was not)."""

    years: list[int] = dataclasses.field(default_factory=list)
    not_covered: list[int] = dataclasses.field(default_factory=list)
    masks: dict[int, dict[str, np.ndarray]] | None = None
    coverage: dict[int, dict[str, np.ndarray]] | None = None
    classes: dict[int, dict[str, np.ndarray]] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None


def scored_years(
    year_a: int, year_b: int, *, reference: Path | None, source: str, gmw_version: str
) -> tuple[list[int], list[int]]:
    """The requested years a reference scores, and those it does not cover.

    A ``reference`` file scores both. WorldCover covers 2020 and 2021; GMW v3 its epochs
    (:data:`~tessera_apps.validate.GMW_V3_YEARS`); GMW v4.1.12 every year 1985 to 2025.
    """
    years = [year_a, year_b]
    if reference is not None:
        return years, []
    if source == "worldcover":
        have: Sequence[int] = tuple(validate.WORLDCOVER_VERSIONS)
    elif gmw_version.strip().lower() == "v3":
        have = validate.GMW_V3_YEARS
    else:
        have = validate.GMW_V4_YEARS
    return [y for y in years if y in have], [y for y in years if y not in have]


def _same_product(seeds: SeedSet, source: str | None) -> str | None:
    """The product the seeds and the reference share, from the seeds' ``source`` properties."""
    text = " ".join(seeds.sources).lower()
    if source == "worldcover" and "worldcover" in text:
        return "ESA WorldCover"
    if source == "gmw" and ("gmw" in text or "mangrove watch" in text):
        return "Global Mangrove Watch"
    return None


# --------------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WetlandResult:
    """What :func:`run_wetland` did.

    Attributes:
        report: The report (read back from disk when the run was skipped).
        paths: The standard outputs.
        extra_paths: Outputs beyond the standard set, by name (``seeds``).
        skipped: The outputs were current, so nothing was read or written.
        validation_written: ``wetland_validation.json`` belongs to this report.
        validation_redone: The outputs were current except the validation, which was redone from
            them without reading a chunk.
    """

    report: dict[str, Any]
    paths: OutputPaths
    extra_paths: dict[str, Path]
    skipped: bool
    validation_written: bool
    validation_redone: bool = False

    @property
    def summary(self) -> dict[str, Any]:
        """The report's summary block."""
        return dict(self.report.get("summary") or {})


@dataclasses.dataclass(frozen=True)
class _Options:
    seeds: SeedSet
    score: str
    threshold_pct: float
    folds: int
    fold_cell_m: float
    min_area_ha: float
    shoreline: Path | None
    coast_km: float
    reference: Path | None
    reference_source: str
    reference_classes: tuple[int, ...]
    gmw_version: str
    seed_buffer_m: float
    worldcover_template: str
    strict_validate: bool
    max_zones: int
    large: bool


@dataclasses.dataclass
class _State:
    table: pd.DataFrame | None = None
    ref: _Reference = dataclasses.field(default_factory=_Reference)
    shoreline: list[BaseGeometry] | None = None
    shoreline_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    #: The Natural Earth coastline, already loaded (or failed) before the run was prepared.
    coast_preloaded: bool = False


def _check_args(
    *,
    year_a: int,
    year_b: int,
    score: str,
    threshold_pct: float,
    folds: int,
    fold_cell_m: float,
    min_area_ha: float,
    coast_km: float,
    seed_buffer_m: float,
    reference: Path | None,
    reference_source: str,
    reference_classes: Sequence[int],
    gmw_version: str,
    shoreline: Path | None,
    validate_run: bool,
) -> tuple[int, ...]:
    """Refuse arguments the app cannot use, before the store is opened; returns the reference classes.

    Raises:
        AppInputError: Says what to change.
    """
    if year_b <= year_a:
        raise AppInputError(f"year_b ({year_b}) must be after year_a ({year_a}).")
    if year_a < FIRST_YEAR or year_b > LAST_YEAR:
        raise AppInputError(f"Years must lie in {FIRST_YEAR} to {LAST_YEAR}; got year_a {year_a}, year_b {year_b}.")
    if score not in SCORES:
        raise AppInputError(f"score {score!r} must be one of {', '.join(SCORES)}.")
    if not (math.isfinite(threshold_pct) and 0.0 < threshold_pct <= 100.0):
        raise AppInputError(f"threshold_pct ({threshold_pct}) must lie in (0, 100].")
    if int(folds) != folds or folds < 1:
        raise AppInputError(f"folds ({folds}) must be a whole number, 1 or more (1 sets the threshold in sample).")
    if not (math.isfinite(fold_cell_m) and fold_cell_m >= io.PIXEL_M):
        raise AppInputError(f"fold_cell_m ({fold_cell_m}) must be at least the {io.PIXEL_M:g} m pixel.")
    if not (math.isfinite(min_area_ha) and min_area_ha >= 0.0):
        raise AppInputError(f"min_area_ha ({min_area_ha}) must be zero or more.")
    if not (math.isfinite(coast_km) and coast_km > 0.0):
        raise AppInputError(f"coast_km ({coast_km}) must be above zero.")
    if not (math.isfinite(seed_buffer_m) and 0.0 <= seed_buffer_m <= 10_000.0):
        raise AppInputError(f"seed_buffer_m ({seed_buffer_m}) must lie in 0 to 10,000 m.")
    if shoreline is not None and not shoreline.is_file():
        raise AppInputError(f"Shoreline file {shoreline} does not exist; pass a vector file of lines or polygons.")
    if reference is not None and not reference.is_file():
        raise AppInputError(
            f"Reference file {reference} does not exist; pass a vector file or a GeoTIFF with --reference."
        )
    if reference_source not in REFERENCE_SOURCES:
        raise AppInputError(f"reference_source {reference_source!r} must be one of {', '.join(REFERENCE_SOURCES)}.")
    if gmw_version.strip().lower() not in validate.GMW_RECORDS:
        raise AppInputError(f"gmw_version {gmw_version!r} must be one of {', '.join(sorted(validate.GMW_RECORDS))}.")
    try:
        classes = tuple(validate._check_worldcover_classes(reference_classes))
    except ValueError as exc:
        raise AppInputError(f"{exc} Pass --reference-classes like 95 or 95,90.") from exc
    if classes != DEFAULT_REFERENCE_CLASSES and (reference is not None or reference_source != "worldcover"):
        raise AppInputError(
            "--reference-classes applies to the WorldCover reference only (--reference-source worldcover)."
        )
    if validate_run:
        years, _missing = scored_years(
            year_a, year_b, reference=reference, source=reference_source, gmw_version=gmw_version
        )
        if not years:
            if reference_source == "worldcover":
                raise AppInputError(
                    f"ESA WorldCover maps 2020 (v100) and 2021 (v200) only, so it cannot score {year_a} or {year_b}. "
                    "Include 2020 or 2021 as a year, validate against GMW (--reference-source gmw), or pass "
                    "--reference."
                )
            raise AppInputError(
                f"GMW {gmw_version} does not cover {year_a} or {year_b}. GMW v3 has epochs 1996, 2007 to 2010 and 2015 "
                "to 2020; v4.1.12 is annual 1985 to 2025 (--gmw-version v4.1.12). Versions are never mixed."
            )
    return classes


def run_wetland(
    aoi: str | Path | LoadedAOI,
    year_a: int,
    year_b: int,
    out_dir: Path | str,
    *,
    seeds: str | Path | SeedSet | None = None,
    score: str = DEFAULT_SCORE,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    folds: int = DEFAULT_FOLDS,
    fold_cell_m: float = FOLD_CELL_M,
    min_area_ha: float = DEFAULT_MIN_AREA_HA,
    shoreline: Path | str | None = None,
    coast_km: float = DEFAULT_COAST_KM,
    validate: bool = False,
    reference: Path | str | None = None,
    reference_source: str = "worldcover",
    reference_classes: Sequence[int] = DEFAULT_REFERENCE_CLASSES,
    gmw_version: str = "v3",
    seed_buffer_m: float = DEFAULT_SEED_BUFFER_M,
    worldcover_url_template: str | None = None,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    max_area_km2: float | None = io.MAX_AOI_KM2,
    max_zones: int = 4,
    snapshot_id: str | None = None,
    reader: Reader | None = None,
    block: tuple[int, int] = io.DEFAULT_BLOCK,
    strict_validate: bool = False,
) -> WetlandResult:
    """Map a wetland type in ``year_a`` and ``year_b`` from seeds, and how its extent moved in between.

    Writes to ``out_dir``: ``wetland_<a>_<b>.tif`` (float32 COG, NaN nodata, mosaicked across zones:
    ``score_<a>``, ``score_<b>``, ``extent_<a>``, ``extent_<b>`` and ``change``, which is +1 gain, -1
    loss, 0 stable on pixels classified in both years and NaN elsewhere), ``wetland_summary.parquet``
    and ``.csv`` (one row per year, the change columns repeated so the csv stands alone),
    ``wetland_seeds.parquet`` (every seed pixel: where it is, its fold, scores, whether it was used),
    ``wetland_preview.png``, ``wetland_validation.json`` when validating, and ``wetland_report.json``
    last. A rerun with the same arguments skips unless ``force``; a validating rerun whose outputs are
    current except ``validation.json`` redoes only the validation, from the outputs on disk; and a
    rerun without ``shoreline`` whose outputs lack the coastal fraction (Natural Earth could not be
    loaded) retries Natural Earth first, and is redone when it loads.

    Args:
        aoi: AOI file (any CRS), demo name (``everglades``) or a loaded AOI.
        year_a: First year; the seeds are read for it to set the reference and the threshold.
        year_b: Second year, after ``year_a``.
        out_dir: Output directory.
        seeds: Seeds file (points or polygons, any CRS), ``"demo"``, a bundled seed set name, or None
            for the AOI demo's bundled seeds.
        score: ``cosine`` (default) or ``mahalanobis`` (Ledoit-Wolf, 500 or more seed pixels).
        threshold_pct: Percentile of the held-out seed scores that sets the threshold.
        folds: Spatial folds for the held-out scores; below 2 sets the threshold in sample.
        fold_cell_m: Side of the grid cells that block the seeds into spatial folds
            (:func:`spatial_blocks`): seeds in one cell, or touching, are never split across folds.
        min_area_ha: Extent patches smaller than this are dropped after a 3 x 3 opening.
        shoreline: Vector file of shoreline lines (or land or water polygons) for the coastal
            fraction; the Natural Earth 10 m coastline when None.
        coast_km: Coastal buffer distance.
        validate: Score each year's extent against the reference.
        reference: Vector file or GeoTIFF of the wetland type, instead of ``reference_source``.
        reference_source: ``worldcover`` (2020, 2021) or ``gmw``.
        reference_classes: WorldCover classes that count as the wetland type (95 mangroves, 90
            herbaceous wetland).
        gmw_version: ``v3`` or ``v4.1.12`` for ``reference_source="gmw"``.
        seed_buffer_m: Pixels within this distance of a seed pixel are left out of the validation.
        worldcover_url_template: WorldCover tile URL or local path template;
            ``validate.WORLDCOVER_URL_TEMPLATE`` when None.
        seed: Seed of the fold assignment.
        force: Run even when the outputs are current.
        max_area_km2: AOI size limit; None lifts it (and the seed chunk limit).
        max_zones: Zone group limit, for the AOI and for each seed polygon.
        snapshot_id: Icechunk snapshot to pin.
        reader: Reader to use instead of opening the live store.
        block: Block shape of the streaming reads.
        strict_validate: With ``validate``, stop before any chunk read when the reference cannot be
            loaded, instead of running without validation.

    Returns:
        The report, output paths and whether the run was skipped.

    Raises:
        AppInputError: A bad argument, AOI, seeds, shoreline or reference file, or output directory,
            or seeds too few or too many for the score, before the store is opened or before any
            chunk read.
        AOIError: The AOI is too large, touches too many zones, or covers no pixel.
        YearsNotPublishedError: A year is missing for a zone the AOI or the seeds touch (before any
            chunk read).
        StrictValidationError: ``strict_validate`` is set and the reference cannot be loaded.
        WetlandError: Too few valid seed pixels once read (before the AOI is streamed), or no AOI pixel
            comparable between the years.
    """
    ref_path = None if reference is None else Path(reference)
    shore_path = None if shoreline is None else Path(shoreline)
    template = _worldcover_template(worldcover_url_template)
    classes = _check_args(
        year_a=year_a,
        year_b=year_b,
        score=score,
        threshold_pct=threshold_pct,
        folds=folds,
        fold_cell_m=fold_cell_m,
        min_area_ha=min_area_ha,
        coast_km=coast_km,
        seed_buffer_m=seed_buffer_m,
        reference=ref_path,
        reference_source=reference_source,
        reference_classes=reference_classes,
        gmw_version=gmw_version,
        shoreline=shore_path,
        validate_run=validate,
    )
    loaded = load_aoi(aoi)
    seed_set = load_seeds(seeds, aoi_demo=loaded.demo)
    out = Path(out_dir)
    paths = export.output_paths(out, APP, year_a, year_b)
    extras = {"seeds": out / f"{APP}_seeds.parquet"}
    params = {
        "year_a": year_a,
        "year_b": year_b,
        "seeds": seed_set.digest_record(),
        "score": score,
        "threshold_pct": threshold_pct,
        "folds": int(folds),
        "fold_cell_m": float(fold_cell_m),
        "min_area_ha": min_area_ha,
        "shoreline": None
        if shore_path is None
        else {"file": shore_path.name, "sha256": export.file_sha256(shore_path)},
        "coast_km": coast_km,
        "reference": None if ref_path is None else {"file": ref_path.name, "sha256": export.file_sha256(ref_path)},
        "reference_source": None if ref_path is not None else reference_source,
        "reference_classes": list(classes),
        "gmw_version": gmw_version.strip().lower(),
        "seed_buffer_m": seed_buffer_m,
        "worldcover_url_template": template,
        "block": list(block),
        "method": {
            "min_seed_pixels": MIN_SEED_PIXELS,
            "min_mahalanobis_pixels": MIN_MAHALANOBIS_PIXELS,
            "fold_blocks": "cells and touching seeds",
            "regime_probe_blocks": REGIME_PROBE_BLOCKS,
            "opening": 3,
        },
    }
    opts = _Options(
        seeds=seed_set,
        score=score,
        threshold_pct=float(threshold_pct),
        folds=int(folds),
        fold_cell_m=float(fold_cell_m),
        min_area_ha=float(min_area_ha),
        shoreline=shore_path,
        coast_km=float(coast_km),
        reference=ref_path,
        reference_source=reference_source,
        reference_classes=classes,
        gmw_version=gmw_version.strip().lower(),
        seed_buffer_m=float(seed_buffer_m),
        worldcover_template=template,
        strict_validate=strict_validate,
        max_zones=max_zones,
        large=max_area_km2 is None,
    )
    state = _State()
    if not force and shore_path is None and _coast_failed_before(paths.report):
        # The outputs on disk lack the coastal fraction (survey N9) because Natural Earth could not be
        # loaded; the coastline is not in the digest, so they would count as current for good.
        force = _retry_coastline(loaded.geometry, opts, state)
    ctx = prepare_run(
        APP,
        loaded,
        [year_a, year_b],
        year_a,
        year_b,
        out,
        params,
        validate=validate,
        force=force,
        max_area_km2=max_area_km2,
        max_zones=max_zones,
        snapshot_id=snapshot_id,
        reader=reader,
        seed=seed,
        block=block,
        extra_outputs=extras,
        # The stream reads both years of every block; the radar regime probe reads a fraction of a
        # year. The seed chunks are added by the preflight, once they are placed.
        year_reads=3,
        code_modules=(sys.modules[__name__],),
        memory_estimate=lambda plan: memory_estimate(plan, block=block, coast_km=coast_km),
        allow_validation_only=True,
        preflight=lambda c: _preflight(c, state, opts),
    )
    if ctx is None:
        report = json.loads(paths.report.read_text(encoding="utf-8"))
        return WetlandResult(
            report=report,
            paths=paths,
            extra_paths=extras,
            skipped=True,
            validation_written="validation" in (report.get("outputs") or {}),
        )
    if ctx.validation_only:
        return _redo_validation(ctx, state, opts)
    return _run(ctx, state, opts)


def _worldcover_template(template: str | None) -> str:
    """The WorldCover tile template: the given one, else :data:`validate.WORLDCOVER_URL_TEMPLATE` read now."""
    return validate.WORLDCOVER_URL_TEMPLATE if template is None else template


# --------------------------------------------------------------------------------------------
# Preflight: seeds, shoreline and the validation reference, before any chunk read
# --------------------------------------------------------------------------------------------


def _preflight(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Place the seeds, check their years and size, load the shoreline and the validation reference."""
    table = place_seeds(ctx.reader, opts.seeds, max_zones=opts.max_zones, cell_m=opts.fold_cell_m)
    on_grid = (table["row"] >= 0) & ~table["duplicate"]
    zones = sorted(set(table.loc[table["row"] >= 0, "zone"].astype(str)))
    years = [ctx.year_a, ctx.year_b]
    try:
        io.reader_check_years(ctx.reader, zones, years)
    except io.YearsNotPublishedError as exc:
        for zone in sorted(set(exc.missing) - set(ctx.plan.zone_names)):
            n = int((table["zone"] == zone).sum())
            exc.add_note(
                f"Zone {zone} holds {n} of the {len(table)} seed pixels and the AOI does not touch it; the seeds are "
                f"read for {years[0]} and {years[1]}. Drop those seeds or choose other years."
            )
        raise
    n_pixels = int(on_grid.sum())
    if n_pixels < MIN_SEED_PIXELS:
        raise AppInputError(
            f"The seeds cover {n_pixels} distinct pixels on the store grid; the app needs at least {MIN_SEED_PIXELS} "
            "valid seed pixels (brief section 6). Add points or polygons."
        )
    if opts.score == "mahalanobis" and n_pixels < MIN_MAHALANOBIS_PIXELS:
        raise AppInputError(
            f"--score mahalanobis needs at least {MIN_MAHALANOBIS_PIXELS} valid seed pixels and the seeds cover only "
            f"{n_pixels}: with 128 dimensions the covariance of fewer pixels is singular or barely shrunk, and its "
            "inverse is meaningless (survey B6). Use --score cosine, or add seed polygons."
        )
    chunks = seed_chunks(ctx.reader, table)
    if len(chunks) > MAX_SEED_CHUNKS and not opts.large:
        raise AppInputError(
            f"The seeds fall in {len(chunks)} store chunks, over the {MAX_SEED_CHUNKS} the app reads without "
            "--i-know-this-is-large (each costs about 8.7 MB a year). Keep the seeds closer together."
        )
    seed_bytes = 0
    for zone, _r, _c in chunks:
        ch, cw = ctx.reader.chunk_shape(zone)
        seed_bytes += ch * cw * io.BYTES_PER_PIXEL
    if not ctx.validation_only:
        ctx.estimated_bytes += seed_bytes * len(years)
    state.table = table
    logger.info(
        "%s: %d seed pixels (%d polygon features, %d points) in %d chunks of %s, at most %.2f GB to read for them.",
        APP,
        n_pixels,
        len(opts.seeds.polygons),
        len(opts.seeds.points),
        len(chunks),
        ", ".join(zones),
        seed_bytes * len(years) / 1e9,
    )
    if not ctx.validation_only:
        _load_shoreline(ctx, state, opts)
    _load_reference(ctx, state.ref, opts)


def _coast_bbox(aoi: BaseGeometry, coast_km: float) -> tuple[float, float, float, float]:
    """The AOI's lon/lat bounds widened by ``coast_km`` (and 0.01 degree), for the shoreline read."""
    west, south, east, north = (float(v) for v in aoi.bounds)
    lat = math.radians(max(abs(south), abs(north)))
    dlat = coast_km / 111.32 + 0.01
    dlon = coast_km / (111.32 * max(math.cos(lat), 0.01)) + 0.01
    return (max(west - dlon, -180.0), max(south - dlat, -90.0), min(east + dlon, 180.0), min(north + dlat, 90.0))


def _natural_earth(aoi: BaseGeometry, coast_km: float) -> tuple[list[BaseGeometry] | None, dict[str, Any]]:
    """The Natural Earth 10 m coastline around the AOI and its record, or None and the error."""
    try:
        lines = validate.natural_earth_coastline(_coast_bbox(aoi, coast_km), scale="10m")
    except validate.ReferenceDataError as exc:
        return None, {"source": "natural_earth_10m", "approximate": True, "error": str(exc)}
    return lines, {
        "source": "natural_earth_10m",
        "url": validate.NATURAL_EARTH_URL.format(scale="10m"),
        "approximate": True,
        "note": COAST_NOTE,
    }


def _coast_failed_before(report_path: Path) -> bool:
    """Whether the report on disk records a coastal fraction that failed (Natural Earth not loaded)."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    summary = report.get("summary") if isinstance(report, dict) else None
    coast = summary.get("coastal") if isinstance(summary, dict) else None
    return isinstance(coast, dict) and bool(coast.get("error"))


def _retry_coastline(aoi: BaseGeometry, opts: _Options, state: _State) -> bool:
    """Load the Natural Earth coastline for a rerun whose outputs on disk lack the coastal fraction.

    Returns:
        True when it loads now, so the outputs are not current and the run is redone (the lines are
        kept for it); False when it still fails, so the outputs stay as they are and an otherwise
        identical rerun still skips, without reading the store again for the same gap.
    """
    lines, info = _natural_earth(aoi, opts.coast_km)
    state.shoreline, state.shoreline_info, state.coast_preloaded = lines, info, True
    if lines is None:
        logger.warning(
            "%s: the outputs on disk have no coastal fraction and the Natural Earth coastline still cannot be loaded "
            "(%s); pass --shoreline with a local shoreline file.",
            APP,
            info["error"],
        )
        return False
    logger.info(
        "%s: the outputs on disk have no coastal fraction (Natural Earth could not be loaded then); it loads now, so "
        "the run is redone.",
        APP,
    )
    return True


def _load_shoreline(ctx: RunContext, state: _State, opts: _Options) -> None:
    """The shoreline lines: the ``--shoreline`` file, else Natural Earth (a failure warns, the run goes on)."""
    if opts.shoreline is not None:
        lines = read_shoreline(opts.shoreline, _coast_bbox(ctx.aoi.geometry, opts.coast_km))
        state.shoreline = lines
        state.shoreline_info = {
            "source": "file",
            "file": opts.shoreline.name,
            "sha256": export.file_sha256(opts.shoreline),
            "approximate": False,
        }
        return
    if not state.coast_preloaded:
        state.shoreline, state.shoreline_info = _natural_earth(ctx.aoi.geometry, opts.coast_km)
    if state.shoreline is None:
        ctx.warn(
            "The coastal fraction was not computed: the Natural Earth coastline could not be loaded "
            f"({state.shoreline_info.get('error')})."
        )


#: Suffixes of a ``--reference`` file read as a raster (the same as :func:`aoi_reference`).
RASTER_REFERENCE_SUFFIXES = frozenset({".tif", ".tiff", ".vrt"})


def file_reference(plan_zone: ZonePlan, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """A ``--reference`` file on one zone plan's grid: the wetland mask and the pixels it covers.

    The mask is :func:`~tessera_apps.apps._common.aoi_reference` (a vector file's polygons, or a
    raster's non-zero pixels). A raster covers only the AOI pixels inside its footprint and not on
    its nodata, so the pixels it does not map are left out of the scores instead of counting as
    not-wetland; a vector file covers the whole AOI (outside its polygons is not-wetland).

    Returns:
        ``(mask, coverage)``: bool arrays of ``plan_zone.shape``, False outside the AOI mask.

    Raises:
        AppInputError: The file cannot be read as a reference.
    """
    mask = aoi_reference(plan_zone, path)
    if path.suffix.lower() not in RASTER_REFERENCE_SUFFIXES:
        return mask, plan_zone.mask.copy()
    try:
        _values, covered = validate._raster_on_plan(path, plan_zone, band=1, resampling=Resampling.nearest)
    except (OSError, ValueError, RuntimeError, rasterio.errors.RasterioError) as exc:
        raise AppInputError(
            f"Could not read the reference file {path}: {exc}. Pass a polygon file or a GeoTIFF with --reference."
        ) from exc
    return mask, covered & plan_zone.mask


def _load_file_reference(plan: AOIPlan, ref: _Reference, path: Path) -> None:
    """Fill ``ref`` from a ``--reference`` file, the same mask and coverage for every scored year.

    Raises:
        AppInputError: The file cannot be read.
        validate.ReferenceDataError: The file covers no AOI pixel.
    """
    masks: dict[str, np.ndarray] = {}
    cover: dict[str, np.ndarray] = {}
    for z in plan.zones:
        masks[z.zone], cover[z.zone] = file_reference(z, path)
    raster = path.suffix.lower() in RASTER_REFERENCE_SUFFIXES
    covered = sum(int(c.sum()) for c in cover.values())
    if covered == 0:
        raise validate.ReferenceDataError(
            f"The reference raster {path.name} has no data over the AOI: it does not overlap it, or is nodata there."
        )
    ref.masks = {year: dict(masks) for year in ref.years}
    ref.coverage = {year: dict(cover) for year in ref.years}
    ref.info = {
        "source": "file",
        "file": path.name,
        "sha256": export.file_sha256(path),
        "kind": "raster" if raster else "vector",
        "coverage": "raster footprint less its nodata pixels" if raster else "the whole AOI",
        "coverage_pixels": covered,
    }


def _load_reference(ctx: RunContext, ref: _Reference, opts: _Options) -> None:
    """Load the validation reference on every AOI zone plan: the preflight, before any chunk read.

    A reference with no coverage over the AOI counts as not loaded: a WorldCover or GMW tile that
    is missing reads as no data, so a template that points at no file would otherwise score nothing
    without a word, and a ``--reference`` raster that does not overlap the AOI (or is nodata over
    it) would score every AOI pixel as not-wetland. A raster's coverage is its footprint less its
    nodata pixels (:func:`file_reference`); a vector file covers the whole AOI.

    Raises:
        AppInputError: The ``--reference`` file cannot be read.
        StrictValidationError: ``strict_validate`` is set and the reference cannot be loaded.
    """
    if not ctx.validate:
        return
    plan = ctx.plan
    years, missing = scored_years(
        ctx.year_a, ctx.year_b, reference=opts.reference, source=opts.reference_source, gmw_version=opts.gmw_version
    )
    ref.years, ref.not_covered = years, missing
    try:
        if opts.reference is not None:
            _load_file_reference(plan, ref, opts.reference)
            return
        masks_y: dict[int, dict[str, np.ndarray]] = {}
        cover_y: dict[int, dict[str, np.ndarray]] = {}
        class_y: dict[int, dict[str, np.ndarray]] = {}
        for year in years:
            masks_y[year], cover_y[year], class_y[year] = {}, {}, {}
            for z in plan.zones:
                if opts.reference_source == "worldcover":
                    codes = validate.worldcover_map(z, year, url_template=opts.worldcover_template)
                    masks_y[year][z.zone] = np.isin(codes, opts.reference_classes) & z.mask
                    cover_y[year][z.zone] = (codes != 0) & z.mask
                    class_y[year][z.zone] = np.where(z.mask, codes, np.uint8(0)).astype(np.uint8)
                else:
                    masks_y[year][z.zone] = validate.gmw_extent(z, year, version=opts.gmw_version)
                    cover_y[year][z.zone] = validate.gmw_coverage(z, year, version=opts.gmw_version)
            if not any(m.any() for m in cover_y[year].values()):
                raise validate.ReferenceDataError(
                    f"The {opts.reference_source} reference has no data over the AOI for {year}"
                    + (
                        f" from {opts.worldcover_template}: its tiles were not found there"
                        if opts.reference_source == "worldcover"
                        else ""
                    )
                    + "."
                )
    except validate.ReferenceDataError as exc:
        if opts.strict_validate:
            raise StrictValidationError(
                f"--strict-validate is set and the validation reference could not be loaded: {exc}"
            ) from exc
        ref.error = str(exc)
        logger.warning("Validation will be skipped, the reference could not be loaded: %s", exc)
        return
    ref.masks, ref.coverage = masks_y, cover_y
    if opts.reference_source == "worldcover":
        ref.classes = class_y
        ref.info = {
            "source": "worldcover",
            "dataset": "ESA WorldCover",
            "versions": {str(y): validate.WORLDCOVER_VERSIONS[y] for y in years},
            "url_template": opts.worldcover_template,
            "classes": list(opts.reference_classes),
            "class_names": {str(c): validate.WORLDCOVER_CLASSES[c] for c in opts.reference_classes},
            "coverage": "class code not 0",
        }
    else:
        ref.info = {
            "source": "gmw",
            "dataset": "Global Mangrove Watch",
            "version": opts.gmw_version,
            "zenodo_record": validate.GMW_RECORDS[opts.gmw_version],
        }


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Zone:
    """Both years' scores on one zone plan's grid, filled block by block."""

    plan: ZonePlan
    score_a: np.ndarray
    score_b: np.ndarray


def _stream(ctx: RunContext, model: SeedModel, seed_radar: bool) -> tuple[dict[str, _Zone], dict[str, int]]:
    """One pass over every block of every zone, reading both years at once."""
    a, b = ctx.year_a, ctx.year_b
    totals = dict.fromkeys(BLOCK_COUNTS, 0)
    zones: dict[str, _Zone] = {}
    for zp in ctx.plan.zones:
        st = _Zone(
            plan=zp,
            score_a=np.full(zp.shape, np.nan, dtype=np.float32),
            score_b=np.full(zp.shape, np.nan, dtype=np.float32),
        )
        for blk in io.iter_plan_blocks(ctx.reader, zp, [a, b], layers=LAYERS, block=ctx.block):
            ctx.mark_read([a, b])
            r = slice(blk.rows.start - zp.rows.start, blk.rows.stop - zp.rows.start)
            c = slice(blk.cols.start - zp.cols.start, blk.cols.stop - zp.cols.start)
            sa, sb, counts = score_block(blk.windows[a], blk.windows[b], blk.mask, model, seed_radar)
            st.score_a[r, c] = sa
            st.score_b[r, c] = sb
            for k, v in counts.items():
                totals[k] += v
        zones[zp.zone] = st
    return zones, totals


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def _read_seeds(ctx: RunContext, table: pd.DataFrame, seed_radar: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Read every seed pixel in both years and mark which are used (survey B5 applied to the seeds).

    Returns:
        ``(table, emb_a, emb_b)``: the table with ``valid``, ``regime``, ``used``, ``drop_reason``,
        ``valid_year_b`` and ``regime_year_b`` added, and the dequantized vectors of each year.
    """
    a, b = ctx.year_a, ctx.year_b
    lon = table["lon"].to_numpy(dtype=np.float64)
    lat = table["lat"].to_numpy(dtype=np.float64)
    zones = sorted(set(table.loc[table["row"] >= 0, "zone"].astype(str)))
    pa = io.read_points(ctx.reader, lon, lat, a, with_radar=True)
    ctx.mark_read([a], zones=zones)
    pb = io.read_points(ctx.reader, lon, lat, b, with_radar=True)
    ctx.mark_read([b], zones=zones)
    assert pa.radar_present is not None and pb.radar_present is not None
    out = table.copy()
    dup = out["duplicate"].to_numpy(dtype=bool)
    va, vb = pa.valid & ~dup, pb.valid & ~dup
    out["valid"] = va
    out["regime"] = np.where(va, np.where(pa.radar_present, "radar", "optical"), "no_data")
    used = va & (pa.radar_present == bool(seed_radar))
    out["used"] = used
    out["drop_reason"] = np.where(dup, "duplicate", np.where(used, "", np.where(va, "regime", "no_data")))
    out["valid_year_b"] = vb
    out["regime_year_b"] = np.where(vb, np.where(pb.radar_present, "radar", "optical"), "no_data")
    out["_ok_b"] = vb & (pb.radar_present == bool(seed_radar))
    return out, pa.embeddings, pb.embeddings


def _write_seeds(path: Path, table: pd.DataFrame, aoi: BaseGeometry) -> Path:
    lon = table["lon"].to_numpy(dtype=np.float64)
    lat = table["lat"].to_numpy(dtype=np.float64)
    frame = pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "feature": table["feature"].to_numpy(dtype=np.int64),
            "kind": table["kind"].astype(str).to_numpy(),
            "group": table["group"].astype(str).to_numpy(),
            "fold": table["fold"].to_numpy(dtype=np.int16),
            "zone": table["zone"].astype(str).to_numpy(),
            "row": table["row"].to_numpy(dtype=np.int64),
            "col": table["col"].to_numpy(dtype=np.int64),
            "inside_aoi": shapely.contains_xy(aoi, lon, lat) if len(table) else np.zeros(0, dtype=bool),
            "valid": table["valid"].to_numpy(dtype=bool),
            "regime": table["regime"].astype(str).to_numpy(),
            "used": table["used"].to_numpy(dtype=bool),
            "drop_reason": table["drop_reason"].astype(str).to_numpy(),
            "score_in_sample": table["score_in_sample"].to_numpy(dtype=np.float64),
            "score_heldout": table["score_heldout"].to_numpy(dtype=np.float64),
            "valid_year_b": table["valid_year_b"].to_numpy(dtype=bool),
            "regime_year_b": table["regime_year_b"].astype(str).to_numpy(),
            "score_year_b": table["score_year_b"].to_numpy(dtype=np.float64),
        }
    )
    with export.atomic_path(path) as tmp:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
    return path


def _score_record(opts: _Options, model: SeedModel) -> dict[str, Any]:
    if opts.score == "cosine":
        reason = (
            "The default (survey B6): 1 minus the cosine similarity to the mean of the dequantized seed vectors, on "
            "the raw int8 embeddings."
        )
    else:
        reason = (
            f"Requested, with {model.n} valid seed pixels (at least {MIN_MAHALANOBIS_PIXELS} are required): the "
            "Mahalanobis distance under the Ledoit-Wolf shrunk covariance of the seed vectors (survey B6), shrinkage "
            f"{model.shrinkage:.4f}."
        )
    return {"type": opts.score, "reason": reason, "shrinkage": model.shrinkage, "seed_pixels_fitted": model.n}


def _seed_summary(table: pd.DataFrame, choice: ThresholdChoice, opts: _Options, model: SeedModel) -> dict[str, Any]:
    used = table["used"].to_numpy(dtype=bool)
    reasons = table["drop_reason"].astype(str)
    ins = table["score_in_sample"].to_numpy(dtype=np.float64)
    held = table["score_heldout"].to_numpy(dtype=np.float64)
    sb = table["score_year_b"].to_numpy(dtype=np.float64)
    in_b = np.isfinite(sb) & used
    with np.errstate(invalid="ignore"):
        inside_a = float(np.mean(ins[used] <= choice.threshold)) if used.any() else None
        inside_held = float(np.mean(held[np.isfinite(held)] <= choice.threshold)) if np.isfinite(held).any() else None
        inside_b = float(np.mean(sb[in_b] <= choice.threshold)) if in_b.any() else None
    return {
        "pixels": len(table),
        "pixels_on_grid": int((table["row"] >= 0).sum()),
        "used": int(used.sum()),
        "dropped_no_data": int((reasons == "no_data").sum()),
        "dropped_regime": int((reasons == "regime").sum()),
        "dropped_duplicate": int((reasons == "duplicate").sum()),
        # The spatial blocks the used seed pixels fall in (survey B6): the fold unit.
        "spatial_blocks": int(table.loc[used, "group"].nunique()),
        "fold_cell_m": opts.fold_cell_m,
        "polygon_features_used": int(table.loc[used & (table["kind"] == "polygon").to_numpy(), "feature"].nunique()),
        "folds_requested": opts.folds,
        "folds": choice.folds,
        "polygon_pixels": int((table["kind"] == "polygon").sum()),
        "point_pixels": int((table["kind"] == "point").sum()),
        "inside_aoi": int(table["inside_aoi"].sum()) if "inside_aoi" in table else None,
        "score": model.kind,
        "threshold": choice.threshold,
        "threshold_pct": choice.percentile,
        "in_sample_percentile": choice.in_sample_percentile,
        "heldout_percentile": choice.heldout_percentile,
        "share_in_sample_within_threshold": inside_a,
        "share_heldout_within_threshold": inside_held,
        "pixels_scored_year_b": int(in_b.sum()),
        "share_year_b_within_threshold": inside_b,
    }


def _run(ctx: RunContext, state: _State, opts: _Options) -> WetlandResult:
    year_a, year_b = ctx.year_a, ctx.year_b
    plan = ctx.plan
    assert state.table is not None
    names = [z.zone for z in plan.zones]

    # The AOI's majority radar regime first (cheap), then the seeds in both years, filtered to it.
    t = time.perf_counter()
    probe = probe_radar_regime(ctx, blocks=REGIME_PROBE_BLOCKS)
    seed_radar = probe["majority_radar"]
    if seed_radar is None:
        raise WetlandError(
            f"No AOI pixel has data in both {year_a} and {year_b} with the same radar regime "
            f"({probe['pixels_mismatch']:,} switch regime); choose other years or another AOI."
        )
    t = ctx.lap("regime_probe", t)
    table, emb_a, emb_b = _read_seeds(ctx, state.table, bool(seed_radar))
    t = ctx.lap("seeds_read", t)
    used = table["used"].to_numpy(dtype=bool)
    n_used = int(used.sum())
    reasons = table["drop_reason"].astype(str)
    if n_used < MIN_SEED_PIXELS:
        raise WetlandError(
            f"Only {n_used} seed pixels are usable ({int((reasons == 'no_data').sum())} have no {year_a} data, "
            f"{int((reasons == 'regime').sum())} are not in the AOI's {regime_name(seed_radar)} radar regime, "
            f"{int((reasons == 'duplicate').sum())} repeat a pixel); the app needs at least {MIN_SEED_PIXELS} "
            "(brief section 6). Add seeds, or move them onto land with data."
        )
    if opts.score == "mahalanobis" and n_used < MIN_MAHALANOBIS_PIXELS:
        raise WetlandError(
            f"--score mahalanobis needs at least {MIN_MAHALANOBIS_PIXELS} valid seed pixels and only {n_used} are "
            "usable after dropping pixels with no data or in the other radar regime (survey B6). Use --score cosine, "
            "or add seed polygons."
        )
    if int((reasons == "regime").sum()):
        ctx.warn(
            f"{int((reasons == 'regime').sum())} seed pixels were dropped because their {year_a} radar regime is not "
            f"the AOI's ({regime_name(seed_radar)}, survey B5)."
        )
    if int((reasons == "no_data").sum()):
        ctx.warn(f"{int((reasons == 'no_data').sum())} seed pixels have no {year_a} data and were dropped.")

    x = emb_a[used].astype(np.float64)
    groups = table["group"].astype(str).to_numpy()[used]
    fold_used, n_folds = assign_folds(groups, opts.folds, ctx.seed)
    model = fit_seed_model(x, opts.score)
    in_sample = model.score_vectors(x)
    held = heldout_scores(x, fold_used, opts.score) if n_folds >= 2 else np.full(n_used, np.nan)
    choice = choose_threshold(in_sample, held, opts.threshold_pct, n_folds)
    if choice.method != "heldout":
        ctx.warn(
            f"The threshold is the in-sample {opts.threshold_pct:g}th percentile of the seed scores, not a held-out "
            f"one: {choice.fallback_reason}. In-sample thresholds are too tight and recall suffers (survey B6); add "
            f"seeds from more places (seeds in one {opts.fold_cell_m:g} m cell, or touching, form one spatial block), "
            "or pass a smaller --fold-cell-m for a small AOI."
        )
    if opts.score == "mahalanobis" and n_folds >= 2:
        smallest = min(int((fold_used != k).sum()) for k in range(n_folds))
        if smallest < MIN_MAHALANOBIS_PIXELS:
            ctx.warn(
                f"The held-out Mahalanobis scores of some fold rest on {smallest} training pixels, under the "
                f"{MIN_MAHALANOBIS_PIXELS} the final score requires; the threshold may be loose."
            )
    n = len(table)
    fold_all = np.full(n, -1, dtype=np.int64)
    fold_all[used] = fold_used
    ins_all = np.full(n, np.nan)
    ins_all[used] = in_sample
    held_all = np.full(n, np.nan)
    held_all[used] = held
    ok_b = table["_ok_b"].to_numpy(dtype=bool) & used
    sb_all = np.full(n, np.nan)
    if ok_b.any():
        sb_all[ok_b] = model.score_vectors(emb_b[ok_b].astype(np.float64))
    table["fold"] = fold_all
    table["score_in_sample"] = ins_all
    table["score_heldout"] = held_all
    table["score_year_b"] = sb_all
    table["inside_aoi"] = (
        shapely.contains_xy(ctx.aoi.geometry, table["lon"].to_numpy(), table["lat"].to_numpy())
        if n
        else np.zeros(0, dtype=bool)
    )
    del emb_a, emb_b, x
    t = ctx.lap("reference", t)

    zones, counts = _stream(ctx, model, bool(seed_radar))
    t = ctx.lap("stream", t)
    score_a = {z: st.score_a for z, st in zones.items()}
    score_b = {z: st.score_b for z, st in zones.items()}
    classified = {z: np.isfinite(score_a[z]) & np.isfinite(score_b[z]) for z in names}
    n_classified = sum(int(m.sum()) for m in classified.values())
    if n_classified == 0:
        raise WetlandError(
            f"No AOI pixel is valid in both {year_a} and {year_b} in the seeds' {regime_name(seed_radar)} radar "
            "regime; choose other years or another AOI."
        )
    thr = choice.threshold
    raw_a = {z: candidate_extent(score_a[z], thr, classified[z]) for z in names}
    raw_b = {z: candidate_extent(score_b[z], thr, classified[z]) for z in names}
    dst = max(plan.zones, key=lambda zp: zp.n_pixels).zone
    min_pixels = metrics.area_to_pixels(opts.min_area_ha)
    extent_a, _, _, _ = mosaic_and_clean(raw_a, plan, min_pixels=min_pixels, dst_zone=dst)
    extent_b, _, _, _ = mosaic_and_clean(raw_b, plan, min_pixels=min_pixels, dst_zone=dst)
    change = {z: change_map(extent_a[z], extent_b[z], classified[z]) for z in names}
    t = ctx.lap("clean", t)

    n_raw_a = sum(int(m.sum()) for m in raw_a.values())
    n_raw_b = sum(int(m.sum()) for m in raw_b.values())
    del raw_a, raw_b
    n_a = sum(int(m.sum()) for m in extent_a.values())
    n_b = sum(int(m.sum()) for m in extent_b.values())
    n_gain = sum(int((change[z] == 1).sum()) for z in names)
    n_loss = sum(int((change[z] == -1).sum()) for z in names)
    n_stable = sum(int((extent_a[z] & extent_b[z]).sum()) for z in names)
    if n_a == 0:
        ctx.warn(f"No {year_a} extent patch survived the threshold and the minimum area.")

    # Coastal fraction on each zone grid.
    coast: dict[str, Any] = {"coast_km": opts.coast_km, **state.shoreline_info}
    frac_a = frac_b = None
    if state.shoreline is not None:
        lines = state.shoreline
        near: dict[str, np.ndarray] = {}
        for zp in plan.zones:
            dist = shoreline_distance_m(zp, lines, opts.coast_km * 1000.0)
            near[zp.zone] = dist <= opts.coast_km * 1000.0
            del dist
        frac_a = coastal_fraction(extent_a, near)
        frac_b = coastal_fraction(extent_b, near)
        coast.update(
            {
                "lines": len(lines),
                "length_km": _lines_length_km(lines),
                "median_segment_m": _median_segment_m(lines),
                "aoi_pixels_within_buffer": sum(
                    int((near[z] & zp.mask).sum()) for z, zp in zip(names, plan.zones, strict=True)
                ),
            }
        )
        del near
        if not lines:
            ctx.warn(f"No shoreline lies within {opts.coast_km:g} km of the AOI, so the coastal fraction is 0.")
    coast["fraction_year_a"] = frac_a
    coast["fraction_year_b"] = frac_b
    t = ctx.lap("coastal", t)

    validation: dict[str, Any] | None = None
    validation_written = False
    exclusion: dict[str, np.ndarray] | None = None
    if ctx.validate:
        exclusion = _exclusion(plan, table, opts.seed_buffer_m)
        validation, validation_written = _validate(
            ctx, state, opts, extent_a=extent_a, extent_b=extent_b, classified=classified, exclusion=exclusion
        )
        t = ctx.lap("validation", t)

    band_names = [f"score_{year_a}", f"score_{year_b}", f"extent_{year_a}", f"extent_{year_b}", "change"]
    mosaic = BandMosaic(plan, len(band_names), dst_zone=dst)
    mosaic.put(0, score_a)
    mosaic.put(1, score_b)
    nan = np.float32(np.nan)
    mosaic.put(2, {z: np.where(classified[z], extent_a[z].astype(np.float32), nan) for z in names})
    mosaic.put(3, {z: np.where(classified[z], extent_b[z].astype(np.float32), nan) for z in names})
    mosaic.put(4, change)
    del score_a, score_b, zones, change
    assert mosaic.data is not None
    with rasterio.Env(GDAL_CACHEMAX=WRITE_CACHE_BYTES):
        export.write_geotiff(ctx.paths.tif, mosaic.data, mosaic.transform, mosaic.crs, band_names=band_names)
    net_ha = (n_b - n_a) * HA_PER_PIXEL
    net_pct = _ratio(n_b - n_a, n_a) * 100.0 if n_a else math.nan
    common = {
        "net_change_ha": net_ha,
        "net_change_pct": net_pct,
        "gain_ha": n_gain * HA_PER_PIXEL,
        "loss_ha": n_loss * HA_PER_PIXEL,
        "stable_ha": n_stable * HA_PER_PIXEL,
        "threshold": thr,
        "score": opts.score,
        "classified_pixels": n_classified,
    }
    rows = [
        {"year": year, "extent_ha": n_px * HA_PER_PIXEL, "extent_pixels": n_px, "coastal_fraction": frac, **common}
        for year, n_px, frac in ((year_a, n_a, frac_a), (year_b, n_b, frac_b))
    ]
    summary_table = pd.DataFrame(rows)
    summary_table["coastal_fraction"] = summary_table["coastal_fraction"].astype(np.float64)
    export.write_table(summary_table, ctx.paths.summary_parquet.with_suffix(""))
    _write_seeds(ctx.extra_outputs["seeds"], table, ctx.aoi.geometry)
    step = stride_for_preview(mosaic.data.shape[1:])
    ref_image = None
    ref_year = None
    if state.ref.masks is not None and state.ref.years:
        ref_year = year_a if year_a in state.ref.years else state.ref.years[0]
        ref_mosaic, _, _ = export.mosaic_zones(
            [GridArray(state.ref.masks[ref_year][z.zone], z) for z in plan.zones],
            dst_zone=dst,
            nodata=export.BOOL_NODATA,
        )
        ref_image = ref_mosaic[::step, ::step].copy()
        del ref_mosaic
    _write_preview(
        ctx.paths.preview,
        score_a=mosaic.data[0, ::step, ::step],
        score_b=mosaic.data[1, ::step, ::step],
        extent_a=mosaic.data[2, ::step, ::step],
        extent_b=mosaic.data[3, ::step, ::step],
        change=mosaic.data[4, ::step, ::step],
        reference=ref_image,
        reference_year=ref_year,
        heldout=table["score_heldout"].to_numpy(dtype=np.float64),
        in_sample=table["score_in_sample"].to_numpy(dtype=np.float64),
        choice=choice,
        year_a=year_a,
        year_b=year_b,
    )
    del mosaic
    ctx.lap("write", t)

    radar = {
        "seed_regime": regime_name(seed_radar),
        "probe": {k: v for k, v in probe.items() if k != "majority_radar"},
        "seeds_radar": int((table["regime"] == "radar").sum()),
        "seeds_optical": int((table["regime"] == "optical").sum()),
        "seeds_no_data": int((table["regime"] == "no_data").sum()),
        "seeds_dropped_regime": int((reasons == "regime").sum()),
        "valid_both_pixels": counts["valid_both"],
        "regime_mismatch_pixels": counts["regime_mismatch"],
        "regime_mismatch_fraction": _ratio(counts["regime_mismatch"], counts["valid_both"]),
        "other_regime_pixels": counts["other_regime"],
        "other_regime_fraction": _ratio(counts["other_regime"], counts["valid_both"]),
        "unclassified_other_regime_year_a": counts["other_regime_a"],
        "unclassified_other_regime_year_b": counts["other_regime_b"],
    }
    if counts["regime_mismatch"]:
        ctx.warn(
            f"{counts['regime_mismatch']:,} valid AOI pixels switch radar regime between {year_a} and {year_b} and "
            "were not classified (survey B5)."
        )
    if counts["other_regime"]:
        ctx.warn(
            f"{counts['other_regime']:,} valid AOI pixels are {regime_name(not seed_radar)} in both years, not the "
            f"seeds' regime ({regime_name(seed_radar)}), and were not classified."
        )
    note = storm_note(ctx.aoi.geometry, year_a, year_b)
    seeds_record = {
        **opts.seeds.record(),
        **_seed_summary(table, choice, opts, model),
    }
    extent_record = {
        str(year): {
            "extent_ha": n_px * HA_PER_PIXEL,
            "extent_pixels": n_px,
            "extent_pixels_before_cleaning": n_raw,
            "coastal_fraction": frac,
        }
        for year, n_px, n_raw, frac in ((year_a, n_a, n_raw_a, frac_a), (year_b, n_b, n_raw_b, frac_b))
    }
    summary = {
        "aoi_area_ha": plan.area_ha,
        "aoi_pixels": plan.n_pixels,
        "zones": plan.zone_names,
        "year_a": year_a,
        "year_b": year_b,
        "extent": extent_record,
        "extent_ha_year_a": n_a * HA_PER_PIXEL,
        "extent_ha_year_b": n_b * HA_PER_PIXEL,
        "net_change_ha": net_ha,
        "net_change_pct": net_pct,
        "gain_ha": n_gain * HA_PER_PIXEL,
        "loss_ha": n_loss * HA_PER_PIXEL,
        "stable_ha": n_stable * HA_PER_PIXEL,
        "gain_pixels": n_gain,
        "loss_pixels": n_loss,
        "classified_pixels": n_classified,
        "classified_fraction": _ratio(n_classified, plan.n_pixels),
        "extent_definition": (
            f"Classified in both years (valid, and in the {regime_name(seed_radar)} radar regime in both), score at "
            f"or below {thr:.6g} ({opts.score}); then a 3 x 3 opening and patches of at least {opts.min_area_ha:g} "
            "ha, on the mosaic of every zone. Gain is extent in year_b only, loss in year_a only."
        ),
        "min_area_ha": opts.min_area_ha,
        "min_pixels": min_pixels,
        "coastal": coast,
        "coastal_fraction_year_a": frac_a,
        "coastal_fraction_year_b": frac_b,
        "seeds": seeds_record,
        "score": _score_record(opts, model),
        "threshold": choice.record(),
        "radar": radar,
        "storm_note": note,
        "validation": validation,
    }
    report = finish_run(ctx, summary=summary, extra=ctx.extra_outputs, validation_written=validation_written)
    return WetlandResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=validation_written,
    )


def _exclusion(plan: AOIPlan, table: pd.DataFrame, buffer_m: float) -> dict[str, np.ndarray]:
    on = table[table["row"] >= 0]
    centres = np.stack(
        [on["centre_lon"].to_numpy(dtype=np.float64), on["centre_lat"].to_numpy(dtype=np.float64)], axis=1
    )
    return {z.zone: seed_exclusion(z, centres, buffer_m) for z in plan.zones}


def false_positive_classes(
    plan: AOIPlan,
    extent: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    classes: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """The WorldCover class of every scored extent pixel outside the reference, pooled over zones.

    Returns:
        ``{"<code>": {"name", "pixels", "share"}}`` in code order, shares of all false positives.
    """
    counts = np.zeros(256, dtype=np.int64)
    for z in plan.zones:
        fp = np.asarray(extent[z.zone], dtype=bool) & ~np.asarray(reference[z.zone], dtype=bool)
        fp &= np.asarray(scored[z.zone], dtype=bool)
        counts += np.bincount(np.asarray(classes[z.zone])[fp].astype(np.int64), minlength=256)
    total = int(counts.sum())
    out: dict[str, dict[str, Any]] = {}
    for code in np.flatnonzero(counts):
        c = int(code)
        out[str(c)] = {
            "name": validate.WORLDCOVER_CLASSES.get(c, "no data" if c == 0 else "unknown"),
            "pixels": int(counts[c]),
            "share": int(counts[c]) / total,
        }
    return out


def _version_mix(
    plan: AOIPlan, ref: _Reference, scored: Mapping[int, Mapping[str, np.ndarray]]
) -> dict[str, Any] | None:
    """How the WorldCover maps of two scored years, from two product versions, compare (None for one year).

    Over the pixels scored in both years: the share where the two maps agree on the reference
    classes, each map's share of reference pixels, and the share where their class codes agree.
    """
    years = sorted(y for y in ref.years if y in validate.WORLDCOVER_VERSIONS)
    versions = {str(y): validate.WORLDCOVER_VERSIONS[y] for y in years}
    if len(set(versions.values())) < 2 or ref.masks is None or ref.classes is None:
        return None
    y0, y1 = years[0], years[-1]
    both = agree = codes_agree = in0 = in1 = 0
    for z in plan.zones:
        m = np.asarray(scored[y0][z.zone], dtype=bool) & np.asarray(scored[y1][z.zone], dtype=bool)
        r0, r1 = ref.masks[y0][z.zone][m], ref.masks[y1][z.zone][m]
        both += int(m.sum())
        agree += int((r0 == r1).sum())
        in0 += int(r0.sum())
        in1 += int(r1.sum())
        codes_agree += int((ref.classes[y0][z.zone][m] == ref.classes[y1][z.zone][m]).sum())
    return {
        "years": [y0, y1],
        "versions": versions,
        "pixels_scored_both_years": both,
        "reference_agreement": _ratio(agree, both),
        "class_code_agreement": _ratio(codes_agree, both),
        "reference_share": {str(y0): _ratio(in0, both), str(y1): _ratio(in1, both)},
    }


def _validate(
    ctx: RunContext,
    state: _State,
    opts: _Options,
    *,
    extent_a: Mapping[str, np.ndarray],
    extent_b: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
    exclusion: Mapping[str, np.ndarray],
    extra_caveats: Sequence[str] = (),
) -> tuple[dict[str, Any], bool]:
    """Score each covered year, write ``validation.json``, and return the summary block and whether it was written."""
    ref = state.ref
    if ref.masks is None or ref.coverage is None or ref.info is None:
        ctx.warn(f"{VALIDATION_SKIPPED}: {ref.error}")
        return {"error": ref.error}, False
    plan = ctx.plan
    a = ctx.year_a
    scores: dict[str, Any] = {}
    per_year: dict[str, Any] = {}
    scored_by_year: dict[int, dict[str, np.ndarray]] = {}
    for year in ref.years:
        extent = extent_a if year == a else extent_b
        parts: dict[str, validate.BinaryScores] = {}
        with_seeds: list[validate.BinaryScores] = []
        scored: dict[str, np.ndarray] = {}
        scored_by_year[year] = scored
        excluded = 0
        for z in plan.zones:
            base = np.asarray(classified[z.zone], dtype=bool) & ref.coverage[year][z.zone] & z.mask
            keep = base & ~exclusion[z.zone]
            excluded += int((base & exclusion[z.zone]).sum())
            scored[z.zone] = keep
            pred = np.asarray(extent[z.zone], dtype=bool)
            parts[z.zone] = validate.binary_scores(pred, ref.masks[year][z.zone], valid=keep)
            with_seeds.append(validate.binary_scores(pred, ref.masks[year][z.zone], valid=base))
        merged = validate.merge_scores(list(parts.values()))
        merged_seeds = validate.merge_scores(with_seeds)
        n_ref_aoi = sum(int((ref.masks[year][z.zone] & z.mask).sum()) for z in plan.zones)
        n_ref_unclassified = sum(
            int((ref.masks[year][z.zone] & z.mask & ~np.asarray(classified[z.zone], dtype=bool)).sum())
            for z in plan.zones
        )
        entry: dict[str, Any] = {
            "recall": merged.recall,
            "precision": merged.precision,
            "iou": merged.iou,
            "f1": merged.f1,
            "pixels_scored": merged.n,
            "reference_pixels_scored": merged.tp + merged.fn,
            "extent_pixels_scored": merged.tp + merged.fp,
            "seed_excluded_pixels": excluded,
            "recall_with_seeds": merged_seeds.recall,
            "precision_with_seeds": merged_seeds.precision,
            "reference_pixels_in_aoi": n_ref_aoi,
            "reference_pixels_unclassified": n_ref_unclassified,
            "coverage_pixels": sum(int((ref.coverage[year][z.zone] & z.mask).sum()) for z in plan.zones),
        }
        scores[f"extent_{year}"] = merged
        scores[f"extent_{year}_per_zone"] = parts
        scores[f"extent_{year}_with_seeds"] = merged_seeds
        if ref.classes is not None:
            fp = false_positive_classes(plan, extent, ref.masks[year], scored, ref.classes[year])
            scores[f"false_positive_classes_{year}"] = fp
            entry["false_positive_classes"] = fp
        per_year[str(year)] = entry
    head = a if a in ref.years else ref.years[0]
    top = per_year[str(head)]
    summary: dict[str, Any] = {
        "source": ref.info["source"],
        "year": head,
        "recall": top["recall"],
        "precision": top["precision"],
        "iou": top["iou"],
        "f1": top["f1"],
        "recall_with_seeds": top["recall_with_seeds"],
        "years_scored": list(ref.years),
        "years_not_covered": list(ref.not_covered),
        "per_year": per_year,
        "seed_buffer_m": opts.seed_buffer_m,
    }
    caveats = [SCOPE_CAVEAT, f"Seed buffer: {opts.seed_buffer_m:g} m around every seed pixel centre."]
    version_mix = _version_mix(plan, ref, scored_by_year) if ref.info["source"] == "worldcover" else None
    if version_mix is not None:
        summary["version_mix"] = version_mix
    product = _same_product(opts.seeds, ref.info["source"])
    if product is not None:
        caveats.append(SAME_PRODUCT_CAVEAT.format(product=product))
    if ref.info["source"] == "worldcover":
        caveats.append(WORLDCOVER_CAVEAT)
        caveats.append(FALSE_POSITIVE_CAVEAT)
        if tuple(opts.reference_classes) != DEFAULT_REFERENCE_CLASSES:
            names = ", ".join(f"{c} ({validate.WORLDCOVER_CLASSES[c]})" for c in opts.reference_classes)
            caveats.append(OTHER_CLASSES_CAVEAT.format(classes=names))
        caveats.append(
            "Reference: ESA WorldCover "
            + ", ".join(f"{y} {v}" for y, v in ref.info["versions"].items())
            + f", tiles from {ref.info['url_template']}."
        )
        if version_mix is not None:
            caveats.append(VERSION_MIX_CAVEAT)
    elif ref.info["source"] == "gmw":
        caveats.append(GMW_CAVEAT)
        caveats.append(f"Reference: Global Mangrove Watch {ref.info['version']} (Zenodo {ref.info['zenodo_record']}).")
    else:
        caveats.append(FILE_CAVEAT)
        if ref.info.get("kind") == "raster":
            caveats.append(FILE_RASTER_COVERAGE_CAVEAT.format(pixels=ref.info["coverage_pixels"], aoi=plan.n_pixels))
        else:
            caveats.append(FILE_VECTOR_COVERAGE_CAVEAT.format(aoi=plan.n_pixels))
        caveats.append(
            f"Reference: the file {ref.info['file']} (sha256 {ref.info['sha256']}), supplied with --reference."
        )
    if ref.not_covered:
        caveats.append(
            f"The reference has no map for {', '.join(str(y) for y in ref.not_covered)}, so that year's extent is not "
            "scored."
        )
    caveats += list(extra_caveats)
    record = {
        **ref.info,
        "years_scored": list(ref.years),
        "years_not_covered": list(ref.not_covered),
        "seed_buffer_m": opts.seed_buffer_m,
        "seeds": opts.seeds.source,
        "version_mix": version_mix,
    }
    validate.write_validation_json(ctx.paths.validation, scores=scores, reference=record, caveats=caveats)
    return summary, True


def _redo_reason(previous: Mapping[str, Any]) -> str:
    """Why the validation of an otherwise current run is redone: a key of :data:`REDO_REASONS`.

    The earlier run's summary says it: no validation block (it did not validate, which the digest
    does not cover), a validation block with an error (its reference could not be loaded), or a
    complete one (its ``validation.json`` went missing or was changed since).
    """
    summary = previous.get("summary")
    validation = summary.get("validation") if isinstance(summary, Mapping) else None
    if not isinstance(validation, Mapping):
        return "not_validated"
    if validation.get("error"):
        return "reference_failed"
    return "missing"


def _redo_validation(ctx: RunContext, state: _State, opts: _Options) -> WetlandResult:
    """Redo only the validation of a run whose other outputs are current, without reading a chunk.

    Both years' extents and the classified pixels come back from the run's GeoTIFF
    (:func:`zone_values_from_raster`), and the seed exclusion from the seeds, placed again from
    metadata. When the reference still cannot be loaded, nothing is written and the run counts as
    skipped.
    """
    previous = json.loads(ctx.paths.report.read_text(encoding="utf-8"))
    if state.ref.masks is None:
        logger.warning(
            "%s: outputs in %s are current; the validation reference still cannot be loaded (%s), so nothing "
            "was redone.",
            APP,
            ctx.paths.report.parent,
            state.ref.error,
        )
        return WetlandResult(
            report=previous,
            paths=ctx.paths,
            extra_paths=dict(ctx.extra_outputs),
            skipped=True,
            validation_written=False,
        )
    assert state.table is not None
    plan = ctx.plan
    ext_a = zone_values_from_raster(ctx.paths.tif, plan, band=3)
    ext_b = zone_values_from_raster(ctx.paths.tif, plan, band=4)
    change = zone_values_from_raster(ctx.paths.tif, plan, band=5)
    classified = {z: np.isfinite(v) for z, v in change.items()}
    exclusion = _exclusion(plan, state.table, opts.seed_buffer_m)
    validation, written = _validate(
        ctx,
        state,
        opts,
        extent_a={z: v == 1 for z, v in ext_a.items()},
        extent_b={z: v == 1 for z, v in ext_b.items()},
        classified=classified,
        exclusion=exclusion,
        extra_caveats=[REDONE_CAVEAT.format(reason=REDO_REASONS[_redo_reason(previous)])],
    )
    summary = dict(previous.get("summary") or {})
    summary["validation"] = validation
    old = [w for w in previous.get("warnings") or [] if not str(w).startswith(VALIDATION_SKIPPED)]
    report = finish_validation_only(
        ctx, previous, summary=summary, warnings=[*old, *ctx.warnings], validation_written=written
    )
    return WetlandResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=written,
        validation_redone=True,
    )


# --------------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------------


def _write_preview(
    path: Path,
    *,
    score_a: np.ndarray,
    score_b: np.ndarray,
    extent_a: np.ndarray,
    extent_b: np.ndarray,
    change: np.ndarray,
    reference: np.ndarray | None,
    reference_year: int | None,
    heldout: np.ndarray,
    in_sample: np.ndarray,
    choice: ThresholdChoice,
    year_a: int,
    year_b: int,
) -> Path:
    """Write the six-panel preview with a matplotlib Figure (never pyplot).

    Panels: the ``year_a`` score map, the ``year_a`` extent over the reference outline, the
    ``year_b`` extent, the gain and loss map, the held-out seed score histogram with the threshold
    and the in-sample percentile, and the ``year_b`` score map. Every image comes in already strided.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    grey = export.PREVIEW_NAN_RGB
    thr = choice.threshold
    fig = Figure(figsize=(18, 11), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    # Three times the threshold keeps the contrast around it; everything less seed-like saturates.
    finite = np.concatenate([score_a[np.isfinite(score_a)], score_b[np.isfinite(score_b)]])
    vmax = 3.0 * thr if thr > 0 else (float(np.percentile(finite, 99)) if finite.size else 1.0)
    vmax = max(vmax, 1e-6)
    score_cmap = mpl.colormaps["viridis_r"].with_extremes(bad=grey)
    for ax, img, year in ((axes[0, 0], score_a, year_a), (axes[1, 2], score_b, year_b)):
        shown = ax.imshow(np.ma.masked_invalid(img), cmap=score_cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
        bar = fig.colorbar(shown, ax=ax, label=f"score; threshold {thr:.4g} (red, {choice.method})", shrink=0.85)
        bar.ax.axhline(thr, color="#d7191c", linewidth=1.5)
        ax.set_title(f"Score {year} (lower is more seed-like)")
        ax.set_axis_off()
    ext_cmap = mpl.colors.ListedColormap(["#e8e4d8", "#1a9641"]).with_extremes(bad=grey)
    for ax, img, year, ref in (
        (axes[0, 1], extent_a, year_a, reference if reference_year == year_a else None),
        (axes[0, 2], extent_b, year_b, reference if reference_year == year_b else None),
    ):
        ax.imshow(np.ma.masked_invalid(img), cmap=ext_cmap, vmin=-0.5, vmax=1.5, interpolation="nearest")
        fin = img[np.isfinite(img)]
        share = float(np.mean(fin == 1)) if fin.size else math.nan
        title = f"Extent {year}: {share:.0%} of classified pixels"
        if ref is not None:
            outline = (ref == 1).astype(np.float64)
            if outline.any():
                ax.contour(outline, levels=[0.5], colors=["#2166ac"], linewidths=0.6)
            title += f",\nreference {reference_year} outline in blue"
        ax.set_title(title)
        ax.set_axis_off()
    ax = axes[1, 0]
    cat = np.where(np.isfinite(change), change + 1.0, np.nan)
    stable = np.isfinite(change) & (change == 0) & (extent_a == 1)
    cat = np.where(stable, 3.0, cat)
    cmap = mpl.colors.ListedColormap(["#d7191c", "#efefef", "#2c7bb6", "#a6d96a"]).with_extremes(bad=grey)
    ax.imshow(np.ma.masked_invalid(cat), cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title(f"Change {year_a} to {year_b}: gain blue, loss red, stable extent green")
    ax.set_axis_off()
    ax = axes[1, 1]
    held = heldout[np.isfinite(heldout)]
    ins = in_sample[np.isfinite(in_sample)]
    data = held if held.size else ins
    if data.size:
        ax.hist(data, bins=40, color="#636363", alpha=0.85, label="held-out seed scores" if held.size else "in-sample")
    ax.axvline(thr, color="#d7191c", linewidth=1.5, label=f"threshold {thr:.4g} ({choice.method})")
    ax.axvline(
        choice.in_sample_percentile,
        color="#2166ac",
        linestyle="--",
        linewidth=1.2,
        label=f"in-sample p{choice.percentile:g} {choice.in_sample_percentile:.4g}",
    )
    ax.set_xlabel("score")
    ax.set_ylabel("seed pixels")
    ax.legend(loc="upper right")
    ax.set_title(f"Seed scores: {data.size} pixels, {choice.folds} spatial folds")
    final = Path(path)
    with export.atomic_path(final) as tmp, tmp.open("wb") as fh:
        fig.savefig(fh, format="png", dpi=100)
    return final


__all__ = [
    "APP",
    "DEFAULT_REFERENCE_CLASSES",
    "FOLD_CELL_M",
    "MIN_MAHALANOBIS_PIXELS",
    "MIN_SEED_PIXELS",
    "SCORES",
    "SEED_COLUMNS",
    "SeedModel",
    "SeedSet",
    "ThresholdChoice",
    "WetlandError",
    "WetlandResult",
    "assign_folds",
    "candidate_extent",
    "change_map",
    "choose_threshold",
    "coastal_fraction",
    "false_positive_classes",
    "file_reference",
    "fit_seed_model",
    "heldout_scores",
    "load_seeds",
    "memory_estimate",
    "place_seeds",
    "read_shoreline",
    "run_wetland",
    "score_block",
    "scored_years",
    "seed_chunks",
    "seed_exclusion",
    "shoreline_distance_m",
    "spatial_blocks",
    "storm_note",
]
