"""Read the TESSERA v1.1 global embeddings through the ``geotessera`` library.

The published store is the dClimate Icechunk repository at :data:`STORE_URL`. It holds one Zarr
group per UTM zone and hemisphere (``10N``, ``20S``). Each group carries int8 ``embeddings``
(time, northing, easting, band), float32 ``scales`` (true value is ``embeddings * scales``, NaN
means no data), and three uint16 observation counts. Rows run north to south, the time axis is
calendar years 2017 to 2025, and a zone lists the years it has published in
``attrs["years_complete"]``. ``docs/prebuild-survey.md`` records how each of these facts was checked.

Every embedding read goes through ``geotessera`` (pinned ``>=0.11,<0.12``), as follows.

- **Identity through** ``GeoTesseraZarr``. :meth:`TesseraReader.open` constructs
  ``geotessera.GeoTesseraZarr(store_url=STORE_URL)`` with the explicit URL, never the library
  default, and before any chunk read requires ``gt.dataset.name == "1.1-dclimate"``; anything else
  raises :class:`DatasetMismatchError`, which the CLI turns into exit code 2. The reader's
  :meth:`~TesseraReader.provenance`, and so every report's ``store`` block, records ``gt.url``, the
  dataset name, ``gt.model_version``, ``gt.build_version`` and the snapshot id actually read.
  geotessera reads the two versions at the head of main, so the reader checks them against the
  snapshot it reads: another model is refused, and another build is recorded as the snapshot's
  (geotessera's value kept as ``head_build_version``).
- **Reads through geotessera's** ``IcechunkStore``, **pinned and counted**. ``GeoTesseraZarr`` in
  0.11 opens the branch head with a session of its own: it takes no snapshot or branch, and given a
  zarr ``Store`` it takes the plain Zarr path, not Icechunk. So it can neither pin a snapshot nor
  have its bytes counted (the upstream issue asks for both). The reader therefore constructs the
  class ``GeoTesseraZarr`` uses inside, the public :class:`geotessera.icechunk.IcechunkStore`, at the
  snapshot asked for (``--snapshot``) or at the branch head as it stands when the reader opens (a
  read-only session never moves, so a run reads one snapshot throughout and records it). Its public
  ``root`` is replaced, before the first read, by the same group opened through
  :class:`CountingStore` (:class:`CountedIcechunkStore`), and every chunk read goes through its
  ``zone_arrays(zone)``: the arrays behind ``open_zone(zone)``, which is what
  ``GeoTesseraZarr.open_zone`` returns for an Icechunk store. The private ``gt._icechunk`` is never
  touched. No retry layer is added and no ``progress=`` is passed (survey, skill rules).
- **The merged grid.** geotessera presents each zone as one grid in the northern CRS, both
  hemisphere groups merged with southern northings negative, embeddings ordered
  ``(time, band, y, x)``. The apps keep this package's per-hemisphere grids (:class:`ZoneGrid`, the
  nominal-zone masks, the chunk-aligned blocks), so a window of a zone group maps onto the merged
  grid by a whole-pixel offset worked out from the two grids' ``spatial:transform`` attributes:
  ``merged_row = group_row + (top_merged - top_group) / 10``, ``top_group`` taken in the northern
  CRS (less the 10,000,000 m false northing for a south group), and likewise for columns from the
  west edges. Rows centred north of the equator come from the north group and the rest from the
  south group. Each south group reaches 14.72 km north of the equator, so a south-zone window that
  extends above the equator holds the north group's pixels there, or no data in a zone with no
  north group (whose merged grid starts at the equator); those rows lie outside the zone's nominal
  band, so plan masks and point reads never use them. Point reads start their chunk window at the
  south group's first own row, so they never fetch the north group's chunk for them; a cached read
  whose chunk-aligned cover reaches above that row does (live, a south group's chunk row 5 takes
  the north group's bottom chunk row with it, about twice the bytes of that chunk row). Snapshots
  before ``B6AFRJGX5XQC0A716PA0`` register ``spatial:transform`` at the pixel centre, which puts a
  north group's bottom row on the equator where geotessera serves it from neither group; a read of
  that row is refused rather than returned as no data.
- **Embeddings in strips.** geotessera fills its own ``(band, y, x)`` output for every read (with
  an ``np.ix_`` assignment of the transposed zarr block, most of the extra CPU), and the apps want
  ``(rows, cols, band)``, C-contiguous, as zarr returned it before. So the reader asks for whole
  chunk rows of at most 64 MiB at a time (one chunk row of a 2048-column block) and transposes
  each strip into one preallocated output: the transient is a strip, not the window. At most four
  embeddings reads run at once. Measured on one 512 x 2048 block of every layer against the 0.1.0
  reader, peak memory is the same for 1 year (within 5 percent) and lower for 2, 4 and 9 years (by
  5 to 20 percent at 2 and 4, by 20 to 45 percent at 9, where the limit applies). The CPU is about
  twice (1.7 to 3.6 times), and the wall time is longer where the network does not hide it, most
  for a 1-year block (live, about 4.4 s against 2 s); the 0.2.0 checkpoint A report has the figures.
- **Observation counts.** geotessera exposes ``s2_obs_count``, ``s1_asc_obs_count`` and
  ``s1_desc_obs_count`` only on the zone dataset (``open_zone`` and ``zone_arrays``), not through
  ``read_region`` or ``sample_points``. They are the one family of layers this package reads that
  the library does not wrap in a read call, and the package reads them from that zone dataset, on
  the same store object as the embeddings, because every app's radar-regime check needs them.
- **int8 throughout.** Blocks stay int8 with their scales, and the cosine paths work on them;
  nothing here dequantises (:meth:`Window.dequantize` does, for the means that need it).
  ``read_region_quantized`` would return the same int8, but only for one lon/lat box in the zone
  of its centre, with no observation counts and no chunk alignment.
- **Published years.** ``years_complete`` comes from ``IcechunkStore.incomplete_years()``, read
  once per reader: the store's years less each group's incomplete ones. It reads only attributes
  (about 0.1 s for all 120 groups live), where ``open_zone``'s ``geotessera:years_incomplete``
  costs 0.7 to 1.4 s per zone and reads the time coordinate's chunks. geotessera reads a group with
  no ``years_complete`` attribute as complete; this reader refuses it.

AOI reads are planned first (:func:`plan_aoi`), which touches metadata only, so callers can check
years, zone count and size before any chunk read. A plan is then read whole (:func:`read_plan`)
when small, or block by block (:func:`iter_plan_blocks`) when large. Layers, years and point chunks
are fetched concurrently on a small thread pool, since each zarr call is a few serial round trips
(shard index, then data). :attr:`TesseraReader.bytes_read` counts chunk payload bytes (see
:class:`CountingStore`).
"""

from __future__ import annotations

import asyncio
import atexit
import dataclasses
import hashlib
import itertools
import logging
import math
import os
import re
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np
import shapely
import zarr
from geotessera.icechunk import FALSE_NORTHING, IcechunkStore
from pyproj import CRS, Geod, Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from zarr.abc.store import Store
from zarr.core.buffer import default_buffer_prototype
from zarr.core.sync import sync
from zarr.storage import WrapperStore

from tessera_apps.cache import ChunkCache, ChunkKey

if TYPE_CHECKING:
    from zarr.abc.store import ByteRequest
    from zarr.core.buffer import Buffer, BufferPrototype

logger = logging.getLogger(__name__)

STORE_BUCKET = "tessera-embeddings"
STORE_PREFIX = "v1.1/dclimate.icechunk"
STORE_REGION = "us-west-2"
STORE_URL = f"s3://{STORE_BUCKET}/{STORE_PREFIX}"
#: The geotessera dataset the store must be (``gt.dataset.name``): the dClimate v1.1 run.
DATASET_NAME = "1.1-dclimate"

FIRST_YEAR = 2017
LAST_YEAR = 2025
YEARS = tuple(range(FIRST_YEAR, LAST_YEAR + 1))

PIXEL_M = 10.0
#: Hectares per 10 m pixel.
HA_PER_PIXEL = PIXEL_M * PIXEL_M / 10_000.0
N_BANDS = 128

#: Latitude limits of the UTM groups. Polar regions are not in the store.
NORTH_LIMIT = 84.0
SOUTH_LIMIT = -80.0

RADAR_LAYERS = ("s1_asc_obs_count", "s1_desc_obs_count")
OBS_LAYERS = ("s2_obs_count", *RADAR_LAYERS)
ALL_LAYERS = ("embeddings", "scales", *OBS_LAYERS)
#: Layers read by default. The observation counts cost about 2 percent of an embeddings chunk and
#: every app needs the radar counts to check that compared years share a radar regime.
DEFAULT_LAYERS = ALL_LAYERS
#: In-memory bytes per pixel of one year of every layer.
BYTES_PER_PIXEL = N_BANDS + 4 + 2 * len(OBS_LAYERS)

#: Default largest window read in one piece; anything bigger goes through :func:`iter_plan_blocks`.
MAX_WINDOW_PIXELS = 1024 * 1024
#: Default block shape for streaming reads: two chunk rows by one shard width.
DEFAULT_BLOCK = (512, 2048)
#: AOIs larger than this are refused unless the caller raises the limit (brief section 6).
MAX_AOI_KM2 = 10_000.0
#: A zone window may be at most this many times the AOI's own pixel count (plus a floor below), so
#: a few small, far-apart polygons or a long thin diagonal cannot allocate a continent-sized mask.
MAX_WINDOW_TO_AOI = 8.0
_WINDOW_FLOOR_PX = 4_000_000

_READ_THREADS = int(os.environ.get("TESSERA_APPS_READ_THREADS", "16"))
_POOL = ThreadPoolExecutor(max_workers=_READ_THREADS, thread_name_prefix="tessera-read")
#: Point chunk groups run here, never on _POOL: each group's read submits its layers to _POOL and
#: waits, and a pool whose workers all wait on their own queue deadlocks.
_GROUP_POOL = ThreadPoolExecutor(max_workers=max(1, _READ_THREADS // 2), thread_name_prefix="tessera-group")
_LOCAL = threading.local()
#: Embeddings reads through geotessera in flight at once, across every reader. Each read holds its
#: output and, one strip at a time, geotessera's (band, y, x) array and the zarr block it fills it
#: from. The strips alone keep a 512 x 2048 block of 1 to 9 years within 5 percent of the 0.1.0
#: reader's peak memory or below it; the limit takes effect from 5 years in flight and lowers a
#: 9-year block's peak by about a third more (the 0.2.0 checkpoint A report has the measurements).
#: The other layers are 30 times smaller and are not gated.
_EMBEDDING_READS = threading.BoundedSemaphore(max(1, int(os.environ.get("TESSERA_APPS_EMBEDDING_READS", "4"))))
#: Most embeddings bytes one strip of a read asks geotessera for (whole chunk rows, at least one).
#: One live chunk row of a 2048-column block is exactly this, so a 512 x 2048 block reads in two
#: strips, while a point read's chunk (8 MiB) and the tests' small windows read in one. A strip costs
#: one more shard index fetch and round trip, and in a store whose inner chunks are smaller than
#: zarr's 1 MiB coalescing gap it can also fetch the chunks between the ones it needs; the live
#: embeddings chunks are about 8 MB, so it never does there.
_STRIP_BYTES = 64 << 20
#: Chunk groups fetched together by :func:`read_points`; bounds memory at about 8.7 MB per group.
_POINT_BATCH = 32


class TesseraError(Exception):
    """Base class for errors raised by this package."""


class YearsNotPublishedError(TesseraError):
    """One or more requested zone-years are not in the store."""

    def __init__(self, missing: dict[str, list[int]]) -> None:
        self.missing = missing
        parts = [f"{zone}: {', '.join(str(y) for y in years)}" for zone, years in sorted(missing.items())]
        super().__init__(
            "Requested years are not published for every zone the AOI touches ("
            + "; ".join(parts)
            + "). A year missing from a zone's years_complete is a publication gap, not an empty year. "
            "Choose years that every zone lists, or move the AOI."
        )


class AOIError(TesseraError):
    """The AOI cannot be served as given."""


class ReadTooLargeError(TesseraError):
    """A single window read would exceed the in-memory limit."""


class StoreUnavailableError(TesseraError):
    """The store cannot be opened: it cannot be reached, or the snapshot asked for is not in it."""


class DatasetMismatchError(TesseraError):
    """geotessera does not identify the store as the dClimate v1.1 dataset this package reads.

    Attributes:
        found: The dataset geotessera named, or ``"none"`` when it named none.
    """

    def __init__(self, found: str | None, message: str | None = None) -> None:
        self.found = found if found else "none"
        if message is None:
            message = f"geotessera identifies {STORE_URL} as dataset {self.found}, not {DATASET_NAME}."
        super().__init__(
            f"{message} This package reads only the dClimate v1.1 run: embeddings from different datasets live in "
            "unrelated spaces and cannot be compared. Check that geotessera is version 0.11 (tessera_apps pins "
            ">=0.11,<0.12)."
        )


# --------------------------------------------------------------------------------------------
# Zones
# --------------------------------------------------------------------------------------------


def canonicalize_zone(zone: str) -> str:
    """Normalise a zone id to its group name, for example ``"7s"`` to ``"07S"``.

    Raises:
        ValueError: The id is malformed or the zone number is outside 1 to 60.
    """
    text = zone.strip().upper()
    m = re.fullmatch(r"(\d{1,2})([NS])", text)
    if not m:
        raise ValueError(f"Zone {zone!r} is malformed: expected '<1-60><N|S>', for example '10N' or '20S'.")
    number = int(m.group(1))
    if not 1 <= number <= 60:
        raise ValueError(f"Zone number {number} in {zone!r} is out of range 1 to 60.")
    return f"{number:02d}{m.group(2)}"


def zone_number(lon: float) -> int:
    """UTM zone number for a longitude: pure six-degree bands, no MGRS exceptions.

    Longitude 180 belongs to zone 60.

    Raises:
        ValueError: The longitude is not finite or outside -180 to 180 (for example 0 to 360 data).
    """
    if not math.isfinite(lon) or not -180.0 <= lon <= 180.0:
        raise ValueError(f"Longitude {lon!r} is outside -180 to 180; convert 0 to 360 longitudes first.")
    return min(math.floor((lon + 180.0) / 6.0) + 1, 60)


def zone_for(lon: float, lat: float) -> str:
    """Nominal zone group for a point, for example ``zone_for(-120.5, 38.7) == "10N"``.

    The hemisphere is north for ``lat >= 0``. Norway and Svalbard exceptions do not apply: the store
    uses pure six-degree bands.

    Raises:
        ValueError: The point is not finite or outside lon -180 to 180, lat -90 to 90.
    """
    if not math.isfinite(lat) or not -90.0 <= lat <= 90.0:
        raise ValueError(f"Latitude {lat!r} is outside -90 to 90.")
    return f"{zone_number(lon):02d}{'N' if lat >= 0 else 'S'}"


def zone_epsg(zone: str) -> int:
    """EPSG code of a zone group: 326xx north, 327xx south."""
    z = canonicalize_zone(zone)
    return (32600 if z.endswith("N") else 32700) + int(z[:2])


def zone_band(zone: str) -> Polygon:
    """The zone's nominal footprint in lon/lat: its six-degree band in its hemisphere."""
    z = canonicalize_zone(zone)
    n = int(z[:2])
    west = -180.0 + 6.0 * (n - 1)
    south, north = (0.0, NORTH_LIMIT) if z.endswith("N") else (SOUTH_LIMIT, 0.0)
    return box(west, south, west + 6.0, north)


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    """Only the areal parts of a geometry.

    Intersections and ``make_valid`` can return lines and points alongside polygons, for example
    where an AOI edge lies on a band edge. Rasterizing a line burns every pixel it crosses, so
    those parts must go before any mask is built.
    """
    parts: list[BaseGeometry] = []
    stack = [geometry]
    while stack:
        g = stack.pop()
        if g.geom_type in ("Polygon", "MultiPolygon"):
            parts.append(g)
        elif g.geom_type == "GeometryCollection":
            stack.extend(shapely.get_parts(g))
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def zones_for_geometry(geometry: BaseGeometry) -> list[str]:
    """Nominal zones whose band overlaps a lon/lat geometry, sorted.

    A polygon that only touches a band edge does not count; a point on a band edge belongs to the
    zone :func:`zone_for` gives it.
    """
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [zone_for(geometry.x, geometry.y)]  # type: ignore[attr-defined]
    west, south, east, north = geometry.bounds
    zones = []
    for n in range(zone_number(max(west, -180.0)), zone_number(min(east, 180.0)) + 1):
        for hemi in ("N", "S"):
            if (hemi == "N" and north < 0) or (hemi == "S" and south >= 0):
                continue
            name = f"{n:02d}{hemi}"
            piece = geometry.intersection(zone_band(name))
            if piece.is_empty:
                continue
            if geometry.area > 0 and _polygonal(piece).area == 0:
                continue
            zones.append(name)
    return sorted(zones)


# --------------------------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ZoneGrid:
    """Pixel grid of one zone group, read from its ``spatial:transform`` attribute.

    Rows run north to south: row 0 is the top edge, so a larger northing is a smaller row. A point
    on a pixel edge belongs to the pixel south and east of it (floor convention).
    """

    zone: str
    epsg: int
    left: float
    top: float
    height: int
    width: int
    pixel: float = PIXEL_M

    @classmethod
    def from_attrs(cls, zone: str, attrs: dict[str, Any], shape: tuple[int, int]) -> ZoneGrid:
        """Build the grid from a zone group's attributes and its (height, width).

        Snapshots before ``B6AFRJGX5XQC0A716PA0`` stored ``spatial:transform`` at the pixel centre
        rather than the corner. When ``spatial:bbox`` shows the origin is half a pixel inside the
        grid edge, the corner from ``spatial:bbox`` is used instead.

        Raises:
            TesseraError: The transform is rotated, northing is not descending, the CRS does not match
                the zone, or the transform disagrees with ``spatial:bbox`` by anything other than the
                known half pixel.
        """
        z = canonicalize_zone(zone)
        try:
            a, b, c, d, e, f = (float(v) for v in attrs["spatial:transform"][:6])
        except (KeyError, TypeError, ValueError) as exc:
            raise TesseraError(f"Zone {z} has no usable spatial:transform attribute.") from exc
        if b != 0.0 or d != 0.0:
            raise TesseraError(f"Zone {z} has a rotated transform {attrs['spatial:transform']}; not supported.")
        if a <= 0 or e >= 0 or not math.isclose(a, -e):
            raise TesseraError(
                f"Zone {z} transform {attrs['spatial:transform']} does not have square pixels with northing "
                "descending; this reader assumes row 0 is the north edge."
            )
        epsg = zone_epsg(z)
        code = attrs.get("proj:code")
        if code is not None and str(code).upper() != f"EPSG:{epsg}":
            raise TesseraError(f"Zone {z} declares {code}, expected EPSG:{epsg}.")
        bbox = attrs.get("spatial:bbox")
        if bbox is not None:
            west, north = float(bbox[0]), float(bbox[3])
            if (c, f) != (west, north):
                if math.isclose(c - west, a / 2) and math.isclose(north - f, a / 2):
                    logger.info("Zone %s stores a pixel-centre transform; using the spatial:bbox corner.", z)
                    c, f = west, north
                else:
                    raise TesseraError(
                        f"Zone {z} transform origin ({c}, {f}) disagrees with spatial:bbox corner ({west}, {north})."
                    )
        return cls(zone=z, epsg=epsg, left=c, top=f, height=int(shape[0]), width=int(shape[1]), pixel=a)

    @property
    def crs(self) -> CRS:
        """The zone's projected CRS."""
        return CRS.from_epsg(self.epsg)

    @property
    def transform(self) -> Affine:
        """Affine transform of the whole group (pixel corner origin)."""
        return Affine(self.pixel, 0.0, self.left, 0.0, -self.pixel, self.top)

    def window_transform(self, row0: int, col0: int) -> Affine:
        """Affine transform of a window whose top-left pixel is ``(row0, col0)``."""
        return Affine(self.pixel, 0.0, self.left + col0 * self.pixel, 0.0, -self.pixel, self.top - row0 * self.pixel)

    def rowcol(self, x: float, y: float) -> tuple[int, int]:
        """Row and column of the pixel containing projected ``(x, y)``. May fall outside the grid."""
        return math.floor((self.top - y) / self.pixel), math.floor((x - self.left) / self.pixel)

    def window_for_bounds(self, xmin: float, ymin: float, xmax: float, ymax: float) -> tuple[slice, slice]:
        """Rows and columns covering projected bounds, snapped outward and clipped to the grid."""
        row0 = max(math.floor((self.top - ymax) / self.pixel), 0)
        row1 = min(math.ceil((self.top - ymin) / self.pixel), self.height)
        col0 = max(math.floor((xmin - self.left) / self.pixel), 0)
        col1 = min(math.ceil((xmax - self.left) / self.pixel), self.width)
        return slice(row0, max(row0, row1)), slice(col0, max(col0, col1))

    def to_lonlat(self) -> Transformer:
        """Transformer from this zone's CRS to lon/lat."""
        return Transformer.from_crs(self.epsg, 4326, always_xy=True)

    def from_lonlat(self) -> Transformer:
        """Transformer from lon/lat to this zone's CRS."""
        return Transformer.from_crs(4326, self.epsg, always_xy=True)


# --------------------------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Window:
    """One year of one rectangular window of a zone group.

    ``mask`` marks the pixels inside the AOI and inside this zone's nominal band; it is ``None`` for
    a plain rectangular read. Layers that were not read are ``None``.
    """

    zone: str
    year: int
    rows: slice
    cols: slice
    grid: ZoneGrid
    scales: np.ndarray
    embeddings: np.ndarray | None = None
    s2_obs_count: np.ndarray | None = None
    s1_asc_obs_count: np.ndarray | None = None
    s1_desc_obs_count: np.ndarray | None = None
    mask: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the window."""
        return int(self.scales.shape[0]), int(self.scales.shape[1])

    @property
    def transform(self) -> Affine:
        """Affine transform of the window."""
        return self.grid.window_transform(self.rows.start, self.cols.start)

    @property
    def crs(self) -> CRS:
        """Projected CRS of the window."""
        return self.grid.crs

    @property
    def valid(self) -> np.ndarray:
        """Pixels with data: a finite scale and, when embeddings were read, a non-zero vector.

        A NaN scale means no data. The embeddings fill value is 0, so an all-zero vector is never
        treated as valid even if its scale were finite.
        """
        ok = np.isfinite(self.scales)
        if self.embeddings is not None:
            ok &= np.any(self.embeddings != 0, axis=-1)
        return ok

    @property
    def radar_present(self) -> np.ndarray:
        """Pixels with at least one Sentinel-1 observation, ascending or descending, that year.

        Raises:
            TesseraError: The radar count layers were not read.
        """
        if self.s1_asc_obs_count is None or self.s1_desc_obs_count is None:
            raise TesseraError("radar_present needs s1_asc_obs_count and s1_desc_obs_count; add them to layers.")
        return (self.s1_asc_obs_count.astype(np.int32) + self.s1_desc_obs_count.astype(np.int32)) > 0

    @property
    def radar_free(self) -> np.ndarray:
        """Valid pixels embedded from optical data only (no Sentinel-1 that year)."""
        return self.valid & ~self.radar_present

    def dequantize(self) -> np.ndarray:
        """Float32 embeddings ``embeddings * scales`` of shape (H, W, 128), NaN where not valid."""
        if self.embeddings is None:
            raise TesseraError("dequantize needs the embeddings layer.")
        out = self.embeddings.astype(np.float32) * self.scales[..., None]
        out[~self.valid] = np.nan
        return out


# --------------------------------------------------------------------------------------------
# Byte counting
# --------------------------------------------------------------------------------------------


#: Last path segments of zarr metadata documents, which :class:`CountingStore` does not count.
METADATA_KEYS = frozenset({"zarr.json", ".zarray", ".zgroup", ".zattrs", ".zmetadata"})


def is_metadata_key(key: str) -> bool:
    """Whether a store key names a zarr metadata document rather than chunk data."""
    return key.rsplit("/", 1)[-1] in METADATA_KEYS


@dataclasses.dataclass
class ByteCounter:
    """Running total of chunk payload bytes returned by the store."""

    bytes: int = 0
    requests: int = 0

    _lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, buf: Buffer | None) -> None:
        """Count one returned buffer. Safe to call from several threads, as zarr does."""
        if buf is not None:
            n = len(buf)
            with self._lock:
                self.bytes += n
                self.requests += 1


class CountingStore(WrapperStore[Any]):
    """A zarr store wrapper that counts the chunk bytes every read returns.

    zarr reads through ``get`` and ``get_partial_values`` on the async path and ``get_sync`` on the
    sync path; all three are counted. The coalescing range readers (``get_ranges`` and
    ``get_ranges_sync``, which zarr 3.4 sharding uses for partial-shard reads) run zarr's own
    coalescer above this wrapper, so each merged fetch, gaps included, passes through the counted
    ``get`` or ``get_sync``. Shard indexes are chunk-key reads and are counted. Copies made by
    ``with_read_only`` share the same counter.

    Metadata documents (``zarr.json``) are not counted. Icechunk serves them from the snapshot its
    session already holds, so they cost no store traffic, and geotessera opens an array afresh on
    every read, so counting them would add the same in-memory document again on each read (about
    1 KB per array, and 35 KB per zone group, whose attributes carry the run provenance). Icechunk's
    own manifest and snapshot fetches happen below the zarr store interface and are not counted
    either; they are small next to chunk payloads.
    """

    def __init__(self, store: Any, counter: ByteCounter | None = None) -> None:  # noqa: ANN401
        super().__init__(store)
        self.counter = counter if counter is not None else ByteCounter()

    def _with_store(self, store: Any) -> Self:  # noqa: ANN401
        return type(self)(store, self.counter)

    def _count(self, key: str, buf: Buffer | None) -> None:
        if not is_metadata_key(key):
            self.counter.add(buf)

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        """Read one key and count it."""
        buf = await self._store.get(key, prototype, byte_range)
        self._count(key, buf)
        return buf

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        """Read several ranges and count them."""
        key_ranges = list(key_ranges)
        bufs = await self._store.get_partial_values(prototype, key_ranges)
        for (key, _), buf in zip(key_ranges, bufs, strict=True):
            self._count(key, buf)
        return list(bufs)

    def get_sync(
        self, key: str, *, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Synchronous read of one key, counted."""
        buf = self._store.get_sync(key, prototype=prototype, byte_range=byte_range)
        self._count(key, buf)
        return buf

    # The two range readers coalesce here, over the counted get / get_sync, instead of forwarding to
    # the wrapped store, so the gap bytes inside a merged fetch are counted too. ``None`` keeps zarr's
    # defaults, as WrapperStore documents.
    async def get_ranges(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_concurrency: int | None = None,
        max_gap_bytes: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> AsyncIterator[Sequence[tuple[int, Buffer | None]]]:
        """Read coalesced ranges through the counted ``get``."""
        async for group in Store.get_ranges(
            self,
            key,
            byte_ranges,
            prototype=prototype,
            **_coalesce_options(
                max_concurrency=max_concurrency, max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes
            ),
        ):
            yield group

    def get_ranges_sync(
        self,
        key: str,
        byte_ranges: Sequence[ByteRequest | None],
        *,
        prototype: BufferPrototype,
        max_gap_bytes: int | None = None,
        max_coalesced_bytes: int | None = None,
    ) -> Sequence[tuple[int, Buffer | None]]:
        """Read coalesced ranges through the counted ``get_sync``."""
        return Store.get_ranges_sync(
            self,
            key,
            byte_ranges,
            prototype=prototype,
            **_coalesce_options(max_gap_bytes=max_gap_bytes, max_coalesced_bytes=max_coalesced_bytes),
        )


def _coalesce_options(**options: int | None) -> dict[str, int]:
    return {k: v for k, v in options.items() if v is not None}


def _icechunk_message(exc: BaseException) -> str:
    """An icechunk error's message without the Rust stack it appends after a ``context:`` line.

    Args:
        exc: The error icechunk raised.

    Returns:
        The message, stripped; the error's class name when the message is empty.
    """
    text = str(exc).split("\ncontext:", 1)[0].strip()
    return text or type(exc).__name__


# --------------------------------------------------------------------------------------------
# The store, through geotessera
# --------------------------------------------------------------------------------------------


class CountedIcechunkStore(IcechunkStore):
    """geotessera's :class:`~geotessera.icechunk.IcechunkStore` with every read it makes counted.

    The constructor is geotessera's own: it opens a read-only session at ``snapshot_id``, or at the
    head of ``branch``, and lists the zone groups. Then the public ``root`` is replaced by the same
    group opened through a :class:`CountingStore`. geotessera reads every zone group, attribute and
    array through ``root`` and builds a zone's merged view on the zone's first ``open_zone`` or
    ``zone_arrays``, so every chunk read after construction is counted. The few reads geotessera
    makes inside its constructor (the root group, the group list and the first group's time
    coordinate) come before the swap and are not.

    Args:
        location: ``s3://`` URL or local path of an Icechunk repository.
        branch: Branch to read when no snapshot is given.
        snapshot_id: Snapshot to read.
        counter: Counter to add to; a new one by default.
        wrap: Applied to the session store inside the counter, so a wrapper (the tests' key log)
            sees every read the counter sees.
    """

    def __init__(
        self,
        location: str | os.PathLike[str],
        *,
        branch: str = "main",
        snapshot_id: str | None = None,
        counter: ByteCounter | None = None,
        wrap: Callable[[Store], Store] | None = None,
    ) -> None:
        super().__init__(location, branch=branch, snapshot_id=snapshot_id)
        self.counter = counter if counter is not None else ByteCounter()
        inner = self.session.store if wrap is None else wrap(self.session.store)
        self.counting = CountingStore(inner, self.counter)
        self.root = zarr.open_group(self.counting, mode="r", use_consolidated=False)

    @property
    def snapshot_id(self) -> str:
        """The snapshot this store reads, whether it was asked for or resolved from the branch."""
        return str(self.session.snapshot_id)


def check_dataset(gt: Any) -> dict[str, str]:  # noqa: ANN401
    """The store identity ``GeoTesseraZarr`` reports, once it is known to be the dClimate v1.1 dataset.

    Args:
        gt: A ``geotessera.GeoTesseraZarr`` opened on :data:`STORE_URL`.

    Returns:
        ``url``, ``dataset``, ``model_version`` and ``build_version``, as geotessera gives them.

    Raises:
        DatasetMismatchError: ``gt.dataset`` is None or names another dataset.
    """
    dataset = getattr(gt, "dataset", None)
    name = None if dataset is None else getattr(dataset, "name", None)
    if name != DATASET_NAME:
        raise DatasetMismatchError(None if name is None else str(name))
    return {
        "url": str(gt.url),
        "dataset": str(name),
        "model_version": str(gt.model_version),
        "build_version": str(gt.build_version),
    }


def _unwrap(store: Store) -> tuple[Store, Callable[[Store], Store] | None]:
    """Split a chain of zarr wrapper stores into the store at the bottom and a function re-applying the chain.

    Each wrapper is re-applied with ``WrapperStore._with_store``, the hook zarr itself uses to copy a
    wrapper onto another store (``with_read_only``), so a wrapper keeps whatever it shares between
    copies, such as a key log's list.
    """
    chain: list[WrapperStore[Any]] = []
    while isinstance(store, WrapperStore) and len(chain) < 16:
        chain.append(store)
        store = store._store
    if not chain:
        return store, None

    def rewrap(inner: Store) -> Store:
        for wrapper in reversed(chain):
            inner = wrapper._with_store(inner)
        return inner

    return store, rewrap


_COPIES: dict[str, Path] = {}
_COPIES_LOCK = threading.Lock()
_COPIES_ROOT: list[Path] = []
#: Writes in flight while a store is copied into a local Icechunk repository.
_COPY_CONCURRENCY = 64


def _copies_root() -> Path:
    """The temporary directory holding this process's local store copies, removed at exit."""
    if not _COPIES_ROOT:
        root = Path(tempfile.mkdtemp(prefix="tessera-apps-stores-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
        _COPIES_ROOT.append(root)
    return _COPIES_ROOT[0]


async def _store_items(store: Store) -> list[tuple[str, Buffer]]:
    prototype = default_buffer_prototype()
    keys = sorted([key async for key in store.list()])
    items = []
    for key in keys:
        buf = await store.get(key, prototype)
        if buf is not None:
            items.append((key, buf))
    return items


async def _write_items(target: Store, items: Sequence[tuple[str, Buffer]]) -> None:
    # Icechunk needs a node's metadata before its children and chunks: metadata goes first, parents
    # before children, then the chunks concurrently.
    meta = sorted((kv for kv in items if is_metadata_key(kv[0])), key=lambda kv: (kv[0].count("/"), kv[0]))
    for key, buf in meta:
        await target.set(key, buf)
    gate = asyncio.Semaphore(_COPY_CONCURRENCY)

    async def put(key: str, buf: Buffer) -> None:
        async with gate:
            await target.set(key, buf)

    await asyncio.gather(*(put(key, buf) for key, buf in items if not is_metadata_key(key)))


def _materialise(store: Store) -> Path:
    """Copy a zarr store, key for key and byte for byte, into a local Icechunk repository.

    The copy lives in a temporary directory removed when the process exits. A store whose keys and
    bytes match an earlier copy reuses it, so the tests' shared stores are copied once.

    Args:
        store: Any listable zarr store (not a wrapper; see :meth:`TesseraReader.from_store`).

    Returns:
        The repository's path, for :class:`CountedIcechunkStore`.

    Raises:
        TesseraError: The store cannot be listed, or Icechunk refuses one of its keys.
    """
    import icechunk

    if not store.supports_listing:
        raise TesseraError(f"{type(store).__name__} cannot be listed, so it cannot be copied for the reader.")
    items = sync(_store_items(store))
    digest = hashlib.sha256()
    for key, buf in items:
        data = buf.as_numpy_array()
        digest.update(key.encode())
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    name = digest.hexdigest()[:32]
    with _COPIES_LOCK:
        path = _COPIES.get(name)
        if path is None:
            path = _copies_root() / name / "store.icechunk"
            try:
                repo = icechunk.Repository.create(icechunk.local_filesystem_storage(str(path)))
                session = repo.writable_session("main")
                sync(_write_items(session.store, items))
                session.commit(f"tessera_apps.io: a copy of a {type(store).__name__} for TesseraReader.from_store")
            except (icechunk.IcechunkError, ValueError) as exc:
                shutil.rmtree(path.parent, ignore_errors=True)
                what = type(store).__name__
                reason = _icechunk_message(exc)
                raise TesseraError(f"Cannot copy the {what} into a local Icechunk repository: {reason}.") from exc
            _COPIES[name] = path
    return path


@dataclasses.dataclass(frozen=True)
class _ZoneView:
    """Where one zone group sits on geotessera's merged view of its zone: its arrays and the offsets."""

    arrays: dict[str, Any]
    row_offset: int
    col_offset: int


def _whole_pixels(value: float, what: str) -> int:
    """An offset in pixels that must be whole, or a TesseraError naming it."""
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        raise TesseraError(f"{what} is {value} pixels, not a whole number; the grids are out of phase.")
    return int(rounded)


def _strip_edges(start: int, stop: int, own0: int, step: int) -> list[int]:
    """Row edges of the strips :meth:`TesseraReader._read_job` reads embeddings in.

    From the group's first own row on (see :meth:`TesseraReader.own_rows`) the edges are multiples
    of ``step``, a whole number of the group's chunk rows. Above it, where geotessera serves a south
    group's rows from the north group, they step up from ``own0``: on the live grids the north group
    ends on the equator at a chunk edge, so those are its chunk-row edges. Either way no source chunk
    row is split between two strips.

    Args:
        start: First group row of the window.
        stop: Row past the window's last.
        own0: The group's first own row.
        step: Strip height in rows, a multiple of the chunk height.

    Returns:
        ``[start, ..., stop]``, ascending; just ``[start]`` when the window is empty.
    """
    edges = [start]
    r = start
    while r < stop:
        r = min(stop, own0 - ((own0 - r - 1) // step) * step if r < own0 else (r // step + 1) * step)
        edges.append(r)
    return edges


# --------------------------------------------------------------------------------------------
# Readers
# --------------------------------------------------------------------------------------------


class Reader(Protocol):
    """What the apps need from a backend. :class:`TesseraReader` is the implementation."""

    @property
    def bytes_read(self) -> int:
        """Payload bytes fetched from the store so far."""
        ...

    def grid(self, zone: str) -> ZoneGrid:
        """Pixel grid of a zone group."""
        ...

    def chunk_shape(self, zone: str) -> tuple[int, int]:
        """(rows, cols) of one inner chunk in a zone group."""
        ...

    def years_complete(self, zone: str) -> list[int]:
        """Years a zone has published."""
        ...

    def read_windows(
        self, zone: str, years: Sequence[int], rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> dict[int, Window]:
        """Read several years of one rectangular window."""
        ...

    def read_window(
        self, zone: str, year: int, rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> Window:
        """Read one year of a rectangular window."""
        ...

    def provenance(self) -> dict[str, Any]:
        """Store identity for reports."""
        ...


@dataclasses.dataclass(frozen=True)
class _Job:
    """One zarr read: a layer of one year over one window."""

    zone: str
    name: str
    year: int
    t: int
    r0: int
    r1: int
    c0: int
    c1: int


class TesseraReader:
    """Reader for the published store through geotessera (or any zarr store with the same layout).

    Use :meth:`open` for the live store and :meth:`from_store` for any other zarr store, such as the
    in-memory stores the tests build. Both read through a :class:`CountedIcechunkStore`, so every
    chunk read goes through geotessera's merged zone arrays and nothing goes through Dask.
    """

    def __init__(
        self,
        store: CountedIcechunkStore,
        *,
        provenance: dict[str, Any] | None = None,
        cache: ChunkCache | None = None,
    ) -> None:
        self._store = store
        self._counter = store.counter
        self._provenance = dict(provenance or {})
        if cache is not None and cache.namespace == ChunkCache.UNPINNED:
            snapshot = self._provenance.get("snapshot_id")
            if not snapshot:
                raise ValueError(
                    "A chunk cache needs a namespace that identifies the data: pass ChunkCache(..., namespace=...) "
                    "or a snapshot_id, so two stores never share cached chunks."
                )
            cache = cache.for_namespace(str(snapshot))
        self._cache = cache
        self._groups: dict[str, zarr.Group] = {}
        self._arrays: dict[tuple[str, str], zarr.Array] = {}
        self._attrs: dict[str, dict[str, Any]] = {}
        self._grids: dict[str, ZoneGrid] = {}
        self._views: dict[str, _ZoneView] = {}
        self._incomplete: dict[str, list[int]] | None = None
        self._lock = threading.Lock()

    @classmethod
    def open(
        cls,
        *,
        snapshot_id: str | None = None,
        branch: str = "main",
        cache: ChunkCache | None = None,
    ) -> TesseraReader:
        """Open the published store anonymously, through geotessera.

        ``GeoTesseraZarr(store_url=STORE_URL)`` gives the store's identity and must name the dClimate
        v1.1 dataset before anything else happens. The reads then go through a
        :class:`CountedIcechunkStore` at ``snapshot_id``, or at the head of ``branch`` as it stands
        now; either way the snapshot read is recorded in :meth:`provenance`.

        Args:
            snapshot_id: Pin an Icechunk snapshot. Pin one for reproducible runs; the id is recorded
                in :meth:`provenance` either way.
            branch: Branch to read when no snapshot is pinned.
            cache: Optional on-disk chunk cache. The reader uses it under a namespace named after the
                snapshot (``reader.cache``), so a new snapshot never serves stale chunks; hit and miss
                counts are shared with the object passed in.

        Raises:
            StoreUnavailableError: The store cannot be reached, ``snapshot_id`` is not a valid id or
                not one of the store's snapshots, or ``branch`` does not exist. The message is
                icechunk's, without its internal stack.
            DatasetMismatchError: geotessera does not identify the store as ``1.1-dclimate``, or the
                snapshot read holds another model than the one geotessera reports.
        """
        import icechunk
        from geotessera import GeoTesseraZarr

        where = f"snapshot {snapshot_id}" if snapshot_id else f"branch {branch!r}"

        def unavailable(exc: BaseException) -> StoreUnavailableError:
            hint = f"Check that this machine can reach S3 in {STORE_REGION} (no AWS credentials are needed)"
            if snapshot_id:
                hint += " and that the snapshot id is one of the store's (tessera-apps info prints the current one)"
            return StoreUnavailableError(
                f"Cannot open the store {STORE_URL} at {where}: {_icechunk_message(exc)}. {hint}."
            )

        try:
            gt = GeoTesseraZarr(store_url=STORE_URL)
        except (icechunk.IcechunkError, ValueError) as exc:
            raise unavailable(exc) from exc
        identity = check_dataset(gt)
        try:
            store = CountedIcechunkStore(STORE_URL, branch=branch, snapshot_id=snapshot_id)
        except (icechunk.IcechunkError, ValueError) as exc:
            raise unavailable(exc) from exc
        except KeyError as exc:
            raise TesseraError(f"geotessera cannot open {STORE_URL} at {where}: {exc!r}.") from exc
        snapshot = store.snapshot_id
        model = store.root.attrs.get("geoemb:model")
        if model is not None and str(model) != identity["model_version"]:
            raise DatasetMismatchError(
                identity["dataset"],
                f"Snapshot {snapshot} of {STORE_URL} holds model {model}, but geotessera reports the store's model as "
                f"{identity['model_version']} (from the head of main).",
            )
        # GeoTesseraZarr reads build_version at the head of main, which a pinned snapshot (or a head
        # that moved since) need not share. The store block describes the data read, so it takes the
        # snapshot's own value and keeps geotessera's beside it when the two differ.
        build = store.root.attrs.get("geoemb:build_version")
        if build is not None and str(build) != identity["build_version"]:
            logger.warning(
                "Snapshot %s holds build %s; geotessera reports build %s from the head of main. Recording %s.",
                snapshot,
                build,
                identity["build_version"],
                build,
            )
            identity = {**identity, "build_version": str(build), "head_build_version": identity["build_version"]}
        provenance = {
            "backend": "geotessera",
            **identity,
            "snapshot_id": snapshot,
            "branch": None if snapshot_id else branch,
            "icechunk_version": getattr(icechunk, "__version__", None),
            "zarr_version": zarr.__version__,
        }
        if cache is not None:
            cache = cache.for_namespace(snapshot)
        logger.info("Reading %s (%s) at snapshot %s through geotessera.", STORE_URL, identity["dataset"], snapshot)
        return cls(store, provenance=provenance, cache=cache)

    @classmethod
    def from_store(cls, store: Any, *, cache: ChunkCache | None = None, **provenance: Any) -> TesseraReader:  # noqa: ANN401
        """Read any zarr store with the published layout (for example a ``MemoryStore`` in tests).

        The store is copied, key for key, into a local Icechunk repository (once per distinct
        content, see :func:`_materialise`), which is then read exactly as :meth:`open` reads the
        published store: through a :class:`CountedIcechunkStore` and geotessera's zone arrays. Wrapper
        stores around ``store`` (zarr ``WrapperStore`` subclasses, such as the tests' key log) are
        re-applied around the copy's session store, inside the byte counter, so they see every read
        the reader makes. The dataset check of :meth:`open` does not apply: a local copy is not a
        published dataset, and no ``GeoTesseraZarr`` is opened.

        A cache must carry an explicit namespace, or ``snapshot_id`` must be passed in
        ``provenance`` (the copy's own snapshot id names a throwaway copy, not the data).
        """
        base, rewrap = _unwrap(store)
        try:
            counted = CountedIcechunkStore(_materialise(base), wrap=rewrap)
        except (KeyError, ValueError) as exc:
            raise TesseraError(f"geotessera cannot open the copy of the {type(base).__name__}: {exc!r}.") from exc
        prov = {"backend": type(store).__name__, **provenance}
        return cls(counted, provenance=prov, cache=cache)

    # ---- metadata -------------------------------------------------------------------------

    @property
    def bytes_read(self) -> int:
        """Chunk payload bytes fetched from the store so far (cache hits are free; see :class:`CountingStore`)."""
        return self._counter.bytes

    @property
    def cache(self) -> ChunkCache | None:
        """The chunk cache in use, if any."""
        return self._cache

    @property
    def store(self) -> CountedIcechunkStore:
        """The geotessera store every read goes through."""
        return self._store

    def _group(self, zone: str) -> zarr.Group:
        z = canonicalize_zone(zone)
        if z not in self._groups:
            if z not in self._store.groups:
                raise TesseraError(f"Zone group {z} is not in the store.")
            node = self._store.root[z]
            if not isinstance(node, zarr.Group):
                raise TesseraError(f"{z} is not a zarr group.")
            self._groups[z] = node
        return self._groups[z]

    def _array(self, zone: str, name: str) -> zarr.Array:
        key = (canonicalize_zone(zone), name)
        if key not in self._arrays:
            node = self._group(zone)[name]
            if not isinstance(node, zarr.Array):
                raise TesseraError(f"{zone}/{name} is not a zarr array.")
            self._arrays[key] = node
        return self._arrays[key]

    def zone_attrs(self, zone: str) -> dict[str, Any]:
        """Attributes of a zone group, read once and cached."""
        z = canonicalize_zone(zone)
        if z not in self._attrs:
            self._attrs[z] = dict(self._group(z).attrs)
        return self._attrs[z]

    def root_attrs(self) -> dict[str, Any]:
        """Attributes of the root group (model provenance)."""
        return dict(self._store.root.attrs)

    def grid(self, zone: str) -> ZoneGrid:
        """Pixel grid of a zone group.

        Raises:
            TesseraError: The group is not in the store, its transform is unusable (see
                :meth:`ZoneGrid.from_attrs`), or its ``spatial:shape``, which geotessera builds its
                merged view from, disagrees with the arrays.
        """
        z = canonicalize_zone(zone)
        if z not in self._grids:
            shape = self._array(z, "scales").shape
            attrs = self.zone_attrs(z)
            grid = ZoneGrid.from_attrs(z, attrs, (shape[1], shape[2]))
            declared = attrs.get("spatial:shape")
            if declared is not None and [int(v) for v in declared] != [grid.height, grid.width]:
                raise TesseraError(
                    f"Zone {z} declares spatial:shape {list(declared)} but its arrays are {grid.height} x {grid.width}."
                )
            self._grids[z] = grid
        return self._grids[z]

    def chunk_shape(self, zone: str) -> tuple[int, int]:
        """(rows, cols) of one inner chunk of the embeddings array (256 x 256 in the live store)."""
        chunks = self._array(zone, "embeddings").chunks
        return int(chunks[1]), int(chunks[2])

    def store_years(self) -> list[int]:
        """Calendar years on the store's time axis, as geotessera decodes them (``IcechunkStore.years``).

        Raises:
            TesseraError: The axis is not consecutive years from 2017: the store layout has changed.
        """
        years = [int(y) for y in self._store.years]
        expected = list(range(FIRST_YEAR, FIRST_YEAR + len(years)))
        if not years or years != expected:
            raise TesseraError(
                f"time coordinate decodes to years {years}, expected consecutive years from {FIRST_YEAR}; the store "
                "layout has changed."
            )
        return years

    def _incomplete_years(self) -> dict[str, list[int]]:
        """The store's ``incomplete_years()`` (geotessera), read once: attributes of every group, no chunk."""
        if self._incomplete is None:
            found = self._store.incomplete_years()
            self._incomplete = {canonicalize_zone(str(g)): [int(y) for y in ys] for g, ys in found.items()}
        return self._incomplete

    def years_complete(self, zone: str) -> list[int]:
        """Years a zone group has published: the store's years less those geotessera finds incomplete.

        Raises:
            TesseraError: The group is not in the store, or has no ``years_complete`` attribute
                (which geotessera would read as complete).
        """
        z = canonicalize_zone(zone)
        if self.zone_attrs(z).get("years_complete") is None:
            raise TesseraError(f"Zone {z} has no years_complete attribute.")
        missing = set(self._incomplete_years().get(z, []))
        return [y for y in self.store_years() if y not in missing]

    def year_index(self, zone: str, year: int) -> int:
        """Position of a calendar year on a zone's time axis."""
        z = canonicalize_zone(zone)
        self._group(z)
        try:
            return self.store_years().index(year)
        except ValueError as exc:
            raise YearsNotPublishedError({z: [year]}) from exc

    def check_years(self, zones: Iterable[str], years: Iterable[int]) -> None:
        """Raise unless every year is in ``years_complete`` for every zone. Reads attributes only.

        Raises:
            YearsNotPublishedError: Names each missing zone-year.
        """
        reader_check_years(self, zones, years)

    def provenance(self) -> dict[str, Any]:
        """Store and model identity for reports.

        For the published store: geotessera's ``url``, ``dataset``, ``model_version`` and
        ``build_version``, and the Icechunk ``snapshot_id`` read, which is the data version
        (``checkpoint_id`` names the model weights and never changes). geotessera takes the two
        versions from the head of main; the model must match the snapshot read (:meth:`open` refuses
        another), and when the snapshot's ``geoemb:build_version`` differs, ``build_version`` is the
        snapshot's and ``head_build_version`` keeps geotessera's. The model keys of the root
        attributes at that snapshot are added for the report's ``model`` block.
        """
        attrs = self.root_attrs()
        out = dict(self._provenance)
        for key in ("checkpoint_id", "geoemb:model", "geoemb:dimensions", "geoemb:build_version"):
            if key in attrs:
                out[key] = attrs[key]
        return out

    def run_commits(self, zone: str, years: Iterable[int]) -> dict[int, str | None]:
        """Pipeline commit that assembled each zone-year, from ``attrs["runs"]``."""
        runs = self.zone_attrs(zone).get("runs") or {}
        out: dict[int, str | None] = {}
        for y in years:
            run = runs.get(str(y)) or {}
            out[y] = (run.get("code") or {}).get("commit")
        return out

    # ---- geotessera's merged view -----------------------------------------------------------

    def _zone_view(self, zone: str) -> _ZoneView:
        """The zone group's place on geotessera's merged zone, built on the zone's first read."""
        z = canonicalize_zone(zone)
        view = self._views.get(z)
        if view is None:
            with self._lock:
                view = self._views.get(z)
                if view is None:
                    view = self._build_view(z)
                    self._views[z] = view
        return view

    def _build_view(self, z: str) -> _ZoneView:
        self.grid(z)
        number = int(z[:2])
        try:
            merged = self._store.open_zone(number)
            arrays = dict(self._store.zone_arrays(number))
        except (KeyError, ValueError) as exc:
            raise TesseraError(f"geotessera cannot build its view of zone {number:02d}: {exc}") from exc
        for name, dims in (("embeddings", ("time", "band", "y", "x")), ("scales", ("time", "y", "x"))):
            if name not in arrays or tuple(arrays[name].dims) != dims:
                got = None if name not in arrays else tuple(arrays[name].dims)
                raise TesseraError(f"geotessera's view of zone {number:02d} has {name} dims {got}, expected {dims}.")
        px, _, west, _, _, top = (float(v) for v in merged.attrs["spatial:transform"])
        # The group's transform as geotessera reads it (not the corner ZoneGrid may correct it to),
        # in the northern CRS, so the offsets are taken between the two grids geotessera itself aligns.
        raw = [float(v) for v in self.zone_attrs(z)["spatial:transform"]]
        group_top = raw[5] - (FALSE_NORTHING if z.endswith("S") else 0.0)
        row_offset = _whole_pixels((top - group_top) / px, f"The row offset of {z} on geotessera's zone {number:02d}")
        col_offset = _whole_pixels((raw[2] - west) / px, f"The column offset of {z} on geotessera's zone {number:02d}")
        logger.debug("%s on geotessera zone %02d: offset (%d, %d)", z, number, row_offset, col_offset)
        return _ZoneView(arrays=arrays, row_offset=row_offset, col_offset=col_offset)

    def own_rows(self, zone: str) -> tuple[int, int]:
        """The group rows geotessera reads from the group itself, as ``(start, stop)``. Attributes only.

        geotessera takes rows centred north of the equator from the north group and the rest from the
        south group, judging centres by each group's ``spatial:transform`` as stored. Every row of a
        live north group qualifies; a south group's first 1,472 rows (north of the equator) do not.
        """
        z = canonicalize_zone(zone)
        grid = self.grid(z)
        t = [float(v) for v in self.zone_attrs(z)["spatial:transform"]]
        group_top = t[5] - (FALSE_NORTHING if z.endswith("S") else 0.0)
        # Group row i is centred at group_top - (i + 0.5) * pixel: north of the equator for i < edge.
        edge = group_top / t[0] - 0.5
        if z.endswith("N"):
            return 0, min(grid.height, max(0, math.ceil(edge)))
        return min(grid.height, max(0, math.floor(edge) + 1)), grid.height

    def _check_served(self, zone: str, r1: int) -> None:
        """Refuse north-group rows that geotessera does not serve from the north group.

        Live north groups end at northing 0, so geotessera serves every row from them. Snapshots before
        ``B6AFRJGX5XQC0A716PA0`` register ``spatial:transform`` at the pixel centre, half a pixel off,
        which puts a north group's bottom row on the equator, and geotessera then serves it from
        neither group. South-group rows north of the equator are allowed: geotessera serves them from
        the north group (see the module docstring).
        """
        stop = self.own_rows(zone)[1]
        if zone.endswith("N") and r1 > stop:
            raise TesseraError(
                f"Rows {stop} to {r1} of zone {zone} are centred on or south of the equator by its spatial:transform, "
                "so geotessera does not read them from that group. Snapshots before B6AFRJGX5XQC0A716PA0 register "
                "the transform at the pixel centre, which puts a north group's bottom row on the equator; pin "
                "B6AFRJGX5XQC0A716PA0 or a later snapshot."
            )

    # ---- reads ----------------------------------------------------------------------------

    def read_window(
        self, zone: str, year: int, rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> Window:
        """Read one year of a rectangular window.

        Args:
            zone: Zone group, for example ``"10N"``.
            year: Calendar year; must be in the zone's ``years_complete``.
            rows: Row range on the zone grid (row 0 is the north edge).
            cols: Column range on the zone grid.
            layers: Arrays to read. ``scales`` is always read.

        Raises:
            YearsNotPublishedError: The year is not published for this zone.
            TesseraError: The window is empty or outside the grid, or a layer name is unknown.
        """
        return self.read_windows(zone, [year], rows, cols, layers=layers)[year]

    def read_windows(
        self, zone: str, years: Sequence[int], rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> dict[int, Window]:
        """Read several years of one window, fetching every (year, layer) concurrently.

        All years are checked before anything is fetched. See :meth:`read_window` for arguments.
        """
        z = canonicalize_zone(zone)
        unknown = set(layers) - set(ALL_LAYERS)
        if unknown:
            raise TesseraError(f"Unknown layers {sorted(unknown)}; choose from {ALL_LAYERS}.")
        years = list(dict.fromkeys(years))
        self.check_years([z], years)
        grid = self.grid(z)
        r0, r1, c0, c1 = _check_window(grid, rows, cols)
        if self._cache is not None:  # a miss reads the chunk-aligned cover
            ch = self.chunk_shape(z)[0]
            self._check_served(z, min(-(-r1 // ch) * ch, grid.height))
        else:
            self._check_served(z, r1)
        names = ["scales", *[n for n in layers if n != "scales"]]
        jobs = [_Job(z, n, y, self.year_index(z, y), r0, r1, c0, c1) for y in years for n in names]
        arrays = self._run(jobs)
        out: dict[int, Window] = {}
        for y in years:
            data = {j.name: a for j, a in zip(jobs, arrays, strict=True) if j.year == y}
            out[y] = Window(zone=z, year=y, rows=slice(r0, r1), cols=slice(c0, c1), grid=grid, **data)
        return out

    def _run(self, jobs: Sequence[_Job]) -> list[np.ndarray]:
        """Run read jobs on the shared thread pool, through the cache when there is one."""
        fn = self._read_job_cached if self._cache is not None else self._read_job
        # Never wait on the pool from inside one of its own tasks: run in this thread instead.
        if len(jobs) == 1 or getattr(_LOCAL, "in_read_pool", False):
            return [fn(job) for job in jobs]

        def task(job: _Job) -> np.ndarray:
            _LOCAL.in_read_pool = True
            try:
                return fn(job)
            finally:
                _LOCAL.in_read_pool = False

        return list(_POOL.map(task, jobs))

    def _read_job(self, job: _Job) -> np.ndarray:
        """One layer of one year over one window, from geotessera's merged zone arrays.

        The window moves onto the merged grid by the zone view's offsets. Group rows that land above
        the merged grid's top edge (a south group's rows north of the equator, in a zone with no
        north group) are served by no group and come back as fill, as geotessera returns any row no
        group serves; a window reaching past the merged grid's other edges is refused. Embeddings
        are read in strips of whole chunk rows, as many as fit in :data:`_STRIP_BYTES` and at least
        one (:func:`_strip_edges`), each strip's ``(band, y, x)`` block transposed into one
        preallocated C-contiguous ``(y, x, band)`` output, so the transient memory is one strip, not
        the window; at most ``TESSERA_APPS_EMBEDDING_READS`` (4) of those reads run at once.

        Raises:
            TesseraError: geotessera's view has no such array, or the window falls outside it.
        """
        view = self._zone_view(job.zone)
        array = view.arrays.get(job.name)
        if array is None:
            raise TesseraError(f"geotessera's view of zone {job.zone[:2]} has no {job.name} array.")
        height, width = (int(array.shape[array.dims.index(d)]) for d in ("y", "x"))
        ro, co = view.row_offset, view.col_offset
        if job.r1 + ro > height or job.c0 + co < 0 or job.c1 + co > width:
            raise TesseraError(
                f"Rows {job.r0}:{job.r1}, cols {job.c0}:{job.c1} of zone {job.zone} fall outside geotessera's "
                f"{height} x {width} view of zone {job.zone[:2]} (the group sits at offset ({ro}, {co}) on it)."
            )
        n = job.r1 - job.r0
        lead = min(max(0, -(job.r0 + ro)), n)  # rows above the merged grid
        cols = slice(job.c0 + co, job.c1 + co)
        if job.name != "embeddings":
            if lead == 0:
                return np.asarray(array[job.t, job.r0 + ro : job.r1 + ro, cols])
            out = np.full((n, job.c1 - job.c0), array.fill_value, dtype=array.dtype)
            if lead < n:
                out[lead:] = array[job.t, job.r0 + lead + ro : job.r1 + ro, cols]
            return out
        bands = int(array.shape[array.dims.index("band")])
        chunk = self.chunk_shape(job.zone)[0]
        per_strip = max(1, _STRIP_BYTES // (chunk * (job.c1 - job.c0) * bands * array.dtype.itemsize))
        edges = _strip_edges(job.r0 + lead, job.r1, self.own_rows(job.zone)[0], chunk * per_strip)
        with _EMBEDDING_READS:
            out = np.empty((n, job.c1 - job.c0, bands), dtype=array.dtype)
            out[:lead] = array.fill_value
            for r, stop in itertools.pairwise(edges):
                out[r - job.r0 : stop - job.r0] = np.moveaxis(array[job.t, :, r + ro : stop + ro, cols], 0, -1)
        return out

    def _read_job_cached(self, job: _Job) -> np.ndarray:
        """Serve a job from whole cached chunks; on any miss, fetch the chunk-aligned cover in one call.

        Cached chunks hold what an uncached read of them returns. So in a south group whose first own
        row is inside a chunk row (row 1472 of chunk row 5, live), a cover that takes that chunk row
        also fetches the north group's chunk serving its rows above the equator, though only rows
        inside the zone's band are ever used.
        """
        cache = self._cache
        assert cache is not None
        ch, cw = self.chunk_shape(job.zone)
        grid = self.grid(job.zone)
        arr = self._array(job.zone, job.name)
        tail = tuple(arr.shape[3:])
        chunk_ids = [
            (cr, cc)
            for cr in range(job.r0 // ch, (job.r1 - 1) // ch + 1)
            for cc in range(job.c0 // cw, (job.c1 - 1) // cw + 1)
        ]

        def bounds(cr: int, cc: int) -> tuple[int, int, int, int]:
            return cr * ch, min((cr + 1) * ch, grid.height), cc * cw, min((cc + 1) * cw, grid.width)

        blocks: dict[tuple[int, int], np.ndarray] = {}
        for cr, cc in chunk_ids:
            br0, br1, bc0, bc1 = bounds(cr, cc)
            got = cache.get(
                ChunkKey(job.zone, job.year, cr, cc), job.name, shape=(br1 - br0, bc1 - bc0, *tail), dtype=arr.dtype
            )
            if got is not None:
                blocks[(cr, cc)] = got
        if len(blocks) < len(chunk_ids):
            ar0, ac0 = (job.r0 // ch) * ch, (job.c0 // cw) * cw
            ar1, ac1 = min(((job.r1 - 1) // ch + 1) * ch, grid.height), min(((job.c1 - 1) // cw + 1) * cw, grid.width)
            cover = self._read_job(dataclasses.replace(job, r0=ar0, r1=ar1, c0=ac0, c1=ac1))
            for cr, cc in chunk_ids:
                if (cr, cc) in blocks:
                    continue
                br0, br1, bc0, bc1 = bounds(cr, cc)
                block = np.ascontiguousarray(cover[br0 - ar0 : br1 - ar0, bc0 - ac0 : bc1 - ac0])
                cache.put(ChunkKey(job.zone, job.year, cr, cc), job.name, block)
                blocks[(cr, cc)] = block
        out = np.empty((job.r1 - job.r0, job.c1 - job.c0, *tail), dtype=arr.dtype)
        for (cr, cc), block in blocks.items():
            br0, br1, bc0, bc1 = bounds(cr, cc)
            ir0, ir1 = max(job.r0, br0), min(job.r1, br1)
            ic0, ic1 = max(job.c0, bc0), min(job.c1, bc1)
            out[ir0 - job.r0 : ir1 - job.r0, ic0 - job.c0 : ic1 - job.c0] = block[
                ir0 - br0 : ir1 - br0, ic0 - bc0 : ic1 - bc0
            ]
        return out


def _check_window(grid: ZoneGrid, rows: slice, cols: slice) -> tuple[int, int, int, int]:
    r0, r1 = rows.start, rows.stop
    c0, c1 = cols.start, cols.stop
    if None in (r0, r1, c0, c1) or rows.step not in (None, 1) or cols.step not in (None, 1):
        raise TesseraError("Windows need explicit start and stop and no step.")
    if not (0 <= r0 < r1 <= grid.height and 0 <= c0 < c1 <= grid.width):
        raise TesseraError(
            f"Window rows {r0}:{r1}, cols {c0}:{c1} is empty or outside zone {grid.zone} "
            f"({grid.height} x {grid.width})."
        )
    return int(r0), int(r1), int(c0), int(c1)


# --------------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------------


def iter_strips(rows: slice, *, strip_rows: int = 512, align: int = 256) -> Iterator[slice]:
    """Split a row range into strips whose inner edges fall on multiples of ``align``.

    Edges are aligned in absolute grid rows, so no chunk is fetched by two strips. A strip is at
    most ``strip_rows`` tall; the first one is shorter when ``rows.start`` is not aligned.
    """
    if strip_rows <= 0 or align <= 0 or strip_rows % align:
        raise ValueError("strip_rows must be a positive multiple of align.")
    start, stop = int(rows.start), int(rows.stop)
    while start < stop:
        end = min(stop, (start // align) * align + strip_rows)
        yield slice(start, end)
        start = end


def iter_blocks(
    rows: slice,
    cols: slice,
    *,
    block: tuple[int, int] = DEFAULT_BLOCK,
    align: tuple[int, int] = (256, 256),
) -> Iterator[tuple[slice, slice]]:
    """Split a window into 2-D blocks with chunk-aligned inner edges, row-major."""
    for rs in iter_strips(rows, strip_rows=block[0], align=align[0]):
        for cs in iter_strips(cols, strip_rows=block[1], align=align[1]):
            yield rs, cs


# --------------------------------------------------------------------------------------------
# AOI plans
# --------------------------------------------------------------------------------------------


def _aligned_span(lo: int, hi: int, step: int, clip_lo: int, clip_hi: int) -> tuple[int, int]:
    """Expand ``[lo, hi)`` outward to multiples of ``step``, then clip to ``[clip_lo, clip_hi)``."""
    return max((lo // step) * step, clip_lo), min(-(-hi // step) * step, clip_hi)


@dataclasses.dataclass(frozen=True)
class ZonePlan:
    """The part of an AOI served by one zone group: a window and a pixel mask on it.

    ``mask`` is True for pixels whose centre lies inside the AOI and inside the zone's nominal band,
    so a multi-zone AOI counts every ground pixel exactly once.
    """

    zone: str
    grid: ZoneGrid
    rows: slice
    cols: slice
    mask: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the window."""
        return self.rows.stop - self.rows.start, self.cols.stop - self.cols.start

    @property
    def transform(self) -> Affine:
        """Affine transform of the window."""
        return self.grid.window_transform(self.rows.start, self.cols.start)

    @property
    def n_pixels(self) -> int:
        """Pixels inside the AOI."""
        return int(self.mask.sum())

    @property
    def area_ha(self) -> float:
        """Area inside the AOI in hectares."""
        return self.n_pixels * HA_PER_PIXEL

    def submask(self, rows: slice, cols: slice) -> np.ndarray:
        """The plan mask over a window given in zone grid rows and columns."""
        return self.mask[
            rows.start - self.rows.start : rows.stop - self.rows.start,
            cols.start - self.cols.start : cols.stop - self.cols.start,
        ]

    def chunks_touched(self, chunk_shape: tuple[int, int]) -> int:
        """Inner chunks that hold at least one AOI pixel: the least any reader must fetch."""
        ch, cw = chunk_shape
        rr, cc = np.nonzero(self.mask)
        if rr.size == 0:
            return 0
        keys = ((rr + self.rows.start) // ch).astype(np.int64) * 1_000_000 + (cc + self.cols.start) // cw
        return int(np.unique(keys).size)

    def blocks(
        self, chunk_shape: tuple[int, int], block: tuple[int, int] = DEFAULT_BLOCK
    ) -> Iterator[tuple[slice, slice]]:
        """Windows :func:`iter_plan_blocks` reads: chunk-aligned blocks trimmed to their AOI pixels.

        Blocks with no AOI pixel are dropped, and each kept block is shrunk to the chunk-aligned
        bounding box of its AOI pixels, so whole chunk rows and columns outside the AOI are never read.
        """
        ch, cw = chunk_shape
        for rs, cs in iter_blocks(self.rows, self.cols, block=block, align=chunk_shape):
            sub = self.submask(rs, cs)
            any_r = np.flatnonzero(sub.any(axis=1))
            if any_r.size == 0:
                continue
            any_c = np.flatnonzero(sub.any(axis=0))
            r0, r1 = _aligned_span(rs.start + int(any_r[0]), rs.start + int(any_r[-1]) + 1, ch, rs.start, rs.stop)
            c0, c1 = _aligned_span(cs.start + int(any_c[0]), cs.start + int(any_c[-1]) + 1, cw, cs.start, cs.stop)
            yield slice(r0, r1), slice(c0, c1)

    def chunks_fetched(self, chunk_shape: tuple[int, int], block: tuple[int, int] = DEFAULT_BLOCK) -> int:
        """Inner chunks :func:`iter_plan_blocks` fetches per year and layer with this block shape."""
        ch, cw = chunk_shape
        total = 0
        for rs, cs in self.blocks(chunk_shape, block):
            total += (-(-rs.stop // ch) - rs.start // ch) * (-(-cs.stop // cw) - cs.start // cw)
        return total


@dataclasses.dataclass(frozen=True)
class AOIPlan:
    """A lon/lat AOI split by nominal zone. Building one reads metadata only."""

    geometry: BaseGeometry
    zones: list[ZonePlan]

    @property
    def zone_names(self) -> list[str]:
        """Zone groups the AOI touches."""
        return [p.zone for p in self.zones]

    @property
    def n_pixels(self) -> int:
        """Pixels inside the AOI across all zones."""
        return sum(p.n_pixels for p in self.zones)

    @property
    def area_ha(self) -> float:
        """AOI area in hectares, counted on the pixel grid."""
        return self.n_pixels * HA_PER_PIXEL

    def estimated_bytes(self, reader: Reader, n_years: int = 1, *, block: tuple[int, int] = DEFAULT_BLOCK) -> int:
        """In-memory bytes :func:`iter_plan_blocks` reads for ``n_years`` of every layer.

        Counts every chunk the block reader fetches at full in-memory size. Bytes on the wire are
        normally a few percent lower (compression), and chunks the store elides (ocean, no data)
        cost nothing.
        """
        total = 0
        for p in self.zones:
            ch, cw = reader.chunk_shape(p.zone)
            total += p.chunks_fetched((ch, cw), block) * ch * cw * BYTES_PER_PIXEL
        return total * n_years


def plan_aoi(
    reader: Reader,
    geometry: BaseGeometry,
    *,
    max_zones: int = 4,
    max_area_km2: float | None = MAX_AOI_KM2,
    densify_deg: float = 0.001,
) -> AOIPlan:
    """Split a lon/lat geometry into per-zone windows and masks without reading any chunk.

    Args:
        reader: Store reader; only zone metadata is read.
        geometry: AOI in EPSG:4326 (lon, lat). Only its polygonal parts count.
        max_zones: Refuse AOIs that touch more zone groups than this.
        max_area_km2: Refuse AOIs larger than this (``None`` for no limit).
        densify_deg: Edge spacing used before projecting, so long edges follow the UTM curvature.

    Raises:
        AOIError: The AOI is empty, has no area, is outside the UTM latitude range, is too large,
            touches too many zones, or is spread so thinly that its window would be many times its
            own size.
    """
    if geometry.is_empty:
        raise AOIError("The AOI is empty.")
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    geometry = _polygonal(geometry)
    if geometry.is_empty or geometry.area == 0:
        raise AOIError("The AOI has no area; pass a polygon, or use read_points for points.")
    west, south, east, north = geometry.bounds
    if west < -180 or east > 180 or south < SOUTH_LIMIT or north > NORTH_LIMIT:
        raise AOIError(
            f"AOI bounds {geometry.bounds} fall outside lon -180 to 180, lat {SOUTH_LIMIT} to {NORTH_LIMIT}. "
            "Check the AOI is in EPSG:4326 with longitude first."
        )
    area_km2 = abs(Geod(ellps="WGS84").geometry_area_perimeter(geometry)[0]) / 1e6
    if max_area_km2 is not None and area_km2 > max_area_km2:
        raise AOIError(
            f"The AOI covers {area_km2:,.0f} km2, over the {max_area_km2:,.0f} km2 limit. "
            "Split it, or raise the limit if you mean it."
        )
    names = zones_for_geometry(geometry)
    if len(names) > max_zones:
        raise AOIError(
            f"The AOI touches {len(names)} zone groups ({', '.join(names)}); the limit is {max_zones}. "
            "Split the AOI into smaller parts."
        )
    # First pass: project every zone's part and check its window, so a sparse part in any zone is
    # refused before a single mask is allocated.
    parts: list[tuple[str, ZoneGrid, BaseGeometry, slice, slice]] = []
    for name in names:
        piece = _polygonal(geometry.intersection(zone_band(name)))
        if piece.is_empty or piece.area == 0:
            continue
        grid = reader.grid(name)
        projected = _polygonal(
            shapely.transform(shapely.segmentize(piece, densify_deg), grid.from_lonlat().transform, interleaved=False)
        )
        rows, cols = grid.window_for_bounds(*projected.bounds)
        height, width = rows.stop - rows.start, cols.stop - cols.start
        if height <= 0 or width <= 0:
            logger.warning("AOI part in %s falls outside the zone grid; skipped.", name)
            continue
        aoi_px = projected.area / (grid.pixel * grid.pixel)
        if height * width > max(MAX_WINDOW_TO_AOI * aoi_px, _WINDOW_FLOOR_PX):
            raise AOIError(
                f"The AOI part in {name} has a {height} x {width} pixel window for about "
                f"{aoi_px:,.0f} AOI pixels: its parts are far apart, or it is long and thin. Split it into "
                "compact pieces and run each separately."
            )
        parts.append((name, grid, projected, rows, cols))
    # Second pass: rasterize pixel centres inside each part.
    plans = []
    for name, grid, projected, rows, cols in parts:
        mask = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=np.uint8)
        rasterize(
            [(projected, 1)],
            out=mask,
            transform=grid.window_transform(rows.start, cols.start),
            all_touched=False,
        )
        bool_mask = mask.view(bool)
        if bool_mask.any():
            plans.append(ZonePlan(zone=name, grid=grid, rows=rows, cols=cols, mask=bool_mask))
    if not plans:
        raise AOIError("The AOI covers no whole pixel centre in any zone group.")
    return AOIPlan(geometry=geometry, zones=plans)


def read_plan(
    reader: Reader,
    plan: ZonePlan,
    year: int,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    max_pixels: int = MAX_WINDOW_PIXELS,
) -> Window:
    """Read one year of a zone plan in one piece, with the AOI mask attached.

    Raises:
        ReadTooLargeError: The window exceeds ``max_pixels``; stream it with :func:`iter_plan_blocks`.
    """
    h, w = plan.shape
    if h * w > max_pixels:
        raise ReadTooLargeError(
            f"Window {h} x {w} in {plan.zone} exceeds {max_pixels} pixels; stream it with iter_plan_blocks."
        )
    win = reader.read_window(plan.zone, year, plan.rows, plan.cols, layers=layers)
    win.mask = plan.mask
    return win


def read_polygon(
    reader: Reader,
    geometry: BaseGeometry,
    year: int,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    max_zones: int = 4,
    max_pixels: int = MAX_WINDOW_PIXELS,
) -> list[Window]:
    """Read one year of a lon/lat polygon: one masked window per zone group it touches.

    The polygon's bounding window is read, and ``Window.mask`` marks the pixels whose centre lies
    inside the polygon and inside the zone's nominal band. Years and window sizes are checked for
    every zone before anything is fetched.
    """
    aoi = plan_aoi(reader, geometry, max_zones=max_zones)
    reader_check_years(reader, aoi.zone_names, [year])
    for p in aoi.zones:
        if p.shape[0] * p.shape[1] > max_pixels:
            raise ReadTooLargeError(
                f"Window {p.shape[0]} x {p.shape[1]} in {p.zone} exceeds {max_pixels} pixels; "
                "stream it with iter_plan_blocks."
            )
    return [read_plan(reader, p, year, layers=layers, max_pixels=max_pixels) for p in aoi.zones]


def read_bbox(
    reader: Reader,
    bbox: tuple[float, float, float, float],
    year: int,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    max_zones: int = 4,
    max_pixels: int = MAX_WINDOW_PIXELS,
) -> list[Window]:
    """Read one year of a lon/lat box ``(west, south, east, north)``; see :func:`read_polygon`."""
    west, south, east, north = bbox
    if not (west < east and south < north):
        raise AOIError(f"bbox {bbox} must be (west, south, east, north) with west < east and south < north.")
    return read_polygon(
        reader, box(west, south, east, north), year, layers=layers, max_zones=max_zones, max_pixels=max_pixels
    )


@dataclasses.dataclass
class Block:
    """One block of a zone plan: the same window read for each requested year."""

    zone: str
    rows: slice
    cols: slice
    mask: np.ndarray
    windows: dict[int, Window]


def iter_plan_blocks(
    reader: Reader,
    plan: ZonePlan,
    years: Sequence[int],
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    block: tuple[int, int] = DEFAULT_BLOCK,
) -> Iterator[Block]:
    """Stream a zone plan in chunk-aligned blocks, reading every requested year per block.

    Years are checked when this is called, before the first block is fetched. Blocks with no AOI
    pixel are skipped, and each block is trimmed to the chunks that hold its AOI pixels (see
    :meth:`ZonePlan.blocks`). Each year's ``Window.mask`` is the block's slice of the plan mask.
    For a multi-zone AOI, check years for every zone first with ``reader.check_years``.
    """
    reader_check_years(reader, [plan.zone], years)
    windows = list(plan.blocks(reader.chunk_shape(plan.zone), block))
    return _iter_blocks(reader, plan, list(dict.fromkeys(years)), layers, windows)


def _iter_blocks(
    reader: Reader, plan: ZonePlan, years: list[int], layers: Sequence[str], windows: list[tuple[slice, slice]]
) -> Iterator[Block]:
    for rs, cs in windows:
        sub = plan.submask(rs, cs)
        wins = reader.read_windows(plan.zone, years, rs, cs, layers=layers)
        for win in wins.values():
            win.mask = sub
        yield Block(zone=plan.zone, rows=rs, cols=cs, mask=sub, windows=wins)


def reader_check_years(reader: Reader, zones: Iterable[str], years: Iterable[int]) -> None:
    """Raise :class:`YearsNotPublishedError` unless every zone lists every year. Metadata only."""
    years = sorted(set(years))
    bad_range = [y for y in years if y not in YEARS]
    missing: dict[str, list[int]] = {}
    for zone in zones:
        have = set(reader.years_complete(zone))
        gaps = sorted({*bad_range, *(y for y in years if y not in have)})
        if gaps:
            missing[canonicalize_zone(zone)] = gaps
    if missing:
        raise YearsNotPublishedError(missing)


# --------------------------------------------------------------------------------------------
# Points
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class PointRead:
    """Embeddings sampled at points. Rows with no data are NaN and ``valid`` is False."""

    embeddings: np.ndarray
    valid: np.ndarray
    zones: list[str]
    radar_present: np.ndarray | None
    chunks_read: int


def _point_zone(lon: float, lat: float) -> str:
    """Zone a point is read from.

    A point exactly on the equator sits on the edge between the north group's bottom row and a row
    of the south group. Under the floor convention it belongs to the pixel south of the edge, which
    only the south group holds.
    """
    zone = zone_for(lon, lat)
    return f"{zone[:2]}S" if lat == 0.0 else zone


def read_points(
    reader: Reader,
    lons: Sequence[float] | np.ndarray,
    lats: Sequence[float] | np.ndarray,
    year: int,
    *,
    with_radar: bool = True,
) -> PointRead:
    """Sample embeddings at lon/lat points, reading each chunk once however many points share it.

    Points are grouped by (zone, chunk row, chunk col) and each group costs one chunk read per
    layer, so the cost scales with the number of distinct chunks, not the number of points. Groups
    are fetched concurrently in batches. Each point is taken from its nominal zone. Years are
    checked for every zone before any chunk is read.

    Raises:
        ValueError: A coordinate is not finite or outside lon -180 to 180, lat -90 to 90.
        YearsNotPublishedError: The year is missing for a zone some point falls in.
    """
    lon = np.asarray(lons, dtype=np.float64)
    lat = np.asarray(lats, dtype=np.float64)
    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError("lons and lats must be 1-D and the same length.")
    n = lon.size
    zones = [_point_zone(float(x), float(y)) for x, y in zip(lon, lat, strict=True)]
    reader_check_years(reader, set(zones), [year])

    out = np.full((n, N_BANDS), np.nan, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    radar = np.zeros(n, dtype=bool) if with_radar else None
    groups: dict[tuple[str, int, int], list[tuple[int, int, int]]] = {}
    zone_arr = np.asarray(zones)
    for zone in sorted(set(zones)):
        idx = np.flatnonzero(zone_arr == zone)
        grid = reader.grid(zone)
        ch, cw = reader.chunk_shape(zone)
        xs, ys = grid.from_lonlat().transform(lon[idx], lat[idx])
        for point, x, y in zip(idx, np.atleast_1d(xs), np.atleast_1d(ys), strict=True):
            r, c = grid.rowcol(float(x), float(y))
            if 0 <= r < grid.height and 0 <= c < grid.width:
                groups.setdefault((zone, r // ch, c // cw), []).append((int(point), r, c))

    layers = ("embeddings", "scales", *(RADAR_LAYERS if with_radar else ()))
    keys = list(groups)
    for start in range(0, len(keys), _POINT_BATCH):
        batch = keys[start : start + _POINT_BATCH]
        windows = list(_GROUP_POOL.map(lambda key: _read_chunk(reader, key, groups[key], year, layers), batch))
        for key, win in zip(batch, windows, strict=True):
            ok = win.valid
            rp = win.radar_present if radar is not None else None
            assert win.embeddings is not None
            for i, r, c in groups[key]:
                lr, lc = r - win.rows.start, c - win.cols.start
                if ok[lr, lc]:
                    out[i] = win.embeddings[lr, lc].astype(np.float32) * win.scales[lr, lc]
                    valid[i] = True
                if radar is not None and rp is not None:
                    radar[i] = bool(rp[lr, lc])
    logger.info("read_points: %d points from %d chunks", n, len(groups))
    return PointRead(embeddings=out, valid=valid, zones=zones, radar_present=radar, chunks_read=len(groups))


def _read_chunk(
    reader: Reader, key: tuple[str, int, int], points: Sequence[tuple[int, int, int]], year: int, layers: Sequence[str]
) -> Window:
    """Read the chunk a group of points falls in.

    The window is the whole chunk, except that it starts no higher than the group's first own row
    (``own_rows``, when the reader has it) unless a point lies above that. In a south group the rows
    above it are served from the north group, so reading them would fetch the north group's bottom
    chunk for pixels no point uses; the points of a south zone always lie at or below it.

    Args:
        reader: The reader.
        key: ``(zone, chunk row, chunk col)``.
        points: ``(point index, row, col)`` of each point in the chunk.
        year: Year to read.
        layers: Layers to read.
    """
    zone, cr, cc = key
    grid = reader.grid(zone)
    ch, cw = reader.chunk_shape(zone)
    start = cr * ch
    own_rows = getattr(reader, "own_rows", None)  # the tests' metadata-only readers have none
    if own_rows is not None:
        start = max(start, min(int(own_rows(zone)[0]), min(r for _, r, _ in points)))
    rows = slice(start, min((cr + 1) * ch, grid.height))
    cols = slice(cc * cw, min((cc + 1) * cw, grid.width))
    return reader.read_window(zone, year, rows, cols, layers=layers)


__all__ = [
    "ALL_LAYERS",
    "DATASET_NAME",
    "DEFAULT_BLOCK",
    "DEFAULT_LAYERS",
    "FIRST_YEAR",
    "HA_PER_PIXEL",
    "LAST_YEAR",
    "MAX_AOI_KM2",
    "N_BANDS",
    "PIXEL_M",
    "STORE_URL",
    "YEARS",
    "AOIError",
    "AOIPlan",
    "Block",
    "ByteCounter",
    "CountedIcechunkStore",
    "CountingStore",
    "DatasetMismatchError",
    "PointRead",
    "ReadTooLargeError",
    "Reader",
    "StoreUnavailableError",
    "TesseraError",
    "TesseraReader",
    "Window",
    "YearsNotPublishedError",
    "ZoneGrid",
    "ZonePlan",
    "canonicalize_zone",
    "check_dataset",
    "iter_blocks",
    "iter_plan_blocks",
    "iter_strips",
    "plan_aoi",
    "read_bbox",
    "read_plan",
    "read_points",
    "read_polygon",
    "reader_check_years",
    "zone_band",
    "zone_epsg",
    "zone_for",
    "zones_for_geometry",
]
