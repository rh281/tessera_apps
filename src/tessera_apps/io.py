"""Read the TESSERA v1.1 global embeddings from the published Icechunk store.

The store holds one Zarr group per UTM zone and hemisphere (``10N``, ``20S``). Each group carries
int8 ``embeddings`` (time, northing, easting, band), float32 ``scales`` (true value is
``embeddings * scales``, NaN means no data), and three uint16 observation counts. Rows run north to
south, the time axis is calendar years 2017 to 2025, and a zone lists the years it has published in
``attrs["years_complete"]``. ``docs/prebuild-survey.md`` records how each of these facts was checked.

Everything here reads through :class:`TesseraReader`, which opens one group at a time with plain
zarr (never Dask) and counts the bytes it pulls from the store. AOI reads are planned first
(:func:`plan_aoi`), which touches metadata only, so callers can check years, zone count and size
before any chunk read. A plan is then read whole (:func:`read_plan`) when small, or block by block
(:func:`iter_plan_blocks`) when large. Layers, years and point chunks are fetched concurrently on a
small thread pool, since each zarr call is a few serial round trips (shard index, then data).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import threading
from collections.abc import AsyncIterator, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np
import shapely
import zarr
from pyproj import CRS, Geod, Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from zarr.abc.store import Store
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


@dataclasses.dataclass
class ByteCounter:
    """Running total of payload bytes returned by the store."""

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
    """A zarr store wrapper that counts the bytes every read returns.

    zarr reads through ``get`` and ``get_partial_values`` on the async path and ``get_sync`` on the
    sync path; all three are counted. The coalescing range readers (``get_ranges`` and
    ``get_ranges_sync``, which zarr 3.4 sharding uses for partial-shard reads) run zarr's own
    coalescer above this wrapper, so each merged fetch, gaps included, passes through the counted
    ``get`` or ``get_sync``. Copies made by ``with_read_only`` share the same counter. Icechunk's
    manifest and snapshot fetches happen below the zarr store interface and are not counted; they
    are small next to chunk payloads.
    """

    def __init__(self, store: Any, counter: ByteCounter | None = None) -> None:  # noqa: ANN401
        super().__init__(store)
        self.counter = counter if counter is not None else ByteCounter()

    def _with_store(self, store: Any) -> Self:  # noqa: ANN401
        return type(self)(store, self.counter)

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        """Read one key and count it."""
        buf = await self._store.get(key, prototype, byte_range)
        self.counter.add(buf)
        return buf

    async def get_partial_values(
        self, prototype: BufferPrototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]
    ) -> list[Buffer | None]:
        """Read several ranges and count them."""
        bufs = await self._store.get_partial_values(prototype, key_ranges)
        for buf in bufs:
            self.counter.add(buf)
        return list(bufs)

    def get_sync(
        self, key: str, *, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        """Synchronous read of one key, counted."""
        buf = self._store.get_sync(key, prototype=prototype, byte_range=byte_range)
        self.counter.add(buf)
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


def _find_counter(store: Any) -> ByteCounter | None:  # noqa: ANN401
    """The counter of the first CountingStore in a chain of wrapper stores."""
    seen = 0
    while store is not None and seen < 16:
        if isinstance(store, CountingStore):
            return store.counter
        store = getattr(store, "_store", None)
        seen += 1
    return None


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
    """Reader for the published Icechunk store (or any zarr store with the same layout).

    Use :meth:`open` for the live store and :meth:`from_store` for any other zarr store, such as the
    in-memory stores the tests build. Arrays are read with plain zarr indexing, so nothing goes
    through Dask.
    """

    def __init__(
        self,
        root: zarr.Group,
        *,
        counter: ByteCounter | None = None,
        provenance: dict[str, Any] | None = None,
        cache: ChunkCache | None = None,
    ) -> None:
        self._root = root
        found = counter if counter is not None else _find_counter(root.store)
        if found is None:
            logger.warning("The store is not wrapped in a CountingStore; bytes_read will stay 0. Use from_store.")
            found = ByteCounter()
        self._counter = found
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
        self._time_index: dict[str, dict[int, int]] = {}

    @classmethod
    def open(
        cls,
        *,
        snapshot_id: str | None = None,
        branch: str = "main",
        cache: ChunkCache | None = None,
    ) -> TesseraReader:
        """Open the published store anonymously.

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
        """
        import icechunk

        storage = icechunk.s3_storage(bucket=STORE_BUCKET, prefix=STORE_PREFIX, region=STORE_REGION, anonymous=True)
        try:
            # Pass no config: a config here replaces the one saved with the store wholesale.
            repo = icechunk.Repository.open(storage)
            session = repo.readonly_session(snapshot_id=snapshot_id) if snapshot_id else repo.readonly_session(branch)
            counting = CountingStore(session.store)
            root = zarr.open_group(counting, mode="r")
        except icechunk.IcechunkError as exc:
            where = f"snapshot {snapshot_id}" if snapshot_id else f"branch {branch!r}"
            hint = f"Check that this machine can reach S3 in {STORE_REGION} (no AWS credentials are needed)"
            if snapshot_id:
                hint += " and that the snapshot id is one of the store's (tessera-apps info prints the current one)"
            raise StoreUnavailableError(
                f"Cannot open the store {STORE_URL} at {where}: {_icechunk_message(exc)}. {hint}."
            ) from exc
        provenance = {
            "backend": "icechunk",
            "store_url": STORE_URL,
            "snapshot_id": session.snapshot_id,
            "branch": None if snapshot_id else branch,
            "icechunk_version": getattr(icechunk, "__version__", None),
            "zarr_version": zarr.__version__,
        }
        if cache is not None:
            cache = cache.for_namespace(str(session.snapshot_id))
        return cls(root, counter=counting.counter, provenance=provenance, cache=cache)

    @classmethod
    def from_store(cls, store: Any, *, cache: ChunkCache | None = None, **provenance: Any) -> TesseraReader:  # noqa: ANN401
        """Wrap any zarr store (for example a ``MemoryStore`` in tests) with byte counting.

        A cache must carry an explicit namespace, or ``snapshot_id`` must be passed in
        ``provenance``.
        """
        counting = CountingStore(store)
        root = zarr.open_group(counting, mode="r")
        prov = {"backend": type(store).__name__, **provenance}
        return cls(root, counter=counting.counter, provenance=prov, cache=cache)

    # ---- metadata -------------------------------------------------------------------------

    @property
    def bytes_read(self) -> int:
        """Payload bytes fetched from the store so far (cache hits are free)."""
        return self._counter.bytes

    @property
    def cache(self) -> ChunkCache | None:
        """The chunk cache in use, if any."""
        return self._cache

    def _group(self, zone: str) -> zarr.Group:
        z = canonicalize_zone(zone)
        if z not in self._groups:
            try:
                node = self._root[z]
            except KeyError as exc:
                raise TesseraError(f"Zone group {z} is not in the store.") from exc
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
        return dict(self._root.attrs)

    def grid(self, zone: str) -> ZoneGrid:
        """Pixel grid of a zone group."""
        z = canonicalize_zone(zone)
        if z not in self._grids:
            shape = self._array(z, "scales").shape
            self._grids[z] = ZoneGrid.from_attrs(z, self.zone_attrs(z), (shape[1], shape[2]))
        return self._grids[z]

    def chunk_shape(self, zone: str) -> tuple[int, int]:
        """(rows, cols) of one inner chunk of the embeddings array (256 x 256 in the live store)."""
        chunks = self._array(zone, "embeddings").chunks
        return int(chunks[1]), int(chunks[2])

    def years_complete(self, zone: str) -> list[int]:
        """Years listed in the zone's ``years_complete`` attribute, sorted."""
        raw = self.zone_attrs(zone).get("years_complete")
        if raw is None:
            raise TesseraError(f"Zone {canonicalize_zone(zone)} has no years_complete attribute.")
        return sorted(int(y) for y in raw)

    def year_index(self, zone: str, year: int) -> int:
        """Position of a calendar year on a zone's time axis."""
        z = canonicalize_zone(zone)
        if z not in self._time_index:
            n = int(self._array(z, "scales").shape[0])
            years = _decode_years(self._group(z), n)
            self._time_index[z] = {y: i for i, y in enumerate(years)}
        try:
            return self._time_index[z][year]
        except KeyError as exc:
            raise YearsNotPublishedError({z: [year]}) from exc

    def check_years(self, zones: Iterable[str], years: Iterable[int]) -> None:
        """Raise unless every year is in ``years_complete`` for every zone. Reads attributes only.

        Raises:
            YearsNotPublishedError: Names each missing zone-year.
        """
        reader_check_years(self, zones, years)

    def provenance(self) -> dict[str, Any]:
        """Store and model identity for reports.

        ``checkpoint_id`` names the model weights and never changes, so the data version is the
        Icechunk ``snapshot_id``.
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
        arr = self._array(job.zone, job.name)
        if arr.ndim == 4:
            return np.asarray(arr[job.t, job.r0 : job.r1, job.c0 : job.c1, :])
        return np.asarray(arr[job.t, job.r0 : job.r1, job.c0 : job.c1])

    def _read_job_cached(self, job: _Job) -> np.ndarray:
        """Serve a job from whole cached chunks; on any miss, fetch the chunk-aligned cover in one call."""
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


def _decode_years(group: zarr.Group, n: int) -> list[int]:
    """Calendar years on a zone's time axis.

    Decodes the ``time`` coordinate when it carries CF units, and otherwise falls back to the
    documented layout: ``n`` consecutive years from 2017.

    Raises:
        TesseraError: The time coordinate decodes to anything other than ``n`` years from 2017.
    """
    fallback = list(range(FIRST_YEAR, FIRST_YEAR + n))
    try:
        node = group["time"]
    except KeyError:
        return fallback
    if not isinstance(node, zarr.Array):
        return fallback
    values = np.asarray(node[:])
    if np.issubdtype(values.dtype, np.datetime64):
        decoded = [int(v) for v in values.astype("datetime64[Y]").astype(int) + 1970]
        if decoded != fallback:
            raise TesseraError(f"time coordinate decodes to years {decoded}, expected {fallback}.")
        return decoded
    units = str(node.attrs.get("units", ""))
    m = re.fullmatch(r"\s*(\w+)\s+since\s+(\d{4}-\d{2}-\d{2})(?:[ T][\d:.]+)?\s*", units)
    if not m:
        return fallback
    unit = {"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms", "seconds": "s", "days": "D"}.get(
        m.group(1)
    )
    if unit is None:
        return fallback
    stamps = np.datetime64(m.group(2)) + values.astype(np.int64).astype(f"timedelta64[{unit}]")
    years = [int(v) for v in stamps.astype("datetime64[Y]").astype(int) + 1970]
    if years != fallback:
        raise TesseraError(
            f"time coordinate decodes to years {years}, expected {fallback}; the store layout has changed."
        )
    return years


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
        windows = list(_GROUP_POOL.map(lambda key: _read_chunk(reader, key, year, layers), batch))
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


def _read_chunk(reader: Reader, key: tuple[str, int, int], year: int, layers: Sequence[str]) -> Window:
    zone, cr, cc = key
    grid = reader.grid(zone)
    ch, cw = reader.chunk_shape(zone)
    rows = slice(cr * ch, min((cr + 1) * ch, grid.height))
    cols = slice(cc * cw, min((cc + 1) * cw, grid.width))
    return reader.read_window(zone, year, rows, cols, layers=layers)


__all__ = [
    "ALL_LAYERS",
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
    "CountingStore",
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
