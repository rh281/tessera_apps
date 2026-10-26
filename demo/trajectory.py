"""The hosted demo's map click: one pixel's wildfire recovery trajectory, read from one store chunk.

The Streamlit page (``demo/app.py``) calls :func:`run_click` when someone clicks the map. This
module imports no Streamlit, so the offline tests run it on the repository's synthetic stores.

What a click does, using the package's own pieces:

- **Where.** The clicked point's nominal zone (:func:`tessera_apps.io.zone_for`), its pixel on that
  zone group's grid (:class:`tessera_apps.io.ZoneGrid`) and the one inner chunk that holds it
  (256 x 256 pixels, 2.56 km, in the live store; :meth:`TesseraReader.chunk_shape`). The window is
  that chunk, clipped to the rows the group serves itself (:meth:`TesseraReader.own_rows`), as
  :func:`tessera_apps.io.read_points` reads a point's chunk. A longitude outside -180 to 180 (a
  click on another copy of the world) is wrapped into it first.
- **Which years.** The pre-fire year ``year_a`` and every year from it through ``through`` (2025 by
  default): the fire year, the post-fire year ``year_b`` and the recovery years after it. Every one
  is checked against the zone's ``years_complete`` (:meth:`TesseraReader.check_years`) before any
  chunk is read. ``year_a - 1`` is read too when the window's own rules are used (below) and either
  the threshold needs its stable-pair fallback or the window has a burned and unburned split, whose
  unburned reference the wildfire app checks against that year.
- **The read.** :meth:`TesseraReader.read_windows` with the wildfire app's layers
  (:data:`tessera_apps.apps.wildfire.LAYERS`), through a :class:`TesseraReader` opened anonymously
  at :data:`SNAPSHOT` unless ``TESSERA_APPS_DEMO_SNAPSHOT`` names another. The bytes are the
  reader's own count of chunk payload bytes (:attr:`TesseraReader.bytes_read`). A
  :class:`WindowCache` keeps the windows read, keyed by snapshot, zone, chunk and year, so a click in
  a chunk and years already read fetches nothing.
- **The numbers.** The wildfire app's own functions on the window, as its full run applies them to
  the AOI: the distance between ``year_a`` and ``year_b`` and each year's similarity to ``year_a``
  (:mod:`tessera_apps.metrics`, on pixels valid in both years with the same radar regime), the
  burn threshold (:func:`~tessera_apps.apps.wildfire.select_threshold`), the 3 x 3 opening and 1 ha
  minimum area (:func:`~tessera_apps.apps._common.mosaic_and_clean`), the unburned reference and
  its median similarity per year (``s_ref_t``), and the pixel's recovery index ``R_t``
  (:func:`~tessera_apps.apps.wildfire.recovery_index`).

Which threshold and reference: a click that a precomputed run covers (:class:`RunSettings`: inside
the Caldor demo AOI, at the snapshot and the years of the run in ``demo/precomputed``) takes that
run's threshold, as the app's ``--threshold`` does, and that run's ``s_ref_t``, so its burned
verdict and ``R_t`` are the full run's. Every other click applies the app's rules to the window
alone: Otsu with the bimodality check, else the stable-pair fallback
(:func:`~tessera_apps.apps.wildfire.select_threshold`), the window's own unburned reference
(:func:`~tessera_apps.apps.wildfire.recovery_reference`) and its medians
(:func:`~tessera_apps.apps.wildfire.recovery_metrics`), with the app's warnings about them (a
suspect fallback, a reference that drifted, a radar-free scar). Those can differ from a run over a
whole fire. The similarities never depend on the window and match a full run's exactly. Every
failure the page can meet raises :class:`ClickError`, whose message says what to change.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry

from tessera_apps import io, metrics
from tessera_apps.apps import wildfire
from tessera_apps.apps._common import AppInputError, mosaic_and_clean
from tessera_apps.demo import demo_aoi
from tessera_apps.io import FIRST_YEAR, LAST_YEAR, AOIPlan, TesseraReader, Window, ZonePlan

logger = logging.getLogger(__name__)

#: The store snapshot every number in docs/checkpoints.md since checkpoint A comes from, and the
#: one ``demo/precomputed`` was made at.
SNAPSHOT = "RC6FS07ZB0T76EXT8XRG"
#: Environment variable that names another snapshot for the click to read.
SNAPSHOT_ENV = "TESSERA_APPS_DEMO_SNAPSHOT"
#: Environment variable that sets the window cache's size in MB.
CACHE_MB_ENV = "TESSERA_APPS_DEMO_CACHE_MB"
#: Default window cache size: about 28 window-years of the live store (one is about 8.9 MB in
#: memory with the wildfire layers), so four chunks at the default years.
DEFAULT_CACHE_MB = 256
#: The Caldor demo's years, the page's defaults.
DEFAULT_YEAR_A = 2020
DEFAULT_YEAR_B = 2022
#: The wildfire app's default minimum burned patch (``--min-area-ha``).
MIN_AREA_HA = 1.0
#: The precomputed demos beside this module (made by ``demo/precompute.py``).
PRECOMPUTED = Path(__file__).resolve().parent / "precomputed"
#: The precomputed wildfire demo whose threshold and unburned reference a click inside its AOI takes.
PRECOMPUTED_RUN = "caldor"

REFERENCE_NOTE = (
    "The threshold, the burned verdict and the unburned reference here are this window's own: the wildfire app's "
    "rules applied to the {size} window, not to a whole fire. They can differ from a run over a whole fire{such_as}, "
    "and so can this pixel's recovery index. Its similarities to the pre-fire year do not depend on the window: they "
    "are what a full run computes for this pixel."
)
RUN_NOTE = (
    "This pixel is inside the {aoi} at the run's snapshot and years ({year_a} and {year_b}), so the click takes "
    "{label}'s threshold ({threshold:.3f}, by {method} over its whole AOI) and its unburned reference's median "
    "similarity in each year (from {source}). This pixel's burned verdict and recovery index are therefore that "
    "run's. Only the 3 x 3 opening and the {min_area:g} ha minimum area are applied to the {size} window, so a patch "
    "cut by the window's edge can still differ from the run. The similarities to the pre-fire year are what the run "
    "computes for this pixel."
)
RADAR_NOTE = (
    "The pixel's Sentinel-1 presence changes between {year_a} and {years}. A pixel that gains or loses radar "
    "shifts by 0.13 to 0.25 of cosine distance with no change on the ground (survey B5), so the wildfire app "
    "leaves such a pair out: those years are drawn hollow and given no recovery index."
)
#: How the page's messages name the wildfire app's year arguments.
PAGE_NAMES = {
    "recovery_through": "the last recovery year",
    "year_a": "the pre-fire year",
    "year_b": "the post-fire year",
    "fire_year": "the fire year",
}


class ClickError(Exception):
    """A click the trajectory cannot answer. The message says what to change.

    Attributes:
        kind: ``"outside"`` (outside the store's coverage), ``"no_data"`` (the store has no data at
            the pixel in a year the curve needs), ``"years"`` (an argument or a year the zone has not
            published) or ``"store"`` (the store, or the precomputed run's settings, cannot be read).
        bytes_read: Chunk payload bytes the click read before it stopped (0 unless ``"no_data"``).
        wall_s: Seconds the click took before it stopped.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.bytes_read = 0
        self.wall_s = 0.0


# --------------------------------------------------------------------------------------------
# The precomputed run's settings
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunSettings:
    """A precomputed wildfire run's threshold and unburned reference, for the clicks it covers.

    A click inside the run's AOI, at its snapshot, its ``year_a`` and ``year_b``, and with recovery
    years the run has, takes the run's threshold (as the app's ``--threshold`` does, so no fallback
    read) and the run's median similarity of its unburned reference per year, so its burned verdict
    and recovery index are the run's, not the window's own.

    Attributes:
        name: The demo's name, for example ``"caldor"``.
        snapshot: Snapshot the run read.
        year_a: The run's pre-fire year.
        year_b: The run's post-fire year.
        geometry: The run's AOI in lon/lat.
        threshold: The run's burn threshold.
        method: How the run chose it, for example ``"otsu"``.
        reference_median: Year to the median similarity of the run's unburned reference (``s_ref_t``).
        reference_p10: Year to its 10th percentile, the level a burned pixel must reach to count as
            recovered.
        reference_pixels: Year to the reference's pixel count with a similarity that year.
        source: The files the settings come from, relative to the repository root.
    """

    name: str
    snapshot: str
    year_a: int
    year_b: int
    geometry: BaseGeometry
    threshold: float
    method: str
    reference_median: Mapping[int, float]
    reference_p10: Mapping[int, float]
    reference_pixels: Mapping[int, int]
    source: str

    @property
    def label(self) -> str:
        """How the page names the run, for example "the precomputed Caldor run"."""
        return f"the precomputed {self.name.capitalize()} run"

    def covers(self, snapshot: str, lon: float, lat: float, *, year_a: int, year_b: int, through: int) -> bool:
        """Whether a click at (``lon``, ``lat``) with these years and snapshot takes this run's settings."""
        if snapshot != self.snapshot or (year_a, year_b) != (self.year_a, self.year_b):
            return False
        if any(y not in self.reference_median for y in range(year_b, through + 1)):
            return False
        return bool(self.geometry.covers(Point(lon, lat)))

    def record(self) -> dict[str, Any]:
        """Plain values for the page."""
        return {
            "name": self.name,
            "label": self.label,
            "snapshot": self.snapshot,
            "year_a": self.year_a,
            "year_b": self.year_b,
            "threshold": self.threshold,
            "method": self.method,
            "reference_pixels": dict(self.reference_pixels),
            "source": self.source,
        }


def load_run_settings(directory: Path = PRECOMPUTED, name: str = PRECOMPUTED_RUN) -> RunSettings:
    """The settings of the precomputed wildfire run ``name``, from ``index.json`` and its summary table.

    The threshold is the index's headline ``threshold`` (from the run's report), and must equal the
    summary table's ``threshold`` in every year; each year's ``s_ref_t`` is the table's
    ``median_sim_unburned``. The AOI is the bundled demo AOI the index names.

    Args:
        directory: The ``demo/precomputed`` folder.
        name: The demo's name.

    Returns:
        The settings.

    Raises:
        ValueError: The files are missing, malformed or disagree; the message says how to restore them.
    """
    try:
        index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
        entry = next(d for d in index["demos"] if d["name"] == name)
        if entry["app"] != "wildfire":
            raise ValueError(f"the {name} demo is a {entry['app']} run, not a wildfire run")
        threshold = float(next(h["value"] for h in entry["headline"] if h["key"] == "threshold"))
        table = entry["files"]["summary_csv"]["path"]
        with (directory / table).open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{table} has no rows")
        methods = {r["threshold_method"] for r in rows}
        if len(methods) != 1 or any(float(r["threshold"]) != threshold for r in rows):
            raise ValueError(f"{table}'s threshold does not match index.json's headline threshold {threshold!r}")
        settings = RunSettings(
            name=name,
            snapshot=str(entry["snapshot"]),
            year_a=int(entry["years"]["year_a"]),
            year_b=int(entry["years"]["year_b"]),
            geometry=demo_aoi(entry["aoi"]["demo"]),
            threshold=threshold,
            method=methods.pop(),
            reference_median={int(r["year"]): float(r["median_sim_unburned"]) for r in rows},
            reference_p10={int(r["year"]): float(r["unburned_sim_p10"]) for r in rows},
            reference_pixels={int(r["year"]): int(r["n_unburned"]) for r in rows},
            source=f"demo/precomputed/{table} and demo/precomputed/index.json",
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise ValueError(
            f"Cannot read the precomputed {name} run's threshold and reference from {directory} "
            f"({type(exc).__name__}: {exc}). Restore demo/precomputed from the repository, or remake it with "
            "python demo/precompute.py --work DIR."
        ) from exc
    if settings.year_b not in settings.reference_median:
        raise ValueError(
            f"The precomputed {name} run's summary table has no row for its post-fire year {settings.year_b}. Restore "
            "demo/precomputed from the repository, or remake it with python demo/precompute.py --work DIR."
        )
    return settings


# --------------------------------------------------------------------------------------------
# The store and the cache
# --------------------------------------------------------------------------------------------


def resolve_snapshot(env: Mapping[str, str] | None = None) -> str:
    """The snapshot the click reads: ``TESSERA_APPS_DEMO_SNAPSHOT`` when set and not empty, else :data:`SNAPSHOT`.

    Args:
        env: Environment to read; ``os.environ`` by default.

    Returns:
        The snapshot id.
    """
    value = (os.environ if env is None else env).get(SNAPSHOT_ENV, "").strip()
    return value or SNAPSHOT


def cache_bytes(env: Mapping[str, str] | None = None) -> int:
    """The window cache's size in bytes, from ``TESSERA_APPS_DEMO_CACHE_MB`` or :data:`DEFAULT_CACHE_MB`.

    Args:
        env: Environment to read; ``os.environ`` by default.

    Returns:
        The size in bytes.

    Raises:
        ValueError: The variable is not a whole number of MB, zero or more.
    """
    text = (os.environ if env is None else env).get(CACHE_MB_ENV, "").strip()
    if not text:
        return DEFAULT_CACHE_MB * 2**20
    try:
        mb = int(text)
    except ValueError as exc:
        raise ValueError(f"{CACHE_MB_ENV}={text!r} is not a whole number of MB; set it to 0 or more.") from exc
    if mb < 0:
        raise ValueError(f"{CACHE_MB_ENV}={mb} is negative; set it to 0 or more.")
    return mb * 2**20


#: A cache key: (snapshot, zone, chunk row, chunk col, year).
CacheKey = tuple[str, str, int, int, int]


def _window_nbytes(win: Window) -> int:
    arrays = (win.scales, win.embeddings, win.s2_obs_count, win.s1_asc_obs_count, win.s1_desc_obs_count)
    return sum(int(a.nbytes) for a in arrays if a is not None)


class WindowCache:
    """Windows read by earlier clicks, in memory, least recently used out first.

    Keyed by (snapshot, zone, chunk row, chunk col, year), so a click in a chunk and years already
    read fetches nothing, and a new snapshot never serves another's data. Safe to share between
    threads (the page shares one between its sessions).

    Args:
        max_bytes: Most bytes of window arrays held; 0 keeps nothing.
    """

    def __init__(self, max_bytes: int = DEFAULT_CACHE_MB * 2**20) -> None:
        self.max_bytes = max_bytes
        self._items: collections.OrderedDict[CacheKey, Window] = collections.OrderedDict()
        self._nbytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        """Windows held."""
        return len(self._items)

    @property
    def nbytes(self) -> int:
        """Bytes of window arrays held."""
        return self._nbytes

    def get(self, key: CacheKey) -> Window | None:
        """The window under ``key``, or None, counting a hit or a miss."""
        with self._lock:
            win = self._items.get(key)
            if win is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return win

    def put(self, key: CacheKey, win: Window) -> None:
        """Keep a window, dropping the least recently used ones past :attr:`max_bytes`."""
        size = _window_nbytes(win)
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._nbytes -= _window_nbytes(old)
            if size > self.max_bytes:
                return
            self._items[key] = win
            self._nbytes += size
            while self._nbytes > self.max_bytes and self._items:
                _, dropped = self._items.popitem(last=False)
                self._nbytes -= _window_nbytes(dropped)


@dataclasses.dataclass
class DemoStore:
    """The reader the clicks share, the snapshot it reads, the window cache, the precomputed runs and a lock.

    The lock serialises reads, so each click's bytes are its own and two sessions clicking the same
    chunk read it once.
    """

    reader: TesseraReader
    snapshot: str
    cache: WindowCache
    #: Precomputed runs whose settings a click they cover takes (:class:`RunSettings`).
    runs: tuple[RunSettings, ...] = ()
    lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    #: Seconds :func:`open_store` took to open the store (metadata only).
    open_s: float = 0.0
    #: Clicks run on this store so far.
    clicks: int = 0

    @classmethod
    def from_reader(
        cls, reader: TesseraReader, *, cache: WindowCache | None = None, runs: Sequence[RunSettings] = ()
    ) -> DemoStore:
        """Wrap an open reader; its provenance must name the snapshot it reads.

        Raises:
            ValueError: The reader's provenance has no ``snapshot_id``.
        """
        snapshot = reader.provenance().get("snapshot_id")
        if not snapshot:
            raise ValueError("The reader's provenance names no snapshot_id; the cache key needs one.")
        return cls(
            reader=reader,
            snapshot=str(snapshot),
            cache=cache if cache is not None else WindowCache(),
            runs=tuple(runs),
        )


def open_store(
    snapshot: str | None = None, *, cache: WindowCache | None = None, precomputed: Path = PRECOMPUTED
) -> DemoStore:
    """Open the published store anonymously at ``snapshot`` (:func:`resolve_snapshot` when None).

    Opening reads the store's metadata, never a chunk. It also reads the precomputed Caldor run's
    threshold and reference (:func:`load_run_settings`) from ``precomputed``.

    Raises:
        ClickError: ``kind`` ``"store"``: the precomputed run's files cannot be read, the store
            cannot be reached, the snapshot is not one of its snapshots, or geotessera does not
            identify it as the dClimate v1.1 dataset.
    """
    snap = snapshot or resolve_snapshot()
    try:
        runs = (load_run_settings(precomputed),)
    except ValueError as exc:
        raise ClickError("store", str(exc)) from exc
    t = time.perf_counter()
    try:
        reader = TesseraReader.open(snapshot_id=snap)
    except (io.StoreUnavailableError, io.DatasetMismatchError) as exc:
        raise ClickError(
            "store",
            f"{exc} The demo reads snapshot {snap}; unset {SNAPSHOT_ENV} to read the default {SNAPSHOT}.",
        ) from exc
    store = DemoStore.from_reader(reader, cache=cache if cache is not None else WindowCache(cache_bytes()), runs=runs)
    store.open_s = time.perf_counter() - t
    logger.info("Opened %s at snapshot %s in %.2f s (metadata only).", io.STORE_URL, store.snapshot, store.open_s)
    return store


# --------------------------------------------------------------------------------------------
# The window
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Located:
    """A clicked point on its zone group's grid and the chunk window that holds it."""

    lon: float
    lat: float
    zone: str
    row: int
    col: int
    chunk: tuple[int, int]
    rows: slice
    cols: slice
    grid: io.ZoneGrid

    @property
    def local(self) -> tuple[int, int]:
        """The pixel's (row, col) inside the window."""
        return self.row - self.rows.start, self.col - self.cols.start

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) of the window."""
        return self.rows.stop - self.rows.start, self.cols.stop - self.cols.start

    def window_lonlat(self) -> BaseGeometry:
        """The window's outline in lon/lat (its zone-grid square, densified before projecting)."""
        g = self.grid
        square = box(
            g.left + self.cols.start * g.pixel,
            g.top - self.rows.stop * g.pixel,
            g.left + self.cols.stop * g.pixel,
            g.top - self.rows.start * g.pixel,
        )
        return shapely.transform(shapely.segmentize(square, 10 * g.pixel), g.to_lonlat().transform, interleaved=False)


def wrap_longitude(lon: float) -> float:
    """``lon`` wrapped into -180 to 180, as a click on another copy of the world needs; NaN stays NaN."""
    if not math.isfinite(lon) or -180.0 <= lon <= 180.0:
        return lon
    return ((lon + 180.0) % 360.0) - 180.0


def _outside(lon: float, lat: float, why: str) -> ClickError:
    return ClickError(
        "outside",
        f"The point ({lon:.5f}, {lat:.5f}) is outside the store's coverage: {why}. Click on land between "
        f"{io.SOUTH_LIMIT:g} and {io.NORTH_LIMIT:g} degrees latitude, for example inside the Caldor outline.",
    )


def locate(reader: TesseraReader, lon: float, lat: float) -> Located:
    """The clicked point's zone, pixel and chunk window. Reads zone metadata only.

    Raises:
        ClickError: ``kind`` ``"outside"``: the point is outside the UTM latitudes the store covers,
            in a zone the store has no group for, or off its zone group's grid.
    """
    if not (math.isfinite(lon) and math.isfinite(lat) and -180.0 <= lon <= 180.0):
        raise _outside(lon, lat, "the longitude is not between -180 and 180")
    if not io.SOUTH_LIMIT <= lat <= io.NORTH_LIMIT:
        raise _outside(lon, lat, f"the store has no polar zones (it covers {io.SOUTH_LIMIT:g} to {io.NORTH_LIMIT:g})")
    zone = io.zone_for(lon, lat)
    if zone not in reader.store.groups:
        raise _outside(lon, lat, f"the store has no zone group {zone}")
    grid = reader.grid(zone)
    x, y = grid.from_lonlat().transform(lon, lat)
    row, col = grid.rowcol(float(x), float(y))
    own0, own1 = reader.own_rows(zone)
    if not (own0 <= row < own1 and 0 <= col < grid.width):
        raise _outside(lon, lat, f"it falls off zone group {zone}'s grid (row {row}, column {col})")
    ch, cw = reader.chunk_shape(zone)
    cr, cc = row // ch, col // cw
    rows = slice(max(cr * ch, own0), min((cr + 1) * ch, own1))
    cols = slice(cc * cw, min((cc + 1) * cw, grid.width))
    return Located(lon=lon, lat=lat, zone=zone, row=row, col=col, chunk=(cr, cc), rows=rows, cols=cols, grid=grid)


@dataclasses.dataclass
class _Reads:
    """What one click fetched and what it took from the cache."""

    bytes: int = 0
    seconds: float = 0.0
    fetched: list[int] = dataclasses.field(default_factory=list)
    cached: list[int] = dataclasses.field(default_factory=list)


def _windows(store: DemoStore, where: Located, years: Sequence[int], reads: _Reads) -> dict[int, Window]:
    """The window for each year: from the cache, else one :meth:`TesseraReader.read_windows` call for the rest."""
    out: dict[int, Window] = {}
    with store.lock:
        missing: list[int] = []
        for y in years:
            win = store.cache.get((store.snapshot, where.zone, *where.chunk, y))
            if win is None:
                missing.append(y)
            else:
                out[y] = win
                reads.cached.append(y)
        if missing:
            before = store.reader.bytes_read
            t = time.perf_counter()
            got = store.reader.read_windows(where.zone, missing, where.rows, where.cols, layers=wildfire.LAYERS)
            reads.seconds += time.perf_counter() - t
            reads.bytes += store.reader.bytes_read - before
            reads.fetched.extend(missing)
            for y, win in got.items():
                store.cache.put((store.snapshot, where.zone, *where.chunk, y), win)
                out[y] = win
    return out


# --------------------------------------------------------------------------------------------
# The trajectory
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class YearPoint:
    """One year of the clicked pixel's curve.

    Attributes:
        year: Calendar year.
        role: ``"stable pair"`` (``year_a - 1``, read for the threshold fallback or the reference
            check), ``"pre-fire"`` (``year_a``), ``"before the fire"`` (after ``year_a`` and before
            the fire year), ``"fire"`` (the fire year, :func:`~tessera_apps.apps.wildfire.default_fire_year`,
            when it is before ``year_b``), ``"post-fire"`` (``year_b``) or ``"recovery"``.
        similarity: The pixel's cosine similarity to its own ``year_a`` vector; None without data.
        radar: The pixel has Sentinel-1 that year; None without data.
        radar_matches: Its radar regime is the same as in ``year_a``; None without data.
        comparable: Valid in this year and ``year_a`` with the same radar regime, as the wildfire
            app requires of a pair it compares.
        reference_median: Median similarity of the unburned reference used (``s_ref_t``: the
            window's own, or the precomputed run's), for ``year_b`` and later when the click has a
            burned and unburned split.
        recovery_index: The pixel's ``R_t``, for ``year_b`` and later when it burned; None where the
            app leaves it undefined.
        window_recovered_fraction: Share of the window's burned pixels recovered that year.
        window_median_recovery_index: Median ``R_t`` of the window's burned pixels.
    """

    year: int
    role: str
    similarity: float | None
    radar: bool | None
    radar_matches: bool | None
    comparable: bool
    reference_median: float | None = None
    recovery_index: float | None = None
    window_recovered_fraction: float | None = None
    window_median_recovery_index: float | None = None


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """The answer to one click: the pixel's curve, the window's figures, and what the click read.

    Attributes:
        lon: Clicked longitude, wrapped into -180 to 180.
        lat: Clicked latitude.
        zone: Zone group read.
        row: The pixel's row on the zone grid.
        col: The pixel's column on the zone grid.
        chunk: (chunk row, chunk col) of the window.
        rows: (start, stop) rows of the window on the zone grid.
        cols: (start, stop) columns of the window on the zone grid.
        window_lonlat: The window's outline in lon/lat.
        snapshot: The store snapshot read.
        year_a: Pre-fire year.
        year_b: Post-fire year.
        fire_year: The fire's year (:func:`~tessera_apps.apps.wildfire.default_fire_year`).
        through: Last recovery year.
        years_fetched: Years this click read from the store.
        years_cached: Years this click took from the cache.
        bytes_read: Chunk payload bytes this click read (the reader's count).
        read_s: Seconds spent reading.
        wall_s: Seconds the click took in all.
        points: The curve, one point per year.
        threshold: The threshold choice (:meth:`ThresholdChoice.record`; method ``"override"`` when
            it is the precomputed run's), or None.
        threshold_note: Why no threshold could be chosen, when none was.
        run: The precomputed run whose threshold and reference the click took
            (:meth:`RunSettings.record`), or None when they are the window's own.
        window_pixels: Pixels in the window.
        classified_pixels: Window pixels with a distance (valid in both years, same regime).
        burned_pixels: Window pixels burned after the opening and the minimum area.
        reference_pixels: Pixels of the unburned reference used: the window's own, or the
            precomputed run's in ``year_b``.
        pixel_distance: The pixel's distance between ``year_a`` and ``year_b``; None when not classified.
        pixel_burned: Whether the pixel burned; None when it is not classified or there is no
            threshold.
        split: The click has burned pixels and an unburned reference, so ``R_t`` exists.
        reference_check: The wildfire app's check of the window's own reference, when it uses one
            and has a split: ``years`` (``[year_a - 1, year_a]``), ``pre_fire_median``,
            ``post_fire_median`` and ``drift`` (pre minus post, None when not computed), and
            ``not_run`` (why not, or None).
        radar_free_scar: Share of the window's burned pixels that are radar-free in ``year_a`` and
            in ``year_b``, or None without burned pixels.
        notes: What the page should say about this click (no split, did not burn, warnings, the
            settings used).
    """

    lon: float
    lat: float
    zone: str
    row: int
    col: int
    chunk: tuple[int, int]
    rows: tuple[int, int]
    cols: tuple[int, int]
    window_lonlat: BaseGeometry
    snapshot: str
    year_a: int
    year_b: int
    fire_year: int
    through: int
    years_fetched: list[int]
    years_cached: list[int]
    bytes_read: int
    read_s: float
    wall_s: float
    points: list[YearPoint]
    threshold: dict[str, Any] | None
    threshold_note: str | None
    run: dict[str, Any] | None
    window_pixels: int
    classified_pixels: int
    burned_pixels: int
    reference_pixels: int
    pixel_distance: float | None
    pixel_burned: bool | None
    split: bool
    reference_check: dict[str, Any] | None
    radar_free_scar: tuple[float, float] | None
    notes: list[str]

    @property
    def from_cache(self) -> bool:
        """Every window came from the cache: the click read nothing."""
        return not self.years_fetched

    @property
    def window_km(self) -> tuple[float, float]:
        """(height, width) of the window in km."""
        return (self.rows[1] - self.rows[0]) * io.PIXEL_M / 1000.0, (self.cols[1] - self.cols[0]) * io.PIXEL_M / 1000.0

    def point(self, year: int) -> YearPoint:
        """The curve's point for ``year``.

        Raises:
            KeyError: The curve has no such year.
        """
        for p in self.points:
            if p.year == year:
                return p
        raise KeyError(year)

    def rows_for_table(self) -> list[dict[str, Any]]:
        """The curve as plain rows for the page's table."""
        return [dataclasses.asdict(p) for p in self.points]


def _years_list(years: Sequence[int], last: str = "and") -> str:
    """``[2022, 2023, 2024]`` as ``"2022, 2023 and 2024"``."""
    text = [str(y) for y in years]
    return text[0] if len(text) == 1 else f"{', '.join(text[:-1])} {last} {text[-1]}"


def _pre_fire_years_that_work(published: Sequence[int], through: int) -> list[int]:
    """Pre-fire years ``y`` with every year from ``y`` to ``through`` published and room for a post-fire year."""
    have = set(published)
    return [y for y in sorted(have) if y + 1 <= through and all(x in have for x in range(y, through + 1))]


def _years_error(
    exc: io.YearsNotPublishedError, zone: str, published: Sequence[int], *, year_a: int, year_b: int, through: int
) -> ClickError:
    """The page's message for years the zone has not published, naming years that work."""
    missing = sorted({y for years in exc.missing.values() for y in years})
    have = _years_list(published) if published else "no year"
    text = (
        f"Zone {zone} has not published {_years_list(missing)} (it publishes {have}). The click reads every year from "
        f"the pre-fire year ({year_a}) to the last recovery year ({through}), so each of them must be published."
    )
    works = _pre_fire_years_that_work(published, through)
    if works:
        text += f" With the last recovery year {through}, a pre-fire year of {_years_list(works, 'or')} works here."
    else:
        for later in sorted(published, reverse=True):
            works = _pre_fire_years_that_work(published, later)
            if works:
                text += (
                    f" No pre-fire year works with the last recovery year {through}; with the last recovery year "
                    f"{later}, a pre-fire year of {_years_list(works, 'or')} does."
                )
                break
        else:
            text += " No two consecutive years are published here, so no click in this zone can work."
    have_set = set(published)
    stop = max((t for t in range(year_b, through) if all(x in have_set for x in range(year_a, t + 1))), default=None)
    if stop is not None:
        text += (
            f" Or keep the pre-fire year {year_a} and the post-fire year {year_b} and set the last recovery year to "
            f"{stop}."
        )
    return ClickError("years", f"{text} Change the years above the map, or click in another zone.")


def _check_years(year_a: int, year_b: int, through: int) -> None:
    """Refuse years the click cannot use, in the page's words (the year inputs above the map).

    Raises:
        ClickError: ``kind`` ``"years"``.
    """
    message = None
    if year_b <= year_a:
        message = (
            f"The post-fire year ({year_b}) must be after the pre-fire year ({year_a}): the pre-fire year is the year "
            "before the fire and the post-fire year the year after it."
        )
    elif year_a < FIRST_YEAR or year_b > LAST_YEAR:
        message = (
            f"Years must lie in {FIRST_YEAR} to {LAST_YEAR}; got the pre-fire year {year_a} and the post-fire year "
            f"{year_b}."
        )
    elif not year_b <= through <= LAST_YEAR:
        message = (
            f"The last recovery year ({through}) must lie between the post-fire year ({year_b}) and {LAST_YEAR}, the "
            "last year in the store."
        )
    if message is not None:
        raise ClickError("years", f"{message} Change the years above the map.")


def _page_words(text: str) -> str:
    """The wildfire app's message with its argument names replaced by the page's."""
    for name, words in PAGE_NAMES.items():
        text = text.replace(name, words)
    return text


def _no_data(where: Located, year: int, which: str) -> ClickError:
    return ClickError(
        "no_data",
        f"The store has no data at ({where.lon:.5f}, {where.lat:.5f}) in {year}, the {which} year (zone {where.zone}, "
        f"row {where.row}, column {where.col}: a NaN scale or an all-zero vector, such as open water or a gap in "
        f"the {year} embeddings). Click on land, or choose another {which} year.",
    )


def _role(year: int, year_a: int, year_b: int, fire_year: int) -> str:
    if year < year_a:
        return "stable pair"
    if year == year_a:
        return "pre-fire"
    if year == year_b:
        return "post-fire"
    if year > year_b:
        return "recovery"
    if year < fire_year:
        return "before the fire"
    return "fire" if year == fire_year else "after the fire"


def _method_name(method: str) -> str:
    """How the page names a threshold method: ``otsu`` as Otsu, the others as the app records them."""
    return "Otsu" if method == "otsu" else method


def _optional(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _median(values: np.ndarray) -> float | None:
    """Median of the finite values, in float64 as the wildfire report takes it; None when there are none."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else None


def _figures_against(
    s_burned: np.ndarray, s_post_burned: np.ndarray, s_ref: float, s_ref_p10: float
) -> dict[str, float | int]:
    """:func:`~tessera_apps.apps.wildfire.recovery_metrics`, given the reference's median and 10th percentile."""
    sb = np.asarray(s_burned, dtype=np.float64).ravel()
    sp = np.asarray(s_post_burned, dtype=np.float64).ravel()
    keep = np.isfinite(sb) & np.isfinite(sp)
    sb, sp = sb[keep], sp[keep]
    r = wildfire.recovery_index(sb, sp, s_ref)
    defined = r[np.isfinite(r)]
    return {
        "median_sim_unburned": s_ref,
        "recovered_fraction": float(np.mean(sb >= s_ref_p10)) if sb.size else math.nan,
        "median_recovery_index": float(np.median(defined)) if defined.size else math.nan,
    }


def run_click(
    store: DemoStore,
    lon: float,
    lat: float,
    *,
    year_a: int = DEFAULT_YEAR_A,
    year_b: int = DEFAULT_YEAR_B,
    through: int = LAST_YEAR,
) -> Trajectory:
    """The wildfire recovery trajectory of the pixel at (``lon``, ``lat``), from the one chunk that holds it.

    Args:
        store: The shared reader, snapshot, cache and precomputed runs.
        lon: Clicked longitude; one outside -180 to 180 is wrapped into it.
        lat: Clicked latitude.
        year_a: The year before the fire.
        year_b: The year after the fire.
        through: The last recovery year.

    Returns:
        The trajectory.

    Raises:
        ClickError: Says what to change: an argument or a year the zone has not published
            (``"years"``), a point outside the store's coverage (``"outside"``), or no data at the
            pixel in ``year_a`` or ``year_b`` (``"no_data"``). Each is raised before any chunk read,
            except ``"no_data"``, which needs the read.
    """
    t0 = time.perf_counter()
    reads = _Reads()
    lon = wrap_longitude(lon)
    with store.lock:
        store.clicks += 1
    try:
        return _click(store, lon, lat, year_a=year_a, year_b=year_b, through=through, reads=reads, t0=t0)
    except ClickError as exc:
        exc.bytes_read = reads.bytes
        exc.wall_s = time.perf_counter() - t0
        logger.info(
            "Click at (%.5f, %.5f) stopped (%s) after reading %d bytes in %.2f s.",
            lon,
            lat,
            exc.kind,
            exc.bytes_read,
            exc.wall_s,
        )
        raise


def _click(
    store: DemoStore,
    lon: float,
    lat: float,
    *,
    year_a: int,
    year_b: int,
    through: int,
    reads: _Reads,
    t0: float,
) -> Trajectory:
    """The body of :func:`run_click`, counting what it reads in ``reads``."""
    _check_years(year_a, year_b, through)
    fire_year = wildfire.default_fire_year(year_a, year_b)
    try:
        wildfire._check_args(
            year_a=year_a,
            year_b=year_b,
            fire_year=fire_year,
            recovery_through=through,
            threshold=None,
            min_area_ha=MIN_AREA_HA,
            recovery_sim=wildfire.DEFAULT_RECOVERY_SIM,
            reference=None,
        )
    except AppInputError as exc:
        raise ClickError("years", f"{_page_words(str(exc))} Change the years above the map.") from exc
    reader = store.reader
    where = locate(reader, lon, lat)
    years = list(range(year_a, through + 1))
    published = reader.years_complete(where.zone)
    try:
        reader.check_years([where.zone], years)
    except io.YearsNotPublishedError as exc:
        raise _years_error(exc, where.zone, published, year_a=year_a, year_b=year_b, through=through) from exc
    prev = year_a - 1
    prev_ok = prev >= FIRST_YEAR and prev in published
    run = next(
        (r for r in store.runs if r.covers(store.snapshot, lon, lat, year_a=year_a, year_b=year_b, through=through)),
        None,
    )
    fallback = prev_ok and run is None

    wins = _windows(store, where, years, reads)
    lr, lc = where.local
    wa, wb = wins[year_a], wins[year_b]
    if not wa.valid[lr, lc]:
        raise _no_data(where, year_a, "pre-fire")
    if not wb.valid[lr, lc]:
        raise _no_data(where, year_b, "post-fire")

    # The wildfire app's stream (wildfire._stream) on one block: pixels valid in both years whose
    # radar regime matches year_a's, NaN elsewhere.
    h, w = where.shape
    window_mask = np.ones((h, w), dtype=bool)
    dist = metrics.cosine_distance_map(wa, wb)
    same_b = metrics.same_radar_regime(wa, wb)
    dist_cls = np.where(np.isfinite(dist) & same_b, dist, np.float32(np.nan)).astype(np.float32)
    raw_sims: dict[int, np.ndarray] = {}
    used_sims: dict[int, np.ndarray] = {}
    same: dict[int, np.ndarray] = {}
    for y in years:
        sim = metrics.cosine_similarity_map(wa, wins[y])
        same[y] = metrics.same_radar_regime(wa, wins[y])
        raw_sims[y] = sim
        used_sims[y] = np.where(np.isfinite(sim) & same[y], sim, np.float32(np.nan)).astype(np.float32)

    # The stable pair, year_a - 1 to year_a, read at most once: for the threshold fallback and for
    # the check of the window's own unburned reference.
    pre: dict[str, Any] = {}

    def read_prev() -> None:
        if "win" not in pre:
            w_prev = _windows(store, where, [prev], reads)[prev]
            pre["win"] = w_prev
            pre["dist"] = wildfire._prefire_distance(w_prev, wa, window_mask)

    def stable_values() -> np.ndarray:
        read_prev()
        d = pre["dist"]
        return np.asarray(d[np.isfinite(d)])

    # The threshold: the precomputed run's, else Otsu with the bimodality check, else the stable pair.
    values = dist_cls[np.isfinite(dist_cls)]
    stable_fn: Callable[[], np.ndarray] | None = stable_values if fallback else None
    choice: wildfire.ThresholdChoice | None
    threshold_note = None
    try:
        choice = wildfire.select_threshold(
            values,
            override=None if run is None else run.threshold,
            stable_pair=(prev, year_a) if fallback else None,
            stable_values=stable_fn,
        )
    except wildfire.WildfireError as exc:
        choice, threshold_note = None, str(exc)

    plan = _window_plan(where, window_mask)
    classified = np.isfinite(dist_cls)
    ref = np.zeros((h, w), dtype=bool)
    if choice is not None:
        raw = classified & (dist_cls > choice.threshold)
        cleaned, _, _, _ = mosaic_and_clean(
            {where.zone: raw}, plan, min_pixels=metrics.area_to_pixels(MIN_AREA_HA), dst_zone=where.zone
        )
        burned = cleaned[where.zone]
        if run is None:
            ref = wildfire.recovery_reference(plan, {where.zone: dist_cls}, choice.threshold, {where.zone: burned})[
                where.zone
            ]
    else:
        burned = np.zeros((h, w), dtype=bool)
    n_burned = int(burned.sum())
    n_ref = int(ref.sum()) if run is None else int(run.reference_pixels[year_b])
    split = choice is not None and n_burned > 0 and n_ref > 0
    pixel_classified = bool(classified[lr, lc])
    pixel_burned = bool(burned[lr, lc]) if pixel_classified and choice is not None else None

    # The wildfire app's check of its own reference: as similar to itself after the fire as before?
    reference_check: dict[str, Any] | None = None
    if run is None and split:
        reference_check = {
            "years": [prev, year_a],
            "pre_fire_median": None,
            "post_fire_median": None,
            "drift": None,
            "not_run": None,
        }
        if prev_ok:
            read_prev()
            pre_med = _median((np.float32(1.0) - pre["dist"])[ref])
            post_med = _median((np.float32(1.0) - dist_cls)[ref])
            reference_check.update(pre_fire_median=pre_med, post_fire_median=post_med)
            if pre_med is not None and post_med is not None:
                reference_check["drift"] = pre_med - post_med
            else:
                reference_check["not_run"] = (
                    f"no pixel of the window's unburned reference is comparable between {prev} and {year_a}"
                )
        elif prev < FIRST_YEAR:
            reference_check["not_run"] = f"{prev} is before {FIRST_YEAR}, the store's first year"
        else:
            reference_check["not_run"] = f"zone {where.zone} has not published {prev}"
    radar_free_scar = None
    if n_burned:
        radar_free_scar = (
            float((wa.radar_free & burned).sum()) / n_burned,
            float((wb.radar_free & burned).sum()) / n_burned,
        )

    points: list[YearPoint] = []
    if "win" in pre:
        prev_sim = metrics.cosine_similarity_map(wa, pre["win"])
        prev_same = metrics.same_radar_regime(wa, pre["win"])
        valid_prev = bool(np.isfinite(prev_sim[lr, lc]))
        points.append(
            YearPoint(
                year=prev,
                role=_role(prev, year_a, year_b, fire_year),
                similarity=_optional(float(prev_sim[lr, lc])),
                radar=bool(pre["win"].radar_present[lr, lc]) if valid_prev else None,
                radar_matches=bool(prev_same[lr, lc]) if valid_prev else None,
                comparable=valid_prev and bool(prev_same[lr, lc]),
            )
        )
    s_post = np.float32(1.0) - dist_cls
    mismatched: list[int] = []
    no_data: list[int] = []
    for y in years:
        s_raw = float(raw_sims[y][lr, lc])
        valid = math.isfinite(s_raw)
        matches = bool(same[y][lr, lc]) if valid else None
        if matches is False:
            mismatched.append(y)
        if not valid:
            no_data.append(y)
        point = YearPoint(
            year=y,
            role=_role(y, year_a, year_b, fire_year),
            similarity=_optional(s_raw),
            radar=bool(wins[y].radar_present[lr, lc]) if valid else None,
            radar_matches=matches,
            comparable=bool(valid and matches),
        )
        if y >= year_b and split:
            s_y = s_post if y == year_b else used_sims[y]
            fin = np.isfinite(s_y)
            s_b, s_p = s_y[burned & fin], s_post[burned & fin]
            if run is None:
                figures = wildfire.recovery_metrics(s_b, s_p, s_y[ref & fin])
            else:
                figures = _figures_against(s_b, s_p, run.reference_median[y], run.reference_p10[y])
            s_ref = float(figures["median_sim_unburned"])
            r_pixel = None
            if pixel_burned and math.isfinite(float(s_y[lr, lc])):
                r_pixel = _optional(float(wildfire.recovery_index(s_y[lr, lc], s_post[lr, lc], s_ref)))
            point = dataclasses.replace(
                point,
                reference_median=_optional(s_ref),
                recovery_index=r_pixel,
                window_recovered_fraction=_optional(float(figures["recovered_fraction"])),
                window_median_recovery_index=_optional(float(figures["median_recovery_index"])),
            )
        points.append(point)

    notes = _notes(
        where,
        year_a=year_a,
        year_b=year_b,
        choice=choice,
        threshold_note=threshold_note,
        run=run,
        other_runs=store.runs,
        n_burned=n_burned,
        n_ref=n_ref,
        pixel_classified=pixel_classified,
        pixel_burned=pixel_burned,
        pixel_distance=float(dist_cls[lr, lc]),
        mismatched=mismatched,
        no_data=no_data,
        reference_check=reference_check,
        radar_free_scar=radar_free_scar,
    )
    wall = time.perf_counter() - t0
    logger.info(
        "Click at (%.5f, %.5f): zone %s chunk %s, read %d bytes (%s) in %.2f s, cached %s; %s; %.2f s in all.",
        lon,
        lat,
        where.zone,
        where.chunk,
        reads.bytes,
        reads.fetched,
        reads.seconds,
        reads.cached,
        "the window's own threshold and reference" if run is None else f"{run.label}'s threshold and reference",
        wall,
    )
    return Trajectory(
        lon=lon,
        lat=lat,
        zone=where.zone,
        row=where.row,
        col=where.col,
        chunk=where.chunk,
        rows=(where.rows.start, where.rows.stop),
        cols=(where.cols.start, where.cols.stop),
        window_lonlat=where.window_lonlat(),
        snapshot=store.snapshot,
        year_a=year_a,
        year_b=year_b,
        fire_year=fire_year,
        through=through,
        years_fetched=sorted(reads.fetched),
        years_cached=sorted(reads.cached),
        bytes_read=reads.bytes,
        read_s=reads.seconds,
        wall_s=wall,
        points=sorted(points, key=lambda p: p.year),
        threshold=None if choice is None else choice.record(),
        threshold_note=threshold_note,
        run=None if run is None else run.record(),
        window_pixels=h * w,
        classified_pixels=int(classified.sum()),
        burned_pixels=n_burned,
        reference_pixels=n_ref,
        pixel_distance=_optional(float(dist_cls[lr, lc])),
        pixel_burned=pixel_burned,
        split=split,
        reference_check=reference_check,
        radar_free_scar=radar_free_scar,
        notes=notes,
    )


def _window_plan(where: Located, mask: np.ndarray) -> AOIPlan:
    """The window as a one-zone AOI plan, so the app's cleaning and reference rules apply to it."""
    zone_plan = ZonePlan(zone=where.zone, grid=where.grid, rows=where.rows, cols=where.cols, mask=mask)
    return AOIPlan(geometry=where.window_lonlat(), zones=[zone_plan])


def _threshold_notes(choice: wildfire.ThresholdChoice, *, year_a: int, year_b: int) -> list[str]:
    """The window's own threshold, as the wildfire app reports it (without its --threshold advice)."""
    if choice.method != "stable_p99":
        return [f"Threshold: {choice.note}."] if choice.note else []
    flagged = choice.fallback_flagged_share
    if choice.fallback_suspect:
        otsu_text = "" if choice.otsu is None else f" (Otsu would put it at {choice.otsu.threshold:.3f})"
        return [
            f"Threshold: the stable pair {choice.stable_pair} has a 99th percentile distance of "
            f"{choice.threshold:.3f}, at or above the {wildfire.FALLBACK_FLAG_PERCENTILE:g}th percentile of the "
            f"{year_a} to {year_b} distances, so the fallback flags only {flagged or 0.0:.1%} of pixels. Either "
            f"nothing changed between {year_a} and {year_b}, or the stable pair is not stable{otsu_text}."
        ]
    share = (
        ""
        if flagged is None
        else f" Here {flagged:.1%} of the window's classified pixels are over it before the 3 x 3 opening and the "
        "minimum area."
    )
    return [
        f"The threshold is the app's fallback: {choice.note}. The fallback marks the window's most-changed pixels as "
        f"burned even where nothing burned.{share}"
    ]


def _notes(
    where: Located,
    *,
    year_a: int,
    year_b: int,
    choice: wildfire.ThresholdChoice | None,
    threshold_note: str | None,
    run: RunSettings | None,
    other_runs: Sequence[RunSettings],
    n_burned: int,
    n_ref: int,
    pixel_classified: bool,
    pixel_burned: bool | None,
    pixel_distance: float,
    mismatched: Sequence[int],
    no_data: Sequence[int],
    reference_check: Mapping[str, Any] | None,
    radar_free_scar: tuple[float, float] | None,
) -> list[str]:
    """What the page says about a click, in the order it says it; the last note names the settings used."""
    h, w = where.shape
    size = f"{h * io.PIXEL_M / 1000:g} km by {w * io.PIXEL_M / 1000:g} km"
    whose = "the window's" if run is None else f"{run.label}'s"
    notes: list[str] = []
    no_curve = "There is no recovery index; the similarity curve is still shown."
    if choice is None:
        notes.append(f"The window has no burned and unburned split: {threshold_note} {no_curve}")
    elif n_burned == 0:
        method = choice.method if run is None else f"{run.label}'s, by {_method_name(run.method)}"
        notes.append(
            f"The window has no burned and unburned split: no pixel's {year_a} to {year_b} distance is above the "
            f"threshold {choice.threshold:.3f} ({method}) in a patch of at least {MIN_AREA_HA:g} ha after "
            f"the 3 x 3 opening. {no_curve}"
        )
    elif run is None and n_ref == 0:
        notes.append(
            f"The window has no burned and unburned split: no pixel is unburned by the app's rule (distance at most "
            f"{wildfire.REFERENCE_DISTANCE_SHARE:g} x the threshold, more than {wildfire.REFERENCE_BUFFER_PX} pixels "
            f"from any burned pixel), so there is no unburned reference in this window. {no_curve}"
        )
    elif run is None and n_ref < wildfire.MIN_REFERENCE_PIXELS:
        notes.append(
            f"The window's unburned reference has only {n_ref} pixels, under the {wildfire.MIN_REFERENCE_PIXELS} the "
            "wildfire app warns at, so its recovery figures are unreliable."
        )
    if run is None and choice is not None:
        notes.extend(_threshold_notes(choice, year_a=year_a, year_b=year_b))
    if reference_check is not None:
        drift = reference_check.get("drift")
        if reference_check.get("not_run"):
            notes.append(
                "The wildfire app's check of its unburned reference (is it as similar to itself after the fire as "
                f"before it?) was not run on the window's: {reference_check['not_run']}."
            )
        elif drift is not None and drift > wildfire.REFERENCE_DRIFT_WARN:
            prev, a = reference_check["years"]
            notes.append(
                f"The window's unburned reference is less similar to itself after the fire (median "
                f"{reference_check['post_fire_median']:.3f}, {year_a} to {year_b}) than before it "
                f"({reference_check['pre_fire_median']:.3f}, {prev} to {a}), by {drift:.3f}, more than the "
                f"{wildfire.REFERENCE_DRIFT_WARN:g} the wildfire app warns at: it may still hold fire-affected land, "
                "which inflates the recovery index here."
            )
    if radar_free_scar is not None and max(radar_free_scar) > wildfire.RADAR_FREE_FLAG:
        notes.append(
            f"More than {wildfire.RADAR_FREE_FLAG:.0%} of the window's scar is radar-free (optical only): "
            f"{radar_free_scar[0]:.1%} of its burned pixels in {year_a} and {radar_free_scar[1]:.1%} in {year_b}. "
            "Expect more noise, as the wildfire app warns."
        )
    if not pixel_classified:
        notes.append(
            f"This pixel is not classified: its radar regime differs between {year_a} and {year_b}, so the wildfire "
            "app compares neither its distance nor its recovery. The similarity curve is still shown."
        )
    elif pixel_burned is False and choice is not None and n_burned > 0:
        why = (
            f"its {year_a} to {year_b} distance {pixel_distance:.3f} is at or under the threshold"
            if pixel_distance <= choice.threshold
            else f"its {year_a} to {year_b} distance {pixel_distance:.3f} is over the threshold, but its patch is "
            f"under {MIN_AREA_HA:g} ha after the 3 x 3 opening"
        )
        notes.append(
            f"This pixel did not burn by {whose} threshold {choice.threshold:.3f} ({why}), so it has no "
            "recovery index. The similarity curve is still shown."
        )
    if no_data:
        notes.append(
            f"The store has no data at this pixel in {', '.join(str(y) for y in no_data)}; the curve skips those years."
        )
    if mismatched:
        notes.append(RADAR_NOTE.format(year_a=year_a, years=", ".join(str(y) for y in mismatched)))
    if run is not None:
        notes.append(
            RUN_NOTE.format(
                aoi=f"{run.name.capitalize()} demo AOI",
                year_a=run.year_a,
                year_b=run.year_b,
                label=run.label,
                threshold=run.threshold,
                method=_method_name(run.method),
                source=run.source,
                min_area=MIN_AREA_HA,
                size=size,
            )
        )
    else:
        such_as = "".join(
            f", such as {r.label} (threshold {r.threshold:.3f} by {_method_name(r.method)} over "
            f"its whole AOI), whose settings the page takes inside its AOI at {r.year_a} and {r.year_b}"
            for r in other_runs[:1]
        )
        notes.append(REFERENCE_NOTE.format(size=size, such_as=such_as))
    return notes


__all__ = [
    "CACHE_MB_ENV",
    "DEFAULT_YEAR_A",
    "DEFAULT_YEAR_B",
    "PRECOMPUTED",
    "SNAPSHOT",
    "SNAPSHOT_ENV",
    "ClickError",
    "DemoStore",
    "Located",
    "RunSettings",
    "Trajectory",
    "WindowCache",
    "YearPoint",
    "cache_bytes",
    "load_run_settings",
    "locate",
    "open_store",
    "resolve_snapshot",
    "run_click",
    "wrap_longitude",
]
