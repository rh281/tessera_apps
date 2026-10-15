"""Plumbing every app shares: load the AOI, prepare a run, clean masks across zones, write the report.

An app run has the same shape whatever the app (brief section 4 intro and section 6, as corrected by
the survey):

1. :func:`load_aoi` reads the AOI from a vector file in any CRS or a bundled demo name, and records
   the sha256 of the file's bytes and of the geometry it loaded.
2. :func:`prepare_run` opens the reader, plans the AOI (zones, size, window checks), checks every
   year the app will read against ``years_complete`` for every zone the AOI touches before any chunk
   is read (survey S2, S3), and returns ``None`` when the outputs of an identical earlier run are
   still current (survey S19). The arguments digest covers the params, the years, the AOI file and
   geometry, the seed, the store snapshot and the code that shapes the outputs. Otherwise it checks
   that the output directory is writable, runs the app's preflight (for example loading a
   validation reference), clears the previous run's report and returns a :class:`RunContext`, all
   before any chunk read.
3. The app streams blocks, writes its rasters, tables and preview, and for a multi-zone AOI cleans
   its masks with :func:`mosaic_and_clean`, which runs on the mosaic of every zone and never zone
   by zone (checkpoint 2, metrics M5).
4. :func:`finish_run` writes ``<app>_report.json`` last, with the arguments digest, provenance (the
   years actually read), output hashes, timings and the chunk payload bytes read.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import rasterio
import rasterio.crs
import rasterio.errors
import shapely
from pyproj import CRS
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from tessera_apps import __version__, demo, export, io, metrics, validate
from tessera_apps.export import GridArray, OutputPaths
from tessera_apps.io import AOIPlan, Reader, ZonePlan

logger = logging.getLogger(__name__)

#: What ``bytes_read`` in a report counts.
BYTES_READ_NOTE = (
    "Chunk payload bytes returned by the zarr store (io.CountingStore), gap bytes of merged range "
    "requests included. Icechunk manifest and snapshot fetches below the zarr interface are not counted."
)

#: Modules whose source shapes every app's outputs; their hash is part of every arguments digest.
FRAMEWORK_MODULES = (
    "tessera_apps.io",
    "tessera_apps.metrics",
    "tessera_apps.export",
    "tessera_apps.validate",
    "tessera_apps.apps._common",
)

#: A run warns before any chunk read when its memory estimate exceeds this share of available memory.
MEMORY_WARN_SHARE = 0.9


class AppInputError(ValueError):
    """An argument or input file an app cannot use. The message says what to change."""


class StrictValidationError(RuntimeError):
    """``--strict-validate`` is set and the validation reference could not be loaded.

    Raised before any chunk read, so a strict run that cannot validate costs nothing.
    """


# --------------------------------------------------------------------------------------------
# AOI
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LoadedAOI:
    """An AOI ready for :func:`tessera_apps.io.plan_aoi`.

    Attributes:
        geometry: The AOI in EPSG:4326 (lon, lat): the union of the polygonal parts of its features.
        path: The file it was read from (for a demo, the bundled file when it is on disk), or None.
        sha256: Hex sha256 of the file's bytes (for a demo, of the bundled GeoJSON).
        name: The demo name, or the file's stem.
        properties: Properties of the AOI feature (a demo's ``aoi`` feature, or a file's only
            feature); empty when the file holds several AOI features.
        demo: The bundled demo's name when the AOI is a demo, else None.
    """

    geometry: BaseGeometry
    path: Path | None
    sha256: str
    name: str
    properties: dict[str, Any]
    demo: str | None = None

    @property
    def source(self) -> str:
        """Where the AOI came from: the file path, or ``demo:<name>``."""
        if self.demo is not None:
            return f"demo:{self.demo}"
        return str(self.path) if self.path is not None else self.name

    @property
    def geometry_sha256(self) -> str:
        """Hex sha256 of the loaded geometry (:func:`geometry_sha256`).

        ``sha256`` covers one file's bytes only. A shapefile's AOI also depends on its sidecars (the
        ``.prj`` CRS, the ``.dbf`` role column), so the run digest covers this hash too: any edit
        that changes the AOI that was loaded changes it (survey S19).
        """
        return geometry_sha256(self.geometry)


def geometry_sha256(geometry: BaseGeometry) -> str:
    """Hex sha256 of a lon/lat geometry: normalised, coordinates rounded to 1e-9 degrees, as 2-D WKB.

    The rounding (about 0.1 mm) keeps the hash stable across tiny floating point differences in a
    reprojection, while any real change to the AOI changes it.
    """
    g = shapely.normalize(geometry)
    g = shapely.transform(g, lambda xy: np.round(xy, 9) + 0.0)  # + 0.0 turns -0.0 into 0.0
    wkb = shapely.to_wkb(g, output_dimension=2, byte_order=1, include_srid=False)
    return hashlib.sha256(bytes(wkb)).hexdigest()


def _polygon_union(geometries: Sequence[BaseGeometry]) -> BaseGeometry:
    """Union of the polygonal parts of ``geometries``; invalid geometries are repaired first."""
    parts = []
    for g in geometries:
        if g is None or g.is_empty:
            continue
        poly = io._polygonal(g if g.is_valid else shapely.make_valid(g))
        if not poly.is_empty and poly.area > 0:
            parts.append(poly)
    if not parts:
        return shapely.Polygon()
    return parts[0] if len(parts) == 1 else shapely.union_all(parts)


def _plain_properties(props: Mapping[str, Any]) -> dict[str, Any]:
    """Feature properties as plain JSON values (numpy scalars, NaN and timestamps converted)."""
    out = {}
    for k, v in props.items():
        try:
            out[str(k)] = export.to_jsonable(v)
        except (TypeError, ValueError):
            out[str(k)] = str(v)
    return out


def _load_demo(key: str) -> LoadedAOI:
    resource = files("tessera_apps.data").joinpath(f"{key}.geojson")
    data = resource.read_bytes()
    on_disk = Path(str(resource))
    geometry = _polygon_union([demo.demo_aoi(key)])
    return LoadedAOI(
        geometry=geometry,
        path=on_disk if on_disk.is_file() else None,
        sha256=hashlib.sha256(data).hexdigest(),
        name=key,
        properties=_plain_properties(demo.demo_properties(key)),
        demo=key,
    )


def load_aoi(spec: str | Path | LoadedAOI) -> LoadedAOI:
    """Load an AOI from a vector file or a bundled demo name.

    A file (GeoJSON, GeoPackage, shapefile, anything GDAL reads) may be in any CRS; it is reprojected
    to EPSG:4326 and the polygonal parts of its features are unioned (lines and points are dropped).
    A file that declares no CRS is taken as lon/lat when its coordinates fit, and refused otherwise.
    When the file has a ``role`` column and some feature has ``role == "aoi"`` (the bundled demo
    FeatureCollections), only those features count. A bundled demo name (``caldor``, ``rondonia``,
    ``everglades``, any case) loads that demo unless a regular file of that name exists: a directory
    of that name (for example an earlier ``--out caldor``) does not hide the demo.

    Args:
        spec: Path, demo name, or an already loaded AOI (returned as is).

    Returns:
        The AOI and its file's sha256.

    Raises:
        AppInputError: The path does not exist and is not a demo name, the file has no polygon, or
            its coordinates are not lon/lat and it declares no CRS.
    """
    if isinstance(spec, LoadedAOI):
        return spec
    text = str(spec).strip()
    path = Path(text).expanduser()
    if text.lower() in demo.DEMO_NAMES and not path.is_file():
        return _load_demo(text.lower())
    if not path.exists():
        raise AppInputError(
            f"AOI {text!r} is neither an existing file nor a bundled demo ({', '.join(demo.DEMO_NAMES)}). "
            "Pass a GeoJSON, GeoPackage or shapefile path, or a demo name."
        )
    if not path.is_file():
        raise AppInputError(f"AOI {path} is not a file.")
    import geopandas as gpd

    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # pyogrio raises its own DataSourceError and friends
        raise AppInputError(f"Could not read the AOI file {path}: {exc}") from exc
    if frame.crs is None:
        if len(frame):
            west, south, east, north = frame.total_bounds
            if not (-180.0 <= west <= east <= 180.0 and -90.0 <= south <= north <= 90.0):
                raise AppInputError(
                    f"{path} declares no CRS and its coordinates are not lon/lat; save it with a CRS (a .prj file "
                    "for a shapefile)."
                )
        logger.warning("%s declares no CRS; its coordinates look like lon/lat, so EPSG:4326 is assumed.", path)
        frame = frame.set_crs("EPSG:4326")
    else:
        frame = frame.to_crs("EPSG:4326")
    if "role" in frame.columns and (frame["role"] == "aoi").any():
        frame = frame[frame["role"] == "aoi"]
    geometry = _polygon_union([g for g in frame.geometry if g is not None])
    if geometry.is_empty:
        raise AppInputError(f"{path} holds no polygon with an area; an AOI must be a polygon or multipolygon.")
    props: dict[str, Any] = {}
    if len(frame) == 1:
        props = _plain_properties({k: v for k, v in frame.iloc[0].items() if k != frame.geometry.name})
    return LoadedAOI(geometry=geometry, path=path, sha256=export.file_sha256(path), name=path.stem, properties=props)


# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class RunContext:
    """Everything an app needs once :func:`prepare_run` has decided to run.

    Attributes:
        app: App name (``wildfire`` and so on); output names start with it.
        aoi: The loaded AOI.
        reader: The store reader.
        plan: The AOI split by zone.
        paths: Output paths.
        digest: The arguments digest recorded in the report (:func:`export.run_args_digest`).
        params: The params the digest covers, including the AOI geometry hash, the code record and
            the data snapshot id.
        years_read: Every year checked up front (all the app may read).
        year_a: First year of the comparison.
        year_b: Second year.
        validate: Whether validation was requested.
        seed: Random seed.
        block: Block shape for streaming reads.
        extra_outputs: Outputs beyond :class:`export.OutputPaths`, by name.
        t0: ``time.perf_counter()`` at the start of :func:`prepare_run`.
        bytes_at_start: ``reader.bytes_read`` at the start, so a reused reader reports this run only.
        estimated_bytes: The plan's in-memory upper bound for ``year_reads`` year-reads of every block.
        validation_only: Every output is current except ``validation.json`` (an earlier validating
            run could not load its reference): the app only redoes the validation, from its own
            outputs, without reading a chunk. The previous report is kept until it is rewritten.
        estimated_memory: The app's peak memory estimate in bytes, or None.
        years_used: Years the app actually read (:meth:`mark_read`); the report's provenance lists
            these, and ``years_read`` as the years checked.
        warnings: Warnings for the report.
        timings: Stage timings in seconds for the report.
    """

    app: str
    aoi: LoadedAOI
    reader: Reader
    plan: AOIPlan
    paths: OutputPaths
    digest: str
    params: dict[str, Any]
    years_read: list[int]
    year_a: int
    year_b: int
    validate: bool
    seed: int
    block: tuple[int, int]
    extra_outputs: dict[str, Path]
    t0: float
    bytes_at_start: int
    estimated_bytes: int
    validation_only: bool = False
    estimated_memory: int | None = None
    years_used: set[int] = dataclasses.field(default_factory=set)
    warnings: list[str] = dataclasses.field(default_factory=list)
    timings: dict[str, float] = dataclasses.field(default_factory=dict)

    def warn(self, message: str) -> None:
        """Record a warning for the report and log it."""
        self.warnings.append(message)
        logger.warning("%s: %s", self.app, message)

    def lap(self, stage: str, since: float) -> float:
        """Record the seconds since ``since`` under ``stage`` and return the current time."""
        now = time.perf_counter()
        self.timings[stage] = round(self.timings.get(stage, 0.0) + now - since, 3)
        return now

    def mark_read(self, years: Iterable[int]) -> None:
        """Record that the app read chunks of ``years`` (for the report's provenance)."""
        self.years_used.update(int(y) for y in years)

    @property
    def bytes_read(self) -> int:
        """Chunk payload bytes this run has read so far."""
        return int(self.reader.bytes_read) - self.bytes_at_start


def _extras_current(report_path: Path, extras: Mapping[str, Path]) -> bool:
    """Whether every extra output exists, is not newer than the report, and has the recorded sha256."""
    if not extras:
        return True
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    recorded = report.get("extra_outputs") if isinstance(report, dict) else None
    if not isinstance(recorded, dict):
        return False
    report_time = report_path.stat().st_mtime_ns
    for name, path in extras.items():
        entry = recorded.get(name)
        if not isinstance(entry, dict) or not path.is_file() or path.stat().st_mtime_ns > report_time:
            return False
        if entry.get("sha256") != export.file_sha256(path):
            return False
    return True


def _open_reader(snapshot_id: str | None) -> Reader:
    return io.TesseraReader.open(snapshot_id=snapshot_id)


def code_record(modules: Iterable[ModuleType | str] = ()) -> dict[str, str]:
    """The code that shapes an app's outputs, for the arguments digest (survey S19).

    An upgrade, or an edit during development (the version stays ``0.1.0.dev0`` throughout), that
    changes the shared logic must not let a rerun keep outputs made by the old code. So the record
    holds the package version and the sha256 of the source of :data:`FRAMEWORK_MODULES` plus the
    app's own modules.

    Args:
        modules: The app's modules (module objects or dotted names).

    Returns:
        ``{"tessera_apps_version", "sources_sha256"}``.
    """
    import importlib

    given: dict[str, ModuleType | None] = dict.fromkeys(FRAMEWORK_MODULES)
    for m in modules:
        if isinstance(m, str):
            given.setdefault(m, None)
        else:
            given[m.__name__] = m
    digest = hashlib.sha256()
    for name in sorted(given):
        module = given[name] or sys.modules.get(name) or importlib.import_module(name)
        digest.update(name.encode("utf-8") + b"\0")
        source = getattr(module, "__file__", None)
        try:
            digest.update(Path(source).read_bytes() if source else b"")
        except OSError:
            digest.update(b"unreadable")
        digest.update(b"\0")
    return {"tessera_apps_version": __version__, "sources_sha256": digest.hexdigest()}


def available_memory() -> int | None:
    """Bytes of memory available to this process now (``MemAvailable`` on Linux), None when unknown."""
    try:
        with Path("/proc/meminfo").open(encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def check_out_dir(out_dir: Path | str) -> Path:
    """Create ``out_dir`` and prove a file can be written there, before any chunk is read.

    Raises:
        AppInputError: The directory cannot be created (a path component is a file, a read-only or
            missing mount) or written; the message names it and says to pass another ``--out``.
    """
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
        fd, probe = tempfile.mkstemp(prefix=".tessera_apps_write_check_", dir=out)
        os.close(fd)
        Path(probe).unlink()
    except OSError as exc:
        reason = exc.strerror or str(exc)
        raise AppInputError(
            f"Cannot write to the output directory {out} ({reason}). Pass a writable directory with --out."
        ) from exc
    return out


def prepare_run(
    app: str,
    aoi: str | Path | LoadedAOI,
    years_read: Sequence[int],
    year_a: int,
    year_b: int,
    out_dir: Path | str,
    params: Mapping[str, Any],
    *,
    validate: bool,
    force: bool,
    max_area_km2: float | None,
    max_zones: int = 4,
    snapshot_id: str | None = None,
    reader: Reader | None = None,
    seed: int = 0,
    block: tuple[int, int] = io.DEFAULT_BLOCK,
    extra_outputs: Mapping[str, Path] | None = None,
    year_reads: int | None = None,
    code_modules: Sequence[ModuleType | str] = (),
    memory_estimate: Callable[[AOIPlan], int] | None = None,
    allow_validation_only: bool = False,
    preflight: Callable[[RunContext], None] | None = None,
) -> RunContext | None:
    """Open the store, plan the AOI, check every year, and decide whether to run.

    Nothing here reads an embedding chunk: zone grids, ``years_complete`` and root attributes are
    metadata. In order:

    1. Open the reader (``TesseraReader.open(snapshot_id=...)``) unless one is passed in.
    2. Plan the AOI with :func:`io.plan_aoi`: refuses an AOI over ``max_area_km2`` (None lifts the
       limit), over ``max_zones`` zone groups, or spread too thinly.
    3. Check every year in ``years_read`` against ``years_complete`` for every zone the AOI touches
       (:func:`io.reader_check_years`), so a publication gap stops the run before any chunk read.
    4. Compute the arguments digest over ``params`` plus the AOI geometry hash
       (:attr:`LoadedAOI.geometry_sha256`), the code record (:func:`code_record`) and the store's
       snapshot id, and over the years, the AOI file's sha256 and the seed. Unless ``force``, return
       None when :func:`export.outputs_current` says an earlier run with the same digest left every
       output intact (and every extra output still has the sha256 its report recorded).
    5. With ``allow_validation_only``, a validating run whose outputs are all current except
       ``validation.json`` gets a context with ``validation_only`` set: the app redoes only the
       validation and reads no chunk (survey S19, an earlier run could not load its reference).
    6. Check that ``out_dir`` can be created and written (:func:`check_out_dir`), log the plan
       (zones, AOI hectares, estimated bytes and, with ``memory_estimate``, the peak memory, with a
       warning when it exceeds the memory available), and run ``preflight``.
    7. Unless ``validation_only``, remove the previous report and ``validation.json``
       (:func:`export.begin_run`). A failing preflight leaves them in place.

    Args:
        app: App name, the prefix of every output.
        aoi: AOI path, demo name or loaded AOI.
        years_read: Every year the app may read.
        year_a: First year (names the output tif).
        year_b: Second year.
        out_dir: Output directory.
        params: Thresholds and options that change the outputs.
        validate: Whether the run will validate (it then needs ``validation.json`` to be current).
        force: Run even when the outputs are current.
        max_area_km2: AOI size limit; None for no limit (``--i-know-this-is-large``).
        max_zones: Zone group limit.
        snapshot_id: Icechunk snapshot to pin when opening the reader.
        reader: A reader to use instead of opening the live store (tests pass an in-memory one).
        seed: Random seed recorded in the digest.
        block: Block shape the app streams with, for the byte estimate.
        extra_outputs: Outputs beyond the standard set, by name, checked for the skip.
        year_reads: How many times the app may read one year of every block, for the byte estimate
            (an app that reads a year twice counts it twice); one per year in ``years_read`` when None.
        code_modules: The app's own modules, hashed into the digest with the framework's.
        memory_estimate: Returns the app's peak memory in bytes for a plan.
        allow_validation_only: The app can redo only its validation from its own outputs.
        preflight: Called with the context after the plan is logged and before ``begin_run``, still
            before any chunk read: an app loads its validation reference here, so a bad reference
            fails early and leaves the previous outputs alone.

    Returns:
        The run context, or None when the outputs are current.

    Raises:
        AOIError: The AOI is too large, touches too many zones, or has no pixel.
        YearsNotPublishedError: A year is missing for some zone; names each zone-year.
        AppInputError: ``out_dir`` cannot be written, or the preflight refused an input.
    """
    t0 = time.perf_counter()
    loaded = load_aoi(aoi)
    if reader is None:
        reader = _open_reader(snapshot_id)
    elif snapshot_id is not None:
        logger.warning("A reader was passed in, so snapshot_id %s is not applied; the reader's is used.", snapshot_id)
    bytes_at_start = int(reader.bytes_read)
    plan = io.plan_aoi(reader, loaded.geometry, max_zones=max_zones, max_area_km2=max_area_km2)
    years = sorted({int(y) for y in years_read})
    io.reader_check_years(reader, plan.zone_names, years)

    paths = export.output_paths(Path(out_dir), app, year_a, year_b)
    extras = {k: Path(v) for k, v in (extra_outputs or {}).items()}
    run_params = {
        **dict(params),
        "aoi_geometry_sha256": loaded.geometry_sha256,
        "code": code_record(code_modules),
        "data_snapshot_id": reader.provenance().get("snapshot_id"),
    }
    digest = export.run_args_digest(run_params, years=years, aoi_sha256=loaded.sha256, seed=seed)
    if not force and _extras_current(paths.report, extras):
        if export.outputs_current(paths, digest, need_validation=validate):
            logger.info("%s: outputs in %s are current for these arguments; pass --force to rerun.", app, out_dir)
            return None
        validation_only = (
            allow_validation_only and validate and export.outputs_current(paths, digest, need_validation=False)
        )
    else:
        validation_only = False
    check_out_dir(out_dir)

    n_reads = len(years) if year_reads is None else year_reads
    estimated = 0 if validation_only else plan.estimated_bytes(reader, n_reads, block=block)
    ctx = RunContext(
        app=app,
        aoi=loaded,
        reader=reader,
        plan=plan,
        paths=paths,
        digest=digest,
        params=run_params,
        years_read=years,
        year_a=int(year_a),
        year_b=int(year_b),
        validate=validate,
        seed=seed,
        block=(int(block[0]), int(block[1])),
        extra_outputs=extras,
        t0=t0,
        bytes_at_start=bytes_at_start,
        estimated_bytes=int(estimated),
        validation_only=validation_only,
    )
    zones_text = ", ".join(f"{z.zone} {z.shape[0]} x {z.shape[1]}" for z in plan.zones)
    if validation_only:
        logger.info(
            "%s: outputs in %s are current except the validation, which an earlier run could not load; "
            "only the validation is redone, from the outputs on disk, without reading a chunk.",
            app,
            out_dir,
        )
    else:
        logger.info(
            "%s: AOI %s, %s ha in %d zone(s) (%s), %s pixels; years %s; at most %.2f GB of chunks to read.",
            app,
            loaded.source,
            f"{plan.area_ha:,.0f}",
            len(plan.zones),
            zones_text,
            f"{plan.n_pixels:,}",
            ", ".join(str(y) for y in years),
            estimated / 1e9,
        )
        if memory_estimate is not None:
            ctx.estimated_memory = int(memory_estimate(plan))
            available = available_memory()
            logger.info(
                "%s: peak memory about %.1f GB (%s available).",
                app,
                ctx.estimated_memory / 1e9,
                "unknown" if available is None else f"{available / 1e9:.1f} GB",
            )
            if available is not None and ctx.estimated_memory > MEMORY_WARN_SHARE * available:
                ctx.warn(
                    f"This run needs about {ctx.estimated_memory / 1e9:.1f} GB of memory and "
                    f"{available / 1e9:.1f} GB is available: it may be killed after reading "
                    f"{estimated / 1e9:.1f} GB. Free memory, or split the AOI."
                )
    if preflight is not None:
        preflight(ctx)
    if not validation_only:
        removed = export.begin_run(paths)
        if removed:
            logger.info("%s: removed the previous run's %s.", app, ", ".join(p.name for p in removed))
    return ctx


def _provenance(ctx: RunContext) -> dict[str, Any]:
    used = sorted(ctx.years_used) if ctx.years_used else list(ctx.years_read)
    provenance = export.provenance_record(
        ctx.reader,
        aoi_path=ctx.aoi.path,
        years=used,
        params=ctx.params,
        seed=ctx.seed,
        zones=ctx.plan.zone_names,
    )
    # The digest and AOI hash are the ones the run was prepared with: the demo AOI may not be a
    # file on disk, and the file may have changed since it was loaded.
    provenance["aoi"] = {
        "name": ctx.aoi.name,
        "source": ctx.aoi.source,
        "sha256": ctx.aoi.sha256,
        "geometry_sha256": ctx.aoi.geometry_sha256,
    }
    provenance["args_digest"] = ctx.digest
    provenance["years_checked"] = list(ctx.years_read)
    return provenance


def finish_run(
    ctx: RunContext,
    *,
    summary: Mapping[str, Any],
    extra: Mapping[str, Path],
    validation_written: bool,
) -> dict[str, Any]:
    """Write ``<app>_report.json``, after every other output, and return it.

    The report holds the arguments digest (what :func:`prepare_run` compares to skip a rerun), the
    provenance block (:func:`export.provenance_record`: ``years`` lists the years the app read,
    recorded with :meth:`RunContext.mark_read`, and ``years_checked`` every year checked up front),
    the app's summary, the params, wall time and stage timings, the chunk payload bytes read, the
    sha256 of every standard output (:func:`export.outputs_record`) and of each extra output, and
    the warnings.

    Args:
        ctx: From :func:`prepare_run`.
        summary: The app's results, plain values.
        extra: Extra output files this run wrote, by name (also listed in ``ctx.extra_outputs``).
        validation_written: Whether this run wrote ``validation.json``.

    Returns:
        The report as plain JSON values.

    Raises:
        FileNotFoundError: A standard or extra output is missing.
    """
    wall = time.perf_counter() - ctx.t0
    read = ctx.bytes_read
    extras = {name: Path(p) for name, p in extra.items()}
    report = {
        "app": ctx.app,
        "args_digest": ctx.digest,
        "provenance": _provenance(ctx),
        "aoi": {
            "name": ctx.aoi.name,
            "source": ctx.aoi.source,
            "sha256": ctx.aoi.sha256,
            "area_ha": ctx.plan.area_ha,
            "pixels": ctx.plan.n_pixels,
            "zones": ctx.plan.zone_names,
        },
        "summary": dict(summary),
        "params": ctx.params,
        "timings": {"wall_s": round(wall, 3), **ctx.timings},
        "bytes_read": read,
        "bytes_read_note": BYTES_READ_NOTE,
        "estimated_bytes": ctx.estimated_bytes,
        "estimated_memory": ctx.estimated_memory,
        "outputs": export.outputs_record(ctx.paths, validation=validation_written),
        "extra_outputs": {name: {"file": p.name, "sha256": export.file_sha256(p)} for name, p in extras.items()},
        "warnings": list(ctx.warnings),
    }
    export.write_json(ctx.paths.report, report)
    rate = read / wall / 1e6 if wall > 0 else math.nan
    logger.info("%s: read %.2f GB in %.1f s (%.0f MB/s).", ctx.app, read / 1e9, wall, rate)
    return cast(dict[str, Any], export.to_jsonable(report))


def finish_validation_only(
    ctx: RunContext,
    previous: Mapping[str, Any],
    *,
    summary: Mapping[str, Any],
    warnings: Sequence[str],
    validation_written: bool,
) -> dict[str, Any]:
    """Rewrite the report of a ``validation_only`` run, after ``validation.json``, and return it.

    Everything the earlier run recorded stays (its digest, provenance, params, timings and bytes
    read describe how the outputs were made); the summary, the warnings and the output hashes are
    replaced, and ``validation_redone`` records when the validation was redone, how long it took and
    the chunk bytes it read (none).

    Args:
        ctx: From :func:`prepare_run`, with ``validation_only`` set.
        previous: The report on disk.
        summary: The new summary.
        warnings: The new warnings.
        validation_written: Whether ``validation.json`` was written.

    Returns:
        The report as plain JSON values.
    """
    if not ctx.validation_only:
        raise ValueError("finish_validation_only needs a validation_only context.")
    report = dict(previous)
    report["summary"] = dict(summary)
    report["warnings"] = list(warnings)
    report["outputs"] = export.outputs_record(ctx.paths, validation=validation_written)
    report["validation_redone"] = {
        "utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "wall_s": round(time.perf_counter() - ctx.t0, 3),
        "bytes_read": ctx.bytes_read,
    }
    export.write_json(ctx.paths.report, report)
    logger.info("%s: validation redone without reading a chunk; report rewritten.", ctx.app)
    return cast(dict[str, Any], export.to_jsonable(report))


# --------------------------------------------------------------------------------------------
# Masks across zones
# --------------------------------------------------------------------------------------------


_KEPT = 1
_DROPPED = 2
_RING = np.ones((3, 3), dtype=bool)


def _write_scratch_tif(path: Path, data: np.ndarray, transform: Affine, crs: CRS) -> None:
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": "uint8",
        "crs": rasterio.crs.CRS.from_wkt(crs.to_wkt()),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)


def mosaic_and_clean(
    per_zone_masks: Mapping[str, np.ndarray],
    plan: AOIPlan,
    *,
    min_pixels: int,
    opening: int = 3,
    dst_zone: str | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, Affine, CRS]:
    """Clean per-zone binary masks as one AOI: opening and minimum component size across zone seams.

    :func:`metrics.clean_mask` must see the whole AOI at once: run zone by zone, a patch cut by a
    zone seam (120 W at Caldor) is split and each part is tested against ``min_pixels`` on its own.
    So the zones' masks are mosaicked onto one grid (:func:`export.mosaic_zones`, uint8 with nodata
    255, which is treated as background), cleaned there, and the result is mapped back onto each
    zone plan with :func:`validate.reference_from_raster`, which reprojects with GDAL's exact
    transformer. A pixel is kept only if it was set in its zone's mask and the mosaic pixel under its
    centre belongs to, or touches, a surviving patch, so cleaning never adds a pixel and a
    reprojected pixel that lands on a neighbour's mosaic pixel still follows its own patch. A
    single-zone AOI is cleaned directly.

    Args:
        per_zone_masks: Bool mask on each zone plan's grid, keyed by zone. Pixels outside the plan's
            AOI mask are ignored.
        plan: The AOI plan.
        min_pixels: Components with fewer pixels (counted on the mosaic) are dropped.
        opening: Side of the square opening; 1 skips it.
        dst_zone: Zone whose grid the mosaic uses (default: the zone with the most AOI pixels).

    Returns:
        ``(cleaned, mosaic, transform, crs)``: the cleaned bool mask per zone (False outside the AOI),
        the cleaned uint8 mosaic (1 kept, 0 not, 255 outside the AOI), its transform and its CRS.

    Raises:
        KeyError: A zone of the plan has no mask.
    """
    raws: dict[str, np.ndarray] = {}
    for z in plan.zones:
        mask = np.asarray(per_zone_masks[z.zone], dtype=bool)
        if mask.shape != z.shape:
            raise ValueError(f"Mask for {z.zone} has shape {mask.shape}; the plan window is {z.shape}.")
        raws[z.zone] = mask & z.mask
    nodata = export.BOOL_NODATA
    if len(plan.zones) == 1:
        z = plan.zones[0]
        kept = metrics.clean_mask(raws[z.zone], min_pixels=min_pixels, opening=opening) & z.mask
        mosaic = np.where(z.mask, kept.astype(np.uint8), np.uint8(nodata)).astype(np.uint8)
        return {z.zone: kept}, mosaic, z.transform, z.grid.crs
    pieces = [GridArray(raws[z.zone], z) for z in plan.zones]
    joined, transform, crs = export.mosaic_zones(pieces, dst_zone=dst_zone, nodata=nodata)
    inside = joined != nodata
    kept_mosaic = metrics.clean_mask(joined == 1, min_pixels=min_pixels, opening=opening)
    mosaic = np.where(inside, kept_mosaic.astype(np.uint8), np.uint8(nodata)).astype(np.uint8)
    # Map back a state per mosaic pixel: 1 kept, 2 dropped by the cleaning, 0 neither. A pixel of a
    # reprojected zone can land on a mosaic pixel that took its value from a neighbour, so each
    # state is first grown one pixel into the background (kept wins where both reach), and a pixel
    # then follows the patch next to it instead of its neighbour's value.
    state = np.zeros(joined.shape, dtype=np.uint8)
    state[(joined == 1) & ~kept_mosaic] = _DROPPED
    state[kept_mosaic] = _KEPT
    background = state == 0
    for value in (_DROPPED, _KEPT):
        state[background & ndimage.binary_dilation(state == value, structure=_RING)] = value
    back = _mosaic_to_zones(state, transform, crs, plan, lambda a: a == _KEPT)
    cleaned = {z.zone: back[z.zone] & raws[z.zone] for z in plan.zones}
    return cleaned, mosaic, transform, crs


def _mosaic_to_zones(
    state: np.ndarray,
    transform: Affine,
    crs: CRS,
    plan: AOIPlan,
    predicate: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
    """``predicate`` of a uint8 mosaic, mapped onto every zone plan's grid (GDAL's exact transformer)."""
    with tempfile.TemporaryDirectory(prefix="tessera_apps_mosaic_") as tmp:
        path = Path(tmp) / "state.tif"
        _write_scratch_tif(path, state, transform, crs)
        return {z.zone: validate.reference_from_raster(z, path, predicate) for z in plan.zones}


def grow_across_zones(
    per_zone_masks: Mapping[str, np.ndarray],
    plan: AOIPlan,
    *,
    pixels: int,
    dst_zone: str | None = None,
) -> dict[str, np.ndarray]:
    """AOI pixels within ``pixels`` steps (8-connected) of a set pixel of any zone's mask.

    Like :func:`mosaic_and_clean`, a multi-zone AOI is grown on the mosaic of every zone, so a set
    pixel just across a zone seam reaches its neighbours on the other side; the result is mapped
    back onto each zone plan and joined with each zone's own growth, so no set pixel or its own
    neighbourhood is ever lost to the reprojection.

    Args:
        per_zone_masks: Bool mask on each zone plan's grid, keyed by zone (outside the AOI ignored).
        plan: The AOI plan.
        pixels: Growth in pixels; 0 returns the masks.
        dst_zone: Zone whose grid the mosaic uses (default: the zone with the most AOI pixels).

    Returns:
        Bool mask per zone, False outside the plan's AOI mask.

    Raises:
        ValueError: ``pixels`` is negative, or a mask does not match its zone window.
    """
    if pixels < 0:
        raise ValueError(f"pixels must not be negative, got {pixels}.")
    raws: dict[str, np.ndarray] = {}
    for z in plan.zones:
        mask = np.asarray(per_zone_masks[z.zone], dtype=bool)
        if mask.shape != z.shape:
            raise ValueError(f"Mask for {z.zone} has shape {mask.shape}; the plan window is {z.shape}.")
        raws[z.zone] = mask & z.mask

    def grow(m: np.ndarray) -> np.ndarray:
        return ndimage.binary_dilation(m, structure=_RING, iterations=pixels) if pixels and m.any() else m.copy()

    own = {z.zone: grow(raws[z.zone]) & z.mask for z in plan.zones}
    if len(plan.zones) == 1 or not any(m.any() for m in raws.values()):
        return own
    nodata = export.BOOL_NODATA
    joined, transform, crs = export.mosaic_zones(
        [GridArray(raws[z.zone], z) for z in plan.zones], dst_zone=dst_zone, nodata=nodata
    )
    state = grow(joined == 1).astype(np.uint8)
    back = _mosaic_to_zones(state, transform, crs, plan, lambda a: a == 1)
    return {z.zone: own[z.zone] | (back[z.zone] & z.mask) for z in plan.zones}


def zone_values_from_raster(path: Path | str, plan: AOIPlan, *, band: int) -> dict[str, np.ndarray]:
    """One float band of a raster an app wrote (its mosaic), mapped back onto every zone plan's grid.

    Nearest resampling with GDAL's exact transformer, so on the mosaic's own zone the values come
    back exactly; on another zone a pixel takes the mosaic pixel under its centre.

    Args:
        path: The raster (for example ``wildfire_<a>_<b>.tif``).
        plan: The AOI plan.
        band: 1-based band.

    Returns:
        Float32 array per zone, NaN outside the plan's AOI mask and on the raster's nodata.
    """
    with rasterio.open(path) as ds:
        data = ds.read(band).astype(np.float32, copy=False)
        src_transform, src_crs, src_nodata = ds.transform, ds.crs, ds.nodata
    out: dict[str, np.ndarray] = {}
    for z in plan.zones:
        dst = np.full(z.shape, np.nan, dtype=np.float32)
        reproject(
            data,
            dst,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan if src_nodata is None else src_nodata,
            dst_transform=z.transform,
            dst_crs=rasterio.crs.CRS.from_wkt(z.grid.crs.to_wkt()),
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
            tolerance=0.0,
        )
        dst[~z.mask] = np.nan
        out[z.zone] = dst
    return out


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def stride_for_preview(shape: Sequence[int], max_side: int = export.PREVIEW_MAX_SIDE) -> int:
    """Step that strides an image of ``shape`` (rows, cols, ...) down to at most ``max_side`` per side.

    Raises:
        ValueError: ``max_side`` is below 1.
    """
    if max_side < 1:
        raise ValueError(f"max_side must be at least 1, got {max_side}.")
    longest = max(int(shape[0]), int(shape[1]))
    return max(1, math.ceil(longest / max_side))


def aoi_reference(plan_zone: ZonePlan, reference_path: Path | str) -> np.ndarray:
    """A user-supplied reference on one zone plan's grid: the ``--reference`` path of every app.

    Vector files (any CRS) are rasterized by the pixel-centre rule
    (:func:`validate.reference_from_file`). A GeoTIFF (``.tif``, ``.tiff``, ``.vrt``) counts every
    non-zero, non-nodata pixel as reference (:func:`validate.reference_from_raster`).

    The user named the file, so any failure to read it (a missing file, an HTML error page or a
    truncated download saved under the name, a raster GDAL cannot open, a projected file with no
    CRS) is an input error that names the file.

    Returns:
        Bool array of ``plan_zone.shape``, False outside the plan's AOI mask.

    Raises:
        AppInputError: The file cannot be read as a reference.
    """
    path = Path(reference_path)
    try:
        if path.suffix.lower() in {".tif", ".tiff", ".vrt"}:
            return validate.reference_from_raster(plan_zone, path, lambda a: a != 0)
        return validate.reference_from_file(plan_zone, path)
    except (OSError, ValueError, RuntimeError, rasterio.errors.RasterioError) as exc:
        raise AppInputError(
            f"Could not read the reference file {path}: {exc}. Pass a polygon file (GeoJSON, GeoPackage, "
            "shapefile) or a GeoTIFF with --reference."
        ) from exc


__all__ = [
    "BYTES_READ_NOTE",
    "FRAMEWORK_MODULES",
    "MEMORY_WARN_SHARE",
    "AppInputError",
    "LoadedAOI",
    "RunContext",
    "StrictValidationError",
    "aoi_reference",
    "available_memory",
    "check_out_dir",
    "code_record",
    "finish_run",
    "finish_validation_only",
    "geometry_sha256",
    "grow_across_zones",
    "load_aoi",
    "mosaic_and_clean",
    "prepare_run",
    "stride_for_preview",
    "zone_values_from_raster",
]
