"""Write app outputs: Cloud Optimized GeoTIFFs, summary tables, JSON reports and preview PNGs.

Every writer is atomic: it writes a temporary file in the destination directory and moves it into
place with :meth:`pathlib.Path.replace`, so a crash never leaves a partial output under the final
name, and an existing output stays intact until its replacement is complete.

Per-pixel results live on :class:`~tessera_apps.io.ZonePlan` grids, one per UTM zone. A multi-zone
AOI is joined into one raster with :func:`mosaic_zones`, which takes each ground pixel from its
nominal zone only. :func:`provenance_record`, :func:`run_args_digest` and :func:`outputs_current`
give the reproducibility block and the skip-if-current rule every app report follows (survey S19).
An app run goes:

1. ``digest = run_args_digest(...)``; skip the run if ``outputs_current(paths, digest, ...)``.
2. :func:`begin_run` removes the previous run's report and ``validation.json``.
3. Write the tif, summary tables and preview, then ``validation.json`` when the run validates.
4. Write ``report.json`` last, with ``"args_digest"`` and ``"outputs": outputs_record(...)``.

A rerun then skips only when every output exists, none is newer than the report, the report records
the same argument digest, and every output the report lists still has the sha256 it recorded. A run
that needs ``validation.json`` also needs the report to list it, so a validation file left by an
earlier run with other arguments is never taken as this run's.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import enum
import hashlib
import importlib.metadata
import json
import math
import operator
import platform
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any, cast

import numpy as np
import pandas as pd
import rasterio
import rasterio.crs
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_copy
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject, transform_bounds

from tessera_apps import __version__
from tessera_apps.io import Reader, ZonePlan, canonicalize_zone

type JSONValue = bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None

#: Largest side, in pixels, of the image drawn into a preview PNG. Larger images are strided down.
PREVIEW_MAX_SIDE = 2048
#: Smallest side, in pixels, a preview image is drawn at. Smaller images are scaled up (nearest).
PREVIEW_MIN_SIDE = 512
#: Colour of NaN pixels in a scalar preview: a neutral grey that no default colormap produces.
PREVIEW_NAN_RGB = (0.8, 0.8, 0.8)
_PREVIEW_DPI = 100

#: Nodata written for bool rasters when the caller gives none.
BOOL_NODATA = 255

#: Distributions whose versions every report records. ``gdal`` is added from rasterio.
PROVENANCE_LIBRARIES = (
    "icechunk",
    "zarr",
    "numcodecs",
    "numpy",
    "rasterio",
    "pyproj",
    "shapely",
    "scipy",
    "scikit-learn",
    "pandas",
    "geotessera",
)
_MODEL_KEYS = ("checkpoint_id", "geoemb:model", "geoemb:dimensions", "geoemb:build_version")
_SECRET_KEY = re.compile(r"secret|token|passw|credential|access[_-]?key|api[_-]?key|authorization", re.IGNORECASE)
_APP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_PREDICTOR_CODECS = frozenset({"DEFLATE", "LZW", "ZSTD", "LZMA"})
_HASH_BLOCK = 1 << 20


# --------------------------------------------------------------------------------------------
# Atomic files
# --------------------------------------------------------------------------------------------


def _scratch_name(path: Path, tag: str) -> Path:
    """A unique hidden name next to ``path``. Nothing is created, so the writer applies the umask."""
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.{tag}"


@contextlib.contextmanager
def atomic_path(path: Path | str) -> Iterator[Path]:
    """Yield a temporary path in ``path``'s directory and move it onto ``path`` if the block succeeds.

    The parent directory is created if needed. On any exception (including KeyboardInterrupt) the
    temporary file is removed and ``path`` is left as it was: absent, or the previous complete file.

    Args:
        path: Final destination.

    Yields:
        The temporary path to write. The caller creates it.
    """
    final = Path(path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = _scratch_name(final, "tmp")
    try:
        yield tmp
        tmp.replace(final)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _scratch_file(path: Path) -> Iterator[Path]:
    """A temporary path next to ``path`` that is always removed afterwards."""
    tmp = _scratch_name(path, "src.tif")
    try:
        yield tmp
    finally:
        tmp.unlink(missing_ok=True)
        Path(f"{tmp}.aux.xml").unlink(missing_ok=True)


# --------------------------------------------------------------------------------------------
# GeoTIFF
# --------------------------------------------------------------------------------------------


def _is_nan(value: float | int | None) -> bool:
    return isinstance(value, float | np.floating) and math.isnan(value)


def _check_nodata(dtype: np.dtype[Any], nodata: float | int | None) -> None:
    """Raise unless ``nodata`` can be stored in ``dtype``."""
    if nodata is None:
        return
    if not isinstance(nodata, int | float | np.integer | np.floating):
        raise TypeError(f"nodata must be a number, got {type(nodata).__name__}.")
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if _is_nan(nodata) or not float(nodata).is_integer() or not info.min <= nodata <= info.max:
            raise ValueError(
                f"nodata {nodata} cannot be stored in {dtype}: use an integer in {info.min}..{info.max}, "
                "or a float dtype for NaN."
            )
    elif dtype.kind == "f":
        if math.isfinite(nodata) and abs(float(nodata)) > float(np.finfo(dtype).max):
            raise ValueError(f"nodata {nodata} overflows {dtype}.")
    else:
        raise TypeError(f"Cannot write {dtype} data.")


def _is_nodata(a: np.ndarray, nodata: float | int) -> np.ndarray:
    """Element-wise nodata test. In float arrays NaN always counts as nodata."""
    if a.dtype.kind == "f":
        return np.isnan(a) if _is_nan(nodata) else (np.isnan(a) | (a == nodata))
    return a == nodata


def _prepare_raster(data: np.ndarray, nodata: float | int | None) -> tuple[np.ndarray, float | int | None]:
    """Return ``data`` as (bands, rows, cols) in a GeoTIFF dtype, plus the nodata to declare."""
    masked = np.ma.getmaskarray(data) if isinstance(data, np.ma.MaskedArray) else None
    arr = np.asarray(data.data if isinstance(data, np.ma.MaskedArray) else data)
    if arr.ndim == 2:
        arr = arr[None]
        if masked is not None:
            masked = masked[None]
    elif arr.ndim != 3:
        raise ValueError(f"data must be (H, W) or (B, H, W); got shape {arr.shape}.")
    if 0 in arr.shape:
        raise ValueError(f"data is empty: shape {arr.shape}.")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
        if nodata is None:
            nodata = BOOL_NODATA
    elif arr.dtype.kind == "f":
        if arr.dtype.itemsize < 4:
            arr = arr.astype(np.float32)
        if nodata is None:
            nodata = math.nan
        elif not _is_nan(nodata):
            _check_nodata(arr.dtype, nodata)
            arr = np.where(np.isnan(arr), arr.dtype.type(nodata), arr)
    elif arr.dtype.kind not in "iu":
        raise TypeError(f"Cannot write {arr.dtype} data to a GeoTIFF; use bool, integer or float.")
    _check_nodata(arr.dtype, nodata)
    if masked is not None and masked.any():
        if nodata is None:
            raise ValueError("A masked integer array needs an explicit nodata value.")
        arr = arr.copy()
        arr[masked] = nodata
    return arr, nodata


def _rio_crs(crs: CRS | rasterio.crs.CRS | str | int) -> rasterio.crs.CRS:
    """Convert a pyproj CRS, rasterio CRS, EPSG code or user string to a rasterio CRS."""
    if isinstance(crs, rasterio.crs.CRS):
        return crs
    if isinstance(crs, bool):
        raise TypeError("crs must be a CRS, an EPSG code or a string, not a bool.")
    if isinstance(crs, int | np.integer):
        return rasterio.crs.CRS.from_epsg(int(crs))
    return rasterio.crs.CRS.from_user_input(crs)


def _cog_copy(src: Path, dst: Path, options: Mapping[str, str]) -> None:
    """Translate a GeoTIFF into a Cloud Optimized GeoTIFF with GDAL's COG driver."""
    rio_copy(str(src), str(dst), driver="COG", **options)


def write_geotiff(
    path: Path,
    data: np.ndarray,
    transform: Affine,
    crs: CRS | rasterio.crs.CRS | str | int,
    *,
    nodata: float | int | None = None,
    band_names: Sequence[str] | None = None,
    compress: str = "deflate",
    blocksize: int = 512,
    overviews: bool = True,
) -> Path:
    """Write a Cloud Optimized GeoTIFF, atomically.

    The array goes to a scratch tiled GeoTIFF in the destination directory, which GDAL's COG driver
    then copies into a temporary COG that replaces ``path`` once complete. Overviews are built
    with AVERAGE resampling for float data and NEAREST otherwise (masks, classes, counts).

    Args:
        path: Output file.
        data: (H, W) or (B, H, W). bool is written as uint8 (0, 1). float16 is widened to float32.
            A masked array's masked pixels are written as nodata.
        transform: Affine transform of the top-left pixel corner.
        crs: pyproj or rasterio CRS, an EPSG code, or any string rasterio understands.
        nodata: Nodata value. When None: NaN for float data, 255 for bool data, none for integers.
            A finite nodata on float data also replaces NaN pixels.
        band_names: One description per band.
        compress: GDAL compression (deflate, lzw, zstd, lzma, none, ...). Deflate, LZW, ZSTD and
            LZMA also use the predictor that suits the dtype (2 for integers, 3 for floats).
        blocksize: Tile size in pixels, a multiple of 16.
        overviews: Build internal overviews (``OVERVIEWS=AUTO``) or none.

    Returns:
        ``path``.

    Raises:
        ValueError: Bad shape, nodata that does not fit the dtype, a band name count that does not
            match, or a bad blocksize.
        TypeError: A dtype GeoTIFF cannot hold, or a transform that is not an ``Affine``.
    """
    final = Path(path)
    if not isinstance(transform, Affine):
        raise TypeError(f"transform must be an affine.Affine, got {type(transform).__name__}.")
    if blocksize <= 0 or blocksize % 16:
        raise ValueError(f"blocksize must be a positive multiple of 16, got {blocksize}.")
    bands, nodata = _prepare_raster(data, nodata)
    count, height, width = bands.shape
    names = None if band_names is None else [str(n) for n in band_names]
    if names is not None and len(names) != count:
        raise ValueError(f"{len(names)} band names for {count} bands.")
    rio_crs = _rio_crs(crs)
    codec = compress.upper()
    options = {
        "BLOCKSIZE": str(blocksize),
        "COMPRESS": codec,
        "OVERVIEWS": "AUTO" if overviews else "NONE",
        "RESAMPLING": "AVERAGE" if bands.dtype.kind == "f" else "NEAREST",
        "BIGTIFF": "IF_SAFER",
        "NUM_THREADS": "ALL_CPUS",
    }
    if codec in _PREDICTOR_CODECS:
        options["PREDICTOR"] = "YES"
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": bands.dtype.name,
        "crs": rio_crs,
        "transform": transform,
        "nodata": nodata,
        "tiled": True,
        "blockxsize": blocksize,
        "blockysize": blocksize,
        "compress": "DEFLATE",
        "zlevel": 1,
        "bigtiff": "IF_SAFER",
    }
    with atomic_path(final) as out, _scratch_file(final) as src:
        with rasterio.open(src, "w", **profile) as ds:
            ds.write(bands)
            for i, name in enumerate(names or [], start=1):
                ds.set_band_description(i, name)
        _cog_copy(src, out, options)
    return final


# --------------------------------------------------------------------------------------------
# Mosaics across zones
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class GridArray:
    """A per-pixel result on a zone plan's grid.

    Attributes:
        data: (H, W) or (B, H, W) with (H, W) equal to ``plan.shape``.
        plan: Gives the zone, transform, CRS, and the mask of pixels that belong to this zone.
            Pixels outside ``plan.mask`` are never written by :func:`mosaic_zones`.
    """

    data: np.ndarray
    plan: ZonePlan

    def __post_init__(self) -> None:
        """Check the data matches the plan's window."""
        shape = np.shape(self.data)
        if len(shape) not in (2, 3) or tuple(shape[-2:]) != tuple(self.plan.shape):
            raise ValueError(f"data shape {shape} does not match plan {self.plan.zone} window {self.plan.shape}.")


def _as_bands(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
    return arr if arr.ndim == 3 else arr[None]


def _empty_pixels(a: np.ndarray, nodata: float | int) -> np.ndarray:
    """(H, W) True where every band of a (B, H, W) array is nodata."""
    return np.asarray(_is_nodata(a, nodata).all(axis=0))


def _same_grid(a: ZonePlan, b: ZonePlan) -> bool:
    ga, gb = a.grid, b.grid
    return (ga.epsg, ga.left, ga.top, ga.pixel) == (gb.epsg, gb.left, gb.top, gb.pixel)


def _neighbour(a: np.ndarray, dr: int, dc: int, fill: float | int) -> np.ndarray:
    """Array whose ``[..., r, c]`` is ``a[..., r + dr, c + dc]``, and ``fill`` past the edge."""
    h, w = a.shape[-2:]
    out = np.full_like(a, fill)
    out[..., max(0, -dr) : h - max(0, dr), max(0, -dc) : w - max(0, dc)] = a[
        ..., max(0, dr) : h - max(0, -dr), max(0, dc) : w - max(0, -dc)
    ]
    return out


_RING = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _grow_one_pixel(src: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Copy of (B, H, W) ``src`` whose pixels next to ``mask`` (8-connected) take a mask neighbour's value."""
    out = src.copy()
    done = mask.copy()
    for dr, dc in _RING:
        take = ~done & _neighbour(mask, dr, dc, False)
        out[:, take] = _neighbour(src, dr, dc, 0)[:, take]
        done |= take
    return out


def _in_zone_band(lon: np.ndarray, lat: np.ndarray, zone: str) -> np.ndarray:
    """Vectorised ``io.zone_for(lon, lat) == zone``; False where a coordinate is not finite."""
    finite = np.isfinite(lon) & np.isfinite(lat)
    number = np.clip(np.floor((np.where(finite, lon, 0.0) + 180.0) / 6.0).astype(np.int64) + 1, 1, 60)
    north = lat >= 0
    return finite & (number == int(zone[:2])) & (north if zone.endswith("N") else ~north)


def _seam_holes(owner: np.ndarray, rows: slice, cols: slice, zid: int) -> np.ndarray:
    """Unowned pixels of ``owner[rows, cols]`` with a 4-neighbour owned by a zone other than ``zid``."""
    h, w = owner.shape
    pad = np.full((rows.stop - rows.start + 2, cols.stop - cols.start + 2), -1, dtype=owner.dtype)
    a0, a1 = max(rows.start - 1, 0), min(rows.stop + 1, h)
    b0, b1 = max(cols.start - 1, 0), min(cols.stop + 1, w)
    pad[a0 - rows.start + 1 : a1 - rows.start + 1, b0 - cols.start + 1 : b1 - cols.start + 1] = owner[a0:a1, b0:b1]
    other = (pad >= 0) & (pad != zid)
    near = other[:-2, 1:-1] | other[2:, 1:-1] | other[1:-1, :-2] | other[1:-1, 2:]
    return (pad[1:-1, 1:-1] < 0) & near


def _join_zone_pieces(
    parts: Sequence[tuple[np.ndarray, ZonePlan]], dtype: np.dtype[Any], nodata: float | int
) -> list[tuple[np.ndarray, ZonePlan]]:
    """Paste the pieces of each zone onto one canvas, one ``(data, plan)`` per zone in first-seen order.

    Each canvas covers the union of its pieces' windows; it holds a piece's values on that piece's
    mask pixels (earlier pieces win where masks overlap) and ``nodata`` everywhere else, and its
    plan's mask is the union of the masks. A zone with a single piece keeps that piece's window.

    Raises:
        ValueError: Two pieces of one zone are on different grids.
    """
    by_zone: dict[str, list[tuple[np.ndarray, ZonePlan]]] = {}
    for data, plan in parts:
        by_zone.setdefault(plan.zone, []).append((data, plan))
    joined: list[tuple[np.ndarray, ZonePlan]] = []
    for zone, group in by_zone.items():
        first = group[0][1]
        if any(not _same_grid(plan, first) for _, plan in group[1:]):
            raise ValueError(f"Two pieces in {zone} are on different grids; they cannot be pasted together.")
        rows = slice(min(p.rows.start for _, p in group), max(p.rows.stop for _, p in group))
        cols = slice(min(p.cols.start for _, p in group), max(p.cols.stop for _, p in group))
        canvas = np.full((group[0][0].shape[0], rows.stop - rows.start, cols.stop - cols.start), nodata, dtype=dtype)
        mask = np.zeros(canvas.shape[1:], dtype=bool)
        for data, plan in group:
            r = slice(plan.rows.start - rows.start, plan.rows.stop - rows.start)
            c = slice(plan.cols.start - cols.start, plan.cols.stop - cols.start)
            take = plan.mask & ~mask[r, c]
            canvas[:, r, c][:, take] = data[:, take]
            mask[r, c] |= take
        joined.append((canvas, ZonePlan(zone=zone, grid=first.grid, rows=rows, cols=cols, mask=mask)))
    return joined


def mosaic_zones(
    pieces: Sequence[GridArray],
    *,
    dst_zone: str | None = None,
    nodata: float | int,
    resampling: Resampling = Resampling.nearest,
) -> tuple[np.ndarray, Affine, CRS]:
    """Join per-zone results of a multi-zone AOI into one raster on one zone's grid.

    Each piece contributes only its ``plan.mask`` pixels, so where zone windows overlap a ground
    pixel is taken from its nominal zone and never written twice. The destination grid is the
    ``dst_zone`` grid itself (pixel edges aligned to that zone's transform), extended to cover the
    union of every piece's window, reprojected. Pieces in ``dst_zone`` are pasted exactly and own
    their mask pixels (even where their value is nodata); pieces in other zones are reprojected
    with ``rasterio.warp.reproject`` (GDAL's exact transformer, as the reference reads in
    :mod:`tessera_apps.validate` use, so nearest takes the source pixel under each destination
    pixel centre) and fill only destination pixels that nothing owns yet and whose centre lies in
    that piece's zone band (``io.zone_for``). Where pieces collide, pieces in ``dst_zone`` win,
    then earlier pieces.

    Seams: a destination pixel whose centre lies just inside another zone's band can map onto a
    source pixel whose own centre lies in the destination band, so neither mask covers it. Such a
    pixel (unowned, next to a pixel of a different zone, centre in the band of the piece's zone)
    takes the value of the nearest mask pixel of its nominal zone, one pixel away at most. At
    Caldor (10N and 11N, 28 million pixels) that is about 1,000 pixels along the 120 W seam.

    Blocks: several pieces may share a zone (for example the blocks
    :func:`~tessera_apps.io.iter_plan_blocks` reads) if they share its grid. Pieces of a zone other
    than ``dst_zone`` are first pasted onto one canvas covering the union of their windows (earlier
    pieces win where they overlap, the mask is the union of their masks) and reprojected as one
    array, so the seam rule above sees across block edges and a split near the seam leaves no hole.

    Args:
        pieces: One or more results; pieces of one zone must be on the same grid.
        dst_zone: Zone whose grid to use. Defaults to the zone with the most AOI pixels (ties go to
            the zone listed first). Must be the zone of at least one piece.
        nodata: Fill value for pixels no piece covers. Must fit the output dtype; NaN needs float
            data. bool pieces are converted to uint8 first, so use for example 255, and preview
            such a mosaic with ``write_preview_png(path, data, nodata=255, vmin=0, vmax=1)`` so the
            nodata pixels are drawn grey instead of stretching the colour scale.
        resampling: Resampling for pieces in other zones. Nearest keeps classes and masks exact;
            bilinear or cubic suit continuous layers (distances, similarities).

    Returns:
        ``(data, transform, crs)``: data is (H, W) when every piece is 2-D, else (B, H, W), in the
        common dtype of the pieces; crs is the destination zone's pyproj CRS. A single piece comes
        back on its own window with its non-mask pixels set to nodata.

    Raises:
        ValueError: No pieces, mixed band counts, nodata that does not fit, a ``dst_zone`` no piece
            is in, or two pieces of one zone on different grids.
    """
    if not pieces:
        raise ValueError("mosaic_zones needs at least one piece.")
    arrays = [_as_bands(p.data) for p in pieces]
    if len({a.shape[0] for a in arrays}) != 1:
        raise ValueError(f"Pieces have different band counts: {[a.shape[0] for a in arrays]}.")
    dtype = np.result_type(*arrays)
    if _is_nan(nodata) and dtype.kind != "f":
        raise ValueError(f"nodata NaN needs float data; the pieces are {dtype}.")
    _check_nodata(dtype, nodata)
    squeeze = all(np.ndim(p.data) == 2 for p in pieces)

    counts: dict[str, int] = {}
    for p in pieces:
        counts[p.plan.zone] = counts.get(p.plan.zone, 0) + p.plan.n_pixels
    if dst_zone is None:
        dst = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        dst = canonicalize_zone(dst_zone)
        if dst not in counts:
            raise ValueError(f"dst_zone {dst} is not the zone of any piece ({', '.join(counts)}).")
    anchor = next(p.plan for p in pieces if p.plan.zone == dst)
    for p in pieces:
        if p.plan.zone == dst and not _same_grid(p.plan, anchor):
            raise ValueError(f"Two pieces in {dst} are on different grids; they cannot be pasted together.")
    grid = anchor.grid
    px = grid.pixel
    dst_crs = grid.crs

    if len(pieces) == 1:
        out = arrays[0].astype(dtype, copy=True)
        out[:, ~anchor.mask] = nodata
        return (out[0] if squeeze else out), anchor.transform, dst_crs

    home = [(arrays[i], p.plan) for i, p in enumerate(pieces) if p.plan.zone == dst]
    away = _join_zone_pieces([(arrays[i], p.plan) for i, p in enumerate(pieces) if p.plan.zone != dst], dtype, nodata)
    parts = home + away

    # Window of each part on the destination grid, in grid rows and columns (may be negative or
    # past the zone group's edge: the output is a file, not a store read).
    spans: list[tuple[int, int, int, int]] = []
    for _, plan in parts:
        if plan.zone == dst:
            spans.append((plan.rows.start, plan.rows.stop, plan.cols.start, plan.cols.stop))
            continue
        h, w = plan.shape
        left, bottom, right, top = array_bounds(h, w, plan.transform)
        xmin, ymin, xmax, ymax = transform_bounds(
            _rio_crs(plan.grid.crs), _rio_crs(dst_crs), left, bottom, right, top, densify_pts=21
        )
        spans.append(
            (
                math.floor((grid.top - ymax) / px),
                math.ceil((grid.top - ymin) / px),
                math.floor((xmin - grid.left) / px),
                math.ceil((xmax - grid.left) / px),
            )
        )
    r0 = min(s[0] for s in spans)
    c0 = min(s[2] for s in spans)
    shape = (max(s[1] for s in spans) - r0, max(s[3] for s in spans) - c0)
    out = np.full((arrays[0].shape[0], *shape), nodata, dtype=dtype)
    zone_ids = {z: k for k, z in enumerate(counts)}
    owner = np.full(shape, -1, dtype=np.int16)

    def window(i: int) -> tuple[slice, slice]:
        pr0, pr1, pc0, pc1 = spans[i]
        return slice(pr0 - r0, pr1 - r0), slice(pc0 - c0, pc1 - c0)

    def warp(i: int, src: np.ndarray) -> np.ndarray:
        rows, cols = window(i)
        tmp = np.full((src.shape[0], rows.stop - rows.start, cols.stop - cols.start), nodata, dtype=dtype)
        reproject(
            source=src,
            destination=tmp,
            src_transform=parts[i][1].transform,
            src_crs=_rio_crs(parts[i][1].grid.crs),
            src_nodata=nodata,
            dst_transform=grid.window_transform(spans[i][0], spans[i][2]),
            dst_crs=_rio_crs(dst_crs),
            dst_nodata=nodata,
            resampling=resampling,
            # Exact transformer, as in validate's reference reads: GDAL's default approximation
            # (0.125 pixel) picks a neighbouring source pixel for about 2 percent of away-zone
            # pixels at Caldor scale, so nearest would no longer mean the pixel holding the centre.
            tolerance=0.0,
        )
        return tmp

    for i in range(len(home)):
        rows, cols = window(i)
        src, plan = parts[i]
        take = plan.mask & (owner[rows, cols] < 0)
        out[:, rows, cols][:, take] = src[:, take]
        owner[rows, cols][take] = zone_ids[dst]
    to_lonlat = grid.to_lonlat()

    def in_band(i: int, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
        """Which window pixels ``(rr, cc)`` of part ``i`` have their centre in that part's zone band."""
        rows, cols = window(i)
        x = grid.left + (c0 + cols.start + cc + 0.5) * px
        y = grid.top - (r0 + rows.start + rr + 0.5) * px
        lon, lat = to_lonlat.transform(x, y)
        return _in_zone_band(np.asarray(lon), np.asarray(lat), parts[i][1].zone)

    away_ids = range(len(home), len(parts))
    for i in away_ids:
        rows, cols = window(i)
        tmp = warp(i, parts[i][0])
        take = (owner[rows, cols] < 0) & ~_empty_pixels(tmp, nodata)
        rr, cc = np.nonzero(take)
        outside = ~in_band(i, rr, cc)
        take[rr[outside], cc[outside]] = False
        out[:, rows, cols][:, take] = tmp[:, take]
        owner[rows, cols][take] = zone_ids[parts[i][1].zone]
    for i in away_ids:
        src, plan = parts[i]
        zid = zone_ids[plan.zone]
        rows, cols = window(i)
        holes = _seam_holes(owner, rows, cols, zid)
        if not holes.any():
            continue
        rr, cc = np.nonzero(holes)
        keep = in_band(i, rr, cc)
        rr, cc = rr[keep], cc[keep]
        if rr.size == 0:
            continue
        grown = warp(i, _grow_one_pixel(src, plan.mask))
        hit = ~_is_nodata(grown[:, rr, cc], nodata).all(axis=0)
        rr, cc = rr[hit], cc[hit]
        out[:, rows.start + rr, cols.start + cc] = grown[:, rr, cc]
        owner[rows.start + rr, cols.start + cc] = zid
    return (out[0] if squeeze else out), grid.window_transform(r0, c0), dst_crs


# --------------------------------------------------------------------------------------------
# Tables and JSON
# --------------------------------------------------------------------------------------------


def write_table(df: pd.DataFrame, out_stem: Path) -> tuple[Path, Path]:
    """Write a table as ``<out_stem>.parquet`` and ``<out_stem>.csv``, each atomically.

    A named or non-default index is written as ordinary leading column(s) in both files, so the two
    files hold the same columns; a plain unnamed RangeIndex is dropped.

    Args:
        df: The table.
        out_stem: Output path without suffix, for example ``out/wildfire_summary``.

    Returns:
        ``(parquet_path, csv_path)``.
    """
    stem = Path(out_stem)
    parquet = stem.parent / f"{stem.name}.parquet"
    csv = stem.parent / f"{stem.name}.csv"
    if not (isinstance(df.index, pd.RangeIndex) and df.index.name is None):
        df = df.reset_index()
    with atomic_path(parquet) as tmp:
        df.to_parquet(tmp, engine="pyarrow", index=False)
    with atomic_path(csv) as tmp:
        df.to_csv(tmp, index=False, lineterminator="\n", encoding="utf-8")
    return parquet, csv


def _json_key(key: object) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, bool | np.bool_):
        return "true" if key else "false"
    if isinstance(key, int | np.integer):
        return str(int(key))
    if isinstance(key, float | np.floating):
        return repr(float(key))
    if isinstance(key, PurePath):
        return str(key)
    if isinstance(key, enum.Enum):
        return _json_key(key.value)
    raise TypeError(f"Cannot use a {type(key).__name__} as a JSON object key.")


def to_jsonable(obj: object) -> JSONValue:
    """Convert ``obj`` into plain JSON types.

    Handles numpy scalars and arrays, ``Path``, ``datetime``/``date``/``time`` (ISO 8601),
    ``numpy.datetime64``, pandas ``Timestamp``, ``NaT`` and ``NA`` (null), enums (their value),
    dataclass instances (their fields), tuples and sets (lists; sets sorted when possible), and
    mappings (keys become strings). NaN and infinities become null, since JSON has neither.

    Raises:
        TypeError: A value or key of a type with no JSON form.
        ValueError: Two keys of one mapping map to the same string, for example ``1`` and ``"1"``.
    """
    if obj is None or isinstance(obj, str | bool):
        return obj
    if obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.datetime64):
        return None if np.isnat(obj) else str(np.datetime_as_string(obj))
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "M":
            text = np.datetime_as_string(obj).astype(object)
            text[np.isnat(obj)] = None
            return to_jsonable(text.tolist())
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, dt.datetime | dt.date | dt.time):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})
    if isinstance(obj, Mapping):
        out: dict[str, JSONValue] = {}
        for k, v in obj.items():
            key = _json_key(k)
            if key in out:
                raise ValueError(f"Two keys map to the JSON key {key!r}.")
            out[key] = to_jsonable(v)
        return out
    if isinstance(obj, set | frozenset):
        items = list(obj)
        with contextlib.suppress(TypeError):
            items = sorted(items)
        return [to_jsonable(v) for v in items]
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    raise TypeError(f"Object of type {type(obj).__name__} has no JSON form.")


def write_json(path: Path, obj: object) -> Path:
    """Write ``obj`` as UTF-8 JSON with sorted keys and indent 1, atomically.

    Values are converted by :func:`to_jsonable` first, so numpy types, paths, datetimes and
    dataclasses are accepted and NaN is written as null.

    Returns:
        ``path``.
    """
    final = Path(path)
    text = json.dumps(to_jsonable(obj), sort_keys=True, indent=1, allow_nan=False, ensure_ascii=False) + "\n"
    with atomic_path(final) as tmp:
        tmp.write_text(text, encoding="utf-8")
    return final


# --------------------------------------------------------------------------------------------
# Preview PNG
# --------------------------------------------------------------------------------------------


def write_preview_png(
    path: Path,
    image: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    legend: str | None = None,
    nodata: float | None = None,
) -> Path:
    """Write a quick-look PNG, atomically, with matplotlib's Agg canvas (pyplot is never used).

    An (H, W, 3) or (H, W, 4) uint8 image is drawn as is. An (H, W) bool, integer or float image is
    mapped through ``cmap`` with a colorbar. Missing pixels are drawn in the neutral grey
    :data:`PREVIEW_NAN_RGB`: NaN, pixels equal to ``nodata``, and masked pixels of a numpy masked
    array. Missing pixels never set the default colour limits, so an integer mosaic such as the
    uint8 (0, 1, 255) array :func:`mosaic_zones` gives for bool pieces with ``nodata=255`` keeps its
    0 and 1 apart. Images whose longer side exceeds :data:`PREVIEW_MAX_SIDE` are strided down to
    fit; small images are scaled up with nearest-neighbour so the figure stays legible.

    Args:
        path: Output PNG.
        image: The image, optionally a numpy masked array.
        title: Title above the image.
        cmap: Matplotlib colormap name for scalar images.
        vmin: Lower colour limit for scalar images (default: minimum of the drawn pixels, 0 for bool).
        vmax: Upper colour limit for scalar images (default: maximum of the drawn pixels, 1 for bool).
        legend: Colorbar label for scalar images, or a caption under an RGB image.
        nodata: Value of missing pixels: a scalar image's pixels equal to it, or a colour image's
            pixels whose R, G and B all equal it, are drawn grey.

    Returns:
        ``path``.

    Raises:
        ValueError: The image is empty, or is neither (H, W) nor (H, W, 3|4) uint8.
        TypeError: ``nodata`` is not a number, or is a bool.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    hidden = np.ma.getmaskarray(image) if isinstance(image, np.ma.MaskedArray) else None
    img = np.asarray(image.data if isinstance(image, np.ma.MaskedArray) else image)
    if img.ndim == 3:
        if img.shape[2] not in (3, 4) or img.dtype != np.uint8:
            raise ValueError(f"A colour image must be (H, W, 3) or (H, W, 4) uint8; got {img.shape} {img.dtype}.")
    elif img.ndim != 2:
        raise ValueError(f"image must be (H, W) or (H, W, 3); got shape {img.shape}.")
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError(f"image is empty: shape {img.shape}.")
    if nodata is not None and (
        isinstance(nodata, bool | np.bool_) or not isinstance(nodata, int | float | np.integer | np.floating)
    ):
        raise TypeError(f"nodata must be a number, got {type(nodata).__name__}.")
    step = max(1, math.ceil(max(img.shape[0], img.shape[1]) / PREVIEW_MAX_SIDE))
    img = img[::step, ::step]
    scalar = img.ndim == 2
    missing = np.zeros(img.shape[:2], dtype=bool)
    if hidden is not None:
        hidden = hidden[::step, ::step]
        missing |= hidden.any(axis=2) if hidden.ndim == 3 else hidden
    if nodata is not None and not _is_nan(nodata):
        missing |= (img == nodata) if scalar else (img[..., :3] == nodata).all(axis=2)
    h, w = img.shape[0], img.shape[1]
    scale = max(1.0, PREVIEW_MIN_SIDE / max(h, w))
    extra_w = 1.6 if scalar else 0.4
    extra_h = 0.4 + (0.4 if title else 0.0) + (0.3 if legend and not scalar else 0.0)
    fig = Figure(
        figsize=(w * scale / _PREVIEW_DPI + extra_w, h * scale / _PREVIEW_DPI + extra_h),
        dpi=_PREVIEW_DPI,
        layout="constrained",
    )
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.set_axis_off()
    if scalar:
        is_bool = img.dtype == np.bool_
        as_float = img.astype(np.float64)
        values = np.ma.masked_array(as_float, mask=missing | ~np.isfinite(as_float))
        if is_bool:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        if values.count() == 0:
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax
        colours = mpl.colormaps[cmap].with_extremes(bad=PREVIEW_NAN_RGB)
        shown = ax.imshow(values, cmap=colours, vmin=vmin, vmax=vmax, interpolation="nearest")
        bar = fig.colorbar(shown, ax=ax, label=legend, shrink=0.9)
        if is_bool:
            bar.set_ticks([0, 1])
    else:
        if missing.any():
            img = img.copy()
            img[missing, :3] = np.round(np.array(PREVIEW_NAN_RGB) * 255).astype(np.uint8)
            if img.shape[2] == 4:
                img[missing, 3] = 255
        ax.imshow(img, interpolation="nearest")
        if legend:
            fig.supxlabel(legend, fontsize="small")
    if title:
        ax.set_title(title)
    final = Path(path)
    with atomic_path(final) as tmp, tmp.open("wb") as fh:
        fig.savefig(fh, format="png", dpi=_PREVIEW_DPI)
    return final


# --------------------------------------------------------------------------------------------
# Digests, output names and provenance
# --------------------------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def args_digest(params: Mapping[str, Any]) -> str:
    """Hex sha256 of ``params`` as canonical JSON.

    Canonical means :func:`to_jsonable` conversion (numpy scalars equal Python ones, NaN is null),
    sorted keys and no whitespace, so the digest does not depend on key order.
    """
    text = json.dumps(to_jsonable(dict(params)), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_args_digest(
    params: Mapping[str, Any], *, years: Sequence[int], aoi_sha256: str | None, seed: int | None
) -> str:
    """The digest an app records in ``report.json`` and compares in :func:`outputs_current`.

    It covers the params (thresholds and options), the years used (sorted, unique), the AOI file's
    sha256 and the random seed, so changing any of them forces a rerun. :func:`provenance_record`
    records the same value, so an app can compute it before reading anything.
    """
    return args_digest(
        {
            "params": dict(params),
            "years": sorted({operator.index(y) for y in years}),
            "aoi_sha256": aoi_sha256,
            "seed": seed,
        }
    )


@dataclasses.dataclass(frozen=True)
class OutputPaths:
    """The files one app run writes (brief section 4)."""

    tif: Path
    summary_parquet: Path
    summary_csv: Path
    report: Path
    preview: Path
    validation: Path

    def required(self, *, need_validation: bool) -> tuple[Path, ...]:
        """Outputs a complete run leaves, with the report last."""
        paths: tuple[Path, ...] = (self.tif, self.summary_parquet, self.summary_csv, self.preview)
        if need_validation:
            paths = (*paths, self.validation)
        return (*paths, self.report)


def output_paths(out_dir: Path, app: str, year_a: int, year_b: int) -> OutputPaths:
    """Output names for an app run: ``<app>_<a>_<b>.tif``, ``<app>_summary.parquet`` and so on.

    Raises:
        ValueError: ``app`` is not a plain name (letters, digits, ``_`` and ``-``).
        TypeError: A year is not an integer.
    """
    if not _APP_NAME.match(app):
        raise ValueError(f"app name {app!r} must be letters, digits, '_' or '-'.")
    a, b = operator.index(year_a), operator.index(year_b)
    d = Path(out_dir)
    return OutputPaths(
        tif=d / f"{app}_{a}_{b}.tif",
        summary_parquet=d / f"{app}_summary.parquet",
        summary_csv=d / f"{app}_summary.csv",
        report=d / f"{app}_report.json",
        preview=d / f"{app}_preview.png",
        validation=d / f"{app}_validation.json",
    )


def outputs_record(paths: OutputPaths, *, validation: bool) -> dict[str, str]:
    """The ``"outputs"`` entry of a report: the sha256 of every output this run wrote.

    Call it just before writing ``report.json`` and store the result under the report's top-level
    ``"outputs"`` key. :func:`outputs_current` checks each listed hash, and only a report that lists
    ``"validation"`` vouches for ``validation.json``: pass ``validation=True`` only when this run
    wrote that file (a ``--validate`` run whose reference could not be fetched passes False).

    Args:
        paths: From :func:`output_paths`.
        validation: Whether this run wrote ``validation.json``.

    Returns:
        ``{field: sha256}`` keyed by :class:`OutputPaths` field name (``"tif"``, ``"summary_parquet"``,
        ``"summary_csv"``, ``"preview"`` and, when ``validation`` is True, ``"validation"``).

    Raises:
        FileNotFoundError: One of those outputs does not exist.
    """
    fields = ["tif", "summary_parquet", "summary_csv", "preview"] + (["validation"] if validation else [])
    return {name: file_sha256(getattr(paths, name)) for name in fields}


def begin_run(paths: OutputPaths) -> tuple[Path, ...]:
    """Remove the previous run's ``report.json`` and ``validation.json`` before a run writes anything.

    Call it once a run has decided not to skip (or was forced). Without the old report, a run that
    crashes part way leaves outputs that :func:`outputs_current` never takes as current. Without the
    old ``validation.json``, a run that does not validate, or whose reference fetch fails, does not
    leave a validation file scored for other arguments next to its outputs.

    Args:
        paths: From :func:`output_paths`.

    Returns:
        The files that were removed.
    """
    removed = []
    for path in (paths.report, paths.validation):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        removed.append(path)
    return tuple(removed)


def outputs_current(paths: OutputPaths, digest: str, *, need_validation: bool) -> bool:
    """Whether a previous run already wrote these outputs with the same arguments.

    True only when all of these hold:

    - every required output exists and ``report.json`` parses as a JSON object;
    - the report records ``"args_digest": digest`` (at the top level or in its ``"provenance"`` block);
    - no other output is newer than the report (an equal mtime is fine: coarse clocks tie). Apps
      write the report last, so this catches a later run with other arguments that crashed after
      replacing some outputs but before its report;
    - every required output listed in the report's ``"outputs"`` (from :func:`outputs_record`) still
      has the sha256 recorded there;
    - with ``need_validation``, the report lists ``"validation"``. ``validation.json`` is the one
      output a run may leave unwritten, so only the report's own hash ties it to ``digest``: a file
      left by an earlier run with other arguments, or by a run whose reference fetch failed, does
      not count.

    Args:
        paths: From :func:`output_paths`.
        digest: From :func:`run_args_digest` for this run.
        need_validation: Whether this run must also have ``validation.json``.
    """
    required = paths.required(need_validation=need_validation)
    if not all(p.is_file() for p in required):
        return False
    try:
        report = json.loads(paths.report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict):
        return False
    recorded = report.get("args_digest")
    if recorded is None and isinstance(report.get("provenance"), dict):
        recorded = report["provenance"].get("args_digest")
    if recorded != digest:
        return False
    report_time = paths.report.stat().st_mtime_ns
    if not all(p.stat().st_mtime_ns <= report_time for p in required if p != paths.report):
        return False
    hashes = report.get("outputs")
    if hashes is None:
        hashes = {}
    if not isinstance(hashes, dict):
        return False
    if need_validation and "validation" not in hashes:
        return False
    for field in dataclasses.fields(paths):
        path = getattr(paths, field.name)
        if field.name in hashes and path in required and path != paths.report:
            try:
                if hashes[field.name] != file_sha256(path):
                    return False
            except OSError:
                return False
    return True


def library_versions() -> dict[str, str | None]:
    """Installed versions of the libraries that shape results, None when not installed.

    Versions come from package metadata, so nothing is imported (geotessera stays unimported).
    """
    out: dict[str, str | None] = {}
    for name in PROVENANCE_LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    out["gdal"] = rasterio.__gdal_version__
    return out


def provenance_record(
    reader: Reader,
    *,
    aoi_path: Path | None,
    years: Sequence[int],
    params: Mapping[str, Any],
    seed: int | None,
    zones: Sequence[str],
) -> dict[str, Any]:
    """The reproducibility block for a report (brief section 6 as corrected by survey S4).

    Records the package, Python and library versions; the store identity from
    ``reader.provenance()`` (backend, store URL, snapshot id: the data version) apart from the model
    keys, which go under ``"model"`` (``checkpoint_id`` names the model weights); the pipeline commit
    of every zone-year when the reader has ``run_commits``; the years, zones, params and seed; the
    AOI file's name and sha256; :func:`run_args_digest`; and a UTC timestamp. Nothing is read from
    the environment, and reader provenance keys that look like credentials are dropped.

    Args:
        reader: The reader the app used.
        aoi_path: The AOI file, or None for an AOI built in code.
        years: Every year the app read.
        params: Thresholds and options that change the outputs.
        seed: Random seed, or None.
        zones: Zone groups the AOI touches.

    Returns:
        A dict of plain JSON values.
    """
    years_used = sorted({operator.index(y) for y in years})
    zone_names = sorted({canonicalize_zone(z) for z in zones})
    aoi = None if aoi_path is None else Path(aoi_path)
    aoi_sha = None if aoi is None else file_sha256(aoi)
    store = {str(k): v for k, v in reader.provenance().items() if not _SECRET_KEY.search(str(k))}
    model = {k: store.pop(k) for k in _MODEL_KEYS if k in store}
    run_commits = getattr(reader, "run_commits", None)
    commits = None
    if callable(run_commits):
        commits = {z: dict(run_commits(z, years_used)) for z in zone_names}
    record = {
        "tessera_apps_version": __version__,
        "python_version": platform.python_version(),
        "libraries": library_versions(),
        "store": store,
        "model": model,
        "pipeline_commits": commits,
        "years": years_used,
        "zones": zone_names,
        "params": dict(params),
        "seed": seed,
        "aoi": {"name": None if aoi is None else aoi.name, "sha256": aoi_sha},
        "args_digest": run_args_digest(params, years=years_used, aoi_sha256=aoi_sha, seed=seed),
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    return cast(dict[str, Any], to_jsonable(record))
