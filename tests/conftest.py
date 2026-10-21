"""Mock stores with the published store's layout, built in memory from fixtures or synthetic data."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from pyproj import Transformer
from typer import Typer
from typer.testing import CliRunner, Result
from zarr.storage import MemoryStore, WrapperStore

from tessera_apps.io import FIRST_YEAR, N_BANDS, YEARS, TesseraReader, zone_epsg

FIXTURES = Path(__file__).parent / "fixtures"


class KeyLog(WrapperStore[Any]):
    """Records every key read, so tests can assert that no chunk was fetched."""

    def __init__(self, store: Any, log: list[str] | None = None) -> None:
        super().__init__(store)
        self.log = log if log is not None else []

    def _with_store(self, store: Any) -> KeyLog:
        return type(self)(store, self.log)

    async def get(self, key, prototype, byte_range=None):
        self.log.append(key)
        return await self._store.get(key, prototype, byte_range)

    async def get_partial_values(self, prototype, key_ranges):
        key_ranges = list(key_ranges)
        self.log.extend(k for k, _ in key_ranges)
        return await self._store.get_partial_values(prototype, key_ranges)

    async def get_ranges(self, key, byte_ranges, **kwargs):
        self.log.append(key)
        async for group in self._store.get_ranges(key, byte_ranges, **kwargs):
            yield group

    def get_sync(self, key, **kwargs):
        self.log.append(key)
        return self._store.get_sync(key, **kwargs)

    def get_ranges_sync(self, key, byte_ranges, **kwargs):
        self.log.append(key)
        yield from self._store.get_ranges_sync(key, byte_ranges, **kwargs)

    def chunk_keys(self) -> list[str]:
        return [k for k in self.log if "/c/" in k or k.split("/")[-1].startswith("c")]


def _time_values(n: int) -> np.ndarray:
    stamps = np.array([f"{FIRST_YEAR + i}-01-01" for i in range(n)], dtype="datetime64[ns]")
    return stamps.astype(np.int64)


def add_zone(
    root: zarr.Group,
    zone: str,
    *,
    left: float,
    top: float,
    height: int,
    width: int,
    fill: Callable[[str, int, tuple[int, ...]], np.ndarray],
    years_complete: list[int] | None = None,
    chunk: int = 32,
    shard: int = 64,
) -> zarr.Group:
    """Create a zone group shaped like the live store, filling each layer with ``fill(name, t, shape)``."""
    g = root.create_group(zone)
    g.attrs.update(
        {
            "crs": f"EPSG:{zone_epsg(zone)}",
            "proj:code": f"EPSG:{zone_epsg(zone)}",
            "years_complete": list(YEARS) if years_complete is None else years_complete,
            "spatial:transform": [10.0, 0.0, left, 0.0, -10.0, top],
            "spatial:shape": [height, width],
            "runs": {str(y): {"code": {"commit": f"abc{y}"}} for y in YEARS},
        }
    )
    n = len(YEARS)
    t = g.create_array("time", shape=(n,), dtype="int64", fill_value=0)
    t[:] = _time_values(n)
    t.attrs["units"] = "nanoseconds since 1970-01-01"
    specs = {
        "embeddings": ((n, height, width, N_BANDS), "int8", 0, (1, chunk, chunk, N_BANDS), (1, shard, shard, N_BANDS)),
        "scales": ((n, height, width), "float32", np.nan, (1, chunk, chunk), (1, shard, shard)),
        "s2_obs_count": ((n, height, width), "uint16", 0, (1, chunk, chunk), (1, shard, shard)),
        "s1_asc_obs_count": ((n, height, width), "uint16", 0, (1, chunk, chunk), (1, shard, shard)),
        "s1_desc_obs_count": ((n, height, width), "uint16", 0, (1, chunk, chunk), (1, shard, shard)),
    }
    for name, (shape, dtype, fv, chunks, shards) in specs.items():
        arr = g.create_array(name, shape=shape, dtype=dtype, fill_value=fv, chunks=chunks, shards=shards)
        for ti in range(n):
            arr[ti] = fill(name, ti, shape[1:])
    return g


@pytest.fixture(scope="session")
def caldor_fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "caldor_64.npz") as d:
        return {k: d[k] for k in d.files}


PAD_ROWS, PAD_COLS = 16, 32


@pytest.fixture
def caldor_store(caldor_fixture) -> tuple[MemoryStore, KeyLog]:
    """10N mock group: the real 64 x 64 Caldor window, padded with no-data, at its true position."""
    fx = caldor_fixture
    a, _, c, _, _, f = fx["transform"]
    h = w = fx["scales"].shape[1]
    height, width = h + 2 * PAD_ROWS, w + 2 * PAD_COLS
    years = [int(y) for y in fx["years"]]

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        if name == "scales":
            out = np.full(shape, np.nan, dtype=np.float32)
        else:
            out = np.zeros(shape, dtype=fx[name].dtype)
        year = FIRST_YEAR + ti
        if year in years:
            out[PAD_ROWS : PAD_ROWS + h, PAD_COLS : PAD_COLS + w] = fx[name][years.index(year)]
        return out

    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    root.attrs.update({"checkpoint_id": "tessera_v1_1_aws_encoder.pt", "geoemb:dimensions": 128})
    add_zone(
        root,
        str(fx["zone"]),
        left=c - PAD_COLS * a,
        top=f + PAD_ROWS * a,
        height=height,
        width=width,
        fill=fill,
        years_complete=years,
    )
    return mem, KeyLog(mem)


@pytest.fixture
def caldor_reader(caldor_store) -> TesseraReader:
    _, logged = caldor_store
    return TesseraReader.from_store(logged)


BOUNDARY = (-120.0, 38.85)


def _synthetic(zone_marker: int, seed: int) -> Callable[[str, int, tuple[int, ...]], np.ndarray]:
    rng = np.random.default_rng(seed)

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        if name == "embeddings":
            e = rng.integers(-60, 60, size=shape, dtype=np.int8)
            e[..., 0] = zone_marker
            return e
        if name == "scales":
            s = np.full(shape, 0.01, dtype=np.float32)
            s[:4, :4] = np.nan
            return s
        if name == "s1_asc_obs_count":
            return np.full(shape, 0 if ti in (5, 6, 7) else 30, dtype=np.uint16)
        if name == "s1_desc_obs_count":
            return np.zeros(shape, dtype=np.uint16)
        return np.full(shape, 50, dtype=np.uint16)

    return fill


@pytest.fixture(scope="session")
def _boundary_mem() -> MemoryStore:
    """Synthetic 10N and 11N groups around 120 W, 38.85 N. Band 0 is +100 in 10N and -100 in 11N.

    11N is missing 2023 from years_complete, and both zones are radar-free in 2022 to 2024.
    """
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    size = 192
    for zone, marker, years in (("10N", 100, None), ("11N", -100, [y for y in YEARS if y != 2023])):
        x, y = Transformer.from_crs(4326, zone_epsg(zone), always_xy=True).transform(*BOUNDARY)
        left = (x // 10) * 10 - size / 2 * 10
        top = (y // 10) * 10 + size / 2 * 10
        add_zone(
            root,
            zone,
            left=left,
            top=top,
            height=size,
            width=size,
            fill=_synthetic(marker, int(zone[:2])),
            years_complete=years,
        )
    return mem


@pytest.fixture
def boundary_store(_boundary_mem) -> tuple[MemoryStore, KeyLog]:
    """The session's boundary store behind a fresh key log."""
    return _boundary_mem, KeyLog(_boundary_mem)


@pytest.fixture
def boundary_reader(boundary_store) -> TesseraReader:
    _, logged = boundary_store
    return TesseraReader.from_store(logged)


# --------------------------------------------------------------------------------------------
# Hansen tile pieces under the caldor fixture (deforest tests)
# --------------------------------------------------------------------------------------------

HANSEN_LAYER_NAMES = ("treecover2000", "lossyear", "datamask")


@pytest.fixture(scope="session")
def caldor_hansen() -> dict[str, Any]:
    """The Hansen GFC tile pieces under the caldor fixture store (``scripts/make_hansen_fixture.py``).

    ``treecover2000``, ``lossyear`` and ``datamask`` on Hansen's own 0.00025 degree grid, with the
    piece's ``west``, ``north``, ``res``, ``tile`` and ``version``.
    """
    with np.load(FIXTURES / "caldor_64_hansen.npz") as d:
        return {k: d[k] for k in d.files}


def write_hansen_tiles(
    directory: Path,
    pieces: dict[str, Any],
    *,
    edit: Callable[[str, np.ndarray], np.ndarray] | None = None,
) -> str:
    """Write Hansen tile pieces as local tiles named like the real ones; returns the ``url_template``.

    ``edit(layer, array)`` may change a layer's values before it is written (for example to put water
    in the datamask).
    """
    import rasterio
    from rasterio.transform import from_origin

    directory.mkdir(parents=True, exist_ok=True)
    version, tile = str(pieces["version"]), str(pieces["tile"])
    transform = from_origin(float(pieces["west"]), float(pieces["north"]), float(pieces["res"]), float(pieces["res"]))
    for layer in HANSEN_LAYER_NAMES:
        data = np.array(pieces[layer], dtype=np.uint8)
        if edit is not None:
            data = np.asarray(edit(layer, data), dtype=np.uint8)
        profile = {
            "driver": "GTiff",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": transform,
        }
        with rasterio.open(directory / f"Hansen_{version}_{layer}_{tile}.tif", "w", **profile) as ds:
            ds.write(data, 1)
    return str(directory / "Hansen_{version}_{layer}_{tile}.tif")


def fixture_window_lonlat(fx: dict[str, np.ndarray], rows: tuple[int, int], cols: tuple[int, int]) -> Any:
    """The lon/lat polygon of a window of the caldor fixture grid, ``rows`` and ``cols`` as (start, stop)."""
    import shapely
    from shapely.geometry import box

    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    square = box(c + cols[0] * a, f - rows[1] * a, c + cols[1] * a, f - rows[0] * a)
    to_ll = Transformer.from_crs(int(fx["epsg"]), 4326, always_xy=True)
    return shapely.transform(shapely.segmentize(square, 10.0), to_ll.transform, interleaved=False)


# --------------------------------------------------------------------------------------------
# ESA WorldCover tile pieces over a zone grid (wetland tests)
# --------------------------------------------------------------------------------------------


def write_worldcover_tiles(
    directory: Path,
    epsg: int,
    left: float,
    top: float,
    codes_by_year: dict[int, np.ndarray],
    *,
    pixel: float = 10.0,
    margin: int = 20,
) -> str:
    """WorldCover tile pieces whose pixels take the code of the zone grid pixel under their centre.

    ``codes_by_year[year]`` is a uint8 (H, W) array on a zone grid with corner ``(left, top)`` in
    ``epsg``; each piece lies on the global 1/12000 degree WorldCover grid, covers that window plus
    ``margin`` WorldCover pixels (0, no data, outside it) and is named like the real tile of its
    centre, for the year's version. Returns the ``url_template`` for the loaders.
    """
    import math

    import rasterio
    from rasterio.transform import from_origin

    from tessera_apps import validate

    directory.mkdir(parents=True, exist_ok=True)
    n = validate.WORLDCOVER_PIXELS_PER_DEG
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    to_zone = Transformer.from_crs(4326, epsg, always_xy=True)
    for year, codes in codes_by_year.items():
        h, w = codes.shape
        xs = [left, left + w * pixel]
        ys = [top - h * pixel, top]
        lons, lats = to_ll.transform([xs[0], xs[0], xs[1], xs[1]], [ys[0], ys[1], ys[0], ys[1]])
        x0, x1 = math.floor(min(lons) * n) - margin, math.ceil(max(lons) * n) + margin
        y0, y1 = math.floor(min(lats) * n) - margin, math.ceil(max(lats) * n) + margin
        cols, rows = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y1, y0, -1) - 0.5)
        zx, zy = to_zone.transform(cols / n, rows / n)
        zr = np.floor((top - np.asarray(zy)) / pixel).astype(np.int64)
        zc = np.floor((np.asarray(zx) - left) / pixel).astype(np.int64)
        inside = (zr >= 0) & (zr < h) & (zc >= 0) & (zc < w)
        data = np.zeros(zr.shape, dtype=np.uint8)
        data[inside] = codes[zr[inside], zc[inside]]
        tile = validate.worldcover_tile((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
        profile = {
            "driver": "GTiff",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_origin(x0 / n, y1 / n, 1.0 / n, 1.0 / n),
            "nodata": 0,
        }
        name = f"ESA_WorldCover_10m_{year}_{validate.WORLDCOVER_VERSIONS[year]}_{tile}_Map.tif"
        with rasterio.open(directory / name, "w", **profile) as ds:
            ds.write(data, 1)
    return str(directory / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")


# --------------------------------------------------------------------------------------------
# Designed stores whose pixels follow a class map (restore tests)
# --------------------------------------------------------------------------------------------

#: Size of the table of noise vectors a designed zone draws from; prime, so the per-pixel index cycles slowly.
NOISE_TABLE = 509


def designed_zone(
    root: zarr.Group,
    zone: str,
    *,
    left: float,
    top: float,
    height: int,
    width: int,
    pattern: Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, ...]],
    prototypes: Callable[[int], np.ndarray],
    noise_sd: float = 6.0,
    chunk: int = 64,
    shard: int | None = None,
    years_complete: list[int] | None = None,
    seed: int = 0,
) -> zarr.Group:
    """A zone group whose pixel vectors follow a class map, built strip by strip so large grids stay cheap.

    ``pattern(rows, cols, year)`` returns ``(classes, data, radar)``: the int class of every pixel (an
    index into ``prototypes(year)``, a (K, 128) array of float vectors), where the pixel has data, and
    where it has Sentinel-1. A pixel's vector is its class prototype plus a noise vector of standard
    deviation ``noise_sd``, taken from a seeded table of :data:`NOISE_TABLE` vectors by a hash of the
    pixel and the year (so repeated vectors keep a large store small once compressed), rounded and
    clipped to int8. Scales vary per pixel, so means of raw int8 vectors differ from dequantized ones.
    One chunk per shard by default, so a KeyLog names exactly the chunks read.
    """
    rng = np.random.default_rng(seed)
    table = np.round(rng.normal(0.0, noise_sd, size=(NOISE_TABLE, N_BANDS))).astype(np.int16)

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        year = FIRST_YEAR + ti
        protos = np.round(np.asarray(prototypes(year), dtype=np.float64)).astype(np.int16)
        h, w = shape[0], shape[1]
        if name == "embeddings":
            out = np.zeros(shape, dtype=np.int8)
        elif name == "scales":
            out = np.full(shape, np.nan, dtype=np.float32)
        else:
            out = np.zeros(shape, dtype=np.uint16)
        step = 128
        for r0 in range(0, h, step):
            rows, cols = np.mgrid[r0 : min(r0 + step, h), 0:w]
            classes, data, radar = pattern(rows, cols, year)[:3]
            if name == "embeddings":
                idx = (rows * 7919 + cols * 104729 + year * 1_000_003) % NOISE_TABLE
                e = protos[classes] + table[idx]
                e = np.clip(e, -127, 127).astype(np.int8)
                e[~data] = 0
                out[r0 : r0 + rows.shape[0]] = e
            elif name == "scales":
                s = 0.01 * (1.0 + 0.05 * ((rows * 7 + cols * 3) % 11))
                out[r0 : r0 + rows.shape[0]] = np.where(data, s, np.nan)
            elif name == "s1_asc_obs_count":
                out[r0 : r0 + rows.shape[0]] = np.where(radar, 30, 0)
            elif name == "s2_obs_count":
                out[r0 : r0 + rows.shape[0]] = 50
        return out

    return add_zone(
        root,
        zone,
        left=left,
        top=top,
        height=height,
        width=width,
        fill=fill,
        years_complete=years_complete,
        chunk=chunk,
        shard=chunk if shard is None else shard,
    )


def designed_store(
    zones: dict[str, dict[str, Any]],
) -> tuple[TesseraReader, KeyLog, MemoryStore]:
    """An in-memory store of :func:`designed_zone` groups, ``{zone: designed_zone keyword arguments}``.

    Returns:
        ``(reader, log, mem)``: a reader behind a fresh KeyLog, the log and the store.
    """
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    root.attrs.update({"checkpoint_id": "tessera_v1_1_aws_encoder.pt", "geoemb:dimensions": N_BANDS})
    for zone, kwargs in zones.items():
        designed_zone(root, zone, **kwargs)
    log = KeyLog(mem)
    return TesseraReader.from_store(log), log, mem


def write_tch_tiles(
    directory: Path,
    epsg: int,
    left: float,
    top: float,
    heights_by_year: dict[int, np.ndarray],
    *,
    pixel: float = 10.0,
    margin: int = 8,
) -> str:
    """GLAD canopy height tile pieces whose pixels take the height of the zone grid pixel under their centre.

    ``heights_by_year[year]`` is a uint8 (H, W) array on a zone grid with corner ``(left, top)`` in
    ``epsg``; each piece lies on the global 0.00025 degree Hansen grid, covers that window plus
    ``margin`` pixels (0 outside it), and is saved as ``<directory>/<year>/<tile>.tif`` with the tile
    named like the real one of its centre. Returns the ``url_template`` for the loader.
    """
    import math

    import rasterio
    from rasterio.transform import from_origin

    from tessera_apps import validate

    res = validate.HANSEN_RES_DEG
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    to_zone = Transformer.from_crs(4326, epsg, always_xy=True)
    for year, heights in heights_by_year.items():
        h, w = heights.shape
        xs = [left, left + w * pixel]
        ys = [top - h * pixel, top]
        lons, lats = to_ll.transform([xs[0], xs[0], xs[1], xs[1]], [ys[0], ys[1], ys[0], ys[1]])
        x0, x1 = math.floor(min(lons) / res) - margin, math.ceil(max(lons) / res) + margin
        y0, y1 = math.floor(min(lats) / res) - margin, math.ceil(max(lats) / res) + margin
        cols, rows = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y1, y0, -1) - 0.5)
        zx, zy = to_zone.transform(cols * res, rows * res)
        zr = np.floor((top - np.asarray(zy)) / pixel).astype(np.int64)
        zc = np.floor((np.asarray(zx) - left) / pixel).astype(np.int64)
        inside = (zr >= 0) & (zr < h) & (zc >= 0) & (zc < w)
        data = np.zeros(zr.shape, dtype=np.uint8)
        data[inside] = heights[zr[inside], zc[inside]]
        tile = validate.hansen_tile((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
        profile = {
            "driver": "GTiff",
            "width": data.shape[1],
            "height": data.shape[0],
            "count": 1,
            "dtype": "uint8",
            "crs": "EPSG:4326",
            "transform": from_origin(x0 * res, y1 * res, res, res),
        }
        (directory / str(year)).mkdir(parents=True, exist_ok=True)
        with rasterio.open(directory / str(year) / f"{tile}.tif", "w", **profile) as ds:
            ds.write(data, 1)
    return str(directory / "{year}" / "{tile}.tif")


# --------------------------------------------------------------------------------------------
# Restore scenarios: a site, its ring, a target stand and a crop patch on a designed 10N grid
# --------------------------------------------------------------------------------------------

_RESTORE_RNG = np.random.default_rng(2026)
#: Land type prototypes: the site starts as pasture and moves towards forest from the planting year.
PASTURE = _RESTORE_RNG.integers(-40, 41, size=N_BANDS).astype(np.float64)
FOREST = _RESTORE_RNG.integers(-40, 41, size=N_BANDS).astype(np.float64)
CROP = _RESTORE_RNG.integers(-40, 41, size=N_BANDS).astype(np.float64)
#: Added to every vector in a radar-free year of a whole-grid switch, as the survey B5 regime shift does.
RADAR_SHIFT = _RESTORE_RNG.integers(-20, 21, size=N_BANDS).astype(np.float64)
RESTORE_CENTRE = (-121.0, 38.5)
RESTORE_EPSG = 32610
RESTORE_PLANTED = 2020
#: Years the site takes to become forest, from the year before planting.
RESTORE_RAMP = 5
#: Grid boxes ((row0, row1), (col0, col1)) of the 192 x 192 scenario: the site in the middle, a forest
#: target stand outside the ring, and crop filling the top-right 64 x 64 chunk of the ring.
SITE_BOX = ((86, 106), (86, 106))
TARGET_BOX = ((160, 186), (10, 36))
CROP_BOX = ((0, 64), (128, 192))
#: The ring the scenario tests use (the default 1 to 3 km needs the large grid).
SMALL_RING = {"ring_inner_km": 0.1, "ring_outer_km": 0.5}

CLASS_PASTURE, CLASS_FOREST, CLASS_CROP, CLASS_SITE = 0, 1, 2, 3


def restore_corner(centre: tuple[float, float] = RESTORE_CENTRE) -> tuple[float, float]:
    """A 10N grid corner on whole kilometres near ``centre``, so 200 m fold cells start on grid pixels."""
    x, y = Transformer.from_crs(4326, RESTORE_EPSG, always_xy=True).transform(*centre)
    return float((x // 1000) * 1000), float((y // 1000) * 1000)


def in_box(rows: np.ndarray, cols: np.ndarray, box_rc: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    (r0, r1), (c0, c1) = box_rc
    return (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)


def site_weight(year: int, planted: int = RESTORE_PLANTED, ramp: int = RESTORE_RAMP) -> float:
    """How far the moving site is from pasture (0) to forest (1) in ``year``."""
    return min(max((year - (planted - 1)) / ramp, 0.0), 1.0)


def restore_pattern(
    *,
    moves: bool = True,
    crop: bool = True,
    site: tuple[tuple[int, int], tuple[int, int]] = SITE_BOX,
    target: tuple[tuple[int, int], tuple[int, int]] | None = TARGET_BOX,
    crop_box: tuple[tuple[int, int], tuple[int, int]] = CROP_BOX,
    holes: tuple[tuple[int, tuple[tuple[int, int], tuple[int, int]]], ...] = (),
    radar_free: tuple[tuple[int, tuple[tuple[int, int], tuple[int, int]]], ...] = (),
    switch_years: tuple[int, ...] = (),
    planted: int = RESTORE_PLANTED,
) -> tuple[Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, ...]], Callable[[int], np.ndarray]]:
    """The class map and prototypes of a restore scenario, for :func:`designed_zone`.

    Pasture everywhere, the site (moving from pasture to forest from ``planted`` over
    :data:`RESTORE_RAMP` years when ``moves``, pasture throughout otherwise), a forest ``target`` and a
    ``crop`` patch. ``holes`` are ``(year, box)`` with no data, ``radar_free`` ``(year, box)`` without
    Sentinel-1, and ``switch_years`` years in which the whole grid is radar-free and every vector is
    shifted by :data:`RADAR_SHIFT`.
    """

    def prototypes(year: int) -> np.ndarray:
        w = site_weight(year, planted) if moves else 0.0
        protos = np.stack([PASTURE, FOREST, CROP, (1.0 - w) * PASTURE + w * FOREST])
        return protos + (RADAR_SHIFT if year in switch_years else 0.0)

    def pattern(rows: np.ndarray, cols: np.ndarray, year: int) -> tuple[np.ndarray, ...]:
        classes = np.full(rows.shape, CLASS_PASTURE, dtype=np.int64)
        if crop:
            classes[in_box(rows, cols, crop_box)] = CLASS_CROP
        if target is not None:
            classes[in_box(rows, cols, target)] = CLASS_FOREST
        classes[in_box(rows, cols, site)] = CLASS_SITE
        data = np.ones(rows.shape, dtype=bool)
        for y, b in holes:
            if y == year:
                data &= ~in_box(rows, cols, b)
        radar = np.full(rows.shape, year not in switch_years, dtype=bool)
        for y, b in radar_free:
            if y == year:
                radar &= ~in_box(rows, cols, b)
        return classes, data, radar

    return pattern, prototypes


def restore_store(
    *,
    size: int = 192,
    chunk: int = 64,
    years_complete: list[int] | None = None,
    centre: tuple[float, float] = RESTORE_CENTRE,
    **pattern_kwargs: Any,
) -> tuple[TesseraReader, KeyLog, MemoryStore]:
    """A one-zone (10N) restore scenario store: :func:`restore_pattern` on a ``size`` grid at :func:`restore_corner`."""
    pattern, prototypes = restore_pattern(**pattern_kwargs)
    left, top = restore_corner(centre)
    return designed_store(
        {
            "10N": {
                "left": left,
                "top": top,
                "height": size,
                "width": size,
                "pattern": pattern,
                "prototypes": prototypes,
                "chunk": chunk,
                "years_complete": years_complete,
            }
        }
    )


def grid_polygon(
    left: float, top: float, box_rc: tuple[tuple[int, int], tuple[int, int]], epsg: int = RESTORE_EPSG
) -> Any:
    """The lon/lat polygon of grid pixels ``((row0, row1), (col0, col1))`` of a grid with corner (left, top)."""
    import shapely
    from shapely.geometry import box

    (r0, r1), (c0, c1) = box_rc
    square = box(left + c0 * 10.0, top - r1 * 10.0, left + c1 * 10.0, top - r0 * 10.0)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    return shapely.transform(shapely.segmentize(square, 10.0), to_ll.transform, interleaved=False)


def write_polygon(path: Path, geometry: Any, properties: dict[str, Any] | None = None) -> Path:
    """Write one lon/lat geometry as a GeoJSON FeatureCollection."""
    import json

    from shapely.geometry import mapping

    feature = {"type": "Feature", "properties": properties or {}, "geometry": mapping(geometry)}
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
    return path


#: An ANSI escape sequence (colour, bold and the like).
ANSI_ESCAPE = re.compile(rb"\x1b\[[0-9;]*m")


class PlainCliRunner(CliRunner):
    """A CliRunner whose captured output carries no ANSI escapes.

    Typer and rich colour their output where they detect CI (``GITHUB_ACTIONS``, ``FORCE_COLOR``,
    ``PY_COLORS``) even when it is captured, and the escape codes split the words the CLI tests
    match (an option name such as ``--aoi``, a table cell). The tests check words, not colours.
    """

    def invoke(
        self,
        app: Typer,
        args: str | Sequence[str] | None = None,
        input: bytes | str | None = None,
        env: Mapping[str, str | None] | None = None,
        catch_exceptions: bool = True,
        color: bool = False,
        **extra: Any,
    ) -> Result:
        result = super().invoke(
            app, args, input=input, env=env, catch_exceptions=catch_exceptions, color=color, **extra
        )
        result.stdout_bytes = ANSI_ESCAPE.sub(b"", result.stdout_bytes)
        if result.stderr_bytes is not None:
            result.stderr_bytes = ANSI_ESCAPE.sub(b"", result.stderr_bytes)
        result.output_bytes = ANSI_ESCAPE.sub(b"", result.output_bytes)
        return result
