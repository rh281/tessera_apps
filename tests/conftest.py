"""Mock stores with the published store's layout, built in memory from fixtures or synthetic data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from pyproj import Transformer
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
