"""Block streaming and point sampling tests for tessera_apps.io, measured at the store.

The block tests check what iter_plan_blocks returns (every year, the block mask) and what it fetches:
blocks with no AOI pixel are skipped, kept blocks are trimmed to the chunks holding their AOI pixels,
every block edge is chunk aligned, and the bytes read stay within ``AOIPlan.estimated_bytes``. The
point tests check that read_points fetches each chunk once, concurrently, from each point's nominal
zone, after checking years. In-memory stores only, no network.
"""

from __future__ import annotations

import collections
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import shapely
import zarr
from pyproj import Transformer
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from zarr.storage import MemoryStore

import tessera_apps.io as tio
from tessera_apps.io import (
    ALL_LAYERS,
    BYTES_PER_PIXEL,
    DEFAULT_BLOCK,
    DEFAULT_LAYERS,
    RADAR_LAYERS,
    PointRead,
    TesseraReader,
    Window,
    YearsNotPublishedError,
    ZoneGrid,
    ZonePlan,
    iter_blocks,
    iter_plan_blocks,
    plan_aoi,
    read_plan,
    read_points,
    zone_epsg,
    zone_for,
)

from .conftest import BOUNDARY, KeyLog, _synthetic, add_zone

#: Inner chunk edge of every mock store here, and a block of two by two chunks.
CHUNK = 32
SMALL_BLOCK = (64, 64)
#: Layers read_points fetches with radar.
POINT_LAYERS = ("embeddings", "scales", *RADAR_LAYERS)

#: An L in (col, row) pixel coordinates of the boundary 10N grid: a bar down the west side and a foot
#: along the south, both west of the 10N/11N band edge (column 94 or more on that grid).
L_PIXELS = Polygon([(8, 8), (30, 8), (30, 165), (88, 165), (88, 184), (8, 184)])
L_PIXEL_COUNT = 22 * 157 + 80 * 19
#: A thin diagonal strip in the same pixel coordinates.
DIAGONAL_PIXELS = LineString([(6, 10), (90, 180)]).buffer(3.0, cap_style="flat")
#: A thin lon/lat diagonal across 120 W, with a part in each of 10N and 11N.
CROSSING = LineString([(-120.010, 38.8575), (-119.990, 38.8425)]).buffer(0.0003, cap_style="flat")


# ---- helpers ----------------------------------------------------------------------------------


def _pixel_aoi(grid: ZoneGrid, shape: BaseGeometry) -> BaseGeometry:
    """A lon/lat AOI from a shape drawn in (col, row) pixel coordinates of a zone grid.

    Edges on whole pixel lines keep every pixel centre half a pixel off the boundary, so the plan
    mask holds exactly the pixels inside the shape.
    """

    def to_utm(xy: np.ndarray) -> np.ndarray:
        return np.column_stack([grid.left + xy[:, 0] * grid.pixel, grid.top - xy[:, 1] * grid.pixel])

    utm = shapely.segmentize(shapely.transform(shape, to_utm), grid.pixel)
    return shapely.transform(utm, grid.to_lonlat().transform, interleaved=False)


def _aoi(reader: TesseraReader, name: str) -> BaseGeometry:
    if name == "crossing":
        return CROSSING
    return _pixel_aoi(reader.grid("10N"), {"L": L_PIXELS, "diagonal": DIAGONAL_PIXELS}[name])


def _chunks(rows: slice, cols: slice, chunk: tuple[int, int] = (CHUNK, CHUNK)) -> set[tuple[int, int]]:
    """Inner chunks a window overlaps, as (chunk row, chunk col)."""
    ch, cw = chunk
    return {
        (r, c)
        for r in range(rows.start // ch, (rows.stop - 1) // ch + 1)
        for c in range(cols.start // cw, (cols.stop - 1) // cw + 1)
    }


def _chunks_with_pixels(p: ZonePlan, chunk: tuple[int, int] = (CHUNK, CHUNK)) -> set[tuple[int, int]]:
    """Brute force: the chunk of every AOI pixel."""
    rr, cc = np.nonzero(p.mask)
    rows = ((rr + p.rows.start) // chunk[0]).tolist()
    cols = ((cc + p.cols.start) // chunk[1]).tolist()
    return set(zip(rows, cols, strict=True))


def _reference_blocks(p: ZonePlan, chunk: tuple[int, int], block: tuple[int, int]) -> list[tuple[slice, slice]]:
    """What ZonePlan.blocks should yield, worked out from chunk indices.

    Each region iter_blocks hands out is dropped when it holds no AOI pixel, and otherwise shrunk to
    the rectangle of chunks spanned by the chunks of its AOI pixels, clipped to the region.
    """
    ch, cw = chunk
    out = []
    for rs, cs in iter_blocks(p.rows, p.cols, block=block, align=chunk):
        rr, cc = np.nonzero(p.submask(rs, cs))
        if rr.size == 0:
            continue
        crs, ccs = (rr + rs.start) // ch, (cc + cs.start) // cw
        out.append(
            (
                slice(max(int(crs.min()) * ch, rs.start), min((int(crs.max()) + 1) * ch, rs.stop)),
                slice(max(int(ccs.min()) * cw, cs.start), min((int(ccs.max()) + 1) * cw, cs.stop)),
            )
        )
    return out


def _lonlat_of_pixels(
    grid: ZoneGrid, rows: Sequence[int] | np.ndarray, cols: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lon/lat of pixel centres of a zone grid."""
    xs = grid.left + (np.asarray(cols) + 0.5) * grid.pixel
    ys = grid.top - (np.asarray(rows) + 0.5) * grid.pixel
    lons, lats = grid.to_lonlat().transform(xs, ys)
    return np.asarray(lons), np.asarray(lats)


def _fetched(log: KeyLog, zone: str, layer: str) -> collections.Counter[tuple[int, int, int]]:
    """Fetches of one layer as (time index, chunk row, chunk col), on a store with one chunk per shard."""
    prefix = f"{zone}/{layer}/c/"
    out: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for key in log.log:
        if key.startswith(prefix):
            t, r, c = (int(v) for v in key[len(prefix) :].split("/")[:3])
            out[(t, r, c)] += 1
    return out


@pytest.fixture(scope="module")
def _chunked_mem() -> MemoryStore:
    """The boundary store's 10N group again, with one inner chunk per shard.

    Every store key is then a single inner chunk, so the key log names exactly the chunks fetched.
    Same generator and seed as boundary_store, so the data match its 10N group.
    """
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    size = 192
    x, y = Transformer.from_crs(4326, zone_epsg("10N"), always_xy=True).transform(*BOUNDARY)
    add_zone(
        root,
        "10N",
        left=(x // 10) * 10 - size / 2 * 10,
        top=(y // 10) * 10 + size / 2 * 10,
        height=size,
        width=size,
        fill=_synthetic(100, 10),
        chunk=CHUNK,
        shard=CHUNK,
    )
    return mem


@pytest.fixture
def chunked(_chunked_mem) -> tuple[KeyLog, TesseraReader]:
    log = KeyLog(_chunked_mem)
    return log, TesseraReader.from_store(log)


# ---- blocks: contents ---------------------------------------------------------------------------


@pytest.mark.parametrize("aoi_name", ["L", "crossing"])
def test_every_block_year_equals_read_plan_under_the_block_mask(boundary_reader, aoi_name):
    years = [2020, 2022]  # embeddings differ every year; 2020 has radar, 2022 has none
    plan = plan_aoi(boundary_reader, _aoi(boundary_reader, aoi_name))
    n_blocks = 0
    for p in plan.zones:
        whole = {y: read_plan(boundary_reader, p, y) for y in years}
        assert not np.array_equal(whole[2020].embeddings, whole[2022].embeddings)
        assert not np.array_equal(whole[2020].s1_asc_obs_count, whole[2022].s1_asc_obs_count)
        covered = np.zeros(p.shape, dtype=np.int32)
        for blk in iter_plan_blocks(boundary_reader, p, years, block=SMALL_BLOCK):
            n_blocks += 1
            r = slice(blk.rows.start - p.rows.start, blk.rows.stop - p.rows.start)
            c = slice(blk.cols.start - p.cols.start, blk.cols.stop - p.cols.start)
            np.testing.assert_array_equal(blk.mask, p.mask[r, c])
            assert blk.mask.any()
            covered[r, c] += 1
            assert sorted(blk.windows) == years
            for y, win in blk.windows.items():
                assert win.year == y
                assert (win.rows, win.cols) == (blk.rows, blk.cols)
                assert win.mask is not None, f"{p.zone} {y}: block window has no mask"
                np.testing.assert_array_equal(win.mask, blk.mask)
                for layer in ALL_LAYERS:
                    np.testing.assert_array_equal(
                        getattr(win, layer), getattr(whole[y], layer)[r, c], err_msg=f"{p.zone} {y} {layer}"
                    )
        # Blocks never overlap, and every AOI pixel is in one of them.
        assert covered.max() == 1
        assert (covered[p.mask] == 1).all()
    assert n_blocks > 2


# ---- blocks: what is fetched --------------------------------------------------------------------


def test_blocks_skip_and_trim_chunks_without_aoi_pixels(chunked):
    log, reader = chunked
    (p,) = plan_aoi(reader, _aoi(reader, "L")).zones
    assert p.n_pixels == L_PIXEL_COUNT  # the plan mask is the L as drawn
    chunk = reader.chunk_shape("10N")
    assert chunk == (CHUNK, CHUNK)

    regions = list(iter_blocks(p.rows, p.cols, block=SMALL_BLOCK, align=chunk))
    skipped = set().union(*(_chunks(rs, cs) for rs, cs in regions if not p.submask(rs, cs).any()))
    kept_regions = set().union(*(_chunks(rs, cs) for rs, cs in regions if p.submask(rs, cs).any()))
    expected = set().union(*(_chunks(rs, cs) for rs, cs in _reference_blocks(p, chunk, SMALL_BLOCK)))
    trimmed_off = kept_regions - expected
    # The two block regions east of the bar hold no AOI pixel, and the bar's blocks have a chunk
    # column (and the foot's east block a chunk row) with none.
    assert skipped == {(0, 2), (1, 2), (2, 2), (3, 2)}
    assert trimmed_off == {(0, 1), (1, 1), (2, 1), (3, 1), (4, 2)}
    assert not (skipped | trimmed_off) & _chunks_with_pixels(p)
    assert len(expected) == 9

    log.log.clear()
    blocks = list(iter_plan_blocks(reader, p, [2020, 2021], block=SMALL_BLOCK))
    assert len(blocks) == 4
    for layer in ALL_LAYERS:
        fetched = _fetched(log, "10N", layer)
        for t in (3, 4):  # 2020 and 2021 on the time axis
            per_year = {(r, c) for tt, r, c in fetched if tt == t}
            assert not per_year & skipped, f"{layer}: chunks of a block with no AOI pixel were fetched"
            assert not per_year & trimmed_off, f"{layer}: chunks trimmed off a block were fetched"
            assert per_year == expected, layer
        assert set(fetched.values()) == {1}, f"{layer}: a chunk was fetched twice in one year"
    assert p.chunks_fetched(chunk, SMALL_BLOCK) == len(expected)


@pytest.mark.parametrize("block", [DEFAULT_BLOCK, SMALL_BLOCK])
@pytest.mark.parametrize("shape", ["L", "diagonal"])
def test_chunks_fetched_is_what_the_store_serves(chunked, shape, block):
    log, reader = chunked
    (p,) = plan_aoi(reader, _aoi(reader, shape)).zones
    chunk = reader.chunk_shape("10N")
    log.log.clear()
    for _ in iter_plan_blocks(reader, p, [2020, 2021], block=block, layers=["embeddings"]):
        pass
    fetched = _fetched(log, "10N", "embeddings")
    assert set(fetched.values()) == {1}
    for t in (3, 4):
        assert len({(r, c) for tt, r, c in fetched if tt == t}) == p.chunks_fetched(chunk, block)
    assert p.chunks_touched(chunk) <= p.chunks_fetched(chunk, block)


@pytest.mark.parametrize("block", [SMALL_BLOCK, (96, 64)])
@pytest.mark.parametrize("aoi_name", ["L", "crossing"])
def test_block_edges_are_plan_edges_or_chunk_multiples(
    boundary_reader: TesseraReader, aoi_name: str, block: tuple[int, int]
) -> None:
    plan = plan_aoi(boundary_reader, _aoi(boundary_reader, aoi_name))
    for p in plan.zones:
        ch, cw = boundary_reader.chunk_shape(p.zone)
        assert p.rows.start % ch or p.cols.start % cw  # the plan origin is off the chunk grid
        seen: set[tuple[int, int]] = set()
        blocks = list(iter_plan_blocks(boundary_reader, p, [2020], block=block, layers=["scales"]))
        assert len(blocks) > 1
        for blk in blocks:
            for edge, plan_edges, step in (
                (blk.rows.start, (p.rows.start, p.rows.stop), ch),
                (blk.rows.stop, (p.rows.start, p.rows.stop), ch),
                (blk.cols.start, (p.cols.start, p.cols.stop), cw),
                (blk.cols.stop, (p.cols.start, p.cols.stop), cw),
            ):
                assert edge in plan_edges or edge % step == 0, f"{p.zone} block {blk.rows}, {blk.cols}"
            assert blk.rows.stop - blk.rows.start <= block[0]
            assert blk.cols.stop - blk.cols.start <= block[1]
            chunks = _chunks(blk.rows, blk.cols, (ch, cw))
            assert not chunks & seen, f"{p.zone}: a chunk is read by two blocks"
            seen |= chunks


def test_iter_plan_blocks_refuses_unpublished_years_when_called(boundary_store, boundary_reader):
    _, log = boundary_store
    plan = plan_aoi(boundary_reader, CROSSING)
    assert plan.zone_names == ["10N", "11N"]
    p10, p11 = plan.zones
    iter_plan_blocks(boundary_reader, p10, [2023])  # 10N publishes 2023
    with pytest.raises(YearsNotPublishedError) as err:
        iter_plan_blocks(boundary_reader, p11, [2022, 2023])  # the call raises, with no next()
    assert err.value.missing == {"11N": [2023]}
    with pytest.raises(YearsNotPublishedError) as err:
        iter_plan_blocks(boundary_reader, p10, [2020, 2016])
    assert err.value.missing == {"10N": [2016]}
    assert log.chunk_keys() == []


# ---- blocks: the byte estimate ------------------------------------------------------------------


@pytest.mark.parametrize("years", [[2020], [2020, 2021]])
@pytest.mark.parametrize("block", [DEFAULT_BLOCK, SMALL_BLOCK])
def test_block_reads_stay_within_estimated_bytes(boundary_reader, block, years):
    plan = plan_aoi(boundary_reader, CROSSING)
    assert plan.zone_names == ["10N", "11N"]
    estimate = plan.estimated_bytes(boundary_reader, len(years), block=block)
    fetched = touched = 0
    before = boundary_reader.bytes_read
    for p in plan.zones:
        chunk = boundary_reader.chunk_shape(p.zone)
        blocks = list(iter_plan_blocks(boundary_reader, p, years, block=block))  # every layer
        in_blocks = sum(len(_chunks(b.rows, b.cols, chunk)) for b in blocks)
        assert in_blocks == p.chunks_fetched(chunk, block)
        fetched += in_blocks
        touched += p.chunks_touched(chunk)
    read = boundary_reader.bytes_read - before
    assert 0 < read <= estimate
    assert estimate == len(years) * fetched * CHUNK * CHUNK * BYTES_PER_PIXEL
    if block == DEFAULT_BLOCK:
        # Each zone is one block here, so whole windows are read. An estimate from the chunks that
        # hold AOI pixels would be below what the block reader really fetches.
        assert touched < fetched
        assert read > len(years) * touched * CHUNK * CHUNK * BYTES_PER_PIXEL


# ---- chunk counts on hand-built plans -----------------------------------------------------------


def _hand_plan(mask: np.ndarray, row0: int, col0: int) -> ZonePlan:
    grid = ZoneGrid(zone="10N", epsg=32610, left=0.0, top=0.0, height=1024, width=1024)
    rows, cols = slice(row0, row0 + mask.shape[0]), slice(col0, col0 + mask.shape[1])
    return ZonePlan(zone="10N", grid=grid, rows=rows, cols=cols, mask=mask)


def _l_mask() -> np.ndarray:
    m = np.zeros((96, 96), dtype=bool)
    m[:, :5] = True
    m[-5:, :] = True
    return m


def _corner_mask() -> np.ndarray:
    """Single pixels on the first row and column of a chunk, where trimming is easiest to get wrong."""
    m = np.zeros((128, 128), dtype=bool)
    m[0, 0] = m[64, 64] = m[96, 32] = m[127, 127] = True
    return m


#: name: (mask, row0, col0, chunks holding an AOI pixel at 32 x 32)
HAND = {
    "full": (np.ones((96, 96), dtype=bool), 0, 0, 9),
    "offset": (np.ones((96, 96), dtype=bool), 16, 16, 16),
    "diagonal": (np.eye(96, dtype=bool), 0, 0, 3),
    "L": (_l_mask(), 40, 8, 7),
    "corners": (_corner_mask(), 0, 0, 4),
    "bar": (np.ones((4, 96), dtype=bool), 8, 0, 3),  # wide and short: catches swapped axes
}


@pytest.mark.parametrize("name", list(HAND))
def test_chunks_touched_is_exact(name):
    mask, row0, col0, expected = HAND[name]
    p = _hand_plan(mask, row0, col0)
    assert len(_chunks_with_pixels(p)) == expected
    assert p.chunks_touched((CHUNK, CHUNK)) == expected
    # A non-square chunk catches swapped axes.
    assert p.chunks_touched((16, 48)) == len(_chunks_with_pixels(p, (16, 48)))


@pytest.mark.parametrize("block", [SMALL_BLOCK, DEFAULT_BLOCK])
@pytest.mark.parametrize("name", list(HAND))
def test_blocks_match_the_reference_and_cover_every_aoi_pixel(name, block):
    mask, row0, col0, _ = HAND[name]
    p = _hand_plan(mask, row0, col0)
    chunk = (CHUNK, CHUNK)
    blocks = list(p.blocks(chunk, block))
    assert blocks == _reference_blocks(p, chunk, block)
    assert p.chunks_fetched(chunk, block) == sum(len(_chunks(rs, cs)) for rs, cs in blocks)
    assert p.chunks_touched(chunk) <= p.chunks_fetched(chunk, block)
    covered = np.zeros(p.shape, dtype=np.int32)
    for rs, cs in blocks:
        covered[rs.start - row0 : rs.stop - row0, cs.start - col0 : cs.stop - col0] += 1
    assert covered.max() == 1
    assert (covered[mask] == 1).all()


# ---- points ------------------------------------------------------------------------------------


def _two_chunk_points(grid: ZoneGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """40 points in chunk (1, 1) of 10N and 10 in chunk (2, 1): two chunks, in two shards."""
    rng = np.random.default_rng(0)
    rows = np.concatenate([rng.integers(32, 64, 40), rng.integers(64, 96, 10)])
    cols = rng.integers(32, 64, 50)
    lons, lats = _lonlat_of_pixels(grid, rows, cols)
    assert {zone_for(lo, la) for lo, la in zip(lons, lats, strict=True)} == {"10N"}
    return rows, cols, lons, lats


def test_points_fetch_what_one_window_over_their_chunks_fetches(boundary_store, boundary_reader):
    _, log = boundary_store
    rows, cols, lons, lats = _two_chunk_points(boundary_reader.grid("10N"))
    # Warm the metadata and the time axis, so both reads below fetch chunk data only.
    boundary_reader.read_window("10N", 2020, slice(0, 1), slice(0, 1))

    log.log.clear()
    before = boundary_reader.bytes_read
    pts = read_points(boundary_reader, lons, lats, 2020)
    point_keys = collections.Counter(log.chunk_keys())
    point_bytes = boundary_reader.bytes_read - before

    log.log.clear()
    before = boundary_reader.bytes_read
    win = boundary_reader.read_window("10N", 2020, slice(32, 96), slice(32, 64), layers=POINT_LAYERS)
    window_keys = collections.Counter(log.chunk_keys())
    window_bytes = boundary_reader.bytes_read - before

    def embeddings_fetches(keys: collections.Counter[str]) -> int:
        return sum(n for key, n in keys.items() if "/embeddings/" in key)

    assert embeddings_fetches(point_keys) > 0
    assert embeddings_fetches(point_keys) == embeddings_fetches(window_keys)
    assert point_keys == window_keys
    assert point_bytes == window_bytes
    assert pts.chunks_read == 2
    assert pts.valid.all()
    np.testing.assert_allclose(pts.embeddings, win.dequantize()[rows - 32, cols - 32], rtol=1e-6)


def test_points_check_every_zone_year_before_any_chunk(boundary_store, boundary_reader):
    _, log = boundary_store
    lons, lats = [-120.005, -119.997], [38.85, 38.85]
    assert [zone_for(lo, la) for lo, la in zip(lons, lats, strict=True)] == ["10N", "11N"]
    # The 10N point is on the 10N mock grid and 10N publishes 2023, so without the up-front check its
    # chunk would be fetched before the 11N point failed.
    grid = boundary_reader.grid("10N")
    r, c = grid.rowcol(*grid.from_lonlat().transform(lons[0], lats[0]))
    assert 0 <= r < grid.height and 0 <= c < grid.width
    assert 2023 in boundary_reader.years_complete("10N")
    assert 2023 not in boundary_reader.years_complete("11N")
    with pytest.raises(YearsNotPublishedError) as err:
        read_points(boundary_reader, lons, lats, 2023)
    assert err.value.missing == {"11N": [2023]}
    assert log.chunk_keys() == []


def test_points_either_side_of_the_band_edge_come_from_their_own_zone(boundary_reader):
    lons, lats = [-120.003, -119.997], [38.85, 38.85]
    # Each point is also on the other zone's mock grid, so only the zone choice decides its value.
    for lon, lat in zip(lons, lats, strict=True):
        for zone in ("10N", "11N"):
            g = boundary_reader.grid(zone)
            r, c = g.rowcol(*g.from_lonlat().transform(lon, lat))
            assert 4 <= r < g.height and 4 <= c < g.width
    pts = read_points(boundary_reader, lons, lats, 2020)
    assert pts.zones == ["10N", "11N"]
    assert pts.valid.all()
    assert pts.chunks_read == 2
    np.testing.assert_allclose(pts.embeddings[:, 0], [1.0, -1.0], rtol=1e-6)
    for i, zone in enumerate(pts.zones):
        g = boundary_reader.grid(zone)
        r, c = g.rowcol(*g.from_lonlat().transform(lons[i], lats[i]))
        win = boundary_reader.read_window(zone, 2020, slice(r, r + 1), slice(c, c + 1))
        np.testing.assert_allclose(pts.embeddings[i], win.dequantize()[0, 0], rtol=1e-6)


@pytest.mark.parametrize(("year", "radar"), [(2021, True), (2022, False)])
def test_point_radar_flag_follows_the_year(boundary_reader, year, radar):
    lons, lats = [-120.003, -119.997, -120.005], [38.85, 38.85, 38.852]
    pts = read_points(boundary_reader, lons, lats, year)
    assert pts.valid.all()
    assert pts.radar_present is not None
    assert pts.radar_present.tolist() == [radar] * 3
    for i, zone in enumerate(pts.zones):
        g = boundary_reader.grid(zone)
        r, c = g.rowcol(*g.from_lonlat().transform(lons[i], lats[i]))
        win = boundary_reader.read_window(zone, year, slice(r, r + 1), slice(c, c + 1))
        assert bool(win.radar_present[0, 0]) is radar
    assert read_points(boundary_reader, lons, lats, year, with_radar=False).radar_present is None


class _GatedReader:
    """Delegates to a reader, but each window read first waits until ``parties`` reads are waiting.

    With ``strict`` a read that times out raises ``BrokenBarrierError``; otherwise the gate is best
    effort and a timed-out read just goes ahead.
    """

    def __init__(self, inner: TesseraReader, parties: int, *, timeout: float, strict: bool = True) -> None:
        self.inner = inner
        self.barrier = threading.Barrier(parties, timeout=timeout)
        self.strict = strict
        self.calls: list[tuple[str, tuple[int, ...], int, int, int, int]] = []
        self._lock = threading.Lock()

    @property
    def bytes_read(self) -> int:
        return self.inner.bytes_read

    def grid(self, zone: str) -> ZoneGrid:
        return self.inner.grid(zone)

    def chunk_shape(self, zone: str) -> tuple[int, int]:
        return self.inner.chunk_shape(zone)

    def years_complete(self, zone: str) -> list[int]:
        return self.inner.years_complete(zone)

    def provenance(self) -> dict[str, Any]:
        return self.inner.provenance()

    def _gate(self, zone: str, years: Sequence[int], rows: slice, cols: slice) -> None:
        with self._lock:
            self.calls.append((zone, tuple(years), rows.start, rows.stop, cols.start, cols.stop))
        try:
            self.barrier.wait()
        except threading.BrokenBarrierError:
            if self.strict:
                raise

    def read_windows(
        self, zone: str, years: Sequence[int], rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> dict[int, Window]:
        self._gate(zone, years, rows, cols)
        return self.inner.read_windows(zone, years, rows, cols, layers=layers)

    def read_window(
        self, zone: str, year: int, rows: slice, cols: slice, *, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> Window:
        self._gate(zone, [year], rows, cols)
        return self.inner.read_window(zone, year, rows, cols, layers=layers)


def test_point_chunks_are_read_concurrently_and_each_once(boundary_reader, monkeypatch):
    # Pin the group pool's size so the test does not depend on TESSERA_APPS_READ_THREADS.
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="test-group")
    monkeypatch.setattr(tio, "_GROUP_POOL", pool)
    grid = boundary_reader.grid("10N")
    # Three points in each of chunks (1, 1), (2, 0) and (4, 2).
    rows = [40, 45, 50, 72, 80, 90, 135, 140, 150]
    cols = [40, 45, 50, 10, 20, 30, 70, 75, 80]
    lons, lats = _lonlat_of_pixels(grid, rows, cols)
    gated = _GatedReader(boundary_reader, parties=3, timeout=5.0)
    try:
        pts = read_points(gated, lons, lats, 2020)
    except threading.BrokenBarrierError:
        pytest.fail("read_points read its chunk groups one at a time")
    assert pts.chunks_read == 3
    assert pts.valid.all()
    assert sorted(gated.calls) == [
        ("10N", (2020,), 32, 64, 32, 64),
        ("10N", (2020,), 64, 96, 0, 32),
        ("10N", (2020,), 128, 160, 64, 96),
    ]


def test_point_flags_are_computed_once_per_chunk(
    boundary_reader: TesseraReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: collections.Counter[str] = collections.Counter()
    for name in ("valid", "radar_present"):
        original = getattr(Window, name)

        def counted(self: Window, _name: str = name, _original: property = original) -> np.ndarray:
            counts[_name] += 1
            return _original.fget(self)  # type: ignore[misc]

        monkeypatch.setattr(tio.Window, name, property(counted))
    _, _, lons, lats = _two_chunk_points(boundary_reader.grid("10N"))
    pts = read_points(boundary_reader, lons, lats, 2020)
    assert pts.valid.all()
    assert pts.radar_present is not None and pts.radar_present.all()
    assert pts.chunks_read == 2
    # Once per chunk group at most, never once per point (50 here).
    assert counts["valid"] <= pts.chunks_read
    assert counts["radar_present"] <= pts.chunks_read


def test_a_full_batch_of_point_chunks_does_not_deadlock(
    boundary_reader: TesseraReader, monkeypatch: pytest.MonkeyPatch
) -> None:
    workers = 16  # the default TESSERA_APPS_READ_THREADS
    # A private pool, so a deadlock can be broken below without touching the shared one.
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="test-read")
    monkeypatch.setattr(tio, "_POOL", pool)
    # One point in each of 32 chunks (one batch): 16 in 10N west of 120 W, 16 in 11N east of it.
    lons: list[float] = []
    lats: list[float] = []
    for zone, chunk_cols in (("10N", (0, 1, 2)), ("11N", (3, 4, 5))):
        cells = [(cr * CHUNK + 12, cc * CHUNK + 12) for cr in range(6) for cc in chunk_cols][:16]
        lo, la = _lonlat_of_pixels(boundary_reader.grid(zone), [r for r, _ in cells], [c for _, c in cells])
        assert {zone_for(x, y) for x, y in zip(lo, la, strict=True)} == {zone}
        lons.extend(lo.tolist())
        lats.extend(la.tolist())
    assert len(lons) == tio._POINT_BATCH
    # The gate holds the first reads until every worker has one, the order the pool reaches by itself
    # most of the time; it makes the outcome certain. A reader that never fills the gate just waits
    # out its one second timeout.
    gated = _GatedReader(boundary_reader, parties=workers, timeout=1.0, strict=False)
    result: dict[str, PointRead | BaseException] = {}

    def run() -> None:
        try:
            result["points"] = read_points(gated, lons, lats, 2020)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    finished = not thread.is_alive()
    pool.shutdown(wait=False, cancel_futures=True)  # cancels the queued reads, which frees stuck workers
    thread.join(timeout=5.0)
    assert finished, "read_points did not finish 32 chunk groups within 5 s"
    assert "error" not in result, repr(result.get("error"))
    pts = result["points"]
    assert isinstance(pts, PointRead)
    assert pts.chunks_read == 32
    assert pts.valid.all()
