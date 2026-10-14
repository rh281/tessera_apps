"""Store-level tests for tessera_apps.io: bytes counted against ground truth, the chunk cache, and
the year and time-axis guards. In-memory stores only, no network.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import logging
import sys
import threading
import types
from collections.abc import Callable, Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import zarr
from shapely.geometry import box
from zarr.abc.store import ByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, cpu
from zarr.storage import MemoryStore

from tessera_apps.cache import ChunkCache, ChunkKey
from tessera_apps.io import (
    ALL_LAYERS,
    CountingStore,
    TesseraError,
    TesseraReader,
    Window,
    YearsNotPublishedError,
    ZoneGrid,
    _decode_years,
    iter_plan_blocks,
    plan_aoi,
)

from .conftest import BOUNDARY, KeyLog, add_zone

FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"
CHUNK, SHARD, SIZE = 32, 64, 128
#: Year read by most tests, and its index on the 2017 to 2025 time axis.
YEAR, T = 2020, 3

Call = tuple[str, str, ByteRequest | None, int]


# ---- ground-truth store -----------------------------------------------------------------------


class Tally:
    """Every buffer a backing store handed out, as (method, key, byte range, length)."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    @property
    def bytes(self) -> int:
        return sum(c[3] for c in self.calls)


class TallyStore(MemoryStore):
    """A MemoryStore that records the length of every buffer its ``get`` and ``get_sync`` return.

    It sits below the CountingStore, so it sees each fetch as the backend serves it, merged ranges
    and their gaps included. That is the ground truth ``bytes_read`` must equal.
    """

    def __init__(
        self,
        store_dict: MutableMapping[str, Buffer] | None = None,
        *,
        read_only: bool = False,
        tally: Tally | None = None,
    ) -> None:
        super().__init__(store_dict, read_only=read_only)
        self.tally = tally if tally is not None else Tally()

    def with_read_only(self, read_only: bool = False) -> TallyStore:
        return type(self)(self._store_dict, read_only=read_only, tally=self.tally)

    async def get(
        self, key: str, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        buf = await super().get(key, prototype, byte_range)
        if buf is not None:
            self.tally.calls.append(("get", key, byte_range, len(buf)))
        return buf

    def get_sync(
        self, key: str, *, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        buf = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        if buf is not None:
            self.tally.calls.append(("get_sync", key, byte_range, len(buf)))
        return buf


class AsyncOnlyTallyStore(TallyStore):
    """The same store with sync IO switched off, as IcechunkStore has it, so zarr takes the async path."""

    _supports_sync_io = False


def _random_fill(seed: int) -> Callable[[str, int, tuple[int, ...]], np.ndarray]:
    rng = np.random.default_rng(seed)

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        if name == "embeddings":
            e = rng.integers(-127, 128, size=shape, dtype=np.int8)
            e[..., 0] = np.where(e[..., 0] == 0, 1, e[..., 0])  # no all-zero vector anywhere
            return e
        if name == "scales":
            return rng.uniform(0.001, 0.1, size=shape).astype(np.float32)
        return rng.integers(0, 60, size=shape).astype(np.uint16)

    return fill


@functools.cache
def _zone_keys(seed: int) -> dict[str, Buffer]:
    """Keys of one 128 x 128 10N group of random data: 32 px chunks in 64 px shards, all stored."""
    store = MemoryStore()
    root = zarr.open_group(store, mode="w")
    fill = _random_fill(seed)
    add_zone(root, "10N", left=500_000.0, top=4_300_000.0, height=SIZE, width=SIZE, fill=fill, chunk=CHUNK, shard=SHARD)
    return dict(store._store_dict)


def _zone_store[S: MemoryStore](seed: int, cls: type[S]) -> S:
    """A fresh store object over a copy of the seeded group's keys (built once per seed)."""
    return cls(dict(_zone_keys(seed)))


MODES: dict[str, tuple[type[TallyStore], str | None, str]] = {
    # zarr's default pipeline reads through the async get even when the store can read synchronously.
    "batched": (TallyStore, None, "get"),
    "batched-async-only": (AsyncOnlyTallyStore, None, "get"),
    # The fused pipeline reads a sync-capable store through get_sync and get_ranges_sync.
    "fused-sync": (TallyStore, FUSED, "get_sync"),
}


@pytest.fixture(params=list(MODES))
def tally_reader(request) -> Iterator[tuple[TesseraReader, TallyStore, str]]:
    cls, pipeline, method = MODES[request.param]
    store = _zone_store(1, cls)
    ctx = zarr.config.set({"codec_pipeline.path": pipeline}) if pipeline else contextlib.nullcontext()
    with ctx:
        yield TesseraReader.from_store(store), store, method


def _read(reader: TesseraReader, store: TallyStore, rows: slice, cols: slice) -> tuple[int, int, list[Call]]:
    """Read one window; return (bytes counted, bytes the backing store returned, its calls)."""
    counted0, truth0, n0 = reader.bytes_read, store.tally.bytes, len(store.tally.calls)
    reader.read_window("10N", YEAR, rows, cols)
    return reader.bytes_read - counted0, store.tally.bytes - truth0, store.tally.calls[n0:]


def _embeddings_calls(calls: list[Call]) -> list[Call]:
    return [c for c in calls if "/embeddings/c/" in c[1]]


# ---- bytes read ---------------------------------------------------------------------------------


def test_bytes_read_equals_what_the_store_returned(tally_reader):
    reader, store, method = tally_reader
    reader.grid("10N")
    assert reader.bytes_read == store.tally.bytes > 0  # metadata is counted too

    # Partial shards: a 32 x 32 window across two shards, so chunks arrive through ranged reads.
    counted, truth, calls = _read(reader, store, slice(16, 48), slice(48, 80))
    emb = _embeddings_calls(calls)
    assert any(isinstance(c[2], RangeByteRequest) for c in emb)
    assert {c[0] for c in emb} == {method}
    assert truth > 32 * 32 * 128  # chunk payload, not just shard indexes
    assert counted == truth

    # A whole shard: one unranged read per layer.
    counted, truth, calls = _read(reader, store, slice(64, 128), slice(0, 64))
    emb = _embeddings_calls(calls)
    assert [c[2] for c in emb] == [None]
    assert {c[0] for c in emb} == {method}
    assert truth > 64 * 64 * 128
    assert counted == truth

    assert reader.bytes_read == store.tally.bytes
    if isinstance(store, AsyncOnlyTallyStore):
        assert {c[0] for c in store.tally.calls} == {"get"}


def _shard_index(store: MemoryStore, key: str, n: int) -> np.ndarray:
    """(offset, nbytes) of each inner chunk of an n x n shard, from the index at the shard's end."""
    buf = MemoryStore.get_sync(store, key, byte_range=SuffixByteRequest(16 * n * n + 4))  # untallied
    assert buf is not None
    return np.frombuffer(buf.to_bytes()[:-4], dtype="<u8").reshape(n, n, 2).astype(np.int64)


def _apart_pair(index: np.ndarray) -> tuple[slice, slice, int, int]:
    """A window over two neighbouring chunks of shard (0, 0) whose bytes are apart in the shard.

    Returns the window's rows and cols and the byte span one merged fetch of the two covers.
    """
    pairs = (((0, 1), slice(0, 32), slice(0, 64)), ((1, 0), slice(0, 64), slice(0, 32)))  # side by side, stacked
    for (r, c), rows, cols in pairs:
        start = int(min(index[0, 0, 0], index[r, c, 0]))
        end = int(max(index[0, 0].sum(), index[r, c].sum()))
        if end - start > int(index[0, 0, 1] + index[r, c, 1]):
            return rows, cols, start, end
    pytest.fail("No two neighbouring chunks are apart in the shard; pick another pair for this layout.")


def test_gap_bytes_inside_a_coalesced_fetch_are_counted(tally_reader):
    reader, store, _ = tally_reader
    key = f"10N/embeddings/c/{T}/0/0/0"
    rows, cols, start, end = _apart_pair(_shard_index(store, key, SHARD // CHUNK))
    counted, truth, calls = _read(reader, store, rows, cols)
    fetched = [(br.start, br.end) for _, k, br, _ in calls if k == key and isinstance(br, RangeByteRequest)]
    assert fetched == [(start, end)]  # one merged fetch that spans the gap
    assert counted == truth


class _MeetingBuffer(cpu.Buffer):
    """A buffer whose len() waits, up to a timeout, until every reader is inside one.

    ByteCounter.add reads the running total, calls len(buf), then writes the total back. zarr's
    Buffer.__len__ is Python code, so a thread switch can land between the read and the write.
    Holding each reader in len() until all are there makes that interleaving certain. Left to
    chance it is rare: pinned to one CPU, a 16,000-call hammer lost no update in most runs.
    """

    meeting: threading.Barrier | None = None

    def __len__(self) -> int:
        if self.meeting is not None:
            try:
                self.meeting.wait()
            except threading.BrokenBarrierError:
                pass  # a counter that serialises add() keeps the others out, so carry on alone
        return super().__len__()


class _MeetingStore(MemoryStore):
    """A MemoryStore whose get_sync hands out buffers that meet at one barrier."""

    def __init__(self, meeting: threading.Barrier) -> None:
        super().__init__()
        self.meeting = meeting

    def get_sync(
        self, key: str, *, prototype: BufferPrototype | None = None, byte_range: ByteRequest | None = None
    ) -> Buffer | None:
        buf = super().get_sync(key, prototype=prototype, byte_range=byte_range)
        if buf is None:
            return None
        out = _MeetingBuffer(buf.as_numpy_array())
        out.meeting = self.meeting
        return out


def test_concurrent_sync_reads_are_all_counted():
    # Readers that are all inside ByteCounter.add at once: each must still be counted.
    payload = b"x" * 1000
    n_threads = 4
    meeting = threading.Barrier(n_threads, timeout=2.0)
    mem = _MeetingStore(meeting)
    mem.set_sync("k", cpu.Buffer.from_bytes(payload))
    store = CountingStore(mem)
    start = threading.Barrier(n_threads)

    def work(_: int) -> None:
        start.wait()
        store.get_sync("k")

    with ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(work, range(n_threads)))
    assert store.counter.bytes == n_threads * len(payload)
    assert store.counter.requests == n_threads


# ---- where the counter comes from ---------------------------------------------------------------


def test_reader_adopts_the_counter_of_a_counting_store_under_its_group(caplog: pytest.LogCaptureFixture) -> None:
    store = _zone_store(2, TallyStore)
    wrappers: tuple[Callable[[Any], Any], ...] = (lambda s: s, KeyLog)  # directly, and under another wrapper
    for wrap in wrappers:
        truth0 = store.tally.bytes
        counting = CountingStore(store)
        with caplog.at_level(logging.WARNING, logger="tessera_apps.io"):
            reader = TesseraReader(zarr.open_group(wrap(counting), mode="r"))
        reader.read_window("10N", YEAR, slice(0, 40), slice(0, 40))
        assert reader.bytes_read == counting.counter.bytes == store.tally.bytes - truth0
        assert reader.bytes_read > 32 * 32 * 128
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_reader_over_an_uncounted_store_warns(caplog):
    store = _zone_store(2, MemoryStore)
    with caplog.at_level(logging.WARNING, logger="tessera_apps.io"):
        reader = TesseraReader(zarr.open_group(store, mode="r"))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and r.name == "tessera_apps.io"]
    assert len(warnings) == 1 and "CountingStore" in warnings[0].getMessage()
    reader.read_window("10N", YEAR, slice(0, 40), slice(0, 40))
    assert reader.bytes_read == 0  # as the warning says


# ---- cache namespaces and validation ------------------------------------------------------------


def test_unpinned_cache_is_refused_without_a_snapshot(tmp_path):
    store = _zone_store(3, MemoryStore)
    with pytest.raises(ValueError, match="namespace"):
        TesseraReader.from_store(store, cache=ChunkCache(tmp_path))
    with pytest.raises(ValueError, match="namespace"):
        TesseraReader(zarr.open_group(CountingStore(store), mode="r"), cache=ChunkCache(tmp_path))
    assert not list(tmp_path.iterdir())


def test_unpinned_cache_takes_the_snapshot_namespace(tmp_path):
    # Two stores with the same layout and different data share one cache directory.
    stores = {"snapA": _zone_store(4, MemoryStore), "snapB": _zone_store(5, MemoryStore)}
    rows, cols = slice(20, 70), slice(30, 90)
    for snap, store in stores.items():
        reader = TesseraReader.from_store(store, cache=ChunkCache(tmp_path), snapshot_id=snap)
        assert reader.cache is not None and reader.cache.namespace == snap
        got = reader.read_window("10N", YEAR, rows, cols)
        ref = TesseraReader.from_store(store).read_window("10N", YEAR, rows, cols)
        for name in ALL_LAYERS:
            np.testing.assert_array_equal(getattr(got, name), getattr(ref, name), err_msg=f"{snap} {name}")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snapA", "snapB"]
    # A cache that already names its data keeps that namespace.
    named = TesseraReader.from_store(stores["snapA"], cache=ChunkCache(tmp_path, namespace="mine"), snapshot_id="x")
    assert named.cache is not None and named.cache.namespace == "mine"


def test_chunk_cache_get_checks_shape_and_dtype(tmp_path):
    cache = ChunkCache(tmp_path, namespace="t")
    key = ChunkKey("10N", 2020, 0, 0)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.put(key, "scales", arr)
    assert cache.get(key, "scales", shape=(4, 3)) is None
    assert cache.get(key, "scales", dtype=np.dtype(np.float64)) is None
    np.testing.assert_array_equal(cache.get(key, "scales", shape=(3, 4), dtype=np.dtype(np.float32)), arr)
    assert (cache.hits, cache.misses) == (1, 2)


def test_cached_chunk_with_wrong_shape_or_dtype_is_a_miss_and_is_overwritten(tmp_path):
    store = _zone_store(6, MemoryStore)
    cache = ChunkCache(tmp_path, namespace="t")
    key = ChunkKey("10N", YEAR, 1, 1)  # rows and cols 32:64
    # Leftovers from another layout: too small, too big, and the right shape in the wrong dtype.
    cache.put(key, "embeddings", np.ones((16, 16, 128), dtype=np.int8))
    cache.put(key, "s2_obs_count", np.full((40, 40), 7, dtype=np.uint16))
    cache.put(key, "scales", np.zeros((32, 32), dtype=np.float64))
    reader = TesseraReader.from_store(store, cache=cache)
    plain = TesseraReader.from_store(store)
    rows, cols = slice(36, 60), slice(40, 64)  # inside chunk (1, 1)
    got = reader.read_window("10N", YEAR, rows, cols)
    ref = plain.read_window("10N", YEAR, rows, cols)
    for name in ALL_LAYERS:
        np.testing.assert_array_equal(getattr(got, name), getattr(ref, name), err_msg=name)
    assert (cache.hits, cache.misses) == (0, len(ALL_LAYERS))
    # The bad entries now hold the whole chunk, in its own shape and dtype.
    whole = plain.read_window("10N", YEAR, slice(32, 64), slice(32, 64))
    for name in ("embeddings", "s2_obs_count", "scales"):
        want = getattr(whole, name)
        np.testing.assert_array_equal(cache.get(key, name, shape=want.shape, dtype=want.dtype), want, err_msg=name)


def test_namespaces_derived_from_a_cache_share_its_stats(tmp_path):
    parent = ChunkCache(tmp_path)
    child = parent.for_namespace("a")
    grandchild = child.for_namespace("b")
    key = ChunkKey("10N", 2020, 0, 0)
    assert child.get(key, "scales") is None
    grandchild.put(key, "scales", np.ones((2, 2), dtype=np.float32))
    assert grandchild.get(key, "scales") is not None
    assert parent.get(key, "scales") is None  # another namespace: a miss
    assert (parent.hits, parent.misses) == (1, 2)
    assert (child.hits, child.misses) == (grandchild.hits, grandchild.misses) == (1, 2)


def test_open_namespaces_the_callers_cache_and_shares_its_stats(tmp_path, monkeypatch):
    store = _zone_store(7, MemoryStore)

    class Repo:
        def readonly_session(self, branch=None, *, snapshot_id=None):
            return types.SimpleNamespace(store=store, snapshot_id=snapshot_id or "SNAP123")

    fake = types.SimpleNamespace(
        s3_storage=lambda **kw: kw,
        Repository=types.SimpleNamespace(open=lambda storage: Repo()),
        __version__="0.0-test",
    )
    monkeypatch.setitem(sys.modules, "icechunk", fake)
    mine = ChunkCache(tmp_path)
    reader = TesseraReader.open(cache=mine)
    assert reader.cache is not None and reader.cache is not mine and reader.cache.namespace == "SNAP123"
    assert reader.provenance()["snapshot_id"] == "SNAP123"
    rows, cols = slice(10, 50), slice(10, 50)
    first = reader.read_window("10N", YEAR, rows, cols)
    after_first = reader.bytes_read
    second = reader.read_window("10N", YEAR, rows, cols)
    assert after_first > 0 and reader.bytes_read == after_first
    assert mine.misses > 0 and mine.hits > 0
    assert (mine.hits, mine.misses) == (reader.cache.hits, reader.cache.misses)
    np.testing.assert_array_equal(second.embeddings, first.embeddings)
    assert [p.name for p in tmp_path.iterdir()] == ["SNAP123"]


# ---- cache keying and fetch count ---------------------------------------------------------------


def test_cache_keys_entries_by_year_and_zone(boundary_store, tmp_path):
    mem, _ = boundary_store
    cache = ChunkCache(tmp_path, namespace="t")
    cached = TesseraReader.from_store(mem, cache=cache)
    plain = TesseraReader.from_store(mem)
    rows, cols = slice(40, 80), slice(40, 80)
    # 2022 is radar-free in the mock and 2021 is not, so a year mix-up shows in every layer.
    assert plain.read_window("10N", 2021, rows, cols).radar_present.all()
    assert not plain.read_window("10N", 2022, rows, cols).radar_present.any()
    for zone, year in (("10N", 2021), ("10N", 2022), ("11N", 2022), ("10N", 2021)):
        got = cached.read_window(zone, year, rows, cols)
        ref = plain.read_window(zone, year, rows, cols)
        for name in ALL_LAYERS:
            np.testing.assert_array_equal(getattr(got, name), getattr(ref, name), err_msg=f"{zone} {year} {name}")
    assert cache.hits > 0
    many = cached.read_windows("10N", [2020, 2021, 2022], rows, cols)
    for year, got in many.items():
        np.testing.assert_array_equal(got.embeddings, plain.read_window("10N", year, rows, cols).embeddings)


def test_cold_cache_fetches_each_year_and_layer_once(boundary_store, tmp_path):
    mem, _ = boundary_store
    cached_log, plain_log = KeyLog(mem), KeyLog(mem)
    cached = TesseraReader.from_store(cached_log, cache=ChunkCache(tmp_path, namespace="t"))
    plain = TesseraReader.from_store(plain_log)
    years = [2020, 2021]
    rows, cols = slice(40, 100), slice(40, 100)  # 3 x 3 chunks of 32 px; the chunk-aligned cover is 32:128
    got = cached.read_windows("10N", years, rows, cols)
    plain.read_windows("10N", years, slice(32, 128), slice(32, 128))
    # The cold read makes exactly the store requests of one plain read of the cover per year and
    # layer, not one read per chunk.
    assert collections.Counter(cached_log.chunk_keys()) == collections.Counter(plain_log.chunk_keys())
    emb = [k for k in cached_log.chunk_keys() if f"10N/embeddings/c/{T}/" in k]
    assert 0 < len(emb) < 9
    ref = plain.read_windows("10N", years, rows, cols)
    cached_log.log.clear()
    again = cached.read_windows("10N", years, rows, cols)
    assert cached_log.chunk_keys() == []
    for year in years:
        for name in ALL_LAYERS:
            np.testing.assert_array_equal(getattr(got[year], name), getattr(ref[year], name), err_msg=name)
            np.testing.assert_array_equal(getattr(again[year], name), getattr(ref[year], name), err_msg=name)


# ---- years --------------------------------------------------------------------------------------


def test_read_window_refuses_an_unpublished_year_before_any_chunk(boundary_store, boundary_reader):
    _, log = boundary_store
    with pytest.raises(YearsNotPublishedError) as err:
        boundary_reader.read_window("11N", 2023, slice(0, 8), slice(0, 8))
    assert err.value.missing == {"11N": [2023]}
    with pytest.raises(YearsNotPublishedError) as err:
        boundary_reader.read_windows("11N", [2022, 2023], slice(0, 8), slice(0, 8))
    assert err.value.missing == {"11N": [2023]}
    assert log.chunk_keys() == []
    # 2023 is on the 11N time axis: only years_complete marks it as a gap.
    assert boundary_reader.year_index("11N", 2023) == 6


def test_iter_plan_blocks_checks_years_when_called(boundary_store, boundary_reader):
    _, log = boundary_store
    aoi = box(BOUNDARY[0] + 0.0005, BOUNDARY[1] - 0.002, BOUNDARY[0] + 0.003, BOUNDARY[1] + 0.002)
    (p,) = plan_aoi(boundary_reader, aoi).zones
    assert p.zone == "11N"
    with pytest.raises(YearsNotPublishedError) as err:
        iter_plan_blocks(boundary_reader, p, [2022, 2023])  # raises on the call, before any next()
    assert err.value.missing == {"11N": [2023]}
    assert log.chunk_keys() == []


# ---- window flags -------------------------------------------------------------------------------


def _window(
    scales: np.ndarray,
    *,
    embeddings: np.ndarray | None = None,
    asc: np.ndarray | None = None,
    desc: np.ndarray | None = None,
) -> Window:
    h, w = scales.shape
    grid = ZoneGrid(zone="10N", epsg=32610, left=500_000.0, top=4_300_000.0, height=h, width=w)
    return Window(
        zone="10N",
        year=2021,
        rows=slice(0, h),
        cols=slice(0, w),
        grid=grid,
        scales=scales,
        embeddings=embeddings,
        s1_asc_obs_count=asc,
        s1_desc_obs_count=desc,
    )


def test_radar_present_counts_either_pass():
    # Descending only (Rondonia's pattern), ascending only, neither, and a sum past the uint16 range.
    asc = np.array([[0, 30, 0, 65535]], dtype=np.uint16)
    desc = np.array([[59, 0, 0, 1]], dtype=np.uint16)
    win = _window(np.full((1, 4), 0.02, dtype=np.float32), asc=asc, desc=desc)
    np.testing.assert_array_equal(win.radar_present, [[True, True, False, True]])
    np.testing.assert_array_equal(win.radar_free, [[False, False, True, False]])


def test_dequantize_is_nan_for_a_zero_vector_with_a_finite_scale():
    emb = np.zeros((1, 2, 128), dtype=np.int8)
    emb[0, 1] = (np.arange(128) - 64).astype(np.int8)
    win = _window(np.array([[0.02, 0.02]], dtype=np.float32), embeddings=emb)
    np.testing.assert_array_equal(win.valid, [[False, True]])
    deq = win.dequantize()
    assert np.isnan(deq[0, 0]).all()
    np.testing.assert_allclose(deq[0, 1], emb[0, 1].astype(np.float32) * np.float32(0.02))


# ---- time axis ----------------------------------------------------------------------------------


def _stamps(first: int, n: int = 9) -> np.ndarray:
    return np.array([f"{first + i}-01-01" for i in range(n)], dtype="datetime64[ns]")


def _time_group(values: np.ndarray, units: str | None = None) -> zarr.Group:
    g = zarr.open_group(MemoryStore(), mode="w")
    t = g.create_array("time", shape=values.shape, dtype=values.dtype)
    t[:] = values
    if units is not None:
        t.attrs["units"] = units
    return g


def test_decode_years_checks_a_cf_time_axis():
    ns = "nanoseconds since 1970-01-01"
    assert _decode_years(_time_group(_stamps(2017).astype(np.int64), ns), 9) == list(range(2017, 2026))
    with pytest.raises(TesseraError, match=r"years \[2018, .* layout has changed"):
        _decode_years(_time_group(_stamps(2018).astype(np.int64), ns), 9)
    # The right stamps under the wrong unit decode far outside 2017 to 2025.
    with pytest.raises(TesseraError, match="layout has changed"):
        _decode_years(_time_group(_stamps(2017).astype(np.int64), "days since 1970-01-01"), 9)


def test_decode_years_checks_a_datetime64_time_axis():
    assert _decode_years(_time_group(_stamps(2017)), 9) == list(range(2017, 2026))
    with pytest.raises(TesseraError, match=r"years \[2018, "):
        _decode_years(_time_group(_stamps(2018)), 9)
