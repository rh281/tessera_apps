"""Store-level tests for tessera_apps.io: bytes counted against ground truth, the read path through
geotessera, the chunk cache, the dataset check and the year and time-axis guards. In-memory stores
copied into local Icechunk repositories, no network.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import sys
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator, MutableMapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import geotessera
import numpy as np
import pytest
import zarr
from geotessera.icechunk import ZoneArray
from pyproj import Transformer
from shapely.geometry import box
from zarr.abc.store import ByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype, cpu
from zarr.storage import MemoryStore, WrapperStore

from tessera_apps.cache import ChunkCache, ChunkKey
from tessera_apps.export import provenance_record
from tessera_apps.io import (
    ALL_LAYERS,
    YEARS,
    CountingStore,
    DatasetMismatchError,
    StoreUnavailableError,
    TesseraError,
    TesseraReader,
    Window,
    YearsNotPublishedError,
    ZoneGrid,
    _strip_edges,
    check_dataset,
    is_metadata_key,
    iter_plan_blocks,
    plan_aoi,
    read_points,
    zone_epsg,
)

from .conftest import BOUNDARY, KeyLog, add_zone

FUSED = "zarr.core.codec_pipeline.FusedCodecPipeline"
CHUNK, SHARD, SIZE = 32, 64, 128
#: Year read by most tests, and its index on the 2017 to 2025 time axis.
YEAR, T = 2020, 3

Call = tuple[str, str, ByteRequest | None, int]


# ---- ground-truth stores ------------------------------------------------------------------------


class Tally:
    """Every buffer a backing store handed out, as (method, key, byte range, length)."""

    def __init__(self) -> None:
        self.calls: list[Call] = []

    @property
    def bytes(self) -> int:
        return sum(c[3] for c in self.calls)

    @property
    def chunk_bytes(self) -> int:
        """Bytes of chunk keys only (zarr v3 ``<array>/c/<coords>``): what CountingStore must count.

        Classified here, not with ``tessera_apps.io.is_metadata_key``, so a key the reader wrongly
        takes for metadata goes missing from its count and not from the truth.
        """
        return sum(c[3] for c in self.calls if "/c/" in c[1])


class TallyStore(MemoryStore):
    """A MemoryStore that records the length of every buffer its ``get`` and ``get_sync`` return.

    Put directly under a CountingStore, it sees each fetch as the backend serves it, merged ranges
    and their gaps included: the ground truth the counter must equal, whichever zarr read path runs.
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


class TallyWrapper(WrapperStore[Any]):
    """Records the length of every buffer the store under it returns.

    ``TesseraReader.from_store`` re-applies it around the local Icechunk copy's session store, inside
    the reader's CountingStore, so it sees each fetch as Icechunk serves it: the reader's ground truth.
    """

    def __init__(self, store: Any, tally: Tally | None = None) -> None:
        super().__init__(store)
        self.tally = tally if tally is not None else Tally()

    def _with_store(self, store: Any) -> TallyWrapper:
        return type(self)(store, self.tally)

    def _record(self, method: str, key: str, byte_range: ByteRequest | None, buf: Buffer | None) -> None:
        if buf is not None:
            self.tally.calls.append((method, key, byte_range, len(buf)))

    async def get(self, key, prototype, byte_range=None):
        buf = await self._store.get(key, prototype, byte_range)
        self._record("get", key, byte_range, buf)
        return buf

    async def get_partial_values(self, prototype, key_ranges: Iterable[tuple[str, ByteRequest | None]]):
        key_ranges = list(key_ranges)
        bufs = await self._store.get_partial_values(prototype, key_ranges)
        for (key, byte_range), buf in zip(key_ranges, bufs, strict=True):
            self._record("get_partial_values", key, byte_range, buf)
        return bufs

    def get_sync(self, key, *, prototype=None, byte_range=None):
        buf = self._store.get_sync(key, prototype=prototype, byte_range=byte_range)
        self._record("get_sync", key, byte_range, buf)
        return buf


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


#: zarr read paths for the CountingStore unit tests: (backing store, codec pipeline, method used).
MODES: dict[str, tuple[type[TallyStore], str | None, str]] = {
    # zarr's default pipeline reads through the async get even when the store can read synchronously.
    "batched": (TallyStore, None, "get"),
    "batched-async-only": (AsyncOnlyTallyStore, None, "get"),
    # The fused pipeline reads a sync-capable store through get_sync and get_ranges_sync.
    "fused-sync": (TallyStore, FUSED, "get_sync"),
}
#: Codec pipelines for the reader tests. The reader always reads an Icechunk session store, which
#: has no sync IO, so both pipelines take the async get.
READER_PIPELINES: dict[str, str | None] = {"batched": None, "fused": FUSED}


def _pipeline(path: str | None) -> contextlib.AbstractContextManager[Any]:
    return zarr.config.set({"codec_pipeline.path": path}) if path else contextlib.nullcontext()


@pytest.fixture(params=list(READER_PIPELINES))
def tally_reader(request) -> Iterator[tuple[TesseraReader, TallyWrapper, MemoryStore]]:
    """A reader over the seeded zone, a tally just under its counter, and the source store."""
    mem = _zone_store(1, MemoryStore)
    wrapper = TallyWrapper(mem)
    with _pipeline(READER_PIPELINES[request.param]):
        yield TesseraReader.from_store(wrapper), wrapper, mem


def _read(reader: TesseraReader, wrapper: TallyWrapper, rows: slice, cols: slice) -> tuple[int, int, list[Call]]:
    """Read one window; return (bytes counted, chunk bytes the store under the counter returned, its calls)."""
    counted0, truth0, n0 = reader.bytes_read, wrapper.tally.chunk_bytes, len(wrapper.tally.calls)
    reader.read_window("10N", YEAR, rows, cols)
    return reader.bytes_read - counted0, wrapper.tally.chunk_bytes - truth0, wrapper.tally.calls[n0:]


def _embeddings_calls(calls: list[Call]) -> list[Call]:
    return [c for c in calls if "/embeddings/c/" in c[1]]


# ---- bytes read ---------------------------------------------------------------------------------


def test_bytes_read_equals_what_the_store_returned(tally_reader):
    reader, wrapper, _ = tally_reader
    reader.grid("10N")
    # Metadata only so far: served from the Icechunk snapshot, and not counted.
    assert wrapper.tally.bytes > 0 and wrapper.tally.chunk_bytes == 0 and reader.bytes_read == 0

    # Partial shards: a 32 x 32 window across two shards, so chunks arrive through ranged reads.
    counted, truth, calls = _read(reader, wrapper, slice(16, 48), slice(48, 80))
    emb = _embeddings_calls(calls)
    assert any(isinstance(c[2], RangeByteRequest) for c in emb)
    assert {c[0] for c in emb} == {"get"}
    assert truth > 32 * 32 * 128  # chunk payload, not just shard indexes
    assert counted == truth

    # A whole shard: one unranged read per layer.
    counted, truth, calls = _read(reader, wrapper, slice(64, 128), slice(0, 64))
    emb = _embeddings_calls(calls)
    assert [c[2] for c in emb] == [None]
    for layer in ALL_LAYERS:  # every layer's chunks are in the truth, not just the embeddings
        assert [c[2] for c in calls if f"10N/{layer}/c/" in c[1]] == [None], layer
    assert truth > 64 * 64 * 128
    assert counted == truth

    assert reader.bytes_read == wrapper.tally.chunk_bytes


def test_only_zarr_metadata_documents_are_metadata_keys():
    for key in ("zarr.json", "10N/zarr.json", "10N/scales/zarr.json", ".zattrs", "10N/.zarray"):
        assert is_metadata_key(key), key
    for key in ("10N/scales/c/3/0/0", "10N/embeddings/c/3/0/0/0", "10N/time/c/0", "10N/s2_obs_count/c/0/1/1"):
        assert not is_metadata_key(key), key


@pytest.mark.parametrize("mode", list(MODES))
def test_counting_store_counts_what_the_backend_returned(mode):
    # The counter itself, on every zarr read path, over a store that can also read synchronously.
    cls, pipeline, method = MODES[mode]
    store = _zone_store(1, cls)
    with _pipeline(pipeline):
        counting = CountingStore(store)
        emb = zarr.open_group(counting, mode="r")["10N"]["embeddings"]
        assert store.tally.bytes > 0 and counting.counter.bytes == 0  # metadata is not counted
        np.asarray(emb[T, 16:48, 48:80, :])
        calls = _embeddings_calls(store.tally.calls)
        assert any(isinstance(c[2], RangeByteRequest) for c in calls)
        assert {c[0] for c in calls} == {method}
        assert counting.counter.bytes == store.tally.chunk_bytes > 32 * 32 * 128
        np.asarray(emb[T, 64:128, 0:64, :])
        assert counting.counter.bytes == store.tally.chunk_bytes > 2 * 64 * 64 * 128
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
    # The local Icechunk copy holds the source store's shards byte for byte, so its shard index applies.
    reader, wrapper, mem = tally_reader
    key = f"10N/embeddings/c/{T}/0/0/0"
    rows, cols, start, end = _apart_pair(_shard_index(mem, key, SHARD // CHUNK))
    counted, truth, calls = _read(reader, wrapper, rows, cols)
    fetched = [(br.start, br.end) for _, k, br, _ in calls if k == key and isinstance(br, RangeByteRequest)]
    assert fetched == [(start, end)]  # one merged fetch that spans the gap
    assert counted == truth


def test_gap_bytes_are_counted_on_the_sync_path():
    store = _zone_store(1, TallyStore)
    key = f"10N/embeddings/c/{T}/0/0/0"
    rows, cols, start, end = _apart_pair(_shard_index(store, key, SHARD // CHUNK))
    with _pipeline(FUSED):
        counting = CountingStore(store)
        emb = zarr.open_group(counting, mode="r")["10N"]["embeddings"]
        np.asarray(emb[T, rows, cols, :])
    calls = store.tally.calls
    fetched = [(br.start, br.end) for _, k, br, _ in calls if k == key and isinstance(br, RangeByteRequest)]
    assert fetched == [(start, end)]
    assert counting.counter.bytes == store.tally.chunk_bytes


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


# ---- the read path: a local Icechunk copy, read through geotessera --------------------------------


def test_from_store_reapplies_wrappers_inside_the_counter():
    mem = _zone_store(2, MemoryStore)
    log = KeyLog(TallyWrapper(mem))
    reader = TesseraReader.from_store(log)
    # The copy is read from the store under the wrappers: they have seen only the root group opened.
    assert log.chunk_keys() == [] and log._store.tally.chunk_bytes == 0
    reader.read_window("10N", YEAR, slice(0, 40), slice(0, 40))
    inner = reader.store.counting._store
    assert isinstance(inner, KeyLog) and inner.log is log.log
    assert isinstance(inner._store, TallyWrapper) and inner._store.tally is log._store.tally
    assert reader.bytes_read == log._store.tally.chunk_bytes > 32 * 32 * 128
    assert any(f"10N/embeddings/c/{T}/" in k for k in log.chunk_keys())
    assert reader.provenance()["backend"] == "KeyLog"


def test_reads_go_through_geotessera_zone_arrays(monkeypatch):
    mem = _zone_store(2, MemoryStore)
    reader = TesseraReader.from_store(mem)
    calls: list[tuple[str, Any]] = []
    outer = ZoneArray._outer

    def spy(self: ZoneArray, key: Any) -> np.ndarray:
        calls.append((self.name, key))
        return outer(self, key)

    monkeypatch.setattr(ZoneArray, "_outer", spy)
    rows, cols = slice(20, 70), slice(30, 90)
    win = reader.read_window("10N", YEAR, rows, cols)
    assert sorted(name for name, _ in calls) == sorted(ALL_LAYERS)
    for name, key in calls:
        # geotessera's own axis order: (time, band, y, x) for embeddings, (time, y, x) otherwise.
        assert key == ((T, slice(None), rows, cols) if name == "embeddings" else (T, rows, cols))
    # The same values as zarr reading the source group directly, embeddings back in (y, x, band) order.
    group = zarr.open_group(mem, mode="r")["10N"]
    for name in ALL_LAYERS:
        got = getattr(win, name)
        assert got.flags.c_contiguous, name
        np.testing.assert_array_equal(got, group[name][T, rows, cols], err_msg=name)


def test_from_store_skips_the_published_dataset_check(monkeypatch):
    # The switch between the two paths: open() asks GeoTesseraZarr for the published dataset's identity
    # and refuses anything but 1.1-dclimate; a local copy is not a published dataset, so from_store
    # never opens a GeoTesseraZarr and records no dataset.
    def no_gt(*args, **kwargs):
        raise AssertionError("from_store must not open GeoTesseraZarr")

    monkeypatch.setattr("geotessera.store.GeoTesseraZarr.__init__", no_gt)
    reader = TesseraReader.from_store(_zone_store(2, MemoryStore))
    assert reader.read_window("10N", YEAR, slice(0, 8), slice(0, 8)).embeddings is not None
    assert "dataset" not in reader.provenance() and reader.provenance()["backend"] == "MemoryStore"


def test_from_store_copies_each_content_once():
    mem = _zone_store(3, MemoryStore)
    first = TesseraReader.from_store(mem)
    again = TesseraReader.from_store(KeyLog(MemoryStore(dict(mem._store_dict))))
    assert again.store.url == first.store.url  # same keys and bytes: one copy
    rows, cols = slice(0, 8), slice(0, 8)
    before = first.read_window("10N", YEAR, rows, cols).embeddings
    assert before is not None
    root = zarr.open_group(mem, mode="r+")
    root["10N"]["embeddings"][T, 0:8, 0:8, :] = 5
    changed = TesseraReader.from_store(mem)
    assert changed.store.url != first.store.url
    after = changed.read_window("10N", YEAR, rows, cols).embeddings
    assert after is not None and (after == 5).all()
    # A reader reads the copy it was made from, so a later change to the source does not reach it.
    np.testing.assert_array_equal(first.read_window("10N", YEAR, rows, cols).embeddings, before)


# ---- geotessera's merged view: offsets, the rows it serves, and the guards in front of it -------

#: Band 0 of every embedding in :func:`_hemisphere_store`: which group a pixel came from.
HEMI_MARKER = {"N": 100, "S": -100}


def _hemisphere_fill(zone: str) -> Callable[[str, int, tuple[int, ...]], np.ndarray]:
    """Layers that are a function of the group's own row and column, so a misplaced read shows.

    Embeddings band 0 is the hemisphere marker, band 1 ``row % 100``, band 2 ``col % 100``; every
    other band and layer is a value between 1 and 120 that also changes with the year.
    """

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        rows, cols = np.indices(shape[:2])
        v = (rows * 7 + cols * 3 + ti) % 120 + 1
        if name == "scales":
            return v.astype(np.float32) / 1000.0
        if name == "embeddings":
            e = np.broadcast_to(v[..., None], shape).astype(np.int8)
            e[..., 0], e[..., 1], e[..., 2] = HEMI_MARKER[zone[-1]], rows % 100, cols % 100
            return e
        return v.astype(np.uint16)

    return fill


def _hemisphere_store(groups: dict[str, tuple[float, float, int, int]]) -> MemoryStore:
    """Zone groups given as ``{zone: (left, top, height, width)}``, 32 px chunks in 64 px shards."""
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    for zone, (left, top, height, width) in groups.items():
        add_zone(root, zone, left=left, top=top, height=height, width=width, fill=_hemisphere_fill(zone))
    return mem


def _source(mem: MemoryStore, zone: str, name: str, rows: slice, cols: slice) -> np.ndarray:
    """A layer of a group of the source store, read with plain zarr: what the reader must return."""
    array = zarr.open_group(mem, mode="r")[f"{zone}/{name}"]
    assert isinstance(array, zarr.Array)
    return np.asarray(array[T, rows, cols])


def _assert_window_is(win: Window, mem: MemoryStore, zone: str, rows: slice, cols: slice) -> None:
    for name in ALL_LAYERS:
        np.testing.assert_array_equal(getattr(win, name), _source(mem, zone, name, rows, cols), err_msg=name)


def _assert_no_data(win: Window) -> None:
    assert win.embeddings is not None and not win.valid.any() and (win.embeddings == 0).all()
    assert win.s2_obs_count is not None and (win.s2_obs_count == 0).all()


def _point_at(zone: str, left: float, top: float, row: int, col: int) -> tuple[float, float]:
    """lon/lat of the centre of a group pixel."""
    lon, lat = Transformer.from_crs(zone_epsg(zone), 4326, always_xy=True).transform(
        left + (col + 0.5) * 10.0, top - (row + 0.5) * 10.0
    )
    return float(lon), float(lat)


#: A south group whose first 48 rows lie north of the equator: its top is 480 m north of it.
S_TOP = 10_000_480.0


def test_a_south_only_zone_reaching_north_of_the_equator(tmp_path):
    # With no 20N group, geotessera's merged grid starts at the equator, so the group's first 48 rows
    # lie above it: no group serves them. They must read as no data, not wrap to the zone's bottom.
    mem = _hemisphere_store({"20S": (500_000.0, S_TOP, SIZE, 64)})
    reader = TesseraReader.from_store(mem)
    assert reader.own_rows("20S") == (48, SIZE)
    cols = slice(0, 16)
    _assert_no_data(reader.read_window("20S", YEAR, slice(0, 32), cols))
    across = reader.read_window("20S", YEAR, slice(32, 96), cols)
    assert across.embeddings is not None and across.scales is not None
    _assert_no_data(dataclasses.replace(across, **{n: getattr(across, n)[:16] for n in ALL_LAYERS}))
    for name in ALL_LAYERS:
        got = getattr(across, name)[16:]
        np.testing.assert_array_equal(got, _source(mem, "20S", name, slice(48, 96), cols), err_msg=name)
    # A cached read of in-band rows fetches the chunk-aligned cover, rows 32 to 64, which starts
    # above the merged grid: the in-band rows are still the group's own.
    cached = TesseraReader.from_store(mem, cache=ChunkCache(tmp_path, namespace="t"))
    _assert_window_is(cached.read_window("20S", YEAR, slice(50, 60), cols), mem, "20S", slice(50, 60), cols)
    _assert_window_is(cached.read_window("20S", YEAR, slice(50, 60), cols), mem, "20S", slice(50, 60), cols)
    # A point just south of the equator, uncached and cached.
    lon, lat = _point_at("20S", 500_000.0, S_TOP, 50, 5)
    assert lat < 0
    want = (
        _source(mem, "20S", "embeddings", slice(50, 51), slice(5, 6))[0, 0]
        * _source(mem, "20S", "scales", slice(50, 51), slice(5, 6))[0, 0]
    )
    for r in (reader, cached):
        pts = read_points(r, [lon], [lat], YEAR)
        assert pts.valid.all()
        np.testing.assert_array_equal(pts.embeddings[0], want.astype(np.float32))


def test_a_window_past_the_bottom_of_geotesseras_view_is_refused(monkeypatch):
    # A north group reaching 32 rows south of the equator, in a zone with no south group: geotessera's
    # view ends at the equator. _check_served refuses those rows first; behind it, the read itself
    # refuses rather than return a window cut short.
    mem = _hemisphere_store({"20N": (500_000.0, 640.0, 96, 64)})
    reader = TesseraReader.from_store(mem)
    assert reader.own_rows("20N") == (0, 64)
    with pytest.raises(TesseraError, match="centred on or south of the equator"):
        reader.read_window("20N", YEAR, slice(48, 80), slice(0, 16))
    monkeypatch.setattr(TesseraReader, "_check_served", lambda self, zone, r1: None)
    with pytest.raises(TesseraError, match="fall outside geotessera's 64 x 64 view of zone 20"):
        reader.read_window("20N", YEAR, slice(48, 80), slice(0, 16))
    _assert_window_is(
        reader.read_window("20N", YEAR, slice(48, 64), slice(0, 16)), mem, "20N", slice(48, 64), slice(0, 16)
    )


def test_point_reads_near_the_equator_read_only_the_south_groups_own_rows():
    # 32S chunk row 1 (rows 32 to 64) straddles the equator at row 48, and geotessera serves its rows
    # 32 to 47 from 32N. A point in it lies at or below row 48, so reading the whole chunk would also
    # fetch 32N's bottom chunk for pixels no point uses.
    groups = {"32N": (500_000.0, 640.0, 64, 64), "32S": (500_000.0, S_TOP, SIZE, 64)}
    mem = _hemisphere_store(groups)
    plain = TesseraReader.from_store(mem)
    straddle = plain.read_window("32S", YEAR, slice(32, 64), slice(0, 16)).embeddings
    assert straddle is not None
    assert (straddle[:16, :, 0] == HEMI_MARKER["N"]).all() and (straddle[16:, :, 0] == HEMI_MARKER["S"]).all()
    log = KeyLog(mem)
    reader = TesseraReader.from_store(log)
    lon, lat = _point_at("32S", 500_000.0, S_TOP, 50, 5)
    pts = read_points(reader, [lon], [lat], YEAR)
    assert pts.zones == ["32S"] and pts.valid.all()
    want = _source(mem, "32S", "embeddings", slice(50, 51), slice(5, 6))[0, 0]
    np.testing.assert_array_equal(
        pts.embeddings[0, :3], want[:3] * _source(mem, "32S", "scales", slice(50, 51), slice(5, 6))[0, 0]
    )
    keys = log.chunk_keys()
    assert any(k.startswith("32S/embeddings/c/") for k in keys)
    assert [k for k in keys if k.startswith("32N/") and "/time/" not in k] == []


@pytest.mark.parametrize("east", ["N", "S"])
def test_column_offsets_come_from_each_groups_west_edge(east):
    # The group whose west edge is 64 pixels east sits 64 columns into geotessera's merged grid,
    # which starts at the westmost edge; a wrong sign or a zero offset reads other columns.
    left = {"N": 500_000.0, "S": 500_000.0}
    left[east] += 640.0
    mem = _hemisphere_store({"32N": (left["N"], 1280.0, SIZE, SIZE), "32S": (left["S"], 10_000_000.0, SIZE, SIZE)})
    reader = TesseraReader.from_store(mem)
    for zone in ("32N", "32S"):
        for rows, cols in ((slice(0, 32), slice(0, 32)), (slice(32, 64), slice(96, 128))):
            _assert_window_is(reader.read_window(zone, YEAR, rows, cols), mem, zone, rows, cols)


def test_embeddings_are_read_in_chunk_row_strips(monkeypatch):
    # The memory bound of the read path: each embeddings read asks geotessera for whole chunk rows of
    # at most _STRIP_BYTES at a time, so its transient arrays are a strip, not the window. Live that
    # is one chunk row of a 2048-column block; here the limit is set so that it is one chunk row too.
    # The other layers are read whole.
    import tessera_apps.io as tio

    assert tio._STRIP_BYTES == 256 * 2048 * 128  # one live chunk row of a DEFAULT_BLOCK
    monkeypatch.setattr(tio, "_STRIP_BYTES", 32 * 60 * 128)
    mem = _zone_store(2, MemoryStore)
    reader = TesseraReader.from_store(mem)
    keys: dict[str, list[Any]] = collections.defaultdict(list)
    outer = ZoneArray._outer

    def spy(self: ZoneArray, key: Any) -> np.ndarray:
        keys[self.name].append(key)
        return outer(self, key)

    monkeypatch.setattr(ZoneArray, "_outer", spy)
    rows, cols = slice(20, 100), slice(30, 90)
    win = reader.read_window("10N", YEAR, rows, cols)
    strips = [(k[2].start, k[2].stop) for k in keys["embeddings"]]
    assert strips == [(20, 32), (32, 64), (64, 96), (96, 100)]
    assert all(k[:2] == (T, slice(None)) and k[3] == cols for k in keys["embeddings"])
    assert keys["scales"] == [(T, rows, cols)]
    assert win.embeddings is not None and win.embeddings.flags.c_contiguous
    _assert_window_is(win, mem, "10N", rows, cols)


def test_embeddings_reads_in_flight_are_limited(monkeypatch):
    # The other half of the memory bound: TESSERA_APPS_EMBEDDING_READS embeddings reads at once,
    # while the small layers keep the pool busy. Set to 1 here, so any leak past it shows.
    import tessera_apps.io as tio

    monkeypatch.setattr(tio, "_EMBEDDING_READS", threading.BoundedSemaphore(1))
    reader = TesseraReader.from_store(_zone_store(2, MemoryStore))
    reader.grid("10N")
    lock = threading.Lock()
    now: collections.Counter[str] = collections.Counter()
    peak: collections.Counter[str] = collections.Counter()
    outer = ZoneArray._outer

    def spy(self: ZoneArray, key: Any) -> np.ndarray:
        kind = "embeddings" if self.name == "embeddings" else "other"
        with lock:
            now[kind] += 1
            now["all"] += 1
            peak[kind], peak["all"] = max(peak[kind], now[kind]), max(peak["all"], now["all"])
        try:
            time.sleep(0.02)
            return outer(self, key)
        finally:
            with lock:
                now[kind] -= 1
                now["all"] -= 1

    monkeypatch.setattr(ZoneArray, "_outer", spy)
    wins = reader.read_windows("10N", list(YEARS), slice(0, 64), slice(0, 64))
    assert peak["embeddings"] == 1 and peak["all"] > 2
    for year, t in ((2017, 0), (2025, 8)):
        np.testing.assert_array_equal(
            wins[year].embeddings,
            zarr.open_group(_zone_store(2, MemoryStore), mode="r")["10N"]["embeddings"][t, 0:64, 0:64],
        )


def test_strip_edges_follow_each_source_groups_chunk_rows():
    assert _strip_edges(20, 100, 0, 32) == [20, 32, 64, 96, 100]
    assert _strip_edges(64, 96, 0, 32) == [64, 96]
    # Above the first own row the edges step up from it: the north group's chunk-row edges when
    # it ends on the equator at a chunk edge, as the live groups do (own row 1472, 256 px chunks).
    assert _strip_edges(1000, 1600, 1472, 256) == [1000, 1216, 1472, 1536, 1600]
    assert _strip_edges(1300, 1536, 1472, 256) == [1300, 1472, 1536]
    assert _strip_edges(40, 40, 0, 32) == [40]
    # Strips of several chunk rows keep their edges on multiples of the strip height.
    assert _strip_edges(20, 200, 0, 64) == [20, 64, 128, 192, 200]


def test_a_group_without_years_complete_is_refused_before_any_chunk():
    # geotessera reads a group with no years_complete as complete in every year, so a publication
    # gap would read as a year of no data; the reader refuses the group instead.
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    fill = _random_fill(11)

    def gap_2023(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        data = fill(name, ti, shape)
        if YEARS[ti] == 2023:
            return np.full(shape, np.nan if name == "scales" else 0, dtype=data.dtype)
        return data

    g = add_zone(root, "10N", left=500_000.0, top=4_300_000.0, height=SIZE, width=SIZE, fill=gap_2023)
    del g.attrs["years_complete"]
    log = KeyLog(mem)
    reader = TesseraReader.from_store(log)
    for call in (
        lambda: reader.years_complete("10N"),
        lambda: reader.check_years(["10N"], [YEAR]),
        lambda: reader.read_window("10N", 2023, slice(0, 8), slice(0, 8)),
    ):
        with pytest.raises(TesseraError, match="Zone 10N has no years_complete attribute"):
            call()
    assert log.chunk_keys() == []


@pytest.mark.parametrize("declared", [[96, SIZE], [160, SIZE], [SIZE, 96]])
def test_a_spatial_shape_that_disagrees_with_the_arrays_is_refused(declared):
    # geotessera builds its merged view from spatial:shape, so a group declaring fewer rows than its
    # arrays hold would come back cut short, inside a Window whose rows say otherwise.
    mem = _zone_store(12, MemoryStore)
    zarr.open_group(mem, mode="r+")["10N"].attrs["spatial:shape"] = declared
    log = KeyLog(mem)
    reader = TesseraReader.from_store(log)
    for call in (lambda: reader.grid("10N"), lambda: reader.read_window("10N", YEAR, slice(90, 110), slice(0, 32))):
        with pytest.raises(TesseraError, match=rf"declares spatial:shape \[{declared[0]}, {declared[1]}\]"):
            call()
    assert log.chunk_keys() == []


def test_a_merged_view_in_another_axis_order_is_refused():
    # Without a dimension named 'band', geotessera keeps the source order (time, y, x, bands), and the
    # reader's (time, band, y, x) indexing would read a transposed block of the right shape.
    mem = _zone_store(13, MemoryStore)
    group = zarr.open_group(mem, mode="r+")["10N"]
    assert isinstance(group, zarr.Group)
    old = group["embeddings"]
    assert isinstance(old, zarr.Array)
    data, chunks, shards = old[:], old.chunks, old.shards
    del group["embeddings"]
    new = group.create_array(
        "embeddings",
        shape=data.shape,
        dtype=data.dtype,
        fill_value=0,
        chunks=chunks,
        shards=shards,
        dimension_names=("time", "northing", "easting", "bands"),
    )
    new[:] = data
    log = KeyLog(mem)
    reader = TesseraReader.from_store(log)
    with pytest.raises(TesseraError, match=r"has embeddings dims \('time', 'y', 'x', 'bands'\), expected"):
        reader.read_window("10N", YEAR, slice(0, 32), slice(0, 16))
    assert [k for k in log.chunk_keys() if "/time/" not in k] == []


# ---- cache namespaces and validation ------------------------------------------------------------


def test_unpinned_cache_is_refused_without_a_snapshot(tmp_path):
    store = _zone_store(3, MemoryStore)
    with pytest.raises(ValueError, match="namespace"):
        TesseraReader.from_store(store, cache=ChunkCache(tmp_path))
    with pytest.raises(ValueError, match="namespace"):
        TesseraReader(TesseraReader.from_store(store).store, cache=ChunkCache(tmp_path))
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


class FakeIcechunkError(Exception):
    """What the fake icechunk modules raise."""


MODEL = "https://geotessera.org/model/1.1"


def _with_root_attrs(store: MemoryStore, **attrs: Any) -> MemoryStore:
    zarr.open_group(store, mode="r+").attrs.update(attrs)
    return store


def _fake_icechunk(monkeypatch: pytest.MonkeyPatch, head: Any, pinned: Any = None) -> None:
    """An icechunk module whose repository serves ``head`` at the branch and ``pinned`` at any snapshot.

    geotessera's ``IcechunkStore`` opens sessions with ``icechunk.s3_storage``, ``Repository.open``
    and ``readonly_session``, so both ``GeoTesseraZarr`` and the reader's store open these.
    """

    class Repo:
        def readonly_session(self, branch=None, *, snapshot_id=None):
            if snapshot_id is not None:
                return types.SimpleNamespace(store=head if pinned is None else pinned, snapshot_id=snapshot_id)
            return types.SimpleNamespace(store=head, snapshot_id="SNAP123")

    fake = types.SimpleNamespace(
        s3_storage=lambda **kw: kw,
        Repository=types.SimpleNamespace(open=lambda storage: Repo()),
        IcechunkError=FakeIcechunkError,
        __version__="0.0-test",
    )
    monkeypatch.setitem(sys.modules, "icechunk", fake)


def test_open_namespaces_the_callers_cache_and_shares_its_stats(tmp_path, monkeypatch):
    store = _with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": MODEL, "geoemb:build_version": "0.1.0"})
    _fake_icechunk(monkeypatch, store)
    mine = ChunkCache(tmp_path)
    reader = TesseraReader.open(cache=mine)
    assert reader.cache is not None and reader.cache is not mine and reader.cache.namespace == "SNAP123"
    prov = reader.provenance()
    assert prov["snapshot_id"] == "SNAP123" and prov["branch"] == "main"
    # The identity GeoTesseraZarr gives, recorded for the report's store block.
    assert prov["backend"] == "geotessera"
    assert prov["url"] == "s3://tessera-embeddings/v1.1/dclimate.icechunk"
    assert prov["dataset"] == "1.1-dclimate"
    assert (prov["model_version"], prov["build_version"]) == (MODEL, "0.1.0")
    # ...and so in every report: geotessera's identity under store, its version under libraries.
    rec = provenance_record(reader, aoi_path=None, years=[YEAR], params={}, seed=None, zones=["10N"])
    assert {k: rec["store"][k] for k in ("url", "dataset", "model_version", "build_version", "snapshot_id")} == {
        "url": "s3://tessera-embeddings/v1.1/dclimate.icechunk",
        "dataset": "1.1-dclimate",
        "model_version": MODEL,
        "build_version": "0.1.0",
        "snapshot_id": "SNAP123",
    }
    assert rec["libraries"]["geotessera"] == geotessera.__version__ is not None
    rows, cols = slice(10, 50), slice(10, 50)
    first = reader.read_window("10N", YEAR, rows, cols)
    after_first = reader.bytes_read
    second = reader.read_window("10N", YEAR, rows, cols)
    assert after_first > 0 and reader.bytes_read == after_first
    assert mine.misses > 0 and mine.hits > 0
    assert (mine.hits, mine.misses) == (reader.cache.hits, reader.cache.misses)
    np.testing.assert_array_equal(second.embeddings, first.embeddings)
    assert [p.name for p in tmp_path.iterdir()] == ["SNAP123"]


def test_open_pins_the_snapshot_asked_for(monkeypatch):
    head = _with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": MODEL})
    pinned = _with_root_attrs(_zone_store(8, MemoryStore), **{"geoemb:model": MODEL})
    _fake_icechunk(monkeypatch, head, pinned)
    reader = TesseraReader.open(snapshot_id="PINNED")
    prov = reader.provenance()
    assert prov["snapshot_id"] == "PINNED" and prov["branch"] is None
    rows, cols = slice(0, 16), slice(0, 16)
    got = reader.read_window("10N", YEAR, rows, cols).embeddings
    np.testing.assert_array_equal(got, zarr.open_group(pinned, mode="r")["10N"]["embeddings"][T, rows, cols])


def test_the_store_block_records_the_build_of_the_snapshot_read(monkeypatch):
    # GeoTesseraZarr reads build_version at the head of main; a pinned snapshot of another build is
    # read and recorded as what it is, with geotessera's head value beside it.
    head = _with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": MODEL, "geoemb:build_version": "0.2.0"})
    pinned = _with_root_attrs(_zone_store(8, MemoryStore), **{"geoemb:model": MODEL, "geoemb:build_version": "0.1.0"})
    _fake_icechunk(monkeypatch, head, pinned)
    prov = TesseraReader.open(snapshot_id="OLD").provenance()
    assert (prov["build_version"], prov["head_build_version"], prov["geoemb:build_version"]) == (
        "0.1.0",
        "0.2.0",
        "0.1.0",
    )
    rec = provenance_record(
        TesseraReader.open(snapshot_id="OLD"), aoi_path=None, years=[YEAR], params={}, seed=None, zones=["10N"]
    )
    assert (rec["store"]["build_version"], rec["store"]["head_build_version"]) == ("0.1.0", "0.2.0")
    assert rec["model"]["geoemb:build_version"] == "0.1.0"
    # The head itself, or a snapshot of the same build: geotessera's value, and nothing beside it.
    same = TesseraReader.open().provenance()
    assert same["build_version"] == "0.2.0" and "head_build_version" not in same


def _cambridge() -> Any:
    from geotessera.registry import DATASETS

    return next(d for d in DATASETS if d.name == "1.1-cambridge")


@pytest.mark.parametrize("found", ["none", "1.1-cambridge"])
def test_open_refuses_a_store_geotessera_does_not_call_dclimate_v1_1(monkeypatch, found):
    log = KeyLog(_with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": MODEL}))
    _fake_icechunk(monkeypatch, log)
    dataset = None if found == "none" else _cambridge()
    monkeypatch.setattr("geotessera.store.dataset_for_location", lambda location: dataset)
    with pytest.raises(DatasetMismatchError) as info:
        TesseraReader.open()
    assert isinstance(info.value, TesseraError) and info.value.found == found
    assert f"as dataset {found}, not 1.1-dclimate" in str(info.value)
    # Refused before the reader read anything: no layer chunk, only what GeoTesseraZarr opens with.
    assert [k for k in log.chunk_keys() if "/time/" not in k] == []


def test_open_refuses_a_snapshot_of_another_model(monkeypatch):
    head = _with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": MODEL})
    pinned = _with_root_attrs(_zone_store(7, MemoryStore), **{"geoemb:model": "https://geotessera.org/model/2.0"})
    _fake_icechunk(monkeypatch, head, pinned)
    with pytest.raises(DatasetMismatchError) as info:
        TesseraReader.open(snapshot_id="OLD")
    assert "Snapshot OLD" in str(info.value) and "model/2.0" in str(info.value) and info.value.found == "1.1-dclimate"


def test_check_dataset_names_what_geotessera_found():
    gt = types.SimpleNamespace(
        dataset=types.SimpleNamespace(name="1.1-dclimate"),
        url="s3://tessera-embeddings/v1.1/dclimate.icechunk",
        model_version=MODEL,
        build_version="0.1.0",
    )
    assert check_dataset(gt) == {
        "url": "s3://tessera-embeddings/v1.1/dclimate.icechunk",
        "dataset": "1.1-dclimate",
        "model_version": MODEL,
        "build_version": "0.1.0",
    }
    with pytest.raises(DatasetMismatchError, match=r"as dataset none, not"):
        check_dataset(types.SimpleNamespace(**{**vars(gt), "dataset": None}))
    with pytest.raises(DatasetMismatchError, match=r"as dataset 2\.0-2B-L~beta1, not"):
        check_dataset(types.SimpleNamespace(**{**vars(gt), "dataset": types.SimpleNamespace(name="2.0-2B-L~beta1")}))


@pytest.mark.parametrize("snapshot", [None, "BOGUS"])
def test_open_names_the_store_when_icechunk_cannot_open_it(monkeypatch, snapshot):
    def refuse(storage):
        raise FakeIcechunkError(
            "object store error dispatch failure: CONNECT tunnel failed\n\ncontext:\n   0: icechunk::refs::fetch_branch"
        )

    fake = types.SimpleNamespace(
        s3_storage=lambda **kw: kw,
        Repository=types.SimpleNamespace(open=refuse),
        IcechunkError=FakeIcechunkError,
        __version__="0.0-test",
    )
    monkeypatch.setitem(sys.modules, "icechunk", fake)
    with pytest.raises(StoreUnavailableError) as info:
        TesseraReader.open(snapshot_id=snapshot)
    message = str(info.value)
    assert isinstance(info.value, TesseraError) and isinstance(info.value.__cause__, FakeIcechunkError)
    assert message.startswith("Cannot open the store s3://tessera-embeddings/v1.1/dclimate.icechunk at ")
    assert "dispatch failure: CONNECT tunnel failed. Check that this machine can reach S3 in us-west-2" in message
    assert "context" not in message and "icechunk::" not in message
    assert ("snapshot BOGUS" in message and "tessera-apps info" in message) == (snapshot is not None)
    assert ("branch 'main'" in message) == (snapshot is None)


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


def _time_store(values: np.ndarray | None, units: str | None = None) -> MemoryStore:
    """The seeded 10N group with its time coordinate replaced by ``values`` (or removed, for None)."""
    store = _zone_store(9, MemoryStore)
    group = zarr.open_group(store, mode="r+")["10N"]
    assert isinstance(group, zarr.Group)
    del group["time"]
    if values is not None:
        t = group.create_array("time", shape=values.shape, dtype=values.dtype, dimension_names=("time",))
        t[:] = values
        if units is not None:
            t.attrs["units"] = units
    return store


def test_cf_time_axis_is_decoded_by_geotessera_and_checked():
    ns = "nanoseconds since 1970-01-01"
    ok = TesseraReader.from_store(_time_store(_stamps(2017).astype(np.int64), ns))
    assert ok.store_years() == list(range(2017, 2026)) and ok.year_index("10N", 2025) == 8
    shifted = TesseraReader.from_store(_time_store(_stamps(2018).astype(np.int64), ns))
    with pytest.raises(TesseraError, match=r"years \[2018, .* layout has changed"):
        shifted.read_window("10N", 2020, slice(0, 8), slice(0, 8))
    # The right stamps under the wrong unit decode far outside 2017 to 2025.
    wrong_unit = TesseraReader.from_store(_time_store(_stamps(2017).astype(np.int64), "days since 1970-01-01"))
    with pytest.raises(TesseraError, match="layout has changed"):
        wrong_unit.years_complete("10N")


def test_datetime64_time_axis_is_decoded_by_geotessera_and_checked():
    assert TesseraReader.from_store(_time_store(_stamps(2017))).store_years() == list(range(2017, 2026))
    with pytest.raises(TesseraError, match=r"years \[2018, "):
        TesseraReader.from_store(_time_store(_stamps(2018))).store_years()


def test_a_zone_group_without_a_time_axis_is_refused():
    with pytest.raises(TesseraError, match="geotessera cannot open the copy"):
        TesseraReader.from_store(_time_store(None))
