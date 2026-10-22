# GeoTesseraZarr cannot pin an Icechunk snapshot, or count the bytes it reads from one

**Repository:** ucam-eo/geotessera
**Version:** 0.11.0 (line numbers below are from the 0.11.0 wheel)

## What we need

`tessera_apps` (https://github.com/rh281/tessera_apps) reads the dClimate v1.1 store, `s3://tessera-embeddings/v1.1/dclimate.icechunk`, through geotessera. For a run to be reproducible it has to read one named Icechunk snapshot, not whatever the head of `main` is that day: the head has moved at least three times since 2026-10-14 (`B6AFRJGX5XQC0A716PA0`, `RC6FS07ZB0T76EXT8XRG`, `4VS54EAWZK79M1WXVCTG`, and on 2026-10-21 `23XVEH2WHSRCS8237E8G`). It also has to report the bytes it pulled from the store, which means wrapping the zarr store geotessera reads through. `GeoTesseraZarr` supports neither for an Icechunk location.

## What happens in 0.11.0

1. **No snapshot or branch argument.** `GeoTesseraZarr.__init__(store_url=DEFAULT_STORE, cache_dir=None, cache_max_size=None, via_gateway=None)` (`store.py:940-946`) has no way to name a snapshot or a branch. For an Icechunk location it builds `IcechunkStore(store_url)` with no arguments (`store.py:954-957`), so it always opens the head of `main`, although `IcechunkStore.__init__(location, *, branch="main", snapshot_id=None)` (`icechunk.py:296`) and `open_session` (`icechunk.py:59-84`) already accept both.
2. **A store object takes the plain Zarr path.** `store_url` may be a `zarr.abc.store.Store` (`store.py:949`), but `is_icechunk_location` accepts only a `str` or path ending in `.icechunk` (`icechunk.py:45-49`). A `Store` that wraps an Icechunk session store (for example a byte-counting `zarr.storage.WrapperStore`) is therefore opened as a plain Zarr store (`store.py:961-967`): geotessera looks for `utmNN` groups, finds the repository's `10N`/`20S` groups instead, and `gt.dataset` is None because `dataset_for_location` returns None for anything that is not a `str` or path (`registry.py:708-711`).
3. **A pre-built `IcechunkStore` cannot be passed in.** It is neither a path nor a zarr `Store`, so `store_url=IcechunkStore(url, snapshot_id=...)` raises `TypeError` in `os.fspath` (`store.py:949-950`).
4. **The only handle is private.** The `IcechunkStore` a `GeoTesseraZarr` uses is `gt._icechunk` (`store.py:937`, `:957`), so a caller cannot reach its session, snapshot id or `root` without depending on a private attribute.

The result is that a caller who needs a pinned snapshot, or a byte count, has to bypass `GeoTesseraZarr` for its reads: we construct `geotessera.icechunk.IcechunkStore(url, snapshot_id=...)` ourselves, replace its public `root` with the same group opened through a counting `WrapperStore`, and read through its `zone_arrays()`. `GeoTesseraZarr` is then used only to check `gt.dataset.name == "1.1-dclimate"` and to record `gt.url`, `gt.model_version` and `gt.build_version`, which it reads from the head of `main`, not from the snapshot we read.

## Proposed change

- Add `snapshot_id: str | None = None` and `branch: str = "main"` to `GeoTesseraZarr.__init__` and to `store.open_zone`, passed through to `IcechunkStore` for Icechunk locations (and refused with a clear error for plain Zarr locations, where they mean nothing).
- Accept an `IcechunkStore` instance as `store_url`, using it as `self._icechunk` and keeping `gt.dataset` from its `url` (`dataset_for_location(store.url)`), so a caller can pass a store it has configured itself (snapshot, branch, a wrapped `root` for byte counting).
- Expose the snapshot actually read, for example `gt.snapshot_id` (from `session.snapshot_id`), so exports and reports can record the data version alongside `gt.dataset.name`.
- Optionally, accept a `store_wrapper: Callable[[zarr.abc.store.Store], zarr.abc.store.Store]` applied to the Icechunk session store before `zarr.open_group`, which covers byte counting and request logging without replacing `root`.

## A second, smaller point

Snapshots of the dClimate store before `B6AFRJGX5XQC0A716PA0` register `spatial:transform` at the pixel centre. `IcechunkStore._build` (`icechunk.py:356-371`) decides which hemisphere group serves a row from the row centre computed from that transform (`centres > 0` for N, `centres < 0` for S), so for those snapshots the north group's bottom row has a computed centre of exactly 0 and is served by neither group: it reads as fill. Taking the grid corner from `spatial:bbox` when it and the transform disagree by exactly half a pixel (which is what tessera_apps does for its own grids), or using `centres >= 0` for the north group, would keep that row.

## A third point: the CPU of a `ZoneArray` read

`ZoneArray._outer` (`icechunk.py:179-204`) allocates its output with `np.full` (`:182`) and copies each part's zarr block into it with `out[np.ix_(*where)] = np.transpose(block, self._order)` (`:202`). For embeddings that is a fancy-index assignment of a transposed block, since the source is `(time, northing, easting, band)` and the output `(time, band, y, x)`. On one 512 x 2048 x 128 int8 read from a local copy of a live-shaped group, the zarr read alone took 0.86 s (1.26 s CPU) and the `ZoneArray` read 2.29 s (2.63 s CPU); the `np.ix_` assignment by itself took 1.26 s, and a plain slice assignment of the same transposed block 0.62 s. A caller that wants band-last arrays (as `tessera_apps`' cosine paths do) then pays a second transpose, about 0.16 s. Across 1 to 9 years of every layer, reads through `ZoneArray` cost 1.7 to 2.7 times the CPU of direct zarr reads.

Two changes would remove most of it:

- When every axis of a part's selection is a contiguous run (which `_compact` already detects, `icechunk.py:136-140`), assign through slices instead of `np.ix_`.
- Accept an output order (for example `order="source"`, giving `(time, y, x, band)`) or an `out=` array, so a caller that wants the source layout skips both transposes and the `np.full` of an array it is about to overwrite.
