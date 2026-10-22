"""Tests for tessera_apps.io against in-memory stores with the live store's layout. No network."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from pyproj import Geod, Transformer
from shapely.geometry import Point, box

from tessera_apps.cache import ChunkCache, ChunkKey
from tessera_apps.io import (
    AOIError,
    ByteCounter,
    CountingStore,
    ReadTooLargeError,
    TesseraError,
    TesseraReader,
    YearsNotPublishedError,
    ZoneGrid,
    canonicalize_zone,
    iter_blocks,
    iter_plan_blocks,
    iter_strips,
    plan_aoi,
    read_bbox,
    read_plan,
    read_points,
    read_polygon,
    zone_epsg,
    zone_for,
    zones_for_geometry,
)

from .conftest import BOUNDARY, PAD_COLS, PAD_ROWS

# ---- zones ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "zone"),
    [
        (-120.5, 38.7, "10N"),  # Caldor
        (-120.0, 38.85, "11N"),  # a band edge belongs to the band east of it
        (-120.0000001, 38.85, "10N"),
        (-63.9, -8.76, "20S"),  # Porto Velho
        (-81.3, 25.8, "17N"),  # Everglades
        (10.0, 0.0, "32N"),  # the equator is north
        (10.0, -0.0001, "32S"),
        (-180.0, 10.0, "01N"),
        (180.0, 10.0, "60N"),  # clamped
        (5.0, 60.0, "31N"),  # MGRS would say 32V: no exceptions here
        (10.0, 78.0, "32N"),  # MGRS would say 33X: no exceptions here
    ],
)
def test_zone_for(lon, lat, zone):
    assert zone_for(lon, lat) == zone


def test_zone_for_rejects_nan():
    with pytest.raises(ValueError):
        zone_for(float("nan"), 0.0)


def test_canonicalize_and_epsg():
    assert canonicalize_zone(" 7s ") == "07S"
    assert canonicalize_zone("33N") == "33N"
    assert zone_epsg("10N") == 32610
    assert zone_epsg("20S") == 32720
    for bad in ("utm15", "61N", "0N", "15", "15X"):
        with pytest.raises(ValueError):
            canonicalize_zone(bad)


def test_zones_for_geometry():
    assert zones_for_geometry(box(-120.1, 38.8, -119.9, 38.9)) == ["10N", "11N"]
    assert zones_for_geometry(box(-120.5, 38.6, -120.1, 38.9)) == ["10N"]
    # Touching a band edge without crossing it does not add the neighbour.
    assert zones_for_geometry(box(-121.0, 38.6, -120.0, 38.9)) == ["10N"]
    assert zones_for_geometry(box(10.0, -0.1, 10.1, 0.1)) == ["32N", "32S"]
    assert zones_for_geometry(Point(-120.0, 38.85)) == ["11N"]


# ---- grid and slice ordering ---------------------------------------------------------------


def _attrs(transform, code="EPSG:32610"):
    return {"spatial:transform": transform, "proj:code": code}


def test_grid_rows_run_north_to_south():
    g = ZoneGrid.from_attrs("10N", _attrs([10, 0, 163840, 0, -10, 9338880]), (933888, 67584))
    r_north, _ = g.rowcol(700000, 4300000)
    r_south, _ = g.rowcol(700000, 4200000)
    assert r_north < r_south
    assert g.rowcol(163840, 9338880) == (0, 0)
    assert g.rowcol(163845, 9338875) == (0, 0)
    rows, cols = g.window_for_bounds(700000, 4200000, 700100, 4200050)
    assert rows.stop - rows.start == 5 and cols.stop - cols.start == 10
    assert g.window_transform(rows.start, cols.start).f == 4200050


def test_grid_rejects_ascending_northing_and_wrong_crs():
    with pytest.raises(TesseraError, match="descending"):
        ZoneGrid.from_attrs("10N", _attrs([10, 0, 0, 0, 10, 0]), (10, 10))
    with pytest.raises(TesseraError, match="EPSG:32610"):
        ZoneGrid.from_attrs("10N", _attrs([10, 0, 0, 0, -10, 0], code="EPSG:32611"), (10, 10))


def test_window_matches_store_and_north_is_row_zero(caldor_reader, caldor_fixture):
    fx = caldor_fixture
    grid = caldor_reader.grid("10N")
    win = caldor_reader.read_window("10N", 2020, slice(PAD_ROWS, PAD_ROWS + 64), slice(PAD_COLS, PAD_COLS + 64))
    t = list(fx["years"]).index(2020)
    np.testing.assert_array_equal(win.embeddings, fx["embeddings"][t])
    np.testing.assert_array_equal(win.scales, fx["scales"][t])
    # The window transform puts row 0 at the fixture's north edge.
    assert win.transform.f == pytest.approx(fx["transform"][5])
    assert win.transform.e < 0
    assert grid.rowcol(fx["transform"][2] + 5, fx["transform"][5] - 5) == (PAD_ROWS, PAD_COLS)


def test_bbox_read_lines_up_with_the_lonlat_point(caldor_reader, caldor_fixture):
    fx = caldor_fixture
    a, _, c, _, _, f = fx["transform"]
    # Centre of fixture pixel (10, 20), in lon/lat.
    x, y = c + 20.5 * a, f - 10.5 * a
    lon, lat = Transformer.from_crs(32610, 4326, always_xy=True).transform(x, y)
    pts = read_points(caldor_reader, [lon], [lat], 2022)
    t = list(fx["years"]).index(2022)
    expect = fx["embeddings"][t, 10, 20].astype(np.float32) * fx["scales"][t, 10, 20]
    np.testing.assert_allclose(pts.embeddings[0], expect, rtol=1e-6)


# ---- NaN and validity ----------------------------------------------------------------------


def test_valid_excludes_nan_scales_and_zero_vectors(caldor_reader):
    win = caldor_reader.read_window("10N", 2020, slice(0, PAD_ROWS + 8), slice(0, PAD_COLS + 8))
    assert not win.valid[:PAD_ROWS].any()
    assert win.valid[PAD_ROWS:, PAD_COLS:].all()
    deq = win.dequantize()
    assert np.isnan(deq[:PAD_ROWS]).all()
    assert np.isfinite(deq[PAD_ROWS:, PAD_COLS:]).all()
    # A zero vector with a finite scale is still not valid.
    win.embeddings[PAD_ROWS, PAD_COLS] = 0
    assert not win.valid[PAD_ROWS, PAD_COLS]


def test_fixture_is_real_and_radar_complete(caldor_fixture):
    fx = caldor_fixture
    assert list(fx["years"]) == list(range(2017, 2026))
    assert np.isfinite(fx["scales"]).all()
    assert (fx["scales"] > 0).all()
    assert ((fx["s1_asc_obs_count"].astype(int) + fx["s1_desc_obs_count"]) > 0).all()


def test_radar_flags(boundary_reader):
    w21 = boundary_reader.read_window("10N", 2021, slice(10, 20), slice(10, 20))
    w22 = boundary_reader.read_window("10N", 2022, slice(10, 20), slice(10, 20))
    assert w21.radar_present.all() and not w21.radar_free.any()
    assert not w22.radar_present.any() and w22.radar_free.all()
    w = boundary_reader.read_window("10N", 2022, slice(10, 20), slice(10, 20), layers=["embeddings"])
    with pytest.raises(TesseraError, match="radar"):
        _ = w.radar_present


# ---- years ----------------------------------------------------------------------------------


def test_missing_year_fails_before_any_chunk_read(boundary_store, boundary_reader):
    _, log = boundary_store
    aoi = box(BOUNDARY[0] - 0.003, BOUNDARY[1] - 0.002, BOUNDARY[0] + 0.003, BOUNDARY[1] + 0.002)
    with pytest.raises(YearsNotPublishedError) as err:
        read_polygon(boundary_reader, aoi, 2023)
    assert err.value.missing == {"11N": [2023]}
    assert "11N: 2023" in str(err.value)
    assert log.chunk_keys() == []


@pytest.mark.parametrize("year", [2016, 2026])
def test_years_outside_store_range_are_refused(caldor_reader, year):
    with pytest.raises(YearsNotPublishedError):
        caldor_reader.read_window("10N", year, slice(0, 8), slice(0, 8))


def test_time_axis_decoding(caldor_reader):
    assert [caldor_reader.year_index("10N", y) for y in (2017, 2020, 2025)] == [0, 3, 8]
    # geotessera decodes the time coordinate (IcechunkStore.years); the reader checks it runs from 2017.
    assert caldor_reader.store_years() == list(range(2017, 2026))


def test_bad_window_and_layer(caldor_reader):
    with pytest.raises(TesseraError):
        caldor_reader.read_window("10N", 2020, slice(0, 10_000), slice(0, 8))
    with pytest.raises(TesseraError):
        caldor_reader.read_window("10N", 2020, slice(5, 5), slice(0, 8))
    with pytest.raises(TesseraError, match="Unknown layers"):
        caldor_reader.read_window("10N", 2020, slice(0, 8), slice(0, 8), layers=["ndvi"])
    with pytest.raises(TesseraError, match="not in the store"):
        caldor_reader.grid("11N")


# ---- strips and blocks ----------------------------------------------------------------------


@pytest.mark.parametrize(("start", "stop"), [(0, 2000), (300, 1500), (256, 257), (511, 513), (100, 200)])
def test_iter_strips_cover_disjoint_and_aligned(start, stop):
    strips = list(iter_strips(slice(start, stop), strip_rows=512, align=256))
    assert strips[0].start == start and strips[-1].stop == stop
    for a, b in itertools.pairwise(strips):
        assert a.stop == b.start
        assert a.stop % 256 == 0  # every inner edge is on a chunk boundary
    for s in strips:
        assert 0 < s.stop - s.start <= 512
        # A strip spans at most two chunk rows.
        assert (s.stop - 1) // 256 - s.start // 256 <= 1


def test_iter_strips_validates_args():
    with pytest.raises(ValueError):
        list(iter_strips(slice(0, 10), strip_rows=300, align=256))


def test_iter_blocks_tile_the_window():
    seen = np.zeros((1000, 3000), dtype=int)
    for rs, cs in iter_blocks(slice(100, 1100), slice(50, 3050), block=(512, 2048), align=(256, 256)):
        seen[rs.start - 100 : rs.stop - 100, cs.start - 50 : cs.stop - 50] += 1
        assert rs.stop - rs.start <= 512 and cs.stop - cs.start <= 2048
    assert (seen == 1).all()


def test_streamed_blocks_equal_one_piece_read(boundary_reader):
    aoi = box(BOUNDARY[0] - 0.008, BOUNDARY[1] - 0.006, BOUNDARY[0] - 0.0005, BOUNDARY[1] + 0.006)
    (p,) = plan_aoi(boundary_reader, aoi).zones
    whole = read_plan(boundary_reader, p, 2020)
    got = np.zeros_like(whole.embeddings)
    n_blocks = 0
    for blk in iter_plan_blocks(boundary_reader, p, [2020, 2021], block=(64, 64)):
        n_blocks += 1
        r, c = blk.rows.start - p.rows.start, blk.cols.start - p.cols.start
        got[r : r + blk.mask.shape[0], c : c + blk.mask.shape[1]] = blk.windows[2020].embeddings
        assert set(blk.windows) == {2020, 2021}
        assert blk.mask.any()
    assert n_blocks > 1
    np.testing.assert_array_equal(got[p.mask], whole.embeddings[p.mask])


# ---- AOI plans across zones -----------------------------------------------------------------


def test_multizone_plan_counts_each_pixel_once(boundary_reader, boundary_store):
    _, log = boundary_store
    aoi = box(BOUNDARY[0] - 0.003, BOUNDARY[1] - 0.002, BOUNDARY[0] + 0.003, BOUNDARY[1] + 0.002)
    plan = plan_aoi(boundary_reader, aoi)
    assert log.chunk_keys() == []  # planning reads metadata only
    assert plan.zone_names == ["10N", "11N"]
    area_m2 = abs(Geod(ellps="WGS84").geometry_area_perimeter(aoi)[0])
    assert plan.n_pixels * 100 == pytest.approx(area_m2, rel=0.02)
    for p in plan.zones:
        to_ll = Transformer.from_crs(p.grid.epsg, 4326, always_xy=True)
        rr, cc = np.nonzero(p.mask)
        xs = p.grid.left + (p.cols.start + cc + 0.5) * 10
        ys = p.grid.top - (p.rows.start + rr + 0.5) * 10
        lons, _ = to_ll.transform(xs, ys)
        if p.zone == "10N":
            assert (lons < -120.0).all()
        else:
            assert (lons >= -120.0).all()


def test_multizone_read_takes_each_pixel_from_its_nominal_zone(boundary_reader):
    aoi = box(BOUNDARY[0] - 0.003, BOUNDARY[1] - 0.002, BOUNDARY[0] + 0.003, BOUNDARY[1] + 0.002)
    wins = read_bbox(boundary_reader, aoi.bounds, 2020)
    assert [w.zone for w in wins] == ["10N", "11N"]
    for w in wins:
        marker = 100 if w.zone == "10N" else -100
        assert (w.embeddings[w.mask][:, 0] == marker).all()


def test_aoi_guards(boundary_reader):
    with pytest.raises(AOIError, match="limit is 1"):
        plan_aoi(boundary_reader, box(-120.003, 38.848, -119.997, 38.852), max_zones=1)
    with pytest.raises(AOIError, match="EPSG:4326"):
        plan_aoi(boundary_reader, box(500000, 4300000, 501000, 4301000))
    with pytest.raises(AOIError, match="no area"):
        plan_aoi(boundary_reader, Point(-120.0, 38.85))
    with pytest.raises(AOIError):
        read_bbox(boundary_reader, (-119.9, 38.8, -120.0, 38.9), 2020)
    aoi = box(BOUNDARY[0] - 0.003, BOUNDARY[1] - 0.002, BOUNDARY[0] - 0.0005, BOUNDARY[1] + 0.002)
    with pytest.raises(ReadTooLargeError, match="iter_plan_blocks"):
        read_polygon(boundary_reader, aoi, 2020, max_pixels=100)


def test_estimated_bytes_counts_touched_chunks(boundary_reader):
    aoi = box(BOUNDARY[0] - 0.003, BOUNDARY[1] - 0.002, BOUNDARY[0] - 0.0005, BOUNDARY[1] + 0.002)
    plan = plan_aoi(boundary_reader, aoi)
    (p,) = plan.zones
    n = p.chunks_touched((32, 32))
    rows = range(p.rows.start // 32, (p.rows.stop - 1) // 32 + 1)
    cols = range(p.cols.start // 32, (p.cols.stop - 1) // 32 + 1)
    assert 0 < n <= len(rows) * len(cols)
    assert plan.estimated_bytes(boundary_reader, n_years=2) == 2 * n * 32 * 32 * (128 + 4 + 6)


# ---- points ---------------------------------------------------------------------------------


def test_points_are_batched_by_chunk(boundary_reader):
    grid = boundary_reader.grid("10N")
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    # 40 points inside chunk (1, 1) of the mock grid (chunks are 32 x 32), and 10 inside chunk (2, 1).
    rng = np.random.default_rng(0)
    rows = np.concatenate([rng.integers(32, 64, 40), rng.integers(64, 96, 10)])
    cols = rng.integers(32, 64, 50)
    xs = grid.left + (cols + 0.5) * 10
    ys = grid.top - (rows + 0.5) * 10
    lons, lats = to_ll.transform(xs, ys)
    assert all(zone_for(lo, la) == "10N" for lo, la in zip(lons, lats, strict=True))
    pts = read_points(boundary_reader, lons, lats, 2020)
    assert pts.chunks_read == 2
    assert pts.valid.all()
    win = boundary_reader.read_window("10N", 2020, slice(0, 192), slice(0, 192))
    np.testing.assert_allclose(pts.embeddings, win.dequantize()[rows, cols], rtol=1e-6)
    assert pts.radar_present is not None and pts.radar_present.all()


def test_points_without_data_are_nan(boundary_reader):
    grid = boundary_reader.grid("10N")
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    lon_nan, lat_nan = to_ll.transform(grid.left + 15, grid.top - 15)  # inside the NaN corner
    pts = read_points(boundary_reader, [lon_nan, -121.9], [lat_nan, 38.85], 2020)  # second is off the mock grid
    assert not pts.valid.any()
    assert np.isnan(pts.embeddings).all()


def test_points_check_years_first(boundary_store, boundary_reader):
    _, log = boundary_store
    with pytest.raises(YearsNotPublishedError):
        read_points(boundary_reader, [-120.5, -119.999], [38.85, 38.85], 2023)
    assert log.chunk_keys() == []


# ---- bytes and cache ------------------------------------------------------------------------


def test_bytes_are_counted(caldor_reader):
    before = caldor_reader.bytes_read
    caldor_reader.read_window("10N", 2020, slice(0, 32), slice(0, 32))
    assert caldor_reader.bytes_read > before


def test_counting_store_copies_share_the_counter():
    from zarr.storage import MemoryStore

    counter = ByteCounter()
    s = CountingStore(MemoryStore(), counter)
    assert s.with_read_only(True).counter is counter


def test_cache_serves_second_read_without_store_bytes(caldor_store, tmp_path):
    _, logged = caldor_store
    cache = ChunkCache(tmp_path, namespace="test")
    reader = TesseraReader.from_store(logged, cache=cache)
    plain = TesseraReader.from_store(logged)
    rows, cols = slice(PAD_ROWS - 3, PAD_ROWS + 40), slice(PAD_COLS + 5, PAD_COLS + 70)
    first = reader.read_window("10N", 2021, rows, cols)
    after_first = reader.bytes_read
    second = reader.read_window("10N", 2021, rows, cols)
    assert reader.bytes_read == after_first
    assert cache.hits > 0
    ref = plain.read_window("10N", 2021, rows, cols)
    for w in (first, second):
        np.testing.assert_array_equal(w.embeddings, ref.embeddings)
        np.testing.assert_array_equal(w.scales, ref.scales)
        np.testing.assert_array_equal(w.s1_desc_obs_count, ref.s1_desc_obs_count)


def test_chunk_cache_roundtrip_and_namespaces(tmp_path):
    cache = ChunkCache(tmp_path, namespace="snapA")
    key = ChunkKey("10N", 2020, 3, 4)
    assert cache.get(key, "scales") is None
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.put(key, "scales", arr)
    np.testing.assert_array_equal(cache.get(key, "scales"), arr)
    assert cache.for_namespace("snapB").get(key, "scales") is None
    assert not list(tmp_path.rglob("*.tmp"))


def test_provenance_and_run_commits(caldor_reader):
    prov = caldor_reader.provenance()
    assert prov["checkpoint_id"] == "tessera_v1_1_aws_encoder.pt"
    assert caldor_reader.run_commits("10N", [2020]) == {2020: "abc2020"}
