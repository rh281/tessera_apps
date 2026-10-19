"""Offline tests of the wetland app's parts, from seed loading to the validation against WorldCover.

Pure parts run on synthetic arrays: seed loading in any CRS, the spatial folds, the cosine and the
Ledoit-Wolf Mahalanobis scores against hand computations, the held-out threshold and its in-sample
fallback, gain and loss, the shoreline distance in metres, the seed exclusion buffer and the
false-positive class composition. Runs use designed in-memory stores, whose embeddings follow a
wetland map given per pixel, near the Everglades (zone 17N) and across the 120 W seam (10N and
11N), with a WorldCover tile written over the same ground. ``tests/test_apps_smoke.py`` runs the
whole app on the real caldor fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import zarr
from pyproj import Transformer
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from sklearn.covariance import EmpiricalCovariance, ledoit_wolf_shrinkage
from zarr.storage import MemoryStore

from tessera_apps import demo, export, io, metrics, validate
from tessera_apps.apps import wetland as wl
from tessera_apps.apps._common import AppInputError, StrictValidationError, mosaic_and_clean
from tessera_apps.io import FIRST_YEAR, N_BANDS, TesseraReader, Window, ZoneGrid, plan_aoi, read_points, zone_epsg

from .conftest import BOUNDARY, KeyLog, add_zone, write_worldcover_tiles

# --------------------------------------------------------------------------------------------
# Designed stores
# --------------------------------------------------------------------------------------------

_RNG = np.random.default_rng(1234)
PROTO_WET = _RNG.integers(-40, 41, size=N_BANDS)
PROTO_DRY = _RNG.integers(-40, 41, size=N_BANDS)
#: A vector per 32 x 32 pixel region added to wet pixels, so seeds from different places differ.
OFFSETS = _RNG.integers(-12, 13, size=(16, N_BANDS))

EVERGLADES = (-81.5, 25.9)
SIZE = 128


def _cell_aligned_corner(epsg: int, centre: tuple[float, float]) -> tuple[float, float]:
    """A grid corner whose 1.28 km window lies inside one 2 km point-fold cell, near ``centre``."""
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(*centre)
    cell = wl.POINT_FOLD_CELL_M
    return math.floor(x / cell) * cell + 360.0, math.floor(y / cell) * cell + 1640.0


def wetland_store(pattern, *, zones=("17N",), centre=EVERGLADES, size=SIZE, chunk=32, offsets=True, corner=None):
    """In-memory zone groups whose pixels follow ``pattern(lon, lat, rows, cols, year)``.

    The pattern returns ``(wet, data, radar)`` bool arrays, or ``(wet, data, radar, exact)`` where
    ``exact`` pixels carry no noise. A wet pixel is ``PROTO_WET`` plus its region's offset (when
    ``offsets``) plus noise, a dry one ``PROTO_DRY`` plus noise; scales vary per pixel so a mean of
    raw int8 vectors differs from the mean of the dequantized ones.

    Returns:
        ``(reader, log, mem)``.
    """
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    root.attrs.update({"checkpoint_id": "tessera_v1_1_aws_encoder.pt", "geoemb:dimensions": N_BANDS})
    for zone in zones:
        epsg = zone_epsg(zone)
        if corner is not None:
            left, top = corner
        else:
            x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(*centre)
            left, top = (x // 10) * 10 - size / 2 * 10, (y // 10) * 10 + size / 2 * 10
        rows, cols = np.mgrid[0:size, 0:size]
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
            left + (cols + 0.5) * 10.0, top - (rows + 0.5) * 10.0
        )
        noise = np.random.default_rng(int(zone[:2]))
        region = ((rows // 32) * 4 + cols // 32) % OFFSETS.shape[0]

        def fill(name, ti, shape, lon=lon, lat=lat, rows=rows, cols=cols, noise=noise, region=region):
            out = pattern(lon, lat, rows, cols, FIRST_YEAR + ti)
            wet, data, radar = out[:3]
            exact = out[3] if len(out) > 3 else np.zeros(wet.shape, dtype=bool)
            if name == "embeddings":
                wet_vec = PROTO_WET + (OFFSETS[region] if offsets else 0)
                base = np.where(wet[..., None], wet_vec, PROTO_DRY).astype(np.float64)
                jitter = noise.normal(0.0, 4.0, size=shape)
                e = np.where(exact[..., None], base, base + jitter)
                e = np.clip(np.round(e), -127, 127).astype(np.int8)
                e[~data] = 0
                return e
            if name == "scales":
                s = 0.01 * (1.0 + 0.05 * ((rows * 7 + cols * 3) % 11))
                return np.where(data, s, np.nan).astype(np.float32)
            if name == "s1_asc_obs_count":
                return np.where(radar, 30, 0).astype(np.uint16)
            if name == "s1_desc_obs_count":
                return np.zeros(shape, dtype=np.uint16)
            return np.full(shape, 50, dtype=np.uint16)

        add_zone(root, zone, left=left, top=top, height=size, width=size, fill=fill, chunk=chunk)
    log = KeyLog(mem)
    return TesseraReader.from_store(log), log, mem


def box_rc(rows, cols, r, c):
    return (rows >= r[0]) & (rows < r[1]) & (cols >= c[0]) & (cols < c[1])


STAND_A = ((8, 120), (8, 56))
STAND_B = ((16, 64), (72, 120))
LOSS = ((40, 64), (72, 120))  # stand B's south half, gone from 2021
GAIN = ((80, 112), (72, 120))  # new from 2021
RADAR_FREE_A = ((10, 14), (10, 20))  # radar-free in 2020 only: two rows of the first seed square
RADAR_FREE_B = ((96, 108), (16, 40))  # radar-free in 2021 and 2024


def wet_map(rows, cols, year):
    wet = box_rc(rows, cols, *STAND_A) | box_rc(rows, cols, *STAND_B)
    if year >= 2021:
        wet = (wet & ~box_rc(rows, cols, *LOSS)) | box_rc(rows, cols, *GAIN)
    return wet


def main_pattern(lon, lat, rows, cols, year):
    data = rows >= 3
    radar = np.ones(rows.shape, dtype=bool)
    if year == 2020:
        radar &= ~box_rc(rows, cols, *RADAR_FREE_A)
    if year in (2021, 2024):
        radar &= ~box_rc(rows, cols, *RADAR_FREE_B)
    return wet_map(rows, cols, year), data, radar


SEED_SQUARES = ((12, 12), (40, 24), (70, 36), (100, 44), (20, 80), (28, 104))
AOI_RC = ((4, 124), (4, 124))


@pytest.fixture(scope="module")
def main_mem():
    corner = _cell_aligned_corner(zone_epsg("17N"), EVERGLADES)
    _, _, mem = wetland_store(main_pattern, corner=corner)
    return mem


@pytest.fixture
def main_store(main_mem):
    log = KeyLog(main_mem)
    return TesseraReader.from_store(log), log


def grid_box_lonlat(grid: ZoneGrid, rows: tuple[int, int], cols: tuple[int, int]) -> Polygon:
    """A window of a zone grid as a lon/lat polygon (corners only: over a kilometre it is straight)."""
    square = box(
        grid.left + cols[0] * grid.pixel,
        grid.top - rows[1] * grid.pixel,
        grid.left + cols[1] * grid.pixel,
        grid.top - rows[0] * grid.pixel,
    )
    return shapely.transform(shapely.segmentize(square, 10.0), grid.to_lonlat().transform, interleaved=False)


def centre_lonlat(grid: ZoneGrid, r: float, c: float) -> tuple[float, float]:
    x, y = grid.to_lonlat().transform(grid.left + (c + 0.5) * grid.pixel, grid.top - (r + 0.5) * grid.pixel)
    return float(x), float(y)


def write_fc(path: Path, geometries, props=None, crs: str | None = None) -> Path:
    import geopandas as gpd

    frame = gpd.GeoDataFrame(props or {"i": list(range(len(geometries)))}, geometry=list(geometries), crs="EPSG:4326")
    if crs is not None:
        frame = frame.to_crs(crs)
    frame.to_file(path, driver="GeoJSON")
    return path


def seed_polygons(grid: ZoneGrid, squares=SEED_SQUARES, side=6) -> list[Polygon]:
    return [grid_box_lonlat(grid, (r, r + side), (c, c + side)) for r, c in squares]


@pytest.fixture
def main_files(main_store, tmp_path):
    reader, _ = main_store
    grid = reader.grid("17N")
    aoi = write_fc(tmp_path / "aoi.geojson", [grid_box_lonlat(grid, *AOI_RC)])
    seeds = write_fc(
        tmp_path / "seeds.geojson",
        seed_polygons(grid),
        {"source": ["test stands"] * len(SEED_SQUARES)},
    )
    return aoi, seeds, grid


# --------------------------------------------------------------------------------------------
# WorldCover over the designed 17N store
# --------------------------------------------------------------------------------------------


def wc_codes(rows, cols, year):
    """The designed WorldCover map on the 17N grid: mostly the wet map, with known disagreements."""
    inside = (rows >= 0) & (rows < SIZE) & (cols >= 0) & (cols < SIZE)
    code = np.where(wet_map(rows, cols, year), 95, 10).astype(np.uint8)
    code[box_rc(rows, cols, (0, SIZE), (60, 66))] = 90  # dry herbaceous stripe
    code[box_rc(rows, cols, (88, 96), (8, 30))] = 90  # part of stand A mapped as herbaceous wetland
    code[box_rc(rows, cols, (112, 120), (40, 56))] = 10  # part of stand A mapped as tree cover
    code[rows >= 122] = 80
    code[box_rc(rows, cols, (60, 70), (100, 110))] = 0  # a coverage hole
    if year == 2021:
        code[box_rc(rows, cols, (64, 72), (8, 56))] = 90  # a 2021-only disagreement
    return np.where(inside, code, np.uint8(0))


def write_worldcover(directory: Path, grid: ZoneGrid, years=(2020, 2021)) -> str:
    """WorldCover tile pieces over the designed 17N store window; returns the template."""
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    codes = {year: wc_codes(rows, cols, year) for year in years}
    return write_worldcover_tiles(directory, grid.epsg, grid.left, grid.top, codes, pixel=grid.pixel)


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    """Reference cache under tmp_path; no network: WorldCover from local tiles only, Natural Earth refused."""
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))

    def blocked(request, timeout):
        raise AssertionError(f"unexpected network access: {request.full_url}")

    monkeypatch.setattr(validate, "_urlopen", blocked)


# --------------------------------------------------------------------------------------------
# Seeds: loading
# --------------------------------------------------------------------------------------------


def test_load_seeds_takes_points_and_polygons_in_any_crs(tmp_path):
    square = box(-81.51, 25.89, -81.505, 25.895)
    pts = [Point(-81.52, 25.88), MultiPoint([(-81.53, 25.87), (-81.531, 25.871)])]
    line = LineString([(-81.5, 25.9), (-81.49, 25.91)])
    path = write_fc(tmp_path / "seeds_utm.geojson", [square, *pts, line], crs="EPSG:32617")
    seeds = wl.load_seeds(path)
    assert [f for f, _ in seeds.polygons] == [0]
    assert seeds.polygons[0][1].symmetric_difference(square).area < 1e-12
    assert [(f, round(x, 6), round(y, 6)) for f, x, y in seeds.points] == [
        (1, -81.52, 25.88),
        (2, -81.53, 25.87),
        (2, -81.531, 25.871),
    ]
    assert seeds.dropped_features == 1 and seeds.path == path and seeds.sha256 == export.file_sha256(path)
    assert seeds.demo is None and seeds.source == str(path)
    # The same seeds in lon/lat give the same geometry hash; moving one point changes it.
    same = wl.load_seeds(write_fc(tmp_path / "seeds_ll.geojson", [square, *pts, line]))
    assert same.geometry_sha256 == seeds.geometry_sha256
    moved = wl.load_seeds(write_fc(tmp_path / "moved.geojson", [square, Point(-81.52, 25.881), pts[1], line]))
    assert moved.geometry_sha256 != seeds.geometry_sha256


def test_load_seeds_keeps_role_seed_features_and_repairs_polygons(tmp_path):
    bowtie = Polygon([(-81.5, 25.9), (-81.49, 25.91), (-81.49, 25.9), (-81.5, 25.91)])
    path = write_fc(
        tmp_path / "roles.geojson",
        [box(-81.6, 25.8, -81.3, 26.0), bowtie, Point(-81.5, 25.9)],
        {"role": ["aoi", "seed", "seed"]},
    )
    seeds = wl.load_seeds(path)
    assert [f for f, _ in seeds.polygons] == [0] and seeds.polygons[0][1].is_valid
    assert seeds.polygons[0][1].area == pytest.approx(0.00005, rel=1e-6)
    assert [f for f, _, _ in seeds.points] == [1]


def test_load_seeds_refuses_files_without_points_or_polygons(tmp_path):
    lines = write_fc(tmp_path / "lines.geojson", [LineString([(-81.5, 25.9), (-81.4, 25.95)])])
    with pytest.raises(AppInputError, match="no point or polygon"):
        wl.load_seeds(lines)
    with pytest.raises(AppInputError, match="neither an existing file"):
        wl.load_seeds(tmp_path / "missing.geojson")
    bad = tmp_path / "broken.geojson"
    bad.write_text('{"type": "Featu', encoding="utf-8")
    with pytest.raises(AppInputError, match="Could not read the seeds file"):
        wl.load_seeds(bad)


def test_load_seeds_without_crs(tmp_path):
    lonlat = tmp_path / "lonlat.csv"
    lonlat.write_text("WKT,id\n" + '"POINT (-81.5 25.9)",1\n', encoding="utf-8")
    seeds = wl.load_seeds(lonlat)
    assert [(round(x, 6), round(y, 6)) for _, x, y in seeds.points] == [(-81.5, 25.9)]
    projected = tmp_path / "projected.csv"
    projected.write_text("WKT,id\n" + '"POINT (450000 2864000)",1\n', encoding="utf-8")
    with pytest.raises(AppInputError, match="declares no CRS"):
        wl.load_seeds(projected)


def test_bundled_seeds_are_the_default_for_the_everglades_demo():
    default = wl.load_seeds(None, aoi_demo="everglades")
    assert default.demo == "everglades_seeds" and default.source == "demo:everglades_seeds"
    assert len(default.polygons) == 12 and not default.points
    assert all("WorldCover" in s for s in default.sources)
    named = wl.load_seeds("everglades_seeds")
    assert (
        named.geometry_sha256 == default.geometry_sha256 == wl.load_seeds("demo", aoi_demo="everglades").geometry_sha256
    )
    assert named.sha256 == export.file_sha256(demo.demo_path("everglades_seeds"))
    with pytest.raises(AppInputError, match="No --seeds"):
        wl.load_seeds(None, aoi_demo="caldor")
    with pytest.raises(AppInputError, match="--seeds demo"):
        wl.load_seeds("demo", aoi_demo=None)


def test_bundled_everglades_seeds_and_demo_years():
    fc = demo.demo_seeds("everglades")
    assert demo.seeds_name("everglades") == "everglades_seeds" == demo.seeds_name("Everglades_Seeds")
    assert demo.demo_path("everglades_seeds").is_file() and demo.demo_path("everglades").is_file()
    feats = fc["features"]
    assert len(feats) == 12 and {f["properties"]["role"] for f in feats} == {"seed"}
    aoi = demo.demo_aoi("everglades")
    for f in feats:
        props = f["properties"]
        poly = shapely.geometry.shape(f["geometry"])
        assert aoi.contains(poly) and props["pixels"] == 100
        assert "WorldCover 2020 v100" in props["source"] and "class 95" in props["source"]
        assert "same product" in props["note"] and "Global Mangrove Watch" in props["note"]
        assert props["method"] and props["date"]
        # About 100 m on a side.
        utm = shapely.transform(poly, Transformer.from_crs(4326, 32617, always_xy=True).transform, interleaved=False)
        assert utm.area == pytest.approx(10_000.0, rel=1e-3)
    assert sum(f["properties"]["pixels"] for f in feats) == 1200
    props = demo.demo_properties("everglades")
    assert (props["year_a"], props["year_b"]) == (2020, 2024)
    assert "WorldCover" in props["years_note"] and "2018" in props["years_note"]
    with pytest.raises(ValueError, match="No bundled seeds"):
        demo.demo_seeds("rondonia")
    with pytest.raises(ValueError, match="Unknown bundled data"):
        demo.demo_path("atlantis")


# --------------------------------------------------------------------------------------------
# Seeds: placing on the store grid and the spatial folds
# --------------------------------------------------------------------------------------------


def test_place_seeds_samples_polygon_pixel_centres_and_point_pixels(main_store, tmp_path):
    reader, log = main_store
    grid = reader.grid("17N")
    square = grid_box_lonlat(grid, (20, 26), (30, 37))
    in_pixel = centre_lonlat(grid, 49.8, 60.2)  # inside pixel (50, 60), off its centre
    sliver = grid_box_lonlat(grid, (70.2, 70.4), (80.2, 80.4))
    far_x, far_y = Transformer.from_crs(4326, 32617, always_xy=True).transform(*in_pixel)
    far = Transformer.from_crs(32617, 4326, always_xy=True).transform(far_x + 2500.0, far_y)
    path = write_fc(
        tmp_path / "s.geojson",
        [square, Point(*in_pixel), sliver, Point(*in_pixel), Point(*far)],
    )
    table = wl.place_seeds(reader, wl.load_seeds(path))
    assert list(table.columns) == list(wl.PLACEMENT_COLUMNS)
    poly = table[table["group"] == "polygon 0"]
    plan = plan_aoi(reader, square, max_area_km2=None).zones[0]
    rr, cc = np.nonzero(plan.mask)
    assert len(poly) == 42 == plan.n_pixels
    assert sorted(zip(poly["row"], poly["col"], strict=True)) == sorted(
        zip(rr + plan.rows.start, cc + plan.cols.start, strict=True)
    )
    assert set(poly["row"]) == set(range(20, 26)) and set(poly["col"]) == set(range(30, 37))
    np.testing.assert_allclose(poly["lon"], poly["centre_lon"])
    # A point samples the pixel it falls in; the pixel centre is recorded beside the point.
    pt = table[(table["kind"] == "point") & (table["feature"] == 1)].iloc[0]
    assert (pt["row"], pt["col"]) == (50, 60) and (pt["lon"], pt["lat"]) == pytest.approx(in_pixel)
    assert (pt["centre_lon"], pt["centre_lat"]) == pytest.approx(centre_lonlat(grid, 50, 60))
    # A sliver with no pixel centre samples the pixel under it, in its polygon's group.
    sl = table[table["feature"] == 2]
    assert len(sl) == 1 and (sl.iloc[0]["row"], sl.iloc[0]["col"]) == (70, 80) and sl.iloc[0]["group"] == "polygon 2"
    # The same pixel twice is a duplicate; a point off the zone grid gets row -1 and its own 2 km cell.
    assert table["duplicate"].tolist() == [False] * 44 + [True, False]
    off = table.iloc[-1]
    assert off["row"] == -1 and off["col"] == -1 and math.isnan(off["centre_lon"])
    assert off["group"] != pt["group"] and pt["group"].startswith("cell 17N ")
    cx, cy = math.floor(far_x / 2000.0), math.floor(far_y / 2000.0)
    assert pt["group"] == f"cell 17N {cx} {cy}"
    assert not log.chunk_keys()


def test_point_seeds_group_by_two_km_cells(main_store, tmp_path):
    reader, _ = main_store
    to_ll = Transformer.from_crs(32617, 4326, always_xy=True)
    x0, y0 = 450_100.0, 2_864_100.0
    xy = [(x0, y0), (x0 + 900, y0 + 900), (x0 + 2000, y0), (x0, y0 - 2000), (x0 + 1850, y0 + 1850)]
    pts = [Point(*to_ll.transform(x, y)) for x, y in xy]
    table = wl.place_seeds(reader, wl.load_seeds(write_fc(tmp_path / "p.geojson", pts)))
    groups = table["group"].tolist()
    assert groups[0] == groups[1] == groups[4] == "cell 17N 225 1432"
    assert groups[2] == "cell 17N 226 1432" and groups[3] == "cell 17N 225 1431"


def test_place_seeds_refuses_oversized_polygons(main_store, tmp_path, monkeypatch):
    reader, _ = main_store
    big = box(-81.6, 25.8, -81.5, 25.9)  # about 110 km2
    with pytest.raises(AppInputError, match="Seed polygon 0"):
        wl.place_seeds(reader, wl.load_seeds(write_fc(tmp_path / "big.geojson", [big])))
    grid = reader.grid("17N")
    monkeypatch.setattr(wl, "MAX_SEED_PIXELS", 50)
    with pytest.raises(AppInputError, match="more than 50 pixels"):
        wl.place_seeds(reader, wl.load_seeds(write_fc(tmp_path / "s.geojson", seed_polygons(grid)[:2])))


def test_assign_folds_keeps_each_group_in_one_fold():
    groups = np.array([f"polygon {i % 7}" for i in range(140)])
    folds, k = wl.assign_folds(groups, 5, seed=0)
    assert k == 5 and set(folds.tolist()) == set(range(5))
    for g in np.unique(groups):
        assert np.unique(folds[groups == g]).size == 1
    other, _ = wl.assign_folds(groups, 5, seed=1)
    assert not np.array_equal(folds, other)
    again, _ = wl.assign_folds(groups, 5, seed=0)
    np.testing.assert_array_equal(folds, again)
    # More folds than groups: one fold per group. One group, or folds below 2: the fallback.
    three, k3 = wl.assign_folds(groups[groups < "polygon 3"], 5, seed=0)
    assert k3 == 3 and np.unique(three).size == 3
    one, k1 = wl.assign_folds(np.array(["polygon 0"] * 9), 5, seed=0)
    assert k1 == 1 and not one.any()
    _, kf = wl.assign_folds(groups, 1, seed=0)
    assert kf == 1


# --------------------------------------------------------------------------------------------
# Scores and the threshold
# --------------------------------------------------------------------------------------------


def window_of(emb: np.ndarray, scales: np.ndarray) -> Window:
    h, w = scales.shape
    grid = ZoneGrid(zone="17N", epsg=32617, left=0.0, top=0.0, height=h, width=w)
    return Window(zone="17N", year=2020, rows=slice(0, h), cols=slice(0, w), grid=grid, scales=scales, embeddings=emb)


def test_cosine_score_is_one_minus_cosine_to_the_dequantized_mean():
    rng = np.random.default_rng(0)
    emb = rng.integers(-60, 61, size=(30, N_BANDS)).astype(np.int8)
    scales = rng.uniform(0.005, 0.05, size=30).astype(np.float32)
    x = emb.astype(np.float64) * scales[:, None]
    model = wl.fit_seed_model(x, "cosine")
    mean = x.mean(axis=0)
    np.testing.assert_allclose(model.mean, mean)
    assert not np.allclose(mean / np.linalg.norm(mean), emb.mean(axis=0) / np.linalg.norm(emb.mean(axis=0)))
    y = rng.normal(size=(12, N_BANDS))
    hand = 1.0 - (y @ mean) / (np.linalg.norm(y, axis=1) * np.linalg.norm(mean))
    np.testing.assert_allclose(model.score_vectors(y), hand, rtol=1e-5, atol=1e-6)
    # On a window of int8 embeddings the per-pixel scale cancels.
    pix = rng.integers(-60, 61, size=(4, 5, N_BANDS)).astype(np.int8)
    sc = rng.uniform(0.005, 0.05, size=(4, 5)).astype(np.float32)
    sc[0, 0] = np.nan
    got = model.score_window(window_of(pix, sc))
    deq = pix.reshape(-1, N_BANDS).astype(np.float64)
    want = 1.0 - (deq @ mean) / (np.linalg.norm(deq, axis=1) * np.linalg.norm(mean))
    np.testing.assert_allclose(got.reshape(-1)[1:], want[1:], rtol=1e-5, atol=1e-6)
    assert np.isnan(got[0, 0]) and got.dtype == np.float32
    assert np.isnan(model.score_vectors(np.zeros((1, N_BANDS)))).all()


def test_mahalanobis_score_uses_the_ledoit_wolf_shrunk_covariance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, N_BANDS)) @ rng.normal(size=(N_BANDS, N_BANDS)) * 0.1 + rng.normal(size=N_BANDS)
    model = wl.fit_seed_model(x, "mahalanobis")
    mean = x.mean(axis=0)
    cov = np.cov(x, rowvar=False, bias=True)
    alpha = ledoit_wolf_shrinkage(x)
    mu = np.trace(cov) / N_BANDS
    shrunk = (1.0 - alpha) * cov + alpha * mu * np.eye(N_BANDS)
    precision = np.linalg.inv(shrunk)
    y = rng.normal(size=(20, N_BANDS)) + mean
    d = y - mean
    hand = np.sqrt(np.einsum("ij,jk,ik->i", d, precision, d))
    assert 0.0 < alpha < 1.0 and model.shrinkage == pytest.approx(alpha)
    np.testing.assert_allclose(model.mean, mean)
    np.testing.assert_allclose(model.score_vectors(y), hand, rtol=1e-6)
    # Without the shrinkage the distances are not the same (the sample covariance is invertible here).
    raw = EmpiricalCovariance().fit(x)
    unshrunk = np.sqrt(raw.mahalanobis(y))
    assert np.abs(unshrunk - hand).max() > 0.05 * hand.mean()
    # On a window, the vectors are dequantized first.
    pix = np.clip(np.round(y[:6].reshape(2, 3, N_BANDS) / 0.02), -127, 127).astype(np.int8)
    sc = np.full((2, 3), 0.02, dtype=np.float32)
    got = model.score_window(window_of(pix, sc))
    np.testing.assert_allclose(got.reshape(-1), model.score_vectors(pix.reshape(6, -1) * 0.02), rtol=1e-5)
    masked = model.score_window(window_of(pix, sc), np.array([[True, False, True], [True, True, False]]))
    assert np.isnan(masked[0, 1]) and np.isnan(masked[1, 2]) and np.isfinite(masked[0, 0])
    with pytest.raises(ValueError, match="at least 2"):
        wl.fit_seed_model(x[:1], "mahalanobis")
    with pytest.raises(ValueError, match="must be one of"):
        wl.fit_seed_model(x, "euclid")


def test_heldout_scores_fit_each_fold_on_the_others():
    rng = np.random.default_rng(2)
    folds = np.repeat(np.arange(4), 10)
    centres = rng.normal(size=(4, N_BANDS)) * 3.0 + 10.0
    x = centres[folds] + rng.normal(size=(40, N_BANDS))
    held = wl.heldout_scores(x, folds, "cosine")
    for k in range(4):
        mean = x[folds != k].mean(axis=0)
        xs = x[folds == k]
        hand = 1.0 - (xs @ mean) / (np.linalg.norm(xs, axis=1) * np.linalg.norm(mean))
        np.testing.assert_allclose(held[folds == k], hand, rtol=1e-5, atol=1e-7)
    in_sample = wl.fit_seed_model(x, "cosine").score_vectors(x)
    assert np.median(held) > 1.2 * np.median(in_sample)
    assert np.isnan(wl.heldout_scores(x, np.zeros(40), "cosine")).all()


def test_choose_threshold_uses_the_pooled_heldout_percentile():
    rng = np.random.default_rng(3)
    ins = rng.uniform(0.0, 0.1, size=200)
    held = rng.uniform(0.0, 0.3, size=200)
    held[:5] = np.nan
    c = wl.choose_threshold(ins, held, 95.0, 5)
    assert c.method == "heldout" and c.threshold == pytest.approx(np.percentile(held[5:], 95))
    assert c.in_sample_percentile == pytest.approx(np.percentile(ins, 95)) and c.threshold > c.in_sample_percentile
    assert c.heldout_pixels == 195 and c.folds == 5 and c.fallback_reason is None
    assert c.record()["fallback_ran"] is False
    fb = wl.choose_threshold(ins, held, 90.0, 1)
    assert fb.method == "in_sample" and fb.threshold == pytest.approx(np.percentile(ins, 90))
    assert "fewer than two" in fb.fallback_reason and fb.record()["fallback_ran"] is True
    none = wl.choose_threshold(ins, np.full(200, np.nan), 95.0, 3)
    assert none.method == "in_sample" and "held out" in none.fallback_reason


# --------------------------------------------------------------------------------------------
# Extent, change, shoreline, exclusion
# --------------------------------------------------------------------------------------------


def test_change_map_codes_gain_loss_stable_and_unclassified():
    a = np.array([[1, 1, 0, 0, 1]], dtype=bool)
    b = np.array([[1, 0, 1, 0, 1]], dtype=bool)
    cl = np.array([[1, 1, 1, 1, 0]], dtype=bool)
    got = wl.change_map(a, b, cl)
    np.testing.assert_array_equal(got[0, :4], [0.0, -1.0, 1.0, 0.0])
    assert np.isnan(got[0, 4]) and got.dtype == np.float32
    score = np.array([0.1, 0.2, np.nan, 0.3], dtype=np.float32)
    assert wl.candidate_extent(score, 0.2, np.array([True, True, True, False])).tolist() == [True, True, False, False]


def test_shoreline_distance_is_in_metres(main_store):
    reader, _ = main_store
    grid = reader.grid("17N")
    plan = plan_aoi(reader, grid_box_lonlat(grid, (20, 60), (20, 60))).zones[0]
    # A north-south line along the centres of zone column 30.
    x = grid.left + 30.5 * grid.pixel
    to_ll = grid.to_lonlat()
    line = LineString([to_ll.transform(x, grid.top - 0.0), to_ll.transform(x, grid.top - SIZE * grid.pixel)])
    dist = wl.shoreline_distance_m(plan, [line], 200.0)
    cols = np.arange(plan.cols.start, plan.cols.stop)
    expected = np.abs(cols - 30) * grid.pixel
    np.testing.assert_allclose(dist, np.broadcast_to(expected, dist.shape), atol=1e-4)
    # The coastal fraction of an extent at known distances: within 50 m is 5 pixels either side.
    extent = np.zeros(plan.shape, dtype=bool)
    extent[:, (cols >= 22) & (cols < 42)] = True
    near = dist <= 50.0
    frac = wl.coastal_fraction({"17N": extent}, {"17N": near})
    assert frac == pytest.approx(11 / 20)
    assert wl.coastal_fraction({"17N": np.zeros(plan.shape, bool)}, {"17N": near}) is None
    # A line beyond the padded window is no shoreline: every distance is inf.
    far = LineString(
        [to_ll.transform(grid.left - 5000.0, grid.top), to_ll.transform(grid.left - 5000.0, grid.top - 1000)]
    )
    assert np.isinf(wl.shoreline_distance_m(plan, [far], 200.0)).all()
    assert np.isinf(wl.shoreline_distance_m(plan, [], 200.0)).all()


def test_read_shoreline_takes_lines_and_polygon_outlines(tmp_path):
    sq = box(-81.5, 25.9, -81.49, 25.91)
    path = write_fc(tmp_path / "shore.geojson", [sq, LineString([(-81.6, 25.8), (-81.55, 25.85)]), Point(0, 0)])
    lines = wl.read_shoreline(path)
    assert len(lines) == 2 and all(g.geom_type == "LineString" for g in lines)
    assert lines[0].equals(LineString(sq.exterior.coords))
    with pytest.raises(AppInputError, match="no line"):
        wl.read_shoreline(write_fc(tmp_path / "pts.geojson", [Point(0, 0)]))


def test_seed_exclusion_buffers_seed_pixel_centres_in_metres(main_store):
    reader, _ = main_store
    grid = reader.grid("17N")
    plan = plan_aoi(reader, grid_box_lonlat(grid, (20, 60), (20, 60))).zones[0]
    seeds = np.array([centre_lonlat(grid, 30, 30), centre_lonlat(grid, 18, 40)])  # the second outside the AOI
    rows, cols = np.mgrid[plan.rows, plan.cols]
    for buffer_m in (0.0, 30.0, 45.0):
        got = wl.seed_exclusion(plan, seeds, buffer_m)
        d = np.minimum(np.hypot(rows - 30, cols - 30), np.hypot(rows - 18, cols - 40)) * grid.pixel
        expected = (d <= buffer_m + 1e-9) & plan.mask
        np.testing.assert_array_equal(got, expected)
    assert wl.seed_exclusion(plan, seeds, 0.0).sum() == 1
    # The seed outside the AOI (row 18) still excludes the AOI pixels within 30 m of it.
    reached = wl.seed_exclusion(plan, seeds[1:], 30.0)
    assert reached[20 - plan.rows.start, 40 - plan.cols.start] and reached[21 - plan.rows.start, 40 - plan.cols.start]
    assert not reached[22 - plan.rows.start, 40 - plan.cols.start] and reached.sum() > 2
    assert not wl.seed_exclusion(plan, np.empty((0, 2)), 200.0).any()


def test_false_positive_classes_count_worldcover_codes(main_store):
    reader, _ = main_store
    grid = reader.grid("17N")
    plan = plan_aoi(reader, grid_box_lonlat(grid, (0, 4), (0, 4)))
    z = plan.zones[0]
    extent = np.zeros(z.shape, bool)
    extent[:, :3] = True
    ref = np.zeros(z.shape, bool)
    ref[:, 0] = True
    codes = np.zeros(z.shape, np.uint8)
    codes[:, 1] = 90
    codes[:2, 2] = 10
    scored = np.ones(z.shape, bool)
    scored[3, :] = False
    got = wl.false_positive_classes(plan, {z.zone: extent}, {z.zone: ref}, {z.zone: scored & z.mask}, {z.zone: codes})
    assert got == {
        "0": {"name": "no data", "pixels": 1, "share": 1 / 6},
        "10": {"name": "tree cover", "pixels": 2, "share": 2 / 6},
        "90": {"name": "herbaceous wetland", "pixels": 3, "share": 3 / 6},
    }


def test_storm_note_for_south_florida_only():
    aoi = demo.demo_aoi("everglades")
    note = wl.storm_note(aoi, 2020, 2024)
    assert "Irma" in note and "10,760 ha" in note and "Ian" in note and "spans Ian" in note
    assert "spans Ian" not in wl.storm_note(aoi, 2020, 2021)
    assert "spans Irma" in wl.storm_note(aoi, 2017, 2019)
    assert wl.storm_note(demo.demo_aoi("caldor"), 2020, 2024) is None


# --------------------------------------------------------------------------------------------
# Runs on the designed 17N store
# --------------------------------------------------------------------------------------------


def run_main(main_store, main_files, tmp_path, year_b=2024, **kw):
    reader, _ = main_store
    aoi, seeds, _ = main_files
    kw.setdefault("seeds", seeds)
    kw.setdefault("min_area_ha", 0.05)
    kw.setdefault("block", (32, 32))
    kw.setdefault("shoreline", None)
    return wl.run_wetland(aoi, 2020, year_b, kw.pop("out", tmp_path / "out"), reader=reader, **kw)


@pytest.fixture
def no_coast(monkeypatch):
    def refused(*args, **kwargs):
        raise validate.ReferenceUnavailableError("naturalearth.s3.amazonaws.com refused the request (403)")

    monkeypatch.setattr(validate, "natural_earth_coastline", refused)


def test_threshold_extent_change_and_radar_guard_on_the_designed_store(main_store, main_files, tmp_path, no_coast):
    reader, log = main_store
    res = run_main(main_store, main_files, tmp_path)
    s = res.summary
    plan = plan_aoi(reader, shapely.geometry.shape(json.loads(main_files[0].read_text())["features"][0]["geometry"]))
    (zp,) = plan.zones
    seeds = pd.read_parquet(res.extra_paths["seeds"])
    assert list(seeds.columns) == list(wl.SEED_COLUMNS)
    assert len(seeds) == 216 and (seeds["kind"] == "polygon").all()

    # Seeds: the two rows of the first square that are radar-free in 2020 are dropped (survey B5).
    dropped = seeds[~seeds["used"]]
    assert len(dropped) == 12 and (dropped["drop_reason"] == "regime").all() and (dropped["regime"] == "optical").all()
    assert set(dropped["row"]) == {12, 13} and set(dropped["col"]) == set(range(12, 18))
    assert set(dropped["group"]) == {"polygon 0"}
    used = seeds[seeds["used"]]
    assert s["seeds"]["used"] == 204 and s["seeds"]["dropped_regime"] == 12 and s["seeds"]["groups"] == 6
    for g, part in used.groupby("group"):
        assert part["fold"].nunique() == 1, g
    assert s["threshold"]["folds"] == 5 and s["seeds"]["folds"] == 5

    # The reference and the held-out scores, from an independent read of the seed pixels.
    pts = read_points(TesseraReader.from_store(main_store[1]._store), used["lon"], used["lat"], 2020)
    x = pts.embeddings.astype(np.float64)
    mean = x.mean(axis=0)
    hand_in = 1.0 - (x @ mean) / (np.linalg.norm(x, axis=1) * np.linalg.norm(mean))
    np.testing.assert_allclose(used["score_in_sample"], hand_in, rtol=1e-5, atol=1e-6)
    folds = used["fold"].to_numpy()
    hand_held = np.empty(len(used))
    for k in np.unique(folds):
        m = x[folds != k].mean(axis=0)
        xs = x[folds == k]
        hand_held[folds == k] = 1.0 - (xs @ m) / (np.linalg.norm(xs, axis=1) * np.linalg.norm(m))
    np.testing.assert_allclose(used["score_heldout"], hand_held, rtol=1e-5, atol=1e-6)
    thr = s["threshold"]["threshold"]
    assert s["threshold"]["method"] == "heldout"
    assert thr == pytest.approx(np.percentile(hand_held, 95))
    assert s["threshold"]["in_sample_percentile"] == pytest.approx(np.percentile(hand_in, 95))
    assert thr > 1.2 * s["threshold"]["in_sample_percentile"]
    assert s["seeds"]["share_heldout_within_threshold"] == pytest.approx(np.mean(hand_held <= thr))

    # The bands: scores where valid in the seeds' regime, extent = clean(score <= thr) over classified.
    with rasterio.open(res.paths.tif) as ds:
        assert ds.descriptions == ("score_2020", "score_2024", "extent_2020", "extent_2024", "change")
        score_a, score_b, ext_a, ext_b, change = ds.read()
    classified = np.isfinite(change)
    rows, cols = np.mgrid[zp.rows, zp.cols]
    mismatch = box_rc(rows, cols, *RADAR_FREE_B) | box_rc(rows, cols, *RADAR_FREE_A)
    no_data = rows < 3
    assert not classified[mismatch].any() and not classified[no_data].any()
    assert classified[zp.mask & ~mismatch & ~no_data].all()
    assert np.isnan(score_b[box_rc(rows, cols, *RADAR_FREE_B)]).all()
    assert np.isfinite(score_a[box_rc(rows, cols, *RADAR_FREE_B) & zp.mask]).all()
    assert s["radar"]["regime_mismatch_pixels"] == int((mismatch & zp.mask & ~no_data).sum())
    assert s["classified_pixels"] == int(classified.sum())
    min_px = metrics.area_to_pixels(0.05)
    for score, ext in ((score_a, ext_a), (score_b, ext_b)):
        raw = classified & (score <= thr)
        np.testing.assert_array_equal(ext == 1, metrics.clean_mask(raw, min_pixels=min_px) & zp.mask)
        assert set(np.unique(ext[classified]).tolist()) == {0.0, 1.0} and np.isnan(ext[~classified]).all()
    gain = (ext_b == 1) & (ext_a == 0)
    loss = (ext_a == 1) & (ext_b == 0)
    np.testing.assert_array_equal(change == 1, gain)
    np.testing.assert_array_equal(change == -1, loss)
    assert s["gain_pixels"] == int(gain.sum()) and s["loss_pixels"] == int(loss.sum())
    assert s["gain_ha"] == pytest.approx(gain.sum() * 0.01) and s["loss_ha"] == pytest.approx(loss.sum() * 0.01)
    n_a, n_b = int((ext_a == 1).sum()), int((ext_b == 1).sum())
    assert s["net_change_ha"] == pytest.approx((n_b - n_a) * 0.01)
    assert s["net_change_pct"] == pytest.approx(100.0 * (n_b - n_a) / n_a)
    # The designed change shows up where it was put: gain in GAIN, loss in LOSS, wet extent stable.
    in_gain = box_rc(rows, cols, *GAIN) & classified
    in_loss = box_rc(rows, cols, *LOSS) & classified
    assert gain[in_gain].mean() > 0.6 and loss[in_loss].mean() > 0.6
    # Elsewhere only threshold noise on land wet in both years flips, and rarely: never on dry land.
    both_wet = wet_map(rows, cols, 2020) & wet_map(rows, cols, 2024) & classified
    for flips, region in ((gain, GAIN), (loss, LOSS)):
        outside = flips & ~box_rc(rows, cols, *region)
        assert not (outside & ~both_wet).any() and outside.sum() < 0.1 * both_wet.sum()
    wet_a = wet_map(rows, cols, 2020) & classified
    assert (ext_a == 1)[wet_a].mean() > 0.6 and not (ext_a == 1)[~wet_map(rows, cols, 2020)].any()

    # One summary row per year with the change columns repeated, the same in parquet and csv.
    table = pd.read_parquet(res.paths.summary_parquet)
    pd.testing.assert_frame_equal(table, pd.read_csv(res.paths.summary_csv), check_dtype=False)
    assert table["year"].tolist() == [2020, 2024]
    assert table["extent_ha"].tolist() == pytest.approx([n_a * 0.01, n_b * 0.01])
    for col in ("net_change_ha", "gain_ha", "loss_ha", "threshold"):
        assert table[col].nunique() == 1 and table[col].iloc[0] == pytest.approx(s[col] if col != "threshold" else thr)
    assert table["coastal_fraction"].isna().all()
    assert any("Natural Earth coastline could not be loaded" in w for w in res.report["warnings"])
    assert any("switch radar regime" in w for w in res.report["warnings"])
    assert any("seed pixels were dropped" in w for w in res.report["warnings"])
    assert s["storm_note"] and "Irma" in s["storm_note"]
    assert s["score"]["type"] == "cosine" and s["score"]["shrinkage"] is None
    assert res.report["estimated_memory"] == wl.memory_estimate(plan, block=(32, 32), coast_km=1.0)
    assert 0 < res.report["bytes_read"] <= res.report["estimated_bytes"]
    assert res.report["provenance"]["years"] == [2020, 2024]

    # Rerunning with the same arguments skips without reading a chunk; another fold seed reruns.
    before = len(log.chunk_keys())
    again = run_main(main_store, main_files, tmp_path)
    assert again.skipped and again.report == res.report and len(log.chunk_keys()) == before
    other = run_main(main_store, main_files, tmp_path, seed=3)
    assert not other.skipped and other.report["args_digest"] != res.report["args_digest"]


def test_seeds_outside_the_aoi_are_read_and_used(main_store, main_files, tmp_path, no_coast):
    reader, _ = main_store
    grid = reader.grid("17N")
    aoi = write_fc(tmp_path / "small_aoi.geojson", [grid_box_lonlat(grid, (60, 120), (4, 60))])
    res = wl.run_wetland(
        aoi, 2020, 2024, tmp_path / "out", seeds=main_files[1], reader=reader, block=(32, 32), min_area_ha=0.05
    )
    seeds = pd.read_parquet(res.extra_paths["seeds"])
    outside = ~seeds["inside_aoi"]
    assert outside.sum() == 4 * 36 and seeds.loc[outside & (seeds["group"] != "polygon 0"), "used"].all()
    assert res.summary["seeds"]["inside_aoi"] == 72


def test_point_seeds_in_one_cell_fall_back_to_the_in_sample_threshold(main_store, main_files, tmp_path, no_coast):
    reader, _ = main_store
    grid = reader.grid("17N")
    pts = [Point(*centre_lonlat(grid, r, c)) for r in range(20, 100, 8) for c in (20, 40)]
    seeds = write_fc(tmp_path / "points.geojson", pts)
    res = run_main(main_store, main_files, tmp_path, seeds=seeds)
    s = res.summary
    table = pd.read_parquet(res.extra_paths["seeds"])
    assert (table["kind"] == "point").all() and table["group"].nunique() == 1
    assert s["threshold"]["method"] == "in_sample" and s["threshold"]["fallback_ran"]
    assert s["threshold"]["folds"] == 1 and table["score_heldout"].isna().all()
    assert s["threshold"]["threshold"] == pytest.approx(np.percentile(table["score_in_sample"], 95))
    assert any("in-sample" in w and "not a held-out" in w for w in res.report["warnings"])
    # --folds 1 with many polygons falls back too.
    res1 = run_main(main_store, main_files, tmp_path, folds=1, out=tmp_path / "f1")
    assert res1.summary["threshold"]["method"] == "in_sample"


def test_mahalanobis_needs_500_seed_pixels_before_any_chunk_read(main_store, main_files, tmp_path):
    reader, log = main_store
    with pytest.raises(AppInputError, match="at least 500"):
        run_main(main_store, main_files, tmp_path, score="mahalanobis")
    assert not log.chunk_keys() and not (tmp_path / "out" / "wetland_report.json").exists()


def test_mahalanobis_runs_with_enough_seed_pixels(main_store, main_files, tmp_path, no_coast):
    reader, _ = main_store
    grid = reader.grid("17N")
    big = [grid_box_lonlat(grid, (r, r + 12), (c, c + 12)) for r, c in ((16, 16), (44, 30), (72, 12), (20, 84))]
    seeds = write_fc(tmp_path / "big.geojson", big)
    res = run_main(main_store, main_files, tmp_path, seeds=seeds, score="mahalanobis")
    s = res.summary
    table = pd.read_parquet(res.extra_paths["seeds"])
    used = table[table["used"]]
    assert s["seeds"]["used"] == len(used) >= 500 and s["score"]["type"] == "mahalanobis"
    pts = read_points(TesseraReader.from_store(main_store[1]._store), used["lon"], used["lat"], 2020)
    model = wl.fit_seed_model(pts.embeddings, "mahalanobis")
    assert s["score"]["shrinkage"] == pytest.approx(model.shrinkage)
    np.testing.assert_allclose(used["score_in_sample"], model.score_vectors(pts.embeddings), rtol=1e-6)
    assert "Ledoit-Wolf" in s["score"]["reason"]
    with rasterio.open(res.paths.tif) as ds:
        score_a, ext_a = ds.read(1), ds.read(3)
    rows, cols = np.mgrid[0 : score_a.shape[0], 0 : score_a.shape[1]]
    wet = np.isfinite(score_a) & (score_a <= s["threshold"]["threshold"])
    assert np.isfinite(score_a).any() and wet.any()


def test_too_few_valid_seed_pixels_stop_before_the_aoi_stream(main_store, main_files, tmp_path, no_coast):
    reader, log = main_store
    grid = reader.grid("17N")
    # Seed pixels on the no-data rows: placed (so the geometry check passes), but without data.
    seeds = write_fc(
        tmp_path / "nodata.geojson", [grid_box_lonlat(grid, (0, 3), (60, 64)), Point(*centre_lonlat(grid, 50, 50))]
    )
    with pytest.raises(wl.WetlandError, match="Only 1 seed pixels are usable"):
        run_main(main_store, main_files, tmp_path, seeds=seeds)
    assert not (tmp_path / "out" / "wetland_summary.parquet").exists()
    # Fewer than 5 seed pixels by geometry stop before any chunk read.
    log.log.clear()
    few = write_fc(tmp_path / "few.geojson", [Point(*centre_lonlat(grid, 50, 50 + i)) for i in range(4)])
    with pytest.raises(AppInputError, match="at least 5"):
        run_main(main_store, main_files, tmp_path, seeds=few)
    assert not log.chunk_keys()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"year_b": 2020}, "must be after"),
        ({"score": "euclid"}, "must be one of"),
        ({"threshold_pct": 0.0}, "threshold_pct"),
        ({"folds": 0}, "folds"),
        ({"min_area_ha": -1.0}, "min_area_ha"),
        ({"coast_km": 0.0}, "coast_km"),
        ({"seed_buffer_m": -5.0}, "seed_buffer_m"),
        ({"reference_source": "osm"}, "reference_source"),
        ({"gmw_version": "v2"}, "gmw_version"),
        ({"reference_classes": [96]}, "no class"),
        ({"reference_classes": [90], "reference_source": "gmw"}, "WorldCover reference only"),
        ({"validate": True, "year_b": 2024, "year_a": 2022}, r"2020 \(v100\) and 2021 \(v200\) only"),
        ({"validate": True, "year_b": 2024, "year_a": 2022, "reference_source": "gmw"}, "GMW v3 does not cover"),
        ({"shoreline": "missing.geojson"}, "Shoreline file"),
        ({"reference": "missing.tif"}, "Reference file"),
    ],
)
def test_bad_arguments_stop_before_the_store_is_read(main_store, main_files, tmp_path, kwargs, match):
    reader, log = main_store
    aoi, seeds, _ = main_files
    kw = {"seeds": seeds, "reader": reader, **kwargs}
    a = kw.pop("year_a", 2020)
    b = kw.pop("year_b", 2024)
    if isinstance(kw.get("shoreline"), str):
        kw["shoreline"] = tmp_path / kw["shoreline"]
    if isinstance(kw.get("reference"), str):
        kw["reference"] = tmp_path / kw["reference"]
    with pytest.raises(AppInputError, match=match):
        wl.run_wetland(aoi, a, b, tmp_path / "out", **kw)
    assert not log.chunk_keys()


def test_missing_years_for_a_seed_zone_stop_before_any_chunk_read(boundary_store, boundary_reader, tmp_path):
    _, log = boundary_store
    lon0, lat0 = BOUNDARY
    aoi = write_fc(tmp_path / "aoi.geojson", [box(lon0 - 0.008, lat0 - 0.004, lon0 - 0.002, lat0 + 0.004)])
    seeds = write_fc(tmp_path / "seeds.geojson", [Point(lon0 + 0.002 + i * 0.0002, lat0) for i in range(6)])
    with pytest.raises(io.YearsNotPublishedError) as info:
        wl.run_wetland(aoi, 2021, 2023, tmp_path / "out", seeds=seeds, reader=boundary_reader)
    assert info.value.missing == {"11N": [2023]}
    notes = " ".join(getattr(info.value, "__notes__", []))
    assert "Zone 11N holds 6 of the 6 seed pixels" in notes and "the AOI does not touch it" in notes
    assert not log.chunk_keys()


def test_too_many_seed_chunks_need_the_large_flag(main_store, main_files, tmp_path, monkeypatch, no_coast):
    reader, log = main_store
    monkeypatch.setattr(wl, "MAX_SEED_CHUNKS", 3)
    with pytest.raises(AppInputError, match="store chunks, over the 3 the app reads without --i-know-this-is-large"):
        run_main(main_store, main_files, tmp_path)
    assert not log.chunk_keys()
    res = run_main(main_store, main_files, tmp_path, max_area_km2=None)
    assert not res.skipped


def test_coastal_fraction_from_a_shoreline_file(main_store, main_files, tmp_path):
    reader, _ = main_store
    grid = reader.grid("17N")
    to_ll = grid.to_lonlat()
    x = grid.left + 40.5 * grid.pixel
    line = LineString([to_ll.transform(x, grid.top), to_ll.transform(x, grid.top - SIZE * grid.pixel)])
    shore = write_fc(tmp_path / "shore.geojson", [line])
    res = run_main(main_store, main_files, tmp_path, shoreline=shore, coast_km=0.1)
    s = res.summary
    with rasterio.open(res.paths.tif) as ds:
        ext_a, ext_b = ds.read(3) == 1, ds.read(4) == 1
        transform = ds.transform
    cols = np.floor((transform.c - grid.left) / grid.pixel).astype(int) + np.arange(ext_a.shape[1])
    near = np.broadcast_to(np.abs(cols - 40) * grid.pixel <= 100.0, ext_a.shape)
    assert s["coastal"]["source"] == "file" and s["coastal"]["approximate"] is False
    assert s["coastal_fraction_year_a"] == pytest.approx((ext_a & near).sum() / ext_a.sum())
    assert s["coastal_fraction_year_b"] == pytest.approx((ext_b & near).sum() / ext_b.sum())
    assert 0.1 < s["coastal_fraction_year_a"] < 0.9
    table = pd.read_parquet(res.paths.summary_parquet)
    assert table["coastal_fraction"].tolist() == pytest.approx(
        [s["coastal_fraction_year_a"], s["coastal_fraction_year_b"]]
    )
    assert s["coastal"]["lines"] == 1 and s["coastal"]["length_km"] == pytest.approx(1.28, rel=0.01)


def test_coastal_fraction_from_natural_earth_is_labelled_approximate(main_store, main_files, tmp_path, monkeypatch):
    reader, _ = main_store
    grid = reader.grid("17N")
    to_ll = grid.to_lonlat()
    calls = []

    def fake(bbox, *, scale="10m", source=None):
        calls.append((bbox, scale))
        x = grid.left + 40.5 * grid.pixel
        return [LineString([to_ll.transform(x, grid.top), to_ll.transform(x, grid.top - SIZE * grid.pixel)])]

    monkeypatch.setattr(validate, "natural_earth_coastline", fake)
    res = run_main(main_store, main_files, tmp_path, coast_km=0.1)
    coast = res.summary["coastal"]
    assert coast["source"] == "natural_earth_10m" and coast["approximate"] is True and "approximate" in coast["note"]
    assert coast["median_segment_m"] == pytest.approx(1280.0, rel=0.01)
    assert res.summary["coastal_fraction_year_a"] is not None
    ((bbox, scale),) = calls
    aoi = shapely.geometry.shape(json.loads(main_files[0].read_text())["features"][0]["geometry"])
    assert scale == "10m" and box(*bbox).contains(aoi.buffer(0.0009))


# --------------------------------------------------------------------------------------------
# Validation on the designed store
# --------------------------------------------------------------------------------------------


@pytest.fixture
def wc_template(main_store, tmp_path):
    reader, _ = main_store
    return write_worldcover(tmp_path / "wc", reader.grid("17N"))


def test_validation_excludes_the_seeds_and_scores_each_worldcover_year(
    main_store, main_files, wc_template, tmp_path, no_coast
):
    reader, log = main_store
    res = run_main(
        main_store,
        main_files,
        tmp_path,
        year_b=2021,
        validate=True,
        worldcover_url_template=wc_template,
        seed_buffer_m=50.0,
    )
    assert res.validation_written
    s = res.summary
    val = s["validation"]
    plan = plan_aoi(reader, shapely.geometry.shape(json.loads(main_files[0].read_text())["features"][0]["geometry"]))
    (zp,) = plan.zones
    with rasterio.open(res.paths.tif) as ds:
        ext = {2020: ds.read(3) == 1, 2021: ds.read(4) == 1}
        classified = np.isfinite(ds.read(5))
    seeds = pd.read_parquet(res.extra_paths["seeds"])
    sr, sc = seeds["row"].to_numpy() - zp.rows.start, seeds["col"].to_numpy() - zp.cols.start
    rr, cc = np.mgrid[0 : zp.shape[0], 0 : zp.shape[1]]
    d = cdist(np.stack([rr.ravel(), cc.ravel()], 1), np.stack([sr, sc], 1)).min(axis=1).reshape(zp.shape) * 10.0
    excluded = d <= 50.0
    assert val["years_scored"] == [2020, 2021] and val["years_not_covered"] == []
    assert val["source"] == "worldcover" and val["year"] == 2020 and val["seed_buffer_m"] == 50.0
    written = json.loads(res.paths.validation.read_text())
    for year in (2020, 2021):
        codes = validate.worldcover_map(zp, year, url_template=wc_template)
        ref = (codes == 95) & zp.mask
        cover = (codes != 0) & zp.mask
        base = classified & cover & zp.mask
        keep = base & ~excluded
        tp = int((ext[year] & ref & keep).sum())
        fp = int((ext[year] & ~ref & keep).sum())
        fn = int((~ext[year] & ref & keep).sum())
        row = val["per_year"][str(year)]
        assert row["recall"] == pytest.approx(tp / (tp + fn)) and row["precision"] == pytest.approx(tp / (tp + fp))
        assert row["iou"] == pytest.approx(tp / (tp + fp + fn)) and row["f1"] == pytest.approx(
            2 * tp / (2 * tp + fp + fn)
        )
        assert row["pixels_scored"] == int(keep.sum()) and row["seed_excluded_pixels"] == int((base & excluded).sum())
        assert row["seed_excluded_pixels"] > 216
        tp_s = int((ext[year] & ref & base).sum())
        fn_s = int((~ext[year] & ref & base).sum())
        assert row["recall_with_seeds"] == pytest.approx(tp_s / (tp_s + fn_s))
        fp_codes = codes[ext[year] & ~ref & keep]
        values, counts = np.unique(fp_codes, return_counts=True)
        assert {k: v["pixels"] for k, v in row["false_positive_classes"].items()} == {
            str(int(a)): int(b) for a, b in zip(values, counts, strict=True)
        }
        assert set(row["false_positive_classes"]) >= {"90", "10"}
        score = written["scores"][f"extent_{year}"]
        assert (score["tp"], score["fp"], score["fn"]) == (tp, fp, fn)
        assert written["scores"][f"false_positive_classes_{year}"] == row["false_positive_classes"]
    # The two years are scored against their own maps: 2021 differs where only its map changed.
    assert val["per_year"]["2020"]["recall"] != val["per_year"]["2021"]["recall"]
    assert (val["recall"], val["precision"]) == (
        val["per_year"]["2020"]["recall"],
        val["per_year"]["2020"]["precision"],
    )
    ref_rec = written["reference"]
    assert ref_rec["versions"] == {"2020": "v100", "2021": "v200"} and ref_rec["classes"] == [95]
    caveats = " ".join(written["caveats"])
    for text in ("leaving out the seed pixels", "itself a model", "herbaceous wetland (90)", "50 m"):
        assert text in caveats, text
    assert "same product" not in caveats  # the test seeds do not come from WorldCover
    assert res.report["outputs"]["validation"] == export.file_sha256(res.paths.validation)


def test_validation_against_worldcover_2020_only_when_year_b_is_not_mapped(
    main_store, main_files, wc_template, tmp_path, no_coast
):
    res = run_main(main_store, main_files, tmp_path, validate=True, worldcover_url_template=wc_template)
    val = res.summary["validation"]
    assert val["years_scored"] == [2020] and val["years_not_covered"] == [2024] and set(val["per_year"]) == {"2020"}
    caveats = " ".join(json.loads(res.paths.validation.read_text())["caveats"])
    assert "no map for 2024" in caveats


def test_same_product_seeds_and_other_reference_classes_get_caveats(
    main_store, main_files, wc_template, tmp_path, no_coast
):
    reader, _ = main_store
    seeds = write_fc(
        tmp_path / "wc_seeds.geojson",
        seed_polygons(reader.grid("17N")),
        {"source": ["ESA WorldCover 2020 v100, class 95"] * len(SEED_SQUARES)},
    )
    res = run_main(
        main_store,
        main_files,
        tmp_path,
        seeds=seeds,
        validate=True,
        worldcover_url_template=wc_template,
        reference_classes=[95, 90],
    )
    written = json.loads(res.paths.validation.read_text())
    caveats = " ".join(written["caveats"])
    assert "same product as the reference (ESA WorldCover)" in caveats
    assert "different wetland type" in caveats and "90 (herbaceous wetland)" in caveats
    assert written["reference"]["classes"] == [90, 95]


def test_validation_against_a_reference_file_scores_both_years(main_store, main_files, tmp_path, no_coast):
    reader, _ = main_store
    grid = reader.grid("17N")
    ref = write_fc(tmp_path / "ref.geojson", [grid_box_lonlat(grid, *STAND_A)])
    res = run_main(main_store, main_files, tmp_path, validate=True, reference=ref)
    val = res.summary["validation"]
    assert val["source"] == "file" and val["years_scored"] == [2020, 2024]
    assert "false_positive_classes" not in val["per_year"]["2020"]
    assert "undated" in " ".join(json.loads(res.paths.validation.read_text())["caveats"])


def test_gmw_reference_scores_the_years_its_version_covers(main_store, main_files, tmp_path, monkeypatch, no_coast):
    calls = []

    def extent(plan, year, *, version="v3", source=None):
        calls.append((year, version))
        return plan.mask.copy()

    monkeypatch.setattr(validate, "gmw_extent", extent)
    monkeypatch.setattr(validate, "gmw_coverage", lambda plan, year, *, version="v3", source=None: plan.mask.copy())
    res = run_main(main_store, main_files, tmp_path, validate=True, reference_source="gmw")
    val = res.summary["validation"]
    assert val["source"] == "gmw" and val["years_scored"] == [2020] and val["years_not_covered"] == [2024]
    assert calls == [(2020, "v3")]
    calls.clear()
    res4 = run_main(
        main_store,
        main_files,
        tmp_path,
        validate=True,
        reference_source="gmw",
        gmw_version="v4.1.12",
        out=tmp_path / "v4",
    )
    assert res4.summary["validation"]["years_scored"] == [2020, 2024]
    assert calls == [(2020, "v4.1.12"), (2024, "v4.1.12")]
    assert "versions are never mixed" in " ".join(json.loads(res4.paths.validation.read_text())["caveats"])


def test_an_unreachable_reference_warns_and_a_rerun_redoes_only_the_validation(
    main_store, main_files, wc_template, tmp_path, monkeypatch, no_coast
):
    reader, log = main_store
    real = validate.worldcover_map

    def refused(*args, **kwargs):
        raise validate.ReferenceUnavailableError("esa-worldcover.s3.eu-central-1.amazonaws.com refused (403)")

    monkeypatch.setattr(validate, "worldcover_map", refused)
    first = run_main(main_store, main_files, tmp_path, validate=True, worldcover_url_template=wc_template)
    assert not first.validation_written and not first.paths.validation.exists()
    assert "error" in first.summary["validation"]
    assert any(w.startswith(wl.VALIDATION_SKIPPED) for w in first.report["warnings"])
    # Strict: stop before any chunk read.
    log.log.clear()
    with pytest.raises(StrictValidationError):
        run_main(
            main_store,
            main_files,
            tmp_path,
            validate=True,
            worldcover_url_template=wc_template,
            strict_validate=True,
            out=tmp_path / "strict",
        )
    assert not log.chunk_keys()
    # Reachable again: only the validation is redone, from the outputs on disk, without a chunk read.
    monkeypatch.setattr(validate, "worldcover_map", real)
    log.log.clear()
    again = run_main(main_store, main_files, tmp_path, validate=True, worldcover_url_template=wc_template)
    assert again.validation_redone and again.validation_written and not log.chunk_keys()
    assert "per_year" in again.summary["validation"]
    assert not any(w.startswith(wl.VALIDATION_SKIPPED) for w in again.report["warnings"])
    assert "redone from the outputs" in " ".join(json.loads(again.paths.validation.read_text())["caveats"])
    fresh = run_main(
        main_store, main_files, tmp_path, validate=True, worldcover_url_template=wc_template, out=tmp_path / "fresh"
    )
    assert fresh.summary["validation"]["per_year"] == again.summary["validation"]["per_year"]
    third = run_main(main_store, main_files, tmp_path, validate=True, worldcover_url_template=wc_template)
    assert third.skipped and third.validation_written


# --------------------------------------------------------------------------------------------
# Across the 120 W seam: the extent is cleaned on the mosaic, seeds exclude across the seam
# --------------------------------------------------------------------------------------------

LON0, LAT0 = BOUNDARY
STAND = box(LON0 - 0.0050, LAT0 - 0.0012, LON0 - 0.0020, LAT0 + 0.0012)  # in 10N, for the seeds
SEAM_PATCH = box(LON0 - 0.00045, LAT0 - 0.00032, LON0 + 0.00045, LAT0 + 0.00032)  # about 8 x 8 px each side


def seam_pattern(lon, lat, rows, cols, year):
    stand = shapely.contains_xy(STAND, lon, lat)
    patch = shapely.contains_xy(SEAM_PATCH, lon, lat)
    on = np.ones(lon.shape, dtype=bool)
    return stand | patch, on, on, patch


def test_a_patch_across_the_zone_seam_is_kept_whole(tmp_path, no_coast):
    reader, _, _ = wetland_store(seam_pattern, zones=("10N", "11N"), centre=BOUNDARY, offsets=False)
    aoi = write_fc(tmp_path / "aoi.geojson", [box(LON0 - 0.0058, LAT0 - 0.0018, LON0 + 0.0020, LAT0 + 0.0018)])
    seeds = write_fc(
        tmp_path / "seeds.geojson",
        [box(LON0 - 0.0048 + i * 0.001, LAT0 - 0.0002, LON0 - 0.0043 + i * 0.001, LAT0 + 0.0003) for i in range(3)],
    )
    plan = plan_aoi(reader, shapely.geometry.shape(json.loads(aoi.read_text())["features"][0]["geometry"]))
    assert plan.zone_names == ["10N", "11N"]
    halves = [int((shapely.contains_xy(SEAM_PATCH, *_centres(z)) & z.mask).sum()) for z in plan.zones]
    min_area = 0.01 * (max(halves) + 5)
    assert sum(halves) * 0.01 > min_area > max(halves) * 0.01
    res = wl.run_wetland(
        aoi, 2020, 2024, tmp_path / "out", seeds=seeds, reader=reader, block=(32, 32), min_area_ha=min_area
    )
    s = res.summary
    with rasterio.open(res.paths.tif) as ds:
        ext_a = ds.read(3)
        transform, crs = ds.transform, ds.crs
    h, w = ext_a.shape
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs, ys = transform.c + cols * transform.a, transform.f + rows * transform.e
    lon, lat = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True).transform(xs, ys)
    in_patch = shapely.contains_xy(SEAM_PATCH.buffer(-0.00005), lon, lat)
    assert in_patch.sum() > 30 and (ext_a[in_patch] == 1).all()
    assert s["extent"]["2020"]["extent_pixels"] >= sum(halves) - 4
    # The same run cleaned zone by zone would drop both halves of the patch.
    per_zone = {}
    for z in plan.zones:
        raw = shapely.contains_xy(SEAM_PATCH, *_centres(z)) & z.mask
        per_zone[z.zone] = metrics.clean_mask(raw, min_pixels=metrics.area_to_pixels(min_area))
    assert not any(m.any() for m in per_zone.values())
    whole, _, _, _ = mosaic_and_clean(
        {z.zone: shapely.contains_xy(SEAM_PATCH, *_centres(z)) & z.mask for z in plan.zones},
        plan,
        min_pixels=metrics.area_to_pixels(min_area),
    )
    assert sum(int(m.sum()) for m in whole.values()) == sum(halves)


def _centres(z):
    rows, cols = np.mgrid[z.rows, z.cols]
    x = z.grid.left + (cols + 0.5) * z.grid.pixel
    y = z.grid.top - (rows + 0.5) * z.grid.pixel
    return z.grid.to_lonlat().transform(x, y)


def test_seeds_across_the_seam_exclude_their_neighbours_in_the_other_zone(tmp_path):
    reader, _, _ = wetland_store(seam_pattern, zones=("10N", "11N"), centre=BOUNDARY, offsets=False)
    aoi = box(LON0 - 0.003, LAT0 - 0.001, LON0 + 0.003, LAT0 + 0.001)
    plan = plan_aoi(reader, aoi)
    east = plan.zones[1]
    assert east.zone == "11N"
    seed = np.array([[LON0 - 0.0001, LAT0]])  # in 10N, about 9 m west of the seam
    got = wl.seed_exclusion(east, seed, 60.0)
    lon, lat = _centres(east)
    geod_d = np.hypot(
        (np.asarray(lon) - seed[0, 0]) * 111_320.0 * math.cos(math.radians(LAT0)),
        (np.asarray(lat) - seed[0, 1]) * 110_950.0,
    )
    assert got.any() and got[(geod_d < 45.0) & east.mask].all() and not got[geod_d > 75.0].any()


def test_moving_the_seeds_file_keeps_the_run_current(main_store, main_files, tmp_path, no_coast):
    first = run_main(main_store, main_files, tmp_path)
    moved = tmp_path / "elsewhere" / "seeds.geojson"
    moved.parent.mkdir()
    moved.write_bytes(main_files[1].read_bytes())
    again = run_main(main_store, main_files, tmp_path, seeds=moved)
    assert again.skipped and again.report == first.report
    edited = tmp_path / "edited.geojson"
    write_fc(edited, seed_polygons(main_store[0].grid("17N"))[:5])
    assert not run_main(main_store, main_files, tmp_path, seeds=edited).skipped
