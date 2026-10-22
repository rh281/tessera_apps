"""The plumbing every app shares: AOI loading, run preparation, cleaning across zone seams, the report."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import tempfile
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import geotessera
import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import LineString, Point, box, mapping

from tessera_apps import export, metrics, validate
from tessera_apps.apps import _common
from tessera_apps.apps._common import (
    AppInputError,
    LoadedAOI,
    aoi_reference,
    code_record,
    finish_run,
    load_aoi,
    mosaic_and_clean,
    prepare_run,
    stride_for_preview,
)
from tessera_apps.demo import DEMO_NAMES, demo_aoi, demo_properties
from tessera_apps.io import TesseraReader, YearsNotPublishedError, plan_aoi

from .conftest import BOUNDARY, KeyLog


def write_geojson(path: Path, features: list[tuple[object, dict]]) -> Path:
    """A lon/lat FeatureCollection of (geometry, properties) pairs."""
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": props, "geometry": mapping(g)} for g, props in features],
    }
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


LON0, LAT0 = BOUNDARY
TWO_ZONE = box(LON0 - 0.004, LAT0 - 0.003, LON0 + 0.003, LAT0 + 0.003)


# --------------------------------------------------------------------------------------------
# load_aoi
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".gpkg", ".geojson"])
def test_load_aoi_reprojects_a_web_mercator_file(tmp_path, suffix):
    to_merc = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = to_merc.transform(-120.5, 38.7)
    poly = box(x0, y0, x0 + 3000.0, y0 + 2000.0)
    path = tmp_path / f"aoi{suffix}"
    gpd.GeoDataFrame({"name": ["box"]}, geometry=[poly], crs="EPSG:3857").to_file(path)

    loaded = load_aoi(path)

    expected = shapely.transform(poly, Transformer.from_crs(3857, 4326, always_xy=True).transform, interleaved=False)
    assert loaded.geometry.symmetric_difference(expected).area < 1e-12
    west, south, _, _ = loaded.geometry.bounds
    assert west == pytest.approx(-120.5, abs=1e-9) and south == pytest.approx(38.7, abs=1e-9)
    assert loaded.path == path and loaded.name == "aoi" and loaded.demo is None
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert loaded.properties == {"name": "box"}
    # Stable: loading twice gives the same hash and geometry.
    again = load_aoi(str(path))
    assert again.sha256 == loaded.sha256 and again.geometry.equals(loaded.geometry)


def test_load_aoi_keeps_polygons_only_and_unions_them(tmp_path):
    a = box(-120.50, 38.70, -120.49, 38.71)
    b = box(-120.48, 38.70, -120.47, 38.71)
    path = write_geojson(
        tmp_path / "parts.geojson",
        [(a, {"k": 1}), (b, {"k": 2}), (LineString([(-121, 38), (-119, 39)]), {}), (Point(-120.0, 38.0), {})],
    )
    loaded = load_aoi(path)
    assert loaded.geometry.geom_type == "MultiPolygon"
    assert loaded.geometry.symmetric_difference(a.union(b)).area < 1e-15
    assert loaded.properties == {}


def test_load_aoi_uses_the_aoi_role_of_a_feature_collection(tmp_path):
    aoi = box(-120.50, 38.70, -120.49, 38.71)
    elsewhere = box(-100.0, 40.0, -99.0, 41.0)
    path = write_geojson(
        tmp_path / "demo_like.geojson",
        [(elsewhere, {"role": "outline"}), (aoi, {"role": "aoi", "year_a": 2020})],
    )
    loaded = load_aoi(path)
    assert loaded.geometry.equals(aoi)
    assert loaded.properties == {"role": "aoi", "year_a": 2020}


@pytest.mark.parametrize("name", DEMO_NAMES)
def test_load_aoi_demo_names(name):
    loaded = load_aoi(name)
    assert loaded.geometry.equals(demo_aoi(name))
    assert loaded.name == name and loaded.demo == name and loaded.source == f"demo:{name}"
    bundled = files("tessera_apps.data").joinpath(f"{name}.geojson")
    assert loaded.sha256 == hashlib.sha256(bundled.read_bytes()).hexdigest()
    assert loaded.path is not None and loaded.path.read_bytes() == bundled.read_bytes()
    assert loaded.properties == demo_properties(name)
    # Case and whitespace do not matter, and the hash is stable.
    assert load_aoi(f" {name.upper()} ").sha256 == loaded.sha256


def test_load_aoi_passes_a_loaded_aoi_through():
    loaded = load_aoi("caldor")
    assert load_aoi(loaded) is loaded


@pytest.mark.filterwarnings("ignore:'crs' was not provided:UserWarning")
def test_load_aoi_errors(tmp_path):
    with pytest.raises(AppInputError, match="neither an existing file nor a bundled demo"):
        load_aoi(tmp_path / "missing.geojson")
    with pytest.raises(AppInputError, match="neither"):
        load_aoi("atlantis")
    points = write_geojson(tmp_path / "points.geojson", [(Point(-120.0, 38.0), {})])
    with pytest.raises(AppInputError, match="no polygon"):
        load_aoi(points)
    # A shapefile without a .prj in metres cannot be placed; in lon/lat it is taken as EPSG:4326.
    projected = tmp_path / "projected.shp"
    gpd.GeoDataFrame(geometry=[box(700_000, 4_200_000, 701_000, 4_201_000)]).to_file(projected)
    with pytest.raises(AppInputError, match="declares no CRS"):
        load_aoi(projected)
    lonlat = tmp_path / "lonlat.shp"
    poly = box(-120.5, 38.7, -120.49, 38.71)
    gpd.GeoDataFrame(geometry=[poly]).to_file(lonlat)
    assert load_aoi(lonlat).geometry.symmetric_difference(poly).area < 1e-15


# --------------------------------------------------------------------------------------------
# prepare_run and finish_run
# --------------------------------------------------------------------------------------------


@pytest.fixture
def two_zone_aoi(tmp_path) -> Path:
    return write_geojson(tmp_path / "two_zone.geojson", [(TWO_ZONE, {})])


def test_prepare_run_checks_every_year_before_any_chunk_read(boundary_store, boundary_reader, two_zone_aoi, tmp_path):
    _, log = boundary_store
    with pytest.raises(YearsNotPublishedError) as info:
        prepare_run(
            "demoapp",
            two_zone_aoi,
            [2020, 2021, 2023],
            2020,
            2021,
            tmp_path / "out",
            {},
            validate=False,
            force=False,
            max_area_km2=None,
            reader=boundary_reader,
        )
    assert info.value.missing == {"11N": [2023]}
    assert log.chunk_keys() == []
    assert not (tmp_path / "out").exists()


def test_prepare_run_plans_and_logs_without_reading_chunks(
    boundary_store, boundary_reader, two_zone_aoi, tmp_path, caplog
):
    _, log = boundary_store
    with caplog.at_level(logging.INFO, logger="tessera_apps"):
        ctx = prepare_run(
            "demoapp",
            two_zone_aoi,
            [2021, 2020],
            2020,
            2021,
            tmp_path / "out",
            {"k": 1},
            validate=False,
            force=False,
            max_area_km2=None,
            reader=boundary_reader,
        )
    assert ctx is not None
    assert log.chunk_keys() == []
    assert ctx.plan.zone_names == ["10N", "11N"]
    assert ctx.years_read == [2020, 2021]
    assert ctx.estimated_bytes == ctx.plan.estimated_bytes(boundary_reader, 2)
    assert ctx.params == {
        "k": 1,
        "aoi_geometry_sha256": ctx.aoi.geometry_sha256,
        "code": code_record(),
        "data_snapshot_id": None,
    }
    assert ctx.digest == export.run_args_digest(ctx.params, years=[2020, 2021], aoi_sha256=ctx.aoi.sha256, seed=0)
    assert not ctx.validation_only
    assert "10N" in caplog.text and "11N" in caplog.text and "GB" in caplog.text


def test_prepare_run_refuses_a_large_aoi(caldor_reader, caldor_store, tmp_path):
    _, log = caldor_store
    big = write_geojson(tmp_path / "big.geojson", [(box(-121.0, 38.0, -120.0, 39.0), {})])
    with pytest.raises(Exception, match="km2"):
        prepare_run(
            "demoapp",
            big,
            [2020],
            2020,
            2021,
            tmp_path,
            {},
            validate=False,
            force=False,
            max_area_km2=100.0,
            reader=caldor_reader,
        )
    assert log.chunk_keys() == []


def _write_outputs(ctx, extra: Path | None = None) -> None:
    z = ctx.plan.zones[0]
    export.write_geotiff(ctx.paths.tif, np.ones(z.shape, dtype=np.float32), z.transform, z.grid.crs)
    export.write_table(pd.DataFrame({"year": [2021]}), ctx.paths.summary_parquet.with_suffix(""))
    export.write_preview_png(ctx.paths.preview, np.zeros((4, 4)))
    if extra is not None:
        export.write_geotiff(extra, np.zeros(z.shape, dtype=np.uint8), z.transform, z.grid.crs, nodata=255)


def _prepare(reader, aoi, out, params, **kw):
    return prepare_run(
        "demoapp",
        aoi,
        [2020, 2021],
        2020,
        2021,
        out,
        params,
        validate=kw.pop("validate", False),
        force=kw.pop("force", False),
        max_area_km2=None,
        reader=reader,
        **kw,
    )


def test_finish_run_writes_the_report_last_and_reruns_skip(boundary_reader, two_zone_aoi, tmp_path):
    out = tmp_path / "out"
    extra = out / "demoapp_extra.tif"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {"k": 1}, extra_outputs={"extra": extra})
    assert ctx is not None
    _write_outputs(ctx, extra)
    ctx.warn("something to note")
    # One chunk read inside the run: bytes_read counts chunk payload only, so a run that reads none reports 0.
    boundary_reader.read_window("10N", 2020, slice(0, 32), slice(0, 32))
    report = finish_run(ctx, summary={"burned": 1}, extra={"extra": extra}, validation_written=False)

    on_disk = json.loads(ctx.paths.report.read_text())
    assert on_disk == report
    assert report["args_digest"] == ctx.digest == report["provenance"]["args_digest"]
    assert report["summary"] == {"burned": 1}
    assert report["warnings"] == ["something to note"]
    assert report["provenance"]["aoi"]["sha256"] == ctx.aoi.sha256
    assert report["provenance"]["store"]["backend"] == "KeyLog"
    assert isinstance(report["bytes_read"], int) and report["bytes_read"] == boundary_reader.bytes_read > 0
    assert "chunk payload" in report["bytes_read_note"].lower()
    assert report["timings"]["wall_s"] > 0
    for name in ("tif", "summary_parquet", "summary_csv", "preview"):
        assert report["outputs"][name] == export.file_sha256(getattr(ctx.paths, name))
    assert "validation" not in report["outputs"]
    assert report["extra_outputs"] == {"extra": {"file": extra.name, "sha256": export.file_sha256(extra)}}
    # Written last: no output is newer than the report.
    report_time = ctx.paths.report.stat().st_mtime_ns
    for p in (*ctx.paths.required(need_validation=False), extra):
        assert p.stat().st_mtime_ns <= report_time

    # Same arguments: skip. Different params: run.
    assert _prepare(boundary_reader, two_zone_aoi, out, {"k": 1}, extra_outputs={"extra": extra}) is None
    assert _prepare(boundary_reader, two_zone_aoi, out, {"k": 2}, extra_outputs={"extra": extra}) is not None


@pytest.mark.parametrize("change", ["bytes", "deleted"])
def test_prepare_run_reruns_when_an_extra_output_changes(boundary_reader, two_zone_aoi, tmp_path, change):
    out = tmp_path / "out"
    extra = out / "demoapp_extra.tif"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {}, extra_outputs={"extra": extra})
    _write_outputs(ctx, extra)
    finish_run(ctx, summary={}, extra={"extra": extra}, validation_written=False)
    assert _prepare(boundary_reader, two_zone_aoi, out, {}, extra_outputs={"extra": extra}) is None
    # The report vouches for the extra output: other bytes (even with an old mtime), or none, rerun.
    if change == "bytes":
        z = ctx.plan.zones[0]
        export.write_geotiff(extra, np.ones(z.shape, dtype=np.uint8), z.transform, z.grid.crs, nodata=255)
        os.utime(extra, ns=(0, 0))
    else:
        extra.unlink()
    assert _prepare(boundary_reader, two_zone_aoi, out, {}, extra_outputs={"extra": extra}) is not None


def test_prepare_run_skip_force_and_validation(boundary_reader, two_zone_aoi, tmp_path):
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    assert _prepare(boundary_reader, two_zone_aoi, out, {}) is None
    # A validating run needs a validation.json the report vouches for; a stray one does not count.
    stray = ctx.paths.validation
    stray.write_text("{}")
    os.utime(stray, ns=(0, 0))
    assert _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True) is not None
    # That run's begin_run removed the old report and the stray validation file.
    assert not ctx.paths.report.exists() and not stray.exists()

    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    forced = _prepare(boundary_reader, two_zone_aoi, out, {}, force=True)
    assert forced is not None and not ctx.paths.report.exists()


def test_prepare_run_opens_the_live_reader_with_the_snapshot(monkeypatch, boundary_reader, two_zone_aoi, tmp_path):
    seen = {}

    def fake_open(**kwargs):
        seen.update(kwargs)
        return boundary_reader

    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(fake_open))
    ctx = prepare_run(
        "demoapp",
        two_zone_aoi,
        [2020],
        2020,
        2021,
        tmp_path,
        {},
        validate=False,
        force=False,
        max_area_km2=None,
        snapshot_id="SNAP",
    )
    assert ctx is not None and seen == {"snapshot_id": "SNAP"}
    assert ctx.reader is boundary_reader


# --------------------------------------------------------------------------------------------
# mosaic_and_clean
# --------------------------------------------------------------------------------------------


SEAM_PATCH = box(LON0 - 0.0012, LAT0 - 0.0008, LON0 + 0.0008, LAT0 + 0.0008)
ISOLATED = box(LON0 - 0.0035, LAT0 + 0.0015, LON0 - 0.0026, LAT0 + 0.0024)


def test_mosaic_and_clean_judges_a_seam_patch_whole(boundary_reader):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    assert plan.zone_names == ["10N", "11N"]
    seam = {z.zone: validate.reference_from_geometries(z, [SEAM_PATCH]) for z in plan.zones}
    lone = {z.zone: validate.reference_from_geometries(z, [ISOLATED]) for z in plan.zones}
    raw = {z: seam[z] | lone[z] for z in seam}
    parts = {z: int(m.sum()) for z, m in seam.items()}
    n_lone = sum(int(m.sum()) for m in lone.values())
    min_pixels = 250
    # The patch is under min_pixels in each zone but over it in total; the isolated patch is under.
    assert all(0 < n < min_pixels for n in parts.values()), parts
    assert sum(parts.values()) > min_pixels and 0 < n_lone < min_pixels

    cleaned, mosaic, transform, crs = mosaic_and_clean(raw, plan, min_pixels=min_pixels)

    for z in plan.zones:
        kept = cleaned[z.zone]
        assert kept.shape == z.shape and kept.dtype == bool
        assert not (kept & ~raw[z.zone]).any(), "cleaning must never add a pixel"
        assert not (kept & lone[z.zone]).any(), "the isolated patch is dropped"
        # The seam patch survives on both sides, give or take its eroded corners.
        assert kept[seam[z.zone]].mean() > 0.9, z.zone
    assert crs == plan.zones[0].grid.crs  # 10N has more AOI pixels
    assert mosaic.dtype == np.uint8 and set(np.unique(mosaic)) <= {0, 1, export.BOOL_NODATA}
    assert abs(int((mosaic != export.BOOL_NODATA).sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels
    assert abs(int((mosaic == 1).sum()) - sum(parts.values())) <= 0.05 * sum(parts.values())

    # Zone by zone, the same patch would be dropped on both sides: this is what the mosaic avoids.
    per_zone = {z.zone: metrics.clean_mask(raw[z.zone], min_pixels=min_pixels) for z in plan.zones}
    assert not any(m.any() for m in per_zone.values())


@pytest.mark.parametrize("density", [0.3, 0.6])
def test_mosaic_and_clean_maps_ragged_masks_back_exactly(boundary_reader, density):
    # Speckle has ragged edges everywhere, so many 11N pixels land on a 10N-grid mosaic pixel that
    # took its value from a neighbour. With nothing to clean, every pixel must come back as it was.
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    rng = np.random.default_rng(int(density * 10))
    raw = {z.zone: (rng.random(z.shape) < density) & z.mask for z in plan.zones}
    cleaned, mosaic, _, _ = mosaic_and_clean(raw, plan, min_pixels=0, opening=1)
    for z in plan.zones:
        np.testing.assert_array_equal(cleaned[z.zone], raw[z.zone])
    # With cleaning, nothing is added, and the home zone follows the mosaic exactly.
    cleaned, mosaic, transform, _ = mosaic_and_clean(raw, plan, min_pixels=12, opening=1)
    home = plan.zones[0]
    row_off = round((transform.f - home.transform.f) / 10.0)
    col_off = round((home.transform.c - transform.c) / 10.0)
    window = mosaic[row_off : row_off + home.shape[0], col_off : col_off + home.shape[1]]
    np.testing.assert_array_equal(cleaned[home.zone], (window == 1) & home.mask)
    for z in plan.zones:
        assert not (cleaned[z.zone] & ~raw[z.zone]).any()
    away = plan.zones[1]
    kept_away = int(cleaned[away.zone].sum())
    assert kept_away > 0
    assert abs(kept_away - (int((mosaic == 1).sum()) - int(cleaned[home.zone].sum()))) <= 0.03 * kept_away


def test_mosaic_and_clean_single_zone_is_clean_mask(boundary_reader):
    plan = plan_aoi(boundary_reader, box(LON0 - 0.004, LAT0 - 0.003, LON0 - 0.0005, LAT0 + 0.003))
    assert plan.zone_names == ["10N"]
    (z,) = plan.zones
    raw = validate.reference_from_geometries(z, [SEAM_PATCH, ISOLATED])
    noise = np.zeros(z.shape, dtype=bool)
    noise[5, 5] = noise[40, 7] = True
    cleaned, mosaic, transform, crs = mosaic_and_clean({"10N": raw | noise}, plan, min_pixels=50)
    np.testing.assert_array_equal(cleaned["10N"], metrics.clean_mask(raw | noise, min_pixels=50) & z.mask)
    assert not cleaned["10N"][5, 5] and not cleaned["10N"][40, 7]
    assert transform == z.transform and crs == z.grid.crs
    np.testing.assert_array_equal(mosaic == export.BOOL_NODATA, ~z.mask)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "max_side", "step"),
    [((100, 50), 2048, 1), ((2048, 10), 2048, 1), ((2049, 10), 2048, 2), ((10, 6000), 2048, 3), ((7, 7, 3), 3, 3)],
)
def test_stride_for_preview(shape, max_side, step):
    assert stride_for_preview(shape, max_side) == step
    assert max(len(range(0, shape[0], step)), len(range(0, shape[1], step))) <= max_side


def test_stride_for_preview_rejects_zero():
    with pytest.raises(ValueError, match="max_side"):
        stride_for_preview((10, 10), 0)


def test_aoi_reference_reads_vectors_and_rasters(boundary_reader, tmp_path):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    vec = write_geojson(tmp_path / "ref.geojson", [(SEAM_PATCH, {})])
    for z in plan.zones:
        np.testing.assert_array_equal(aoi_reference(z, vec), validate.reference_from_geometries(z, [SEAM_PATCH]))
    z = plan.zones[0]
    data = validate.reference_from_geometries(z, [SEAM_PATCH]).astype(np.uint8)
    tif = export.write_geotiff(tmp_path / "ref.tif", data, z.transform, z.grid.crs, nodata=255)
    np.testing.assert_array_equal(aoi_reference(z, tif), data.astype(bool) & z.mask)


def test_loaded_aoi_source():
    geom = box(0, 0, 1, 1)
    assert LoadedAOI(geom, Path("/x/a.geojson"), "0", "a", {}).source == "/x/a.geojson"
    assert LoadedAOI(geom, None, "0", "a", {}, demo="caldor").source == "demo:caldor"


# --------------------------------------------------------------------------------------------
# Review fixes: AOI identity, reruns, early failures
# --------------------------------------------------------------------------------------------


def test_load_aoi_demo_name_wins_over_a_directory(tmp_path, monkeypatch):
    # `--out caldor` creates ./caldor/; an identical rerun must still find the demo.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "caldor").mkdir()
    loaded = load_aoi("caldor")
    assert loaded.demo == "caldor" and loaded.geometry.equals(demo_aoi("caldor"))
    assert load_aoi("Caldor").demo == "caldor"
    # A regular file of that name is still a file.
    write_geojson(tmp_path / "rondonia", [(box(-120.5, 38.7, -120.49, 38.71), {})])
    assert load_aoi("rondonia").demo is None


def _hash(geometry) -> str:
    return LoadedAOI(geometry, None, "x", "a", {}).geometry_sha256


def test_geometry_hash_is_stable_and_follows_the_geometry():
    a = box(-120.5, 38.7, -120.49, 38.71)
    assert _hash(a) == _hash(shapely.Polygon(list(a.exterior.coords)[::-1]))  # orientation and start vertex
    assert _hash(a) == _hash(shapely.transform(a, lambda c: c + 1e-12))  # floating point noise
    assert _hash(a) != _hash(box(-120.5, 38.7, -120.49, 38.7101))


def _prepare_shp(reader, path, out):
    return prepare_run(
        "demoapp",
        path,
        [2020, 2021],
        2020,
        2021,
        out,
        {},
        validate=False,
        force=False,
        max_area_km2=None,
        reader=reader,
    )


@pytest.mark.parametrize("sidecar", [".dbf", ".prj"])
def test_editing_a_shapefile_sidecar_reruns(boundary_reader, tmp_path, sidecar):
    # The AOI of a shapefile depends on its .prj (CRS) and .dbf (the role column), but the AOI file
    # hash covers the .shp alone: the digest must follow the geometry that was loaded (survey S19).
    utm10 = "+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs"
    x0, y0 = Transformer.from_crs(4326, 32610, always_xy=True).transform(LON0 - 0.003, LAT0 - 0.002)
    small = box(x0, y0, x0 + 150, y0 + 150)
    large = box(x0, y0, x0 + 300, y0 + 300)
    first = tmp_path / "a" / "aoi.shp"
    first.parent.mkdir()
    gpd.GeoDataFrame({"role": ["aoi", "other"]}, geometry=[small, large], crs=utm10).to_file(first)
    if sidecar == ".dbf":
        other = gpd.GeoDataFrame({"role": ["other", "aoi"]}, geometry=[small, large], crs=utm10)
    else:  # the same coordinates, read 200 m further west
        shifted = "+proj=tmerc +lat_0=0 +lon_0=-123 +k=0.9996 +x_0=500200 +y_0=0 +datum=WGS84 +units=m +no_defs"
        other = gpd.GeoDataFrame({"role": ["aoi", "other"]}, geometry=[small, large], crs=shifted)
    second = tmp_path / "b" / "aoi.shp"
    second.parent.mkdir()
    other.to_file(second)
    assert first.read_bytes() == second.read_bytes(), "only the sidecar differs"

    out = tmp_path / "out"
    before = load_aoi(first)
    ctx = _prepare_shp(boundary_reader, first, out)
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    assert _prepare_shp(boundary_reader, first, out) is None
    (first.with_suffix(sidecar)).write_bytes(second.with_suffix(sidecar).read_bytes())
    after = load_aoi(first)
    assert after.sha256 == before.sha256 and not after.geometry.equals(before.geometry)
    rerun = _prepare_shp(boundary_reader, first, out)
    assert rerun is not None and rerun.params["aoi_geometry_sha256"] == after.geometry_sha256


def test_prepare_run_reruns_on_a_new_store_snapshot(_boundary_mem, two_zone_aoi, tmp_path):
    out = tmp_path / "out"

    def prepare(snapshot: str):
        reader = TesseraReader.from_store(KeyLog(_boundary_mem), snapshot_id=snapshot)
        return _prepare(reader, two_zone_aoi, out, {})

    ctx = prepare("SNAP_A")
    assert ctx.params["data_snapshot_id"] == "SNAP_A"
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    assert prepare("SNAP_A") is None
    assert prepare("SNAP_B") is not None, "an unpinned rerun after the store's main branch moved must rerun"


def test_a_code_change_reruns(boundary_reader, two_zone_aoi, tmp_path, monkeypatch):
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    assert ctx.params["code"] == code_record()
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    assert _prepare(boundary_reader, two_zone_aoi, out, {}) is None
    monkeypatch.setattr(_common, "__version__", "9.9.9")
    assert _prepare(boundary_reader, two_zone_aoi, out, {}) is not None


def test_a_geotessera_upgrade_reruns(boundary_reader, two_zone_aoi, tmp_path, monkeypatch):
    # geotessera's merged view decides which group serves each row, and the pin admits 0.11.x
    # releases, so outputs made through another geotessera are not current.
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    assert ctx.params["code"]["geotessera_version"] == geotessera.__version__
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    assert _prepare(boundary_reader, two_zone_aoi, out, {}) is None
    monkeypatch.setattr(geotessera, "__version__", "0.11.99")
    assert _prepare(boundary_reader, two_zone_aoi, out, {}) is not None


def test_code_record_hashes_the_module_sources(tmp_path):
    source = tmp_path / "fake_app_module.py"
    source.write_text("THRESHOLD = 1\n")
    spec = importlib.util.spec_from_file_location("fake_app_module", source)
    module = importlib.util.module_from_spec(spec)
    first = code_record([module])
    assert first == code_record([module]) and first != code_record()
    assert set(first) == {"tessera_apps_version", "geotessera_version", "sources_sha256"}
    source.write_text("THRESHOLD = 2\n")
    assert code_record([module])["sources_sha256"] != first["sources_sha256"]


def test_finish_run_records_the_years_read_not_only_the_years_checked(boundary_reader, two_zone_aoi, tmp_path):
    ctx = prepare_run(
        "demoapp",
        two_zone_aoi,
        [2019, 2020, 2021],
        2020,
        2021,
        tmp_path / "out",
        {},
        validate=False,
        force=False,
        max_area_km2=None,
        reader=boundary_reader,
    )
    ctx.mark_read([2021, 2020])
    _write_outputs(ctx)
    report = finish_run(ctx, summary={}, extra={}, validation_written=False)
    prov = report["provenance"]
    assert prov["years"] == [2020, 2021]
    assert prov["years_checked"] == [2019, 2020, 2021]
    assert set(prov["pipeline_commits"]["10N"]) == {"2020", "2021"}


@pytest.mark.parametrize("how", ["file_in_the_path", "read_only"])
def test_prepare_run_refuses_an_unwritable_out_dir_before_any_chunk_read(
    boundary_store, boundary_reader, two_zone_aoi, tmp_path, monkeypatch, how
):
    _, log = boundary_store
    if how == "file_in_the_path":
        blocker = tmp_path / "results.txt"
        blocker.write_text("not a directory")
        out = blocker / "out"
    else:
        out = tmp_path / "out"

        def read_only(*args, **kwargs):
            raise OSError(30, "Read-only file system")

        monkeypatch.setattr(tempfile, "mkstemp", read_only)
    with pytest.raises(AppInputError, match="--out"):
        _prepare(boundary_reader, two_zone_aoi, out, {})
    assert log.chunk_keys() == []


def test_a_failing_preflight_keeps_the_previous_outputs(boundary_reader, two_zone_aoi, tmp_path):
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    seen = []

    def refuse(c):
        seen.append(c.plan.zone_names)
        raise AppInputError("bad reference")

    with pytest.raises(AppInputError, match="bad reference"):
        _prepare(boundary_reader, two_zone_aoi, out, {"k": 2}, preflight=refuse)
    assert seen == [["10N", "11N"]]
    assert ctx.paths.report.exists(), "begin_run must not run before the preflight succeeds"


def test_prepare_run_redoes_only_a_missing_validation(boundary_store, boundary_reader, two_zone_aoi, tmp_path):
    _, log = boundary_store
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    # Everything but validation.json is current: an app that can redo its validation is told so,
    # and the report stays until the app rewrites it.
    again = _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True, allow_validation_only=True)
    assert again is not None and again.validation_only and again.estimated_bytes == 0
    assert ctx.paths.report.exists()
    assert log.chunk_keys() == []
    # Not for an app that cannot, nor when forced, nor when other outputs are stale.
    assert not _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True, force=True).validation_only
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    plain = _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True)
    assert not plain.validation_only and not ctx.paths.report.exists()
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    finish_run(ctx, summary={}, extra={}, validation_written=False)
    stale = _prepare(boundary_reader, two_zone_aoi, out, {"k": 2}, validate=True, allow_validation_only=True)
    assert not stale.validation_only


def test_finish_validation_only_rewrites_the_report_last(boundary_reader, two_zone_aoi, tmp_path):
    out = tmp_path / "out"
    ctx = _prepare(boundary_reader, two_zone_aoi, out, {})
    _write_outputs(ctx)
    ctx.warn("Validation skipped: host unreachable")
    first = finish_run(ctx, summary={"validation": {"error": "x"}}, extra={}, validation_written=False)
    redo = _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True, allow_validation_only=True)
    export.write_json(redo.paths.validation, {"scores": {}})
    report = _common.finish_validation_only(
        redo, first, summary={"validation": {"iou": 0.7}}, warnings=[], validation_written=True
    )
    assert report["args_digest"] == first["args_digest"] and report["provenance"] == first["provenance"]
    assert report["summary"] == {"validation": {"iou": 0.7}} and report["warnings"] == []
    assert report["outputs"]["validation"] == export.file_sha256(redo.paths.validation)
    assert report["validation_redone"]["bytes_read"] == 0
    assert _prepare(boundary_reader, two_zone_aoi, out, {}, validate=True) is None
    with pytest.raises(ValueError, match="validation_only"):
        _common.finish_validation_only(ctx, first, summary={}, warnings=[], validation_written=False)


def test_prepare_run_estimates_memory_and_warns_before_any_chunk_read(
    boundary_store, boundary_reader, two_zone_aoi, tmp_path, monkeypatch, caplog
):
    _, log = boundary_store
    seen = []

    def estimate(plan):
        seen.append(plan.zone_names)
        return 5_000_000_000

    monkeypatch.setattr(_common, "available_memory", lambda: 8_000_000_000)
    with caplog.at_level(logging.INFO, logger="tessera_apps"):
        ctx = _prepare(boundary_reader, two_zone_aoi, tmp_path / "a", {}, memory_estimate=estimate)
    assert seen == [["10N", "11N"]] and ctx.estimated_memory == 5_000_000_000
    assert ctx.warnings == [] and "peak memory about 5.0 GB" in caplog.text
    monkeypatch.setattr(_common, "available_memory", lambda: 4_000_000_000)
    ctx = _prepare(boundary_reader, two_zone_aoi, tmp_path / "b", {}, memory_estimate=estimate)
    assert any("5.0 GB of memory" in w and "4.0 GB is available" in w for w in ctx.warnings)
    assert log.chunk_keys() == []


def test_available_memory_is_positive_or_unknown():
    mem = _common.available_memory()
    assert mem is None or mem > 0


# --------------------------------------------------------------------------------------------
# Review fixes: cleaning and growing across zones, references
# --------------------------------------------------------------------------------------------


def test_mosaic_and_clean_treats_outside_the_aoi_as_background(boundary_reader):
    # A small patch on the AOI's edge touches the mosaic's nodata (outside the AOI); nodata must not
    # join it to a large component that passes min_pixels.
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    corner = box(LON0 - 0.0045, LAT0 + 0.0022, LON0 - 0.0032, LAT0 + 0.0035) & TWO_ZONE
    raw = {z.zone: validate.reference_from_geometries(z, [corner]) for z in plan.zones}
    n = sum(int(m.sum()) for m in raw.values())
    assert 9 < n < 200
    cleaned, mosaic, _, _ = mosaic_and_clean(raw, plan, min_pixels=200)
    assert not any(m.any() for m in cleaned.values())
    assert not (mosaic == 1).any()


def test_mosaic_and_clean_opens_every_zone_and_the_seam(boundary_reader):
    # The multi-zone path applies the 3 x 3 opening like the single-zone path: one-pixel lines in
    # each zone and across the seam, and a spur on a patch, are removed; the patch itself is kept as
    # metrics.clean_mask keeps it.
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    raw = {z.zone: np.zeros(z.shape, dtype=bool) for z in plan.zones}
    for z in plan.zones:
        rows = np.flatnonzero(z.mask.any(axis=1))
        r = int(rows[len(rows) // 4])
        raw[z.zone][r] = z.mask[r]  # a full-width one-pixel line, which also crosses the seam
        assert raw[z.zone][r].sum() >= 20
    home = plan.zones[0]
    rr, cc = np.nonzero(home.mask)
    r0, c0 = int(np.median(rr)), int(np.min(cc)) + 5
    patch = np.zeros(home.shape, dtype=bool)
    patch[r0 : r0 + 10, c0 : c0 + 10] = True
    spur = np.zeros(home.shape, dtype=bool)
    spur[r0 + 4, c0 + 10 : c0 + 20] = True
    raw[home.zone] |= (patch | spur) & home.mask
    cleaned, _, _, _ = mosaic_and_clean(raw, plan, min_pixels=10)
    away = plan.zones[1]
    assert not cleaned[away.zone].any(), "a one-pixel line cannot survive a 3 x 3 opening"
    np.testing.assert_array_equal(cleaned[home.zone], patch & home.mask)


def test_grow_across_zones_reaches_over_the_seam(boundary_reader):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    z10, z11 = plan.zones
    seed = box(LON0 + 0.00005, LAT0 - 0.0002, LON0 + 0.00025, LAT0 + 0.0002)  # just east of 120 W
    masks = {z.zone: validate.reference_from_geometries(z, [seed]) for z in plan.zones}
    assert masks["11N"].any() and not masks["10N"].any()
    grown = _common.grow_across_zones(masks, plan, pixels=4)
    assert grown["10N"].any(), "growth crosses the seam"
    assert (grown["11N"] >= masks["11N"]).all() and not (grown["11N"] & ~z11.mask).any()
    # In one zone it is a plain dilation, clipped to the AOI.
    expected = ndimage.binary_dilation(masks["11N"], structure=np.ones((3, 3), bool), iterations=4) & z11.mask
    np.testing.assert_array_equal(
        _common.grow_across_zones({"11N": masks["11N"]}, _single(plan, z11), pixels=4)["11N"], expected
    )
    assert _common.grow_across_zones(masks, plan, pixels=0)["11N"].sum() == masks["11N"].sum()
    with pytest.raises(ValueError, match="pixels"):
        _common.grow_across_zones(masks, plan, pixels=-1)
    assert z10.zone == "10N"


def _single(plan, zone_plan):
    return type(plan)(geometry=plan.geometry, zones=[zone_plan])


def test_zone_values_from_raster_round_trips_the_home_zone(boundary_reader, tmp_path):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    rng = np.random.default_rng(3)
    values = {z.zone: np.where(z.mask, rng.random(z.shape), np.nan).astype(np.float32) for z in plan.zones}
    data, transform, crs = export.mosaic_zones(
        [export.GridArray(values[z.zone], z) for z in plan.zones], dst_zone="10N", nodata=np.nan
    )
    tif = export.write_geotiff(tmp_path / "m.tif", data, transform, crs)
    back = _common.zone_values_from_raster(tif, plan, band=1)
    np.testing.assert_array_equal(back["10N"], values["10N"])
    # The other zone takes the mosaic pixel under each centre: random speckle, where every pixel
    # differs from its neighbours, is the worst case for that round trip.
    away = plan.zones[1]
    same = back["11N"][away.mask] == values["11N"][away.mask]
    assert same.mean() > 0.9
    assert np.isnan(back["11N"][~away.mask]).all()


def test_raster_reference_counts_any_nonzero_value(boundary_reader, tmp_path):
    # A Hansen lossyear raster stores the loss year (21 for 2021), not 1.
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    z = plan.zones[0]
    patch = validate.reference_from_geometries(z, [box(LON0 - 0.002, LAT0 - 0.001, LON0, LAT0 + 0.001)])
    assert patch.sum() > 50
    data = np.where(patch, 21, 0).astype(np.uint8)
    data[0, 0] = 3
    tif = export.write_geotiff(tmp_path / "lossyear.tif", data, z.transform, z.grid.crs, nodata=255)
    np.testing.assert_array_equal(aoi_reference(z, tif), data.astype(bool) & z.mask)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("perimeter.geojson", b"<html><body>502 Bad Gateway</body></html>"),
        ("perimeter.geojson", b'{"type": "FeatureCollection", "features": [{"type": "Fea'),
        ("perimeter.tif", b"II*\x00 not really a tiff"),
    ],
)
def test_aoi_reference_names_an_unreadable_file(boundary_reader, tmp_path, name, content):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(AppInputError, match=name.replace(".", r"\.")):
        aoi_reference(plan.zones[0], path)


@pytest.mark.filterwarnings("ignore:'crs' was not provided:UserWarning")
def test_aoi_reference_refuses_a_projected_file_without_a_crs(boundary_reader, tmp_path):
    plan = plan_aoi(boundary_reader, TWO_ZONE)
    path = tmp_path / "projected.shp"
    gpd.GeoDataFrame(geometry=[box(700_000, 4_200_000, 701_000, 4_201_000)]).to_file(path)
    with pytest.raises(AppInputError, match="no CRS"):
        aoi_reference(plan.zones[0], path)
