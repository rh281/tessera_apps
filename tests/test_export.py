"""Offline tests for export.py: COG writer, zone mosaics, tables, JSON, previews, digests, provenance."""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import geotessera
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.crs
import shapely
from PIL import Image
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_bounds
from rio_cogeo.cogeo import cog_validate
from shapely.geometry import box

import tessera_apps
from tessera_apps import export
from tessera_apps.export import (
    GridArray,
    OutputPaths,
    args_digest,
    atomic_path,
    begin_run,
    file_sha256,
    mosaic_zones,
    output_paths,
    outputs_current,
    outputs_record,
    provenance_record,
    run_args_digest,
    to_jsonable,
    write_geotiff,
    write_json,
    write_preview_png,
    write_table,
)
from tessera_apps.io import ZoneGrid, ZonePlan, plan_aoi, zone_epsg
from tessera_apps.validate import write_validation_json

from .conftest import BOUNDARY

T10 = Affine(10.0, 0.0, 600_000.0, 0.0, -10.0, 4_300_000.0)


def _files(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


def _structure(path: Path) -> dict[str, str]:
    with rasterio.open(path) as ds:
        return ds.tags(ns="IMAGE_STRUCTURE")


# --------------------------------------------------------------------------------------------
# write_geotiff
# --------------------------------------------------------------------------------------------


def test_cog_float_round_trip_is_valid(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.random((2, 700, 600)).astype(np.float32)
    data[:, :20, :30] = np.nan
    path = write_geotiff(tmp_path / "a.tif", data, T10, CRS.from_epsg(32610), band_names=["distance", "similarity"])
    assert path == tmp_path / "a.tif"
    assert _files(tmp_path) == ["a.tif"]
    valid, errors, warnings = cog_validate(str(path))
    assert valid and errors == [] and warnings == []
    with rasterio.open(path) as ds:
        assert ds.driver == "GTiff"
        assert ds.count == 2 and ds.dtypes == ("float32", "float32")
        assert ds.transform == T10
        assert ds.crs.to_epsg() == 32610
        assert np.isnan(ds.nodata)
        assert ds.descriptions == ("distance", "similarity")
        assert ds.block_shapes == [(512, 512), (512, 512)]
        assert ds.compression.value == "DEFLATE"
        assert ds.overviews(1) == [2]
        np.testing.assert_array_equal(ds.read(), data)
    tags = _structure(path)
    assert tags["LAYOUT"] == "COG"
    assert tags["OVERVIEW_RESAMPLING"] == "AVERAGE"
    assert tags["PREDICTOR"] == "3"


def test_cog_bool_is_uint8_with_nodata_255_and_nearest_overviews(tmp_path):
    rng = np.random.default_rng(2)
    data = rng.random((1100, 900)) > 0.5
    path = write_geotiff(tmp_path / "m.tif", data, T10, 32610)
    assert cog_validate(str(path))[0]
    with rasterio.open(path) as ds:
        assert ds.count == 1 and ds.dtypes == ("uint8",)
        assert ds.nodata == 255
        np.testing.assert_array_equal(ds.read(1), data.astype(np.uint8))
        assert ds.overviews(1)
        assert set(np.unique(ds.read(1, out_shape=(550, 450)))) <= {0, 1}
    assert _structure(path)["OVERVIEW_RESAMPLING"] == "NEAREST"


def test_cog_bool_respects_explicit_nodata(tmp_path):
    path = write_geotiff(tmp_path / "m.tif", np.eye(40, dtype=bool), T10, 32610, nodata=0)
    with rasterio.open(path) as ds:
        assert ds.nodata == 0
        assert ds.read(1).sum() == 40


def test_cog_integer_data_keeps_dtype_and_has_no_nodata(tmp_path):
    data = np.arange(50 * 60, dtype=np.uint16).reshape(50, 60)
    path = write_geotiff(tmp_path / "c.tif", data, T10, "EPSG:32610")
    with rasterio.open(path) as ds:
        assert ds.dtypes == ("uint16",)
        assert ds.nodata is None
        np.testing.assert_array_equal(ds.read(1), data)
    assert _structure(path)["PREDICTOR"] == "2"


@pytest.mark.parametrize(
    ("dtype", "written"), [(np.float16, "float32"), (np.float32, "float32"), (np.float64, "float64")]
)
def test_cog_float_with_finite_nodata_replaces_nan(tmp_path, dtype, written):
    data = np.ones((30, 40), dtype=dtype)
    data[3, 4] = np.nan
    data[0, :5] = 0.25
    path = write_geotiff(tmp_path / "f.tif", data, T10, 32610, nodata=-9999.0)
    with rasterio.open(path) as ds:
        assert ds.dtypes == (written,)
        assert ds.nodata == -9999.0
        out = ds.read(1)
        valid = ds.read_masks(1)
    assert out[3, 4] == -9999.0
    assert np.isfinite(out).all()
    # GDAL's mask band (what GIS tools use) marks exactly the NaN pixel as nodata.
    assert valid[3, 4] == 0 and (valid > 0).sum() == data.size - 1
    np.testing.assert_array_equal(
        np.delete(out.ravel(), 3 * 40 + 4), np.delete(data.astype(written).ravel(), 3 * 40 + 4)
    )
    assert np.isnan(data[3, 4])  # the caller's array is untouched


def test_cog_float16_is_widened(tmp_path):
    path = write_geotiff(tmp_path / "h.tif", np.full((8, 8), 0.5, dtype=np.float16), T10, 32610)
    with rasterio.open(path) as ds:
        assert ds.dtypes == ("float32",)
        assert (ds.read(1) == 0.5).all()


def test_cog_masked_array_pixels_become_nodata(tmp_path):
    data = np.ma.masked_array(np.arange(20, dtype=np.int16).reshape(4, 5), mask=np.eye(4, 5, dtype=bool))
    with pytest.raises(ValueError, match="explicit nodata"):
        write_geotiff(tmp_path / "x.tif", data, T10, 32610)
    path = write_geotiff(tmp_path / "x.tif", data, T10, 32610, nodata=-1)
    with rasterio.open(path) as ds:
        out = ds.read(1)
    assert (out[np.eye(4, 5, dtype=bool)] == -1).all()
    assert (out[~np.eye(4, 5, dtype=bool)] == data.data[~np.eye(4, 5, dtype=bool)]).all()
    fdata = np.ma.masked_array(np.ones((4, 5), dtype=np.float32), mask=np.eye(4, 5, dtype=bool))
    with rasterio.open(write_geotiff(tmp_path / "y.tif", fdata, T10, 32610)) as ds:
        assert np.isnan(ds.read(1)[np.eye(4, 5, dtype=bool)]).all()


@pytest.mark.parametrize(
    "crs", [32610, "EPSG:32610", CRS.from_epsg(32610), rasterio.crs.CRS.from_epsg(32610), np.int64(32610)]
)
def test_cog_accepts_crs_forms(tmp_path, crs):
    path = write_geotiff(tmp_path / "c.tif", np.zeros((4, 4), dtype=np.uint8), T10, crs)
    with rasterio.open(path) as ds:
        assert ds.crs.to_epsg() == 32610


def test_cog_options_blocksize_compress_and_no_overviews(tmp_path):
    data = np.random.default_rng(3).random((700, 650)).astype(np.float32)
    path = write_geotiff(tmp_path / "o.tif", data, T10, 32610, blocksize=256, compress="lzw", overviews=False)
    valid, errors, _ = cog_validate(str(path))
    assert valid and errors == []
    with rasterio.open(path) as ds:
        assert ds.block_shapes == [(256, 256)]
        assert ds.compression.value == "LZW"
        assert ds.overviews(1) == []
    path = write_geotiff(tmp_path / "n.tif", data, T10, 32610, compress="none")
    with rasterio.open(path) as ds:
        assert ds.compression is None
    assert "PREDICTOR" not in _structure(path)


def test_cog_passes_rio_cogeo_cli(tmp_path):
    rio = Path(sys.executable).parent / "rio"
    if not rio.exists():
        pytest.skip("rio CLI not installed next to this interpreter")
    data = np.random.default_rng(4).random((900, 800)).astype(np.float32)
    path = write_geotiff(tmp_path / "cli.tif", data, T10, 32610)
    res = subprocess.run([str(rio), "cogeo", "validate", str(path)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
    assert "is a valid cloud optimized GeoTIFF" in res.stdout


@pytest.mark.parametrize(
    ("data", "kwargs", "exc"),
    [
        (np.zeros(5), {}, ValueError),
        (np.zeros((1, 2, 3, 4)), {}, ValueError),
        (np.zeros((0, 4)), {}, ValueError),
        (np.zeros((4, 4), dtype=np.uint8), {"nodata": 300}, ValueError),
        (np.zeros((4, 4), dtype=np.uint8), {"nodata": -1}, ValueError),
        (np.zeros((4, 4), dtype=np.int16), {"nodata": 1.5}, ValueError),
        (np.zeros((4, 4), dtype=np.int16), {"nodata": float("nan")}, ValueError),
        (np.zeros((4, 4), dtype=np.float32), {"nodata": 1e300}, ValueError),
        (np.zeros((2, 4, 4)), {"band_names": ["only one"]}, ValueError),
        (np.zeros((4, 4)), {"blocksize": 100}, ValueError),
        (np.zeros((4, 4)), {"blocksize": 0}, ValueError),
        (np.zeros((4, 4), dtype=np.complex64), {}, TypeError),
        (np.zeros((4, 4)), {"nodata": "0"}, TypeError),
    ],
)
def test_cog_rejects_bad_input_and_writes_nothing(tmp_path, data, kwargs, exc):
    with pytest.raises(exc):
        write_geotiff(tmp_path / "bad.tif", data, T10, 32610, **kwargs)
    assert _files(tmp_path) == []


def test_cog_rejects_non_affine_transform(tmp_path):
    with pytest.raises(TypeError):
        write_geotiff(tmp_path / "bad.tif", np.zeros((4, 4)), (10, 0, 0, 0, -10, 0), 32610)  # type: ignore[arg-type]


def test_cog_rewrite_replaces_and_leaves_no_temp_files(tmp_path):
    out = tmp_path / "sub" / "r.tif"
    write_geotiff(out, np.zeros((16, 16), dtype=np.uint8), T10, 32610)
    write_geotiff(out, np.full((16, 16), 7, dtype=np.uint8), T10, 32610)
    assert _files(out.parent) == ["r.tif"]
    with rasterio.open(out) as ds:
        assert (ds.read(1) == 7).all()


def _failing_copy(src, dst, options):
    Path(dst).write_bytes(b"II*\x00partial")
    raise RuntimeError("disk full")


def test_cog_failure_mid_write_leaves_no_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_cog_copy", _failing_copy)
    with pytest.raises(RuntimeError, match="disk full"):
        write_geotiff(tmp_path / "f.tif", np.zeros((16, 16), dtype=np.uint8), T10, 32610)
    assert _files(tmp_path) == []


def test_cog_failure_keeps_the_previous_file(tmp_path, monkeypatch):
    out = write_geotiff(tmp_path / "f.tif", np.full((16, 16), 3, dtype=np.uint8), T10, 32610)
    before = out.read_bytes()
    monkeypatch.setattr(export, "_cog_copy", _failing_copy)
    with pytest.raises(RuntimeError):
        write_geotiff(out, np.zeros((16, 16), dtype=np.uint8), T10, 32610)
    assert out.read_bytes() == before
    assert _files(tmp_path) == ["f.tif"]


def test_atomic_path_cleans_up_on_interrupt(tmp_path):
    target = tmp_path / "x.txt"
    target.write_text("old")
    with pytest.raises(KeyboardInterrupt), atomic_path(target) as tmp:
        assert tmp.parent == tmp_path and tmp.name.startswith(".x.txt.")
        tmp.write_text("new")
        raise KeyboardInterrupt
    assert target.read_text() == "old"
    assert _files(tmp_path) == ["x.txt"]
    with atomic_path(tmp_path / "deep" / "y.txt") as tmp:
        tmp.write_text("ok")
    assert (tmp_path / "deep" / "y.txt").read_text() == "ok"
    assert _files(tmp_path / "deep") == ["y.txt"]


# --------------------------------------------------------------------------------------------
# mosaic_zones
# --------------------------------------------------------------------------------------------

BASE = 1_000_000


def _ids(plan: ZonePlan, k: int) -> np.ndarray:
    """Every window pixel gets a value: zone code k * BASE plus its flat index (mask or not)."""
    h, w = plan.shape
    return (k * BASE + np.arange(h * w, dtype=np.int32)).reshape(h, w)


def _centres(transform: Affine, rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return transform.c + (cc + 0.5) * transform.a, transform.f + (rr + 0.5) * transform.e


def to_dst_from_ll(epsg: int):
    return Transformer.from_crs(4326, epsg, always_xy=True).transform


def _plan_centres(plan: ZonePlan, rr: np.ndarray, cc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _centres(plan.transform, rr, cc)


@pytest.mark.parametrize(("dst_zone", "dst_name"), [(None, "10N"), ("11n", "11N")])
def test_mosaic_two_zones_takes_every_pixel_once_from_its_nominal_zone(boundary_reader, dst_zone, dst_name):
    lon0, lat0 = BOUNDARY
    aoi = box(lon0 - 0.004, lat0 - 0.002, lon0 + 0.002, lat0 + 0.002)
    plan = plan_aoi(boundary_reader, aoi)
    assert plan.zone_names == ["10N", "11N"]
    by_zone = {p.zone: p for p in plan.zones}
    assert by_zone["10N"].n_pixels > by_zone["11N"].n_pixels
    code = {"10N": 1, "11N": 2}
    # Listed with the destination zone last: the destination pieces must still win collisions.
    pieces = sorted((GridArray(_ids(p, code[p.zone]), p) for p in plan.zones), key=lambda g: g.plan.zone == dst_name)
    out, transform, crs = mosaic_zones(pieces, dst_zone=dst_zone, nodata=-1)
    dst, other = by_zone[dst_name], by_zone["11N" if dst_name == "10N" else "10N"]
    assert crs == dst.grid.crs
    assert out.ndim == 2 and out.dtype == np.int32

    # Aligned to the destination zone's own grid, and covering every piece's window, reprojected.
    assert (transform.a, transform.b, transform.d, transform.e) == (10.0, 0.0, 0.0, -10.0)
    col_off = (transform.c - dst.grid.left) / 10
    row_off = (dst.grid.top - transform.f) / 10
    assert col_off == int(col_off) and row_off == int(row_off)
    left, bottom, right, top = array_bounds(out.shape[0], out.shape[1], transform)
    for p in plan.zones:
        b = transform_bounds(p.grid.crs, dst.grid.crs, *array_bounds(*p.shape, p.transform), densify_pts=21)
        assert left <= b[0] + 1e-6 and bottom <= b[1] + 1e-6 and b[2] <= right + 1e-6 and b[3] <= top + 1e-6

    # Destination-zone pixels are an exact copy of that piece's mask pixels, and nothing else of it.
    r0, c0 = dst.rows.start - int(row_off), dst.cols.start - int(col_off)
    assert r0 >= 0 and c0 >= 0
    sub = out[r0 : r0 + dst.shape[0], c0 : c0 + dst.shape[1]]
    np.testing.assert_array_equal(sub[dst.mask], _ids(dst, code[dst.zone])[dst.mask])
    zone_of = np.where(out >= 0, out // BASE, 0)
    assert (zone_of == code[dst.zone]).sum() == dst.n_pixels

    # Other-zone pixels: only that piece's mask pixels. Each lands within half a pixel diagonal of its
    # source (nearest neighbour), except seam pixels next to the destination zone, which take the
    # nearest mask pixel of their own zone, one pixel further at most.
    rr, cc = np.nonzero(zone_of == code[other.zone])
    n_other = rr.size
    assert abs(n_other - other.n_pixels) <= 0.03 * other.n_pixels
    flat = out[rr, cc] - code[other.zone] * BASE
    sr, sc = np.divmod(flat, other.shape[1])
    assert other.mask[sr, sc].all()
    to_dst = Transformer.from_crs(other.grid.epsg, dst.grid.epsg, always_xy=True)
    sx, sy = to_dst.transform(*_plan_centres(other, sr, sc))
    dx, dy = _centres(transform, rr, cc)
    dist = np.hypot(sx - dx, sy - dy)
    far = dist > 7.2
    assert dist.max() <= 7.2 + 14.2
    assert far.sum() <= 0.05 * n_other
    home = np.pad(zone_of == code[dst.zone], 1)
    next_to_home = home[rr, cc + 1] | home[rr + 2, cc + 1] | home[rr + 1, cc] | home[rr + 1, cc + 2]
    assert next_to_home[far].all()
    assert np.unique(flat).size >= 0.95 * other.n_pixels

    # Every ground pixel exactly once: no holes inside the AOI, nothing outside it, and each value
    # from the zone whose band holds the pixel centre.
    filled = out >= 0
    assert abs(int(filled.sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels
    to_ll = Transformer.from_crs(dst.grid.epsg, 4326, always_xy=True)
    aoi_dst = shapely.transform(shapely.segmentize(aoi, 1e-4), to_dst_from_ll(dst.grid.epsg), interleaved=False)
    grr, gcc = np.indices(out.shape)
    gx, gy = _centres(transform, grr, gcc)
    inner = shapely.contains_xy(aoi_dst.buffer(-15), gx, gy)
    outer = shapely.contains_xy(aoi_dst.buffer(15), gx, gy)
    assert filled[inner].all()
    assert not filled[~outer].any()
    lon, _ = to_ll.transform(gx[filled], gy[filled])
    zones_filled = zone_of[filled]
    assert (zones_filled[lon < lon0 - 1e-7] == 1).all()
    assert (zones_filled[lon >= lon0 + 1e-7] == 2).all()


def test_mosaic_default_zone_is_the_one_with_most_pixels(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.001, lat0 - 0.002, lon0 + 0.004, lat0 + 0.002))
    by_zone = {p.zone: p for p in plan.zones}
    assert by_zone["11N"].n_pixels > by_zone["10N"].n_pixels
    pieces = [GridArray(np.ones(p.shape, dtype=np.float32), p) for p in plan.zones]
    _, transform, crs = mosaic_zones(pieces, nodata=np.nan)
    assert crs.to_epsg() == 32611
    assert (transform.c - by_zone["11N"].grid.left) % 10 == 0


def test_mosaic_multiband_float_and_nan_nodata(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.003, lat0 - 0.002, lon0 + 0.003, lat0 + 0.002))
    pieces = []
    for k, p in enumerate(plan.zones, start=1):
        data = np.stack([np.full(p.shape, k, np.float32), np.full(p.shape, 10 * k, np.float32)])
        pieces.append(GridArray(data, p))
    out, _, _ = mosaic_zones(pieces, nodata=np.nan, resampling=Resampling.nearest)
    assert out.ndim == 3 and out.shape[0] == 2 and out.dtype == np.float32
    filled = np.isfinite(out[0])
    np.testing.assert_array_equal(np.isfinite(out[1]), filled)
    np.testing.assert_array_equal(out[1][filled], 10 * out[0][filled])
    assert set(np.unique(out[0][filled])) == {1.0, 2.0}
    assert abs(int(filled.sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels


def test_mosaic_nodata_inside_the_nominal_zone_is_not_back_filled(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.002, lon0 + 0.002, lat0 + 0.002))
    by_zone = {p.zone: p for p in plan.zones}
    p10, p11 = by_zone["10N"], by_zone["11N"]
    eleven = GridArray(np.full(p11.shape, 2, np.float32), p11)
    valid, _, _ = mosaic_zones([GridArray(np.ones(p10.shape, np.float32), p10), eleven], nodata=np.nan)
    # 10N has no valid value anywhere (for example no embedding there): its pixels stay nodata and
    # are not taken from the 11N piece next door, so 11N covers exactly what it covers above.
    empty, _, _ = mosaic_zones([GridArray(np.full(p10.shape, np.nan, np.float32), p10), eleven], nodata=np.nan)
    assert (valid == 1).sum() == p10.n_pixels
    np.testing.assert_array_equal(np.isfinite(empty), valid == 2)


def test_mosaic_earlier_piece_wins_within_another_zone(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.002, lon0 + 0.002, lat0 + 0.002))
    by_zone = {p.zone: p for p in plan.zones}
    p10, p11 = by_zone["10N"], by_zone["11N"]
    # Two overlapping 11N pieces (say overlapping blocks): the one listed first keeps every pixel.
    pieces = [
        GridArray(np.full(p11.shape, 2, np.int32), p11),
        GridArray(np.full(p10.shape, 1, np.int32), p10),
        GridArray(np.full(p11.shape, 3, np.int32), p11),
    ]
    out, _, crs = mosaic_zones(pieces, dst_zone="10N", nodata=0)  # 11N would win the pixel count
    assert crs.to_epsg() == 32610
    assert set(np.unique(out)) == {0, 1, 2}
    assert abs(int((out == 2).sum()) - p11.n_pixels) <= 0.03 * p11.n_pixels


def test_mosaic_never_fills_a_pixel_from_another_zones_band(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.002, lon0 + 0.002, lat0 + 0.002))
    by_zone = {p.zone: p for p in plan.zones}
    p10, p11 = by_zone["10N"], by_zone["11N"]
    # Drop the eastmost 10N pixel of every row: those pixels lie in the 10N band but no piece covers
    # them, so they stay nodata. 11N must not fill them, by nearest neighbour or at the seam.
    mask = p10.mask.copy()
    rows = np.flatnonzero(mask.any(axis=1))
    last = np.array([np.flatnonzero(mask[r])[-1] for r in rows])
    mask[rows, last] = False
    pieces = [
        GridArray(np.full(p10.shape, 1, np.int32), dataclasses.replace(p10, mask=mask)),
        GridArray(np.full(p11.shape, 2, np.int32), p11),
    ]
    out, transform, _ = mosaic_zones(pieces, nodata=0)
    r0 = round((p10.grid.top - transform.f) / 10)
    c0 = round((transform.c - p10.grid.left) / 10)
    sub = out[p10.rows.start - r0 :, p10.cols.start - c0 :]
    assert (sub[rows, last] == 0).all()
    assert (sub[:, : p10.shape[1]][mask] == 1).all()


class _GridOnlyReader:
    """Live-store zone grids and nothing else: enough for plan_aoi anywhere."""

    def grid(self, zone):
        north = zone.endswith("N")
        return ZoneGrid(
            zone=zone,
            epsg=zone_epsg(zone),
            left=163_840.0,
            top=9_338_880.0 if north else 10_014_720.0,
            height=933_888 if north else 890_880,
            width=67_584,
        )


@pytest.mark.parametrize("dst_zone", [None, "31S"])
def test_mosaic_across_the_equator_takes_each_hemisphere_from_its_group(dst_zone):
    aoi = box(3.0, -0.002, 3.004, 0.003)
    plan = plan_aoi(_GridOnlyReader(), aoi)  # type: ignore[arg-type]
    assert plan.zone_names == ["31N", "31S"]
    code = {"31N": 1, "31S": 2}
    pieces = [GridArray(_ids(p, code[p.zone]), p) for p in plan.zones]
    out, transform, crs = mosaic_zones(pieces, dst_zone=dst_zone, nodata=-1)
    assert crs.to_epsg() == (32631 if dst_zone is None else 32731)
    filled = out >= 0
    assert abs(int(filled.sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels
    rr, cc = np.nonzero(filled)
    _, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(*_centres(transform, rr, cc))
    zone_of = out[rr, cc] // BASE
    assert (zone_of[lat >= 0] == 1).all()
    assert (zone_of[lat < 0] == 2).all()
    assert (zone_of == 2).sum() == plan.zones[1].n_pixels  # the two grids align: every pixel lands once


def test_mosaic_bool_pieces_become_uint8(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.003, lat0 - 0.002, lon0 + 0.003, lat0 + 0.002))
    pieces = [GridArray(np.ones(p.shape, dtype=bool), p) for p in plan.zones]
    out, _, _ = mosaic_zones(pieces, nodata=255)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) == {1, 255}
    with pytest.raises(ValueError):
        mosaic_zones(pieces, nodata=np.nan)
    with pytest.raises(ValueError):
        mosaic_zones(pieces, nodata=-1)


def test_mosaic_single_piece_is_returned_masked(boundary_reader):
    lon0, lat0 = BOUNDARY
    (p,) = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.002, lon0 - 0.0005, lat0 + 0.002)).zones
    data = np.random.default_rng(5).random(p.shape).astype(np.float32)
    keep = data.copy()
    out, transform, crs = mosaic_zones([GridArray(data, p)], nodata=np.nan)
    np.testing.assert_array_equal(data, keep)
    assert out.shape == p.shape
    assert transform == p.transform
    assert crs == p.grid.crs
    np.testing.assert_array_equal(out[p.mask], data[p.mask])
    assert np.isnan(out[~p.mask]).all()
    assert (~p.mask).any()


def _plan(grid: ZoneGrid, rows: slice, cols: slice, mask: np.ndarray | None = None) -> ZonePlan:
    shape = (rows.stop - rows.start, cols.stop - cols.start)
    return ZonePlan(
        zone=grid.zone, grid=grid, rows=rows, cols=cols, mask=np.ones(shape, bool) if mask is None else mask
    )


GRID10 = ZoneGrid(zone="10N", epsg=32610, left=500_000.0, top=4_300_000.0, height=1000, width=1000)


def test_mosaic_same_zone_pieces_paste_exactly_and_first_wins():
    mask_a = np.ones((4, 5), bool)
    mask_a[3, 4] = False  # inside the overlap with B: B fills it
    a = GridArray(np.full((4, 5), 1, np.uint8), _plan(GRID10, slice(10, 14), slice(20, 25), mask_a))
    b = GridArray(np.full((4, 5), 2, np.uint8), _plan(GRID10, slice(12, 16), slice(23, 28)))
    out, transform, crs = mosaic_zones([a, b], nodata=0)
    assert out.shape == (6, 8)
    assert transform == GRID10.window_transform(10, 20)
    assert crs == GRID10.crs
    expect = np.zeros((6, 8), np.uint8)
    expect[2:6, 3:8] = 2
    expect[0:4, 0:5][mask_a] = 1
    np.testing.assert_array_equal(out, expect)
    assert out[3, 4] == 2 and out[2, 3] == 1


def _column_blocks(piece: GridArray, cuts: list[int]) -> list[GridArray]:
    """Split a piece into column blocks at window columns ``cuts`` (each block keeps its own mask)."""
    plan = piece.plan
    edges = [0, *cuts, plan.shape[1]]
    blocks = []
    for a, b in itertools.pairwise(edges):
        sub = ZonePlan(
            zone=plan.zone,
            grid=plan.grid,
            rows=plan.rows,
            cols=slice(plan.cols.start + a, plan.cols.start + b),
            mask=plan.mask[:, a:b].copy(),
        )
        blocks.append(GridArray(piece.data[..., a:b], sub))
    return blocks


def _inner_aoi(aoi, out: np.ndarray, transform: Affine, crs: CRS, metres: float = 15) -> np.ndarray:
    """Output pixels whose centre lies more than ``metres`` inside the AOI."""
    to_dst = Transformer.from_crs(4326, crs, always_xy=True).transform
    aoi_dst = shapely.transform(shapely.segmentize(aoi, 1e-4), to_dst, interleaved=False)
    idx = np.indices(out.shape[-2:])
    return shapely.contains_xy(aoi_dst.buffer(-metres), *_centres(transform, idx[0], idx[1]))


@pytest.mark.parametrize(("dst_name", "away_name"), [("11N", "10N"), ("10N", "11N")])
def test_mosaic_away_zone_split_into_column_blocks_matches_the_whole_piece(boundary_reader, dst_name, away_name):
    # The docstring allows blocks of one zone. Split the away zone next to the slanted 120 W seam,
    # where a block edge used to cut the seam back-fill and leave nodata holes inside the AOI.
    lon0, lat0 = BOUNDARY
    aoi = box(lon0 - 0.004, lat0 - 0.004, lon0 + 0.004, lat0 + 0.004)
    plan = plan_aoi(boundary_reader, aoi)
    code = {"10N": 1, "11N": 2}
    whole = {p.zone: GridArray(_ids(p, code[p.zone]), p) for p in plan.zones}
    ref, transform, crs = mosaic_zones([whole["10N"], whole["11N"]], dst_zone=dst_name, nodata=-1)
    inner = _inner_aoi(aoi, ref, transform, crs)
    assert inner.sum() > 1000 and (ref[inner] >= 0).all()
    width = whole[away_name].plan.shape[1]
    # The seam is the east edge of the 10N window and the west edge of the 11N one.
    cuts = range(width - 8, width) if away_name == "10N" else range(1, 9)
    for cut in cuts:
        pieces = [whole[dst_name], *_column_blocks(whole[away_name], [cut])]
        out, t2, c2 = mosaic_zones(pieces, dst_zone=dst_name, nodata=-1)
        assert (t2, c2) == (transform, crs)
        holes = inner & (out < 0)
        assert not holes.any(), f"cut at column {cut}: {int(holes.sum())} AOI pixels left as nodata"
        np.testing.assert_array_equal(out, ref)
    # Many narrow blocks, listed out of order and mixed with the destination piece.
    blocks = _column_blocks(whole[away_name], list(range(3, width, 3)))
    out, _, _ = mosaic_zones([*blocks[::2], whole[dst_name], *blocks[1::2]], dst_zone=dst_name, nodata=-1)
    np.testing.assert_array_equal(out, ref)


def test_mosaic_blocks_from_zone_plan_blocks_leave_no_seam_holes(boundary_reader):
    # Chunk-aligned, AOI-trimmed blocks as iter_plan_blocks reads them, in both zones.
    lon0, lat0 = BOUNDARY
    aoi = box(lon0 - 0.004, lat0 - 0.004, lon0 + 0.004, lat0 + 0.004)
    plan = plan_aoi(boundary_reader, aoi)
    pieces = []
    for p in plan.zones:
        values = np.full(p.shape, 1 if p.zone == "10N" else 2, np.uint8)
        for rows, cols in p.blocks((8, 8), (16, 16)):
            sub = ZonePlan(zone=p.zone, grid=p.grid, rows=rows, cols=cols, mask=p.submask(rows, cols).copy())
            r = slice(rows.start - p.rows.start, rows.stop - p.rows.start)
            c = slice(cols.start - p.cols.start, cols.stop - p.cols.start)
            pieces.append(GridArray(values[r, c], sub))
    assert len(pieces) > 8
    whole = [GridArray(np.full(p.shape, 1 if p.zone == "10N" else 2, np.uint8), p) for p in plan.zones]
    for dst in ("10N", "11N"):
        ref, transform, crs = mosaic_zones(whole, dst_zone=dst, nodata=255)
        out, t2, _ = mosaic_zones(pieces, dst_zone=dst, nodata=255)
        # The blocks may cover a smaller window than the plans: compare on the reference grid.
        dr, dc = round((t2.f - transform.f) / -10), round((t2.c - transform.c) / 10)
        assert dr >= 0 and dc >= 0
        grown = np.full_like(ref, 255)
        grown[dr : dr + out.shape[0], dc : dc + out.shape[1]] = out
        inner = _inner_aoi(aoi, ref, transform, crs)
        assert not (inner & (grown == 255)).any()
        np.testing.assert_array_equal(grown, ref)


def test_mosaic_honours_the_resampling_argument(boundary_reader):
    # A smooth east-west ramp in the away zone: nearest copies source values, bilinear blends them.
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.004, lon0 + 0.004, lat0 + 0.004))
    z = {p.zone: p for p in plan.zones}
    p11 = z["11N"]
    ramp = (10.0 + 0.5 * np.arange(p11.shape[1], dtype=np.float32))[None, :].repeat(p11.shape[0], axis=0)
    ramp += 0.01 * np.arange(p11.shape[0], dtype=np.float32)[:, None]
    pieces = [GridArray(np.ones(z["10N"].shape, np.float32), z["10N"]), GridArray(ramp, p11)]
    results = {}
    for method in (Resampling.nearest, Resampling.bilinear):
        out, _, _ = mosaic_zones(pieces, dst_zone="10N", nodata=np.nan, resampling=method)
        results[method] = out[np.isfinite(out) & (out >= 5)]
    near, bil = results[Resampling.nearest], results[Resampling.bilinear]
    source = ramp[p11.mask]
    assert near.size >= 0.97 * p11.n_pixels
    assert np.isin(near, source).all()
    blended = ~np.isin(bil, ramp)
    assert blended.mean() > 0.5  # most pixels fall between source centres
    assert source.min() - 1e-4 <= bil.min() and bil.max() <= source.max() + 1e-4


def test_mosaic_keeps_away_zone_pixels_with_some_nodata_bands(boundary_reader):
    # A year stack where one year is missing on every other 11N column: the other year must survive
    # exactly as if it had been mosaicked alone, and the missing year stays missing.
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.004, lat0 - 0.002, lon0 + 0.002, lat0 + 0.002))
    z = {p.zone: p for p in plan.zones}
    p10, p11 = z["10N"], z["11N"]
    first = {k: np.full(p.shape, 1.0 if k == "10N" else 2.0, np.float32) for k, p in z.items()}
    second = {k: np.full(p.shape, 10.0 if k == "10N" else 20.0, np.float32) for k, p in z.items()}
    second["11N"][:, ::2] = np.nan
    single, _, _ = mosaic_zones([GridArray(first[k], p) for k, p in z.items()], dst_zone="10N", nodata=np.nan)
    stack = [GridArray(np.stack([first[k], second[k]]), p) for k, p in z.items()]
    out, _, _ = mosaic_zones(stack, dst_zone="10N", nodata=np.nan)
    np.testing.assert_array_equal(out[0], single)
    n11 = int((out[0] == 2).sum())
    assert abs(n11 - p11.n_pixels) <= 0.03 * p11.n_pixels
    eleven = out[0] == 2
    assert 0.3 < np.isnan(out[1][eleven]).mean() < 0.7
    np.testing.assert_array_equal(out[1][out[0] == 1], 10.0)
    assert (out[1][eleven & ~np.isnan(out[1])] == 20).all()
    assert (out[0] == 1).sum() == p10.n_pixels


def test_mosaic_rejects_bad_input(boundary_reader):
    p = _plan(GRID10, slice(0, 4), slice(0, 4))
    with pytest.raises(ValueError):
        mosaic_zones([], nodata=0)
    with pytest.raises(ValueError):
        GridArray(np.zeros((3, 4)), p)
    with pytest.raises(ValueError):
        GridArray(np.zeros(4), p)
    two_band = GridArray(np.zeros((2, 4, 4)), p)
    one_band = GridArray(np.zeros((4, 4)), _plan(GRID10, slice(4, 8), slice(0, 4)))
    with pytest.raises(ValueError, match="band counts"):
        mosaic_zones([two_band, one_band], nodata=np.nan)
    with pytest.raises(ValueError, match="dst_zone"):
        mosaic_zones([one_band], dst_zone="11N", nodata=np.nan)
    shifted = dataclasses.replace(GRID10, left=GRID10.left + 5)
    with pytest.raises(ValueError, match="different grids"):
        mosaic_zones([one_band, GridArray(np.zeros((4, 4)), _plan(shifted, slice(0, 4), slice(0, 4)))], nodata=0)
    # Pieces of a zone other than the destination must share a grid too: they are pasted together.
    grid11 = ZoneGrid(zone="11N", epsg=32611, left=200_000.0, top=4_300_000.0, height=1000, width=1000)
    away = [
        GridArray(np.zeros((4, 4)), _plan(grid11, slice(0, 4), slice(0, 4))),
        GridArray(np.zeros((4, 4)), _plan(dataclasses.replace(grid11, top=grid11.top + 5), slice(0, 4), slice(0, 4))),
    ]
    with pytest.raises(ValueError, match="11N are on different grids"):
        mosaic_zones([one_band, *away], dst_zone="10N", nodata=0)
    ints = GridArray(np.zeros((4, 4), np.int16), p)
    with pytest.raises(ValueError):
        mosaic_zones([ints], nodata=np.nan)
    with pytest.raises(ValueError):
        mosaic_zones([ints], nodata=40_000)


# --------------------------------------------------------------------------------------------
# Tables and JSON
# --------------------------------------------------------------------------------------------


def test_write_table_round_trip(tmp_path):
    df = pd.DataFrame(
        {
            "year": [2021, 2022, 2023],
            "recovery": [0.1, np.nan, 0.75],
            "label": ["a", "b", "c"],
            "flag": [True, False, True],
        }
    )
    parquet, csv = write_table(df, tmp_path / "out" / "wildfire.v1_summary")
    assert parquet == tmp_path / "out" / "wildfire.v1_summary.parquet"
    assert csv == tmp_path / "out" / "wildfire.v1_summary.csv"
    assert _files(tmp_path / "out") == ["wildfire.v1_summary.csv", "wildfire.v1_summary.parquet"]
    pd.testing.assert_frame_equal(pd.read_parquet(parquet), df, check_dtype=False)
    back = pd.read_csv(csv)
    assert list(back.columns) == list(df.columns)
    np.testing.assert_array_equal(back["year"], df["year"])
    np.testing.assert_allclose(back["recovery"], df["recovery"])
    assert csv.read_text().splitlines()[0] == "year,recovery,label,flag"


def test_write_table_keeps_a_meaningful_index(tmp_path):
    df = pd.DataFrame({"area_ha": [1.5, 2.5]}, index=pd.Index([2020, 2022], name="year"))
    parquet, csv = write_table(df, tmp_path / "t")
    assert list(pd.read_parquet(parquet).columns) == ["year", "area_ha"]
    assert list(pd.read_csv(csv).columns) == ["year", "area_ha"]
    assert pd.read_csv(csv)["year"].tolist() == [2020, 2022]
    plain = pd.DataFrame({"x": [1, 2]})
    _, csv2 = write_table(plain, tmp_path / "p")
    assert list(pd.read_csv(csv2).columns) == ["x"]


def test_write_table_keeps_a_named_multiindex(tmp_path):
    # A per-zone, per-year summary from groupby: both key levels become leading columns, in order.
    df = pd.DataFrame(
        {"zone": ["10N", "10N", "11N", "11N"], "year": [2020, 2022, 2020, 2022], "area_ha": [1.0, 2.0, 3.0, 4.5]}
    )
    summary = df.groupby(["zone", "year"]).sum()
    assert summary.index.nlevels == 2
    parquet, csv = write_table(summary, tmp_path / "s")
    for back in (pd.read_parquet(parquet), pd.read_csv(csv)):
        assert list(back.columns) == ["zone", "year", "area_ha"]
        assert back["zone"].tolist() == ["10N", "10N", "11N", "11N"]
        assert back["year"].tolist() == [2020, 2022, 2020, 2022]
        assert back["area_ha"].tolist() == [1.0, 2.0, 3.0, 4.5]
    assert csv.read_text().splitlines()[0] == "zone,year,area_ha"


def test_write_table_failure_leaves_no_temp_file(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise OSError("no space")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    with pytest.raises(OSError):
        write_table(pd.DataFrame({"x": [1]}), tmp_path / "t")
    assert _files(tmp_path) == ["t.parquet"]


class Colour(enum.Enum):
    """An enum value for JSON."""

    RED = "red"


@dataclasses.dataclass(frozen=True)
class Scores:
    """A dataclass for JSON."""

    f1: float | None
    n: int


def test_write_json_handles_numpy_paths_dates_and_nan(tmp_path):
    obj = {
        "b": np.int64(3),
        "a": np.float32(0.5),
        "flag": np.bool_(True),
        "arr": np.array([[1.0, np.nan], [np.inf, 2.0]]),
        "ints": np.arange(3, dtype=np.uint8),
        "path": Path("/data/aoi.geojson"),
        "when": dt.datetime(2026, 10, 15, 12, 30, tzinfo=dt.UTC),
        "day": dt.date(2021, 8, 14),
        "nan": float("nan"),
        "ninf": -np.inf,
        "years": {2022: 0.5, np.int64(2020): 1.0},
        "tuple": (1, 2),
        "set": {3, 1, 2},
        "scores": Scores(f1=float("nan"), n=4),
        "colour": Colour.RED,
        "d64": np.datetime64("2021-08-14"),
        "d64s": np.array(["2021-01-01", "NaT"], dtype="datetime64[D]"),
        "ts": pd.Timestamp("2021-08-14T00:00:00"),
        "nat": pd.NaT,
        "na": pd.NA,
        "none": None,
    }
    path = write_json(tmp_path / "r.json", obj)
    text = path.read_text(encoding="utf-8")
    assert text.startswith('{\n "a": 0.5,\n "arr": [\n  [\n')
    assert text.endswith("}\n")
    assert "NaN" not in text and "Infinity" not in text
    back = json.loads(text)
    assert list(back) == sorted(back)
    assert back["b"] == 3 and back["a"] == 0.5 and back["flag"] is True
    assert back["arr"] == [[1.0, None], [None, 2.0]]
    assert back["ints"] == [0, 1, 2]
    assert back["path"] == "/data/aoi.geojson"
    assert back["when"] == "2026-10-15T12:30:00+00:00"
    assert back["day"] == "2021-08-14"
    assert back["nan"] is None and back["ninf"] is None
    assert back["years"] == {"2020": 1.0, "2022": 0.5}
    assert back["tuple"] == [1, 2] and back["set"] == [1, 2, 3]
    assert back["scores"] == {"f1": None, "n": 4}
    assert back["colour"] == "red"
    assert back["d64"] == "2021-08-14"
    assert back["d64s"] == ["2021-01-01", None]
    assert back["ts"] == "2021-08-14T00:00:00"
    assert back["nat"] is None and back["na"] is None and back["none"] is None
    assert _files(tmp_path) == ["r.json"]


def test_to_jsonable_rejects_what_json_cannot_hold():
    with pytest.raises(TypeError):
        to_jsonable({"x": object()})
    with pytest.raises(TypeError):
        to_jsonable(1 + 2j)
    with pytest.raises(TypeError):
        to_jsonable({(1, 2): "tuple key"})
    with pytest.raises(ValueError, match="JSON key"):
        to_jsonable({1: "a", "1": "b"})
    assert to_jsonable({True: 1, 1.5: 2, Path("p"): 3}) == {"true": 1, "1.5": 2, "p": 3}


def test_write_json_failure_keeps_old_file(tmp_path):
    path = write_json(tmp_path / "r.json", {"a": 1})
    with pytest.raises(TypeError):
        write_json(path, {"a": object()})
    assert json.loads(path.read_text()) == {"a": 1}
    assert _files(tmp_path) == ["r.json"]


# --------------------------------------------------------------------------------------------
# Preview PNG
# --------------------------------------------------------------------------------------------


def _png(path: Path) -> np.ndarray:
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _count(px: np.ndarray, rgb: tuple[int, int, int]) -> int:
    return int((px == np.array(rgb, dtype=np.uint8)).all(axis=-1).sum())


def test_preview_rgb_is_drawn_as_is(tmp_path):
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = (255, 0, 0)
    img[:, 32:] = (0, 0, 255)
    path = write_preview_png(tmp_path / "p.png", img, title="PCA 2020", legend="false colour")
    px = _png(path)
    red, blue = _count(px, (255, 0, 0)), _count(px, (0, 0, 255))
    assert red > 50_000 and blue > 50_000
    assert abs(red - blue) < 0.05 * red
    assert _files(tmp_path) == ["p.png"]


def test_preview_rgb_pixels_are_not_resampled(tmp_path):
    img = np.random.default_rng(6).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    px = _png(write_preview_png(tmp_path / "r.png", img))
    source = {tuple(c) for c in img.reshape(-1, 3).tolist()}
    flat = px.reshape(-1, 3)
    exact = np.fromiter((tuple(c) in source for c in flat.tolist()), dtype=bool, count=len(flat))
    # Nearest-neighbour upscaling draws each source pixel as a block of its exact colour.
    assert exact.sum() >= 0.8 * 512 * 512
    assert len({tuple(c) for c in flat[exact].tolist()}) >= 0.95 * len(source)


def test_preview_scalar_nan_is_grey(tmp_path):
    grey = tuple(round(255 * v) for v in export.PREVIEW_NAN_RGB)
    img = np.linspace(0, 1, 40 * 60).reshape(40, 60)
    path = write_preview_png(tmp_path / "a.png", img, title="distance", legend="cosine distance", cmap="magma")
    assert _count(_png(path), grey) < 200  # anti-aliased text only
    img[:, :30] = np.nan
    path = write_preview_png(tmp_path / "b.png", img, vmin=0, vmax=1)
    assert _count(_png(path), grey) > 20_000


def test_preview_bool_and_all_nan(tmp_path):
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 10:20] = True
    assert _png(write_preview_png(tmp_path / "m.png", mask, cmap="gray")).size
    assert _png(write_preview_png(tmp_path / "n.png", np.full((10, 10), np.nan))).size


def _near(px: np.ndarray, rgb: tuple[int, ...], tol: int = 2) -> int:
    """Pixels within ``tol`` of ``rgb`` in every channel."""
    return int((np.abs(px.astype(int) - np.array(rgb[:3], dtype=int)).max(axis=-1) <= tol).sum())


def _cmap(name: str, x: float) -> tuple[int, ...]:
    import matplotlib as mpl

    return tuple(int(v) for v in mpl.colormaps[name](x, bytes=True)[:3])


# A 512 x 512 drawing area: more than 200,000 pixels of one colour means the image, not the colorbar.
IMAGE_PIXELS = 200_000


@pytest.mark.parametrize(("value", "colour"), [(True, 1.0), (False, 0.0)])
def test_preview_uniform_bool_mask_uses_its_own_colour(tmp_path, value, colour):
    # A fully burned (or fully unburned) AOI must not be drawn like the other class or the middle.
    px = _png(write_preview_png(tmp_path / "m.png", np.full((50, 50), value)))
    assert _near(px, _cmap("viridis", colour)) > IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 1.0 - colour)) < 2_000  # colorbar only
    assert _near(px, _cmap("viridis", 0.5)) < 2_000


def test_preview_uses_the_given_cmap(tmp_path):
    px = _png(write_preview_png(tmp_path / "m.png", np.ones((50, 50), dtype=bool), cmap="magma"))
    assert _near(px, _cmap("magma", 1.0)) > IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 1.0), tol=8) == 0
    ramp = np.linspace(0, 1, 50 * 50).reshape(50, 50)
    px = _png(write_preview_png(tmp_path / "r.png", ramp, cmap="cividis"))
    assert _near(px, _cmap("cividis", 0.0)) > 500 and _near(px, _cmap("cividis", 1.0)) > 500
    assert _near(px, _cmap("viridis", 0.0), tol=8) == 0


def test_preview_honours_explicit_limits(tmp_path):
    # Data narrower than the limits must not be stretched over the whole colormap.
    img = np.full((50, 50), 0.2)
    img[:, 25:] = 0.4
    px = _png(write_preview_png(tmp_path / "l.png", img, vmin=0.0, vmax=1.0))
    assert _near(px, _cmap("viridis", 0.2)) > 0.4 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 0.4)) > 0.4 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 0.0)) < 2_000 and _near(px, _cmap("viridis", 1.0)) < 2_000
    # Without limits the same data spans the colormap: the check above can tell the two apart.
    px = _png(write_preview_png(tmp_path / "a.png", img))
    assert _near(px, _cmap("viridis", 0.0)) > 0.4 * IMAGE_PIXELS
    # A one-sided limit keeps the data maximum as the other end.
    px = _png(write_preview_png(tmp_path / "v.png", img, vmin=0.0))
    assert _near(px, _cmap("viridis", 0.5)) > 0.4 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 1.0)) > 0.4 * IMAGE_PIXELS


def _burn_mosaic() -> np.ndarray:
    """A uint8 burned-mask mosaic as mosaic_zones gives for bool pieces: 0, 1 and nodata 255."""
    m = np.zeros((50, 50), np.uint8)
    m[:, 20:35] = 1
    m[:, 35:] = 255
    return m


def test_preview_integer_nodata_is_grey_and_keeps_classes_apart(tmp_path):
    grey = tuple(round(255 * v) for v in export.PREVIEW_NAN_RGB)
    m = _burn_mosaic()
    px = _png(write_preview_png(tmp_path / "n.png", m, nodata=255, legend="burned"))
    # 0 and 1 take the two ends of the colormap; nodata is grey and does not stretch the scale.
    assert _near(px, grey, tol=1) > 0.25 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 1.0)) > 0.25 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 0.0)) > 0.35 * IMAGE_PIXELS
    burned = _near(px, _cmap("viridis", 1.0))
    assert abs(burned - _near(px, grey, tol=1)) < 0.1 * burned  # 15 columns each
    # The same mosaic without nodata shows the hazard: 0 and 1 look alike and 255 is yellow.
    px = _png(write_preview_png(tmp_path / "raw.png", m))
    assert _near(px, grey, tol=1) < 200
    # Fully burned with nodata around it: explicit limits keep "burned" yellow.
    m[:, :35] = 1
    px = _png(write_preview_png(tmp_path / "all.png", m, nodata=255, vmin=0, vmax=1))
    assert _near(px, _cmap("viridis", 1.0)) > 0.6 * IMAGE_PIXELS
    # A float image with a finite nodata (and NaN) is handled the same way.
    f = np.where(_burn_mosaic() == 255, -9999.0, _burn_mosaic().astype(float))
    f[0, 0] = np.nan
    px = _png(write_preview_png(tmp_path / "f.png", f, nodata=-9999.0))
    assert _near(px, _cmap("viridis", 1.0)) > 0.25 * IMAGE_PIXELS
    assert _near(px, grey, tol=1) > 0.25 * IMAGE_PIXELS


def test_preview_honours_masked_arrays(tmp_path):
    grey = tuple(round(255 * v) for v in export.PREVIEW_NAN_RGB)
    m = _burn_mosaic()
    masked = np.ma.masked_equal(m, 255)
    px = _png(write_preview_png(tmp_path / "m.png", masked))
    assert _near(px, grey, tol=1) > 0.25 * IMAGE_PIXELS
    assert _near(px, _cmap("viridis", 1.0)) > 0.25 * IMAGE_PIXELS
    assert m[0, 40] == 255  # the caller's array is untouched
    # Colour images: masked pixels and pixels whose R, G and B all equal nodata are grey.
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:, :20] = (255, 0, 0)
    px = _png(write_preview_png(tmp_path / "c.png", rgb, nodata=0))
    assert _near(px, grey, tol=1) > 0.4 * IMAGE_PIXELS and _near(px, (0, 0, 0), tol=0) < 2_000
    hidden = np.zeros(rgb.shape, bool)
    hidden[:, :20, 0] = True  # one masked channel hides the pixel
    px = _png(write_preview_png(tmp_path / "d.png", np.ma.masked_array(rgb, mask=hidden)))
    assert _near(px, grey, tol=1) > 0.4 * IMAGE_PIXELS and _near(px, (255, 0, 0), tol=0) < 2_000
    assert (rgb[:, :20] == (255, 0, 0)).all()
    rgba = np.concatenate([rgb, np.full((40, 40, 1), 0, np.uint8)], axis=2)
    px = _png(write_preview_png(tmp_path / "e.png", rgba, nodata=0))
    assert _near(px, grey, tol=1) > 0.4 * IMAGE_PIXELS
    with pytest.raises(TypeError):
        write_preview_png(tmp_path / "x.png", m, nodata="255")  # type: ignore[arg-type]


def test_preview_downsamples_large_images(tmp_path):
    img = np.zeros((5000, 300), dtype=bool)
    px = _png(write_preview_png(tmp_path / "big.png", img))
    assert 1667 <= px.shape[0] <= export.PREVIEW_MAX_SIDE  # 5000 rows strided by 3
    img = np.zeros((300, 2048), dtype=np.float32)
    px = _png(write_preview_png(tmp_path / "edge.png", img))
    assert px.shape[1] >= 2048  # exactly the limit: not strided


def test_preview_leaves_pyplot_alone(tmp_path):
    import matplotlib
    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend()
    before = plt.get_fignums()
    write_preview_png(tmp_path / "p.png", np.ones((20, 20)))
    write_preview_png(tmp_path / "q.png", np.zeros((20, 20, 3), dtype=np.uint8))
    assert plt.get_fignums() == before
    assert matplotlib.get_backend() == backend


def test_preview_module_does_not_import_pyplot(tmp_path):
    code = (
        "import sys, numpy as np\n"
        "from tessera_apps.export import write_preview_png\n"
        f"write_preview_png({str(tmp_path / 'x.png')!r}, np.ones((8, 8)))\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "img",
    [np.zeros((4, 4, 3), dtype=np.float32), np.zeros((4, 4, 5), dtype=np.uint8), np.zeros(4), np.zeros((0, 3))],
)
def test_preview_rejects_bad_images(tmp_path, img):
    with pytest.raises(ValueError):
        write_preview_png(tmp_path / "x.png", img)
    assert _files(tmp_path) == []


# --------------------------------------------------------------------------------------------
# Digests and output names
# --------------------------------------------------------------------------------------------


def test_file_sha256_matches_hashlib(tmp_path):
    data = os.urandom(3 * (1 << 20) + 5)
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_args_digest_is_canonical():
    d = args_digest({"threshold": 0.3, "min_area_ha": 1, "years": [2020, 2022]})
    assert len(d) == 64 and int(d, 16) >= 0
    assert d == args_digest({"years": [2020, 2022], "min_area_ha": 1, "threshold": 0.3})
    assert d == args_digest({"years": np.array([2020, 2022]), "min_area_ha": np.int64(1), "threshold": 0.3})
    assert d != args_digest({"threshold": 0.31, "min_area_ha": 1, "years": [2020, 2022]})
    assert args_digest({"t": float("nan")}) == args_digest({"t": None})
    assert args_digest({"a": {"y": 1, "x": 2}}) == args_digest({"a": {"x": 2, "y": 1}})


def test_run_args_digest_covers_params_years_aoi_and_seed():
    base = run_args_digest({"t": 0.3}, years=[2022, 2020], aoi_sha256="ab", seed=0)
    assert base == run_args_digest({"t": 0.3}, years=[2020, 2022, 2020], aoi_sha256="ab", seed=0)
    assert base != run_args_digest({"t": 0.4}, years=[2020, 2022], aoi_sha256="ab", seed=0)
    assert base != run_args_digest({"t": 0.3}, years=[2020, 2023], aoi_sha256="ab", seed=0)
    assert base != run_args_digest({"t": 0.3}, years=[2020, 2022], aoi_sha256="cd", seed=0)
    assert base != run_args_digest({"t": 0.3}, years=[2020, 2022], aoi_sha256="ab", seed=1)


def test_output_paths_use_the_brief_names(tmp_path):
    p = output_paths(tmp_path, "wildfire", 2020, np.int64(2022))
    assert p == OutputPaths(
        tif=tmp_path / "wildfire_2020_2022.tif",
        summary_parquet=tmp_path / "wildfire_summary.parquet",
        summary_csv=tmp_path / "wildfire_summary.csv",
        report=tmp_path / "wildfire_report.json",
        preview=tmp_path / "wildfire_preview.png",
        validation=tmp_path / "wildfire_validation.json",
    )
    assert p.required(need_validation=False)[-1] == p.report
    assert p.validation not in p.required(need_validation=False)
    assert p.validation in p.required(need_validation=True)
    for bad in ("", "../x", "a b", "a/b", "_x"):
        with pytest.raises(ValueError):
            output_paths(tmp_path, bad, 2020, 2022)
    with pytest.raises(TypeError):
        output_paths(tmp_path, "wildfire", 2020.0, 2022)  # type: ignore[arg-type]


def _write_outputs(p: OutputPaths, report: object, *, validation: bool) -> None:
    for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview) + ((p.validation,) if validation else ()):
        f.write_bytes(b"x")
    past = p.tif.stat().st_mtime_ns - 10_000_000_000
    for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview) + ((p.validation,) if validation else ()):
        os.utime(f, ns=(past, past))
    write_json(p.report, report)


def test_outputs_current_semantics(tmp_path):
    p = output_paths(tmp_path, "deforest", 2019, 2022)
    digest = run_args_digest({"t": 0.3}, years=[2019, 2022], aoi_sha256="ab", seed=0)
    assert not outputs_current(p, digest, need_validation=False)

    _write_outputs(p, {"args_digest": digest}, validation=False)
    assert outputs_current(p, digest, need_validation=False)
    assert not outputs_current(p, "0" * 64, need_validation=False)
    assert not outputs_current(p, digest, need_validation=True)  # no validation.json yet

    # A validation.json the report does not list is not this run's, whatever its mtime.
    p.validation.write_bytes(b"{}")
    os.utime(p.validation, ns=(p.report.stat().st_mtime_ns - 1,) * 2)
    assert not outputs_current(p, digest, need_validation=True)
    assert outputs_current(p, digest, need_validation=False)

    write_json(p.report, {"args_digest": digest, "outputs": outputs_record(p, validation=True)})
    assert outputs_current(p, digest, need_validation=True)
    assert outputs_current(p, digest, need_validation=False)

    p.preview.unlink()
    assert not outputs_current(p, digest, need_validation=False)


def test_outputs_current_accepts_outputs_with_the_same_mtime_as_the_report(tmp_path):
    # Coarse clocks (1 s on FAT or HFS+, a jiffy on older Linux kernels) give the report and the
    # outputs written just before it the same mtime: a complete run must still be current.
    p = output_paths(tmp_path, "wildfire", 2020, 2022)
    digest = "d" * 64
    for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview, p.validation):
        f.write_bytes(b"x")
    write_json(p.report, {"args_digest": digest, "outputs": outputs_record(p, validation=True)})
    stamp = p.report.stat().st_mtime_ns
    for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview, p.validation):
        os.utime(f, ns=(stamp, stamp))
    assert outputs_current(p, digest, need_validation=False)
    assert outputs_current(p, digest, need_validation=True)
    os.utime(p.preview, ns=(stamp + 1, stamp + 1))
    assert not outputs_current(p, digest, need_validation=False)


THRESHOLD_APP_YEARS = (2020, 2022)


def _app_run(p: OutputPaths, *, threshold: float, validate: bool, fetch_ok: bool = True, begin: bool = True) -> str:
    """What a checkpoint 3 app does, with the real writers: returns the run's args digest.

    The ``--validate`` flag is not part of the digest (``need_validation`` covers it), so a run
    with and a run without validation at one threshold share a digest.
    """
    digest = run_args_digest({"threshold": threshold}, years=THRESHOLD_APP_YEARS, aoi_sha256="ab", seed=0)
    if outputs_current(p, digest, need_validation=validate):
        return digest
    if begin:
        begin_run(p)
    burned = np.random.default_rng(int(threshold * 100)).random((32, 32)) > threshold
    write_geotiff(p.tif, burned, T10, 32610, overviews=False)
    write_table(
        pd.DataFrame({"threshold": [threshold], "burned_px": [int(burned.sum())]}), p.summary_parquet.with_suffix("")
    )
    write_preview_png(p.preview, burned, title=f"threshold {threshold}")
    wrote_validation = False
    if validate and fetch_ok:
        write_validation_json(
            p.validation, scores={"burned": {"iou": threshold}}, reference={"threshold": threshold}, caveats=[]
        )
        wrote_validation = True
    write_json(p.report, {"args_digest": digest, "outputs": outputs_record(p, validation=wrote_validation)})
    return digest


def _validated_threshold(p: OutputPaths) -> float:
    return json.loads(p.validation.read_text())["reference"]["threshold"]


@pytest.mark.parametrize("begin", [True, False])
def test_outputs_current_rejects_a_validation_json_left_by_another_run(tmp_path, begin):
    p = output_paths(tmp_path, "wildfire", *THRESHOLD_APP_YEARS)
    # 1. --threshold 0.30 --validate: validation.json scored at 0.30.
    x = _app_run(p, threshold=0.30, validate=True, begin=begin)
    assert outputs_current(p, x, need_validation=True)
    assert _validated_threshold(p) == 0.30
    # 2. --threshold 0.45 without --validate: new tif, tables, preview and report.
    y = _app_run(p, threshold=0.45, validate=False, begin=begin)
    assert y != x
    assert outputs_current(p, y, need_validation=False)
    assert p.validation.exists() is not begin  # begin_run removes the stale file; without it, it stays
    # 3. --threshold 0.45 --validate must not be skipped, even with the 0.30 file still on disk and
    # older than the report.
    assert not outputs_current(p, y, need_validation=True)
    _app_run(p, threshold=0.45, validate=True, begin=begin)
    assert _validated_threshold(p) == 0.45
    assert outputs_current(p, y, need_validation=True)
    # A rerun with the first arguments is not current either: the outputs are the 0.45 run's.
    assert not outputs_current(p, x, need_validation=True)
    assert not outputs_current(p, x, need_validation=False)


@pytest.mark.parametrize("begin", [True, False])
def test_outputs_current_rejects_validation_from_before_a_failed_reference_fetch(tmp_path, begin):
    p = output_paths(tmp_path, "wildfire", *THRESHOLD_APP_YEARS)
    _app_run(p, threshold=0.30, validate=True, begin=begin)
    # A --validate run at 0.45 whose reference fetch fails (NIFC 403) still writes its report.
    y = _app_run(p, threshold=0.45, validate=True, fetch_ok=False, begin=begin)
    assert outputs_current(p, y, need_validation=False)
    assert not outputs_current(p, y, need_validation=True)
    assert p.validation.exists() is not begin


def test_outputs_current_checks_the_hashes_the_report_lists(tmp_path):
    p = output_paths(tmp_path, "wildfire", *THRESHOLD_APP_YEARS)
    y = _app_run(p, threshold=0.45, validate=True)
    assert outputs_current(p, y, need_validation=True)
    # An older validation.json copied back over this run's (same name, older mtime): rejected.
    stamp = p.validation.stat().st_mtime_ns
    old = p.validation.read_text().replace("0.45", "0.3")
    p.validation.write_text(old)
    os.utime(p.validation, ns=(stamp, stamp))
    assert not outputs_current(p, y, need_validation=True)
    assert outputs_current(p, y, need_validation=False)  # not required, so not checked
    # Any listed output whose bytes changed is rejected too.
    _app_run(p, threshold=0.45, validate=True)
    assert outputs_current(p, y, need_validation=True)
    stamp = p.summary_csv.stat().st_mtime_ns
    p.summary_csv.write_text("threshold,burned_px\n0.45,0\n")
    os.utime(p.summary_csv, ns=(stamp, stamp))
    assert not outputs_current(p, y, need_validation=False)
    # A malformed "outputs" entry is never current.
    write_json(p.report, {"args_digest": y, "outputs": ["validation"]})
    assert not outputs_current(p, y, need_validation=False)
    write_json(p.report, {"args_digest": y, "outputs": None})
    assert not outputs_current(p, y, need_validation=True)


def test_outputs_record_and_begin_run(tmp_path):
    p = output_paths(tmp_path, "deforest", 2019, 2022)
    with pytest.raises(FileNotFoundError):
        outputs_record(p, validation=False)
    for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview):
        f.write_bytes(f.name.encode())
    record = outputs_record(p, validation=False)
    assert record == {
        name: hashlib.sha256(getattr(p, name).name.encode()).hexdigest()
        for name in ("tif", "summary_parquet", "summary_csv", "preview")
    }
    with pytest.raises(FileNotFoundError):
        outputs_record(p, validation=True)
    p.validation.write_text("{}")
    assert outputs_record(p, validation=True)["validation"] == hashlib.sha256(b"{}").hexdigest()

    write_json(p.report, {"args_digest": "d" * 64})
    assert begin_run(p) == (p.report, p.validation)
    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(
        f.name for f in (p.tif, p.summary_parquet, p.summary_csv, p.preview)
    )
    assert begin_run(p) == ()


def test_outputs_current_rejects_outputs_newer_than_the_report(tmp_path):
    p = output_paths(tmp_path, "wildfire", 2020, 2022)
    digest = "d" * 64
    _write_outputs(p, {"args_digest": digest}, validation=False)
    assert outputs_current(p, digest, need_validation=False)
    later = p.report.stat().st_mtime_ns + 5_000_000_000
    os.utime(p.tif, ns=(later, later))
    assert not outputs_current(p, digest, need_validation=False)


@pytest.mark.parametrize(
    ("report", "current"),
    [
        ({"provenance": {"args_digest": "d" * 64}}, True),
        ({"args_digest": "e" * 64}, False),
        ({"provenance": "d" * 64}, False),
        (["d" * 64], False),
        ({}, False),
    ],
)
def test_outputs_current_reads_the_digest_from_the_report(tmp_path, report, current):
    p = output_paths(tmp_path, "wetland", 2018, 2024)
    _write_outputs(p, report, validation=False)
    assert outputs_current(p, "d" * 64, need_validation=False) is current


def test_outputs_current_survives_a_corrupt_report(tmp_path):
    p = output_paths(tmp_path, "wetland", 2018, 2024)
    _write_outputs(p, {"args_digest": "d" * 64}, validation=False)
    p.report.write_text('{"args_digest": "dd', encoding="utf-8")
    assert not outputs_current(p, "d" * 64, need_validation=False)
    p.report.write_bytes(b"\xff\xfe\x00")
    assert not outputs_current(p, "d" * 64, need_validation=False)


# --------------------------------------------------------------------------------------------
# provenance_record
# --------------------------------------------------------------------------------------------


def test_provenance_record_with_caldor_reader(caldor_reader, tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sekrit-value-123")
    aoi = tmp_path / "caldor.geojson"
    aoi.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    params = {"threshold": np.float64(0.281), "min_area_ha": 1}
    start = dt.datetime.now(dt.UTC).replace(microsecond=0)
    rec = provenance_record(caldor_reader, aoi_path=aoi, years=[2022, 2020, 2022], params=params, seed=7, zones=["10n"])

    assert rec["tessera_apps_version"] == tessera_apps.__version__
    assert rec["python_version"] == platform.python_version()
    libs = rec["libraries"]
    assert libs["numpy"] == np.__version__
    assert libs["rasterio"] == rasterio.__version__
    assert libs["gdal"] == rasterio.__gdal_version__
    for name in ("icechunk", "zarr", "geotessera"):
        assert name in libs
    assert libs["geotessera"] == geotessera.__version__ and libs["geotessera"] is not None
    assert rec["store"] == {"backend": "KeyLog"}
    assert rec["model"] == {"checkpoint_id": "tessera_v1_1_aws_encoder.pt", "geoemb:dimensions": 128}
    assert rec["pipeline_commits"] == {"10N": {"2020": "abc2020", "2022": "abc2022"}}
    assert rec["years"] == [2020, 2022]
    assert rec["zones"] == ["10N"]
    assert rec["params"] == {"threshold": 0.281, "min_area_ha": 1}
    assert rec["seed"] == 7
    assert rec["aoi"] == {"name": "caldor.geojson", "sha256": hashlib.sha256(aoi.read_bytes()).hexdigest()}
    assert rec["args_digest"] == run_args_digest(params, years=[2020, 2022], aoi_sha256=rec["aoi"]["sha256"], seed=7)
    created = dt.datetime.fromisoformat(rec["created_utc"])
    assert created.utcoffset() == dt.timedelta(0)
    assert start <= created <= dt.datetime.now(dt.UTC)

    text = json.dumps(rec, allow_nan=False)
    assert "sekrit-value-123" not in text
    write_json(tmp_path / "report.json", {"provenance": rec})

    aoi.write_text('{"type": "FeatureCollection", "features": [null]}', encoding="utf-8")
    again = provenance_record(caldor_reader, aoi_path=aoi, years=[2020, 2022], params=params, seed=7, zones=["10N"])
    assert again["args_digest"] != rec["args_digest"]


class _BareReader:
    def provenance(self):
        return {
            "backend": "fake",
            "snapshot_id": "SNAP",
            "aws_secret_access_key": "hidden",
            "session_token": "hidden",
            "Password": "hidden",
            "geoemb:model": "https://geotessera.org/model/1.1",
        }


def test_provenance_record_without_run_commits_or_aoi_and_drops_secrets():
    rec = provenance_record(_BareReader(), aoi_path=None, years=[2021], params={}, seed=None, zones=["11N", "10N"])  # type: ignore[arg-type]
    assert rec["pipeline_commits"] is None
    assert rec["aoi"] == {"name": None, "sha256": None}
    assert rec["store"] == {"backend": "fake", "snapshot_id": "SNAP"}
    assert rec["model"] == {"geoemb:model": "https://geotessera.org/model/1.1"}
    assert rec["zones"] == ["10N", "11N"]
    assert "hidden" not in json.dumps(rec)
    assert rec["args_digest"] == run_args_digest({}, years=[2021], aoi_sha256=None, seed=None)
