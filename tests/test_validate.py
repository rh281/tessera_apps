"""Offline tests for validate.py: scoring, reference rasters, and every dataset loader against local fakes.

No test touches the network: an autouse fixture replaces the module's URL opener with one that
fails, and tests that exercise a loader's HTTP path install a :class:`FakeWeb` that answers in the
documented formats (ArcGIS REST, OGC WFS 2.0, Zenodo REST).
"""

from __future__ import annotations

import email.message
import http.server
import io
import json
import math
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.response
import uuid
import zipfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.errors
import rasterio.io
import shapely
from affine import Affine
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, mapping, shape
from sklearn import metrics as skm

from tessera_apps import validate as v
from tessera_apps.demo import DEMO_NAMES, demo_aoi, demo_feature, demo_geojson, demo_properties
from tessera_apps.io import ZoneGrid, ZonePlan, plan_aoi

# --------------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Cache under tmp_path, and no real network: any unexpected URL open fails the test."""
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))

    def blocked(request, timeout):
        raise AssertionError(f"unexpected network access: {request.get_method()} {request.full_url}")

    monkeypatch.setattr(v, "_urlopen", blocked)


def _headers(length: int | None = None) -> email.message.Message:
    h = email.message.Message()
    if length is not None:
        h["Content-Length"] = str(length)
    return h


def http_error(url: str, code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, reason, _headers(), None)


def proxy_refusal() -> urllib.error.URLError:
    return urllib.error.URLError(OSError("Tunnel connection failed: 403 Forbidden"))


Handler = Callable[[str, dict[str, str]], Any]


class FakeWeb:
    """Stands in for ``validate._urlopen``: routes by scheme, host and path, records every call."""

    def __init__(self) -> None:
        self.routes: dict[str, Handler] = {}
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    def route(self, url: str, handler: Handler) -> None:
        self.routes[url] = handler

    def __call__(self, request, timeout):
        """Answer one request like urllib.request.urlopen would."""
        url = request.full_url
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}{parts.path}"
        params = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
        if request.data:
            params.update(urllib.parse.parse_qsl(request.data.decode("ascii"), keep_blank_values=True))
        method = request.get_method()
        self.calls.append((method, base, params))
        handler = self.routes.get(base)
        if handler is None:
            raise http_error(url, 404, "Not Found")
        result = handler(method, params)
        if isinstance(result, BaseException):
            raise result
        body = result if isinstance(result, bytes) else json.dumps(result).encode("utf-8")
        return urllib.response.addinfourl(io.BytesIO(body), _headers(len(body)), url, 200)

    def urls(self, method: str | None = None) -> list[str]:
        return [b for m, b, _ in self.calls if method is None or m == method]


@pytest.fixture
def web(monkeypatch) -> FakeWeb:
    fake = FakeWeb()
    monkeypatch.setattr(v, "_urlopen", fake)
    return fake


def offline(monkeypatch) -> None:
    def blocked(request, timeout):
        raise AssertionError(f"network used after caching: {request.full_url}")

    monkeypatch.setattr(v, "_urlopen", blocked)


BOUNDARY_LON, BOUNDARY_LAT = -120.0, 38.85


@pytest.fixture
def two_zone(boundary_reader) -> list[ZonePlan]:
    """A small AOI straddling 120 W: one plan in 10N, one in 11N."""
    aoi = box(BOUNDARY_LON - 0.006, BOUNDARY_LAT - 0.005, BOUNDARY_LON + 0.006, BOUNDARY_LAT + 0.005)
    plans = plan_aoi(boundary_reader, aoi).zones
    assert [p.zone for p in plans] == ["10N", "11N"]
    return plans


@pytest.fixture
def plan10(boundary_reader) -> ZonePlan:
    """A small AOI wholly inside 10N, west of the zone edge."""
    aoi = box(BOUNDARY_LON - 0.008, BOUNDARY_LAT - 0.004, BOUNDARY_LON - 0.002, BOUNDARY_LAT + 0.004)
    (plan,) = plan_aoi(boundary_reader, aoi).zones
    return plan


def centres_lonlat(plan: ZonePlan) -> tuple[np.ndarray, np.ndarray]:
    h, w = plan.shape
    t = plan.transform
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    lon, lat = plan.grid.to_lonlat().transform(x, y)
    return np.asarray(lon), np.asarray(lat)


def lookup(
    arr: np.ndarray, west: float, north: float, res: float, lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Value of a north-up lon/lat raster at each point, and whether the point is inside it."""
    c = np.floor((lon - west) / res).astype(np.int64)
    r = np.floor((north - lat) / res).astype(np.int64)
    inside = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.zeros(lon.shape, dtype=arr.dtype)
    out[inside] = arr[r[inside], c[inside]]
    return out, inside


def write_lonlat_tif(path: Path, data: np.ndarray, west: float, north: float, res: float, nodata=None) -> Path:
    bands = data[None] if data.ndim == 2 else data
    profile = {
        "driver": "GTiff",
        "width": bands.shape[2],
        "height": bands.shape[1],
        "count": bands.shape[0],
        "dtype": bands.dtype.name,
        "crs": "EPSG:4326",
        "transform": from_origin(west, north, res, res),
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)
    return path


def no_temp_files(root: Path) -> bool:
    return not [p for p in root.rglob("*") if ".part" in p.name or ".tmp" in p.name]


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------


def test_binary_scores_counts_and_ratios():
    pred = np.array([1, 1, 0, 0, 1], dtype=bool)
    ref = np.array([1, 0, 1, 0, 1], dtype=bool)
    s = v.binary_scores(pred, ref)
    assert (s.tp, s.fp, s.fn, s.tn, s.n) == (2, 1, 1, 1, 5)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.f1 == pytest.approx(2 / 3)
    assert s.iou == pytest.approx(0.5)
    assert s.accuracy == pytest.approx(0.6)


def test_binary_scores_asymmetric_ratios():
    # 3 TP, 1 FP, 4 FN, 2 TN: every ratio differs, so a swapped term shows up.
    pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    ref = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
    s = v.binary_scores(pred, ref)
    assert (s.tp, s.fp, s.fn, s.tn) == (3, 1, 4, 2)
    assert s.precision == pytest.approx(3 / 4)
    assert s.recall == pytest.approx(3 / 7)
    assert s.f1 == pytest.approx(6 / 11)
    assert s.iou == pytest.approx(3 / 8)
    assert s.accuracy == pytest.approx(5 / 10)


def test_binary_scores_all_false_gives_none_not_nan():
    z = np.zeros((4, 5), dtype=bool)
    s = v.binary_scores(z, z)
    assert (s.tp, s.fp, s.fn, s.tn, s.n) == (0, 0, 0, 20, 20)
    assert s.precision is None and s.recall is None and s.f1 is None and s.iou is None
    assert s.accuracy == 1.0


def test_binary_scores_empty_valid_mask():
    a = np.ones((3, 3), dtype=bool)
    s = v.binary_scores(a, a, np.zeros((3, 3), dtype=bool))
    assert s.n == 0
    assert (s.precision, s.recall, s.f1, s.iou, s.accuracy) == (None, None, None, None, None)


def test_binary_scores_no_prediction_with_reference():
    ref = np.array([True, True, False])
    s = v.binary_scores(np.zeros(3, dtype=bool), ref)
    assert s.precision is None
    assert s.recall == 0.0 and s.f1 == 0.0 and s.iou == 0.0
    s2 = v.binary_scores(ref, np.zeros(3, dtype=bool))
    assert s2.recall is None and s2.precision == 0.0


def test_binary_scores_valid_mask_drops_pixels():
    pred = np.array([[1, 1], [1, 0]], dtype=bool)
    ref = np.array([[1, 0], [0, 0]], dtype=bool)
    valid = np.array([[1, 0], [0, 1]], dtype=bool)
    s = v.binary_scores(pred, ref, valid)
    assert (s.tp, s.fp, s.fn, s.tn, s.n) == (1, 0, 0, 1, 2)
    assert s.precision == 1.0


def test_binary_scores_accepts_01_numbers_and_rejects_other_values():
    s = v.binary_scores(np.array([1, 0, 1], dtype=np.uint8), np.array([1.0, 0.0, 0.0]))
    assert (s.tp, s.fp) == (1, 1)
    with pytest.raises(ValueError, match="only 0 and 1"):
        v.binary_scores(np.array([1, 255], dtype=np.uint8), np.array([1, 0], dtype=bool))
    with pytest.raises(ValueError, match="only 0 and 1"):
        v.binary_scores(np.array([1.0, np.nan]), np.array([1, 0], dtype=bool))
    with pytest.raises(ValueError, match="Shapes differ"):
        v.binary_scores(np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="Shapes differ"):
        v.binary_scores(np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), np.zeros(4, dtype=bool))
    with pytest.raises(TypeError):
        v.binary_scores(np.array(["a", "b"]), np.array(["a", "b"]))


def test_binary_scores_match_sklearn():
    rng = np.random.default_rng(1)
    pred = rng.random((60, 70)) < 0.3
    ref = rng.random((60, 70)) < 0.4
    valid = rng.random((60, 70)) < 0.8
    s = v.binary_scores(pred, ref, valid)
    p, r = pred[valid], ref[valid]
    assert s.precision == pytest.approx(skm.precision_score(r, p))
    assert s.recall == pytest.approx(skm.recall_score(r, p))
    assert s.f1 == pytest.approx(skm.f1_score(r, p))
    assert s.iou == pytest.approx(skm.jaccard_score(r, p))
    assert s.accuracy == pytest.approx(skm.accuracy_score(r, p))
    assert s.n == int(valid.sum())


def test_merge_scores_equals_scoring_the_concatenation():
    rng = np.random.default_rng(2)
    parts, preds, refs = [], [], []
    for shape_ in ((10, 12), (7, 30), (1, 1)):
        p = rng.random(shape_) < 0.4
        r = rng.random(shape_) < 0.5
        m = rng.random(shape_) < 0.7
        parts.append(v.binary_scores(p, r, m))
        preds.append(p[m])
        refs.append(r[m])
    merged = v.merge_scores(parts)
    whole = v.binary_scores(np.concatenate(preds), np.concatenate(refs))
    assert merged == whole
    empty = v.merge_scores([])
    assert empty.n == 0 and empty.precision is None and empty.accuracy is None


def test_from_counts_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        v.BinaryScores.from_counts(1, -1, 0, 0)


def test_confusion_matrix_matches_sklearn_with_unlisted_labels_and_valid():
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 6, size=(40, 50))
    pred = rng.integers(0, 6, size=(40, 50))
    valid = rng.random((40, 50)) < 0.75
    labels = [4, 1, 2, 9]  # 0, 3 and 5 are left out; 9 never occurs
    cm = v.confusion_matrix(pred, ref, labels, valid)
    expected = skm.confusion_matrix(ref[valid], pred[valid], labels=labels)
    np.testing.assert_array_equal(cm, expected)
    assert cm.dtype == np.int64
    full = v.confusion_matrix(pred, ref, list(range(6)))
    np.testing.assert_array_equal(full, skm.confusion_matrix(ref.ravel(), pred.ravel(), labels=list(range(6))))
    assert full.sum() == ref.size


def test_per_class_f1_matches_sklearn():
    rng = np.random.default_rng(4)
    ref = rng.integers(0, 4, size=500)
    pred = np.where(rng.random(500) < 0.6, ref, rng.integers(0, 4, size=500))
    labels = [0, 1, 2, 3, 7]
    cm = v.confusion_matrix(pred, ref, labels)
    f1 = v.per_class_f1(cm, labels)
    expected = skm.f1_score(ref, pred, labels=labels, average=None, zero_division=np.nan)
    for lab, e in zip(labels, expected, strict=True):
        if math.isnan(e):
            assert f1[lab] is None
        else:
            assert f1[lab] == pytest.approx(e)
    assert f1[7] is None
    assert list(f1) == labels


def test_confusion_matrix_leaves_out_values_above_every_label():
    # 255 is the usual nodata: larger than every label, so it must be dropped, not indexed past the end.
    pred = np.array([0, 1, 255, 2, 2, 1, 0, 255])
    ref = np.array([0, 1, 2, 255, 2, 0, 0, 255])
    for labels in ([0, 1, 2], [2, 0, 1]):
        cm = v.confusion_matrix(pred, ref, labels)
        np.testing.assert_array_equal(cm, skm.confusion_matrix(ref, pred, labels=labels))
        assert cm.sum() == 5
    rng = np.random.default_rng(5)
    ref2 = rng.choice([0, 1, 2, 255], size=(30, 40))
    pred2 = rng.choice([0, 1, 2, 255], size=(30, 40))
    np.testing.assert_array_equal(
        v.confusion_matrix(pred2, ref2, [0, 1, 2]),
        skm.confusion_matrix(ref2.ravel(), pred2.ravel(), labels=[0, 1, 2]),
    )


def test_numeric_valid_mask_equals_the_bool_mask():
    # A uint8 0/1 coverage read from a raster must select pixels, not index rows 0 and 1.
    rng = np.random.default_rng(6)
    pred = rng.random((6, 7)) < 0.5
    ref = rng.random((6, 7)) < 0.5
    valid = rng.random((6, 7)) < 0.5
    as_uint8 = valid.astype(np.uint8)
    expected = v.binary_scores(pred, ref, valid)
    assert expected.n == int(valid.sum())
    assert v.binary_scores(pred, ref, as_uint8) == expected
    assert v.binary_scores(pred, ref, valid.astype(np.float32)) == expected
    labels_p = rng.integers(0, 3, size=(6, 7))
    labels_r = rng.integers(0, 3, size=(6, 7))
    cm_bool = v.confusion_matrix(labels_p, labels_r, [0, 1, 2], valid)
    np.testing.assert_array_equal(v.confusion_matrix(labels_p, labels_r, [0, 1, 2], as_uint8), cm_bool)
    np.testing.assert_array_equal(cm_bool, skm.confusion_matrix(labels_r[valid], labels_p[valid], labels=[0, 1, 2]))
    assert cm_bool.sum() == int(valid.sum())
    with pytest.raises(ValueError, match="only 0 and 1"):
        v.binary_scores(pred, ref, as_uint8 * 2)


def test_confusion_matrix_and_f1_reject_bad_input():
    a = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match="repeat"):
        v.confusion_matrix(a, a, [0, 1, 0])
    with pytest.raises(ValueError, match="non-empty"):
        v.confusion_matrix(a, a, [])
    with pytest.raises(ValueError, match="Shapes differ"):
        v.confusion_matrix(a, np.zeros(5, dtype=int), [0, 1])
    with pytest.raises(ValueError, match="expected"):
        v.per_class_f1(np.zeros((2, 2), dtype=int), [0, 1, 2])


def test_write_validation_json(tmp_path):
    out = tmp_path / "out" / "wildfire_validation.json"
    scores = v.binary_scores(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool))
    path = v.write_validation_json(
        out,
        scores={"burned": scores, "extra": np.float32(np.nan), "count": np.int64(3)},
        reference={"dataset": "NIFC", "year": np.int64(2021), "file": Path("/x/y.geojson")},
        caveats="Perimeters include unburned islands.",
    )
    assert path == out
    data = json.loads(out.read_text())
    assert set(data) == {"scores", "reference", "caveats"}
    assert data["scores"]["burned"]["precision"] is None
    assert data["scores"]["burned"]["tn"] == 4
    assert data["scores"]["extra"] is None
    assert data["scores"]["count"] == 3
    assert data["reference"] == {"dataset": "NIFC", "year": 2021, "file": "/x/y.geojson"}
    assert data["caveats"] == ["Perimeters include unburned islands."]
    assert "NaN" not in out.read_text()
    assert sorted(p.name for p in out.parent.iterdir()) == ["wildfire_validation.json"]
    v.write_validation_json(out, scores={}, reference={}, caveats=["a", "b"])
    assert json.loads(out.read_text())["caveats"] == ["a", "b"]


# --------------------------------------------------------------------------------------------
# Reference rasters on the AOI grid
# --------------------------------------------------------------------------------------------


def test_reference_from_geometries_pixel_centre_rule(plan10):
    plan = plan10
    t = plan.transform
    r, c = plan.shape[0] // 2 - 2, plan.shape[1] // 2 - 2
    assert plan.mask[r : r + 5, c : c + 5].all()

    def xc(col):
        return t.c + (col + 0.5) * t.a

    def yc(row):
        return t.f + (row + 0.5) * t.e

    # Edges sit 2 m past the centres of row r and column c (touching them, centres outside), and 2 m
    # past the centres of row r + 3 and column c + 3 (centres inside).
    rect = box(xc(c) + 2, yc(r + 3) - 2, xc(c + 3) + 2, yc(r) - 2)
    lonlat = shapely.transform(shapely.segmentize(rect, 1.0), plan.grid.to_lonlat().transform, interleaved=False)
    ref = v.reference_from_geometries(plan, [lonlat])
    expected = np.zeros(plan.shape, dtype=bool)
    expected[r + 1 : r + 4, c + 1 : c + 4] = True
    np.testing.assert_array_equal(ref, expected)
    assert ref.dtype == bool and ref.shape == plan.shape


def test_reference_from_geometries_multi_zone_matches_centre_lookup(two_zone):
    diamond = Polygon(
        [
            (BOUNDARY_LON - 0.004, BOUNDARY_LAT),
            (BOUNDARY_LON + 0.001, BOUNDARY_LAT + 0.0035),
            (BOUNDARY_LON + 0.0045, BOUNDARY_LAT - 0.0005),
            (BOUNDARY_LON - 0.0005, BOUNDARY_LAT - 0.004),
        ]
    )
    totals = []
    for plan in two_zone:
        ref = v.reference_from_geometries(plan, [diamond])
        lon, lat = centres_lonlat(plan)
        expected = shapely.contains_xy(diamond, lon, lat) & plan.mask
        bad = np.argwhere(ref != expected)
        for rr, cc in bad:
            d_m = diamond.boundary.distance(Point(lon[rr, cc], lat[rr, cc])) * 111_000
            assert d_m < 0.5, f"pixel {rr, cc} in {plan.zone} differs {d_m:.2f} m from the edge"
        assert not (ref & ~plan.mask).any()
        totals.append(int(ref.sum()))
    assert all(t > 100 for t in totals)


def test_reference_from_geometries_keeps_polygons_only(plan10):
    lon, lat = centres_lonlat(plan10)
    cx, cy = float(lon.mean()), float(lat.mean())
    poly = box(cx - 0.001, cy - 0.001, cx + 0.001, cy + 0.001)
    line = LineString([(cx - 0.003, cy - 0.003), (cx + 0.003, cy + 0.003)])
    alone = v.reference_from_geometries(plan10, [poly])
    assert alone.sum() > 50
    mixed = v.reference_from_geometries(plan10, [shapely.GeometryCollection([poly, line, Point(cx, cy + 0.003)])])
    np.testing.assert_array_equal(mixed, alone)
    assert not v.reference_from_geometries(plan10, [line, Point(cx, cy)]).any()
    assert not v.reference_from_geometries(plan10, []).any()
    assert not v.reference_from_geometries(plan10, [box(10, 10, 11, 11)]).any()
    # Two overlapping parts burn their union; an invalid bow tie is repaired, not dropped.
    left = box(cx - 0.001, cy - 0.001, cx, cy + 0.001)
    right = box(cx - 0.0002, cy - 0.001, cx + 0.001, cy + 0.001)
    np.testing.assert_array_equal(v.reference_from_geometries(plan10, [left, right]), alone)
    bowtie = Polygon(
        [(cx - 0.001, cy - 0.001), (cx + 0.001, cy + 0.001), (cx + 0.001, cy - 0.001), (cx - 0.001, cy + 0.001)]
    )
    assert v.reference_from_geometries(plan10, [bowtie]).sum() > 10


def test_reference_from_file_reprojects_epsg3857(tmp_path, two_zone):
    poly = box(BOUNDARY_LON - 0.003, BOUNDARY_LAT - 0.002, BOUNDARY_LON + 0.002, BOUNDARY_LAT + 0.003)
    path = tmp_path / "ref.geojson"
    gpd.GeoDataFrame({"name": ["x"]}, geometry=[poly], crs="EPSG:4326").to_crs("EPSG:3857").to_file(
        path, driver="GeoJSON"
    )
    assert gpd.read_file(path).crs.to_epsg() == 3857
    for plan in two_zone:
        got = v.reference_from_file(plan, path)
        np.testing.assert_array_equal(got, v.reference_from_geometries(plan, [poly]))
        assert got.sum() > 100


@pytest.mark.filterwarnings("ignore:'crs' was not provided")
def test_reference_from_file_without_crs(tmp_path, plan10, caplog):
    lon, lat = centres_lonlat(plan10)
    poly = box(
        float(lon.mean()) - 0.001, float(lat.mean()) - 0.001, float(lon.mean()) + 0.001, float(lat.mean()) + 0.001
    )
    path = tmp_path / "nocrs.shp"
    gpd.GeoDataFrame(geometry=[poly]).to_file(path)
    with caplog.at_level("WARNING"):
        got = v.reference_from_file(plan10, path)
    np.testing.assert_array_equal(got, v.reference_from_geometries(plan10, [poly]))
    assert "EPSG:4326 is assumed" in caplog.text
    far = tmp_path / "projected_nocrs.shp"
    gpd.GeoDataFrame(geometry=[box(500000, 4300000, 501000, 4301000)]).to_file(far)
    with pytest.raises(v.ReferenceDataError, match="no CRS"):
        v.reference_from_file(plan10, far)
    with pytest.raises(FileNotFoundError):
        v.reference_from_file(plan10, tmp_path / "missing.geojson")


@pytest.fixture
def lonlat_raster(tmp_path, two_zone):
    """A 4326 raster around the boundary with a fine pattern, a nodata block and a gap to its east."""
    res = 0.0001
    west, north = BOUNDARY_LON - 0.03, BOUNDARY_LAT + 0.02
    h, w = 400, 320  # east edge at -119.998: the 11N part is partly outside
    rows, cols = np.mgrid[0:h, 0:w]
    data = ((rows // 4 + cols // 3) % 7).astype(np.uint8)
    data[150:190, 240:260] = 255  # inside the 10N part
    data[150:190, 302:318] = 255  # inside the 11N part
    path = write_lonlat_tif(tmp_path / "ref.tif", data, west, north, res, nodata=255)
    return path, data, west, north, res


def test_reference_from_raster_values_nodata_and_extent(lonlat_raster, two_zone, monkeypatch):
    path, data, west, north, res = lonlat_raster
    windows = []
    original = rasterio.io.DatasetReader.read

    def spy(self, *args, **kwargs):
        windows.append(kwargs.get("window"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(rasterio.io.DatasetReader, "read", spy)
    for plan in two_zone:
        got = v.reference_from_raster(plan, path, lambda a: a >= 3)
        lon, lat = centres_lonlat(plan)
        vals, inside = lookup(data, west, north, res, lon, lat)
        expected = inside & (vals >= 3) & (vals != 255) & plan.mask
        mismatch = int((got != expected).sum())
        assert mismatch == 0, f"{plan.zone}: {mismatch} pixels disagree with the centre lookup"
        assert not (got & ~plan.mask).any()
    nodata_hit = (vals == 255) & inside & two_zone[-1].mask
    assert nodata_hit.any(), "the nodata block must fall inside the 11N plan for this test to mean anything"
    assert not got[nodata_hit].any()
    outside = ~inside & two_zone[-1].mask
    assert outside.any() and not got[outside].any()
    assert windows and all(w is not None and w.width * w.height < data.size / 4 for w in windows)


def test_reference_from_raster_projected_source(tmp_path, plan10):
    # A raster in the zone's own CRS on a 5 m grid: each 10 m pixel takes the value under its centre.
    t = plan10.transform
    h, w = plan10.shape
    data = np.zeros((2 * h, 2 * w), dtype=np.int16)
    data[:, w:] = 7
    path = tmp_path / "utm.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2 * w,
        height=2 * h,
        count=1,
        dtype="int16",
        crs="EPSG:32610",
        transform=from_origin(t.c, t.f, 5.0, 5.0),
    ) as dst:
        dst.write(data, 1)
    got = v.reference_from_raster(plan10, path, lambda a: a == 7)
    expected = np.zeros(plan10.shape, dtype=bool)
    expected[:, w // 2 :] = True
    np.testing.assert_array_equal(got, expected & plan10.mask)


def test_reference_from_raster_no_overlap_bad_band_bad_predicate(tmp_path, plan10, lonlat_raster):
    far = write_lonlat_tif(tmp_path / "far.tif", np.ones((10, 10), np.uint8), 10.0, 10.0, 0.01)
    assert not v.reference_from_raster(plan10, far, lambda a: a == 1).any()
    path = lonlat_raster[0]
    with pytest.raises(ValueError, match="band 2"):
        v.reference_from_raster(plan10, path, lambda a: a == 1, band=2)
    with pytest.raises(ValueError, match="predicate returned shape"):
        v.reference_from_raster(plan10, path, lambda a: a[:1] == 1)


def test_reference_from_geometries_densifies_long_edges():
    """A long straight lon/lat edge is a curve in UTM; projecting only its end points cuts a chord.

    A 78 km wide window at 38.85 N in 10N: the parallel bows about 95 m (10 rows) between the
    window's edges, so a polygon whose south edge is that parallel must follow it to the pixel.
    """
    lat0, west, east = 38.85, -123.45, -122.55
    fwd = Transformer.from_crs(4326, 32610, always_xy=True)
    lons = np.linspace(west, east, 200)
    xs, ys = fwd.transform(lons, np.full(lons.shape, lat0))
    assert ys.max() - ys.min() > 80  # the parallel is curved in UTM
    left = math.floor(xs.min() / 10) * 10
    width = math.ceil((xs.max() - left) / 10)
    top = math.ceil((ys.max() + 900) / 10) * 10
    height = 200  # 2 km: the curved parallel stays inside across the whole width
    assert top - height * 10 < ys.min() - 500
    grid = ZoneGrid(zone="10N", epsg=32610, left=left, top=top, height=height, width=width)
    plan = ZonePlan(
        zone="10N", grid=grid, rows=slice(0, height), cols=slice(0, width), mask=np.ones((height, width), bool)
    )
    poly = box(-124.0, lat0, -122.0, 39.5)  # only its south edge crosses the window
    got = v.reference_from_geometries(plan, [poly])
    lon, lat = centres_lonlat(plan)
    expected = shapely.contains_xy(poly, lon, lat)
    assert expected.any() and (~expected).any()
    wrong = got != expected
    off_m = np.abs(lat[wrong] - lat0) * 111_000
    assert not wrong.any() or off_m.max() < 0.01, f"{int(wrong.sum())} pixels wrong, up to {off_m.max():.1f} m"
    # Each column's edge row follows the curve: rows differ by several pixels across the window.
    edge_rows = got.sum(axis=0)
    assert edge_rows.max() - edge_rows.min() >= 5


def test_reference_from_file_reads_the_named_layer(tmp_path, plan10):
    lon, lat = centres_lonlat(plan10)
    cx, cy = float(lon.mean()), float(lat.mean())
    first = box(cx - 0.002, cy - 0.002, cx, cy)
    second = box(cx, cy, cx + 0.002, cy + 0.002)
    path = tmp_path / "two.gpkg"
    gpd.GeoDataFrame({"n": [1]}, geometry=[first], crs="EPSG:4326").to_file(path, layer="first", driver="GPKG")
    gpd.GeoDataFrame({"n": [2]}, geometry=[second], crs="EPSG:4326").to_file(path, layer="second", driver="GPKG")
    exp_first = v.reference_from_geometries(plan10, [first])
    exp_second = v.reference_from_geometries(plan10, [second])
    assert exp_first.sum() > 50 and exp_second.sum() > 50 and not (exp_first & exp_second).any()
    np.testing.assert_array_equal(v.reference_from_file(plan10, path, layer="second"), exp_second)
    np.testing.assert_array_equal(v.reference_from_file(plan10, path, layer="first"), exp_first)
    with pytest.warns(UserWarning, match="More than one layer"):
        np.testing.assert_array_equal(v.reference_from_file(plan10, path), exp_first)


def test_reference_from_raster_every_pixel_with_nodata(tmp_path, two_zone):
    """A unique id in every source pixel and nodata blocks: each 10 m pixel takes exactly the source
    pixel under its centre. An approximate transformer picks a neighbour for a few pixels, which the
    ids show even where a coarse predicate would not.
    """
    res = 0.0001
    west, north = BOUNDARY_LON - 0.03, BOUNDARY_LAT + 0.02
    nodata = np.iinfo(np.uint32).max
    data = np.arange(1, 400 * 320 + 1, dtype=np.uint32).reshape(400, 320)
    data[150:190, 240:260] = nodata  # inside the 10N part
    data[150:190, 302:318] = nodata  # inside the 11N part
    path = write_lonlat_tif(tmp_path / "ids.tif", data, west, north, res, nodata=int(nodata))
    for plan in two_zone:
        seen: list[np.ndarray] = []

        def odd(a: np.ndarray, seen: list[np.ndarray] = seen) -> np.ndarray:
            seen.append(a.copy())
            return a % 2 == 1

        got = v.reference_from_raster(plan, path, odd)
        lon, lat = centres_lonlat(plan)
        vals, inside = lookup(data, west, north, res, lon, lat)
        has_data = inside & (vals != nodata)
        assert (inside & ~has_data & plan.mask).any(), "a nodata block must fall inside each plan"
        wrong = int((seen[0] != np.where(has_data, vals, 0)).sum())
        assert wrong == 0, f"{plan.zone}: {wrong} pixels took a neighbouring source pixel"
        np.testing.assert_array_equal(got, has_data & (vals % 2 == 1) & plan.mask)


def test_reference_from_raster_south_up(tmp_path, two_zone):
    # Row 0 is the south edge (positive y pixel size), as xarray writes ascending latitudes.
    rng = np.random.default_rng(8)
    res = 0.0001
    west, south = BOUNDARY_LON - 0.03, BOUNDARY_LAT - 0.02
    data = rng.integers(0, 250, size=(400, 600), dtype=np.uint8)
    path = tmp_path / "south_up.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=600,
        height=400,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(res, 0.0, west, 0.0, res, south),
    ) as dst:
        dst.write(data, 1)
    with rasterio.open(path) as ds:
        assert ds.transform.e > 0
    north_up = data[::-1]
    for plan in two_zone:
        got = v.reference_from_raster(plan, path, lambda a: a % 2 == 1)
        lon, lat = centres_lonlat(plan)
        vals, inside = lookup(north_up, west, south + 400 * res, res, lon, lat)
        assert inside[plan.mask].all()
        assert int((got != (inside & (vals % 2 == 1) & plan.mask)).sum()) == 0, plan.zone


def test_plan_lonlat_bounds(plan10):
    west, south, east, north = v.plan_lonlat_bounds(plan10)
    lon, lat = centres_lonlat(plan10)
    assert west < lon.min() < lon.max() < east
    assert south < lat.min() < lat.max() < north
    w2, s2, e2, n2 = v.plan_lonlat_bounds(plan10, margin_deg=0.01)
    assert (w2, s2, e2, n2) == pytest.approx((west - 0.01, south - 0.01, east + 0.01, north + 0.01))


# --------------------------------------------------------------------------------------------
# Network plumbing
# --------------------------------------------------------------------------------------------


def test_cache_dir_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "c"))
    assert v.cache_dir() == tmp_path / "c"
    assert v.reference_dir("hansen") == tmp_path / "c" / "reference" / "hansen"
    monkeypatch.delenv("TESSERA_APPS_CACHE")
    assert v.cache_dir() == Path.home() / ".cache" / "tessera_apps"


def test_gdal_options_from_environment(monkeypatch):
    for var in (
        "GDAL_HTTP_PROXY",
        "HTTPS_PROXY",
        "https_proxy",
        "CURL_CA_BUNDLE",
        "GDAL_CURL_CA_BUNDLE",
        "SSL_CERT_FILE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/bundle.pem")
    before = dict(__import__("os").environ)
    opts = v.gdal_options()
    assert opts["GDAL_HTTP_PROXY"] == "http://proxy.local:3128"
    assert opts["CURL_CA_BUNDLE"] == "/etc/bundle.pem"
    assert opts["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
    assert dict(__import__("os").environ) == before
    monkeypatch.setenv("GDAL_HTTP_PROXY", "http://other:1")
    monkeypatch.setenv("GDAL_CURL_CA_BUNDLE", "/gdal.pem")
    opts = v.gdal_options()
    assert "GDAL_HTTP_PROXY" not in opts and "CURL_CA_BUNDLE" not in opts
    monkeypatch.delenv("GDAL_HTTP_PROXY")
    monkeypatch.delenv("HTTPS_PROXY")
    monkeypatch.setenv("https_proxy", "http://lower:8")
    assert v.gdal_options()["GDAL_HTTP_PROXY"] == "http://lower:8"


def test_gdal_options_bound_every_request(monkeypatch):
    """Without timeouts a /vsicurl/ read on a stalled connection blocks forever."""
    names = ("GDAL_HTTP_CONNECTTIMEOUT", "GDAL_HTTP_TIMEOUT", "GDAL_HTTP_LOW_SPEED_TIME", "GDAL_HTTP_LOW_SPEED_LIMIT")
    for name in (*names, "GDAL_HTTP_MAX_RETRY"):
        monkeypatch.delenv(name, raising=False)
    opts = v.gdal_options()
    for name in names:
        assert 0 < float(opts[name]) <= 600, name
    assert float(opts["GDAL_HTTP_TIMEOUT"]) * (int(opts["GDAL_HTTP_MAX_RETRY"]) + 1) <= 1800
    # A value the user set is left to GDAL, which reads the environment itself.
    monkeypatch.setenv("GDAL_HTTP_TIMEOUT", "7")
    monkeypatch.setenv("GDAL_HTTP_MAX_RETRY", "0")
    opts = v.gdal_options()
    assert "GDAL_HTTP_TIMEOUT" not in opts and "GDAL_HTTP_MAX_RETRY" not in opts
    assert "GDAL_HTTP_LOW_SPEED_TIME" in opts


class RangeServer:
    """A local HTTP server for ``/vsicurl/`` reads: HEAD, ranged GET, and failures or stalls on demand."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.fail_from: dict[str, int] = {}  # a GET whose range reaches this byte answers HTTP 503
        self.stall_s = 0.0  # every GET waits this long, sends nothing and closes
        self.gets: list[tuple[str, str | None]] = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):
                """Keep the test output quiet."""

            def _empty(self, code: int) -> None:
                self.send_response(code)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_HEAD(self):
                """The file's size, as a static file server answers."""
                body = owner.files.get(self.path.split("?")[0])
                if body is None:
                    self._empty(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()

            def do_GET(self):
                """A byte range of the file, or a stall or HTTP 503 when configured."""
                path = self.path.split("?")[0]
                owner.gets.append((path, self.headers.get("Range")))
                body = owner.files.get(path)
                if body is None:
                    self._empty(404)
                    return
                if owner.stall_s:
                    time.sleep(owner.stall_s)
                    self.close_connection = True
                    return
                m = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
                start, end = (int(m.group(1)), min(int(m.group(2)), len(body) - 1)) if m else (0, len(body) - 1)
                if path in owner.fail_from and end >= owner.fail_from[path]:
                    self._empty(503)
                    return
                self.send_response(206 if m else 200)
                self.send_header("Content-Length", str(end - start + 1))
                if m:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                self.wfile.write(body[start : end + 1])

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def add(self, name: str, body: bytes, *, fail_from: int | None = None) -> str:
        """Serve ``body`` under a fresh path (GDAL caches by URL) and return its URL."""
        path = f"/{uuid.uuid4().hex}/{name}"
        self.files[path] = body
        if fail_from is not None:
            self.fail_from[path] = fail_from
        return f"http://127.0.0.1:{self.server.server_port}{path}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def range_server(monkeypatch) -> Iterator[RangeServer]:
    """A loopback server, reached without any proxy, with GDAL retries off so failures are quick."""
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy", "GDAL_HTTP_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("GDAL_HTTP_MAX_RETRY", "0")
    server = RangeServer()
    yield server
    server.close()


#: With VSI_CACHE on (as gdal_options sets it), GDAL's first request for a remote GeoTIFF is its first
#: 32 KB; tags stored further in and the pixel data come in later requests.
GDAL_OPEN_BYTES = 32768


def geotiff_bytes(data: np.ndarray, west: float, north: float, res: float, **profile) -> bytes:
    """An uncompressed single-band EPSG:4326 GeoTIFF in memory."""
    with rasterio.MemoryFile() as mf:
        with mf.open(
            driver="GTiff",
            width=data.shape[1],
            height=data.shape[0],
            count=1,
            dtype=data.dtype.name,
            crs="EPSG:4326",
            transform=from_origin(west, north, res, res),
            **profile,
        ) as dst:
            dst.write(data, 1)
        return mf.read()


def late_georef_bytes(west: float, north: float, res: float) -> bytes:
    """A GeoTIFF whose georeferencing tags lie far past GDAL's first read, as in real Hansen tiles.

    One row per strip makes the strip arrays long (40000 rows: 240 KB), and GDAL writes the GeoTIFF
    tags after them.
    """
    body = geotiff_bytes(np.ones((40000, 8), dtype=np.uint8), west, north, res, blockysize=1)
    with rasterio.MemoryFile(body) as mf, mf.open() as ds:
        assert ds.crs is not None  # complete, the file is fine
    return body


def test_remote_stall_becomes_an_error_naming_the_host(range_server, plan10, monkeypatch):
    for name, value in (
        ("GDAL_HTTP_TIMEOUT", "3"),
        ("GDAL_HTTP_LOW_SPEED_TIME", "1"),
        ("GDAL_HTTP_CONNECTTIMEOUT", "3"),
    ):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setitem(v._GDAL_DEFAULTS, name, value)
    lon, lat = centres_lonlat(plan10)
    url = range_server.add(
        "stall.tif", geotiff_bytes(np.ones((80, 80), np.uint8), lon.min() - 0.001, lat.max() + 0.001, 0.0001)
    )
    range_server.stall_s = 20.0  # without a timeout GDAL waits for the server to give up
    began = time.monotonic()
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.reference_from_raster(plan10, url, lambda a: a == 1)
    assert time.monotonic() - began < 10
    assert "127.0.0.1" in str(info.value) and "local path" in str(info.value)
    assert range_server.gets


def test_remote_raster_read_failures_name_the_host(range_server, plan10):
    lon, lat = centres_lonlat(plan10)
    west, north, res = float(lon.min()) - 0.001, float(lat.max()) + 0.001, 0.00002
    data = np.random.default_rng(9).integers(1, 250, size=(800, 500), dtype=np.uint8)
    body = geotiff_bytes(data, west, north, res)
    whole = v.reference_from_raster(plan10, range_server.add("ok.tif", body), lambda a: a % 2 == 1)
    assert whole.any()
    # The connection fails after the open: the window read must not escape as a bare GDAL error.
    broken = range_server.add("broken.tif", body, fail_from=GDAL_OPEN_BYTES)
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.reference_from_raster(plan10, broken, lambda a: a % 2 == 1)
    msg = str(info.value)
    assert "127.0.0.1" in msg and broken in msg and "local path" in msg and "Read failed" in msg
    # The request for the GeoTIFF tags fails: GDAL opens the file with no CRS and an identity
    # transform. That is a failed read of the host, not a file without a CRS.
    late = range_server.add("late.tif", late_georef_bytes(west, north, res), fail_from=GDAL_OPEN_BYTES)
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning), pytest.raises(v.ReferenceUnavailableError) as info:
        v.reference_from_raster(plan10, late, lambda a: a == 1)
    msg = str(info.value)
    assert "127.0.0.1" in msg and "georeferencing" in msg and "local path" in msg and "has no CRS" not in msg


def test_hansen_remote_read_failures_name_the_host(range_server, plan10, monkeypatch):
    monkeypatch.setattr(v, "_exists", lambda url, **kwargs: True)
    assert v.hansen_tiles(v._hansen_bounds(plan10)) == ["40N_130W"]
    data = np.random.default_rng(10).integers(0, 26, size=(TILE_H, TILE_W), dtype=np.uint8)
    body = geotiff_bytes(data, TILE_WEST, TILE_NORTH, HANSEN_RES)

    def template_for(body: bytes, fail_from: int | None) -> str:
        url = range_server.add(f"Hansen_{v.HANSEN_VERSION}_lossyear_40N_130W.tif", body, fail_from=fail_from)
        return url.replace(v.HANSEN_VERSION, "{version}").replace("lossyear", "{layer}").replace("40N_130W", "{tile}")

    got = v.hansen_layer(plan10, "lossyear", url_template=template_for(body, None))
    lon, lat = centres_lonlat(plan10)
    vals, inside = lookup(data, TILE_WEST, TILE_NORTH, HANSEN_RES, lon, lat)
    assert inside.all() and (got == vals).all()
    # The plan's rows of this tile lie past GDAL's first read, so the window read is what fails.
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.hansen_layer(plan10, "lossyear", url_template=template_for(body, GDAL_OPEN_BYTES))
    msg = str(info.value)
    assert "127.0.0.1" in msg and "Hansen GFC" in msg and "url_template" in msg and "--reference" in msg
    assert "Read failed" in msg
    # Lost GeoTIFF tags are a failed read of the host, not a tile on the wrong grid.
    late = late_georef_bytes(TILE_WEST, TILE_NORTH, HANSEN_RES)
    with pytest.warns(rasterio.errors.NotGeoreferencedWarning), pytest.raises(v.ReferenceUnavailableError) as info:
        v.hansen_layer(plan10, "lossyear", url_template=template_for(late, GDAL_OPEN_BYTES))
    msg = str(info.value)
    assert "127.0.0.1" in msg and "georeferencing" in msg and "url_template" in msg
    assert "degree pixels" not in msg
    # Only the good read was cached; the failed ones left nothing behind.
    assert len(list(v.reference_dir("hansen").rglob("*.tif"))) == 1 and no_temp_files(v.cache_dir())


def test_check_bbox():
    assert v._check_bbox((-1, -2, 3, 4)) == (-1.0, -2.0, 3.0, 4.0)
    bad_boxes = (
        (1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, 0, 0, 1),  # zero width
        (0, 1, 1, 1),  # zero height
        (0, 0, 1),
        (-181, 0, 0, 1),
        (0, 0, float("nan"), 1),
        (0, -91, 1, 0),
    )
    for bad in bad_boxes:
        with pytest.raises(ValueError):
            v._check_bbox(bad)


def test_download_is_atomic_on_a_dropped_connection(monkeypatch, tmp_path):
    class Dropping:
        """A response that sends one block, then loses the connection."""

        headers = _headers(10_000)

        def __init__(self):
            self.sent = False

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, *exc):
            """Context manager exit."""
            return False

        def read(self, n=-1):
            if self.sent:
                raise ConnectionResetError("connection reset by peer")
            self.sent = True
            return b"x" * 100

    monkeypatch.setattr(v, "_urlopen", lambda request, timeout: Dropping())
    dest = tmp_path / "d" / "file.zip"
    with pytest.raises(v.ReferenceUnavailableError, match=r"Could not reach host\.test for Thing"):
        v._download("https://host.test/file.zip", dest, dataset="Thing", hint="Do X.")
    assert not dest.exists() and no_temp_files(tmp_path)


def test_download_rejects_a_truncated_body(monkeypatch, tmp_path):
    def short(request, timeout):
        return urllib.response.addinfourl(io.BytesIO(b"abc"), _headers(10), request.full_url, 200)

    monkeypatch.setattr(v, "_urlopen", short)
    dest = tmp_path / "file.bin"
    with pytest.raises(v.ReferenceUnavailableError, match="stopped after 3 of 10 bytes"):
        v._download("https://host.test/file.bin", dest, dataset="Thing", hint="Do X.")
    assert not dest.exists() and no_temp_files(tmp_path)


def test_download_reuses_existing_file(tmp_path):
    dest = tmp_path / "have.zip"
    dest.write_bytes(b"cached")
    assert v._download("https://host.test/have.zip", dest, dataset="Thing", hint="") == dest
    assert dest.read_bytes() == b"cached"


def test_exists_distinguishes_missing_from_unreachable(web):
    web.route("https://host.test/there.tif", lambda m, p: b"")
    assert v._exists("https://host.test/there.tif", dataset="D", hint="H") is True
    assert v._exists("https://host.test/missing.tif", dataset="D", hint="H") is False
    web.route("https://host.test/forbidden.tif", lambda m, p: http_error("u", 403, "Forbidden"))
    with pytest.raises(v.ReferenceUnavailableError, match="HTTP 403"):
        v._exists("https://host.test/forbidden.tif", dataset="D", hint="H")
    assert web.urls("HEAD")


# --------------------------------------------------------------------------------------------
# Hansen
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "tile"),
    [
        (-120.5, 38.7, "40N_130W"),
        (-120.0, 38.85, "40N_120W"),  # on the 120 W edge: the tile to its east
        (-119.99, 38.85, "40N_120W"),
        (-63.11, -8.67, "00N_070W"),
        (-81.5, 25.9, "30N_090W"),
        (151.2, -33.9, "30S_150E"),
        (0.0, 0.0, "00N_000E"),
        (-0.01, -0.01, "00N_010W"),
        (5.0, 10.0, "10N_000E"),  # on a latitude edge: the tile whose top edge it is
        (5.0, 10.001, "20N_000E"),
        (-180.0, -55.0, "50S_180W"),
        (180.0, 60.0, "60N_170E"),
        (25.0, -40.0, "40S_020E"),
    ],
)
def test_hansen_tile_names(lon, lat, tile):
    assert v.hansen_tile(lon, lat) == tile


def test_hansen_tile_rejects_bad_points():
    with pytest.raises(ValueError):
        v.hansen_tile(200.0, 0.0)
    with pytest.raises(ValueError):
        v.hansen_tile(0.0, float("nan"))


@pytest.mark.parametrize(
    ("bbox", "tiles"),
    [
        ((-63.20087, -8.75983, -63.01913, -8.58017), ["00N_070W"]),
        ((-120.679, 38.524, -119.908, 38.900), ["40N_130W", "40N_120W"]),
        ((-81.62, 25.75, -81.32, 25.97), ["30N_090W"]),
        ((-0.5, -0.5, 0.5, 0.5), ["10N_010W", "10N_000E", "00N_010W", "00N_000E"]),
        ((-70.0, -10.0, -60.0, 0.0), ["00N_070W"]),  # exactly one tile: edges pull in nothing
        ((-70.0, -10.0, -59.99, 0.01), ["10N_070W", "10N_060W", "00N_070W", "00N_060W"]),
        ((10.0, 30.0, 20.0, 40.0), ["40N_010E"]),
        ((-125.0, 35.0, -125.0, 35.0), ["40N_130W"]),  # zero area: point rule
        ((-120.0, 40.0, -120.0, 40.0), ["40N_120W"]),
        ((-120.0, 38.0, -119.0, 38.0), ["40N_120W"]),
        ((170.5, 1.0, 180.0, 2.0), ["10N_170E"]),
    ],
)
def test_hansen_tiles_for_bbox(bbox, tiles):
    assert v.hansen_tiles(bbox) == tiles


def test_hansen_tiles_rejects_bad_bbox():
    with pytest.raises(ValueError):
        v.hansen_tiles((1.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        v.hansen_tiles((0.0, 0.0, 1.0))


HANSEN_RES = v.HANSEN_RES_DEG
TILE_WEST, TILE_NORTH, TILE_H, TILE_W = BOUNDARY_LON - 0.05, BOUNDARY_LAT + 0.05, 400, 200


def hansen_pattern(layer: str, tile: str) -> np.ndarray:
    rows, cols = np.mgrid[0:TILE_H, 0:TILE_W]
    if layer == "lossyear":
        return ((rows // 3 + cols // 3 + (7 if tile.endswith("120W") else 0)) % 26).astype(np.uint8)
    if layer == "treecover2000":
        return ((rows * 7 + cols * 3) % 101).astype(np.uint8)
    if layer == "datamask":
        return np.where((rows // 4 + cols // 4) % 5 == 0, 2, 1).astype(np.uint8)
    return ((rows + cols) % 2).astype(np.uint8)


def tile_origin(tile: str) -> float:
    return TILE_WEST if tile.endswith("130W") else BOUNDARY_LON


@pytest.fixture
def hansen_dir(tmp_path) -> tuple[Path, str]:
    """Small pieces of tiles 40N_130W and 40N_120W around the 120 W edge, on the real Hansen grid."""
    root = tmp_path / "hansen_tiles"
    root.mkdir()
    for tile in ("40N_130W", "40N_120W"):
        for layer in ("lossyear", "treecover2000", "datamask"):
            write_lonlat_tif(
                root / f"Hansen_{v.HANSEN_VERSION}_{layer}_{tile}.tif",
                hansen_pattern(layer, tile),
                tile_origin(tile),
                TILE_NORTH,
                HANSEN_RES,
            )
    return root, str(root / "Hansen_{version}_{layer}_{tile}.tif")


def hansen_expected(plan: ZonePlan, layer: str) -> np.ndarray:
    lon, lat = centres_lonlat(plan)
    out = np.zeros(plan.shape, dtype=np.uint8)
    for tile in ("40N_130W", "40N_120W"):
        vals, inside = lookup(hansen_pattern(layer, tile), tile_origin(tile), TILE_NORTH, HANSEN_RES, lon, lat)
        out[inside] = vals[inside]
    return out


def test_hansen_layer_mosaics_tiles_onto_each_zone(hansen_dir, two_zone):
    _, template = hansen_dir
    for plan in two_zone:
        got = v.hansen_layer(plan, "lossyear", url_template=template)
        assert got.dtype == np.uint8 and got.shape == plan.shape
        expected = hansen_expected(plan, "lossyear")
        assert (got != expected).sum() == 0
        assert (got[plan.mask] == expected[plan.mask]).mean() > 0.999


def test_hansen_loss_year_window(hansen_dir, two_zone):
    _, template = hansen_dir
    for plan in two_zone:
        lossyear = hansen_expected(plan, "lossyear")
        got = v.hansen_loss(plan, 2019, 2022, url_template=template)
        expected = (lossyear >= 20) & (lossyear <= 22) & plan.mask
        assert (got != expected).sum() == 0
        on_mask = plan.mask & (v.hansen_layer(plan, "lossyear", url_template=template) == lossyear)
        for code in (19, 20, 22, 23):
            pix = on_mask & (lossyear == code)
            assert pix.any(), f"lossyear {code} absent from the {plan.zone} plan"
            assert got[pix].all() == (code in (20, 21, 22))
            assert got[pix].any() == (code in (20, 21, 22))
        assert not (got & ~plan.mask).any()
    since_2000 = v.hansen_loss(two_zone[0], 2000, 2025, url_template=template)
    lossyear = hansen_expected(two_zone[0], "lossyear")
    assert (since_2000 != ((lossyear >= 1) & two_zone[0].mask)).sum() == 0


def test_hansen_forest_and_valid(hansen_dir, two_zone):
    _, template = hansen_dir
    for plan in two_zone:
        cover = hansen_expected(plan, "treecover2000")
        lossyear = hansen_expected(plan, "lossyear")
        datamask = hansen_expected(plan, "datamask")
        forest = v.hansen_forest(plan, 2019, url_template=template)
        expected = (cover >= 30) & ~((lossyear >= 1) & (lossyear <= 19)) & plan.mask
        assert (forest != expected).sum() == 0
        strict = v.hansen_forest(plan, 2019, threshold=75, url_template=template)
        assert (strict != ((cover >= 75) & ~((lossyear >= 1) & (lossyear <= 19)) & plan.mask)).sum() == 0
        assert strict.sum() < forest.sum()
        f2000 = v.hansen_forest(plan, 2000, url_template=template)
        assert (f2000 != ((cover >= 30) & plan.mask)).sum() == 0
        valid = v.hansen_valid(plan, url_template=template)
        assert (valid != ((datamask == 1) & plan.mask)).sum() == 0
        assert valid.any() and (~valid & plan.mask).any()


def test_hansen_tile_pieces_ending_inside_the_window(tmp_path, two_zone):
    """Tiles whose north and west edges fall inside the window are pasted at the right offsets."""
    root = tmp_path / "partial"
    root.mkdir()
    north, south_rows = 38.8515, 200  # 38.8515 is on the Hansen grid and inside both plans
    pieces = {"40N_130W": BOUNDARY_LON - 0.005, "40N_120W": BOUNDARY_LON + 0.001}
    patterns = {}
    for tile, west in pieces.items():
        rows, cols = np.mgrid[0:south_rows, 0:40]
        patterns[tile] = ((rows // 2 + cols // 2) % 26 + 1).astype(np.uint8)
        write_lonlat_tif(
            root / f"Hansen_{v.HANSEN_VERSION}_lossyear_{tile}.tif", patterns[tile], west, north, HANSEN_RES
        )
    template = str(root / "Hansen_{version}_{layer}_{tile}.tif")
    for plan in two_zone:
        got = v.hansen_layer(plan, "lossyear", url_template=template)
        lon, lat = centres_lonlat(plan)
        expected = np.zeros(plan.shape, dtype=np.uint8)
        for tile, west in pieces.items():
            vals, inside = lookup(patterns[tile], west, north, HANSEN_RES, lon, lat)
            expected[inside] = vals[inside]
        assert (lat > north).any() and (expected == 0).any() and (expected > 0).any()
        assert (got != expected).sum() == 0


def test_hansen_missing_tile_reads_as_zero(hansen_dir, two_zone):
    root, template = hansen_dir
    for p in root.glob("*40N_120W.tif"):
        p.unlink()
    east = two_zone[1]
    assert not v.hansen_layer(east, "lossyear", url_template=template)[east.mask].any()
    assert not v.hansen_valid(east, url_template=template).any()
    west = two_zone[0]
    assert v.hansen_valid(west, url_template=template).any()


def test_hansen_cache_is_reused_and_written_atomically(hansen_dir, two_zone):
    root, template = hansen_dir
    plan = two_zone[0]
    first = v.hansen_layer(plan, "treecover2000", url_template=template)
    cached = list(v.reference_dir("hansen").rglob("*.tif"))
    assert len(cached) == 1 and cached[0].parent.name == v.HANSEN_VERSION
    with rasterio.open(cached[0]) as ds:
        assert ds.crs.to_epsg() == 4326
        assert ds.res == pytest.approx((HANSEN_RES, HANSEN_RES))
        assert ds.transform.c / HANSEN_RES == pytest.approx(round(ds.transform.c / HANSEN_RES))
    for p in root.iterdir():
        p.unlink()
    np.testing.assert_array_equal(v.hansen_layer(plan, "treecover2000", url_template=template), first)
    assert no_temp_files(v.cache_dir())
    # A different layer or plan is a different cache entry.
    assert not v.hansen_layer(two_zone[1], "treecover2000", url_template=template).any()


def test_hansen_cache_is_per_url_template(hansen_dir, two_zone, tmp_path):
    """A run with a wrong local template caches zeros; a rerun with the right one must read its tiles."""
    _, template = hansen_dir
    plan = two_zone[0]
    wrong = str(tmp_path / "nowhere" / "Hansen_{version}_{layer}_{tile}.tif")
    assert not v.hansen_layer(plan, "lossyear", url_template=wrong).any()
    got = v.hansen_layer(plan, "lossyear", url_template=template)
    expected = hansen_expected(plan, "lossyear")
    assert expected.any() and (got != expected).sum() == 0
    assert len(list(v.reference_dir("hansen").rglob("*.tif"))) == 2


def test_hansen_http_404_reads_zero_and_unreachable_names_host(web, two_zone):
    template = "https://tiles.example.test/{version}/Hansen_{version}_{layer}_{tile}.tif"
    plan = two_zone[0]
    got = v.hansen_layer(plan, "lossyear", url_template=template)
    assert not got.any()
    heads = web.urls("HEAD")
    assert heads == [
        f"https://tiles.example.test/{v.HANSEN_VERSION}/Hansen_{v.HANSEN_VERSION}_lossyear_40N_130W.tif",
        f"https://tiles.example.test/{v.HANSEN_VERSION}/Hansen_{v.HANSEN_VERSION}_lossyear_40N_120W.tif",
    ]

    def refuse(method, params):
        return proxy_refusal()

    web.route(heads[0].replace("lossyear", "datamask"), refuse)
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.hansen_valid(plan, url_template=template)
    msg = str(info.value)
    assert "tiles.example.test" in msg and "Hansen GFC" in msg and "403 Forbidden" in msg
    assert "url_template" in msg and "--reference" in msg


def test_hansen_argument_checks(two_zone):
    plan = two_zone[0]
    with pytest.raises(ValueError, match="before"):
        v.hansen_loss(plan, 2022, 2019)
    with pytest.raises(ValueError, match="before"):
        v.hansen_loss(plan, 2020, 2020)
    with pytest.raises(ValueError, match="2000 to 2025"):
        v.hansen_loss(plan, 2019, 2026)
    with pytest.raises(ValueError, match="2000 to 2024"):
        v.hansen_loss(plan, 2019, 2025, version="GFC-2024-v1.12")
    with pytest.raises(ValueError, match="outside"):
        v.hansen_loss(plan, 1999, 2005)
    with pytest.raises(ValueError, match="percentage"):
        v.hansen_forest(plan, 2019, threshold=101)
    with pytest.raises(ValueError, match="outside"):
        v.hansen_forest(plan, 1999)
    with pytest.raises(ValueError, match="multi-band"):
        v.hansen_layer(plan, "first")
    with pytest.raises(ValueError, match="not like"):
        v.hansen_loss(plan, 2019, 2020, version="v1.13")


# --------------------------------------------------------------------------------------------
# ArcGIS REST fakes (NIFC, USFS FACTS)
# --------------------------------------------------------------------------------------------

_OPS = {
    "=": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def eval_where(where: str, props: dict[str, Any], types: dict[str, str]) -> bool:
    """A small SQL-92 evaluator for the where clauses ArcGIS accepts; type errors raise TypeError."""
    for clause in re.split(r"\s+AND\s+", where.strip()):
        m = re.fullmatch(r"UPPER\((\w+)\) LIKE '%(.*)%'", clause)
        if m:
            if m.group(2).replace("''", "'") not in str(props[m.group(1)]).upper():
                return False
            continue
        m = re.fullmatch(r"(\w+)\s*(>=|<=|=|<|>)\s*DATE '(\d{4})-(\d{2})-(\d{2})'", clause)
        if m:
            if types[m.group(1)] != "esriFieldTypeDate":
                raise TypeError(clause)
            ms = datetime(int(m.group(3)), int(m.group(4)), int(m.group(5)), tzinfo=UTC).timestamp() * 1000
            if not _OPS[m.group(2)](props[m.group(1)], ms):
                return False
            continue
        m = re.fullmatch(r"(\w+)\s*(>=|<=|=|<|>)\s*'(.*)'", clause)
        if m:
            if types[m.group(1)] != "esriFieldTypeString":
                raise TypeError(clause)
            if not _OPS[m.group(2)](str(props[m.group(1)]), m.group(3)):
                return False
            continue
        m = re.fullmatch(r"(\w+)\s*(>=|<=|=|<|>)\s*(-?\d+)", clause)
        if m:
            if types[m.group(1)] in ("esriFieldTypeString", "esriFieldTypeDate"):
                raise TypeError(clause)
            if not _OPS[m.group(2)](props[m.group(1)], int(m.group(3))):
                return False
            continue
        raise AssertionError(f"clause the fake does not understand: {clause!r}")
    return True


class FakeArcGIS:
    """One ArcGIS layer (FeatureServer or MapServer) per the REST API: layer JSON and /query."""

    def __init__(
        self,
        web: FakeWeb,
        layer_url: str,
        fields: list[dict[str, str]],
        features: list[dict[str, Any]],
        *,
        max_records: int = 2,
        paging: bool = True,
        flag_at_top: bool = False,
        ignore_offset: bool = False,
    ) -> None:
        self.layer_url = layer_url
        self.fields = [{"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"}, *fields]
        self.types = {f["name"]: f["type"] for f in self.fields}
        self.features = features
        self.max_records = max_records
        self.paging = paging
        self.flag_at_top = flag_at_top
        self.ignore_offset = ignore_offset
        self.queries: list[dict[str, str]] = []
        web.route(layer_url, self.layer)
        web.route(layer_url + "/query", self.query)

    def layer(self, method, params):
        assert params.get("f") == "json"
        return {
            "currentVersion": 11.3,
            "name": "layer",
            "type": "Feature Layer",
            "objectIdField": "OBJECTID",
            "maxRecordCount": self.max_records,
            "advancedQueryCapabilities": {"supportsPagination": self.paging, "supportsOrderBy": True},
            "fields": self.fields,
        }

    def _matching(self, params):
        west, south, east, north = (float(x) for x in params["geometry"].split(","))
        assert params["geometryType"] == "esriGeometryEnvelope" and params["inSR"] == "4326"
        assert params["spatialRel"] == "esriSpatialRelIntersects"
        env = box(west, south, east, north)
        return [
            f
            for f in self.features
            if eval_where(params["where"], f["properties"], self.types) and shape(f["geometry"]).intersects(env)
        ]

    def query(self, method: str, params: dict[str, str]) -> Any:
        self.queries.append(params)
        assert method == "POST"
        try:
            if params.get("returnIdsOnly") == "true":
                ids = [f["properties"]["OBJECTID"] for f in self._matching(params)]
                return {"objectIdFieldName": "OBJECTID", "objectIds": ids[::-1]}
            assert params["outSR"] == "4326" and params["f"] == "geojson" and params["returnGeometry"] == "true"
            if "objectIds" in params:
                wanted = {int(i) for i in params["objectIds"].split(",")}
                assert len(wanted) <= self.max_records
                return {"type": "FeatureCollection", "features": [f for f in self.features if f["id"] in wanted]}
            if not self.paging:
                raise AssertionError("paging parameters sent to a layer without pagination")
            matches = self._matching(params)
            matches.sort(key=lambda f: f["properties"][params["orderByFields"]])
            count = int(params["resultRecordCount"])
            assert count <= self.max_records
            offset = 0 if self.ignore_offset else int(params["resultOffset"])
            page = matches[offset : offset + count]
            more = offset + count < len(matches)
            out: dict[str, Any] = {"type": "FeatureCollection", "features": page}
            if more and self.flag_at_top:
                out["exceededTransferLimit"] = True
            elif more:
                out["properties"] = {"exceededTransferLimit": True}
            return out
        except TypeError as exc:
            return {"error": {"code": 400, "message": "Unable to complete operation.", "details": [str(exc)]}}


def arcgis_feature(oid: int, geom, **props) -> dict[str, Any]:
    return {"type": "Feature", "id": oid, "geometry": mapping(geom), "properties": {"OBJECTID": oid, **props}}


CALDOR_BBOX = (-120.68, 38.52, -119.9, 38.9)


def nifc_features(year_type: str = "int") -> list[dict[str, Any]]:
    def year(y):
        return y if year_type == "int" else str(y)

    def fy(y):
        return {"FIRE_YEAR_INT" if year_type == "int" else "FIRE_YEAR": year(y)}

    return [
        arcgis_feature(1, box(-120.6, 38.6, -120.3, 38.8), INCIDENT="Caldor", **fy(2021)),
        arcgis_feature(2, box(-120.3, 38.7, -120.0, 38.85), INCIDENT="CALDOR", **fy(2021)),
        arcgis_feature(3, box(-120.0, 38.75, -119.95, 38.88), INCIDENT="Caldor Fire", **fy(2021)),
        arcgis_feature(4, box(-120.5, 38.55, -120.45, 38.6), INCIDENT="Mosquito", **fy(2021)),
        arcgis_feature(5, box(-120.5, 38.55, -120.45, 38.6), INCIDENT="Caldor", **fy(2020)),
        arcgis_feature(6, box(-121.6, 39.6, -121.5, 39.7), INCIDENT="Caldor", **fy(2021)),
        arcgis_feature(7, box(-120.4, 38.6, -120.35, 38.65), INCIDENT="Dixie", **fy(2021)),
    ]


INT_FIELDS = [
    {"name": "INCIDENT", "type": "esriFieldTypeString"},
    {"name": "FIRE_YEAR_INT", "type": "esriFieldTypeSmallInteger"},
]


def test_nifc_discovers_fields_pages_and_filters(web):
    fake = FakeArcGIS(web, v.NIFC_HISTORY_URL, INT_FIELDS, nifc_features())
    geoms = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="caldor")
    assert sorted(g.bounds for g in geoms) == sorted(
        shape(f["geometry"]).bounds for f in nifc_features() if f["id"] in (1, 2, 3)
    )
    where = fake.queries[0]["where"]
    assert where == "FIRE_YEAR_INT=2021 AND UPPER(INCIDENT) LIKE '%CALDOR%'"
    assert [q["resultOffset"] for q in fake.queries] == ["0", "2"]
    assert all(q["orderByFields"] == "OBJECTID" and q["resultRecordCount"] == "2" for q in fake.queries)
    assert web.urls()[0] == v.NIFC_HISTORY_URL
    every = v.nifc_perimeters(CALDOR_BBOX, year=2021)
    assert len(every) == 5  # Caldor x3, Mosquito, Dixie; feature 6 is outside the bbox, 5 is 2020


def test_nifc_string_year_field_and_flag_at_top(web):
    fields = [{"name": "INCIDENT", "type": "esriFieldTypeString"}, {"name": "FIRE_YEAR", "type": "esriFieldTypeString"}]
    fake = FakeArcGIS(web, v.NIFC_HISTORY_URL, fields, nifc_features("str"), flag_at_top=True, max_records=1)
    geoms = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")
    assert len(geoms) == 3
    assert fake.queries[0]["where"] == "FIRE_YEAR='2021' AND UPPER(INCIDENT) LIKE '%CALDOR%'"
    assert len(fake.queries) == 3


def test_nifc_query_url_and_quotes_in_names(web):
    feats = [arcgis_feature(1, box(-120.6, 38.6, -120.3, 38.8), INCIDENT="O'Brien", FIRE_YEAR_INT=2021)]
    FakeArcGIS(web, v.NIFC_HISTORY_URL, INT_FIELDS, feats)
    geoms = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="o'brien", url=v.NIFC_HISTORY_URL + "/query")
    assert len(geoms) == 1


def test_nifc_caches_by_query(web, monkeypatch):
    FakeArcGIS(web, v.NIFC_HISTORY_URL, INT_FIELDS, nifc_features())
    first = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")
    files = list(v.reference_dir("nifc").glob("*.geojson"))
    assert len(files) == 1 and no_temp_files(v.cache_dir())
    offline(monkeypatch)
    again = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")
    assert [g.wkt for g in again] == [g.wkt for g in first]
    with pytest.raises(AssertionError, match="network used"):
        v.nifc_perimeters(CALDOR_BBOX, year=2020, name="Caldor")


def test_nifc_cache_is_per_bbox_and_empty_answers_are_not_cached(web):
    fake = FakeArcGIS(web, v.NIFC_HISTORY_URL, INT_FIELDS, nifc_features())
    caldor = v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")
    asked = len(fake.queries)
    far = v.nifc_perimeters((-121.7, 39.5, -121.4, 39.8), year=2021, name="Caldor")
    assert len(caldor) == 3 and len(far) == 1
    assert far[0].equals(box(-121.6, 39.6, -121.5, 39.7))
    assert len(fake.queries) > asked
    cached = sorted(v.reference_dir("nifc").iterdir())
    assert len(cached) == 2
    # Nothing there: answered, but not cached, so a later run asks again.
    assert v.nifc_perimeters((-100.0, 30.0, -99.0, 31.0), year=2021) == []
    assert sorted(v.reference_dir("nifc").iterdir()) == cached
    asked = len(fake.queries)
    assert v.nifc_perimeters((-100.0, 30.0, -99.0, 31.0), year=2021) == []
    assert len(fake.queries) > asked


def test_usfs_cache_is_per_bbox(web):
    fake = FakeArcGIS(web, v.USFS_REFORESTATION_URL, USFS_FIELDS, usfs_features(), paging=True)
    wide = v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2019, completed_to=2020)
    asked = len(fake.queries)
    narrow = v.usfs_reforestation((-120.505, 38.795, -120.485, 38.815), completed_from=2019, completed_to=2020)
    assert sorted(wide["OBJECTID"]) == [1, 3, 9] and list(narrow["OBJECTID"]) == [1]
    assert len(fake.queries) > asked


def test_nifc_errors(web):
    FakeArcGIS(web, v.NIFC_HISTORY_URL, [{"name": "INCIDENT", "type": "esriFieldTypeString"}], [])
    with pytest.raises(v.ReferenceDataError, match="no fire year field"):
        v.nifc_perimeters(CALDOR_BBOX, year=2021)
    web.route(v.NIFC_HISTORY_URL, lambda m, p: {"error": {"code": 499, "message": "Token Required", "details": []}})
    with pytest.raises(v.ReferenceDataError, match="returned error 499: Token Required"):
        v.nifc_perimeters(CALDOR_BBOX, year=2021)
    web.route(v.NIFC_HISTORY_URL, lambda m, p: b"<html>maintenance</html>")
    with pytest.raises(v.ReferenceDataError, match="did not return JSON"):
        v.nifc_perimeters(CALDOR_BBOX, year=2021)
    web.route(v.NIFC_HISTORY_URL, lambda m, p: proxy_refusal())
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")
    msg = str(info.value)
    assert "services3.arcgis.com" in msg and "NIFC" in msg and "--reference" in msg and "403" in msg
    with pytest.raises(ValueError):
        v.nifc_perimeters((0, 0, 0, 0), year=2021)


def test_nifc_server_type_error_surfaces(web):
    # A string year field queried as a number is an ArcGIS 400 error, not an empty answer.
    fields = [{"name": "INCIDENT", "type": "esriFieldTypeString"}, {"name": "FIRE_YEAR", "type": "esriFieldTypeString"}]
    fake = FakeArcGIS(web, v.NIFC_HISTORY_URL, fields, nifc_features("str"))
    fake.types["FIRE_YEAR"] = "esriFieldTypeString"
    with pytest.raises(v.ReferenceDataError, match="Unable to complete"):
        v._arcgis_query(
            v.NIFC_HISTORY_URL,
            fake.layer("GET", {"f": "json"}),
            where="FIRE_YEAR=2021",
            bbox=CALDOR_BBOX,
            dataset="NIFC",
            hint="",
        )


def test_arcgis_paging_detects_a_server_ignoring_offset(web):
    FakeArcGIS(web, v.NIFC_HISTORY_URL, INT_FIELDS, nifc_features(), ignore_offset=True)
    with pytest.raises(v.ReferenceDataError, match="same page twice"):
        v.nifc_perimeters(CALDOR_BBOX, year=2021, name="Caldor")


USFS_FIELDS = [
    {"name": "ACTIVITY", "type": "esriFieldTypeString"},
    {"name": "DATE_COMPLETED", "type": "esriFieldTypeDate"},
    {"name": "NBR_UNITS_ACCOMPLISHED", "type": "esriFieldTypeDouble"},
]


def ms(y, m=6, d=1):
    return int(datetime(y, m, d, tzinfo=UTC).timestamp() * 1000)


def usfs_features() -> list[dict[str, Any]]:
    return [
        arcgis_feature(1, box(-120.5, 38.8, -120.49, 38.81), ACTIVITY="Plant Trees", DATE_COMPLETED=ms(2019)),
        arcgis_feature(3, box(-120.48, 38.8, -120.47, 38.81), ACTIVITY="Plant Trees", DATE_COMPLETED=ms(2020, 12, 31)),
        arcgis_feature(5, box(-120.46, 38.8, -120.45, 38.81), ACTIVITY="Plant Trees", DATE_COMPLETED=ms(2018, 12, 31)),
        arcgis_feature(7, box(-120.44, 38.8, -120.43, 38.81), ACTIVITY="Plant Trees", DATE_COMPLETED=ms(2021, 1, 1)),
        arcgis_feature(9, box(-120.42, 38.8, -120.41, 38.81), ACTIVITY="Fill-in", DATE_COMPLETED=ms(2020, 1, 1)),
    ]


def test_usfs_reforestation_by_object_id_without_pagination(web, monkeypatch):
    fake = FakeArcGIS(web, v.USFS_REFORESTATION_URL, USFS_FIELDS, usfs_features(), paging=False)
    frame = v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2019, completed_to=2020)
    assert sorted(frame["OBJECTID"]) == [1, 3, 9]
    assert frame.crs.to_epsg() == 4326
    assert isinstance(frame, gpd.GeoDataFrame)
    assert pd.api.types.is_datetime64_any_dtype(frame["DATE_COMPLETED"])
    assert frame["DATE_COMPLETED"].min().year == 2019
    ids_query = fake.queries[0]
    assert ids_query["returnIdsOnly"] == "true"
    assert ids_query["where"] == "DATE_COMPLETED >= DATE '2019-01-01' AND DATE_COMPLETED < DATE '2021-01-01'"
    assert [q["objectIds"] for q in fake.queries[1:]] == ["1,3", "9"]
    offline(monkeypatch)
    again = v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2019, completed_to=2020)
    assert sorted(again["OBJECTID"]) == [1, 3, 9]
    assert pd.api.types.is_datetime64_any_dtype(again["DATE_COMPLETED"])


def test_usfs_reforestation_paged_layer_and_edges(web):
    FakeArcGIS(web, v.USFS_REFORESTATION_URL, USFS_FIELDS, usfs_features(), paging=True)
    frame = v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2021, completed_to=2021)
    assert list(frame["OBJECTID"]) == [7]
    empty = v.usfs_reforestation((-100.0, 30.0, -99.0, 31.0), completed_from=2019, completed_to=2020)
    assert len(empty) == 0 and empty.crs.to_epsg() == 4326
    with pytest.raises(ValueError, match="before"):
        v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2020, completed_to=2019)


def test_usfs_unreachable_and_missing_field(web):
    FakeArcGIS(web, v.USFS_REFORESTATION_URL, [{"name": "ACTIVITY", "type": "esriFieldTypeString"}], [])
    with pytest.raises(v.ReferenceDataError, match="no DATE_COMPLETED"):
        v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2019, completed_to=2020)
    web.route(v.USFS_REFORESTATION_URL, lambda m, p: proxy_refusal())
    with pytest.raises(v.ReferenceUnavailableError, match=r"apps\.fs\.usda\.gov.*--reference"):
        v.usfs_reforestation((-120.6, 38.7, -120.3, 38.9), completed_from=2019, completed_to=2020)


# --------------------------------------------------------------------------------------------
# EFFIS WFS fake
# --------------------------------------------------------------------------------------------

EFFIS_TEST_URL = "https://effis.example.test/effis"


def wfs_capabilities(names: list[str], formats: list[str], count_default: int | None = None) -> bytes:
    values = "".join(f"<ows:Value>{f}</ows:Value>" for f in formats)
    types = "".join(f"<FeatureType><Name>{n}</Name><Title>{n}</Title></FeatureType>" for n in names)
    constraint = (
        ""
        if count_default is None
        else f'<ows:Constraint name="CountDefault"><ows:NoValues/><ows:DefaultValue>{count_default}'
        "</ows:DefaultValue></ows:Constraint>"
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<wfs:WFS_Capabilities version="2.0.0" xmlns:wfs="http://www.opengis.net/wfs/2.0"
    xmlns:ows="http://www.opengis.net/ows/1.1" xmlns="http://www.opengis.net/wfs/2.0">
  <ows:OperationsMetadata>
    <ows:Operation name="GetFeature">
      <ows:Parameter name="outputFormat"><ows:AllowedValues>{values}</ows:AllowedValues></ows:Parameter>
    </ows:Operation>
    <ows:Constraint name="ImplementsResultPaging">
      <ows:NoValues/><ows:DefaultValue>TRUE</ows:DefaultValue>
    </ows:Constraint>
    {constraint}
  </ows:OperationsMetadata>
  <FeatureTypeList>{types}</FeatureTypeList>
</wfs:WFS_Capabilities>""".encode()


def effis_features() -> list[dict[str, Any]]:
    def f(i, geom, date):
        return {
            "type": "Feature",
            "id": f"ba.{i}",
            "geometry": mapping(geom),
            "properties": {"id": i, "FIREDATE": date},
        }

    return [
        f(1, box(23.1, 38.8, 23.3, 39.0), "2021-08-03 11:20:00"),
        f(2, box(23.2, 38.9, 23.25, 38.95), "2021-08-10"),
        f(3, box(23.3, 38.75, 23.35, 38.8), "2020-07-14 09:00:00"),
        f(4, box(23.05, 38.72, 23.07, 38.74), "14/09/2021"),
        f(5, box(25.0, 40.0, 25.1, 40.1), "2021-08-01"),
    ]


#: An empty GetFeature answer as MapServer writes it for WFS 2.0 (GML 3.2).
MAPSERVER_EMPTY_GML = b"""<?xml version="1.0" encoding="UTF-8"?>
<wfs:FeatureCollection xmlns:ms="http://mapserver.gis.umn.edu/mapserver" xmlns:gml="http://www.opengis.net/gml/3.2"
   xmlns:wfs="http://www.opengis.net/wfs/2.0" timeStamp="2024-05-01T10:00:00" numberMatched="0" numberReturned="0">
</wfs:FeatureCollection>"""


class FakeWFS:
    """A WFS 2.0 endpoint: GetCapabilities and paged GetFeature in GeoJSON or GML.

    ``cap`` is a server-side limit on features per response (MapServer ``wfs_maxfeatures``,
    GeoServer ``maxFeatures``), applied whatever ``COUNT`` asks for; ``advertise_cap`` also lists it
    as ``CountDefault``. ``matched`` adds ``numberMatched`` to each page. ``ignore_start`` serves the
    first page for every ``STARTINDEX``.
    """

    def __init__(
        self,
        web: FakeWeb,
        url: str,
        features: list[dict[str, Any]],
        *,
        formats: list[str],
        name: str = "ms:effis.nrt.ba.poly",
        swap: bool = False,
        tmp: Path | None = None,
        cap: int | None = None,
        advertise_cap: bool = False,
        matched: bool = False,
        ignore_start: bool = False,
    ) -> None:
        self.features = features
        self.formats = formats
        self.name = name
        self.swap = swap
        self.tmp = tmp
        self.cap = cap
        self.advertise_cap = advertise_cap
        self.matched = matched
        self.ignore_start = ignore_start
        self.requests: list[dict[str, str]] = []
        web.route(url, self.handle)

    def get_features(self) -> list[dict[str, str]]:
        return [r for r in self.requests if r["REQUEST"] == "GetFeature"]

    def handle(self, method, params):
        params = {k.upper(): val for k, val in params.items()}
        self.requests.append(params)
        assert params["SERVICE"] == "WFS"
        if params["REQUEST"] == "GetCapabilities":
            return wfs_capabilities(
                ["ms:effis.nrt.ba.poly.30d", self.name, "ms:modis.ba.poly"],
                self.formats,
                self.cap if self.advertise_cap else None,
            )
        assert params["REQUEST"] == "GetFeature" and params["VERSION"] == "2.0.0"
        assert params["TYPENAMES"] == self.name
        lat0, lon0, lat1, lon1, crs = params["BBOX"].split(",")
        assert crs == "urn:ogc:def:crs:EPSG::4326"
        env = box(float(lon0), float(lat0), float(lon1), float(lat1))
        hits = [f for f in self.features if shape(f["geometry"]).intersects(env)]
        start, count = int(params["STARTINDEX"]), int(params["COUNT"])
        if self.ignore_start:
            start = 0
        if self.cap is not None:
            count = min(count, self.cap)
        page = hits[start : start + count]
        if self.swap:
            page = [
                {**f, "geometry": mapping(shapely.transform(shape(f["geometry"]), lambda xy: xy[:, ::-1]))}
                for f in page
            ]
        if "OUTPUTFORMAT" in params:
            assert params["OUTPUTFORMAT"] in self.formats
            out = {"type": "FeatureCollection", "features": page}
            if self.matched:
                out.update(numberMatched=len(hits), numberReturned=len(page))
            return out
        if not page:
            if self.matched:
                return MAPSERVER_EMPTY_GML.replace(b'numberMatched="0"', f'numberMatched="{len(hits)}"'.encode())
            return b'<?xml version="1.0"?><wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"/>'
        path = self.tmp / f"page_{uuid.uuid4().hex}.gml"
        gpd.GeoDataFrame.from_features(page, crs="EPSG:4326").to_file(path, driver="GML")
        body = path.read_bytes()
        if self.matched:
            root = re.search(rb"<(\w+:)?FeatureCollection\b", body)
            assert root is not None
            body = body[: root.end()] + f' numberMatched="{len(hits)}"'.encode() + body[root.end() :]
        return body


GREECE_BBOX = (23.0, 38.7, 23.5, 39.1)


def test_effis_geojson_paging_and_year_filter(web):
    fake = FakeWFS(
        web,
        EFFIS_TEST_URL,
        effis_features(),
        formats=["text/xml; subtype=gml/3.2.1", "application/json; subtype=geojson"],
    )
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL, page_size=2)
    assert sorted(g.bounds for g in geoms) == sorted(
        shape(f["geometry"]).bounds for f in effis_features() if f["properties"]["id"] in (1, 2, 4)
    )
    gets = [r for r in fake.requests if r["REQUEST"] == "GetFeature"]
    # Four features intersect the bbox (one is from 2020): two full pages, then an empty one.
    assert [r["STARTINDEX"] for r in gets] == ["0", "2", "4"]
    assert all(r["COUNT"] == "2" for r in gets)
    assert gets[0]["BBOX"] == "38.7,23.0,39.1,23.5,urn:ogc:def:crs:EPSG::4326"
    assert gets[0]["OUTPUTFORMAT"] == "application/json; subtype=geojson"
    assert gets[0]["SRSNAME"] == "urn:ogc:def:crs:EPSG::4326"


def test_effis_swaps_latitude_first_output(web):
    FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["geojson"], swap=True)
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    assert len(geoms) == 3
    assert all(box(*GREECE_BBOX).contains(g) for g in geoms)


def test_effis_gml_when_no_json_format(web, tmp_path, monkeypatch):
    FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["text/xml; subtype=gml/3.2.1"], tmp=tmp_path)
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2020, url=EFFIS_TEST_URL)
    assert len(geoms) == 1
    assert geoms[0].equals(box(23.3, 38.75, 23.35, 38.8))
    assert list(v.reference_dir("effis").glob("*.geojson"))
    offline(monkeypatch)
    assert len(v.effis_burnt_areas(GREECE_BBOX, year=2020, url=EFFIS_TEST_URL)) == 1


EFFIS_2021 = sorted(shape(f["geometry"]).bounds for f in effis_features() if f["properties"]["id"] in (1, 2, 4))
GML = ["text/xml; subtype=gml/3.2.1"]


@pytest.mark.parametrize("formats", [["application/json; subtype=geojson"], GML], ids=["geojson", "gml"])
def test_effis_pages_past_a_server_cap_below_count(web, tmp_path, formats):
    """A server that sends fewer features than COUNT (wfs_maxfeatures) has not necessarily sent them all."""
    fake = FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=formats, tmp=tmp_path, cap=2)
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)  # COUNT 1000
    assert sorted(g.bounds for g in geoms) == EFFIS_2021
    gets = fake.get_features()
    assert [r["STARTINDEX"] for r in gets] == ["0", "2", "4"]  # 2 + 2 features, then an empty page
    assert all(r["COUNT"] == "1000" for r in gets)
    assert len(list(v.reference_dir("effis").glob("*.geojson"))) == 1


@pytest.mark.parametrize("formats", [["geojson"], GML], ids=["geojson", "gml"])
def test_effis_uses_count_default_and_number_matched(web, tmp_path, formats):
    fake = FakeWFS(
        web, EFFIS_TEST_URL, effis_features(), formats=formats, tmp=tmp_path, cap=2, advertise_cap=True, matched=True
    )
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    assert sorted(g.bounds for g in geoms) == EFFIS_2021
    gets = fake.get_features()
    assert [r["COUNT"] for r in gets] == ["2", "2"]  # the advertised cap
    assert [r["STARTINDEX"] for r in gets] == ["0", "2"]  # numberMatched (4) reached: no empty page asked for


@pytest.mark.parametrize("page_size", [4, 2, 3, 1000])
def test_effis_gml_empty_pages(web, tmp_path, page_size):
    """An empty GML page (no burnt area in the bbox, or a full last page) is an empty list, not a crash."""
    fake = FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=GML, tmp=tmp_path)
    geoms = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL, page_size=page_size)
    assert sorted(g.bounds for g in geoms) == EFFIS_2021
    assert len(fake.get_features()) == -(-4 // page_size) + 1  # the pages, then one empty page
    assert v.effis_burnt_areas((10.0, 45.0, 10.5, 45.5), year=2021, url=EFFIS_TEST_URL, page_size=page_size) == []


@pytest.mark.parametrize(
    "body",
    [
        MAPSERVER_EMPTY_GML,
        b'<?xml version="1.0"?><wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"/>',
        b'<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs" xmlns:gml="http://www.opengis.net/gml"'
        b' numberOfFeatures="0"><gml:boundedBy><gml:null>missing</gml:null></gml:boundedBy></wfs:FeatureCollection>',
    ],
    ids=["mapserver", "geoserver", "wfs11"],
)
def test_parse_wfs_features_empty_gml(body):
    assert v._parse_wfs_features(body, as_json=False) == []
    assert v._parse_wfs_page(MAPSERVER_EMPTY_GML, as_json=False) == ([], 0)


def test_parse_wfs_features_rejects_bodies_it_cannot_read():
    with pytest.raises(v.ReferenceDataError, match="did not return GML"):
        v._parse_wfs_features(b"<html><body>maintenance", as_json=False)
    with pytest.raises(v.ReferenceDataError, match="could not read the GML"):
        v._parse_wfs_features(
            b'<wfs:FeatureCollection xmlns:wfs="http://www.opengis.net/wfs/2.0"><wfs:member>junk</wfs:member>'
            b"</wfs:FeatureCollection>",
            as_json=False,
        )
    with pytest.raises(v.ReferenceDataError, match="not an object"):
        v._parse_wfs_features(b"[1, 2]", as_json=True)


def test_effis_refuses_a_server_that_ignores_startindex(web):
    FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["geojson"], cap=2, ignore_start=True)
    with pytest.raises(v.ReferenceDataError, match="same page twice"):
        v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    assert not v.reference_dir("effis").exists() or not list(v.reference_dir("effis").iterdir())


def test_effis_paging_that_fails_part_way_is_not_cached(web):
    fake = FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["geojson"], cap=2)

    def flaky(method, params):
        return proxy_refusal() if params.get("STARTINDEX") == "2" else fake.handle(method, params)

    web.route(EFFIS_TEST_URL, flaky)
    with pytest.raises(v.ReferenceUnavailableError, match=r"effis\.example\.test"):
        v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    assert not v.reference_dir("effis").exists() or not list(v.reference_dir("effis").iterdir())
    web.route(EFFIS_TEST_URL, fake.handle)
    assert sorted(g.bounds for g in v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)) == EFFIS_2021


def test_effis_cache_is_per_bbox(web):
    fake = FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["geojson"])
    wide = v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    asked = len(fake.get_features())
    small = v.effis_burnt_areas((23.04, 38.71, 23.08, 38.75), year=2021, url=EFFIS_TEST_URL)
    assert len(wide) == 3 and len(small) == 1
    assert small[0].equals(box(23.05, 38.72, 23.07, 38.74))
    assert len(fake.get_features()) > asked


def test_effis_errors(web):
    FakeWFS(web, EFFIS_TEST_URL, effis_features(), formats=["geojson"], name="ms:other.layer")
    with pytest.raises(v.ReferenceDataError, match=r"does not list effis\.nrt\.ba\.poly"):
        v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    report = (
        b'<?xml version="1.0"?><ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/1.1">'
        b"<ows:Exception><ows:ExceptionText>msWFSGetFeature(): invalid bbox</ows:ExceptionText></ows:Exception>"
        b"</ows:ExceptionReport>"
    )

    def broken(method, params):
        if params["REQUEST"] == "GetCapabilities":
            return wfs_capabilities(["ms:effis.nrt.ba.poly"], ["geojson"])
        return report

    web.route(EFFIS_TEST_URL, broken)
    with pytest.raises(v.ReferenceDataError, match="invalid bbox"):
        v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)

    def no_date(method, params):
        if params["REQUEST"] == "GetCapabilities":
            return wfs_capabilities(["ms:effis.nrt.ba.poly"], ["geojson"])
        feature = {"type": "Feature", "geometry": mapping(box(23.1, 38.8, 23.2, 38.9)), "properties": {"DATE": "2021"}}
        return {"type": "FeatureCollection", "features": [feature] if params["STARTINDEX"] == "0" else []}

    web.route(EFFIS_TEST_URL, no_date)
    with pytest.raises(v.ReferenceDataError, match="no FIREDATE"):
        v.effis_burnt_areas(GREECE_BBOX, year=2021, url=EFFIS_TEST_URL)
    web.route(v.EFFIS_WFS_URL, lambda m, p: proxy_refusal())
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.effis_burnt_areas(GREECE_BBOX, year=2021)
    assert "maps.effis.emergency.copernicus.eu" in str(info.value) and "--reference" in str(info.value)


@pytest.mark.parametrize(
    ("value", "year"),
    [
        ("2021-08-03 11:20:00", 2021),
        ("14/09/2019", 2019),
        ("20180704", 2018),
        (datetime(2017, 1, 2, tzinfo=UTC), 2017),
        (pd.Timestamp("2016-03-04"), 2016),
        (ms(2015), 2015),
        (np.int64(ms(2014)), 2014),
        (2013, 2013),
        (None, None),
        ("unknown", None),
        (float("nan"), None),
        (12, None),
    ],
)
def test_year_of(value, year):
    assert v._year_of(value) == year


# --------------------------------------------------------------------------------------------
# ESA WorldCover
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lon", "lat", "tile"),
    [
        (-81.5, 25.9, "N24W084"),  # the Everglades demo AOI
        (-84.0, 25.0, "N24W084"),  # on a tile's west edge: that tile
        (-81.0, 25.0, "N24W081"),  # on its east edge: the tile to the east
        (-81.5, 27.0, "N24W084"),  # on a latitude edge: the tile whose top edge it is
        (-81.5, 27.001, "N27W084"),
        (-81.5, 24.0, "N21W084"),
        (0.0, 0.0, "S03E000"),
        (-0.01, -0.01, "S03W003"),
        (-59.0, -2.0, "S03W060"),
        (151.2, -33.9, "S36E150"),
        (2.9, 5.5, "N03E000"),
        (3.0, 5.5, "N03E003"),
        (180.0, 60.0, "N57E177"),
        (-180.0, -55.0, "S57W180"),
        (10.0, 90.0, "N87E009"),
        (10.0, -90.0, "S90E009"),
    ],
)
def test_worldcover_tile_names(lon, lat, tile):
    assert v.worldcover_tile(lon, lat) == tile


def test_worldcover_tile_rejects_bad_points():
    for lon, lat in ((200.0, 0.0), (0.0, float("nan")), (0.0, 91.0)):
        with pytest.raises(ValueError, match="lon/lat"):
            v.worldcover_tile(lon, lat)


@pytest.mark.parametrize(
    ("bbox", "tiles"),
    [
        ((-81.62, 25.75, -81.32, 25.97), ["N24W084"]),
        ((-84.0, 24.0, -81.0, 27.0), ["N24W084"]),  # exactly one tile: edges pull in nothing
        ((-84.0, 24.0, -80.99, 27.01), ["N27W084", "N27W081", "N24W084", "N24W081"]),
        ((-0.5, -0.5, 0.5, 0.5), ["N00W003", "N00E000", "S03W003", "S03E000"]),
        ((-120.05, 38.8, -119.95, 38.9), ["N36W123", "N36W120"]),
        ((-81.0, 27.0, -81.0, 27.0), ["N24W081"]),  # zero area: the point rule
        ((-82.0, 25.0, -81.0, 25.0), ["N24W084"]),
        ((178.0, 10.0, 180.0, 11.0), ["N09E177"]),
        ((-180.0, -60.0, -177.0, -57.0), ["S60W180"]),
    ],
)
def test_worldcover_tiles_for_bbox(bbox, tiles):
    assert v.worldcover_tiles(bbox) == tiles


def test_worldcover_tiles_rejects_bad_bbox():
    with pytest.raises(ValueError, match="lon/lat"):
        v.worldcover_tiles((1.0, 0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="west, south"):
        v.worldcover_tiles((0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        v.worldcover_tiles((0.0, 0.0, float("inf"), 1.0))


WC_RES = 1.0 / v.WORLDCOVER_PIXELS_PER_DEG
WC_NORTH, WC_H, WC_W = 38.9, 1200, 600
WC_WEST = {"N36W123": -120.05, "N36W120": -120.0}
WC_CODES = np.array([10, 20, 30, 80, 90, 95, 95, 40], dtype=np.uint8)


def wc_pattern(year: int, tile: str) -> np.ndarray:
    """Class codes laid in 3 x 4 pixel blocks, shifted by year and tile, with no-data (0) stripes."""
    rows, cols = np.mgrid[0:WC_H, 0:WC_W]
    shift = (3 if year == 2021 else 0) + (5 if tile == "N36W120" else 0)
    codes = WC_CODES[(rows // 3 + cols // 4 + shift) % WC_CODES.size]
    return np.where((rows // 50) % 7 == 3, np.uint8(0), codes).astype(np.uint8)


def wc_name(year: int, tile: str) -> str:
    return f"ESA_WorldCover_10m_{year}_{v.WORLDCOVER_VERSIONS[year]}_{tile}_Map.tif"


@pytest.fixture
def worldcover_dir(tmp_path) -> tuple[Path, str]:
    """Pieces of WorldCover tiles N36W123 and N36W120 around the 120 W edge, for 2020 (v100) and 2021 (v200)."""
    root = tmp_path / "worldcover_tiles"
    root.mkdir()
    for year in (2020, 2021):
        for tile, west in WC_WEST.items():
            write_lonlat_tif(root / wc_name(year, tile), wc_pattern(year, tile), west, WC_NORTH, WC_RES, nodata=0)
    return root, str(root / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")


def wc_expected(plan: ZonePlan, year: int) -> np.ndarray:
    lon, lat = centres_lonlat(plan)
    out = np.zeros(plan.shape, dtype=np.uint8)
    for tile, west in WC_WEST.items():
        vals, inside = lookup(wc_pattern(year, tile), west, WC_NORTH, WC_RES, lon, lat)
        out[inside] = vals[inside]
    return out


def test_worldcover_map_mosaics_tiles_onto_each_zone(worldcover_dir, two_zone):
    _, template = worldcover_dir
    for plan in two_zone:
        for year in (2020, 2021):
            got = v.worldcover_map(plan, year, url_template=template)
            expected = wc_expected(plan, year)
            assert got.dtype == np.uint8 and got.shape == plan.shape
            assert (got != expected).sum() == 0
            assert set(np.unique(expected[plan.mask]).tolist()) >= {0, 10, 80, 90, 95}


def test_worldcover_classes_and_valid(worldcover_dir, two_zone):
    _, template = worldcover_dir
    for plan in two_zone:
        codes = wc_expected(plan, 2020)
        mangrove = v.worldcover_classes(plan, 2020, [95], url_template=template)
        assert mangrove.dtype == np.bool_ and (mangrove != ((codes == 95) & plan.mask)).sum() == 0
        wet = v.worldcover_classes(plan, 2020, (90, 95), url_template=template)
        assert (wet != (np.isin(codes, [90, 95]) & plan.mask)).sum() == 0 and wet.sum() > mangrove.sum()
        valid = v.worldcover_valid(plan, 2020, url_template=template)
        assert (valid != ((codes != 0) & plan.mask)).sum() == 0
        assert valid.any() and (~valid & plan.mask).any()
        assert not (mangrove & ~plan.mask).any() and not (valid & ~plan.mask).any()


def test_worldcover_years_read_their_own_version(worldcover_dir, two_zone, tmp_path):
    """2020 comes from v100 and 2021 from v200, never one map for both years."""
    root, template = worldcover_dir
    plan = two_zone[0]
    m20 = v.worldcover_map(plan, 2020, url_template=template)
    m21 = v.worldcover_map(plan, 2021, url_template=template)
    assert (m20 != m21)[plan.mask].mean() > 0.5
    assert (m20 != wc_expected(plan, 2020)).sum() == 0 and (m21 != wc_expected(plan, 2021)).sum() == 0
    # Without the v200 tiles, 2021 has no data at all: it is never filled from the 2020 map.
    only = tmp_path / "only_2020"
    only.mkdir()
    for p in root.glob("*_2020_v100_*"):
        (only / p.name).write_bytes(p.read_bytes())
    only_2020 = str(only / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
    assert not v.worldcover_map(plan, 2021, url_template=only_2020).any()
    assert not v.worldcover_valid(plan, 2021, url_template=only_2020).any()
    np.testing.assert_array_equal(v.worldcover_map(plan, 2020, url_template=only_2020), m20)


def test_worldcover_refuses_other_years_and_classes(worldcover_dir, two_zone):
    _, template = worldcover_dir
    plan = two_zone[0]
    for year in (2018, 2019, 2022, 2024):
        with pytest.raises(ValueError, match=r"2020 \(v100\) and 2021 \(v200\) only"):
            v.worldcover_map(plan, year, url_template=template)
        with pytest.raises(ValueError, match="2020"):
            v.worldcover_classes(plan, year, [95], url_template=template)
        with pytest.raises(ValueError, match="2021"):
            v.worldcover_valid(plan, year, url_template=template)
    with pytest.raises(ValueError, match="no class"):
        v.worldcover_classes(plan, 2020, [95, 96], url_template=template)
    with pytest.raises(ValueError, match="at least one"):
        v.worldcover_classes(plan, 2020, [], url_template=template)
    assert not list(v.reference_dir("worldcover").rglob("*.tif"))


def test_worldcover_missing_tile_reads_as_no_data(worldcover_dir, two_zone):
    root, template = worldcover_dir
    for p in root.glob("*N36W120*"):
        p.unlink()
    west, east = two_zone
    assert not v.worldcover_map(east, 2020, url_template=template)[east.mask].any()
    assert not v.worldcover_valid(east, 2020, url_template=template).any()
    assert not v.worldcover_classes(east, 2020, [95], url_template=template).any()
    assert v.worldcover_valid(west, 2020, url_template=template).any()


def test_worldcover_cache_is_reused_and_written_atomically(worldcover_dir, two_zone):
    root, template = worldcover_dir
    plan = two_zone[0]
    first = v.worldcover_map(plan, 2020, url_template=template)
    cached = list(v.reference_dir("worldcover").rglob("*.tif"))
    assert len(cached) == 1 and cached[0].parent.name == "v100" and cached[0].name.startswith("map_2020_")
    with rasterio.open(cached[0]) as ds:
        assert ds.crs.to_epsg() == 4326 and ds.nodata == 0
        assert ds.res == pytest.approx((WC_RES, WC_RES))
        assert ds.transform.c * v.WORLDCOVER_PIXELS_PER_DEG == pytest.approx(round(ds.transform.c / WC_RES))
        assert ds.transform.f * v.WORLDCOVER_PIXELS_PER_DEG == pytest.approx(round(ds.transform.f / WC_RES))
    for p in root.iterdir():
        p.unlink()
    np.testing.assert_array_equal(v.worldcover_map(plan, 2020, url_template=template), first)
    assert no_temp_files(v.cache_dir())
    # Another year or another plan is another window, read from the (now missing) local tiles: no
    # data, and not cached, so the tiles are read once they are back.
    assert not v.worldcover_map(plan, 2021, url_template=template).any()
    assert not v.worldcover_map(two_zone[1], 2020, url_template=template).any()
    names = sorted(p.parent.name for p in v.reference_dir("worldcover").rglob("*.tif"))
    assert names == ["v100"] and no_temp_files(v.cache_dir())


def test_worldcover_local_tiles_put_in_place_later_are_read(worldcover_dir, two_zone, tmp_path):
    """A window built while a local tile was missing is never cached (finding F2), wholly or partly."""
    root, _ = worldcover_dir
    plan = two_zone[0]
    assert v.worldcover_tiles(v.plan_lonlat_bounds(plan, margin_deg=0.001)) == ["N36W123", "N36W120"]
    later = tmp_path / "downloaded_later"
    later.mkdir()
    template = str(later / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
    # No tile yet: no data, nothing cached.
    assert not v.worldcover_map(plan, 2020, url_template=template).any()
    assert not list(v.reference_dir("worldcover").rglob("*.tif"))
    # One of the two tiles: only its part of the window, still not cached.
    west = wc_name(2020, "N36W123")
    (later / west).write_bytes((root / west).read_bytes())
    partial = v.worldcover_map(plan, 2020, url_template=template)
    lon, _ = centres_lonlat(plan)
    full = wc_expected(plan, 2020)
    assert partial.any() and not partial[lon >= -120.0].any()
    np.testing.assert_array_equal(partial[lon < -120.0], full[lon < -120.0])
    assert not list(v.reference_dir("worldcover").rglob("*.tif"))
    # Both tiles: the whole window, now cached and reused.
    east = wc_name(2020, "N36W120")
    (later / east).write_bytes((root / east).read_bytes())
    np.testing.assert_array_equal(v.worldcover_map(plan, 2020, url_template=template), full)
    assert len(list(v.reference_dir("worldcover").rglob("*.tif"))) == 1
    for p in later.iterdir():
        p.unlink()
    np.testing.assert_array_equal(v.worldcover_map(plan, 2020, url_template=template), full)
    assert no_temp_files(v.cache_dir())


def test_worldcover_cache_is_per_url_template(worldcover_dir, two_zone, tmp_path):
    _, template = worldcover_dir
    plan = two_zone[0]
    wrong = str(tmp_path / "nowhere" / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
    assert not v.worldcover_map(plan, 2020, url_template=wrong).any()
    got = v.worldcover_map(plan, 2020, url_template=template)
    assert got.any() and (got != wc_expected(plan, 2020)).sum() == 0


def test_worldcover_rejects_a_tile_off_its_grid(tmp_path, two_zone):
    root = tmp_path / "off_grid"
    root.mkdir()
    for tile, west in WC_WEST.items():
        write_lonlat_tif(root / wc_name(2020, tile), wc_pattern(2020, tile), west + WC_RES / 3, WC_NORTH, WC_RES)
    template = str(root / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
    with pytest.raises(v.ReferenceDataError, match="WorldCover grid"):
        v.worldcover_map(two_zone[0], 2020, url_template=template)
    coarse = tmp_path / "coarse"
    coarse.mkdir()
    write_lonlat_tif(coarse / wc_name(2020, "N36W123"), np.full((10, 10), 95, np.uint8), -120.05, WC_NORTH, 0.001)
    with pytest.raises(v.ReferenceDataError, match="1/12000"):
        v.worldcover_map(
            two_zone[0], 2020, url_template=str(coarse / "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif")
        )


def test_worldcover_http_404_reads_no_data_and_unreachable_names_host(web, two_zone):
    template = "https://wc.example.test/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    plan = two_zone[0]
    assert not v.worldcover_map(plan, 2021, url_template=template).any()
    heads = web.urls("HEAD")
    assert heads == [
        "https://wc.example.test/v200/2021/map/ESA_WorldCover_10m_2021_v200_N36W123_Map.tif",
        "https://wc.example.test/v200/2021/map/ESA_WorldCover_10m_2021_v200_N36W120_Map.tif",
    ]

    def refuse(method, params):
        return proxy_refusal()

    web.route(heads[0].replace("2021", "2020").replace("v200", "v100"), refuse)
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.worldcover_classes(plan, 2020, [95], url_template=template)
    msg = str(info.value)
    assert "wc.example.test" in msg and "ESA WorldCover" in msg and "403 Forbidden" in msg
    assert "url_template" in msg and "--reference" in msg


def test_worldcover_remote_tile_read_through_vsicurl(range_server, plan10, monkeypatch):
    monkeypatch.setattr(v, "_exists", lambda url, **kwargs: True)
    data = wc_pattern(2020, "N36W123")
    name = wc_name(2020, "N36W123")
    url = range_server.add(name, geotiff_bytes(data, WC_WEST["N36W123"], WC_NORTH, WC_RES))
    assert url.endswith(name)
    template = url[: -len(name)] + "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    assert v.worldcover_tiles(v.plan_lonlat_bounds(plan10, margin_deg=0.001)) == ["N36W123"]
    got = v.worldcover_map(plan10, 2020, url_template=template)
    lon, lat = centres_lonlat(plan10)
    vals, inside = lookup(data, WC_WEST["N36W123"], WC_NORTH, WC_RES, lon, lat)
    assert inside.all() and (got == vals).all()


# --------------------------------------------------------------------------------------------
# Global Mangrove Watch
# --------------------------------------------------------------------------------------------


def test_gmw_version_and_year_checks(plan10):
    with pytest.raises(ValueError, match=r"v3 has epochs.*v4\.1\.12"):
        v.gmw_extent(plan10, 2024)
    with pytest.raises(ValueError, match=r"v3 has epochs"):
        v.gmw_extent(plan10, 2011, version="v3")
    with pytest.raises(ValueError, match=r"v4\.1\.12 covers 1985 to 2025"):
        v.gmw_extent(plan10, 1984, version="v4.1.12")
    with pytest.raises(ValueError, match="not supported"):
        v.gmw_extent(plan10, 2018, version="v4")
    assert v._gmw_version("V3", 2018) == "v3"
    assert v._gmw_version("v4.1.12", 2024) == "v4.1.12"
    for year in (1996, 2007, 2010, 2015, 2020):
        assert v._gmw_version("v3", year) == "v3"


def gmw_polys(plan: ZonePlan) -> tuple[Polygon, Polygon]:
    lon, lat = centres_lonlat(plan)
    cx, cy = float(lon.mean()), float(lat.mean())
    return box(cx - 0.002, cy - 0.002, cx, cy + 0.002), box(cx, cy - 0.002, cx + 0.002, cy)


def test_gmw_vector_source_file_and_directory(tmp_path, plan10):
    a, b = gmw_polys(plan10)
    d = tmp_path / "gmw"
    d.mkdir()
    gpd.GeoDataFrame({"PXLVAL": [1]}, geometry=[a], crs="EPSG:4326").to_file(d / "gmw_v3_2018_vec.gpkg")
    gpd.GeoDataFrame({"PXLVAL": [1]}, geometry=[b], crs="EPSG:4326").to_file(d / "gmw_v3_2020_vec.gpkg")
    exp_a = v.reference_from_geometries(plan10, [a])
    exp_b = v.reference_from_geometries(plan10, [b])
    assert exp_a.sum() > 50 and exp_b.sum() > 50 and not (exp_a & exp_b).all()
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018, source=d / "gmw_v3_2018_vec.gpkg"), exp_a)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018, source=d), exp_a)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2020, source=d), exp_b)
    np.testing.assert_array_equal(v.gmw_coverage(plan10, 2020, source=d), plan10.mask)
    with pytest.raises(FileNotFoundError, match="gmw_v3_2016_vec"):
        v.gmw_extent(plan10, 2016, source=d)
    with pytest.raises(FileNotFoundError):
        v.gmw_extent(plan10, 2018, source=tmp_path / "nothing")


def test_gmw_v4_raster_band_is_year_minus_1984(tmp_path, plan10):
    lon, lat = centres_lonlat(plan10)
    res = 0.00025
    west, north = float(lon.min()) - 0.001, float(lat.max()) + 0.001
    h = int((north - float(lat.min()) + 0.002) / res)
    w = int((float(lon.max()) + 0.001 - west) / res) // 2  # covers only the west half of the plan
    bands = np.zeros((41, h, w), dtype=np.uint8)
    bands[2018 - 1985, :, : w // 2] = 1
    bands[2024 - 1985, :, w // 2 :] = 1
    bands[2019 - 1985] = 1
    path = write_lonlat_tif(tmp_path / "GMW_N39W121_v4.1.12.tif", bands, west, north, res)
    e2018 = v.gmw_extent(plan10, 2018, version="v4.1.12", source=path)
    e2024 = v.gmw_extent(plan10, 2024, version="v4.1.12", source=tmp_path)
    cov = v.gmw_coverage(plan10, 2024, version="v4.1.12", source=path)
    vals18, inside = lookup(bands[2018 - 1985], west, north, res, lon, lat)
    vals24, _ = lookup(bands[2024 - 1985], west, north, res, lon, lat)
    assert (e2018 != ((vals18 == 1) & inside & plan10.mask)).sum() == 0
    assert (e2024 != ((vals24 == 1) & inside & plan10.mask)).sum() == 0
    assert (cov != (inside & plan10.mask)).sum() == 0
    assert e2018.any() and e2024.any() and not (e2018 & e2024).any()
    assert cov.any() and (plan10.mask & ~cov).any()
    (tmp_path / "one").mkdir()
    single = write_lonlat_tif(tmp_path / "one" / "x.tif", bands[0], west, north, res)
    assert not v.gmw_extent(plan10, 2018, version="v4.1.12", source=single).any()
    three = write_lonlat_tif(tmp_path / "three.tif", bands[:3], west, north, res)
    with pytest.raises(v.ReferenceDataError, match="3 bands"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=three)


def gmw_grid(plan: ZonePlan) -> tuple[float, float, float, int, int]:
    """A GMW-like 0.00025 degree grid around a plan: (west, north, res, height, width)."""
    lon, lat = centres_lonlat(plan)
    res = 0.00025
    west, north = float(lon.min()) - 0.001, float(lat.max()) + 0.001
    h = int((north - float(lat.min()) + 0.002) / res)
    w = int((float(lon.max()) + 0.001 - west) / res) + 1
    return west, north, res, h, w


def gmw_expected(plan: ZonePlan, arr: np.ndarray, west: float, north: float, res: float) -> np.ndarray:
    lon, lat = centres_lonlat(plan)
    vals, inside = lookup(arr, west, north, res, lon, lat)
    return (vals == 1) & inside & plan.mask


def test_gmw_v4_folder_reads_only_the_years_single_band_tiles(tmp_path, plan10):
    """Per-year single-band tiles of one cell: each year reads its own tile, never the union."""
    west, north, res, h, w = gmw_grid(plan10)
    y2018 = np.zeros((h, w), np.uint8)
    y2018[:, : w // 2] = 1
    y2024 = np.zeros((h, w), np.uint8)
    y2024[:, w // 3 :] = 1
    flat = tmp_path / "flat"
    flat.mkdir()
    write_lonlat_tif(flat / "GMW_N39W121_2018_v4.1.12.tif", y2018, west, north, res)
    write_lonlat_tif(flat / "GMW_N39W121_2024_v4.1.12.tif", y2024, west, north, res)
    nested = tmp_path / "nested"  # the year in a folder name counts too
    for year, arr in ((2018, y2018), (2024, y2024)):
        (nested / str(year)).mkdir(parents=True)
        write_lonlat_tif(nested / str(year) / "GMW_N39W121_v4.1.12.tif", arr, west, north, res)
    exp18 = gmw_expected(plan10, y2018, west, north, res)
    exp24 = gmw_expected(plan10, y2024, west, north, res)
    assert exp18.any() and exp24.any() and (exp18 != exp24).any()
    for source in (flat, nested):
        np.testing.assert_array_equal(v.gmw_extent(plan10, 2018, version="v4.1.12", source=source), exp18)
        np.testing.assert_array_equal(v.gmw_extent(plan10, 2024, version="v4.1.12", source=source), exp24)
    with pytest.raises(FileNotFoundError, match="other years"):
        v.gmw_extent(plan10, 2020, version="v4.1.12", source=flat)


def test_gmw_never_reads_another_version_from_a_folder(tmp_path, plan10):
    west, north, res, h, w = gmw_grid(plan10)
    bands = np.zeros((41, h, w), np.uint8)
    bands[2018 - 1985, :, : w // 2] = 1
    d = tmp_path / "gmw"
    d.mkdir()
    write_lonlat_tif(d / "GMW_N39W121_v4.1.12.tif", bands, west, north, res)
    v3 = write_lonlat_tif(d / "GMW_N39W121_2018_v3.tif", np.ones((h, w), np.uint8), west, north, res)
    write_lonlat_tif(d / "GMW_N39W121_2018_v4.0.tif", np.ones((h, w), np.uint8), west, north, res)
    expected = gmw_expected(plan10, bands[2018 - 1985], west, north, res)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018, version="v4.1.12", source=d), expected)
    everything = gmw_expected(plan10, np.ones((h, w), np.uint8), west, north, res)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018, version="v3", source=d), everything)
    with pytest.raises(v.ReferenceDataError, match=r"named for v3, not GMW v4\.1\.12"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=v3)
    vec = tmp_path / "vec"
    vec.mkdir()
    a, _ = gmw_polys(plan10)
    gpd.GeoDataFrame({"PXLVAL": [1]}, geometry=[a], crs="EPSG:4326").to_file(vec / "gmw_v3_2018_vec.gpkg")
    with pytest.raises(FileNotFoundError, match="another version"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=vec)


def test_gmw_single_band_v4_files_must_be_dated(tmp_path, plan10, caplog):
    west, north, res, h, w = gmw_grid(plan10)
    ones = np.ones((h, w), np.uint8)
    undated = tmp_path / "undated"
    undated.mkdir()
    tile = write_lonlat_tif(undated / "GMW_N39W121_v4.1.12.tif", ones, west, north, res)
    with pytest.raises(v.ReferenceDataError, match="no year in its name"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=undated)
    with caplog.at_level("WARNING"):  # passed by itself, it is read as given
        got = v.gmw_extent(plan10, 2018, version="v4.1.12", source=tile)
    np.testing.assert_array_equal(got, gmw_expected(plan10, ones, west, north, res))
    assert "no year in its name" in caplog.text
    other = write_lonlat_tif(tmp_path / "GMW_N39W121_2024_v4.1.12.tif", ones, west, north, res)
    with pytest.raises(v.ReferenceDataError, match=r"named for \[2024\], not 2018"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=other)
    span = write_lonlat_tif(tmp_path / "GMW_N39W121_2015_2020.tif", ones, west, north, res)
    with pytest.raises(v.ReferenceDataError, match="one band"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=span)
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    write_lonlat_tif(mixed / "GMW_N39W121_v4.1.12.tif", np.zeros((41, h, w), np.uint8), west, north, res)
    write_lonlat_tif(mixed / "GMW_N39W121_2018_v4.1.12.tif", ones, west, north, res)
    with pytest.raises(v.ReferenceDataError, match="mixes 41-band tiles"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=mixed)
    twice = tmp_path / "twice"
    (twice / "a").mkdir(parents=True)
    (twice / "b").mkdir()
    write_lonlat_tif(twice / "a" / "GMW_N39W121_2018.tif", ones, west, north, res)
    write_lonlat_tif(twice / "b" / "gmw_n39w121_2018_v4.1.12.tif", ones, west, north, res)
    with pytest.raises(v.ReferenceDataError, match="two single-band files for cell N39W121"):
        v.gmw_extent(plan10, 2018, version="v4.1.12", source=twice)


@pytest.mark.parametrize("kind", ["single", "annual"])
def test_gmw_raster_coverage_is_the_tile_footprint(tmp_path, plan10, kind):
    """A 0/1 tile that declares nodata=0 still covers its non-mangrove pixels."""
    west, north, res, h, w = gmw_grid(plan10)
    w = w // 2  # the tile covers the west half of the plan
    if kind == "single":
        data = np.zeros((h, w), np.uint8)
        data[:, : w // 2] = 1
        layers = {2018: data}
        path = write_lonlat_tif(tmp_path / "GMW_N39W121_2018_v4.1.12.tif", data, west, north, res, nodata=0)
    else:
        data = np.zeros((41, h, w), np.uint8)
        data[2018 - 1985, :, : w // 2] = 1  # nothing in 2024
        layers = {2018: data[2018 - 1985], 2024: data[2024 - 1985]}
        path = write_lonlat_tif(tmp_path / "GMW_N39W121_v4.1.12.tif", data, west, north, res, nodata=0)
    lon, lat = centres_lonlat(plan10)
    _, inside = lookup(layers[2018], west, north, res, lon, lat)
    footprint = inside & plan10.mask
    assert footprint.any() and (plan10.mask & ~footprint).any()
    for year, arr in layers.items():
        extent = v.gmw_extent(plan10, year, version="v4.1.12", source=path)
        coverage = v.gmw_coverage(plan10, year, version="v4.1.12", source=path)
        np.testing.assert_array_equal(extent, gmw_expected(plan10, arr, west, north, res))
        np.testing.assert_array_equal(coverage, footprint)
        assert (coverage & ~extent).any()
        # Scored inside the coverage, predicting mangrove everywhere is not a perfect score.
        scores = v.binary_scores(np.ones(plan10.shape, bool), extent, coverage)
        assert scores.fp > 0 and scores.precision is not None and scores.precision < 1


def test_pick_gmw_files():
    files = [
        ("gmw_v3_2016_vec.zip", "u16"),
        ("gmw_v3_2018_gpkg.zip", "u18g"),
        ("gmw_v3_2018_vec.gpkg", "u18p"),
        ("gmw_v3_2018_vec.zip", "u18"),
        ("gmw_v3_2018_rasters.zip", "u18r"),
    ]
    bbox = (-81.62, 25.75, -81.32, 25.97)
    assert v._pick_gmw_files(files, "v3", 2018, bbox) == [("gmw_v3_2018_vec.zip", "u18")]
    assert v._pick_gmw_files(files[:3], "v3", 2018, bbox) == [("gmw_v3_2018_vec.gpkg", "u18p")]
    with pytest.raises(v.ReferenceDataError, match="gmw_v3_2020_vec"):
        v._pick_gmw_files(files, "v3", 2020, bbox)
    tiles = [
        ("GMW_N26W082_v4.1.12.tif", "a"),  # top-left name of the cell holding the bbox
        ("GMW_N25W082_v4.1.12.tif", "b"),  # bottom-left name of the same cell
        ("GMW_N27W082_v4.1.12.tif", "c"),
        ("GMW_N26W080_v4.1.12.tif", "d"),
        ("GMW_S26W082_v4.1.12.tif", "e"),
        ("readme.txt", "f"),
    ]
    assert sorted(k for k, _ in v._pick_gmw_files(tiles, "v4.1.12", 2018, bbox)) == [
        "GMW_N25W082_v4.1.12.tif",
        "GMW_N26W082_v4.1.12.tif",
    ]
    assert v._pick_gmw_files(tiles, "v4.1.12", 2018, (10.2, 5.2, 10.4, 5.4)) == []
    with pytest.raises(v.ReferenceDataError, match="per-tile rasters"):
        v._pick_gmw_files([("gmw_v4_all.zip", "z")], "v4.1.12", 2018, bbox)


def test_pick_gmw_files_keeps_one_year_and_one_version():
    """If the record ships per-year tiles, only the asked year's are downloaded, and never another version."""
    bbox = (-81.62, 25.75, -81.32, 25.97)
    tiles = [
        ("GMW_N26W082_2018_v4.1.12.tif", "a18"),
        ("GMW_N26W082_2020_v4.1.12.tif", "a20"),
        ("GMW_N26W082_2024_v4.1.12.tif", "a24"),
        ("GMW_N25W082_2018_v4.1.12.tif", "b18"),
        ("GMW_N26W082_2018_v3.tif", "v3"),
        ("GMW_N26W082_2018_v4.0.tif", "v40"),
        ("GMW_N26W082_2018_v4.tif", "v4"),  # v4 fits v4.1.12
    ]
    assert sorted(u for _, u in v._pick_gmw_files(tiles, "v4.1.12", 2018, bbox)) == ["a18", "b18", "v4"]
    assert [u for _, u in v._pick_gmw_files(tiles, "v4.1.12", 2024, bbox)] == ["a24"]
    with pytest.raises(v.ReferenceDataError, match="named for other years, none for 2022"):
        v._pick_gmw_files(tiles, "v4.1.12", 2022, bbox)
    span = [("GMW_N26W082_1985_2025_v4.1.12.tif", "all")]  # a span of years is one file per cell
    assert v._pick_gmw_files(span, "v4.1.12", 2018, bbox) == span
    assert v._pick_gmw_files(tiles, "v4.1.12", 2018, (10.2, 5.2, 10.4, 5.4)) == []


def test_cell_names_cover_both_corner_conventions():
    assert v._cell_names((-81.62, 25.75, -81.32, 25.97)) == {"N25W082", "N26W082"}
    assert v._cell_names((-0.5, -0.5, 0.5, 0.5)) == {"S01W001", "N00W001", "S01E000", "N00E000", "N01W001", "N01E000"}


def shapefile_zip(path: Path, frame: gpd.GeoDataFrame, stem: str) -> bytes:
    work = path.parent / f"{stem}_work"
    work.mkdir()
    frame.to_file(work / f"{stem}.shp")
    with zipfile.ZipFile(path, "w") as zf:
        for p in sorted(work.iterdir()):
            zf.write(p, p.name)
    return path.read_bytes()


def test_gmw_downloads_from_zenodo_listing(web, tmp_path, plan10, monkeypatch):
    a, b = gmw_polys(plan10)
    blob = shapefile_zip(
        tmp_path / "gmw.zip", gpd.GeoDataFrame({"PXLVAL": [1, 1]}, geometry=[a, b], crs="EPSG:4326"), "gmw_v3_2018_vec"
    )
    api = v.ZENODO_RECORD_API.format(record_id="6894273")
    content = "https://zenodo.org/api/records/6894273/files/gmw_v3_2018_vec.zip/content"
    record = {
        "id": 6894273,
        "files": [
            {"key": "gmw_v3_2017_vec.zip", "size": 1, "links": {"self": content.replace("2018", "2017")}},
            {"key": "gmw_v3_2018_vec.zip", "size": len(blob), "links": {"self": content}},
        ],
    }
    web.route(api, lambda m, p: record)
    web.route(content, lambda m, p: blob)
    got = v.gmw_extent(plan10, 2018)
    np.testing.assert_array_equal(got, v.reference_from_geometries(plan10, [a, b]))
    assert web.urls() == [api, content]
    cached = v.reference_dir("gmw") / "v3" / "gmw_v3_2018_vec.zip"
    assert cached.read_bytes() == blob and no_temp_files(v.cache_dir())
    # A rerun, and the coverage call, need no network at all: Zenodo may be down or blocked by then.
    offline(monkeypatch)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018), got)
    np.testing.assert_array_equal(v.gmw_coverage(plan10, 2018), plan10.mask)


def test_gmw_v3_file_already_in_the_cache_needs_no_listing(tmp_path, plan10, monkeypatch):
    """A cache copied from a machine with network access (file only, no listing) works offline."""
    a, b = gmw_polys(plan10)
    blob = shapefile_zip(
        tmp_path / "gmw.zip", gpd.GeoDataFrame({"PXLVAL": [1, 1]}, geometry=[a, b], crs="EPSG:4326"), "gmw_v3_2018_vec"
    )
    folder = v.reference_dir("gmw") / "v3"
    folder.mkdir(parents=True)
    (folder / "gmw_v3_2018_vec.zip").write_bytes(blob)
    offline(monkeypatch)
    np.testing.assert_array_equal(v.gmw_extent(plan10, 2018), v.reference_from_geometries(plan10, [a, b]))


def test_gmw_zenodo_entries_layout_and_v4_tiles(web, tmp_path, boundary_reader, monkeypatch):
    plan = plan_aoi(
        boundary_reader, box(BOUNDARY_LON - 0.008, BOUNDARY_LAT - 0.004, BOUNDARY_LON - 0.002, BOUNDARY_LAT + 0.004)
    ).zones[0]
    lon, lat = centres_lonlat(plan)
    res = 0.00025
    # Named for the 1 degree cell N39W121 (top-left), but holding only a small piece of it.
    west, north = -120.02, 38.86
    bands = np.zeros((41, 80, 80), dtype=np.uint8)
    bands[2024 - 1985, :, 40:] = 1  # east of 120.01 W
    tile = write_lonlat_tif(tmp_path / "GMW_N39W121_v4.1.12.tif", bands, west, north, res)
    api = v.ZENODO_RECORD_API.format(record_id="21346457")
    link = "https://zenodo.org/api/records/21346457/files/GMW_N39W121_v4.1.12.tif/content"
    record = {
        "files": {
            "entries": {"GMW_N39W121_v4.1.12.tif": {"key": "GMW_N39W121_v4.1.12.tif", "links": {"content": link}}}
        }
    }
    web.route(api, lambda m, p: record)
    web.route(link, lambda m, p: tile.read_bytes())
    got = v.gmw_extent(plan, 2024, version="v4.1.12")
    vals, inside = lookup(bands[2024 - 1985], west, north, res, lon, lat)
    assert (got != ((vals == 1) & inside & plan.mask)).sum() == 0
    assert got.any()
    assert web.urls() == [api, link]
    # The listing is cached with the tile, so coverage and later years need no network.
    offline(monkeypatch)
    np.testing.assert_array_equal(v.gmw_coverage(plan, 2024, version="v4.1.12"), inside & plan.mask)
    assert not v.gmw_extent(plan, 2018, version="v4.1.12").any()


def test_gmw_unreachable_and_bad_listing(web, plan10):
    web.route(v.ZENODO_RECORD_API.format(record_id="6894273"), lambda m, p: proxy_refusal())
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.gmw_extent(plan10, 2018)
    msg = str(info.value)
    assert "zenodo.org" in msg and "GMW v3" in msg and "source=" in msg and "records/6894273" in msg
    web.route(v.ZENODO_RECORD_API.format(record_id="6894273"), lambda m, p: {"files": []})
    with pytest.raises(v.ReferenceDataError, match="no downloadable files"):
        v.gmw_extent(plan10, 2018)
    web.route(v.ZENODO_RECORD_API.format(record_id="6894273"), lambda m, p: b"not json")
    with pytest.raises(v.ReferenceDataError, match="not JSON"):
        v.gmw_extent(plan10, 2018)


# --------------------------------------------------------------------------------------------
# Natural Earth
# --------------------------------------------------------------------------------------------


def coastline_frame() -> gpd.GeoDataFrame:
    lines = [
        LineString([(-81.7, 25.8), (-81.5, 25.85), (-81.2, 25.9)]),  # crosses the bbox
        LineString([(-81.5, 25.76), (-81.4, 25.77)]),  # inside
        LineString([(-80.0, 25.0), (-79.9, 25.1)]),  # outside
    ]
    return gpd.GeoDataFrame({"featurecla": ["Coastline"] * 3, "scalerank": [0, 0, 0]}, geometry=lines, crs="EPSG:4326")


EVERGLADES_BBOX = (-81.62, 25.75, -81.32, 25.97)


def test_natural_earth_downloads_clips_and_caches(web, tmp_path, monkeypatch):
    blob = shapefile_zip(tmp_path / "ne.zip", coastline_frame(), "ne_10m_coastline")
    url = v.NATURAL_EARTH_URL.format(scale="10m")
    web.route(url, lambda m, p: blob)
    lines = v.natural_earth_coastline(EVERGLADES_BBOX)
    assert web.urls() == ["https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip"]
    assert len(lines) == 2 and all(isinstance(g, LineString) for g in lines)
    area = box(*EVERGLADES_BBOX)
    assert all(area.buffer(1e-9).contains(g) for g in lines)
    total = sum(g.length for g in lines)
    expected = coastline_frame().geometry.iloc[0].intersection(area).length + coastline_frame().geometry.iloc[1].length
    assert total == pytest.approx(expected)
    assert (v.reference_dir("natural_earth") / "ne_10m_coastline.zip").read_bytes() == blob
    offline(monkeypatch)
    assert len(v.natural_earth_coastline(EVERGLADES_BBOX)) == 2
    local = v.natural_earth_coastline(EVERGLADES_BBOX, source=tmp_path / "ne.zip")
    assert [g.wkt for g in local] == [g.wkt for g in lines]


def test_natural_earth_splits_multipart_pieces(tmp_path):
    zig = LineString([(-81.7, 25.8), (-81.5, 25.8), (-81.5, 26.1), (-81.4, 26.1), (-81.4, 25.8), (-81.2, 25.8)])
    frame = gpd.GeoDataFrame(geometry=[zig], crs="EPSG:4326")
    path = tmp_path / "zig.gpkg"
    frame.to_file(path)
    lines = v.natural_earth_coastline(EVERGLADES_BBOX, source=path)
    assert len(lines) == 2 and all(g.geom_type == "LineString" for g in lines)


def test_natural_earth_errors(web, tmp_path):
    with pytest.raises(ValueError, match="scale"):
        v.natural_earth_coastline(EVERGLADES_BBOX, scale="5m")
    with pytest.raises(FileNotFoundError):
        v.natural_earth_coastline(EVERGLADES_BBOX, source=tmp_path / "none.zip")
    web.route(v.NATURAL_EARTH_URL.format(scale="50m"), lambda m, p: proxy_refusal())
    with pytest.raises(v.ReferenceUnavailableError) as info:
        v.natural_earth_coastline(EVERGLADES_BBOX, scale="50m")
    assert "naturalearth.s3.amazonaws.com" in str(info.value) and "source=" in str(info.value)
    assert not (v.reference_dir("natural_earth") / "ne_50m_coastline.zip").exists()


# --------------------------------------------------------------------------------------------
# Demo AOIs
# --------------------------------------------------------------------------------------------


def test_demo_aois():
    assert set(DEMO_NAMES) == {"caldor", "rondonia", "everglades", "itapebi", "jaru"}
    rondonia = demo_aoi("rondonia")
    assert rondonia.bounds == pytest.approx((-63.20087, -8.75983, -63.01913, -8.58017))
    assert demo_aoi(" Everglades ").bounds == pytest.approx(EVERGLADES_BBOX)
    caldor = demo_aoi("caldor")
    outline = demo_feature("caldor", "hansen_2021_loss_outline")
    assert isinstance(outline, MultiPolygon)
    assert caldor.contains(outline.buffer(-0.001))
    assert demo_properties("rondonia")["year_a"] == 2019 and demo_properties("rondonia")["year_b"] == 2022
    assert demo_geojson("everglades")["type"] == "FeatureCollection"
    with pytest.raises(ValueError, match="Unknown demo"):
        demo_aoi("atlantis")
    with pytest.raises(ValueError, match="no feature with role"):
        demo_feature("rondonia", "seeds")


def test_demo_aoi_without_roles_falls_back_to_first_feature(monkeypatch):
    import tessera_apps.demo as demo

    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "x"}, "geometry": mapping(box(0, 0, 1, 1))}],
    }
    monkeypatch.setattr(demo, "demo_geojson", lambda name: fc)
    assert demo.demo_aoi("caldor").equals(box(0, 0, 1, 1))
    assert demo.demo_properties("caldor") == {"name": "x"}
