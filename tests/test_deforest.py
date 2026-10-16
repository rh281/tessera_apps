"""Offline tests of the deforest app's parts, from the label rules to the validation window.

Pure parts run on synthetic arrays: the Hansen label classes, the label ring, cluster sampling, the
labels CSV, the classifier and its spatially held-out estimate, the radar regime guard, the
cross-check cut, the conversion rule, cleaning across a zone seam and the patch-size summary. The
parts that need a run (the validation window, the regime guard on labels, early failures, the
transfer to year_b, the cross-check against the ring, the options and the estimates) use the
caldor fixture store of ``tests/conftest.py`` with the Hansen tile pieces under it served from
local tiles (``url_template``). Designed in-memory stores, whose embeddings follow a forest map
given per pixel, cover the regime probe's fallback, a conversion patch across a zone seam and
labels read across one. ``tests/test_apps_smoke.py`` runs the whole app.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import zarr
from pyproj import Transformer
from scipy.special import expit
from shapely.geometry import Point, box, mapping
from zarr.storage import MemoryStore

from tessera_apps import export, io, metrics, validate
from tessera_apps.apps import deforest as df
from tessera_apps.apps._common import AppInputError, BandMosaic, StrictValidationError, spread_indices
from tessera_apps.io import (
    BYTES_PER_PIXEL,
    FIRST_YEAR,
    TesseraReader,
    Window,
    ZoneGrid,
    plan_aoi,
    read_plan,
    zone_epsg,
)

from .conftest import BOUNDARY, PAD_COLS, PAD_ROWS, KeyLog, add_zone, fixture_window_lonlat, write_hansen_tiles

F, N, X = df.FOREST, df.NONFOREST, df.NO_LABEL


# --------------------------------------------------------------------------------------------
# Label classes
# --------------------------------------------------------------------------------------------


def classes(tc, ly, dm=1, year_a=2019, **kw) -> list[int]:
    tc = np.atleast_1d(np.asarray(tc, dtype=np.uint8))
    ly = np.broadcast_to(np.asarray(ly, dtype=np.uint8), tc.shape)
    dm = np.broadcast_to(np.asarray(dm, dtype=np.uint8), tc.shape)
    return df.label_classes(tc, ly, dm, year_a, **kw).tolist()


def test_label_classes_cover_cuts():
    # Forest at 50 percent or more, non-forest below 10, nothing in between.
    assert classes([0, 9, 10, 30, 49, 50, 100], 0) == [N, N, X, X, X, F, F]


def test_label_classes_loss_years():
    # year_a 2019 is lossyear 19. Loss up to 17 (year_a - 2) reads as non-forest by 2019; loss in 18
    # or 19 is not a label whatever the cover; loss after 2019 leaves a 2019 forest label.
    assert classes([80] * 7, [0, 1, 17, 18, 19, 20, 25]) == [F, N, N, X, X, F, F]
    assert classes([5] * 5, [0, 17, 18, 19, 20]) == [N, N, X, X, N]
    assert classes([30] * 4, [0, 17, 18, 21]) == [X, N, X, X]


def test_label_classes_follow_year_a():
    assert classes([80] * 4, [19, 20, 21, 22], year_a=2022) == [N, N, X, X]
    assert classes([80] * 3, [16, 17, 18], year_a=2018) == [N, X, X]


def test_label_classes_only_on_land():
    tc = np.array([80, 80, 80, 5, 5, 5], dtype=np.uint8)
    dm = np.array([1, 0, 2, 1, 0, 2], dtype=np.uint8)
    assert df.label_classes(tc, np.zeros(6, dtype=np.uint8), dm, 2019).tolist() == [F, X, X, N, X, X]
    early = np.full(6, 5, dtype=np.uint8)
    assert df.label_classes(tc, early, dm, 2019).tolist() == [N, X, X, N, X, X]


def test_label_classes_custom_cuts():
    assert classes([25, 29, 30, 69, 70], 0, forest_cover=70, nonforest_cover=30) == [N, N, X, X, F]


def test_label_classes_refuse_bad_input():
    with pytest.raises(ValueError, match="Shapes differ"):
        df.label_classes(np.zeros(3), np.zeros(4), np.zeros(3), 2019)
    with pytest.raises(ValueError, match="must not exceed"):
        df.label_classes(np.zeros(3), np.zeros(3), np.zeros(3), 2019, forest_cover=10, nonforest_cover=20)


# --------------------------------------------------------------------------------------------
# The label ring
# --------------------------------------------------------------------------------------------

RONDONIA_BOX = box(-63.12, -8.68, -63.10, -8.66)  # about 2.2 x 2.2 km, zone 20S


def utm(geometry, epsg: int):
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    return shapely.transform(geometry, t.transform, interleaved=False)


def test_label_ring_lies_outside_the_aoi_up_to_its_outer_edge():
    ring = df.label_ring(RONDONIA_BOX, outer_km=1.0)
    assert (ring.zone, ring.epsg, ring.inner_km, ring.outer_km) == ("20S", 32720, 0.0, 1.0)
    assert ring.geometry.intersection(RONDONIA_BOX).area < 1e-12
    aoi_m = utm(RONDONIA_BOX, 32720)
    ring_m = utm(ring.geometry, 32720)
    assert ring_m.area == pytest.approx(aoi_m.buffer(1000.0, quad_segs=16).area - aoi_m.area, rel=0.005)
    xmin, ymin, xmax, ymax = aoi_m.bounds
    for x in np.linspace(xmin - 1500, xmax + 1500, 41):
        for y in np.linspace(ymin - 1500, ymax + 1500, 41):
            p = Point(x, y)
            d = aoi_m.distance(p)
            if d == 0.0:
                assert not ring_m.contains(p), "a point of the AOI is in the ring"
            elif 5.0 < d < 990.0:
                assert ring_m.contains(p), d
            elif d > 1010.0:
                assert not ring_m.contains(p), d


def test_label_ring_inner_edge():
    ring = df.label_ring(RONDONIA_BOX, outer_km=1.0, inner_km=0.4)
    ring_m = utm(ring.geometry, 32720)
    aoi_m = utm(RONDONIA_BOX, 32720)
    x0, _, _, y1 = aoi_m.bounds
    centre_y = (aoi_m.bounds[1] + y1) / 2
    assert not ring_m.contains(Point(x0 - 200.0, centre_y))
    assert ring_m.contains(Point(x0 - 600.0, centre_y))
    assert not ring_m.contains(Point(x0 - 1100.0, centre_y))
    with pytest.raises(ValueError, match="inner_km < outer_km"):
        df.label_ring(RONDONIA_BOX, outer_km=0.4, inner_km=0.4)


def test_label_ring_is_buffered_in_the_main_zone():
    assert df.label_ring(box(-120.01, 38.80, -119.95, 38.85), outer_km=0.5).zone == "11N"
    assert df.label_ring(box(-120.05, 38.80, -119.99, 38.85), outer_km=0.5).zone == "10N"


def boundary_aoi():
    """An AOI of about 350 x 330 m across the 10N / 11N seam of the boundary store, mostly in 10N."""
    lon0, lat0 = BOUNDARY
    return box(lon0 - 0.0025, lat0 - 0.0015, lon0 + 0.0015, lat0 + 0.0015)


def zone_centres(z):
    rr, cc = np.nonzero(z.mask)
    x = z.grid.left + (cc + z.cols.start + 0.5) * z.grid.pixel
    y = z.grid.top - (rr + z.rows.start + 0.5) * z.grid.pixel
    lon, lat = z.grid.to_lonlat().transform(x, y)
    return shapely.points(lon, lat), rr, cc


def overlap(a, b) -> int:
    """AOI-mask pixels two zone plans of one zone share."""
    r0, r1 = max(a.rows.start, b.rows.start), min(a.rows.stop, b.rows.stop)
    c0, c1 = max(a.cols.start, b.cols.start), min(a.cols.stop, b.cols.stop)
    if r0 >= r1 or c0 >= c1:
        return 0
    ma = a.mask[r0 - a.rows.start : r1 - a.rows.start, c0 - a.cols.start : c1 - a.cols.start]
    mb = b.mask[r0 - b.rows.start : r1 - b.rows.start, c0 - b.cols.start : c1 - b.cols.start]
    return int((ma & mb).sum())


def test_plan_label_ring_spans_both_zones_and_holds_no_aoi_pixel(boundary_reader):
    aoi = boundary_aoi()
    aoi_plan = plan_aoi(boundary_reader, aoi)
    assert aoi_plan.zone_names == ["10N", "11N"]
    ring = df.label_ring(aoi, outer_km=0.25)
    assert ring.zone == "10N"
    plan = df.plan_label_ring(boundary_reader, ring, aoi_plan)
    assert plan.zone_names == ["10N", "11N"]
    aoi_m = utm(aoi, ring.epsg)
    for z in plan.zones:
        pts, _, _ = zone_centres(z)
        assert not shapely.contains(aoi, pts).any()
        d = shapely.distance(aoi_m, utm(pts, ring.epsg))
        assert d.min() > 0.0 and d.max() <= 250.0 + 8.0
        a = next(p for p in aoi_plan.zones if p.zone == z.zone)
        assert overlap(z, a) == 0
    ring_px = utm(ring.geometry, ring.epsg).area / 100.0
    assert plan.n_pixels == pytest.approx(ring_px, rel=0.03)


def test_plan_label_ring_removes_aoi_pixels_from_any_ring(boundary_reader):
    aoi = boundary_aoi()
    aoi_plan = plan_aoi(boundary_reader, aoi)
    covering = df.LabelRing(geometry=aoi.buffer(0.002), zone="10N", epsg=32610, inner_km=0.0, outer_km=0.2)
    plan = df.plan_label_ring(boundary_reader, covering, aoi_plan)
    assert plan.n_pixels == plan_aoi(boundary_reader, covering.geometry).n_pixels - aoi_plan.n_pixels
    for z in plan.zones:
        assert overlap(z, next(p for p in aoi_plan.zones if p.zone == z.zone)) == 0


def test_plan_label_ring_refuses_a_ring_off_the_store(boundary_reader):
    aoi_plan = plan_aoi(boundary_reader, boundary_aoi())
    far = df.LabelRing(geometry=box(-120.5, 38.0, -120.49, 38.01), zone="10N", epsg=32610, inner_km=0, outer_km=1)
    with pytest.raises(df.DeforestError, match="--labels"):
        df.plan_label_ring(boundary_reader, far, aoi_plan)


# --------------------------------------------------------------------------------------------
# Cluster sampling
# --------------------------------------------------------------------------------------------


def test_choose_label_chunks_caps_and_represents_the_rare_class():
    forest = np.full(100, 500)
    non = np.zeros(100)
    non[37] = 5
    for seed in range(6):
        got = df.choose_label_chunks(forest, non, 4, np.random.default_rng(seed))
        assert len(got) == 4 and len(set(got)) == 4
        assert got[0] == 37, "the rarer class draws first"


def test_choose_label_chunks_is_seeded():
    forest, non = np.arange(1, 51), np.arange(50, 0, -1)
    first = df.choose_label_chunks(forest, non, 10, np.random.default_rng(1))
    assert first == df.choose_label_chunks(forest, non, 10, np.random.default_rng(1))
    others = {tuple(df.choose_label_chunks(forest, non, 10, np.random.default_rng(s))) for s in range(2, 6)}
    assert others - {tuple(first)}


def test_choose_label_chunks_stops_when_the_classes_run_out():
    assert sorted(df.choose_label_chunks([3, 0, 0, 2], [0, 0, 4, 0], 10, np.random.default_rng(0))) == [0, 2, 3]
    assert df.choose_label_chunks([0, 0], [0, 0], 5, np.random.default_rng(0)) == []
    with pytest.raises(ValueError):
        df.choose_label_chunks([1, 2], [1], 1, np.random.default_rng(0))


def test_choose_label_chunks_alternates_between_the_classes():
    # The rare class lives in its own chunks: always drawing by its weights would pick only them and
    # sample no forest label at all. Alternating gives two chunks of each class.
    forest = np.r_[np.full(10, 1000), np.zeros(10)]
    non = np.r_[np.zeros(10), np.full(10, 10)]
    for seed in range(5):
        got = df.choose_label_chunks(forest, non, 4, np.random.default_rng(seed))
        assert len(got) == 4
        assert sum(1 for i in got if i < 10) == 2 and sum(1 for i in got if i >= 10) == 2, (seed, got)
        assert got[0] >= 10 and got[1] < 10, "the rarer class draws first, then the classes alternate"


GRID = ZoneGrid(zone="20S", epsg=32720, left=500_000.0, top=9_050_000.0, height=100_000, width=100_000)
ROW0, COL0 = 32 * 31, 32 * 62  # chunk aligned


def class_grid(seed: int = 0) -> np.ndarray:
    """128 x 128 classes (16 chunks of 32 x 32): forest scattered everywhere, non-forest in one chunk."""
    rng = np.random.default_rng(seed)
    c = np.where(rng.random((128, 128)) < 0.6, F, X).astype(np.int8)
    c[96:128, 0:32] = np.where(rng.random((32, 32)) < 0.5, N, X)
    return c


def sample(seed: int = 3, max_chunks: int = 5, per_class: int = 40, grid=None):
    cl = class_grid() if grid is None else grid
    m = df.ClassMap(zone="20S", grid=GRID, row0=ROW0, col0=COL0, classes=cl, chunk_shape=(32, 32))
    return df.sample_hansen_labels([m], max_chunks=max_chunks, per_class=per_class, seed=seed)


def test_sample_hansen_labels_caps_chunks_and_pixels_per_class():
    cl = class_grid()
    labels, info = sample()
    chunks = set(zip(labels["chunk_row"], labels["chunk_col"], strict=True))
    assert len(chunks) <= 5 and info["chunks_used"] == 5
    assert info["chunks_with_labels"] == 16 and info["chunks_with_nonforest"] == 1 and info["chunks_with_forest"] == 15
    assert {f"20S/{r}/{c}" for r, c in chunks} <= set(info["chunks"])
    n_f = int((labels["label"] == F).sum())
    n_n = int((labels["label"] == N).sum())
    assert n_f == min(40, info["pixels_in_chunks_forest"]) == 40 == info["sampled_forest"]
    assert n_n == min(40, info["pixels_in_chunks_nonforest"]) == info["sampled_nonforest"] > 0
    # Every label is a pixel of its class at its pixel centre, inside a chosen chunk, unweighted.
    assert (cl[labels["row"] - ROW0, labels["col"] - COL0] == labels["label"]).all()
    assert (labels["chunk_row"] == labels["row"] // 32).all() and (labels["chunk_col"] == labels["col"] // 32).all()
    x, y = GRID.from_lonlat().transform(labels["lon"].to_numpy(), labels["lat"].to_numpy())
    back = [GRID.rowcol(float(a), float(b)) for a, b in zip(x, y, strict=True)]
    assert back == list(zip(labels["row"], labels["col"], strict=True))
    assert (labels["weight"] == 1.0).all() and (labels["zone"] == "20S").all()
    assert not labels.duplicated(["row", "col"]).any()


def test_sample_hansen_labels_takes_every_pixel_when_under_the_cap():
    labels, info = sample(max_chunks=16, per_class=10_000)
    assert info["chunks_used"] == 16
    assert len(labels) == info["ring_pixels_forest"] + info["ring_pixels_nonforest"]


def test_sample_hansen_labels_is_seeded():
    a, _ = sample(seed=7)
    b, _ = sample(seed=7)
    pd.testing.assert_frame_equal(a, b)
    c, _ = sample(seed=8)
    assert not a[["row", "col"]].equals(c[["row", "col"]])


def test_sample_hansen_labels_across_zones_and_empty_maps():
    cl = class_grid()
    maps = [
        df.ClassMap(zone="21S", grid=GRID, row0=ROW0, col0=COL0, classes=cl, chunk_shape=(32, 32)),
        df.ClassMap(
            zone="20S", grid=GRID, row0=ROW0, col0=COL0, classes=np.full((8, 8), X, np.int8), chunk_shape=(32, 32)
        ),
        df.ClassMap(zone="20S", grid=GRID, row0=ROW0, col0=COL0, classes=cl, chunk_shape=(32, 32)),
    ]
    labels, info = df.sample_hansen_labels(maps, max_chunks=40, per_class=20_000, seed=0)
    assert info["chunks_with_labels"] == 32 and info["chunks_used"] == 32
    assert set(labels["zone"]) == {"20S", "21S"}
    assert info["chunks"] == sorted(info["chunks"])
    none, info = df.sample_hansen_labels([maps[1]], max_chunks=5, per_class=5, seed=0)
    assert len(none) == 0 and info["chunks_used"] == 0 and list(none.columns) == list(df.LABEL_COLUMNS)


def test_chunk_tiles_cover_an_unaligned_window_once():
    m = df.ClassMap(zone="20S", grid=GRID, row0=37, col0=70, classes=np.zeros((50, 90), np.int8), chunk_shape=(32, 16))
    tiles = df.chunk_tiles(m)
    seen = np.zeros((50, 90), dtype=int)
    for kr, kc, rows, cols in tiles:
        seen[rows, cols] += 1
        # Every pixel of a tile lies in that store chunk, and the tile is the whole chunk inside the window.
        assert {(r + 37) // 32 for r in range(rows.start, rows.stop)} == {kr}
        assert {(c + 70) // 16 for c in range(cols.start, cols.stop)} == {kc}
    assert (seen == 1).all()
    assert [t[:2] for t in tiles] == [(kr, kc) for kr in range(1, 3) for kc in range(4, 10)]
    assert df.chunk_tiles(dataclasses.replace(m, classes=np.zeros((0, 5), np.int8))) == []


def test_sample_hansen_labels_on_a_non_square_unaligned_map():
    # 64 x 160 (not square, not chunk aligned, 32 x 48 chunks): forest in the west, non-forest in the east. A row or
    # column decode that mixes up the height and the width puts labels on the wrong pixels, and a
    # chunk split that does assigns pixels to chunks the cap never chose.
    h, w, row0, col0 = 64, 160, 37, 11
    rng = np.random.default_rng(4)
    cl = np.where(rng.random((h, w)) < 0.7, np.where(np.arange(w)[None, :] < 80, F, N), X).astype(np.int8)
    loss = rng.integers(0, 26, size=(h, w), dtype=np.uint8)
    m = df.ClassMap(zone="20S", grid=GRID, row0=row0, col0=col0, classes=cl, chunk_shape=(32, 48), lossyear=loss)
    labels, info = df.sample_hansen_labels([m], max_chunks=4, per_class=150, seed=1)
    assert info["chunks_used"] == 4 < info["chunks_with_labels"]
    assert (cl[labels["row"] - row0, labels["col"] - col0] == labels["label"]).all()
    assert (labels["hansen_lossyear"] == loss[labels["row"] - row0, labels["col"] - col0]).all()
    used = {f"20S/{r}/{c}" for r, c in zip(labels["chunk_row"], labels["chunk_col"], strict=True)}
    assert used <= set(info["chunks"]) and len(info["chunks"]) == 4
    assert (labels["chunk_row"] == labels["row"] // 32).all() and (labels["chunk_col"] == labels["col"] // 48).all()
    assert labels["row"].between(row0, row0 + h - 1).all() and labels["col"].between(col0, col0 + w - 1).all()
    assert {F, N} == set(labels["label"]) and int((labels["label"] == F).sum()) == 150
    # Without a lossyear the column is -1.
    plain, _ = df.sample_hansen_labels([dataclasses.replace(m, lossyear=None)], max_chunks=4, per_class=150, seed=1)
    assert (plain["hansen_lossyear"] == -1).all() and plain[["row", "col"]].equals(labels[["row", "col"]])


def test_label_stage_memory_stays_within_its_estimate():
    # The ring's label stage: three Hansen layers, the class map, the lossyear kept for the transfer
    # check, and the sampling. Its numpy peak per window pixel must stay within the memory model's
    # MEMORY_RING_PER_PIXEL (before the margin), for every ring pixel labelled: the sampler before
    # held about 90 bytes per labelled pixel, so a 50 km ring peaked at 3 times the estimate.
    h, w = 1100, 1700
    cover = np.random.default_rng(0).choice(np.array([5, 90], np.uint8), size=(h, w))
    tracemalloc.start()
    try:
        tc, dm, ly = cover.copy(), np.ones((h, w), np.uint8), np.zeros((h, w), np.uint8)
        classes = df.label_classes(tc, ly, dm, 2019)
        del tc, dm
        m = df.ClassMap(zone="20S", grid=GRID, row0=37, col0=11, classes=classes, chunk_shape=(256, 256), lossyear=ly)
        labels, info = df.sample_hansen_labels([m], max_chunks=24, per_class=500, seed=0)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert info["ring_pixels_forest"] + info["ring_pixels_nonforest"] == h * w and len(labels) == 1000
    assert peak / (h * w) <= df.MEMORY_RING_PER_PIXEL


# --------------------------------------------------------------------------------------------
# Labels CSV
# --------------------------------------------------------------------------------------------


def csv_text(n_forest: int = 25, n_non: int = 25, header: str = "lon,lat,label", extra: str = "") -> str:
    lines = [header]
    for i in range(n_forest + n_non):
        lines.append(f"{-63.1 + i * 1e-4:.6f},{-8.67 + i * 1e-4:.6f},{1 if i < n_forest else 0}{extra}")
    return "\n".join(lines) + "\n"


def test_read_labels_csv_reads_a_good_file(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(csv_text(header="LON, Lat ,Label,Weight", extra=",2.5"), encoding="utf-8")
    frame = df.read_labels_csv(path)
    assert list(frame.columns) == ["lon", "lat", "label", "weight"] and len(frame) == 50
    assert frame["label"].tolist() == [1] * 25 + [0] * 25 and (frame["weight"] == 2.5).all()
    assert frame["lon"].iloc[3] == pytest.approx(-63.0997)
    path.write_text(csv_text(), encoding="utf-8")
    assert (df.read_labels_csv(path)["weight"] == 1.0).all()
    assert len(df.read_labels_csv(path, min_per_class=25)) == 50


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (None, "does not exist"),
        ("", "Could not read"),
        ("lon,lat,label\n", "has no rows"),
        (csv_text(header="lon,lat,class"), "no label column"),
        (csv_text(header="x,y,label"), "no lon, lat column"),
        (csv_text().replace("-8.67", "85.0", 1), "outside lon -180 to 180"),
        (csv_text().replace("-63.1", "-200.1", 1), "outside lon -180 to 180"),
        (csv_text().replace("-63.100000", "", 1), "not a number on line"),
        (csv_text().replace(",1\n", ",2\n", 1), "label must be 1"),
        (csv_text().replace(",0\n", ",forest\n", 1), "label must be 1"),
        (csv_text(header="lon,lat,label,weight", extra=",-1"), "weight must be a positive"),
        (csv_text(n_forest=19), "19 forest and 25 non-forest"),
        (csv_text(n_non=0), "25 forest and 0 non-forest"),
    ],
)
def test_read_labels_csv_errors(tmp_path, content, match):
    path = tmp_path / "labels.csv"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(AppInputError, match=match) as info:
        df.read_labels_csv(path)
    assert "labels.csv" in str(info.value)


def test_read_labels_csv_names_the_bad_lines(tmp_path):
    path = tmp_path / "labels.csv"
    rows = csv_text().splitlines()
    rows[3] = rows[3].rsplit(",", 1)[0] + ",7"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(AppInputError, match=r"line\(s\) 4 hold 7"):
        df.read_labels_csv(path)


# --------------------------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------------------------


def blobs(n_groups: int = 10, per_group: int = 40, seed: int = 0):
    """Two Gaussian classes 4 sigma apart along one direction, both in every group."""
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    y = np.tile(np.arange(per_group) % 2, n_groups)
    g = np.repeat(np.arange(n_groups), per_group)
    direction = rng.normal(size=128)
    direction /= np.linalg.norm(direction)
    x = rng.normal(size=(n, 128)) + 4.0 * y[:, None] * direction
    return x, y, g


def test_fit_forest_model_on_separable_labels():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x, y, g = blobs()
    fit = df.fit_forest_model(x, y, g, seed=0)
    rec = fit.record
    assert rec["converged"] and rec["labels"] == 400 and rec["labels_forest"] == rec["labels_nonforest"] == 200
    assert rec["training"]["accuracy"] > 0.97 and rec["heldout"]["balanced_accuracy"] > 0.95
    assert rec["heldout"]["folds"] == 5 and rec["heldout"]["groups"] == 10 and rec["heldout"]["folds_skipped"] == 0
    assert rec["heldout"]["n"] == 400 and np.isfinite(fit.heldout_prob).all()
    assert set(fit.folds.tolist()) == {0, 1, 2, 3, 4}
    assert rec["weighted"] is False
    # The folded linear model is the scaler and the regression, fitted once on every label.
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(class_weight="balanced", max_iter=df.MAX_ITER, random_state=0).fit(scaler.transform(x), y)
    expected = lr.predict_proba(scaler.transform(x))[:, 1]
    np.testing.assert_allclose(fit.model.probability_vectors(x), expected, atol=1e-8)


def test_held_out_estimate_holds_out_whole_chunks():
    # No class signal beyond chunk identity: each chunk has its own random centre and one class. A
    # held-out estimate that lets a chunk into both the training and test parts memorises the
    # centres and scores near 1; held out by chunk it scores near chance.
    rng = np.random.default_rng(1)
    n_groups, per = 20, 20
    centres = rng.normal(scale=3.0, size=(n_groups, 128))
    x = np.repeat(centres, per, axis=0) + rng.normal(scale=0.3, size=(n_groups * per, 128))
    y = np.repeat(np.arange(n_groups) % 2, per)
    g = np.repeat(np.arange(n_groups), per)
    order = rng.permutation(x.shape[0])
    x, y, g = x[order], y[order], g[order]
    fit = df.fit_forest_model(x, y, g)
    assert fit.record["training"]["accuracy"] > 0.95
    assert fit.record["heldout"]["balanced_accuracy"] < 0.75
    for grp in range(n_groups):
        assert np.unique(fit.folds[g == grp]).size == 1


def test_held_out_folds_without_both_classes_are_not_scored():
    x, _, g = blobs(n_groups=6)
    y = (g != 0).astype(int)  # every non-forest label is in chunk 0
    fit = df.fit_forest_model(x, y, g)
    held = fit.record["heldout"]
    assert held["folds"] == 5 and held["folds_skipped"] >= 1
    assert np.isnan(fit.heldout_prob[g == 0]).all() and held["labels_unscored"] >= int((g == 0).sum())
    assert held["recall_nonforest"] is None and held["balanced_accuracy"] is None


def test_no_held_out_estimate_from_one_chunk():
    x, y, _ = blobs(n_groups=1)
    fit = df.fit_forest_model(x, y, np.zeros(len(y)))
    assert fit.record["heldout"]["folds"] == 0 and fit.record["heldout"]["n"] == 0
    assert (fit.folds == -1).all() and np.isnan(fit.heldout_prob).all()
    with pytest.raises(ValueError, match="both classes"):
        df.fit_forest_model(x, np.ones(len(y)), np.arange(len(y)))


def test_fit_forest_model_uses_the_weights():
    x, y, g = blobs()
    flipped = y.copy()
    flipped[:40] = 1 - flipped[:40]  # 40 wrong labels in chunk 0
    w = np.ones(len(y))
    plain = df.fit_forest_model(x, flipped, g, weights=w)
    assert plain.record["weighted"] is False
    w[:40] = 1e-6
    weighted = df.fit_forest_model(x, flipped, g, weights=w)
    assert weighted.record["weighted"] is True
    p_plain = plain.model.probability_vectors(x[:40])
    p_weighted = weighted.model.probability_vectors(x[:40])
    # With the wrong labels weighted down, chunk 0 is predicted as its true class more confidently.
    assert np.mean(np.abs(p_weighted - y[:40])) < np.mean(np.abs(p_plain - y[:40]))


def test_fit_forest_model_balances_imbalanced_classes():
    # 360 forest and 40 non-forest labels, as hand labelling often gives: the fit must be sklearn's
    # class_weight="balanced" one, which an unweighted fit (biased toward forest) is not.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(5)
    y = np.r_[np.ones(360, dtype=int), np.zeros(40, dtype=int)]
    direction = rng.normal(size=128)
    x = rng.normal(size=(400, 128)) + 1.2 * y[:, None] * direction / np.linalg.norm(direction)
    g = np.arange(400) % 8
    fit = df.fit_forest_model(x, y, g, seed=0)
    scaled = StandardScaler().fit(x).transform(x)
    kw = {"max_iter": df.MAX_ITER, "random_state": 0}
    balanced = LogisticRegression(class_weight="balanced", **kw).fit(scaled, y).predict_proba(scaled)[:, 1]
    unweighted = LogisticRegression(**kw).fit(scaled, y).predict_proba(scaled)[:, 1]
    got = fit.model.probability_vectors(x)
    np.testing.assert_allclose(got, balanced, atol=1e-6)
    assert np.max(np.abs(got - unweighted)) > 0.05
    assert fit.record["labels_forest"] == 360 and fit.record["labels_nonforest"] == 40


def window(emb: np.ndarray, scales: np.ndarray, radar: np.ndarray | None = None, year: int = 2019) -> Window:
    h, w = scales.shape
    asc = np.full((h, w), 30, dtype=np.uint16) if radar is None else np.where(radar, 30, 0).astype(np.uint16)
    return Window(
        zone="20S",
        year=year,
        rows=slice(0, h),
        cols=slice(0, w),
        grid=GRID,
        scales=scales,
        embeddings=emb,
        s1_asc_obs_count=asc,
        s1_desc_obs_count=np.zeros((h, w), dtype=np.uint16),
    )


def test_forest_model_probability_works_on_the_int8_embeddings(monkeypatch):
    rng = np.random.default_rng(2)
    emb = rng.integers(-128, 128, size=(9, 7, 128), dtype=np.int8)
    scales = rng.uniform(0.001, 0.02, size=(9, 7)).astype(np.float32)
    scales[0, 0] = np.nan
    emb[1, 1] = 0
    scales[2, 2] = 0.0
    model = df.ForestModel(weights=rng.normal(size=128) * 0.05, intercept=0.3)
    monkeypatch.setattr(df, "PREDICT_STRIP_BYTES", 7 * 128 * 4 * 2)  # two rows per strip
    p = model.probability(window(emb, scales))
    assert p.dtype == np.float32 and p.shape == (9, 7)
    deq = emb.astype(np.float64) * scales[..., None].astype(np.float64)
    expected = expit(deq @ model.weights + model.intercept)
    bad = np.zeros((9, 7), dtype=bool)
    bad[0, 0] = bad[1, 1] = bad[2, 2] = True
    assert np.isnan(p[bad]).all() and np.isfinite(p[~bad]).all()
    np.testing.assert_allclose(p[~bad], expected[~bad], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(model.probability_vectors(deq[~bad]), expected[~bad])


# --------------------------------------------------------------------------------------------
# Transfer to year_b at the labels, and the ring label cut
# --------------------------------------------------------------------------------------------


def test_hansen_stable_labels():
    label = np.array([F, F, F, F, F, N, N, N, F])
    lossyear = np.array([0, 20, 22, 23, 25, 0, 15, 21, -1])
    # year_b 2022: forest lost in 2020 or 2022 changed; lost in 2023 or 2025 is still forest in 2022.
    assert df.hansen_stable_labels(label, lossyear, 2022).tolist() == [1, 0, 0, 1, 1, 1, 1, 1, 0]
    assert df.hansen_stable_labels(label, lossyear, 2024).tolist() == [1, 0, 0, 0, 1, 1, 1, 1, 0]
    with pytest.raises(ValueError):
        df.hansen_stable_labels(label, lossyear[:3], 2022)


def test_vector_distance_is_the_cosine_distance_of_rows():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=(5, 128)), rng.normal(size=(5, 128))
    b[1] = 3.0 * a[1]
    b[2] = -a[2]
    a[3] = 0.0
    a[4, 0] = np.nan
    d = df.vector_distance(a, b)
    expected = 1.0 - np.sum(a[0] * b[0]) / (np.linalg.norm(a[0]) * np.linalg.norm(b[0]))
    assert d[0] == pytest.approx(expected) and d[1] == pytest.approx(0.0, abs=1e-12) and d[2] == pytest.approx(2.0)
    assert np.isnan(d[3]) and np.isnan(d[4])


def test_label_transfer_scores_year_b_against_the_held_out_year_a():
    y = np.array([1] * 6 + [0] * 6)
    scored = np.ones(12, dtype=bool)
    scored[0] = False  # not Hansen-stable, or not valid in year_b: never scored
    pa = np.where(y == 1, 0.9, 0.1)
    held = pa.copy()
    held[1] = np.nan  # its fold was not scored: left out of the held-out comparison only
    pb = pa.copy()
    pb[[2, 3]] = 0.2  # two stable forest labels read as non-forest in year_b
    pb[6] = 0.8  # and one non-forest label as forest
    pb[11] = np.nan  # no year_b probability: not scored
    got = df.label_transfer(y, scored, pa, held, pb, forest_prob=0.5)
    assert (got["labels"], got["labels_forest"], got["labels_nonforest"]) == (10, 5, 5)
    assert got["year_a_in_sample"]["balanced_accuracy"] == 1.0
    assert got["year_b"]["recall_forest"] == pytest.approx(3 / 5) and got["year_b"]["recall_nonforest"] == 0.8
    assert got["year_a_heldout"]["n"] == 9 and got["year_a_heldout"]["balanced_accuracy"] == 1.0
    # The drop compares the same 9 labels: year_b recalls 2/4 forest and 4/5 non-forest there.
    assert got["drop_labels"] == 9 and got["drop"] == pytest.approx(1.0 - (2 / 4 + 4 / 5) / 2)
    none = df.label_transfer(y, scored, pa, np.full(12, np.nan), pb)
    assert none["drop"] is None and none["year_a_heldout"]["n"] == 0
    with pytest.raises(ValueError):
        df.label_transfer(y, scored[:3], pa, held, pb)


def test_ring_label_cut_uses_stable_forest_labels_only():
    y = np.array([1] * 101 + [0, 0, 1, 1])
    d = np.r_[np.linspace(0.0, 0.1, 101), [0.9, 0.9, 0.9, np.nan]]
    scored = np.r_[np.ones(101, dtype=bool), [True, True, False, True]]
    cut, n = df.ring_label_cut(d, y, scored)
    assert n == 101 and cut == pytest.approx(float(np.percentile(np.linspace(0.0, 0.1, 101), 95)))
    assert df.ring_label_cut(d, y, scored, percentile=50.0)[0] == pytest.approx(0.05)
    assert df.ring_label_cut(d, y, np.zeros(y.size, dtype=bool)) == (None, 0)


def test_cross_check_removed_splits_the_flips_by_hansen_loss():
    ly = np.array([[0, 19, 20, 21], [22, 23, 10, 0]], dtype=np.uint8)
    scored = np.array([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=bool)
    below = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool)
    got = df.cross_check_removed(ly, scored, below, 2019, 2022)
    assert got == {
        "flips_below_cut": 7,
        "loss_in_window": 3,
        "loss_year_a": 1,
        "loss_other_years": 2,
        "no_loss": 1,
        "loss_lag_aligned": 3,
    }
    halves = df.cross_check_removed([ly[:1], ly[1:]], [scored[:1], scored[1:]], [below[:1], below[1:]], 2019, 2022)
    assert halves == got


# --------------------------------------------------------------------------------------------
# Radar regime
# --------------------------------------------------------------------------------------------


def test_training_regime_is_the_majority_of_same_regime_pixels():
    assert df.training_regime(10, 5) is True
    assert df.training_regime(5, 10) is False
    assert df.training_regime(5, 5) is True
    assert df.training_regime(0, 0) is None
    assert [df.regime_name(r) for r in (True, False, None)] == ["radar", "optical", "none"]


def test_apply_regime_keeps_valid_labels_of_the_training_regime():
    labels = pd.DataFrame({"label": [1, 1, 0, 0]})
    out = df.apply_regime(labels, np.array([True, True, False, True]), np.array([True, False, True, True]), True)
    assert out["used"].tolist() == [True, False, False, True]
    assert out["drop_reason"].tolist() == ["", "regime", "no_data", ""]
    assert out["regime"].tolist() == ["radar", "optical", "no_data", "radar"]
    optical = df.apply_regime(labels, np.array([True, True, False, True]), np.array([True, False, True, True]), False)
    assert optical["used"].tolist() == [False, True, False, False]
    with pytest.raises(ValueError):
        df.apply_regime(labels, np.ones(3, dtype=bool), np.ones(4, dtype=bool), True)


def test_classify_block_applies_the_radar_regime_guard():
    rng = np.random.default_rng(3)
    emb_a = rng.integers(-60, 60, size=(2, 3, 128), dtype=np.int8)
    emb_b = rng.integers(-60, 60, size=(2, 3, 128), dtype=np.int8)
    scales_a = np.full((2, 3), 0.01, dtype=np.float32)
    scales_b = scales_a.copy()
    scales_b[1, 0] = np.nan  # no data in year_b
    radar_a = np.array([[True, True, False], [True, True, True]])
    radar_b = np.array([[True, False, False], [True, True, True]])
    mask = np.array([[True, True, True], [True, True, False]])
    model = df.ForestModel(weights=np.zeros(128), intercept=2.0)
    wa, wb = window(emb_a, scales_a, radar_a), window(emb_b, scales_b, radar_b, year=2022)
    pa, pb, d, counts = df.classify_block(wa, wb, mask, model, True)
    assert np.isfinite(pa).tolist() == [[True, True, False], [True, True, False]]
    assert np.isfinite(pb).tolist() == [[True, False, False], [False, True, False]]
    assert np.isfinite(d).tolist() == [[True, False, True], [False, True, False]]
    np.testing.assert_allclose(pa[np.isfinite(pa)], expit(2.0), rtol=1e-6)
    expected_d = metrics.cosine_distance_map(wa, wb)
    np.testing.assert_allclose(d[np.isfinite(d)], expected_d[np.isfinite(d)])
    assert counts == {
        "valid_a": 5,
        "valid_b": 4,
        "valid_both": 4,
        "regime_mismatch": 1,
        "other_regime": 1,
        "other_regime_a": 1,
        "other_regime_b": 2,
    }
    # Trained on optical-only labels, only the pixel optical in both years is classified.
    pa, pb, d, counts = df.classify_block(wa, wb, mask, model, False)
    assert (np.isfinite(pa) & np.isfinite(pb)).tolist() == [[False, False, True], [False, False, False]]
    assert counts["other_regime"] == 2 and counts["regime_mismatch"] == 1
    # A pixel that switches regime is never a conversion, whatever the probabilities.
    sure = df.ForestModel(weights=np.zeros(128), intercept=5.0)
    pa, _, d, _ = df.classify_block(wa, wb, mask, sure, True)
    _, pb_low, _, _ = df.classify_block(wa, wb, mask, df.ForestModel(weights=np.zeros(128), intercept=-5.0), True)
    assert not df.conversion_mask(pa, pb_low, d, cut=None)[0, 1]


# --------------------------------------------------------------------------------------------
# Cross-check cut, conversion rule, cleaning, patches
# --------------------------------------------------------------------------------------------


def test_cross_check_cut_is_a_percentile_of_stable_forest_only():
    stable_d = np.linspace(0.0, 0.1, 101)
    dist = np.concatenate([stable_d, [0.8, 0.8, 0.9, 0.9, np.nan, 0.95]])
    pa = np.concatenate([np.full(101, 0.9), [0.9, 0.9, 0.1, 0.1, 0.9, 0.5]])
    pb = np.concatenate([np.full(101, 0.8), [0.1, 0.2, 0.1, 0.2, 0.9, np.nan]])
    cut, n = df.cross_check_cut(dist, pa, pb)
    assert n == 101 and cut == pytest.approx(float(np.percentile(stable_d, 95)))
    half, _ = df.cross_check_cut(dist, pa, pb, percentile=50.0)
    assert half == pytest.approx(0.05)
    # Per-zone arrays are pooled.
    zoned, n2 = df.cross_check_cut([dist[:50], dist[50:]], [pa[:50], pa[50:]], [pb[:50], pb[50:]])
    assert (zoned, n2) == (cut, n)
    # The forest cut moves the stable set: at 0.85 only year_a > 0.85 and year_b > 0.85 count.
    pb2 = pb.copy()
    pb2[:50] = 0.9
    strict, n3 = df.cross_check_cut(dist, pa, pb2, forest_prob=0.85)
    assert n3 == 50 and strict == pytest.approx(float(np.percentile(stable_d[:50], 95)))
    assert df.cross_check_cut(dist, np.full(dist.shape, 0.1), pb) == (None, 0)


def test_conversion_mask_rule():
    pa = np.array([0.9, 0.9, 0.9, 0.2, 0.9, np.nan, 0.9, 0.5, 0.9])
    pb = np.array([0.1, 0.8, 0.1, 0.1, 0.1, 0.1, np.nan, 0.1, 0.5])
    d = np.array([0.5, 0.5, 0.1, 0.5, np.nan, 0.5, 0.5, 0.5, 0.5])
    got = df.conversion_mask(pa, pb, d, cut=0.3)
    assert got.tolist() == [True, False, False, False, False, False, False, True, False]
    # No cut: the distance only has to be finite.
    assert df.conversion_mask(pa, pb, d, cut=None).tolist() == [
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    # The forest probability cut applies to both years.
    strict = df.conversion_mask(pa, pb, d, cut=0.3, forest_prob=0.95)
    assert not strict.any()
    loose = df.conversion_mask(pa, pb, d, cut=0.3, forest_prob=0.15)
    assert loose.tolist() == [True, False, False, True, False, False, False, True, False]
    high = df.conversion_mask(pa, pb, d, cut=0.3, forest_prob=0.85)
    assert high.tolist() == [True, True, False, False, False, False, False, False, True]


def centres_in(z, geometry) -> np.ndarray:
    pts, rr, cc = zone_centres(z)
    out = np.zeros(z.shape, dtype=bool)
    hit = shapely.contains(geometry, pts)
    out[rr[hit], cc[hit]] = True
    return out


def test_clean_conversion_judges_a_patch_across_the_zone_seam_whole(boundary_reader):
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, boundary_aoi())
    patch = box(lon0 - 0.0004, lat0 - 0.0003, lon0 + 0.0004, lat0 + 0.0004)
    speck = box(lon0 - 0.0020, lat0 + 0.0008, lon0 - 0.0018, lat0 + 0.0010)
    raw = {z.zone: centres_in(z, patch) | centres_in(z, speck) for z in plan.zones}
    halves = [int(centres_in(z, patch).sum()) for z in plan.zones]
    assert min(halves) > 15
    min_pixels = max(halves) + 5
    assert sum(halves) > min_pixels + 10
    for z in plan.zones:
        assert not metrics.clean_mask(raw[z.zone], min_pixels=min_pixels).any(), "zone by zone it would vanish"
    cleaned, mosaic = df.clean_conversion(raw, plan, min_pixels=min_pixels)
    for z in plan.zones:
        assert cleaned[z.zone].any(), z.zone
        assert not (cleaned[z.zone] & ~raw[z.zone]).any(), "cleaning never adds a pixel"
        assert not (cleaned[z.zone] & centres_in(z, speck)).any(), "the speck is removed"
    sizes = metrics.component_sizes(mosaic == 1)
    assert sizes.size == 1 and sizes[0] >= min_pixels
    # Without a minimum area the patch is kept whole and the opening still takes the speck.
    kept, _ = df.clean_conversion(raw, plan, min_pixels=0)
    assert sum(int(kept[z.zone].sum()) for z in plan.zones) >= sum(int(cleaned[z.zone].sum()) for z in plan.zones)


def test_patch_size_summary_bins_and_statistics():
    sizes = np.array([10, 50, 99, 100, 499, 500, 1999, 2000, 9999, 10000, 25000])
    s = df.patch_size_summary(sizes)
    assert s["histogram"] == {"<0.5": 1, "0.5-1": 2, "1-5": 2, "5-20": 2, "20-100": 2, ">100": 2}
    assert s["count"] == 11 and s["max_ha"] == pytest.approx(250.0) and s["median_ha"] == pytest.approx(5.0)
    assert s["p90_ha"] == pytest.approx(float(np.percentile(sizes * 0.01, 90)))
    assert s["total_ha"] == pytest.approx(sizes.sum() * 0.01)
    empty = df.patch_size_summary([])
    assert empty["count"] == 0 and empty["median_ha"] is None and empty["max_ha"] is None
    assert set(empty["histogram"].values()) == {0} and list(empty["histogram"]) == list(df.PATCH_BIN_LABELS)


def test_loss_year_breakdown_counts_scored_pixels_per_loss_year():
    ly = np.array([[0, 0, 19, 19], [20, 20, 22, 25]], dtype=np.uint8)
    scored = np.array([[True, True, True, True], [True, True, True, False]])
    conv = np.array([[False, True, True, False], [True, True, False, True]])
    flips = np.array([[True, True, True, True], [True, False, False, True]])
    got = df.loss_year_breakdown(ly, scored, conv, flips)
    assert list(got) == ["no_loss", "2019", "2020", "2022"]
    assert got["no_loss"] == {"pixels": 2, "converted_share": 0.5, "flip_share": 1.0}
    assert got["2019"] == {"pixels": 2, "converted_share": 0.5, "flip_share": 1.0}
    assert got["2020"] == {"pixels": 2, "converted_share": 1.0, "flip_share": 0.5}
    assert got["2022"] == {"pixels": 1, "converted_share": 0.0, "flip_share": 0.0}
    halves = df.loss_year_breakdown(
        [ly[:1], ly[1:]], [scored[:1], scored[1:]], [conv[:1], conv[1:]], [flips[:1], flips[1:]]
    )
    assert halves == got
    below = flips & ~conv
    with_cut = df.loss_year_breakdown(ly, scored, conv, flips, below)
    assert with_cut["no_loss"]["below_cut_share"] == 0.5 and with_cut["2019"]["below_cut_share"] == 0.5
    assert with_cut["2020"]["below_cut_share"] == 0.0 and with_cut["2022"]["below_cut_share"] == 0.0
    assert {k: {n: v for n, v in row.items() if n != "below_cut_share"} for k, row in with_cut.items()} == got


def test_spread_indices_spreads_over_the_range():
    assert spread_indices(25, 8) == [0, 3, 7, 10, 14, 17, 21, 24]
    assert spread_indices(6, 8) == [0, 1, 2, 3, 4, 5]
    assert spread_indices(36, 8)[-1] == 35 and 1 not in spread_indices(36, 8)
    assert spread_indices(0, 8) == [] and spread_indices(5, 0) == []


def test_band_mosaic_equals_the_mosaic_of_the_whole_stack(boundary_reader):
    # The deforest COG is mosaicked through one index band (_common.BandMosaic) and gathered band by
    # band: the result must be the one-shot mosaic of the whole stack, seam fills included, with the
    # bands put in any order, on the zone grid asked for.
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(boundary_reader, box(lon0 - 0.006, lat0 - 0.005, lon0 + 0.0065, lat0 + 0.004))
    assert plan.zone_names == ["10N", "11N"]
    rng = np.random.default_rng(0)
    for dst in ("10N", "11N"):
        for density in (0.3, 1.0):
            stacks = {}
            for z in plan.zones:
                keep = (rng.random(z.shape) < density) & z.mask
                stack = np.where(keep[None], rng.random((4, *z.shape)), np.nan).astype(np.float32)
                stack[1:][rng.random((3, *z.shape)) < 0.3] = np.nan
                stacks[z.zone] = stack
            whole, transform, crs = export.mosaic_zones(
                [export.GridArray(stacks[z.zone], z) for z in plan.zones], dst_zone=dst, nodata=np.nan
            )
            bands = BandMosaic(plan, 4, dst_zone=dst)
            for k in (3, 0, 2, 1):
                bands.put(k, {z: stacks[z][k] for z in stacks})
            assert bands.data is not None
            np.testing.assert_array_equal(bands.data, whole)
            assert bands.transform == transform and bands.crs == crs
    # By default the mosaic is on the grid of the zone with the most AOI pixels.
    big = max(plan.zones, key=lambda z: z.n_pixels)
    default = BandMosaic(plan, 1)
    assert default.crs == big.grid.crs and default.transform == BandMosaic(plan, 1, dst_zone=big.zone).transform


def test_memory_estimate_grows_with_the_window_and_the_ring(boundary_reader):
    plan = plan_aoi(boundary_reader, boundary_aoi())
    base = df.memory_estimate(plan, block=(32, 32), ring_km=None)
    assert base > df.MEMORY_FIXED
    assert df.memory_estimate(plan, block=(32, 32), ring_km=50.0) > base
    assert df.memory_estimate(plan, block=(512, 2048), ring_km=None) >= base


# --------------------------------------------------------------------------------------------
# Runs on the caldor fixture: validation window, regime guard on labels, early failures
# --------------------------------------------------------------------------------------------

AOI_ROWS = AOI_COLS = (12, 52)
RUN_KW = {"label_ring_km": 0.1, "labels_per_class": 100, "min_area_ha": 0.05, "block": (32, 32)}


def chunk_years(log: KeyLog, layer: str = "embeddings") -> list[int]:
    years = set()
    for key in log.chunk_keys():
        parts = key.split("/")
        if parts[1] == layer and parts[2] == "c":
            years.add(FIRST_YEAR + int(parts[3]))
    return sorted(years)


def write_aoi(path: Path, geometry) -> Path:
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(geometry)}],
    }
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def window_aoi(caldor_fixture, tmp_path):
    geometry = fixture_window_lonlat(caldor_fixture, AOI_ROWS, AOI_COLS)
    return write_aoi(tmp_path / "aoi.geojson", geometry), geometry


def piece_inside(pieces, geometry) -> np.ndarray:
    """Hansen piece pixels whose centre lies inside a lon/lat geometry."""
    h, w = pieces["lossyear"].shape
    r = float(pieces["res"])
    lon = float(pieces["west"]) + (np.arange(w) + 0.5) * r
    lat = float(pieces["north"]) - (np.arange(h) + 0.5) * r
    lon2, lat2 = np.meshgrid(lon, lat)
    return shapely.contains_xy(geometry, lon2, lat2)


def test_validation_scores_loss_after_year_a_through_year_b(caldor_reader, caldor_hansen, window_aoi, tmp_path):
    path, geometry = window_aoi
    inside = piece_inside(caldor_hansen, geometry)
    lossyear = caldor_hansen["lossyear"]
    # Put year_a's own loss (2019) on some AOI pixels with none: it must not count in the headline
    # window, and does in the lag-aligned one.
    marked = inside & (lossyear == 0) & (np.arange(lossyear.shape[1])[None, :] % 3 == 0)
    assert marked.sum() > 20
    # And water (datamask 2) on the west half of the AOI: only land is scored.
    lon = float(caldor_hansen["west"]) + (np.arange(lossyear.shape[1]) + 0.5) * float(caldor_hansen["res"])
    water = inside & (lon[None, :] < geometry.centroid.x)

    def edit(layer, data):
        if layer == "lossyear":
            return np.where(marked, 19, data)
        return np.where(water, 2, data) if layer == "datamask" else data

    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen, edit=edit)
    res = df.run_deforest(
        path, 2019, 2022, tmp_path / "out", validate=True, reader=caldor_reader, hansen_url_template=template, **RUN_KW
    )
    (z,) = plan_aoi(caldor_reader, geometry).zones
    ly = validate.hansen_layer(z, "lossyear", url_template=template)
    tc = validate.hansen_layer(z, "treecover2000", url_template=template)
    land = validate.hansen_valid(z, url_template=template)
    assert 0.3 < land.sum() / z.mask.sum() < 0.7, "about half the AOI is water"
    window = (ly >= 20) & (ly <= 22) & land
    lagged = (ly >= 19) & (ly <= 21) & land
    only_b = (ly == 22) & land
    assert ((ly == 19) & land).sum() > 20 and ((ly > 22) & land).sum() > 0
    assert window.sum() > only_b.sum() > 0
    val = json.loads(res.paths.validation.read_text())
    with rasterio.open(res.paths.tif) as ds:
        prob_a, prob_b, _, converted = ds.read()
    classified = np.isfinite(converted)
    assert classified[z.mask].all(), "every AOI pixel is classified on this window"
    done = converted == 1
    assert (done & ~land & z.mask).any(), "some conversion lies on the water half and must not be scored"
    n_land = int((classified & land).sum())
    conv = val["scores"]["conversion"]
    assert conv["n"] == n_land < int(z.mask.sum()) and res.summary["validation"]["pixels_scored"] == n_land
    assert conv["tp"] + conv["fn"] == int(window.sum())
    assert conv["tp"] == int((done & window).sum()) and conv["tp"] + conv["fp"] == int((done & land).sum())
    lag = val["scores"]["conversion_lag_aligned"]
    assert lag["n"] == n_land and lag["tp"] + lag["fn"] == int(lagged.sum()) and lag["tp"] == int((done & lagged).sum())
    # The lag window gains year_a's loss and loses year_b's.
    gained, lost = int(((ly == 19) & land).sum()), int(((ly == 22) & land).sum())
    assert lag["tp"] + lag["fn"] == conv["tp"] + conv["fn"] + gained - lost
    only = val["scores"]["conversion_year_b_only"]
    assert only["n"] == n_land
    assert only["tp"] + only["fn"] == int(only_b.sum()) and only["tp"] == int((done & only_b).sum())
    assert only["precision"] <= conv["precision"]
    for year, prob, key in ((2019, prob_a, "forest_agreement_year_a"), (2022, prob_b, "forest_agreement_year_b")):
        proxy = (tc >= 50) & ~((ly >= 1) & (ly <= year - 2000)) & land
        forest = np.isfinite(prob) & (prob >= 0.5)
        agree = val["scores"][key]
        assert agree["pixels"] == n_land
        assert agree["accuracy"] == pytest.approx(float(np.mean(forest[land] == proxy[land])))
        assert agree["recall_forest"] == pytest.approx(float(forest[proxy].mean()))
    # The per loss year breakdown: 2019 (year_a) is scored but outside the window; water is not scored.
    breakdown = val["scores"]["loss_year_breakdown"]
    year_a_loss = (ly == 19) & land
    assert breakdown["2019"]["pixels"] == int(year_a_loss.sum())
    assert breakdown["2019"]["converted_share"] == pytest.approx(float((converted[year_a_loss] == 1).mean()))
    flips = (prob_a >= 0.5) & (prob_b < 0.5)
    assert breakdown["2021"]["flip_share"] == pytest.approx(float(flips[(ly == 21) & land].mean()))
    assert breakdown["no_loss"]["pixels"] == int(((ly == 0) & land).sum())
    assert sum(v["pixels"] for v in breakdown.values()) == n_land
    # The flips the cross-check cut removed, on land, split by Hansen loss.
    with rasterio.open(res.paths.tif) as ds:
        dist = ds.read(3)
    below = flips & ~(dist > res.summary["cross_check"]["cut"]) & land
    codes = ly[below].astype(int)
    removed = val["scores"]["cross_check_removed"]
    assert removed["flips_below_cut"] == int(below.sum()) > 0
    assert removed["loss_in_window"] == int(((codes >= 20) & (codes <= 22)).sum())
    assert removed["loss_lag_aligned"] == int(((codes >= 19) & (codes <= 21)).sum())
    assert removed["no_loss"] == int((codes == 0).sum()) and removed["loss_year_a"] == int((codes == 19).sum())
    assert res.summary["validation"]["loss_year_breakdown"] == breakdown
    assert any("annual embedding" in c for c in val["caveats"])
    assert any("lag_aligned" in c and "2019 through 2021" in c for c in val["caveats"])
    assert not any("the window a conversion between the two embeddings covers" in c for c in val["caveats"])
    assert val["reference"]["loss_window"]["lossyear_codes"] == [20, 21, 22]
    assert val["reference"]["lag_aligned_window"]["lossyear_codes"] == [19, 20, 21]
    assert val["reference"]["year_b_only_code"] == 22 and val["reference"]["source"] == "hansen"
    assert res.summary["validation"]["year_b_only"]["recall"] == only["recall"]


def prob_b_of(res) -> np.ndarray:
    with rasterio.open(res.paths.tif) as ds:
        return ds.read(2)


def caldor_reader_with(fx: dict[str, np.ndarray], edit) -> tuple[TesseraReader, KeyLog]:
    """The caldor fixture store of conftest, with ``edit(name, year, array)`` applied to each fixture layer."""
    a, _, c, _, _, f = fx["transform"]
    h = w = fx["scales"].shape[1]
    years = [int(y) for y in fx["years"]]

    def fill(name: str, ti: int, shape: tuple[int, ...]) -> np.ndarray:
        out = np.full(shape, np.nan, dtype=np.float32) if name == "scales" else np.zeros(shape, dtype=fx[name].dtype)
        year = FIRST_YEAR + ti
        if year in years:
            out[PAD_ROWS : PAD_ROWS + h, PAD_COLS : PAD_COLS + w] = edit(name, year, fx[name][years.index(year)].copy())
        return out

    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    add_zone(
        root,
        str(fx["zone"]),
        left=c - PAD_COLS * a,
        top=f + PAD_ROWS * a,
        height=h + 2 * PAD_ROWS,
        width=w + 2 * PAD_COLS,
        fill=fill,
        years_complete=years,
    )
    log = KeyLog(mem)
    return TesseraReader.from_store(log), log


def test_labels_outside_the_aoi_regime_are_dropped(caldor_fixture, caldor_hansen, window_aoi, tmp_path):
    # In 2019 the west of the window had no radar: ring labels there are optical-only, the AOI is
    # mostly radar in both years, so they are dropped, and the AOI pixels there switch regime.
    west = 20

    def edit(name, year, arr):
        if year == 2019 and name in ("s1_asc_obs_count", "s1_desc_obs_count"):
            arr[:, :west] = 0
        return arr

    reader, log = caldor_reader_with(caldor_fixture, edit)
    path, geometry = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)
    res = df.run_deforest(path, 2019, 2022, tmp_path / "out", reader=reader, hansen_url_template=template, **RUN_KW)
    s = res.summary
    radar = s["radar"]
    assert radar["training_regime"] == "radar"
    labels = pd.read_parquet(res.extra_paths["labels"])
    store_west = PAD_COLS + west
    dropped = labels[labels["drop_reason"] == "regime"]
    assert len(dropped) == s["labels"]["dropped_regime"] == radar["labels_dropped_regime"] > 0
    assert (dropped["col"] < store_west).all() and (dropped["regime"] == "optical").all()
    used = labels[labels["used"]]
    assert (used["regime"] == "radar").all() and (used["col"] >= store_west).all()
    assert s["classifier"]["labels"] == len(used)
    assert (labels.loc[~labels["used"], "fold"] == -1).all() and (used["fold"] >= 0).all()
    (z,) = plan_aoi(reader, geometry).zones
    cols = np.arange(z.cols.start, z.cols.stop)[None, :] + np.zeros(z.shape, dtype=int)
    switching = z.mask & (cols < store_west)
    assert radar["regime_mismatch_pixels"] == int(switching.sum()) > 0
    with rasterio.open(res.paths.tif) as ds:
        prob_a, prob_b, dist, converted = ds.read()
    assert np.isnan(converted[switching]).all() and np.isnan(prob_a[switching]).all()
    assert np.isfinite(prob_b[switching]).all() and np.isnan(dist[switching]).all()
    assert np.isfinite(converted[z.mask & ~switching]).all()
    assert s["classified_pixels"] == int((z.mask & ~switching).sum())
    # year_a's statistics count its own classified pixels only (not the switching west).
    cls_a = np.isfinite(prob_a) & z.mask
    n_forest_a = int(((prob_a >= 0.5) & cls_a).sum())
    assert s["forest"]["2019"]["classified_pixels"] == int(cls_a.sum()) < int(z.mask.sum())
    assert s["forest"]["2019"]["forest_fraction"] == pytest.approx(n_forest_a / cls_a.sum())
    assert s["forest_fraction_year_a"] == pytest.approx(n_forest_a / cls_a.sum())
    warnings = res.report["warnings"]
    assert any("dropped because their 2019 radar regime" in w for w in warnings)
    assert any("switch radar regime" in w and "survey B5" in w for w in warnings)


def test_too_few_usable_labels_stop_before_the_aoi_stream(caldor_store, caldor_reader, window_aoi, tmp_path):
    _, log = caldor_store
    path, _ = window_aoi
    rows = ["lon,lat,label"]
    # Five usable labels per class on the fixture window, twenty per class in its no-data padding.
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    a, _, c, _, _, f = 10.0, 0.0, 731280.0, 0.0, -10.0, 4286160.0
    for label in (1, 0):
        for i in range(5):
            lon, lat = to_ll.transform(c + (2 + i) * a + 5, f - (2 + 3 * label) * a - 5)
            rows.append(f"{lon:.7f},{lat:.7f},{label}")
        for i in range(20):
            lon, lat = to_ll.transform(c - (5 + i) * a - 5, f + (3 + label) * a)
            rows.append(f"{lon:.7f},{lat:.7f},{label}")
    csv = tmp_path / "labels.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(df.DeforestError, match="Only 5 forest and 5 non-forest labels are usable"):
        df.run_deforest(path, 2019, 2022, tmp_path / "out", labels=csv, reader=caldor_reader, **RUN_KW)
    assert chunk_years(log) == [2019], "only the label chunks were read, not the AOI stream"


def test_a_label_ring_without_labels_stops_before_any_chunk_read(
    caldor_store, caldor_reader, caldor_hansen, window_aoi, tmp_path
):
    _, log = caldor_store
    path, _ = window_aoi
    # Land everywhere, but every pixel between the cover cuts with no loss: no label of either class.
    layers = {"treecover2000": 30, "lossyear": 0, "datamask": 1}
    template = write_hansen_tiles(
        tmp_path / "between", caldor_hansen, edit=lambda layer, d: np.full_like(d, layers[layer])
    )
    with pytest.raises(df.DeforestError, match="holds 0 forest and 0 non-forest") as info:
        df.run_deforest(
            path, 2019, 2022, tmp_path / "out", reader=caldor_reader, hansen_url_template=template, **RUN_KW
        )
    assert "--label-ring-km" in str(info.value) and log.chunk_keys() == []


def test_a_label_ring_on_water_or_missing_tiles_names_the_version(
    caldor_store, caldor_reader, caldor_hansen, window_aoi, tmp_path
):
    _, log = caldor_store
    path, _ = window_aoi
    water = write_hansen_tiles(
        tmp_path / "water", caldor_hansen, edit=lambda layer, d: np.full_like(d, 2) if layer == "datamask" else d
    )
    nowhere = str(tmp_path / "nowhere" / "Hansen_{version}_{layer}_{tile}.tif")
    for template in (water, nowhere):
        with pytest.raises(df.DeforestError, match="no land pixel") as info:
            df.run_deforest(
                path, 2019, 2022, tmp_path / "out", reader=caldor_reader, hansen_url_template=template, **RUN_KW
            )
        message = str(info.value)
        assert validate.HANSEN_VERSION in message and template in message and "--hansen-version" in message
        assert "--labels" in message and "Widen --label-ring-km" not in message
    assert log.chunk_keys() == []


def test_unreachable_hansen_labels_stop_before_any_chunk_read(
    caldor_store, caldor_reader, window_aoi, tmp_path, monkeypatch
):
    _, log = caldor_store
    path, _ = window_aoi

    def unreachable(*args, **kwargs):
        raise validate.ReferenceUnavailableError("storage.googleapis.com refused the request (403)")

    monkeypatch.setattr(validate, "hansen_layer", unreachable)
    with pytest.raises(df.DeforestError, match="--labels") as info:
        df.run_deforest(path, 2019, 2022, tmp_path / "out", reader=caldor_reader, **RUN_KW)
    assert "storage.googleapis.com" in str(info.value) and log.chunk_keys() == []


def test_a_bad_labels_csv_stops_before_the_store_is_read(caldor_store, caldor_reader, window_aoi, tmp_path):
    _, log = caldor_store
    path, _ = window_aoi
    bad = tmp_path / "labels.csv"
    bad.write_text(csv_text(n_non=3), encoding="utf-8")
    with pytest.raises(AppInputError, match="3 non-forest"):
        df.run_deforest(path, 2019, 2022, tmp_path / "out", labels=bad, reader=caldor_reader)
    assert log.chunk_keys() == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"year_a": 2022, "year_b": 2022}, "must be after year_a"),
        ({"year_a": 2016}, "Years must lie in 2017 to 2025"),
        ({"labels_per_class": 10}, "at least 20"),
        ({"label_chunks": 0}, "label_chunks"),
        ({"label_ring_km": 0.0}, "label ring"),
        ({"label_ring_km": 60.0}, "label ring"),
        ({"label_ring_km": 1.0, "label_ring_inner_km": 1.0}, "label ring"),
        ({"forest_cover": 20, "nonforest_cover": 30}, "Cover cuts"),
        ({"forest_cover": 120}, "Cover cuts"),
        ({"forest_prob": 1.0}, "forest_prob"),
        ({"stable_percentile": 0.0}, "stable_percentile"),
        ({"min_area_ha": -1.0}, "min_area_ha"),
        ({"hansen_version": "v1.13"}, "--hansen-version"),
        ({"hansen_version": "GFC-2021-v1.9", "year_a": 2022, "year_b": 2024}, "cannot label year_a 2022"),
        ({"hansen_version": "GFC-2021-v1.9", "validate": True}, "cannot validate year_b 2022"),
        ({"reference": "missing.geojson"}, "does not exist"),
    ],
)
def test_bad_arguments_stop_before_the_store_is_read(caldor_store, caldor_reader, window_aoi, tmp_path, kwargs, match):
    _, log = caldor_store
    path, _ = window_aoi
    args = {"year_a": 2019, "year_b": 2022, **kwargs}
    year_a, year_b = args.pop("year_a"), args.pop("year_b")
    if "reference" in args:
        args["reference"] = tmp_path / args["reference"]
    with pytest.raises(AppInputError, match=match):
        df.run_deforest(path, year_a, year_b, tmp_path / "out", reader=caldor_reader, **args)
    assert log.chunk_keys() == []


def cells_csv(path: Path, fx, cells: dict[int, tuple[tuple[int, int], tuple[int, int]]]) -> Path:
    """A labels CSV with one row per fixture-grid pixel centre of each ``{label: (rows, cols)}`` window."""
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    to_ll = Transformer.from_crs(int(fx["epsg"]), 4326, always_xy=True)
    lines = ["lon,lat,label"]
    for label, (rows, cols) in cells.items():
        for r in range(*rows):
            for col in range(*cols):
                lon, lat = to_ll.transform(c + (col + 0.5) * a, f - (r + 0.5) * a)
                lines.append(f"{lon:.8f},{lat:.8f},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_own_labels_and_reference_never_read_hansen(caldor_fixture, caldor_reader, window_aoi, tmp_path):
    # GFC-2021 records loss through 2021, and the tiles do not exist: with --labels and --reference
    # neither is needed.
    path, geometry = window_aoi
    ref = write_aoi(tmp_path / "ref.geojson", geometry.buffer(-0.0001))
    csv = cells_csv(tmp_path / "labels.csv", caldor_fixture, {1: ((26, 34), (0, 6)), 0: ((0, 6), (40, 48))})
    res = df.run_deforest(
        path,
        2019,
        2022,
        tmp_path / "out",
        validate=True,
        reference=ref,
        labels=csv,
        hansen_version="GFC-2021-v1.9",
        hansen_url_template=str(tmp_path / "nowhere" / "Hansen_{version}_{layer}_{tile}.tif"),
        reader=caldor_reader,
        **RUN_KW,
    )
    s = res.summary
    assert s["label_source"] == "csv" and s["labels"]["rows"] == 96 and s["labels"]["inside_aoi"] == 0
    assert s["validation"]["source"] == "file" and res.validation_written
    val = json.loads(res.paths.validation.read_text())
    assert "conversion_year_b_only" not in val["scores"] and "forest_agreement_year_a" not in val["scores"]
    assert val["reference"]["file"] == "ref.geojson"
    assert any("supplied with --labels" in c for c in val["caveats"])
    assert s["hansen"] == {
        "version": "GFC-2021-v1.9",
        "url_template": str(tmp_path / "nowhere" / "Hansen_{version}_{layer}_{tile}.tif"),
        "labels": False,
        "validation": False,
    }


# --------------------------------------------------------------------------------------------
# Runs: the label stage's early failures and its notes
# --------------------------------------------------------------------------------------------


def test_hansen_without_land_skips_or_stops_the_validation(
    caldor_store, caldor_reader, caldor_fixture, caldor_hansen, window_aoi, tmp_path
):
    # A version the template has no tiles for (a 404 on the bucket, or no file locally) reads as
    # ocean: the reference has no land pixel, so the validation must be skipped with a warning, or
    # refused under --strict-validate before any chunk read, never scored over 0 pixels in silence.
    _, log = caldor_store
    path, _ = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)  # the default version only
    csv = cells_csv(tmp_path / "labels.csv", caldor_fixture, {1: ((26, 34), (0, 6)), 0: ((0, 6), (40, 48))})
    version = "GFC-2024-v1.12"
    kw = {**RUN_KW, "labels": csv, "validate": True, "reader": caldor_reader, "hansen_url_template": template}
    res = df.run_deforest(path, 2019, 2022, tmp_path / "out", hansen_version=version, **kw)
    assert not res.validation_written and not res.paths.validation.exists()
    error = res.summary["validation"]["error"]
    assert "no land pixel" in error and version in error and template in error and "--hansen-version" in error
    assert any(w.startswith(df.VALIDATION_SKIPPED) and version in w for w in res.report["warnings"])
    reads = len(log.chunk_keys())
    with pytest.raises(StrictValidationError, match="no land pixel"):
        df.run_deforest(path, 2019, 2022, tmp_path / "strict", hansen_version=version, strict_validate=True, **kw)
    assert len(log.chunk_keys()) == reads and not (tmp_path / "strict" / "deforest_report.json").exists()
    # With the version the tiles are under, the same run validates.
    ok = df.run_deforest(path, 2019, 2022, tmp_path / "ok", strict_validate=True, **kw)
    assert ok.validation_written and ok.summary["validation"]["pixels_scored"] > 0


def test_a_non_default_hansen_version_reaches_the_labels_and_the_validation(
    caldor_reader, caldor_hansen, window_aoi, tmp_path
):
    path, geometry = window_aoi
    version = "GFC-2024-v1.12"
    # The default version's tiles sit next to them, with no loss anywhere: reading them instead
    # would change both the labels (no cleared non-forest, no lossyear) and the reference (empty).
    template = write_hansen_tiles(
        tmp_path / "hansen", caldor_hansen, edit=lambda layer, d: np.zeros_like(d) if layer == "lossyear" else d
    )
    write_hansen_tiles(tmp_path / "hansen", {**caldor_hansen, "version": np.array(version)})
    res = df.run_deforest(
        path,
        2019,
        2022,
        tmp_path / "out",
        validate=True,
        reader=caldor_reader,
        hansen_url_template=template,
        hansen_version=version,
        **RUN_KW,
    )
    plan = plan_aoi(caldor_reader, geometry)
    (z,) = plan.zones
    loss = validate.hansen_loss(z, 2019, 2022, version=version, url_template=template)
    assert loss.sum() > 0 and not validate.hansen_loss(z, 2019, 2022, url_template=template).any()
    assert res.summary["validation"]["reference_pixels_in_aoi"] == int(loss.sum())
    val = json.loads(res.paths.validation.read_text())
    assert val["reference"]["version"] == version and res.summary["labels"]["hansen_version"] == version
    labels = pd.read_parquet(res.extra_paths["labels"])
    (rz,) = df.plan_label_ring(caldor_reader, df.label_ring(geometry, outer_km=0.1), plan).zones
    layers = [validate.hansen_layer(rz, n, version=version, url_template=template) for n in df_layers()]
    at = (labels["row"].to_numpy() - rz.rows.start, labels["col"].to_numpy() - rz.cols.start)
    assert (df.label_classes(*layers, 2019)[at] == labels["label"].to_numpy()).all()
    assert (labels["hansen_lossyear"].to_numpy() == layers[1][at]).all() and (labels["hansen_lossyear"] > 0).any()


def df_layers() -> tuple[str, str, str]:
    """The Hansen layers label_classes takes, in its order."""
    return ("treecover2000", "lossyear", "datamask")


SEAM_WEST_AOI = box(-120.004, 38.849, -120.001, 38.851)  # wholly in 10N, about 100 m west of 120 W


def test_year_a_missing_where_only_the_label_ring_reaches_names_the_ring(
    boundary_store, boundary_reader, tmp_path, monkeypatch
):
    # 11N lacks 2023 in the boundary store. The AOI is in 10N, but its 0.3 km ring reaches 11N:
    # the error must say the ring is why, and what to change, before any Hansen or chunk read.
    _, log = boundary_store
    calls: list[str] = []

    def hansen_layer(plan, layer, **kwargs):
        calls.append(layer)
        raise AssertionError("Hansen must not be read")

    monkeypatch.setattr(validate, "hansen_layer", hansen_layer)
    aoi = write_aoi(tmp_path / "aoi.geojson", SEAM_WEST_AOI)
    assert plan_aoi(boundary_reader, SEAM_WEST_AOI).zone_names == ["10N"]
    with pytest.raises(io.YearsNotPublishedError) as info:
        df.run_deforest(aoi, 2023, 2024, tmp_path / "out", reader=boundary_reader, label_ring_km=0.3)
    assert info.value.missing == {"11N": [2023]}
    notes = " ".join(getattr(info.value, "__notes__", []))
    assert "Zone 11N is read only for the Hansen labels" in notes and "0 to 0.3 km" in notes
    assert "--label-ring-km" in notes and "--labels" in notes
    assert log.chunk_keys() == [] and calls == []


def lonlat_csv(path: Path, rows: list[tuple[float, float, int]]) -> Path:
    path.write_text("lon,lat,label\n" + "".join(f"{x:.7f},{y:.7f},{v}\n" for x, y, v in rows), encoding="utf-8")
    return path


def test_year_a_missing_where_only_csv_labels_fall_names_the_file(boundary_store, boundary_reader, tmp_path):
    _, log = boundary_store
    aoi = write_aoi(tmp_path / "aoi.geojson", SEAM_WEST_AOI)
    rows = [(-120.0035 + (i % 10) * 1e-4, 38.8515 + (i // 10) * 1e-4, i % 2) for i in range(50)]  # 10N
    rows += [(-119.9995 + i * 1e-4, 38.8505, 1) for i in range(7)]  # 11N
    csv = lonlat_csv(tmp_path / "hand.csv", rows)
    with pytest.raises(io.YearsNotPublishedError) as info:
        df.run_deforest(aoi, 2023, 2024, tmp_path / "out", reader=boundary_reader, labels=csv)
    notes = " ".join(getattr(info.value, "__notes__", []))
    assert "Zone 11N holds 7 of the 57 labels in hand.csv" in notes and "Drop those labels" in notes
    assert log.chunk_keys() == []


def test_sampled_chunks_holding_one_class_stop_before_any_chunk_read(
    caldor_store, caldor_reader, window_aoi, tmp_path, monkeypatch
):
    # The ring holds both classes, but each store chunk only one: one sampled chunk cannot train.
    _, log = caldor_store
    path, _ = window_aoi

    def hansen_layer(plan, layer, **kwargs):
        cols = np.broadcast_to(np.arange(plan.cols.start, plan.cols.stop)[None, :], plan.shape)
        forest = (cols // 32) % 2 == 0
        values = {"treecover2000": np.where(forest, 90, 0), "lossyear": 0, "datamask": 1}[layer]
        return np.broadcast_to(values, plan.shape).astype(np.uint8)

    monkeypatch.setattr(validate, "hansen_layer", hansen_layer)
    with pytest.raises(df.DeforestError, match="ring chunks sampled hold") as info:
        df.run_deforest(path, 2019, 2022, tmp_path / "out", reader=caldor_reader, **{**RUN_KW, "label_chunks": 1})
    assert "--label-chunks" in str(info.value) and log.chunk_keys() == []


# --------------------------------------------------------------------------------------------
# Runs: denominators, transfer and the cross-check against the ring
# --------------------------------------------------------------------------------------------


def test_year_b_pixels_out_of_regime_set_the_year_b_statistics(caldor_fixture, caldor_hansen, window_aoi, tmp_path):
    # In 2022 the west of the window has no radar: its AOI pixels are classified in 2019 only. Each
    # year's statistics must use that year's classified pixels, the conversion share the year_a
    # forest classified in both years, and each forest agreement that year's pixels.
    west = 20

    def edit(name, year, arr):
        if year == 2022 and name in ("s1_asc_obs_count", "s1_desc_obs_count"):
            arr[:, :west] = 0
        return arr

    reader, _ = caldor_reader_with(caldor_fixture, edit)
    path, geometry = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)
    res = df.run_deforest(
        path, 2019, 2022, tmp_path / "out", validate=True, reader=reader, hansen_url_template=template, **RUN_KW
    )
    s = res.summary
    (z,) = plan_aoi(reader, geometry).zones
    inside = z.mask
    with rasterio.open(res.paths.tif) as ds:
        prob_a, prob_b, _, converted = ds.read()
    cls = {2019: np.isfinite(prob_a) & inside, 2022: np.isfinite(prob_b) & inside}
    assert int(cls[2019].sum()) == 1600 > int(cls[2022].sum()) > 0
    table = pd.read_parquet(res.paths.summary_parquet).set_index("year")
    for year, prob in ((2019, prob_a), (2022, prob_b)):
        n_forest = int(((prob >= 0.5) & cls[year]).sum())
        row = s["forest"][str(year)]
        assert row["forest_pixels"] == n_forest and row["classified_pixels"] == int(cls[year].sum())
        assert row["forest_fraction"] == pytest.approx(n_forest / cls[year].sum())
        assert table.loc[year, "forest_fraction"] == pytest.approx(n_forest / cls[year].sum())
    both = np.isfinite(converted) & inside
    assert s["classified_pixels"] == int(both.sum()) == int(cls[2022].sum())
    forest_a_both = (prob_a >= 0.5) & both
    assert s["forest_pixels_year_a_classified_both"] == int(forest_a_both.sum())
    assert s["conversion_fraction_of_forest"] == pytest.approx((converted == 1).sum() / forest_a_both.sum())
    land = validate.hansen_valid(z, url_template=template)
    ly = validate.hansen_layer(z, "lossyear", url_template=template)
    tc = validate.hansen_layer(z, "treecover2000", url_template=template)
    val = json.loads(res.paths.validation.read_text())
    for year, prob, key in ((2019, prob_a, "forest_agreement_year_a"), (2022, prob_b, "forest_agreement_year_b")):
        scope = cls[year] & land
        proxy = (tc >= 50) & ~((ly >= 1) & (ly <= year - 2000))
        assert val["scores"][key]["pixels"] == int(scope.sum())
        assert val["scores"][key]["accuracy"] == pytest.approx(float(np.mean((prob >= 0.5)[scope] == proxy[scope])))


def test_a_classifier_that_transfers_badly_to_year_b_is_reported(caldor_fixture, caldor_hansen, window_aoi, tmp_path):
    # Every 2022 vector is turned around: the 2019 classifier then calls the 2022 forest non-forest
    # and the other way round. Its 2019 scores still look good, so only the transfer check at the
    # Hansen-stable labels, and the 2022 forest agreement, can show it.
    def edit(name, year, arr):
        if name == "embeddings" and year == 2022:
            return (-np.clip(arr.astype(np.int16), -127, 127)).astype(np.int8)
        return arr

    reader, _ = caldor_reader_with(caldor_fixture, edit)
    path, _ = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)
    res = df.run_deforest(
        path, 2019, 2022, tmp_path / "out", validate=True, reader=reader, hansen_url_template=template, **RUN_KW
    )
    s = res.summary
    tr = s["transfer"]
    assert tr["available"] and s["classifier"]["heldout"]["balanced_accuracy"] > 0.8
    assert tr["year_b"]["balanced_accuracy"] < 0.3 and tr["drop"] > 0.5
    assert min(tr["drop_labels_forest"], tr["drop_labels_nonforest"]) >= df.MIN_TRANSFER_LABELS
    warning = next(w for w in res.report["warnings"] if "transfers poorly to 2022" in w)
    shown = tr["year_a_heldout"]["balanced_accuracy"] - tr["drop"]
    assert f"accuracy is {shown:.3f}, {tr['drop']:.3f} below" in warning and "--label-chunks" in warning
    assert s["validation"]["forest_agreement_year_b"]["accuracy"] < 0.5
    assert s["validation"]["forest_agreement_year_a"]["accuracy"] > 0.6


def test_a_cross_check_cut_far_above_the_ring_label_cut_warns(caldor_reader, caldor_hansen, window_aoi, tmp_path):
    # On the fixture (a burn scar) many AOI pixels the classifier still calls forest in 2022 burned:
    # at the 90th percentile the AOI's cut sits well above the same percentile of the ring's
    # Hansen-stable forest labels.
    path, _ = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)
    res = df.run_deforest(
        path,
        2019,
        2022,
        tmp_path / "out",
        reader=caldor_reader,
        hansen_url_template=template,
        **{**RUN_KW, "stable_percentile": 90.0},
    )
    cross = res.summary["cross_check"]
    assert cross["ring_label_cut_labels"] >= df.MIN_RING_CUT_LABELS
    assert cross["cut_to_ring_label_cut"] == pytest.approx(cross["cut"] / cross["ring_label_cut"])
    assert cross["cut_to_ring_label_cut"] > df.CUT_WARN_RATIO
    warning = next(w for w in res.report["warnings"] if "times the 90th percentile" in w)
    assert f"{cross['cut']:.3f}" in warning and "survey N13" in warning and "--stable-percentile" in warning
    assert cross["removed_by_hansen_loss"] is None, "no validation, no split by Hansen loss"


# --------------------------------------------------------------------------------------------
# What run_deforest hands prepare_run: the digest params, the estimates
# --------------------------------------------------------------------------------------------


class _PreparedError(Exception):
    """Stops run_deforest at prepare_run."""


def prepared(monkeypatch, *args, **kwargs) -> dict:
    """What run_deforest passes prepare_run (its params, memory estimate and year reads), without running."""
    seen: dict = {}

    def stop(*a, **k):
        seen.update(params=a[6], memory_estimate=k["memory_estimate"], year_reads=k["year_reads"])
        raise _PreparedError

    monkeypatch.setattr(df, "prepare_run", stop)
    with pytest.raises(_PreparedError):
        df.run_deforest(*args, **kwargs)
    return seen


#: Every option that shapes the outputs, with a value other than its default.
DIGEST_CHANGES = [
    ("labels_per_class", 300),
    ("label_chunks", 12),
    ("label_ring_km", 2.0),
    ("label_ring_inner_km", 0.05),
    ("forest_cover", 60),
    ("nonforest_cover", 20),
    ("forest_prob", 0.7),
    ("stable_percentile", 80.0),
    ("min_area_ha", 1.0),
    ("hansen_version", "GFC-2024-v1.12"),
    ("hansen_url_template", "/elsewhere/Hansen_{version}_{layer}_{tile}.tif"),
    ("block", (64, 64)),
]


@pytest.mark.parametrize(("option", "value"), DIGEST_CHANGES)
def test_every_output_shaping_option_reaches_the_digest(
    monkeypatch, caldor_reader, window_aoi, tmp_path, option, value
):
    # A rerun skips when the digest of these params matches, so an option missing from them would
    # let a changed run return the old outputs.
    path, _ = window_aoi
    base = prepared(monkeypatch, path, 2019, 2022, tmp_path / "o", reader=caldor_reader)["params"]
    changed = prepared(monkeypatch, path, 2019, 2022, tmp_path / "o", reader=caldor_reader, **{option: value})["params"]
    expected = list(value) if isinstance(value, tuple) else value
    assert changed[option] == expected != base[option]
    assert {k: v for k, v in changed.items() if k != option} == {k: v for k, v in base.items() if k != option}


def test_the_digest_params_cover_every_option(monkeypatch, caldor_reader, caldor_fixture, window_aoi, tmp_path):
    # Runtime switches and what the digest covers on its own (years, seed, the AOI) aside, every
    # keyword of run_deforest is in the params; files are there by name and sha256.
    path, geometry = window_aoi
    runtime = {"aoi", "year_a", "year_b", "out_dir", "validate", "seed", "force", "max_area_km2", "max_zones"}
    runtime |= {"snapshot_id", "reader", "strict_validate"}
    covered = {name for name, _ in DIGEST_CHANGES} | {"labels", "reference"}
    assert set(inspect.signature(df.run_deforest).parameters) - runtime == covered
    csv = cells_csv(tmp_path / "labels.csv", caldor_fixture, {1: ((26, 34), (0, 6)), 0: ((0, 6), (40, 48))})
    ref = write_aoi(tmp_path / "ref.geojson", geometry.buffer(-0.0001))
    params = prepared(monkeypatch, path, 2019, 2022, tmp_path / "o", reader=caldor_reader, labels=csv, reference=ref)
    assert covered <= set(params["params"])
    assert params["params"]["labels"] == {"file": "labels.csv", "sha256": export.file_sha256(csv)}
    assert params["params"]["reference"] == {"file": "ref.geojson", "sha256": export.file_sha256(ref)}


def test_the_memory_estimate_charges_the_ring_only_for_hansen_labels(
    monkeypatch, caldor_reader, caldor_fixture, window_aoi, tmp_path
):
    path, geometry = window_aoi
    plan = plan_aoi(caldor_reader, geometry)
    kw = {"reader": caldor_reader, "label_ring_km": 20.0, "block": (32, 32)}
    hansen = prepared(monkeypatch, path, 2019, 2022, tmp_path / "o", **kw)
    wide = df.memory_estimate(plan, block=(32, 32), ring_km=20.0)
    assert hansen["memory_estimate"](plan) == wide > df.memory_estimate(plan, block=(32, 32), ring_km=None)
    assert hansen["year_reads"] == 3
    csv = cells_csv(tmp_path / "labels.csv", caldor_fixture, {1: ((26, 34), (0, 6)), 0: ((0, 6), (40, 48))})
    own = prepared(monkeypatch, path, 2019, 2022, tmp_path / "o", labels=csv, **kw)
    assert own["memory_estimate"](plan) == df.memory_estimate(plan, block=(32, 32), ring_km=None)


def test_non_default_options_shape_the_run(caldor_reader, caldor_hansen, window_aoi, tmp_path):
    path, geometry = window_aoi
    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen)
    kw = {
        "label_ring_km": 0.12,
        "label_ring_inner_km": 0.03,
        "labels_per_class": 60,
        "label_chunks": 3,
        "forest_cover": 60,
        "nonforest_cover": 20,
        "forest_prob": 0.6,
        "stable_percentile": 80.0,
        "min_area_ha": 0.1,
        "block": (32, 32),
        "seed": 2,
    }
    out = tmp_path / "out"
    res = df.run_deforest(path, 2019, 2022, out, reader=caldor_reader, hansen_url_template=template, **kw)
    s = res.summary
    for key, value in kw.items():
        if key not in ("block", "seed"):
            assert res.report["params"][key] == value, key
    plan = plan_aoi(caldor_reader, geometry)
    (z,) = plan.zones
    inside = z.mask
    with rasterio.open(res.paths.tif) as ds:
        prob_a, prob_b, dist, converted = ds.read()
    # Forest is p >= 0.6, not 0.5.
    assert s["forest_prob"] == 0.6
    for year, prob in ((2019, prob_a), (2022, prob_b)):
        n = int((inside & (prob >= 0.6)).sum())
        assert s["forest"][str(year)]["forest_pixels"] == n != int((inside & (prob >= 0.5)).sum())
    # The cut is the 80th percentile of the distance of pixels forest at 0.6 in both years.
    stable = inside & (prob_a >= 0.6) & (prob_b >= 0.6)
    d = dist[stable].astype(np.float64)
    cut = s["cross_check"]["cut"]
    assert cut == pytest.approx(float(np.percentile(d, 80)), rel=1e-6) and cut < float(np.percentile(d, 95))
    assert s["cross_check"]["percentile"] == 80.0 and s["cross_check"]["stable_forest_pixels"] == int(stable.sum())
    # The conversion is that rule, cleaned at 0.1 ha (10 pixels).
    raw = inside & (prob_a >= 0.6) & (prob_b < 0.6) & (dist > cut)
    np.testing.assert_array_equal(converted == 1, metrics.clean_mask(raw, min_pixels=10) & inside)
    assert s["min_pixels"] == 10
    # The labels: classes at the 60 and 20 percent cuts, from 3 of the 6 ring chunks, 30 to 120 m out.
    ring = df.label_ring(geometry, outer_km=0.12, inner_km=0.03)
    (rz,) = df.plan_label_ring(caldor_reader, ring, plan).zones
    layers = [validate.hansen_layer(rz, n, url_template=template) for n in df_layers()]
    custom = df.label_classes(*layers, 2019, forest_cover=60, nonforest_cover=20)
    default = df.label_classes(*layers, 2019)
    lab = s["labels"]
    assert lab["ring_pixels_forest"] == int((custom == F)[rz.mask].sum()) != int((default == F)[rz.mask].sum())
    assert lab["ring_pixels_nonforest"] == int((custom == N)[rz.mask].sum()) != int((default == N)[rz.mask].sum())
    labels = pd.read_parquet(res.extra_paths["labels"])
    at = (labels["row"].to_numpy() - rz.rows.start, labels["col"].to_numpy() - rz.cols.start)
    assert (custom[at] == labels["label"].to_numpy()).all()
    assert lab["chunks_used"] == 3 < lab["chunks_with_labels"] and labels["source_chunk"].nunique() <= 3
    assert (lab["used_forest"], lab["used_nonforest"]) == (60, 60)
    assert lab["ring"]["inner_km"] == 0.03 and lab["ring"]["outer_km"] == 0.12
    half_diagonal = 5.0 * np.sqrt(2.0)
    gap = shapely.distance(utm(geometry, ring.epsg), utm(shapely.points(labels["lon"], labels["lat"]), ring.epsg))
    assert gap.min() >= 30.0 - half_diagonal and gap.max() <= 120.0 + half_diagonal
    # A rerun with one option changed is not skipped.
    again = df.run_deforest(
        path, 2019, 2022, out, reader=caldor_reader, hansen_url_template=template, **{**kw, "stable_percentile": 95.0}
    )
    assert not again.skipped and again.report["args_digest"] != res.report["args_digest"]
    assert again.summary["cross_check"]["cut"] > cut


# --------------------------------------------------------------------------------------------
# Runs on designed stores: the regime probe's fallback, a zone seam, labels across a seam
# --------------------------------------------------------------------------------------------

LON0, LAT0 = BOUNDARY
STRIP = LON0 - 0.0018  # non-forest west of this, in every year
PATCH = box(LON0 - 0.0006, LAT0 - 0.0005, LON0 + 0.0006, LAT0 + 0.0005)  # forest cleared between 2018 and 2020


def designed_store(pattern, *, zones=("10N", "11N"), chunk=32, size=192, centre=BOUNDARY):
    """In-memory zone groups around ``centre`` whose pixels follow ``pattern``.

    ``pattern(lon, lat, rows, cols, year)`` gives ``(forest, data, radar)`` bool arrays on a zone's
    grid. A forest pixel's embedding points along band 1 and a non-forest one against it, with small
    noise elsewhere, so a linear classifier separates them and an unchanged pixel stays close in
    cosine distance.

    Returns:
        ``(reader, log, mem)``: a reader behind a key log, the log, and the store itself.
    """
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    for zone in zones:
        epsg = zone_epsg(zone)
        x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(*centre)
        left, top = (x // 10) * 10 - size / 2 * 10, (y // 10) * 10 + size / 2 * 10
        rows, cols = np.mgrid[0:size, 0:size]
        lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
            left + (cols + 0.5) * 10.0, top - (rows + 0.5) * 10.0
        )
        noise = np.random.default_rng(int(zone[:2]))

        def fill(name, ti, shape, lon=lon, lat=lat, rows=rows, cols=cols, noise=noise):
            forest, data, radar = pattern(lon, lat, rows, cols, FIRST_YEAR + ti)
            if name == "embeddings":
                e = noise.integers(-1, 2, size=shape, dtype=np.int8)
                e[..., 1] = np.where(forest, 60, -60)
                return e
            if name == "scales":
                return np.where(data, 0.01, np.nan).astype(np.float32)
            if name == "s1_asc_obs_count":
                return np.where(radar, 30, 0).astype(np.uint16)
            if name == "s1_desc_obs_count":
                return np.zeros(shape, dtype=np.uint16)
            return np.full(shape, 50, dtype=np.uint16)

        add_zone(root, zone, left=left, top=top, height=size, width=size, fill=fill, chunk=chunk)
    log = KeyLog(mem)
    return TesseraReader.from_store(log), log, mem


def seam_pattern(lon, lat, rows, cols, year):
    forest = lon >= STRIP
    if year >= 2020:
        forest = forest & ~shapely.contains_xy(PATCH, lon, lat)
    everywhere = np.ones(lon.shape, dtype=bool)
    return forest, everywhere, everywhere


def grid_box_lonlat(grid: ZoneGrid, rows: tuple[int, int], cols: tuple[int, int]):
    """A window of a zone grid as a lon/lat polygon."""
    square = box(
        grid.left + cols[0] * grid.pixel,
        grid.top - rows[1] * grid.pixel,
        grid.left + cols[1] * grid.pixel,
        grid.top - rows[0] * grid.pixel,
    )
    return shapely.transform(shapely.segmentize(square, 10.0), grid.to_lonlat().transform, interleaved=False)


def test_the_regime_probe_reads_on_until_a_block_has_data(tmp_path):
    # 8 x 8 chunks and blocks: the AOI has more blocks than the probe's spread sample, and only one
    # block outside that sample has data. The probe must read on until it finds it.
    size, chunk, centre = 128, 8, (-120.5, 38.85)
    full, _, _ = designed_store(
        lambda lon, lat, r, c, y: (c < 64, r >= 0, r >= 0), zones=("10N",), chunk=chunk, size=size, centre=centre
    )
    grid = full.grid("10N")
    aoi = grid_box_lonlat(grid, (64, 112), (24, 72))
    plan = plan_aoi(full, aoi)
    (z,) = plan.zones
    jobs = list(z.blocks(full.chunk_shape("10N"), (8, 8)))
    first = spread_indices(len(jobs), df.REGIME_PROBE_BLOCKS)
    assert len(jobs) > df.REGIME_PROBE_BLOCKS
    target = next(i for i in range(len(jobs)) if i not in first and z.submask(*jobs[i]).sum() >= 20)
    rs, cs = jobs[target]

    def pattern(lon, lat, rows, cols, year):
        data = (rows < 40) | ((rows >= rs.start) & (rows < rs.stop) & (cols >= cs.start) & (cols < cs.stop))
        return cols < 64, data, np.ones(lon.shape, dtype=bool)

    reader, _, _ = designed_store(pattern, zones=("10N",), chunk=chunk, size=size, centre=centre)
    to_ll = grid.to_lonlat()
    cells = [(r, c, 1) for r in range(10, 15) for c in range(10, 20)] + [
        (r, c, 0) for r in range(10, 15) for c in (90, 95)
    ]
    cells += [(r, c, 0) for r in range(20, 25) for c in range(80, 88)]
    rows = []
    for r, c, label in cells:
        x, y = to_ll.transform(grid.left + (c + 0.5) * 10.0, grid.top - (r + 0.5) * 10.0)
        rows.append((x, y, label))
    csv = lonlat_csv(tmp_path / "labels.csv", rows)
    res = df.run_deforest(
        write_aoi(tmp_path / "aoi.geojson", aoi), 2018, 2020, tmp_path / "out", labels=csv, reader=reader, block=(8, 8)
    )
    probe = res.summary["radar"]["probe"]
    assert probe["blocks_total"] == len(jobs)
    assert probe["blocks_read"] == len(first) + sum(1 for i in range(target) if i not in first) + 1 > len(first)
    assert res.summary["classified_pixels"] == int(z.submask(rs, cs).sum()) > 0
    # The estimate counts the label chunks once, for year_a: labels from a CSV are not read in year_b.
    lab = res.summary["labels"]
    assert lab["label_read_years"] == [2018] and lab["label_read_chunks"] > 0
    assert res.report["estimated_bytes"] == plan.estimated_bytes(reader, 3, block=(8, 8)) + (
        lab["label_read_chunks"] * chunk * chunk * BYTES_PER_PIXEL
    )
    assert res.summary["transfer"]["available"] is False and "--labels" in res.summary["transfer"]["reason"]


def test_deforest_across_a_zone_seam(tmp_path, monkeypatch):
    # A patch cleared across 120 W: the COG is on the grid of the zone with most AOI pixels (10N),
    # each band is the mosaic of the zones' own values, and the patch is one patch, kept whole.
    reader, _, mem = designed_store(seam_pattern)
    aoi = box(LON0 - 0.0025, LAT0 - 0.0015, LON0 + 0.0015, LAT0 + 0.0015)
    # 100 forest labels north of the AOI on both sides of the seam, 100 non-forest ones west of it.
    rows = [(LON0 - 0.0012 + i * 0.00026, LAT0 + 0.0018 + j * 0.0001, 1) for i in range(10) for j in range(10)]
    rows += [(LON0 - 0.0034 + i * 0.0002, LAT0 - 0.0014 + j * 0.00015, 0) for i in range(5) for j in range(20)]
    csv = lonlat_csv(tmp_path / "labels.csv", rows)
    puts: dict[int, dict[str, np.ndarray]] = {}
    real_put = BandMosaic.put

    def spy_put(self, i, per_zone):
        puts[i] = {zone: np.array(v, dtype=np.float32) for zone, v in per_zone.items()}
        return real_put(self, i, per_zone)

    monkeypatch.setattr(BandMosaic, "put", spy_put)
    res = df.run_deforest(
        write_aoi(tmp_path / "aoi.geojson", aoi),
        2018,
        2020,
        tmp_path / "out",
        labels=csv,
        reader=reader,
        block=(32, 32),
        min_area_ha=0.1,
    )
    plan = plan_aoi(reader, aoi)
    assert plan.zone_names == ["10N", "11N"] == res.summary["zones"]
    assert max(plan.zones, key=lambda z: z.n_pixels).zone == "10N"
    with rasterio.open(res.paths.tif) as ds:
        data, crs, transform = ds.read(), ds.crs, ds.transform
    assert crs.to_epsg() == 32610 and sorted(puts) == [0, 1, 2, 3]
    for i in range(4):
        whole, t, _ = export.mosaic_zones(
            [export.GridArray(puts[i][z.zone], z) for z in plan.zones], dst_zone="10N", nodata=np.nan
        )
        np.testing.assert_array_equal(data[i], whole)
        assert transform == t
    quiet = TesseraReader.from_store(mem)
    for z in plan.zones:
        expected = metrics.cosine_distance_map(read_plan(quiet, z, 2018), read_plan(quiet, z, 2020))
        np.testing.assert_allclose(puts[2][z.zone][z.mask], expected[z.mask], atol=1e-6)
        assert np.isnan(puts[2][z.zone][~z.mask]).all()
        conv = puts[3][z.zone] == 1
        assert (conv & centres_in(z, PATCH)).sum() > 10, z.zone
        assert not (conv & ~centres_in(z, PATCH.buffer(0.00015))).any(), z.zone
    assert res.summary["patches"]["count"] == 1
    assert res.summary["converted_pixels"] == sum(int((puts[3][z.zone] == 1).sum()) for z in plan.zones)


def test_labels_read_across_a_seam_are_in_the_provenance(tmp_path, monkeypatch):
    # The AOI is wholly in 10N; its Hansen label ring reaches 11N, whose 2018 and 2020 chunks are
    # read for the labels (year_a to train, year_b for the transfer check): the provenance must name
    # 11N and its pipeline commits.
    reader, _, _ = designed_store(seam_pattern)

    def hansen_layer(plan, layer, **kwargs):
        rows, cols = np.mgrid[plan.rows, plan.cols]
        x = plan.grid.left + (cols + 0.5) * plan.grid.pixel
        y = plan.grid.top - (rows + 0.5) * plan.grid.pixel
        lon, _ = plan.grid.to_lonlat().transform(x, y)
        values = {"treecover2000": np.where(lon >= STRIP, 90, 0), "lossyear": 0, "datamask": 1}[layer]
        return np.broadcast_to(values, plan.shape).astype(np.uint8)

    monkeypatch.setattr(validate, "hansen_layer", hansen_layer)
    aoi = box(LON0 - 0.0025, LAT0 - 0.0015, LON0 - 0.0004, LAT0 + 0.0015)
    res = df.run_deforest(
        write_aoi(tmp_path / "aoi.geojson", aoi),
        2018,
        2020,
        tmp_path / "out",
        reader=reader,
        label_ring_km=0.1,
        labels_per_class=100,
        block=(32, 32),
        min_area_ha=0.1,
    )
    assert res.report["aoi"]["zones"] == ["10N"] and res.summary["labels"]["ring"]["zones"] == ["10N", "11N"]
    labels = pd.read_parquet(res.extra_paths["labels"])
    assert (labels["zone"] == "11N").any() and (labels["zone"] == "10N").any()
    prov = res.report["provenance"]
    assert prov["zones"] == ["10N", "11N"]
    assert prov["zone_years_beyond_aoi"] == {"11N": [2018, 2020]}
    assert prov["pipeline_commits"]["11N"] == {"2018": "abc2018", "2020": "abc2020"}
    assert prov["pipeline_commits"]["10N"] == {"2018": "abc2018", "2020": "abc2020"}


def test_window_caveat_reports_measured_alignment_not_an_assumption():
    from tessera_apps.apps.deforest import _window_caveat

    breakdown = {"2019": {"pixels": 100, "converted_share": 0.014}, "2022": {"pixels": 100, "converted_share": 0.731}}
    text = _window_caveat(2019, 2022, [20, 21, 22], [19, 20, 21], breakdown, 0.657, 0.384)
    assert "1.4% of Hansen 2019 loss" in text and "73.1% of Hansen 2022 loss" in text
    assert "the headline window scores higher (IoU 0.657 against 0.384)" in text
    flipped = _window_caveat(2019, 2022, [20, 21, 22], [19, 20, 21], breakdown, 0.30, 0.50)
    assert "lag-aligned window (2019 to 2021) scores higher" in flipped
    assert "n/a" in _window_caveat(2019, 2022, [20, 21, 22], [19, 20, 21], {}, None, 0.5)
