"""Offline tests of the deforest app's parts, from the label rules to the validation window.

Pure parts run on synthetic arrays: the Hansen label classes, the label ring, cluster sampling, the
labels CSV, the classifier and its spatially held-out estimate, the radar regime guard, the
cross-check cut, the conversion rule, cleaning across a zone seam and the patch-size summary. The
parts that need a run (the validation window, the regime guard on labels, early failures) use the
caldor fixture store of ``tests/conftest.py`` with the Hansen tile pieces under it served from
local tiles (``url_template``); ``tests/test_apps_smoke.py`` runs the whole app.
"""

from __future__ import annotations

import json
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

from tessera_apps import metrics, validate
from tessera_apps.apps import deforest as df
from tessera_apps.apps._common import AppInputError
from tessera_apps.io import FIRST_YEAR, TesseraReader, Window, ZoneGrid, plan_aoi

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
    # Put year_a's own loss (2019) on some AOI pixels with none: it must not count.
    marked = inside & (lossyear == 0) & (np.arange(lossyear.shape[1])[None, :] % 3 == 0)
    assert marked.sum() > 20

    def edit(layer, data):
        return np.where(marked, 19, data) if layer == "lossyear" else data

    template = write_hansen_tiles(tmp_path / "hansen", caldor_hansen, edit=edit)
    res = df.run_deforest(
        path, 2019, 2022, tmp_path / "out", validate=True, reader=caldor_reader, hansen_url_template=template, **RUN_KW
    )
    (z,) = plan_aoi(caldor_reader, geometry).zones
    ly = validate.hansen_layer(z, "lossyear", url_template=template)
    tc = validate.hansen_layer(z, "treecover2000", url_template=template)
    window = (ly >= 20) & (ly <= 22) & z.mask
    only_b = (ly == 22) & z.mask
    assert ((ly == 19) & z.mask).sum() > 20 and ((ly > 22) & z.mask).sum() > 0
    assert window.sum() > only_b.sum() > 0
    val = json.loads(res.paths.validation.read_text())
    with rasterio.open(res.paths.tif) as ds:
        prob_a, converted = ds.read(1), ds.read(4)
    classified = np.isfinite(converted)
    assert classified[z.mask].all(), "every AOI pixel is classified on this window"
    conv = val["scores"]["conversion"]
    assert conv["tp"] + conv["fn"] == int(window.sum())
    assert conv["tp"] == int(((converted == 1) & window).sum()) and conv["tp"] + conv["fp"] == int(
        (converted == 1).sum()
    )
    only = val["scores"]["conversion_year_b_only"]
    assert only["tp"] + only["fn"] == int(only_b.sum()) and only["tp"] == int(((converted == 1) & only_b).sum())
    assert only["precision"] <= conv["precision"]
    proxy = (tc >= 50) & ~((ly >= 1) & (ly <= 19)) & z.mask
    agree = val["scores"]["forest_agreement_year_a"]
    forest = np.isfinite(prob_a) & (prob_a >= 0.5)
    assert agree["accuracy"] == pytest.approx(float(np.mean(forest[z.mask] == proxy[z.mask])))
    assert agree["recall_forest"] == pytest.approx(float(forest[proxy].mean()))
    # The per loss year breakdown: 2019 (year_a) is scored but outside the window.
    breakdown = val["scores"]["loss_year_breakdown"]
    year_a_loss = (ly == 19) & z.mask
    assert breakdown["2019"]["pixels"] == int(year_a_loss.sum())
    assert breakdown["2019"]["converted_share"] == pytest.approx(float((converted[year_a_loss] == 1).mean()))
    flips = (prob_a >= 0.5) & (prob_b_of(res) < 0.5)
    assert breakdown["2021"]["flip_share"] == pytest.approx(float(flips[(ly == 21) & z.mask].mean()))
    assert sum(v["pixels"] for v in breakdown.values()) == conv["n"]
    assert res.summary["validation"]["loss_year_breakdown"] == breakdown
    assert any("annual embedding" in c for c in val["caveats"])
    assert val["reference"]["loss_window"]["lossyear_codes"] == [20, 21, 22]
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


def test_an_empty_label_ring_stops_before_any_chunk_read(
    caldor_store, caldor_reader, caldor_hansen, window_aoi, tmp_path
):
    _, log = caldor_store
    path, _ = window_aoi
    water = write_hansen_tiles(
        tmp_path / "water", caldor_hansen, edit=lambda layer, d: np.full_like(d, 2) if layer == "datamask" else d
    )
    with pytest.raises(df.DeforestError, match="holds 0 forest and 0 non-forest"):
        df.run_deforest(path, 2019, 2022, tmp_path / "out", reader=caldor_reader, hansen_url_template=water, **RUN_KW)
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
