"""Offline tests of the restore app, from its pure parts to whole runs on designed stores.

Pure parts run on hand-made arrays: the ring in metres, the held-out control threshold (checked
against the wetland app's held-out scores), the placebos and their band, the verdict rule, the null
halves, the target progress, the per-pixel fractions, the trend and the canopy check. Runs use the
designed 10N stores of ``tests/conftest.py`` (:func:`~tests.conftest.restore_store`), whose site moves
from pasture to forest from 2020 (or stays pasture), next to a ring of pasture with a crop chunk and
a forest target stand, with known gaps in data and radar. The control is matched on 2018, the year
before the 2019 baseline, so stage 1 reads 2018 and stage 2 reads 2019 to 2025. One run uses a 1024
x 1024 grid with 256 pixel chunks and the default 1 to 3 km ring. The GLAD canopy height loader
reads local tiles written by :func:`~tests.conftest.write_tch_tiles`. Designed stores for the
review's failure cases (a ring mixing pasture with a near-pasture land type, a control half of whose
ring moves like the site) sit next to the tests that use them.
"""

from __future__ import annotations

import json
import math
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.errors
import rasterio.transform
import shapely
from pyproj import Transformer
from scipy.stats import theilslopes
from shapely.geometry import LineString, Point, box

from tessera_apps import demo, export, io, validate
from tessera_apps.apps import restore as rs
from tessera_apps.apps import wetland as wl
from tessera_apps.apps._common import AppInputError, StrictValidationError, load_aoi
from tessera_apps.io import FIRST_YEAR, N_BANDS, TesseraReader, plan_aoi

from .conftest import (
    CLASS_PASTURE,
    CLASS_SITE,
    CROP_BOX,
    PASTURE,
    RESTORE_EPSG,
    RESTORE_PLANTED,
    SITE_BOX,
    SMALL_RING,
    TARGET_BOX,
    KeyLog,
    designed_store,
    grid_polygon,
    in_box,
    restore_corner,
    restore_pattern,
    restore_store,
    write_polygon,
    write_tch_tiles,
    write_worldcover_tiles,
)

YEARS = list(range(RESTORE_PLANTED - 1, 2026))
#: The year the control is matched on: the one before the baseline.
MATCH = RESTORE_PLANTED - 2
KW = {**SMALL_RING, "block": (64, 64)}
#: A site pixel and four ring pixels with no data in 2021 (the panel rule drops them from every year).
HOLES = ((2021, ((90, 91), (90, 91))), (2021, ((50, 52), (60, 62))))
#: A 4 x 4 site block and a 4 x 4 ring block without Sentinel-1 in 2022 (another radar pattern).
RADAR_FREE = ((2022, ((100, 104), (100, 104))), (2022, ((140, 144), (60, 64))))
SWITCH = (2022, 2023, 2024)


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    """Reference cache under tmp_path; no network: canopy tiles from local files only."""
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))

    def blocked(request, timeout):
        raise AssertionError(f"unexpected network access: {request.full_url}")

    monkeypatch.setattr(validate, "_urlopen", blocked)


@pytest.fixture(scope="module")
def working_mem():
    return restore_store(holes=HOLES, radar_free=RADAR_FREE)[2]


@pytest.fixture(scope="module")
def null_mem():
    return restore_store(moves=False)[2]


@pytest.fixture(scope="module")
def switch_mem():
    return restore_store(switch_years=SWITCH)[2]


def _reader(mem):
    log = KeyLog(mem)
    return TesseraReader.from_store(log), log


@pytest.fixture
def working(working_mem):
    return _reader(working_mem)


@pytest.fixture
def files(tmp_path):
    left, top = restore_corner()
    site = write_polygon(tmp_path / "site.geojson", grid_polygon(left, top, SITE_BOX), {"name": "site"})
    target = write_polygon(tmp_path / "target.geojson", grid_polygon(left, top, TARGET_BOX), {"name": "target"})
    return site, target


def run(reader, files, tmp_path, out="out", **kw):
    site, target = files
    args = {**KW, "planted": RESTORE_PLANTED, "target": target, "reader": reader, **kw}
    return rs.run_restore(site, tmp_path / out, **args)


def trajectory(result) -> pd.DataFrame:
    return pd.DataFrame(result.summary["trajectory"]).set_index("year")


def tif_bands(path: Path) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """The bands of the output COG by name, and the grid (row, col) of its top-left pixel on the scenario grid."""
    left, top = restore_corner()
    with rasterio.open(path) as ds:
        bands = {name: ds.read(i + 1) for i, name in enumerate(ds.descriptions)}
        t = ds.transform
    return bands, (round((top - t.f) / 10.0), round((t.c - left) / 10.0))


def fetched(log: KeyLog, zone: str = "10N", layer: str = "embeddings") -> set[tuple[int, int, int]]:
    """(year, chunk row, chunk col) of every chunk read, on a store with one chunk per shard."""
    prefix = f"{zone}/{layer}/c/"
    out = set()
    for key in log.log:
        if key.startswith(prefix):
            t, r, c = (int(v) for v in key[len(prefix) :].split("/")[:3])
            out.add((FIRST_YEAR + t, r, c))
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# --------------------------------------------------------------------------------------------
# Years, the ring and the polygons
# --------------------------------------------------------------------------------------------


def test_trajectory_years_defaults_and_planting_limits():
    assert rs.trajectory_years(2020, None, None) == (2019, 2025, list(range(2019, 2026)))
    assert rs.trajectory_years(2019, 2017, 2023) == (2017, 2023, list(range(2017, 2024)))
    assert rs.trajectory_years(2018, None, None)[0] == 2017
    for planted, a, b, match in (
        (2017, None, None, "no pre-planting year"),
        (2016, None, None, "no pre-planting year"),
        (2026, None, None, "after the store's last year"),
        (2020, 2020, None, "must be before the planting year"),
        (2020, 2016, None, "before the store's first year"),
        (2020, None, 2019, "planting year .2020. or later"),
        (2020, None, 2026, "after the store's last year"),
    ):
        with pytest.raises(AppInputError, match=match):
            rs.trajectory_years(planted, a, b)


def _metres(geometry, epsg=RESTORE_EPSG):
    tr = Transformer.from_crs(4326, epsg, always_xy=True)
    return shapely.transform(geometry, tr.transform, interleaved=False)


@pytest.mark.parametrize(("inner", "outer"), [(1.0, 3.0), (0.25, 0.75)])
def test_ring_is_inner_to_outer_km_from_the_site_in_metres(inner, outer):
    left, top = restore_corner()
    site = grid_polygon(left, top, SITE_BOX)
    ring = rs.ring_geometry(site, inner, outer)
    site_m, ring_m = _metres(site), _metres(ring)
    # The buffer is a polygon of 16 segments a quarter circle, inside the circle by r (1 - cos(pi / 64)).
    sagitta = inner * 1000.0 * (1.0 - math.cos(math.pi / 64))
    assert inner * 1000.0 - sagitta - 1.0 <= ring_m.distance(site_m) <= inner * 1000.0 + 1.0
    far = max(site_m.distance(Point(xy)) for xy in ring_m.exterior.coords)
    assert far == pytest.approx(outer * 1000.0, abs=1.0)
    assert not ring_m.intersects(site_m.buffer(inner * 1000.0 - 2.0))
    expected = site_m.buffer(outer * 1000.0, quad_segs=16).area - site_m.buffer(inner * 1000.0, quad_segs=16).area
    assert ring_m.area == pytest.approx(expected, rel=1e-3)


def test_a_ring_from_zero_km_starts_at_the_site_edge():
    left, top = restore_corner()
    site = grid_polygon(left, top, SITE_BOX)
    ring = rs.ring_geometry(site, 0.0, 0.5)
    assert _metres(ring).intersection(_metres(site)).area < 1.0
    assert _metres(ring).distance(_metres(site)) < 1.0
    with pytest.raises(ValueError, match="inner"):
        rs.ring_geometry(site, 0.5, 0.5)


def test_main_zone_is_the_zone_with_most_of_the_site():
    west_heavy = box(-120.05, 38.0, -119.99, 38.05)
    assert rs.main_zone(west_heavy) == "10N"
    assert rs.main_zone(box(-120.01, 38.0, -119.9, 38.05)) == "11N"


def test_polygon_files_with_points_or_lines_are_refused(tmp_path):
    left, top = restore_corner()
    square = grid_polygon(left, top, SITE_BOX)
    mixed = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": shapely.geometry.mapping(square)},
            {"type": "Feature", "properties": {}, "geometry": shapely.geometry.mapping(Point(-121.0, 38.5))},
        ],
    }
    path = tmp_path / "mixed.geojson"
    path.write_text(json.dumps(mixed), encoding="utf-8")
    with pytest.raises(AppInputError, match="Point features; the site must be polygons only"):
        rs.load_polygon(path, "site")
    line = write_polygon(tmp_path / "line.geojson", LineString([(-121.0, 38.5), (-121.01, 38.51)]))
    with pytest.raises(AppInputError, match="LineString features; the target must be polygons only"):
        rs.load_polygon(line, "target")
    with pytest.raises(AppInputError, match="neither an existing file nor a bundled demo"):
        rs.load_polygon(tmp_path / "missing.geojson", "site")
    with pytest.raises(AppInputError, match="needs a bundled demo site"):
        rs.load_polygon("demo", "control")
    loaded = rs.load_polygon(write_polygon(tmp_path / "ok.geojson", square), "control")
    assert loaded.geometry.equals_exact(square, 1e-9) and len(loaded.sha256) == 64


# --------------------------------------------------------------------------------------------
# Pure parts: the control threshold, placebos, the band and the verdict
# --------------------------------------------------------------------------------------------


def _vectors(n: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.integers(-40, 41, size=N_BANDS)
    emb = np.clip(base + rng.normal(0, 8, size=(n, N_BANDS)), -127, 127).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, size=n).astype(np.float32)
    return emb, scale


def test_pooled_sum_is_the_sum_of_dequantized_vectors():
    emb, scale = _vectors(70_000)
    expected = (emb.astype(np.float64) * scale[:, None].astype(np.float64)).sum(axis=0)
    np.testing.assert_allclose(rs.pooled_sum(emb, scale), expected, rtol=1e-12)
    assert rs.pooled_sum(emb[:0], scale[:0]).tolist() == [0.0] * N_BANDS


def test_heldout_similarity_matches_the_wetland_heldout_scores():
    emb, scale = _vectors(300)
    folds = np.repeat(np.arange(4), 75)
    deq = emb.astype(np.float64) * scale[:, None]
    held = rs.heldout_similarity(emb, scale, folds)
    np.testing.assert_allclose(held, 1.0 - wl.heldout_scores(deq, folds, "cosine"), atol=1e-6)
    # Each fold is scored against the others only: its own pixels do not pull the mean towards it.
    other = deq[folds != 0].sum(axis=0)
    np.testing.assert_allclose(held[folds == 0], [cos(v, other) for v in deq[folds == 0]], atol=1e-6)
    assert np.isnan(rs.heldout_similarity(emb, scale, np.zeros(300, dtype=int))).all()


def test_control_threshold_is_the_heldout_percentile_with_an_in_sample_fallback():
    rng = np.random.default_rng(0)
    in_sample = rng.uniform(0.90, 0.99, size=500)
    held = in_sample - rng.uniform(0.0, 0.02, size=500)
    choice = rs.control_threshold(in_sample, held, 5.0, 4)
    assert choice.method == "heldout" and choice.folds == 4
    assert 1.0 - choice.threshold == pytest.approx(np.percentile(held, 5.0), abs=1e-12)
    assert 1.0 - choice.in_sample_percentile == pytest.approx(np.percentile(in_sample, 5.0), abs=1e-12)
    fallback = rs.control_threshold(in_sample, np.full(500, np.nan), 10.0, 1)
    assert fallback.method == "in_sample"
    assert 1.0 - fallback.threshold == pytest.approx(np.percentile(in_sample, 10.0), abs=1e-12)


def test_draw_placebos_takes_the_nearest_control_pixels_and_is_seeded():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 1000, size=(400, 2))
    plc = rs.draw_placebos(xy, 30, 12, seed=7)
    members, centres = plc.members, plc.centres
    assert members.shape == (30, 12) and centres.shape == (30,) and (plc.sizes == 12).all()
    for row, c in zip(members, centres, strict=True):
        d = np.hypot(*(xy - xy[c]).T)
        assert set(row) == set(np.argsort(d)[:12]) and row[0] == c
    # Distinct centres while the pool has enough of them.
    assert np.unique(centres).size == 30
    again = rs.draw_placebos(xy, 30, 12, seed=7)
    other = rs.draw_placebos(xy, 30, 12, seed=8)
    np.testing.assert_array_equal(members, again.members)
    assert not np.array_equal(members, other.members)
    with pytest.raises(ValueError, match="need 1 to 400"):
        rs.draw_placebos(xy, 3, 401, seed=0)
    # More placebos than pixels: every pixel is a centre once before any repeats.
    few = rs.draw_placebos(xy[:25], 40, 5, seed=1)
    assert np.unique(few.centres).size == 25 and few.centres.size == 40


def test_placebos_keep_the_site_footprint_in_a_sparse_pool():
    """Review S3: in a thinned pool the k nearest pixels spread far wider than the site; a disk of its area does not."""
    rows, cols = np.mgrid[0:120, 0:120]
    keep = np.random.default_rng(0).random((60, 60))[rows // 2, cols // 2] < 0.25  # 75% thinned in 2 x 2 blocks
    xy = np.stack([cols[keep] * 10.0 + 5.0, -(rows[keep] * 10.0 + 5.0)], axis=1)
    site_px = 400
    radius = math.sqrt(site_px * 100.0 / math.pi)
    plc = rs.draw_placebos(xy, 50, site_px, seed=3, radius_m=radius, min_pixels=25)
    for members, centre, size, reach in zip(plc.members, plc.centres, plc.sizes, plc.reach_m, strict=True):
        inside = members[members >= 0]
        assert inside.size == size >= 25 and members[0] == centre
        d = np.hypot(*(xy[inside] - xy[centre]).T)
        assert d.max() <= radius and reach == pytest.approx(d.max())
        # Every pool pixel within the radius belongs to the placebo (none is passed over for a farther one).
        assert int((np.hypot(*(xy - xy[centre]).T) < radius).sum()) == size
    assert np.median(plc.sizes) < 0.5 * site_px
    # Without the radius the placebo reaches about twice the site's radius.
    plain = rs.draw_placebos(xy, 50, site_px, seed=3)
    assert np.median(plain.reach_m) > 1.7 * radius
    with pytest.raises(ValueError, match="too sparse"):
        rs.draw_placebos(xy, 5, site_px, seed=3, radius_m=15.0, min_pixels=25)


def test_placebo_divergence_ignores_padding():
    rng = np.random.default_rng(11)
    emb = rng.integers(-60, 61, size=(50, 2, N_BANDS)).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, size=(50, 2)).astype(np.float32)
    padded = rs.placebo_divergence(emb, scale, np.array([[0, 1, 2, -1, -1], [3, 4, 5, 6, 7]]))
    np.testing.assert_allclose(padded[0], rs.placebo_divergence(emb, scale, np.array([[0, 1, 2]]))[0], atol=1e-12)
    np.testing.assert_allclose(padded[1], rs.placebo_divergence(emb, scale, np.array([[3, 4, 5, 6, 7]]))[0])


def test_placebo_divergence_is_against_the_pool_less_the_placebo():
    rng = np.random.default_rng(11)
    n, years = 90, 3
    emb = rng.integers(-60, 61, size=(n, years, N_BANDS)).astype(np.int8)
    scale = rng.uniform(0.005, 0.02, size=(n, years)).astype(np.float32)
    members = np.array([[0, 1, 2, 3], [10, 20, 30, 40], [5, 6, 7, 89]])
    got = rs.placebo_divergence(emb, scale, members)
    assert got.shape == (3, years)
    for p, idx in enumerate(members):
        rest = np.setdiff1d(np.arange(n), idx)
        for t in range(years):
            deq = emb[:, t].astype(np.float64) * scale[:, t, None]
            expected = 1.0 - cos(deq[idx].mean(axis=0), deq[rest].mean(axis=0))
            assert got[p, t] == pytest.approx(expected, abs=1e-12)


def test_placebo_band_is_the_5_50_95_percentiles_per_year():
    rng = np.random.default_rng(1)
    change = rng.normal(size=(200, 5))
    band = rs.placebo_band(change)
    np.testing.assert_allclose(band, np.percentile(change, [5, 50, 95], axis=0))


def test_trailing_runs():
    sig = np.array([[1, 1, 0, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 1]], dtype=bool)
    assert rs.trailing_runs(sig).tolist() == [2, 0, 5, 1]
    assert int(rs.trailing_runs(np.array([True, False, True, True, True]))) == 3


VERDICT_YEARS = list(range(2019, 2026))
P95 = np.full(7, 0.01)


def _verdict(change, planted=2020, min_post=3, years=VERDICT_YEARS, p95=P95):
    return rs.call_verdict(years, np.asarray(change, dtype=float), p95, planted=planted, min_post_years=min_post)


def test_the_verdict_rule_on_hand_made_trajectories():
    steady = _verdict([0, 0.02, 0.03, 0.05, 0.08, 0.1, 0.12])
    assert steady.verdict == "diverging from control since 2020" and steady.since == 2020
    assert steady.post_years == list(range(2020, 2026)) and steady.significant_years == steady.post_years
    late = _verdict([0, 0, 0.005, 0.02, 0.005, 0.03, 0.04])
    assert late.verdict == "diverging from control since 2024" and late.since == 2024
    assert late.significant_years == [2022, 2024, 2025]
    two = _verdict([0, 0, 0, 0, 0, 0.02, 0.02])
    assert two.verdict == "diverging from control since 2024"
    single = _verdict([0, 0, 0, 0, 0, 0, 0.5])
    assert single.verdict == "no divergence detected" and single.since is None
    assert single.detail == "divergence in 2025 only; one year is not enough to call it."
    dip = _verdict([0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.0])
    assert dip.verdict == "no divergence detected" and "2020, 2021, 2022, 2023, 2024" in dip.detail
    flat = _verdict([0, 0.001, -0.002, 0.0, 0.005, 0.009, 0.01])
    assert flat.verdict == "no divergence detected" and flat.significant_years == []
    assert "within the placebo band" in flat.detail
    # One-sided: a divergence that falls far below the band is not significant (review T1).
    falling = _verdict([0, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05])
    assert falling.verdict == "no divergence detected" and falling.significant_years == []
    # A band below zero (placebos smaller than the site, a year with less spread): a falling divergence
    # above the band is still not significant; it must also be above zero (review S2).
    low = np.full(7, -0.03)
    assert _verdict([0, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01], p95=low).significant_years == []
    assert _verdict([0, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], p95=low).verdict == (
        "diverging from control since 2020"
    )
    # A single trailing year after earlier significant ones names them all (review S5).
    gap = _verdict([0, 0.05, 0.05, 0.05, 0.05, 0.0, 0.05])
    assert gap.verdict == "no divergence detected" and gap.significant_years == [2020, 2021, 2022, 2023, 2025]
    assert "2020, 2021, 2022, 2023, 2025" in gap.detail and "only" not in gap.detail
    assert "ending in 2025 is that one year" in gap.detail
    # Only post-planting years count: divergence that started before planting is called from planting on.
    early = _verdict([0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], planted=2022)
    assert early.verdict == "diverging from control since 2022" and early.post_years == [2022, 2023, 2024, 2025]
    assert early.significant_years == [2022, 2023, 2024, 2025]
    # Too few post-planting years: insufficient, however strong the change.
    short = _verdict([0, 0.5, 0.6, 0.7], years=[2021, 2022, 2023, 2024], p95=np.full(4, 0.01), planted=2023)
    assert short.verdict == "insufficient years" and short.since is None and "needs at least 3" in short.detail
    assert "or lower --min-post-years" in short.detail
    # One post-planting year: lowering --min-post-years (floor 2) cannot help, so the text does not suggest it.
    one = _verdict([0, 0.5], years=[2024, 2025], p95=np.full(2, 0.01), planted=2025)
    assert one.verdict == "insufficient years" and "1 post-planting year(s) (2025 to 2025)" in one.detail
    assert "lower --min-post-years" not in one.detail and "cannot go below 2" in one.detail
    two_post = _verdict(
        [0, 0.5, 0.6, 0.7], years=[2021, 2022, 2023, 2024], p95=np.full(4, 0.01), planted=2023, min_post=2
    )
    assert two_post.verdict == "diverging from control since 2023"
    # Equal to the 95th percentile is not above it.
    assert _verdict([0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).verdict == "no divergence detected"


def test_placebo_verdicts_and_the_false_positive_rate():
    change = np.array(
        [
            [0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.02],
            [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02],
            [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05],
            [0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01],
        ]
    )
    texts, diverging = rs.placebo_verdicts(VERDICT_YEARS, change, P95, planted=2020, min_post_years=3)
    assert texts == [
        "diverging from control since 2020",
        "diverging from control since 2024",
        "no divergence detected",
        "no divergence detected",
        "no divergence detected",
        "no divergence detected",
    ]
    assert diverging.tolist() == [True, True, False, False, False, False]
    assert rs.false_positive_rate(diverging[:4], False) == 0.5
    # The same zero floor as the site's rule, below a band under zero.
    _, low = rs.placebo_verdicts(VERDICT_YEARS, change[4:5] * 0.2, np.full(7, -0.03), planted=2020, min_post_years=3)
    assert not low.any()
    assert rs.false_positive_rate(diverging, True) is None
    texts, diverging = rs.placebo_verdicts(VERDICT_YEARS, change, P95, planted=2024, min_post_years=3)
    assert texts == ["insufficient years"] * 6 and not diverging.any()


def test_null_halves_alternate_angular_sectors():
    angles = np.deg2rad(np.arange(8) * 45 + 10)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 100.0 + [500.0, 200.0]
    assert rs.null_halves(xy, (500.0, 200.0)).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert rs.null_halves(xy, (500.0, 200.0), sectors=4).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_axis_halves_split_across_the_long_axis():
    # A 30 x 10 block of pixel centres, long along x.
    xs, ys = np.meshgrid(np.arange(30) * 10.0, np.arange(10) * 10.0)
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1)
    for split in ("median", "centroid"):
        halves = rs.axis_halves(xy, split=split)
        assert (halves == 0).sum() == (halves == 1).sum() == 150
        # Contiguous: one half is the west end, the other the east end.
        west, east = xy[halves == halves[0], 0], xy[halves != halves[0], 0]
        assert west.max() < east.min()
    # The median split balances an uneven shape; the centroid split follows the area.
    lopsided = np.concatenate([xy, np.stack([np.full(60, 400.0), np.arange(60) * 1.0], axis=1)])
    med = rs.axis_halves(lopsided, split="median")
    assert abs(int((med == 0).sum()) - int((med == 1).sum())) <= 1
    # A supplied control off to one side of a site falls in one sector, but splits by its axis (review S4).
    angles = np.mod(np.arctan2(xy[:, 1] + 50.0, xy[:, 0] + 1000.0), 2 * np.pi)
    assert np.unique(np.floor(angles / (np.pi / 4))).size == 1
    assert set(rs.null_halves(xy, (-1000.0, -50.0)).tolist()) == {0}
    with pytest.raises(ValueError, match="split"):
        rs.axis_halves(xy, split="sectors")
    assert rs.axis_halves(xy[:1]).tolist() == [0]


def test_target_progress_is_unclipped():
    progress = rs.target_progress([0.1, 0.5, 0.95, 1.0], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.1])
    assert progress[:3].tolist() == pytest.approx([0.0, 0.5, 1.0625])
    assert np.isnan(progress[3])
    assert rs.target_progress([0.0], [0.2], [0.6])[0] == pytest.approx(-0.5)


def test_departed_and_closer_fractions():
    control = np.linspace(0.8, 1.0, 101)
    p5 = np.percentile(control, 5)
    site = np.array([p5 - 0.01, p5 + 0.01, 0.9, np.nan])
    assert rs.departed_fraction(site, control) == pytest.approx(1 / 3)
    assert math.isnan(rs.departed_fraction(np.array([np.nan]), control))
    assert rs.closer_fraction(np.array([0.9, 0.5, 0.7, np.nan]), np.array([0.8, 0.6, 0.7, 0.1])) == pytest.approx(1 / 3)


def test_theil_sen_trend_and_spearman():
    t = rs.theil_sen_trend([2020, 2021, 2022, 2023], np.array([0.0, 0.1, 0.2, 0.31]))
    assert t["theil_sen_slope"] == pytest.approx(0.1, abs=0.006) and t["spearman_rho"] == pytest.approx(1.0)
    assert t["n_years"] == 4
    flat = rs.theil_sen_trend([2020, 2021, 2022], np.zeros(3))
    assert flat["theil_sen_slope"] == 0.0 and flat["spearman_rho"] is None
    assert rs.theil_sen_trend([2025], np.array([0.2]))["theil_sen_slope"] is None


def test_canopy_check_on_synthetic_heights():
    years = [2019, 2020, 2021, 2022, 2023]
    change = np.array([0.0, 0.01, 0.05, 0.1, 0.2])
    rng = np.random.default_rng(0)

    def heights(tall_share: float, n: int = 1000, tall: float = 12.0) -> np.ndarray:
        h = np.zeros(n, dtype=np.float32)
        h[: round(tall_share * n)] = tall
        return rng.permutation(h)

    site_share = {2019: 0.0, 2020: 0.0, 2021: 0.1, 2022: 0.26, 2023: 0.6}
    site = {y: heights(s) for y, s in site_share.items()}
    site[2023][:50] = np.nan  # not covered: left out of the means
    control = {y: heights(0.05) for y in years}
    check = rs.canopy_check(years, change, {"site": site, "control": control})
    assert check["years"] == years and check["canopy_onset_year"] == 2022
    row = check["per_year"]["2022"]
    assert row["site_tall_share"] == pytest.approx(0.26) and row["control_tall_share"] == pytest.approx(0.05)
    assert row["site_mean_m"] == pytest.approx(0.26 * 12) and row["gap_mean_m"] == pytest.approx((0.26 - 0.05) * 12)
    assert check["per_year"]["2023"]["site_pixels_covered"] == 950
    assert check["per_year"]["2019"]["gap_change_mean_m"] == 0.0
    gap = np.array([check["per_year"][str(y)]["gap_change_mean_m"] for y in years])
    assert check["pearson_divergence_vs_mean_gap_change"] == pytest.approx(np.corrcoef(change, gap)[0, 1])
    from scipy.stats import spearmanr

    # 2019 and 2020 tie (no tall site pixel either year), so rho is just below 1.
    assert check["spearman_divergence_vs_mean_gap_change"] == pytest.approx(spearmanr(change, gap)[0])
    assert 0.95 < check["spearman_divergence_vs_mean_gap_change"] < 1.0
    flat = rs.canopy_check(years, change, {"site": control, "control": control})
    assert flat["canopy_onset_year"] is None and flat["pearson_divergence_vs_mean_gap_change"] is None
    # The onset is the site's lead over the control, not the site's own share: a tall control delays it.
    tall_control = {y: heights(0.3) for y in years}
    lead = rs.canopy_check(years, change, {"site": site, "control": tall_control})
    assert lead["canopy_onset_year"] == 2023 and lead["per_year"]["2022"]["gap_tall_share"] == pytest.approx(-0.04)
    # The onset year and the sign come first (lead decision: the few years say little more).
    assert list(check)[:3] == ["canopy_onset_year", "correlation_sign", "correlation_years"]
    assert check["correlation_sign"] == "positive" and check["correlation_years"] == 5
    assert flat["correlation_sign"] is None
    falling = rs.canopy_check(years, -change, {"site": site, "control": control})
    assert falling["correlation_sign"] == "negative"


def test_canopy_boundaries_5m_and_a_lead_of_exactly_0_2():
    site = {2020: np.zeros(1000, np.float32), 2021: np.r_[np.full(250, 5.0), np.zeros(750)].astype(np.float32)}
    control = {2020: np.zeros(1000, np.float32), 2021: np.r_[np.full(50, 5.0), np.zeros(950)].astype(np.float32)}
    check = rs.canopy_check([2020, 2021], np.array([0.0, 0.1]), {"site": site, "control": control})
    assert check["per_year"]["2021"]["site_tall_share"] == pytest.approx(0.25) and check["tall_m"] == 5
    assert check["canopy_onset_year"] == 2021


def test_majority_pattern_breaks_ties_towards_radar():
    radar = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
    assert rs.majority_pattern(radar).tolist() == [True, True, True]
    assert rs.majority_pattern(radar[:2]).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        rs.majority_pattern(radar[:0])


def test_valid_years_refuses_a_zero_vector_with_a_finite_scale():
    emb = np.ones((3, 2, N_BANDS), dtype=np.int8)
    emb[0, 1] = 0
    scale = np.full((3, 2), 0.01, dtype=np.float32)
    scale[2, 0] = np.nan
    pix = rs._Pixels(
        role=np.ones(3, np.int8),
        zone=np.zeros(3, np.int16),
        row=np.zeros(3, np.int64),
        col=np.zeros(3, np.int64),
        emb=emb,
        scale=scale,
        radar=np.ones((3, 2), bool),
    )
    assert rs._valid_years(pix).tolist() == [[True, False], [True, True], [False, True]]


def test_chunk_runs_join_neighbouring_chunks_and_never_add_an_empty_one():
    rows = np.array([5, 10, 70, 200, 70, 130])
    cols = np.array([5, 70, 10, 10, 250, 130])
    zones = np.zeros(6, dtype=np.int16)
    shapes = {0: (64, 64)}
    assert rs.chunk_runs(zones, rows, cols, shapes, 4) == [
        (0, 0, 0, 1),
        (0, 1, 0, 0),
        (0, 1, 3, 3),
        (0, 2, 2, 2),
        (0, 3, 0, 0),
    ]
    assert rs.chunk_runs(zones, rows, cols, shapes, 1)[:2] == [(0, 0, 0, 0), (0, 0, 1, 1)]
    two = rs.chunk_runs(np.array([0, 1]), np.array([5, 5]), np.array([5, 70]), {0: (64, 64), 1: (64, 64)}, 4)
    assert two == [(0, 0, 0, 0), (1, 0, 1, 1)]


# --------------------------------------------------------------------------------------------
# GLAD tree canopy height
# --------------------------------------------------------------------------------------------


def _site_plan(reader):
    left, top = restore_corner()
    (zp,) = plan_aoi(reader, grid_polygon(left, top, ((60, 130), (60, 130)))).zones
    return zp


def _tch_heights(value_by_year: dict[int, int], size: int = 192) -> dict[int, np.ndarray]:
    rows, cols = np.mgrid[0:size, 0:size]
    out = {}
    for year, v in value_by_year.items():
        h = np.where(cols < 96, 15, 0).astype(np.uint8)
        h[(rows >= 80) & (rows < 120)] = v
        out[year] = h
    return out


def test_glad_tch_reads_local_tiles_onto_the_plan_grid(working, tmp_path):
    reader, _ = working
    zp = _site_plan(reader)
    left, top = restore_corner()
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, _tch_heights({2020: 7, 2021: 9}))
    assert template.endswith("{year}/{tile}.tif")
    h = validate.glad_tch(zp, 2020, url_template=template)
    assert h.dtype == np.float32 and h.shape == zp.shape
    assert np.isnan(h[~zp.mask]).all() and np.isfinite(h[zp.mask]).all()
    # The 30 m layer is resampled by nearest: away from the edges of the designed patches it is exact.
    rows = np.arange(zp.rows.start, zp.rows.stop)[:, None] + np.zeros(zp.shape[1], dtype=int)
    cols = np.arange(zp.cols.start, zp.cols.stop)[None, :] + np.zeros((zp.shape[0], 1), dtype=int)
    interior = zp.mask & (np.abs(cols - 96) > 3) & ((rows < 77) | (rows > 82)) & ((rows < 117) | (rows > 122))
    expected = np.where((rows >= 80) & (rows < 120), 7, np.where(cols < 96, 15, 0))
    np.testing.assert_array_equal(h[interior], expected[interior])
    assert np.nanmax(validate.glad_tch(zp, 2021, url_template=template)) == 15
    # Cached: the windows are read again without the tiles.
    for path in (tmp_path / "tch").rglob("*.tif"):
        path.unlink()
    np.testing.assert_array_equal(validate.glad_tch(zp, 2020, url_template=template), h)
    with pytest.raises(validate.ReferenceDataError, match="No GLAD tree canopy height tile for 2022"):
        validate.glad_tch(zp, 2022, url_template=template)
    with pytest.raises(ValueError, match="2015 to 2024"):
        validate.glad_tch(zp, 2025, url_template=template)


def test_glad_tch_pixels_no_tile_covers_are_nan_not_zero(working, tmp_path):
    reader, _ = working
    zp = _site_plan(reader)
    left, top = restore_corner()
    # Tile pieces over the west half of the grid only (plus a margin of 8 layer pixels, about 22 m each).
    west = {2020: np.full((192, 96), 12, dtype=np.uint8)}
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, west)
    h = validate.glad_tch(zp, 2020, url_template=template)
    cols = np.arange(zp.cols.start, zp.cols.stop)[None, :] + np.zeros((zp.shape[0], 1), dtype=int)
    covered = zp.mask & (cols < 90)  # away from the edge, where the piece's own margin pixels hold 0
    beyond = zp.mask & (cols > 96 + 30)
    assert (h[covered] == 12).all()
    assert beyond.any() and np.isnan(h[beyond]).all()


def test_glad_tch_missing_tiles_are_not_cached(working, tmp_path):
    reader, _ = working
    zp = _site_plan(reader)
    left, top = restore_corner()
    template = str(tmp_path / "later" / "{year}" / "{tile}.tif")
    bbox = validate.plan_lonlat_bounds(zp)
    assert validate.glad_tch_reachable(bbox, 2020, url_template=template) == 0
    with pytest.raises(validate.ReferenceDataError):
        validate.glad_tch(zp, 2020, url_template=template)
    write_tch_tiles(tmp_path / "later", RESTORE_EPSG, left, top, _tch_heights({2020: 7}))
    assert validate.glad_tch_reachable(bbox, 2020, url_template=template) == 1
    assert np.nanmax(validate.glad_tch(zp, 2020, url_template=template)) == 15


def test_glad_tch_reads_a_missing_remote_tile_as_not_covered(working, monkeypatch):
    reader, _ = working
    zp = _site_plan(reader)
    heads = []

    def not_found(request, timeout):
        heads.append((request.get_method(), request.full_url))
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(validate, "_urlopen", not_found)
    template = "https://tiles.invalid/TCH/{year}/{tile}.tif"
    with pytest.raises(validate.ReferenceDataError, match="No GLAD tree canopy height tile for 2020"):
        validate.glad_tch(zp, 2020, url_template=template)
    assert heads == [("HEAD", "https://tiles.invalid/TCH/2020/40N_130W.tif")]
    assert validate.glad_tch_reachable(validate.plan_lonlat_bounds(zp), 2020, url_template=template) == 0


def test_glad_tch_unreachable_host_names_it(working, monkeypatch):
    reader, _ = working
    zp = _site_plan(reader)

    def refused(request, timeout):
        raise urllib.error.URLError("blocked by the proxy")

    monkeypatch.setattr(validate, "_urlopen", refused)
    with pytest.raises(validate.ReferenceUnavailableError, match=r"storage\.googleapis\.com"):
        validate.glad_tch(zp, 2020)
    with pytest.raises(validate.ReferenceUnavailableError):
        validate.glad_tch_reachable(validate.plan_lonlat_bounds(zp), 2020)
    assert validate.GLAD_TCH_URL_TEMPLATE.format(year=2020, tile="40N_130W").endswith(
        "Annual_Layers_2015_2024/TCH/2020/40N_130W.tif"
    )


# --------------------------------------------------------------------------------------------
# Runs on the designed stores
# --------------------------------------------------------------------------------------------


def test_a_working_site_diverges_from_its_matched_control(working, working_mem, files, tmp_path):
    reader, log = working
    result = run(reader, files, tmp_path)
    s = result.summary
    assert s["verdict"] == "diverging from control since 2020" and s["diverging_since"] == 2020
    assert s["years"] == YEARS and s["post_planting_years"] == YEARS[1:] and s["planted"] == 2020
    assert s["match_year"] == MATCH == s["control"]["match_year"]
    traj = trajectory(result)
    assert traj.index.tolist() == YEARS
    assert traj["significant"].tolist() == [False] + [True] * 6
    assert (traj["divergence_change"].iloc[1:] > traj["placebo_p95"].iloc[1:]).all()
    assert traj.loc[2019, "divergence_change"] == 0.0 and traj.loc[2019, "placebo_p95"] == 0.0
    # The trend runs over the post-planting years only (review T13).
    post = YEARS[1:]
    assert s["trend"]["n_years"] == len(post)
    assert s["trend"]["theil_sen_slope"] == pytest.approx(theilslopes(traj.loc[post, "divergence_change"], post)[0])

    # The divergence is 1 - cos between the dequantized means of the role 1 and role 2 pixels, by hand.
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role = bands["role"]
    quiet = TesseraReader.from_store(working_mem)
    h, w = role.shape
    for year in YEARS:
        win = quiet.read_window("10N", year, slice(r0, r0 + h), slice(c0, c0 + w), layers=("embeddings", "scales"))
        deq = win.embeddings.astype(np.float64) * win.scales[..., None].astype(np.float64)
        d = 1.0 - cos(deq[role == rs.ROLE_SITE].mean(axis=0), deq[role == rs.ROLE_CONTROL].mean(axis=0))
        assert traj.loc[year, "divergence"] == pytest.approx(d, abs=1e-9)

    # Target progress rises from 0 towards 1; the site's pixels come closer to the target than to the control.
    progress = traj["target_progress"]
    assert abs(progress[2019]) < 0.02 and progress[2024] > 0.95 and progress.loc[:2024].is_monotonic_increasing
    assert traj.loc[2019, "target_similarity"] < 0.5 and traj.loc[2025, "target_similarity"] > 0.99
    closer = traj["closer_to_target_fraction"]
    assert closer.is_monotonic_increasing and closer[2019] == 0.0 and closer[2023] == 1.0
    assert (traj["control_closer_to_target_fraction"] == 0.0).all()
    assert (traj["departed_fraction"].iloc[1:] == 1.0).all()
    assert (traj["n_site"] == 400 - 1 - 16).all() and (traj["n_target"] == 26 * 26).all()
    assert s["target"]["halves"] is not None and sum(s["target"]["halves"]) == 26 * 26
    assert s["trend"]["theil_sen_slope"] > 0.1 and s["trend"]["spearman_rho"] > 0.9

    # The null pair stays within the band; the placebos rarely call divergence.
    null = s["null_test"]
    assert null["null_verdict"] == "no divergence detected" and null["null_near_zero"] is True
    assert null["null_to_site_ratio"] < 0.01 and sum(null["half_pixels"]) == traj["n_control"].iloc[0]
    assert null["split"] == "sectors" and null["half_to_site_pixels"] > 10 and "easy to pass" in null["caveat"]
    post_null = [v for y, v in null["divergence_change"].items() if int(y) >= 2020]
    assert null["null_max_abs_change"] == pytest.approx(max(abs(v) for v in post_null))
    plc = s["placebos"]
    assert plc["false_positive_rate"] <= 0.1 and plc["pixels_cap"] == 383 and plc["distinct_centres"] == 200
    # Placebos of the site's footprint: a disk of its area, at most the site's pixel count.
    assert plc["radius_m"] == pytest.approx(math.sqrt(400 * 100.0 / math.pi))
    assert 25 <= plc["pixels_min"] <= plc["pixels_median"] <= plc["pixels_max"] <= 383
    assert (
        0.8 < plc["footprint_ratio_median"] <= 1.0
        and "cannot exceed about 5 percent" in plc["false_positive_rate_note"]
    )
    # Review S6: the rate is bounded by construction, and the placebo note says so.
    assert "cannot exceed about 5 percent" in plc["note"] and s["placebo_false_positive_rate_note"] == rs.FPR_NOTE
    placebos = pd.read_parquet(result.extra_paths["placebos"])
    assert (placebos["reach_m"] <= plc["radius_m"]).all()
    assert not placebos.groupby("placebo")[["centre_x", "centre_y"]].first().duplicated().any()
    assert s["limitation"] == rs.LIMITATION and "not an accuracy" in rs.LIMITATION


def test_a_site_that_does_nothing_shows_no_divergence(null_mem, files, tmp_path):
    reader, _ = _reader(null_mem)
    result = run(reader, files, tmp_path)
    s = result.summary
    assert s["verdict"] == "no divergence detected" and s["diverging_since"] is None
    traj = trajectory(result)
    assert not traj.loc[2020:, "significant"].all()
    null = s["null_test"]
    assert null["null_verdict"] == "no divergence detected" and null["null_near_zero"]
    assert null["null_max_abs_change"] < 1e-3
    # Every year draws fresh noise here, so after the baseline a site that did nothing sits near 0.05.
    departed = traj.loc[2020:, "departed_fraction"]
    assert (departed - 0.05).abs().max() < 0.04
    assert s["placebos"]["false_positive_rate"] <= 0.06
    assert s["placebo_false_positive_rate"] == s["placebos"]["false_positive_rate"]
    assert (traj["closer_to_target_fraction"] == 0.0).all()
    assert traj["target_progress"].abs().max() < 0.02
    # The change since the baseline is the column to read (lead decision); the note says why the level is offset.
    np.testing.assert_allclose(
        traj["departed_fraction_change"], traj["departed_fraction"] - traj.loc[2019, "departed_fraction"]
    )
    assert traj["departed_fraction_change"].abs().max() < 0.04 and traj.loc[2019, "departed_fraction_change"] == 0.0
    assert s["departed_fraction_baseline"] == traj.loc[2019, "departed_fraction"]
    assert s["departed_fraction_change"] == {str(y): v for y, v in traj["departed_fraction_change"].items()}
    assert "trims the control's low tail" in s["departed_fraction_note"]


def test_a_small_target_splits_into_contiguous_halves_and_warns(working, files, tmp_path, monkeypatch):
    reader, _ = working
    left, top = restore_corner()
    # A 6 x 6 stand inside one 200 m cell: the old cell folds could not split it; a line across its axis can.
    small = write_polygon(tmp_path / "small_target.geojson", grid_polygon(left, top, ((160, 166), (10, 16))))
    result = run(reader, files, tmp_path, target=small)
    s = result.summary
    traj = trajectory(result)
    assert s["target"]["pixels"] == 36 and s["target"]["halves"] == [18, 18] and s["target"]["small_target"]
    assert traj["target_progress"].notna().all() and traj["target_halves_similarity"].notna().all()
    # Review S7: a small target biases the 100 percent level low, so the run says the progress overstates.
    warnings = " ".join(result.report["warnings"])
    assert "The target holds 36 used pixels, under 200" in warnings and "overstates" in warnings
    big = run(reader, files, tmp_path, out="big")
    assert not big.summary["target"]["small_target"] and not any(
        "used pixels, under" in w for w in big.report["warnings"]
    )
    # A half under the minimum: only the raw similarity, no progress.
    monkeypatch.setattr(rs, "MIN_TARGET_HALF_PIXELS", 20)
    split = run(reader, files, tmp_path, out="nosplit", target=small)
    traj = trajectory(split)
    assert split.summary["target"]["halves"] is None
    assert traj["target_progress"].isna().all() and traj["target_halves_similarity"].isna().all()
    assert traj["target_similarity"].notna().all() and traj.loc[2025, "target_similarity"] > 0.99
    warnings = " ".join(split.report["warnings"])
    assert (
        "cannot be split into two spatial halves for the 100 percent level (one half would hold 18 pixels" in warnings
    )
    assert traj.loc[2025, "closer_to_target_fraction"] == 1.0


def test_too_few_post_planting_years_is_insufficient(null_mem, files, tmp_path):
    reader, _ = _reader(null_mem)
    result = run(reader, files, tmp_path, planted=2024)
    s = result.summary
    assert s["verdict"] == "insufficient years" and s["year_a"] == 2023 and s["post_planting_years"] == [2024, 2025]
    assert "needs at least 3" in s["verdict_detail"] and s["placebos"]["false_positive_rate"] is None
    assert set(pd.read_parquet(result.paths.summary_parquet)["verdict"]) == {"insufficient years"}
    late_reader, _, _ = restore_store(planted=2024)
    two = run(late_reader, files, tmp_path, out="two", planted=2024, min_post_years=2)
    assert two.summary["verdict"] == "diverging from control since 2024"


def test_a_site_already_unlike_its_ring_at_the_baseline_keeps_no_control(working, files, tmp_path):
    reader, log = working
    # Planted "2024" on the store whose site has been turning into forest since 2020: by 2022, the year the
    # control is matched on, no ring pixel is like it, so no control can be matched, and the run stops before stage 2.
    with pytest.raises(rs.RestoreError, match="Only 0 ring pixels are kept") as info:
        run(reader, files, tmp_path, planted=2024)
    text = str(info.value)
    assert "widen the ring" in text
    # Lead decision: a site cleared or prepared in the match year is named, with an earlier --year-a.
    assert (
        "prepared or cleared for planting in 2022" in text and "pass --year-a 2022 (two years before planting)" in text
    )
    assert {y for y, _, _ in fetched(log)} == {2022}


def test_the_control_filter_rejects_other_land_with_a_heldout_threshold(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path)
    ctrl = result.summary["control"]
    assert ctrl["source"] == "ring" and ctrl["filtered"] and ctrl["threshold_source"] == "heldout"
    # The 20 x 20 site straddles two 200 m cells each way: four spatial blocks, four folds.
    assert ctrl["site_cells"] == 4 and ctrl["folds"] == 4 and ctrl["heldout_pixels"] == 400
    th = ctrl["threshold"]
    assert th["method"] == "heldout" and th["heldout_percentile"] == th["threshold"]
    assert th["threshold"] > th["in_sample_percentile"]  # held-out similarities are lower, so the cut is looser
    assert ctrl["threshold_similarity"] == pytest.approx(1.0 - th["threshold"])
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role, base = bands["role"], bands["baseline_similarity"]
    rows = np.arange(role.shape[0])[:, None] + r0
    cols = np.arange(role.shape[1])[None, :] + c0
    site = (rows >= SITE_BOX[0][0]) & (rows < SITE_BOX[0][1]) & (cols >= SITE_BOX[1][0]) & (cols < SITE_BOX[1][1])
    ring = np.isfinite(base) & ~site
    passes = (1.0 - base.astype(np.float64)) <= th["threshold"]
    assert ctrl["candidate_pixels_valid_match_year"] == int(ring.sum())
    assert ctrl["kept_pixels"] == int((ring & passes).sum())
    assert ctrl["rejected_pixels"] == int((ring & ~passes).sum()) == int((role == rs.ROLE_REJECTED).sum())
    crop = (rows < CROP_BOX[0][1]) & (cols >= CROP_BOX[1][0]) & ring
    assert crop.sum() > 300 and (role[crop] == rs.ROLE_REJECTED).all()
    # Comparable pasture is kept: the rejections outside the crop are the noise tail, about 5 percent.
    pasture = ring & ~crop
    assert 0.0 < float(np.mean(role[pasture] == rs.ROLE_REJECTED)) < 0.1


def test_a_site_in_one_cell_falls_back_to_the_in_sample_threshold(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path, fold_cell_m=4000.0)
    ctrl = result.summary["control"]
    assert ctrl["threshold_source"] == "in_sample" and ctrl["site_cells"] == 1 and ctrl["folds"] == 1
    assert ctrl["threshold"]["fallback_reason"] == "the site's pixels fall in 1 cell(s) of 4000 m"
    assert any("in-sample 5th percentile" in w for w in result.report["warnings"])
    assert ctrl["threshold"]["threshold"] == ctrl["threshold"]["in_sample_percentile"]
    folds_one = run(reader, files, tmp_path, out="one", folds=1)
    assert folds_one.summary["control"]["threshold"]["fallback_reason"] == "--folds is below 2"


def test_panel_and_regime_drops_are_counted_and_shown(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path)
    s = result.summary
    regime = s["regime"]
    assert regime["radar_free_years"] == [] and set(regime["majority_pattern"].values()) == {"radar"}
    site = regime["dropped"]["site"]
    assert site == {"read": 400, "panel": 1, "regime": 16, "used": 383}
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role, base = bands["role"], bands["baseline_similarity"]
    kept = (1.0 - base.astype(np.float64)) <= s["control"]["threshold"]["threshold"]

    def block(box_rc):
        (a, b), (c, d) = box_rc
        return slice(a - r0, b - r0), slice(c - c0, d - c0)

    hole, free = block(HOLES[1][1]), block(RADAR_FREE[1][1])
    ctrl = regime["dropped"]["control"]
    assert ctrl["panel"] == int(kept[hole].sum()) and ctrl["regime"] == int(kept[free].sum()) and ctrl["regime"] > 8
    assert ctrl["used"] == int((role == rs.ROLE_CONTROL).sum()) == ctrl["read"] - ctrl["panel"] - ctrl["regime"]
    # A pixel without data in one year is dropped from every year: role 4 and no similarity to the control.
    assert role[block(HOLES[0][1])].item() == rs.ROLE_DROPPED
    assert (role[block(RADAR_FREE[0][1])] == rs.ROLE_DROPPED).all()
    for year in YEARS:
        sim = bands[f"similarity_to_control_{year}"]
        assert np.isnan(sim[block(HOLES[0][1])]).all() and np.isfinite(sim[role == rs.ROLE_SITE]).all()
    warnings = " ".join(result.report["warnings"])
    assert "17 of 400 site pixels were dropped: 1 lack data" in warnings and "16 have another radar pattern" in warnings


def test_a_whole_grid_radar_switch_warns_and_keeps_the_verdict(switch_mem, files, tmp_path):
    reader, _ = _reader(switch_mem)
    result = run(reader, files, tmp_path)
    s = result.summary
    assert s["regime"]["radar_free_years"] == list(SWITCH)
    assert [s["regime"]["majority_pattern"][str(y)] for y in YEARS] == ["radar"] * 3 + ["optical"] * 3 + ["radar"]
    assert any("no Sentinel-1 in 2022, 2023, 2024" in w for w in result.report["warnings"])
    assert s["verdict"] == "diverging from control since 2020"
    assert s["null_test"]["null_near_zero"] and s["regime"]["dropped"]["site"]["regime"] == 0
    assert s["regime"]["site_radar_free_fraction"] == {str(y): float(y in SWITCH) for y in YEARS}


def test_stage_two_reads_only_the_chunks_with_a_kept_pixel(working, files, tmp_path):
    reader, log = working
    result = run(reader, files, tmp_path)
    bands, (r0, c0) = tif_bands(result.paths.tif)
    # Every pixel read after stage 1: the site, the kept control (role 4 here are pixels the panel or
    # regime rule dropped after stage 2), and the target.
    rr, cc = np.nonzero(np.isin(bands["role"], (rs.ROLE_SITE, rs.ROLE_CONTROL, rs.ROLE_DROPPED)))
    kept_chunks = {(int(r), int(c)) for r, c in zip((rr + r0) // 64, (cc + c0) // 64, strict=True)}
    kept_chunks.add((TARGET_BOX[0][0] // 64, TARGET_BOX[1][0] // 64))
    reads = fetched(log)
    later = {(r, c) for y, r, c in reads if y != MATCH}
    for year in YEARS:
        assert {(r, c) for y, r, c in reads if y == year} == kept_chunks
    assert later == kept_chunks
    # The crop chunk holds only rejected ring pixels: read for the match year, never again.
    crop_chunk = (0, 2)
    assert (MATCH, *crop_chunk) in reads and crop_chunk not in later
    assert {y for y, _, _ in reads} == {MATCH, *YEARS}
    assert result.summary["chunk_windows_stage2"] > 0
    assert result.report["provenance"]["years"] == [MATCH, *YEARS]
    assert 0 < result.report["bytes_read"] <= result.report["estimated_bytes"]


def test_a_target_inside_the_ring_is_left_out_of_the_control(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    inside = ((130, 150), (70, 90))
    target = write_polygon(tmp_path / "ring_target.geojson", grid_polygon(left, top, inside))
    result = run(reader, files, tmp_path, target=target)
    s = result.summary
    bands, (r0, c0) = tif_bands(result.paths.tif)
    (a, b), (c, d) = inside
    block = bands["role"][a - r0 : b - r0, c - c0 : d - c0]
    assert (block == rs.ROLE_TARGET).all() and int((bands["role"] == rs.ROLE_TARGET).sum()) == 400
    assert (
        s["target"]["pixels"] == 400
        and np.isfinite(bands["baseline_similarity"][a - r0 : b - r0, c - c0 : d - c0]).all()
    )
    plain = run(reader, files, tmp_path, out="plain", target=None)
    assert s["control"]["candidate_pixels"] == plain.summary["control"]["candidate_pixels"] - 400
    # The pasture target is like the site, so it would have passed the filter: it is out of the control only by role.
    assert s["control"]["kept_pixels"] < plain.summary["control"]["kept_pixels"] - 300


def test_a_supplied_control_is_used_as_given_with_its_balance(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "control.geojson", grid_polygon(left, top, ((20, 60), (20, 60))))
    result = run(reader, files, tmp_path, control=control)
    s = result.summary
    ctrl = s["control"]
    assert ctrl["source"] == "file" and not ctrl["filtered"] and ctrl["rejected_pixels"] == 0
    assert ctrl["kept_pixels"] == 1600 and ctrl["candidate_pixels"] == 1600 and ctrl["ring_inner_km"] is None
    assert ctrl["share_below_threshold"] < 0.2
    assert 0.0 <= ctrl["balance"]["baseline_divergence_rank"] <= 1.0
    # Review S2: the rank shares the placebos' size bias, and its note says so.
    assert "biased low when placebos are smaller than the site" in ctrl["balance"]["note"]
    assert ctrl["balance"]["baseline_divergence"] == trajectory(result).loc[2019, "divergence"]
    assert not any("poorly matched" in w for w in result.report["warnings"])
    assert s["verdict"] == "diverging from control since 2020"
    bands, _ = tif_bands(result.paths.tif)
    assert rs.ROLE_REJECTED not in np.unique(bands["role"])
    assert result.report["params"]["control"]["sha256"] == export.file_sha256(control)
    assert result.report["params"]["ring_inner_km"] is None


def test_placebos_hold_a_third_of_a_small_control_and_warn(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "control.geojson", grid_polygon(left, top, ((20, 50), (20, 50))))
    result = run(reader, files, tmp_path, control=control)
    s = result.summary
    assert s["control"]["kept_pixels"] == 900 and s["placebos"]["pixels_cap"] == 300 < s["site"]["pixels_used"]
    warning = next(w for w in result.report["warnings"] if w.startswith("Placebos hold a median of"))
    assert "a third of the control pool), fewer than the site's 383" in warning
    # Review S2: smaller placebos shift the band; the warning no longer calls the test conservative.
    assert "moves the band rather than only widening it" in warning and "conservative" not in warning
    placebos = pd.read_parquet(result.extra_paths["placebos"])
    assert (placebos["pixels"] <= 300).all() and placebos["placebo"].nunique() == 200
    assert placebos["pixels"].median() == s["placebos"]["pixels_median"]


def test_a_poorly_matched_control_warns(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "crop.geojson", grid_polygon(left, top, ((0, 60), (130, 190))))
    result = run(reader, files, tmp_path, control=control)
    assert any("supplied control is poorly matched" in w for w in result.report["warnings"])
    assert result.summary["control"]["share_below_threshold"] == 1.0
    assert result.summary["control"]["balance"]["baseline_divergence_rank"] == 1.0


@pytest.mark.parametrize(
    ("control_box", "target_box", "match"),
    [
        (((80, 100), (80, 100)), TARGET_BOX, "site and the control overlap"),
        (((20, 60), (20, 60)), ((40, 80), (40, 80)), "control and the target overlap"),
        (None, ((100, 120), (100, 120)), "site and the target overlap"),
    ],
)
def test_overlapping_polygons_stop_before_any_read(working, files, tmp_path, control_box, target_box, match):
    reader, log = working
    before = len(log.log)
    left, top = restore_corner()
    control = (
        None if control_box is None else write_polygon(tmp_path / "c.geojson", grid_polygon(left, top, control_box))
    )
    target = write_polygon(tmp_path / "t.geojson", grid_polygon(left, top, target_box))
    with pytest.raises(AppInputError, match=match):
        run(reader, files, tmp_path, control=control, target=target)
    assert len(log.log) == before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"planted": 2017}, "no pre-planting year"),
        ({"planted": None}, "--planted YYYY is required"),
        ({"year_a": 2020}, "must be before the planting year"),
        ({"year_b": 2019}, "planting year .2020. or later"),
        ({"ring_inner_km": 0.5, "ring_outer_km": 0.5}, "ring needs"),
        ({"ring_outer_km": 11.0}, "ring needs"),
        ({"control_pct": 0.0}, "control-pct"),
        ({"folds": 0}, "--folds"),
        ({"fold_cell_m": 5.0}, "--fold-cell-m"),
        ({"placebos": 10}, "--placebos"),
        ({"min_post_years": 1}, "--min-post-years"),
    ],
)
def test_bad_arguments_stop_before_the_store_is_read(working, files, tmp_path, kwargs, match):
    reader, log = working
    before = len(log.log)
    with pytest.raises(AppInputError, match=match):
        run(reader, files, tmp_path, **kwargs)
    assert len(log.log) == before


def test_a_tiny_site_is_refused(working, tmp_path):
    reader, log = working
    before = len(log.log)
    left, top = restore_corner()
    tiny = write_polygon(tmp_path / "tiny.geojson", grid_polygon(left, top, ((90, 94), (90, 94))))
    with pytest.raises(AppInputError, match=r"needs at least 0\.25 ha"):
        rs.run_restore(tiny, tmp_path / "o", planted=2020, reader=reader, **KW)
    assert len(log.log) == before


def test_a_2018_planting_warns_about_the_2017_baseline(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path, planted=2018)
    assert result.summary["year_a"] == 2017
    assert any("baseline year is 2017" in w and "survey S16" in w for w in result.report["warnings"])


def test_a_years_complete_gap_names_the_zone_year_and_role_before_any_chunk_read(files, tmp_path):
    reader, log, _ = restore_store(years_complete=[y for y in range(2017, 2026) if y != 2023])
    with pytest.raises(io.YearsNotPublishedError, match="10N: 2023") as info:
        run(reader, files, tmp_path)
    notes = " ".join(info.value.__notes__)
    assert "Zone 10N is touched by the site, ring and target" in notes and "2018 to 2025" in notes
    assert "the control is matched on 2018, the year before the baseline" in notes
    assert log.chunk_keys() == []
    assert run(reader, files, tmp_path, year_b=2022).summary["year_b"] == 2022
    # The match year is checked too: a store without 2018 stops before any read.
    reader, log, _ = restore_store(years_complete=[y for y in range(2017, 2026) if y != 2018])
    with pytest.raises(io.YearsNotPublishedError, match="10N: 2018"):
        run(reader, files, tmp_path, out="nomatch")
    assert log.chunk_keys() == []


SEAM = (-120.0, 38.85)


def _seam_store(missing_11n=(2023,)):
    """Empty 10N and 11N grids around 120 W, 38.85 N; 11N leaves out ``missing_11n`` from years_complete."""

    def empty(rows, cols, year):
        z = np.zeros(rows.shape, dtype=bool)
        return np.zeros(rows.shape, dtype=np.int64), z, z

    zones = {}
    for zone in ("10N", "11N"):
        x, y = Transformer.from_crs(4326, io.zone_epsg(zone), always_xy=True).transform(*SEAM)
        zones[zone] = {
            "left": (x // 10) * 10 - 128 * 10,
            "top": (y // 10) * 10 + 128 * 10,
            "height": 256,
            "width": 256,
            "pattern": empty,
            "prototypes": lambda year: np.zeros((1, N_BANDS)),
            "years_complete": [y for y in range(2017, 2026) if zone == "10N" or y not in missing_11n],
        }
    return designed_store(zones)


def test_a_target_in_a_zone_without_the_years_stops_before_any_chunk_read(tmp_path):
    reader, log, _ = _seam_store()
    lon, lat = SEAM
    site = write_polygon(tmp_path / "s.geojson", box(lon - 0.012, lat - 0.002, lon - 0.010, lat + 0.000))
    target = write_polygon(tmp_path / "t.geojson", box(lon + 0.004, lat - 0.002, lon + 0.007, lat + 0.001))
    with pytest.raises(io.YearsNotPublishedError, match="11N: 2023") as info:
        rs.run_restore(site, tmp_path / "o", planted=2020, target=target, reader=reader, **KW)
    assert "Zone 11N is touched by the target" in " ".join(info.value.__notes__)
    assert log.chunk_keys() == []
    with pytest.raises(AppInputError, match="touch 2 zone groups"):
        rs.run_restore(site, tmp_path / "o", planted=2020, target=target, reader=reader, max_zones=1, **KW)
    assert log.chunk_keys() == []


def test_a_control_too_small_to_keep_enough_pixels_stops_before_any_read(working, files, tmp_path):
    # Review F5: 64 pixel centres can never hold 100 kept control pixels, so the preflight refuses it.
    reader, log = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "small.geojson", grid_polygon(left, top, ((20, 28), (20, 28))))
    with pytest.raises(AppInputError, match="The control covers 64 pixel centres") as info:
        run(reader, files, tmp_path, control=control)
    assert "pass a larger control polygon" in str(info.value)
    assert log.chunk_keys() == []


#: Every grid pixel except the site, as boxes: with these as match-year holes the ring has no match-year data.
AROUND_SITE = (
    ((0, SITE_BOX[0][0]), (0, 192)),
    ((SITE_BOX[0][1], 192), (0, 192)),
    (SITE_BOX[0], (0, SITE_BOX[1][0])),
    (SITE_BOX[0], (SITE_BOX[1][1], 192)),
)


def test_too_few_kept_ring_pixels_stop_before_stage_two(files, tmp_path):
    # No ring pixel has match-year data, so none can be filtered and none is kept: the run stops after stage 1.
    reader, log, _ = restore_store(holes=tuple((MATCH, b) for b in AROUND_SITE))
    with pytest.raises(
        rs.RestoreError, match=r"Only 0 ring pixels are kept \(0 have 2018 data, of \d+ in the ring\)"
    ) as info:
        run(reader, files, tmp_path)
    assert "Only 0% of the ring has 2018 data: Pass --match-year with a year of fuller data" in str(info.value)
    assert {y for y, _, _ in fetched(log)} == {MATCH}


def test_a_supplied_control_needs_no_match_year_data(files, tmp_path):
    # 121 pixel centres, 33 of them without data in the match year: the control is used as given, all 121 of
    # them (the match year only sets the ring's filter); its balance rests on the 88 with data.
    reader, log, _ = restore_store(holes=((MATCH, ((20, 23), (20, 31))),))
    left, top = restore_corner()
    control = write_polygon(tmp_path / "c.geojson", grid_polygon(left, top, ((20, 31), (20, 31))))
    result = run(reader, files, tmp_path, control=control)
    s = result.summary
    assert s["control"]["kept_pixels"] == 121 and s["regime"]["dropped"]["control"]["used"] == 121
    assert s["match_year_data"]["control"] == {
        "pixels": 121,
        "with_data": 88,
        "without_data": 33,
        "share_with_data": 88 / 121,
    }
    assert s["control"]["candidate_pixels_valid_match_year"] == 88 and "ring" not in s["match_year_data"]
    assert s["control"]["selection_drift_pixels"] == {"site": 400, "control": 88}
    bands, (r0, c0) = tif_bands(result.paths.tif)
    hole = bands["baseline_similarity"][20 - r0 : 23 - r0, 20 - c0 : 31 - c0]
    assert np.isnan(hole).all() and (bands["role"][20 - r0 : 23 - r0, 20 - c0 : 31 - c0] == rs.ROLE_CONTROL).all()
    assert not any("supplied control's pixels" in w for w in result.report["warnings"])
    # Under half with match-year data: the balance rests on a few, and the run says so.
    sparse, _, _ = restore_store(holes=((MATCH, ((20, 27), (20, 31))),))
    few = run(sparse, files, tmp_path, out="few", control=control)
    warning = next(w for w in few.report["warnings"] if "supplied control's pixels" in w)
    assert "Only 36% of the supplied control's pixels (44 of 121) have 2018 data" in warning
    assert "rests on them alone" in warning and few.summary["control"]["kept_pixels"] == 121


def test_a_site_without_match_year_data_stops_before_stage_two(tmp_path, files):
    reader, log, _ = restore_store(holes=((MATCH, SITE_BOX),))
    with pytest.raises(rs.RestoreError, match="Only 0 of the site's 400 pixels have 2018 data") as info:
        run(reader, files, tmp_path)
    text = str(info.value)
    assert "the year the control is matched on" in text
    assert "Pass --match-year with a year of fuller data (2017 to 2019; the default is 2018)" in text
    assert {y for y, _, _ in fetched(log)} == {MATCH}


def test_a_target_without_match_year_data_is_used(tmp_path, files):
    # Review T9 then; now the target needs data in the trajectory years only, and its match-year gap is reported.
    reader, log, _ = restore_store(holes=((MATCH, TARGET_BOX),))
    result = run(reader, files, tmp_path)
    s = result.summary
    assert s["match_year_data"]["target"] == {
        "pixels": 676,
        "with_data": 0,
        "without_data": 676,
        "share_with_data": 0.0,
    }
    assert trajectory(result)["n_target"].eq(676).all() and s["regime"]["dropped"]["target"]["panel"] == 0
    assert s["verdict"] == "diverging from control since 2020"
    # The target's own match-year gap needs no warning: it plays no part in the filter.
    assert not any("target's pixels" in w for w in result.report["warnings"])


def test_too_few_pixels_after_the_panel_rule_stop_the_run(files, tmp_path):
    # Review T9: the minimums are checked again once the panel rule has dropped pixels without data in a year.
    reader, _, _ = restore_store(holes=((2021, ((20, 23), (20, 31))),))
    left, top = restore_corner()
    control = write_polygon(tmp_path / "c.geojson", grid_polygon(left, top, ((20, 31), (20, 31))))
    with pytest.raises(rs.RestoreError, match="Only 88 control pixels have data in every year from 2019 to 2025"):
        run(reader, files, tmp_path, control=control)
    reader, _, _ = restore_store(holes=((2021, ((86, 105), (86, 106))),))
    with pytest.raises(rs.RestoreError, match="Only 20 site pixels have data in every year from 2019 to 2025"):
        run(reader, files, tmp_path)
    # A hole in the baseline year itself shows once stage 2 has read it.
    reader, _, _ = restore_store(holes=((2019, SITE_BOX),))
    with pytest.raises(rs.RestoreError, match="No site pixel has data in every year from 2019 to 2025"):
        run(reader, files, tmp_path)


#: A sparse match year: no data north of row 100 (the north 14 rows of the site, about half the ring, the crop)
#: nor on the target.
SPARSE_MATCH = ((MATCH, ((0, 100), (0, 192))), (MATCH, TARGET_BOX))


@pytest.fixture(scope="module")
def sparse_match_mem():
    return restore_store(holes=SPARSE_MATCH)[2]


def test_a_sparse_match_year_filters_the_ring_only(sparse_match_mem, files, tmp_path):
    reader, log = _reader(sparse_match_mem)
    result = run(reader, files, tmp_path)
    s = result.summary
    cover = s["match_year_data"]
    assert cover["year"] == MATCH and "match year's only use" in cover["note"]
    assert cover["site"] == {"pixels": 400, "with_data": 120, "without_data": 280, "share_with_data": 0.3}
    assert cover["target"] == {"pixels": 676, "with_data": 0, "without_data": 676, "share_with_data": 0.0}
    ring = cover["ring"]
    assert ring["with_data"] == s["control"]["candidate_pixels_valid_match_year"] and ring["share_with_data"] < 0.5
    assert ring["pixels"] == s["control"]["candidate_pixels"] == ring["with_data"] + ring["without_data"]
    # Site and target pixels need the trajectory years only: every one of them is used.
    assert s["regime"]["dropped"]["site"] == {"read": 400, "panel": 0, "regime": 0, "used": 400}
    assert s["site"]["pixels_valid_match_year"] == 120 and s["site"]["pixels_used"] == 400
    traj = trajectory(result)
    assert traj["n_site"].eq(400).all() and traj["n_target"].eq(676).all()
    assert s["verdict"] == "diverging from control since 2020"
    warnings = result.report["warnings"]
    site_warning = next(w for w in warnings if "of the site's pixels" in w)
    assert (
        "Only 30% of the site's pixels (120 of 400) have 2018 data, the year the control is matched on" in site_warning
    )
    assert "rests on them alone" in site_warning and "every site pixel with data in each year from 2019" in site_warning
    assert "--match-year" in site_warning
    ring_warning = next(w for w in warnings if "of the ring's candidate pixels" in w)
    assert "have 2018 data" in ring_warning and "cannot be filtered" in ring_warning and "--match-year" in ring_warning

    # The ring pixels without match-year data are never kept, and their chunks are not read again unless a kept,
    # site or target pixel shares them; the site's own northern rows are.
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role, base = bands["role"], bands["baseline_similarity"]
    rows = np.arange(role.shape[0])[:, None] + r0
    cols = np.arange(role.shape[1])[None, :] + c0
    site = in_box(rows, cols, SITE_BOX)
    assert (role[site] == rs.ROLE_SITE).all()
    north = np.broadcast_to(rows < 100, role.shape)
    ring_north = north & ~site & np.isfinite(role)
    assert np.isnan(base[north]).all() and ring_north.sum() > 1000
    assert (role[ring_north] == rs.ROLE_DROPPED).all()
    later = {(r, c) for y, r, c in fetched(log) if y != MATCH}
    assert (1, 1) in later  # the site's chunk
    # The site's match-year mean is over its 120 pixels with data: their similarity in the COG is to that mean.
    deq, _ = _window(sparse_match_mem, MATCH, slice(100, 106), slice(86, 106))
    flat = deq.reshape(-1, N_BANDS)
    mean = flat.mean(axis=0)
    by_hand = flat @ mean / (np.linalg.norm(flat, axis=1) * np.linalg.norm(mean))
    np.testing.assert_allclose(base[100 - r0 : 106 - r0, 86 - c0 : 106 - c0].reshape(-1), by_hand, atol=1e-5)

    # The selection drift is over the used pixels with match-year data in both years, not the whole site.
    ctrl = s["control"]
    assert ctrl["selection_drift_pixels"]["site"] == 120
    assert ctrl["selection_drift_pixels"]["control"] == traj["n_control"].iloc[0]
    assert ctrl["baseline_divergence_match_pixels"] != ctrl["baseline_divergence"]
    assert ctrl["selection_drift"] == pytest.approx(
        ctrl["baseline_divergence_match_pixels"] - ctrl["match_year_divergence"]
    )


def test_a_full_match_year_gives_no_coverage_warning(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path)
    cover = result.summary["match_year_data"]
    assert cover["site"]["without_data"] == 0 and cover["ring"]["without_data"] == 0
    assert cover["target"]["share_with_data"] == 1.0
    assert not any("have 2018 data, the year the control is matched on" in w for w in result.report["warnings"])
    ctrl = result.summary["control"]
    assert ctrl["baseline_divergence_match_pixels"] == ctrl["baseline_divergence"]


def test_match_year_option_sets_the_filter_year(working, files, tmp_path):
    reader, log = working
    first = run(reader, files, tmp_path)
    assert first.report["params"]["match_year"] == MATCH == first.summary["match_year"]
    # An earlier, non-adjacent match year: read for stage 1 only, the baseline stays 2019.
    early = run(reader, files, tmp_path, match_year=2017)
    s = early.summary
    assert not early.skipped and s["match_year"] == 2017 == s["control"]["match_year"] and s["year_a"] == 2019
    assert early.report["params"]["match_year"] == 2017 and early.report["provenance"]["years"] == [2017, *YEARS]
    assert any(
        "The control is matched on 2017, the store's first year (--match-year; the baseline is 2019)" in w
        for w in early.report["warnings"]
    )
    assert s["control"]["selection_drift"] is not None and s["verdict"] == "diverging from control since 2020"
    reads = len(log.log)
    assert run(reader, files, tmp_path, match_year=2017).skipped and len(log.log) == reads
    # The baseline itself: allowed, with the regression-to-the-mean warning, and the drift has no year to sit in.
    same = run(reader, files, tmp_path, out="same", match_year=2019)
    s = same.summary
    assert s["match_year"] == 2019 and s["control"]["selection_drift"] is None
    assert same.report["provenance"]["years"] == YEARS
    warning = next(w for w in same.report["warnings"] if "matched on the baseline year itself" in w)
    assert "--match-year 2019" in warning and "regression to the mean" in warning and "2018" in warning
    # The default (None) is the year before the baseline, and the first year when the baseline is the first.
    assert rs.resolve_match_year(2019, None) == 2018 and rs.resolve_match_year(2017, None) == 2017
    assert rs.resolve_match_year(2019, 2017) == 2017 and rs.resolve_match_year(2019, 2019) == 2019


def test_a_supplied_control_matched_on_the_baseline_does_not_warn_of_selection(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "control.geojson", grid_polygon(left, top, ((20, 60), (20, 60))))
    result = run(reader, files, tmp_path, control=control, match_year=2019)
    assert result.summary["match_year"] == 2019
    assert not any("matched on the baseline year itself" in w for w in result.report["warnings"])


def test_a_match_year_equal_to_the_baseline_drops_pixels_without_its_data(files, tmp_path):
    # With the match year on the baseline, a pixel without that year's data has no baseline: the panel drops it.
    reader, _, _ = restore_store(holes=((2019, ((86, 90), (86, 106))),))
    result = run(reader, files, tmp_path, match_year=2019)
    s = result.summary
    assert s["match_year_data"]["site"]["without_data"] == 80
    assert s["regime"]["dropped"]["site"] == {"read": 400, "panel": 80, "regime": 0, "used": 320}
    # The default match year (2018) has data there, and the baseline gap drops the same pixels by the panel rule.
    default = run(reader, files, tmp_path, out="default")
    assert default.summary["match_year_data"]["site"]["without_data"] == 0
    assert default.summary["regime"]["dropped"]["site"]["panel"] == 80


@pytest.mark.parametrize(
    ("match_year", "message"),
    [
        (2020, r"--match-year 2020 is after the baseline year \(2019\)"),
        (2016, "--match-year 2016 is outside the store's years, 2017 to 2025"),
        (2026, "--match-year 2026 is outside the store's years"),
        (2018.5, "must be a year"),
    ],
)
def test_bad_match_years_stop_before_the_store_is_read(working, files, tmp_path, match_year, message):
    reader, log = working
    before = len(log.log)
    with pytest.raises(AppInputError, match=message):
        run(reader, files, tmp_path, match_year=match_year)
    assert len(log.log) == before


def test_a_match_year_missing_from_years_complete_stops_before_any_chunk_read(files, tmp_path):
    reader, log, _ = restore_store(years_complete=list(range(2018, 2026)))
    with pytest.raises(io.YearsNotPublishedError, match="10N: 2017") as info:
        run(reader, files, tmp_path, match_year=2017)
    notes = " ".join(info.value.__notes__)
    assert "2017 (the year the control is matched on, --match-year) and every year from 2019 to 2025" in notes
    assert "--match-year" in notes and log.chunk_keys() == []
    assert run(reader, files, tmp_path).summary["match_year"] == 2018


def test_reruns_skip_and_follow_the_control_and_target_files(working, files, tmp_path):
    reader, log = working
    first = run(reader, files, tmp_path)
    reads = len(log.log)
    again = run(reader, files, tmp_path)
    assert again.skipped and again.report == first.report and len(log.log) == reads
    params = first.report["params"]
    assert params["target"]["sha256"] and params["site"]["sha256"] == first.report["provenance"]["aoi"]["sha256"]
    left, top = restore_corner()
    _, target = files
    write_polygon(target, grid_polygon(left, top, ((160, 186), (12, 38))), {"name": "moved"})
    moved = run(reader, files, tmp_path)
    assert not moved.skipped and moved.report["params"]["target"]["sha256"] != params["target"]["sha256"]
    forced = run(reader, files, tmp_path, force=True)
    assert not forced.skipped


def test_reruns_follow_every_argument_of_the_verdict(working, files, tmp_path):
    # Review T6: with year_a fixed the extent and years stay the same, so planted is only in the params.
    reader, _ = working
    assert not run(reader, files, tmp_path, year_a=2018).skipped
    moved = run(reader, files, tmp_path, year_a=2018, planted=2019)
    assert not moved.skipped and moved.summary["planted"] == 2019
    more = run(reader, files, tmp_path, year_a=2018, planted=2019, min_post_years=4)
    assert not more.skipped
    pct = run(reader, files, tmp_path, year_a=2018, planted=2019, min_post_years=4, control_pct=10.0)
    assert not pct.skipped and pct.summary["control"]["control_pct"] == 10.0
    assert run(reader, files, tmp_path, year_a=2018, planted=2019, min_post_years=4, control_pct=10.0).skipped


def test_the_same_files_under_another_path_skip(working, files, tmp_path, monkeypatch):
    # Review F3: the digest covers the files' names and bytes, not where they lie.
    reader, log = working
    first = run(reader, files, tmp_path)
    assert "source" not in first.report["params"]["site"] and "source" not in first.report["params"]["target"]
    reads = len(log.log)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    relative = (Path("site.geojson"), Path("target.geojson"))
    assert run(reader, relative, tmp_path).skipped
    dotted = (Path("./sub/../site.geojson"), Path("sub/../target.geojson"))
    assert run(reader, dotted, tmp_path).skipped and len(log.log) == reads


def test_a_change_in_the_wetland_code_invalidates_the_outputs(working, files, tmp_path, monkeypatch):
    # Review T6: restore uses wetland's fold helpers, so wetland's code is part of the digest.
    reader, _ = working
    run(reader, files, tmp_path)
    assert run(reader, files, tmp_path).skipped
    edited = tmp_path / "wetland_edited.py"
    edited.write_text(Path(wl.__file__).read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    monkeypatch.setattr(wl, "__file__", str(edited))
    assert not run(reader, files, tmp_path).skipped


def test_control_pct_sets_the_filter(working, files, tmp_path):
    # Review T5: the option reaches the threshold: the 20th percentile cut is tighter than the 5th.
    reader, _ = working
    five = run(reader, files, tmp_path, out="five")
    twenty = run(reader, files, tmp_path, out="twenty", control_pct=20.0)
    th5, th20 = five.summary["control"], twenty.summary["control"]
    assert th20["threshold_similarity"] > th5["threshold_similarity"] and th20["kept_pixels"] < th5["kept_pixels"]
    # The cut is the 20th percentile of the held-out similarities: the 80th of the distances.
    assert th20["threshold"]["percentile"] == 80.0 and th5["threshold"]["percentile"] == 95.0


def test_a_demo_site_gives_its_planting_year_and_target(working, tmp_path, monkeypatch):
    from importlib import resources

    import tessera_apps.apps._common as common

    left, top = restore_corner()
    features = [
        {
            "type": "Feature",
            "properties": {"role": "aoi", "name": "test site", "planted": 2020},
            "geometry": shapely.geometry.mapping(grid_polygon(left, top, SITE_BOX)),
        },
        {
            "type": "Feature",
            "properties": {"role": "target"},
            "geometry": shapely.geometry.mapping(grid_polygon(left, top, TARGET_BOX)),
        },
    ]
    data = tmp_path / "data"
    data.mkdir()
    (data / "testsite.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    real = resources.files

    def files(package):
        return data if package == "tessera_apps.data" else real(package)

    monkeypatch.setattr(demo, "DEMO_NAMES", (*demo.DEMO_NAMES, "testsite"))
    monkeypatch.setattr(demo, "files", files)
    monkeypatch.setattr(common, "files", files)
    reader, _ = working
    result = rs.run_restore("testsite", tmp_path / "o", target="demo", reader=reader, **KW)
    s = result.summary
    assert s["planted"] == 2020 and s["verdict"] == "diverging from control since 2020"
    assert result.report["aoi"]["source"] == "demo:testsite" and s["target"]["pixels"] == 26 * 26
    with pytest.raises(AppInputError, match="no feature with role 'control'"):
        rs.run_restore("testsite", tmp_path / "o2", control="demo", reader=reader, **KW)


# --------------------------------------------------------------------------------------------
# The bundled restore demos and the <demo>:<role> site
# --------------------------------------------------------------------------------------------


#: The UTM zone each bundled restore demo lies in.
DEMO_ZONES = {"itapebi": "24S", "jaru": "20S"}


@pytest.mark.parametrize("name", ["itapebi", "jaru"])
def test_the_bundled_restore_demos_load_with_their_roles(name):
    assert name in demo.DEMO_NAMES and name in demo.RESTORE_DEMOS
    assert demo.demo_roles(name) == ["aoi", "target", "negative"]
    site = load_aoi(name)
    aoi_props = demo.demo_feature_properties(name, "aoi")
    assert site.demo == name and site.demo_role is None and site.source == f"demo:{name}"
    assert site.properties["planted"] == 2019 and site.properties["synthetic"] is True and site.properties == aoi_props
    assert site.path is not None and site.path.is_file() and site.sha256 == export.file_sha256(site.path)
    geoms = {}
    for role in ("aoi", "target", "negative"):
        g = demo.demo_feature(name, role)
        props = demo.demo_feature_properties(name, role)
        assert g.geom_type in ("Polygon", "MultiPolygon") and g.is_valid, role
        assert rs.main_zone(g) == DEMO_ZONES[name] and io.zones_for_geometry(g) == [DEMO_ZONES[name]]
        assert rs.geodesic_area_ha(g) == pytest.approx(props["area_ha"], rel=0.03), role
        geoms[role] = g
    assert demo.demo_feature_properties(name, "negative")["planted"] == 2019
    assert "planted" not in demo.demo_feature_properties(name, "target")
    # Three separate areas, each larger than the app's smallest site.
    for a, b in (("aoi", "target"), ("aoi", "negative"), ("target", "negative")):
        assert rs.overlap_m2(geoms[a], geoms[b], DEMO_ZONES[name]) == 0.0, (a, b)
    assert min(rs.geodesic_area_ha(g) for g in geoms.values()) > rs.MIN_SITE_HA
    with pytest.raises(ValueError, match="no feature with role 'control'"):
        demo.demo_feature_properties(name, "control")


def test_a_demo_role_loads_as_the_site_with_its_own_properties():
    negative = rs.load_polygon("itapebi:negative", "site")
    assert negative.demo == "itapebi" and negative.demo_role == "negative" and negative.name == "itapebi:negative"
    assert negative.source == "demo:itapebi:negative" and negative.properties["planted"] == 2019
    assert negative.geometry.equals(demo.demo_feature("itapebi", "negative"))
    site = load_aoi("itapebi")
    assert negative.sha256 != site.sha256 and negative.geometry_sha256 != site.geometry_sha256
    # Case and spaces do not matter, and the aoi role is the demo itself.
    assert rs.load_polygon(" Itapebi:Negative ", "site").sha256 == negative.sha256
    assert rs.load_polygon("itapebi:aoi", "site") == site
    # --target demo and --control demo resolve against the same demo file, whatever feature is the site.
    target = rs.load_polygon("demo", "target", demo_name=negative.demo)
    assert target.geometry.equals(demo.demo_feature("itapebi", "target")) and target.source == "demo:itapebi:target"
    assert target.properties == {}
    assert rs.demo_role_spec("jaru:negative") == ("jaru", "negative") and rs.demo_role_spec("site.geojson") is None
    assert rs.demo_role_spec("atlantis:negative") is None


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("itapebi:forest", r"--aoi itapebi:forest: Demo 'itapebi' has no feature with role 'forest'; roles present"),
        ("jaru:", "has no feature with role ''"),
        ("atlantis:negative", "neither an existing file nor a bundled demo"),
    ],
)
def test_an_unknown_demo_role_stops_before_any_read(working, tmp_path, spec, message):
    reader, log = working
    before = len(log.log)
    with pytest.raises(AppInputError, match=message):
        rs.run_restore(spec, tmp_path / "o", reader=reader, **KW)
    assert len(log.log) == before


def test_demo_names_elsewhere_keep_their_meaning_and_errors(working, tmp_path):
    reader, log = working
    before = len(log.log)
    # Another app's AOI: a restore demo name is the demo's AOI polygon, as every demo name is ...
    site = load_aoi("jaru")
    assert site.geometry.equals(demo.demo_aoi("jaru")) and site.demo == "jaru" and site.demo_role is None
    # ... but the <demo>:<role> form is restore's alone, and says so.
    with pytest.raises(AppInputError, match=r"only the restore app takes <demo>:<role>.*Pass 'itapebi' for the demo"):
        load_aoi("itapebi:negative")
    with pytest.raises(AppInputError, match="neither an existing file nor a bundled demo"):
        load_aoi("atlantis:negative")
    # Wetland has no seeds for a restore demo; restore has no planting year or target for another app's demo.
    with pytest.raises(AppInputError, match="the AOI has no bundled seeds"):
        wl.load_seeds(None, aoi_demo="itapebi")
    with pytest.raises(AppInputError, match="The demo 'caldor' gives no planting year; pass --planted YYYY"):
        rs.run_restore("caldor", tmp_path / "o", reader=reader, **KW)
    with pytest.raises(AppInputError, match="--target demo: Demo 'everglades' has no feature with role 'target'"):
        rs.run_restore("everglades", tmp_path / "o", planted=2020, target="demo", reader=reader, **KW)
    with pytest.raises(AppInputError, match="The demo feature itapebi:target gives no planting year"):
        rs.run_restore("itapebi:target", tmp_path / "o", reader=reader, **KW)
    with pytest.raises(AppInputError, match="--control demo: Demo 'itapebi' has no feature with role 'control'"):
        rs.run_restore("itapebi:negative", tmp_path / "o", control="demo", reader=reader, **KW)
    assert len(log.log) == before


def test_a_demo_role_site_runs_with_the_demo_target(working, tmp_path, monkeypatch):
    from importlib import resources

    import tessera_apps.apps._common as common

    left, top = restore_corner()
    features = [
        {
            "type": "Feature",
            "properties": {"role": "aoi", "name": "test site", "planted": 2021},
            "geometry": shapely.geometry.mapping(grid_polygon(left, top, ((40, 60), (130, 150)))),
        },
        {
            "type": "Feature",
            "properties": {"role": "target"},
            "geometry": shapely.geometry.mapping(grid_polygon(left, top, TARGET_BOX)),
        },
        {
            "type": "Feature",
            "properties": {"role": "negative", "planted": 2020},
            "geometry": shapely.geometry.mapping(grid_polygon(left, top, SITE_BOX)),
        },
    ]
    data = tmp_path / "data"
    data.mkdir()
    (data / "testsite.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    real = resources.files

    def files(package):
        return data if package == "tessera_apps.data" else real(package)

    monkeypatch.setattr(demo, "DEMO_NAMES", (*demo.DEMO_NAMES, "testsite"))
    monkeypatch.setattr(demo, "files", files)
    monkeypatch.setattr(common, "files", files)
    reader, _ = working
    # The "negative" feature here is the designed store's moving site: planted comes from its own properties.
    result = rs.run_restore("testsite:negative", tmp_path / "o", target="demo", reader=reader, **KW)
    s = result.summary
    assert s["planted"] == 2020 and s["verdict"] == "diverging from control since 2020"
    assert (
        result.report["aoi"]["source"] == "demo:testsite:negative" and s["site"]["source"] == "demo:testsite:negative"
    )
    assert s["target"]["pixels"] == 26 * 26 and result.report["params"]["target"]["name"] == "testsite target"
    assert result.report["params"]["site"]["name"] == "testsite:negative"


def test_canopy_check_against_local_glad_tiles(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    rows, cols = np.mgrid[0:192, 0:192]
    heights = {}
    for year in range(2015, 2025):
        h = np.zeros((192, 192), dtype=np.uint8)
        (a, b), (c, d) = TARGET_BOX
        # Two heights in the target, so a stale window shows as wrong values (review T10).
        h[(rows >= a) & (rows < b) & (cols >= c) & (cols < d)] = np.where(cols < 23, 20, 30)[
            (rows >= a) & (rows < b) & (cols >= c) & (cols < d)
        ]
        (a, b), (c, d) = SITE_BOX
        h[(rows >= a) & (rows < b) & (cols >= c) & (cols < d)] = {2022: 6, 2023: 8, 2024: 10}.get(year, 0)
        # The crop the filter rejects is tall: it must not reach the control's heights (review T7).
        h[in_box(rows, cols, CROP_BOX)] = 25
        heights[year] = h
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, heights)
    result = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    s = result.summary
    canopy = s["canopy"]
    assert canopy["years"] == list(range(2019, 2025)) and canopy["canopy_onset_year"] == 2022
    assert canopy["correlation_sign"] == "positive" and canopy["correlation_years"] == 6
    assert canopy["pearson_divergence_vs_mean_gap_change"] > 0.9
    assert canopy["per_year"]["2021"]["site_mean_m"] == 0.0
    # The tall crop is rejected by the filter, so the kept control holds no tall pixel but a few next to it.
    assert (
        canopy["per_year"]["2024"]["site_tall_share"] > 0.8 and canopy["per_year"]["2024"]["control_tall_share"] < 0.02
    )
    # Every target pixel is covered with its own heights: (13 columns at 20 m + 13 at 30 m) / 26.
    per = canopy["per_year"]["2024"]
    assert per["target_pixels_covered"] == s["target"]["pixels"] and per["target_mean_m"] == pytest.approx(
        25.0, abs=1.0
    )
    assert per["control_pixels_covered"] == s["regime"]["dropped"]["control"]["used"] and per["control_mean_m"] < 0.5
    table = pd.read_parquet(result.paths.summary_parquet).set_index("year")
    for col in ("site_tch_mean", "control_tch_mean", "site_tall_share", "control_tall_share", "target_tch_mean"):
        assert np.isnan(table.loc[2025, col]) and np.isfinite(table.loc[2024, col]), col
    assert table.loc[2024, "site_tch_mean"] == pytest.approx(canopy["per_year"]["2024"]["site_mean_m"])
    val = json.loads(result.paths.validation.read_text())
    assert set(val["scores"]) == {"site_sized_null", "null_test", "canopy"}
    assert val["scores"]["canopy"]["canopy_onset_year"] == 2022
    site_sized = val["scores"]["site_sized_null"]
    assert site_sized["placebo_false_positive_rate"] == s["placebo_false_positive_rate"]
    assert "cannot exceed about 5 percent" in site_sized["note"]
    assert val["reference"]["canopy"]["url_template"] == template and val["reference"]["global_reference"] is None
    caveats = " ".join(val["caveats"])
    assert "not an accuracy" in caveats and "5 m or more" in caveats and "ends in 2024" in caveats
    assert "at most 8 years" in caveats and "easy to pass" in caveats
    assert s["validation"]["canopy_onset_year"] == 2022 and result.validation_written
    assert s["validation"]["canopy_correlation_sign"] == "positive"
    assert s["validation"]["placebo_false_positive_rate"] == s["placebo_false_positive_rate"]


def test_canopy_pixels_no_tile_covers_are_left_out(working, files, tmp_path):
    # Review T7: pixels of a missing tile are NaN in the loader and stored as 255 in the run; they must not
    # enter the means as 255 m.
    reader, _ = working
    left, top = restore_corner()
    west = {y: np.full((192, 120), 12, dtype=np.uint8) for y in range(2015, 2025)}
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, west)
    result = run(reader, files, tmp_path, validate=True, tch_url_template=template, target=None)
    per = result.summary["canopy"]["per_year"]["2020"]
    used = result.summary["regime"]["dropped"]["control"]["used"]
    assert 0 < per["control_pixels_covered"] < used
    # Covered pixels read 12 m, or 0 in the tile's margin; none reads 255.
    assert 0.0 < per["control_mean_m"] <= 12.0 and per["site_mean_m"] == 12.0


def test_a_broken_local_canopy_tile_is_recorded_and_not_retried(working, files, tmp_path):
    # Review T10 and F1: a truncated tile fails the check; an identical rerun keeps the outputs (a broken tile
    # needs --force), and --force once the tile is replaced redoes it.
    reader, log = working
    left, top = restore_corner()
    heights = {y: np.full((192, 192), 9, dtype=np.uint8) for y in range(2015, 2025)}
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, heights)
    tile = next((tmp_path / "tch" / "2019").glob("*.tif"))
    good = tile.read_bytes()
    tile.write_bytes(good[: len(good) // 2])
    first = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert first.summary["canopy"]["error"] and first.summary["canopy"]["error_kind"] == "bad_tile"
    reads = len(log.log)
    assert run(reader, files, tmp_path, validate=True, tch_url_template=template).skipped and len(log.log) == reads
    tile.write_bytes(good)
    fixed = run(reader, files, tmp_path, validate=True, tch_url_template=template, force=True)
    assert fixed.summary["canopy"].get("error") is None


def test_a_canopy_year_without_tiles_is_not_retried_while_still_missing(working, files, tmp_path):
    # Review F1: tiles for 2015 to 2022 only; the check needs 2019 to 2024. A rerun must not read every chunk again
    # while 2023 is still missing, and does once it is there.
    reader, log = working
    left, top = restore_corner()
    some = {y: np.full((192, 192), 9, dtype=np.uint8) for y in range(2015, 2023)}
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, some)
    first = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert "No GLAD tree canopy height tile for 2023" in first.summary["canopy"]["error"]
    assert first.summary["canopy"]["error_kind"] == "missing_tile"
    reads = len(log.log)
    for _ in range(2):
        assert run(reader, files, tmp_path, validate=True, tch_url_template=template).skipped
    assert len(log.log) == reads
    # 2023 alone is not enough: 2024 is needed too.
    write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, {2023: np.full((192, 192), 9, dtype=np.uint8)})
    assert run(reader, files, tmp_path, validate=True, tch_url_template=template).skipped
    write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, {2024: np.full((192, 192), 9, dtype=np.uint8)})
    redone = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert not redone.skipped and redone.summary["canopy"].get("error") is None


def test_an_unreachable_canopy_host_warns_and_strict_stops_before_reading(working, files, tmp_path, monkeypatch):
    reader, log = working

    def refused(request, timeout):
        raise urllib.error.URLError("blocked by the proxy")

    monkeypatch.setattr(validate, "_urlopen", refused)
    with pytest.raises(StrictValidationError, match=r"storage\.googleapis\.com"):
        run(reader, files, tmp_path, validate=True, strict_validate=True)
    assert log.chunk_keys() == [] and not (tmp_path / "out" / "restore_report.json").exists()
    result = run(reader, files, tmp_path, validate=True)
    assert result.validation_written and result.summary["canopy"]["error"]
    assert any(w.startswith(rs.CANOPY_SKIPPED) for w in result.report["warnings"])
    val = json.loads(result.paths.validation.read_text())
    assert val["scores"]["null_test"]["null_near_zero"] and val["reference"]["canopy"]["error"]
    assert "site_tch_mean" not in pd.read_csv(result.paths.summary_csv).columns
    # No canopy check asked for: nothing to load.
    plain = run(reader, files, tmp_path, out="plain", validate=True, canopy=False)
    assert plain.summary["canopy"] is None and plain.validation_written


def test_a_failed_canopy_check_is_redone_once_the_tiles_are_there(working, files, tmp_path):
    reader, _ = working
    template = str(tmp_path / "tch" / "{year}" / "{tile}.tif")
    first = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert first.summary["canopy"]["error"]
    still = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert still.skipped
    left, top = restore_corner()
    flat = {y: np.full((192, 192), 9, dtype=np.uint8) for y in range(2015, 2025)}
    write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, flat)
    redone = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    assert not redone.skipped and redone.summary["canopy"].get("error") is None
    assert redone.summary["canopy"]["per_year"]["2020"]["site_mean_m"] == 9.0


# --------------------------------------------------------------------------------------------
# The default 1 to 3 km ring on a 1024 x 1024 grid with 256 pixel chunks
# --------------------------------------------------------------------------------------------

BIG_SITE = ((497, 527), (497, 527))
#: A whole chunk of crop inside the ring (chunk (1, 3)), so its pruning is the filter's doing (review T14).
BIG_CROP = ((256, 512), (768, 1024))


@pytest.fixture(scope="module")
def big_mem():
    return restore_store(size=1024, chunk=256, site=BIG_SITE, target=None, crop_box=BIG_CROP)[2]


def test_the_default_ring_on_a_large_grid_prunes_stage_two_chunks(big_mem, tmp_path, monkeypatch):
    import tracemalloc

    from tessera_apps.apps import _common

    reader, log = _reader(big_mem)
    left, top = restore_corner()
    site = write_polygon(tmp_path / "site.geojson", grid_polygon(left, top, BIG_SITE))
    # Traced numpy memory per stage (review F2): the peak of each stage, and what is held at its end.
    stages: dict[str, tuple[int, int]] = {}
    lap = _common.RunContext.lap

    def traced_lap(self, stage, since):
        stages[stage] = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        return lap(self, stage, since)

    monkeypatch.setattr(_common.RunContext, "lap", traced_lap)
    tracemalloc.start()
    try:
        result = rs.run_restore(site, tmp_path / "out", planted=2020, reader=reader)
        peak = max(p for _, p in stages.values())
    finally:
        tracemalloc.stop()
    # The estimate's variable part covers the run's peak (the earlier model said 0.57 GB for a 1.0 GB peak) ...
    assert peak < result.report["estimated_memory"] - rs.MEMORY_FIXED
    # ... and once stage 2 has filled the pool, no later stage holds a second copy of it: the panel rule
    # compacts it in place and the null halves are summed through masks (earlier, about 1.75 times the pool).
    s = result.summary
    pool_bytes = (s["control"]["kept_pixels"] + s["site"]["pixels_valid_match_year"]) * (len(YEARS) + 1) * 133
    later = max(stages[k][1] for k in ("panel", "placebos", "trajectories", "write"))
    assert later < stages["stage2"][0] + 0.5 * pool_bytes
    assert s["verdict"] == "diverging from control since 2020"
    assert s["control"]["ring_inner_km"] == 1.0 and s["control"]["ring_outer_km"] == 3.0
    assert s["control"]["kept_pixels"] > 150_000 and s["placebos"]["pixels_cap"] == 900
    assert s["null_test"]["null_near_zero"] and s["placebos"]["false_positive_rate"] <= 0.1
    reads = fetched(log)
    later = {(r, c) for y, r, c in reads if y != MATCH}
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role = bands["role"]
    rr, cc = np.nonzero(np.isin(role, (rs.ROLE_SITE, rs.ROLE_CONTROL, rs.ROLE_REJECTED, rs.ROLE_DROPPED)))
    in_crop = ((rr + r0) // 256 == 1) & ((cc + c0) // 256 == 3)
    # The crop chunk holds ring candidates, every one rejected by the filter: read for the match year, never after.
    assert in_crop.sum() > 5000 and (role[rr[in_crop], cc[in_crop]] == rs.ROLE_REJECTED).all()
    assert (MATCH, 1, 3) in reads and (1, 3) not in later
    # The corner chunks outside the ring are never read after the match year either.
    assert later == {(r, c) for r in range(4) for c in range(4)} - {(0, 0), (0, 3), (3, 0), (3, 3), (1, 3)}
    for year in YEARS:
        assert {(r, c) for y, r, c in reads if y == year} == later


# --------------------------------------------------------------------------------------------
# Cases from the review: designed stores where an earlier rule gave a wrong answer
# --------------------------------------------------------------------------------------------


def _window(mem, year: int, rows: slice, cols: slice) -> tuple[np.ndarray, np.ndarray]:
    """Dequantized float64 vectors and the raw int8 ones of a window of the designed 10N grid."""
    win = TesseraReader.from_store(mem).read_window("10N", year, rows, cols, layers=("embeddings", "scales"))
    assert win.embeddings is not None
    return win.embeddings.astype(np.float64) * win.scales[..., None].astype(np.float64), win.embeddings


def _mixed_ring_store(distance: float = 40.0, share: float = 0.5):
    """A pasture site that does nothing, in a ring of pasture mixed pixel by pixel with a near-pasture land type B.

    B is ``distance`` from pasture (pasture's norm is about 265); the store draws fresh noise per pixel and year,
    so B pixels that look like the site in one year by their noise look like B again the next (review S1).
    """
    rng = np.random.default_rng(7)
    delta = rng.normal(0.0, 1.0, N_BANDS)
    near = PASTURE + delta * distance / np.linalg.norm(delta)
    mixed = rng.random((192, 192)) < share

    def pattern(rows, cols, year):
        classes = np.where(mixed[rows, cols], 1, 0).astype(np.int64)
        classes[in_box(rows, cols, SITE_BOX)] = 0
        return classes, np.ones(rows.shape, bool), np.ones(rows.shape, bool)

    left, top = restore_corner()
    return designed_store(
        {
            "10N": {
                "left": left,
                "top": top,
                "height": 192,
                "width": 192,
                "pattern": pattern,
                "prototypes": lambda year: np.stack([PASTURE, near]),
            }
        }
    )


def test_a_ring_of_near_site_land_does_not_make_a_quiet_site_diverge(files, tmp_path):
    """Review S1: filtering on the baseline's own noise made this do-nothing site 'diverging since 2020'."""
    reader, _, _ = _mixed_ring_store()
    result = run(reader, files, tmp_path, target=None)
    s = result.summary
    assert s["verdict"] == "no divergence detected", s["significant_years"]
    ctrl = s["control"]
    assert ctrl["match_year"] == MATCH and 4000 < ctrl["kept_pixels"] < ctrl["candidate_pixels_valid_match_year"]
    # The B pixels the filter let in by their match-year noise drift back by the baseline: that drift shows
    # between the match year and the baseline, outside dD.
    assert ctrl["selection_drift"] > 1e-4 and ctrl["selection_drift"] == pytest.approx(
        ctrl["baseline_divergence"] - ctrl["match_year_divergence"]
    )
    assert "regression to the mean" in ctrl["match_note"]
    traj = trajectory(result)
    assert traj["divergence_change"].abs().max() < ctrl["selection_drift"]


def test_a_2017_baseline_shares_the_filter_year_and_says_so(working, files, tmp_path):
    reader, _ = working
    result = run(reader, files, tmp_path, planted=2018)
    s = result.summary
    assert s["year_a"] == s["match_year"] == 2017 and s["control"]["selection_drift"] is None
    warning = next(w for w in result.report["warnings"] if "baseline year is 2017" in w)
    assert "share one year" in warning and "regression to the mean" in warning
    # A 2018 baseline is matched on 2017: that warns about the match year, not the baseline (review T13).
    later = run(reader, files, tmp_path, out="later", year_a=2018)
    assert later.summary["match_year"] == 2017
    assert not any("baseline year is" in w for w in later.report["warnings"])
    assert any("The control is matched on 2017" in w for w in later.report["warnings"])


def _heterogeneous_target(tmp_path):
    """A stand of 36 x 26 pixels: 10 rows of pasture north of the 26 rows of the forest target box."""
    left, top = restore_corner()
    (a, b), (c, d) = TARGET_BOX
    return write_polygon(tmp_path / "mixed_target.geojson", grid_polygon(left, top, ((a - 10, b), (c, d))))


@pytest.mark.parametrize("store", ["working", "switch"])
def test_target_levels_and_the_closer_band_by_hand(store, working_mem, switch_mem, files, tmp_path):
    """Review T2: s_T, c_T, tau and the progress from the role band and the target window, by hand."""
    mem = working_mem if store == "working" else switch_mem
    reader, _ = _reader(mem)
    target = _heterogeneous_target(tmp_path)
    result = run(reader, files, tmp_path, target=target)
    traj = trajectory(result)
    s = result.summary
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role = bands["role"]
    h, w = role.shape
    (a, b), (c, d) = TARGET_BOX
    rows = slice(a - 10, b)
    # The halves are the two contiguous ends of the stand: the long axis runs north to south.
    assert s["target"]["halves"] == [18 * 26, 18 * 26] and "centroid" in s["target"]["split"]
    for year in YEARS:
        deq, raw = _window(mem, year, slice(r0, r0 + h), slice(c0, c0 + w))
        tdeq, _ = _window(mem, year, rows, slice(c, d))
        tgt = tdeq.reshape(-1, N_BANDS).mean(axis=0)
        north, south = tdeq[:18].reshape(-1, N_BANDS).mean(axis=0), tdeq[18:].reshape(-1, N_BANDS).mean(axis=0)
        site = deq[role == rs.ROLE_SITE].mean(axis=0)
        ctrl = deq[role == rs.ROLE_CONTROL].mean(axis=0)
        st, ct, tau = cos(site, tgt), cos(ctrl, tgt), cos(north, south)
        assert traj.loc[year, "target_similarity"] == pytest.approx(st, abs=1e-9)
        assert traj.loc[year, "control_target_similarity"] == pytest.approx(ct, abs=1e-9)
        assert traj.loc[year, "target_halves_similarity"] == pytest.approx(tau, abs=1e-9) and tau < 0.99
        assert traj.loc[year, "target_progress"] == pytest.approx((st - ct) / (tau - ct), rel=1e-9)
        if year == YEARS[-1]:
            # closer_to_target_<year_b>: a site pixel more like the target mean than the control's.
            px = raw[role == rs.ROLE_SITE]
            expect = (cosines(px, tgt) > cosines(px, ctrl)).astype(np.float32)
            np.testing.assert_array_equal(bands[f"closer_to_target_{year}"][role == rs.ROLE_SITE], expect)
    if store == "switch":
        # The control mean moves in the radar-free years: the 0 percent level is that year's, not the baseline's.
        assert abs(traj.loc[2023, "control_target_similarity"] - traj.loc[2019, "control_target_similarity"]) > 1e-3


def cosines(raw: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Float32 cosines of int8 vectors to a reference, as the app computes them (the fixtures have exact ties)."""
    from tessera_apps import metrics

    return metrics.cosine_similarity(raw, ref).astype(np.float64)


def test_control_pixels_closer_to_a_pasture_target_by_hand(working, working_mem, files, tmp_path):
    """Review T3: the control's own closer share, its null, is computed (not 0 by construction)."""
    reader, _ = working
    left, top = restore_corner()
    inside = ((130, 150), (70, 90))  # a pasture stand in the ring: control pixels split between the two means
    target = write_polygon(tmp_path / "ring_target.geojson", grid_polygon(left, top, inside))
    result = run(reader, files, tmp_path, target=target)
    traj = trajectory(result)
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role = bands["role"]
    h, w = role.shape
    for year in (YEARS[0], YEARS[-1]):
        deq, raw = _window(working_mem, year, slice(r0, r0 + h), slice(c0, c0 + w))
        tgt = deq[role == rs.ROLE_TARGET].sum(axis=0)
        ctrl = deq[role == rs.ROLE_CONTROL].sum(axis=0)
        px = raw[role == rs.ROLE_CONTROL]
        expect = float(np.mean(cosines(px, tgt) > cosines(px, ctrl)))
        assert 0.05 < expect < 0.95
        assert traj.loc[year, "control_closer_to_target_fraction"] == pytest.approx(expect, abs=1e-12)


@pytest.fixture(scope="module")
def null_switch_mem():
    return restore_store(moves=False, switch_years=SWITCH)[2]


@pytest.mark.parametrize("store", ["switch", "null_switch"])
def test_per_year_control_cosines_and_departed_by_hand(store, switch_mem, null_switch_mem, files, tmp_path):
    """Review T3: the similarity_to_control bands and departed_fraction use that year's control mean."""
    mem = switch_mem if store == "switch" else null_switch_mem
    reader, _ = _reader(mem)
    result = run(reader, files, tmp_path)
    traj = trajectory(result)
    bands, (r0, c0) = tif_bands(result.paths.tif)
    role = bands["role"]
    h, w = role.shape
    for year in YEARS:
        deq, raw = _window(mem, year, slice(r0, r0 + h), slice(c0, c0 + w))
        ctrl = deq[role == rs.ROLE_CONTROL].sum(axis=0)
        site_c = cosines(raw[role == rs.ROLE_SITE], ctrl)
        ctrl_c = cosines(raw[role == rs.ROLE_CONTROL], ctrl)
        np.testing.assert_allclose(bands[f"similarity_to_control_{year}"][role == rs.ROLE_SITE], site_c, atol=1e-6)
        np.testing.assert_allclose(bands[f"similarity_to_control_{year}"][role == rs.ROLE_CONTROL], ctrl_c, atol=1e-6)
        expect = float(np.mean(site_c < np.percentile(ctrl_c, 5.0)))
        assert traj.loc[year, "departed_fraction"] == pytest.approx(expect, abs=1e-12), year
    if store == "null_switch":
        # A site that did nothing stays near its baseline share in the radar-free years too.
        assert traj["departed_fraction_change"].abs().max() < 0.04


def test_a_null_site_with_a_switch_is_judged_one_sided(null_switch_mem, files, tmp_path):
    """Review T1 and S2: the table's significant column is the verdict's own rule, one-sided with a zero floor."""
    reader, _ = _reader(null_switch_mem)
    result = run(reader, files, tmp_path)
    traj = trajectory(result)
    assert result.summary["verdict"] == "no divergence detected"
    change, p95 = traj["divergence_change"], traj["placebo_p95"]
    # The switch pulls dD below zero, far below the band: not significant.
    assert change[2022] < -abs(p95[2022]) and not traj.loc[2022, "significant"]
    assert traj["significant"].tolist() == ((change > np.maximum(p95, 0.0)) & (change.index >= 2020)).tolist()
    # In 2024 the band sits below zero and the site's falling change is above it: the zero floor keeps it out.
    assert p95[2024] < change[2024] < 0.0 and not traj.loc[2024, "significant"]
    placebos = pd.read_parquet(result.extra_paths["placebos"])
    p = placebos[placebos["year"] == 2024]
    assert not (p["significant"] & (p["divergence_change"] <= 0)).any()


def _half_ring_moves_store():
    """The working scenario, plus every pasture pixel in the even angular sectors around the site moving like it."""
    pattern, prototypes = restore_pattern()

    def pat(rows, cols, year):
        classes, data, radar = pattern(rows, cols, year)
        classes = classes.copy()
        angle = np.mod(np.arctan2(-(rows + 0.5 - 96.0), cols + 0.5 - 96.0), 2 * np.pi)
        even = (np.floor(angle / (np.pi / 4)).astype(int) % 8) % 2 == 0
        classes[even & (classes == CLASS_PASTURE)] = CLASS_SITE
        return classes, data, radar

    left, top = restore_corner()
    return designed_store(
        {"10N": {"left": left, "top": top, "height": 192, "width": 192, "pattern": pat, "prototypes": prototypes}}
    )


def test_the_null_test_fails_when_half_the_control_moves(files, tmp_path):
    """Review T4: the null pair can be called diverging, and then null_near_zero is False."""
    reader, _, _ = _half_ring_moves_store()
    result = run(reader, files, tmp_path, validate=True, canopy=False)
    null = result.summary["null_test"]
    assert null["null_verdict"].startswith("diverging") and null["null_near_zero"] is False
    assert null["significant_years"] and null["null_max_abs_change"] > 0.01
    post = [v for y, v in null["divergence_change"].items() if int(y) >= 2020]
    assert null["null_max_abs_change"] == pytest.approx(max(abs(v) for v in post))
    assert result.summary["validation"]["null_near_zero"] is False


def test_a_compact_supplied_control_gets_a_null_test(working, files, tmp_path):
    """Review S4: a control off to one side lies in one angular sector; its halves come from its own long axis."""
    reader, _ = working
    left, top = restore_corner()
    # North-east of the site, inside one 45 degree sector around its centroid.
    control = write_polygon(tmp_path / "ne.geojson", grid_polygon(left, top, ((40, 60), (130, 160))))
    result = run(reader, files, tmp_path, control=control, validate=True, canopy=False, strict_validate=True)
    null = result.summary["null_test"]
    assert "error" not in null and null["split"] == "long_axis" and null["sectors"] is None
    assert abs(null["half_pixels"][0] - null["half_pixels"][1]) <= 1 and sum(null["half_pixels"]) == 600
    assert null["null_verdict"] == "no divergence detected"
    val = json.loads(result.paths.validation.read_text())
    assert val["scores"]["null_test"]["split"] == "long_axis"


def test_strict_validation_stops_when_the_null_test_cannot_run(working, files, tmp_path, monkeypatch):
    reader, _ = working
    monkeypatch.setattr(rs, "null_halves", lambda xy, centre, sectors=8: np.zeros(len(xy), dtype=np.int8))
    with pytest.raises(StrictValidationError, match="null test could not run"):
        run(reader, files, tmp_path, validate=True, canopy=False, strict_validate=True)
    assert not (tmp_path / "out" / "restore_report.json").exists()
    loose = run(reader, files, tmp_path, validate=True, canopy=False)
    assert loose.summary["null_test"]["error"] and loose.summary["validation"]["null_error"]
    assert any("null test could not run" in c for c in json.loads(loose.paths.validation.read_text())["caveats"])


def test_the_majority_pattern_is_the_sites(files, tmp_path):
    """Review T8: most of the site lacks radar in 2022 while most of the ring has it; the site's pattern wins."""
    reader, _, _ = restore_store(radar_free=((2022, ((86, 100), (86, 106))), (2022, ((36, 60), (40, 100)))))
    result = run(reader, files, tmp_path, target=None)
    reg = result.summary["regime"]
    assert reg["majority_pattern"]["2022"] == "optical" and reg["radar_free_years"] == [2022]
    assert reg["dropped"]["site"]["regime"] == 6 * 20 and reg["dropped"]["site"]["used"] == 14 * 20
    ctrl = reg["dropped"]["control"]
    assert ctrl["used"] == ctrl["read"] - ctrl["panel"] - ctrl["regime"] and ctrl["used"] < 24 * 60


def test_target_pixels_follow_the_regime_rule(files, tmp_path):
    """Review T8: a radar-free block in the target is dropped from the target mean, and counted."""
    reader, _, _ = restore_store(radar_free=((2022, ((160, 164), (10, 14))),))
    result = run(reader, files, tmp_path)
    tgt = result.summary["regime"]["dropped"]["target"]
    assert tgt == {"read": 676, "panel": 0, "regime": 16, "used": 660}
    assert (trajectory(result)["n_target"] == 676 - 16).all() and result.summary["target"]["pixels"] == 660


def test_the_report_aoi_is_the_site(working, files, tmp_path):
    """Review F6: provenance and the report's aoi block describe the site; the extent is labelled as such."""
    reader, _ = working
    result = run(reader, files, tmp_path)
    site = load_aoi(files[0])
    prov = result.report["provenance"]["aoi"]
    assert prov["geometry_sha256"] == site.geometry_sha256 == result.report["params"]["site"]["geometry_sha256"]
    assert prov["sha256"] == site.sha256 and prov["name"] == "site"
    aoi = result.report["aoi"]
    assert aoi["pixels"] == 400 and aoi["area_ha"] == pytest.approx(4.0, rel=0.01)
    assert aoi["extent_pixels"] == result.summary["extent"]["pixels"] > 400
    assert aoi["extent_area_ha"] == result.summary["extent"]["area_ha"]
    # The extent's own geometry is still what the digest covers.
    assert result.report["params"]["aoi_geometry_sha256"] != site.geometry_sha256


def test_an_extent_error_with_a_supplied_control_does_not_advise_the_ring(working, files, tmp_path):
    """Review F7: --ring-outer-km does nothing with a supplied control, so the note does not suggest it."""
    reader, log = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "c.geojson", grid_polygon(left, top, ((10, 60), (10, 60))))
    with pytest.raises(io.AOIError) as info:
        run(reader, files, tmp_path, control=control, max_area_km2=0.05)
    notes = " ".join(info.value.__notes__)
    assert "supplied control" in notes and "--ring-outer-km" not in notes and "closer to the site" in notes
    with pytest.raises(io.AOIError) as ring:
        run(reader, files, tmp_path, max_area_km2=0.05)
    assert "--ring-outer-km" in " ".join(ring.value.__notes__)
    assert log.chunk_keys() == []


def _tile_edge_plan(tmp_path):
    """A 10N plan across the 40 N edge between two GLAD tiles, on an empty designed store."""

    def empty(rows, cols, year):
        z = np.zeros(rows.shape, dtype=bool)
        return np.zeros(rows.shape, dtype=np.int64), z, z

    lon, lat = -121.0, 40.0
    x, y = Transformer.from_crs(4326, RESTORE_EPSG, always_xy=True).transform(lon, lat)
    reader, _, _ = designed_store(
        {
            "10N": {
                "left": (x // 10) * 10 - 1280,
                "top": (y // 10) * 10 + 1280,
                "height": 256,
                "width": 256,
                "pattern": empty,
                "prototypes": lambda year: np.zeros((1, N_BANDS)),
            }
        }
    )
    (zp,) = plan_aoi(reader, box(lon - 0.003, lat - 0.002, lon + 0.003, lat + 0.002)).zones
    return zp


def _write_tile_rows(directory: Path, tile: str, bounds, south: float, north: float, value: int) -> None:
    res = validate.HANSEN_RES_DEG
    x0, x1 = math.floor(bounds[0] / res) - 4, math.ceil(bounds[2] / res) + 4
    y0, y1 = round(south / res), round(north / res)
    data = np.full((y1 - y0, x1 - x0), value, dtype=np.uint8)
    (directory / "2020").mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "width": data.shape[1], "height": data.shape[0], "count": 1, "dtype": "uint8"}
    transform = rasterio.transform.from_origin(x0 * res, y1 * res, res, res)
    with rasterio.open(directory / "2020" / f"{tile}.tif", "w", crs="EPSG:4326", transform=transform, **profile) as ds:
        ds.write(data, 1)


def test_glad_tch_a_partly_covered_local_window_reads_a_tile_added_later(tmp_path):
    """Review F8: a local window with one of its two tiles missing is not cached, so the tile added later is read."""
    zp = _tile_edge_plan(tmp_path)
    bounds = validate._hansen_bounds(zp)
    north_tile, south_tile = validate.hansen_tiles(bounds)
    template = str(tmp_path / "tch" / "{year}" / "{tile}.tif")
    _write_tile_rows(tmp_path / "tch", south_tile, bounds, bounds[1] - 0.01, 40.0, 7)
    first = validate.glad_tch(zp, 2020, url_template=template)
    covered = int(np.isfinite(first).sum())
    assert 0 < covered < int(zp.mask.sum())
    _write_tile_rows(tmp_path / "tch", north_tile, bounds, 40.0, bounds[3] + 0.01, 7)
    second = validate.glad_tch(zp, 2020, url_template=template)
    assert int(np.isfinite(second).sum()) == int(zp.mask.sum()) and np.nanmax(second) == 7


def test_glad_tch_reads_each_window_and_only_the_tile_values(working, tmp_path):
    """Review T10: the cache is per window (bounds in the key), resampling is nearest, years start in 2015."""
    reader, _ = working
    left, top = restore_corner()
    a = _site_plan(reader)
    (b,) = plan_aoi(reader, grid_polygon(left, top, ((20, 50), (140, 170)))).zones
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, _tch_heights({2020: 7}))
    ha = validate.glad_tch(a, 2020, url_template=template)
    hb = validate.glad_tch(b, 2020, url_template=template)
    assert np.isfinite(hb[b.mask]).all() and set(np.unique(hb[b.mask]).tolist()) == {0.0}
    assert set(np.unique(ha[a.mask]).tolist()) == {0.0, 7.0, 15.0}
    with pytest.raises(ValueError, match="2015 to 2024"):
        validate.glad_tch(a, 2014, url_template=template)


def test_glad_tch_a_broken_local_tile_raises(working, tmp_path):
    reader, _ = working
    zp = _site_plan(reader)
    left, top = restore_corner()
    rng = np.random.default_rng(0)
    heights = {2020: rng.integers(0, 30, (192, 192)).astype(np.uint8)}
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, heights)
    tile = next((tmp_path / "tch" / "2020").glob("*.tif"))
    tile.write_bytes(tile.read_bytes()[: tile.stat().st_size // 2])
    with pytest.raises(rasterio.errors.RasterioIOError):
        validate.glad_tch(zp, 2020, url_template=template)


# --------------------------------------------------------------------------------------------
# scripts/check_restore_demos.py, offline on the designed store and local tiles
# --------------------------------------------------------------------------------------------


def test_check_restore_demos_evidence_on_local_layers(tmp_path):
    from scripts import check_restore_demos as crd

    reader, _, _ = restore_store(holes=((2017, SITE_BOX),), radar_free=((2018, SITE_BOX),))
    left, top = restore_corner()
    rows, cols = np.mgrid[0:192, 0:192]
    # Taller and greener a few pixels beyond the site, so the 30 m and 9 m grids cover every site pixel with it.
    around = in_box(rows, cols, ((SITE_BOX[0][0] - 6, SITE_BOX[0][1] + 6), (SITE_BOX[1][0] - 6, SITE_BOX[1][1] + 6)))
    heights = {y: np.where(around & (y >= 2021), 10, 0).astype(np.uint8) for y in validate.GLAD_TCH_YEARS}
    tch = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, heights)
    # Hansen pieces on the same grid: tree cover 80 on the target, loss in 2021 on a 10 x 10 block of the site.
    cover = np.where(in_box(rows, cols, TARGET_BOX), 80, 0).astype(np.uint8)
    loss = np.where(in_box(rows, cols, ((90, 100), (90, 100))), 21, 0).astype(np.uint8)
    write_tch_tiles(tmp_path / "hansen", RESTORE_EPSG, left, top, {2000: cover, 2001: loss})
    (tmp_path / "hansen" / "2000").rename(tmp_path / "hansen" / "treecover2000")
    (tmp_path / "hansen" / "2001").rename(tmp_path / "hansen" / "lossyear")
    hansen = str(tmp_path / "hansen" / "{layer}" / "{tile}.tif")
    grass = np.full((192, 192), 30, dtype=np.uint8)
    wc = write_worldcover_tiles(
        tmp_path / "wc", RESTORE_EPSG, left, top, {2020: grass, 2021: np.where(around, 10, grass)}
    )
    site = grid_polygon(left, top, SITE_BOX)
    ev = crd.polygon_evidence(
        reader, site, tch_template=tch, hansen_template=hansen, worldcover_template=wc, store_years=range(2017, 2026)
    )
    assert ev["zones"] == ["10N"] and ev["pixels"] == 400 and ev["area_ha"] == pytest.approx(4.0, rel=0.01)
    assert ev["glad_tch"]["2020"] == {"mean_m": 0.0, "tall_share": 0.0, "pixels_covered": 400}
    assert ev["glad_tch"]["2021"] == {"mean_m": 10.0, "tall_share": 1.0, "pixels_covered": 400}
    assert ev["hansen"]["treecover2000_mean"] == 0.0 and set(ev["hansen"]["loss_pixels_by_year"]) == {"2021"}
    assert 60 <= ev["hansen"]["loss_pixels"] <= 140 and ev["hansen"]["version"] == validate.HANSEN_VERSION
    assert ev["worldcover"]["2020"] == {"grassland": 1.0} and ev["worldcover"]["2021"] == {"tree cover": 1.0}
    store = ev["store"]["per_year"]
    assert store["2017"] == {"data_share": 0.0, "radar_share": 0.0, "radar_share_of_data": None}
    assert store["2018"] == {"data_share": 1.0, "radar_share": 0.0, "radar_share_of_data": 0.0}
    assert all(store[str(y)]["radar_share"] == 1.0 for y in range(2019, 2026))
    assert ev["store"]["missing_years"] == []
    target = crd.polygon_evidence(
        reader, grid_polygon(left, top, TARGET_BOX), tch_template=tch, hansen_template=hansen, worldcover_template=wc
    )
    assert target["hansen"]["treecover2000_mean"] == pytest.approx(80.0) and target["hansen"]["loss_pixels"] == 0
    # A layer that cannot be loaded is recorded with its error; the rest is still computed.
    broken = crd.polygon_evidence(
        reader,
        site,
        tch_template=str(tmp_path / "none" / "{year}" / "{tile}.tif"),
        hansen_template=hansen,
        worldcover_template=wc,
    )
    assert (
        "GladTileMissingError" in broken["glad_tch"]["error"]
        and broken["hansen"]["loss_pixels"] == ev["hansen"]["loss_pixels"]
    )
    text = crd.demo_tables("test", {"label": "a test site", "polygons": {"aoi": ev, "broken": broken}})
    assert "GLAD canopy height, mean m / share of 5 m or more" in text and "10.0 / 100%" in text
    assert "0% / 0%" in text and "100% / 0%" in text and "grassland 100%" in text and "2021:" in text
    assert "broken" in text and "error" in text


def test_check_restore_demos_polygons_follow_the_app():
    from scripts import check_restore_demos as crd

    for name in demo.RESTORE_DEMOS:
        polys = crd.demo_polygons(name)
        assert list(polys) == list(crd.ROLES)
        zone = rs.main_zone(polys["aoi"])
        ring = rs.ring_geometry(polys["aoi"], 1.0, 3.0, zone=zone)
        assert polys["ring"].symmetric_difference(ring.difference(polys["target"])).area < 1e-12
        assert rs.overlap_m2(polys["ring"], polys["aoi"], zone) < 1.0
    table = crd.format_table(["a", "bb"], [["x", "1"], ["yyy", "22"]])
    assert table.splitlines() == ["a    bb", "---  --", "x     1", "yyy  22"]


def test_correlation_sign_needs_a_correlation_of_some_size():
    # A Spearman rho of 0.04 over eight years is no sign at all (the Itapebi negative read "positive").
    assert rs.correlation_sign(0.04) == "none" and rs.correlation_sign(-0.29) == "none"
    assert rs.correlation_sign(0.3) == "positive" and rs.correlation_sign(0.97) == "positive"
    assert rs.correlation_sign(-0.3) == "negative"
    assert rs.correlation_sign(None) is None and rs.correlation_sign(math.nan) is None


def test_the_null_caveat_follows_the_size_of_the_halves():
    # At Itapebi the smaller half held 0.41 times the site's pixels: the halves test is not easy to pass there.
    assert "about as demanding" in rs.null_caveat(0.41) and "easy to pass" not in rs.null_caveat(0.41)
    assert "easy to pass" in rs.null_caveat(rs.NULL_EASY_RATIO) and "easy to pass" in rs.null_caveat(12.0)
    assert rs.null_caveat(None) == rs.NULL_CAVEAT


def test_a_site_already_moving_at_the_baseline_is_warned_about(working, files, tmp_path):
    reader, _ = working
    # The site starts turning into forest in 2020; told it was planted in 2021, the baseline is 2020 and
    # the match year 2019, so the control is matched but the site already differs at the baseline.
    result = run(reader, files, tmp_path, planted=2021)
    s = result.summary
    warnings = " ".join(result.report["warnings"])
    assert s["control"]["balance"]["baseline_divergence_rank"] >= rs.BASELINE_RANK_WARN
    assert "already differs from its control at the baseline 2020" in warnings
    assert "pass an earlier --year-a (and --match-year)" in warnings
    assert "further from the control between 2019 and 2020" in warnings
    # A site matched at a quiet baseline gets no such warning.
    quiet = run(reader, files, tmp_path, out="quiet")
    assert "already differs from its control" not in " ".join(quiet.report["warnings"])
