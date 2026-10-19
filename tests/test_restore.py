"""Offline tests of the restore app, from its pure parts to whole runs on designed stores.

Pure parts run on hand-made arrays: the ring in metres, the held-out control threshold (checked
against the wetland app's held-out scores), the placebos and their band, the verdict rule, the null
halves, the target progress, the per-pixel fractions, the trend and the canopy check. Runs use the
designed 10N stores of ``tests/conftest.py`` (:func:`~tests.conftest.restore_store`), whose site moves
from pasture to forest from 2020 (or stays pasture), next to a ring of pasture with a crop chunk and
a forest target stand, with known gaps in data and radar. One run uses a 1024 x 1024 grid with 256
pixel chunks and the default 1 to 3 km ring. The GLAD canopy height loader reads local tiles written
by :func:`~tests.conftest.write_tch_tiles`.
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
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point, box

from tessera_apps import demo, export, io, validate
from tessera_apps.apps import restore as rs
from tessera_apps.apps import wetland as wl
from tessera_apps.apps._common import AppInputError, StrictValidationError
from tessera_apps.io import FIRST_YEAR, N_BANDS, TesseraReader, plan_aoi

from .conftest import (
    CROP_BOX,
    RESTORE_EPSG,
    RESTORE_PLANTED,
    SITE_BOX,
    SMALL_RING,
    TARGET_BOX,
    KeyLog,
    designed_store,
    grid_polygon,
    restore_corner,
    restore_store,
    write_polygon,
    write_tch_tiles,
)

YEARS = list(range(RESTORE_PLANTED - 1, 2026))
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
    members, centres = rs.draw_placebos(xy, 30, 12, seed=7)
    assert members.shape == (30, 12) and centres.shape == (30,)
    for row, c in zip(members, centres, strict=True):
        d = np.hypot(*(xy - xy[c]).T)
        assert set(row) == set(np.argsort(d)[:12]) and row[0] == c
    again, _ = rs.draw_placebos(xy, 30, 12, seed=7)
    other, _ = rs.draw_placebos(xy, 30, 12, seed=8)
    np.testing.assert_array_equal(members, again)
    assert not np.array_equal(members, other)
    with pytest.raises(ValueError, match="need 1 to 400"):
        rs.draw_placebos(xy, 3, 401, seed=0)


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
    # Only post-planting years count: divergence that started before planting is called from planting on.
    early = _verdict([0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07], planted=2022)
    assert early.verdict == "diverging from control since 2022" and early.post_years == [2022, 2023, 2024, 2025]
    # Too few post-planting years: insufficient, however strong the change.
    short = _verdict([0, 0.5, 0.6, 0.7], years=[2021, 2022, 2023, 2024], p95=np.full(4, 0.01), planted=2023)
    assert short.verdict == "insufficient years" and short.since is None and "needs at least 3" in short.detail
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
        ]
    )
    texts, diverging = rs.placebo_verdicts(VERDICT_YEARS, change, P95, planted=2020, min_post_years=3)
    assert texts == [
        "diverging from control since 2020",
        "diverging from control since 2024",
        "no divergence detected",
        "no divergence detected",
    ]
    assert diverging.tolist() == [True, True, False, False]
    assert rs.false_positive_rate(diverging, False) == 0.5
    assert rs.false_positive_rate(diverging, True) is None
    texts, diverging = rs.placebo_verdicts(VERDICT_YEARS, change, P95, planted=2024, min_post_years=3)
    assert texts == ["insufficient years"] * 4 and not diverging.any()


def test_null_halves_alternate_angular_sectors():
    angles = np.deg2rad(np.arange(8) * 45 + 10)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 100.0 + [500.0, 200.0]
    assert rs.null_halves(xy, (500.0, 200.0)).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert rs.null_halves(xy, (500.0, 200.0), sectors=4).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


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


def test_majority_pattern_breaks_ties_towards_radar():
    radar = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool)
    assert rs.majority_pattern(radar).tolist() == [True, True, True]
    assert rs.majority_pattern(radar[:2]).tolist() == [True, True, False]
    with pytest.raises(ValueError):
        rs.majority_pattern(radar[:0])


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
    traj = trajectory(result)
    assert traj.index.tolist() == YEARS
    assert traj["significant"].tolist() == [False] + [True] * 6
    assert (traj["divergence_change"].iloc[1:] > traj["placebo_p95"].iloc[1:]).all()
    assert traj.loc[2019, "divergence_change"] == 0.0 and traj.loc[2019, "placebo_p95"] == 0.0

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
    assert s["placebos"]["false_positive_rate"] <= 0.1 and s["placebos"]["pixels_each"] == 383
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
    # The filter trims the control's low tail at the baseline, so the baseline value sits above 0.05 (see the note).
    assert s["departed_fraction_baseline"] == traj.loc[2019, "departed_fraction"] > 0.05
    assert "trims the control's low tail" in s["departed_fraction_note"]


def test_a_target_in_one_cell_reports_its_similarity_without_progress(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    small = write_polygon(tmp_path / "small_target.geojson", grid_polygon(left, top, ((160, 166), (10, 16))))
    result = run(reader, files, tmp_path, target=small)
    s = result.summary
    traj = trajectory(result)
    assert s["target"]["pixels"] == 36 and s["target"]["cells"] == 1 and s["target"]["halves"] is None
    assert traj["target_progress"].isna().all() and traj["target_halves_similarity"].isna().all()
    assert traj["target_similarity"].notna().all() and traj.loc[2025, "target_similarity"] > 0.99
    warnings = " ".join(result.report["warnings"])
    assert "cannot be split into two spatial halves for the 100 percent level (its pixels fall in one 200 m" in warnings
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
    # Planted "2024" on the store whose site has been turning into forest since 2020: by 2023 no ring pixel is
    # like it, so no control can be matched, and the run stops before stage 2.
    with pytest.raises(rs.RestoreError, match="Only 0 ring pixels are kept") as info:
        run(reader, files, tmp_path, planted=2024)
    assert "widen the ring" in str(info.value)
    assert {y for y, _, _ in fetched(log)} == {2023}


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
    assert ctrl["candidate_pixels_valid_year_a"] == int(ring.sum())
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
    later = {(r, c) for y, r, c in reads if y != 2019}
    for year in YEARS[1:]:
        assert {(r, c) for y, r, c in reads if y == year} == kept_chunks
    assert later == kept_chunks
    # The crop chunk holds only rejected ring pixels: read for the baseline, never again.
    crop_chunk = (0, 2)
    assert (2019, *crop_chunk) in reads and crop_chunk not in later
    assert {y for y, _, _ in reads} == set(YEARS)
    assert result.summary["chunk_windows_stage2"] > 0
    assert result.report["provenance"]["years"] == YEARS
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
    assert s["control"]["kept_pixels"] == 900 and s["placebos"]["pixels_each"] == 300 < s["site"]["pixels_used"]
    assert any("Each placebo holds 300 pixels, fewer than the site's 383" in w for w in result.report["warnings"])
    placebos = pd.read_parquet(result.extra_paths["placebos"])
    assert (placebos["pixels"] == 300).all() and placebos["placebo"].nunique() == 200


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
    assert "Zone 10N is touched by the site, ring and target" in notes and "2019 to 2025" in notes
    assert log.chunk_keys() == []
    assert run(reader, files, tmp_path, year_b=2022).summary["year_b"] == 2022


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


def test_too_few_kept_control_pixels_stop_before_stage_two(working, files, tmp_path):
    reader, log = working
    left, top = restore_corner()
    control = write_polygon(tmp_path / "small.geojson", grid_polygon(left, top, ((20, 28), (20, 28))))
    with pytest.raises(rs.RestoreError, match="Only 64 control pixels are kept") as info:
        run(reader, files, tmp_path, control=control)
    assert "pass a larger control polygon" in str(info.value)
    assert {y for y, _, _ in fetched(log)} == {2019}


def test_a_site_without_baseline_data_stops_before_stage_two(tmp_path, files):
    reader, log, _ = restore_store(holes=((2019, SITE_BOX),))
    with pytest.raises(rs.RestoreError, match="Only 0 of the site's 400 pixels have 2019 data"):
        run(reader, files, tmp_path)
    assert {y for y, _, _ in fetched(log)} == {2019}


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


def test_canopy_check_against_local_glad_tiles(working, files, tmp_path):
    reader, _ = working
    left, top = restore_corner()
    rows, cols = np.mgrid[0:192, 0:192]
    heights = {}
    for year in range(2015, 2025):
        h = np.zeros((192, 192), dtype=np.uint8)
        (a, b), (c, d) = TARGET_BOX
        h[(rows >= a) & (rows < b) & (cols >= c) & (cols < d)] = 20
        (a, b), (c, d) = SITE_BOX
        h[(rows >= a) & (rows < b) & (cols >= c) & (cols < d)] = {2022: 6, 2023: 8, 2024: 10}.get(year, 0)
        heights[year] = h
    template = write_tch_tiles(tmp_path / "tch", RESTORE_EPSG, left, top, heights)
    result = run(reader, files, tmp_path, validate=True, tch_url_template=template)
    s = result.summary
    canopy = s["canopy"]
    assert canopy["years"] == list(range(2019, 2025)) and canopy["canopy_onset_year"] == 2022
    assert canopy["pearson_divergence_vs_mean_gap_change"] > 0.9
    assert canopy["per_year"]["2021"]["site_mean_m"] == 0.0 and canopy["per_year"]["2024"]["target_mean_m"] > 15.0
    assert canopy["per_year"]["2024"]["site_tall_share"] > 0.8 and canopy["per_year"]["2024"]["control_tall_share"] == 0
    table = pd.read_parquet(result.paths.summary_parquet).set_index("year")
    for col in ("site_tch_mean", "control_tch_mean", "site_tall_share", "control_tall_share", "target_tch_mean"):
        assert np.isnan(table.loc[2025, col]) and np.isfinite(table.loc[2024, col]), col
    assert table.loc[2024, "site_tch_mean"] == pytest.approx(canopy["per_year"]["2024"]["site_mean_m"])
    val = json.loads(result.paths.validation.read_text())
    assert set(val["scores"]) == {"null_test", "canopy"} and val["scores"]["canopy"]["canopy_onset_year"] == 2022
    assert val["reference"]["canopy"]["url_template"] == template and val["reference"]["global_reference"] is None
    caveats = " ".join(val["caveats"])
    assert "not an accuracy" in caveats and "5 m or more" in caveats and "ends in 2024" in caveats
    assert s["validation"]["canopy_onset_year"] == 2022 and result.validation_written


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
BIG_CROP = ((0, 256), (768, 1024))


@pytest.fixture(scope="module")
def big_mem():
    return restore_store(size=1024, chunk=256, site=BIG_SITE, target=None, crop_box=BIG_CROP)[2]


def test_the_default_ring_on_a_large_grid_prunes_stage_two_chunks(big_mem, tmp_path):
    reader, log = _reader(big_mem)
    left, top = restore_corner()
    site = write_polygon(tmp_path / "site.geojson", grid_polygon(left, top, BIG_SITE))
    result = rs.run_restore(site, tmp_path / "out", planted=2020, reader=reader)
    s = result.summary
    assert s["verdict"] == "diverging from control since 2020"
    assert s["control"]["ring_inner_km"] == 1.0 and s["control"]["ring_outer_km"] == 3.0
    assert s["control"]["kept_pixels"] > 150_000 and s["placebos"]["pixels_each"] == 900
    assert s["null_test"]["null_near_zero"] and s["placebos"]["false_positive_rate"] <= 0.1
    reads = fetched(log)
    later = {(r, c) for y, r, c in reads if y != 2019}
    # The crop chunk and the corner chunks outside the ring are read for the baseline at most, never after.
    assert (2019, 0, 3) in reads and (0, 3) not in later
    assert later == {(r, c) for r in range(4) for c in range(4)} - {(0, 0), (0, 3), (3, 0), (3, 3)}
    for year in YEARS[1:]:
        assert {(r, c) for y, r, c in reads if y == year} == later
