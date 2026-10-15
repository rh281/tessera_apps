"""Tests for tessera_apps.metrics: pure numpy, no network.

Realistic checks use the real 64 x 64 Caldor window (tests/fixtures/caldor_64.npz), where 2020 to
2022 is about half burned. Edge cases use small synthetic windows.
"""

from __future__ import annotations

import dataclasses
import math
import tracemalloc
from collections.abc import Callable

import numpy as np
import pytest

from tessera_apps import metrics
from tessera_apps.io import TesseraError, Window, ZoneGrid, zone_epsg
from tessera_apps.metrics import (
    OtsuResult,
    PCABasis,
    area_to_pixels,
    clean_mask,
    component_sizes,
    cosine_distance_map,
    cosine_similarity,
    cosine_similarity_map,
    cosine_to_vector,
    fit_pca,
    mean_embedding,
    otsu_threshold,
    pca_rgb,
    same_radar_regime,
    trajectory,
)

D = 128


def make_window(
    emb: np.ndarray,
    scales: np.ndarray | None = None,
    *,
    year: int = 2020,
    zone: str = "10N",
    row0: int = 0,
    col0: int = 0,
    asc: np.ndarray | None = None,
    desc: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> Window:
    emb = np.asarray(emb)
    h, w = emb.shape[:2]
    if scales is None:
        scales = np.full((h, w), 0.02, dtype=np.float32)
    grid = ZoneGrid(
        zone=zone, epsg=zone_epsg(zone), left=500_000.0, top=4_300_000.0, height=row0 + h + 8, width=col0 + w + 8
    )
    return Window(
        zone=zone,
        year=year,
        rows=slice(row0, row0 + h),
        cols=slice(col0, col0 + w),
        grid=grid,
        scales=np.asarray(scales, dtype=np.float32),
        embeddings=emb,
        s1_asc_obs_count=asc,
        s1_desc_obs_count=desc,
        mask=mask,
    )


def random_emb(shape: tuple[int, ...], seed: int = 0, d: int = D) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.integers(-128, 128, size=(*shape, d), dtype=np.int8)
    e[..., 0] = np.where(e[..., 0] == 0, 1, e[..., 0])  # never an all-zero vector by accident
    return e


def ref_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Float64 cosine without any of the module's tricks."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


@pytest.fixture(scope="module")
def caldor(caldor_fixture) -> dict[int, Window]:
    """The real Caldor window as one Window per year, at its true zone position."""
    fx = caldor_fixture
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    row0, col0 = int(fx["row0"]), int(fx["col0"])
    h, w = fx["scales"].shape[1:]
    grid = ZoneGrid(
        zone=str(fx["zone"]),
        epsg=int(fx["epsg"]),
        left=c - col0 * a,
        top=f + row0 * a,
        height=row0 + h + 1,
        width=col0 + w + 1,
    )
    out = {}
    for i, year in enumerate(int(y) for y in fx["years"]):
        out[year] = Window(
            zone=str(fx["zone"]),
            year=year,
            rows=slice(row0, row0 + h),
            cols=slice(col0, col0 + w),
            grid=grid,
            scales=fx["scales"][i],
            embeddings=fx["embeddings"][i],
            s2_obs_count=fx["s2_obs_count"][i],
            s1_asc_obs_count=fx["s1_asc_obs_count"][i],
            s1_desc_obs_count=fx["s1_desc_obs_count"][i],
        )
    return out


def traced_peak(fn: Callable[[], object]) -> int:
    """Peak bytes allocated (numpy buffers included) while ``fn`` runs, above what was live before."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] - base
    finally:
        if not was_tracing:
            tracemalloc.stop()


def sub_window(win: Window, rows: slice) -> Window:
    """A row band of a window, as a block of the same zone grid."""
    return Window(
        zone=win.zone,
        year=win.year,
        rows=slice(win.rows.start + rows.start, win.rows.start + rows.stop),
        cols=win.cols,
        grid=win.grid,
        scales=win.scales[rows],
        embeddings=None if win.embeddings is None else win.embeddings[rows],
    )


# ---- cosine_similarity --------------------------------------------------------------------


def test_cosine_identical_opposite_orthogonal():
    a = np.zeros((3, D), dtype=np.int8)
    b = np.zeros((3, D), dtype=np.int8)
    a[0, :5] = [3, -7, 1, 0, 9]
    b[0, :5] = [3, -7, 1, 0, 9]
    a[1, :5] = [3, -7, 1, 0, 9]
    b[1, :5] = [-3, 7, -1, 0, -9]
    a[2, 0] = 5
    b[2, 1] = -11
    out = cosine_similarity(a, b)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, -1.0, 0.0], atol=1e-7)


def test_cosine_int8_extremes_do_not_overflow():
    lo = np.full(D, -128, dtype=np.int8)
    hi = np.full(D, 127, dtype=np.int8)
    assert cosine_similarity(lo, lo) == pytest.approx(1.0, abs=1e-7)
    assert cosine_similarity(hi, hi) == pytest.approx(1.0, abs=1e-7)
    assert cosine_similarity(lo, hi) == pytest.approx(-1.0, abs=1e-7)
    # Alternating extremes: a dot product of -128 * 127 * D must not wrap around.
    alt = np.where(np.arange(D) % 2 == 0, -128, 127).astype(np.int8)
    stack_a = np.stack([alt, lo, alt])
    stack_b = np.stack([lo, alt, -(alt.astype(np.int16)).clip(-128, 127).astype(np.int8)])
    np.testing.assert_allclose(cosine_similarity(stack_a, stack_b), ref_cos(stack_a, stack_b), atol=1e-6)


def test_cosine_long_uint8_vectors_do_not_overflow():
    # 40,000 components of 255 squared sum past 2**31, so int32 accumulation would wrap.
    a = np.full(40_000, 255, dtype=np.uint8)
    b = a.copy()
    b[::2] = 1
    assert float(cosine_similarity(a, a)) == pytest.approx(1.0, abs=1e-7)
    assert float(cosine_similarity(a, b)) == pytest.approx(float(ref_cos(a, b)), abs=1e-7)
    stack = np.stack([a, b])
    np.testing.assert_allclose(cosine_similarity(stack, stack[::-1]), ref_cos(stack, stack[::-1]), atol=1e-7)
    np.testing.assert_allclose(cosine_similarity(stack, b.astype(np.float64)), ref_cos(stack, b[None]), atol=1e-7)


def test_cosine_matches_float64_reference_for_random_int8():
    a = random_emb((17, 23), seed=1)
    b = random_emb((17, 23), seed=2)
    np.testing.assert_allclose(cosine_similarity(a, b), ref_cos(a, b), atol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cosine_float_inputs(dtype):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(9, 11, 16)).astype(dtype)
    b = rng.normal(size=(9, 11, 16)).astype(dtype)
    out = cosine_similarity(a, b)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref_cos(a, b), atol=1e-6)


@pytest.mark.parametrize("dtype", [np.int16, np.uint16, np.uint8, np.int32, np.int64])
def test_cosine_other_integer_dtypes(dtype):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(4)
    hi = min(int(info.max), 2**40)
    lo = max(int(info.min), -(2**40))
    a = rng.integers(lo, hi, size=(50, 32), dtype=dtype, endpoint=True)
    b = rng.integers(lo, hi, size=(50, 32), dtype=dtype, endpoint=True)
    a[0] = info.max
    b[0] = info.max
    np.testing.assert_allclose(cosine_similarity(a, b), ref_cos(a, b), atol=1e-6)
    assert cosine_similarity(a[0], b[0]) == pytest.approx(1.0, abs=1e-7)


def test_cosine_bool_inputs():
    a = np.array([[True, True, False, False], [True, False, False, False]])
    b = np.array([[True, False, True, False], [False, True, False, False]])
    np.testing.assert_allclose(cosine_similarity(a, b), [0.5, 0.0], atol=1e-7)


def test_cosine_zero_vector_is_nan():
    a = random_emb((4,), seed=5)
    b = random_emb((4,), seed=6)
    a[1] = 0
    b[2] = 0
    out = cosine_similarity(a, b)
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert np.all(np.isfinite(out[[0, 3]]))
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    outf = cosine_similarity(af, bf)
    assert np.isnan(outf[1]) and np.isnan(outf[2])
    np.testing.assert_allclose(outf[[0, 3]], out[[0, 3]], atol=1e-6)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cosine_non_finite_component_is_nan(bad):
    rng = np.random.default_rng(7)
    a = rng.normal(size=(3, 8))
    b = rng.normal(size=(3, 8))
    a[0, 3] = bad
    b[2, 7] = bad
    out = cosine_similarity(a, b)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == pytest.approx(ref_cos(a[1], b[1]), abs=1e-6)


def test_cosine_tiny_and_huge_floats_neither_underflow_nor_overflow():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(2, 8))
    b = rng.normal(size=(2, 8))
    expected = ref_cos(a, b)
    np.testing.assert_allclose(cosine_similarity(a * 1e-200, b * 1e-200), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(a * 1e200, b * 1e200), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(a * 1e-200, b * 1e200), expected, atol=1e-6)


def test_cosine_broadcasts_a_reference_vector():
    emb = random_emb((6, 7), seed=9)
    ref = np.random.default_rng(10).normal(size=D)
    expected = ref_cos(emb, ref[None, None, :])
    np.testing.assert_allclose(cosine_similarity(emb, ref), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(ref, emb), expected, atol=1e-6)
    ref_int = random_emb((), seed=11)
    np.testing.assert_allclose(cosine_similarity(emb, ref_int), ref_cos(emb, ref_int[None, None, :]), atol=1e-6)


def test_cosine_reference_vector_edge_cases():
    emb = random_emb((3, 4), seed=12)
    emb[0, 0] = 0
    ref = np.random.default_rng(13).normal(size=D)
    out = cosine_similarity(emb, ref)
    assert np.isnan(out[0, 0])
    assert np.isfinite(out).sum() == 11
    assert np.all(np.isnan(cosine_similarity(emb, np.zeros(D))))
    bad = ref.copy()
    bad[5] = np.nan
    assert np.all(np.isnan(cosine_similarity(emb, bad)))
    embf = emb.astype(np.float64) * 1e-3
    embf[1, 1, 0] = np.inf
    outf = cosine_similarity(embf, ref)
    assert np.isnan(outf[1, 1]) and np.isnan(outf[0, 0])
    np.testing.assert_allclose(outf[2], out[2], atol=1e-6)
    # Huge reference values are scaled before squaring.
    np.testing.assert_allclose(cosine_similarity(emb, ref * 1e300), out, atol=1e-6)
    assert cosine_similarity(ref, ref).shape == ()
    assert float(cosine_similarity(ref, ref)) == pytest.approx(1.0, abs=1e-7)


def test_cosine_outer_broadcast():
    a = random_emb((4, 1), seed=14)
    b = random_emb((1, 5), seed=15)
    out = cosine_similarity(a, b)
    assert out.shape == (4, 5)
    np.testing.assert_allclose(out, ref_cos(np.broadcast_to(a, (4, 5, D)), np.broadcast_to(b, (4, 5, D))), atol=1e-6)


def test_cosine_values_stay_in_range():
    a = random_emb((50,), seed=16)
    out = cosine_similarity(a, a)
    assert np.all(out <= 1.0) and np.all(out >= -1.0)
    np.testing.assert_array_equal(out, np.ones(50, dtype=np.float32))
    rng = np.random.default_rng(17)
    f = rng.normal(size=(200, 7)) * 10 ** rng.uniform(-5, 5, size=(200, 1))
    outf = cosine_similarity(f, f * 3.0)
    assert np.all(outf <= 1.0) and np.all(outf >= 0.9999)
    assert np.all(cosine_similarity(f, -f) >= -1.0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.zeros((3, 4)), np.zeros((3, 5))),
        (np.zeros((3, 0)), np.zeros((3, 0))),
        (np.float64(1.0), np.zeros(4)),
        (np.zeros((3, 4)), np.zeros((2, 4))),
    ],
)
def test_cosine_rejects_bad_shapes(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


@pytest.mark.parametrize("budget", [1, 300, 1000, 5000])
def test_cosine_strips_give_the_same_answer(monkeypatch, budget):
    a = random_emb((5, 6, 7), seed=18, d=16)
    b = random_emb((5, 6, 7), seed=19, d=16)
    a[1, 2, 3] = 0
    ref = np.random.default_rng(20).normal(size=16)
    whole_int = cosine_similarity(a, b)
    whole_float = cosine_similarity(a.astype(np.float32), b.astype(np.float64))
    whole_ref = cosine_similarity(a, ref)
    whole_outer = cosine_similarity(a[:, :, :1], b[:, :1, :])
    monkeypatch.setattr(metrics, "_STRIP_ELEMENTS", budget)
    np.testing.assert_array_equal(cosine_similarity(a, b), whole_int)
    np.testing.assert_array_equal(cosine_similarity(a.astype(np.float32), b.astype(np.float64)), whole_float)
    np.testing.assert_array_equal(cosine_similarity(a, ref), whole_ref)
    np.testing.assert_array_equal(cosine_similarity(a[:, :, :1], b[:, :1, :]), whole_outer)
    np.testing.assert_allclose(whole_int, ref_cos(a, b), atol=1e-6)


def test_cosine_mixes_int8_and_float_arrays():
    # A raw int8 year against a dequantized one: the float kernel must take both.
    a = random_emb((5, 6), seed=50)
    b = random_emb((5, 6), seed=51).astype(np.float64) * 0.03
    expected = ref_cos(a, b)
    np.testing.assert_allclose(cosine_similarity(a, b), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(b, a), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(a, b.astype(np.float32)), expected, atol=1e-6)
    np.testing.assert_allclose(cosine_similarity(a, a.astype(np.float64) * 0.02), 1.0, atol=1e-6)


def test_cosine_maps_stay_within_the_strip_budget():
    # The spec asks for strips so a large block never needs a full float64 copy. These windows hold
    # 16 M int8 elements each, so one float64 copy of either is 128 MB; the strips stay near 40 MB
    # at most (cosine_to_vector) and under 10 MB for the int8 pair kernel.
    rng = np.random.default_rng(52)
    shape = (128, 1024, D)
    a = make_window(rng.integers(-128, 128, size=shape, dtype=np.int8))
    b = make_window(rng.integers(-128, 128, size=shape, dtype=np.int8), year=2022)
    ref = rng.normal(size=D)
    limit = 48 * 2**20
    assert traced_peak(lambda: cosine_distance_map(a, b)) < limit
    assert traced_peak(lambda: cosine_similarity(a.embeddings, b.embeddings)) < limit
    assert traced_peak(lambda: cosine_to_vector(a, ref)) < limit


def test_pieces_cover_every_index_once():
    for lead, inner, budget in [((5, 6, 7), 16, 300), ((3, 4), 10, 25), ((9,), 4, 7), ((2, 3), 1000, 10), ((), 4, 1)]:
        seen = np.zeros(lead, dtype=np.int64)
        for idx in metrics._pieces(lead, inner, budget):
            piece = seen[idx]
            assert piece.size * inner <= max(budget, inner)
            seen[idx] += 1
        np.testing.assert_array_equal(seen, np.ones(lead, dtype=np.int64))
    assert list(metrics._pieces((0, 4), 3, 10)) == []


# ---- window maps --------------------------------------------------------------------------


def test_similarity_map_marks_invalid_pixels_nan():
    a_emb = random_emb((4, 5), seed=21)
    b_emb = random_emb((4, 5), seed=22)
    a_sc = np.full((4, 5), 0.02, dtype=np.float32)
    b_sc = np.full((4, 5), 0.03, dtype=np.float32)
    a_sc[0, 0] = np.nan  # no data in a
    b_sc[0, 1] = np.nan  # no data in b
    a_emb[1, 1] = 0  # fill vector in a, finite scale
    b_sc[2, 2] = 0.0  # non-positive scale
    a_sc[3, 3] = -0.02
    a = make_window(a_emb, a_sc)
    b = make_window(b_emb, b_sc, year=2022)
    sim = cosine_similarity_map(a, b)
    bad = np.zeros((4, 5), dtype=bool)
    bad[[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]] = True
    np.testing.assert_array_equal(np.isnan(sim), bad)
    np.testing.assert_allclose(sim[~bad], ref_cos(a_emb, b_emb)[~bad], atol=1e-6)
    dist = cosine_distance_map(a, b)
    assert dist.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(dist), bad)
    np.testing.assert_allclose(dist[~bad], 1.0 - sim[~bad], atol=1e-7)


def test_similarity_map_does_not_apply_window_mask():
    emb = random_emb((3, 3), seed=23)
    mask = np.zeros((3, 3), dtype=bool)
    a = make_window(emb, mask=mask)
    b = make_window(emb, year=2022, mask=mask)
    np.testing.assert_allclose(cosine_similarity_map(a, b), np.ones((3, 3)), atol=1e-7)


@pytest.mark.parametrize(
    "other",
    [
        {"zone": "11N"},
        {"row0": 1},
        {"col0": 2},
    ],
)
def test_similarity_map_rejects_different_windows(other):
    emb = random_emb((3, 3), seed=24)
    a = make_window(emb)
    b = make_window(emb, year=2022, **other)
    with pytest.raises(ValueError, match="Windows differ"):
        cosine_similarity_map(a, b)
    with pytest.raises(ValueError, match="Windows differ"):
        cosine_distance_map(a, b)


def test_similarity_map_needs_embeddings():
    emb = random_emb((2, 2), seed=25)
    a = make_window(emb)
    b = make_window(emb, year=2022)
    b.embeddings = None
    with pytest.raises(ValueError, match="no embeddings"):
        cosine_similarity_map(a, b)
    b.embeddings = emb[:, :1]
    with pytest.raises(ValueError, match="do not match"):
        cosine_similarity_map(a, b)


def test_raw_cosine_equals_dequantized_cosine_on_caldor(caldor):
    a, b = caldor[2020], caldor[2022]
    raw = cosine_similarity_map(a, b)
    deq = cosine_similarity(a.dequantize(), b.dequantize())
    assert np.isfinite(raw).all()
    np.testing.assert_allclose(raw, deq, atol=1e-6)
    np.testing.assert_allclose(raw, ref_cos(a.dequantize(), b.dequantize()), atol=1e-6)


def test_caldor_burn_is_bimodal_and_stable_pixels_are_similar(caldor):
    dist = cosine_distance_map(caldor[2020], caldor[2022])
    res = otsu_threshold(dist)
    assert res.bimodal
    assert 0.15 < res.threshold < 0.40
    assert res.eta > 0.7
    assert 0.35 < res.high_mode_share < 0.75  # about half the window burned
    assert res.low_mode_share + res.high_mode_share == pytest.approx(1.0)
    burned = dist > res.threshold
    assert float(np.median(1.0 - dist[burned])) < 0.7
    assert float(np.median(1.0 - dist[~burned])) > 0.85
    # Before the fire, year to year, the forest is stable at a cosine around 0.96.
    pre = cosine_similarity_map(caldor[2019], caldor[2020])
    assert 0.94 < float(np.median(pre)) < 0.98


def test_cosine_to_vector_window_and_array(caldor):
    win = caldor[2020]
    mean, n = mean_embedding(win)
    out = cosine_to_vector(win, mean)
    assert out.dtype == np.float32 and out.shape == (64, 64)
    np.testing.assert_allclose(out, ref_cos(win.dequantize(), mean[None, None, :]), atol=1e-6)
    assert 0.8 < float(np.median(out)) < 1.0
    np.testing.assert_allclose(cosine_to_vector(win.embeddings, mean), out, atol=1e-7)
    np.testing.assert_allclose(cosine_to_vector(win.dequantize(), mean), out, atol=1e-6)


def test_cosine_to_vector_marks_invalid_window_pixels():
    emb = random_emb((3, 4), seed=26)
    sc = np.full((3, 4), 0.02, dtype=np.float32)
    sc[1, 2] = np.nan
    sc[2, 3] = -1.0
    emb[0, 0] = 0
    ref = np.random.default_rng(27).normal(size=D)
    out = cosine_to_vector(make_window(emb, sc), ref)
    bad = np.zeros((3, 4), dtype=bool)
    bad[[0, 1, 2], [0, 2, 3]] = True
    np.testing.assert_array_equal(np.isnan(out), bad)
    np.testing.assert_allclose(out[~bad], ref_cos(emb, ref[None, None, :])[~bad], atol=1e-6)


@pytest.mark.parametrize(
    "ref",
    [np.zeros(D), np.full(D, np.nan), np.ones((2, D)), np.ones(D - 1)],
)
def test_cosine_to_vector_rejects_bad_reference(ref):
    win = make_window(random_emb((2, 2), seed=28))
    with pytest.raises(ValueError):
        cosine_to_vector(win, ref)


# ---- mean_embedding -----------------------------------------------------------------------


def test_mean_embedding_matches_dequantized_mean_on_caldor(caldor):
    win = caldor[2022]
    mean, n = mean_embedding(win)
    assert n == 64 * 64
    assert mean.dtype == np.float64 and mean.shape == (D,)
    expected = win.embeddings.astype(np.float64).reshape(-1, D) * win.scales.astype(np.float64).reshape(-1, 1)
    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-10, atol=1e-12)


def test_mean_embedding_dequantizes():
    emb = np.zeros((1, 2, 4), dtype=np.int8)
    emb[0, 0, 0] = 1
    emb[0, 1, 1] = 1
    scales = np.array([[1.0, 3.0]], dtype=np.float32)
    mean, n = mean_embedding(make_window(emb, scales))
    assert n == 2
    np.testing.assert_allclose(mean, [0.5, 1.5, 0.0, 0.0])


def test_mean_embedding_skips_invalid_pixels():
    emb = np.full((2, 3, 4), 10, dtype=np.int8)
    emb[0, 0] = [100, 100, 100, 100]
    emb[0, 1] = 0
    emb[0, 2] = [50, 50, 50, 50]
    emb[1, 0] = [-90, -90, -90, -90]
    scales = np.full((2, 3), 0.5, dtype=np.float32)
    scales[0, 0] = np.nan  # no data
    scales[0, 1] = 7.0  # fill vector with a finite scale
    scales[0, 2] = 0.0  # zero scale
    scales[1, 0] = -0.5  # negative scale
    mean, n = mean_embedding(make_window(emb, scales))
    assert n == 2
    np.testing.assert_allclose(mean, [5.0] * 4)


def test_mean_embedding_mask_and_window_mask():
    emb = np.arange(1, 7, dtype=np.int8).reshape(2, 3, 1) * np.ones((1, 1, 4), dtype=np.int8)
    win_mask = np.array([[True, True, False], [False, False, False]])
    win = make_window(emb, np.ones((2, 3), dtype=np.float32), mask=win_mask)
    mean, n = mean_embedding(win)
    assert n == 2
    np.testing.assert_allclose(mean, [1.5] * 4)
    # An explicit mask is combined with Window.mask, never used in its place.
    explicit = np.array([[False, True, True], [False, True, True]])
    mean, n = mean_embedding(win, explicit)
    assert n == 1
    np.testing.assert_allclose(mean, [2.0] * 4)
    with pytest.raises(ValueError, match="No valid pixels"):
        mean_embedding(win, ~win_mask)
    win.mask = None
    mean, n = mean_embedding(win)
    assert n == 6
    np.testing.assert_allclose(mean, [3.5] * 4)
    mean, n = mean_embedding(win, explicit)
    assert n == 4
    np.testing.assert_allclose(mean, [4.0] * 4)


def test_mean_embedding_pools_blocks_not_block_means(caldor):
    win = caldor[2021]
    whole, n_whole = mean_embedding(win)
    top, bottom = sub_window(win, slice(0, 10)), sub_window(win, slice(10, 64))
    pooled, n = mean_embedding([top, bottom])
    assert n == n_whole == 64 * 64
    np.testing.assert_allclose(pooled, whole, rtol=1e-10, atol=1e-12)
    mt, _ = mean_embedding(top)
    mb, _ = mean_embedding(bottom)
    assert not np.allclose(pooled, (mt + mb) / 2, rtol=1e-4)
    masks = [np.zeros((10, 64), dtype=bool), np.ones((54, 64), dtype=bool)]
    masks[0][0, 0] = True
    pooled_m, n_m = mean_embedding([top, bottom], masks)
    assert n_m == 54 * 64 + 1
    sel = np.zeros((64, 64), dtype=bool)
    sel[0, 0] = True
    sel[10:] = True
    whole_m, _ = mean_embedding(win, sel)
    np.testing.assert_allclose(pooled_m, whole_m, rtol=1e-10, atol=1e-12)


def test_mean_embedding_strips_give_the_same_answer(caldor, monkeypatch):
    win = caldor[2023]
    mask = np.random.default_rng(29).random((64, 64)) > 0.3
    whole, n = mean_embedding(win, mask)
    monkeypatch.setattr(metrics, "_STRIP_ELEMENTS", 1000)
    stripped, n2 = mean_embedding(win, mask)
    assert n2 == n == int(mask.sum())
    np.testing.assert_allclose(stripped, whole, rtol=1e-12, atol=1e-14)


def test_mean_embedding_errors():
    emb = random_emb((2, 2), seed=30)
    win = make_window(emb)
    with pytest.raises(ValueError, match="No valid pixels"):
        mean_embedding(win, np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="No valid pixels"):
        mean_embedding(make_window(emb, np.full((2, 2), np.nan, dtype=np.float32)))
    with pytest.raises(ValueError, match="mask shape"):
        mean_embedding(win, np.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="one per window"):
        mean_embedding([win, win], np.ones((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="2 windows"):
        mean_embedding([win, win], [np.ones((2, 2), dtype=bool)])
    with pytest.raises(ValueError, match="No windows"):
        mean_embedding([])
    with pytest.raises(TypeError):
        mean_embedding([win, emb])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="lengths differ"):
        mean_embedding([win, make_window(random_emb((2, 2), seed=31, d=8))])
    win.embeddings = None
    with pytest.raises(ValueError, match="no embeddings"):
        mean_embedding(win)


# ---- trajectory ---------------------------------------------------------------------------


def test_trajectory_on_caldor(caldor):
    out = trajectory(caldor, 2020)
    assert list(out) == sorted(caldor)
    np.testing.assert_array_equal(out[2020], np.ones((64, 64), dtype=np.float32))
    for year, sim in out.items():
        assert sim.dtype == np.float32
        if year != 2020:
            np.testing.assert_array_equal(sim, cosine_similarity_map(caldor[year], caldor[2020]))
    burned = (
        cosine_distance_map(caldor[2020], caldor[2022])
        > otsu_threshold(cosine_distance_map(caldor[2020], caldor[2022])).threshold
    )
    assert np.median(out[2022][burned]) < np.median(out[2022][~burned]) - 0.15
    assert np.median(out[2019]) > 0.94


def test_trajectory_masks_and_invalid_pixels():
    e20 = random_emb((3, 3), seed=32)
    e21 = random_emb((3, 3), seed=33)
    e22 = e20.copy()
    s20 = np.full((3, 3), 0.02, dtype=np.float32)
    s21 = s20.copy()
    s22 = s20.copy()
    s20[0, 0] = np.nan  # invalid in the reference year: NaN in every year
    s21[1, 1] = np.nan  # invalid in 2021 only
    e22[2, 2] = 0  # fill vector in 2022 only
    wins = {
        2022: make_window(e22, s22, year=2022),
        2020: make_window(e20, s20, year=2020),
        2021: make_window(e21, s21, year=2021),
    }
    mask = np.ones((3, 3), dtype=bool)
    mask[0, 2] = False
    out = trajectory(wins, 2020, mask=mask)
    assert list(out) == [2020, 2021, 2022]
    ref_nan = np.zeros((3, 3), dtype=bool)
    ref_nan[0, 0] = ref_nan[0, 2] = True
    np.testing.assert_array_equal(np.isnan(out[2020]), ref_nan)
    assert np.all(out[2020][~ref_nan] == 1.0)
    nan21 = ref_nan.copy()
    nan21[1, 1] = True
    np.testing.assert_array_equal(np.isnan(out[2021]), nan21)
    np.testing.assert_allclose(out[2021][~nan21], ref_cos(e21, e20)[~nan21], atol=1e-6)
    nan22 = ref_nan.copy()
    nan22[2, 2] = True
    np.testing.assert_array_equal(np.isnan(out[2022]), nan22)
    np.testing.assert_allclose(out[2022][~nan22], 1.0, atol=1e-7)


def test_trajectory_uses_window_masks_when_no_mask_given():
    e = random_emb((2, 3), seed=34)
    m20 = np.array([[True, True, True], [False, True, True]])
    m21 = np.array([[True, False, True], [True, True, True]])
    wins = {2020: make_window(e, mask=m20), 2021: make_window(e, year=2021, mask=m21)}
    out = trajectory(wins, 2021)
    np.testing.assert_array_equal(np.isnan(out[2020]), ~(m20 & m21))
    np.testing.assert_array_equal(np.isnan(out[2021]), ~(m20 & m21))
    # An explicit mask is combined with the windows' masks, never used in their place.
    out = trajectory(wins, 2021, mask=np.ones((2, 3), dtype=bool))
    for year in (2020, 2021):
        np.testing.assert_array_equal(np.isnan(out[year]), ~(m20 & m21))
    explicit = np.array([[False, True, True], [True, True, False]])
    out = trajectory(wins, 2021, mask=explicit)
    for year in (2020, 2021):
        np.testing.assert_array_equal(np.isnan(out[year]), ~(m20 & m21 & explicit))
    plain = {y: Window(**{**w.__dict__, "mask": None}) for y, w in wins.items()}
    np.testing.assert_array_equal(np.isnan(trajectory(plain, 2021, mask=explicit)[2020]), ~explicit)


def test_trajectory_errors():
    e = random_emb((2, 2), seed=35)
    wins = {2020: make_window(e), 2021: make_window(e, year=2021)}
    with pytest.raises(ValueError, match="Reference year"):
        trajectory(wins, 2019)
    with pytest.raises(ValueError, match="mask shape"):
        trajectory(wins, 2020, mask=np.ones((3, 3), dtype=bool))
    wins[2022] = make_window(e, year=2022, col0=1)
    with pytest.raises(ValueError, match="Windows differ"):
        trajectory(wins, 2020)


# ---- explicit masks never reach past Window.mask -------------------------------------------


def _two_zone_burn() -> tuple[np.ndarray, dict[str, dict[int, Window]], np.ndarray]:
    """One 4 x 8 ground strip seen by two zones that overlap on ground columns 2 to 5.

    Zone 10N covers ground columns 0 to 5 and owns 0 to 3 (its Window.mask); zone 11N covers 2 to
    7 and owns 4 to 7, as the nominal bands split an AOI at 120 W. A burn on ground columns 2 to 6
    crosses the seam. Returns the 2022 ground vectors, the windows per zone and year, and the burn.
    """
    ground20 = random_emb((4, 8), seed=60)
    ground22 = ground20.copy()
    burn = np.zeros((4, 8), dtype=bool)
    burn[:, 2:7] = True
    ground22[burn] = random_emb((int(burn.sum()),), seed=61)
    own = np.zeros((4, 8), dtype=bool)
    wins: dict[str, dict[int, Window]] = {}
    for zone, cols, owned in (("10N", slice(0, 6), slice(0, 4)), ("11N", slice(2, 8), slice(4, 8))):
        mask = own.copy()
        mask[:, owned] = True
        wins[zone] = {
            year: make_window(g[:, cols], year=year, zone=zone, mask=mask[:, cols])
            for year, g in ((2020, ground20), (2022, ground22))
        }
    return ground22, wins, burn


def test_masks_from_cosine_maps_stay_inside_window_masks() -> None:
    ground22, wins, burn = _two_zone_burn()
    burned: dict[str, np.ndarray] = {}
    for zone, w in wins.items():
        own = w[2022].mask
        assert own is not None
        dist = cosine_distance_map(w[2020], w[2022])
        assert np.isfinite(dist).all()  # the cosine map ignores Window.mask, as specified
        burned[zone] = dist > 0.5
        assert (burned[zone] & ~own).any()  # so the burn reaches past the zone's own pixels
    # Pooled over both zones, every burned ground pixel counts once.
    mean, n = mean_embedding([wins["10N"][2022], wins["11N"][2022]], [burned["10N"], burned["11N"]])
    assert n == int(burn.sum()) == 20
    expected = ground22[burn].astype(np.float64).mean(axis=0) * np.float64(np.float32(0.02))
    np.testing.assert_allclose(mean, expected, rtol=1e-12)
    basis = fit_pca([wins["10N"][2022], wins["11N"][2022]], [burned["10N"], burned["11N"]])
    np.testing.assert_allclose(basis.mean, expected, rtol=1e-6, atol=1e-6)  # the sample is float32
    for w in wins.values():
        win, m = w[2022], burned[w[2022].zone]
        own = win.mask
        assert own is not None
        n_zone = int((m & own).sum())
        assert mean_embedding(win, m)[1] == n_zone
        traj = trajectory(w, 2020, mask=m)
        for sim in traj.values():
            np.testing.assert_array_equal(np.isfinite(sim), m & own)
        rgb = pca_rgb(win, basis, mask=m)
        assert not rgb[~own].any()
        plain = dataclasses.replace(win, mask=None)
        np.testing.assert_array_equal(rgb, pca_rgb(plain, basis, mask=m & own))
        np.testing.assert_array_equal(fit_pca(win, m).mean, fit_pca(plain, m & own).mean)
        # Without Window.mask the explicit mask alone applies.
        assert mean_embedding(plain, m)[1] == int(m.sum()) > n_zone


# ---- radar regime -------------------------------------------------------------------------


def test_same_radar_regime_cases():
    emb = random_emb((1, 5), seed=36)
    # Pixels: both radar (asc), both radar (desc vs asc), both radar-free, a only, b only.
    a = make_window(
        emb,
        asc=np.array([[30, 0, 0, 12, 0]], dtype=np.uint16),
        desc=np.array([[0, 59, 0, 0, 0]], dtype=np.uint16),
    )
    b = make_window(
        emb,
        year=2022,
        asc=np.array([[25, 31, 0, 0, 0]], dtype=np.uint16),
        desc=np.array([[0, 0, 0, 0, 3]], dtype=np.uint16),
    )
    out = same_radar_regime(a, b)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, [[True, True, True, False, False]])
    np.testing.assert_array_equal(same_radar_regime(b, a), out)


def test_same_radar_regime_errors():
    emb = random_emb((1, 2), seed=37)
    counts = np.ones((1, 2), dtype=np.uint16)
    a = make_window(emb, asc=counts, desc=counts)
    with pytest.raises(TesseraError):
        same_radar_regime(a, make_window(emb, year=2022))
    with pytest.raises(ValueError, match="Windows differ"):
        same_radar_regime(a, make_window(emb, row0=3, asc=counts, desc=counts))


def test_caldor_has_radar_every_year(caldor):
    for year in caldor:
        assert same_radar_regime(caldor[2020], caldor[year]).all()


# ---- Otsu ---------------------------------------------------------------------------------


def test_otsu_two_values():
    v = np.array([0.0] * 30 + [1.0] * 70)
    res = otsu_threshold(v)
    assert isinstance(res, OtsuResult)
    assert res.threshold == pytest.approx(0.5)
    assert res.eta == pytest.approx(1.0)
    assert res.low_mode_share == pytest.approx(0.3)
    assert res.high_mode_share == pytest.approx(0.7)
    assert res.bimodal


def test_otsu_threshold_sits_in_the_gap():
    v = np.array([0.0, 0.1, 0.2, 0.9, 1.0])
    res = otsu_threshold(v)
    assert res.threshold == pytest.approx(0.55)
    assert res.high_mode_share == pytest.approx(0.4)


def test_otsu_ignores_non_finite_values():
    rng = np.random.default_rng(38)
    v = np.concatenate([rng.normal(0.05, 0.02, 500), rng.normal(0.45, 0.05, 300)])
    clean = otsu_threshold(v)
    dirty = otsu_threshold(np.concatenate([v, [np.nan, np.inf, -np.inf, np.nan]]).reshape(4, 201))
    assert dirty == clean


def test_otsu_split_maximises_between_class_variance():
    rng = np.random.default_rng(39)
    v = np.concatenate([rng.gamma(2.0, 0.02, 3000), rng.normal(0.4, 0.08, 1200)])
    res = otsu_threshold(v, bins=64)
    edges = np.linspace(v.min(), v.max(), 65)
    best = 0.0
    for e in edges[1:-1]:
        lo, hi = v[v < e], v[v >= e]
        eta = lo.size * hi.size / v.size**2 * (hi.mean() - lo.mean()) ** 2 / v.var()
        best = max(best, eta)
    assert res.eta == pytest.approx(best, rel=1e-9)
    high = v > res.threshold
    assert high.mean() == res.high_mode_share
    assert (~high).mean() == res.low_mode_share
    eta_at_t = high.mean() * (1 - high.mean()) * (v[high].mean() - v[~high].mean()) ** 2 / v.var()
    assert res.eta == pytest.approx(eta_at_t, rel=1e-9)
    assert 0.15 < res.threshold < 0.3


def test_otsu_uses_exact_bin_sums_not_bin_centres():
    # With 4 bins, bin centres put the best split at 0.5; the exact sums put it at 0.75.
    v = np.array([0.0, 0.31, 0.36, 0.49, 0.53, 0.89, 0.93, 1.0])
    res = otsu_threshold(v, bins=4)
    assert res.threshold == pytest.approx(0.71)
    assert res.high_mode_share == pytest.approx(0.375)


def test_otsu_values_on_bin_edges():
    # With 4 bins these values rescale exactly onto the histogram edges 0, 0.25, ..., 1, so the
    # low class must use the same half-open bins as the histogram: u < edge, not u <= edge.
    v = np.repeat(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), [40, 10, 5, 10, 35])
    res = otsu_threshold(v, bins=4)
    etas = {}
    for e in (1.0, 2.0, 3.0):
        lo, hi = v[v < e], v[v >= e]
        etas[e] = lo.size * hi.size / v.size**2 * (hi.mean() - lo.mean()) ** 2 / v.var()
    assert max(etas, key=etas.__getitem__) == 2.0
    assert res.eta == pytest.approx(etas[2.0], rel=1e-12)
    assert res.threshold == 1.5
    assert res.low_mode_share == res.high_mode_share == 0.5
    assert np.mean(v > res.threshold) == res.high_mode_share


@pytest.mark.parametrize("chunk", [3, 1000])
def test_otsu_chunks_give_the_same_answer(caldor, monkeypatch, chunk):
    dist = cosine_distance_map(caldor[2020], caldor[2022])
    rng = np.random.default_rng(54)
    noisy = np.concatenate([rng.normal(0.05, 0.02, 3000), rng.normal(0.4, 0.05, 900)]).reshape(60, 65)
    noisy[rng.random(noisy.shape) < 0.05] = np.nan
    edges = np.repeat(np.arange(5.0), [40, 10, 5, 10, 35]).reshape(5, 20).astype(np.float32)
    cases = [(dist, 256), (noisy, 256), (noisy.T, 64), (edges, 4), (np.array([-1e308, 1e308, 1e308]), 256)]
    whole = [otsu_threshold(v, bins=bins) for v, bins in cases]
    monkeypatch.setattr(metrics, "_OTSU_CHUNK", chunk)
    for (v, bins), w in zip(cases, whole, strict=True):
        r = otsu_threshold(v, bins=bins)
        assert (r.threshold, r.low_mode_share, r.high_mode_share, r.bimodal) == (
            w.threshold,
            w.low_mode_share,
            w.high_mode_share,
            w.bimodal,
        )
        assert r.eta == pytest.approx(w.eta, rel=1e-12)


def test_otsu_streams_large_inputs():
    # 4 M float32 values (16 MB). Holding even one float64 copy of them would take 32 MB; the
    # chunked passes need a few MB whatever the input size.
    rng = np.random.default_rng(53)
    v = rng.random(4 * 2**20, dtype=np.float32)
    half = v.size // 2
    v[:half] *= 0.2
    v[half:] = 0.6 + 0.4 * v[half:]
    v[::97] = np.nan
    peak = traced_peak(lambda: otsu_threshold(v))
    assert peak < v.nbytes
    res = otsu_threshold(v)
    assert res.bimodal and 0.2 < res.threshold < 0.6
    assert res.high_mode_share == pytest.approx(0.5, abs=0.001)


def test_otsu_eta_on_unimodal_distributions():
    rng = np.random.default_rng(40)
    normal = otsu_threshold(rng.normal(0.0, 1.0, 400_000))
    assert normal.threshold == pytest.approx(0.0, abs=0.05)
    assert normal.eta == pytest.approx(2 / math.pi, abs=0.005)
    uniform = otsu_threshold(rng.uniform(0.0, 1.0, 400_000))
    assert uniform.threshold == pytest.approx(0.5, abs=0.01)
    assert uniform.eta == pytest.approx(0.75, abs=0.005)
    assert not otsu_threshold(rng.normal(0.0, 1.0, 10_000), min_eta=0.7).bimodal


def test_otsu_small_high_mode_is_not_bimodal():
    v = np.concatenate([np.full(9990, 0.05), np.full(10, 0.6)])
    v[:5000] += 0.01
    res = otsu_threshold(v)
    assert res.high_mode_share == pytest.approx(0.001)
    assert res.eta > 0.9
    assert not res.bimodal
    assert otsu_threshold(v, min_share=0.001).bimodal
    assert not otsu_threshold(v, min_share=0.001, min_eta=0.999).bimodal


def test_otsu_extreme_ranges():
    one = 1.0
    nxt = float(np.nextafter(one, 2.0))
    res = otsu_threshold(np.array([one, one, nxt]))
    assert res.threshold == one
    assert res.high_mode_share == pytest.approx(1 / 3)
    assert res.eta == pytest.approx(1.0)
    huge = otsu_threshold(np.array([-1e308, -1e308, 1e308, 1e308, 1e308]))
    assert -1e308 <= huge.threshold < 1e308
    assert huge.high_mode_share == pytest.approx(0.6)
    assert huge.eta == pytest.approx(1.0)
    tiny = otsu_threshold(np.array([0.0, 5e-324, 5e-324]))
    assert tiny.threshold == 0.0
    assert tiny.high_mode_share == pytest.approx(2 / 3)
    shifted = otsu_threshold(np.array([0.0, 0.1, 0.2, 0.9, 1.0]) * 1e-9 + 1e6)
    assert shifted.high_mode_share == pytest.approx(0.4)
    assert 1e6 + 0.2e-9 < shifted.threshold < 1e6 + 0.9e-9


@pytest.mark.parametrize("values", [np.array([]), np.array([3.0, 3.0, 3.0]), np.array([np.nan, 1.0, np.inf])])
def test_otsu_needs_two_distinct_values(values):
    with pytest.raises(ValueError, match="two distinct"):
        otsu_threshold(values)


def test_otsu_rejects_too_few_bins():
    with pytest.raises(ValueError, match="bins"):
        otsu_threshold(np.array([0.0, 1.0]), bins=1)


# ---- masks --------------------------------------------------------------------------------


def test_area_to_pixels():
    assert area_to_pixels(1.0) == 100
    assert area_to_pixels(0.5) == 50
    assert area_to_pixels(0.07) == 7
    assert area_to_pixels(0.015) == 2
    assert area_to_pixels(0.0) == 0
    # Areas between whole pixel counts round up, never to the nearest count.
    assert area_to_pixels(0.012) == 2
    assert area_to_pixels(0.025) == 3
    assert area_to_pixels(0.0001) == 1
    assert area_to_pixels(0.094) == 10  # above the 9 pixels a 3 x 3 opening keeps anyway
    assert area_to_pixels(0.105) == 11
    for bad in (-0.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            area_to_pixels(bad)


def test_clean_mask_opening_removes_speckle_and_thin_lines():
    m = np.zeros((20, 20), dtype=bool)
    m[1, 1] = True  # single pixel
    m[5, 2:12] = m[6, 2:12] = True  # two-pixel-wide line
    m[10:13, 10:13] = True  # 3 x 3 square
    out = clean_mask(m, min_pixels=0)
    expected = np.zeros_like(m)
    expected[10:13, 10:13] = True
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(clean_mask(m, min_pixels=0, opening=1), m)
    np.testing.assert_array_equal(clean_mask(m, min_pixels=0, opening=0), m)
    two = clean_mask(m, min_pixels=0, opening=2)
    assert not two[1, 1] and two[5:7, 2:12].all() and two[10:13, 10:13].all()


def test_clean_mask_drops_small_components_at_the_boundary():
    m = np.zeros((12, 12), dtype=bool)
    m[1:4, 1:4] = True  # 9 pixels
    m[6:10, 6:10] = True  # 16 pixels
    out9 = clean_mask(m, min_pixels=9)
    np.testing.assert_array_equal(out9, m)
    out10 = clean_mask(m, min_pixels=10)
    assert not out10[1:4, 1:4].any() and out10[6:10, 6:10].all()
    assert not clean_mask(m, min_pixels=17).any()


def test_clean_mask_uses_eight_connectivity():
    m = np.zeros((10, 10), dtype=bool)
    m[1:4, 1:4] = True
    m[4:7, 4:7] = True  # touches the first square at one corner only
    np.testing.assert_array_equal(clean_mask(m, min_pixels=18), m)
    assert not clean_mask(m, min_pixels=19).any()


def test_clean_mask_keeps_objects_on_the_image_edge_and_leaves_input_alone():
    m = np.zeros((8, 8), dtype=np.uint8)
    m[:3, :] = 1
    before = m.copy()
    out = clean_mask(m, min_pixels=5)
    assert out.dtype == np.bool_
    np.testing.assert_array_equal(out, m.astype(bool))
    np.testing.assert_array_equal(m, before)
    full = np.ones((5, 5), dtype=bool)
    np.testing.assert_array_equal(clean_mask(full, min_pixels=25), full)


def test_clean_mask_returns_a_new_array_for_a_bool_input():
    m = np.ones((4, 4), dtype=bool)
    for kw in ({"min_pixels": 0, "opening": 1}, {"min_pixels": 1, "opening": 0}, {"min_pixels": 5, "opening": 0}):
        out = clean_mask(m, **kw)
        assert not np.shares_memory(out, m)
        out[:] = False
        assert m.all()
    empty = np.zeros((4, 4), dtype=bool)
    assert not np.shares_memory(clean_mask(empty, min_pixels=5, opening=0), empty)


def test_clean_mask_on_parts_drops_a_patch_the_whole_grid_keeps():
    # A 150 pixel patch cut by a zone seam (or a block edge) into 70 + 80 pixels: each part alone is
    # under min_pixels, which is why a multi-zone AOI must be cleaned on one grid (the mosaic).
    m = np.zeros((14, 20), dtype=bool)
    m[2:12, 3:18] = True
    seam = 10
    min_pixels = area_to_pixels(1.0)
    whole = clean_mask(m, min_pixels=min_pixels)
    np.testing.assert_array_equal(whole, m)
    left = clean_mask(m[:, :seam], min_pixels=min_pixels)
    right = clean_mask(m[:, seam:], min_pixels=min_pixels)
    assert int(m[:, :seam].sum()) == 70 and int(m[:, seam:].sum()) == 80
    assert not left.any() and not right.any()
    doc = " ".join((clean_mask.__doc__ or "").split())
    assert "zone seam" in doc and "mosaic" in doc  # the usage note warns about zones, not just blocks


def test_clean_mask_errors():
    with pytest.raises(ValueError, match="2-D"):
        clean_mask(np.zeros((2, 2, 2), dtype=bool), min_pixels=1)
    with pytest.raises(ValueError):
        clean_mask(np.zeros((2, 2), dtype=bool), min_pixels=-1)
    with pytest.raises(ValueError):
        clean_mask(np.zeros((2, 2), dtype=bool), min_pixels=1, opening=-3)


def test_component_sizes():
    m = np.zeros((10, 10), dtype=bool)
    m[0:3, 0:3] = True
    m[7:9, 7:9] = True
    m[5, 0] = True
    sizes = component_sizes(m)
    assert sizes.dtype == np.int64
    assert sorted(sizes.tolist()) == [1, 4, 9]
    assert sizes.sum() == m.sum()
    diag = np.zeros((6, 6), dtype=bool)
    diag[0:2, 0:2] = True
    diag[2:4, 2:4] = True
    assert component_sizes(diag).tolist() == [8]
    assert component_sizes(np.zeros((4, 4), dtype=bool)).size == 0
    with pytest.raises(ValueError, match="2-D"):
        component_sizes(np.zeros(4, dtype=bool))


# ---- PCA ----------------------------------------------------------------------------------


def _known_axes(n: int = 4000, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sd = np.array([10.0, 5.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    return rng.normal(size=(n, 8)) * sd + np.arange(8)


def test_fit_pca_recovers_known_axes():
    x = _known_axes()
    basis = fit_pca(x)
    assert isinstance(basis, PCABasis)
    assert basis.components.shape == (3, 8)
    np.testing.assert_allclose(basis.mean, x.mean(axis=0), rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(basis.components @ basis.components.T, np.eye(3), atol=1e-9)
    for i in range(3):
        assert basis.components[i, i] > 0.99
    xc = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    np.testing.assert_allclose(np.abs(np.sum(vt[:3] * basis.components, axis=1)), 1.0, atol=1e-6)
    proj = xc @ basis.components.T
    np.testing.assert_allclose(basis.lo, np.percentile(proj, 2, axis=0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(basis.hi, np.percentile(proj, 98, axis=0), rtol=1e-5, atol=1e-5)
    assert not basis.mean.flags.writeable and not basis.components.flags.writeable


def test_fit_pca_sign_is_fixed(caldor):
    for win in caldor.values():
        comps = fit_pca(win).components
        for c in comps:
            assert c[int(np.argmax(np.abs(c)))] > 0
    flipped = fit_pca(-_known_axes())
    for c in flipped.components:
        assert c[int(np.argmax(np.abs(c)))] > 0


def test_fit_pca_sample_is_seeded(caldor):
    win = caldor[2020]
    a = fit_pca(win, sample=500, seed=3)
    b = fit_pca(win, sample=500, seed=3)
    c = fit_pca(win, sample=500, seed=4)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.components, b.components)
    np.testing.assert_array_equal(a.lo, b.lo)
    assert not np.array_equal(a.mean, c.mean)
    full0 = fit_pca(win, seed=0)
    full1 = fit_pca(win, seed=1)
    np.testing.assert_array_equal(full0.mean, full1.mean)
    np.testing.assert_allclose(full0.mean, mean_embedding(win)[0], rtol=1e-5, atol=1e-6)


def test_fit_pca_pools_windows_and_respects_masks(caldor):
    win = caldor[2024]
    top, bottom = sub_window(win, slice(0, 30)), sub_window(win, slice(30, 64))
    pooled = fit_pca([top, bottom])
    whole = fit_pca(win)
    np.testing.assert_allclose(pooled.mean, whole.mean, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(pooled.components, whole.components, atol=1e-5)
    np.testing.assert_allclose(pooled.lo, whole.lo, rtol=1e-5)
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :20] = True
    masked = fit_pca(win, mask)
    np.testing.assert_allclose(masked.mean, mean_embedding(win, mask)[0], rtol=1e-5, atol=1e-6)
    win_masked = Window(**{**win.__dict__, "mask": mask})
    np.testing.assert_allclose(fit_pca(win_masked).mean, masked.mean, rtol=1e-12)


def test_fit_pca_subsampled_blocks_pool_like_the_whole(caldor):
    # A streamed AOI over `sample` pixels is always subsampled. Blocks in row order select the same
    # pixels as the whole window, so the pooled basis is identical, not just close.
    win = caldor[2024]
    top, bottom = sub_window(win, slice(0, 30)), sub_window(win, slice(30, 64))
    three = [sub_window(win, slice(0, 7)), sub_window(win, slice(7, 40)), sub_window(win, slice(40, 64))]
    for sample in (1000, 3000):
        whole = fit_pca(win, sample=sample, seed=5)
        for blocks in ([top, bottom], three):
            pooled = fit_pca(blocks, sample=sample, seed=5)
            np.testing.assert_array_equal(pooled.mean, whole.mean)
            np.testing.assert_array_equal(pooled.components, whole.components)
            np.testing.assert_array_equal(pooled.lo, whole.lo)
            np.testing.assert_array_equal(pooled.hi, whole.hi)


def test_fit_pca_uses_dequantized_vectors():
    emb = random_emb((6, 6), seed=42, d=8)
    sc1 = np.full((6, 6), 0.01, dtype=np.float32)
    sc2 = sc1.copy()
    sc2[:3] = 0.05
    b1 = fit_pca(make_window(emb, sc1))
    b2 = fit_pca(make_window(emb, sc2))
    deq = emb.astype(np.float64) * sc2[..., None].astype(np.float64)
    np.testing.assert_allclose(b2.mean, deq.reshape(-1, 8).mean(axis=0), rtol=1e-6)
    assert not np.allclose(b1.mean * 1.0, b2.mean)


def test_fit_pca_errors():
    emb = random_emb((3, 3), seed=43)
    with pytest.raises(ValueError, match="No valid pixels"):
        fit_pca(make_window(emb, np.full((3, 3), np.nan, dtype=np.float32)))
    with pytest.raises(ValueError, match="at least 3"):
        fit_pca(np.ones((5, 2)))
    with pytest.raises(ValueError, match="sample"):
        fit_pca(emb, sample=0)
    with pytest.raises(ValueError, match="lengths differ"):
        fit_pca([make_window(emb), make_window(random_emb((3, 3), seed=44, d=8))])
    with pytest.raises(ValueError):
        fit_pca(np.ones(8))


def test_pca_rgb_stretch_and_invalid_pixels(caldor):
    win = caldor[2020]
    rgb = pca_rgb(win)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)
    for ch in range(3):
        assert 0.005 < (rgb[..., ch] == 0).mean() < 0.04
        assert 0.005 < (rgb[..., ch] == 255).mean() < 0.04
    scales = win.scales.copy()
    scales[5, 6] = np.nan
    holed = Window(**{**win.__dict__, "scales": scales})
    mask = np.ones((64, 64), dtype=bool)
    mask[10:12, :] = False
    basis = fit_pca(win)
    out = pca_rgb(holed, basis, mask=mask)
    assert (out[5, 6] == 0).all()
    assert (out[10:12] == 0).all()
    keep = mask.copy()
    keep[5, 6] = False
    np.testing.assert_array_equal(out[keep], pca_rgb(win, basis)[keep])
    win_masked = Window(**{**win.__dict__, "mask": mask})
    np.testing.assert_array_equal(pca_rgb(win_masked, basis), pca_rgb(win, basis, mask=mask))


def test_pca_rgb_matches_manual_projection(caldor):
    win = caldor[2022]
    basis = fit_pca(caldor[2020])
    rgb = pca_rgb(win, basis)
    x = win.dequantize().astype(np.float64).reshape(-1, D)
    p = (x - basis.mean) @ basis.components.T
    expected = np.clip(np.rint((p - basis.lo) / (basis.hi - basis.lo) * 255), 0, 255).astype(np.uint8)
    diff = np.abs(rgb.reshape(-1, 3).astype(int) - expected.astype(int))
    assert diff.max() <= 1
    assert (diff == 0).mean() > 0.999


def test_pca_rgb_shared_basis_gives_comparable_colours(caldor):
    basis = fit_pca([caldor[2020], caldor[2022]])
    before = pca_rgb(caldor[2020], basis)
    # A window whose pixels are the 2020 vectors at a different quantisation scale looks identical.
    scaled = Window(**{**caldor[2022].__dict__, "embeddings": caldor[2020].embeddings, "scales": caldor[2020].scales})
    np.testing.assert_array_equal(pca_rgb(scaled, basis), before)
    after = pca_rgb(caldor[2022], basis)
    burned = cosine_distance_map(caldor[2020], caldor[2022]) > 0.3
    change = np.abs(after.astype(int) - before.astype(int)).sum(axis=-1)
    assert np.median(change[burned]) > np.median(change[~burned])


def test_pca_rgb_strips_give_the_same_answer(caldor, monkeypatch):
    win = caldor[2021]
    basis = fit_pca(win)
    whole = pca_rgb(win, basis)
    monkeypatch.setattr(metrics, "_STRIP_ELEMENTS", 1000)
    np.testing.assert_array_equal(pca_rgb(win, basis), whole)


def test_pca_rgb_single_valid_pixel_and_arrays():
    emb = np.zeros((3, 3, 8), dtype=np.int8)
    emb[1, 1] = [5, -3, 2, 0, 1, 1, 0, 7]
    win = make_window(emb)
    basis = fit_pca(win)
    rgb = pca_rgb(win, basis)
    np.testing.assert_array_equal(rgb[1, 1], [128, 128, 128])
    rgb[1, 1] = 0
    assert not rgb.any()
    x = _known_axes(500)
    arr = x.reshape(20, 25, 8).copy()
    arr[0, 0, 3] = np.nan
    out = pca_rgb(arr)
    assert out.shape == (20, 25, 3)
    assert (out[0, 0] == 0).all()
    mask = np.ones((20, 25), dtype=bool)
    mask[3] = False
    assert (pca_rgb(arr, mask=mask)[3] == 0).all()


def test_pca_rgb_auto_basis_is_fitted_on_the_mask(caldor):
    win = caldor[2022]
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :24] = True
    auto = pca_rgb(win, mask=mask)
    np.testing.assert_array_equal(auto, pca_rgb(win, fit_pca(win, mask), mask=mask))
    assert not np.array_equal(auto[mask], pca_rgb(win, fit_pca(win), mask=mask)[mask])
    arr = win.dequantize()
    np.testing.assert_allclose(fit_pca(arr, mask).mean, arr[mask].astype(np.float64).mean(axis=0), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(pca_rgb(arr, mask=mask), pca_rgb(arr, fit_pca(arr, mask), mask=mask))
    # The window's own mask drives the automatic basis too, alone or with an explicit mask.
    owned = dataclasses.replace(win, mask=mask)
    np.testing.assert_array_equal(pca_rgb(owned), auto)
    rows = np.zeros((64, 64), dtype=bool)
    rows[10:50] = True
    np.testing.assert_array_equal(pca_rgb(owned, mask=rows), pca_rgb(win, fit_pca(win, mask & rows), mask=mask & rows))


def test_pca_rgb_rejects_a_basis_of_another_length(caldor):
    basis = fit_pca(_known_axes())
    with pytest.raises(ValueError, match="dimensions"):
        pca_rgb(caldor[2020], basis)
