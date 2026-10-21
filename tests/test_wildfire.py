"""The wildfire app's parts: recovery metrics, threshold choice, the radar regime guard, multi-zone runs.

Two stores are used. The session boundary store of ``tests/conftest.py`` (random vectors in 10N and
11N around 120 W, radar-free in 2022 to 2024, 11N missing 2023) tests the radar guard and the year
checks. A synthetic store built here has a burn scar that crosses the 10N/11N seam, recovers on a
known schedule, and a strip of 10N rows that loses radar in 2022, so the multi-zone outputs, the
validation and the mismatch count can be checked against exact expectations.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
import zarr
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box, mapping
from zarr.storage import MemoryStore

from tessera_apps import export, metrics, validate
from tessera_apps.apps import _common
from tessera_apps.apps import wildfire as wf
from tessera_apps.apps._common import AppInputError, StrictValidationError
from tessera_apps.apps.wildfire import (
    MAX_VALLEY_RATIO,
    WildfireError,
    default_fire_year,
    memory_estimate,
    recovery_index,
    recovery_metrics,
    recovery_reference,
    run_wildfire,
    score_against_reference,
    select_threshold,
    valley_ratio,
)
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app
from tessera_apps.demo import demo_feature
from tessera_apps.io import FIRST_YEAR, YEARS, TesseraReader, YearsNotPublishedError, plan_aoi, zone_epsg

from .conftest import BOUNDARY, KeyLog, PlainCliRunner, add_zone

LON0, LAT0 = BOUNDARY

# --------------------------------------------------------------------------------------------
# Recovery metrics
# --------------------------------------------------------------------------------------------


def test_recovery_index_values_clip_and_guard():
    s_post = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.88, 0.881, 0.9, np.nan, 0.5])
    s_t = np.array([0.5, 0.7, 0.9, 1.0, 0.0, 0.89, 0.89, 0.95, 0.7, np.nan])
    r = recovery_index(s_t, s_post, 0.9)
    assert r.dtype == np.float32
    np.testing.assert_allclose(r[:3], [0.0, 0.5, 1.0], atol=1e-6)
    assert r[3] == pytest.approx(1.25)
    assert r[4] == -1.0  # (0 - 0.5) / 0.4 = -1.25, clipped
    # Denominator 0.9 - 0.88 = 0.02 is defined, 0.019 is not, and a negative one is not.
    assert r[5] == pytest.approx(0.5, abs=1e-5)
    assert np.isnan(r[6]) and np.isnan(r[7])
    assert np.isnan(r[8]) and np.isnan(r[9])
    assert recovery_index(np.array([3.0]), np.array([0.0]), 1.0)[0] == 2.0  # clipped at 2


def test_recovery_index_guard_is_configurable():
    r = recovery_index(np.array([0.95]), np.array([0.85]), 0.9, min_denominator=0.1)
    assert np.isnan(r[0])
    assert recovery_index(np.array([0.95]), np.array([0.85]), 0.9, min_denominator=0.01)[0] == 2.0


def test_recovery_metrics_calibrates_against_unburned_land():
    # Unburned land is itself only about 0.85 to 0.95 similar to its own earlier vector.
    unburned = np.linspace(0.80, 0.95, 101)
    p10 = np.percentile(unburned, 10)
    burned = np.array([0.82, 0.85, 0.70, 0.95, np.nan])
    post = np.array([0.40, 0.40, 0.40, 0.40, 0.40])
    m = recovery_metrics(burned, post, unburned)
    assert m["n_burned"] == 4 and m["n_unburned"] == 101
    assert m["unburned_sim_p10"] == pytest.approx(p10)
    # Recovered means inside normal year-to-year variation: three of four reach p10 = 0.815 ...
    assert m["recovered_fraction"] == pytest.approx(0.75)
    # ... while the brief's fixed 0.9 counts only one.
    assert m["sim90_fraction"] == pytest.approx(0.25)
    assert m["median_sim_burned"] == pytest.approx(np.median([0.82, 0.85, 0.70, 0.95]))
    assert m["median_sim_unburned"] == pytest.approx(0.875)
    expected_r = (np.array([0.82, 0.85, 0.70, 0.95]) - 0.4) / (0.875 - 0.4)
    assert m["median_recovery_index"] == pytest.approx(float(np.median(np.clip(expected_r, -1, 2))))
    assert m["n_recovery_index_undefined"] == 0
    # The brief's level is a parameter.
    assert recovery_metrics(burned, post, unburned, recovery_sim=0.8)["sim90_fraction"] == 0.75


def test_recovery_metrics_undefined_index_and_empty_inputs():
    unburned = np.full(10, 0.9)
    m = recovery_metrics(np.array([0.95, 0.6]), np.array([0.89, 0.3]), unburned)
    assert m["n_recovery_index_undefined"] == 1
    assert m["median_recovery_index"] == pytest.approx((0.6 - 0.3) / (0.9 - 0.3))
    empty = recovery_metrics(np.array([]), np.array([]), np.array([]))
    assert empty["n_burned"] == 0 and empty["n_unburned"] == 0
    for key in ("median_sim_burned", "recovered_fraction", "median_recovery_index", "sim90_fraction"):
        assert math.isnan(empty[key])
    no_ref = recovery_metrics(np.array([0.5]), np.array([0.2]), np.array([np.nan]))
    assert math.isnan(no_ref["recovered_fraction"]) and math.isnan(no_ref["median_recovery_index"])
    with pytest.raises(ValueError, match="s_post_burned"):
        recovery_metrics(np.array([0.5, 0.6]), np.array([0.2]), unburned)


# --------------------------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------------------------


def _bimodal(n: int = 20_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0.08, 0.02, int(n * 0.7)), rng.normal(0.45, 0.06, n - int(n * 0.7))])


def _unimodal(n: int = 20_000, seed: int = 1) -> np.ndarray:
    # Stable land: a skewed, single population of small distances.
    return np.random.default_rng(seed).lognormal(np.log(0.05), 0.5, n)


def test_valley_ratio_tells_a_valley_from_a_single_population():
    assert valley_ratio(_bimodal(), 0.25) < 0.2
    assert valley_ratio(_unimodal(), 0.06) == pytest.approx(1.0)
    assert valley_ratio(np.array([0.1]), 0.1) == 1.0
    assert valley_ratio(np.full(100, 0.3), 0.3) == 1.0


def test_select_threshold_uses_otsu_when_bimodal():
    called = []
    choice = select_threshold(
        _bimodal(), stable_pair=(2019, 2020), stable_values=lambda: called.append(1) or _unimodal()
    )
    assert choice.method == "otsu" and choice.bimodal and not choice.fallback_ran and choice.fallback_available
    assert called == []
    assert 0.15 < choice.threshold < 0.35
    assert choice.threshold == choice.otsu.threshold


def test_select_threshold_falls_back_to_the_stable_pair():
    values = _unimodal(seed=2)
    stable = _unimodal(seed=3)
    otsu_alone = select_threshold(values)
    assert otsu_alone.otsu is not None and otsu_alone.otsu.bimodal, "Otsu's own flag passes a single population"
    choice = select_threshold(values, stable_pair=(2019, 2020), stable_values=lambda: stable)
    assert choice.method == "stable_p99" and choice.fallback_ran and choice.bimodal is False
    assert choice.threshold == pytest.approx(float(np.percentile(stable, 99)))
    assert choice.stable_p99 == choice.threshold and choice.stable_pair == (2019, 2020)
    assert choice.n_stable_values == stable.size
    assert choice.record()["method"] == "stable_p99"


def test_select_threshold_without_a_fallback_keeps_otsu_with_a_note():
    choice = select_threshold(_unimodal())
    assert choice.method == "otsu" and choice.bimodal is False and not choice.fallback_available
    assert "no stable-pair fallback" in choice.note
    empty_pair = select_threshold(_unimodal(), stable_values=lambda: np.array([np.nan]))
    assert empty_pair.method == "otsu" and "no comparable pixel" in empty_pair.note
    # The pass ran and found nothing to compare: the note must not say there was no fallback.
    assert empty_pair.fallback_ran and "no stable-pair fallback" not in empty_pair.note
    assert empty_pair.n_stable_values is None and empty_pair.fallback_suspect is None


def _shoulder(seed: int = 7) -> np.ndarray:
    # A narrow mode of unchanged pixels and a broad, skewed mode of burned ones (the Caldor fixture
    # with year_b as the fire year looks like this): Otsu's split lands on the high mode's shoulder.
    rng = np.random.default_rng(seed)
    low = np.abs(rng.normal(0.012, 0.012, 2000))
    high = rng.gamma(4.0, 0.035, 2000) + 0.01
    return np.concatenate([low, high])


def _density_at(values: np.ndarray, threshold: float) -> float:
    """Smoothed density at the threshold over the lower class peak (the check the valley replaced)."""
    lo, hi = np.percentile(values, [0.5, 99.5])
    bins = int(np.clip(round(math.sqrt(values.size) / 2), 16, 128))
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    smooth = ndimage.gaussian_filter1d(counts.astype(float), 1.0, mode="nearest")
    k = int(np.searchsorted(edges, threshold, side="right") - 1)
    return float(smooth[k] / min(smooth[: k + 1].max(), smooth[k:].max()))


def test_valley_ratio_finds_the_valley_when_otsu_sits_on_a_shoulder():
    values = _shoulder()
    otsu = metrics.otsu_threshold(values)
    assert otsu.bimodal and _density_at(values, otsu.threshold) > MAX_VALLEY_RATIO, "the scenario"
    assert valley_ratio(values, otsu.threshold) < 0.75
    called = []
    choice = select_threshold(values, stable_pair=(2019, 2020), stable_values=lambda: called.append(1) or values)
    assert choice.method == "otsu" and choice.bimodal and called == []
    # A single population still has no valley between its "peaks", wherever the split falls.
    for t in (0.03, 0.05, 0.08, 0.12):
        assert valley_ratio(_unimodal(), t) == pytest.approx(1.0, abs=0.02)


def test_select_threshold_reports_a_fallback_that_flags_almost_nothing():
    values = _unimodal(seed=4)
    noisy_pair = _unimodal(seed=5) * 3.0  # a "stable" pair that varies more than the pair tested
    suspect = select_threshold(values, stable_pair=(2019, 2020), stable_values=lambda: noisy_pair)
    assert suspect.method == "stable_p99" and suspect.threshold == pytest.approx(np.percentile(noisy_pair, 99))
    assert suspect.fallback_suspect is True and suspect.fallback_flagged_share < 0.005
    assert suspect.record()["fallback_suspect"] is True
    quiet_pair = _unimodal(seed=6) * 0.8
    fine = select_threshold(values, stable_pair=(2019, 2020), stable_values=lambda: quiet_pair)
    assert fine.fallback_suspect is False
    assert fine.fallback_flagged_share == pytest.approx(float(np.mean(values > fine.threshold)))


def test_select_threshold_override_skips_the_fallback():
    def boom() -> np.ndarray:
        raise AssertionError("the fallback must not run with an override")

    choice = select_threshold(_unimodal(), override=0.3, stable_values=boom)
    assert choice.method == "override" and choice.threshold == 0.3 and choice.otsu is not None
    constant = select_threshold(np.full(50, 0.2), override=0.5)
    assert constant.method == "override" and constant.otsu is None
    with pytest.raises(WildfireError, match="--threshold"):
        select_threshold(np.full(50, 0.2))
    with pytest.raises(WildfireError):
        select_threshold(np.array([]))
    fallback = select_threshold(np.full(50, 0.2), stable_values=lambda: np.linspace(0, 1, 101))
    assert fallback.method == "stable_p99" and fallback.threshold == pytest.approx(0.99)


# --------------------------------------------------------------------------------------------
# Arguments and the up-front checks
# --------------------------------------------------------------------------------------------


@pytest.fixture
def no_store(monkeypatch):
    def refuse(**kwargs):
        raise AssertionError("the store must not be opened")

    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(refuse))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"year_a": 2022, "year_b": 2022}, "must be after year_a"),
        ({"year_a": 2022, "year_b": 2020}, "must be after year_a"),
        ({"year_a": 2016, "year_b": 2018}, "2017 to 2025"),
        ({"recovery_through": 2026}, "recovery_through"),
        ({"recovery_through": 2021}, "recovery_through"),
        ({"fire_year": 2019}, "fire_year"),
        ({"fire_year": 2023}, "fire_year"),
        ({"threshold": 0.0}, "threshold"),
        ({"threshold": float("nan")}, "threshold"),
        ({"min_area_ha": -1.0}, "min_area_ha"),
        ({"recovery_sim": 1.5}, "recovery_sim"),
        ({"reference": "missing.geojson"}, "does not exist"),
    ],
)
def test_bad_arguments_fail_before_the_store_opens(no_store, tmp_path, kwargs, message):
    args = {"year_a": 2020, "year_b": 2022, **kwargs}
    with pytest.raises(AppInputError, match=message):
        run_wildfire("caldor", args.pop("year_a"), args.pop("year_b"), tmp_path, **args)


def _two_zone_aoi(tmp_path: Path, geometry=None) -> Path:
    geometry = geometry or box(LON0 - 0.004, LAT0 - 0.003, LON0 + 0.003, LAT0 + 0.003)
    feature = {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
    fc = {"type": "FeatureCollection", "features": [feature]}
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(fc))
    return path


def test_recovery_years_are_checked_before_any_chunk_read(boundary_store, boundary_reader, tmp_path):
    # 11N has no 2023. A run whose recovery reaches 2023 must stop before reading anything.
    _, log = boundary_store
    with pytest.raises(YearsNotPublishedError) as info:
        run_wildfire(_two_zone_aoi(tmp_path), 2020, 2021, tmp_path / "out", threshold=0.5, reader=boundary_reader)
    assert info.value.missing == {"11N": [2023]}
    assert log.chunk_keys() == []
    assert not (tmp_path / "out").exists()


def test_the_fallback_year_is_checked_up_front_and_explained(boundary_store, boundary_reader, tmp_path):
    _, log = boundary_store
    aoi = _two_zone_aoi(tmp_path)
    with pytest.raises(YearsNotPublishedError) as info:
        run_wildfire(aoi, 2024, 2025, tmp_path / "out", reader=boundary_reader)
    assert info.value.missing == {"11N": [2023]}
    assert any("--threshold" in note for note in info.value.__notes__)
    assert log.chunk_keys() == []
    # With a threshold there is no fallback, so 2023 is not needed.
    result = run_wildfire(aoi, 2024, 2025, tmp_path / "out", threshold=0.5, reader=boundary_reader)
    assert 2023 not in result.report["provenance"]["years"]


def test_every_argument_is_in_the_digest(monkeypatch, tmp_path):
    seen = []

    class StopRunError(Exception):
        pass

    def capture(app, aoi, years_read, year_a, year_b, out_dir, params, **kwargs):
        seen.append((params, sorted(years_read)))
        raise StopRunError

    monkeypatch.setattr(wf, "prepare_run", capture)
    ref = _two_zone_aoi(tmp_path)
    other_ref = tmp_path / "other.geojson"
    other_ref.write_text(ref.read_text().replace("{}", '{"x": 1}'))
    variants = [
        {},
        {"threshold": 0.3},
        {"min_area_ha": 2.0},
        {"fire_year": 2020},
        {"fire_name": "Caldor"},
        {"reference": ref},
        {"reference": other_ref},
        {"recovery_through": 2024},
        {"recovery_sim": 0.8},
        {"block": (256, 256)},
    ]
    for v in variants:
        with pytest.raises(StopRunError):
            run_wildfire("caldor", 2020, 2022, tmp_path, **v)
    digests = {_common.export.run_args_digest(params, years=years, aoi_sha256="x", seed=0) for params, years in seen}
    assert len(digests) == len(variants)


# --------------------------------------------------------------------------------------------
# Radar regime guard (survey B5), on the boundary store
# --------------------------------------------------------------------------------------------


def test_pixels_that_switch_radar_regime_are_not_classified(boundary_reader, tmp_path):
    aoi = _two_zone_aoi(tmp_path)
    # Random vectors every year: every pixel is far from its earlier self, so a 0.5 threshold
    # burns everything that is compared at all.
    same = run_wildfire(
        aoi, 2020, 2021, tmp_path / "same", threshold=0.5, recovery_through=2022, reader=boundary_reader
    )
    assert same.summary["radar"]["radar_mismatch_fraction"] == 0.0
    assert same.summary["burned_fraction"] > 0.95
    # Recovery years get the same guard: 2022 lost radar, so no burned pixel is compared with 2020.
    assert same.summary["radar"]["radar_mismatch_fraction_by_year"] == {"2021": 0.0, "2022": 1.0}
    rows = {r["year"]: r for r in same.summary["recovery"]}
    assert rows[2021]["n_burned"] == same.summary["burned_pixels"] > 0
    assert rows[2022]["n_burned"] == 0 and rows[2022]["n_unburned"] == 0
    assert rows[2022]["radar_mismatch_fraction"] == 1.0
    with rasterio.open(same.paths.tif) as ds:
        assert ds.descriptions[-1] == "recovery_index_2022" and np.isnan(ds.read(4)).all()
    # 2021 has radar and 2022 does not: every pixel switches regime, so none is classified.
    switch = run_wildfire(
        aoi, 2021, 2022, tmp_path / "switch", threshold=0.5, recovery_through=2022, reader=boundary_reader
    )
    s = switch.summary
    assert s["radar"]["radar_mismatch_fraction"] == 1.0
    assert s["radar"]["radar_mismatch_pixels"] == s["valid_pixels_year_a"] > 0
    assert s["classified_pixels"] == 0 and s["burned_pixels"] == 0 and s["burned_area_ha"] == 0
    assert any("switch radar regime" in w for w in switch.report["warnings"])
    with rasterio.open(switch.paths.tif) as ds:
        assert np.isnan(ds.read(1)).all() and np.isnan(ds.read(2)).all()


def test_no_comparable_pixel_without_a_threshold_is_an_error(boundary_reader, tmp_path):
    with pytest.raises(WildfireError, match="switch regime"):
        run_wildfire(
            _two_zone_aoi(tmp_path), 2021, 2022, tmp_path / "out", recovery_through=2022, reader=boundary_reader
        )


# --------------------------------------------------------------------------------------------
# A synthetic scar across the 10N/11N seam
# --------------------------------------------------------------------------------------------

SIZE = 128
BURN = box(LON0 - 0.003, LAT0 - 0.0015, LON0 + 0.002, LAT0 + 0.0015)
SCAR_AOI = box(LON0 - 0.0045, LAT0 - 0.0028, LON0 + 0.0035, LAT0 + 0.0028)
#: Weight of the pre-fire vector in burned pixels each year (fire in 2021, full recovery by 2025).
PRE_FIRE_WEIGHT = {2021: 0.0, 2022: 0.0, 2023: 0.3, 2024: 0.6, 2025: 1.0}
#: Rows of the 10N group (from its top) that have no radar in 2022 only.
RADAR_GAP_ROWS = 40
MIN_AREA_HA = 10.0


def _scar_fill(zone: str, left: float, top: float, seed: int):
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    lon, lat = Transformer.from_crs(zone_epsg(zone), 4326, always_xy=True).transform(
        left + (cols + 0.5) * 10.0, top - (rows + 0.5) * 10.0
    )
    burn = shapely.contains_xy(BURN, lon, lat)
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(SIZE, SIZE, 128))
    fire = rng.normal(size=(SIZE, SIZE, 128))
    scales: dict[int, np.ndarray] = {}

    def fill(name, ti, shape):
        year = FIRST_YEAR + ti
        if name == "embeddings":
            w = np.where(burn, PRE_FIRE_WEIGHT.get(year, 1.0), 1.0)[..., None]
            noise = np.random.default_rng((seed, ti)).normal(size=base.shape)
            v = w * base + (1.0 - w) * fire + 0.25 * noise
            scale = np.abs(v).max(axis=-1) / 127.0
            scales[ti] = scale.astype(np.float32)
            return np.clip(np.rint(v / scale[..., None]), -127, 127).astype(np.int8)
        if name == "scales":
            return scales[ti]
        if name == "s1_asc_obs_count":
            radar = np.full(shape, 30, dtype=np.uint16)
            if zone == "10N" and year == 2022:
                radar[:RADAR_GAP_ROWS] = 0
            return radar
        if name == "s1_desc_obs_count":
            return np.zeros(shape, dtype=np.uint16)
        return np.full(shape, 50, dtype=np.uint16)

    return fill


@pytest.fixture(scope="module")
def scar_store() -> MemoryStore:
    mem = MemoryStore()
    root = zarr.open_group(mem, mode="w")
    for zone, seed in (("10N", 10), ("11N", 11)):
        x, y = Transformer.from_crs(4326, zone_epsg(zone), always_xy=True).transform(*BOUNDARY)
        left = (x // 10) * 10 - SIZE / 2 * 10
        top = (y // 10) * 10 + SIZE / 2 * 10
        add_zone(root, zone, left=left, top=top, height=SIZE, width=SIZE, fill=_scar_fill(zone, left, top, seed))
    return mem


@pytest.fixture(scope="module")
def scar_run(scar_store, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scar")
    aoi = _two_zone_aoi(tmp, SCAR_AOI)
    ref = tmp / "burn.geojson"
    ref.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(BURN)}))
    reader = TesseraReader.from_store(KeyLog(scar_store))
    result = run_wildfire(
        aoi,
        2020,
        2022,
        tmp / "out",
        min_area_ha=MIN_AREA_HA,
        validate=True,
        reference=ref,
        reader=reader,
        block=(64, 64),
    )
    plan = plan_aoi(reader, SCAR_AOI)
    return result, plan, reader


def test_scar_store_preconditions(scar_run):
    _, plan, _ = scar_run
    assert plan.zone_names == ["10N", "11N"]
    min_pixels = round(MIN_AREA_HA / 0.01)
    parts = {}
    for z in plan.zones:
        parts[z.zone] = int(validate.reference_from_geometries(z, [BURN]).sum())
    # Each zone's part of the scar is under the minimum area; the whole scar is over it.
    assert all(0 < n < min_pixels for n in parts.values()), parts
    assert sum(parts.values()) > min_pixels


def test_scar_across_the_seam_is_found_whole_and_validated_in_both_zones(scar_run):
    result, plan, _ = scar_run
    s = result.summary
    assert s["threshold"]["method"] == "otsu" and s["threshold"]["bimodal"]
    assert s["zones"] == ["10N", "11N"]
    assert s["aoi_pixels"] == plan.n_pixels
    assert s["burned_fraction"] == s["burned_pixels"] / s["classified_pixels"]
    val = json.loads(result.paths.validation.read_text())
    scores = val["scores"]
    # Scored over every classified pixel of both zones, and the scar is found on both sides.
    assert scores["burned"]["n"] == s["classified_pixels"] == plan.n_pixels - s["radar"]["radar_mismatch_pixels"]
    assert set(scores["per_zone"]) == {"10N", "11N"}
    for zone in ("10N", "11N"):
        part = scores["per_zone"][zone]
        assert part["tp"] > 0 and part["recall"] > 0.9, (zone, part)
    assert sum(p["n"] for p in scores["per_zone"].values()) == scores["burned"]["n"]
    assert scores["burned"]["iou"] > 0.9
    assert scores["recall_inside_reference"] > 0.9
    assert scores["burned_outside_reference_fraction"] < 0.05
    assert scores["iou_below_0_6"] is False
    assert any("outline" in c for c in val["caveats"])
    assert s["burned_pixels"] == scores["burned"]["tp"] + scores["burned"]["fp"]
    # burn.geojson is not Hansen-derived, so it gets no Hansen caveat.
    assert val["reference"]["hansen_derived"] is False
    assert not any("stand-replacing" in c for c in val["caveats"])
    # Otsu saw the distances of both zones.
    assert s["threshold"]["n_values"] == s["classified_pixels"]


def test_scar_mosaic_holds_every_aoi_pixel_once(scar_run):
    result, plan, _ = scar_run
    s = result.summary
    with rasterio.open(result.paths.tif) as ds:
        assert ds.crs.to_epsg() == zone_epsg("10N")
        burned = ds.read(2)
        recovery = ds.read(6)
    covered = int(np.isfinite(burned).sum())
    expected = s["classified_pixels"]
    assert abs(covered - expected) <= 0.01 * expected
    assert abs(int(np.nansum(burned)) - s["burned_pixels"]) <= 0.02 * s["burned_pixels"]
    assert abs(int(np.isfinite(recovery).sum()) - s["burned_pixels"]) <= 0.02 * s["burned_pixels"]
    with rasterio.open(result.paths.tif) as main:
        crs, transform, shape = main.crs, main.transform, main.shape
    for key in ("pca_a", "pca_b"):
        with rasterio.open(result.extra_paths[key]) as ds:
            rgb = ds.read()
            # The false colour shares the main COG's grid (10N), so the layers overlay.
            assert (ds.crs, ds.transform, ds.shape) == (crs, transform, shape), key
        assert abs(int(rgb.any(axis=0).sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels


def test_scar_recovery_follows_the_schedule(scar_run):
    result, _, _ = scar_run
    rows = {r["year"]: r for r in result.summary["recovery"]}
    assert sorted(rows) == [2022, 2023, 2024, 2025]
    r = [rows[y]["median_recovery_index"] for y in sorted(rows)]
    assert r[0] == 0.0 and r[0] < r[1] < r[2] < r[3]
    assert 0.25 < r[1] < 0.55 and 0.65 < r[2] < 0.95 and r[3] > 0.9
    assert rows[2022]["recovered_fraction"] == 0.0 and rows[2024]["recovered_fraction"] < 0.05
    assert rows[2025]["recovered_fraction"] > 0.8
    # Unburned land stays about 1 / 1.0625 similar to itself, whatever the year.
    for row in rows.values():
        assert 0.9 < row["median_sim_unburned"] < 0.97
    # The reference is far from the scar and was stable before the fire too; its pre-fire check read
    # 2019 on the four sampled blocks, which here are all of them.
    ref = result.summary["recovery_reference"]
    assert ref["pixels"] == rows[2022]["n_unburned"] > 1000
    assert ref["pre_fire_similarity"]["coverage"] == "every block"
    assert 0.9 < ref["pre_fire_similarity"]["median"] < 0.97
    assert result.summary["validation"]["recovery_reference_inside_fraction"] == 0.0


def test_scar_radar_gap_is_counted_and_excluded(scar_run):
    result, plan, _ = scar_run
    radar = result.summary["radar"]
    z10 = next(z for z in plan.zones if z.zone == "10N")
    gap = int(z10.mask[: max(0, RADAR_GAP_ROWS - z10.rows.start)].sum())
    assert gap > 0
    assert radar["radar_mismatch_pixels"] == gap
    assert radar["radar_mismatch_fraction"] == pytest.approx(gap / plan.n_pixels)
    # Only 2022 lost radar: the later recovery years compare like with like again.
    by_year = radar["radar_mismatch_fraction_by_year"]
    assert by_year["2022"] == pytest.approx(gap / plan.n_pixels)
    assert by_year["2023"] == by_year["2024"] == by_year["2025"] == 0.0
    assert radar["radar_free_fraction_in_scar_year_a"] == 0.0


def test_score_against_reference_counts_every_zone(boundary_reader):
    plan = plan_aoi(boundary_reader, box(LON0 - 0.004, LAT0 - 0.003, LON0 + 0.003, LAT0 + 0.003))
    burned = {z.zone: z.mask.copy() for z in plan.zones}
    reference = {z.zone: np.zeros(z.shape, dtype=bool) for z in plan.zones}
    classified = {z.zone: z.mask.copy() for z in plan.zones}
    for z in plan.zones:
        rr, cc = np.nonzero(z.mask)
        reference[z.zone][rr[:100], cc[:100]] = True
        classified[z.zone][rr[:10], cc[:10]] = False  # ten reference pixels left unclassified
    scores = score_against_reference(plan, burned, reference, classified)
    merged = scores["burned"]
    assert merged.n == plan.n_pixels - 20
    assert merged.tp == 180 and merged.fn == 0
    assert scores["reference_pixels"] == 200 and scores["reference_pixels_unclassified"] == 20
    assert scores["recall_inside_reference"] == pytest.approx(1.0)
    assert scores["burned_outside_reference_fraction"] == pytest.approx(1 - 200 / plan.n_pixels)
    assert scores["iou_below_0_6"] is True
    assert set(scores["per_zone"]) == {"10N", "11N"}


def test_boundary_store_years():
    # The session store the radar tests rely on: radar-free 2022 to 2024, 11N missing 2023.
    assert YEARS[5:8] == (2022, 2023, 2024)


# --------------------------------------------------------------------------------------------
# Review fixes
# --------------------------------------------------------------------------------------------


def test_default_fire_year_is_never_year_a():
    assert default_fire_year(2020, 2022) == 2021
    assert default_fire_year(2020, 2021) == 2021  # year_b is the fire year (survey N11 warns)
    assert default_fire_year(2017, 2020) == 2019


def test_recovery_reference_keeps_only_land_shown_unchanged(boundary_reader):
    # Distances on the boundary plan: a burn just east of 120 W (11N), land just under the threshold
    # far from it, speckle over the threshold that cleaning removed, and stable land. The reference
    # keeps stable land only; its buffer around the burn reaches across the seam into 10N, and the
    # speckle carves nothing out.
    plan = plan_aoi(boundary_reader, box(LON0 - 0.004, LAT0 - 0.003, LON0 + 0.003, LAT0 + 0.003))
    burn = box(LON0 + 0.00005, LAT0 - 0.0008, LON0 + 0.0012, LAT0 + 0.0008)
    near_burn = box(LON0 - 0.0005, LAT0 - 0.0008, LON0 - 0.00005, LAT0 + 0.0008)  # 10N, within 50 m
    just_under = box(LON0 - 0.0035, LAT0 + 0.0015, LON0 - 0.0025, LAT0 + 0.0025)
    dist, burned, speckle = {}, {}, {}
    rng = np.random.default_rng(1)
    for z in plan.zones:
        d = np.where(z.mask, 0.05, np.nan).astype(np.float32)
        d[validate.reference_from_geometries(z, [just_under])] = 0.35
        burned[z.zone] = validate.reference_from_geometries(z, [burn])
        d[burned[z.zone]] = 0.6
        speckle[z.zone] = (rng.random(z.shape) < 0.01) & z.mask & ~burned[z.zone] & (d < 0.1)
        d[speckle[z.zone]] = 0.6
        dist[z.zone] = d
    ref = recovery_reference(plan, dist, 0.4, burned)
    for z in plan.zones:
        r = ref[z.zone]
        assert not (r & burned[z.zone]).any() and not (r & speckle[z.zone]).any()
        assert not (r & validate.reference_from_geometries(z, [just_under])).any(), "just under the threshold"
        assert not (r & ~z.mask).any()
    z10 = plan.zones[0]
    assert validate.reference_from_geometries(z10, [near_burn]).sum() > 20
    assert not (ref["10N"] & validate.reference_from_geometries(z10, [near_burn])).any(), "buffer crosses the seam"
    stable = sum(int((np.isfinite(d) & (d < 0.1)).sum()) for d in dist.values())
    assert 0.6 * stable < sum(int(r.sum()) for r in ref.values()) < stable
    # Without a burn nothing is buffered, and the distance cut is half the threshold.
    calm = {z: np.where(np.isfinite(d), np.minimum(d, 0.35), np.nan).astype(np.float32) for z, d in dist.items()}
    none = {z: np.zeros_like(b) for z, b in burned.items()}
    ref = recovery_reference(plan, calm, 0.4, none)
    assert sum(int(r.sum()) for r in ref.values()) == stable


def test_band_mosaic_equals_the_mosaic_of_the_whole_stack(boundary_reader):
    # The main COG is mosaicked through one index band and gathered band by band; the result must be
    # the one-shot mosaic of the whole stack, seam fills included, even for ragged speckle.
    plan = plan_aoi(boundary_reader, box(LON0 - 0.006, LAT0 - 0.005, LON0 + 0.0065, LAT0 + 0.004))
    assert plan.zone_names == ["10N", "11N"]
    rng = np.random.default_rng(0)
    for density in (0.3, 1.0):
        classified = {z.zone: (rng.random(z.shape) < density) & z.mask for z in plan.zones}
        stacks = {}
        for z in plan.zones:
            stack = np.full((5, *z.shape), np.nan, dtype=np.float32)
            stack[0] = np.where(classified[z.zone], rng.random(z.shape), np.nan)
            for k in range(1, 5):
                stack[k] = np.where(classified[z.zone] & (rng.random(z.shape) < 0.5), rng.random(z.shape), np.nan)
            stacks[z.zone] = stack
        whole, transform, crs = export.mosaic_zones(
            [export.GridArray(stacks[z.zone], z) for z in plan.zones], dst_zone="10N", nodata=np.nan
        )
        bands = wf._BandMosaic(plan, "10N", classified, 5)
        for k in (3, 0, 4, 1, 2):
            bands.put(k, {z: stacks[z][k] for z in stacks})
        np.testing.assert_array_equal(bands.data, whole)
        assert bands.transform == transform and bands.crs == crs


def _fake_plan(*shapes):
    return type("Plan", (), {"zones": [type("Zone", (), {"shape": shape})() for shape in shapes]})()


def test_memory_estimate_takes_the_larger_stage():
    plan = _fake_plan((1000, 2000), (500, 500))
    px = 1000 * 2000 + 500 * 500
    block_px = 512 * 2000  # the widest zone's block, trimmed to its window
    stream = (wf.MEMORY_STREAM_PER_PIXEL + 3 * wf.MEMORY_PER_LATER_YEAR_STREAM) * px
    stream += wf.MEMORY_PER_BLOCK_PIXEL_YEAR * block_px * 6
    write = (wf.MEMORY_WRITE_PER_PIXEL + 3 * wf.MEMORY_PER_LATER_YEAR_WRITE) * px
    base = memory_estimate(plan, n_recovery_years=4, years_per_block=6, block=(512, 2048))
    assert base == int(wf.MEMORY_FIXED + wf.MEMORY_MARGIN * max(stream, write))
    assert memory_estimate(plan, n_recovery_years=6, years_per_block=8, block=(512, 2048)) > base
    assert memory_estimate(plan, n_recovery_years=4, years_per_block=6, block=(256, 256)) <= base
    # Caldor's live plan (10N 4348 x 6023, 11N 4202 x 932, 2020 to 2022): about 4.8 GB, under the
    # review's measured 5.9 GB for the code before it, and the write stage sets it.
    caldor = memory_estimate(
        _fake_plan((4348, 6023), (4202, 932)), n_recovery_years=4, years_per_block=6, block=(512, 2048)
    )
    assert 4.0e9 < caldor < 5.5e9


def test_looks_like_hansen_by_name_or_content(tmp_path):
    by_name = tmp_path / "hansen_2021_loss.geojson"
    by_name.write_text('{"type": "FeatureCollection", "features": []}')
    by_content = tmp_path / "outline.geojson"
    by_content.write_text('{"type": "FeatureCollection", "name": "Hansen GFC outline", "features": []}')
    neither = tmp_path / "perimeter.geojson"
    neither.write_text('{"type": "FeatureCollection", "features": []}')
    assert wf._looks_like_hansen(by_name) and wf._looks_like_hansen(by_content)
    assert not wf._looks_like_hansen(neither)


def _ten_n_only(tmp_path: Path) -> Path:
    return _two_zone_aoi(tmp_path, box(LON0 - 0.004, LAT0 - 0.003, LON0 - 0.0005, LAT0 + 0.003))


def test_radar_free_scar_is_measured_flagged_and_warned(boundary_reader, tmp_path):
    # 10N is radar-free in 2022 to 2024 and the vectors are random, so 0.5 burns everything.
    result = run_wildfire(
        _ten_n_only(tmp_path),
        2022,
        2024,
        tmp_path / "out",
        threshold=0.5,
        recovery_through=2024,
        reader=boundary_reader,
    )
    s = result.summary
    radar = s["radar"]
    assert s["burned_pixels"] > 0 and radar["radar_mismatch_pixels"] == 0
    assert radar["radar_free_fraction_in_scar_year_a"] == 1.0 == radar["radar_free_fraction_in_scar_year_b"]
    assert radar["radar_free_flag"] is True
    assert any("radar-free" in w for w in result.report["warnings"])


def test_the_stable_pair_excludes_radar_regime_switches(boundary_reader, tmp_path):
    # year_a 2022 is radar-free and 2021 is not: every stable-pair pixel switches regime, so the
    # fallback has nothing to compare (it must not take the p99 of regime shifts, survey B5).
    result = run_wildfire(
        _ten_n_only(tmp_path), 2022, 2024, tmp_path / "out", recovery_through=2024, reader=boundary_reader
    )
    th = result.summary["threshold"]
    assert th["bimodal"] is False and th["fallback_ran"] is True
    assert th["method"] == "otsu" and th["n_stable_values"] is None
    assert "no comparable pixel" in th["note"]
    ref = result.summary["recovery_reference"]
    assert ref["pre_fire_similarity"]["pixels"] == 0 and ref["pre_fire_similarity"]["median"] is None


def test_multi_zone_validation_is_redone_close_to_a_full_run(scar_store, scar_run, tmp_path):
    # A run without validation, then the same arguments with validation: the validation is redone
    # from the GeoTIFF without reading the store. Across zones a few seam pixels may move.
    full, _, _ = scar_run
    aoi = _two_zone_aoi(tmp_path, SCAR_AOI)
    ref = tmp_path / "burn.geojson"
    ref.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(BURN)}))
    log = KeyLog(scar_store)
    reader = TesseraReader.from_store(log)
    kw = {"min_area_ha": MIN_AREA_HA, "reference": ref, "reader": reader, "block": (64, 64)}
    first = run_wildfire(aoi, 2020, 2022, tmp_path / "out", validate=False, **kw)
    assert not first.validation_written
    reads = len(log.chunk_keys())
    redone = run_wildfire(aoi, 2020, 2022, tmp_path / "out", validate=True, **kw)
    assert redone.validation_redone and len(log.chunk_keys()) == reads
    got = json.loads(redone.paths.validation.read_text())["scores"]
    want = json.loads(full.paths.validation.read_text())["scores"]
    for key in ("tp", "fp", "fn", "tn", "n"):
        assert got["burned"][key] == pytest.approx(want["burned"][key], rel=0.01, abs=5), key
    assert got["burned"]["iou"] == pytest.approx(want["burned"]["iou"], abs=0.01)


# --------------------------------------------------------------------------------------------
# Burn severity (--severity): MTBS dnbr6 codes
# --------------------------------------------------------------------------------------------

#: Pixels of each code laid out in every zone of the boundary plan, and how many of them are burned.
#: Background (0) is mapped land outside the burn; the rest of the AOI is not covered by the map.
LAYOUT = {1: (70, 7), 2: (60, 12), 3: (50, 25), 4: (40, 36), 5: (30, 3), 6: (20, 4), 0: (40, 5)}
#: Pixels of each code left unclassified: the last of their run, never burned.
UNCLASSIFIED = {0: 3, 1: 6, 3: 10, 5: 4, 6: 2}
#: AOI pixels after the laid-out runs are not covered by the map: the first are burned, the last unclassified.
BURNED_UNMAPPED = 5
UNCLASSIFIED_UNMAPPED = 2
UNMAPPED = wf.SEVERITY_UNMAPPED


def _laid_out(plan, layout=LAYOUT, unclassified=UNCLASSIFIED):
    """Severity, burned and classified masks per zone with the known counts of ``layout``."""
    severity, burned, classified = {}, {}, {}
    for z in plan.zones:
        rr, cc = np.nonzero(z.mask)
        assert len(rr) > sum(n for n, _ in layout.values()) + BURNED_UNMAPPED + UNCLASSIFIED_UNMAPPED
        s = np.full(z.shape, UNMAPPED, dtype=np.uint8)
        b = np.zeros(z.shape, dtype=bool)
        v = z.mask.copy()
        start = 0
        for code, (n, n_burned) in layout.items():
            k = unclassified.get(code, 0)
            assert n_burned + k <= n
            s[rr[start : start + n], cc[start : start + n]] = code
            b[rr[start : start + n_burned], cc[start : start + n_burned]] = True
            v[rr[start + n - k : start + n], cc[start + n - k : start + n]] = False
            start += n
        b[rr[start : start + BURNED_UNMAPPED], cc[start : start + BURNED_UNMAPPED]] = True
        v[rr[len(rr) - UNCLASSIFIED_UNMAPPED :], cc[len(rr) - UNCLASSIFIED_UNMAPPED :]] = False
        # Outside the AOI mask nothing counts, whatever the arrays hold.
        s[~z.mask] = 4
        b[~z.mask] = True
        severity[z.zone], burned[z.zone], classified[z.zone] = s, b, v
    return severity, burned, classified


@pytest.fixture
def boundary_plan(boundary_reader):
    plan = plan_aoi(boundary_reader, box(LON0 - 0.004, LAT0 - 0.003, LON0 + 0.003, LAT0 + 0.003))
    assert plan.zone_names == ["10N", "11N"] and all(int(z.mask.sum()) > 400 for z in plan.zones)
    return plan


def test_severity_scores_count_every_class_and_merge_zones(boundary_plan):
    plan = boundary_plan
    severity, burned, classified = _laid_out(plan)
    sc = wf.score_against_severity(plan, burned, severity, classified)
    assert sc["min_class"] == 3 and sc["min_class_name"] == "moderate"
    zones = len(plan.zones)
    px = {c: (n - UNCLASSIFIED.get(c, 0)) * zones for c, (n, _) in LAYOUT.items()}
    hit = {c: k * zones for c, (_, k) in LAYOUT.items()}
    assert set(sc["classes"]) == {"0", "1", "2", "3", "4", "5"}
    for c in (0, 1, 2, 3, 4, 5):
        row = sc["classes"][str(c)]
        assert (row["pixels"], row["detected_pixels"]) == (px[c], hit[c]), c
        assert row["detection_rate"] == pytest.approx(hit[c] / px[c])
        assert row["hectares"] == pytest.approx(px[c] * 0.01)
        assert row["detected_hectares"] == pytest.approx(hit[c] * 0.01)
    # Moderate or worse, over classified mapped pixels in classes 0 to 5: background (0, outside the
    # mapped burn) and increased greenness (5) are unburned land, so detections there are false positives.
    m = sc["min_class_or_worse"]
    ref = px[3] + px[4]
    tp = hit[3] + hit[4]
    fp = hit[0] + hit[1] + hit[2] + hit[5]
    assert (m.tp, m.fn, m.fp) == (tp, ref - tp, fp)
    assert m.n == sum(px[c] for c in (0, 1, 2, 3, 4, 5)) == sc["pixels_scored"]
    assert m.iou == pytest.approx(tp / (tp + fp + ref - tp))
    assert sc["moderate_or_worse"] == m
    low = sc["low_or_worse"]
    ref_low = px[2] + px[3] + px[4]
    tp_low = hit[2] + hit[3] + hit[4]
    assert (low.tp, low.fn, low.fp, low.n) == (tp_low, ref_low - tp_low, hit[0] + hit[1] + hit[5], m.n)
    # Zones are merged by summing counts.
    for key in ("min_class_or_worse", "moderate_or_worse", "low_or_worse"):
        parts = sc["per_zone"][key]
        assert set(parts) == {"10N", "11N"}
        assert validate.merge_scores(list(parts.values())) == sc[key]
        assert all(p.n == m.n // zones for p in parts.values())
    # Pixels the map does not cover and class 6 are excluded and counted.
    n_classified = sum(int(classified[z.zone].sum()) for z in plan.zones)
    unmapped = sum(int(z.mask.sum()) - sum(n for n, _ in LAYOUT.values()) - UNCLASSIFIED_UNMAPPED for z in plan.zones)
    assert sc["excluded"] == {"unmapped_pixels": unmapped, "non_processing_pixels": px[6]}
    assert n_classified == m.n + px[6] + unmapped
    assert sc["unclassified_pixels_in_classes_1_to_5"] == (6 + 10 + 4) * zones
    # Where the detections fall: every burned AOI pixel the app classified, unmapped and class 6 too.
    by_class = sc["detections_by_class"]
    n_burned = sum(hit.values()) + BURNED_UNMAPPED * zones
    assert sc["burned_pixels"] == n_burned
    assert set(by_class) == {"0", "1", "2", "3", "4", "5", "6", "unmapped"}
    assert by_class["unmapped"]["pixels"] == BURNED_UNMAPPED * zones
    for c in (0, 1, 2, 3, 4, 5, 6):
        assert by_class[str(c)]["pixels"] == hit[c]
        assert by_class[str(c)]["share"] == pytest.approx(hit[c] / n_burned)
    assert sum(v["share"] for v in by_class.values()) == pytest.approx(1.0)
    assert sc["stop_rule_min_class"] == 3 and sc["iou_below_0_6"] is (m.iou < 0.6)


def test_severity_min_class_switch_and_classes_0_and_5_are_never_burned(boundary_plan):
    plan = boundary_plan
    severity, burned, classified = _laid_out(plan)
    zones = len(plan.zones)
    default = wf.score_against_severity(plan, burned, severity, classified)
    high = wf.score_against_severity(plan, burned, severity, classified, min_class=4)
    m = high["min_class_or_worse"]
    assert (m.tp, m.fn) == (36 * zones, 4 * zones)
    assert m.fp == (5 + 7 + 12 + 25 + 3) * zones
    low = wf.score_against_severity(plan, burned, severity, classified, min_class=2)
    assert low["min_class_or_worse"] == low["low_or_worse"]
    # Moderate or worse does not move with the min class.
    assert high["moderate_or_worse"] == low["moderate_or_worse"] == default["min_class_or_worse"]
    every = wf.score_against_severity(plan, burned, severity, classified, min_class=1)["min_class_or_worse"]
    # Classes 1 to 4 are burned; background (0) and increased greenness (5) stay unburned even then.
    assert every.tp == (7 + 12 + 25 + 36) * zones and every.fp == (5 + 3) * zones
    assert wf.severity_burned(np.arange(7), 3).tolist() == [False, False, False, True, True, False, False]
    assert not wf.severity_burned(np.array([UNMAPPED]), 1).any()
    for bad in (0, 5, 3.0, True):
        with pytest.raises(AppInputError, match="severity_min_class"):
            wf.score_against_severity(plan, burned, severity, classified, min_class=bad)


def _stop_view(plan, layout, min_class):
    severity, burned, classified = _laid_out(plan, layout, unclassified={})
    sc = wf.score_against_severity(plan, burned, severity, classified, min_class=min_class)
    info = {"file": "s.tif", "sha256": "0" * 64, "proxy": False, "source_tag": None, "nodata": None}
    none = {z.zone: np.zeros(z.shape, dtype=bool) for z in plan.zones}
    validation = {"severity": wf._severity_summary(sc, info, severity, none)}
    return sc, wf._severity_caveats(info, sc), wf.stop_rule(validation), wf.headline(validation)


def test_the_stop_rule_reads_moderate_or_worse_whatever_the_min_class(boundary_plan):
    # The app finds every moderate and high pixel and nothing else: IoU 1 against moderate or worse,
    # 0.5 against low or worse (half the reference missed) and against high (as many false positives).
    only_moderate = {3: (50, 50), 4: (50, 50), 2: (100, 0), 1: (100, 0)}
    for k in (2, 3, 4):
        sc, caveats, rule, head = _stop_view(boundary_plan, only_moderate, k)
        assert sc["moderate_or_worse"].iou == 1.0 and sc["low_or_worse"].iou == 0.5, k
        assert sc["min_class_or_worse"].iou == (1.0 if k == 3 else 0.5), k
        assert rule == {"basis": "severity", "min_class": 3, "iou": 1.0, "below_0_6": False}, k
        assert sc["iou_below_0_6"] is False and not any("stop-and-report" in c for c in caveats), k
        assert ("the stop rule reads IoU 1.000 against moderate or worse" in head["text"]) is (k != 3), k
    # The app also finds the low pixels: IoU 1 against low or worse, 0.5 against moderate or worse,
    # which trips the rule even when the min class is low.
    with_low = {3: (50, 50), 4: (50, 50), 2: (100, 100), 1: (100, 0)}
    for k in (2, 3):
        sc, caveats, rule, _ = _stop_view(boundary_plan, with_low, k)
        assert sc["low_or_worse"].iou == 1.0 and sc["moderate_or_worse"].iou == 0.5, k
        assert rule == {"basis": "severity", "min_class": 3, "iou": 0.5, "below_0_6": True}, k
        assert sc["iou_below_0_6"] is True, k
        assert any("stop-and-report" in c and "severity class 3 (moderate)" in c for c in caveats), k


def test_the_recovery_reference_burned_fraction_counts_burned_classes_only(boundary_plan):
    plan = boundary_plan
    severity, burned, classified = _laid_out(plan)
    sc = wf.score_against_severity(plan, burned, severity, classified, min_class=3)
    # Ten reference pixels in each of 0, 1, 3, 4, 5, 6 and unmapped, per zone.
    reference, s = {}, {}
    for z in plan.zones:
        rr, cc = np.nonzero(z.mask)
        codes = np.full(z.shape, UNMAPPED, dtype=np.uint8)
        ref = np.zeros(z.shape, dtype=bool)
        for i, code in enumerate((0, 1, 3, 4, 5, 6, UNMAPPED)):
            codes[rr[10 * i : 10 * i + 10], cc[10 * i : 10 * i + 10]] = code
            ref[rr[10 * i : 10 * i + 10], cc[10 * i : 10 * i + 10]] = True
        reference[z.zone], s[z.zone] = ref, codes
    info = {"file": "s.tif", "sha256": "0" * 64}
    assert wf._severity_summary(sc, info, s, reference)["recovery_reference_burned_fraction"] == pytest.approx(2 / 7)
    sc1 = wf.score_against_severity(plan, burned, severity, classified, min_class=1)
    assert wf._severity_summary(sc1, info, s, reference)["recovery_reference_burned_fraction"] == pytest.approx(3 / 7)


def test_severity_with_nothing_scored_has_no_ratios(boundary_plan):
    plan = boundary_plan
    for fill in (0, UNMAPPED):
        codes = {z.zone: np.full(z.shape, fill, dtype=np.uint8) for z in plan.zones}
        none = {z.zone: np.zeros(z.shape, dtype=bool) for z in plan.zones}
        sc = wf.score_against_severity(plan, none, codes, none)
        assert sc["pixels_scored"] == 0 and sc["burned_pixels"] == 0
        assert sc["min_class_or_worse"].iou is None and sc["iou_below_0_6"] is False
        assert all(row["detection_rate"] is None for row in sc["classes"].values())
        assert all(v["share"] is None for v in sc["detections_by_class"].values())
    with pytest.raises(ValueError, match="Severity codes"):
        wf.score_against_severity(plan, none, {z.zone: np.full(z.shape, 7, dtype=np.uint8) for z in plan.zones}, none)


# The caldor fixture: synthetic severity rasters on its 10 m grid.


def _fixture_grid(fx):
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    n = fx["scales"].shape[1]
    return rasterio.Affine(a, 0.0, c, 0.0, -a, f), n, int(fx["epsg"])


@pytest.fixture
def fixture_aoi(caldor_fixture, tmp_path):
    """The fixture window minus a two-pixel margin, as a lon/lat GeoJSON, and its geometry."""
    transform, n, epsg = _fixture_grid(caldor_fixture)
    a, c, f = transform.a, transform.c, transform.f
    square = box(c + 2 * a, f - (n - 2) * a, c + (n - 2) * a, f - 2 * a)
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    aoi = shapely.transform(shapely.segmentize(square, 10.0), to_ll.transform, interleaved=False)
    path = tmp_path / "fixture_aoi.geojson"
    path.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(aoi)}))
    return path, aoi


@pytest.fixture
def cli_wildfire_reader(monkeypatch, caldor_reader):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(lambda **kwargs: caldor_reader))
    return caldor_reader


def _write_raster(path: Path, data, transform, crs, *, tags=None, nodata=None) -> Path:
    data = np.asarray(data)
    profile = {
        "driver": "GTiff",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data, 1)
        if tags:
            ds.update_tags(**tags)
    return path


def _blocks(n: int) -> np.ndarray:
    """Codes 1 to 6 in 8 x 8 blocks over an n x n window, with a 4-row band of background on top."""
    rows, cols = np.mgrid[0:n, 0:n]
    s = (1 + (rows // 8 + cols // 8) % 6).astype(np.uint8)
    s[:4] = 0
    return s


#: Columns of the block map on its nodata (255): the map does not cover them.
NODATA_COLUMNS = slice(40, 46)


def _block_map(n: int) -> np.ndarray:
    """:func:`_blocks` with :data:`NODATA_COLUMNS` on nodata 255."""
    s = _blocks(n)
    s[:, NODATA_COLUMNS] = 255
    return s


@pytest.fixture
def block_severity(caldor_fixture, tmp_path) -> Path:
    transform, n, epsg = _fixture_grid(caldor_fixture)
    return _write_raster(tmp_path / "blocks_dnbr6.tif", _block_map(n), transform, epsg, nodata=255)


SEVERITY_KW = {"min_area_ha": 0.05, "block": (32, 32)}


def _on_output_grid(tif: Path, fx, severity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The run's burned band and the fixture-grid severity under each of its pixels (same 10 m grid)."""
    transform, n, _ = _fixture_grid(fx)
    with rasterio.open(tif) as ds:
        flags = ds.read(2)
        t = ds.transform
    dc, dr = round((t.c - transform.c) / 10.0), round((transform.f - t.f) / 10.0)
    rr, cc = np.mgrid[0 : flags.shape[0], 0 : flags.shape[1]]
    rr, cc = rr + dr, cc + dc
    inside = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
    codes = np.full(flags.shape, UNMAPPED, dtype=np.uint8)
    codes[inside] = severity[rr[inside], cc[inside]]
    return flags, codes


def test_severity_run_scores_each_class_on_the_fixture(
    caldor_reader, caldor_fixture, fixture_aoi, block_severity, tmp_path
):
    aoi_path, _ = fixture_aoi
    kw = {**SEVERITY_KW, "reader": caldor_reader, "severity": block_severity}
    result = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw)
    # The severity map alone asks for validation.json, with no outline scores in it.
    assert result.validation_written
    val = json.loads(result.paths.validation.read_text())
    assert set(val["scores"]) == {"severity"} and set(val["reference"]) == {"severity"}
    assert val["reference"]["severity"]["nodata"] == 255
    sev = val["scores"]["severity"]
    flags, codes = _on_output_grid(result.paths.tif, caldor_fixture, _block_map(caldor_fixture["scales"].shape[1]))
    classified = np.isfinite(flags)
    burned = flags == 1
    mapped = codes != UNMAPPED
    assert burned.sum() > 100 and (~burned & classified).sum() > 100, "the fixture is about half burned"
    for c in (0, 1, 2, 3, 4, 5):
        here = classified & (codes == c)
        row = sev["classes"][str(c)]
        assert row["pixels"] == int(here.sum()) > 0
        assert row["detected_pixels"] == int((here & burned).sum())
        assert row["detection_rate"] == pytest.approx(float(burned[here].mean()))
    # Background (the band on top) is scored as unburned land; the nodata columns and class 6 are not scored.
    scored = classified & mapped & (codes <= 5)
    ref = scored & ((codes == 3) | (codes == 4))
    tp = int((burned & ref).sum())
    fp = int((burned & scored & ~ref).sum())
    fn = int((~burned & ref).sum())
    m = sev["min_class_or_worse"]
    assert (m["tp"], m["fp"], m["fn"], m["n"]) == (tp, fp, fn, int(scored.sum()))
    assert int((burned & scored & (codes == 0)).sum()) > 0, "detections on background count as false positives"
    assert sev["excluded"]["non_processing_pixels"] == int((classified & (codes == 6)).sum()) > 0
    assert sev["excluded"]["unmapped_pixels"] == int((classified & ~mapped).sum()) > 0
    assert sev["excluded"]["unmapped_pixels"] != sev["excluded"]["non_processing_pixels"]
    assert sev["detections_by_class"]["0"]["pixels"] == int((burned & (codes == 0)).sum()) > 0
    assert sev["detections_by_class"]["unmapped"]["pixels"] == int((burned & ~mapped).sum())
    # The summary carries the compact view, number for number, and the headline puts the severity IoU first.
    s = result.summary
    view = s["validation"]["severity"]
    assert view["iou"] == pytest.approx(tp / (tp + fp + fn))
    for key in ("iou", "precision", "recall", "f1"):
        assert view[key] == sev["min_class_or_worse"][key], key
    for block in ("moderate_or_worse", "low_or_worse"):
        assert view[block] == {k: sev[block][k] for k in ("iou", "precision", "recall", "f1")}, block
    assert view["low_or_worse"] != view["moderate_or_worse"]
    assert view["pixels_scored"] == sev["pixels_scored"] == m["n"]
    assert view["excluded"] == sev["excluded"]
    assert set(view["detections_by_class"]) == {"0", "1", "2", "3", "4", "5", "6", "unmapped"}
    for c, share in view["detections_by_class"].items():
        assert share == sev["detections_by_class"][c]["share"], c
    assert view["detection_rate_by_class"] == {c: row["detection_rate"] for c, row in sev["classes"].items()}
    assert view["iou_below_0_6"] is sev["iou_below_0_6"] is (sev["moderate_or_worse"]["iou"] < 0.6)
    assert view["proxy"] is False
    assert s["validation"]["stop_rule"] == {
        "basis": "severity",
        "min_class": 3,
        "iou": sev["moderate_or_worse"]["iou"],
        "below_0_6": sev["iou_below_0_6"],
    }
    head = s["headline"]
    assert head["basis"] == "severity" and head["order"] == ["severity_iou", "detection_rate_by_class"]
    assert head["severity_iou"] == view["iou"]
    assert head["text"].startswith("Severity (severity map): IoU")
    assert "high" in head["text"] and "increased greenness" in head["text"]
    assert result.report["params"]["severity"]["file"] == block_severity.name
    # The caveats say what the severity scores can show, and this map is not the proxy.
    assert any("per fire" in c for c in val["caveats"]) and any("blocky" in c for c in val["caveats"])
    assert any("excluded" in c and "false positives" in c for c in val["caveats"])
    assert not any("sentinel2_dnbr" in c for c in val["caveats"])
    assert not any("declares 0 as its nodata" in c for c in val["caveats"])
    # A rerun with the same severity map skips; another min class reruns, and the stop rule stays put.
    assert run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw).skipped
    moved = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw, severity_min_class=4)
    assert not moved.skipped
    assert moved.summary["validation"]["severity"]["min_class"] == 4
    assert moved.summary["validation"]["severity"]["iou"] != view["iou"]
    assert moved.summary["validation"]["stop_rule"] == s["validation"]["stop_rule"]


def test_a_severity_file_whose_nodata_is_0_cannot_score_its_background(
    caldor_reader, caldor_fixture, fixture_aoi, tmp_path
):
    aoi_path, _ = fixture_aoi
    transform, n, epsg = _fixture_grid(caldor_fixture)
    path = _write_raster(tmp_path / "zero_nodata.tif", _blocks(n), transform, epsg, nodata=0)
    result = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", reader=caldor_reader, severity=path, **SEVERITY_KW)
    val = json.loads(result.paths.validation.read_text())
    sev = val["scores"]["severity"]
    flags, codes = _on_output_grid(result.paths.tif, caldor_fixture, _blocks(n))
    assert sev["classes"]["0"]["pixels"] == 0
    assert sev["excluded"]["unmapped_pixels"] == int((np.isfinite(flags) & (codes == 0)).sum()) > 0
    assert any("declares 0 as its nodata" in c for c in val["caveats"])


def test_load_severity_reads_any_crs_with_nearest(caldor_reader, fixture_aoi, tmp_path):
    _, aoi = fixture_aoi
    plan = plan_aoi(caldor_reader, aoi)
    (z,) = plan.zones
    # Class 4 west of the AOI's middle meridian, 2 east of it, as a 0.0001 degree lon/lat raster.
    west, south, east, north = aoi.bounds
    mid = (west + east) / 2
    step = 0.0001
    width, height = round((east - west) / step) + 40, round((north - south) / step) + 40
    t = rasterio.Affine(step, 0, west - 20 * step, 0, -step, north + 20 * step)
    lons = t.c + (np.arange(width) + 0.5) * step
    data = np.where(lons < mid, 4, 2).astype(np.uint8)[None, :].repeat(height, axis=0)
    classes, info = wf.load_severity(plan, _write_raster(tmp_path / "ll.tif", data, t, 4326))
    got = classes[z.zone]
    assert got.dtype == np.uint8 and got.shape == z.shape
    assert (got[~z.mask] == UNMAPPED).all() and set(np.unique(got[z.mask]).tolist()) == {2, 4}
    rows, cols = np.nonzero(z.mask)
    x = z.transform.c + (cols + 0.5) * 10.0
    y = z.transform.f - (rows + 0.5) * 10.0
    lon, _ = Transformer.from_crs(z.grid.crs, 4326, always_xy=True).transform(x, y)
    far = np.abs(lon - mid) > 2 * step
    assert (got[rows, cols][far] == np.where(lon < mid, 4, 2)[far]).all()
    assert info["file"] == "ll.tif" and info["proxy"] is False and info["source_tag"] is None
    assert info["crs"] == "EPSG:4326" and info["sha256"] == export.file_sha256(tmp_path / "ll.tif")
    assert info["nodata"] is None


def test_load_severity_checks_the_codes_over_the_aoi_only(caldor_reader, caldor_fixture, tmp_path):
    transform, n, epsg = _fixture_grid(caldor_fixture)
    a, c, f = transform.a, transform.c, transform.f
    # A triangle AOI: the far corner of its plan window lies outside it.
    tri = shapely.Polygon([(c + 5 * a, f - 5 * a), (c + (n - 5) * a, f - 5 * a), (c + 5 * a, f - (n - 5) * a)])
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    plan = plan_aoi(caldor_reader, shapely.transform(shapely.segmentize(tri, 10.0), to_ll.transform, interleaved=False))
    (z,) = plan.zones
    # Fixture-grid pixel (row, col) of plan pixel (0, 0).
    r0, c0 = round((f - z.transform.f) / a), round((z.transform.c - c) / a)
    corner = np.argwhere(~z.mask)
    corner = corner[np.argmax(corner.sum(axis=1))]
    inside = np.argwhere(z.mask)[0]
    outside = _blocks(n)
    outside[corner[0] + r0, corner[1] + c0] = 255
    classes, info = wf.load_severity(plan, _write_raster(tmp_path / "fill_outside.tif", outside, transform, epsg))
    assert classes[z.zone][tuple(corner)] == UNMAPPED and info["nodata"] is None
    within = _blocks(n)
    within[inside[0] + r0, inside[1] + c0] = 255
    path = _write_raster(tmp_path / "fill_inside.tif", within, transform, epsg)
    with pytest.raises(AppInputError, match=re.escape("codes 0 to 6 (255) over the AOI")):
        wf.load_severity(plan, path)
    # Declared as nodata, the same fill value is simply not mapped.
    path = _write_raster(tmp_path / "fill_nodata.tif", within, transform, epsg, nodata=255)
    classes, info = wf.load_severity(plan, path)
    assert classes[z.zone][tuple(inside)] == UNMAPPED and info["nodata"] == 255


def test_severity_errors_name_the_file_before_any_chunk_read(
    caldor_reader, caldor_store, caldor_fixture, fixture_aoi, tmp_path
):
    _, log = caldor_store
    aoi_path, _ = fixture_aoi
    transform, n, epsg = _fixture_grid(caldor_fixture)
    garbage = tmp_path / "garbage.tif"
    garbage.write_bytes(b"<html>not a GeoTIFF</html>")
    far = _write_raster(tmp_path / "far.tif", _blocks(n), transform @ rasterio.Affine.translation(5_000, 0), epsg)
    seven = _blocks(n)
    seven[30, 30] = 7
    halves = _blocks(n).astype(np.float32)
    halves[40, 40] = 2.5
    only_edges = np.where(_blocks(n) == 6, 6, 0).astype(np.uint8)
    cases = {
        "garbage": (garbage, "Could not read the severity file"),
        "far": (far, "does not overlap the AOI"),
        "seven": (_write_raster(tmp_path / "seven.tif", seven, transform, epsg), "codes 0 to 6 (7)"),
        "halves": (_write_raster(tmp_path / "halves.tif", halves, transform, epsg), "codes 0 to 6 (2.5)"),
        "edges": (_write_raster(tmp_path / "edges.tif", only_edges, transform, epsg), "no pixel in classes 1 to 5"),
    }
    out = tmp_path / "out"
    for name, (path, message) in cases.items():
        with pytest.raises(AppInputError, match=re.escape(message)) as info:
            run_wildfire(aoi_path, 2020, 2022, out, reader=caldor_reader, severity=path, **SEVERITY_KW)
        assert str(path) in str(info.value), name
        assert log.chunk_keys() == [], name
    assert not (out / "wildfire_report.json").exists()


def _unreachable(*args, **kwargs):
    raise validate.ReferenceUnavailableError("services3.arcgis.com refused the request (403)")


def _nine(fx, tmp_path: Path) -> Path:
    """A block map with one value outside the dnbr6 codes, 9, inside the fixture AOI."""
    transform, n, epsg = _fixture_grid(fx)
    bad = _blocks(n)
    bad[20, 20] = 9
    return _write_raster(tmp_path / "nine.tif", bad, transform, epsg)


def test_a_bad_severity_file_is_named_before_a_strict_unreachable_outline(
    cli_wildfire_reader, caldor_store, caldor_fixture, fixture_aoi, tmp_path, monkeypatch
):
    # The severity map is loaded first, so a bad file is a usage error naming it (exit 2), not the
    # strict validation failure the unreachable outline would give (exit 3).
    _, log = caldor_store
    aoi_path, _ = fixture_aoi
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    path = _nine(caldor_fixture, tmp_path)
    with pytest.raises(AppInputError, match=re.escape(str(path))):
        run_wildfire(
            aoi_path,
            2020,
            2022,
            tmp_path / "out",
            reader=cli_wildfire_reader,
            validate=True,
            strict_validate=True,
            severity=path,
            **SEVERITY_KW,
        )
    assert log.chunk_keys() == []
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(tmp_path / "o")]
    result = PlainCliRunner().invoke(app, [*args, "--validate", "--strict-validate", "--severity", str(path)])
    assert result.exit_code == EXIT_USAGE and "nine.tif" in result.stderr, result.output
    assert log.chunk_keys() == []


def test_severity_arguments_fail_before_the_store_opens(no_store, tmp_path):
    with pytest.raises(AppInputError, match=r"Severity file .* does not exist"):
        run_wildfire("caldor", 2020, 2022, tmp_path, severity=tmp_path / "missing.tif")
    with pytest.raises(AppInputError, match="Severity file"):
        run_wildfire("caldor", 2020, 2022, tmp_path, severity=tmp_path)  # a directory
    for bad in (0, 5, -1):
        with pytest.raises(AppInputError, match="severity_min_class"):
            run_wildfire("caldor", 2020, 2022, tmp_path, severity_min_class=bad)


def test_the_sentinel2_proxy_tag_is_named_in_the_caveats(caldor_reader, caldor_fixture, fixture_aoi, tmp_path):
    aoi_path, _ = fixture_aoi
    transform, n, epsg = _fixture_grid(caldor_fixture)
    tags = {"TESSERA_APPS_SOURCE": "sentinel2_dnbr", "TESSERA_APPS_DNBR_OFFSET": "-0.013", "OTHER": "x"}
    proxy = _write_raster(tmp_path / "proxy.tif", _blocks(n), transform, epsg, tags=tags)
    result = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", reader=caldor_reader, severity=proxy, **SEVERITY_KW)
    val = json.loads(result.paths.validation.read_text())
    info = val["reference"]["severity"]
    assert info["proxy"] is True and info["source_tag"] == "sentinel2_dnbr"
    assert info["tags"] == {"TESSERA_APPS_DNBR_OFFSET": "-0.013", "TESSERA_APPS_SOURCE": "sentinel2_dnbr"}
    assert any("sentinel2_dnbr" in c and "not MTBS" in c for c in val["caveats"])
    assert result.summary["headline"]["text"].startswith("Severity (Sentinel-2 dNBR proxy)")
    # Another source tag is not the proxy.
    other = _write_raster(tmp_path / "mtbs.tif", _blocks(n), transform, epsg, tags={"TESSERA_APPS_SOURCE": "mtbs"})
    plain = run_wildfire(aoi_path, 2020, 2022, tmp_path / "o2", reader=caldor_reader, severity=other, **SEVERITY_KW)
    caveats = json.loads(plain.paths.validation.read_text())["caveats"]
    assert not any("sentinel2_dnbr" in c for c in caveats)
    assert plain.summary["validation"]["severity"]["proxy"] is False


def test_severity_sets_the_stop_rule_and_comes_first(caldor_reader, fixture_aoi, tmp_path):
    # A poor outline (a small box) trips the stop rule on its own; a severity map made from the run's
    # own burned band scores IoU 1, and then the stop rule reads the severity IoU instead.
    aoi_path, aoi = fixture_aoi
    west, south, east, north = aoi.bounds
    small = box(west, south, west + (east - west) / 5, south + (north - south) / 5)
    outline = tmp_path / "small.geojson"
    outline.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(small)}))
    kw = {**SEVERITY_KW, "reader": caldor_reader, "validate": True, "reference": outline}
    alone = run_wildfire(aoi_path, 2020, 2022, tmp_path / "alone", **kw)
    v = alone.summary["validation"]
    assert v["iou"] < 0.6
    assert v["stop_rule"] == {
        "basis": "reference",
        "iou": v["iou"],
        "recall_inside_reference": v["recall_inside_reference"],
        "below_0_6": True,
    }
    assert any("stop and report" in w and "recall inside the reference" in w for w in alone.report["warnings"])
    head = alone.summary["headline"]
    assert head["basis"] == "reference" and head["order"] == ["recall_inside_reference", "iou"]
    assert head["text"].index("recall inside") < head["text"].index("IoU")
    with rasterio.open(alone.paths.tif) as ds:
        flags, t, crs = ds.read(2), ds.transform, ds.crs
    perfect = np.where(flags == 1, 4, np.where(np.isfinite(flags), 1, 0)).astype(np.uint8)
    sev = _write_raster(tmp_path / "perfect.tif", perfect, t, crs)
    both = run_wildfire(aoi_path, 2020, 2022, tmp_path / "both", severity=sev, **kw)
    s = both.summary
    assert s["burned_pixels"] == alone.summary["burned_pixels"]
    assert s["validation"]["severity"]["iou"] == 1.0 and s["validation"]["iou"] == v["iou"]
    assert s["validation"]["stop_rule"]["basis"] == "severity"
    assert s["validation"]["stop_rule"]["below_0_6"] is False
    assert not any("stop and report" in w for w in both.report["warnings"])
    val = json.loads(both.paths.validation.read_text())
    assert "burned" in val["scores"] and "severity" in val["scores"]
    assert val["reference"]["file"] == "small.geojson" and val["reference"]["severity"]["file"] == "perfect.tif"
    assert not any("stop-and-report" in c for c in val["caveats"])
    head = s["headline"]
    assert head["order"] == ["severity_iou", "detection_rate_by_class", "recall_inside_reference", "iou"]
    assert head["detection_rate_by_class"]["4"] == 1.0 and head["detection_rate_by_class"]["1"] == 0.0
    assert head["text"].index("Severity") < head["text"].index("recall inside")
    assert s["validation"]["severity"]["recovery_reference_burned_fraction"] == 0.0
    # A severity map that misses the burn trips the stop rule on its own IoU.
    inverted = np.where(flags == 1, 1, np.where(np.isfinite(flags), 4, 0)).astype(np.uint8)
    inv = _write_raster(tmp_path / "inv.tif", inverted, t, crs)
    bad = run_wildfire(aoi_path, 2020, 2022, tmp_path / "bad", severity=inv, **kw)
    assert bad.summary["validation"]["severity"]["iou"] == 0.0
    assert [w for w in bad.report["warnings"] if w.endswith(wf.STOP_WARNING)] == [
        "IoU 0.000 against severity class 3 (moderate) or worse is below 0.6: stop and report (brief section 8)."
    ]
    caveats = json.loads(bad.paths.validation.read_text())["caveats"]
    assert any("stop-and-report" in c and "severity class 3" in c for c in caveats)


def test_severity_survives_an_unreachable_outline_and_the_rerun_retries(
    caldor_reader, caldor_store, fixture_aoi, block_severity, tmp_path, monkeypatch
):
    _, log = caldor_store
    aoi_path, aoi = fixture_aoi
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    kw = {**SEVERITY_KW, "reader": caldor_reader, "validate": True, "severity": block_severity}
    first = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw)
    # The severity scores are written; the report does not vouch for the file, since the outline is missing.
    assert not first.validation_written and first.paths.validation.is_file()
    first_val = json.loads(first.paths.validation.read_text())
    first_scores = first_val["scores"]
    assert set(first_scores) == {"severity"}
    # The file itself says the outline was asked for and why it is missing, not only the report.
    outline = first_val["reference"]["outline"]
    assert outline["scored"] is False and "services3.arcgis.com" in outline["error"]
    assert any(c.startswith(wf.OUTLINE_NOT_SCORED) and "services3.arcgis.com" in c for c in first_val["caveats"])
    assert "validation" not in first.report["outputs"]
    s = first.summary["validation"]
    assert "services3.arcgis.com" in s["error"] and s["severity"]["iou"] is not None
    assert first.summary["headline"]["basis"] == "severity"
    # This map trips the stop rule, so the first run warns once.
    assert s["stop_rule"]["below_0_6"] is True
    assert sum(w.endswith(wf.STOP_WARNING) for w in first.report["warnings"]) == 1
    reads = len(log.chunk_keys())
    # Still unreachable: nothing is read or redone.
    again = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw)
    assert again.skipped and len(log.chunk_keys()) == reads
    # Reachable now: the validation is redone from the outputs on disk, both blocks, without a chunk read.
    answer = [demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)]
    monkeypatch.setattr(validate, "nifc_perimeters", lambda bbox, **k: list(answer))
    redone = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw)
    assert redone.validation_redone and redone.validation_written and len(log.chunk_keys()) == reads
    redone_val = json.loads(redone.paths.validation.read_text())
    scores = redone_val["scores"]
    assert {"burned", "severity"} <= set(scores)
    assert "outline" not in redone_val["reference"]
    assert not any(c.startswith(wf.OUTLINE_NOT_SCORED) for c in redone_val["caveats"])
    assert scores["severity"]["min_class_or_worse"] == first_scores["severity"]["min_class_or_worse"]
    assert redone.summary["headline"]["order"][-2:] == ["recall_inside_reference", "iou"]
    # The redone report carries the stop warning once, not once per run, and no skipped warning.
    warnings = redone.report["warnings"]
    assert sum(w.endswith(wf.STOP_WARNING) for w in warnings) == 1, warnings
    assert not any(w.startswith(wf.VALIDATION_SKIPPED) for w in warnings)
    assert run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", **kw).skipped


def test_adding_validate_after_a_severity_only_run_scores_the_outline(
    caldor_reader, caldor_store, fixture_aoi, block_severity, tmp_path, monkeypatch
):
    _, log = caldor_store
    aoi_path, aoi = fixture_aoi
    out = tmp_path / "out"
    kw = {**SEVERITY_KW, "reader": caldor_reader, "severity": block_severity}
    first = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    assert first.validation_written and set(json.loads(first.paths.validation.read_text())["scores"]) == {"severity"}
    before = first.paths.report.read_text()
    reads = len(log.chunk_keys())
    # The outline cannot be loaded: strict stops, and otherwise nothing is redone; the report is left as it was.
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    with pytest.raises(StrictValidationError):
        run_wildfire(aoi_path, 2020, 2022, out, validate=True, strict_validate=True, **kw)
    assert first.paths.report.read_text() == before
    again = run_wildfire(aoi_path, 2020, 2022, out, validate=True, **kw)
    assert again.skipped and not again.validation_redone and not again.validation_written
    assert again.report == first.report and first.paths.report.read_text() == before
    assert run_wildfire(aoi_path, 2020, 2022, out, **kw).skipped
    # Reachable: adding --validate (strict too) scores the outline from the outputs on disk, without a chunk read.
    answer = [demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)]
    monkeypatch.setattr(validate, "nifc_perimeters", lambda bbox, **k: list(answer))
    redone = run_wildfire(aoi_path, 2020, 2022, out, validate=True, strict_validate=True, **kw)
    assert redone.validation_redone and redone.validation_written and not redone.skipped
    assert len(log.chunk_keys()) == reads
    scores = json.loads(redone.paths.validation.read_text())["scores"]
    assert {"burned", "severity"} <= set(scores)
    assert redone.summary["validation"]["source"] == "nifc" and redone.summary["validation"]["iou"] is not None
    assert "validation" in redone.report["outputs"]
    # Current now for both requests.
    assert run_wildfire(aoi_path, 2020, 2022, out, validate=True, **kw).skipped
    assert run_wildfire(aoi_path, 2020, 2022, out, **kw).skipped
    assert len(log.chunk_keys()) == reads


def test_a_partial_validation_redone_without_validate_still_scores_the_outline_later(
    caldor_reader, caldor_store, fixture_aoi, block_severity, tmp_path, monkeypatch
):
    _, log = caldor_store
    aoi_path, aoi = fixture_aoi
    out = tmp_path / "out"
    kw = {**SEVERITY_KW, "reader": caldor_reader, "severity": block_severity}
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    partial = run_wildfire(aoi_path, 2020, 2022, out, validate=True, **kw)
    assert not partial.validation_written
    reads = len(log.chunk_keys())
    # Without --validate the severity validation is complete, and the report vouches for it.
    severity_only = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    assert severity_only.validation_redone and severity_only.validation_written
    assert "error" not in severity_only.summary["validation"]
    answer = [demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)]
    monkeypatch.setattr(validate, "nifc_perimeters", lambda bbox, **k: list(answer))
    later = run_wildfire(aoi_path, 2020, 2022, out, validate=True, **kw)
    assert later.validation_redone and later.validation_written
    assert "burned" in json.loads(later.paths.validation.read_text())["scores"]
    assert len(log.chunk_keys()) == reads


def test_severity_enters_the_digest(monkeypatch, tmp_path):
    seen = []

    class StopRunError(Exception):
        pass

    def capture(app, aoi, years_read, year_a, year_b, out_dir, params, **kwargs):
        seen.append((params, sorted(years_read), kwargs["validate"]))
        raise StopRunError

    monkeypatch.setattr(wf, "prepare_run", capture)
    one = tmp_path / "one.tif"
    one.write_bytes(b"one")
    two = tmp_path / "two.tif"
    two.write_bytes(b"two")
    copy = tmp_path / "copy" / "one.tif"
    copy.parent.mkdir()
    copy.write_bytes(b"one")
    edited = tmp_path / "edited" / "one.tif"
    edited.parent.mkdir()
    edited.write_bytes(b"one, edited")
    variants = [
        {},
        {"severity": one},
        {"severity": two},
        {"severity": edited},
        {"severity": one, "severity_min_class": 2},
        {"severity": one, "severity_min_class": 4},
    ]
    for v in [*variants, {"severity": copy}, {"severity_min_class": 2}, {"severity": one, "validate": True}]:
        with pytest.raises(StopRunError):
            run_wildfire("caldor", 2020, 2022, tmp_path, **v)
    digests = [_common.export.run_args_digest(p, years=y, aoi_sha256="x", seed=0) for p, y, _ in seen]
    assert len(set(digests[: len(variants)])) == len(variants)
    # The file is digested by its name and content (a copy elsewhere is the same); the min class
    # only counts with a map; --validate does not enter the digest.
    assert digests[len(variants)] == digests[1] and digests[-2] == digests[0] and digests[-1] == digests[1]
    # A severity map asks for validation.json even without --validate.
    assert [v for _, _, v in seen] == [False, True, True, True, True, True, True, False, True]


def _cli_args(aoi_path: Path, out: Path) -> list[str]:
    return ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]


def test_cli_wildfire_severity_options(cli_wildfire_reader, fixture_aoi, block_severity, tmp_path):
    aoi_path, _ = fixture_aoi
    out = tmp_path / "cli"
    opts = ["--min-area-ha", "0.05", "--severity", str(block_severity)]
    result = PlainCliRunner().invoke(app, [*_cli_args(aoi_path, out), *opts, "--severity-min-class", "4"])
    assert result.exit_code == 0, (result.output, result.exception)
    text = result.stdout
    flat = " ".join(text.split())  # rich wraps long lines
    assert "Headline: Severity (severity map): IoU" in flat and "against high or worse" in flat
    assert "the stop rule reads IoU" in flat
    assert "Severity, class 4 (high) or worse: IoU" in flat and "low or worse: IoU" in flat
    assert "moderate or worse (the stop rule's): IoU" in flat
    assert "4 high" in text and "5 increased greenness" in text and "0 background" in text
    report = json.loads((out / "wildfire_report.json").read_text())
    assert report["params"]["severity"] == {
        "file": block_severity.name,
        "sha256": export.file_sha256(block_severity),
        "min_class": 4,
    }
    view = report["summary"]["validation"]["severity"]
    assert view["min_class"] == 4
    assert "Validation (" not in text  # no outline was asked for
    # The table's two columns: the class's detection rate, then the share of the detections in it.
    for c, label in ((4, "4 high"), (0, "0 background")):
        row = next(line for line in text.splitlines() if label in line)
        assert re.findall(r"\d+\.\d%|n/a", row) == [
            f"{view['detection_rate_by_class'][str(c)]:.1%}",
            f"{view['detections_by_class'][str(c)]:.1%}",
        ], row
    found = re.search(r"excluded classified pixels: ([\d,]+) outside the map, ([\d,]+) non-processing", flat)
    assert found is not None, text
    excluded = view["excluded"]
    assert [int(g.replace(",", "")) for g in found.groups()] == [
        excluded["unmapped_pixels"],
        excluded["non_processing_pixels"],
    ]
    assert excluded["unmapped_pixels"] != excluded["non_processing_pixels"]
    assert f"burned pixels outside the map {view['detections_by_class']['unmapped']:.1%}" in flat
    # The default min class is the stop rule's own, so the line does not repeat it.
    default = PlainCliRunner().invoke(app, [*_cli_args(aoi_path, tmp_path / "default"), *opts])
    assert default.exit_code == 0, default.output
    assert "stop rule" not in " ".join(default.stdout.split())


def test_cli_wildfire_prints_a_bracketed_severity_file_name(cli_wildfire_reader, fixture_aoi, block_severity, tmp_path):
    # Rich markup in the file name is printed, not read: "[bold]" and "[v2]" would otherwise vanish.
    aoi_path, _ = fixture_aoi
    named = tmp_path / "s[bold][v2].tif"
    named.write_bytes(block_severity.read_bytes())
    args = [*_cli_args(aoi_path, tmp_path / "o"), "--min-area-ha", "0.05", "--severity", str(named)]
    result = PlainCliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "[bold]" in result.stdout and "[v2]" in result.stdout


def test_cli_wildfire_prints_the_severity_headline_before_the_outline(
    cli_wildfire_reader, fixture_aoi, block_severity, tmp_path
):
    aoi_path, aoi = fixture_aoi
    ref = tmp_path / "outline.geojson"
    outline = demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)
    ref.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(outline)}))
    args = [*_cli_args(aoi_path, tmp_path / "o"), "--min-area-ha", "0.05"]
    result = PlainCliRunner().invoke(app, [*args, "--reference", str(ref), "--severity", str(block_severity)])
    assert result.exit_code == 0, (result.output, result.exception)
    text = result.stdout
    order = [text.index(k) for k in ("Headline: Severity", "Severity, class 3", "Validation (file): recall inside")]
    assert order == sorted(order)
    line = text[text.index("Validation (file)") :]
    assert line.index("recall inside reference") < line.index("IoU")


def test_cli_wildfire_strict_validate_after_a_severity_only_run(
    cli_wildfire_reader, fixture_aoi, block_severity, tmp_path, monkeypatch
):
    # Adding --validate --strict-validate to a --severity run is not "current": with NIFC
    # unreachable it exits 3 instead of 0.
    aoi_path, _ = fixture_aoi
    args = [*_cli_args(aoi_path, tmp_path / "o"), "--min-area-ha", "0.05", "--severity", str(block_severity)]
    assert PlainCliRunner().invoke(app, args).exit_code == 0
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    result = PlainCliRunner().invoke(app, [*args, "--validate", "--strict-validate"])
    assert result.exit_code == EXIT_VALIDATION, result.output
    assert "services3.arcgis.com" in result.stderr


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--severity", "missing.tif"], "does not exist"),
        (["--severity-min-class", "5"], "severity_min_class"),
    ],
)
def test_cli_wildfire_bad_severity_exits_2(cli_wildfire_reader, fixture_aoi, tmp_path, extra, message):
    aoi_path, _ = fixture_aoi
    result = PlainCliRunner().invoke(app, [*_cli_args(aoi_path, tmp_path), *extra])
    assert result.exit_code == EXIT_USAGE and message in result.stderr
    assert not (tmp_path / "wildfire_report.json").exists()


def test_cli_wildfire_unusable_severity_file_exits_2_before_reading(
    cli_wildfire_reader, caldor_store, caldor_fixture, fixture_aoi, tmp_path
):
    _, log = caldor_store
    aoi_path, _ = fixture_aoi
    path = _nine(caldor_fixture, tmp_path)
    result = PlainCliRunner().invoke(app, [*_cli_args(aoi_path, tmp_path / "o"), "--severity", str(path)])
    assert result.exit_code == EXIT_USAGE and "nine.tif" in result.stderr and "0 to 6" in result.stderr
    assert log.chunk_keys() == []


def test_cli_wildfire_passes_the_severity_options(monkeypatch, tmp_path):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        raise AppInputError("stop here")

    monkeypatch.setattr("tessera_apps.apps.run_wildfire", fake)
    sev = tmp_path / "s.tif"
    args = ["wildfire", "--aoi", "caldor", "--year-a", "2020", "--year-b", "2022", "--severity", str(sev)]
    result = PlainCliRunner().invoke(app, [*args, "--severity-min-class", "2"])
    assert result.exit_code == EXIT_USAGE
    assert seen["severity"] == sev and seen["severity_min_class"] == 2 and seen["validate"] is False
    PlainCliRunner().invoke(app, args)
    assert seen["severity_min_class"] == 3


# --------------------------------------------------------------------------------------------
# scripts/make_caldor_dnbr.py, offline: the parts that decide the COG's bytes
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dnbr_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_caldor_dnbr.py"
    spec = importlib.util.spec_from_file_location("make_caldor_dnbr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_the_proxy_cog_bytes_depend_only_on_what_it_is_made_from(
    dnbr_script, caldor_reader, caldor_fixture, fixture_aoi, tmp_path, monkeypatch
):
    s = dnbr_script

    class Clock(dt.datetime):
        ticks = 0

        @classmethod
        def now(cls, tz=None):
            cls.ticks += 1
            return dt.datetime(2026, 1, 1, tzinfo=dt.UTC) + dt.timedelta(minutes=cls.ticks)

    # A clock that moves on every call: nothing the COG holds may read it.
    monkeypatch.setattr(
        s, "dt", types.SimpleNamespace(datetime=Clock, date=dt.date, timedelta=dt.timedelta, UTC=dt.UTC)
    )
    records = [
        {"scene": "S2A_10SGH_20210615_1_L2A", "window": [0, 0, 32, 32]},
        {"scene": "S2B_10SGJ_20210620_1_L2A", "window": None},
    ]
    kw = {"offset": -0.0133, "pre": records, "post": records[:1], "tiles": ["10SGH", "10SGJ"], "aoi_sha256": "ab"}
    tags = s.cog_tags(**kw)
    assert tags == s.cog_tags(**kw)
    assert tags["TESSERA_APPS_SOURCE"] == "sentinel2_dnbr" and not any("CREATED" in k for k in tags)
    assert json.loads(tags["TESSERA_APPS_PRE_SCENES"]) == ["S2A_10SGH_20210615_1_L2A"]
    # The proxy on the fixture's grid at 20 m: classes inside an AOI square, nodata outside it.
    fx_transform, n, _ = _fixture_grid(caldor_fixture)
    grid = s.Grid(rasterio.Affine(20.0, 0.0, fx_transform.c, 0.0, -20.0, fx_transform.f), n // 2, n // 2)
    inside = np.zeros(grid.shape, dtype=bool)
    inside[2:-2, 2:-2] = True
    dnbr = np.linspace(-0.3, 1.0, inside.size, dtype=np.float32).reshape(grid.shape)
    dnbr[5, :] = np.nan
    classes = s.classify(dnbr, inside)
    assert (classes[~inside] == s.NODATA).all()
    assert set(np.unique(classes[inside]).tolist()) == {1, 2, 3, 4, 5, 6}
    a = s.write_cog(tmp_path / "a.tif", classes, grid, tags)
    b = s.write_cog(tmp_path / "b.tif", classes, grid, s.cog_tags(**kw))
    assert export.file_sha256(a) == export.file_sha256(b)
    with rasterio.open(a) as ds:
        assert ds.nodata == s.NODATA and ds.tags()["TESSERA_APPS_SOURCE"] == "sentinel2_dnbr"
        assert not any("CREATED" in k for k in ds.tags())
    # The wildfire app takes the nodata outside the proxy's AOI as unmapped, not as background.
    _, aoi = fixture_aoi
    plan = plan_aoi(caldor_reader, aoi)
    codes, info = wf.load_severity(plan, a)
    (z,) = plan.zones
    got = codes[z.zone][z.mask]
    assert info["proxy"] is True and info["nodata"] == s.NODATA
    assert (got == UNMAPPED).any() and not (got == 0).any()
