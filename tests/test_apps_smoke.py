"""Each app end to end on the bundled real fixture (brief section 3), offline, through its function and its CLI.

The wildfire app runs on ``tests/fixtures/caldor_64.npz``: a real 64 x 64 window of the Caldor scar,
about half burned, with all nine years 2017 to 2025, served by the in-memory store of
``tests/conftest.py``. The AOI is the window minus a two-pixel margin, and the validation reference
is the caldor demo's Hansen 2021 loss outline clipped to it.

The deforest app runs on the same store, with the Hansen GFC tile pieces under the fixture
(``tests/fixtures/caldor_64_hansen.npz``) served as local tiles through the loaders' ``url_template``,
so both its labels and its validation run offline. Its AOI is the central 40 x 40 pixels and its
label ring the 10 pixels around them.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from pyproj import Transformer
from rio_cogeo.cogeo import cog_validate
from shapely.geometry import box, mapping
from typer.testing import CliRunner

from tessera_apps import export, metrics, validate
from tessera_apps.apps import DeforestResult, StrictValidationError, WildfireResult
from tessera_apps.apps import deforest as dfr
from tessera_apps.apps import wildfire as wf
from tessera_apps.apps._common import AppInputError
from tessera_apps.apps.deforest import run_deforest
from tessera_apps.apps.wildfire import memory_estimate, run_wildfire
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app
from tessera_apps.demo import demo_feature
from tessera_apps.io import (
    BYTES_PER_PIXEL,
    FIRST_YEAR,
    TesseraReader,
    iter_plan_blocks,
    plan_aoi,
    read_plan,
    read_points,
)

from .conftest import BOUNDARY, fixture_window_lonlat, write_hansen_tiles

RECOVERY_FIELDS = (
    "median_sim_burned",
    "median_sim_unburned",
    "unburned_sim_p10",
    "recovered_fraction",
    "median_recovery_index",
    "sim90_fraction",
    "radar_mismatch_fraction",
)


def _feature_collection(path: Path, geometry, properties: dict) -> Path:
    feature = {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}
    fc = {"type": "FeatureCollection", "features": [feature]}
    path.write_text(json.dumps(fc), encoding="utf-8")
    return path


def report_warnings(result) -> list[str]:
    return list(result.report.get("warnings") or [])


def years_in(log) -> list[int]:
    """Years whose embedding chunks a KeyLog saw read (the time index is the first chunk coordinate)."""
    years = set()
    for key in log.chunk_keys():
        parts = key.split("/")
        if parts[1] == "embeddings" and parts[2] == "c":
            years.add(FIRST_YEAR + int(parts[3]))
    return sorted(years)


@pytest.fixture
def caldor_aoi(caldor_fixture, tmp_path):
    """The fixture window minus a two-pixel margin, as a lon/lat GeoJSON, and its geometry."""
    fx = caldor_fixture
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    n = fx["scales"].shape[1]
    square = box(c + 2 * a, f - (n - 2) * a, c + (n - 2) * a, f - 2 * a)
    to_ll = Transformer.from_crs(int(fx["epsg"]), 4326, always_xy=True)
    aoi = shapely.transform(shapely.segmentize(square, 10.0), to_ll.transform, interleaved=False)
    return _feature_collection(tmp_path / "caldor_window.geojson", aoi, {"name": "fixture window"}), aoi


@pytest.fixture
def hansen_reference(caldor_aoi, tmp_path) -> Path:
    _, aoi = caldor_aoi
    outline = demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)
    return _feature_collection(tmp_path / "reference.geojson", outline, {"role": "hansen_2021_loss_outline"})


def test_wildfire_end_to_end_on_the_caldor_fixture(caldor_reader, caldor_store, caldor_aoi, hansen_reference, tmp_path):
    aoi_path, aoi = caldor_aoi
    out = tmp_path / "out"
    kwargs = {
        "min_area_ha": 0.05,
        "validate": True,
        "reference": hansen_reference,
        "reader": caldor_reader,
        "block": (32, 32),
    }
    result = run_wildfire(aoi_path, 2020, 2022, out, **kwargs)
    assert not result.skipped and result.validation_written
    paths = result.paths

    # Every output exists.
    for p in (*paths.required(need_validation=True), *result.extra_paths.values()):
        assert p.is_file(), p
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [
            "wildfire_2020_2022.tif",
            "wildfire_pca_2020.tif",
            "wildfire_pca_2022.tif",
            "wildfire_summary.parquet",
            "wildfire_summary.csv",
            "wildfire_preview.png",
            "wildfire_validation.json",
            "wildfire_report.json",
        ]
    )

    # The tif is a valid COG with the named bands.
    ok, errors, _ = cog_validate(str(paths.tif))
    assert ok, errors
    with rasterio.open(paths.tif) as ds:
        assert ds.descriptions == (
            "distance_2020_2022",
            "burned",
            "recovery_index_2022",
            "recovery_index_2023",
            "recovery_index_2024",
            "recovery_index_2025",
        )
        assert ds.dtypes[0] == "float32" and np.isnan(ds.nodata)
        bands = ds.read()
    for key in ("pca_a", "pca_b"):
        ok, errors, _ = cog_validate(str(result.extra_paths[key]))
        assert ok, errors
        with rasterio.open(result.extra_paths[key]) as ds:
            assert ds.count == 3 and ds.dtypes == ("uint8",) * 3 and ds.nodata == 0
            rgb = ds.read()
        assert (rgb.any(axis=0)).sum() == 3600  # every AOI pixel coloured, nothing outside
    # Before and after share one basis, fitted on the AOI's 2020 pixels (all 3600 are sampled here).
    (zone_plan,) = plan_aoi(caldor_reader, aoi).zones
    basis = metrics.fit_pca(read_plan(caldor_reader, zone_plan, 2020))
    for key, year in (("pca_a", 2020), ("pca_b", 2022)):
        expected = np.moveaxis(metrics.pca_rgb(read_plan(caldor_reader, zone_plan, year), basis), -1, 0)
        expected = np.where(zone_plan.mask, np.maximum(expected, 1), 0)
        with rasterio.open(result.extra_paths[key]) as ds:
            assert ds.transform == zone_plan.transform
            got = ds.read().astype(np.int16)
        close = (np.abs(got - expected) <= 1).all(axis=0)
        assert close[zone_plan.mask].mean() > 0.99, key

    summary = result.summary
    plan = plan_aoi(caldor_reader, aoi)
    assert summary["aoi_pixels"] == plan.n_pixels == 3600
    assert summary["classified_pixels"] == 3600
    assert 0.2 < summary["burned_fraction"] < 0.8
    burned = bands[1]
    assert int(np.nansum(burned)) == summary["burned_pixels"]
    assert int(np.isfinite(burned).sum()) == summary["classified_pixels"]
    assert summary["burned_area_ha"] == pytest.approx(summary["burned_pixels"] * 0.01)
    assert summary["threshold"]["method"] == "otsu" and not summary["fallback_ran"]
    # Recovery rasters cover exactly the burned pixels, and R is 0 at year_b by construction.
    for band in bands[2:]:
        assert np.array_equal(np.isfinite(band), burned == 1)
    assert np.all(bands[2][burned == 1] == 0)

    # One row per recovery year, the same in parquet and csv, every recovery field finite.
    table = pd.read_parquet(paths.summary_parquet)
    pd.testing.assert_frame_equal(table, pd.read_csv(paths.summary_csv), check_dtype=False)
    assert table["year"].tolist() == [2022, 2023, 2024, 2025]
    for col in RECOVERY_FIELDS:
        assert np.isfinite(table[col]).all(), col
    assert (table["n_burned"] == summary["burned_pixels"]).all()
    assert (table["threshold"] == summary["threshold"]["threshold"]).all()
    assert (table["burned_fraction"] == summary["burned_fraction"]).all()
    assert table.loc[0, "median_recovery_index"] == 0.0
    # The per-pixel recovery rasters are the table's index, pixel by pixel.
    for k, row in table.iterrows():
        assert float(np.nanmedian(bands[2 + k][burned == 1])) == pytest.approx(row["median_recovery_index"], abs=1e-6)
    assert float(np.nanmedian(bands[5][burned == 1])) != 0.0

    # The distance band holds the cosine distance itself, on every classified pixel.
    (zp,) = plan.zones
    expected = metrics.cosine_distance_map(read_plan(caldor_reader, zp, 2020), read_plan(caldor_reader, zp, 2022))
    assert int(np.isfinite(bands[0]).sum()) == summary["classified_pixels"]
    np.testing.assert_allclose(bands[0][zp.mask], expected[zp.mask], atol=1e-6)
    assert np.all(bands[0][burned == 1] > summary["threshold"]["threshold"])

    # The default fire year is year_b - 1 (survey N11), so no fire-year warning.
    assert summary["fire_year"] == 2021
    assert not any("survey N11" in w or "is the fire year" in w for w in report_warnings(result))

    # Recovery is measured against land shown unchanged: on this window (mostly inside the scar) it
    # stays about 0.95 similar to itself before and after the fire, and no burned pixel reaches its
    # normal variation. The pixels just under the threshold are fire-affected and were left out:
    # with them the reference median fell to about 0.89 and about 10 percent "recovered".
    ref = summary["recovery_reference"]
    assert ref["pixels"] >= 100 and ref["unburned_candidates"] > 2 * ref["pixels"]
    assert ref["max_distance"] == pytest.approx(0.5 * summary["threshold"]["threshold"])
    assert ref["post_fire_similarity"]["median"] > 0.94
    assert ref["pre_fire_similarity"]["years"] == [2019, 2020] and ref["pre_fire_similarity"]["median"] > 0.95
    assert ref["pre_fire_similarity"]["pixels"] > 0
    assert (table["n_unburned"] == ref["pixels"]).all()
    assert (table["median_sim_unburned"] > 0.94).all() and (table["unburned_sim_p10"] > 0.9).all()
    assert (table["recovered_fraction"] == 0.0).all()
    assert ref["pre_minus_post_median"] < wf.REFERENCE_DRIFT_WARN
    assert not any("less similar to itself" in w for w in report_warnings(result))
    assert (table["sim90_fraction"] == 0.0).all() and (table["median_sim_burned"] < 0.7).all()

    # Validation against the Hansen outline (checkpoint 2 measured IoU 0.75 on this window).
    val = json.loads(paths.validation.read_text())
    assert val["scores"]["burned"]["iou"] > 0.6
    assert val["scores"]["burned"]["n"] == 3600
    assert val["reference"]["source"] == "file" and val["reference"]["hansen_derived"] is True
    assert any("stand-replacing" in c for c in val["caveats"])
    assert summary["validation"]["iou"] == val["scores"]["burned"]["iou"]
    # It scores the cleaned mask the app writes, not the raw threshold mask.
    assert summary["burned_pixels_before_cleaning"] > summary["burned_pixels"]
    scores = val["scores"]["burned"]
    assert scores["tp"] + scores["fp"] == summary["burned_pixels"]
    assert summary["validation"]["recovery_reference_inside_fraction"] < 0.05

    report = json.loads(paths.report.read_text())
    assert report == result.report
    assert report["bytes_read"] > 0
    assert report["provenance"]["args_digest"] == report["args_digest"]
    # 2019 is read for the recovery reference's pre-fire check; the years listed are the years read.
    _, log = caldor_store
    assert report["provenance"]["years"] == [2019, 2020, 2022, 2023, 2024, 2025] == years_in(log)
    assert report["provenance"]["years_checked"] == report["provenance"]["years"]
    assert report["provenance"]["store"]["backend"] == "KeyLog"
    assert set(report["extra_outputs"]) == {"pca_a", "pca_b"}
    assert report["estimated_memory"] == memory_estimate(
        plan, n_recovery_years=4, years_per_block=6, block=kwargs["block"]
    )

    # Rerunning with the same arguments skips without reading a chunk.
    chunks_before = len(log.chunk_keys())
    again = run_wildfire(aoi_path, 2020, 2022, out, **kwargs)
    assert again.skipped and again.report == report and again.validation_written
    assert len(log.chunk_keys()) == chunks_before

    # Changing a parameter reruns; so does --force.
    changed = run_wildfire(aoi_path, 2020, 2022, out, **{**kwargs, "min_area_ha": 0.06})
    assert not changed.skipped and changed.report["args_digest"] != report["args_digest"]
    forced = run_wildfire(aoi_path, 2020, 2022, out, **{**kwargs, "min_area_ha": 0.06}, force=True)
    assert not forced.skipped and forced.report["args_digest"] == changed.report["args_digest"]


def test_wildfire_falls_back_to_the_stable_pair_when_nothing_burned(caldor_reader, caldor_aoi, tmp_path):
    # 2018 to 2019 is before the Caldor fire: no change, so Otsu is not bimodal and the 99th
    # percentile of the 2017 to 2018 distances takes over.
    aoi_path, aoi = caldor_aoi
    result = run_wildfire(aoi_path, 2018, 2019, tmp_path / "out", min_area_ha=0.05, reader=caldor_reader)
    th = result.summary["threshold"]
    assert th["method"] == "stable_p99" and th["fallback_ran"] and result.summary["fallback_ran"]
    assert th["stable_pair"] == [2017, 2018]
    assert th["otsu"] is not None and th["bimodal"] is False and th["valley_ratio"] > 0.9

    (plan,) = plan_aoi(caldor_reader, aoi).zones
    (blk,) = iter_plan_blocks(caldor_reader, plan, [2017, 2018])
    d = metrics.cosine_distance_map(blk.windows[2017], blk.windows[2018])
    keep = blk.mask & np.isfinite(d) & metrics.same_radar_regime(blk.windows[2017], blk.windows[2018])
    assert th["threshold"] == pytest.approx(float(np.percentile(d[keep], 99)), rel=1e-6)
    assert th["n_stable_values"] == int(keep.sum())
    # The fallback flags almost nothing; Otsu would have flagged close to half the window. That is
    # right when nothing changed, and wrong when the "stable" pair is not stable, so it is reported.
    assert result.summary["burned_fraction"] < 0.05
    assert th["otsu"]["high_mode_share"] > 0.3
    assert th["fallback_suspect"] is True and th["fallback_flagged_share"] < 0.005
    warnings = report_warnings(result)
    assert any("the stable pair (2017, 2018)" in w and "--threshold" in w for w in warnings)
    assert any("No burned pixel" in w for w in warnings)
    assert 2017 in result.report["provenance"]["years"]


def test_wildfire_without_a_stable_pair_keeps_otsu_and_warns(caldor_reader, caldor_aoi, tmp_path):
    aoi_path, _ = caldor_aoi
    result = run_wildfire(
        aoi_path, 2017, 2018, tmp_path / "out", min_area_ha=0.05, fire_year=2018, reader=caldor_reader
    )
    th = result.summary["threshold"]
    assert th["method"] == "otsu" and not th["fallback_ran"] and not th["fallback_available"]
    # year_b is the fire year here, which the report flags (survey N11).
    assert any("survey N11" in w for w in result.report["warnings"])
    assert th["bimodal"] is False
    assert any("no stable-pair fallback" in w for w in result.report["warnings"])
    assert result.report["provenance"]["years"] == [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


@pytest.fixture
def cli_reader(monkeypatch, caldor_reader):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(lambda **kwargs: caldor_reader))
    return caldor_reader


def test_cli_wildfire_runs_skips_and_validates(cli_reader, caldor_aoi, hansen_reference, tmp_path):
    aoi_path, _ = caldor_aoi
    out = tmp_path / "cli_out"
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    args += ["--min-area-ha", "0.05", "--validate", "--reference", str(hansen_reference)]
    runner = CliRunner()
    first = runner.invoke(app, args)
    assert first.exit_code == 0, (first.output, first.exception)
    assert "Burned:" in first.stdout and "IoU" in first.stdout and "Recovery" in first.stdout
    assert "GB of chunks" in first.stdout and "Recovery reference:" in first.stdout
    report = json.loads((out / "wildfire_report.json").read_text())
    assert (out / "wildfire_validation.json").is_file()
    assert report["summary"]["validation"]["iou"] > 0.6

    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "current" in second.stdout and "--force" in second.stdout
    assert json.loads((out / "wildfire_report.json").read_text()) == report


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--year-a", "2022", "--year-b", "2022"], "must be after year_a"),
        (["--year-a", "2020", "--year-b", "2022", "--recovery-through", "2026"], "recovery_through"),
        (["--year-a", "2020", "--year-b", "2022", "--threshold", "3"], "threshold"),
    ],
)
def test_cli_wildfire_usage_errors_exit_2(cli_reader, caldor_aoi, tmp_path, extra, message):
    aoi_path, _ = caldor_aoi
    result = CliRunner().invoke(app, ["wildfire", "--aoi", str(aoi_path), "--out", str(tmp_path), *extra])
    assert result.exit_code == EXIT_USAGE
    assert message in result.stderr
    assert not (tmp_path / "wildfire_report.json").exists()


def test_cli_wildfire_missing_aoi_exits_2(cli_reader, tmp_path):
    result = CliRunner().invoke(
        app, ["wildfire", "--aoi", str(tmp_path / "nope.geojson"), "--year-a", "2020", "--year-b", "2022"]
    )
    assert result.exit_code == EXIT_USAGE and "neither an existing file" in result.stderr


def _unreachable(*args, **kwargs):
    raise validate.ReferenceUnavailableError("services3.arcgis.com refused the request (403)")


def test_cli_wildfire_unreachable_reference_warns(cli_reader, caldor_aoi, tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    aoi_path, _ = caldor_aoi
    out = tmp_path / "out"
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    result = CliRunner().invoke(app, [*args, "--min-area-ha", "0.05", "--validate"])
    assert result.exit_code == 0, result.output
    # The run itself completes, without a validation file, and says why.
    report = json.loads((out / "wildfire_report.json").read_text())
    assert not (out / "wildfire_validation.json").exists()
    assert "validation" not in report["outputs"]
    assert any("services3.arcgis.com" in w for w in report["warnings"])
    assert "services3.arcgis.com" in report["summary"]["validation"]["error"]
    assert "Validation: not done" in result.stdout


def test_cli_wildfire_strict_validate_stops_before_reading(cli_reader, caldor_store, caldor_aoi, tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    _, log = caldor_store
    aoi_path, _ = caldor_aoi
    out = tmp_path / "out"
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    result = CliRunner().invoke(app, [*args, "--min-area-ha", "0.05", "--validate", "--strict-validate"])
    assert result.exit_code == EXIT_VALIDATION, result.output
    assert "--strict-validate" in result.stderr and "services3.arcgis.com" in result.stderr
    assert log.chunk_keys() == [] and not (out / "wildfire_report.json").exists()
    # The library raises the same before any chunk read.
    with pytest.raises(StrictValidationError):
        run_wildfire(aoi_path, 2020, 2022, out, validate=True, strict_validate=True, reader=cli_reader)
    assert log.chunk_keys() == []


# --------------------------------------------------------------------------------------------
# Review fixes: NIFC, fire year, threshold, early failures, reruns
# --------------------------------------------------------------------------------------------


@pytest.fixture
def fake_nifc(monkeypatch, caldor_aoi):
    """NIFC answering with the Hansen outline clipped to the AOI, recording every query."""
    _, aoi = caldor_aoi
    calls = []
    answer = [demo_feature("caldor", "hansen_2021_loss_outline").intersection(aoi)]

    def fake(bbox, *, year, name=None, url=validate.NIFC_HISTORY_URL):
        calls.append({"bbox": tuple(bbox), "year": year, "name": name})
        return list(answer)

    monkeypatch.setattr(validate, "nifc_perimeters", fake)
    return calls, answer


def test_wildfire_validates_against_nifc_for_the_fire_year(caldor_reader, caldor_aoi, fake_nifc, tmp_path):
    aoi_path, aoi = caldor_aoi
    calls, _ = fake_nifc
    result = run_wildfire(
        aoi_path,
        2020,
        2022,
        tmp_path / "out",
        min_area_ha=0.05,
        validate=True,
        fire_name="Caldor",
        reader=caldor_reader,
        block=(32, 32),
    )
    assert len(calls) == 1
    assert calls[0]["year"] == 2021 and calls[0]["name"] == "Caldor"
    assert calls[0]["bbox"] == pytest.approx(aoi.bounds)
    assert result.summary["fire_year"] == 2021 and result.validation_written
    val = json.loads(result.paths.validation.read_text())
    assert val["reference"]["source"] == "nifc" and val["reference"]["fire_year"] == 2021
    assert val["reference"]["fire_name"] == "Caldor" and val["reference"]["perimeters"] == 1
    assert result.summary["validation"]["source"] == "nifc"
    assert any("unburned islands" in c for c in val["caveats"])
    assert not any("stand-replacing" in c for c in val["caveats"])
    assert val["scores"]["burned"]["iou"] > 0.6


def test_wildfire_refuses_an_empty_nifc_answer(caldor_reader, caldor_aoi, fake_nifc, tmp_path):
    aoi_path, _ = caldor_aoi
    calls, answer = fake_nifc
    answer.clear()
    result = run_wildfire(
        aoi_path, 2020, 2022, tmp_path / "out", min_area_ha=0.05, validate=True, reader=caldor_reader, block=(32, 32)
    )
    assert calls[0]["year"] == 2021 and calls[0]["name"] is None
    assert not result.validation_written and not result.paths.validation.exists()
    assert "no perimeter" in result.summary["validation"]["error"]
    assert any(w.startswith("Validation skipped") and "no perimeter" in w for w in report_warnings(result))


def test_wildfire_with_the_fire_year_as_year_b(caldor_reader, caldor_aoi, fake_nifc, tmp_path):
    # Brief 4.1 allows year_b to be the fire year. Then year_a + 1 is the fire year: NIFC is asked
    # for 2021, the report warns (survey N11), and Otsu finds the burn although the histogram's high
    # mode is broad and Otsu's split sits on its shoulder (the valley is between the peaks).
    aoi_path, _ = caldor_aoi
    calls, _ = fake_nifc
    result = run_wildfire(
        aoi_path, 2020, 2021, tmp_path / "out", min_area_ha=0.05, validate=True, reader=caldor_reader, block=(32, 32)
    )
    s = result.summary
    assert s["fire_year"] == 2021 and calls[0]["year"] == 2021
    warnings = report_warnings(result)
    assert any("year_b (2021) is the fire year" in w and "survey N11" in w for w in warnings)
    assert not any("year_a (2020) is the fire year" in w for w in warnings)
    th = s["threshold"]
    assert th["method"] == "otsu" and th["bimodal"] is True and th["valley_ratio"] < 0.85
    assert not th["fallback_ran"]
    assert 0.3 < s["burned_fraction"] < 0.6
    assert s["validation"]["iou"] > 0.65


def test_a_contaminated_recovery_reference_is_visible(caldor_reader, caldor_aoi, tmp_path, monkeypatch):
    # The reference as it was before the review (every classified pixel at or below the threshold):
    # on this window it holds fire-affected land, so its similarity drops after the fire, recovery
    # is inflated, and the report says so.
    monkeypatch.setitem(wf.recovery_reference.__kwdefaults__, "distance_share", 1.0)
    monkeypatch.setitem(wf.recovery_reference.__kwdefaults__, "buffer_px", 0)
    aoi_path, _ = caldor_aoi
    result = run_wildfire(aoi_path, 2020, 2022, tmp_path / "out", min_area_ha=0.05, reader=caldor_reader)
    ref = result.summary["recovery_reference"]
    assert ref["pixels"] == ref["unburned_candidates"]
    assert ref["pre_minus_post_median"] > wf.REFERENCE_DRIFT_WARN
    assert any("less similar to itself" in w and "Extend the AOI" in w for w in report_warnings(result))
    rows = result.summary["recovery"]
    assert max(r["recovered_fraction"] for r in rows) > 0.05 and min(r["median_sim_unburned"] for r in rows) < 0.92


def test_wildfire_applies_the_minimum_area(caldor_reader, caldor_aoi, tmp_path):
    # After the opening, the scar's main patch and a smaller one (about 440 px) remain: 5 ha (500 px)
    # drops the smaller one.
    aoi_path, _ = caldor_aoi
    kw = {"reader": caldor_reader, "block": (32, 32)}
    small = run_wildfire(aoi_path, 2020, 2022, tmp_path / "a", min_area_ha=0.0, **kw)
    big = run_wildfire(aoi_path, 2020, 2022, tmp_path / "b", min_area_ha=5.0, **kw)
    assert big.summary["min_pixels"] == 500 and small.summary["min_pixels"] == 0
    assert big.summary["burned_pixels"] < small.summary["burned_pixels"]
    with rasterio.open(big.paths.tif) as ds:
        burned = ds.read(2) == 1
    assert metrics.component_sizes(burned).min() >= 500


def test_bytes_read_counts_this_run_only(caldor_reader, caldor_aoi, tmp_path):
    aoi_path, _ = caldor_aoi
    kw = {"threshold": 0.25, "min_area_ha": 0.05, "reader": caldor_reader, "block": (32, 32)}
    first = run_wildfire(aoi_path, 2020, 2022, tmp_path / "a", **kw)
    second = run_wildfire(aoi_path, 2020, 2022, tmp_path / "b", **kw)
    assert first.report["bytes_read"] > 0
    assert second.report["bytes_read"] == pytest.approx(first.report["bytes_read"], rel=0.05)
    assert second.report["bytes_read"] <= second.report["estimated_bytes"]


def test_run_level_warnings(caldor_reader, caldor_aoi, tmp_path):
    aoi_path, aoi = caldor_aoi
    west, south, east, north = aoi.bounds
    corner = box(west, south, west + (east - west) / 4, south + (north - south) / 4)
    ref = _feature_collection(tmp_path / "corner.geojson", corner, {})
    kw = {"min_area_ha": 0.05, "reader": caldor_reader, "block": (32, 32)}
    poor = run_wildfire(aoi_path, 2020, 2022, tmp_path / "a", validate=True, reference=ref, **kw)
    assert poor.summary["validation"]["iou_below_0_6"] is True
    assert any("stop and report" in w for w in report_warnings(poor))
    val = json.loads(poor.paths.validation.read_text())
    assert any("stop-and-report" in c for c in val["caveats"])
    nothing = run_wildfire(aoi_path, 2020, 2022, tmp_path / "b", threshold=1.99, **kw)
    assert nothing.summary["burned_pixels"] == 0
    assert any("No burned pixel" in w for w in report_warnings(nothing))


def test_pca_subsample_is_seeded(caldor_reader, caldor_aoi, tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "PCA_SAMPLE_CAP", 600)
    monkeypatch.setattr(wf, "PCA_SAMPLE_PER_BLOCK", 100)
    aoi_path, _ = caldor_aoi
    kw = {"threshold": 0.25, "min_area_ha": 0.05, "reader": caldor_reader, "block": (32, 32)}
    a = run_wildfire(aoi_path, 2020, 2022, tmp_path / "a", **kw)
    b = run_wildfire(aoi_path, 2020, 2022, tmp_path / "b", **kw)
    assert 0 < a.summary["pca"]["pixels"] <= 600, "the subsampling path ran"
    for key in ("pca_a", "pca_b"):
        assert a.report["extra_outputs"][key]["sha256"] == b.report["extra_outputs"][key]["sha256"]


@pytest.mark.parametrize("content", [b"<html><body>502 Bad Gateway</body></html>", b'{"type": "Featu'])
def test_a_bad_reference_file_stops_before_any_chunk_read(caldor_reader, caldor_store, caldor_aoi, tmp_path, content):
    _, log = caldor_store
    aoi_path, _ = caldor_aoi
    out = tmp_path / "out"
    kw = {"threshold": 0.25, "min_area_ha": 0.05, "reader": caldor_reader, "block": (32, 32)}
    earlier = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    reads = len(log.chunk_keys())
    bad = tmp_path / "perimeter.geojson"
    bad.write_bytes(content)
    with pytest.raises(AppInputError, match=r"perimeter\.geojson"):
        run_wildfire(aoi_path, 2020, 2022, out, validate=True, reference=bad, **kw)
    assert len(log.chunk_keys()) == reads, "no chunk is read before the reference is checked"
    assert json.loads(earlier.paths.report.read_text()) == earlier.report, "the previous outputs are kept"


def test_cli_wildfire_bad_reference_exits_2(cli_reader, caldor_store, caldor_aoi, tmp_path):
    _, log = caldor_store
    aoi_path, _ = caldor_aoi
    bad = tmp_path / "perimeter.geojson"
    bad.write_text("<html>404</html>")
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(tmp_path / "o")]
    result = CliRunner().invoke(app, [*args, "--reference", str(bad)])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "perimeter.geojson" in result.stderr
    assert log.chunk_keys() == []


def test_cli_wildfire_unwritable_out_exits_2(cli_reader, caldor_store, caldor_aoi, tmp_path):
    _, log = caldor_store
    aoi_path, _ = caldor_aoi
    blocker = tmp_path / "results.txt"
    blocker.write_text("a file, not a directory")
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(blocker / "out")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == EXIT_USAGE, result.output
    assert "--out" in result.stderr and "results.txt" in result.stderr
    assert log.chunk_keys() == []


def test_validation_is_redone_from_the_outputs_without_reading(
    caldor_reader, caldor_store, caldor_aoi, fake_nifc, tmp_path, monkeypatch
):
    # A --validate run whose reference could not be fetched is current except validation.json: an
    # identical rerun must not stream the AOI again just to retry the fetch (survey S19).
    _, log = caldor_store
    aoi_path, _ = caldor_aoi
    calls, _ = fake_nifc
    working = validate.nifc_perimeters
    out = tmp_path / "out"
    kw = {"min_area_ha": 0.05, "validate": True, "reader": caldor_reader, "block": (32, 32)}
    monkeypatch.setattr(validate, "nifc_perimeters", _unreachable)
    first = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    assert not first.validation_written and any(w.startswith("Validation skipped") for w in first.report["warnings"])
    reads = len(log.chunk_keys())
    # Still unreachable: nothing is read or written.
    again = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    assert again.skipped and not again.validation_written and again.report == first.report
    assert len(log.chunk_keys()) == reads
    # Reachable again: the validation is redone from the GeoTIFF, still without a chunk read.
    monkeypatch.setattr(validate, "nifc_perimeters", working)
    redone = run_wildfire(aoi_path, 2020, 2022, out, **kw)
    assert redone.validation_redone and redone.validation_written and not redone.skipped
    assert len(log.chunk_keys()) == reads and len(calls) == 1
    report = json.loads(redone.paths.report.read_text())
    assert report == redone.report and report["args_digest"] == first.report["args_digest"]
    assert report["validation_redone"]["bytes_read"] == 0
    assert not any(w.startswith("Validation skipped") for w in report["warnings"])
    assert "validation" in report["outputs"]
    # The scores are those of a full validating run (one zone: identical).
    full = run_wildfire(aoi_path, 2020, 2022, tmp_path / "full", **kw)
    redone_val = json.loads(redone.paths.validation.read_text())
    full_val = json.loads(full.paths.validation.read_text())
    assert redone_val["scores"] == full_val["scores"]
    assert redone.summary["validation"] == full.summary["validation"]
    assert any("redone" in c for c in redone_val["caveats"])
    # And now the outputs are current.
    assert run_wildfire(aoi_path, 2020, 2022, out, **kw).skipped


def test_cli_wildfire_redone_validation_message(cli_reader, caldor_aoi, fake_nifc, tmp_path, monkeypatch):
    aoi_path, _ = caldor_aoi
    out = tmp_path / "out"
    args = ["wildfire", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    args += ["--min-area-ha", "0.05"]
    assert CliRunner().invoke(app, args).exit_code == 0
    result = CliRunner().invoke(app, [*args, "--validate", "--strict-validate"])
    assert result.exit_code == 0, result.output
    assert "validation was redone" in result.stdout and "IoU" in result.stdout


def test_cli_wildfire_store_errors_exit_2_with_notes(monkeypatch, boundary_reader, tmp_path):
    # TesseraError subclasses (missing years, a large AOI) are usage errors with their notes.
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(lambda **kwargs: boundary_reader))
    lon0, lat0 = BOUNDARY
    aoi = _feature_collection(tmp_path / "a.geojson", box(lon0 - 0.004, lat0 - 0.003, lon0 + 0.003, lat0 + 0.003), {})
    args = ["wildfire", "--aoi", str(aoi), "--out", str(tmp_path / "o")]
    missing = CliRunner().invoke(app, [*args, "--year-a", "2024", "--year-b", "2025"])
    assert missing.exit_code == EXIT_USAGE, missing.output
    assert "11N: 2023" in missing.stderr and "--threshold" in missing.stderr
    big = _feature_collection(tmp_path / "big.geojson", box(-121.5, 38.0, -120.0, 39.0), {})
    large = CliRunner().invoke(app, ["wildfire", "--aoi", str(big), "--year-a", "2020", "--year-b", "2022"])
    assert large.exit_code == EXIT_USAGE and "10,000 km2 limit" in large.stderr


def test_cli_wildfire_forwards_every_option(monkeypatch, tmp_path):
    seen = {}
    written = {"value": True}

    def record(aoi, year_a, year_b, out, **kwargs):
        seen.update(kwargs, aoi=aoi, year_a=year_a, year_b=year_b, out=out)
        return WildfireResult(
            report={"summary": {}},
            paths=export.output_paths(Path(out), "wildfire", year_a, year_b),
            extra_paths={},
            skipped=False,
            validation_written=written["value"],
        )

    monkeypatch.setattr("tessera_apps.apps.run_wildfire", record)
    ref = tmp_path / "r.geojson"
    ref.write_text("{}")
    args = ["wildfire", "--aoi", "caldor", "--year-a", "2019", "--year-b", "2021", "--out", str(tmp_path)]
    args += ["--reference", str(ref), "--fire-year", "2020", "--fire-name", "Caldor", "--threshold", "0.3"]
    args += ["--min-area-ha", "2.5", "--recovery-through", "2024", "--force", "--snapshot", "SNAP"]
    args += ["--i-know-this-is-large", "--strict-validate"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen["aoi"] == "caldor" and (seen["year_a"], seen["year_b"], seen["out"]) == (2019, 2021, tmp_path)
    assert seen["validate"] is True and seen["reference"] == ref  # --reference implies --validate
    assert seen["snapshot_id"] == "SNAP" and seen["max_area_km2"] is None and seen["force"] is True
    assert seen["fire_year"] == 2020 and seen["fire_name"] == "Caldor" and seen["threshold"] == 0.3
    assert seen["min_area_ha"] == 2.5 and seen["recovery_through"] == 2024 and seen["strict_validate"] is True
    # Defaults: the 10,000 km2 limit, no snapshot, no force, no validation.
    seen.clear()
    defaults = CliRunner().invoke(app, ["wildfire", "--aoi", "caldor", "--year-a", "2020", "--year-b", "2022"])
    assert defaults.exit_code == 0, defaults.output
    assert seen["max_area_km2"] == 10_000 and seen["snapshot_id"] is None and seen["force"] is False
    assert seen["validate"] is False and seen["strict_validate"] is False and seen["out"] == Path("out")
    # A strict run whose validation was not written still exits 3.
    written["value"] = False
    base = ["wildfire", "--aoi", "caldor", "--year-a", "2020", "--year-b", "2022", "--out", str(tmp_path)]
    strict = CliRunner().invoke(app, [*base, "--validate", "--strict-validate"])
    assert strict.exit_code == EXIT_VALIDATION


# --------------------------------------------------------------------------------------------
# Network tier: a small live window, anonymous (run with -m network)
# --------------------------------------------------------------------------------------------

LIVE_WINDOW = box(-120.375, 38.68, -120.355, 38.695)  # about 290 ha inside the Caldor scar, 10N
LIVE_SNAPSHOT = "B6AFRJGX5XQC0A716PA0"


@pytest.mark.network
def test_wildfire_live_window(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key)
    aoi = _feature_collection(tmp_path / "live.geojson", LIVE_WINDOW, {})
    outline = demo_feature("caldor", "hansen_2021_loss_outline").intersection(LIVE_WINDOW)
    ref = _feature_collection(tmp_path / "ref.geojson", outline, {"role": "hansen_2021_loss_outline"})
    result = run_wildfire(aoi, 2020, 2022, tmp_path / "out", validate=True, reference=ref, snapshot_id=LIVE_SNAPSHOT)
    s = result.summary
    assert s["zones"] == ["10N"] and s["threshold"]["method"] == "otsu"
    assert 0.2 < s["burned_fraction"] < 0.7
    assert s["validation"]["iou"] > 0.6
    assert result.report["provenance"]["store"]["snapshot_id"] == LIVE_SNAPSHOT
    assert 0 < result.report["bytes_read"] <= result.report["estimated_bytes"]
    # The window lies inside the scar: the recovery reference (land shown unchanged) was stable
    # before the fire and its drift after it is reported. Every pixel at or below the threshold
    # (the reference before the review) put 21 percent of the reference inside the Hansen outline
    # and up to 2.3 percent of burned pixels "recovered"; land shown unchanged puts 9 percent inside
    # and 0.2 percent.
    ref = s["recovery_reference"]
    assert ref["pre_fire_similarity"]["median"] > 0.95 and ref["pre_minus_post_median"] is not None
    assert s["validation"]["recovery_reference_inside_fraction"] < 0.15
    assert all(row["recovered_fraction"] < 0.01 for row in s["recovery"])


# --------------------------------------------------------------------------------------------
# Deforest: end to end on the caldor fixture, labels and validation from local Hansen tiles
# --------------------------------------------------------------------------------------------

DEFOREST_ROWS = DEFOREST_COLS = (12, 52)
DEFOREST_KW = {"label_ring_km": 0.1, "labels_per_class": 100, "min_area_ha": 0.05, "block": (32, 32)}
DEFOREST_BANDS = ("forest_prob_2019", "forest_prob_2022", "distance_2019_2022", "converted")
DEFOREST_FILES = (
    "deforest_2019_2022.tif",
    "deforest_summary.parquet",
    "deforest_summary.csv",
    "deforest_labels.parquet",
    "deforest_preview.png",
    "deforest_validation.json",
    "deforest_report.json",
)
LABEL_COLUMNS = [
    "lon",
    "lat",
    "label",
    "weight",
    "source",
    "source_chunk",
    "zone",
    "row",
    "col",
    "valid",
    "regime",
    "used",
    "drop_reason",
    "fold",
    "heldout_prob",
    "inside_aoi",
    "hansen_lossyear",
    "hansen_stable",
    "prob_year_a",
    "prob_year_b",
    "distance",
    "transfer_scored",
]


@pytest.fixture
def deforest_hansen(caldor_hansen, tmp_path, monkeypatch) -> str:
    """The Hansen tile pieces under the fixture as local tiles, with the reference cache under tmp_path."""
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))
    return write_hansen_tiles(tmp_path / "hansen", caldor_hansen)


@pytest.fixture
def deforest_aoi(caldor_fixture, tmp_path):
    """The central 40 x 40 pixels of the fixture window as a lon/lat GeoJSON, and its geometry."""
    geometry = fixture_window_lonlat(caldor_fixture, DEFOREST_ROWS, DEFOREST_COLS)
    return _feature_collection(tmp_path / "deforest_window.geojson", geometry, {"name": "deforest window"}), geometry


def _unreachable_hansen(*args, **kwargs):
    raise validate.ReferenceUnavailableError("storage.googleapis.com refused the request (403)")


def test_deforest_end_to_end_on_the_caldor_fixture(
    caldor_reader, caldor_store, deforest_aoi, deforest_hansen, tmp_path
):
    aoi_path, aoi = deforest_aoi
    out = tmp_path / "out"
    kwargs = {**DEFOREST_KW, "validate": True, "reader": caldor_reader, "hansen_url_template": deforest_hansen}
    result = run_deforest(aoi_path, 2019, 2022, out, **kwargs)
    assert isinstance(result, DeforestResult) and not result.skipped and result.validation_written
    paths = result.paths
    for p in (*paths.required(need_validation=True), *result.extra_paths.values()):
        assert p.is_file(), p
    assert sorted(p.name for p in out.iterdir()) == sorted(DEFOREST_FILES)

    # The tif is a valid float32 COG with the four named bands, on the AOI's window.
    ok, errors, _ = cog_validate(str(paths.tif))
    assert ok, errors
    plan = plan_aoi(caldor_reader, aoi)
    (zp,) = plan.zones
    with rasterio.open(paths.tif) as ds:
        assert ds.descriptions == DEFOREST_BANDS
        assert ds.dtypes == ("float32",) * 4 and np.isnan(ds.nodata)
        assert ds.transform == zp.transform and ds.shape == zp.shape
        prob_a, prob_b, dist, converted = ds.read()
    inside = zp.mask
    s = result.summary
    assert s["aoi_pixels"] == plan.n_pixels == 1600 == s["classified_pixels"]
    assert s["zones"] == ["10N"] and s["label_source"] == "hansen" and (s["year_a"], s["year_b"]) == (2019, 2022)
    for band in (prob_a, prob_b):
        assert np.isfinite(band[inside]).all() and np.isnan(band[~inside]).all()
        assert band[inside].min() >= 0.0 and band[inside].max() <= 1.0
    expected = metrics.cosine_distance_map(read_plan(caldor_reader, zp, 2019), read_plan(caldor_reader, zp, 2022))
    np.testing.assert_allclose(dist[inside], expected[inside], atol=1e-6)
    assert set(np.unique(converted[inside]).tolist()) == {0.0, 1.0}

    # The conversion is the rule then the cleaning: forest in 2019, non-forest in 2022, beyond the cut.
    cut = s["cross_check"]["cut"]
    stable = inside & (prob_a >= 0.5) & (prob_b >= 0.5)
    assert cut == pytest.approx(float(np.percentile(dist[stable].astype(np.float64), 95)), rel=1e-6)
    assert s["cross_check"]["stable_forest_pixels"] == int(stable.sum()) >= dfr.MIN_STABLE_PIXELS
    raw = inside & (prob_a >= 0.5) & (prob_b < 0.5) & (dist > cut)
    flips = inside & (prob_a >= 0.5) & (prob_b < 0.5)
    assert s["converted_pixels_before_cleaning"] == int(raw.sum())
    assert s["cross_check"]["classifier_flips_below_cut"] == int((flips & ~raw).sum()) > 0
    done = converted == 1
    np.testing.assert_array_equal(done, metrics.clean_mask(raw, min_pixels=5) & inside)
    assert s["converted_pixels"] == int(done.sum()) > 100 and s["min_pixels"] == 5
    assert s["converted_ha"] == pytest.approx(done.sum() * 0.01)
    forest_a, forest_b = int((prob_a[inside] >= 0.5).sum()), int((prob_b[inside] >= 0.5).sum())
    assert s["forest"]["2019"] == {
        "forest_ha": pytest.approx(forest_a * 0.01),
        "forest_fraction": pytest.approx(forest_a / 1600),
        "forest_pixels": forest_a,
        "classified_pixels": 1600,
    }
    assert s["forest"]["2022"]["forest_pixels"] == forest_b < forest_a
    assert s["conversion_fraction_of_forest"] == pytest.approx(done.sum() / forest_a)
    sizes = metrics.component_sizes(done)
    patches = s["patches"]
    assert patches["count"] == sizes.size > 0 and sum(patches["histogram"].values()) == sizes.size
    assert patches["max_ha"] == pytest.approx(sizes.max() * 0.01)
    assert patches["median_ha"] == pytest.approx(float(np.median(sizes)) * 0.01)

    # One summary row per year, the conversion columns repeated, the same in parquet and csv.
    table = pd.read_parquet(paths.summary_parquet)
    pd.testing.assert_frame_equal(table, pd.read_csv(paths.summary_csv), check_dtype=False)
    assert table["year"].tolist() == [2019, 2022]
    assert table["forest_ha"].tolist() == pytest.approx([forest_a * 0.01, forest_b * 0.01])
    assert table["forest_fraction"].tolist() == pytest.approx([forest_a / 1600, forest_b / 1600])
    for col, value in (
        ("converted_ha", s["converted_ha"]),
        ("conversion_fraction_of_forest", s["conversion_fraction_of_forest"]),
        ("n_patches", sizes.size),
        ("cross_check_cut", cut),
        ("label_source", "hansen"),
    ):
        assert (table[col] == value).all(), col

    # The labels: sampled in the ring, never inside the AOI, by the Hansen rules, held out by chunk.
    labels = pd.read_parquet(result.extra_paths["labels"])
    assert list(labels.columns) == LABEL_COLUMNS
    assert len(labels) == 200 and int((labels["label"] == 1).sum()) == 100
    assert not labels["inside_aoi"].any() and (labels["source"] == "hansen").all()
    in_window = (
        (labels["row"] >= zp.rows.start)
        & (labels["row"] < zp.rows.stop)
        & (labels["col"] >= zp.cols.start)
        & (labels["col"] < zp.cols.stop)
    )
    picked = labels[in_window]
    assert not zp.mask[picked["row"] - zp.rows.start, picked["col"] - zp.cols.start].any()
    ring = dfr.label_ring(aoi, outer_km=0.1)
    (rz,) = dfr.plan_label_ring(caldor_reader, ring, plan).zones
    assert rz.mask[labels["row"] - rz.rows.start, labels["col"] - rz.cols.start].all()
    layers = {
        name: validate.hansen_layer(rz, name, url_template=deforest_hansen)
        for name in ("treecover2000", "lossyear", "datamask")
    }
    classes = dfr.label_classes(layers["treecover2000"], layers["lossyear"], layers["datamask"], 2019)
    assert (classes[labels["row"] - rz.rows.start, labels["col"] - rz.cols.start] == labels["label"]).all()
    assert labels["used"].all() and (labels["regime"] == "radar").all() and (labels["drop_reason"] == "").all()
    assert labels["fold"].between(0, 4).all() and labels["heldout_prob"].notna().all()
    assert labels.groupby("source_chunk")["fold"].nunique().max() == 1
    assert set(labels["source_chunk"]) <= set(s["labels"]["chunks"])
    # Transfer to 2022 at the labels Hansen keeps unchanged, recomputed from a refit of the same
    # model and an independent read of both years at the labels (a reader that leaves the key log alone).
    ring_loss = layers["lossyear"][labels["row"] - rz.rows.start, labels["col"] - rz.cols.start]
    assert (labels["hansen_lossyear"] == ring_loss).all()
    stable = (labels["label"] == 0) | (ring_loss == 0) | (ring_loss > 22)
    assert (labels["hansen_stable"] == stable).all() and 0 < stable.sum() < len(labels)
    quiet = TesseraReader.from_store(caldor_store[0])
    lon, lat = labels["lon"].to_numpy(), labels["lat"].to_numpy()
    pts_a, pts_b = read_points(quiet, lon, lat, 2019), read_points(quiet, lon, lat, 2022)
    y = labels["label"].to_numpy()
    refit = dfr.fit_forest_model(pts_a.embeddings, y, labels["source_chunk"].to_numpy(), seed=0)
    assert refit.record["heldout"] == s["classifier"]["heldout"]
    np.testing.assert_allclose(labels["prob_year_a"], refit.model.probability_vectors(pts_a.embeddings), rtol=1e-5)
    expected_b = refit.model.probability_vectors(pts_b.embeddings)
    np.testing.assert_allclose(labels["prob_year_b"], expected_b, rtol=1e-5)
    np.testing.assert_allclose(labels["distance"], dfr.vector_distance(pts_a.embeddings, pts_b.embeddings), rtol=1e-5)
    scored = stable.to_numpy() & pts_b.valid
    assert (labels["transfer_scored"] == scored).all()
    tr = s["transfer"]
    assert tr["available"] and tr["years"] == [2019, 2022] and tr["labels"] == int(scored.sum())
    assert (tr["labels_forest"], tr["labels_nonforest"]) == (
        int((scored & (y == 1)).sum()),
        int((scored & (y == 0)).sum()),
    )
    pb = expected_b[scored] >= 0.5
    assert tr["year_b"]["recall_forest"] == pytest.approx(float(pb[y[scored] == 1].mean()))
    assert tr["year_b"]["recall_nonforest"] == pytest.approx(float((~pb[y[scored] == 0]).mean()))
    held_b = refit.heldout_prob[scored] >= 0.5
    held_bal = (held_b[y[scored] == 1].mean() + (~held_b[y[scored] == 0]).mean()) / 2
    assert tr["year_a_heldout"]["balanced_accuracy"] == pytest.approx(float(held_bal))
    assert tr["drop"] == pytest.approx(held_bal - tr["year_b"]["balanced_accuracy"])
    # The ring label cut: the 95th percentile of the distance of the Hansen-stable forest labels.
    ring_set = scored & (y == 1)
    expected_ring_cut = float(np.percentile(dfr.vector_distance(pts_a.embeddings, pts_b.embeddings)[ring_set], 95))
    cross = s["cross_check"]
    assert cross["ring_label_cut"] == pytest.approx(expected_ring_cut, rel=1e-6)
    assert cross["ring_label_cut_labels"] == int(ring_set.sum()) >= dfr.MIN_RING_CUT_LABELS
    assert cross["cut_to_ring_label_cut"] == pytest.approx(cross["cut"] / expected_ring_cut, rel=1e-6)
    lab = s["labels"]
    assert lab["used_forest"] == lab["used_nonforest"] == 100 and lab["chunks_used"] == len(lab["chunks"]) <= 24
    assert lab["ring"]["outer_km"] == 0.1 and lab["ring"]["inner_km"] == 0.0 and lab["ring"]["zones"] == ["10N"]
    clf = s["classifier"]
    assert clf["labels"] == 200 and clf["heldout"]["folds"] == 5 and clf["heldout"]["folds_skipped"] == 0
    assert clf["training"]["accuracy"] > 0.9 and 0.7 < clf["heldout"]["balanced_accuracy"] <= 1.0
    radar = s["radar"]
    assert radar["training_regime"] == "radar" and radar["regime_mismatch_pixels"] == 0
    assert radar["probe"]["pixels_radar"] == 1600 and radar["labels_dropped_regime"] == 0

    # Validation against Hansen loss in (2019, 2022], with the narrower 2022-only comparison.
    val = json.loads(paths.validation.read_text())
    assert set(val["scores"]) == {
        "conversion",
        "conversion_per_zone",
        "conversion_lag_aligned",
        "conversion_lag_aligned_per_zone",
        "conversion_year_b_only",
        "forest_agreement_year_a",
        "forest_agreement_year_b",
        "loss_year_breakdown",
        "cross_check_removed",
    }
    breakdown = val["scores"]["loss_year_breakdown"]
    assert sum(v["pixels"] for v in breakdown.values()) == 1600
    lossyear = validate.hansen_layer(zp, "lossyear", url_template=deforest_hansen)
    below = flips & ~raw
    for year, row in breakdown.items():
        code = 0 if year == "no_loss" else int(year) - 2000
        here = inside & (lossyear == code)
        assert row["pixels"] == int(here.sum()) and row["converted_share"] == pytest.approx(float(done[here].mean()))
        assert row["flip_share"] == pytest.approx(float(flips[here].mean()))
        assert row["below_cut_share"] == pytest.approx(float(below[here].mean()))
    # The flips the cross-check removed, split by Hansen loss (survey N13's cut can remove real loss).
    codes = lossyear[below].astype(int)
    removed = {
        "flips_below_cut": int(below.sum()),
        "loss_in_window": int(((codes >= 20) & (codes <= 22)).sum()),
        "loss_year_a": int((codes == 19).sum()),
        "loss_other_years": int(((codes > 0) & ((codes < 19) | (codes > 22))).sum()),
        "no_loss": int((codes == 0).sum()),
        "loss_lag_aligned": int(((codes >= 19) & (codes <= 21)).sum()),
    }
    assert removed["loss_in_window"] > 0 and removed["no_loss"] > 0
    assert val["scores"]["cross_check_removed"] == removed == s["validation"]["cross_check_removed"]
    assert s["cross_check"]["removed_by_hansen_loss"] == removed
    # The lag-aligned window [2019, 2021] the two annual embeddings mostly see, scored like the headline.
    lag_ref = inside & (lossyear >= 19) & (lossyear <= 21)
    lag = val["scores"]["conversion_lag_aligned"]
    assert lag["n"] == 1600 and lag["tp"] + lag["fn"] == int(lag_ref.sum()) and lag["tp"] == int((done & lag_ref).sum())
    assert lag["tp"] + lag["fp"] == int(done.sum())
    summary_lag = s["validation"]["lag_aligned"]
    assert summary_lag["window"] == {"from_year": 2019, "through_year": 2021, "lossyear_codes": [19, 20, 21]}
    assert (summary_lag["precision"], summary_lag["recall"], summary_lag["iou"]) == (
        lag["precision"],
        lag["recall"],
        lag["iou"],
    )
    assert s["validation"]["window"] == {"after_year": 2019, "through_year": 2022, "lossyear_codes": [20, 21, 22]}
    # Both years' forest maps against Hansen's forest proxy for that year, over that year's pixels.
    cover = validate.hansen_layer(zp, "treecover2000", url_template=deforest_hansen)
    for year, prob, key in ((2019, prob_a, "forest_agreement_year_a"), (2022, prob_b, "forest_agreement_year_b")):
        proxy = (cover >= 50) & ~((lossyear >= 1) & (lossyear <= year - 2000)) & inside
        mapped = prob >= 0.5
        agree = val["scores"][key]
        assert agree["pixels"] == 1600 == s["validation"][key]["pixels"]
        assert agree["accuracy"] == pytest.approx(float(np.mean(mapped[inside] == proxy[inside])))
        assert agree["recall_forest"] == pytest.approx(float(mapped[proxy].mean()))
        assert agree["recall_nonforest"] == pytest.approx(float((~mapped[inside & ~proxy]).mean()))
        assert s["validation"][key]["accuracy"] == agree["accuracy"]
    conv = val["scores"]["conversion"]
    assert conv["n"] == 1600 and conv["tp"] + conv["fp"] == s["converted_pixels"]
    loss = validate.hansen_loss(zp, 2019, 2022, url_template=deforest_hansen)
    assert conv["tp"] + conv["fn"] == int(loss.sum()) == s["validation"]["reference_pixels_in_aoi"]
    assert s["validation"]["precision"] == conv["precision"] > 0.8 and conv["recall"] > 0.3
    assert s["validation"]["iou"] == conv["iou"] and s["validation"]["pixels_scored"] == 1600
    assert s["validation"]["year_b_only"]["precision"] <= conv["precision"]
    assert 0.0 < s["validation"]["forest_agreement_year_a"]["accuracy"] <= 1.0
    ref = val["reference"]
    assert ref["source"] == "hansen" and ref["version"] == validate.HANSEN_VERSION and ref["label_source"] == "hansen"
    assert ref["loss_window"]["lossyear_codes"] == [20, 21, 22] and ref["url_template"] == deforest_hansen
    caveats = " ".join(val["caveats"])
    for text in ("not independent", "30 m", "2022 alone", "salvage logging", "spatially held out", "Tree cover"):
        assert text in caveats, text

    report = json.loads(paths.report.read_text())
    assert report == result.report and report["app"] == "deforest"
    _, log = caldor_store
    assert report["provenance"]["years"] == [2019, 2022] == years_in(log)
    assert report["provenance"]["years_checked"] == [2019, 2022]
    assert set(report["extra_outputs"]) == {"labels"}
    assert 0 < report["bytes_read"] <= report["estimated_bytes"]
    # The estimate is the AOI stream and probe plus the label chunks, read in both years for Hansen labels.
    chunk_bytes = 32 * 32 * BYTES_PER_PIXEL
    assert lab["label_read_years"] == [2019, 2022] and lab["label_read_chunks"] == labels["source_chunk"].nunique()
    assert report["estimated_bytes"] == plan.estimated_bytes(caldor_reader, 3, block=(32, 32)) + (
        2 * lab["label_read_chunks"] * chunk_bytes
    )
    assert report["estimated_memory"] == dfr.memory_estimate(plan, block=(32, 32), ring_km=0.1)
    assert report["params"]["hansen_url_template"] == deforest_hansen
    assert report["params"]["hansen_version"] == validate.HANSEN_VERSION
    assert set(report["timings"]) >= {
        "labels_read",
        "regime_probe",
        "classifier",
        "transfer",
        "stream",
        "clean",
        "write",
    }
    assert report["provenance"]["zones"] == ["10N"] and "zone_years_beyond_aoi" not in report["provenance"]
    assert not report["warnings"]

    # Rerunning with the same arguments skips without reading a chunk.
    chunks_before = len(log.chunk_keys())
    again = run_deforest(aoi_path, 2019, 2022, out, **kwargs)
    assert again.skipped and again.report == report and again.validation_written
    assert len(log.chunk_keys()) == chunks_before
    # The same arguments elsewhere give the same outputs; another seed draws other labels.
    twin = run_deforest(aoi_path, 2019, 2022, tmp_path / "twin", **kwargs)
    assert twin.report["outputs"] == report["outputs"] and twin.report["extra_outputs"] == report["extra_outputs"]
    reseeded = run_deforest(aoi_path, 2019, 2022, out, **kwargs, seed=1)
    assert not reseeded.skipped and reseeded.report["args_digest"] != report["args_digest"]
    other = pd.read_parquet(reseeded.extra_paths["labels"])
    assert not other[["row", "col"]].equals(labels[["row", "col"]])
    assert reseeded.report["provenance"]["seed"] == 1


def test_deforest_with_a_labels_csv(
    caldor_reader, caldor_store, caldor_fixture, deforest_aoi, deforest_hansen, tmp_path
):
    aoi_path, _ = deforest_aoi
    hansen = run_deforest(
        aoi_path,
        2019,
        2022,
        tmp_path / "hansen_run",
        reader=caldor_reader,
        hansen_url_template=deforest_hansen,
        **DEFOREST_KW,
    )
    sampled = pd.read_parquet(hansen.extra_paths["labels"])
    rows = pd.concat([sampled[sampled["label"] == 1].head(30), sampled[sampled["label"] == 0].head(30)])
    frame = rows[["lon", "lat", "label"]].assign(weight=2.0)
    # Two hand labels inside the AOI and five in the store's no-data padding.
    a, _, c, _, _, f = (float(v) for v in caldor_fixture["transform"])
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    extra = [(*to_ll.transform(c + 30.5 * a, f - 30.5 * a), 1), (*to_ll.transform(c + 31.5 * a, f - 31.5 * a), 1)]
    extra += [(*to_ll.transform(c - 10.5 * a, f - (10.5 + i) * a), i % 2) for i in range(5)]
    frame = pd.concat([frame, pd.DataFrame(extra, columns=["lon", "lat", "label"]).assign(weight=2.0)])
    csv = tmp_path / "hand_labels.csv"
    frame.to_csv(csv, index=False)
    nowhere = str(tmp_path / "nowhere" / "Hansen_{version}_{layer}_{tile}.tif")
    out = tmp_path / "csv_run"
    kw = {**DEFOREST_KW, "labels": csv, "reader": caldor_reader, "hansen_url_template": nowhere}
    result = run_deforest(aoi_path, 2019, 2022, out, **kw)
    s = result.summary
    assert s["label_source"] == "csv"
    lab = s["labels"]
    assert lab["file"] == "hand_labels.csv" and lab["sha256"] == export.file_sha256(csv)
    assert (lab["rows"], lab["inside_aoi"], lab["dropped_no_data"]) == (67, 2, 5)
    assert lab["used_forest"] == 32 and lab["used_nonforest"] == 30
    assert s["classifier"]["labels"] == 62 and s["classifier"]["weighted"] is False
    written = pd.read_parquet(result.extra_paths["labels"])
    back = pd.read_csv(csv)
    np.testing.assert_array_equal(written["lon"].to_numpy(), back["lon"].to_numpy())
    np.testing.assert_array_equal(written["label"].to_numpy(), back["label"].to_numpy())
    assert (written["source"] == "csv").all() and int(written["inside_aoi"].sum()) == 2
    assert (written["drop_reason"] == "no_data").sum() == 5 and not written.loc[~written["valid"], "used"].any()
    assert (written["weight"] == 2.0).all()
    assert any("lie inside the AOI" in w for w in result.report["warnings"])
    assert (pd.read_csv(result.paths.summary_csv)["label_source"] == "csv").all()
    assert s["hansen"]["labels"] is False
    # CSV labels are read in year_a only, and carry no year_b truth for the transfer check.
    assert lab["label_read_years"] == [2019] and s["transfer"]["available"] is False
    assert written["prob_year_b"].isna().all() and not written["transfer_scored"].any()
    assert (written["hansen_lossyear"] == -1).all() and not written["hansen_stable"].any()
    plan = plan_aoi(caldor_reader, aoi_path_geometry(aoi_path))
    assert result.report["estimated_bytes"] == plan.estimated_bytes(caldor_reader, 3, block=(32, 32)) + (
        lab["label_read_chunks"] * 32 * 32 * BYTES_PER_PIXEL
    )
    # Editing the labels file reruns.
    assert run_deforest(aoi_path, 2019, 2022, out, **kw).skipped
    csv.write_text(csv.read_text().replace(",2.0", ",1.0"), encoding="utf-8")
    assert not run_deforest(aoi_path, 2019, 2022, out, **kw).skipped


def aoi_path_geometry(path: Path):
    """The lon/lat geometry of an AOI file this module wrote."""
    return shapely.geometry.shape(json.loads(Path(path).read_text())["features"][0]["geometry"])


def test_deforest_validation_is_redone_from_the_outputs_without_reading(
    caldor_reader, caldor_store, deforest_aoi, deforest_hansen, tmp_path, monkeypatch
):
    _, log = caldor_store
    aoi_path, _ = deforest_aoi
    working = validate.hansen_loss
    out = tmp_path / "out"
    kw = {**DEFOREST_KW, "validate": True, "reader": caldor_reader, "hansen_url_template": deforest_hansen}
    monkeypatch.setattr(validate, "hansen_loss", _unreachable_hansen)
    first = run_deforest(aoi_path, 2019, 2022, out, **kw)
    assert not first.validation_written and not first.paths.validation.exists()
    assert "storage.googleapis.com" in first.summary["validation"]["error"]
    assert any(w.startswith("Validation skipped") for w in first.report["warnings"])
    reads = len(log.chunk_keys())
    again = run_deforest(aoi_path, 2019, 2022, out, **kw)
    assert again.skipped and not again.validation_written and again.report == first.report
    assert len(log.chunk_keys()) == reads
    monkeypatch.setattr(validate, "hansen_loss", working)
    redone = run_deforest(aoi_path, 2019, 2022, out, **kw)
    assert redone.validation_redone and redone.validation_written and not redone.skipped
    assert len(log.chunk_keys()) == reads
    report = json.loads(redone.paths.report.read_text())
    assert report == redone.report and report["args_digest"] == first.report["args_digest"]
    assert report["validation_redone"]["bytes_read"] == 0 and "validation" in report["outputs"]
    assert not any(w.startswith("Validation skipped") for w in report["warnings"])
    full = run_deforest(aoi_path, 2019, 2022, tmp_path / "full", **kw)
    redone_val = json.loads(redone.paths.validation.read_text())
    full_val = json.loads(full.paths.validation.read_text())
    assert redone_val["scores"] == full_val["scores"]
    assert redone.summary["validation"] == full.summary["validation"]
    assert any("redone" in c for c in redone_val["caveats"])
    assert any("not independent" in c for c in redone_val["caveats"])
    assert run_deforest(aoi_path, 2019, 2022, out, **kw).skipped


def test_cli_deforest_runs_skips_and_validates(cli_reader, deforest_aoi, deforest_hansen, tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "HANSEN_URL_TEMPLATE", deforest_hansen)
    aoi_path, _ = deforest_aoi
    out = tmp_path / "cli_out"
    args = ["deforest", "--aoi", str(aoi_path), "--year-a", "2019", "--year-b", "2022", "--out", str(out), "--validate"]
    args += ["--label-ring-km", "0.1", "--labels-per-class", "100", "--min-area-ha", "0.05"]
    runner = CliRunner()
    first = runner.invoke(app, args)
    assert first.exit_code == 0, (first.output, first.exception)
    for text in (
        "Deforest 2019 to 2022",
        "Forest: 2019",
        "Converted:",
        "Cross-check:",
        "Labels: Hansen",
        "Classifier:",
        "Validation (hansen, loss after 2019 through 2022)",
        "against loss in 2019 through 2021 (lag-aligned)",
        "against loss in 2022 only",
        "Transfer to 2022",
        "2022 forest map against Hansen's forest proxy",
        "flips the cross-check removed",
        "forest proxy",
        "Conversion patches by size",
        "GB of chunks",
    ):
        assert text in first.stdout, text
    report = json.loads((out / "deforest_report.json").read_text())
    assert report["params"]["hansen_url_template"] == deforest_hansen
    assert (out / "deforest_validation.json").is_file() and report["summary"]["validation"]["precision"] > 0.8
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "current" in second.stdout and "--force" in second.stdout
    assert json.loads((out / "deforest_report.json").read_text()) == report
    # The report stores the histogram with sorted keys; both runs show the bins smallest first.
    bins = re.compile(r"<0\.5|0\.5-1|1-5|5-20|20-100|>100")
    for run in (first, second):
        header = next(line for line in run.stdout.splitlines() if "<0.5" in line)
        assert bins.findall(header) == ["<0.5", "0.5-1", "1-5", "5-20", "20-100", ">100"], header


def test_cli_deforest_errors_exit_2_before_reading(cli_reader, caldor_store, deforest_aoi, tmp_path):
    _, log = caldor_store
    aoi_path, _ = deforest_aoi
    bad = tmp_path / "labels.csv"
    bad.write_text("lon,lat,label\n-120.3,38.69,3\n", encoding="utf-8")
    base = ["deforest", "--aoi", str(aoi_path), "--out", str(tmp_path / "o")]
    cases = [
        (["--year-a", "2019", "--year-b", "2022", "--labels", str(bad)], "labels.csv"),
        (["--year-a", "2022", "--year-b", "2019"], "must be after year_a"),
        (["--year-a", "2019", "--year-b", "2022", "--hansen-version", "v2"], "--hansen-version"),
        (["--year-a", "2019", "--year-b", "2022", "--forest-prob", "1.5"], "forest_prob"),
    ]
    for extra, message in cases:
        result = CliRunner().invoke(app, [*base, *extra])
        assert result.exit_code == EXIT_USAGE, (extra, result.output)
        assert message in result.stderr, (extra, result.stderr)
    assert log.chunk_keys() == [] and not (tmp_path / "o" / "deforest_report.json").exists()


def test_cli_deforest_data_errors_exit_2_before_reading(
    cli_reader, caldor_store, caldor_hansen, deforest_aoi, tmp_path, monkeypatch
):
    # DeforestError is a TesseraError: an unusable label ring or unreachable Hansen labels exit 2 with
    # the --labels hint, before any chunk read, never with a traceback.
    _, log = caldor_store
    aoi_path, _ = deforest_aoi
    args = ["deforest", "--aoi", str(aoi_path), "--year-a", "2019", "--year-b", "2022", "--out", str(tmp_path / "o")]
    args += ["--label-ring-km", "0.1", "--labels-per-class", "100"]
    water = write_hansen_tiles(
        tmp_path / "water", caldor_hansen, edit=lambda layer, d: np.full_like(d, 2) if layer == "datamask" else d
    )
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(validate, "HANSEN_URL_TEMPLATE", water)
    on_water = CliRunner().invoke(app, args)
    assert on_water.exit_code == EXIT_USAGE, (on_water.output, on_water.exception)
    assert "no land pixel" in on_water.stderr and "--labels" in on_water.stderr
    monkeypatch.setattr(validate, "hansen_layer", _unreachable_hansen)
    unreachable = CliRunner().invoke(app, args)
    assert unreachable.exit_code == EXIT_USAGE, (unreachable.output, unreachable.exception)
    assert "storage.googleapis.com" in unreachable.stderr and "--labels" in unreachable.stderr
    assert log.chunk_keys() == [] and not (tmp_path / "o" / "deforest_report.json").exists()


def test_cli_deforest_prints_brackets_in_names(cli_reader, caldor_reader, caldor_fixture, deforest_aoi, tmp_path):
    # File names and paths are printed as given: rich must not read "[draft]" as a style, nor crash
    # on "[/aoi]" as a closing tag, once every output is written.
    _, aoi = deforest_aoi
    odd = tmp_path / "run["
    odd.mkdir()
    aoi_path = _feature_collection(odd / "aoi].geojson", aoi, {})
    a, _, c, _, _, f = (float(v) for v in caldor_fixture["transform"])
    to_ll = Transformer.from_crs(32610, 4326, always_xy=True)
    rows = ["lon,lat,label"]
    for label, (r0, c0) in ((1, (26, 0)), (0, (0, 40))):
        for i in range(48):
            lon, lat = to_ll.transform(c + (c0 + i % 6 + 0.5) * a, f - (r0 + i // 6 + 0.5) * a)
            rows.append(f"{lon:.8f},{lat:.8f},{label}")
    csv = tmp_path / "pts [draft].csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "x[" / "y]"
    args = ["deforest", "--aoi", str(aoi_path), "--year-a", "2019", "--year-b", "2022", "--out", str(out)]
    args += ["--labels", str(csv), "--min-area-ha", "0.05"]
    env = {"COLUMNS": "400"}
    first = CliRunner().invoke(app, args, env=env)
    assert first.exit_code == 0, (first.output, first.exception)
    assert "Labels: pts [draft].csv (96 rows" in first.stdout and str(aoi_path) in first.stdout
    again = CliRunner().invoke(app, args, env=env)
    assert again.exit_code == 0, (again.output, again.exception)
    assert f"Outputs in {out} are current" in again.stdout


def test_cli_deforest_strict_validate_stops_before_reading(
    cli_reader, caldor_store, deforest_aoi, deforest_hansen, tmp_path, monkeypatch
):
    monkeypatch.setattr(validate, "HANSEN_URL_TEMPLATE", deforest_hansen)
    monkeypatch.setattr(validate, "hansen_loss", _unreachable_hansen)
    _, log = caldor_store
    aoi_path, _ = deforest_aoi
    out = tmp_path / "out"
    args = ["deforest", "--aoi", str(aoi_path), "--year-a", "2019", "--year-b", "2022", "--out", str(out)]
    args += ["--label-ring-km", "0.1", "--labels-per-class", "100", "--validate"]
    strict = CliRunner().invoke(app, [*args, "--strict-validate"])
    assert strict.exit_code == EXIT_VALIDATION, strict.output
    assert "--strict-validate" in strict.stderr and "storage.googleapis.com" in strict.stderr
    assert log.chunk_keys() == [] and not (out / "deforest_report.json").exists()
    relaxed = CliRunner().invoke(app, args)
    assert relaxed.exit_code == 0, relaxed.output
    assert "Validation: not done" in relaxed.stdout and not (out / "deforest_validation.json").exists()


def test_cli_deforest_forwards_every_option(monkeypatch, tmp_path):
    seen = {}

    def record(aoi, year_a, year_b, out, **kwargs):
        seen.update(kwargs, aoi=aoi, year_a=year_a, year_b=year_b, out=out)
        return DeforestResult(
            report={"summary": {}},
            paths=export.output_paths(Path(out), "deforest", year_a, year_b),
            extra_paths={},
            skipped=False,
            validation_written=True,
        )

    monkeypatch.setattr("tessera_apps.apps.run_deforest", record)
    ref = tmp_path / "r.geojson"
    ref.write_text("{}")
    labels = tmp_path / "l.csv"
    labels.write_text("lon,lat,label\n")
    args = ["deforest", "--aoi", "rondonia", "--year-a", "2019", "--year-b", "2022", "--out", str(tmp_path)]
    args += ["--reference", str(ref), "--labels", str(labels), "--labels-per-class", "300", "--label-chunks", "12"]
    args += ["--label-ring-km", "7.5", "--forest-cover", "60", "--nonforest-cover", "15", "--forest-prob", "0.6"]
    args += ["--stable-percentile", "90", "--min-area-ha", "1.5", "--hansen-version", "GFC-2024-v1.12", "--seed", "4"]
    args += ["--force", "--snapshot", "SNAP", "--i-know-this-is-large", "--strict-validate"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen["aoi"] == "rondonia" and (seen["year_a"], seen["year_b"], seen["out"]) == (2019, 2022, tmp_path)
    assert seen["validate"] is True and seen["reference"] == ref and seen["labels"] == labels
    assert (seen["labels_per_class"], seen["label_chunks"], seen["label_ring_km"]) == (300, 12, 7.5)
    assert (seen["forest_cover"], seen["nonforest_cover"], seen["forest_prob"]) == (60, 15, 0.6)
    assert (seen["stable_percentile"], seen["min_area_ha"], seen["hansen_version"]) == (90.0, 1.5, "GFC-2024-v1.12")
    assert seen["seed"] == 4 and seen["force"] is True and seen["snapshot_id"] == "SNAP"
    assert seen["max_area_km2"] is None and seen["strict_validate"] is True
    seen.clear()
    defaults = CliRunner().invoke(app, ["deforest", "--aoi", "rondonia", "--year-a", "2019", "--year-b", "2022"])
    assert defaults.exit_code == 0, defaults.output
    assert seen["validate"] is False and seen["reference"] is None and seen["labels"] is None
    assert (seen["labels_per_class"], seen["label_chunks"], seen["label_ring_km"]) == (500, 24, 5.0)
    assert (seen["forest_cover"], seen["nonforest_cover"], seen["forest_prob"]) == (50, 10, 0.5)
    assert (seen["stable_percentile"], seen["min_area_ha"], seen["seed"]) == (95.0, 0.5, 0)
    assert seen["hansen_version"] == validate.HANSEN_VERSION and seen["max_area_km2"] == 10_000
    assert seen["snapshot_id"] is None and seen["force"] is False and seen["strict_validate"] is False
    assert seen["out"] == Path("out")


RONDONIA_WINDOW = box(-63.115, -8.675, -63.105, -8.665)  # about 1.1 x 1.1 km inside the Rondonia demo AOI


@pytest.mark.network
def test_deforest_live_window(tmp_path, monkeypatch):
    # Default label settings: with a cheap ring (1 km, 6 chunks, 200 labels per class) the 2019
    # classifier transferred badly to 2022 on this window (2022 forest map accuracy 0.61 against
    # Hansen) while every 2019 score looked good, so the year_b checks are asserted here too.
    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))
    aoi = _feature_collection(tmp_path / "live.geojson", RONDONIA_WINDOW, {})
    result = run_deforest(aoi, 2019, 2022, tmp_path / "out", validate=True, snapshot_id=LIVE_SNAPSHOT)
    s = result.summary
    assert s["zones"] == ["20S"] and s["label_source"] == "hansen"
    assert s["labels"]["chunks_used"] <= 24 and s["labels"]["used_forest"] >= 20 and s["labels"]["used_nonforest"] >= 20
    assert s["classifier"]["heldout"]["folds"] >= 2
    assert result.report["provenance"]["store"]["snapshot_id"] == LIVE_SNAPSHOT
    assert 0 < result.report["bytes_read"] <= result.report["estimated_bytes"]
    assert s["validation"]["source"] == "hansen" and s["validation"]["pixels_scored"] > 0
    # The year_b forest baseline is plausible: the classifier transfers to 2022.
    assert s["validation"]["forest_agreement_year_b"]["accuracy"] > 0.85
    assert s["validation"]["forest_agreement_year_a"]["accuracy"] > 0.85
    transfer = s["transfer"]
    assert transfer["available"] and transfer["drop"] is not None and transfer["drop"] <= dfr.TRANSFER_WARN_DROP
    assert not any("transfers poorly" in w for w in result.report["warnings"])
    lag = s["validation"]["lag_aligned"]
    assert lag["window"]["lossyear_codes"] == [19, 20, 21] and lag["reference_pixels_scored"] > 0


# --------------------------------------------------------------------------------------------
# Wetland: end to end on the caldor fixture, validation from a local WorldCover tile
# --------------------------------------------------------------------------------------------

WETLAND_FILES = (
    "wetland_2020_2022.tif",
    "wetland_summary.parquet",
    "wetland_summary.csv",
    "wetland_seeds.parquet",
    "wetland_preview.png",
    "wetland_validation.json",
    "wetland_report.json",
)
WETLAND_BANDS = ("score_2020", "score_2022", "extent_2020", "extent_2022", "change")
#: 6 x 6 pixel seed squares (row, col) of the fixture window, in the stands the 2021 fire left alone.
WETLAND_SEEDS = ((4, 4), (4, 52), (14, 40), (22, 6))
WETLAND_KW = {"min_area_ha": 0.05, "block": (32, 32), "seed_buffer_m": 30.0, "coast_km": 0.1}


def fixture_codes(n: int = 64) -> np.ndarray:
    """A WorldCover map on the fixture grid: mangrove north, tree cover south, a wetland strip, water, no data."""
    rows, cols = np.mgrid[0:n, 0:n]
    codes = np.where(rows < 30, 95, 10).astype(np.uint8)
    codes[(cols >= 28) & (cols < 34)] = 90
    codes[(rows >= 56) & (cols < 8)] = 80
    codes[rows >= 60] = 0
    return codes


@pytest.fixture
def wetland_inputs(caldor_fixture, tmp_path, monkeypatch):
    """AOI, seeds, a shoreline along fixture column 10 and WorldCover 2020 tiles over the fixture window."""
    from .conftest import write_worldcover_tiles

    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))
    fx = caldor_fixture
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    aoi = fixture_window_lonlat(fx, (2, 62), (2, 62))
    aoi_path = _feature_collection(tmp_path / "wetland_window.geojson", aoi, {"name": "wetland window"})
    seeds = [fixture_window_lonlat(fx, (r, r + 6), (k, k + 6)) for r, k in WETLAND_SEEDS]
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"source": "fixture stands"}, "geometry": mapping(s)} for s in seeds
        ],
    }
    seeds_path = tmp_path / "seeds.geojson"
    seeds_path.write_text(json.dumps(fc), encoding="utf-8")
    to_ll = Transformer.from_crs(int(fx["epsg"]), 4326, always_xy=True)
    x = c + 10.5 * a
    line = shapely.LineString([to_ll.transform(x, f + 200.0), to_ll.transform(x, f - 64 * a - 200.0)])
    shore = _feature_collection(tmp_path / "shore.geojson", line, {})
    template = write_worldcover_tiles(tmp_path / "worldcover", int(fx["epsg"]), c, f, {2020: fixture_codes()})
    return aoi_path, aoi, seeds_path, shore, template


def test_wetland_end_to_end_on_the_caldor_fixture(
    caldor_reader, caldor_store, caldor_fixture, wetland_inputs, tmp_path
):
    from scipy.spatial.distance import cdist

    from tessera_apps.apps import WetlandResult
    from tessera_apps.apps import wetland as wl

    aoi_path, aoi, seeds_path, shore, template = wetland_inputs
    out = tmp_path / "out"
    kwargs = {
        **WETLAND_KW,
        "seeds": seeds_path,
        "shoreline": shore,
        "validate": True,
        "worldcover_url_template": template,
        "reader": caldor_reader,
    }
    result = wl.run_wetland(aoi_path, 2020, 2022, out, **kwargs)
    assert isinstance(result, WetlandResult) and not result.skipped and result.validation_written
    paths = result.paths
    for p in (*paths.required(need_validation=True), *result.extra_paths.values()):
        assert p.is_file(), p
    assert sorted(p.name for p in out.iterdir()) == sorted(WETLAND_FILES)

    # A valid float32 COG with the five named bands, on the AOI's window.
    ok, errors, _ = cog_validate(str(paths.tif))
    assert ok, errors
    plan = plan_aoi(caldor_reader, aoi)
    (zp,) = plan.zones
    with rasterio.open(paths.tif) as ds:
        assert ds.descriptions == WETLAND_BANDS
        assert ds.dtypes == ("float32",) * 5 and np.isnan(ds.nodata)
        assert ds.transform == zp.transform and ds.shape == zp.shape
        score_a, score_b, ext_a, ext_b, change = ds.read()
    inside = zp.mask
    s = result.summary
    assert s["aoi_pixels"] == plan.n_pixels == 3600 == s["classified_pixels"]
    assert s["zones"] == ["10N"] and (s["year_a"], s["year_b"]) == (2020, 2022)
    for band in (score_a, score_b, change):
        assert np.isfinite(band[inside]).all() and np.isnan(band[~inside]).all()

    # The seeds: 144 pixels in 4 polygons, each polygon in its own fold.
    seeds = pd.read_parquet(result.extra_paths["seeds"])
    assert list(seeds.columns) == list(wl.SEED_COLUMNS)
    assert len(seeds) == 144 and seeds["used"].all() and (seeds["drop_reason"] == "").all()
    assert seeds["inside_aoi"].all() and (seeds["regime"] == "radar").all()
    assert seeds.groupby("group")["fold"].nunique().max() == 1 and seeds["fold"].nunique() == 4
    # The reference is the mean of the dequantized seed vectors; the score map is 1 - cosine to it.
    quiet = TesseraReader.from_store(caldor_store[0])
    pts = read_points(quiet, seeds["lon"].to_numpy(), seeds["lat"].to_numpy(), 2020)
    assert pts.valid.all()
    mean = pts.embeddings.astype(np.float64).mean(axis=0)
    for year, band in ((2020, score_a), (2022, score_b)):
        expected = 1.0 - metrics.cosine_to_vector(read_plan(quiet, zp, year), mean)
        np.testing.assert_allclose(band[inside], expected[inside], atol=1e-5)
    x = pts.embeddings.astype(np.float64)
    np.testing.assert_allclose(
        seeds["score_in_sample"], 1.0 - (x @ mean) / (np.linalg.norm(x, axis=1) * np.linalg.norm(mean)), atol=1e-6
    )
    # The threshold: the 95th percentile of the pooled held-out seed scores, not the in-sample one.
    th = s["threshold"]
    assert th["method"] == "heldout" and th["folds"] == 4 and th["heldout_pixels"] == 144
    thr = th["threshold"]
    assert thr == pytest.approx(float(np.percentile(seeds["score_heldout"], 95)))
    assert th["in_sample_percentile"] == pytest.approx(float(np.percentile(seeds["score_in_sample"], 95)))
    assert thr > th["in_sample_percentile"]
    sb = seeds["score_year_b"].to_numpy()
    pb = read_points(quiet, seeds["lon"].to_numpy(), seeds["lat"].to_numpy(), 2022)
    xb = pb.embeddings.astype(np.float64)
    np.testing.assert_allclose(sb, 1.0 - (xb @ mean) / (np.linalg.norm(xb, axis=1) * np.linalg.norm(mean)), atol=1e-6)
    assert s["seeds"]["share_year_b_within_threshold"] == pytest.approx(float(np.mean(sb <= thr)))

    # Extent: the cleaned score <= threshold; change +1 gain, -1 loss, 0 stable.
    for score, ext in ((score_a, ext_a), (score_b, ext_b)):
        expected = metrics.clean_mask(inside & (score <= thr), min_pixels=5) & inside
        np.testing.assert_array_equal(ext == 1, expected)
    gain, loss = (ext_b == 1) & (ext_a == 0), (ext_a == 1) & (ext_b == 0)
    np.testing.assert_array_equal(change == 1, gain)
    np.testing.assert_array_equal(change == -1, loss)
    assert loss.sum() > 100  # the fire burned stands that were seed-like in 2020
    n_a, n_b = int((ext_a == 1).sum()), int((ext_b == 1).sum())
    assert s["extent"]["2020"]["extent_pixels"] == n_a and s["extent"]["2022"]["extent_pixels"] == n_b
    assert s["extent_ha_year_a"] == pytest.approx(n_a * 0.01) and s["gain_ha"] == pytest.approx(gain.sum() * 0.01)
    assert s["loss_ha"] == pytest.approx(loss.sum() * 0.01) and s["net_change_ha"] == pytest.approx((n_b - n_a) * 0.01)

    # Coastal fraction: extent within 100 m of the shoreline along fixture column 10.
    first_col = round((zp.transform.c - float(caldor_fixture["transform"][2])) / 10.0)
    near = np.broadcast_to(np.abs(np.arange(zp.shape[1]) + first_col - 10) * 10.0 <= 100.0, zp.shape)
    coast = s["coastal"]
    assert coast["source"] == "file" and coast["file"] == "shore.geojson" and coast["coast_km"] == 0.1
    assert s["coastal_fraction_year_a"] == pytest.approx(((ext_a == 1) & near).sum() / n_a)
    assert s["coastal_fraction_year_b"] == pytest.approx(((ext_b == 1) & near).sum() / n_b)

    # One summary row per year, the change columns repeated, the same in parquet and csv.
    table = pd.read_parquet(paths.summary_parquet)
    pd.testing.assert_frame_equal(table, pd.read_csv(paths.summary_csv), check_dtype=False)
    assert table["year"].tolist() == [2020, 2022]
    assert table["extent_ha"].tolist() == pytest.approx([n_a * 0.01, n_b * 0.01])
    assert table["coastal_fraction"].tolist() == pytest.approx(
        [s["coastal_fraction_year_a"], s["coastal_fraction_year_b"]]
    )
    for col in ("net_change_ha", "net_change_pct", "gain_ha", "loss_ha", "threshold", "score"):
        assert table[col].nunique() == 1, col
    assert table["gain_ha"].iloc[0] == pytest.approx(s["gain_ha"]) and table["score"].iloc[0] == "cosine"

    # Validation: WorldCover 2020 only (2022 has no map), seeds and 30 m around them left out.
    val = json.loads(paths.validation.read_text())
    assert set(val["scores"]) == {
        "extent_2020",
        "extent_2020_per_zone",
        "extent_2020_with_seeds",
        "false_positive_classes_2020",
    }
    codes = validate.worldcover_map(zp, 2020, url_template=template)
    ref, cover = (codes == 95) & inside, (codes != 0) & inside
    sr = seeds["row"].to_numpy() - zp.rows.start
    sc = seeds["col"].to_numpy() - zp.cols.start
    rr, cc = np.mgrid[0 : zp.shape[0], 0 : zp.shape[1]]
    dist = cdist(np.stack([rr.ravel(), cc.ravel()], 1), np.stack([sr, sc], 1)).min(axis=1).reshape(zp.shape) * 10.0
    keep = cover & (dist > 30.0)
    pred = ext_a == 1
    tp, fp, fn = int((pred & ref & keep).sum()), int((pred & ~ref & keep).sum()), int((~pred & ref & keep).sum())
    score = val["scores"]["extent_2020"]
    assert (score["tp"], score["fp"], score["fn"], score["n"]) == (tp, fp, fn, int(keep.sum()))
    v = s["validation"]
    assert v["source"] == "worldcover" and v["year"] == 2020
    assert v["years_scored"] == [2020] and v["years_not_covered"] == [2022]
    assert v["recall"] == pytest.approx(tp / (tp + fn)) and v["precision"] == pytest.approx(tp / (tp + fp))
    row = v["per_year"]["2020"]
    assert row["seed_excluded_pixels"] == int((cover & (dist <= 30.0)).sum()) > 144
    fp_codes, fp_counts = np.unique(codes[pred & ~ref & keep], return_counts=True)
    assert {k: w["pixels"] for k, w in row["false_positive_classes"].items()} == {
        str(int(k)): int(n) for k, n in zip(fp_codes, fp_counts, strict=True)
    }
    ref_rec = val["reference"]
    assert ref_rec["source"] == "worldcover" and ref_rec["versions"] == {"2020": "v100"}
    assert ref_rec["url_template"] == template and ref_rec["years_not_covered"] == [2022]
    caveats = " ".join(val["caveats"])
    for text in ("leaving out the seed pixels", "itself a model", "no map for 2022", "30 m"):
        assert text in caveats, text

    # The report: provenance, the seed chunks in the byte estimate, extras and timings.
    report = json.loads(paths.report.read_text())
    assert report == result.report and report["app"] == "wetland"
    _, log = caldor_store
    assert report["provenance"]["years"] == [2020, 2022] == years_in(log)
    assert report["provenance"]["years_checked"] == [2020, 2022]
    assert set(report["extra_outputs"]) == {"seeds"}
    assert 0 < report["bytes_read"] <= report["estimated_bytes"]
    chunk_bytes = 32 * 32 * BYTES_PER_PIXEL
    n_chunks = len(wl.seed_chunks(caldor_reader, wl.place_seeds(caldor_reader, wl.load_seeds(seeds_path))))
    assert (
        report["estimated_bytes"] == plan.estimated_bytes(caldor_reader, 3, block=(32, 32)) + 2 * n_chunks * chunk_bytes
    )
    assert report["estimated_memory"] == wl.memory_estimate(plan, block=(32, 32), coast_km=0.1)
    assert set(report["timings"]) >= {"regime_probe", "seeds_read", "reference", "stream", "clean", "coastal", "write"}
    assert report["params"]["worldcover_url_template"] == template and report["params"]["score"] == "cosine"
    assert s["radar"]["seed_regime"] == "radar" and s["radar"]["regime_mismatch_pixels"] == 0
    assert s["storm_note"] is None and not report["warnings"]

    # Rerunning with the same arguments skips without reading a chunk; the same run elsewhere is identical.
    chunks_before = len(log.chunk_keys())
    again = wl.run_wetland(aoi_path, 2020, 2022, out, **kwargs)
    assert again.skipped and again.report == report and again.validation_written
    assert len(log.chunk_keys()) == chunks_before
    twin = wl.run_wetland(aoi_path, 2020, 2022, tmp_path / "twin", **kwargs)
    assert twin.report["outputs"] == report["outputs"] and twin.report["extra_outputs"] == report["extra_outputs"]
    # Changing an option reruns.
    other = wl.run_wetland(aoi_path, 2020, 2022, out, **{**kwargs, "threshold_pct": 90.0})
    assert not other.skipped and other.summary["threshold"]["threshold"] < thr


def test_cli_wetland_runs_skips_and_validates(cli_reader, wetland_inputs, tmp_path, monkeypatch):
    aoi_path, _, seeds_path, shore, template = wetland_inputs
    monkeypatch.setattr(validate, "WORLDCOVER_URL_TEMPLATE", template)
    out = tmp_path / "cli_out"
    args = ["wetland", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    args += ["--seeds", str(seeds_path), "--shoreline", str(shore), "--coast-km", "0.1", "--validate"]
    args += ["--min-area-ha", "0.05", "--seed-buffer-m", "30"]
    runner = CliRunner()
    first = runner.invoke(app, args, env={"COLUMNS": "200"})
    assert first.exit_code == 0, (first.output, first.exception)
    for text in (
        "Wetland 2020 to 2022",
        "Seeds:",
        "144 of 144 pixels used",
        "spatial folds",
        "held-out seed scores",
        "Extent",
        "Change: net",
        "Coastal fraction within 0.1 km",
        "Radar: seeds' regime radar",
        "Validation (worldcover 2020): recall",
        "false positives by WorldCover class",
        "not scored (no reference map): 2022",
        "GB of chunks",
    ):
        assert text in first.stdout, text
    report = json.loads((out / "wetland_report.json").read_text())
    assert report["params"]["worldcover_url_template"] == template
    assert (out / "wetland_validation.json").is_file() and (out / "wetland_seeds.parquet").is_file()
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "current" in second.stdout and "--force" in second.stdout
    assert json.loads((out / "wetland_report.json").read_text()) == report


def test_cli_wetland_errors_exit_2_before_reading(cli_reader, caldor_store, wetland_inputs, tmp_path):
    _, log = caldor_store
    aoi_path, _, seeds_path, shore, _ = wetland_inputs
    lines = _feature_collection(tmp_path / "lines.geojson", shapely.LineString([(-120.3, 38.6), (-120.2, 38.7)]), {})
    base = ["wetland", "--aoi", str(aoi_path), "--out", str(tmp_path / "o"), "--shoreline", str(shore)]
    ok = ["--year-a", "2020", "--year-b", "2022", "--seeds", str(seeds_path)]
    cases = [
        (["--year-a", "2022", "--year-b", "2020", "--seeds", str(seeds_path)], "must be after year_a"),
        ([*ok, "--reference-classes", "95,x"], "--reference-classes"),
        ([*ok, "--reference-classes", "97"], "no class"),
        ([*ok, "--score", "mahalanobis"], "at least 500"),
        ([*ok, "--score", "knn"], "must be one of"),
        (["--year-a", "2020", "--year-b", "2022", "--seeds", str(lines)], "no point or polygon"),
        (["--year-a", "2020", "--year-b", "2022"], "No --seeds"),
        (["--year-a", "2022", "--year-b", "2024", "--seeds", str(seeds_path), "--validate"], "2020 (v100)"),
        ([*ok, "--reference-source", "gmw", "--gmw-version", "v3.1"], "gmw_version"),
    ]
    for extra, message in cases:
        result = CliRunner().invoke(app, [*base, *extra], env={"COLUMNS": "300"})
        assert result.exit_code == EXIT_USAGE, (extra, result.output)
        assert message in result.stderr, (extra, result.stderr)
    assert log.chunk_keys() == [] and not (tmp_path / "o" / "wetland_report.json").exists()


def test_cli_wetland_strict_validate_stops_before_reading(
    cli_reader, caldor_store, wetland_inputs, tmp_path, monkeypatch
):
    def unreachable(*args, **kwargs):
        raise validate.ReferenceUnavailableError("esa-worldcover.s3.eu-central-1.amazonaws.com refused (403)")

    monkeypatch.setattr(validate, "worldcover_map", unreachable)
    _, log = caldor_store
    aoi_path, _, seeds_path, shore, _ = wetland_inputs
    out = tmp_path / "out"
    args = ["wetland", "--aoi", str(aoi_path), "--year-a", "2020", "--year-b", "2022", "--out", str(out)]
    args += ["--seeds", str(seeds_path), "--shoreline", str(shore), "--validate", "--min-area-ha", "0.05"]
    strict = CliRunner().invoke(app, [*args, "--strict-validate"])
    assert strict.exit_code == EXIT_VALIDATION, strict.output
    assert "--strict-validate" in strict.stderr and "esa-worldcover" in strict.stderr
    assert log.chunk_keys() == [] and not (out / "wetland_report.json").exists()
    relaxed = CliRunner().invoke(app, args)
    assert relaxed.exit_code == 0, relaxed.output
    assert "Validation: not done" in relaxed.stdout and not (out / "wetland_validation.json").exists()


def test_cli_wetland_forwards_every_option(monkeypatch, tmp_path):
    from tessera_apps.apps import WetlandResult

    seen = {}

    def record(aoi, year_a, year_b, out, **kwargs):
        seen.update(kwargs, aoi=aoi, year_a=year_a, year_b=year_b, out=out)
        return WetlandResult(
            report={"summary": {}},
            paths=export.output_paths(Path(out), "wetland", year_a, year_b),
            extra_paths={},
            skipped=False,
            validation_written=True,
        )

    monkeypatch.setattr("tessera_apps.apps.run_wetland", record)
    ref = tmp_path / "r.geojson"
    ref.write_text("{}")
    shore = tmp_path / "s.geojson"
    shore.write_text("{}")
    args = ["wetland", "--aoi", "everglades", "--year-a", "2020", "--year-b", "2024", "--out", str(tmp_path)]
    args += ["--seeds", "my_seeds.gpkg", "--reference", str(ref), "--reference-source", "gmw"]
    args += ["--reference-classes", "95, 90", "--gmw-version", "v4.1.12", "--score", "mahalanobis"]
    args += ["--threshold-pct", "90", "--folds", "3", "--min-area-ha", "0.5", "--seed-buffer-m", "150"]
    args += ["--shoreline", str(shore), "--coast-km", "2.5", "--seed", "7", "--force", "--snapshot", "SNAP"]
    args += ["--i-know-this-is-large", "--strict-validate"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert seen["aoi"] == "everglades" and (seen["year_a"], seen["year_b"], seen["out"]) == (2020, 2024, tmp_path)
    assert seen["seeds"] == "my_seeds.gpkg" and seen["validate"] is True and seen["reference"] == ref
    assert seen["reference_source"] == "gmw" and seen["reference_classes"] == [95, 90]
    assert seen["gmw_version"] == "v4.1.12" and seen["score"] == "mahalanobis"
    assert (seen["threshold_pct"], seen["folds"], seen["min_area_ha"], seen["seed_buffer_m"]) == (90.0, 3, 0.5, 150.0)
    assert seen["shoreline"] == shore and seen["coast_km"] == 2.5 and seen["seed"] == 7
    assert seen["force"] is True and seen["snapshot_id"] == "SNAP" and seen["max_area_km2"] is None
    assert seen["strict_validate"] is True
    seen.clear()
    defaults = CliRunner().invoke(app, ["wetland", "--aoi", "everglades", "--year-a", "2020", "--year-b", "2024"])
    assert defaults.exit_code == 0, defaults.output
    # --aoi everglades with no --seeds: run_wetland takes the bundled everglades seeds (seeds None).
    assert seen["seeds"] is None and seen["validate"] is False and seen["reference"] is None
    assert seen["reference_source"] == "worldcover" and seen["reference_classes"] == [95]
    assert seen["gmw_version"] == "v3" and seen["score"] == "cosine"
    assert (seen["threshold_pct"], seen["folds"], seen["min_area_ha"], seen["seed_buffer_m"]) == (95.0, 5, 0.1, 200.0)
    assert seen["shoreline"] is None and seen["coast_km"] == 1.0 and seen["seed"] == 0
    assert seen["max_area_km2"] == 10_000 and seen["snapshot_id"] is None and seen["force"] is False
    assert seen["strict_validate"] is False and seen["out"] == Path("out")


#: About 1 x 1 km of mangrove coast inside the Everglades demo AOI, around bundled seed 8 (about 56
#: percent WorldCover mangrove in 2020 and 2021).
EVERGLADES_WINDOW = box(-81.3545, 25.8341, -81.3445, 25.8431)


@pytest.mark.network
def test_wetland_live_window(tmp_path, monkeypatch):
    from tessera_apps.apps import wetland as wl

    for key in [k for k in os.environ if k.startswith("AWS_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))
    aoi = _feature_collection(tmp_path / "live.geojson", EVERGLADES_WINDOW, {})
    result = wl.run_wetland(
        aoi, 2020, 2021, tmp_path / "out", seeds="everglades_seeds", validate=True, snapshot_id=LIVE_SNAPSHOT
    )
    s = result.summary
    assert s["zones"] == ["17N"] and s["seeds"]["source"] == "demo:everglades_seeds"
    assert s["seeds"]["pixels"] == 1200 and s["seeds"]["used"] >= 1100 and s["seeds"]["groups"] == 12
    assert s["threshold"]["method"] == "heldout" and s["threshold"]["folds"] == 5
    assert s["threshold"]["threshold"] > s["threshold"]["in_sample_percentile"]
    assert result.report["provenance"]["store"]["snapshot_id"] == LIVE_SNAPSHOT
    assert 0 < result.report["bytes_read"] <= result.report["estimated_bytes"]
    val = s["validation"]
    assert val["source"] == "worldcover" and val["years_scored"] == [2020, 2021]
    assert val["per_year"]["2020"]["pixels_scored"] > 0 and val["recall"] is not None
    assert "same product" in " ".join(json.loads(result.paths.validation.read_text())["caveats"])
    assert s["coastal"]["source"] == "natural_earth_10m" and s["coastal"]["approximate"] is True
    assert s["coastal_fraction_year_a"] is not None and s["storm_note"] and "Irma" in s["storm_note"]
