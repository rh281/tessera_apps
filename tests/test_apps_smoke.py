"""Each app end to end on the bundled real fixture (brief section 3), offline, through its function and its CLI.

The wildfire app runs on ``tests/fixtures/caldor_64.npz``: a real 64 x 64 window of the Caldor scar,
about half burned, with all nine years 2017 to 2025, served by the in-memory store of
``tests/conftest.py``. The AOI is the window minus a two-pixel margin, and the validation reference
is the caldor demo's Hansen 2021 loss outline clipped to it.
"""

from __future__ import annotations

import json
import os
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
from tessera_apps.apps import StrictValidationError, WildfireResult
from tessera_apps.apps import wildfire as wf
from tessera_apps.apps._common import AppInputError
from tessera_apps.apps.wildfire import memory_estimate, run_wildfire
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app
from tessera_apps.demo import demo_feature
from tessera_apps.io import FIRST_YEAR, iter_plan_blocks, plan_aoi, read_plan

from .conftest import BOUNDARY

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
