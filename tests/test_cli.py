"""The CLI wiring works offline."""

from __future__ import annotations

import json
import urllib.error

import pytest
from typer.testing import CliRunner

from tessera_apps import validate
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app

from .conftest import SITE_BOX, TARGET_BOX, KeyLog, grid_polygon, restore_corner, restore_store, write_polygon


def test_help_lists_info():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "info" in result.output


# --------------------------------------------------------------------------------------------
# restore, on a designed store served in place of the live one
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _restore_mem():
    return restore_store()[2]


@pytest.fixture
def restore_cli(_restore_mem, tmp_path, monkeypatch):
    """``TesseraReader.open`` serves the designed restore store; returns the base arguments and the key log."""
    from tessera_apps.io import TesseraReader

    log = KeyLog(_restore_mem)
    reader = TesseraReader.from_store(log)
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(lambda **kwargs: reader))
    monkeypatch.setenv("TESSERA_APPS_CACHE", str(tmp_path / "cache"))

    def refused(request, timeout):
        raise urllib.error.URLError("blocked by the proxy")

    monkeypatch.setattr(validate, "_urlopen", refused)
    left, top = restore_corner()
    site = write_polygon(tmp_path / "site.geojson", grid_polygon(left, top, SITE_BOX))
    target = write_polygon(tmp_path / "target.geojson", grid_polygon(left, top, TARGET_BOX))
    base = ["restore", "--site", str(site), "--out", str(tmp_path / "out"), "--ring-inner-km", "0.1"]
    base += ["--ring-outer-km", "0.5", "--target", str(target)]
    return base, log


def test_restore_help_lists_its_options():
    result = CliRunner().invoke(app, ["restore", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    for option in ("--aoi", "--site", "--planted", "--control", "--target", "--no-canopy", "--placebos"):
        assert option in result.output, option


def test_cli_restore_runs_skips_and_prints_the_verdict(restore_cli, tmp_path):
    base, log = restore_cli
    runner = CliRunner()
    first = runner.invoke(app, [*base, "--planted", "2020"], env={"COLUMNS": "200"})
    assert first.exit_code == 0, (first.output, first.exception)
    lines = first.stdout.splitlines()
    assert lines[0] == "Verdict: diverging from control since 2020"
    for text in (
        "Trajectory",
        "Control: ring 0.1 to 0.5 km",
        "held-out 5th percentile, 4 folds of 4 cells",
        "Placebos: 200 of 400 pixels",
        "Null test (control halves): no divergence detected",
        "progress is unclipped",
        "Limitation: There is no global reference layer",
        "GB of chunks",
    ):
        assert text in first.stdout, text
    report = json.loads((tmp_path / "out" / "restore_report.json").read_text())
    assert report["summary"]["verdict"] == "diverging from control since 2020"
    reads = len(log.log)
    second = runner.invoke(app, [*base, "--planted", "2020"])
    assert second.exit_code == 0 and "current" in second.stdout and len(log.log) == reads


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--planted", "2017"], "no pre-planting year"),
        ([], "--planted YYYY is required"),
        (["--planted", "2020", "--year-a", "2020"], "must be before the planting year"),
        (["--planted", "2020", "--placebos", "5"], "--placebos"),
    ],
)
def test_cli_restore_usage_errors_exit_2_before_reading(restore_cli, extra, message):
    base, log = restore_cli
    result = CliRunner().invoke(app, [*base, *extra])
    assert result.exit_code == EXIT_USAGE, result.output
    assert message in result.output and log.chunk_keys() == []


def test_cli_restore_data_error_exits_2(restore_cli, tmp_path):
    base, log = restore_cli
    left, top = restore_corner()
    small = write_polygon(tmp_path / "small.geojson", grid_polygon(left, top, ((20, 28), (20, 28))))
    result = CliRunner().invoke(app, [*base, "--planted", "2020", "--control", str(small)])
    assert result.exit_code == EXIT_USAGE and "Only 64 control pixels are kept" in result.output


def test_cli_restore_canopy_host_unreachable_warns_or_exits_3(restore_cli, tmp_path):
    base, log = restore_cli
    runner = CliRunner()
    strict = runner.invoke(app, [*base, "--planted", "2020", "--validate", "--strict-validate"])
    assert strict.exit_code == EXIT_VALIDATION, strict.output
    assert "storage.googleapis.com" in strict.output and log.chunk_keys() == []
    loose = runner.invoke(app, [*base, "--planted", "2020", "--validate"], env={"COLUMNS": "200"})
    assert loose.exit_code == 0, loose.output
    assert "Canopy check: not done" in loose.stdout and "Warning:" in loose.stdout
    assert (tmp_path / "out" / "restore_validation.json").is_file()
    # The canopy still cannot be reached: the outputs stay, and --strict-validate says so with exit 3.
    again = runner.invoke(app, [*base, "--planted", "2020", "--validate", "--strict-validate"])
    assert again.exit_code == EXIT_VALIDATION and "current" in again.stdout
    no_canopy = runner.invoke(app, [*base, "--planted", "2020", "--validate", "--no-canopy", "--strict-validate"])
    assert no_canopy.exit_code == 0, no_canopy.output
