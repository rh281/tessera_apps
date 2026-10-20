"""The CLI wiring works offline."""

from __future__ import annotations

import json
import re
import urllib.error

import pytest
from typer.testing import CliRunner

from tessera_apps import validate
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app

from .conftest import SITE_BOX, TARGET_BOX, KeyLog, grid_polygon, restore_corner, restore_store, write_polygon

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


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
    # Rich colours the help where it detects CI (GITHUB_ACTIONS), splitting option names with escape codes.
    text = ANSI_ESCAPE.sub("", result.output)
    for option in (
        "--aoi",
        "--site",
        "--planted",
        "--match-year",
        "--control",
        "--target",
        "--no-canopy",
        "--placebos",
    ):
        assert option in text, option


def test_cli_restore_runs_skips_and_prints_the_verdict(restore_cli, tmp_path):
    base, log = restore_cli
    runner = CliRunner()
    first = runner.invoke(app, [*base, "--planted", "2020"], env={"COLUMNS": "200"})
    assert first.exit_code == 0, (first.output, first.exception)
    lines = ANSI_ESCAPE.sub("", first.stdout).splitlines()
    assert lines[0] == "Verdict: diverging from control since 2020"
    for text in (
        "Trajectory",
        "Control: ring 0.1 to 0.5 km",
        "held-out 5th percentile, 4 folds of 4 cells",
        "pixels kept at a 2018 similarity",
        "Match year 2018 (the control filter's only year): pixels with data, site 400 of 400; ring ",
        "; target 676 of 676",
        "departed chg",
        "Site-sized null (placebos):",
        "of 200 called diverging",
        "at most about 5% by construction",
        "Null test (control halves,",
        "so easy to pass): ratio to the site",
        "no divergence detected, max change",
        "progress is unclipped",
        "Limitation: There is no global reference layer",
        "GB of chunks",
    ):
        assert text in ANSI_ESCAPE.sub("", first.stdout), text
    report = json.loads((tmp_path / "out" / "restore_report.json").read_text())
    assert report["summary"]["verdict"] == "diverging from control since 2020"
    reads = len(log.log)
    second = runner.invoke(app, [*base, "--planted", "2020"])
    assert second.exit_code == 0 and "current" in ANSI_ESCAPE.sub("", second.stdout) and len(log.log) == reads


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--planted", "2017"], "no pre-planting year"),
        ([], "--planted YYYY is required"),
        (["--planted", "2020", "--year-a", "2020"], "must be before the planting year"),
        (["--planted", "2020", "--placebos", "5"], "--placebos"),
        (["--planted", "2020", "--match-year", "2020"], "--match-year 2020 is after the baseline year (2019)"),
        (["--planted", "2020", "--match-year", "2016"], "outside the store's years"),
        (["--aoi", "itapebi:forest"], "Demo 'itapebi' has no feature with role 'forest'"),
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
    assert result.exit_code == EXIT_USAGE and "The control covers 64 pixel centres" in result.output
    assert log.chunk_keys() == []
    # A data error found after the first read (no ring pixel like the site in the match year) also exits 2.
    late = CliRunner().invoke(app, [*base, "--planted", "2024"])
    assert late.exit_code == EXIT_USAGE and "Only 0 ring pixels are kept" in late.output


def test_cli_restore_forwards_every_option(restore_cli, tmp_path, monkeypatch):
    import tessera_apps.apps as apps

    seen: dict = {}

    def fake(aoi, out, **kwargs):
        seen.update(kwargs, aoi=aoi, out=out)
        raise apps.AppInputError("stop here")

    monkeypatch.setattr(apps, "run_restore", fake)
    base, _ = restore_cli
    extra = ["--planted", "2021", "--year-a", "2019", "--year-b", "2024", "--control-pct", "12.5", "--folds", "3"]
    extra += ["--match-year", "2017"]
    extra += ["--fold-cell-m", "150", "--placebos", "50", "--min-post-years", "2", "--seed", "4", "--no-canopy"]
    extra += ["--validate", "--strict-validate", "--force", "--snapshot", "abc", "--i-know-this-is-large"]
    result = CliRunner().invoke(app, [*base, *extra])
    assert result.exit_code == EXIT_USAGE and "stop here" in result.output
    expected = {
        "planted": 2021,
        "year_a": 2019,
        "year_b": 2024,
        "match_year": 2017,
        "control_pct": 12.5,
        "folds": 3,
        "fold_cell_m": 150.0,
        "placebos": 50,
        "min_post_years": 2,
        "seed": 4,
        "canopy": False,
        "validate": True,
        "strict_validate": True,
        "force": True,
        "snapshot_id": "abc",
        "max_area_km2": None,
        "ring_inner_km": 0.1,
        "ring_outer_km": 0.5,
    }
    assert {k: seen[k] for k in expected} == expected


def test_cli_restore_strict_exits_3_when_the_null_test_cannot_run(restore_cli, tmp_path, monkeypatch):
    import numpy as np

    from tessera_apps.apps import restore as rs

    monkeypatch.setattr(rs, "null_halves", lambda xy, centre, sectors=8: np.zeros(len(xy), dtype=np.int8))
    base, _ = restore_cli
    runner = CliRunner()
    args = [*base, "--planted", "2020", "--validate", "--no-canopy"]
    strict = runner.invoke(app, [*args, "--strict-validate"])
    assert strict.exit_code == EXIT_VALIDATION and "null test could not run" in strict.output
    loose = runner.invoke(app, args, env={"COLUMNS": "200"})
    assert loose.exit_code == 0 and "Null test: not run" in loose.stdout
    # The outputs are current and record the failed null test: a strict rerun says so with exit 3.
    again = runner.invoke(app, [*args, "--strict-validate"])
    assert again.exit_code == EXIT_VALIDATION and "current" in again.stdout


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
