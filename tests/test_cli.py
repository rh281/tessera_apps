"""The CLI wiring works offline."""

from __future__ import annotations

import json
import re
import urllib.error

import pytest

from tessera_apps import validate
from tessera_apps.cli import EXIT_USAGE, EXIT_VALIDATION, app

from .conftest import (
    SITE_BOX,
    TARGET_BOX,
    KeyLog,
    PlainCliRunner,
    grid_polygon,
    restore_corner,
    restore_store,
    write_polygon,
)


def test_help_lists_info():
    result = PlainCliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "info" in result.output
    assert "--version" in result.output


def test_version_prints_the_package_version_and_exits_0(monkeypatch):
    from tessera_apps import __version__

    def no_store(**kwargs):
        raise AssertionError("--version must not open the store")

    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(no_store))
    result = PlainCliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output == f"tessera-apps {__version__}\n"
    # Eager: it wins over a command given after it, which then never runs.
    result = PlainCliRunner().invoke(app, ["--version", "info"])
    assert result.exit_code == 0, result.output
    assert result.output == f"tessera-apps {__version__}\n"


def test_version_is_single_sourced_and_has_a_changelog_section():
    import tomllib
    from pathlib import Path

    import tessera_apps

    repo = Path(__file__).resolve().parent.parent
    pyproject = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in pyproject["project"]
    assert pyproject["project"]["dynamic"] == ["version"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "src/tessera_apps/__init__.py"
    # The release notes come from this section, so the newest one must be the package's version.
    sections = re.findall(r"^## (\S+) \((\d{4}-\d{2}-\d{2})\)$", (repo / "CHANGELOG.md").read_text(), re.MULTILINE)
    assert sections, "CHANGELOG.md has no '## X.Y.Z (YYYY-MM-DD)' section"
    assert sections[0][0] == tessera_apps.__version__


def test_console_script_prints_the_version():
    # The installed entry point, not the in-process app: what `pip install` users run.
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from tessera_apps import __version__

    exe = shutil.which("tessera-apps", path=str(Path(sys.executable).parent))
    if exe is None:
        pytest.skip("the tessera-apps console script is not installed next to this Python")
    done = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout == f"tessera-apps {__version__}\n"


def _store_unreachable(**kwargs):
    from tessera_apps.io import StoreUnavailableError

    raise StoreUnavailableError("Cannot open the store s3://tessera-embeddings/v1.1/dclimate.icechunk: no route")


def test_info_exits_2_naming_the_reason_when_the_store_cannot_be_opened(monkeypatch):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(_store_unreachable))
    result = PlainCliRunner().invoke(app, ["info"])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "Error: Cannot open the store" in result.stderr and "no route" in result.stderr
    assert "Traceback" not in result.output


def _another_dataset(**kwargs):
    from tessera_apps.io import DatasetMismatchError

    raise DatasetMismatchError("1.1-cambridge")


def test_info_exits_2_naming_the_dataset_geotessera_found(monkeypatch):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(_another_dataset))
    result = PlainCliRunner().invoke(app, ["info"])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "as dataset 1.1-cambridge, not 1.1-dclimate" in result.stderr
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["deforest", "--aoi", "rondonia", "--year-a", "2019", "--year-b", "2022"],
        ["wetland", "--aoi", "everglades", "--year-a", "2020", "--year-b", "2024"],
        ["wildfire", "--aoi", "caldor", "--year-a", "2020", "--year-b", "2022"],
        ["restore", "--aoi", "itapebi"],
    ],
)
def test_every_app_exits_2_before_writing_when_the_store_is_another_dataset(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(_another_dataset))
    out = tmp_path / "out"
    result = PlainCliRunner().invoke(app, [*argv, "--out", str(out)])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "as dataset 1.1-cambridge, not 1.1-dclimate" in result.stderr
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize(("zone", "message"), [("99X", "is malformed"), ("59S", "Zone group 59S is not in the store")])
def test_info_exits_2_on_a_bad_zone(monkeypatch, zone, message):
    import zarr
    from zarr.storage import MemoryStore

    from tessera_apps.io import TesseraReader

    store = MemoryStore()
    zarr.open_group(store, mode="w")
    reader = TesseraReader.from_store(store)
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(lambda **kwargs: reader))
    result = PlainCliRunner().invoke(app, ["info", "--zone", zone])
    assert result.exit_code == EXIT_USAGE, result.output
    assert message in result.stderr


@pytest.mark.parametrize(
    "argv",
    [
        ["deforest", "--aoi", "rondonia", "--year-a", "2019", "--year-b", "2022"],
        ["wetland", "--aoi", "everglades", "--year-a", "2020", "--year-b", "2024"],
        ["wildfire", "--aoi", "caldor", "--year-a", "2020", "--year-b", "2022"],
        ["restore", "--aoi", "itapebi"],
    ],
)
def test_every_app_exits_2_before_writing_when_the_store_cannot_be_opened(monkeypatch, tmp_path, argv):
    monkeypatch.setattr("tessera_apps.io.TesseraReader.open", staticmethod(_store_unreachable))
    out = tmp_path / "out"
    result = PlainCliRunner().invoke(app, [*argv, "--out", str(out)])
    assert result.exit_code == EXIT_USAGE, result.output
    assert "Error: Cannot open the store" in result.stderr
    assert not out.exists() or not any(out.iterdir())


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
    result = PlainCliRunner().invoke(app, ["restore", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    text = result.output
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
    runner = PlainCliRunner()
    first = runner.invoke(app, [*base, "--planted", "2020"], env={"COLUMNS": "200"})
    assert first.exit_code == 0, (first.output, first.exception)
    lines = first.stdout.splitlines()
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
        assert text in first.stdout, text
    report = json.loads((tmp_path / "out" / "restore_report.json").read_text())
    assert report["summary"]["verdict"] == "diverging from control since 2020"
    reads = len(log.log)
    second = runner.invoke(app, [*base, "--planted", "2020"])
    assert second.exit_code == 0 and "current" in second.stdout and len(log.log) == reads


def test_cli_restore_trajectory_fits_80_columns(restore_cli):
    # A pipe or an 80-column terminal: every trajectory value prints in full, none is cut to an ellipsis.
    base, _ = restore_cli
    result = PlainCliRunner().invoke(app, [*base, "--planted", "2020"], env={"COLUMNS": "80"})
    assert result.exit_code == 0, (result.output, result.exception)
    text = result.stdout
    assert "\u2026" not in text
    table_lines = [line for line in text.splitlines() if line.startswith(("\u2503", "\u2502"))]
    assert table_lines and max(len(line) for line in table_lines) <= 80
    rows = [line for line in table_lines if re.match(r"^\u2502 20\d\d \u2502", line)]
    assert len(rows) == 2 * len({line[2:6] for line in rows}), "both tables have a row per year"
    for header in ("null dD", "departed chg", "closer to target", "radar-free"):
        assert header in text, header


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
    result = PlainCliRunner().invoke(app, [*base, *extra])
    assert result.exit_code == EXIT_USAGE, result.output
    assert message in result.output and log.chunk_keys() == []


def test_cli_restore_data_error_exits_2(restore_cli, tmp_path):
    base, log = restore_cli
    left, top = restore_corner()
    small = write_polygon(tmp_path / "small.geojson", grid_polygon(left, top, ((20, 28), (20, 28))))
    result = PlainCliRunner().invoke(app, [*base, "--planted", "2020", "--control", str(small)])
    assert result.exit_code == EXIT_USAGE and "The control covers 64 pixel centres" in result.output
    assert log.chunk_keys() == []
    # A data error found after the first read (no ring pixel like the site in the match year) also exits 2.
    late = PlainCliRunner().invoke(app, [*base, "--planted", "2024"])
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
    result = PlainCliRunner().invoke(app, [*base, *extra])
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
    runner = PlainCliRunner()
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
    runner = PlainCliRunner()
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
