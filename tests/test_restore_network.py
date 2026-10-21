"""Network tier for the restore app: the bundled Itapebi demo end to end against the live store.

Run with ``pytest -m network tests/test_restore_network.py``. The run reads the live TESSERA store
anonymously (every ``AWS_*`` variable cleared) and GLAD annual tree canopy height from
storage.googleapis.com for the canopy check; it is skipped when either cannot be reached. Outputs
and the reference cache go to temporary directories.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest

from tessera_apps import demo, validate
from tessera_apps.apps import restore as rs
from tessera_apps.cli import app
from tessera_apps.io import TesseraReader

from .conftest import PlainCliRunner

pytestmark = pytest.mark.network

#: The outputs a validated restore run writes, for the Itapebi demo's years (2018 to 2025).
OUTPUTS = (
    "restore_2018_2025.tif",
    "restore_summary.parquet",
    "restore_summary.csv",
    "restore_placebos.parquet",
    "restore_preview.png",
    "restore_validation.json",
    "restore_report.json",
)


@pytest.fixture(scope="module", autouse=True)
def live_env(tmp_path_factory) -> Iterator[None]:
    """A fresh reference cache and no AWS credentials, so the anonymous read is what gets tested."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TESSERA_APPS_CACHE", str(tmp_path_factory.mktemp("cache")))
        for key in [k for k in os.environ if k.startswith("AWS_")]:
            mp.delenv(key)
        yield


@pytest.fixture(scope="module")
def reachable(live_env) -> None:
    """Skip unless the store opens and a GLAD canopy height tile over the demo can be found."""
    try:
        TesseraReader.open()
    except Exception as exc:  # icechunk and the object store raise their own errors
        pytest.skip(f"The TESSERA store cannot be reached: {type(exc).__name__}: {exc}")
    bbox = demo.demo_aoi("itapebi").bounds
    try:
        found = validate.glad_tch_reachable(bbox, 2020)
    except validate.ReferenceDataError as exc:
        pytest.skip(f"The GLAD canopy height host cannot be reached: {exc}")
    if not found:
        pytest.skip("No GLAD canopy height tile for 2020 covers the Itapebi demo.")


def test_the_itapebi_demo_runs_end_to_end_with_validation(reachable, tmp_path):
    out = tmp_path / "out"
    args = ["restore", "--aoi", "itapebi", "--target", "demo", "--validate", "--out", str(out)]
    result = PlainCliRunner().invoke(app, args, env={"COLUMNS": "200"})
    assert result.exit_code == 0, (result.output, result.exception)
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    report = json.loads((out / "restore_report.json").read_text(encoding="utf-8"))
    summary = report["summary"]
    verdict = summary["verdict"]
    assert verdict in (rs.VERDICT_INSUFFICIENT, rs.VERDICT_NONE) or (
        verdict.startswith("diverging from control since ") and summary["diverging_since"] is not None
    ), verdict
    assert summary["planted"] == 2019 and summary["year_a"] == 2018 and summary["match_year"] == 2017
    assert report["aoi"]["source"] == "demo:itapebi" and summary["target"] is not None
    assert summary["match_year_data"]["site"]["pixels"] == summary["site"]["pixels"]
    validation = json.loads((out / "restore_validation.json").read_text(encoding="utf-8"))
    assert "null_test" in validation["scores"] and "canopy" in validation["scores"]
    assert result.stdout.splitlines()[0] == f"Verdict: {verdict}"
