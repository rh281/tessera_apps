"""The CLI wiring works offline."""

from __future__ import annotations

from typer.testing import CliRunner

from tessera_apps.cli import app


def test_help_lists_info():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "info" in result.output
