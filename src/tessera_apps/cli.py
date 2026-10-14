"""Command line entry point: ``tessera-apps``. The app commands arrive from checkpoint 3 on."""

from __future__ import annotations

import json

import typer

from tessera_apps import __version__

app = typer.Typer(help="Climate applications on the TESSERA v1.1 global embeddings.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Climate applications on the TESSERA v1.1 global embeddings."""


@app.command()
def info(
    zone: str = typer.Option("10N", help="Zone group whose published years to list."),
    snapshot: str | None = typer.Option(None, help="Pin an Icechunk snapshot id."),
) -> None:
    """Open the store anonymously and print its provenance and one zone's published years."""
    from tessera_apps.io import TesseraReader

    reader = TesseraReader.open(snapshot_id=snapshot)
    out = {
        "tessera_apps": __version__,
        **reader.provenance(),
        "zone": zone,
        "years_complete": reader.years_complete(zone),
        "bytes_read": reader.bytes_read,
    }
    typer.echo(json.dumps(out, indent=1, default=str))
