"""Command line entry point: ``tessera-apps``.

Exit codes: 0 on success or when the outputs are already current; 2 when an argument, the AOI, a
labels or reference file, the output directory or the store's published years stop the run before
it reads anything, or the data cannot answer the question as asked (no pixel comparable between
the two years, too few usable labels), with a message that says what to change; 3 when
``--strict-validate`` is set and the validation reference could not be loaded (checked before any
chunk read).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer

from tessera_apps import __version__

app = typer.Typer(help="Climate applications on the TESSERA v1.1 global embeddings.", no_args_is_help=True)

#: Exit code for a run stopped by its arguments, AOI, the store's published years, or unusable data.
EXIT_USAGE = 2
#: Exit code for a ``--strict-validate`` run whose reference could not be loaded.
EXIT_VALIDATION = 3
#: Default output directory.
DEFAULT_OUT = Path("out")


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


@contextmanager
def _rich_logging() -> Iterator[None]:
    """Show the package's INFO logs (the run plan, bytes read) on stderr through rich while a command runs."""
    from rich.console import Console
    from rich.logging import RichHandler

    handler = RichHandler(console=Console(stderr=True), show_path=False, markup=False)
    handler.setLevel(logging.INFO)
    pkg = logging.getLogger("tessera_apps")
    old_level = pkg.level
    pkg.addHandler(handler)
    pkg.setLevel(logging.INFO)
    try:
        yield
    finally:
        pkg.removeHandler(handler)
        pkg.setLevel(old_level)


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _error_text(exc: BaseException) -> str:
    return "\n".join([str(exc), *getattr(exc, "__notes__", [])])


def _fmt(value: Any, spec: str = ".3f") -> str:  # noqa: ANN401
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return format(value, spec)


def _print_wildfire(report: Mapping[str, Any], *, skipped: bool, out: Path, validation_redone: bool = False) -> None:
    """The run summary: burned area, threshold, recovery table, validation, cost."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    s = report.get("summary") or {}
    th = s.get("threshold") or {}
    otsu = th.get("otsu") or {}
    if skipped:
        console.print(f"Outputs in {out} are current for these arguments; pass --force to rerun.")
    elif validation_redone:
        console.print(
            f"Outputs in {out} are current for these arguments; the validation was redone from them without "
            "reading the store."
        )
    console.print(
        f"[bold]Wildfire {s.get('year_a')} to {s.get('year_b')}[/bold], AOI {report.get('aoi', {}).get('source')}: "
        f"{_fmt(s.get('aoi_area_ha'), ',.0f')} ha in {', '.join(s.get('zones') or [])}"
    )
    console.print(
        f"Burned: {_fmt(s.get('burned_area_ha'), ',.1f')} ha, {_fmt(s.get('burned_fraction'), '.1%')} of "
        f"{_fmt(s.get('classified_pixels'), ',')} classified pixels"
    )
    console.print(
        f"Threshold: {_fmt(th.get('threshold'))} ({th.get('method')}); Otsu {_fmt(otsu.get('threshold'))}, "
        f"eta {_fmt(otsu.get('eta'))}, valley ratio {_fmt(th.get('valley_ratio'), '.2f')}"
        + (f"; stable pair {th.get('stable_pair')} p99 {_fmt(th.get('stable_p99'))}" if th.get("fallback_ran") else "")
    )
    radar = s.get("radar") or {}
    console.print(
        f"Radar: regime mismatch {_fmt(radar.get('radar_mismatch_fraction'), '.1%')} of valid pixels, radar-free "
        f"share of the scar {_fmt(radar.get('radar_free_fraction_in_scar_year_a'), '.1%')} ({s.get('year_a')}) and "
        f"{_fmt(radar.get('radar_free_fraction_in_scar_year_b'), '.1%')} ({s.get('year_b')})"
    )
    table = Table(title="Recovery of burned pixels (similarity to their own pre-fire vector)")
    for col in ("year", "burned px", "recovered", "median R", "sim > 0.9", "median s burned", "median s reference"):
        table.add_column(col, justify="right")
    for row in s.get("recovery") or []:
        table.add_row(
            str(row.get("year")),
            _fmt(row.get("n_burned"), ","),
            _fmt(row.get("recovered_fraction"), ".1%"),
            _fmt(row.get("median_recovery_index"), ".2f"),
            _fmt(row.get("sim90_fraction"), ".1%"),
            _fmt(row.get("median_sim_burned")),
            _fmt(row.get("median_sim_unburned")),
        )
    console.print(table)
    ref = s.get("recovery_reference") or {}
    if ref:
        pre = ref.get("pre_fire_similarity") or {}
        post = ref.get("post_fire_similarity") or {}
        console.print(
            f"Recovery reference: {_fmt(ref.get('pixels'), ',')} pixels shown unchanged; their median similarity "
            f"is {_fmt(pre.get('median'))} before the fire and {_fmt(post.get('median'))} after it"
        )
    val = s.get("validation")
    if isinstance(val, Mapping) and "iou" in val:
        console.print(
            f"Validation ({val.get('source')}): IoU {_fmt(val.get('iou'))}, precision {_fmt(val.get('precision'))}, "
            f"recall {_fmt(val.get('recall'))}, F1 {_fmt(val.get('f1'))}; recall inside reference "
            f"{_fmt(val.get('recall_inside_reference'))}, burned outside reference "
            f"{_fmt(val.get('burned_outside_reference_fraction'), '.1%')}"
        )
    elif isinstance(val, Mapping) and "error" in val:
        console.print(f"Validation: not done ({val['error']})")
    wall = (report.get("timings") or {}).get("wall_s")
    cost = f"{_fmt((report.get('bytes_read') or 0) / 1e9, '.2f')} GB of chunks in {_fmt(wall, '.1f')} s"
    console.print(f"The run that wrote these outputs read {cost}" if skipped else f"Read {cost}")
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {warning}", markup=True)


@app.command()
def wildfire(
    aoi: Annotated[
        str, typer.Option("--aoi", help="AOI file (GeoJSON, GeoPackage, shapefile; any CRS) or a demo name.")
    ],
    year_a: Annotated[int, typer.Option("--year-a", help="The year before the fire.")],
    year_b: Annotated[int, typer.Option("--year-b", help="The year after the fire (not the fire year itself).")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = DEFAULT_OUT,
    validate: Annotated[
        bool, typer.Option("--validate", help="Score the burned mask against NIFC or --reference.")
    ] = False,
    reference: Annotated[
        Path | None,
        typer.Option(
            "--reference", help="Vector file of burned area to validate against instead of NIFC (implies --validate)."
        ),
    ] = None,
    fire_year: Annotated[
        int | None,
        typer.Option(
            "--fire-year", help="The fire's year (default: year-b minus 1, or year-b when it is year-a plus 1)."
        ),
    ] = None,
    fire_name: Annotated[
        str | None, typer.Option("--fire-name", help="Part of the NIFC incident name, for example Caldor.")
    ] = None,
    threshold: Annotated[
        float | None, typer.Option("--threshold", help="Fixed cosine distance threshold (default: Otsu).")
    ] = None,
    min_area_ha: Annotated[float, typer.Option("--min-area-ha", help="Drop burned patches smaller than this.")] = 1.0,
    recovery_through: Annotated[int, typer.Option("--recovery-through", help="Last recovery year.")] = 2025,
    force: Annotated[bool, typer.Option("--force", help="Rerun even when the outputs are current.")] = False,
    snapshot: Annotated[str | None, typer.Option("--snapshot", help="Pin an Icechunk snapshot id.")] = None,
    i_know_this_is_large: Annotated[
        bool, typer.Option("--i-know-this-is-large", help="Lift the 10,000 km2 AOI limit.")
    ] = False,
    strict_validate: Annotated[
        bool,
        typer.Option(
            "--strict-validate",
            help="Stop with exit code 3, before reading any chunk, when --validate cannot load its reference.",
        ),
    ] = False,
) -> None:
    """Map a burn scar between two years and track its recovery through --recovery-through."""
    from tessera_apps import io
    from tessera_apps.apps import AppInputError, StrictValidationError, run_wildfire

    do_validate = validate or reference is not None
    with _rich_logging():
        try:
            result = run_wildfire(
                aoi,
                year_a,
                year_b,
                out,
                threshold=threshold,
                min_area_ha=min_area_ha,
                fire_year=fire_year,
                fire_name=fire_name,
                validate=do_validate,
                reference=reference,
                recovery_through=recovery_through,
                force=force,
                max_area_km2=None if i_know_this_is_large else io.MAX_AOI_KM2,
                snapshot_id=snapshot,
                strict_validate=strict_validate and do_validate,
            )
        except StrictValidationError as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_VALIDATION) from exc
        except (AppInputError, io.TesseraError) as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_USAGE) from exc
    _print_wildfire(result.report, skipped=result.skipped, out=out, validation_redone=result.validation_redone)
    if do_validate and not result.validation_written and strict_validate:
        raise _fail("Error: --strict-validate is set and the validation reference was not loaded.", EXIT_VALIDATION)


def _print_deforest(report: Mapping[str, Any], *, skipped: bool, out: Path, validation_redone: bool = False) -> None:
    """The run summary: forest both years, conversion, patches, labels, classifier, validation, cost.

    Every value that comes from the user or the outside (paths, the labels file, the Hansen version,
    error text and warnings) is escaped, so a bracket in it is printed, not read as rich markup.
    """
    from rich.console import Console
    from rich.markup import escape
    from rich.table import Table

    def e(value: Any) -> str:  # noqa: ANN401
        return escape(str(value))

    console = Console()
    s = report.get("summary") or {}
    if skipped:
        console.print(f"Outputs in {e(out)} are current for these arguments; pass --force to rerun.")
    elif validation_redone:
        console.print(
            f"Outputs in {e(out)} are current for these arguments; the validation was redone from them without "
            "reading the store."
        )
    year_a, year_b = s.get("year_a"), s.get("year_b")
    console.print(
        f"[bold]Deforest {e(year_a)} to {e(year_b)}[/bold], AOI {e(report.get('aoi', {}).get('source'))}: "
        f"{_fmt(s.get('aoi_area_ha'), ',.0f')} ha in {e(', '.join(s.get('zones') or []))}"
    )
    forest = s.get("forest") or {}
    parts = []
    for year in (year_a, year_b):
        row = forest.get(str(year)) or {}
        parts.append(
            f"{e(year)} {_fmt(row.get('forest_ha'), ',.1f')} ha ({_fmt(row.get('forest_fraction'), '.1%')} of "
            f"{_fmt(row.get('classified_pixels'), ',')} classified pixels)"
        )
    console.print("Forest: " + "; ".join(parts))
    patches = s.get("patches") or {}
    console.print(
        f"Converted: {_fmt(s.get('converted_ha'), ',.1f')} ha, {_fmt(s.get('conversion_fraction_of_forest'), '.1%')} "
        f"of {e(year_a)} forest; {_fmt(patches.get('count'), ',')} patches (median "
        f"{_fmt(patches.get('median_ha'), '.2f')} ha, p90 {_fmt(patches.get('p90_ha'), '.2f')} ha, max "
        f"{_fmt(patches.get('max_ha'), ',.1f')} ha)"
    )
    cross = s.get("cross_check") or {}
    ring_cut = cross.get("ring_label_cut")
    console.print(
        f"Cross-check: distance above {_fmt(cross.get('cut'))}, the {_fmt(cross.get('percentile'), 'g')}th percentile "
        f"of {_fmt(cross.get('stable_forest_pixels'), ',')} stable forest pixels; it removed "
        f"{_fmt(cross.get('classifier_flips_below_cut'), ',')} classifier flips"
        + (
            ""
            if ring_cut is None
            else f"; the same percentile among {_fmt(cross.get('ring_label_cut_labels'), ',')} Hansen-stable ring "
            f"forest labels is {_fmt(ring_cut)}"
        )
    )
    labels = s.get("labels") or {}
    if labels.get("source") == "hansen":
        ring = labels.get("ring") or {}
        where = (
            f"Hansen {e(labels.get('hansen_version'))}, ring {_fmt(ring.get('inner_km'), 'g')} to "
            f"{_fmt(ring.get('outer_km'), 'g')} km around the AOI, {e(labels.get('chunks_used'))} of "
            f"{e(labels.get('chunks_with_labels'))} chunks"
        )
    else:
        where = f"{e(labels.get('file'))} ({e(labels.get('rows'))} rows, {e(labels.get('inside_aoi'))} inside the AOI)"
    radar = s.get("radar") or {}
    console.print(
        f"Labels: {where}; used {_fmt(labels.get('used_forest'), ',')} forest and "
        f"{_fmt(labels.get('used_nonforest'), ',')} non-forest in the {e(radar.get('training_regime'))} regime "
        f"({_fmt(labels.get('dropped_no_data'), ',')} without data, {_fmt(labels.get('dropped_regime'), ',')} in "
        "the other regime)"
    )
    clf = s.get("classifier") or {}
    held = clf.get("heldout") or {}
    train = clf.get("training") or {}
    console.print(
        f"Classifier: training accuracy {_fmt(train.get('accuracy'))}; held out by store chunk "
        f"({e(held.get('folds'))} folds): balanced accuracy {_fmt(held.get('balanced_accuracy'))}, recall forest "
        f"{_fmt(held.get('recall_forest'))}, non-forest {_fmt(held.get('recall_nonforest'))}"
    )
    transfer = s.get("transfer") or {}
    if transfer.get("available"):
        tb = transfer.get("year_b") or {}
        ta = transfer.get("year_a_heldout") or {}
        console.print(
            f"Transfer to {e(year_b)}: on {_fmt(transfer.get('labels'), ',')} Hansen-stable labels, balanced accuracy "
            f"{_fmt(tb.get('balanced_accuracy'))} (recall forest {_fmt(tb.get('recall_forest'))}, non-forest "
            f"{_fmt(tb.get('recall_nonforest'))}) against {_fmt(ta.get('balanced_accuracy'))} held out in {e(year_a)}"
        )
    elif transfer.get("reason"):
        console.print(f"Transfer to {e(year_b)}: not checked ({e(transfer['reason'])})")
    console.print(
        f"Radar: regime mismatch {_fmt(radar.get('regime_mismatch_fraction'), '.1%')} of valid pixels, other regime "
        f"{_fmt(radar.get('other_regime_fraction'), '.1%')} (not classified)"
    )
    hist = patches.get("histogram") or {}
    if hist:
        from tessera_apps.apps.deforest import PATCH_BIN_LABELS

        # The report is written with sorted keys, so a skipped rerun reads the bins back in text
        # order; show them smallest first, as the live run does.
        order = [k for k in PATCH_BIN_LABELS if k in hist] + [k for k in hist if k not in PATCH_BIN_LABELS]
        table = Table(title="Conversion patches by size (ha)")
        for label in order:
            table.add_column(e(label), justify="right")
        table.add_row(*[_fmt(hist[k], ",") for k in order])
        console.print(table)
    val = s.get("validation")
    if isinstance(val, Mapping) and "iou" in val:
        window = val.get("window")
        named = ""
        if isinstance(window, Mapping):
            named = f", loss after {window.get('after_year')} through {window.get('through_year')}"
        console.print(
            f"Validation ({e(val.get('source'))}{e(named)}): IoU {_fmt(val.get('iou'))}, precision "
            f"{_fmt(val.get('precision'))}, recall {_fmt(val.get('recall'))}, F1 {_fmt(val.get('f1'))} over "
            f"{_fmt(val.get('pixels_scored'), ',')} pixels"
        )
        lag = val.get("lag_aligned")
        if isinstance(lag, Mapping):
            lw = lag.get("window") or {}
            console.print(
                f"  against loss in {e(lw.get('from_year'))} through {e(lw.get('through_year'))} (lag-aligned): "
                f"IoU {_fmt(lag.get('iou'))}, precision {_fmt(lag.get('precision'))}, recall {_fmt(lag.get('recall'))}"
            )
        only_b = val.get("year_b_only")
        if isinstance(only_b, Mapping):
            console.print(
                f"  against loss in {e(year_b)} only: precision {_fmt(only_b.get('precision'))}, recall "
                f"{_fmt(only_b.get('recall'))}"
            )
        for year, key in ((year_a, "forest_agreement_year_a"), (year_b, "forest_agreement_year_b")):
            agree = val.get(key)
            if isinstance(agree, Mapping):
                console.print(
                    f"  {e(year)} forest map against Hansen's forest proxy: accuracy {_fmt(agree.get('accuracy'))}, "
                    f"recall forest {_fmt(agree.get('recall_forest'))}, "
                    f"non-forest {_fmt(agree.get('recall_nonforest'))}"
                )
        removed = val.get("cross_check_removed")
        if isinstance(removed, Mapping):
            console.print(
                f"  flips the cross-check removed: {_fmt(removed.get('flips_below_cut'), ',')}, of which "
                f"{_fmt(removed.get('loss_in_window'), ',')} with Hansen loss in the window and "
                f"{_fmt(removed.get('loss_lag_aligned'), ',')} in the lag-aligned one"
            )
    elif isinstance(val, Mapping) and "error" in val:
        console.print(f"Validation: not done ({e(val['error'])})")
    wall = (report.get("timings") or {}).get("wall_s")
    cost = f"{_fmt((report.get('bytes_read') or 0) / 1e9, '.2f')} GB of chunks in {_fmt(wall, '.1f')} s"
    console.print(f"The run that wrote these outputs read {cost}" if skipped else f"Read {cost}")
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {e(warning)}", markup=True)


@app.command()
def deforest(
    aoi: Annotated[
        str, typer.Option("--aoi", help="AOI file (GeoJSON, GeoPackage, shapefile; any CRS) or a demo name.")
    ],
    year_a: Annotated[int, typer.Option("--year-a", help="First year; the labels describe it.")],
    year_b: Annotated[int, typer.Option("--year-b", help="Second year, after --year-a.")],
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = DEFAULT_OUT,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Score the conversion against Hansen loss in (year-a, year-b] or --reference."),
    ] = False,
    reference: Annotated[
        Path | None,
        typer.Option(
            "--reference",
            help="Vector file or GeoTIFF of conversion to validate against instead of Hansen (implies --validate).",
        ),
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(
            "--labels", help="CSV of lon, lat, label (1 forest, 0 non-forest) and optional weight, instead of Hansen."
        ),
    ] = None,
    labels_per_class: Annotated[
        int, typer.Option("--labels-per-class", help="Most Hansen label pixels per class.")
    ] = 500,
    label_chunks: Annotated[
        int, typer.Option("--label-chunks", help="Most store chunks the Hansen labels are sampled from.")
    ] = 24,
    label_ring_km: Annotated[
        float, typer.Option("--label-ring-km", help="Outer edge of the label ring around the AOI, in km.")
    ] = 5.0,
    forest_cover: Annotated[
        int, typer.Option("--forest-cover", help="Lowest Hansen treecover2000 (percent) of a forest label.")
    ] = 50,
    nonforest_cover: Annotated[
        int, typer.Option("--nonforest-cover", help="Hansen treecover2000 (percent) below which a label is non-forest.")
    ] = 10,
    forest_prob: Annotated[
        float, typer.Option("--forest-prob", help="Forest probability at or above which a pixel is forest.")
    ] = 0.5,
    stable_percentile: Annotated[
        float,
        typer.Option(
            "--stable-percentile", help="Percentile of the stable forest distance that sets the cross-check cut."
        ),
    ] = 95.0,
    min_area_ha: Annotated[
        float, typer.Option("--min-area-ha", help="Drop conversion patches smaller than this.")
    ] = 0.5,
    hansen_version: Annotated[
        str, typer.Option("--hansen-version", help="Hansen Global Forest Change version.")
    ] = "GFC-2025-v1.13",
    seed: Annotated[int, typer.Option("--seed", help="Seed of the label sampling and the classifier.")] = 0,
    force: Annotated[bool, typer.Option("--force", help="Rerun even when the outputs are current.")] = False,
    snapshot: Annotated[str | None, typer.Option("--snapshot", help="Pin an Icechunk snapshot id.")] = None,
    i_know_this_is_large: Annotated[
        bool, typer.Option("--i-know-this-is-large", help="Lift the 10,000 km2 AOI limit.")
    ] = False,
    strict_validate: Annotated[
        bool,
        typer.Option(
            "--strict-validate",
            help="Stop with exit code 3, before reading any chunk, when --validate cannot load its reference.",
        ),
    ] = False,
) -> None:
    """Map forest in two years and the forest that became something else in between."""
    from tessera_apps import io
    from tessera_apps.apps import AppInputError, StrictValidationError, run_deforest

    do_validate = validate or reference is not None
    with _rich_logging():
        try:
            result = run_deforest(
                aoi,
                year_a,
                year_b,
                out,
                validate=do_validate,
                reference=reference,
                labels=labels,
                labels_per_class=labels_per_class,
                label_chunks=label_chunks,
                label_ring_km=label_ring_km,
                forest_cover=forest_cover,
                nonforest_cover=nonforest_cover,
                forest_prob=forest_prob,
                stable_percentile=stable_percentile,
                min_area_ha=min_area_ha,
                hansen_version=hansen_version,
                seed=seed,
                force=force,
                max_area_km2=None if i_know_this_is_large else io.MAX_AOI_KM2,
                snapshot_id=snapshot,
                strict_validate=strict_validate and do_validate,
            )
        except StrictValidationError as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_VALIDATION) from exc
        except (AppInputError, io.TesseraError) as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_USAGE) from exc
    _print_deforest(result.report, skipped=result.skipped, out=out, validation_redone=result.validation_redone)
    if do_validate and not result.validation_written and strict_validate:
        raise _fail("Error: --strict-validate is set and the validation reference was not loaded.", EXIT_VALIDATION)
