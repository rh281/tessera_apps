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


def _print_wetland(report: Mapping[str, Any], *, skipped: bool, out: Path, validation_redone: bool = False) -> None:
    """The run summary: seeds, threshold, extent both years, change, coastal fraction, validation, cost.

    Every value that comes from the user or the outside (paths, error text, warnings) is escaped, so
    a bracket in it is printed, not read as rich markup.
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
        f"[bold]Wetland {e(year_a)} to {e(year_b)}[/bold], AOI {e(report.get('aoi', {}).get('source'))}: "
        f"{_fmt(s.get('aoi_area_ha'), ',.0f')} ha in {e(', '.join(s.get('zones') or []))}"
    )
    seeds = s.get("seeds") or {}
    th = s.get("threshold") or {}
    score = s.get("score") or {}
    console.print(
        f"Seeds: {e(seeds.get('source'))}, {_fmt(seeds.get('used'), ',')} of {_fmt(seeds.get('pixels'), ',')} pixels "
        f"used ({_fmt(seeds.get('dropped_no_data'), ',')} without data, {_fmt(seeds.get('dropped_regime'), ',')} in "
        f"the other radar regime, {_fmt(seeds.get('dropped_duplicate'), ',')} repeated)"
    )
    console.print(
        f"  {_fmt(seeds.get('polygon_features'), ',')} polygons and {_fmt(seeds.get('point_seeds'), ',')} points in "
        f"{_fmt(seeds.get('spatial_blocks'), ',')} spatial blocks ({_fmt(seeds.get('fold_cell_m'), 'g')} m cells), "
        f"{e(th.get('folds'))} spatial folds"
    )
    method = "held-out" if th.get("method") == "heldout" else "in-sample (fallback)"
    console.print(
        f"Score: {e(score.get('type'))}; threshold {_fmt(th.get('threshold'), '.4g')}, the "
        f"{_fmt(th.get('percentile'), 'g')}th percentile of the {method} seed scores; the in-sample percentile is "
        f"{_fmt(th.get('in_sample_percentile'), '.4g')}"
    )
    table = Table(title="Extent")
    for col in ("year", "extent ha", "coastal fraction"):
        table.add_column(col, justify="right")
    for year in (year_a, year_b):
        row = (s.get("extent") or {}).get(str(year)) or {}
        table.add_row(e(year), _fmt(row.get("extent_ha"), ",.1f"), _fmt(row.get("coastal_fraction"), ".1%"))
    console.print(table)
    console.print(
        f"Change: net {_fmt(s.get('net_change_ha'), ',.1f')} ha ({_fmt(s.get('net_change_pct'), '+.1f')}%), gain "
        f"{_fmt(s.get('gain_ha'), ',.1f')} ha, loss {_fmt(s.get('loss_ha'), ',.1f')} ha; seeds within the threshold "
        f"in {e(year_b)}: {_fmt(seeds.get('share_year_b_within_threshold'), '.1%')}"
    )
    coast = s.get("coastal") or {}
    if coast.get("error"):
        console.print(f"Coastal fraction: not computed ({e(coast['error'])})")
    else:
        source = coast.get("file") or coast.get("source")
        console.print(
            f"Coastal fraction within {_fmt(coast.get('coast_km'), 'g')} km of the shoreline ({e(source)}"
            + (", approximate" if coast.get("approximate") else "")
            + f"): {_fmt(coast.get('fraction_year_a'), '.1%')} in {e(year_a)}, "
            f"{_fmt(coast.get('fraction_year_b'), '.1%')} in {e(year_b)}"
        )
    radar = s.get("radar") or {}
    console.print(
        f"Radar: seeds' regime {e(radar.get('seed_regime'))}; regime mismatch "
        f"{_fmt(radar.get('regime_mismatch_fraction'), '.1%')} of valid pixels, other regime "
        f"{_fmt(radar.get('other_regime_fraction'), '.1%')} (not classified)"
    )
    val = s.get("validation")
    if isinstance(val, Mapping) and "per_year" in val:
        for year, row in (val.get("per_year") or {}).items():
            console.print(
                f"Validation ({e(val.get('source'))} {e(year)}): recall {_fmt(row.get('recall'))}, precision "
                f"{_fmt(row.get('precision'))}, IoU {_fmt(row.get('iou'))}, F1 {_fmt(row.get('f1'))} over "
                f"{_fmt(row.get('pixels_scored'), ',')} pixels, seeds and {_fmt(val.get('seed_buffer_m'), 'g')} m "
                f"around them left out (recall with them {_fmt(row.get('recall_with_seeds'))})"
            )
            fp = row.get("false_positive_classes") or {}
            if fp:
                top = sorted(fp.items(), key=lambda kv: -float(kv[1].get("share") or 0.0))[:4]
                console.print(
                    "  false positives by WorldCover class: "
                    + ", ".join(f"{e(v.get('name'))} {_fmt(v.get('share'), '.0%')}" for _, v in top)
                )
        mix = val.get("version_mix")
        if isinstance(mix, Mapping):
            versions = mix.get("versions") or {}
            console.print(
                "  the years are scored against different WorldCover versions ("
                + ", ".join(f"{e(y)} {e(v)}" for y, v in versions.items())
                + f"), whose maps agree on the reference classes over {_fmt(mix.get('reference_agreement'), '.1%')} "
                "of the pixels scored in both years: compare the years with that in mind"
            )
        if val.get("years_not_covered"):
            console.print(f"  not scored (no reference map): {e(', '.join(str(y) for y in val['years_not_covered']))}")
    elif isinstance(val, Mapping) and "error" in val:
        console.print(f"Validation: not done ({e(val['error'])})")
    if s.get("storm_note"):
        console.print(f"Note: {e(s['storm_note'])}")
    wall = (report.get("timings") or {}).get("wall_s")
    cost = f"{_fmt((report.get('bytes_read') or 0) / 1e9, '.2f')} GB of chunks in {_fmt(wall, '.1f')} s"
    console.print(f"The run that wrote these outputs read {cost}" if skipped else f"Read {cost}")
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {e(warning)}", markup=True)


def _reference_classes(text: str) -> list[int]:
    try:
        codes = [int(part) for part in text.replace(" ", "").split(",") if part]
    except ValueError as exc:
        raise _fail(
            f"Error: --reference-classes {text!r} must be comma-separated WorldCover class codes, like 95 or 95,90.",
            EXIT_USAGE,
        ) from exc
    if not codes:
        raise _fail("Error: --reference-classes needs at least one WorldCover class code, like 95.", EXIT_USAGE)
    return codes


@app.command()
def wetland(
    aoi: Annotated[
        str, typer.Option("--aoi", help="AOI file (GeoJSON, GeoPackage, shapefile; any CRS) or a demo name.")
    ],
    year_a: Annotated[int, typer.Option("--year-a", help="First year; the seeds set the reference in it.")],
    year_b: Annotated[int, typer.Option("--year-b", help="Second year, after --year-a.")],
    seeds: Annotated[
        str | None,
        typer.Option(
            "--seeds",
            help="Seeds file (points or polygons of the wetland type, any CRS), or 'demo' for the AOI demo's bundled "
            "seeds (the default for --aoi everglades).",
        ),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = DEFAULT_OUT,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Score each year's extent against WorldCover, GMW or --reference."),
    ] = False,
    reference: Annotated[
        Path | None,
        typer.Option(
            "--reference",
            help="Vector file or GeoTIFF of the wetland type to validate against (implies --validate).",
        ),
    ] = None,
    reference_source: Annotated[
        str, typer.Option("--reference-source", help="worldcover (2020, 2021) or gmw.")
    ] = "worldcover",
    reference_classes: Annotated[
        str,
        typer.Option(
            "--reference-classes", help="WorldCover classes of the wetland type: 95 mangroves, 90 herbaceous."
        ),
    ] = "95",
    gmw_version: Annotated[str, typer.Option("--gmw-version", help="GMW version: v3 or v4.1.12.")] = "v3",
    score: Annotated[str, typer.Option("--score", help="cosine, or mahalanobis (500 or more seed pixels).")] = "cosine",
    threshold_pct: Annotated[
        float, typer.Option("--threshold-pct", help="Percentile of the held-out seed scores that sets the threshold.")
    ] = 95.0,
    folds: Annotated[int, typer.Option("--folds", help="Spatial folds of the seeds for the held-out scores.")] = 5,
    fold_cell_m: Annotated[
        float,
        typer.Option(
            "--fold-cell-m",
            help="Grid cell (metres) that blocks the seeds into spatial folds: seeds in one cell, or touching, share "
            "a fold. Smaller for a small AOI.",
        ),
    ] = 2000.0,
    min_area_ha: Annotated[float, typer.Option("--min-area-ha", help="Drop extent patches smaller than this.")] = 0.1,
    seed_buffer_m: Annotated[
        float, typer.Option("--seed-buffer-m", help="Leave pixels this close to a seed out of the validation.")
    ] = 200.0,
    shoreline: Annotated[
        Path | None,
        typer.Option("--shoreline", help="Shoreline lines or polygons (any CRS); Natural Earth 10m when omitted."),
    ] = None,
    coast_km: Annotated[float, typer.Option("--coast-km", help="Coastal buffer for the coastal fraction.")] = 1.0,
    seed: Annotated[int, typer.Option("--seed", help="Seed of the spatial fold assignment.")] = 0,
    force: Annotated[bool, typer.Option("--force", help="Rerun even when the outputs are current.")] = False,
    snapshot: Annotated[str | None, typer.Option("--snapshot", help="Pin an Icechunk snapshot id.")] = None,
    i_know_this_is_large: Annotated[
        bool, typer.Option("--i-know-this-is-large", help="Lift the 10,000 km2 AOI limit and the seed chunk limit.")
    ] = False,
    strict_validate: Annotated[
        bool,
        typer.Option(
            "--strict-validate",
            help="Stop with exit code 3, before reading any chunk, when --validate cannot load its reference.",
        ),
    ] = False,
) -> None:
    """Map a wetland type from seeds in two years, and how its extent moved in between."""
    from tessera_apps import io
    from tessera_apps.apps import AppInputError, StrictValidationError, run_wetland

    do_validate = validate or reference is not None
    classes = _reference_classes(reference_classes)
    with _rich_logging():
        try:
            result = run_wetland(
                aoi,
                year_a,
                year_b,
                out,
                seeds=seeds,
                score=score,
                threshold_pct=threshold_pct,
                folds=folds,
                fold_cell_m=fold_cell_m,
                min_area_ha=min_area_ha,
                shoreline=shoreline,
                coast_km=coast_km,
                validate=do_validate,
                reference=reference,
                reference_source=reference_source,
                reference_classes=classes,
                gmw_version=gmw_version,
                seed_buffer_m=seed_buffer_m,
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
    _print_wetland(result.report, skipped=result.skipped, out=out, validation_redone=result.validation_redone)
    if do_validate and not result.validation_written and strict_validate:
        raise _fail("Error: --strict-validate is set and the validation reference was not loaded.", EXIT_VALIDATION)


def _print_restore(report: Mapping[str, Any], *, skipped: bool, out: Path) -> None:
    """The run summary: the verdict on top, the trajectory table, control, regime, placebos, null test, canopy, cost.

    Every value that comes from the user or the outside (paths, error text, warnings) is escaped, so
    a bracket in it is printed, not read as rich markup.
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
    console.print(f"[bold]Verdict: {e(s.get('verdict'))}[/bold]")
    console.print(f"  {e(s.get('verdict_detail'))}")
    site = s.get("site") or {}
    console.print(
        f"Restore, planted {e(s.get('planted'))}, years {e(s.get('year_a'))} to {e(s.get('year_b'))}; site "
        f"{e(site.get('source'))}: {_fmt(site.get('area_ha'), ',.1f')} ha, {_fmt(site.get('pixels_used'), ',')} of "
        f"{_fmt(site.get('pixels'), ',')} pixels used, in {e(', '.join(s.get('zones') or []))}"
    )
    table = Table(title="Trajectory (divergence from control, change since the baseline, placebo band)")
    columns = (
        ("year", "year", "d"),
        ("post", "post_planting", "b"),
        ("D", "divergence", ".4g"),
        ("dD", "divergence_change", ".4g"),
        ("p05", "placebo_p05", ".4g"),
        ("p50", "placebo_p50", ".4g"),
        ("p95", "placebo_p95", ".4g"),
        ("sig", "significant", "b"),
        ("null dD", "null_divergence_change", ".4g"),
        ("departed", "departed_fraction", ".1%"),
        ("closer to target", "closer_to_target_fraction", ".1%"),
        ("progress", "target_progress", ".2f"),
        ("radar-free", "site_radar_free_fraction", ".0%"),
    )
    for name, _key, _spec in columns:
        table.add_column(name, justify="right")
    for row in s.get("trajectory") or []:
        cells = []
        for _name, key, spec in columns:
            value = row.get(key)
            if spec == "b":
                cells.append("yes" if value else "no")
            elif spec == "d":
                cells.append(e(value))
            else:
                cells.append(_fmt(value, spec))
        table.add_row(*cells)
    console.print(table)
    ctrl = s.get("control") or {}
    if ctrl.get("source") == "ring":
        console.print(
            f"Control: ring {_fmt(ctrl.get('ring_inner_km'), 'g')} to {_fmt(ctrl.get('ring_outer_km'), 'g')} km, "
            f"{_fmt(ctrl.get('kept_pixels'), ',')} of {_fmt(ctrl.get('candidate_pixels_valid_year_a'), ',')} pixels "
            f"kept at a baseline similarity of {_fmt(ctrl.get('threshold_similarity'), '.4f')} or more ("
            + ("held-out" if ctrl.get("threshold_source") == "heldout" else "in-sample, fallback")
            + f" {_fmt(ctrl.get('control_pct'), 'g')}th percentile, {e(ctrl.get('folds'))} folds of "
            f"{e(ctrl.get('site_cells'))} cells)"
        )
    else:
        console.print(
            f"Control: {e(ctrl.get('file'))}, {_fmt(ctrl.get('kept_pixels'), ',')} pixels used as given; "
            f"{_fmt(ctrl.get('share_below_threshold'), '.0%')} of them below the held-out threshold "
            f"{_fmt(ctrl.get('threshold_similarity'), '.4f')}"
        )
    balance = ctrl.get("balance") or {}
    console.print(
        f"  baseline divergence {_fmt(balance.get('baseline_divergence'), '.5f')}, above "
        f"{_fmt(balance.get('baseline_divergence_rank'), '.0%')} of the placebos' (median "
        f"{_fmt(balance.get('placebo_baseline_divergence_p50'), '.5f')})"
    )
    regime = s.get("regime") or {}
    dropped = regime.get("dropped") or {}
    console.print(
        "Panel and radar regime: "
        + "; ".join(
            f"{e(role)} {_fmt(d.get('used'), ',')} used, {_fmt(d.get('panel'), ',')} without data some year, "
            f"{_fmt(d.get('regime'), ',')} other radar pattern"
            for role, d in dropped.items()
            if d.get("read")
        )
    )
    if regime.get("radar_free_years"):
        console.print(f"  radar-free years in the majority pattern: {e(regime['radar_free_years'])}")
    placebos = s.get("placebos") or {}
    trend = s.get("trend") or {}
    console.print(
        f"Placebos: {e(placebos.get('count'))} of {e(placebos.get('pixels_each'))} pixels; false positive rate "
        f"{_fmt(placebos.get('false_positive_rate'), '.1%')}. Trend after planting: Theil-Sen "
        f"{_fmt(trend.get('theil_sen_slope'), '.4f')} per year, Spearman rho {_fmt(trend.get('spearman_rho'), '.2f')}"
    )
    null = s.get("null_test") or {}
    if null.get("error"):
        console.print(f"Null test: not run ({e(null['error'])})")
    else:
        console.print(
            f"Null test (control halves): {e(null.get('null_verdict'))}, max change "
            f"{_fmt(null.get('null_max_abs_change'), '.5f')} against the site's "
            f"{_fmt(null.get('site_max_abs_change'), '.5f')} (ratio {_fmt(null.get('null_to_site_ratio'), '.3f')}), "
            f"near zero: {e(null.get('null_near_zero'))}"
        )
    target = s.get("target")
    if isinstance(target, Mapping):
        console.print(
            f"Target: {_fmt(target.get('pixels'), ',')} pixels; progress is unclipped "
            "(0 is the control's similarity to the target, 1 the target's own halves')"
        )
    canopy = s.get("canopy")
    if isinstance(canopy, Mapping):
        if canopy.get("error"):
            console.print(f"Canopy check: not done ({e(canopy['error'])})")
        else:
            console.print(
                f"Canopy check (GLAD tree canopy height {e((canopy.get('years') or [None])[0])} to "
                f"{e((canopy.get('years') or [None])[-1])}): onset year {e(canopy.get('canopy_onset_year'))}, "
                f"Pearson {_fmt(canopy.get('pearson_divergence_vs_mean_gap_change'), '.2f')} and Spearman "
                f"{_fmt(canopy.get('spearman_divergence_vs_mean_gap_change'), '.2f')} between dD and the mean height "
                "gap change"
            )
    console.print(f"Limitation: {e(s.get('limitation'))}")
    wall = (report.get("timings") or {}).get("wall_s")
    cost = f"{_fmt((report.get('bytes_read') or 0) / 1e9, '.2f')} GB of chunks in {_fmt(wall, '.1f')} s"
    console.print(f"The run that wrote these outputs read {cost}" if skipped else f"Read {cost}")
    for warning in report.get("warnings") or []:
        console.print(f"[yellow]Warning:[/yellow] {e(warning)}", markup=True)


@app.command()
def restore(
    aoi: Annotated[
        str,
        typer.Option(
            "--aoi", "--site", help="The site polygon (GeoJSON, GeoPackage, shapefile; any CRS) or a demo name."
        ),
    ],
    planted: Annotated[
        int | None, typer.Option("--planted", help="Planting year (a demo site's properties may give it).")
    ] = None,
    year_a: Annotated[
        int | None, typer.Option("--year-a", help="Baseline year, before planting (default: planted - 1).")
    ] = None,
    year_b: Annotated[int | None, typer.Option("--year-b", help="Last year (default: 2025).")] = None,
    control: Annotated[
        str | None,
        typer.Option("--control", help="A control polygon used as given (no filter), or 'demo'; the ring otherwise."),
    ] = None,
    target: Annotated[
        str | None, typer.Option("--target", help="A mature reference stand (polygon), or 'demo'.")
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = DEFAULT_OUT,
    validate: Annotated[
        bool, typer.Option("--validate", help="Write the validation file: the null test and the canopy check.")
    ] = False,
    canopy: Annotated[
        bool, typer.Option("--canopy/--no-canopy", help="With --validate, check against GLAD tree canopy height.")
    ] = True,
    ring_inner_km: Annotated[float, typer.Option("--ring-inner-km", help="Inner edge of the control ring.")] = 1.0,
    ring_outer_km: Annotated[
        float, typer.Option("--ring-outer-km", help="Outer edge of the control ring, at most 10 km.")
    ] = 3.0,
    control_pct: Annotated[
        float,
        typer.Option(
            "--control-pct", help="Percentile of the held-out site similarities that sets the control filter."
        ),
    ] = 5.0,
    folds: Annotated[int, typer.Option("--folds", help="Spatial folds of the site's pixels.")] = 5,
    fold_cell_m: Annotated[
        float, typer.Option("--fold-cell-m", help="Grid cell (metres) that blocks the site's pixels into folds.")
    ] = 200.0,
    placebos: Annotated[int, typer.Option("--placebos", help="Placebo pseudo-sites drawn from the control.")] = 200,
    min_post_years: Annotated[int, typer.Option("--min-post-years", help="Post-planting years the verdict needs.")] = 3,
    seed: Annotated[int, typer.Option("--seed", help="Seed of the folds, the placebos and the target halves.")] = 0,
    force: Annotated[bool, typer.Option("--force", help="Rerun even when the outputs are current.")] = False,
    snapshot: Annotated[str | None, typer.Option("--snapshot", help="Pin an Icechunk snapshot id.")] = None,
    i_know_this_is_large: Annotated[
        bool, typer.Option("--i-know-this-is-large", help="Lift the 10,000 km2 limit on the extent and the target.")
    ] = False,
    strict_validate: Annotated[
        bool,
        typer.Option(
            "--strict-validate",
            help="Stop with exit code 3, before reading any chunk, when --validate cannot load the canopy layer.",
        ),
    ] = False,
) -> None:
    """Is a restoration site diverging from a matched control the way a working project should?"""
    from tessera_apps import io
    from tessera_apps.apps import AppInputError, StrictValidationError, run_restore

    with _rich_logging():
        try:
            result = run_restore(
                aoi,
                out,
                planted=planted,
                year_a=year_a,
                year_b=year_b,
                control=control,
                target=target,
                validate=validate,
                canopy=canopy,
                ring_inner_km=ring_inner_km,
                ring_outer_km=ring_outer_km,
                control_pct=control_pct,
                folds=folds,
                fold_cell_m=fold_cell_m,
                placebos=placebos,
                min_post_years=min_post_years,
                seed=seed,
                force=force,
                max_area_km2=None if i_know_this_is_large else io.MAX_AOI_KM2,
                snapshot_id=snapshot,
                strict_validate=strict_validate and validate,
            )
        except StrictValidationError as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_VALIDATION) from exc
        except (AppInputError, io.TesseraError) as exc:
            raise _fail(f"Error: {_error_text(exc)}", EXIT_USAGE) from exc
    _print_restore(result.report, skipped=result.skipped, out=out)
    if validate and canopy and strict_validate:
        canopy_rec = (result.report.get("summary") or {}).get("canopy")
        if isinstance(canopy_rec, Mapping) and canopy_rec.get("error"):
            raise _fail("Error: --strict-validate is set and the canopy check could not be done.", EXIT_VALIDATION)
