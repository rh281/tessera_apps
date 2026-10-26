"""Precompute the hosted demo's outputs: run the five bundled demos and copy what the page shows.

The Streamlit page (``demo/app.py``) serves these files at once; only its map click reads the
store. This script runs each bundled demo with the package's own command line, as the README's
demo commands run it, pinned to one store snapshot, in a work directory outside the repository.
It then copies each run's preview, summary table and validation file into
``demo/precomputed/<name>/`` and writes ``demo/precomputed/index.json``: for each demo the app,
the AOI, the years, the exact command, the snapshot, the package and geotessera versions, the
bytes read and the wall time of the run that wrote the outputs, the headline numbers (each with
what it is scored against and the field of the report or validation file it comes from) and the
run's warnings.

The five commands, run from the work directory (``ID`` is the snapshot):

    tessera-apps wildfire --aoi caldor --year-a 2020 --year-b 2022 --validate
        --reference hansen2021_outline.geojson --severity caldor_s2_dnbr6.tif --snapshot ID --out out/caldor
    tessera-apps deforest --aoi rondonia --year-a 2019 --year-b 2022 --validate --snapshot ID --out out/rondonia
    tessera-apps wetland --aoi everglades --year-a 2020 --year-b 2024 --validate --snapshot ID --out out/everglades
    tessera-apps restore --aoi itapebi --target demo --validate --snapshot ID --out out/itapebi
    tessera-apps restore --aoi jaru --target demo --validate --snapshot ID --out out/jaru

The Caldor command is the README's second wildfire command. NIFC cannot be reached from the
build environment, so its outline is the Hansen 2021 loss outline bundled with the demo, written
as notebook 01 writes it (the file checkpoint A and notebook 01 scored against), and its severity
map is the Sentinel-2 dNBR proxy for MTBS that ``scripts/make_caldor_dnbr.py`` makes. Both are
made in the work directory when missing; ``--dnbr`` copies an existing map there instead of
building one (the script reads about 600 MB of Sentinel-2 imagery).

Every child process runs with every ``AWS_*`` variable removed from its environment: the store
and every reference are read anonymously. Reruns are idempotent: the apps skip a run whose
outputs are current, and the copies and the index are rewritten only when their bytes change.
The script ends by checking that no file under ``demo/precomputed`` holds an absolute local path,
an AWS credential or an em or en dash. Each command's output is appended to
``<work>/logs/<name>.log``, and each invocation adds a line per demo to
``<work>/logs/precompute_runs.jsonl`` (whether it ran or skipped, its wall time, the files copied).

Usage:
    python demo/precompute.py --work DIR [--snapshot ID] [--dnbr PATH] [--dest DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("precompute")

#: The repository root (this file is ``demo/precompute.py``).
REPO = Path(__file__).resolve().parent.parent
#: Where the page reads the precomputed files from.
DEFAULT_DEST = Path(__file__).resolve().parent / "precomputed"
#: The store snapshot every number in docs/checkpoints.md since checkpoint A comes from.
SNAPSHOT = "RC6FS07ZB0T76EXT8XRG"
#: The Caldor validation inputs, by the names checkpoint A and notebook 01 used.
OUTLINE_FILE = "hansen2021_outline.geojson"
DNBR_FILE = "caldor_s2_dnbr6.tif"
DNBR_SCRIPT = REPO / "scripts" / "make_caldor_dnbr.py"
#: The tag ``scripts/make_caldor_dnbr.py`` writes, which the wildfire app reads as "proxy, not MTBS".
DNBR_SOURCE_TAG = "sentinel2_dnbr"
CLI = "tessera-apps"
INDEX_FILE = "index.json"
INDEX_SCHEMA_VERSION = 1


class PrecomputeError(RuntimeError):
    """A step that cannot go on; the message says what to change."""


@dataclass(frozen=True)
class Demo:
    """One bundled demo: its name (the page's dropdown and folder), app and app arguments."""

    name: str
    app: str
    args: tuple[str, ...]

    def argv(self, snapshot: str) -> list[str]:
        """The command's arguments after ``tessera-apps``, with the snapshot and the output folder."""
        return [self.app, *self.args, "--snapshot", snapshot, "--out", f"out/{self.name}"]

    def out_dir(self, work: Path) -> Path:
        """The run's output folder in the work directory."""
        return work / "out" / self.name

    def output_names(self) -> dict[str, str]:
        """The outputs copied for the page, by the key the report's ``outputs`` block gives them."""
        return {
            "preview": f"{self.app}_preview.png",
            "summary_csv": f"{self.app}_summary.csv",
            "validation": f"{self.app}_validation.json",
        }


#: The demos in the page's order.
DEMOS: tuple[Demo, ...] = (
    Demo(
        "caldor",
        "wildfire",
        (
            "--aoi",
            "caldor",
            "--year-a",
            "2020",
            "--year-b",
            "2022",
            "--validate",
            "--reference",
            OUTLINE_FILE,
            "--severity",
            DNBR_FILE,
        ),
    ),
    Demo("rondonia", "deforest", ("--aoi", "rondonia", "--year-a", "2019", "--year-b", "2022", "--validate")),
    Demo("everglades", "wetland", ("--aoi", "everglades", "--year-a", "2020", "--year-b", "2024", "--validate")),
    Demo("itapebi", "restore", ("--aoi", "itapebi", "--target", "demo", "--validate")),
    Demo("jaru", "restore", ("--aoi", "jaru", "--target", "demo", "--validate")),
)


@dataclass(frozen=True)
class Doc:
    """A parsed JSON output and its file name, so every number it gives names where it came from."""

    file: str
    data: Mapping[str, Any]

    def get(self, path: str) -> Any:  # noqa: ANN401
        """The value at a dotted path, such as ``summary.burned_area_ha``.

        Raises:
            PrecomputeError: The field is missing, so no number is invented for it.
        """
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise PrecomputeError(
                    f"{self.file} has no field {path}: the output format changed, so update the headline fields "
                    "in demo/precompute.py to the report's field names."
                )
            node = node[part]
        return node

    def source(self, path: str) -> str:
        """Where a number comes from: the file name and the field."""
        return f"{self.file} {path}"


@dataclass
class RunRecord:
    """What one demo's step did in this invocation (kept in the work directory, not in the index)."""

    name: str
    command: str
    skipped: bool
    elapsed_s: float
    report_bytes_read: int
    report_wall_s: float
    copied: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# Environment and inputs
# ------------------------------------------------------------------------------------------------


def anonymous_env() -> tuple[dict[str, str], list[str]]:
    """The environment for the child processes: this one without any ``AWS_*`` variable.

    Returns:
        The environment, and the names of the variables removed (logged, never their values).
    """
    removed = sorted(k for k in os.environ if k.startswith("AWS_"))
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
    # Plain, wide logs: rich draws the CLI's summary for a terminal otherwise.
    env["NO_COLOR"] = "1"
    env["COLUMNS"] = "160"
    return env, removed


def find_cli() -> str:
    """The ``tessera-apps`` command of the Python environment running this script.

    Raises:
        PrecomputeError: The package is not installed there.
    """
    beside = Path(sys.executable).parent / CLI
    if beside.is_file():
        return str(beside)
    found = shutil.which(CLI)
    if found:
        return found
    raise PrecomputeError(
        f"{CLI} is not installed for {sys.executable}: run this script with the Python of the environment the "
        "package is installed in (from a checkout, .venv/bin/python demo/precompute.py), or install the package "
        "there (uv pip install --no-deps -e .)."
    )


def outline_bytes() -> bytes:
    """The bundled Hansen 2021 loss outline as notebook 01 writes it: a FeatureCollection of the one feature.

    Raises:
        PrecomputeError: The Caldor demo file does not hold exactly one outline feature.
    """
    from tessera_apps.demo import demo_geojson

    collection = demo_geojson("caldor")
    outline = [
        f for f in collection["features"] if (f.get("properties") or {}).get("role") == "hansen_2021_loss_outline"
    ]
    if len(outline) != 1:
        raise PrecomputeError(
            f"caldor.geojson holds {len(outline)} features with role hansen_2021_loss_outline, not 1: restore the "
            "bundled demo file (scripts/make_caldor_aoi.py writes it)."
        )
    return json.dumps({"type": "FeatureCollection", "features": outline}).encode("utf-8")


def ensure_outline(work: Path) -> Path:
    """Write the outline into the work directory unless the same bytes are already there."""
    path = work / OUTLINE_FILE
    data = outline_bytes()
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)
        logger.info("Wrote %s (sha256 %s).", OUTLINE_FILE, sha256_bytes(data))
    return path


def dnbr_source_tag(path: Path) -> str | None:
    """The ``TESSERA_APPS_SOURCE`` tag of a severity GeoTIFF, or None when it has none."""
    import rasterio

    with rasterio.open(path) as src:
        return src.tags().get("TESSERA_APPS_SOURCE")


def check_dnbr(path: Path) -> None:
    """Refuse a severity map that is not the Sentinel-2 dNBR proxy the index labels it as.

    Raises:
        PrecomputeError: The file has no ``TESSERA_APPS_SOURCE=sentinel2_dnbr`` tag.
    """
    tag = dnbr_source_tag(path)
    if tag != DNBR_SOURCE_TAG:
        raise PrecomputeError(
            f"{path} is not the Sentinel-2 dNBR map scripts/make_caldor_dnbr.py writes (its TESSERA_APPS_SOURCE tag "
            f"is {tag!r}, not {DNBR_SOURCE_TAG!r}): delete it, or pass --dnbr with the script's output."
        )


def ensure_dnbr(work: Path, dnbr_from: Path | None, env: Mapping[str, str], logs: Path) -> Path:
    """Put the Caldor severity map in the work directory: copied from ``--dnbr``, or made by the script.

    Args:
        work: The work directory.
        dnbr_from: An existing map to copy, or None to build one when the work directory has none.
        env: The anonymous environment for the script.
        logs: The folder for the script's log.

    Raises:
        PrecomputeError: The map given or found is not the Sentinel-2 proxy, differs from the copy
            already in the work directory, or the script failed.
    """
    path = work / DNBR_FILE
    if dnbr_from is not None:
        if not dnbr_from.is_file():
            raise PrecomputeError(f"--dnbr {dnbr_from} is not a file: give the path of caldor_s2_dnbr6.tif.")
        check_dnbr(dnbr_from)
        if path.is_file():
            if sha256_file(path) != sha256_file(dnbr_from):
                raise PrecomputeError(
                    f"{path} already exists and differs from --dnbr {dnbr_from}: delete one of them, so the run "
                    "is scored against the map you mean."
                )
        else:
            shutil.copyfile(dnbr_from, path)
            logger.info("Copied the severity map from %s (sha256 %s).", dnbr_from, sha256_file(path))
    elif not path.is_file():
        if not DNBR_SCRIPT.is_file():
            raise PrecomputeError(
                f"{DNBR_FILE} is not in {work} and {DNBR_SCRIPT} is missing: run from a checkout of the repository, "
                "or pass --dnbr with an existing map."
            )
        logger.info("Building %s with scripts/make_caldor_dnbr.py (about 600 MB of Sentinel-2 reads).", DNBR_FILE)
        elapsed = run_logged(
            [sys.executable, str(DNBR_SCRIPT), "--out", DNBR_FILE], cwd=work, env=env, log=logs / "make_caldor_dnbr.log"
        )
        logger.info("Built %s in %.0f s (sha256 %s).", DNBR_FILE, elapsed, sha256_file(path))
    check_dnbr(path)
    return path


# ------------------------------------------------------------------------------------------------
# Running and copying
# ------------------------------------------------------------------------------------------------


def run_logged(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> float:
    """Run a command with its output appended to a log file; return its wall time in seconds.

    The log keeps every invocation (each under a header line), so a rerun that skips does not
    overwrite the log of the run that wrote the outputs.

    Raises:
        PrecomputeError: The command exited with a non-zero code; the message has the log's last lines.
    """
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== {stamp} {shlex.join(argv)}\n")
    t0 = time.perf_counter()
    with log.open("a", encoding="utf-8") as fh:
        proc = subprocess.run(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise PrecomputeError(
            f"{shlex.join(argv)} (in {cwd}) exited with code {proc.returncode} after {elapsed:.0f} s. Its log is "
            f"{log}; its last lines:\n{tail}"
        )
    return elapsed


def sha256_bytes(data: bytes) -> str:
    """The hex sha256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """The hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_file(src: Path, dst: Path) -> bool:
    """Copy ``src`` to ``dst`` unless ``dst`` already holds the same bytes; True when it wrote."""
    data = src.read_bytes()
    if dst.is_file() and dst.read_bytes() == data:
        return False
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return True


def write_if_changed(path: Path, text: str) -> bool:
    """Write a text file unless it already holds exactly this text; True when it wrote."""
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return False
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def load_json(path: Path) -> dict[str, Any]:
    """Parse a JSON object file.

    Raises:
        PrecomputeError: The file is missing or is not a JSON object.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PrecomputeError(f"Cannot read {path} ({exc}): rerun the demo with --force in its folder.") from exc
    if not isinstance(data, dict):
        raise PrecomputeError(f"{path} is not a JSON object: rerun the demo with --force in its folder.")
    return data


def run_demo(demo: Demo, *, cli: str, work: Path, snapshot: str, env: Mapping[str, str]) -> RunRecord:
    """Run one demo (the app skips it when its outputs are current) and check what it wrote.

    Raises:
        PrecomputeError: The run failed, or its report does not match the demo, the snapshot or its files.
    """
    out = demo.out_dir(work)
    report_path = out / f"{demo.app}_report.json"
    before = sha256_file(report_path) if report_path.is_file() else None
    argv = demo.argv(snapshot)
    command = shlex.join([CLI, *argv])
    logger.info("%s: %s", demo.name, command)
    elapsed = run_logged([cli, *argv], cwd=work, env=env, log=work / "logs" / f"{demo.name}.log")
    report = Doc(report_path.name, load_json(report_path))
    skipped = before is not None and sha256_file(report_path) == before
    if report.get("app") != demo.app:
        raise PrecomputeError(f"{report_path} is a {report.get('app')} report, not {demo.app}: empty {out} and rerun.")
    read_snapshot = report.get("provenance.store.snapshot_id")
    if read_snapshot != snapshot:
        raise PrecomputeError(f"{report_path} was read at snapshot {read_snapshot}, not {snapshot}: empty {out}.")
    outputs = report.get("outputs")
    for key, name in demo.output_names().items():
        path = out / name
        if not path.is_file():
            why = (report.data.get("summary") or {}).get("validation")
            raise PrecomputeError(
                f"{demo.name}: {name} was not written ({why!r}). The page needs it: rerun once its reference can "
                "be loaded."
            )
        if outputs.get(key) != sha256_file(path):
            raise PrecomputeError(f"{path} is not the file its report vouches for: rerun the demo with --force.")
    record = RunRecord(
        name=demo.name,
        command=command,
        skipped=skipped,
        elapsed_s=round(elapsed, 1),
        report_bytes_read=int(report.get("bytes_read")),
        report_wall_s=float(report.get("timings.wall_s")),
    )
    logger.info(
        "%s: %s in %.1f s; the run that wrote the outputs read %s bytes in %.1f s.",
        demo.name,
        "outputs current, skipped" if skipped else "ran",
        elapsed,
        f"{record.report_bytes_read:,}",
        record.report_wall_s,
    )
    return record


def copy_outputs(demo: Demo, work: Path, dest: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Copy the page's files of one demo into ``dest/<name>/``.

    Returns:
        The index's ``files`` block (path relative to ``dest``, size, sha256) and the names rewritten.
    """
    folder = dest / demo.name
    folder.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    written: list[str] = []
    for key, name in demo.output_names().items():
        src = demo.out_dir(work) / name
        dst = folder / name
        if sync_file(src, dst):
            written.append(name)
        files[key] = {"path": f"{demo.name}/{name}", "bytes": dst.stat().st_size, "sha256": sha256_file(dst)}
    return files, written


# ------------------------------------------------------------------------------------------------
# The index
# ------------------------------------------------------------------------------------------------


#: How the page names a binary score.
SCORE_NAMES = {"iou": "IoU", "precision": "Precision", "recall": "Recall", "f1": "F1"}


def _fmt_ha(value: float) -> str:
    return f"{value:,.1f} ha"


def _fmt_share(value: float) -> str:
    return f"{value:.1%}"


def _fmt_3(value: float) -> str:
    return f"{value:.3f}"


def number(
    doc: Doc,
    path: str,
    key: str,
    label: str,
    fmt: Callable[[Any], str],
    *,
    unit: str | None = None,
    scored_against: str | None = None,
    reference: str | None = None,
) -> dict[str, Any]:
    """One headline number: its value read from ``doc``, how the page shows it, and where it came from."""
    value = doc.get(path)
    item: dict[str, Any] = {"key": key, "label": label, "value": value, "display": fmt(value), "unit": unit}
    if scored_against is not None:
        item["scored_against"] = scored_against
    if reference is not None:
        item["reference"] = reference
    item["source"] = doc.source(path)
    return item


def _same(a: Doc, path_a: str, b: Doc, path_b: str) -> None:
    """Fail when the report and the validation file disagree on a number the page shows."""
    if a.get(path_a) != b.get(path_b):
        raise PrecomputeError(
            f"{a.file} {path_a} ({a.get(path_a)!r}) differs from {b.file} {path_b} ({b.get(path_b)!r}): the outputs "
            "do not come from one run; rerun the demo with --force."
        )


def wildfire_headline(report: Doc, val: Doc) -> tuple[list[dict[str, Any]], list[str]]:
    """Burned area, the threshold, the outline IoU and the severity IoU against moderate or worse.

    Raises:
        PrecomputeError: The validation inputs are not the stand-ins this script labels.
    """
    if not (val.get("reference.hansen_derived") is True and val.get("reference.source") == "file"):
        raise PrecomputeError(
            f"{val.file}: the outline is not the Hansen-derived file this script labels as the NIFC stand-in; "
            "update the labels in wildfire_headline."
        )
    if not (
        val.get("reference.severity.proxy") is True and val.get("reference.severity.source_tag") == DNBR_SOURCE_TAG
    ):
        raise PrecomputeError(
            f"{val.file}: the severity map is not the Sentinel-2 dNBR proxy this script labels as the MTBS stand-in; "
            "update the labels in wildfire_headline."
        )
    _same(report, "summary.validation.iou", val, "scores.burned.iou")
    _same(report, "summary.validation.severity.moderate_or_worse.iou", val, "scores.severity.moderate_or_worse.iou")
    outline = (
        f"the Hansen 2021 loss outline bundled with the demo ({val.get('reference.file')}, sha256 "
        f"{val.get('reference.sha256')[:16]})"
    )
    outline_note = (
        "A stand-in for the NIFC Caldor perimeter, which could not be reached from the build environment: the "
        "outline of Hansen Global Forest Change 2021 forest loss, not a fire perimeter."
    )
    severity = (
        f"moderate or worse burn (classes 3 and 4) in {val.get('reference.severity.file')}, sha256 "
        f"{val.get('reference.severity.sha256')[:16]}"
    )
    severity_note = (
        "A Sentinel-2 dNBR proxy for MTBS, made by scripts/make_caldor_dnbr.py, since MTBS could not be reached "
        "from the build environment. Likely optimistic against real MTBS: the proxy and the embeddings both come "
        "from Sentinel-2."
    )
    method = report.get("summary.threshold.method")
    items = [
        number(report, "summary.burned_area_ha", "burned_area_ha", "Burned area", _fmt_ha, unit="ha"),
        number(
            report,
            "summary.burned_fraction",
            "burned_fraction",
            "Share of classified pixels burned",
            _fmt_share,
            unit="fraction",
        ),
        number(
            report,
            "summary.threshold.threshold",
            "threshold",
            f"Cosine distance threshold (method: {method})",
            _fmt_3,
            unit="cosine distance",
        ),
        number(
            val,
            "scores.severity.moderate_or_worse.iou",
            "severity_iou",
            "IoU against moderate or worse burn",
            _fmt_3,
            unit="IoU",
            scored_against=severity,
            reference=severity_note,
        ),
        number(
            val,
            "scores.burned.iou",
            "outline_iou",
            "IoU against the reference outline",
            _fmt_3,
            unit="IoU",
            scored_against=outline,
            reference=outline_note,
        ),
        number(
            val,
            "scores.recall_inside_reference",
            "outline_recall_inside",
            "Share of the outline mapped as burned",
            _fmt_3,
            unit="recall",
            scored_against=outline,
            reference=outline_note,
        ),
    ]
    labels = [
        "The reference outline is a Hansen 2021 loss stand-in for the NIFC perimeter, which could not be reached.",
        "The severity map is a Sentinel-2 dNBR proxy for MTBS, which could not be reached.",
        f"The demo AOI is provisional: {report.get('aoi.source')} is derived from Hansen loss, not the NIFC perimeter.",
    ]
    return items, labels


def deforest_headline(report: Doc, val: Doc) -> tuple[list[dict[str, Any]], list[str]]:
    """Forest loss area, forest in both years, and the scores against Hansen loss."""
    _same(report, "summary.validation.iou", val, "scores.conversion.iou")
    if val.get("reference.source") != "hansen":
        raise PrecomputeError(f"{val.file}: the reference is not Hansen; update the labels in deforest_headline.")
    year_a, year_b = report.get("summary.year_a"), report.get("summary.year_b")
    window = val.get("reference.loss_window")
    against = (
        f"Hansen Global Forest Change {val.get('reference.version')} loss after {window['after_year']} through "
        f"{window['through_year']} (lossyear codes {', '.join(str(c) for c in window['lossyear_codes'])}), on land "
        "(datamask 1)"
    )
    ring_inner, ring_outer = report.get("summary.labels.ring.inner_km"), report.get("summary.labels.ring.outer_km")
    note = (
        "Hansen Global Forest Change, the brief's reference for this app. The classifier's training labels also "
        f"come from Hansen, sampled only in a ring {ring_inner:g} to {ring_outer:g} km outside the AOI, so the "
        "score inside the AOI is spatially held out but not independent of Hansen."
    )
    scores = [
        number(
            val,
            f"scores.conversion.{m}",
            f"hansen_{m}",
            f"{SCORE_NAMES[m]} against Hansen loss",
            _fmt_3,
            unit=m,
            scored_against=against,
            reference=note,
        )
        for m in ("precision", "recall", "iou", "f1")
    ]
    items = [
        number(
            report,
            "summary.converted_ha",
            "forest_loss_ha",
            f"Forest converted to non-forest, {year_a} to {year_b}",
            _fmt_ha,
            unit="ha",
        ),
        number(
            report,
            "summary.conversion_fraction_of_forest",
            "loss_share_of_forest",
            f"Share of the {year_a} forest converted",
            _fmt_share,
            unit="fraction",
        ),
        number(report, "summary.forest_ha_year_a", "forest_ha_year_a", f"Forest in {year_a}", _fmt_ha, unit="ha"),
        number(report, "summary.forest_ha_year_b", "forest_ha_year_b", f"Forest in {year_b}", _fmt_ha, unit="ha"),
        *scores,
    ]
    return items, []


def wetland_headline(report: Doc, val: Doc) -> tuple[list[dict[str, Any]], list[str]]:
    """Extent in both years, net change, and the scores against the reference the run used."""
    year_a, year_b = report.get("summary.year_a"), report.get("summary.year_b")
    scored = val.get("reference.years_scored")
    if len(scored) != 1:
        raise PrecomputeError(f"{val.file} scores years {scored}; wetland_headline shows one year's scores.")
    year = scored[0]
    _same(report, "summary.validation.iou", val, f"scores.extent_{year}.iou")
    source = val.get("reference.source")
    if source != "worldcover":
        raise PrecomputeError(
            f"{val.file}: the reference is {source!r}, not WorldCover; update the labels in wetland_headline."
        )
    classes = ", ".join(f"{c} {val.get('reference.class_names')[str(c)]}" for c in val.get("reference.classes"))
    versions = val.get("reference.versions")
    not_covered = val.get("reference.years_not_covered")
    against = (
        f"{val.get('reference.dataset')} {year} ({versions[str(year)]}), class {classes}, leaving out the seed "
        f"pixels and {val.get('reference.seed_buffer_m'):g} m around them"
    )
    note = (
        "The reference this run used: the ESA WorldCover mangrove class, a stand-in for Global Mangrove Watch, which "
        "could not be reached from the build environment. The bundled seeds were drawn from WorldCover too, so the "
        "score is not independent of the seeds."
    )
    if not_covered:
        note += f" WorldCover has no map for {', '.join(str(y) for y in not_covered)}, so that year is not scored."
    scores = [
        number(
            val,
            f"scores.extent_{year}.{m}",
            f"reference_{m}",
            f"{SCORE_NAMES[m]} against WorldCover {year} mangroves",
            _fmt_3,
            unit=m,
            scored_against=against,
            reference=note,
        )
        for m in ("iou", "precision", "recall", "f1")
    ]
    items = [
        number(report, "summary.extent_ha_year_a", "extent_ha_year_a", f"Extent in {year_a}", _fmt_ha, unit="ha"),
        number(report, "summary.extent_ha_year_b", "extent_ha_year_b", f"Extent in {year_b}", _fmt_ha, unit="ha"),
        number(
            report,
            "summary.net_change_ha",
            "net_change_ha",
            f"Net change, {year_a} to {year_b}",
            lambda v: f"{v:+,.1f} ha",
            unit="ha",
        ),
        number(report, "summary.gain_ha", "gain_ha", "Gain", _fmt_ha, unit="ha"),
        number(report, "summary.loss_ha", "loss_ha", "Loss", _fmt_ha, unit="ha"),
        *scores,
    ]
    return items, ["The wetland reference is the ESA WorldCover mangrove class, a stand-in for Global Mangrove Watch."]


def restore_headline(report: Doc, val: Doc) -> tuple[list[dict[str, Any]], list[str]]:
    """The verdict, the null test and the canopy check."""
    _same(report, "summary.null_test.null_verdict", val, "scores.null_test.null_verdict")
    _same(report, "summary.canopy.canopy_onset_year", val, "scores.canopy.canopy_onset_year")
    canopy_error = val.get("reference.canopy.error")
    if canopy_error:
        raise PrecomputeError(f"{val.file}: the canopy check did not run ({canopy_error}); rerun when it can.")
    null_against = str(val.get("reference.null_test"))
    null_note = (
        "A check that the method reports nothing where nothing happened, not an accuracy: there is no global "
        "reference layer for restoration outcomes."
    )
    canopy_years = val.get("scores.canopy.years")
    canopy_against = f"{val.get('reference.canopy.dataset')}, {canopy_years[0]} to {canopy_years[-1]}"
    canopy_note = (
        f"GLAD canopy height is a 30 m model that sees only trees of {val.get('scores.canopy.tall_m')} m or more, "
        "so a young planting is invisible to it for its first years: an independent check, not ground truth."
    )
    items = [
        number(report, "summary.verdict", "verdict", "Verdict", str),
        number(report, "summary.verdict_detail", "verdict_detail", "Why", str),
        number(
            val,
            "scores.null_test.null_verdict",
            "null_verdict",
            "Null test verdict",
            str,
            scored_against=null_against,
            reference=null_note,
        ),
        number(
            val,
            "scores.null_test.null_max_abs_change",
            "null_max_abs_change",
            "Null test: largest divergence change",
            lambda v: f"{v:.4f}",
            unit="cosine distance",
            scored_against=null_against,
            reference=null_note,
        ),
        number(
            val,
            "scores.null_test.null_to_site_ratio",
            "null_to_site_ratio",
            "Null test against the site's change",
            _fmt_3,
            unit="ratio",
            scored_against=null_against,
            reference=null_note,
        ),
        number(
            val,
            "scores.site_sized_null.placebo_false_positive_rate",
            "placebo_false_positive_rate",
            "Placebos called diverging",
            _fmt_share,
            unit="fraction",
            scored_against="site-sized pieces of the control through the same verdict rule",
            reference=null_note,
        ),
        number(
            val,
            "scores.canopy.canopy_onset_year",
            "canopy_onset_year",
            "Canopy onset year",
            str,
            unit="year",
            scored_against=canopy_against,
            reference=canopy_note,
        ),
        number(
            val,
            "scores.canopy.correlation_sign",
            "canopy_correlation_sign",
            "Divergence against canopy gap",
            str,
            scored_against=canopy_against,
            reference=canopy_note,
        ),
        number(
            val,
            "scores.canopy.spearman_divergence_vs_tall_gap_change",
            "canopy_spearman_tall",
            f"Spearman rho, divergence against the gap in canopy of {val.get('scores.canopy.tall_m')} m or more",
            _fmt_3,
            unit="rho",
            scored_against=canopy_against,
            reference=canopy_note,
        ),
    ]
    return items, []


HEADLINES: dict[str, Callable[[Doc, Doc], tuple[list[dict[str, Any]], list[str]]]] = {
    "wildfire": wildfire_headline,
    "deforest": deforest_headline,
    "wetland": wetland_headline,
    "restore": restore_headline,
}


def index_entry(demo: Demo, *, work: Path, snapshot: str, files: Mapping[str, Any]) -> dict[str, Any]:
    """The index entry of one demo, every number read from its report or validation file."""
    from tessera_apps.demo import demo_aoi, demo_properties

    out = demo.out_dir(work)
    report = Doc(f"{demo.app}_report.json", load_json(out / f"{demo.app}_report.json"))
    val = Doc(f"{demo.app}_validation.json", load_json(out / f"{demo.app}_validation.json"))
    props = demo_properties(demo.name)
    aoi: dict[str, Any] = {
        "demo": demo.name,
        "title": props.get("name"),
        "source": report.get("aoi.source"),
        "area_ha": report.get("aoi.area_ha"),
        "pixels": report.get("aoi.pixels"),
        "zones": report.get("aoi.zones"),
        "sha256": report.get("aoi.sha256"),
        "bounds_lonlat": [round(float(b), 6) for b in demo_aoi(demo.name).bounds],
    }
    years: dict[str, Any] = {
        "year_a": report.get("summary.year_a"),
        "year_b": report.get("summary.year_b"),
        "read": report.get("provenance.years"),
    }
    labels: list[str] = []
    if demo.app == "wildfire":
        aoi["status"] = props.get("status")
        years["fire_year"] = report.get("summary.fire_year")
        years["recovery"] = report.get("summary.recovery_years")
    if demo.app == "wetland" and props.get("years_note"):
        years["note"] = props["years_note"]
    if demo.app == "restore":
        aoi["extent_area_ha"] = report.get("aoi.extent_area_ha")
        aoi["land_use"] = props.get("land_use")
        aoi["synthetic"] = bool(props.get("synthetic"))
        years["planted"] = report.get("summary.planted")
        years["trajectory"] = report.get("summary.years")
        if aoi["synthetic"]:
            labels.append(
                "Synthetic site: located from GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery because no "
                "public planting polygon could be reached. It is not a documented project."
            )
    items, app_labels = HEADLINES[demo.app](report, val)
    labels.extend(app_labels)
    entry: dict[str, Any] = {
        "name": demo.name,
        "app": demo.app,
        "aoi": aoi,
        "years": years,
        "command": shlex.join([CLI, *demo.argv(snapshot)]),
        "command_cwd": (
            "the work directory, holding hansen2021_outline.geojson and caldor_s2_dnbr6.tif"
            if demo.app == "wildfire"
            else "the work directory"
        ),
        "snapshot": report.get("provenance.store.snapshot_id"),
        "versions": {
            "tessera_apps": report.get("provenance.tessera_apps_version"),
            "geotessera": report.get("provenance.libraries.geotessera"),
        },
        "bytes_read": report.get("bytes_read"),
        "wall_s": report.get("timings.wall_s"),
        "run_created_utc": report.get("provenance.created_utc"),
        "files": dict(files),
        "headline": items,
        "labels": labels,
        "warnings": list(report.get("warnings")),
    }
    if demo.app == "wildfire":
        entry["inputs"] = [
            {
                "file": report.get("validation_params.reference.file"),
                "sha256": report.get("validation_params.reference.sha256"),
                "made_by": "the caldor.geojson feature with role hansen_2021_loss_outline, written as notebook 01 does",
            },
            {
                "file": report.get("validation_params.severity.file"),
                "sha256": report.get("validation_params.severity.sha256"),
                "made_by": f"python scripts/make_caldor_dnbr.py --out {DNBR_FILE}",
            },
        ]
        entry["app_headline"] = report.get("summary.headline.text")
    return entry


def build_index(entries: Sequence[Mapping[str, Any]], *, work: Path, snapshot: str) -> dict[str, Any]:
    """The whole index: the store read and one entry per demo."""
    first = load_json(DEMOS[0].out_dir(work) / f"{DEMOS[0].app}_report.json")
    store = Doc(f"{DEMOS[0].app}_report.json", first).get("provenance.store")
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "about": (
            "Precomputed outputs of the five bundled demos, made by demo/precompute.py. Every number comes from the "
            "run's report or validation file, named in its source field; bytes_read and wall_s are those of the run "
            "that wrote the outputs."
        ),
        "snapshot": snapshot,
        "store": {k: store.get(k) for k in ("url", "dataset", "model_version", "build_version")},
        "demos": list(entries),
    }


# ------------------------------------------------------------------------------------------------
# The final check
# ------------------------------------------------------------------------------------------------

#: Em and en dashes and their look-alikes (figure dash, horizontal bar, minus sign, small and wide forms).
DASHES = "".join(chr(c) for c in (0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0xFE58, 0xFE63, 0xFF0D))
_ABSOLUTE = re.compile(r"(?<![A-Za-z0-9_.])(/home/|/tmp/|/root/|/Users/|/var/folders/|[A-Za-z]:\\\\)")
_AWS_KEY_ID = re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")
_AWS_NAMES = re.compile(r"aws_(secret_access_key|access_key_id|session_token)", re.IGNORECASE)


def png_text(data: bytes) -> str:
    """The text chunks (tEXt, iTXt, zTXt) of a PNG, joined; image data is not text."""
    out: list[str] = []
    i = 8
    while i + 8 <= len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        if kind in (b"tEXt", b"iTXt", b"zTXt"):
            out.append(data[i + 8 : i + 8 + n].decode("latin-1"))
        i += 12 + n
    return "\n".join(out)


def check_tree(dest: Path, secrets: Sequence[str], local_paths: Sequence[str]) -> list[str]:
    """Problems in the published files: an absolute local path, an AWS credential, a dash.

    Args:
        dest: The folder to check (``demo/precomputed``).
        secrets: Values of ``AWS_*`` variables set when the script started; none may appear.
        local_paths: This machine's own paths (the work directory, the repository, the home folder).

    Returns:
        One line per problem; empty when the folder is clean.
    """
    problems: list[str] = []
    for path in sorted(p for p in dest.rglob("*") if p.is_file()):
        rel = path.relative_to(dest)
        raw = path.read_bytes()
        text = png_text(raw) if path.suffix == ".png" else raw.decode("utf-8", errors="replace")
        if _ABSOLUTE.search(text) or any(p in text for p in local_paths):
            problems.append(f"{rel}: holds an absolute local path")
        if any(p.encode() in raw for p in local_paths):
            problems.append(f"{rel}: its bytes hold a local path")
        if _AWS_KEY_ID.search(text) or _AWS_NAMES.search(text) or any(s.encode() in raw for s in secrets):
            problems.append(f"{rel}: holds what looks like an AWS credential")
        found = sorted({c for c in text if c in DASHES} | {c for c in DASHES if f"\\u{ord(c):04x}" in text.lower()})
        if found:
            problems.append(f"{rel}: holds dash characters {', '.join(f'U+{ord(c):04X}' for c in found)}")
    return problems


def tree_size(dest: Path) -> tuple[int, int]:
    """The number of files under ``dest`` and their total size in bytes."""
    files = [p for p in dest.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def precompute(work: Path, *, dest: Path, snapshot: str, dnbr_from: Path | None) -> list[RunRecord]:
    """Run the five demos, copy their files, write the index and check the published folder.

    Raises:
        PrecomputeError: Any step that cannot go on; the message says what to change.
    """
    work = work.expanduser().resolve()
    if work == REPO or REPO in work.parents:
        raise PrecomputeError(
            f"--work {work} is inside the repository {REPO}: give a folder outside it, since the runs write about "
            "0.5 GB of GeoTIFFs that must not be committed."
        )
    dest = dest.expanduser().resolve()
    if dest == work or work in dest.parents:
        raise PrecomputeError(f"--dest {dest} is inside --work {work}: give the repository's demo/precomputed.")
    secrets = [os.environ[k] for k in os.environ if k.startswith("AWS_") and len(os.environ[k]) >= 8]
    env, removed = anonymous_env()
    if removed:
        logger.info("Removed %s from the runs' environment: every read is anonymous.", ", ".join(removed))
    cli = find_cli()
    (work / "logs").mkdir(parents=True, exist_ok=True)
    (work / "out").mkdir(exist_ok=True)
    ensure_outline(work)
    ensure_dnbr(work, dnbr_from, env, work / "logs")

    records: list[RunRecord] = []
    for demo in DEMOS:
        records.append(run_demo(demo, cli=cli, work=work, snapshot=snapshot, env=env))

    dest.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for demo, record in zip(DEMOS, records, strict=True):
        files, written = copy_outputs(demo, work, dest)
        record.copied = written
        entries.append(index_entry(demo, work=work, snapshot=snapshot, files=files))
    index = build_index(entries, work=work, snapshot=snapshot)
    text = json.dumps(index, indent=2, ensure_ascii=False) + "\n"
    wrote = write_if_changed(dest / INDEX_FILE, text)
    logger.info("%s %s.", "Wrote" if wrote else "Unchanged:", dest / INDEX_FILE)

    local_paths = sorted({str(work), str(REPO), str(Path.home())} - {"/"})
    problems = check_tree(dest, secrets, local_paths)
    if problems:
        raise PrecomputeError(
            f"{dest} holds what must not be published; fix the outputs or the index text:\n" + "\n".join(problems)
        )
    n_files, n_bytes = tree_size(dest)
    logger.info(
        "%s: %d files, %s bytes; no absolute local path, AWS credential or dash.", dest, n_files, f"{n_bytes:,}"
    )
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with (work / "logs" / "precompute_runs.jsonl").open("a", encoding="utf-8") as fh:
        for r in records:
            row = {
                "utc": stamp,
                "name": r.name,
                "command": r.command,
                "skipped": r.skipped,
                "elapsed_s": r.elapsed_s,
                "report_bytes_read": r.report_bytes_read,
                "report_wall_s": r.report_wall_s,
                "copied": r.copied,
            }
            fh.write(json.dumps(row) + "\n")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the arguments and precompute; return the exit code (2 when a step cannot go on)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--work", type=Path, required=True, help="Work directory outside the repository, for the runs' outputs."
    )
    ap.add_argument("--snapshot", default=SNAPSHOT, help=f"Icechunk snapshot to pin (default {SNAPSHOT}).")
    ap.add_argument("--dnbr", type=Path, default=None, help=f"An existing {DNBR_FILE} to copy instead of building one.")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Where the page's files go (demo/precomputed).")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.perf_counter()
    try:
        records = precompute(args.work, dest=args.dest, snapshot=args.snapshot, dnbr_from=args.dnbr)
    except PrecomputeError as exc:
        logger.error("Error: %s", exc)
        return 2
    total = sum(r.report_bytes_read for r in records if not r.skipped)
    logger.info(
        "Done in %.0f s: %d demos ran, %d were current; this invocation read %s chunk bytes.",
        time.perf_counter() - t0,
        sum(not r.skipped for r in records),
        sum(r.skipped for r in records),
        f"{total:,}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
