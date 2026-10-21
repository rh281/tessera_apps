"""Check the built wheel and sdist before they are published.

Usage:
    python scripts/check_dist.py DIST_DIR --version 0.1.0 [--list]

Opens the one wheel and the one sdist in ``DIST_DIR`` and exits 1, naming every problem, unless:

- both carry the version given, in their file names, their metadata and ``__version__``;
- the wheel holds the package with its six demo GeoJSON files and ``py.typed``, and the licence in
  its dist-info folder, and nothing else: no tests, notebooks, scripts or ``__pycache__``;
- the sdist holds ``src``, ``tests`` (with the fixtures), ``scripts``, ``docs``, the README, the
  licence, the changelog, ``pyproject.toml``, the lock and the lint configuration, and none of the
  notebooks, their outputs, ``.github`` or ``.claude``;
- every file the source tree would ship (``git ls-files`` plus untracked files git does not ignore)
  is there: the package files in both, the ``src``, ``tests``, ``scripts`` and ``docs`` files in the
  sdist;
- the metadata says ``Requires-Python: >=3.12`` and ``License-Expression: Apache-2.0``, has no
  licence classifier, lists every runtime dependency of ``pyproject.toml`` (and no other), and the
  wheel declares the ``tessera-apps`` console script;
- the long description (the README) is Markdown with no relative link, which would break on PyPI.

Only the standard library is used, so it runs with any Python 3.12 or later.
"""

from __future__ import annotations

import argparse
import configparser
import email.message
import email.parser
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: The distribution name as it appears in the file names and the metadata.
DIST_NAME = "tessera_apps"
#: The package directory inside the wheel (and under ``src/`` in the sdist).
PACKAGE = "tessera_apps"
#: The console script and its entry point.
CONSOLE_SCRIPT = ("tessera-apps", "tessera_apps.cli:app")
REQUIRES_PYTHON = ">=3.12"
LICENSE_EXPRESSION = "Apache-2.0"
LICENSE_FILE = "LICENSE"
#: The bundled demo AOIs and seeds, in ``tessera_apps/data/``.
DEMO_GEOJSON = ("caldor", "rondonia", "everglades", "everglades_seeds", "itapebi", "jaru")
#: Files the wheel must hold under the package directory, whatever the source tree says.
WHEEL_PACKAGE_REQUIRED = (
    "__init__.py",
    "py.typed",
    "cli.py",
    "io.py",
    "cache.py",
    "metrics.py",
    "validate.py",
    "export.py",
    "demo.py",
    "apps/__init__.py",
    "apps/_common.py",
    "apps/wildfire.py",
    "apps/deforest.py",
    "apps/wetland.py",
    "apps/restore.py",
    "data/__init__.py",
    *(f"data/{name}.geojson" for name in DEMO_GEOJSON),
)
#: Files at the top of the sdist that must be there.
SDIST_TOP_REQUIRED = (
    "PKG-INFO",
    "pyproject.toml",
    "README.md",
    LICENSE_FILE,
    "CHANGELOG.md",
    "requirements-lock.txt",
    "ruff.toml",
    "mypy.ini",
)
#: Directories at the top of the sdist that must be there.
SDIST_DIRS = ("src", "tests", "scripts", "docs")
#: Everything allowed at the top of the sdist. Hatchling always adds ``.gitignore``.
SDIST_TOP_ALLOWED = frozenset({*SDIST_TOP_REQUIRED, *SDIST_DIRS, ".gitignore"})
#: Files deeper in the sdist that must be there, whatever the source tree says.
SDIST_REQUIRED = (
    *(f"src/{PACKAGE}/{rel}" for rel in WHEEL_PACKAGE_REQUIRED),
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/fixtures/caldor_64.npz",
    "tests/fixtures/caldor_64_hansen.npz",
    "scripts/check_dist.py",
    "docs/checkpoints.md",
    "docs/releasing.md",
)
#: Path parts that must never appear in either archive.
FORBIDDEN_PARTS = frozenset(
    {
        "__pycache__",
        "notebooks",
        ".github",
        ".claude",
        ".git",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".ipynb_checkpoints",
    }
)
#: File suffixes that must never appear in either archive.
FORBIDDEN_SUFFIXES = (".pyc", ".pyo", ".ipynb")
#: Lowest metadata version with ``License-Expression`` (PEP 639).
MIN_METADATA_VERSION = (2, 4)
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+)*((a|b|rc)[0-9]+)?(\.post[0-9]+)?(\.dev[0-9]+)?$")
_DUNDER_VERSION_RE = re.compile(r"^__version__\s*=\s*[\"']([^\"']+)[\"']", re.MULTILINE)
_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*([^;]*?)\s*(?:;\s*(.*?))?\s*$")
_LINK_RE = re.compile(r"\]\(\s*<?([^)\s>]+)")
_REF_LINK_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)
_HTML_LINK_RE = re.compile(r"""<(?:a|img)\b[^>]*?\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_ABSOLUTE_LINK = ("http://", "https://", "mailto:", "#")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,}).*?^ {0,3}\1[ \t]*$", re.MULTILINE | re.DOTALL)
_CODE_SPAN_RE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)", re.DOTALL)


class CheckError(Exception):
    """The check cannot start: no dist folder, or not exactly one wheel and one sdist in it."""


@dataclass(frozen=True)
class Expected:
    """What the two archives must say and hold.

    Attributes:
        version: The package version, without a leading ``v``.
        dependencies: The runtime dependencies of ``pyproject.toml``, as written there.
        extras: The optional dependency groups of ``pyproject.toml``.
        source_files: Repository-relative POSIX paths of the files the source tree would ship, or
            None to skip that comparison.
    """

    version: str
    dependencies: tuple[str, ...]
    extras: tuple[str, ...] = ()
    source_files: frozenset[str] | None = None


def normalize_name(name: str) -> str:
    """The PEP 503 normalised form of a project name: lower case, runs of ``-_.`` as one ``-``.

    Args:
        name: A project or extra name.

    Returns:
        The normalised name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def normalize_requirement(text: str) -> str:
    """A requirement in one canonical spelling, so two spellings of the same one compare equal.

    The name is normalised, extras and version specifiers are sorted, spaces dropped and the marker's
    quotes made single, so ``icechunk>=2.2,<3`` and ``icechunk<3,>=2.2`` agree.

    Args:
        text: A PEP 508 requirement, from ``pyproject.toml`` or a ``Requires-Dist`` header.

    Returns:
        The canonical spelling.

    Raises:
        ValueError: When ``text`` is not a requirement this function can read.
    """
    match = _REQUIREMENT_RE.match(text)
    if match is None:
        raise ValueError(f"cannot read the requirement {text!r}")
    name, extras, specs, marker = match.groups()
    extra_names = sorted(normalize_name(e.strip()) for e in (extras or "").strip("[]").split(",") if e.strip())
    spec_parts = sorted(s.replace(" ", "") for s in specs.strip("()").split(",") if s.strip())
    out = normalize_name(name)
    if extra_names:
        out += "[" + ",".join(extra_names) + "]"
    out += ",".join(spec_parts)
    if marker:
        out += ";" + re.sub(r"\s+", " ", marker.replace('"', "'")).strip()
    return out


def read_pyproject(path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The runtime dependencies and the extras' names of a ``pyproject.toml``.

    Args:
        path: The file.

    Returns:
        ``(dependencies, extras)``.

    Raises:
        CheckError: When the file is missing or has no ``[project]`` table.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CheckError(f"cannot read {path}: {exc}") from exc
    project = data.get("project")
    if not isinstance(project, dict):
        raise CheckError(f"{path} has no [project] table")
    deps = tuple(str(d) for d in project.get("dependencies", []))
    extras = tuple(str(e) for e in (project.get("optional-dependencies") or {}))
    return deps, extras


def source_files(root: Path) -> frozenset[str]:
    """The files under ``src``, ``tests``, ``scripts`` and ``docs`` that a build of ``root`` ships.

    In a git work tree that is ``git ls-files --cached --others --exclude-standard`` (tracked files
    and untracked ones git does not ignore, which is what hatchling packs), minus deleted files.
    Elsewhere the directories are walked, skipping :data:`FORBIDDEN_PARTS` and
    :data:`FORBIDDEN_SUFFIXES`.

    Args:
        root: The repository root.

    Returns:
        Repository-relative POSIX paths.

    Raises:
        CheckError: When ``root`` holds none of the directories.
    """
    if not any((root / d).is_dir() for d in SDIST_DIRS):
        raise CheckError(f"{root} is not a source tree: it has none of {', '.join(SDIST_DIRS)}")
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SDIST_DIRS],
            capture_output=True,
            check=True,
        )
        listed: Iterable[str] = (p for p in done.stdout.decode("utf-8").split("\0") if p)
    except (OSError, subprocess.CalledProcessError):
        listed = (
            p.relative_to(root).as_posix() for d in SDIST_DIRS if (root / d).is_dir() for p in (root / d).rglob("*")
        )
    return frozenset(p for p in listed if (root / p).is_file() and not forbidden(p))


def forbidden(path: str) -> bool:
    """Whether a member path is one that neither archive may hold.

    Args:
        path: A POSIX path inside an archive.

    Returns:
        True for a path with a part in :data:`FORBIDDEN_PARTS` or a suffix in :data:`FORBIDDEN_SUFFIXES`.
    """
    pure = PurePosixPath(path)
    return bool(FORBIDDEN_PARTS.intersection(pure.parts)) or pure.name.endswith(FORBIDDEN_SUFFIXES)


def find_dists(dist_dir: Path) -> tuple[Path, Path]:
    """The one wheel and the one sdist in ``dist_dir``.

    Args:
        dist_dir: The build's output folder.

    Returns:
        ``(wheel, sdist)``.

    Raises:
        CheckError: When the folder is missing or does not hold exactly one of each.
    """
    if not dist_dir.is_dir():
        raise CheckError(f"{dist_dir} is not a folder; build first, for example `uv build -o {dist_dir}`")
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    problems = []
    for kind, found in (("wheel (*.whl)", wheels), ("sdist (*.tar.gz)", sdists)):
        if len(found) != 1:
            names = ", ".join(p.name for p in found) or "none"
            problems.append(f"expected one {kind} in {dist_dir}, found {len(found)}: {names}")
    if problems:
        raise CheckError("; ".join(problems))
    return wheels[0], sdists[0]


def parse_metadata(text: str) -> email.message.Message:
    """Read a ``METADATA`` or ``PKG-INFO`` file.

    Args:
        text: The file's text.

    Returns:
        The headers, with the long description as the payload.
    """
    return email.parser.Parser().parsestr(text)


def relative_links(markdown: str) -> list[str]:
    """Links in a Markdown text that are not absolute (http, https, mailto) or in-page anchors.

    Fenced code blocks and code spans are left out, since nothing in them is a link.

    Args:
        markdown: The text.

    Returns:
        Each relative link target once, in order of first appearance.
    """
    markdown = _CODE_SPAN_RE.sub("", _FENCE_RE.sub("", markdown))
    found = [*_LINK_RE.findall(markdown), *_REF_LINK_RE.findall(markdown), *_HTML_LINK_RE.findall(markdown)]
    return list(dict.fromkeys(link for link in found if not link.startswith(_ABSOLUTE_LINK)))


def check_metadata(meta: email.message.Message, expected: Expected, where: str) -> list[str]:
    """The problems in a wheel's ``METADATA`` or an sdist's ``PKG-INFO``.

    Args:
        meta: The parsed file.
        expected: What it must say.
        where: The file's name in the messages.

    Returns:
        One message per problem.
    """
    problems = []
    name = meta.get("Name")
    if name is None or normalize_name(name) != normalize_name(DIST_NAME):
        problems.append(f"{where}: Name is {name!r}, expected {DIST_NAME!r}")
    if meta.get("Version") != expected.version:
        problems.append(f"{where}: Version is {meta.get('Version')!r}, expected {expected.version!r}")
    meta_version = meta.get("Metadata-Version") or ""
    try:
        parsed = tuple(int(part) for part in meta_version.split("."))
    except ValueError:
        parsed = ()
    if parsed < MIN_METADATA_VERSION:
        problems.append(
            f"{where}: Metadata-Version is {meta_version!r}; License-Expression needs 2.4 or later (hatchling>=1.27)"
        )
    if meta.get("Requires-Python") != REQUIRES_PYTHON:
        problems.append(f"{where}: Requires-Python is {meta.get('Requires-Python')!r}, expected {REQUIRES_PYTHON!r}")
    if meta.get("License-Expression") != LICENSE_EXPRESSION:
        problems.append(
            f"{where}: License-Expression is {meta.get('License-Expression')!r}, expected {LICENSE_EXPRESSION!r}"
        )
    if LICENSE_FILE not in (meta.get_all("License-File") or []):
        problems.append(f"{where}: no License-File: {LICENSE_FILE} header")
    if meta.get("License") is not None:
        problems.append(f"{where}: has a License header as well as License-Expression (PEP 639 forbids both)")
    licence_classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if licence_classifiers:
        problems.append(f"{where}: licence classifier {licence_classifiers} alongside License-Expression")
    runtime = []
    for req in meta.get_all("Requires-Dist") or []:
        if "extra ==" in req.replace('"', "'"):
            continue
        try:
            runtime.append(normalize_requirement(req))
        except ValueError as exc:
            problems.append(f"{where}: {exc}")
    want = set()
    for dep in expected.dependencies:
        try:
            want.add(normalize_requirement(dep))
        except ValueError as exc:
            problems.append(f"pyproject.toml: {exc}")
    for dep in sorted(want - set(runtime)):
        problems.append(f"{where}: runtime dependency {dep} of pyproject.toml is not a Requires-Dist")
    for dep in sorted(set(runtime) - want):
        problems.append(f"{where}: Requires-Dist {dep} is not a runtime dependency in pyproject.toml")
    provided = {normalize_name(e) for e in meta.get_all("Provides-Extra") or []}
    for extra in sorted({normalize_name(e) for e in expected.extras} - provided):
        problems.append(f"{where}: extra {extra!r} of pyproject.toml is not a Provides-Extra")
    content_type = meta.get("Description-Content-Type") or ""
    if not content_type.startswith("text/markdown"):
        problems.append(f"{where}: Description-Content-Type is {content_type!r}, expected text/markdown")
    payload = meta.get_payload()
    description = payload if isinstance(payload, str) else ""
    if not description.strip():
        problems.append(f"{where}: the long description (the README) is empty")
    for link in relative_links(description):
        problems.append(f"{where}: the long description has a relative link {link!r}, which breaks on PyPI")
    return problems


def _package_version(text: str, where: str, expected: str) -> list[str]:
    match = _DUNDER_VERSION_RE.search(text)
    if match is None:
        return [f"{where}: no __version__ assignment"]
    if match.group(1) != expected:
        return [f"{where}: __version__ is {match.group(1)!r}, expected {expected!r}"]
    return []


def _missing(present: Iterable[str], required: Iterable[str], where: str, why: str = "") -> list[str]:
    have = set(present)
    return [f"{where}: {path} is missing{why}" for path in sorted(set(required) - have)]


def check_wheel(path: Path, expected: Expected) -> list[str]:
    """The problems in a wheel.

    Args:
        path: The ``.whl`` file.
        expected: What it must say and hold.

    Returns:
        One message per problem.
    """
    want_name = f"{DIST_NAME}-{expected.version}-py3-none-any.whl"
    problems = [] if path.name == want_name else [f"wheel file name is {path.name}, expected {want_name}"]
    try:
        with zipfile.ZipFile(path) as zf:
            return problems + _check_wheel_members(zf, expected)
    except (OSError, zipfile.BadZipFile) as exc:
        return [*problems, f"{path.name} is not a readable wheel: {exc}"]


def _check_wheel_members(zf: zipfile.ZipFile, expected: Expected) -> list[str]:
    problems = []
    names = [n for n in zf.namelist() if not n.endswith("/")]
    info_dirs = {n.split("/", 1)[0] for n in names if n.split("/", 1)[0].endswith(".dist-info")}
    info = f"{DIST_NAME}-{expected.version}.dist-info"
    if info_dirs != {info}:
        problems.append(f"wheel: dist-info folders are {sorted(info_dirs)}, expected [{info!r}]")
        if len(info_dirs) == 1:
            # Read the one that is there, so its contents are still checked.
            info = next(iter(info_dirs))
    required = [f"{PACKAGE}/{rel}" for rel in WHEEL_PACKAGE_REQUIRED]
    required += [f"{info}/{n}" for n in ("METADATA", "WHEEL", "RECORD", "entry_points.txt")]
    required.append(f"{info}/licenses/{LICENSE_FILE}")
    problems += _missing(names, required, "wheel")
    if expected.source_files is not None:
        src_prefix = f"src/{PACKAGE}/"
        from_source = [p[len("src/") :] for p in expected.source_files if p.startswith(src_prefix)]
        problems += _missing(names, set(from_source) - set(required), "wheel", " (it is in the source tree)")
        in_wheel = {n for n in names if n.startswith(f"{PACKAGE}/")}
        for extra in sorted(in_wheel - set(from_source)):
            problems.append(f"wheel: {extra} is not in the source tree")
    for name in names:
        top = name.split("/", 1)[0]
        pure = PurePosixPath(name)
        test_like = "tests" in pure.parts or pure.name == "conftest.py" or pure.name.startswith("test_")
        if top not in (PACKAGE, info) and not top.endswith(".dist-info"):
            problems.append(f"wheel: {name} is outside the package and its dist-info folder")
        elif forbidden(name) or test_like:
            problems.append(f"wheel: {name} must not ship")
    members = set(names)
    if f"{info}/METADATA" in members:
        meta = parse_metadata(zf.read(f"{info}/METADATA").decode("utf-8"))
        problems += check_metadata(meta, expected, f"wheel {info}/METADATA")
    if f"{info}/WHEEL" in members:
        wheel_meta = parse_metadata(zf.read(f"{info}/WHEEL").decode("utf-8"))
        if (wheel_meta.get("Root-Is-Purelib") or "").lower() != "true":
            problems.append("wheel: WHEEL does not say Root-Is-Purelib: true")
        if "py3-none-any" not in (wheel_meta.get_all("Tag") or []):
            problems.append(f"wheel: WHEEL tags are {wheel_meta.get_all('Tag')}, expected py3-none-any")
    if f"{info}/entry_points.txt" in members:
        problems += _check_entry_points(zf.read(f"{info}/entry_points.txt").decode("utf-8"))
    init = f"{PACKAGE}/__init__.py"
    if init in members:
        problems += _package_version(zf.read(init).decode("utf-8"), f"wheel {init}", expected.version)
    return problems


class _CaseKeepingParser(configparser.ConfigParser):
    """A ConfigParser that keeps the case of keys, as entry point names are case sensitive."""

    def optionxform(self, optionstr: str) -> str:
        """Keep the key as written.

        Args:
            optionstr: The key.

        Returns:
            The key, unchanged.
        """
        return optionstr


def _check_entry_points(text: str) -> list[str]:
    parser = _CaseKeepingParser(delimiters=("=",), interpolation=None)
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        return [f"wheel entry_points.txt cannot be read: {exc}"]
    script, target = CONSOLE_SCRIPT
    got = parser.get("console_scripts", script, fallback=None)
    if got is None or got.strip() != target:
        return [f"wheel entry_points.txt: console script {script} is {got!r}, expected {target!r}"]
    return []


def _check_sdist_pyproject(text: str, expected: Expected) -> list[str]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return [f"sdist pyproject.toml cannot be read: {exc}"]
    project = data.get("project") or {}
    problems = []
    script, target = CONSOLE_SCRIPT
    if (project.get("scripts") or {}).get(script) != target:
        problems.append(f"sdist pyproject.toml: [project.scripts] {script} is not {target!r}")
    try:
        got = sorted(normalize_requirement(str(d)) for d in project.get("dependencies", []))
        want = sorted(normalize_requirement(d) for d in expected.dependencies)
    except ValueError as exc:
        return [*problems, f"sdist pyproject.toml: {exc}"]
    if got != want:
        problems.append("sdist pyproject.toml: its dependencies differ from the pyproject.toml checked against")
    return problems


def check_sdist(path: Path, expected: Expected) -> list[str]:
    """The problems in an sdist.

    Args:
        path: The ``.tar.gz`` file.
        expected: What it must say and hold.

    Returns:
        One message per problem.
    """
    root = f"{DIST_NAME}-{expected.version}"
    want_name = f"{root}.tar.gz"
    problems = [] if path.name == want_name else [f"sdist file name is {path.name}, expected {want_name}"]
    try:
        with tarfile.open(path, "r:gz") as tf:
            return problems + _check_sdist_members(tf, expected, root)
    except (OSError, tarfile.TarError) as exc:
        return [*problems, f"{path.name} is not a readable sdist: {exc}"]


def _check_sdist_members(tf: tarfile.TarFile, expected: Expected, root: str) -> list[str]:
    problems = []
    members = {}
    tops = {PurePosixPath(m.name).parts[0] for m in tf.getmembers() if PurePosixPath(m.name).parts}
    if tops != {root}:
        problems.append(f"sdist: top folders are {sorted(tops)}, expected [{root!r}]")
        if len(tops) == 1:
            # Read the one that is there, so its contents are still checked.
            root = next(iter(tops))
    for m in tf.getmembers():
        pure = PurePosixPath(m.name)
        if pure.is_absolute() or ".." in pure.parts:
            problems.append(f"sdist: unsafe member path {m.name}")
        elif not (m.isfile() or m.isdir()):
            problems.append(f"sdist: {m.name} is not a regular file or folder")
        elif pure.parts[0] != root:
            problems.append(f"sdist: {m.name} is outside the {root}/ folder")
        elif m.isfile() and len(pure.parts) > 1:
            members[PurePosixPath(*pure.parts[1:]).as_posix()] = m
    names = sorted(members)
    problems += _missing(names, [*SDIST_TOP_REQUIRED, *SDIST_REQUIRED], "sdist")
    for d in SDIST_DIRS:
        if not any(n.startswith(f"{d}/") for n in names):
            problems.append(f"sdist: no {d}/ folder")
    if expected.source_files is not None:
        also = expected.source_files - {*SDIST_TOP_REQUIRED, *SDIST_REQUIRED}
        problems += _missing(names, also, "sdist", " (it is in the source tree)")
    for name in names:
        top = name.split("/", 1)[0]
        if top not in SDIST_TOP_ALLOWED:
            problems.append(f"sdist: {name} must not ship (only {', '.join(sorted(SDIST_TOP_ALLOWED))})")
        elif forbidden(name):
            problems.append(f"sdist: {name} must not ship")

    def read(name: str) -> str | None:
        member = members.get(name)
        fh = tf.extractfile(member) if member is not None else None
        return fh.read().decode("utf-8") if fh is not None else None

    pkg_info = read("PKG-INFO")
    if pkg_info is not None:
        problems += check_metadata(parse_metadata(pkg_info), expected, "sdist PKG-INFO")
    pyproject = read("pyproject.toml")
    if pyproject is not None:
        problems += _check_sdist_pyproject(pyproject, expected)
    init = f"src/{PACKAGE}/__init__.py"
    init_text = read(init)
    if init_text is not None:
        problems += _package_version(init_text, f"sdist {init}", expected.version)
    return problems


def check_dist(dist_dir: Path, expected: Expected) -> list[str]:
    """The problems in the wheel and the sdist in ``dist_dir``.

    Args:
        dist_dir: The build's output folder.
        expected: What both must say and hold.

    Returns:
        One message per problem; empty when both pass.

    Raises:
        CheckError: When the folder does not hold exactly one wheel and one sdist.
    """
    wheel, sdist = find_dists(dist_dir)
    return [*check_wheel(wheel, expected), *check_sdist(sdist, expected)]


def list_members(dist_dir: Path) -> Mapping[str, list[str]]:
    """The file names in the wheel and the sdist, for the build log.

    Args:
        dist_dir: The build's output folder.

    Returns:
        ``{archive name: [member, ...]}``.
    """
    wheel, sdist = find_dists(dist_dir)
    with zipfile.ZipFile(wheel) as zf:
        wheel_names = [n for n in zf.namelist() if not n.endswith("/")]
    with tarfile.open(sdist, "r:gz") as tf:
        sdist_names = [m.name for m in tf.getmembers() if m.isfile()]
    return {wheel.name: wheel_names, sdist.name: sdist_names}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the checks and print the result.

    Args:
        argv: The arguments, without the program name (``sys.argv[1:]`` when None).

    Returns:
        0 when both archives pass, 1 when any problem is found or the check cannot start.
    """
    repo = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dist_dir", type=Path, help="The folder that holds the wheel and the sdist.")
    ap.add_argument("--version", required=True, help="The version both must carry, for example 0.1.0 (no 'v').")
    ap.add_argument(
        "--pyproject", type=Path, default=repo / "pyproject.toml", help="The pyproject.toml the dependencies come from."
    )
    ap.add_argument(
        "--source-root",
        type=Path,
        default=repo,
        help="The source tree whose files both must hold (default: this repo).",
    )
    ap.add_argument("--no-source-check", action="store_true", help="Skip the comparison with the source tree.")
    ap.add_argument("--list", action="store_true", help="Also print every file in both archives.")
    args = ap.parse_args(argv)
    if not _VERSION_RE.match(args.version):
        print(
            f"check_dist: --version {args.version!r} is not a version like 0.1.0 (give it without a 'v')",
            file=sys.stderr,
        )
        return 1
    try:
        deps, extras = read_pyproject(args.pyproject)
        files = None if args.no_source_check else source_files(args.source_root)
        expected = Expected(version=args.version, dependencies=deps, extras=extras, source_files=files)
        problems = check_dist(args.dist_dir, expected)
    except CheckError as exc:
        print(f"check_dist: cannot check: {exc}", file=sys.stderr)
        return 1
    if args.list:
        # After the checks, so an archive that cannot be opened is named in the problems below
        # (check_wheel and check_sdist report it) instead of ending the run in a traceback here.
        try:
            members = list_members(args.dist_dir)
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
            print(f"check_dist: cannot list the archives: {exc}", file=sys.stderr)
        else:
            for archive, names in members.items():
                print(f"{archive} ({len(names)} files):")
                for name in names:
                    print(f"  {name}")
    wheel, sdist = find_dists(args.dist_dir)
    if problems:
        print(f"check_dist: {len(problems)} problem(s) in {wheel.name} and {sdist.name}:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    source = "skipped" if files is None else f"{len(files)} files"
    print(f"check_dist: OK: {wheel.name} and {sdist.name}, version {args.version}; source tree comparison {source}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
