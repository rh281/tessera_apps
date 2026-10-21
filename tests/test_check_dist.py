"""``scripts/check_dist.py`` on small synthetic wheels and sdists, offline.

A good wheel and sdist are written from the checker's own lists, then each test breaks one thing and
expects the checker to name it. The last tests tie the checker's lists to this repository, so a new
module, demo AOI or packaging change cannot leave it behind.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

from scripts import check_dist as cd

VERSION = "1.2.3"
DEPS = ("icechunk>=2.2,<3", "numcodecs[pcodec]>=0.16", "numpy>=2.0")
EXTRAS = ("validate", "dev")
INFO = f"tessera_apps-{VERSION}.dist-info"
ROOT = f"tessera_apps-{VERSION}"
REPO = Path(__file__).resolve().parent.parent
README = (
    "# tessera_apps\n\nSee [notebook 01](https://github.com/rh281/tessera_apps/blob/main/notebooks/01_wildfire.ipynb)"
    " and [the CLI](#command-line).\n\n```bash\nls [a](b)\n```\n"
)


def metadata(**changes: list[str] | None) -> str:
    """A METADATA or PKG-INFO text; a header given as None is left out, a list replaces its values."""
    headers: dict[str, list[str] | None] = {
        "Metadata-Version": ["2.4"],
        "Name": ["tessera_apps"],
        "Version": [VERSION],
        "Requires-Python": [">=3.12"],
        "License-Expression": ["Apache-2.0"],
        "License-File": ["LICENSE"],
        "Classifier": ["Programming Language :: Python :: 3.12", "Typing :: Typed"],
        # Hatchling sorts the specifiers: icechunk<3,>=2.2 for icechunk>=2.2,<3.
        "Requires-Dist": ["icechunk<3,>=2.2", "numcodecs[pcodec]>=0.16", "numpy>=2.0", "pytest>=8; extra == 'dev'"],
        "Provides-Extra": ["dev", "validate"],
        "Description-Content-Type": ["text/markdown"],
    }
    description = changes.pop("description", None)
    headers.update(changes)
    lines = [f"{key}: {value}" for key, values in headers.items() if values is not None for value in values]
    return "\n".join(lines) + "\n\n" + (description[0] if description else README)


def wheel_files(version: str = VERSION) -> dict[str, str]:
    info = f"tessera_apps-{version}.dist-info"
    files = {f"tessera_apps/{rel}": "" for rel in cd.WHEEL_PACKAGE_REQUIRED}
    files["tessera_apps/__init__.py"] = f'__version__ = "{version}"\n'
    files[f"{info}/METADATA"] = metadata(Version=[version])
    files[f"{info}/WHEEL"] = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    files[f"{info}/RECORD"] = ""
    files[f"{info}/entry_points.txt"] = "[console_scripts]\ntessera-apps = tessera_apps.cli:app\n"
    files[f"{info}/licenses/LICENSE"] = "Apache License\n"
    return files


def pyproject_text(deps: tuple[str, ...] = DEPS, script: str = "tessera_apps.cli:app") -> str:
    dep_lines = "".join(f'    "{d}",\n' for d in deps)
    return (
        '[project]\nname = "tessera_apps"\ndynamic = ["version"]\ndependencies = [\n'
        + dep_lines
        + ']\n\n[project.optional-dependencies]\nvalidate = []\ndev = ["pytest>=8"]\n\n'
        + f'[project.scripts]\ntessera-apps = "{script}"\n'
    )


def sdist_files() -> dict[str, str]:
    files = dict.fromkeys(cd.SDIST_TOP_REQUIRED, "")
    files.update(dict.fromkeys(cd.SDIST_REQUIRED, ""))
    files["PKG-INFO"] = metadata()
    files["pyproject.toml"] = pyproject_text()
    files["README.md"] = README
    files[".gitignore"] = "__pycache__/\n"
    files["src/tessera_apps/__init__.py"] = f'__version__ = "{VERSION}"\n'
    return files


def write_wheel(dist: Path, files: dict[str, str], name: str = f"tessera_apps-{VERSION}-py3-none-any.whl") -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / name
    with zipfile.ZipFile(path, "w") as zf:
        for member, text in files.items():
            zf.writestr(member, text)
    return path


def write_sdist(
    dist: Path, files: dict[str, str], name: str = f"{ROOT}.tar.gz", root: str = ROOT, symlink: str | None = None
) -> Path:
    dist.mkdir(parents=True, exist_ok=True)
    path = dist / name
    with tarfile.open(path, "w:gz") as tf:
        for member, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(member if member.startswith("/") or member.startswith("..") else f"{root}/{member}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        if symlink is not None:
            link = tarfile.TarInfo(f"{root}/{symlink}")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            tf.addfile(link)
    return path


def expected(**kwargs: object) -> cd.Expected:
    base: dict[str, object] = {"version": VERSION, "dependencies": DEPS, "extras": EXTRAS}
    base.update(kwargs)
    return cd.Expected(**base)  # type: ignore[arg-type]


@pytest.fixture
def dist(tmp_path):
    """A dist folder with a good wheel and sdist, and an empty file uv writes there."""
    folder = tmp_path / "dist"
    write_wheel(folder, wheel_files())
    write_sdist(folder, sdist_files())
    (folder / ".gitignore").write_text("*\n")
    return folder


def problems_with(tmp_path, *, wheel=None, sdist=None, exp=None, **sdist_kwargs) -> list[str]:
    folder = tmp_path / "broken"
    write_wheel(folder, wheel if wheel is not None else wheel_files())
    write_sdist(folder, sdist if sdist is not None else sdist_files(), **sdist_kwargs)
    return cd.check_dist(folder, exp or expected())


def assert_named(problems: list[str], *fragments: str) -> None:
    for fragment in fragments:
        assert any(fragment in p for p in problems), (fragment, problems)


# --------------------------------------------------------------------------------------------
# the good case, and the command line
# --------------------------------------------------------------------------------------------


def test_good_archives_pass(dist):
    assert cd.check_dist(dist, expected()) == []


def test_main_passes_and_fails_with_messages(dist, tmp_path, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject_text())
    args = [str(dist), "--version", VERSION, "--pyproject", str(pyproject), "--no-source-check"]
    assert cd.main(args) == 0
    assert "check_dist: OK" in capsys.readouterr().out

    pyproject.write_text(pyproject_text((*DEPS, "rich>=13")))
    assert cd.main(args) == 1
    err = capsys.readouterr().err
    assert "problem(s)" in err
    assert "runtime dependency rich>=13 of pyproject.toml is not a Requires-Dist" in err


def test_main_lists_members(dist, tmp_path, capsys):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject_text())
    args = [str(dist), "--version", VERSION, "--pyproject", str(pyproject), "--no-source-check", "--list"]
    assert cd.main(args) == 0
    out = capsys.readouterr().out
    assert f"{ROOT}/tests/fixtures/caldor_64.npz" in out
    assert "tessera_apps/data/jaru.geojson" in out


@pytest.mark.parametrize(
    ("suffix", "named"),
    [("-py3-none-any.whl", "is not a readable wheel"), (".tar.gz", "is not a readable sdist")],
)
def test_main_lists_and_names_an_unreadable_archive(dist, tmp_path, suffix, named, capsys):
    # release.yml passes --list, so a corrupt archive must be named there too, not end in a traceback.
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(pyproject_text())
    (dist / f"{ROOT}{suffix}").write_bytes(b"not a zip")
    args = [str(dist), "--version", VERSION, "--pyproject", str(pyproject), "--no-source-check", "--list"]
    assert cd.main(args) == 1
    err = capsys.readouterr().err
    assert "check_dist: cannot list the archives" in err
    assert "problem(s)" in err
    assert named in err


@pytest.mark.parametrize("version", ["v1.2.3", "1.2.3 ", "latest", ""])
def test_main_refuses_a_malformed_version(dist, version, capsys):
    assert cd.main([str(dist), "--version", version, "--no-source-check"]) == 1
    assert "is not a version like" in capsys.readouterr().err


def test_main_fails_loudly_without_a_dist_folder(tmp_path, capsys):
    assert cd.main([str(tmp_path / "nope"), "--version", VERSION, "--no-source-check"]) == 1
    assert "is not a folder; build first" in capsys.readouterr().err


def test_find_dists_needs_exactly_one_of_each(dist, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(cd.CheckError, match=r"one wheel .* found 0: none.*one sdist .* found 0"):
        cd.find_dists(empty)
    shutil.copy(next(dist.glob("*.whl")), dist / "tessera_apps-1.2.2-py3-none-any.whl")
    with pytest.raises(cd.CheckError, match="found 2"):
        cd.find_dists(dist)


def test_main_fails_on_a_pyproject_without_project(dist, tmp_path, capsys):
    bad = tmp_path / "pyproject.toml"
    bad.write_text("[tool.x]\n")
    assert cd.main([str(dist), "--version", VERSION, "--pyproject", str(bad), "--no-source-check"]) == 1
    assert "has no [project] table" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# the wheel
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("drop", "fragment"),
    [
        ("tessera_apps/data/jaru.geojson", "wheel: tessera_apps/data/jaru.geojson is missing"),
        ("tessera_apps/data/everglades_seeds.geojson", "everglades_seeds.geojson is missing"),
        ("tessera_apps/py.typed", "wheel: tessera_apps/py.typed is missing"),
        ("tessera_apps/apps/restore.py", "wheel: tessera_apps/apps/restore.py is missing"),
        (f"{INFO}/licenses/LICENSE", f"wheel: {INFO}/licenses/LICENSE is missing"),
        (f"{INFO}/entry_points.txt", f"wheel: {INFO}/entry_points.txt is missing"),
    ],
)
def test_wheel_missing_file(tmp_path, drop, fragment):
    files = wheel_files()
    del files[drop]
    assert_named(problems_with(tmp_path, wheel=files), fragment)


@pytest.mark.parametrize(
    ("member", "fragment"),
    [
        ("tests/test_io.py", "is outside the package"),
        ("notebooks/01_wildfire.ipynb", "is outside the package"),
        ("scripts/make_fixture.py", "is outside the package"),
        ("tessera_apps/__pycache__/cli.cpython-312.pyc", "must not ship"),
        ("tessera_apps/cli.pyc", "must not ship"),
        ("tessera_apps/notebooks/x.ipynb", "must not ship"),
        ("tessera_apps/tests/fixtures/caldor_64.npz", "must not ship"),
        ("tessera_apps/conftest.py", "must not ship"),
        ("tessera_apps/test_cli.py", "must not ship"),
    ],
)
def test_wheel_forbidden_file(tmp_path, member, fragment):
    files = wheel_files()
    files[member] = ""
    assert_named(problems_with(tmp_path, wheel=files), f"wheel: {member} {fragment}")


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"Version": ["9.9.9"]}, "Version is '9.9.9', expected '1.2.3'"),
        ({"Name": ["tessera_embeddings"]}, "Name is 'tessera_embeddings'"),
        ({"Requires-Python": [">=3.10"]}, "Requires-Python is '>=3.10', expected '>=3.12'"),
        ({"Requires-Python": None}, "Requires-Python is None"),
        ({"License-Expression": None}, "License-Expression is None, expected 'Apache-2.0'"),
        ({"License-Expression": ["MIT"]}, "License-Expression is 'MIT'"),
        ({"License-File": None}, "no License-File: LICENSE header"),
        ({"License": ["Apache-2.0"]}, "has a License header as well"),
        (
            {"Classifier": ["License :: OSI Approved :: Apache Software License"]},
            "licence classifier ['License :: OSI Approved :: Apache Software License']",
        ),
        ({"Metadata-Version": ["2.1"]}, "Metadata-Version is '2.1'"),
        ({"Requires-Dist": ["icechunk<3,>=2.2", "numpy>=2.0"]}, "numcodecs[pcodec]>=0.16 of pyproject.toml is not"),
        (
            {"Requires-Dist": ["icechunk>=2.2", "numcodecs[pcodec]>=0.16", "numpy>=2.0"]},
            "Requires-Dist icechunk>=2.2 is not a runtime dependency",
        ),
        (
            {"Requires-Dist": [*DEPS, "boto3>=1"]},
            "Requires-Dist boto3>=1 is not a runtime dependency in pyproject.toml",
        ),
        ({"Provides-Extra": ["dev"]}, "extra 'validate' of pyproject.toml is not a Provides-Extra"),
        ({"Description-Content-Type": ["text/x-rst"]}, "Description-Content-Type is 'text/x-rst'"),
        ({"description": ["\n"]}, "the long description (the README) is empty"),
        ({"description": ["See [the notebook](notebooks/01_wildfire.ipynb).\n"]}, "relative link 'notebooks/01_"),
        ({"description": ['<img src="docs/preview.png">\n']}, "relative link 'docs/preview.png'"),
        ({"description": ["See [the docs][d].\n\n[d]: docs/releasing.md\n"]}, "relative link 'docs/releasing.md'"),
    ],
)
def test_wheel_metadata_problem(tmp_path, changes, fragment):
    files = wheel_files()
    files[f"{INFO}/METADATA"] = metadata(**changes)
    assert_named(problems_with(tmp_path, wheel=files), f"wheel {INFO}/METADATA: ", fragment)


def test_requires_dist_matches_whatever_the_spelling(tmp_path):
    files = wheel_files()
    files[f"{INFO}/METADATA"] = metadata(
        **{"Requires-Dist": ["icechunk <3, >=2.2", "NumCodecs[PCodec] >=0.16", "numpy >= 2.0", 'x; extra == "dev"']}
    )
    assert problems_with(tmp_path, wheel=files) == []


def test_wheel_package_version_and_names(tmp_path):
    files = wheel_files()
    files["tessera_apps/__init__.py"] = '__version__ = "0.0.1"\n'
    assert_named(problems_with(tmp_path, wheel=files), "wheel tessera_apps/__init__.py: __version__ is '0.0.1'")

    files["tessera_apps/__init__.py"] = "VERSION = 1\n"
    assert_named(problems_with(tmp_path, wheel=files), "wheel tessera_apps/__init__.py: no __version__ assignment")


def test_wheel_of_another_version_is_named_and_still_read(tmp_path):
    folder = tmp_path / "other"
    write_wheel(folder, wheel_files("1.2.4"), name="tessera_apps-1.2.4-py3-none-any.whl")
    write_sdist(folder, sdist_files())
    problems = cd.check_dist(folder, expected())
    assert_named(
        problems,
        "wheel file name is tessera_apps-1.2.4-py3-none-any.whl, expected tessera_apps-1.2.3-py3-none-any.whl",
        "dist-info folders are ['tessera_apps-1.2.4.dist-info']",
        "METADATA: Version is '1.2.4', expected '1.2.3'",
        "__version__ is '1.2.4'",
    )


def test_wheel_tags_and_purelib(tmp_path):
    files = wheel_files()
    files[f"{INFO}/WHEEL"] = "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: cp312-cp312-linux_x86_64\n"
    assert_named(
        problems_with(tmp_path, wheel=files),
        "WHEEL does not say Root-Is-Purelib: true",
        "WHEEL tags are ['cp312-cp312-linux_x86_64']",
    )


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[console_scripts]\ntessera-apps = tessera_apps.main:run\n", "is 'tessera_apps.main:run'"),
        ("[console_scripts]\nTessera-Apps = tessera_apps.cli:app\n", "is None"),
        ("[gui_scripts]\ntessera-apps = tessera_apps.cli:app\n", "is None"),
        ("not an ini file", "cannot be read"),
    ],
)
def test_wheel_console_script(tmp_path, text, fragment):
    files = wheel_files()
    files[f"{INFO}/entry_points.txt"] = text
    assert_named(problems_with(tmp_path, wheel=files), "entry_points.txt", fragment)


def test_unreadable_wheel(tmp_path):
    folder = tmp_path / "bad"
    folder.mkdir()
    (folder / f"tessera_apps-{VERSION}-py3-none-any.whl").write_bytes(b"not a zip")
    write_sdist(folder, sdist_files())
    assert_named(cd.check_dist(folder, expected()), "is not a readable wheel")


# --------------------------------------------------------------------------------------------
# the sdist
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "drop",
    [
        "CHANGELOG.md",
        "LICENSE",
        "README.md",
        "requirements-lock.txt",
        "ruff.toml",
        "mypy.ini",
        "tests/fixtures/caldor_64.npz",
        "tests/conftest.py",
        "scripts/check_dist.py",
        "docs/releasing.md",
        "src/tessera_apps/data/caldor.geojson",
        "src/tessera_apps/py.typed",
    ],
)
def test_sdist_missing_file(tmp_path, drop):
    files = sdist_files()
    del files[drop]
    assert_named(problems_with(tmp_path, sdist=files), f"sdist: {drop} is missing")


def test_sdist_missing_folder(tmp_path):
    files = {k: v for k, v in sdist_files().items() if not k.startswith("docs/")}
    assert_named(problems_with(tmp_path, sdist=files), "sdist: no docs/ folder", "sdist: docs/releasing.md is missing")


@pytest.mark.parametrize(
    "member",
    [
        "notebooks/01_wildfire.ipynb",
        "notebooks/out/wildfire/wildfire_report.json",
        ".github/workflows/release.yml",
        ".claude/settings.json",
        ".venv/bin/python",
        "out/wildfire_2020_2022.tif",
        "dist/old.whl",
        "src/tessera_apps/__pycache__/io.cpython-312.pyc",
        "tests/__pycache__/conftest.cpython-312.pyc",
        "docs/notes.ipynb",
        "scripts/.ipynb_checkpoints/x.py",
    ],
)
def test_sdist_forbidden_file(tmp_path, member):
    files = sdist_files()
    files[member] = ""
    assert_named(problems_with(tmp_path, sdist=files), f"sdist: {member} must not ship")


def test_sdist_member_outside_its_folder(tmp_path):
    files = sdist_files()
    files["../evil.py"] = ""
    files["/etc/evil"] = ""
    assert_named(
        problems_with(tmp_path, sdist=files, symlink="link"),
        "sdist: unsafe member path ../evil.py",
        "sdist: unsafe member path /etc/evil",
        f"sdist: {ROOT}/link is not a regular file or folder",
    )


def test_sdist_of_another_version_is_named_and_still_read(tmp_path):
    files = sdist_files()
    files["PKG-INFO"] = metadata(Version=["1.2.4"])
    problems = problems_with(tmp_path, sdist=files, name="tessera_apps-1.2.4.tar.gz", root="tessera_apps-1.2.4")
    assert_named(
        problems,
        "sdist file name is tessera_apps-1.2.4.tar.gz, expected tessera_apps-1.2.3.tar.gz",
        "sdist: top folders are ['tessera_apps-1.2.4'], expected ['tessera_apps-1.2.3']",
        "sdist PKG-INFO: Version is '1.2.4'",
    )
    assert not any("is missing" in p for p in problems), problems


def test_sdist_metadata_and_pyproject(tmp_path):
    files = sdist_files()
    files["PKG-INFO"] = metadata(**{"License-Expression": None, "Requires-Python": [">=3.11"]})
    files["pyproject.toml"] = pyproject_text(deps=("numpy>=2.0",), script="tessera_apps.cli:main")
    files["src/tessera_apps/__init__.py"] = '__version__ = "1.2.3.dev0"\n'
    assert_named(
        problems_with(tmp_path, sdist=files),
        "sdist PKG-INFO: License-Expression is None",
        "sdist PKG-INFO: Requires-Python is '>=3.11'",
        "sdist pyproject.toml: [project.scripts] tessera-apps is not 'tessera_apps.cli:app'",
        "sdist pyproject.toml: its dependencies differ",
        "sdist src/tessera_apps/__init__.py: __version__ is '1.2.3.dev0'",
    )


def test_unreadable_sdist(tmp_path):
    folder = tmp_path / "bad"
    write_wheel(folder, wheel_files())
    (folder / f"{ROOT}.tar.gz").write_bytes(b"not a tarball")
    assert_named(cd.check_dist(folder, expected()), "is not a readable sdist")


# --------------------------------------------------------------------------------------------
# the source tree comparison
# --------------------------------------------------------------------------------------------


def test_source_files_must_all_ship(dist):
    source = frozenset(
        [
            *(f"src/tessera_apps/{rel}" for rel in cd.WHEEL_PACKAGE_REQUIRED),
            "src/tessera_apps/apps/new_app.py",
            "tests/test_new_app.py",
            "scripts/make_new_fixture.py",
            "docs/new.md",
        ]
    )
    problems = cd.check_dist(dist, expected(source_files=source))
    assert_named(
        problems,
        "wheel: tessera_apps/apps/new_app.py is missing (it is in the source tree)",
        "sdist: src/tessera_apps/apps/new_app.py is missing (it is in the source tree)",
        "sdist: tests/test_new_app.py is missing (it is in the source tree)",
        "sdist: scripts/make_new_fixture.py is missing (it is in the source tree)",
        "sdist: docs/new.md is missing (it is in the source tree)",
    )
    # A file required anyway is reported once, not twice.
    assert len([p for p in problems if "apps/new_app.py" in p]) == 2


def test_wheel_file_not_in_the_source_tree(tmp_path):
    files = wheel_files()
    files["tessera_apps/stale.py"] = ""
    source = frozenset(f"src/tessera_apps/{rel}" for rel in cd.WHEEL_PACKAGE_REQUIRED)
    problems = problems_with(tmp_path, wheel=files, exp=expected(source_files=source))
    assert problems == ["wheel: tessera_apps/stale.py is not in the source tree"]


def _tree(root: Path) -> None:
    for rel in ("src/tessera_apps/__init__.py", "tests/test_a.py", "scripts/s.py", "docs/d.md", "notebooks/n.ipynb"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")
    (root / "tests/__pycache__").mkdir()
    (root / "tests/__pycache__/test_a.cpython-312.pyc").write_bytes(b"\0")
    (root / "docs/scratch.log").write_text("x\n")
    (root / ".gitignore").write_text("*.log\n__pycache__/\n")


def test_source_files_walk_without_git(tmp_path, monkeypatch):
    _tree(tmp_path)

    def no_git(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(cd.subprocess, "run", no_git)
    # Without git nothing reads .gitignore, so docs/scratch.log counts; the forbidden parts never do.
    assert cd.source_files(tmp_path) == {
        "src/tessera_apps/__init__.py",
        "tests/test_a.py",
        "scripts/s.py",
        "docs/d.md",
        "docs/scratch.log",
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_files_follow_git(tmp_path):
    _tree(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    # Tracked and untracked files count, ignored ones and files outside the four folders do not.
    assert cd.source_files(tmp_path) == {"src/tessera_apps/__init__.py", "tests/test_a.py", "scripts/s.py", "docs/d.md"}


def test_source_files_refuse_a_folder_that_is_not_a_source_tree(tmp_path):
    with pytest.raises(cd.CheckError, match="is not a source tree"):
        cd.source_files(tmp_path)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("icechunk>=2.2,<3", "icechunk<3,>=2.2"),
        ("numcodecs[pcodec]>=0.16", "NumCodecs [PCodec] >= 0.16"),
        ("scikit-learn>=1.7.1", "scikit_learn>=1.7.1"),
        ('foo; python_version < "3.13"', "foo ;python_version  <  '3.13'"),
        ("zarr (>=3.3)", "zarr>=3.3"),
    ],
)
def test_normalize_requirement_same(a, b):
    assert cd.normalize_requirement(a) == cd.normalize_requirement(b)


@pytest.mark.parametrize(("a", "b"), [("numpy>=2.0", "numpy>=2.1"), ("numcodecs[pcodec]", "numcodecs"), ("a", "b")])
def test_normalize_requirement_different(a, b):
    assert cd.normalize_requirement(a) != cd.normalize_requirement(b)


def test_normalize_requirement_refuses_garbage():
    with pytest.raises(ValueError, match="cannot read the requirement"):
        cd.normalize_requirement(">=1.0")


def test_relative_links_skip_code_absolute_links_and_anchors():
    text = (
        "[a](https://x.org/a) [b](http://x.org) [c](#install) [d](mailto:a@b.c) ![e](img/e.png)\n"
        "`[f](not/a/link)` and ``[g](h)``\n\n```python\nx = y[0](z)\n```\n\n~~~\n[i](j)\n~~~\n"
        "[ref]: docs/ref.md\n<a href='LICENSE'>licence</a> [k](img/e.png)\n"
    )
    assert cd.relative_links(text) == ["img/e.png", "docs/ref.md", "LICENSE"]


# --------------------------------------------------------------------------------------------
# the checker's lists against this repository
# --------------------------------------------------------------------------------------------


def test_lists_match_the_repository():
    package = REPO / "src" / "tessera_apps"
    for rel in cd.WHEEL_PACKAGE_REQUIRED:
        assert (package / rel).is_file(), rel
    assert sorted(cd.DEMO_GEOJSON) == sorted(p.stem for p in (package / "data").glob("*.geojson"))
    for rel in [*cd.SDIST_TOP_REQUIRED, *cd.SDIST_REQUIRED]:
        assert rel == "PKG-INFO" or (REPO / rel).is_file(), rel


def test_lists_match_the_hatch_configuration():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = config["tool"]["hatch"]["build"]["targets"]["sdist"]
    included = {entry.strip("/") for entry in sdist["include"]}
    assert included == (set(cd.SDIST_TOP_REQUIRED) - {"PKG-INFO"}) | set(cd.SDIST_DIRS)
    project = config["project"]
    assert project["requires-python"] == cd.REQUIRES_PYTHON
    assert project["license"] == cd.LICENSE_EXPRESSION
    assert project["license-files"] == [cd.LICENSE_FILE]
    assert project["scripts"] == dict([cd.CONSOLE_SCRIPT])
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
