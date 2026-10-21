# Releasing tessera_apps

Releases go to PyPI as [`tessera-apps`](https://pypi.org/project/tessera-apps/) (`pip install tessera_apps` finds it under either spelling). They are published by [`.github/workflows/release.yml`](https://github.com/rh281/tessera_apps/blob/main/.github/workflows/release.yml) with PyPI trusted publishing, so no API token is stored anywhere.

**A version number can never be reused on PyPI.** Once a file for 0.1.0 has been uploaded, PyPI refuses any other file for 0.1.0 for ever, even after the release is deleted. A mistake that reaches PyPI is fixed with a new version (0.1.1), never by re-uploading. TestPyPI follows the same rule, separately.

## What the workflow does

On a push of a tag `vX.Y.Z`:

1. **`ci`** runs the whole of `ci.yml` (lockfile, ruff, mypy, offline tests) on Python 3.12 and 3.13.
2. **`build`** checks that the repository is public (step 0 of the one-time setup) and that the tag is `v` plus `__version__`, and stops, naming the problem, if not. It takes the release notes from the `## X.Y.Z (date)` section of `CHANGELOG.md` and stops if there is none. Then it runs `uv build`, `scripts/check_dist.py` (what the wheel and the sdist must and must not hold, and their metadata) and `twine check --strict`. Last, a smoke test: a fresh Python 3.12 venv, `pip install` of the wheel with its dependencies resolved from PyPI as a user would get them, `tessera-apps --version` and `--help`, and the offline test suite run against the installed wheel.
3. **`publish`** uploads the build to PyPI, in the GitHub environment `pypi`.
4. **`github-release`** creates the GitHub release for the tag, with the wheel and the sdist attached and the changelog section as its notes.

Nothing is published unless every earlier job passed. The workflow can also be run by hand from the Actions tab (`workflow_dispatch`): that is a dry run of steps 1 and 2, which publishes nothing unless the **testpypi** box is ticked, and then only to TestPyPI. Two runs for the same tag never overlap: the second waits.

## One-time setup

These steps need an owner of the PyPI account and an admin of the GitHub repository. Until they are done, the `publish` job fails and nothing is uploaded.

0. **Make the repository public, before the first tag.** The README is the PyPI page and links to this repository about twenty times, the package's Homepage, Source, Issues and Changelog links point here, and the README's "from source" and reproducible install routes clone it. While the repository is private all of those give a 404 to everyone else, and PyPI keeps each release's page as it was uploaded, so a release made while private keeps its dead links for ever. In *Settings*, *General*, *Danger Zone*, choose *Change visibility* and make it public. Then check from a machine that is not signed in to GitHub:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' https://github.com/rh281/tessera_apps
   ```

   It must print `200` (a private repository gives `404`). The `build` job runs the same check on a tag push and stops the release, before anything is published, when the repository is not public; a dry run only warns.
1. **PyPI pending trusted publisher.** Sign in at <https://pypi.org>, open *Your account*, *Publishing*, and add a pending publisher of type GitHub with:
   - PyPI project name: `tessera-apps`
   - Owner: `rh281`
   - Repository name: `tessera_apps`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

   The first successful publish creates the project and turns the pending publisher into a normal one. A pending publisher does not reserve the name, so do the first release soon after.
2. **GitHub environment `pypi`.** In the repository's *Settings*, *Environments*, create `pypi`. Optionally:
   - add **required reviewers**, so a person approves each upload before it starts (GitHub offers them on a public repository, and on a private one only with a paid plan);
   - under *Deployment branches and tags*, choose *Selected branches and tags* and add a tag rule `v*`, so only a release tag can deploy to it.
3. **TestPyPI, for dry runs (optional).** The same on <https://test.pypi.org> (a separate account): a pending publisher with the same project, owner, repository and workflow, and environment name `testpypi`. Then create the GitHub environment `testpypi`. A dry run runs on a branch, not a tag, so if you add a deployment rule there, allow the branch you run it from (`main`).
4. **Tag protection (optional).** A tag ruleset on `v*` that limits who can create release tags.

## Each release

1. **Bump the version** in `src/tessera_apps/__init__.py`, the only place it lives: `__version__ = "X.Y.Z"`. The wheel, the sdist, `tessera-apps --version` and every run report read it from there.
2. **Add the changelog section** at the top of `CHANGELOG.md`, headed `## X.Y.Z (YYYY-MM-DD)`, with what changed and the known limitations. The release notes are taken from it, and both the build and the offline tests fail when the newest section is not the package version.
3. **Regenerate the lock** if a dependency changed: `uv pip compile pyproject.toml --all-extras --python-version 3.12 --no-header -o requirements-lock.txt`.
4. **Commit and push** to `main` (or merge the pull request), and wait for CI to pass.
5. **Dry run (optional).** In the Actions tab, open *release*, *Run workflow*, on `main`. Leave **testpypi** unticked to build and check only, or tick it to upload to TestPyPI as well, then try that build:

   ```bash
   python3.12 -m venv /tmp/tessera-testpypi
   /tmp/tessera-testpypi/bin/pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tessera_apps==X.Y.Z
   /tmp/tessera-testpypi/bin/tessera-apps info
   ```

   TestPyPI keeps the version too, so a second upload of the same version there fails. That is expected: the checks before the upload still say whether the build is good.
6. **Tag and push the tag:**

   ```bash
   git tag -a vX.Y.Z -m "tessera_apps X.Y.Z"
   git push origin vX.Y.Z
   ```

7. **Watch the run** in the Actions tab, and approve the `pypi` deployment if the environment has reviewers.
8. **Check the release** from a clean venv, with no AWS credentials set:

   ```bash
   python3.12 -m venv /tmp/tessera-check
   /tmp/tessera-check/bin/pip install tessera_apps==X.Y.Z
   /tmp/tessera-check/bin/tessera-apps --version
   /tmp/tessera-check/bin/tessera-apps info
   ```

   `--version` prints `tessera-apps X.Y.Z`, and `info` prints the store's snapshot and the published years of zone 10N. Check that the GitHub release has the notes and both files, and that the PyPI page renders the README.

## When something fails

- **Before `publish`** (a private repository, a wrong tag, a missing changelog section, a failing check or test): nothing reached PyPI. Delete the tag (`git push origin :refs/tags/vX.Y.Z` and `git tag -d vX.Y.Z`), fix the problem on `main`, and tag again.
- **In `publish`** (most often the trusted publisher or the environment name does not match): fix the setup on PyPI or GitHub, then use *Re-run failed jobs* on the same run, which uploads the same build. If PyPI already accepted any file for the version, do not move the tag to other code: release X.Y.Z+1.
- **In `github-release`, after PyPI accepted the upload:** re-run the job, or create the release by hand from the run's `dist` artifact with `gh release create vX.Y.Z dist/* --notes-file notes.md`.
- **A bad release on PyPI:** yank it (on PyPI, *Manage*, the release, *Options*, *Yank*) so `pip` stops choosing it, and publish a fixed X.Y.Z+1. Deleting it would not free the number.
