# tessera_apps: pre-build survey

Survey date 2026-10-14. The repo `dClimate/tessera-embeddings` was read-only here. The live store was read anonymously at main snapshot `B6AFRJGX5XQC0A716PA0`. Nothing was built. The companion notebook `TESSERA_community_applications.ipynb` was not available, so its `TesseraReader` and conventions are unchecked.

The brief can be built in outline, but not as written. Three blockers stop the build or the release: dependency floors that admit an icechunk which cannot open the store and a geotessera which silently reads the legacy v1 model, an existing `release.yml` that builds `tessera_embeddings` instead of `tessera_apps`, and `.gitignore` plus hatchling rules that drop every bundled `.geojson`. Two method blockers follow: cross-year cosine is biased wherever a pixel switches between radar and radar-free years, and the wetland Mahalanobis score means nothing below about 129 seed pixels. The two-backend `Reader` design also hides real differences between the Icechunk store and the Source Coop Zarr copy that `GeoTesseraZarr()` reads by default, so choose one backend before writing `io.py`.

Severity key:
- **Blocker:** as specified, the package cannot open the store, cannot ship, or gives wrong answers.
- **Should fix:** change the brief before the step it affects.
- **Note:** wording or a minor point.

## Blockers

### B1. Dependency floors allow icechunk and zarr versions that cannot open the store
- **Brief (§3 Dependencies):** `icechunk>=1.0`, `zarr>=3`.
- **Actually:** The store is in the Icechunk v2 repository format, and icechunk 1.x cannot open it. The repo pins `icechunk>=2.1.1` and `zarr>=3.1.2` (the zarr floor is needed for sharded arrays). geotessera 0.11.0 requires `icechunk>=2.2` and `zarr>=3.3`. icechunk 2.0.4 has a known read slowdown of about 25x (#2158).
- **Evidence:** icechunk 1.1.21 `Repository.open` raised "this repository uses Icechunk v2 format, please upgrade the icechunk library". icechunk 2.3.0 reports `SpecVersion.v2`. See `pyproject.toml:35-37`, geotessera 0.11.0 `Requires-Dist`, and `scripts/scoping/scale_tests/README.md:78-80`.
- **Change:** Use `icechunk>=2.2,<3`, `zarr>=3.3` and `numcodecs[pcodec]>=0.16`.

### B2. `geotessera>=0.9` can silently read the legacy v1 model
- **Brief (§2 skill bullet; §3 Dependencies and Access path):** `geotessera>=0.9`, and the client "defaults to our dataset".
- **Actually:**
  - Only 0.11.0 (2026-10-04) defaults to the dClimate v1.1 data and can open `.icechunk` URLs.
  - 0.9.0 defaults to the legacy v1 store `https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1/zarr`, which holds a different model, and has no Icechunk support.
  - 0.10.0 to 0.10.2 default to `zarr_store_url("v1")`, list dclimate as "coming soon", and have no Icechunk module.
  - With the root lock's `zarr==3.2.1` and `icechunk==2.1.1` as constraints, the resolver picks geotessera 0.10.0. `GeoTesseraZarr()` would then return v1 embeddings while an Icechunk reader returns v1.1, and the two cannot be compared.
  - The 0.x API is still changing: `progress=` is deprecated and the default store changed in 0.11.
- **Evidence:** In the wheels: 0.9.0 `store.py:37`; 0.10.0 `store.py:66` and `registry.py:51`; 0.11.0 `store.py:72` and `registry.py:87-107`. CHANGES.md v0.11.0 says "GeoTessera now defaults to v1.1-dclimate". `uv pip compile` with those constraints resolves `geotessera==0.10.0`.
- **Change:**
  - Pin `geotessera>=0.11,<0.12`.
  - Always pass `store_url` explicitly.
  - Before any read, assert `gt.dataset is not None and gt.dataset.name == "1.1-dclimate"`.
  - Record `geotessera.__version__`, `gt.url` and `gt.dataset.name` in every report.

### B3. The existing release workflow builds the wrong package
- **Brief (§7.1, §8 step 7):** Use the existing release workflow if there is one, otherwise add `release.yml` that publishes on a `v*` tag.
- **Actually:**
  - `.github/workflows/release.yml` exists. It triggers on `v*` and runs a bare `uv build` at the root, which only ever builds `tessera_embeddings` 0.1.0. That stays true even if `tessera_apps` is made a uv workspace member, so this workflow cannot publish `tessera_apps`.
  - It has no test gate (`needs:`) and no check that the tag matches the version.
  - The repo has no tags, and `tessera_embeddings` is not on PyPI.
  - Whether a `v*` push would actually upload `tessera_embeddings` depends on a PyPI pending trusted publisher and on the `pypi` environment's rules. Neither could be checked.
- **Evidence:** `release.yml:3-6,15-17,21-26`; `pyproject.toml:2-3`. Reproduced with uv 0.11.32: a bare `uv build` builds only `tessera_embeddings`, and `uv build --package tessera_apps` writes to the root `dist/`. `git ls-remote` shows 0 tags.
- **Change:**
  - Leave `release.yml` untouched.
  - Add `.github/workflows/release-tessera-apps.yml`, triggered by a tag that does not start with `v` (for example `tessera-apps-v*`).
  - Build with `uv build tessera_apps -o dist` (standalone) or `uv build --package tessera_apps` (workspace).
  - Publish with `pypa/gh-action-pypi-publish` pinned by SHA, and add a step that checks the tag matches the version.
  - A PyPI owner must register a pending trusted publisher for `tessera-apps`: owner dClimate, repo tessera-embeddings, that workflow file and the environment name. Make this an explicit human checkpoint in step 7.
  - Never push a bare `v*` tag.

### B4. `.geojson` files are dropped by git and by hatchling
- **Brief (§3 `data/fixtures/`, §4 demo AOIs, §10 DoD):** One GeoJSON per app in `data/fixtures/`, and the DoD runs `--aoi data/fixtures/caldor.geojson`.
- **Actually:**
  - The root `.gitignore:101` ignores `*.geojson`. It is only negated for `examples/**` and `tests/fixtures/**` (lines 105-108), so `git add` refuses the files.
  - Adding `!tessera_apps/**/*.geojson` fixes git but not the build. Hatchling 1.32.4 applies the nearest `.gitignore` relative to the `tessera_apps/` project root, and it still dropped the files from both the sdist and the wheel, even when they were force-tracked.
  - `[tool.hatch.build] artifacts = ["*.geojson"]` restored them.
  - The same relative matching works in reverse: hatchling included a git-ignored `tessera_apps/tests/fixtures/x.geojson`.
- **Evidence:** `git check-ignore -v` prints `.gitignore:101:*.geojson`. The scratch sdist kept `caldor.npz` but not `caldor.geojson`. The wheel kept `data/caldor.json` but not `data/caldor.geojson`.
- **Change:**
  - In the same commit as the fixtures, add `!tessera_apps/**/*.geojson` to the root `.gitignore`, plus `demo/` if that lives at the root.
  - Add `artifacts = ["*.geojson"]` to `tessera_apps/pyproject.toml`. Alternatively, store the AOIs as `.json`.
  - After `uv build`, list the contents of the wheel and the sdist to confirm the AOIs are there.

### B5. Radar regime switches bias every cross-year comparison (general use; the demos are unaffected)
- **Brief (§2 last bullet; §4.1 step 5; §4.2 steps 1-4; §4.3 step 4; §4.4 steps 2-5):** Optical-only pixels "may be noisier". Wildfire raises a flag if more than 20% of the scar is radar-free. The other apps compare years, or apply a `year_a` model or reference to `year_b`, without checking the radar regime.
- **Actually:**
  - In radar-gap chunks, every pixel has radar in 2017-2021, none in 2022-2024, and radar again in 2025.
  - Median consecutive-year cosine at the 2021 to 2022 and 2024 to 2025 transitions falls to 0.753-0.865. Within a single regime it is 0.91-0.99.
  - The shift is systematic. A scene-wide mean shift explains 70-94% of the change energy at a switch, against 36-62% within a regime. Removing that mean shift brings the median back to 0.97-0.99.
  - The effect adds 0.13-0.25 of cosine distance to every switching pixel. That hides moderate real change and biases any classifier carried across years.
  - Halving radar depth alone does not cause a step: Calaveras (10N) has radar every year and stays at 0.94-0.97.
  - All three demo AOIs are 0% radar-free from 2017 to 2025, so the demos will not reveal this.
- **Evidence:** Live chunk medians for the two transitions:
  - 47N (Siberia): 0.863 and 0.796.
  - 52S: 0.852 and 0.753.
  - 53S: 0.865 and 0.831.
  - ADR-013 (`context_docs/decisions/013-optional-s1-s2-only-pixels.md:130-135`) calls the all-zero radar input "out-of-distribution as a sequence".
  - `docs/single-vs-global.md:346-348` says not to treat the two kinds of pixel as interchangeable.
- **Change:**
  - Add a per-pixel `radar_present = s1_asc_obs_count + s1_desc_obs_count > 0` to io/metrics.
  - For every pair of years an app compares, mask the pixels whose regime differs, or refuse the AOI. This covers wildfire recovery years, deforest training and prediction years, wetland reference and target years, and every restore trajectory year.
  - Report the mismatch fraction in `report.json`.
  - Replace the ">20% radar-free in scar" flag with this check.
  - Keep the obs counts in every npz fixture so the check is tested offline.

### B6. Wetland Mahalanobis is meaningless at the stated seed counts (blocks step 5)
- **Brief (§4.3 steps 2-3 and demo; §6 minimum of 5 seed pixels):**
  - Mahalanobis is the primary score.
  - Cosine is the fallback below 30 seed pixels.
  - The threshold is the 95th percentile of the seeds' own scores.
  - The demo uses 12 seed points.
- **Actually:**
  - With d=128, the sample covariance has rank of at most n-1. For n <= 128, `np.linalg.inv` does not raise; it silently returns a meaningless inverse. At n=12-60, held-out recall roughly equalled the false-positive rate (0.4-0.7 each).
  - Ledoit-Wolf on random seeds gave recall of 0.005-0.08 at n=30, 0.51-0.64 at n=128, 0.84-0.92 at n=500 and 0.93 at n=2000.
  - Seeds drawn from polygons are spatially contiguous. With contiguous seeds and the in-sample 95th percentile, held-out recall was:
    - 0.17-0.42 for Mahalanobis, even at n=2000.
    - 0.00-0.21 for cosine-to-mean at n=12, and 0.60-0.75 at n=2000.
  - So the 12-point demo always falls back to cosine, and a larger seed set alone does not fix the in-sample threshold.
  - Setting the threshold from 12 seeds chosen from GMW also flatters recall against GMW.
- **Evidence:** Live `geoemb:dimensions=128`. Experiments on k-means clusters in live Caldor 512x512 blocks (`methods_check/maha.py`, `skeptic_mr/maha2.py`). The clusters are not real wetland classes.
- **Change:**
  - Default to cosine-to-seed-mean.
  - Allow Mahalanobis only with shrinkage (`sklearn.covariance.LedoitWolf` or OAS) and about 500 or more valid seed pixels.
  - Set thresholds from held-out seed scores using spatially blocked folds, never from in-sample scores.
  - Ship the demo seeds as polygons: at least 30 pixels, ideally several hundred, and more than 500 if Mahalanobis is to be shown.
  - Exclude seed pixels from recall.
  - Apply the same rules to restore's control filter (S17).

## Should fix before starting

### S1. Pick one backend; the §2 facts describe only the Icechunk zone groups
- **Brief (§2 Facts; §3 Access path, one `Reader` protocol, `iter_strips` aligned to 256):** Use `GeoTesseraZarr` for point and patch reads. The facts do not say which backend they describe.
- **Actually:** `GeoTesseraZarr()` defaults to the Source Coop Zarr copy (`https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate`), not the Icechunk repo. The two backends differ as follows:

  | | Icechunk zone groups | Source Coop Zarr copy / geotessera views |
  |---|---|---|
  | Groups | `10N`, `20S` (per hemisphere) | `utm01`..`utm60` (hemispheres merged) |
  | South CRS | EPSG:327xx | 326xx with negative northings. A live Rondonia read returned EPSG:32620 with origin northing -972310.0; the Icechunk `20S` group is EPSG:32720 |
  | Dims | (time, northing, easting, band) | (time, band, y, x) |
  | Embedding chunks | (1,256,256,128) in shards (1,2048,2048,128), Zstd; scales PCodec | (1,128,32,32) in shards (1,128,4096,4096), Blosc zstd |
  | Cold point read | about 8.1 MB | about 0.65 MB measured: 265 KB scales plus 381 KB embeddings, of which 2 x 262 KB are shard indexes |
  | years_complete | `root[zone].attrs["years_complete"]` | `attrs["geotessera:source_groups"]["10N"]["years_complete"]` |
  | `open_zone()` | lazy, no dask | dask-backed, chunked at the shard, which breaks the `chunks=None` rule |
  | `cache_dir` | ignored | used |

  Both backends share int8 with scales and NaN, calendar years 2017-2025, the three `*_obs_count` layers, and a descending north axis. A parity test returned identical embeddings. geotessera's `iter_region` starts its strips at the window top, so they are not chunk-aligned.
- **Evidence:** Fetched `utm10/embeddings/zarr.json` from the `s3.us-west-2` Source Coop mirror. Live `open_zone` calls on both backends. A byte-counting wrapper measured the read costs. geotessera 0.11.0 `store.py:72,587-593,903-946,1334-1381` and `icechunk.py:344-366`. `README.md:87-95`.
- **Change:**
  - Simplest: use one backend, the Icechunk store, through a hand-written `TesseraReader`.
  - If both backends stay, the `Reader` must normalise dim names, the band axis, the hemisphere and CRS convention, chunk geometry (read from `array.chunks`), years_complete lookup, and provenance. Add a southern-AOI test that compares both backends.
  - State in §2 that the facts describe the Icechunk groups.
  - Replace "defaults to our dataset" with "defaults to the dClimate v1.1 run, served from the Source Coop Zarr copy".

### S2. Run the years_complete check in every app, on every backend, for every year read
- **Brief (§3 Access path; §4 intro):** Drop to Icechunk "only when an app needs `years_complete` or the observation counts". Check the requested years.
- **Actually:**
  - Neither trigger needs Icechunk. The obs counts are in `open_zone()` on both backends, and years_complete is on the Zarr copy too.
  - §4 requires the check in every app anyway, so the "only when" condition never applies.
  - geotessera 0.11 does not raise on a missing zone-year. It returns fill, sets `geotessera:years_incomplete`, and only `streaming.py` logs a warning.
  - Wildfire reads every year through 2025, and restore reads `planted - 1` through 2025, not only `year_a` and `year_b`.
- **Evidence:** geotessera 0.11.0 `icechunk.py:13` ("Zone-years missing from a group's years_complete read as fill"), `:180-186`, `:316-325`, `:389-398` and `:452`. Live `incomplete_years()` returned the same 14 gaps as the raw probe. Brief lines 91, 101 and 150.
- **Change:** Add a preflight to every app on every backend. It makes one `zarr.open_group` metadata read per touched zone, checks every year the app will read, runs before any chunk read, and fails with the zone and year.

### S3. "Before any S3 read" is impossible, and nothing reports bytes read
- **Brief (§6 Fail loud; §6 Cost discipline):** Checks stop "before any S3 read". Every app "logs bytes read from S3".
- **Actually:**
  - years_complete is a zone attribute stored on S3, and reading `zone.attrs` also pulls the large `runs` provenance. Checking seed validity needs `scales` reads.
  - icechunk 2.3.0 has no read counter. `Session` exposes `snapshot_id`, `status`, `chunk_coordinates` and `chunk_type`. `Repository.chunk_storage_stats` reports stored bytes only.
  - Host-wide `psutil.net_io_counters` overcounted: 16.3 MB for a read of about 8.1 MB, and 393 MB for a read expected near 208 MB.
  - A `zarr.storage.WrapperStore` subclass that counts the buffers from `get`, `get_partial_values` and the async generator `get_ranges` measured 8.12 MB correctly. zarr 3.4.1 sharding uses `get_ranges`; zarr 3.2.1 uses `byte_getter.get`. The wrapper does not see manifest or snapshot fetches.
- **Evidence:** `context_docs/storage/reading-the-published-store.md:51-56,64-67`; `scripts/diagnostic/published_store_read_bench.py:137-143`; probes `methods_check/bytes_probe.py` and `wrap_probe3.py`.
- **Change:**
  - Reword to "before any embedding chunk read". Run the metadata and scales checks first; the registry parquet can pre-screen coverage.
  - Define the logged figure as "chunk payload bytes", from a zarr-level wrapper store. Otherwise use an estimate: chunks that exist (`session.chunk_coordinates` or the `live_shards` logic) times the chunk size, not the bbox area.

### S4. `checkpoint_id` identifies the model, not the data version
- **Brief (§6 Reproducible):** Record the "store checkpoint_id from the root attrs".
- **Actually:**
  - `checkpoint_id = "tessera_v1_1_aws_encoder.pt"` is the filename of the model weights. It stays the same for the store's whole lifetime.
  - The data version is the Icechunk snapshot id. Main head is `B6AFRJGX5XQC0A716PA0`, a metadata-only commit from 2026-09-14 that moved `spatial:transform` to the pixel corner. Its parent, `QR7F41A6WYZ03VC92T6G`, is the final data commit and the one the repo docs measured.
  - The two snapshots share `checkpoint_id` and `geoemb:build_version` `0.1.0`, yet the 10N transform origin differs: (163845, 9338875) against (163840, 9338880).
  - The Zarr copy records `geotessera:source_snapshot = QR7F41A6WYZ03VC92T6G` and `geotessera:migration_id`.
- **Evidence:** Live root attrs and `repo.ancestry(branch="main")`; `conventions.py:318-321,337-338`; `reading-the-published-store.md:24-27,158-161`.
- **Change:**
  - Record `checkpoint_id` and `geoemb:model` as model provenance.
  - Record the backend URL plus `session.snapshot_id` (or `geotessera:source_snapshot` and `migration_id`) as data provenance.
  - Record `zone.attrs["runs"][year]["code"]["commit"]` for each zone-year used.
  - Support `snapshot_id` pinning via `repo.readonly_session(snapshot_id=...)` or `IcechunkStore(snapshot_id=...)`, and pin it for the demos.

### S5. Fixtures and demo assets are not in the wheel
- **Brief (§3 layout; §4.3 "default bundled seeds"; §10 DoD):** `data/fixtures/` sits outside `src/`. The DoD runs `pip install`, then `--aoi data/fixtures/caldor.geojson`.
- **Actually:**
  - The wheel contains only `src/tessera_apps`, so after `pip install` there is no `data/fixtures/`.
  - The DoD path only resolves with the working directory set to `tessera_apps/` inside a checkout.
  - An installed CLI cannot find any default seeds.
- **Evidence:** The scratch wheel contains `__init__.py`, `cli.py`, `metrics.py`, `data/caldor.json` and dist-info only.
- **Change:**
  - Put demo AOIs, seeds and default reference data under `src/tessera_apps/data/` and load them with `importlib.resources`. Handle `.geojson` as in B4.
  - Optionally accept demo names, for example `--aoi caldor`.
  - Keep the test-only npz files under `tessera_apps/tests/fixtures/`.
  - Reword the DoD to say whether it means a checkout or a pip install.

### S6. ruff, mypy and CI do not cover the new package
- **Brief (§6, §10):** `ruff` is clean against `ruff.toml`, `mypy` is clean against `mypy.ini`, and `pytest` passes offline.
- **Actually:**
  - **ruff 0.15.13:**
    - The root `ruff.toml` applies. Its `known-first-party` lists only `tessera_embeddings`, so I001 merges `tessera_apps` imports into the third-party block.
    - The `tests/**/*.py` ignore is anchored at the root and misses `tessera_apps/tests`, so ANN201 and D103 fire.
    - `.ipynb` files are linted by default (T201, ANN, D103, I001).
    - T201 fires on `print` in `cli.py`.
    - All of these can be fixed, but they add friction.
  - **mypy 2.1.0:**
    - From the repo root, `mypy tessera_apps/src/tessera_apps` names the modules `tessera_apps.src.tessera_apps.*`, and first-party imports silently become `Any`. Only 1 of 5 planted errors was reported.
    - Run from inside `tessera_apps/`, it found all 5.
    - `mypy -p tessera_apps` from the root treats the top-level directory as a namespace package and pulls in the tests.
  - **CI:**
    - `lint.yml` runs `ruff check src/ tests/` and `mypy src/tessera_embeddings/`.
    - `unit.yml` runs `pytest tests/unit`.
    - The architecture tests scan only `src/tessera_embeddings`.
    - `integration.yml` filters on specific `src/` and `tests/` subtrees.
    - Nothing touches `tessera_apps/`. In a workspace, `uv sync --all-extras` installs only the root package; `--all-packages` is needed.
- **Evidence:** `ruff.toml:74-75,80-99`; `mypy.ini:1-10`; `lint.yml:28-33`; `unit.yml:31-40`; `architecture.yml:28-31`; `integration.yml:4-16`.
- **Change:**
  - **ruff:** Either set `known-first-party = ["tessera_embeddings", "tessera_apps"]` and add per-file-ignores for `tessera_apps/tests/**/*.py` (copy the tests entry) and `tessera_apps/notebooks/**/*.ipynb` (at least T201, ANN and D1xx), or give `tessera_apps/pyproject.toml` a `[tool.ruff]` with `extend = "../ruff.toml"`. Write CLI output with `typer.echo` or a rich `Console`.
  - **mypy:** Run `cd tessera_apps && mypy src/tessera_apps tests`, or add `$MYPY_CONFIG_FILE_DIR/tessera_apps/src` to `mypy_path`. Optionally turn off `ignore_missing_imports` for `tessera_apps.*`.
  - **CI:** Add a job that runs ruff, mypy (from inside the package) and offline pytest on Python 3.12 and 3.13. Pin actions by SHA, set `timeout-minutes`, and copy the existing concurrency block.

### S7. The second lockfile departs from the single-lock policy
- **Brief (§3, §8 step 7):** `requirements-lock.txt` lists exact tested versions.
- **Actually:**
  - The repo has a single-lock policy (`README.md:243`; `docs/environment-setup.md:28-31`), and ADR 005 removed the pip-format mirrors of `uv.lock`. A lock for a different package is not a mirror, but it still departs from that policy.
  - `lock-check.yml` watches only the root `pyproject.toml` and `uv.lock`.
  - A workspace lock would move `tessera_embeddings` from icechunk 2.1.1 to 2.3.0 and from zarr 3.2.1 to 3.4.1. That result is approximate, because download.pytorch.org was blocked.
  - The numcodecs/pcodec pair differs between the root lock (0.16.5 / 0.3.6) and a fresh resolve (0.17.0 / 1.0.4). Both pairs decode the scales.
- **Evidence:** `lock-check.yml:4-8,21-22`; `context_docs/decisions/005-multi-lock-file-strategy.md:11-36`; `uv.lock` pins.
- **Change:** Decide between the two options below and record the choice in an ADR. Either way, do not hand-copy versions from `uv.lock`.
  - **Standalone:** Generate the lock with `uv pip compile tessera_apps/pyproject.toml --python-version 3.12 -o tessera_apps/requirements-lock.txt` (82 pins), add a CI drift check, and add `tessera_apps/**` to the lock-check paths.
  - **Workspace:** Add `[tool.uv.workspace] members`, add `tessera_apps/pyproject.toml` to the lock-check paths, and accept the version bump.

### S8. The `[validate]` extra lists the wrong packages, and rio-cogeo is missing
- **Brief (§3; §5; §8 step 2 `rio cogeo validate`):** `[validate]` pulls in `pystac-client` and `planetary-computer`.
- **Actually:**
  - None of the §5 sources uses STAC or Planetary Computer. NIFC is ArcGIS REST, EFFIS is WFS, Hansen is HTTPS GeoTIFFs on GCS, GMW is on Zenodo, and Natural Earth is a zip on S3.
  - planetary-computer's last release is 1.0.0 (2023-07-05), and it pulls in pydantic, pytz and requests.
  - Hansen 1.11 is reportedly also in NASA MAAP STAC (unverified).
  - `rio cogeo validate` needs `rio-cogeo`, which the brief does not list.
  - pyarrow, scikit-image and scikit-learn already arrive transitively through geotessera.
- **Evidence:** PyPI `planetary-computer` JSON; the §5 table; geotessera 0.9.0 to 0.11.0 METADATA.
- **Change:**
  - Make `[validate]` = `requests` or `httpx`, plus `owslib` if WFS stays. Read Hansen through rasterio `/vsicurl/`.
  - Drop `pystac-client` and `planetary-computer`.
  - Add `rio-cogeo` to a dev/test extra, or validate with GDAL.
  - Declare `pyarrow` and `scipy` directly. Implement Otsu in numpy or add `scikit-image`.

### S9. The Caldor AOI crosses into zone 11N
- **Brief (§4.1 demo; §8 checkpoint 1):** "Caldor Fire, El Dorado County". Checkpoint 1 "reads the Caldor AOI for 2020".
- **Actually:**
  - Hansen 2021 loss continues east of 120W, the 10N/11N boundary, onto the Freel Peak, Saxon Creek and Cold Creek slopes NE of Christmas Valley (38.77-38.94N, 120.00-119.93W). That is 12,835 px, about 773 ha, followed by a gap before the separate Tamarack scar.
  - At 38.85N, both 10N and 11N have finite 2022 scales from 120.10W to 119.90W.
  - geotessera's `read_region` serves only the centre zone and warns about it. `read_region_quantized` and `iter_region` do the same without warning.
  - The fire burned in El Dorado, Amador and Alpine counties.
- **Evidence:** Hansen v1.13 tile `40N_120W` with lossyear == 21. Live Icechunk scales. `docs/global-store.md:63-66` (take each point from its nominal group). geotessera `store.py:1265-1381`; `SKILL.md:61-64`. The NIFC perimeter could not be fetched.
- **Change:**
  - Compute the zones from the perimeter bounds, and in checkpoint 1 and the demo, mosaic by nominal zone.
  - Have `read_bbox` split the AOI by zone, and do not use `GeoTesseraZarr.read_region` for multi-zone AOIs.
  - Fix the county text.

### S10. Wildfire validation: the AOI definition and the perimeter semantics
- **Brief (§4.1 Validation; §8 checkpoint 3; §10):** Report IoU and P/R against the NIFC perimeter and stop if IoU < 0.6. The demo uses `--aoi data/fixtures/caldor.geojson`.
- **Actually:**
  - The brief does not say whether `caldor.geojson` is the perimeter or a box.
  - If it is the perimeter, nothing outside it is ever classified. Precision is then inflated by construction, and Otsu also runs only inside it.
  - NIFC perimeters are outer boundaries, "not adjusted for unburned areas within the fire perimeter". A pixel-level mask therefore has an IoU ceiling well below 1, and the 0.6 rule may trip because of how the reference is defined.
- **Evidence:** NIFC metadata from web search snippets (the service itself was blocked).
- **Change:**
  - Make the demo AOI the perimeter's bbox buffered by at least 2 km, and compute all metrics over that AOI.
  - Alongside IoU, report recall inside the perimeter and the fraction of the burned mask that falls outside it, and state the caveat in `validation.json`.
  - Optionally add MTBS as a US reference, or make the stop rule "IoU < 0.6 and recall inside the perimeter < X".

### S11. The wildfire recovery index at 0.9 stays at zero
- **Brief (§4.1 step 4):** Recovery is the fraction of burned pixels whose similarity to pre-fire exceeds 0.9.
- **Actually:**
  - In block A, the median cosine of burned pixels to their 2020 values is 0.559 (2022), 0.586, 0.613 and 0.603 (2025). The fraction above 0.9 is 0.000, 0.0001, 0.0016 and 0.0032.
  - Block B gives 0.000-0.004, and its median is flat.
  - Even unburned pixels compared with 2020 reach only a 2025 median of 0.886-0.907 (p5 0.68-0.71). About half of unburned land fails the 0.9 test, so the index can never approach 1.
- **Evidence:** Live 10N blocks A (-120.36, 38.69) and B (-120.15, 38.80), 2019-2025. `context_docs/inference/v1.1-against-the-published-store.md:62-63`.
- **Change:**
  - Use a continuous metric normalised against an unburned reference in the same regime, for example `R_t = (s_t - s_post) / (s_ref,t - s_post)`, and also report the median-similarity trajectory.
  - If a fraction stays, calibrate its threshold to the per-year distribution of unburned pixels.

### S12. Deforest labels: "outside the AOI, inside its bbox" is empty
- **Brief (§4.2 step 1):** Sample 500 points per class inside the AOI's bbox but outside the AOI. The demo is a 20 x 20 km box.
- **Actually:**
  - For a box AOI, that region is empty apart from reprojection slivers.
  - With a 10 km ring, the 40 km box touches 256-289 chunks. The ring alone is about 175-225 chunks, against 64-81 for the AOI. 1,000 random points touch about 99% of them, roughly 1.5-1.9 GB per year.
  - Training on Hansen-derived labels and then validating against Hansen plausibly inflates agreement. This was not measured.
- **Evidence:** Arithmetic using 8.65 MB per chunk (`docs/global-store.md:105`).
- **Change:**
  - Use an explicit buffer ring (5-10 km) with cluster sampling inside a capped number of chunks.
  - Prefer labels from a source independent of the validation layer, or require a spatially held-out split and state it in the report.

### S13. Hansen: version, validation window and the year_a forest label
- **Brief (§5 deforest row; §4.2 step 1 and Validation; §8 checkpoint 4):** Hansen GFC v1.11. Validate against "Hansen loss for `year_b`" ("Rondonia 2022").
- **Actually:**
  - GFC-2025-v1.13 (loss years 2001-2025) is current. v1.11 and v1.12 still exist, and v1.11 covers the 2019-2022 demo, so the version only matters for a `year_b` of 2024-2025.
  - Conversion between `year_a` and `year_b` corresponds to lossyear in (`year_a`, `year_b`], which is codes 20-22 for the demo. Using lossyear == 22 alone keeps only about 31% of the 2020-2022 loss in the Porto Velho box (0.45% against 1.44%), so conversions in 2020-21 count as false positives.
  - GFC has no per-year forest layer. Its layers are treecover2000, gain (2000-2012), lossyear, datamask, first and last. In the Porto Velho box the derived 2019 forest proxy is 35.5%, against 48.1% with treecover2000 >= 30.
  - The same bucket holds annual GLAD layers for 2015-2024 (`TCH_2015_2024/`, `Annual_Layers_2015_2024/TCH/{year}/`, `LCLU_2015_2024_v01/{year}/`). They are a better label source for `year_a`, but have no 2025.
- **Evidence:** GCS JSON listing of `gs://earthenginepartners-hansen`; v1.13 `download.html`; windowed reads of `00N_070W`.
- **Change:**
  - Default to v1.13, expose the version as a constant or CLI option, and record it in the report.
  - Use the reference `(lossyear > year_a - 2000) & (lossyear <= year_b - 2000)`.
  - Optionally report `year_a`'s own loss as an edge class.
  - Exclude pixels with loss in or before `year_a` from the `year_a` forest labels.
  - Spell out the forest derivation and its threshold (or use the GLAD annual layers), and add the caveat that tree cover is not forest.

### S14. The Rondonia demo AOI has little signal
- **Brief (§4.2 demo):** A 20 x 20 km box "near Porto Velho", 2019 to 2022.
- **Actually:**
  - A box centred on the city (bbox -63.9909,-8.8498,-63.8091,-8.6702) has 1.44% loss in 2020-2022. It is 5.5% water and 35.5% forest by the 2019 proxy; the rest is mostly land cleared earlier.
  - Boxes on the deforestation front have 11 to 15 times more:
    - Centred on 8.670S 63.110W: 22.1% loss (7.3% in 2022 alone), about 61% forest in 2019.
    - Centred on 9.750S 64.190W: 16.7%.
    - Centred on 9.570S 64.370W: 16.7%.
  - All are in Rondonia and in zone 20S.
- **Evidence:** Windowed reads of Hansen v1.13 `00N_070W`; point-in-polygon test against Natural Earth admin-1.
- **Change:** Pin the demo to explicit coordinates, for example a 20 km box centred on 8.67S 63.11W, about 87 km east of Porto Velho. Confirm it is mostly forest in 2019 before freezing the fixture.

### S15. GMW v3 has no 2024 epoch
- **Brief (§5 wetland row; §4.3 Validation and demo; §8 checkpoint 5):** "Per-year extent polygons 1996 to 2020, plus later updates". The demo uses `year_a=2018` and `year_b=2024`.
- **Actually:**
  - v3 has 11 epochs (1996, 2007-2010, 2015-2020) and nothing after 2020.
  - Annual 1985-2025 coverage exists only in GMW v4.1.12 (Zenodo record 21346457): 41 bands with band = year - 1984 (2018 is band 34, 2024 is band 40), about 30 m, EPSG:4326, 1-degree tiles. This comes from web search only.
  - One secondary source mentions per-year v4.1 vector GeoPackages.
  - The survey rated this a blocker. Verification downgraded it, because v3 has a 2018 epoch and checkpoint 5 can still report `year_a` recall.
- **Evidence:** `samapriya/awesome-gee-community-datasets` `docs/projects/mangrove.md` (line 3 and lines 36-46). The JAXA and Zenodo pages, found via search, were blocked here.
- **Change:**
  - Choose one:
    - (a) Validate both years against v4.1.12 raster bands 34 and 40, with a raster loader.
    - (b) Keep v3 and move the demo to 2018 and 2020.
  - Never mix versions across the two years, and name the version in `validate.py`.
  - Fix the §5 row to "11 epochs 1996-2020 (v3); annual 1985-2025 (v4.1, raster)".
  - Hold the seed stands out of the validation mask.

### S16. The restore demo fallback is too recent; also define the planting-year limits
- **Brief (§4.4 demo):** A 2019 or 2020 planting, so at least five post-planting years. The fallback is a synthetic site inside the Caldor replanting areas.
- **Actually:**
  - Caldor replanting began in 2022 at the earliest. USFS project 61602, "Caldor Fire Reforestation Along Mormon Emigrant Trail and Silver Fork Road" (about 200 acres), had its decision signed in February 2022, and volunteer planting started in spring 2022. That leaves at most 3-4 post-planting years, all dominated by the 2021 fire.
  - 2017 is weak. A live Rondonia 20S chunk at (-8.9, -63.7) had few or no valid pixels in 2017 (one read gave 0%, another 12.5%), even though 20S lists 2017 as complete. A Caldor chunk had a median of about 40-43 S2 observations in 2017, against 92-119 in later years.
- **Evidence:** Web search (fs.usda.gov/r05/eldorado/projects/61602; sugarpinefoundation.org); live reads.
- **Change:**
  - Drop the Caldor fallback.
  - Prefer a USFS FACTS "Silviculture Reforestation (All Years)" polygon for tree planting with `DATE_COMPLETED` in 2019 or 2020, for example Eldorado NF plantings in the 2014 King Fire scar (10N). Otherwise use a synthetic site and say so.
  - Refuse `planted <= 2017` and warn for `planted = 2018`.

### S17. The restore control filter is undefined
- **Brief (§4.4 step 1):** Exclude ring pixels outside "the site's own `planted - 1` distribution".
- **Actually:**
  - "Distribution" is undefined, and a covariance-based reading inherits the problems in B6.
  - For a circular 50 ha site the ring is about 29.5 km2, touching 9-16 chunks, up to about 140 MB per year.
  - A cosine filter at the site's own 5th percentile gave held-out recall of 0.89 with random seeds at n=30. With contiguous site pixels it gave only 0.006-0.31, so it would discard most ring pixels.
- **Evidence:** `skeptic_mr/maha2.py` patch-seed test; the ring arithmetic.
- **Change:**
  - Define the filter as cosine to the site's `planted - 1` mean, with the threshold calibrated on spatially held-out site pixels.
  - Require control and site to share the radar regime in every year compared (B5).

### S18. Large AOIs: memory and strip shape
- **Brief (§6 Fail loud; §3 `iter_strips`):** AOIs up to 10,000 km2 run without a flag. `iter_strips` streams 512-row blocks aligned to 256.
- **Actually:**
  - 10,000 km2 is 1e8 px per year: 12.8 GB as int8, 51.2 GB as float32.
  - Five to six years is about 66-80 GB. That is roughly 5-6 minutes at 219 MB/s cross-region, or about 1.8 h at 100 Mbit/s.
  - A 512 x 10,000 strip is 655 MB as int8 and 2.6 GB as float32.
  - Caldor-sized box (about 45 x 49 km, 4,582 x 4,913 px): about 400 chunks, about 3.5 GB per year. One 512-row float32 strip of it is about 1.29 GB.
  - Cosine on int8 cast to float32 is exact (max difference 3.6e-7), because the scales are strictly positive.
- **Evidence:** `docs/global-store.md:103`; `docs/single-vs-global.md:467`; `quantization.py:84-89`; live check.
- **Change:**
  - Stream 2-D blocks (for example 512 x 2048, aligned to the reader's chunk shape) and read every year you need for each block.
  - Compute cosine on int8. Dequantise only for means and covariances, after masking NaN scales.
  - Write rasters block by block.
  - Express the size guard in estimated bytes (chunks touched x 8.65 MB x years), not km2.

### S19. Idempotent skip can leave stale outputs
- **Brief (§6; §4 output names):** Skip any output that already exists unless `--force`.
- **Actually:**
  - Only the `.tif` filename carries the years.
  - A rerun with different years or thresholds would skip the existing summary, report, preview and validation files, leaving the old ones in place.
  - A crash mid-write leaves a partial file that the next run skips.
- **Evidence:** The §4 output list.
- **Change:**
  - Put an args hash (or the years and thresholds) in every filename, or store it in `report.json` and compare before skipping.
  - Write to a temp file and rename it atomically.

## Notes

- **N1. §2, pixel read cost.**
  - Brief: "about 9 MB".
  - Actually: measured on the wire, about 8.1 MB. That is 7.99-8.05 MB of embeddings, 57-80 KB of scales and 2 x 1,028 B of shard index. In memory it is 8.65 MB, about 9 MB with the obs counts, and 0 B for elided (ocean) chunks. `docs/global-store.md:105` mislabels 8.65 MB as the on-the-wire figure.
  - Change: say this, and base estimates on chunks that exist.
- **N2. §2, northing.**
  - Brief: "usually descending".
  - Actually: always descending (`zone_grid.py:169-171`). `sel(northing=slice(lo, hi))` silently returns 0 rows, while `slice(hi, lo)` returned 4,582.
  - Change: write "always descending, row 0 is north; slice(north, south)". Assert this on open, and raise if a selection comes back empty.
- **N3. §2, radar-free pixels.**
  - Brief: "roughly a fifth", identified by the three counts.
  - Actually: radar-free = `isfinite(scales) & (s1_asc_obs_count + s1_desc_obs_count == 0)`. `s2_obs_count` is not needed, and counts are written over refused tiles too.
  - Store `runs` attrs aggregated over 120 groups give 26.6%, 26.5% and 26.0% for 2022-2024, and 3.5-5.7% in other years. 10N is about 6% in 2021, 2022 and 2025.
  - The repo disagrees with itself: "about a quarter" (`docs/global-store.md:185`) against "about a fifth" (`docs/single-vs-global.md:85,345`). `campaign-cost-model.md:879,889` gives 81% radar coverage and 6.8% radar-free pixel-years.
  - Change: reword. A zone summary is at `zone.attrs["runs"][str(year)]["radar_coverage"]["s1_free_pct"]`.
- **N4. §2 and §10, README heading.**
  - Brief: "Quickstart, data access".
  - Actually: the heading at `README.md:47` uses an em dash, not a comma; its anchor is `#quickstart--data-access`, linked from `README.md:30`. A separate `## Quickstart` at `README.md:247` covers the generation pipeline (Earthdata login).
  - Change: cite the anchor, add the pointer inside that section without renaming the heading, and write the pointer without an em dash.
- **N5. §6 and §10, `network` marker.**
  - Actually: the repo has no `network` marker. Its equivalent is `integration` (deselected by default).
  - Verified: declaring `network` with `addopts = "-m 'not network'"` in `tessera_apps/pyproject.toml` works, both from inside the package and as `pytest tessera_apps/tests` from the root. Without it, a plain `pytest` would hit S3.
  - Precedent for a live-store test: `tests/integration/test_published_store_access.py:8-11,38-44` (TESSERA_TEST_PUBLISHED_STORE=1 opt-in; clears `AWS_*`).
  - Change: declare the marker, or use `integration` with an env opt-in. In the live tier, clear `AWS_*` env vars. The root Hypothesis profiles do not apply.
- **N6. §3 and §7.1, root sdist.**
  - Actually: the root `tessera_embeddings` sdist would include `tessera_apps/` (7 files in the scratch build). It already ships 476 files (3.6 MB) of the repo, so the only real cost is the size of the executed notebooks.
  - Change: add `[tool.hatch.build.targets.sdist] exclude = ["tessera_apps", "demo"]` to the root `pyproject.toml`.
- **N7. Repo docs.**
  - Actually: the README "Repo structure" block (`README.md:455-488`) and `docs/README.md` do not list the new package. The repo writes CLIs with argparse `main(argv) -> int`; typer (0.25.1) and rich (15.0.0) are only transitive in `uv.lock`.
  - Change: add `tessera_apps/` and `demo/` to the docs, write a short ADR (package boundary, release tags, lockfile), and name a maintainer in `tessera_apps/README.md`.
- **N8. §5, Hansen tiles.**
  - Actually: tiles are not COGs (uint8, LZW, strips of 1 x 40000, no overviews). The v1.12 and v1.13 `lossyear` layers have 280 tiles, with ocean tiles dropped. v1.12 `lossyear.txt` still lists 504 tiles, and v1.13's lists 280 (`wc -l` prints 279).
  - Change: treat a 404 as "no loss" after checking `datamask`. Read windows via `/vsicurl/`, cache the clipped AOI, and reproject to the UTM 10 m grid with nearest-neighbour resampling.
- **N9. §4.3 step 5, coastline.**
  - Actually: Natural Earth 10m coastline segments in the Everglades AOI have a median length of 1.17 km (p90 3.38 km), which is the same scale as the 1 km buffer.
  - Change: label the metric "approximate", or use OSM land polygons, GSHHG full resolution, or a TESSERA-derived water edge.
- **N10. §4.3 demo, storms.**
  - Actually: Irma (September 2017) caused about 10,760 ha of mangrove dieback in SW Florida (Lagomasino et al. 2021), and Ian (2022) hit the north edge of the AOI. Because `year_a=2018` is post-Irma, gain may be recovery.
  - Change: say so in notebook 03.
- **N11. §4.1 step 1, fire year as `year_b`.**
  - Actually: the median distance of burned pixels from 2020 is 0.174 in 2021 against 0.441 in 2022 (block A), and 0.129 against 0.394 (block B).
  - Change: default `year_b` to the fire year + 1, and warn when the fire year is passed.
- **N12. §4.1 step 2, Otsu.**
  - Actually: on block A the threshold falls from 0.281 to 0.211, 0.177 and 0.160 at burned shares of 10%, 5% and 2%, flagging 17-30% of the non-burned pool. This is a semi-synthetic, pessimistic mix.
  - Change: log the threshold and a bimodality check. When the high mode is small, fall back to the 99th percentile of a stable reference pair.
- **N13. §4.2 step 4, median cross-check.**
  - Actually: half of all pixels exceed the median by definition. At Caldor, stable inter-annual distance has a median of 0.03 and a p99 of 0.15.
  - Change: use the 95th percentile of distance for pixels classed as forest in both years, in the same regime.
- **N14. §6, four-zone guard.**
  - Actually: a compact AOI on the equator at a zone edge touches 4 groups, and only elongated AOIs under the 10,000 km2 cap reach 5. The Porto Velho demo is about 970 km from the equator and about 230 km from 66W.
  - Change: count zones by nominal band and hemisphere, and test the equator and zone-edge cases.
- **N15. §3 lock, numcodecs and pcodec.**
  - Actually: PCodec is only needed on the Icechunk backend. The Zarr copy's scales use Blosc.
  - Change: generate the lock with `uv pip compile` so the two stay paired.

## Facts the build session needs

### Store and access
- **Icechunk store:** `s3://tessera-embeddings/v1.1/dclimate.icechunk`, region `us-west-2`, anonymous. Hosted by AWS Open Data, branch `main` only, 1,070 tags. It opened in 0.85 s with no credentials.
  ```python
  storage = icechunk.s3_storage(bucket="tessera-embeddings", prefix="v1.1/dclimate.icechunk",
                                region="us-west-2", anonymous=True)
  repo = icechunk.Repository.open(storage)          # pass NO config
  session = repo.readonly_session(branch="main")    # or snapshot_id="..."
  root = zarr.open_group(session.store, mode="r")
  ```
- **Config:** Do not pass a config to `Repository.open`, because it replaces the saved reader config wholesale (`README.md:78-80`, `global_store.py:83-91`). To enlarge the chunk cache, start from `icechunk.Repository.fetch_config(storage)` (`tests/integration/test_published_store_access.py:95-101`).
- **Snapshots:** Main head is `B6AFRJGX5XQC0A716PA0`; its parent and the final data commit is `QR7F41A6WYZ03VC92T6G`. The pixel-corner transform origin applies from `B6AFRJGX...` on; older snapshots used the pixel centre.
- **Tags:** `zone-<ZONE>-<YEAR>` exists for each completed cell. `year-<YEAR>-complete` exists for 2022-2025 only, so do not use year tags to decide which years exist (`reading-the-published-store.md:111-113`).
- **Source Coop Zarr copy:** `https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate`. Direct bucket endpoint: `https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tessera/tessera/zarr/v1.1-dclimate`, which geotessera 0.11 uses by default.
- **Tile registry:** `s3://tessera-embeddings/v1.1/dclimate.registry/parts/zone=<ZONE>/year=<YEAR>/<run_id>.parquet`. Take the latest `assembled_at` per (zone, year, tile), and read `refused_px` alongside `embedded` (`docs/global-store.md:219-238`).
- **No tessera_embeddings dependency:** The package is not on PyPI (404), so a PyPI-published `tessera_apps` cannot import it. Mirror its helpers instead.
- **Reading attrs:** Use `zarr.open_group` for attrs (about 0.3 s per group), not `xr.open_zarr`, which takes 1.4 s for 10N because it materialises the 933,888-element northing coordinate.

### Root attributes
- `checkpoint_id="tessera_v1_1_aws_encoder.pt"`
- `geoemb:model="https://geotessera.org/model/1.1"`
- `geoemb:dimensions=128`. The v1.1 model emits 192 values; the first 128 are stored.
- `geoemb:data_type="int8"`
- `geoemb:gsd=10.0`
- `geoemb:spatial_layout="utm_zones"`
- `geoemb:build_version="0.1.0"`
- `optical_min_obs=15`
- `geoemb:source_data`
- `geoemb:quantization={method: per_pixel_scale, scale: {array_name: scales, nodata: NaN}}`
- The Zarr copy adds `geotessera:source_snapshot`, `geotessera:source_store` and `geotessera:migration_id`.

### Zone attributes
- **Keys:**
  - `crs`
  - `years_complete`
  - `zone_scheme="utm_6deg_nominal"`
  - `time_convention="calendar_year"`
  - `layout="global"`
  - `proj:code`, `proj:projjson`, `proj:wkt2`
  - `spatial:bbox`, `spatial:dimensions`, `spatial:registration`, `spatial:shape`, `spatial:transform`, `spatial:transform_type`
  - `zarr_conventions`
  - `runs`: per year, with `assembled_at`, `code` (`commit`), `input_coverage`, `optical_skips`, `radar_coverage` (`s1_free_pct`) and `run_id`
- **Group names:** 120 groups, `01N`..`60N` and `01S`..`60S`.

### Arrays per zone group (10N shown)
| Array | Shape | dtype, fill | Chunks / shards / codec |
|---|---|---|---|
| `embeddings` | (9, 933888, 67584, 128), dims (time, northing, easting, band) | int8, fill 0 | chunks (1,256,256,128), shards (1,2048,2048,128), Zstd |
| `scales` | (9, N, E) | float32, fill NaN | inner (1,256,256), PCodec |
| `s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count` | (9, N, E) | uint16, fill 0 | |
| `s2/s1_asc/s1_desc_month_covered` | (9, N, E, 12) | int8 with attr `dtype="bool"` | |

- **Coordinates:** `time`, `time_bnds`, `northing`, `easting`, `band`, `month` (1..12).
- **Time:** values run from `2017-01-01` to `2025-01-01` as int64 ns. Time index = year - 2017.

### Grid
- Pixel size is 10 m. The transform origin is the pixel corner; coordinate arrays hold pixel centres.
- **North groups:** `spatial:transform` `[10,0,163840,0,-10,9338880]`, shape (933888, 67584), easting 163,840-839,680, northing 0-9,338,880.
- **South groups:** `[10,0,163840,0,-10,10014720]`, shape (890880, 67584), northing 1,105,920-10,014,720 (false northing 10,000,000). A south group reaches about 14.7 km north of the equator.
- `row = int((top - y) // 10)`, `col = int((x - 163840) // 10)`.
- Northing runs 9338875 down to 5 in 10N.
- Shard pitch is 20,480 m and an inner chunk is 2,560 m.
- **Routing:** band = `floor((lon + 180) / 6) + 1`, clamped to 1..60. Hemisphere is N if lat >= 0. There are no MGRS exceptions.
- **Overlap:** Adjacent groups overlap by up to a shard. A mosaic takes each point from its nominal group.
- **Units:** 1 px = 0.01 ha, so 1 ha = 100 px.

### Read costs and value ranges
- **Chunk-year size:** one 256x256 chunk-year is 8,388,608 B of embeddings plus 262,144 B of scales, 8.65 MB in memory. It is about 8.1 MB on the wire, and elided chunks cost 0.
- **Throughput (docs):** a 4096^2 block takes 2.8 s in-region and 9.8 s cross-region. A single point takes 325 ms in-region and 486 ms cross-region (`docs/global-store.md:98-103`).
- **Dask cost:** reading one pixel through Dask takes about 3 s and nearly 2 GB peak, against about 0.2 s and under 200 MB with `chunks=None` (`docs/single-vs-global.md:318-323`).
- **Typical cosines at Caldor:**
  - Two random pixels: median 0.832 (p5 0.589).
  - A pixel against the block mean: median 0.916.
  - Stable forest, year to year: 0.96-0.97.
  - 2020 to 2022 distance is bimodal, with an Otsu threshold of 0.281.
- **Unfilled zone-years (14 of 1,080):**
  - 03S: 2017, 2018, 2020, 2021
  - 08S: 2017, 2018
  - 09S: 2017
  - 24N: 2017
  - 26S: 2017
  - 29S: 2017
  - 31S: 2018-2021

### Demo AOIs
| AOI | Zone(s) | EPSG | years_complete | Hansen tile(s) | Sample chunk origin (row, col) |
|---|---|---|---|---|---|
| Caldor | 10N, plus 11N east of 120W | 32610 (32611) | 2017-2025 both | `40N_130W`, `40N_120W` | 10N (505344, 56576) |
| Rondonia | 20S | 32720 | 2017-2025 | `00N_070W` | 20S (100864, 26880) |
| Everglades / Ten Thousand Islands | 17N | 32617 | 2017-2025 | `30N_090W` | 17N (647936, 28928) |

- **Other useful chunks:**
  - Calaveras stable forest: 10N (509952, 57088).
  - Radar-gap test chunks, 100% radar-free in 2022-2024: 47N (257536, 38912), 52S (311296, 13824), 53S (278016, 53760).
- **Caldor block:** 10N rows 505088-505600, cols 56320-56832.
- **Caldor fire:** ignition 2021-08-14, contained 2021-10-21, 221,835 acres.
- **Caldor radar:** median asc/desc counts are 61/60 in 2020, 31/29 in 2022 and 52/52 in 2025.
- **Rondonia radar:** ascending count is 0 in every year; descending is 59 in 2021 and 29 in 2022.
- **Everglades radar:** ascending 25-32, descending 0-3.
- **Radar-free share:** 0% at all three demo sites.

### Repo helpers to mirror, not import
- `storage/global_store.py:75-101`: `open_global_repo(store_path: str, *, get_credentials=None, region: str | None = None, scatter_initial_credentials: bool = False, anonymous: bool = False) -> icechunk.Repository`. `zarr_store.open_store` (`zarr_store.py:1072-1079`) has no `anonymous` parameter.
- `zone_grid.canonicalize_zone(zone: str) -> str` turns `"7s"` into `"07S"` (`:142-161`).
- `zone_grid.ZONES` and `zone(name) -> ZoneSpec(epsg, hemisphere, utm_zone, easting, northing)`, with `.group_name`, `.crs`, `.width` and `.height` (`:54-82,133-139`).
- `easting_coords(spec)` (ascending) and `northing_coords(spec)` (descending) return pixel centres (`:164-171`).
- `ingest/land_mask.zone_for_cell(lon, lat) -> "NNH"` (`:166-176`).
- `shard_writer.read_years_complete(node: zarr.Group) -> list[int]` (`:551-554`).
- `published_store.calendar_years(group) -> list[int]` (`:171-184`).
- `published_store.live_shards(session, zone, var="scales") -> dict[time_index, frozenset[(shard_y, shard_x)]]` reads manifests only (`:187-209`).
- `published_store.shard_pixel_window(shard, shape, *, shard_px=2048)` (`:212-222`). `COVERAGE_VAR = "scales"`.
- **Trap:** `ingest/roi.py:252-266` `determine_utm_zone()` applies the Norway and Svalbard MGRS exceptions. Do not use it for store routing.

### geotessera 0.11.0 API
- **Constructor:** `from geotessera import GeoTesseraZarr`; `GeoTesseraZarr(store_url=DEFAULT_STORE, cache_dir=None, cache_max_size=None, via_gateway=None)` (`store.py:940-946`). For Icechunk, pass `store_url="s3://tessera-embeddings/v1.1/dclimate.icechunk"`. `store_url` also accepts a zarr `Store` instance, which is how to plug in a byte counter on the Zarr path.
- **Reads:**
  - `sample_points(coords, year, *, progress=True, cross_zone=True, search_px=1, depth=None) -> (N,128) float32` with NaN rows.
  - `sample_at(lon, lat, year, ...)`.
  - `probe(lon, lat, year) -> (vec, status)`.
  - `read_region(bbox, year, *, depth=None) -> (H,W,128) float32, Affine, crs`. Centre zone only.
  - `read_region_quantized(...) -> int8, scales, Affine, crs`.
  - `iter_region(bbox, year, *, depth=None, strip_rows=512)`.
  - `read_patch(lon, lat, year, size_px, *, depth, dst_crs, resampling)`. Merges across zones with nearest-neighbour.
  - `open_zone(*, zone|lon|bbox) -> xr.Dataset` with a `.tessera` accessor.
  - `export_geotiffs(...)`.
- **Attributes:** `years`, `dataset`, `model_version`, `build_version`, `n_bands`, `depths`, `url`.
- **IcechunkStore:** `geotessera.icechunk.IcechunkStore(location, *, branch="main", snapshot_id=None)`, with `.session`, `.root`, `.groups`, `.years`, `.zones()` and `.incomplete_years()`. On a `GeoTesseraZarr` it is `gt._icechunk`, a private attribute.
- **Other helpers:** `geotessera.registry.zarr_store_url(version="v1.1", variant=None)`, `geotessera.registry.icechunk_dataset("v1.1") -> (icechunk_url, registry_url)`, `geotessera.store.TesseraAccessor.dequantise(emb_int8, scales)`, `geotessera.__version__`.
- **Skill rules:**
  - Do not add a retry layer; obstore already retries with backoff (`store.py:78-87`).
  - Do not pass `progress=`. Use `logging.getLogger("geotessera").setLevel(logging.INFO)` instead.
  - `cache_dir` is ignored for Icechunk.
- **Byte counter:** Subclass `zarr.storage.WrapperStore` and override `get`, `get_partial_values` and the async generator `get_ranges(*a, **k)`, which yields groups of `(idx, Buffer)`. Sum `len(buf)`.

### Reference dataset endpoints
- **Hansen v1.13:**
  - URL pattern: `https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/Hansen_GFC-2025-v1.13_{layer}_{NN}{N|S}_{EEE}{E|W}.tif`.
  - Layers: `treecover2000`, `gain`, `lossyear`, `datamask`, `first`, `last`.
  - Tile name is the top-left corner: LAT = `ceil(lat/10)*10`, LON = `floor(lon/10)*10`.
  - Format: EPSG:4326, 0.00025 deg pixels, 40000 x 40000 per tile, uint8. `lossyear` 0 means no loss and N means 2000+N, with values 1..25.
  - Demo tile sizes: `00N_070W` lossyear 55,466,434 B and treecover2000 204,963,567 B; `40N_130W` lossyear 23,848,304 B; `30N_090W` lossyear 34,622,142 B.
  - Through the sandbox proxy, `/vsicurl/` needs `GDAL_HTTP_PROXY` and `CURL_CA_BUNDLE=/root/.ccr/ca-bundle.crt`.
- **GLAD annual layers:** in the same bucket, `TCH_2015_2024/TCH_filter_{year}/`, `Annual_Layers_2015_2024/TCH/{year}/` and `LCLU_2015_2024_v01/{year}/`.
- **NIFC (unverified here):**
  - History layer: `https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/InterAgencyFirePerimeterHistory_All_Years_View/FeatureServer/0`. Its fields include `INCIDENT`, `FIRE_YEAR` or `FIRE_YEAR_INT`, `GIS_ACRES`, `IRWINID` and `UNQE_FIRE_ID`.
  - WFIGS layer: `.../WFIGS_Interagency_Perimeters/FeatureServer/0`, with fields `poly_IncidentName` and `poly_GISAcres`.
  - Example query: `query?where=UPPER(INCIDENT) LIKE '%CALDOR%' AND FIRE_YEAR_INT=2021&outFields=*&returnGeometry=true&outSR=4326&f=geojson`.
  - Discover the field names at runtime with `{layer}?f=json`.
- **EFFIS:**
  - WFS layer `effis.nrt.ba.poly`, not `modis.ba.poly`.
  - Burn severity GeoTIFFs: `https://effis-gwis-cms.s3-eu-west-1.amazonaws.com/effis/applications/data-and-services/severity_{2018..2023}.tiff` (79-107 MB each).
- **GMW:**
  - v3: Zenodo DOI `10.5281/zenodo.6894273`, vectors `gmw_v3_{year}_vec` for 1996, 2007-2010 and 2015-2020.
  - v4.1.12: Zenodo record 21346457, `band = year - 1984`, value 1 = mangrove (from web search).
- **Natural Earth:**
  - `https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip` (3,069,451 B, VERSION 5.0.0-pre9).
  - 50m version: `.../50m_physical/ne_50m_coastline.zip` (455,936 B).
  - GeoJSON: `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson`.
- **USFS FACTS (unverified):** `https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_SilvicultureReforestation_01/MapServer/8`. It has a `DATE_COMPLETED` field and accepts `f=geojson` without a key.

### Pins and tooling
- **Recommended floors:**
  - `geotessera>=0.11,<0.12`
  - `icechunk>=2.2,<3`
  - `zarr>=3.3`
  - `numcodecs[pcodec]>=0.16`
  - `rasterio>=1.4.3`
  - `scikit-learn>=1.7.1` (geotessera's floor)
  - `python>=3.12`, tested on 3.12 and 3.13, the same as the repo's classifiers and CI matrix.
- **Fresh Python 3.12 resolve (82 pins):**
  - geotessera 0.11.0
  - icechunk 2.3.0
  - zarr 3.4.1
  - numcodecs 0.17.0
  - pcodec 1.0.4
  - rasterio 1.5.2
  - typer 0.27.3
  - xarray 2026.9.0
  - numpy 2.5.4
  - pystac-client 0.9.0
  - pydantic 2.14.1
  - No source builds were needed.
- **Root `uv.lock`:**
  - icechunk 2.1.1
  - zarr 3.2.1
  - xarray 2026.4.0
  - numpy 2.4.4
  - rasterio 1.5.0
  - numcodecs 0.16.5
  - pcodec 0.3.6
  - ruff 0.15.13
  - mypy 2.1.0
  - pytest 9.0.3
  - hypothesis 6.152.7
  - typer 0.25.1 and rich 15.0.0 (both transitive)
- **Action SHAs used in the repo:**
  - `pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33` (v1.14.2)
  - `actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1` (v7.0.1)
  - `astral-sh/setup-uv@c18668ad3cf93ea998bef934396af7bb5c839dc7` (v10.2.0)
- **Builds:**
  - `uv build tessera_apps` writes to `tessera_apps/dist/`.
  - `uv build --package tessera_apps` (workspace) writes to the root `dist/`.
  - A bare `uv build` builds only `tessera_embeddings`.
- **Style and lint:**
  - Google docstrings, with D100, D104 and D107 exempt.
  - ANN annotations required; line length 120; double quotes.
  - `from __future__ import annotations` is used throughout `src/`.
  - Per-file-ignore globs in `ruff.toml` are anchored at the root (comment at `ruff.toml:97-99`).
- **Architecture rules:** `python -m tessera_embeddings.architecture_tests --source tessera_apps/src/` needs the root env synced, because it imports pydantic. A skeleton passes with 0 violations.
- **`.gitignore`:** `*.geojson` (line 101) and `*.tiff` (line 103) are ignored. `*.tif`, `*.npz`, `*.parquet`, `*.csv` and `*.png` are not.
- **Repo docs:** There is no CONTRIBUTING.md, CLAUDE.md, AGENTS.md, PR template or pre-commit config. Contributing rules are at `README.md:556-592`.
- **PyPI names:** `tessera-apps` and `tessera_apps` are unclaimed on both PyPI and TestPyPI.
- **Survey scratch** (may not exist on the build machine): `/tmp/claude-0/-home-user-Web/a9566209-8a00-5748-8e9b-c1203b2f19ca/scratchpad/`.
  - `methods_check/caldor_block.npz` holds a 512x512 Caldor block for 2019-2023 and 2025, as int8 plus scales.
  - The `probe*.py`, `maha.py`, `otsu_imb.py` and `wrap_probe3.py` scripts are also there.

## Checked and fine
- The store is anonymous, in us-west-2, and opens with no credentials.
- Embeddings are int8 and the true value is embeddings x scales. A NaN scale means no data, and the embeddings fill is 0.
- The `chunks=None` advice holds. "Multi-gigabyte" slightly overstates "near 2 GB".
- `years_complete` is an attribute on each zone group, not the root. A year missing from it is a publication gap.
- Zone groups are `01N`-`60S` with EPSG 326xx and 327xx. They are pure 6-degree bands with no MGRS exceptions.
- Inner chunks are 256x256x128 inside 2048x2048 shards.
- The time axis is 9 calendar-year slots, 2017-2025, with nothing sub-annual.
- The three obs-count layer names are correct.
- `checkpoint_id` is in the root attrs. What it means is covered in S4.
- Stored embedding dimension is 128.
- Quantisation is symmetric per pixel, so cosine is unaffected (max difference 3.6e-7).
- The cited doc sections exist: `docs/single-vs-global.md:258` and `:528`.
- The `/plugin marketplace add ucam-eo/geotessera` and `/plugin install geotessera@ucam-eo` commands match the manifests at `ucam-eo/geotessera@2814c5b`.
- `geotessera` and `GeoTesseraZarr` exist on PyPI, latest 0.11.0, requiring Python >=3.12.
- Both geotessera backends return identical embeddings for the same point and region.
- `numcodecs[pcodec]` exists, and pcodec 1.0.4 decodes the store's scales.
- icechunk and rasterio ship cp312 wheels on all major platforms.
- `python>=3.12` matches the repo, whose CI tests 3.12 and 3.13.
- The repo is a single package with one lock, not a uv workspace.
- A release workflow with trusted publishing and a `pypi` environment exists. The trigger problem is in B3.
- The architecture tests scan only `src/tessera_embeddings`, and a `tessera_apps/src` skeleton gives 0 violations.
- No existing test walks the whole tree, and the root `testpaths` is `tests/` only.
- `lock-check.yml` does not forbid a second lockfile.
- ruff and mypy settings are as described: Google docstrings, ANN, T20, line length 120, `disallow_untyped_defs`, `ignore_missing_imports`.
- No repo rule bans em dashes, so the ban applies only to the new package.
- There is no CONTRIBUTING.md, CLAUDE.md, AGENTS.md, PR template or pre-commit config.
- `tessera_apps` and `tessera-apps` are unclaimed on PyPI and TestPyPI.
- No demo zone (10N, 11N, 17N, 19S, 20S) has a publication gap. All 14 gaps are in remote zones.
- Demo UTM zones: Caldor 10N, Everglades 17N, Porto Velho 20S. The Everglades and Rondonia boxes each fall in a single zone.
- Hansen GFC is public on GCS with no key, uses 10x10-degree tiles named by their top-left corner, and Porto Velho is in tile `00N_070W`.
- Hansen lossyear values run 1..25 in v1.13. The download page's "1-20" is a typo.
- The Natural Earth 10m coastline downloads without a key from the S3 and GitHub mirrors.
- GMW v3 has a 2018 epoch.
- A 2019 or 2020 planting leaves 5-6 post-planting years in the store.
- The Caldor years make sense: 2020 is entirely pre-fire, 2022 is the first full post-fire year, and 2023-2025 exist.
- All three demo sites have radar in every year. Halving radar depth does not shift the embeddings (Calaveras).
- The wildfire change signal is strong: stable cosine is 0.970, the distance distribution is bimodal, and Otsu gives 0.281.
- The concern that EFFIS WFS cannot return past-year perimeters is refuted. `effis.nrt.ba.poly` serves the archive, per two third-party PRs and the EFFIS WMS capabilities. EFFIS hosts were blocked here.

## Could not verify
The survey sandbox could only reach `storage.googleapis.com`, `*.s3.amazonaws.com`, `raw.githubusercontent.com` and PyPI. The `pytest -m network` tier and the validate loaders need a runner with wider egress.

- **Companion notebook:** `TESSERA_community_applications.ipynb` was not available. Its `TesseraReader`, sections 2, 5, 12 and 12.3, and the FastAPI sketch were not checked.
- **NIFC:** the service was blocked. The Caldor perimeter geometry, field names, GIS acres, and whether the perimeter itself crosses 120W were not checked. The 120W crossing is inferred from Hansen.
- **EFFIS:** the WFS GetCapabilities and its layers were not checked first-hand.
- **GMW:** the Zenodo files for v3 and v4.1.12 (names, formats, sizes, per-year vectors) were not fetched. The v4.1 details come from web search.
- **Restoration sites:** Restor open projects and the USFS FACTS reforestation layer were not queried, so no specific 2019 or 2020 planting polygon is confirmed.
- **STAC catalogs:** the Planetary Computer catalog and the NASA MAAP STAC for Hansen were not checked.
- **Natural Earth official page:** the official page and `naciscdn.org` were not reached. Only the mirrors were checked.
- **GitHub API:** it was blocked, so the following were not checked: whether the `pypi` environment and its protection rules exist, whether a PyPI pending trusted publisher exists for `tessera_embeddings`, and `release.yml` run history.
- **Workspace lock:** a full workspace `uv lock` could not run because download.pytorch.org was blocked.
- **icechunk read statistics:** none found in icechunk 2.3.0.
- **Read timings:** the 325 ms per point figure was not re-measured.
- **Plugin install:** the `/plugin` install commands were not actually run.
- **Radar:** the effect of a changed orbit mix (ascending-only against both) was not measured. Only full switches between radar and radar-free were.
- **Method experiments:** the Mahalanobis and Otsu tests used k-means clusters and two Caldor blocks, not real wetland seeds or the full Caldor AOI.
- **Cambridge validation:** the claim that Cambridge validated radar-free embeddings (ADR-013:117-128) has no citation.
- **pcodec:** 1.0.4 against 0.3.6 decoding was checked on one point and one small region only.
- **CI uv version:** not pinned in the workflows. Local tests used uv 0.11.32.
- **Everglades storms:** some of the storm context rests on background knowledge and web search.

## Proposed edits to the brief
1. **§3 Dependencies:** Change to `geotessera>=0.11,<0.12`, `icechunk>=2.2,<3`, `zarr>=3.3`, `numcodecs[pcodec]>=0.16`. Add `pyarrow` and `scipy`, and add `rio-cogeo` to a dev/test extra. Make `[validate]` = `requests` or `httpx`, plus `owslib` if WFS stays; drop `pystac-client` and `planetary-computer`.
2. **§2 and §3 Access path:** Use one backend, the Icechunk store, through `TesseraReader`. If both backends stay, list what the `Reader` normalises. State that the §2 facts describe the Icechunk groups. Replace "defaults to our dataset" with "defaults to the dClimate v1.1 run via the Source Coop Zarr copy". Always pass `store_url` and assert `gt.dataset.name == "1.1-dclimate"`.
3. **§3 and §4 intro:** Every app, on every backend, checks `years_complete` for every year it will read (not only `year_a` and `year_b`), with `zarr.open_group`, before any chunk read.
4. **§6 Fail loud:** Change "before any S3 read" to "before any embedding chunk read".
5. **§6 Cost:** Define "bytes read" as chunk payload bytes from a zarr wrapper store, or as an estimate from the chunks that exist. Express the large-AOI guard in estimated bytes.
6. **§3 `iter_strips`:** Use 2-D blocks aligned to the reader's chunk shape, read all needed years per block, compute cosine on int8, and dequantise only for means.
7. **§6 Reproducible:** Record `session.snapshot_id` (or the Zarr copy's source snapshot), `runs[year].code.commit`, the backend URL, the geotessera version and `gt.dataset.name`. Label `checkpoint_id` as model provenance. Pin a snapshot for the demos.
8. **§6 Idempotent:** Add an args hash to the output names, or compare against `report.json`, and use atomic writes.
9. **New in §4:** Add a per-pixel radar-regime check for every pair of years compared, with the mismatch fraction in the report. It replaces the 20% flag. Keep obs counts in the fixtures.
10. **§4.1:**
    - Default `year_b` to the fire year + 1.
    - Use a recovery metric normalised against an unburned same-regime reference.
    - Log the Otsu threshold and add a bimodality fallback.
    - Make the demo AOI the perimeter bbox plus 2 km.
    - Add recall-inside and fraction-outside to the validation, along with the perimeter caveat.
    - Exercise the 10N/11N multi-zone path in checkpoint 1.
    - Change the county text to "El Dorado, Amador and Alpine counties".
11. **§4.2:**
    - Sample labels from a 5-10 km buffer ring with clustered sampling.
    - Use Hansen v1.13 as an option, with the reference lossyear in (`year_a`, `year_b`].
    - Spell out the `year_a` forest derivation, or use the GLAD annual layers.
    - Set the cross-check cut from the distances of pixels classed forest in both years.
    - Pin the demo AOI to a 20 km box centred on 8.67S 63.11W.
12. **§4.3:**
    - Default to cosine-to-mean.
    - Allow Mahalanobis only with shrinkage and 500 or more pixels.
    - Set thresholds from spatially held-out seed scores.
    - Ship polygon seeds and exclude them from recall.
    - Choose GMW v4.1.12 (bands 34 and 40) or move the demo to 2018 and 2020, and do not mix versions.
    - Label the Natural Earth 1 km metric as approximate.
    - Note Irma in notebook 03.
13. **§4.4:**
    - Drop the Caldor fallback and prefer a USFS FACTS planting from 2019 or 2020 (King Fire scar, 10N).
    - Refuse `planted <= 2017` and warn for 2018.
    - Define the control filter as cosine to the `planted - 1` mean, calibrated on held-out site pixels, with radar regimes matched.
14. **§3 layout and §10:**
    - Put demo AOIs, seeds and default data under `src/tessera_apps/data/` and load them with `importlib.resources`.
    - Keep test npz files under `tessera_apps/tests/fixtures/`.
    - Add the `.gitignore` negation and hatch `artifacts = ["*.geojson"]` (or use `.json`).
    - Say whether the DoD means a checkout or a pip install.
15. **§6 Tooling:**
    - Spell out the ruff config changes, and run mypy as `cd tessera_apps && mypy src/tessera_apps tests`.
    - Declare a `network` marker with default deselection in `tessera_apps/pyproject.toml`, and clear `AWS_*` in the live tier.
    - Add a CI job for 3.12 and 3.13.
16. **§3 Lockfile:** Choose between a standalone `uv pip compile` lock with a drift check and a workspace member, and record the choice in an ADR. Update the `lock-check.yml` paths to match.
17. **§7.1:**
    - Add `release-tessera-apps.yml` on `tessera-apps-v*` tags, with a tag-matches-version check and SHA-pinned actions.
    - Make registering the PyPI pending publisher an explicit human step.
    - Never push a bare `v*` tag.
18. **§2 and §10 repo integration:**
    - Cite the README heading by its anchor `#quickstart--data-access`.
    - Add the package to "Repo structure" and `docs/README.md`.
    - Exclude `tessera_apps` and `demo` from the root sdist.
19. **§2 wording:**
    - Pixel cost: about 8.65 MB in memory, about 8.1 MB on the wire, 0 for elided chunks.
    - Northing is always descending.
    - Radar-free = finite scales and zero S1 counts, about a quarter of embedded pixels in 2022-2024.
20. **§6 Zone guard:** Count zones by nominal band and hemisphere, and test the equator and zone-edge cases.