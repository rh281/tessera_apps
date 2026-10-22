# tessera_apps

Climate applications built on the [TESSERA](https://geotessera.org) v1.1 global embeddings published by dClimate: 128-dimension, 10 m, annual (2017 to 2025) embeddings in an anonymous Icechunk store at `s3://tessera-embeddings/v1.1/dclimate.icechunk` (AWS `us-west-2`). No GPU, no model training beyond scikit-learn linear models.

**Status: checkpoint 7 of the build brief, version 0.1.0.** All four apps are built, and 0.1.0 is the first release. The hosted demo (checkpoint 8) is next. See [docs/checkpoints.md](https://github.com/rh281/tessera_apps/blob/main/docs/checkpoints.md) for the checkpoint reports and numbers.

| App | Question | Status |
|---|---|---|
| `wildfire` | Where did it burn, and how far has each burned pixel recovered? | built, [notebook 01](https://github.com/rh281/tessera_apps/blob/main/notebooks/01_wildfire.ipynb) |
| `deforest` | Where did forest become something else between two years? | built, [notebook 02](https://github.com/rh281/tessera_apps/blob/main/notebooks/02_deforest.ipynb) |
| `wetland` | Given a few known wetland locations, where else is that wetland, and how has it moved? | built, [notebook 03](https://github.com/rh281/tessera_apps/blob/main/notebooks/03_wetland.ipynb) |
| `restore` | Is a restoration site diverging from a matched control the way a working project should? | built, [notebook 04](https://github.com/rh281/tessera_apps/blob/main/notebooks/04_restore.ipynb) |

## Install

```bash
pip install tessera_apps
```

Python 3.12 or 3.13. No AWS credentials and no AWS account are needed: the store allows anonymous reads. GDAL comes inside the rasterio wheel, so there is nothing else to install. `tessera-apps info` opens the store and prints its snapshot and a zone's published years, which is a quick check that it can be reached.

From source, the latest `main`:

```bash
pip install "tessera_apps @ git+https://github.com/rh281/tessera_apps.git"
```

The reproducible route installs the exact versions the release was tested against from the lock (`requirements-lock.txt`, which also pins the test and notebook tools), then the package itself without resolving anything again:

```bash
git clone https://github.com/rh281/tessera_apps.git
cd tessera_apps
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements-lock.txt
uv pip install -p .venv --no-deps -e .
```

With plain pip, the last three lines are `python3.12 -m venv .venv`, `.venv/bin/pip install -r requirements-lock.txt` and `.venv/bin/pip install --no-deps -e .`.

## Command line

Every app is a `tessera-apps` command with the same shape:

```
tessera-apps <app> --aoi AOI --year-a YEAR --year-b YEAR [--out DIR] [--validate] [--force]
                   [--snapshot ID] [--i-know-this-is-large] [--strict-validate] [app options]
```

| Option | What it does |
|---|---|
| `--aoi` | The area: a vector file (GeoJSON, GeoPackage or shapefile, in any CRS) or a bundled demo name (`caldor`, `rondonia`, `everglades`, `itapebi`, `jaru`). The demos ship inside the package. `restore` also takes `--site`, and `<demo>:<role>` for another feature of a demo file, such as `itapebi:negative`. |
| `--year-a`, `--year-b` | The two years compared, 2017 to 2025. Required by `wildfire`, `deforest` and `wetland`. `restore` defaults them from `--planted`: the year before planting, and 2025. |
| `--out` | The output folder, `out` by default. |
| `--validate` | Score the result against the app's reference dataset and write `<app>_validation.json` (below: what happens when the reference cannot be loaded). Giving a `--reference` file implies it. |
| `--force` | Run again even when the outputs are current. |
| `--snapshot` | Pin an Icechunk snapshot id. By default the run reads the head of `main`, and the report records which snapshot that was. |
| `--i-know-this-is-large` | Lift the 10,000 km2 limit on the AOI (for `wetland` also the limit on seed chunks, for `restore` the limit on the extent and the target). |
| `--strict-validate` | With `--validate`, exit with code 3, before any chunk is read, when the reference cannot be loaded (`restore` also exits 3, before writing outputs, when its null test cannot run). Without it the run goes on and the report says why validation was not done. |

`tessera-apps <app> --help` lists the app's own options, such as `--reference` and `--severity` for `wildfire`, `--labels` for `deforest`, `--seeds` for `wetland`, and `--planted`, `--target` and `--control` for `restore`.

**What a run writes.** In `--out`: a Cloud Optimized GeoTIFF `<app>_<year-a>_<year-b>.tif` with the app's bands, `<app>_summary.parquet` and `<app>_summary.csv`, `<app>_preview.png`, `<app>_validation.json` with `--validate`, and `<app>_report.json`, written last. The report records the package version, the store snapshot, the zones and years read and checked, the parameters and thresholds, the random seed, the sha256 of the AOI file, the bytes read and the wall time. Each app's section below lists its extra outputs. The command also prints a summary and logs the run plan and the bytes read on stderr.

**When the validation reference cannot be loaded** (its host cannot be reached, for example) and `--strict-validate` is not set, the run still finishes, and its report says why validation was not done (`summary.validation.error` and `warnings`). `wildfire`, `deforest` and `wetland` then write no `<app>_validation.json`, with one exception: `wildfire` still writes its `--severity` scores, and the file says that the outline (NIFC or `--reference`) was not scored and why. Rerunning the same command once the reference loads redoes only the validation, from the outputs on disk, without reading the store. With `--validate`, `restore` always writes its validation file, and it names any check that could not run (the null test, the canopy height comparison).

**Reruns are idempotent.** A rerun with the same arguments, input files, package version and store snapshot finds the outputs current, prints so and exits 0 without reading any chunk. Any change to one of those runs it again, and `--force` always does. The report is written last and removed before a run starts, so a run that stopped half way never looks finished.

**Exit codes.**

- `0`: the run finished, or its outputs were already current.
- `1`: an unexpected error, for example the network failing part way through a run; the traceback says what failed. Rerunning the same command runs it again, since a run that stopped before writing its report is never taken as finished.
- `2`: an argument, the AOI, a labels or reference file, the output folder or the store's published years stopped the run before it read anything, the store could not be opened (no route to S3 in `us-west-2`, or a `--snapshot` id the store does not have), or the data cannot answer the question as asked (no pixel comparable between the two years, too few usable labels). The message says what to change.
- `3`: `--strict-validate` is set and the validation could not be done: its reference could not be loaded or, for `restore`, the null test could not run.

**Two more commands.** `tessera-apps info [--zone 10N] [--snapshot ID]` opens the store anonymously and prints its provenance and one zone's published years. `tessera-apps --version` prints the version.

## Wildfire

Maps where it burned between two years and how far each burned pixel has recovered since. It takes the cosine distance between the year before and the year after the fire, sets an Otsu threshold (with a fallback when the distances are not bimodal), and cleans the burned mask to a minimum area. For every later year it measures each burned pixel's similarity to its own pre-fire vector, calibrated against nearby land that did not burn. With `--validate` it scores the burned mask against NIFC perimeters (US) or any `--reference` file. With `--severity` it also scores it against a burn severity map in MTBS dnbr6 classes: detection per class, and IoU against moderate or worse. A perimeter includes unburned islands, so severity is the fairer reference for a pixel map.

```bash
tessera-apps wildfire --aoi caldor --year-a 2020 --year-b 2022 --validate --fire-name Caldor
```

To score against the Hansen 2021 loss outline bundled with the demo and a Sentinel-2 dNBR severity map, as notebook 01 does, write the outline out, build the map (from a checkout of this repository: the script reads about 600 MB of Sentinel-2 imagery anonymously, in about 2 minutes) and pass both. The validation inputs are part of the run's arguments, so this reads the store again and replaces the first command's outputs in `out`:

```bash
python -c 'import geopandas as gpd; from tessera_apps.demo import demo_feature; gpd.GeoSeries([demo_feature("caldor", "hansen_2021_loss_outline")], crs=4326).to_file("outline.geojson")'
python scripts/make_caldor_dnbr.py --out caldor_s2_dnbr6.tif
tessera-apps wildfire --aoi caldor --year-a 2020 --year-b 2022 --validate --reference outline.geojson --severity caldor_s2_dnbr6.tif
```

This writes `out/wildfire_2020_2022.tif` (distance, burned and per-year recovery bands), two PCA false colour COGs, `wildfire_summary.parquet` and `.csv` (one row per year), `wildfire_preview.png`, `wildfire_report.json` and `wildfire_validation.json`. The Caldor demo reads about 23 GB and takes about 8 minutes outside us-west-2. On Caldor the burned mask scores IoU 0.813 against moderate or worse burn in the Sentinel-2 dNBR severity map made by [scripts/make_caldor_dnbr.py](https://github.com/rh281/tessera_apps/blob/main/scripts/make_caldor_dnbr.py) (MTBS itself could not be reached from the build environment). NIFC could not be reached either, so the first command's validation could not be checked there. See [notebook 01](https://github.com/rh281/tessera_apps/blob/main/notebooks/01_wildfire.ipynb) and [docs/checkpoints.md](https://github.com/rh281/tessera_apps/blob/main/docs/checkpoints.md) for the results.

## Deforest

Finds where forest became something else between two years, and the forest baseline in each. It trains a forest / non-forest logistic regression on labels sampled from Hansen in a ring outside the AOI (or on your own `--labels` CSV, for places with no forest layer), classifies both years, and calls conversion where forest in year A is non-forest in year B and the embedding really changed. With `--validate` it scores the conversion against Hansen loss over the same window.

```bash
tessera-apps deforest --aoi rondonia --year-a 2019 --year-b 2022 --validate
```

This writes `out/deforest_2019_2022.tif` (forest probability in both years, distance, converted), `deforest_summary.parquet` and `.csv`, `deforest_labels.parquet`, `deforest_preview.png`, `deforest_report.json` and `deforest_validation.json`. The Rondonia demo reads 1.6 GB in under 2 minutes. See [notebook 02](https://github.com/rh281/tessera_apps/blob/main/notebooks/02_deforest.ipynb), which also shows how to run with hand labels.

## Wetland

Given a few confirmed locations of a wetland type (points or polygons), finds where else it is and how its extent moved between two years. It scores every pixel by cosine distance to the seeds' mean embedding, sets the threshold from seed scores held out by location (so it is not fitted to the seeds themselves), and reports extent, gain, loss and the share near the coast. With `--validate` it scores the extent against ESA WorldCover (2020 and 2021) or Global Mangrove Watch, leaving the seeds out of the score. The `everglades` demo brings its own seeds.

```bash
tessera-apps wetland --aoi everglades --year-a 2020 --year-b 2024 --validate
```

This writes `out/wetland_2020_2024.tif` (score and extent for both years, change), `wetland_summary.parquet` and `.csv`, `wetland_seeds.parquet`, `wetland_preview.png`, `wetland_report.json` and `wetland_validation.json`. The Everglades demo reads 2.4 GB in under 2 minutes. See [notebook 03](https://github.com/rh281/tessera_apps/blob/main/notebooks/03_wetland.ipynb).

## Restore

Asks whether a restoration or reforestation site is diverging from a matched control the way a working project should. It builds the control from a ring 1 to 3 km around the site, keeping only land whose embedding looked like the site's in the year before the baseline. Then, for every year from the year before planting to 2025, it measures the cosine distance between the site's mean embedding and the control's. A year counts as diverging only when the change since the baseline beats 200 placebo sites of the site's size, drawn from the control. The verdict is "diverging from control since YYYY", "no divergence detected" or "insufficient years". With `--target` (a mature stand) it also reports how far the site has moved toward the target, and the share of site pixels closer to the target than to the control. With `--validate` it runs the brief's null test (control against control) and compares the trajectory with GLAD annual tree canopy height.

```bash
tessera-apps restore --aoi itapebi --target demo --validate
tessera-apps restore --aoi itapebi:negative --target demo --validate
```

This writes `out/restore_2018_2025.tif` (pixel roles, similarity to the site in the match year, per-year similarity to the control), `restore_summary.parquet` and `.csv` (one row per year), `restore_placebos.parquet`, `restore_preview.png`, `restore_report.json` and `restore_validation.json`. The demo sites' planting year (2019) comes from their files. The Itapebi demo reads 0.8 GB in under a minute. Both commands write to `out`, so the second replaces the first's outputs; give each its own `--out` to keep both. See [notebook 04](https://github.com/rh281/tessera_apps/blob/main/notebooks/04_restore.ipynb).

**The demo sites are synthetic.** The brief asks for a real project polygon with a 2019 or 2020 planting date. No such polygon could be reached from the build environment: USFS FACTS, Restor, Plant-for-the-Planet and Global Forest Watch are all blocked there. The two bundled sites were therefore located from independent public layers (GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery). They are labelled synthetic in their files. [scripts/check_restore_demos.py](https://github.com/rh281/tessera_apps/blob/main/scripts/check_restore_demos.py) recomputes the evidence for them.

- `itapebi`: a 62 ha eucalyptus block planted on pasture in 2019, in Bahia, Brazil. It stands in for a tree planting, not an ecological restoration.
- `jaru`: a 32 ha paddock of natural regrowth on abandoned pasture in Rondonia, Brazil.

Each file also holds a mature target stand and a stand that stayed pasture (`<demo>:negative`).

There is no global reference layer for restoration outcomes. The null test checks that the method reports nothing where nothing happened; it is not an accuracy.

## Reading the store

```python
from shapely.geometry import box
from tessera_apps.io import TesseraReader, plan_aoi, iter_plan_blocks

reader = TesseraReader.open()                       # anonymous; pin with snapshot_id="..."
plan = plan_aoi(reader, box(-120.4, 38.6, -120.3, 38.7))   # metadata only, no chunk reads
reader.check_years(plan.zone_names, [2020, 2022])   # fails loudly on a publication gap
for zone_plan in plan.zones:
    for block in iter_plan_blocks(reader, zone_plan, [2020, 2022]):
        a, b = block.windows[2020], block.windows[2022]   # int8 embeddings, scales, obs counts
        ...
print(reader.bytes_read, reader.provenance())
```

What `io.py` takes care of:

- **geotessera.** Every embedding read goes through [`geotessera`](https://github.com/ucam-eo/geotessera) 0.11, the University of Cambridge client. `TesseraReader.open` opens `GeoTesseraZarr(store_url="s3://tessera-embeddings/v1.1/dclimate.icechunk")` and refuses to read unless geotessera names the store `1.1-dclimate`; the chunks are read through geotessera's `IcechunkStore` at the snapshot you pin, or at the current head, which the report records. The module docstring of `tessera_apps.io` explains why the reads use `IcechunkStore` rather than `GeoTesseraZarr` itself (it cannot pin a snapshot yet) and where the observation counts come from.
- **Zones.** Pure six-degree UTM bands, `10N` or `20S`, with no MGRS exceptions. An AOI that crosses a zone edge is split, and each pixel is taken from its nominal zone, so nothing is counted twice.
- **Grid.** Rows run north to south. The grid comes from each group's `spatial:transform`, and a transform with ascending northing is refused.
- **No data.** A NaN `scales` value or an all-zero vector is never valid.
- **Years.** Every year is checked against the zone's `years_complete` before any chunk is read.
- **Cost.** Large AOIs are streamed in chunk-aligned blocks. Point reads are batched so each chunk is fetched once. `bytes_read` counts the chunk payload bytes every read returns (not the `zarr.json` metadata, which Icechunk serves from the snapshot it already holds).
- **Radar.** `Window.radar_free` marks pixels embedded without Sentinel-1. Comparing a radar year with a radar-free year biases cosine distance (see the survey, B5).
- **Cache.** `ChunkCache` is an optional on-disk chunk cache, namespaced by Icechunk snapshot.

## Other modules

- `metrics.py`: cosine similarity and distance maps (exact on int8), mean embeddings and per-pixel trajectories, the radar-regime match, Otsu with a bimodality check, mask cleaning, and PCA false colour with a basis shared across years.
- `export.py`: Cloud Optimized GeoTIFFs, a mosaic of per-zone results that takes each ground pixel from its nominal zone, parquet and csv tables, JSON, preview PNGs, provenance records, and idempotent output checks.
- `validate.py`: binary and multi-class scores, reference rasters on the AOI grid from geometries, files or rasters, and loaders for NIFC, EFFIS, Hansen GFC, ESA WorldCover, GLAD annual tree canopy height, Global Mangrove Watch, the Natural Earth coastline and USFS FACTS. Downloads are cached under `~/.cache/tessera_apps/reference/`.
- `demo.py`: the bundled demo AOIs (`caldor`, `rondonia`, `everglades`, and the restore sites `itapebi` and `jaru`).

## Development

From a checkout installed by the reproducible route above:

```bash
.venv/bin/python -m pytest -q          # 1,165 offline tests, in-memory stores and a real 64 x 64 fixture
.venv/bin/python -m pytest -m network  # live store and reference datasets (NIFC, EFFIS, Zenodo, USFS must be reachable), and the Itapebi restore demo
.venv/bin/ruff check src tests scripts && .venv/bin/ruff format --check src tests scripts && .venv/bin/mypy src tests scripts
```

To regenerate the lock after changing dependencies, run `uv pip compile pyproject.toml --all-extras --python-version 3.12 --no-header -o requirements-lock.txt`. CI checks that the committed lock still matches `pyproject.toml`.

## Releasing

Releases are published to PyPI by [.github/workflows/release.yml](https://github.com/rh281/tessera_apps/blob/main/.github/workflows/release.yml) when a `vX.Y.Z` tag is pushed, with trusted publishing and after the full check. [docs/releasing.md](https://github.com/rh281/tessera_apps/blob/main/docs/releasing.md) has the one-time setup and the steps for each release.

## Changelog

See [CHANGELOG.md](https://github.com/rh281/tessera_apps/blob/main/CHANGELOG.md), which also lists the known limitations of each release.

## Where things come from

- [docs/prebuild-survey.md](https://github.com/rh281/tessera_apps/blob/main/docs/prebuild-survey.md) checks the build brief against the store, the source repo, and the reference datasets. It recommends changes to the brief, and the code follows those recommendations wherever the brief and the survey disagree.
- `tests/fixtures/caldor_64.npz` is a real 64 x 64 window inside the Caldor scar, about half burned, with all 9 years and every layer. It was made by [scripts/make_fixture.py](https://github.com/rh281/tessera_apps/blob/main/scripts/make_fixture.py).
- `src/tessera_apps/data/itapebi.geojson` and `jaru.geojson` are the restore demo sites. They are **synthetic**: located from GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery because no public planting polygon could be reached (see the Restore section). [scripts/check_restore_demos.py](https://github.com/rh281/tessera_apps/blob/main/scripts/check_restore_demos.py) recomputes their evidence.
- The Caldor burn severity map used in notebook 01 is made by [scripts/make_caldor_dnbr.py](https://github.com/rh281/tessera_apps/blob/main/scripts/make_caldor_dnbr.py) from the Sentinel-2 L2A archive on AWS. It is a stand-in for MTBS, which could not be reached from the build environment.
- `src/tessera_apps/data/caldor.geojson` is the Caldor demo AOI. It is **provisional**: this build environment cannot reach NIFC, so the AOI is derived from Hansen GFC v1.13 2021 loss by [scripts/make_caldor_aoi.py](https://github.com/rh281/tessera_apps/blob/main/scripts/make_caldor_aoi.py). The AOI is the 2021 scar's bounding box plus 2 km. Replace it with the NIFC perimeter's bounding box plus 2 km when the network tier runs.

## License

Apache-2.0. See [LICENSE](https://github.com/rh281/tessera_apps/blob/main/LICENSE).
