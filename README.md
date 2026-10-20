# tessera_apps

Climate applications built on the [TESSERA](https://geotessera.org) v1.1 global embeddings published by dClimate: 128-dimension, 10 m, annual (2017 to 2025) embeddings in an anonymous Icechunk store at `s3://tessera-embeddings/v1.1/dclimate.icechunk` (AWS `us-west-2`). No GPU, no model training beyond scikit-learn linear models.

**Status: checkpoint 6 of the build brief.** All four apps are built. Packaging and release (checkpoint 7) and the hosted demo (checkpoint 8) are next. See `docs/checkpoints.md` for the checkpoint reports and numbers.

| App | Question | Status |
|---|---|---|
| `wildfire` | Where did it burn, and how far has each burned pixel recovered? | built, [notebook 01](notebooks/01_wildfire.ipynb) |
| `deforest` | Where did forest become something else between two years? | built, [notebook 02](notebooks/02_deforest.ipynb) |
| `wetland` | Given a few known wetland locations, where else is that wetland, and how has it moved? | built, [notebook 03](notebooks/03_wetland.ipynb) |
| `restore` | Is a restoration site diverging from a matched control the way a working project should? | built, [notebook 04](notebooks/04_restore.ipynb) |

## Install

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements-lock.txt
uv pip install -p .venv --no-deps -e .
```

Python 3.12 or 3.13. No AWS credentials are needed. `tessera-apps info` opens the store and prints its snapshot and a zone's published years.

## Wildfire

Maps where it burned between two years and how far each burned pixel has recovered since. It takes the cosine distance between the year before and the year after the fire, sets an Otsu threshold (with a fallback when the distances are not bimodal), and cleans the burned mask to a minimum area. For every later year it measures each burned pixel's similarity to its own pre-fire vector, calibrated against nearby land that did not burn. With `--validate` it scores the burned mask against NIFC perimeters (US) or any `--reference` file. With `--severity` it also scores it against a burn severity map in MTBS dnbr6 classes: detection per class, and IoU against moderate or worse. A perimeter includes unburned islands, so severity is the fairer reference for a pixel map.

```bash
tessera-apps wildfire --aoi caldor --year-a 2020 --year-b 2022 --validate --fire-name Caldor
tessera-apps wildfire --aoi caldor --year-a 2020 --year-b 2022 --validate --reference outline.geojson --severity caldor_s2_dnbr6.tif
```

This writes `out/wildfire_2020_2022.tif` (distance, burned and per-year recovery bands), two PCA false colour COGs, `wildfire_summary.parquet` and `.csv` (one row per year), `wildfire_preview.png`, `wildfire_report.json` and `wildfire_validation.json`. The Caldor demo reads about 23 GB and takes about 8 minutes outside us-west-2. On Caldor the burned mask scores IoU 0.813 against moderate or worse burn in a Sentinel-2 dNBR severity map made by `scripts/make_caldor_dnbr.py` (MTBS itself could not be reached from the build environment). See [notebook 01](notebooks/01_wildfire.ipynb) and `docs/checkpoints.md` for the results.

## Deforest

Finds where forest became something else between two years, and the forest baseline in each. It trains a forest / non-forest logistic regression on labels sampled from Hansen in a ring outside the AOI (or on your own `--labels` CSV, for places with no forest layer), classifies both years, and calls conversion where forest in year A is non-forest in year B and the embedding really changed. With `--validate` it scores the conversion against Hansen loss over the same window.

```bash
tessera-apps deforest --aoi rondonia --year-a 2019 --year-b 2022 --validate
```

This writes `out/deforest_2019_2022.tif` (forest probability in both years, distance, converted), `deforest_summary.parquet` and `.csv`, `deforest_labels.parquet`, `deforest_preview.png`, `deforest_report.json` and `deforest_validation.json`. The Rondonia demo reads 1.6 GB in under 2 minutes. See [notebook 02](notebooks/02_deforest.ipynb), which also shows how to run with hand labels.

## Wetland

Given a few confirmed locations of a wetland type (points or polygons), finds where else it is and how its extent moved between two years. It scores every pixel by cosine distance to the seeds' mean embedding, sets the threshold from seed scores held out by location (so it is not fitted to the seeds themselves), and reports extent, gain, loss and the share near the coast. With `--validate` it scores the extent against ESA WorldCover (2020 and 2021) or Global Mangrove Watch, leaving the seeds out of the score.

```bash
tessera-apps wetland --aoi everglades --year-a 2020 --year-b 2024 --validate
```

This writes `out/wetland_2020_2024.tif` (score and extent for both years, change), `wetland_summary.parquet` and `.csv`, `wetland_seeds.parquet`, `wetland_preview.png`, `wetland_report.json` and `wetland_validation.json`. The Everglades demo reads 2.4 GB in about a minute. See [notebook 03](notebooks/03_wetland.ipynb).

## Restore

Asks whether a restoration or reforestation site is diverging from a matched control the way a working project should. It builds the control from a ring 1 to 3 km around the site, keeping only land whose embedding looked like the site's in the year before the baseline. Then, for every year from the year before planting to 2025, it measures the cosine distance between the site's mean embedding and the control's. A year counts as diverging only when the change since the baseline beats 200 placebo sites of the site's size, drawn from the control. The verdict is "diverging from control since YYYY", "no divergence detected" or "insufficient years". With `--target` (a mature stand) it also reports how far the site has moved toward the target, and the share of site pixels closer to the target than to the control. With `--validate` it runs the brief's null test (control against control) and compares the trajectory with GLAD annual tree canopy height.

```bash
tessera-apps restore --aoi itapebi --target demo --validate
tessera-apps restore --aoi itapebi:negative --target demo --validate
```

This writes `out/restore_2018_2025.tif` (pixel roles, similarity to the site in the match year, per-year similarity to the control), `restore_summary.parquet` and `.csv` (one row per year), `restore_placebos.parquet`, `restore_preview.png`, `restore_report.json` and `restore_validation.json`. The Itapebi demo reads 0.8 GB in under a minute. See [notebook 04](notebooks/04_restore.ipynb).

**The demo sites are synthetic.** The brief asks for a real project polygon with a 2019 or 2020 planting date. No such polygon could be reached from the build environment: USFS FACTS, Restor, Plant-for-the-Planet and Global Forest Watch are all blocked there. The two bundled sites were therefore located from independent public layers (GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery). They are labelled synthetic in their files. `scripts/check_restore_demos.py` recomputes the evidence for them.

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

- **Zones.** Pure six-degree UTM bands, `10N` or `20S`, with no MGRS exceptions. An AOI that crosses a zone edge is split, and each pixel is taken from its nominal zone, so nothing is counted twice.
- **Grid.** Rows run north to south. The grid comes from each group's `spatial:transform`, and a transform with ascending northing is refused.
- **No data.** A NaN `scales` value or an all-zero vector is never valid.
- **Years.** Every year is checked against the zone's `years_complete` before any chunk is read.
- **Cost.** Large AOIs are streamed in chunk-aligned blocks. Point reads are batched so each chunk is fetched once. `bytes_read` counts the payload bytes every read returns.
- **Radar.** `Window.radar_free` marks pixels embedded without Sentinel-1. Comparing a radar year with a radar-free year biases cosine distance (see the survey, B5).
- **Cache.** `ChunkCache` is an optional on-disk chunk cache, namespaced by Icechunk snapshot.

## Other modules

- `metrics.py`: cosine similarity and distance maps (exact on int8), mean embeddings and per-pixel trajectories, the radar-regime match, Otsu with a bimodality check, mask cleaning, and PCA false colour with a basis shared across years.
- `export.py`: Cloud Optimized GeoTIFFs, a mosaic of per-zone results that takes each ground pixel from its nominal zone, parquet and csv tables, JSON, preview PNGs, provenance records, and idempotent output checks.
- `validate.py`: binary and multi-class scores, reference rasters on the AOI grid from geometries, files or rasters, and loaders for NIFC, EFFIS, Hansen GFC, ESA WorldCover, GLAD annual tree canopy height, Global Mangrove Watch, the Natural Earth coastline and USFS FACTS. Downloads are cached under `~/.cache/tessera_apps/reference/`.
- `demo.py`: the bundled demo AOIs (`caldor`, `rondonia`, `everglades`, and the restore sites `itapebi` and `jaru`).

## Development

```bash
.venv/bin/python -m pytest -q          # 1,052 offline tests, in-memory stores and a real 64 x 64 fixture
.venv/bin/python -m pytest -m network  # live store and reference datasets (NIFC, EFFIS, Zenodo, USFS must be reachable), and the Itapebi restore demo
.venv/bin/ruff check src tests scripts && .venv/bin/mypy src tests scripts
```

To regenerate the lock after changing dependencies, run `uv pip compile pyproject.toml --all-extras --python-version 3.12 --no-header -o requirements-lock.txt`.

## Where things come from

- `docs/prebuild-survey.md` checks the build brief against the store, the source repo, and the reference datasets. It recommends changes to the brief, and the code follows those recommendations wherever the brief and the survey disagree.
- `tests/fixtures/caldor_64.npz` is a real 64 x 64 window inside the Caldor scar, about half burned, with all 9 years and every layer. It was made by `scripts/make_fixture.py`.
- `src/tessera_apps/data/itapebi.geojson` and `jaru.geojson` are the restore demo sites. They are **synthetic**: located from GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery because no public planting polygon could be reached (see the Restore section). `scripts/check_restore_demos.py` recomputes their evidence.
- The Caldor burn severity map used in notebook 01 is made by `scripts/make_caldor_dnbr.py` from the Sentinel-2 L2A archive on AWS. It is a stand-in for MTBS, which could not be reached from the build environment.
- `src/tessera_apps/data/caldor.geojson` is the Caldor demo AOI. It is **provisional**: this build environment cannot reach NIFC, so the AOI is derived from Hansen GFC v1.13 2021 loss by `scripts/make_caldor_aoi.py`. The AOI is the 2021 scar's bounding box plus 2 km. Replace it with the NIFC perimeter's bounding box plus 2 km when the network tier runs.

## License

Apache-2.0.
