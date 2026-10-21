# Changelog

All notable changes to `tessera_apps`. Versions follow [semantic versioning](https://semver.org); a version number is never reused on PyPI.

## 0.1.0 (2026-10-21)

The first release: four climate applications on the TESSERA v1.1 global embeddings, read anonymously from the dClimate Icechunk store. No AWS account or credentials are needed.

### Added

- **Four apps, each a `tessera-apps` command and a Python function** (`tessera_apps.apps.run_wildfire`, `run_deforest`, `run_wetland`, `run_restore`):
  - `wildfire`: the burn scar between two years (cosine distance, an Otsu threshold with a bimodality check and a fallback) and each burned pixel's recovery in every later year, calibrated against land that did not burn. Validation against NIFC perimeters, any `--reference` outline, or a burn severity map in MTBS dnbr6 classes (`--severity`).
  - `deforest`: forest in two years from a logistic regression trained on Hansen labels sampled in a ring outside the AOI (or on your own `--labels` CSV), and the forest that became something else. Validation against Hansen loss over the same window.
  - `wetland`: where else a wetland type is, given a few seed points or polygons, and how its extent moved; the threshold comes from seed scores held out by location. Validation against ESA WorldCover or Global Mangrove Watch.
  - `restore`: whether a restoration site diverges from a matched control ring, judged against 200 site-sized placebos, with a null test (control against control), progress toward a mature target stand and a GLAD canopy height check.
- **One command line shape for all four:** `--aoi`, `--year-a`, `--year-b`, `--out`, `--validate`, `--force`, `--snapshot`, `--i-know-this-is-large` and `--strict-validate`; exit code 2, with the reason, for anything that stops a run before it reads (including a store that cannot be opened) and 3 for a strict validation failure; idempotent reruns. Each run writes a Cloud Optimized GeoTIFF, parquet and csv tables, a preview PNG, a report JSON with full provenance and, with `--validate`, a validation JSON. `tessera-apps info` opens the store and prints its snapshot and a zone's published years; `tessera-apps --version` prints the version.
- **The store reader** (`tessera_apps.io`): zone planning for any AOI, a check of every year against the zone's `years_complete` before any chunk is read, streaming of large AOIs in chunk-aligned blocks, batched point reads, radar regime flags, a count of the bytes read, and an optional on-disk chunk cache (`tessera_apps.cache`).
- **Metrics and export** (`tessera_apps.metrics`, `tessera_apps.export`): exact cosine maps on int8, trajectories, Otsu, mask cleaning, PCA false colour with a basis shared across years; COG writing with a zone mosaic, tables, previews and provenance records.
- **Reference data loaders** (`tessera_apps.validate`): NIFC perimeters, EFFIS burnt areas, Hansen Global Forest Change, ESA WorldCover, Global Mangrove Watch, GLAD annual tree canopy height, the Natural Earth coastline and USFS FACTS reforestation, with binary and multi-class scores. Downloads are cached under `~/.cache/tessera_apps/reference/`.
- **Demos bundled in the wheel:** `caldor` (wildfire), `rondonia` (deforest), `everglades` with 12 seed polygons (wetland), and the restore sites `itapebi` and `jaru`, each with a target stand and a negative stand. Pass the name to `--aoi`.
- **Notebooks** 01 to 04, one per app, committed with their executed outputs.
- **Packaging:** Apache-2.0 licence file, `py.typed`, PEP 639 licence metadata, and a release workflow that publishes to PyPI with trusted publishing after the full check, a dist check and a smoke test of the wheel.

### Known limitations

- **The restore demo sites are synthetic.** No public planting polygon could be reached from the build environment, so `itapebi` and `jaru` were located from GLAD canopy height, Hansen, WorldCover and Sentinel-2 imagery. Their files say so, and `scripts/check_restore_demos.py` recomputes the evidence.
- **NIFC, EFFIS, MTBS, GMW and USFS could not be reached from the build environment.** The NIFC, EFFIS, GMW and USFS FACTS loaders are therefore tested offline only, and the validation numbers in [docs/checkpoints.md](https://github.com/rh281/tessera_apps/blob/main/docs/checkpoints.md) use the stand-ins named there: a Hansen 2021 loss outline for the NIFC Caldor perimeter, a Sentinel-2 dNBR proxy for MTBS, and the WorldCover mangrove class for GMW. The Caldor demo AOI is derived from Hansen loss for the same reason and should be redrawn from the NIFC perimeter where NIFC can be reached.
- **The Caldor severity map is a Sentinel-2 dNBR proxy** made by `scripts/make_caldor_dnbr.py`, not MTBS. The burned mask's IoU of 0.813 against moderate or worse burn is likely optimistic against real MTBS, since the proxy and the embeddings both come from Sentinel-2.
- **The wildfire memory estimate is 8 to 10 percent low on Caldor:** 4.76 GB estimated against measured peaks of 5.15 to 5.27 GB.
- **Two items of the build brief are not in this release.** `notebooks/00_retrieval.ipynb` (a copy of the companion notebook `TESSERA_community_applications.ipynb`) is missing because that notebook was never available to the build ([docs/prebuild-survey.md](https://github.com/rh281/tessera_apps/blob/main/docs/prebuild-survey.md)). And the package reads the Icechunk store directly, without `geotessera` (survey B2: before 0.11 it silently reads the legacy v1 model), so the `geotessera` entry the brief asks for in each report's `libraries` is null unless that package happens to be installed, and it never shapes a result.
- **Adding a validation input to a finished run re-reads every chunk.** For example, adding `--severity` or a `--reference` file to a finished Caldor run reads all 23 GB again. Only a rerun after the reference failed to load redoes just the validation, from the outputs on disk (wildfire, deforest and wetland).
