# tessera_apps

Climate applications built on the [TESSERA](https://geotessera.org) v1.1 global embeddings published by dClimate: 128-dimension, 10 m, annual (2017 to 2025) embeddings in an anonymous Icechunk store at `s3://tessera-embeddings/v1.1/dclimate.icechunk` (AWS `us-west-2`). No GPU, no model training beyond scikit-learn linear models.

**Status: checkpoint 2 of the build brief (`io.py`, `metrics.py`, `export.py`, `validate.py`).** The four applications below are planned and not built yet. See `docs/checkpoints.md` for the checkpoint report and numbers.

| App | Question | Status |
|---|---|---|
| `wildfire` | Where did it burn, and how far has each burned pixel recovered? | planned |
| `deforest` | Where did forest become something else between two years? | planned |
| `wetland` | Given a few known wetland locations, where else is that wetland, and how has it moved? | planned |
| `restore` | Is a restoration site diverging from a matched control the way a working project should? | planned |

## Install

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements-lock.txt
uv pip install -p .venv --no-deps -e .
```

Python 3.12 or 3.13. No AWS credentials are needed. `tessera-apps info` opens the store and prints its snapshot and a zone's published years.

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
- `validate.py`: binary and multi-class scores, reference rasters on the AOI grid from geometries, files or rasters, and loaders for NIFC, EFFIS, Hansen GFC, Global Mangrove Watch, the Natural Earth coastline and USFS FACTS. Downloads are cached under `~/.cache/tessera_apps/reference/`.
- `demo.py`: the bundled demo AOIs (`caldor`, `rondonia`, `everglades`).

## Development

```bash
.venv/bin/python -m pytest -q          # 507 offline tests, in-memory stores and a real 64 x 64 fixture
.venv/bin/python -m pytest -m network  # live store and reference datasets (NIFC, EFFIS, Zenodo, USFS must be reachable)
.venv/bin/ruff check src tests scripts && .venv/bin/mypy src tests scripts
```

To regenerate the lock after changing dependencies, run `uv pip compile pyproject.toml --all-extras --python-version 3.12 --no-header -o requirements-lock.txt`.

## Where things come from

- `docs/prebuild-survey.md` checks the build brief against the store, the source repo, and the reference datasets. It recommends changes to the brief, and the code follows those recommendations wherever the brief and the survey disagree.
- `tests/fixtures/caldor_64.npz` is a real 64 x 64 window inside the Caldor scar, about half burned, with all 9 years and every layer. It was made by `scripts/make_fixture.py`.
- `src/tessera_apps/data/caldor.geojson` is the Caldor demo AOI. It is **provisional**: this build environment cannot reach NIFC, so the AOI is derived from Hansen GFC v1.13 2021 loss by `scripts/make_caldor_aoi.py`. The AOI is the 2021 scar's bounding box plus 2 km. Replace it with the NIFC perimeter's bounding box plus 2 km when the network tier runs.

## License

Apache-2.0.
