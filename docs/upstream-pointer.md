# Pointers to tessera_apps for two upstream READMEs

The build brief (section 10) asks for a short pointer to this package in the top-level README of [dClimate/tessera-embeddings](https://github.com/dClimate/tessera-embeddings), in its quickstart section on data access. The 0.2.0 handoff adds a one-line pointer for the README of [ucam-eo/geotessera-examples](https://github.com/ucam-eo/geotessera-examples). Both say that the package reads embeddings through `geotessera` with the v1.1 dClimate store. This repository cannot push to either, so this page holds the text for their maintainers to paste.

**Paste them only once tessera_apps 0.2.0 is on PyPI**, that is when <https://pypi.org/project/tessera-apps/0.2.0/> opens. 0.1.0 reads the store without `geotessera`, so before 0.2.0 is out, `pip install tessera_apps` installs a version the text does not describe.

**Both link to PyPI, not to GitHub.** rh281/tessera_apps is private by the owner's choice, so a link to it gives a 404 to anyone without access, while the PyPI page opens for everyone (its own links to the repository give the same 404). If the repository is made public, the links can point to <https://github.com/rh281/tessera_apps> instead.

## dClimate/tessera-embeddings

### Where it goes

In `README.md` of dClimate/tessera-embeddings, at the end of the section that opens with "The finished global store is public" (its heading is "Quickstart" and "data access", joined by a dash; line 47 at commit `2ea97c2`, and still line 47 at `3b13965`, the head of `main` on 2026-10-23, whose first 120 lines are the same as at `2ea97c2`):

- **after** the paragraph that begins "[The global embeddings store](#the-global-embeddings-store) below covers the layout" (lines 105 and 106),
- **before** the `[sc]: https://source.coop/...` link definition (line 108) and the `---` rule that comes before `## What this is`.

### The text to paste

Everything between the outer fences, as it stands. It is wrapped at the same width as the rest of that README.

````markdown
**Ready-made applications.** [tessera_apps](https://pypi.org/project/tessera-apps/) is a Python
package of four climate applications built on this store: burn scars and their recovery
(`wildfire`), forest that became something else (`deforest`), where else a wetland type is from
a few seed locations (`wetland`), and whether a restoration site diverges from a matched
control (`restore`). It reads the v1.1 dClimate store through
[geotessera](https://github.com/ucam-eo/geotessera) 0.11 and checks that geotessera names it
`1.1-dclimate`. Each app is one command that reads the store anonymously, checks every year
against `years_complete` before reading, and writes Cloud Optimized GeoTIFFs, tables, a preview
and a validation report. No AWS account is needed. The Rondonia demo reads about 1.6 GB in
under 2 minutes:

```bash
pip install tessera_apps
tessera-apps deforest --aoi rondonia --year-a 2019 --year-b 2022 --validate
```
````

### Why this text

- **What it is**, in one sentence, with the link: the four apps and what each answers.
- **geotessera with the v1.1 dClimate store:** since 0.2.0 every embedding read goes through `geotessera` 0.11, opened with `store_url="s3://tessera-embeddings/v1.1/dclimate.icechunk"`, and a run refuses to read unless geotessera names the dataset `1.1-dclimate`.
- **How to get it:** `pip install tessera_apps`, from PyPI. It needs Python 3.12 or 3.13 and nothing else, since GDAL comes inside the rasterio wheel.
- **One demo command** that works straight after the install: the Rondonia AOI ships inside the package. It is the cheapest demo with a validation against an independent layer (Hansen loss). Under 0.2.0 it read 1,550,387,529 bytes in 85 s outside us-west-2 (0.2.0 checkpoint A in [docs/checkpoints.md](https://github.com/rh281/tessera_apps/blob/main/docs/checkpoints.md); 0.1.0 took 81 s in the same session). The Caldor wildfire demo reads 23 GB.
- **No AWS account**, which matches the "No AWS account is needed" line earlier in the same section.

## ucam-eo/geotessera-examples

### Where it goes

In `README.md` of ucam-eo/geotessera-examples, read from a shallow clone of `main` at commit `a4ca84b` (2026-08-27; still the head on 2026-10-23): as a new last row of the "Where next" table (lines 27 to 32), after line 32, the `pumap-viz/` row. The table's other rows point to folders of that repository and this one to a package outside it, so its maintainers may prefer the same line as a sentence after the table, before `## Concepts` (line 52).

### The line to paste

```markdown
| Map burn scars, forest loss, wetlands or restoration with ready-made apps | [tessera_apps](https://pypi.org/project/tessera-apps/) (`pip install tessera_apps`), which uses geotessera with the v1.1 dClimate store |
```

### Why this line

- **One line**, in the table's own two columns: what the reader wants, and where to go.
- **geotessera with the v1.1 dClimate store:** the scripts in that repository open `GeoTesseraZarr` on the Zarr store that geotessera's registry names for a model version (`zarr_store_url(...)` in `quickstart.py`, `classify/`, `solarpanel/` and `teaching/`, at `a4ca84b`). This package opens the dClimate Icechunk store `s3://tessera-embeddings/v1.1/dclimate.icechunk` explicitly, so the line names which store it uses.
