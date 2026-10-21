# A pointer to tessera_apps for the dClimate/tessera-embeddings README

The build brief (section 10) asks for a short pointer to this package in the top-level README of [dClimate/tessera-embeddings](https://github.com/dClimate/tessera-embeddings), in its quickstart section on data access. This repository cannot push there, so this page holds the text for that repository's maintainers to paste.

**Paste it only once both links work:** tessera_apps 0.1.0 is on PyPI (<https://pypi.org/project/tessera-apps/> opens) and <https://github.com/rh281/tessera_apps> opens for someone who is not signed in to GitHub (the repository is public). Before that, the install command fails and the link gives a 404.

## Where it goes

In `README.md` of dClimate/tessera-embeddings, at the end of the section that opens with "The finished global store is public" (its heading is "Quickstart" and "data access", joined by a dash; line 47 at commit `2ea97c2`):

- **after** the paragraph that begins "[The global embeddings store](#the-global-embeddings-store) below covers the layout",
- **before** the `[sc]: https://source.coop/...` link definition and the `---` rule that comes before `## What this is`.

## The text to paste

Everything between the outer fences, as it stands. It is wrapped at the same width as the rest of that README.

````markdown
**Ready-made applications.** [tessera_apps](https://github.com/rh281/tessera_apps) is a Python
package of four climate applications built on this store: burn scars and their recovery
(`wildfire`), forest that became something else (`deforest`), where else a wetland type is from a
few seed locations (`wetland`), and whether a restoration site diverges from a matched control
(`restore`). Each is one command that reads the store anonymously, checks every year against
`years_complete` before reading, and writes Cloud Optimized GeoTIFFs, tables, a preview and a
validation report. It needs no AWS account. The Rondonia demo reads about 1.6 GB:

```bash
pip install tessera_apps
tessera-apps deforest --aoi rondonia --year-a 2019 --year-b 2022 --validate
```
````

## Why this text

- **What it is**, in one sentence, with the link: the four apps and what each answers.
- **How to get it:** `pip install tessera_apps`, from PyPI. It needs Python 3.12 or 3.13 and nothing else, since GDAL comes inside the rasterio wheel.
- **One demo command** that works straight after the install: the Rondonia AOI ships inside the package. It is the cheapest demo with a validation against an independent layer (Hansen loss): 1.6 GB in under 2 minutes outside us-west-2. The Caldor wildfire demo reads 23 GB.
- **No AWS account**, which matches the "No AWS account is needed" line earlier in the same section.
