"""Cut the Hansen GFC tile pieces under the caldor fixture into an npz, for the offline deforest tests.

Usage:
    python scripts/make_hansen_fixture.py --out tests/fixtures/caldor_64_hansen.npz

Reads the ``treecover2000``, ``lossyear`` and ``datamask`` windows of tile ``40N_130W`` that cover
the padded fixture store of ``tests/conftest.py`` (the 64 x 64 window plus its no-data padding) and
a margin, on Hansen's own 0.00025 degree grid, so a test can write them back as local tile pieces
(``validate`` reads tiles through a ``url_template``) and the app reads them as it reads the real
tiles. Needs storage.googleapis.com.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from tessera_apps import validate as v

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "caldor_64.npz"
#: Fixture-grid pixels of padding around the 64 x 64 window, as in tests/conftest.py, plus a margin.
PAD = 40
MARGIN_DEG = 0.002


def main() -> None:
    """Write the npz."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--version", default=v.HANSEN_VERSION)
    args = ap.parse_args()
    with np.load(FIXTURE) as fx:
        a, _, c, _, _, f = (float(x) for x in fx["transform"])
        n = int(fx["scales"].shape[1])
        epsg = int(fx["epsg"])
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    west, south, east, north = to_ll.transform_bounds(c - PAD * a, f - (n + PAD) * a, c + (n + PAD) * a, f + PAD * a)
    r = v.HANSEN_RES_DEG
    west = math.floor((west - MARGIN_DEG) / r) * r
    north = math.ceil((north + MARGIN_DEG) / r) * r
    east = math.ceil((east + MARGIN_DEG) / r) * r
    south = math.floor((south - MARGIN_DEG) / r) * r
    tile = v.hansen_tile((west + east) / 2, (south + north) / 2)
    layers = {}
    with rasterio.Env(**v.gdal_options()):
        for layer in ("treecover2000", "lossyear", "datamask"):
            url = v.HANSEN_URL_TEMPLATE.format(version=args.version, layer=layer, tile=tile)
            with rasterio.open(f"/vsicurl/{url}") as ds:
                col0 = round((west - ds.transform.c) / r)
                row0 = round((ds.transform.f - north) / r)
                width = round((east - west) / r)
                height = round((north - south) / r)
                layers[layer] = ds.read(1, window=Window(col0, row0, width, height))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        **layers,
        west=np.float64(round(west, 6)),
        north=np.float64(round(north, 6)),
        res=np.float64(r),
        tile=np.str_(tile),
        version=np.str_(args.version),
    )
    print(f"{args.out}: tile {tile}, {layers['lossyear'].shape} pixels from ({west:.5f}, {north:.5f})")


if __name__ == "__main__":
    main()
