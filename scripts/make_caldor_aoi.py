"""Derive the bundled Caldor demo AOI from Hansen Global Forest Change 2021 loss.

The brief asks for the NIFC perimeter, but this build environment cannot reach NIFC. Hansen
``lossyear == 21`` (loss in 2021) is public on Google Cloud Storage and outlines the Caldor scar as
the largest connected 2021 loss patch in the window below. The AOI is that patch's bounding box
buffered by ``--buffer-km`` (default 2 km), so the classifier also sees unburned land around the
scar (survey S10). Replace it with the NIFC perimeter bbox when the network tier is run somewhere
NIFC is reachable; the scar outline is also written, for use as a secondary reference.

Usage:
    python scripts/make_caldor_aoi.py --out src/tessera_apps/data/caldor.geojson
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

HANSEN = (
    "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/Hansen_GFC-2025-v1.13_lossyear_{tile}.tif"
)
# Window around the fire: west of the Tamarack scar, which starts east of about 119.85 W.
WINDOW = (-120.8, 38.4, -119.85, 39.1)
RES_DEG = 0.00025
TILES = {"40N_130W": (-130.0, 30.0, -120.0, 40.0), "40N_120W": (-120.0, 30.0, -110.0, 40.0)}


def read_loss(year_code: int) -> tuple[np.ndarray, rasterio.Affine]:
    """Boolean loss mask for one Hansen year code over WINDOW, mosaicked across tiles."""
    west, south, east, north = WINDOW
    res = RES_DEG
    width = round((east - west) / res)
    height = round((north - south) / res)
    out = np.zeros((height, width), dtype=bool)
    transform = rasterio.Affine(res, 0, west, 0, -res, north)
    for tile, (tw, _ts, te, _tn) in TILES.items():
        w, e = max(west, tw), min(east, te)
        if w >= e:
            continue
        with rasterio.open(f"/vsicurl/{HANSEN.format(tile=tile)}") as src:
            win = from_bounds(w, south, e, north, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=win)
        c0 = round((w - west) / res)
        out[: data.shape[0], c0 : c0 + data.shape[1]] |= data == year_code
    return out, transform


def main() -> None:
    """Write the AOI GeoJSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--buffer-km", type=float, default=2.0)
    ap.add_argument("--join-km", type=float, default=1.0)
    ap.add_argument("--min-patch-ha", type=float, default=500.0)
    ap.add_argument("--close-px", type=int, default=12, help="closing radius in Hansen pixels (0.00025 deg)")
    args = ap.parse_args()
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    loss, transform = read_loss(21)
    closed = ndimage.binary_closing(loss, structure=np.ones((3, 3)), iterations=args.close_px)
    labels, n = ndimage.label(closed)
    sizes = ndimage.sum(closed, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    # The scar is broken where loss was patchy (Hansen records stand-replacing loss, not every
    # burned pixel). Keep the largest patch plus any patch of at least --min-patch-ha whose edge is
    # within --join-km of it; this picks up the Christmas Valley lobe east of 120 W and leaves the
    # separate Tamarack fire out.
    lat_mid = (WINDOW[1] + WINDOW[3]) / 2
    ns_m = RES_DEG * 111_320.0
    ew_m = ns_m * np.cos(np.radians(lat_mid))
    px_ha = ns_m * ew_m / 1e4
    # Distance in metres from the main patch, honouring the non-square pixels.
    dist_m = ndimage.distance_transform_edt(labels != biggest, sampling=(ns_m, ew_m))
    near = dist_m <= args.join_km * 1000.0
    keep = [biggest] + [
        i for i in np.unique(labels[near]) if i not in (0, biggest) and sizes[i - 1] * px_ha >= args.min_patch_ha
    ]
    scar = np.isin(labels, keep)
    polys = [shape(g) for g, v in shapes(scar.astype(np.uint8), mask=scar, transform=transform) if v == 1]
    outline = unary_union(polys).simplify(0.001)
    west, south, east, north = outline.bounds
    lat = (south + north) / 2
    dlat = args.buffer_km / 111.32
    dlon = args.buffer_km / (111.32 * np.cos(np.radians(lat)))
    aoi = box(west - dlon, south - dlat, east + dlon, north + dlat)
    scar_ha = float(sum(sizes[i - 1] for i in keep)) * px_ha
    props = {
        "name": "Caldor Fire 2021 demo AOI",
        "source": "Hansen GFC v1.13 lossyear == 21: largest connected patch plus patches of at least "
        f"{args.min_patch_ha:.0f} ha within {args.join_km} km of it; bbox buffered",
        "buffer_km": args.buffer_km,
        "status": "provisional: replace with the NIFC perimeter bbox plus 2 km when NIFC is reachable",
        "scar_bounds": [round(v, 5) for v in outline.bounds],
        "scar_area_ha_approx": round(scar_ha),
        "year_a": 2020,
        "year_b": 2022,
    }
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {**props, "role": "aoi"}, "geometry": mapping(aoi)},
            {"type": "Feature", "properties": {"role": "hansen_2021_loss_outline"}, "geometry": mapping(outline)},
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fc, indent=1))
    print(f"kept {len(keep)} patches; scar bounds {outline.bounds}, about {scar_ha:,.0f} ha")
    print(f"AOI {aoi.bounds}; wrote {args.out}")


if __name__ == "__main__":
    main()
