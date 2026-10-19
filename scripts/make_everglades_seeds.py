"""Derive the bundled Everglades wetland seeds from ESA WorldCover 2020 mangroves.

The brief asks for seeds inside known Global Mangrove Watch stands, but this build environment
cannot reach GMW (zenodo.org is blocked). ESA WorldCover 2020 (v100) is public on AWS Open Data and
maps mangroves as class 95, so the seeds come from it instead:

1. WorldCover 2020 class 95 inside the Everglades demo AOI, on the TESSERA 10 m grid of zone 17N
   (``validate.worldcover_classes``; the zone grid is store metadata, no chunk is read).
2. Erode by 5 pixels (about 50 m, 3 x 3 structuring element) so only stand interiors remain.
3. Label the 8-connected components of what is left.
4. Split the AOI window into a 4 x 3 grid and pick, in each cell, the largest component whose
   centroid falls in it. A cell with no such component (open water) is filled with the unpicked
   component whose centroid lies farthest from those already picked, so 12 components spread over
   the AOI, deterministically.
5. In each component place one 10 x 10 pixel square (100 m on a side, on pixel edges, so exactly
   100 pixel centres) wholly inside the eroded component, at the feasible position nearest its
   centroid.

That gives 12 polygons and 1,200 pixels at 10 m: enough held-out folds for the threshold, and more
than the 500 pixels the shrinkage Mahalanobis score needs (survey B6).

The seeds and the WorldCover validation reference come from the same product, so the wetland
app's score against WorldCover is not independent of its seeds. Replace these seeds with GMW stands
where zenodo.org is reachable.

Usage:
    python scripts/make_everglades_seeds.py --out src/tessera_apps/data/everglades_seeds.geojson
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import box, mapping

from tessera_apps import validate as v
from tessera_apps.demo import demo_aoi
from tessera_apps.io import TesseraReader, ZonePlan, plan_aoi

SNAPSHOT = "B6AFRJGX5XQC0A716PA0"
ERODE_PX = 5
SQUARE_PX = 10
GRID_COLS, GRID_ROWS = 4, 3
EIGHT = np.ones((3, 3), dtype=bool)

SOURCE = "ESA WorldCover 2020 v100, class 95 (mangroves)"
METHOD = (
    "WorldCover 2020 class 95 inside the AOI on the TESSERA zone 17N 10 m grid, eroded by 5 pixels (about 50 m, "
    "3 x 3 structuring element) to keep stand interiors; 8-connected components; the AOI window split into a "
    "4 x 3 grid and the largest component whose centroid falls in each cell picked (a cell with none is filled "
    "with the unpicked component farthest from those already picked); one 10 x 10 pixel (100 m) square wholly "
    "inside each eroded component, at the feasible position nearest its centroid "
    "(scripts/make_everglades_seeds.py)."
)
NOTE = (
    "These seeds and the default WorldCover validation reference come from the same product, so the wetland app's "
    "score against WorldCover is not independent of its seeds. Global Mangrove Watch stands should replace them "
    "where zenodo.org is reachable (it is blocked from the build environment)."
)


def feasible_corners(component: np.ndarray, size: int) -> np.ndarray:
    """Top-left corners (row, col) of every ``size`` x ``size`` window wholly inside ``component``."""
    m = component.astype(np.int64)
    integral = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = m.cumsum(0).cumsum(1)
    sums = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    rows, cols = np.nonzero(sums == size * size)
    return np.stack([rows, cols], axis=1)


def place_square(labels: np.ndarray, label: int, centroid: tuple[float, float]) -> tuple[int, int] | None:
    """The top-left pixel of the feasible square nearest a component's centroid (ties: north, then west)."""
    rr, cc = np.nonzero(labels == label)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    sub = labels[r0:r1, c0:c1] == label
    if sub.shape[0] < SQUARE_PX or sub.shape[1] < SQUARE_PX:
        return None
    corners = feasible_corners(sub, SQUARE_PX)
    if corners.size == 0:
        return None
    centre = corners + (SQUARE_PX - 1) / 2.0 + np.array([r0, c0])
    d2 = (centre[:, 0] - centroid[0]) ** 2 + (centre[:, 1] - centroid[1]) ** 2
    order = np.lexsort((corners[:, 1], corners[:, 0], np.round(d2, 6)))
    best = corners[order[0]]
    return int(best[0] + r0), int(best[1] + c0)


def square_lonlat(plan: ZonePlan, row: int, col: int) -> shapely.Polygon:
    """The lon/lat polygon of a SQUARE_PX square whose top-left pixel is (row, col) of the plan window."""
    t = plan.transform
    x0, y0 = t.c + col * t.a, t.f + row * t.e
    square = box(x0, y0 + SQUARE_PX * t.e, x0 + SQUARE_PX * t.a, y0)
    # Over 100 m the UTM edges are straight in lon/lat to far below a millimetre, so the corners suffice.
    lonlat = shapely.transform(square, plan.grid.to_lonlat().transform, interleaved=False)
    return shapely.set_precision(lonlat, 1e-7)


def main() -> None:
    """Write the seeds GeoJSON."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snapshot", default=SNAPSHOT, help="Icechunk snapshot whose zone grid the seeds follow.")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="Date recorded in the seed properties.")
    args = ap.parse_args()

    reader = TesseraReader.open(snapshot_id=args.snapshot)
    (plan,) = plan_aoi(reader, demo_aoi("everglades")).zones
    mangrove = v.worldcover_classes(plan, 2020, [v.WORLDCOVER_MANGROVE])
    eroded = ndimage.binary_erosion(mangrove, structure=EIGHT, iterations=ERODE_PX) & plan.mask
    labels, n = ndimage.label(eroded, structure=EIGHT)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    centroids = ndimage.center_of_mass(eroded, labels, range(1, n + 1))
    height, width = plan.shape
    print(
        f"{plan.zone} window {height} x {width}: {int(mangrove.sum()):,} mangrove pixels, "
        f"{int(eroded.sum()):,} after eroding {ERODE_PX} px, {n} components"
    )

    # Components large enough to hold a square, largest first (ties by label).
    candidates = [i for i in np.argsort(-sizes[1:], kind="stable") + 1 if sizes[i] >= SQUARE_PX * SQUARE_PX]
    picked: list[tuple[int, str, tuple[int, int]]] = []
    used: set[int] = set()
    empty_cells: list[str] = []
    for gr in range(GRID_ROWS):
        for gc in range(GRID_COLS):
            r0, r1 = gr * height / GRID_ROWS, (gr + 1) * height / GRID_ROWS
            c0, c1 = gc * width / GRID_COLS, (gc + 1) * width / GRID_COLS
            cell = f"row {gr} col {gc}"
            for i in candidates:
                cy, cx = centroids[i - 1]
                if i in used or not (r0 <= cy < r1 and c0 <= cx < c1):
                    continue
                corner = place_square(labels, i, (cy, cx))
                if corner is not None:
                    picked.append((i, cell, corner))
                    used.add(i)
                    break
            else:
                empty_cells.append(cell)
    for cell in empty_cells:
        best, best_d, best_corner = None, -1.0, None
        for i in candidates:
            if i in used:
                continue
            cy, cx = centroids[i - 1]
            d = min(math.hypot(cy - centroids[j - 1][0], cx - centroids[j - 1][1]) for j, _, _ in picked)
            if d > best_d:
                corner = place_square(labels, i, (cy, cx))
                if corner is not None:
                    best, best_d, best_corner = i, d, corner
        if best is None or best_corner is None:
            raise SystemExit(f"No component left to fill grid cell {cell}.")
        picked.append((best, f"{cell} (empty, filled with the farthest component)", best_corner))
        used.add(best)

    features = []
    for k, (label, cell, (row, col)) in enumerate(picked, start=1):
        assert eroded[row : row + SQUARE_PX, col : col + SQUARE_PX].all()
        polygon = square_lonlat(plan, row, col)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "role": "seed",
                    "seed_id": k,
                    "class": "mangrove",
                    "source": SOURCE,
                    "method": METHOD,
                    "date": args.date,
                    "note": NOTE,
                    "grid_cell": cell,
                    "component_pixels_eroded": int(sizes[label]),
                    "zone": plan.zone,
                    "zone_rows": [plan.rows.start + row, plan.rows.start + row + SQUARE_PX],
                    "zone_cols": [plan.cols.start + col, plan.cols.start + col + SQUARE_PX],
                    "pixels": SQUARE_PX * SQUARE_PX,
                    "store_snapshot": args.snapshot,
                },
                "geometry": mapping(polygon),
            }
        )
        print(f"seed {k:2d}: {cell}, component of {int(sizes[label]):,} px, window rows {row} cols {col}")
    fc = {"type": "FeatureCollection", "features": features}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fc, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(features)} seeds, {len(features) * SQUARE_PX * SQUARE_PX:,} pixels, to {args.out}")


if __name__ == "__main__":
    main()
