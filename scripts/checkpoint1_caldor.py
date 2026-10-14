"""Checkpoint 1: read the Caldor AOI for 2020 and print shape, valid fraction, bytes read and seconds."""

import json
import sys
import time
from importlib.resources import files

from shapely.geometry import shape

from tessera_apps.io import TesseraReader, iter_plan_blocks, plan_aoi

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
t0 = time.time()
reader = TesseraReader.open()
aoi = shape(json.loads(files("tessera_apps.data").joinpath("caldor.geojson").read_text())["features"][0]["geometry"])
plan = plan_aoi(reader, aoi)
reader.check_years(plan.zone_names, [year])
print(
    f"snapshot {reader.provenance()['snapshot_id']}, zones {plan.zone_names}, AOI {plan.area_ha:,.0f} ha, "
    f"estimated {plan.estimated_bytes(reader) / 1e9:.2f} GB, planned in {time.time() - t0:.1f} s"
)
for p in plan.zones:
    valid = radar_free = 0
    for blk in iter_plan_blocks(reader, p, [year]):
        w = blk.windows[year]
        valid += int((w.valid & blk.mask).sum())
        radar_free += int((w.radar_free & blk.mask).sum())
    print(
        f"{p.zone}: window {p.shape[0]} x {p.shape[1]}, {p.n_pixels:,} AOI px, valid {valid / p.n_pixels:.4f}, "
        f"radar-free {radar_free / max(valid, 1):.4f}"
    )
print(
    f"read {reader.bytes_read / 1e9:.2f} GB in {time.time() - t0:.0f} s "
    f"({reader.bytes_read / 1e6 / (time.time() - t0):.0f} MB/s)"
)
