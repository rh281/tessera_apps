"""Checkpoint 2: every validation loader on its demo AOI. Prints what each returns, or why it could not.

Plans come from the live store (metadata only). Hansen, Natural Earth and the --reference path read
public data; NIFC, EFFIS, GMW (Zenodo) and USFS need those hosts to be reachable.
"""

import time
from collections.abc import Callable

import numpy as np
from shapely.geometry import box

from tessera_apps import validate as v
from tessera_apps.demo import demo_feature, demo_properties
from tessera_apps.io import TesseraReader, plan_aoi

reader = TesseraReader.open()
print(f"snapshot {reader.provenance()['snapshot_id']}")


def show(label: str, fn: Callable[[], object]) -> None:
    """Run one loader and print the share of AOI pixels it marks, or its error."""
    t0 = time.time()
    try:
        out = fn()
    except v.ReferenceDataError as exc:
        print(f"  {label}: UNAVAILABLE ({str(exc)[:160]})")
        return
    if not isinstance(out, str) and hasattr(out, "__len__"):
        print(f"  {label}: {len(out)} features in {time.time() - t0:.1f} s")
    else:
        print(f"  {label}: {out} in {time.time() - t0:.1f} s")


def share(plans: list, fn: Callable) -> str:
    """Fraction of AOI pixels the per-zone loader marks True, pooled over zones."""
    hit = tot = 0
    for p in plans:
        r = np.asarray(fn(p), dtype=bool)
        assert r.shape == p.shape, (r.shape, p.shape)
        hit += int((r & p.mask).sum())
        tot += p.n_pixels
    return f"raster on {len(plans)} zone(s), {hit:,} of {tot:,} AOI px ({hit / tot:.4f})"


def check_demo(name: str) -> None:
    """Run the loaders that apply to one demo AOI."""
    aoi = demo_feature(name, "aoi")
    props = demo_properties(name)
    plan = plan_aoi(reader, aoi)
    zones = plan.zones
    print(
        f"{name}: zones {plan.zone_names}, {plan.area_ha:,.0f} ha, years {props.get('year_a')} to {props.get('year_b')}"
    )
    if name == "caldor":
        outline = demo_feature("caldor", "hansen_2021_loss_outline")
        show(
            "reference_from_geometries (Hansen 2021 outline)",
            lambda: share(zones, lambda p: v.reference_from_geometries(p, [outline])),
        )
        show("NIFC perimeters 2021", lambda: v.nifc_perimeters(aoi.bounds, year=2021, name="CALDOR"))
        show(
            "EFFIS burnt areas (Europe only; probe)",
            lambda: v.effis_burnt_areas(box(2.0, 42.0, 2.1, 42.1).bounds, year=2022),
        )
        show("Hansen loss 2020 to 2021", lambda: share(zones, lambda p: v.hansen_loss(p, 2020, 2021)))
    if name == "rondonia":
        show("Hansen loss 2019 to 2022", lambda: share(zones, lambda p: v.hansen_loss(p, 2019, 2022)))
        show("Hansen forest 2019", lambda: share(zones, lambda p: v.hansen_forest(p, 2019)))
        show("Hansen valid", lambda: share(zones, v.hansen_valid))
    if name == "everglades":
        show("GMW v3 extent 2018", lambda: share(zones, lambda p: v.gmw_extent(p, 2018)))
        show("Natural Earth 10m coastline", lambda: v.natural_earth_coastline(aoi.bounds))
        show(
            "USFS reforestation 2019 to 2020 (King Fire area probe)",
            lambda: v.usfs_reforestation((-120.7, 38.7, -120.4, 38.9), completed_from=2019, completed_to=2020),
        )


for demo in ("caldor", "rondonia", "everglades"):
    check_demo(demo)
print(f"store bytes read: {reader.bytes_read:,}")
