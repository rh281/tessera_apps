"""Recompute the evidence behind every bundled restore demo polygon from the public layers and the store.

For each restore demo (``tessera_apps.demo.RESTORE_DEMOS``: ``itapebi``, ``jaru``) and each of its
polygons, the site (role ``aoi``), the mature target, the negative (pasture that stayed pasture) and
the site's 1 to 3 km ring (built as the restore app builds it, in the site's main UTM zone, less the
target), on the TESSERA 10 m grid of the zone (the plan is store metadata only):

- **GLAD annual tree canopy height** (Potapov et al.), every year 2015 to 2024: the mean height
  over the pixels the layer covers and the share of 5 m or more (``validate.glad_tch``).
- **Hansen GFC v1.13**: the mean ``treecover2000`` and the 10 m pixels of loss by year
  (``validate.hansen_layer``).
- **ESA WorldCover** 2020 (v100) and 2021 (v200): the share of each class (``validate.worldcover_map``).
- **The TESSERA store**: the zone, and for every year 2017 to 2025 the share of pixels with data
  (a finite, positive scale) and with radar (data and at least one Sentinel-1 observation), from the
  cheap layers only (``scales``, ``s1_asc_obs_count``, ``s1_desc_obs_count``; no embeddings read).
  A zone that does not publish a year leaves that year out, recorded as missing.

A layer that cannot be loaded (a host down, a tile missing) is recorded with its error, and the
rest of the evidence is still computed. Prints one table per demo and layer group; ``--out``
writes everything as JSON (atomically), with the store snapshot, the bytes of chunks read and the
wall time. The store is opened anonymously: no AWS credentials are needed.

Usage:
    python scripts/check_restore_demos.py [--demo itapebi] [--role aoi] [--out evidence.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio.errors
from shapely.geometry.base import BaseGeometry

from tessera_apps import demo, export, io, validate
from tessera_apps.apps import restore as rs

logger = logging.getLogger("check_restore_demos")

#: The polygons checked per demo, in table order: the demo's feature roles, then the ring.
ROLES = ("aoi", "target", "negative", "ring")
#: The site's ring, as the restore app's default.
RING_KM = (rs.DEFAULT_RING_INNER_KM, rs.DEFAULT_RING_OUTER_KM)
#: Store years checked for data and radar.
STORE_YEARS = tuple(range(io.FIRST_YEAR, io.LAST_YEAR + 1))
#: The store layers read: no embeddings.
STORE_LAYERS = ("scales", *io.RADAR_LAYERS)
#: WorldCover years.
WORLDCOVER_YEARS = tuple(sorted(validate.WORLDCOVER_VERSIONS))


def demo_polygons(name: str) -> dict[str, BaseGeometry]:
    """The polygons of one restore demo by role: its ``aoi``, ``target`` and ``negative`` features and the ring.

    The ring is ``RING_KM`` from the site's edge in the site's main zone, less the target, as the
    restore app's control candidates.
    """
    site = demo.demo_feature(name, "aoi")
    out = {role: demo.demo_feature(name, role) for role in ROLES if role != "ring"}
    ring = rs.ring_geometry(site, *RING_KM, zone=rs.main_zone(site))
    out["ring"] = io._polygonal(ring.difference(out["target"]))
    return out


def _masked(plan: io.AOIPlan, layer: Callable[..., np.ndarray], *args: object, **kwargs: object) -> np.ndarray:
    """``layer(zone_plan, *args, **kwargs)`` inside the plan's mask, pooled over its zones."""
    parts = [np.asarray(layer(zp, *args, **kwargs))[zp.mask] for zp in plan.zones]
    return np.concatenate(parts) if parts else np.zeros(0)


def _guarded(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one layer's evidence; a layer that cannot be loaded gives ``{"error": ...}``."""
    try:
        return fn()
    except (validate.ReferenceDataError, rasterio.errors.RasterioError, OSError, ValueError) as exc:
        logger.warning("could not load a layer: %s", exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def canopy_evidence(plan: io.AOIPlan, *, url_template: str) -> dict[str, Any]:
    """GLAD canopy height per year: mean metres and share of 5 m or more over the pixels the layer covers."""
    out: dict[str, Any] = {}
    for year in validate.GLAD_TCH_YEARS:
        h = _masked(plan, validate.glad_tch, year, url_template=url_template)
        h = h[np.isfinite(h)]
        out[str(year)] = {
            "mean_m": float(h.mean()) if h.size else None,
            "tall_share": float(np.mean(h >= validate.GLAD_TCH_MIN_TREE_M)) if h.size else None,
            "pixels_covered": int(h.size),
        }
    return out


def hansen_evidence(plan: io.AOIPlan, *, url_template: str) -> dict[str, Any]:
    """Hansen ``treecover2000`` mean (percent) and loss pixels by year (10 m pixels, nearest from 30 m)."""
    cover = _masked(plan, validate.hansen_layer, "treecover2000", url_template=url_template)
    loss = _masked(plan, validate.hansen_layer, "lossyear", url_template=url_template)
    counts = np.bincount(loss.astype(np.int64), minlength=1) if loss.size else np.zeros(1, dtype=np.int64)
    by_year = {str(2000 + code): int(n) for code, n in enumerate(counts) if code >= 1 and n}
    return {
        "version": validate.HANSEN_VERSION,
        "treecover2000_mean": float(cover.mean()) if cover.size else None,
        "loss_pixels": int(sum(by_year.values())),
        "loss_pixels_by_year": by_year,
    }


def worldcover_evidence(plan: io.AOIPlan, *, url_template: str) -> dict[str, Any]:
    """ESA WorldCover class shares per year, by class name (``no data`` for code 0)."""
    out: dict[str, Any] = {}
    for year in WORLDCOVER_YEARS:
        codes = _masked(plan, validate.worldcover_map, year, url_template=url_template)
        values, counts = np.unique(codes, return_counts=True)
        total = int(counts.sum())
        out[str(year)] = {
            validate.WORLDCOVER_CLASSES.get(int(v), "no data" if int(v) == 0 else f"code {int(v)}"): int(c) / total
            for v, c in zip(values, counts, strict=True)
        }
    return out


def store_evidence(reader: io.Reader, plan: io.AOIPlan, *, years: Sequence[int] = STORE_YEARS) -> dict[str, Any]:
    """Per year the share of the plan's pixels with data and with radar, from the cheap layers only.

    Years a zone does not publish are left out and listed under ``missing_years``.
    """
    published = [y for y in years if all(y in reader.years_complete(z) for z in plan.zone_names)]
    have = dict.fromkeys(published, 0)
    radar = dict.fromkeys(published, 0)
    total = plan.n_pixels
    for zp in plan.zones:
        if not published:
            break
        for blk in io.iter_plan_blocks(reader, zp, published, layers=STORE_LAYERS):
            for year, win in blk.windows.items():
                with np.errstate(invalid="ignore"):
                    ok = np.isfinite(win.scales) & (win.scales > 0) & blk.mask
                have[year] += int(ok.sum())
                radar[year] += int((ok & win.radar_present).sum())
    return {
        "zones": list(plan.zone_names),
        "pixels": int(total),
        "missing_years": [y for y in years if y not in published],
        "per_year": {
            str(y): {
                "data_share": have[y] / total if total else None,
                "radar_share": radar[y] / total if total else None,
                "radar_share_of_data": radar[y] / have[y] if have[y] else None,
            }
            for y in published
        },
    }


def polygon_evidence(
    reader: io.Reader,
    geometry: BaseGeometry,
    *,
    tch_template: str = validate.GLAD_TCH_URL_TEMPLATE,
    hansen_template: str = validate.HANSEN_URL_TEMPLATE,
    worldcover_template: str = validate.WORLDCOVER_URL_TEMPLATE,
    store_years: Sequence[int] = STORE_YEARS,
) -> dict[str, Any]:
    """Every layer's evidence for one lon/lat polygon, on its TESSERA zone plan (metadata only)."""
    plan = io.plan_aoi(reader, geometry)
    return {
        "zones": list(plan.zone_names),
        "area_ha": rs.geodesic_area_ha(geometry),
        "pixels": int(plan.n_pixels),
        "glad_tch": _guarded(lambda: canopy_evidence(plan, url_template=tch_template)),
        "hansen": _guarded(lambda: hansen_evidence(plan, url_template=hansen_template)),
        "worldcover": _guarded(lambda: worldcover_evidence(plan, url_template=worldcover_template)),
        "store": store_evidence(reader, plan, years=store_years),
    }


def _pct(value: object, digits: int = 0) -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "-"
    return f"{100.0 * float(value):.{digits}f}%"


def _num(value: object, spec: str = ".1f") -> str:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        return "-"
    return format(float(value), spec)


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A plain-text table: the first column left-aligned, the others right-aligned."""
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            str(c).ljust(widths[i]) if i == 0 else str(c).rjust(widths[i]) for i, c in enumerate(cells)
        ).rstrip()

    return "\n".join([line(headers), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def _top_classes(shares: object, n: int = 3) -> str:
    if not isinstance(shares, dict) or "error" in shares:
        return "error" if isinstance(shares, dict) else "-"
    top = sorted(shares.items(), key=lambda kv: -float(kv[1]))[:n]
    return ", ".join(f"{name} {_pct(share)}" for name, share in top)


def demo_tables(name: str, evidence: dict[str, Any]) -> str:
    """The printed tables of one demo: summary, canopy height, and store data and radar shares."""
    polys: dict[str, Any] = evidence["polygons"]
    out = [f"== {name}: {evidence.get('label', '')}"]
    summary = []
    for role, ev in polys.items():
        hansen = ev.get("hansen") or {}
        wc = ev.get("worldcover") or {}
        loss = hansen.get("loss_pixels_by_year") or {}
        summary.append(
            [
                role,
                ", ".join(ev["zones"]),
                _num(ev["area_ha"]),
                f"{ev['pixels']:,}",
                "error" if "error" in hansen else _num(hansen.get("treecover2000_mean")),
                "error" if "error" in hansen else (", ".join(f"{y}:{n}" for y, n in loss.items()) or "none"),
                _top_classes(wc.get("2020") if "error" not in wc else wc),
                _top_classes(wc.get("2021") if "error" not in wc else wc),
            ]
        )
    out.append(
        format_table(
            [
                "polygon",
                "zone",
                "ha",
                "pixels",
                "tc2000",
                "Hansen loss px by year",
                "WorldCover 2020",
                "WorldCover 2021",
            ],
            summary,
        )
    )
    canopy_rows = []
    for role, ev in polys.items():
        tch = ev.get("glad_tch") or {}
        if "error" in tch:
            canopy_rows.append([role, *(["error"] * len(validate.GLAD_TCH_YEARS))])
            continue
        canopy_rows.append(
            [
                role,
                *(
                    f"{_num((tch.get(str(y)) or {}).get('mean_m'))} / {_pct((tch.get(str(y)) or {}).get('tall_share'))}"
                    for y in validate.GLAD_TCH_YEARS
                ),
            ]
        )
    out.append("GLAD canopy height, mean m / share of 5 m or more")
    out.append(format_table(["polygon", *(str(y) for y in validate.GLAD_TCH_YEARS)], canopy_rows))
    store_rows = []
    for role, ev in polys.items():
        per = (ev.get("store") or {}).get("per_year") or {}
        cells = []
        for y in STORE_YEARS:
            rec = per.get(str(y))
            cells.append(
                "missing" if rec is None else f"{_pct(rec.get('data_share'))} / {_pct(rec.get('radar_share'))}"
            )
        store_rows.append([role, *cells])
    out.append("TESSERA store, share of pixels with data / with data and radar")
    out.append(format_table(["polygon", *(str(y) for y in STORE_YEARS)], store_rows))
    return "\n".join(out)


def check_demos(
    reader: io.Reader,
    names: Sequence[str],
    roles: Sequence[str] = ROLES,
    *,
    tch_template: str = validate.GLAD_TCH_URL_TEMPLATE,
    hansen_template: str = validate.HANSEN_URL_TEMPLATE,
    worldcover_template: str = validate.WORLDCOVER_URL_TEMPLATE,
) -> dict[str, Any]:
    """The evidence of every polygon of the named demos, by demo and role."""
    out: dict[str, Any] = {}
    for name in names:
        polygons = demo_polygons(name)
        props = demo.demo_feature_properties(name, "aoi")
        record: dict[str, Any] = {
            "label": props.get("name"),
            "planted": props.get("planted"),
            "synthetic": props.get("synthetic"),
            "ring_km": list(RING_KM),
            "polygons": {},
        }
        for role in roles:
            t0 = time.perf_counter()
            before = int(getattr(reader, "bytes_read", 0))
            ev = polygon_evidence(
                reader,
                polygons[role],
                tch_template=tch_template,
                hansen_template=hansen_template,
                worldcover_template=worldcover_template,
            )
            ev["bytes_read"] = int(getattr(reader, "bytes_read", 0)) - before
            ev["wall_s"] = round(time.perf_counter() - t0, 2)
            logger.info("%s %s: %d store bytes in %.1f s", name, role, ev["bytes_read"], ev["wall_s"])
            record["polygons"][role] = ev
        out[name] = record
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check and print (and with ``--out`` write) the evidence."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="append", choices=demo.RESTORE_DEMOS, help="A demo to check (default: all).")
    ap.add_argument("--role", action="append", choices=ROLES, help="A polygon to check (default: all four).")
    ap.add_argument("--out", type=Path, help="Write the evidence as JSON here.")
    ap.add_argument("--snapshot", help="Pin an Icechunk snapshot id.")
    ap.add_argument("--tch-template", default=validate.GLAD_TCH_URL_TEMPLATE, help="GLAD canopy height tiles.")
    ap.add_argument("--hansen-template", default=validate.HANSEN_URL_TEMPLATE, help="Hansen GFC tiles.")
    ap.add_argument("--worldcover-template", default=validate.WORLDCOVER_URL_TEMPLATE, help="WorldCover tiles.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.perf_counter()
    reader = io.TesseraReader.open(snapshot_id=args.snapshot)
    names = list(dict.fromkeys(args.demo or demo.RESTORE_DEMOS))
    roles = list(dict.fromkeys(args.role or ROLES))
    evidence = check_demos(
        reader,
        names,
        roles,
        tch_template=args.tch_template,
        hansen_template=args.hansen_template,
        worldcover_template=args.worldcover_template,
    )
    wall = time.perf_counter() - t0
    for name, record in evidence.items():
        print(demo_tables(name, record))
        print()
    nbytes = int(reader.bytes_read)
    snapshot = reader.provenance().get("snapshot_id")
    print(f"Read {nbytes / 1e6:.1f} MB of store chunks in {wall:.1f} s (snapshot {snapshot}).")
    if args.out is not None:
        export.write_json(
            args.out,
            {
                "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "store_snapshot": reader.provenance().get("snapshot_id"),
                "bytes_read": nbytes,
                "wall_s": round(wall, 2),
                "ring_km": list(RING_KM),
                "templates": {
                    "glad_tch": args.tch_template,
                    "hansen": args.hansen_template,
                    "worldcover": args.worldcover_template,
                },
                "demos": evidence,
            },
        )
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
