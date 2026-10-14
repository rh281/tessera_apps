"""Cut a small window of the live store into an npz fixture for offline tests.

Usage:
    python scripts/make_fixture.py --lon -120.36 --lat 38.69 --size 64 --out tests/fixtures/caldor_64.npz

Reads every published year of every layer for one ``size`` x ``size`` window, chosen inside the
256 x 256 chunk that contains the point so the read costs one chunk per year and layer. With
``--pick-burn`` the window inside that chunk is the one whose 2020 to 2022 cosine distance is
closest to half above ``--threshold``, so the fixture holds both burned and unburned pixels.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from tessera_apps.io import ALL_LAYERS, YEARS, TesseraReader, zone_for


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    num = (af * bf).sum(-1)
    den = np.linalg.norm(af, axis=-1) * np.linalg.norm(bf, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - num / den


def main() -> None:
    """Write the fixture."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pick-burn", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.281)
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    t0 = time.time()
    reader = TesseraReader.open(snapshot_id=args.snapshot)
    zone = zone_for(args.lon, args.lat)
    grid = reader.grid(zone)
    ch, cw = reader.chunk_shape(zone)
    x, y = grid.from_lonlat().transform(args.lon, args.lat)
    r, c = grid.rowcol(x, y)
    cr0, cc0 = (r // ch) * ch, (c // cw) * cw
    s = args.size
    r0, c0 = r - s // 2, c - s // 2
    if args.pick_burn:
        a = reader.read_window(zone, 2020, slice(cr0, cr0 + ch), slice(cc0, cc0 + cw), layers=["embeddings", "scales"])
        b = reader.read_window(zone, 2022, slice(cr0, cr0 + ch), slice(cc0, cc0 + cw), layers=["embeddings", "scales"])
        assert a.embeddings is not None and b.embeddings is not None
        burned = _cosine_distance(a.embeddings, b.embeddings) > args.threshold
        best = None
        for i in range(0, ch - s + 1, 8):
            for j in range(0, cw - s + 1, 8):
                frac = float(burned[i : i + s, j : j + s].mean())
                if best is None or abs(frac - 0.5) < abs(best[0] - 0.5):
                    best = (frac, i, j)
        assert best is not None
        print(f"picked window with burned fraction {best[0]:.2f}")
        r0, c0 = cr0 + best[1], cc0 + best[2]
    else:
        # Keep the window inside one chunk so each year costs one chunk read.
        r0 = min(max(r0, cr0), cr0 + ch - s)
        c0 = min(max(c0, cc0), cc0 + cw - s)

    years = [yr for yr in YEARS if yr in reader.years_complete(zone)]
    stacks: dict[str, list[np.ndarray]] = {name: [] for name in ALL_LAYERS}
    for yr in years:
        win = reader.read_window(zone, yr, slice(r0, r0 + s), slice(c0, c0 + s), layers=ALL_LAYERS)
        for name in ALL_LAYERS:
            arr = getattr(win, name)
            assert arr is not None
            stacks[name].append(arr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prov = reader.provenance()
    payload: dict[str, Any] = {
        "zone": np.array(zone),
        "row0": np.array(r0),
        "col0": np.array(c0),
        "years": np.array(years),
        "transform": np.array(grid.window_transform(r0, c0)[:6]),
        "epsg": np.array(grid.epsg),
        "snapshot_id": np.array(str(prov.get("snapshot_id"))),
        **{name: np.stack(v) for name, v in stacks.items()},
    }
    np.savez_compressed(args.out, **payload)
    mb = reader.bytes_read / 1e6
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), zone {zone} rows {r0}:{r0 + s} "
        f"cols {c0}:{c0 + s}, read {mb:.1f} MB in {time.time() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()
