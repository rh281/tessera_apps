"""Checkpoint 2 integration: io blocks into metrics, metrics into export, and validate on the same grids.

Each module has its own tests; these check that they fit together the way an app will chain them:
``plan_aoi`` and ``iter_plan_blocks`` (io) give windows that metrics accepts, per-zone results go
through ``mosaic_zones`` and ``write_geotiff`` (export), and the reference loaders (validate) return
masks on the same ``ZonePlan`` grids, so outputs and references line up pixel for pixel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
import shapely
from pyproj import Transformer
from rio_cogeo.cogeo import cog_validate
from shapely.geometry import Polygon, box

from tessera_apps import export, metrics, validate
from tessera_apps.demo import demo_feature
from tessera_apps.export import GridArray, mosaic_zones, write_geotiff
from tessera_apps.io import (
    AOIPlan,
    TesseraReader,
    ZoneGrid,
    ZonePlan,
    iter_plan_blocks,
    plan_aoi,
    read_plan,
    zone_epsg,
    zone_for,
)

from .conftest import BOUNDARY, PAD_COLS, PAD_ROWS

BLOCK = (32, 32)


def _stream_distance(reader: TesseraReader, plan: ZonePlan, year_a: int, year_b: int) -> tuple[np.ndarray, list]:
    """Cosine distance on the plan grid, computed block by block as an app streams a large AOI."""
    out = np.full(plan.shape, np.nan, dtype=np.float32)
    blocks = list(iter_plan_blocks(reader, plan, [year_a, year_b], block=BLOCK))
    for blk in blocks:
        r = slice(blk.rows.start - plan.rows.start, blk.rows.stop - plan.rows.start)
        c = slice(blk.cols.start - plan.cols.start, blk.cols.stop - plan.cols.start)
        out[r, c] = metrics.cosine_distance_map(blk.windows[year_a], blk.windows[year_b])
    return out, blocks


def _caldor_plan(reader: TesseraReader, caldor_fixture: dict[str, np.ndarray]) -> AOIPlan:
    """An AOI 2 pixels inside the real 64 x 64 Caldor window, as a lon/lat polygon."""
    fx = caldor_fixture
    a, _, c, _, _, f = (float(v) for v in fx["transform"])
    n = fx["scales"].shape[1]
    square = box(c + 2 * a, f - (n - 2) * a, c + (n - 2) * a, f - 2 * a)
    to_ll = Transformer.from_crs(int(fx["epsg"]), 4326, always_xy=True)
    aoi = shapely.transform(shapely.segmentize(square, 10.0), to_ll.transform, interleaved=False)
    return plan_aoi(reader, aoi)


def test_caldor_blocks_metrics_cog_and_reference_line_up(caldor_reader, caldor_fixture, tmp_path):
    aoi = _caldor_plan(caldor_reader, caldor_fixture)
    (plan,) = aoi.zones
    assert plan.zone == "10N"
    # The plan sits on the fixture window inside the padded group.
    assert plan.rows.start >= PAD_ROWS and plan.cols.start >= PAD_COLS

    dist, blocks = _stream_distance(caldor_reader, plan, 2020, 2022)
    assert len(blocks) > 1

    # Streaming in blocks gives exactly what one whole-window read gives, on every AOI pixel.
    wa, wb = (read_plan(caldor_reader, plan, y) for y in (2020, 2022))
    whole = metrics.cosine_distance_map(wa, wb)
    np.testing.assert_array_equal(dist[plan.mask], whole[plan.mask])
    assert np.isfinite(dist[plan.mask]).all()

    # Pooled block means equal the whole-window mean, and block windows carry their AOI masks.
    mean_blocks, n_blocks = metrics.mean_embedding([b.windows[2020] for b in blocks])
    mean_whole, n_whole = metrics.mean_embedding(wa)
    assert n_blocks == n_whole == plan.n_pixels
    np.testing.assert_allclose(mean_blocks, mean_whole, rtol=1e-12)

    # A block's trajectory against 2020 is the complement of the distance map on its AOI pixels.
    blk = blocks[0]
    traj = metrics.trajectory(blk.windows, 2020)
    d_blk = metrics.cosine_distance_map(blk.windows[2020], blk.windows[2022])
    np.testing.assert_array_equal((np.float32(1) - traj[2022])[blk.mask], d_blk[blk.mask])
    assert np.isnan(traj[2022][~blk.mask]).all()
    assert metrics.same_radar_regime(blk.windows[2020], blk.windows[2022])[blk.mask].all()

    # Otsu on the AOI's distances separates the burn (about half the window).
    otsu = metrics.otsu_threshold(dist[plan.mask])
    assert otsu.bimodal and 0.3 < otsu.high_mode_share < 0.7
    burned = metrics.clean_mask(dist > otsu.threshold, min_pixels=metrics.area_to_pixels(1.0)) & plan.mask

    # One zone: the mosaic is the piece itself with non-AOI pixels set to nodata.
    out, transform, crs = mosaic_zones([GridArray(dist, plan)], nodata=float("nan"))
    assert transform == plan.transform and crs == plan.grid.crs
    np.testing.assert_array_equal(out[plan.mask], dist[plan.mask])
    assert np.isnan(out[~plan.mask]).all()

    tif = write_geotiff(tmp_path / "wildfire_2020_2022.tif", out, transform, crs, band_names=["cosine_distance"])
    valid, errors, _ = cog_validate(str(tif))
    assert valid, errors
    with rasterio.open(tif) as ds:
        assert ds.transform == plan.transform and ds.crs.to_epsg() == plan.grid.epsg
        np.testing.assert_array_equal(ds.read(1), out)

    # validate reads the export back onto the same plan grid without moving a pixel.
    back = validate.reference_from_raster(plan, tif, lambda v: v > otsu.threshold)
    np.testing.assert_array_equal(back, (dist > otsu.threshold) & plan.mask)

    # The Hansen 2021 loss outline covers about half the fixture: the burned mask agrees with it.
    ref = validate.reference_from_geometries(plan, [demo_feature("caldor", "hansen_2021_loss_outline")])
    assert ref.shape == plan.shape and not (ref & ~plan.mask).any()
    scores = validate.binary_scores(burned, ref, valid=plan.mask)
    assert scores.n == plan.n_pixels
    assert scores.iou is not None and scores.iou > 0.6, scores
    path = validate.write_validation_json(
        tmp_path / "wildfire_validation.json",
        scores={"burned": scores},
        reference={"name": "hansen_2021_loss_outline"},
        caveats=["Hansen records stand-replacing loss only."],
    )
    assert json.loads(path.read_text())["scores"]["burned"]["tp"] == scores.tp


@pytest.fixture
def two_zone(boundary_reader) -> tuple[AOIPlan, Polygon]:
    lon0, lat0 = BOUNDARY
    aoi = box(lon0 - 0.004, lat0 - 0.003, lon0 + 0.003, lat0 + 0.003)
    plan = plan_aoi(boundary_reader, aoi)
    assert plan.zone_names == ["10N", "11N"]
    return plan, aoi


def test_two_zone_blocks_mosaic_and_cog(boundary_reader, two_zone, tmp_path):
    plan, _ = two_zone
    boundary_reader.check_years(plan.zone_names, [2020, 2021])
    pieces = []
    for p in plan.zones:
        dist, blocks = _stream_distance(boundary_reader, p, 2020, 2021)
        assert len(blocks) > 1
        whole = metrics.cosine_distance_map(read_plan(boundary_reader, p, 2020), read_plan(boundary_reader, p, 2021))
        np.testing.assert_array_equal(dist[p.mask], whole[p.mask])
        assert np.isfinite(dist[p.mask]).all()
        # The fixture drops radar in 2022: comparing across that change is what same_radar_regime flags.
        assert all(metrics.same_radar_regime(b.windows[2020], b.windows[2021]).all() for b in blocks)
        later = next(iter_plan_blocks(boundary_reader, p, [2021, 2022], block=BLOCK))
        assert not metrics.same_radar_regime(later.windows[2021], later.windows[2022]).any()
        pieces.append(GridArray(dist, p))

    for dst in ("10N", "11N"):
        out, transform, crs = mosaic_zones(pieces, dst_zone=dst, nodata=float("nan"))
        dst_plan = next(p for p in plan.zones if p.zone == dst)
        assert crs == dst_plan.grid.crs
        # Every AOI pixel once, give or take the reprojected seam (export's own tests pin the rule).
        assert abs(int(np.isfinite(out).sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels

        tif = write_geotiff(tmp_path / f"two_zone_{dst}.tif", out, transform, crs)
        valid, errors, _ = cog_validate(str(tif))
        assert valid, errors

        # The mosaic's grid is the destination zone's own grid: reading it back onto that plan is
        # exact on the plan's AOI pixels.
        src = next(g for g in pieces if g.plan.zone == dst)
        back = validate.reference_from_raster(dst_plan, tif, lambda v: v > 0.5)
        np.testing.assert_array_equal(back, (src.data > 0.5) & dst_plan.mask)
        # The other zone comes back through two nearest-neighbour warps: nearly every AOI pixel is covered.
        away = next(p for p in plan.zones if p.zone != dst)
        covered = validate.reference_from_raster(away, tif, np.isfinite)
        assert covered.sum() >= 0.97 * away.n_pixels


def test_two_zone_references_mosaic_and_merged_scores(two_zone, tmp_path):
    plan, aoi = two_zone
    lon0, lat0 = BOUNDARY
    burn = box(lon0 - 0.002, lat0 - 0.0015, lon0 + 0.0015, lat0 + 0.0015)
    shifted = shapely.affinity.translate(burn, xoff=0.0004, yoff=0.0002)

    refs, preds = [], []
    for p in plan.zones:
        # io and validate use the same pixel-centre rule: the AOI rasterizes to the plan mask.
        np.testing.assert_array_equal(validate.reference_from_geometries(p, [aoi]), p.mask)
        ref = validate.reference_from_geometries(p, [burn])
        pred = validate.reference_from_geometries(p, [shifted])
        assert ref.shape == p.shape and ref.any() and not (ref & ~p.mask).any()
        refs.append(ref)
        preds.append(pred)

    parts = [validate.binary_scores(pr, rf, valid=p.mask) for pr, rf, p in zip(preds, refs, plan.zones, strict=True)]
    merged = validate.merge_scores(parts)
    pooled = validate.binary_scores(
        np.concatenate([pr[p.mask] for pr, p in zip(preds, plan.zones, strict=True)]),
        np.concatenate([rf[p.mask] for rf, p in zip(refs, plan.zones, strict=True)]),
    )
    assert merged == pooled
    assert merged.n == plan.n_pixels

    # bool references mosaic to uint8 with nodata 255, and the burn keeps its pixel count.
    out, transform, crs = mosaic_zones(
        [GridArray(rf, p) for rf, p in zip(refs, plan.zones, strict=True)], nodata=export.BOOL_NODATA
    )
    assert out.dtype == np.uint8
    n_ref = sum(int(rf.sum()) for rf in refs)
    assert abs(int((out == 1).sum()) - n_ref) <= 0.03 * n_ref
    assert abs(int((out != export.BOOL_NODATA).sum()) - plan.n_pixels) <= 0.01 * plan.n_pixels
    tif = write_geotiff(tmp_path / "reference.tif", out, transform, crs, nodata=export.BOOL_NODATA)
    valid, errors, _ = cog_validate(str(tif))
    assert valid, errors


def test_export_zone_band_rule_matches_io_zone_for():
    # mosaic_zones decides seam ownership with a vectorised copy of io.zone_for: keep them identical.
    rng = np.random.default_rng(0)
    edges = np.arange(-180.0, 181.0, 6.0)
    lon = np.concatenate([rng.uniform(-180, 180, 2000), edges, np.nextafter(edges, -np.inf)[1:], [180.0]])
    lat = np.concatenate([rng.uniform(-80, 84, lon.size - 3), [0.0, -0.0, np.nextafter(0.0, -1.0)]])
    zones = {zone_for(float(x), float(y)) for x, y in zip(lon, lat, strict=True)}
    for z in zones:
        expected = np.array([zone_for(float(x), float(y)) == z for x, y in zip(lon, lat, strict=True)])
        np.testing.assert_array_equal(export._in_zone_band(lon, lat, z), expected)


@pytest.mark.parametrize("root", [None, "custom-cache"])
def test_reference_cache_and_chunk_cache_share_one_root(root, tmp_path):
    # Chunk caches and reference downloads resolve one root, read when called, so they always agree.
    env = {k: v for k, v in os.environ.items() if k != "TESSERA_APPS_CACHE"}
    env["HOME"] = str(tmp_path / "home")
    if root is not None:
        env["TESSERA_APPS_CACHE"] = str(tmp_path / root)
    code = (
        "from tessera_apps import cache, validate\n"
        "chunks, refs = cache.ChunkCache().root, validate.cache_dir()\n"
        "assert chunks == refs / 'chunks', (chunks, refs)\n"
        "print(validate.cache_dir())\n"
    )
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    expected = tmp_path / root if root is not None else tmp_path / "home" / ".cache" / "tessera_apps"
    assert Path(done.stdout.strip()) == expected


class _GridOnlyReader:
    """Live-store zone grids and nothing else: enough for plan_aoi anywhere, no chunk reads."""

    def grid(self, zone: str) -> ZoneGrid:
        north = zone.endswith("N")
        return ZoneGrid(
            zone=zone,
            epsg=zone_epsg(zone),
            left=163_840.0,
            top=9_338_880.0 if north else 10_014_720.0,
            height=933_888 if north else 890_880,
            width=67_584,
        )


@pytest.mark.parametrize("dst", ["10N", "11N"])
def test_mosaic_takes_the_source_pixel_under_each_centre(dst):
    # validate reads references onto plan grids with GDAL's exact transformer (pixel-centre rule).
    # mosaic_zones must warp the same way: with GDAL's default 0.125 pixel approximation about 50 of
    # these 77,000 away-zone pixels (2 percent at Caldor scale) take a neighbouring source pixel.
    lon0, lat0 = BOUNDARY
    plan = plan_aoi(_GridOnlyReader(), box(lon0 - 0.02, lat0 - 0.02, lon0 + 0.02, lat0 + 0.02))  # type: ignore[arg-type]
    base = 10**8
    pieces = [
        GridArray((k * base + np.arange(p.shape[0] * p.shape[1])).reshape(p.shape), p)
        for k, p in enumerate(plan.zones, 1)
    ]
    out, transform, crs = mosaic_zones(pieces, dst_zone=dst, nodata=-1)
    k_away, away = next((k, p) for k, p in enumerate(plan.zones, 1) if p.zone != dst)
    rr, cc = np.nonzero(out // base == k_away)
    x = transform.c + (cc + 0.5) * transform.a
    y = transform.f + (rr + 0.5) * transform.e
    ax, ay = Transformer.from_crs(crs, away.grid.crs, always_xy=True).transform(x, y)
    sr = np.floor((away.transform.f - ay) / 10).astype(np.int64)
    sc = np.floor((ax - away.transform.c) / 10).astype(np.int64)
    inside = (sr >= 0) & (sr < away.shape[0]) & (sc >= 0) & (sc < away.shape[1])
    under = np.zeros(rr.shape, dtype=bool)
    under[inside] = away.mask[sr[inside], sc[inside]]
    # Seam pixels (no away mask pixel under the centre) are filled from a neighbour by design.
    assert (~under).sum() < 0.01 * rr.size
    np.testing.assert_array_equal(out[rr, cc][under] - k_away * base, (sr * away.shape[1] + sc)[under])
