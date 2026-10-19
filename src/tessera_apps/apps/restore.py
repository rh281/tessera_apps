"""Restore: is a restoration site following the trajectory a working project should, against a matched control?

Brief section 4.4, as corrected by the pre-build survey (B5, S2, S16, S17, S18, S19):

- **Inputs** (:func:`run_restore`). The site polygon (any CRS, polygons only; points and lines are an
  error), the planting year, and optionally a control polygon and a mature reference stand (the
  target). The three must not overlap. ``planted <= 2017`` is refused (no pre-planting year worth
  having) and a 2017 baseline warns (few Sentinel-2 observations, sparse valid pixels in places;
  survey S16). The trajectory covers every year from ``year_a`` (default ``planted - 1``, the
  baseline) to ``year_b`` (default 2025), and every one of them is checked against
  ``years_complete`` for every zone the site, the ring or control, and the target touch, before any
  chunk is read (survey S2). The error names the zone, the year and the role that touches it.
- **Plans.** The site plus a ring (or the site plus the supplied control) is planned as one extent
  with :func:`~tessera_apps.io.plan_aoi` (metadata only), the target separately. The ring is built in
  the site's main UTM zone in metres (outer buffer minus inner buffer, default 1 to 3 km) and less
  the target; its pixels are the control candidates.
- **Stage 1, the baseline** (``year_a`` only, for the site, the ring and the target). The site
  mean ``m_s`` is the dequantized mean of the valid site pixels. The control filter (survey S17) is
  the cosine of every valid ring pixel to ``m_s`` (exact on int8), kept at or above a threshold set
  on held-out site pixels: the site's pixels are blocked into cells of ``fold_cell_m`` (default 200
  m) with :func:`~tessera_apps.apps.wetland.spatial_blocks`, the cells split into ``folds`` folds
  with :func:`~tessera_apps.apps.wetland.assign_folds`, each fold scored against the mean of the
  other folds (:func:`heldout_similarity`), and the threshold is the ``control_pct`` percentile of
  those held-out similarities. A site of one cell falls back to the in-sample percentile with a
  warning (:func:`~tessera_apps.apps.wetland.choose_threshold` models the choice, on distances). A
  supplied control is used as given, and its baseline balance is reported. Too few valid site
  pixels (:data:`MIN_SITE_PIXELS`), kept control pixels (:data:`MIN_CONTROL_PIXELS`) or valid target
  pixels stop the run here, before stage 2.
- **Stage 2, the other years**, read only for the store chunks that hold a kept site, control or
  target pixel (:func:`chunk_runs`): a chunk with no kept pixel is never fetched again.
- **Panel and radar regime** (survey B5). A pixel is used only if it is valid in every trajectory
  year, so the means never change composition. Its radar pattern is its radar presence over the
  years, and only site, control and target pixels with the site's majority pattern are kept, so
  every within-year comparison is between pixels of one regime. When the majority pattern itself
  has radar-free years the run warns: the comparisons stay fair and the placebo band sees the same
  switch, but the trajectory may step at those years.
- **Trajectories** per year: the divergence ``D(t) = 1 - cos(m_s(t), m_c(t))`` and its change since
  the baseline ``dD(t) = D(t) - D(year_a)``; a placebo band from ``placebos`` pseudo-sites, each a
  random control pixel and its nearest control pixels in the main zone's metres
  (:func:`draw_placebos`), whose ``dD_p(t)`` is computed exactly as the site's against the control
  pool less the placebo; a year is significant when ``dD(t)`` exceeds the band's 95th percentile;
  the verdict (:func:`call_verdict`) and the same rule applied to every placebo (the false positive
  rate); the Theil-Sen slope and Spearman rho of ``dD`` after planting; with a target, the site's
  similarity to it, the control's (the 0 percent level) and the target's own two spatial halves'
  (the 100 percent level), giving an unclipped progress; and the per-pixel fractions
  (:func:`departed_fraction`, :func:`closer_fraction`).
- **Null test** (always computed, written to the validation file with ``validate``). The control
  pool is split into two halves by alternating angular sectors around the site centroid
  (:func:`null_halves`), and the same divergence and verdict are computed between the halves. There
  is no global reference layer for restoration outcomes, so this is a sanity check that the method
  reports no divergence where none is expected, not an accuracy (:data:`LIMITATION`).
- **Canopy height check** (with ``validate`` and ``canopy``): GLAD annual tree canopy height
  (:func:`~tessera_apps.validate.glad_tch`) over the kept pixels, the site minus control gap and its
  change since ``year_a``, its correlation with ``dD`` and the first year the site's tall share
  leads the control's by :data:`CANOPY_ONSET_GAP` (:func:`canopy_check`). It is loaded before any
  chunk read; when the host cannot be reached the run completes with a warning (or stops under
  ``strict_validate``), and an identical rerun retries it once the tiles can be reached.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
import shapely
from pyproj import Geod, Transformer
from shapely.geometry.base import BaseGeometry

from tessera_apps import demo, export, io, metrics, validate
from tessera_apps.apps import wetland
from tessera_apps.apps._common import (
    AppInputError,
    BandMosaic,
    LoadedAOI,
    RunContext,
    StrictValidationError,
    finish_run,
    load_aoi,
    prepare_run,
    stride_for_preview,
)
from tessera_apps.apps.wetland import ThresholdChoice, assign_folds, choose_threshold, spatial_blocks
from tessera_apps.export import OutputPaths
from tessera_apps.io import FIRST_YEAR, HA_PER_PIXEL, LAST_YEAR, N_BANDS, AOIPlan, Reader, ZonePlan

logger = logging.getLogger(__name__)

APP = "restore"
#: Layers every read takes: the radar counts give each pixel's regime (survey B5).
LAYERS = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
#: Smallest site, in hectares.
MIN_SITE_HA = 0.25
#: Fewest valid site pixels (and target pixels) a run accepts.
MIN_SITE_PIXELS = 25
#: Fewest kept control pixels a run accepts.
MIN_CONTROL_PIXELS = 100
#: The first planting year accepted: 2017 is the store's first year, so an earlier or equal planting
#: has no pre-planting baseline worth having (survey S16).
FIRST_PLANTED = FIRST_YEAR + 1
DEFAULT_RING_INNER_KM = 1.0
DEFAULT_RING_OUTER_KM = 3.0
#: Widest ring accepted, in km.
MAX_RING_OUTER_KM = 10.0
DEFAULT_CONTROL_PCT = 5.0
DEFAULT_FOLDS = 5
#: Side (metres, in each zone's CRS) of the cells that block the site's pixels into spatial folds:
#: a 50 ha site holds a dozen or more.
DEFAULT_FOLD_CELL_M = 200.0
DEFAULT_PLACEBOS = 200
#: Fewest placebos accepted: below this the 95th percentile of the band rests on one or two values.
MIN_PLACEBOS = 20
DEFAULT_MIN_POST_YEARS = 3
DEFAULT_SEED = 0
#: Percentiles of the placebo band.
BAND_PERCENTILES = (5.0, 50.0, 95.0)
#: Angular sectors around the site centroid that split the control pool for the null test.
NULL_SECTORS = 8
#: Fewest pixels in each half of the target for its 100 percent level.
MIN_TARGET_HALF_PIXELS = 10
#: The site's tall share must lead the control's by this much for the canopy onset year.
CANOPY_ONSET_GAP = 0.2
#: Pixels dequantized at once (2**15 x 128 float64 is 32 MB).
STRIP = 2**15
#: GDAL block cache for the output COG.
WRITE_CACHE_BYTES = 128 * 2**20

#: Codes of the ``role`` band.
ROLE_SITE = 1
ROLE_CONTROL = 2
ROLE_REJECTED = 3
ROLE_DROPPED = 4
ROLE_TARGET = 5
ROLE_NAMES = {
    ROLE_SITE: "site",
    ROLE_CONTROL: "matched control",
    ROLE_REJECTED: "ring rejected by the control filter",
    ROLE_DROPPED: "dropped (no data, or the panel or regime rule)",
    ROLE_TARGET: "target",
}

VERDICT_INSUFFICIENT = "insufficient years"
VERDICT_NONE = "no divergence detected"
VERDICT_DIVERGING = "diverging from control since {year}"

LIMITATION = (
    "There is no global reference layer for restoration outcomes. The null test (the control pool split into two "
    "halves by alternating angular sectors around the site, compared exactly as the site is) is a sanity check that "
    "the method reports no divergence where none is expected, not an accuracy. Divergence from the control says the "
    "site's embeddings moved away from comparable land that started like it; whether that movement is the intended "
    "vegetation is for the target trajectory, the canopy check and a site visit to say."
)
PROGRESS_NOTE = (
    "target_progress = (s_T - c_T) / (tau - c_T), where s_T is the site's cosine similarity to the target mean, c_T "
    "the control's (the 0 percent level) and tau the cosine between the means of two spatial halves of the target "
    "(the 100 percent level, as close as the target is to itself). It is reported unclipped: below 0 the site is "
    "further from the target than the control is, above 1 it is closer than the target's own halves."
)
PLACEBO_NOTE = (
    "Each placebo is a random control pixel and its nearest control pixels (main zone metres), as many as the site's "
    "pixels when the pool allows (n_p = min(site pixels, control pixels // 3)); its divergence is computed exactly as "
    "the site's, against the control pool less the placebo. The band is the 5th, 50th and 95th percentile of the "
    "placebos' divergence change per year, and a year is significant when the site's change exceeds the 95th."
)
DEPARTED_NOTE = (
    "departed_fraction is the share of site pixels whose cosine to the control mean is below the control pixels' own "
    "5th percentile. The control filter drops the ring pixels least like the site at the baseline, which trims the "
    "control's low tail there, so a site that did nothing sits near 0.05 plus about the share the filter trimmed from "
    "comparable land (departed_fraction_baseline gives the baseline value); read later years against it. A supplied "
    "control is not filtered and has no such offset."
)
CANOPY_CAVEATS = (
    "GLAD annual tree canopy height is a Landsat and GEDI model at 30 m that sees only trees of 5 m or more, so a "
    "young planting is invisible to it for its first years and the canopy gap lags the embeddings.",
    "The layer ends in 2024, so 2025 has no canopy value.",
    "A site picked with this layer (the bundled demo) is not independent of it for site choice, although the "
    "embeddings are independent of it.",
)
CANOPY_SKIPPED = "Canopy check skipped, GLAD tree canopy height could not be loaded"

# Peak memory model (survey S18), bytes: stage 1 keeps the int8 vector, scale, radar flag and position of
# every valid pixel of the extent and the target; stage 2 keeps every kept pixel's int8 vectors, scales and
# radar flags for each year, and reads one chunk row of a few chunks for every other year at once; the
# write holds the float32 bands of the mosaic and their per-zone copies; the canopy check holds one uint8
# height window per year and zone.
MEMORY_FIXED = 700_000_000
MEMORY_STAGE1_PER_PIXEL = 160
MEMORY_PER_PIXEL_YEAR = 140
MEMORY_PER_BLOCK_PIXEL_YEAR = 210
MEMORY_WRITE_PER_WINDOW_PIXEL_BAND = 8
MEMORY_MARGIN = 1.4


class RestoreError(io.TesseraError):
    """The data cannot answer the question as asked (too few valid site, control or target pixels)."""


# --------------------------------------------------------------------------------------------
# Inputs: polygons, the ring and the years
# --------------------------------------------------------------------------------------------


def _demo_role_aoi(name: str, role: str) -> LoadedAOI:
    """A bundled demo's feature of one role (``control`` or ``target``) as a loaded polygon."""
    try:
        geometry = demo.demo_feature(name, role)
    except ValueError as exc:
        raise AppInputError(f"--{role} demo: {exc}") from exc
    raw = demo.demo_path(name).read_bytes()
    sha = hashlib.sha256(raw + b"\0" + role.encode("utf-8")).hexdigest()
    poly = io._polygonal(geometry if geometry.is_valid else shapely.make_valid(geometry))
    return LoadedAOI(geometry=poly, path=None, sha256=sha, name=f"{name} {role}", properties={}, demo=name)


def _check_polygonal_file(path: Path, role: str) -> None:
    """Refuse a vector file with any point or line feature: the site, control and target are areas.

    Raises:
        AppInputError: Names the role, the file and the geometry types found.
    """
    import geopandas as gpd

    try:
        frame = gpd.read_file(path)
    except Exception as exc:  # pyogrio raises its own DataSourceError and friends
        raise AppInputError(f"Could not read the {role} file {path}: {exc}") from exc
    if "role" in frame.columns and (frame["role"] == "aoi").any():
        frame = frame[frame["role"] == "aoi"]
    bad: set[str] = set()
    for g in frame.geometry:
        if g is None or g.is_empty:
            continue
        for part in shapely.get_parts(g):
            if part.geom_type not in ("Polygon", "MultiPolygon"):
                bad.add(part.geom_type)
    if bad:
        raise AppInputError(
            f"The {role} file {path} holds {', '.join(sorted(bad))} features; the {role} must be polygons only "
            "(points and lines are an error). Pass the outline of the area."
        )


def load_polygon(spec: str | Path | LoadedAOI, role: str, *, demo_name: str | None = None) -> LoadedAOI:
    """Load the site, control or target: a polygon file in any CRS, a bundled demo, or a loaded AOI.

    For the site a demo name loads the demo's AOI feature (:func:`~tessera_apps.apps._common.load_aoi`);
    for the control and the target, ``"demo"`` loads the feature of that role from the site's demo.

    Args:
        spec: Path, demo name (site) or ``"demo"`` (control, target), or an already loaded AOI.
        role: ``site``, ``control`` or ``target``, for the messages.
        demo_name: The site's demo, for ``"demo"``.

    Raises:
        AppInputError: The file is missing or unreadable, holds points or lines, or has no polygon;
            or ``"demo"`` is given without a demo site.
    """
    if isinstance(spec, LoadedAOI):
        return spec
    text = str(spec).strip()
    if role != "site" and text.lower() == "demo":
        if demo_name is None:
            raise AppInputError(f"--{role} demo needs a bundled demo site (--aoi <demo name>).")
        return _demo_role_aoi(demo_name, role)
    path = Path(text).expanduser()
    if role == "site" and text.lower() in demo.DEMO_NAMES and not path.is_file():
        return load_aoi(text)
    if not path.is_file():
        also = f" nor a bundled demo ({', '.join(demo.DEMO_NAMES)})" if role == "site" else ""
        raise AppInputError(f"The {role} {text!r} is neither an existing file{also}; pass a polygon file (any CRS).")
    _check_polygonal_file(path, role)
    return load_aoi(path)


def geodesic_area_ha(geometry: BaseGeometry) -> float:
    """Area of a lon/lat geometry on the WGS84 ellipsoid, in hectares."""
    return abs(Geod(ellps="WGS84").geometry_area_perimeter(geometry)[0]) / 10_000.0


def main_zone(geometry: BaseGeometry) -> str:
    """The zone holding most of a lon/lat geometry (ties go to the first in sorted order).

    Raises:
        AppInputError: The geometry touches no zone.
    """
    zones = io.zones_for_geometry(geometry)
    if not zones:
        raise AppInputError("The site touches no UTM zone of the store; check it is in lon/lat.")
    areas = [io._polygonal(geometry.intersection(io.zone_band(z))).area for z in zones]
    return zones[int(np.argmax(areas))]


def to_zone_metres(geometry: BaseGeometry, zone: str) -> BaseGeometry:
    """A lon/lat geometry in a zone's UTM metres, densified first so long edges follow the projection."""
    tr = Transformer.from_crs(4326, io.zone_epsg(zone), always_xy=True)
    return shapely.transform(shapely.segmentize(geometry, 0.001), tr.transform, interleaved=False)


def to_lonlat(geometry: BaseGeometry, zone: str) -> BaseGeometry:
    """A geometry in a zone's UTM metres back to lon/lat (densified to 50 m first), repaired if needed."""
    tr = Transformer.from_crs(io.zone_epsg(zone), 4326, always_xy=True)
    out = shapely.transform(shapely.segmentize(geometry, 50.0), tr.transform, interleaved=False)
    return io._polygonal(out if out.is_valid else shapely.make_valid(out))


def ring_geometry(site: BaseGeometry, inner_km: float, outer_km: float, *, zone: str | None = None) -> BaseGeometry:
    """The control ring around a lon/lat site: outer buffer minus inner buffer, built in metres.

    The buffers are taken in the site's main UTM zone (``zone`` overrides it), so the ring is
    ``inner_km`` to ``outer_km`` from the site's edge on the ground, then projected back to lon/lat.
    With ``inner_km`` 0 the ring starts at the site's edge (the site itself is removed).

    Raises:
        ValueError: ``outer_km`` is not above ``inner_km``, or ``inner_km`` is negative.
    """
    if not (0.0 <= inner_km < outer_km):
        raise ValueError(f"The ring needs 0 <= inner ({inner_km}) < outer ({outer_km}).")
    z = zone or main_zone(site)
    site_m = to_zone_metres(site, z)
    outer = site_m.buffer(outer_km * 1000.0, quad_segs=16)
    inner = site_m.buffer(inner_km * 1000.0, quad_segs=16) if inner_km > 0 else site_m
    return to_lonlat(outer.difference(inner), z)


def overlap_m2(a: BaseGeometry, b: BaseGeometry, zone: str) -> float:
    """Area (square metres, in a zone's UTM CRS) where two lon/lat geometries overlap."""
    if not a.intersects(b):
        return 0.0
    return float(to_zone_metres(a, zone).intersection(to_zone_metres(b, zone)).area)


def trajectory_years(planted: int, year_a: int | None, year_b: int | None) -> tuple[int, int, list[int]]:
    """``(year_a, year_b, years)`` with the defaults filled and every rule of the planting year checked.

    ``year_a`` defaults to ``planted - 1`` and must be at most that; ``year_b`` defaults to the store's
    last year and must be at least ``planted``.

    Raises:
        AppInputError: Says which rule failed and what to change.
    """
    if int(planted) != planted:
        raise AppInputError(f"--planted ({planted}) must be a year.")
    planted = int(planted)
    if planted < FIRST_PLANTED:
        raise AppInputError(
            f"--planted {planted} is refused: the store starts in {FIRST_YEAR}, so a planting in {FIRST_YEAR} or "
            "earlier has no pre-planting year to match the control on (survey S16). Choose a site planted in "
            f"{FIRST_PLANTED} or later."
        )
    if planted > LAST_YEAR:
        raise AppInputError(f"--planted {planted} is after the store's last year, {LAST_YEAR}.")
    a = planted - 1 if year_a is None else int(year_a)
    b = LAST_YEAR if year_b is None else int(year_b)
    if a < FIRST_YEAR:
        raise AppInputError(f"--year-a {a} is before the store's first year, {FIRST_YEAR}.")
    if a > planted - 1:
        raise AppInputError(
            f"--year-a {a} must be before the planting year ({planted}): it is the baseline the control is matched "
            f"on. Use {planted - 1} or earlier."
        )
    if b > LAST_YEAR:
        raise AppInputError(f"--year-b {b} is after the store's last year, {LAST_YEAR}.")
    if b < planted:
        raise AppInputError(f"--year-b {b} must be the planting year ({planted}) or later.")
    return a, b, list(range(a, b + 1))


# --------------------------------------------------------------------------------------------
# Pure parts: means, the control filter, placebos, the verdict, the null test, the target
# --------------------------------------------------------------------------------------------


def pooled_sum(emb: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Sum of dequantized vectors ``emb * scale`` over pixels, float64 (D,), in strips.

    A cosine against a mean equals one against the sum, so the trajectories work on sums.

    Args:
        emb: (n, D) int8 vectors.
        scale: (n,) per-pixel scales.
    """
    e = np.asarray(emb)
    s = np.asarray(scale, dtype=np.float64)
    total = np.zeros(e.shape[-1], dtype=np.float64)
    for start in range(0, e.shape[0], STRIP):
        total += s[start : start + STRIP] @ e[start : start + STRIP].astype(np.float64)
    return total


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two float vectors (NaN when either is zero)."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return math.nan
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine of two (n, D) float arrays (NaN for a zero row)."""
    dot = np.einsum("nd,nd->n", a, b)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = dot / (na * nb)
    out = np.clip(out, -1.0, 1.0)
    out[(na == 0) | (nb == 0)] = np.nan
    return out


def pixel_cosine(emb: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine of every int8 pixel vector to one reference vector (exact: the per-pixel scale cancels), float64."""
    if np.asarray(emb).shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return metrics.cosine_similarity(np.asarray(emb), np.asarray(ref, dtype=np.float64)).astype(np.float64)


def heldout_similarity(emb: np.ndarray, scale: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Each pixel's cosine to the dequantized mean of the pixels in the other folds (NaN with one fold).

    The same held-out scheme as :func:`~tessera_apps.apps.wetland.heldout_scores` with the cosine
    score (one minus this), computed from per-fold sums so the vectors are never dequantized whole.

    Args:
        emb: (n, D) int8 vectors.
        scale: (n,) scales.
        folds: (n,) fold of every pixel.
    """
    f = np.asarray(folds)
    out = np.full(f.size, np.nan, dtype=np.float64)
    labels = np.unique(f)
    if labels.size < 2:
        return out
    sums = {int(k): pooled_sum(emb[f == k], scale[f == k]) for k in labels}
    total = np.sum(list(sums.values()), axis=0)
    for k, s in sums.items():
        test = f == k
        out[test] = pixel_cosine(emb[test], total - s)
    return out


def control_threshold(
    site_similarity: np.ndarray, heldout: np.ndarray, control_pct: float, n_folds: int
) -> ThresholdChoice:
    """The control filter's threshold: the ``control_pct`` percentile of the held-out site similarities.

    :func:`~tessera_apps.apps.wetland.choose_threshold` makes the choice on distances (one minus the
    similarity) at the ``100 - control_pct`` percentile, which is the same cut: ring pixels whose
    distance is at or below the returned ``threshold`` (similarity at or above ``1 - threshold``) are
    kept. With fewer than two folds it falls back to the in-sample percentile.
    """
    return choose_threshold(1.0 - np.asarray(site_similarity), 1.0 - np.asarray(heldout), 100.0 - control_pct, n_folds)


def draw_placebos(xy: np.ndarray, n_placebos: int, n_pixels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Placebo pseudo-sites: a random control pixel (seeded) and its nearest control pixels.

    Args:
        xy: (N, 2) control pixel centres in metres.
        n_placebos: How many placebos.
        n_pixels: Pixels per placebo, the centre included.
        seed: Random seed.

    Returns:
        ``(members, centres)``: (P, n_pixels) indices into the control pool, nearest first, and the (P,)
        centre indices.

    Raises:
        ValueError: ``n_pixels`` is not between 1 and N.
    """
    from scipy.spatial import cKDTree

    n = int(np.asarray(xy).shape[0])
    if not 1 <= n_pixels <= n:
        raise ValueError(f"Placebos of {n_pixels} pixels need 1 to {n} control pixels.")
    rng = np.random.default_rng(seed)
    centres = rng.choice(n, size=int(n_placebos), replace=int(n_placebos) > n)
    tree = cKDTree(np.asarray(xy, dtype=np.float64))
    _, idx = tree.query(np.asarray(xy, dtype=np.float64)[centres], k=int(n_pixels))
    members = np.asarray(idx, dtype=np.int64).reshape(len(centres), int(n_pixels))
    return members, np.asarray(centres, dtype=np.int64)


def placebo_divergence(emb: np.ndarray, scale: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Divergence ``D_p(t)`` of every placebo from the control pool less itself, per year.

    Args:
        emb: (N, Y, D) int8 control vectors.
        scale: (N, Y) control scales.
        members: (P, n_p) indices of each placebo's pixels.

    Returns:
        (P, Y) float64 ``1 - cos(sum of the placebo, sum of the rest)``.
    """
    from scipy.sparse import csc_matrix

    n, n_years = scale.shape
    p, k = members.shape
    m = csc_matrix((np.ones(p * k, dtype=np.float64), (np.repeat(np.arange(p), k), members.reshape(-1))), shape=(p, n))
    m.sum_duplicates()
    m.data[:] = 1.0
    out = np.empty((p, n_years), dtype=np.float64)
    for t in range(n_years):
        inside = np.zeros((p, emb.shape[-1]), dtype=np.float64)
        total = np.zeros(emb.shape[-1], dtype=np.float64)
        for start in range(0, n, STRIP):
            stop = min(start + STRIP, n)
            x = emb[start:stop, t, :].astype(np.float64) * scale[start:stop, t, None].astype(np.float64)
            inside += m[:, start:stop] @ x
            total += x.sum(axis=0)
        out[:, t] = 1.0 - cosine_rows(inside, total[None, :] - inside)
    return out


def placebo_band(change: np.ndarray) -> np.ndarray:
    """(3, Y) 5th, 50th and 95th percentiles of the placebos' (P, Y) divergence change per year."""
    return np.percentile(np.asarray(change, dtype=np.float64), BAND_PERCENTILES, axis=0)


def trailing_runs(significant: np.ndarray) -> np.ndarray:
    """Length of the run of True values that ends at the last column, per row of a (..., Y) bool array."""
    sig = np.asarray(significant, dtype=bool)
    flipped = sig[..., ::-1]
    # The first False from the end; a row of all True runs the whole length.
    first_false = np.argmin(flipped, axis=-1)
    return np.where(flipped.all(axis=-1), sig.shape[-1], first_false)


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The verdict on one divergence trajectory (brief step 6).

    Attributes:
        verdict: ``"diverging from control since <year>"``, ``"no divergence detected"`` or
            ``"insufficient years"``.
        since: The year divergence starts, or None.
        detail: A sentence on why.
        post_years: Post-planting years judged (``planted`` to ``year_b``).
        significant_years: Post-planting years whose change exceeds the band's 95th percentile.
    """

    verdict: str
    since: int | None
    detail: str
    post_years: list[int]
    significant_years: list[int]

    def record(self) -> dict[str, Any]:
        """The verdict for the report."""
        return dataclasses.asdict(self)


def call_verdict(
    years: Sequence[int], change: np.ndarray, p95: np.ndarray, *, planted: int, min_post_years: int
) -> Verdict:
    """The verdict over the post-planting years.

    Fewer than ``min_post_years`` post-planting years: insufficient years. Otherwise the earliest
    year ``Y >= planted`` such that every year from ``Y`` to the last is significant (the change
    exceeds the band's 95th percentile), with at least two such years: diverging since ``Y``.
    Otherwise no divergence detected.
    """
    ys = [int(y) for y in years]
    ch = np.asarray(change, dtype=np.float64)
    hi = np.asarray(p95, dtype=np.float64)
    post = np.array([y >= planted for y in ys], dtype=bool)
    post_years = [y for y in ys if y >= planted]
    with np.errstate(invalid="ignore"):
        sig = (ch > hi) & np.isfinite(ch) & np.isfinite(hi)
    sig_years = [y for y, s, p in zip(ys, sig, post, strict=True) if s and p]
    if len(post_years) < min_post_years:
        return Verdict(
            VERDICT_INSUFFICIENT,
            None,
            f"{len(post_years)} post-planting year(s) ({planted} to {ys[-1]}); the verdict needs at least "
            f"{min_post_years}. Wait for more years in the store, or lower --min-post-years.",
            post_years,
            sig_years,
        )
    run = int(trailing_runs(sig[post]))
    last = ys[-1]
    if run >= 2:
        since = last - run + 1
        return Verdict(
            VERDICT_DIVERGING.format(year=since),
            since,
            f"The divergence change exceeds the placebo band's 95th percentile in every year from {since} to {last} "
            f"({run} years).",
            post_years,
            sig_years,
        )
    if run == 1:
        detail = f"divergence in {last} only; one year is not enough to call it."
    elif sig_years:
        detail = (
            f"The divergence change exceeds the placebo band in {', '.join(str(y) for y in sig_years)}, but not in "
            f"every year through {last}."
        )
    else:
        detail = "The divergence change stays within the placebo band in every post-planting year."
    return Verdict(VERDICT_NONE, None, detail, post_years, sig_years)


def placebo_verdicts(
    years: Sequence[int], change: np.ndarray, p95: np.ndarray, *, planted: int, min_post_years: int
) -> tuple[list[str], np.ndarray]:
    """The verdict rule of :func:`call_verdict` applied to every placebo's (P, Y) change.

    Returns:
        ``(verdicts, diverging)``: the verdict text of each placebo and a (P,) bool of those diverging.
    """
    ys = [int(y) for y in years]
    ch = np.asarray(change, dtype=np.float64)
    post = np.array([y >= planted for y in ys], dtype=bool)
    n = ch.shape[0]
    if int(post.sum()) < min_post_years:
        return [VERDICT_INSUFFICIENT] * n, np.zeros(n, dtype=bool)
    with np.errstate(invalid="ignore"):
        sig = (ch > np.asarray(p95, dtype=np.float64)[None, :]) & np.isfinite(ch)
    runs = trailing_runs(sig[:, post])
    diverging = runs >= 2
    texts = [
        VERDICT_DIVERGING.format(year=ys[-1] - int(r) + 1) if d else VERDICT_NONE
        for r, d in zip(runs, diverging, strict=True)
    ]
    return texts, diverging


def false_positive_rate(diverging: np.ndarray, insufficient: bool) -> float | None:
    """Share of placebos the verdict calls diverging (None when there are too few post-planting years)."""
    d = np.asarray(diverging, dtype=bool)
    if insufficient or d.size == 0:
        return None
    return float(d.mean())


def theil_sen_trend(years: Sequence[int], change: np.ndarray) -> dict[str, Any]:
    """Theil-Sen slope (per year, with its 95 percent interval) and Spearman rho of the change.

    The slope needs two years, rho three and a change that is not constant; otherwise they are None.
    """
    from scipy.stats import spearmanr, theilslopes

    x = np.asarray(years, dtype=np.float64)
    y = np.asarray(change, dtype=np.float64)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    out: dict[str, Any] = {
        "theil_sen_slope": None,
        "theil_sen_low": None,
        "theil_sen_high": None,
        "spearman_rho": None,
        "spearman_p": None,
        "n_years": int(x.size),
    }
    if x.size >= 2:
        slope, _intercept, low, high = theilslopes(y, x)
        out.update(theil_sen_slope=float(slope), theil_sen_low=float(low), theil_sen_high=float(high))
    if x.size >= 3 and np.ptp(y) > 0:
        rho, p = spearmanr(x, y)
        out.update(spearman_rho=float(rho), spearman_p=float(p))
    return out


def null_halves(xy: np.ndarray, centre: tuple[float, float], sectors: int = NULL_SECTORS) -> np.ndarray:
    """Half (0 or 1) of every control pixel: alternating angular sectors around the site centroid.

    Args:
        xy: (N, 2) pixel centres in metres.
        centre: The site centroid in the same metres.
        sectors: Number of equal sectors; even sectors form half 0, odd ones half 1.
    """
    d = np.asarray(xy, dtype=np.float64) - np.asarray(centre, dtype=np.float64)[None, :]
    angle = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * math.pi)
    sector = np.floor(angle / (2.0 * math.pi / sectors)).astype(np.int64) % sectors
    return (sector % 2).astype(np.int8)


def target_progress(site_sim: np.ndarray, control_sim: np.ndarray, target_sim: np.ndarray) -> np.ndarray:
    """``(s_T - c_T) / (tau - c_T)`` per year, unclipped; NaN where the denominator is 0 or a term is missing."""
    s = np.asarray(site_sim, dtype=np.float64)
    c = np.asarray(control_sim, dtype=np.float64)
    tau = np.asarray(target_sim, dtype=np.float64)
    den = tau - c
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (s - c) / den
    out[~np.isfinite(out) | (den == 0)] = np.nan
    return out


def departed_fraction(site_sim: np.ndarray, control_sim: np.ndarray, pct: float = 5.0) -> float:
    """Share of site pixels whose cosine to the control mean is below the control pixels' own ``pct`` percentile.

    A site that did nothing sits near ``pct / 100``.
    """
    s = np.asarray(site_sim, dtype=np.float64)
    c = np.asarray(control_sim, dtype=np.float64)
    s, c = s[np.isfinite(s)], c[np.isfinite(c)]
    if s.size == 0 or c.size == 0:
        return math.nan
    return float(np.mean(s < np.percentile(c, pct)))


def closer_fraction(to_target: np.ndarray, to_control: np.ndarray) -> float:
    """Share of pixels whose cosine to the target mean exceeds their cosine to the control mean."""
    a = np.asarray(to_target, dtype=np.float64)
    b = np.asarray(to_control, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        return math.nan
    return float(np.mean(a[ok] > b[ok]))


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Pearson and Spearman correlation of two series (None with fewer than three finite pairs or a constant)."""
    from scipy.stats import spearmanr

    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, None
    pearson = float(np.corrcoef(a, b)[0, 1])
    rho, _p = spearmanr(a, b)
    return pearson, float(rho)


def _height_stats(h: np.ndarray, tall_m: float) -> tuple[float, float, int]:
    v = np.asarray(h, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, 0
    return float(v.mean()), float(np.mean(v >= tall_m)), int(v.size)


def canopy_check(
    years: Sequence[int],
    change: np.ndarray,
    heights: Mapping[str, Mapping[int, np.ndarray]],
    *,
    tall_m: float = validate.GLAD_TCH_MIN_TREE_M,
    onset_gap: float = CANOPY_ONSET_GAP,
) -> dict[str, Any]:
    """The canopy height check on per-pixel heights of the kept pixels.

    Args:
        years: The trajectory years, ``change`` follows them.
        change: The site's divergence change per year.
        heights: ``{"site": {year: heights}, "control": {...}, "target": {...}}``: metres per kept
            pixel, NaN where the layer has no value; the years present are the ones checked (the first
            is the baseline of the gap change).
        tall_m: Height of a tall pixel (the layer's 5 m floor).
        onset_gap: Lead of the site's tall share over the control's that sets the onset year.

    Returns:
        Per-year means, tall shares, gaps and gap changes; Pearson and Spearman correlation of ``dD``
        with the change in the mean height gap and in the tall share gap; ``canopy_onset_year``.
    """
    ys = sorted(int(y) for y in heights["site"])
    by_year = dict(zip([int(y) for y in years], np.asarray(change, dtype=np.float64), strict=True))
    per_year: dict[str, Any] = {}
    gap_mean: list[float] = []
    gap_tall: list[float] = []
    onset = None
    for y in ys:
        row: dict[str, Any] = {}
        for role in ("site", "control", "target"):
            if role not in heights:
                continue
            mean, tall, n = _height_stats(heights[role][y], tall_m)
            row[f"{role}_mean_m"] = mean
            row[f"{role}_tall_share"] = tall
            row[f"{role}_pixels_covered"] = n
        row["gap_mean_m"] = row["site_mean_m"] - row["control_mean_m"]
        row["gap_tall_share"] = row["site_tall_share"] - row["control_tall_share"]
        gap_mean.append(row["gap_mean_m"])
        gap_tall.append(row["gap_tall_share"])
        if onset is None and math.isfinite(row["gap_tall_share"]) and row["gap_tall_share"] >= onset_gap - 1e-12:
            onset = y
        per_year[str(y)] = row
    gm = np.asarray(gap_mean, dtype=np.float64)
    gt = np.asarray(gap_tall, dtype=np.float64)
    gm_change = gm - gm[0] if gm.size else gm
    gt_change = gt - gt[0] if gt.size else gt
    for y, a, b in zip(ys, gm_change, gt_change, strict=True):
        per_year[str(y)]["gap_change_mean_m"] = float(a)
        per_year[str(y)]["gap_change_tall_share"] = float(b)
    dd = np.array([by_year.get(y, math.nan) for y in ys], dtype=np.float64)
    pearson_m, spearman_m = _corr(dd, gm_change)
    pearson_t, spearman_t = _corr(dd, gt_change)
    return {
        "years": ys,
        "tall_m": tall_m,
        "per_year": per_year,
        "pearson_divergence_vs_mean_gap_change": pearson_m,
        "spearman_divergence_vs_mean_gap_change": spearman_m,
        "pearson_divergence_vs_tall_gap_change": pearson_t,
        "spearman_divergence_vs_tall_gap_change": spearman_t,
        "canopy_onset_year": onset,
        "onset_rule": f"first year the site's share of pixels of {tall_m:g} m or more exceeds the control's by "
        f"{onset_gap:g} or more",
    }


# --------------------------------------------------------------------------------------------
# Stage reads
# --------------------------------------------------------------------------------------------


def _valid(win: io.Window) -> np.ndarray:
    return win.valid & (win.scales > 0)


def chunk_runs(
    zones: np.ndarray, rows: np.ndarray, cols: np.ndarray, chunk_shape: Mapping[int, tuple[int, int]], max_run: int
) -> list[tuple[int, int, int, int]]:
    """Windows of whole chunks that hold at least one of the given pixels, a chunk row at a time.

    Chunks of one chunk row that follow each other join one window of at most ``max_run`` chunks,
    so no window holds a chunk without a pixel.

    Args:
        zones: Zone index of every pixel.
        rows: Zone grid row of every pixel.
        cols: Zone grid column of every pixel.
        chunk_shape: (rows, cols) of a chunk per zone index.
        max_run: Most chunks per window.

    Returns:
        ``(zone, chunk_row, first_chunk_col, last_chunk_col)`` per window, sorted.
    """
    keys: set[tuple[int, int, int]] = set()
    for z in np.unique(zones):
        ch, cw = chunk_shape[int(z)]
        sel = zones == z
        for r, c in set(zip((rows[sel] // ch).tolist(), (cols[sel] // cw).tolist(), strict=True)):
            keys.add((int(z), int(r), int(c)))
    out: list[tuple[int, int, int, int]] = []
    for z, r, c in sorted(keys):
        if out and out[-1][0] == z and out[-1][1] == r and out[-1][3] == c - 1 and c - out[-1][2] < max_run:
            out[-1] = (z, r, out[-1][2], c)
        else:
            out.append((z, r, c, c))
    return out


@dataclasses.dataclass
class _Pixels:
    """Pixels of one or more roles: positions and, per trajectory year, int8 vectors, scales and radar flags."""

    role: np.ndarray
    zone: np.ndarray
    row: np.ndarray
    col: np.ndarray
    emb: np.ndarray
    scale: np.ndarray
    radar: np.ndarray

    @property
    def n(self) -> int:
        return int(self.role.size)

    def take(self, sel: np.ndarray) -> _Pixels:
        return _Pixels(*(getattr(self, f.name)[sel] for f in dataclasses.fields(self)))


class _Collector:
    """Stage 1 pixels of one role, gathered block by block (year_a only)."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {k: [] for k in ("zone", "row", "col", "emb", "scale", "radar")}

    def add(self, zone: int, win: io.Window, sel: np.ndarray) -> None:
        rr, cc = np.nonzero(sel)
        if rr.size == 0:
            return
        assert win.embeddings is not None
        self.parts["zone"].append(np.full(rr.size, zone, dtype=np.int16))
        self.parts["row"].append((rr + win.rows.start).astype(np.int64))
        self.parts["col"].append((cc + win.cols.start).astype(np.int64))
        self.parts["emb"].append(np.ascontiguousarray(win.embeddings[rr, cc]))
        self.parts["scale"].append(win.scales[rr, cc].astype(np.float32))
        self.parts["radar"].append(win.radar_present[rr, cc])

    def done(self, role: int) -> _Pixels:
        empty: dict[str, np.ndarray] = {
            "zone": np.zeros(0, np.int16),
            "row": np.zeros(0, np.int64),
            "col": np.zeros(0, np.int64),
            "emb": np.zeros((0, N_BANDS), np.int8),
            "scale": np.zeros(0, np.float32),
            "radar": np.zeros(0, bool),
        }
        arr: dict[str, np.ndarray] = {k: (np.concatenate(v) if v else empty[k]) for k, v in self.parts.items()}
        return _Pixels(
            role=np.full(arr["zone"].size, role, dtype=np.int8),
            zone=arr["zone"],
            row=arr["row"],
            col=arr["col"],
            emb=arr["emb"][:, None, :],
            scale=arr["scale"][:, None],
            radar=arr["radar"][:, None],
        )


def _concat(parts: Sequence[_Pixels]) -> _Pixels:
    return _Pixels(*(np.concatenate([getattr(p, f.name) for p in parts]) for f in dataclasses.fields(_Pixels)))


def _expand_years(pix: _Pixels, n_years: int) -> _Pixels:
    """Pixels with year_a (index 0) filled from stage 1 and every other year empty (NaN scale, zero vector)."""
    emb = np.zeros((pix.n, n_years, pix.emb.shape[-1]), dtype=np.int8)
    scale = np.full((pix.n, n_years), np.nan, dtype=np.float32)
    radar = np.zeros((pix.n, n_years), dtype=bool)
    emb[:, 0] = pix.emb[:, 0]
    scale[:, 0] = pix.scale[:, 0]
    radar[:, 0] = pix.radar[:, 0]
    return dataclasses.replace(pix, emb=emb, scale=scale, radar=radar)


def _valid_years(pix: _Pixels) -> np.ndarray:
    """(n, Y) pixels valid each year: finite, positive scale and a vector that is not all zero."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(pix.scale) & (pix.scale > 0)
    return ok & np.any(pix.emb != 0, axis=-1)


# --------------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------------


def memory_estimate(plan: AOIPlan, *, block: tuple[int, int], n_years: int, target_pixels: int = 0) -> int:
    """Peak memory of a run in bytes, from the extent plan, the block shape, the years and the target size."""
    window_px = sum(z.shape[0] * z.shape[1] for z in plan.zones)
    pixels = plan.n_pixels + int(target_pixels)
    block_px = max((min(block[0], z.shape[0]) * min(block[1], z.shape[1]) for z in plan.zones), default=0)
    stage1 = MEMORY_STAGE1_PER_PIXEL * pixels + MEMORY_PER_BLOCK_PIXEL_YEAR * block_px
    stage2 = MEMORY_PER_PIXEL_YEAR * n_years * pixels + MEMORY_PER_BLOCK_PIXEL_YEAR * block_px * 2
    write = MEMORY_PER_PIXEL_YEAR * n_years * pixels + MEMORY_WRITE_PER_WINDOW_PIXEL_BAND * (n_years + 3) * window_px
    canopy = n_years * (window_px + 2 * int(target_pixels))
    return int(MEMORY_FIXED + MEMORY_MARGIN * (max(stage1, stage2, write) + canopy))


# --------------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RestoreResult:
    """What :func:`run_restore` did.

    Attributes:
        report: The report (read back from disk when the run was skipped).
        paths: The standard outputs.
        extra_paths: Outputs beyond the standard set, by name (``placebos``).
        skipped: The outputs were current, so nothing was read or written.
        validation_written: ``restore_validation.json`` belongs to this report.
    """

    report: dict[str, Any]
    paths: OutputPaths
    extra_paths: dict[str, Path]
    skipped: bool
    validation_written: bool

    @property
    def summary(self) -> dict[str, Any]:
        """The report's summary block."""
        return dict(self.report.get("summary") or {})


@dataclasses.dataclass(frozen=True)
class _Options:
    site: LoadedAOI
    control: LoadedAOI | None
    target: LoadedAOI | None
    planted: int
    years: list[int]
    ring_inner_km: float
    ring_outer_km: float
    control_pct: float
    folds: int
    fold_cell_m: float
    placebos: int
    min_post_years: int
    canopy: bool
    tch_template: str
    strict_validate: bool
    max_zones: int
    max_area_km2: float | None
    zone: str
    ring: BaseGeometry | None
    candidates: BaseGeometry


@dataclasses.dataclass
class _ZoneRoles:
    """Role masks on one extent zone plan."""

    plan: ZonePlan
    site: np.ndarray
    control: np.ndarray
    target: np.ndarray


@dataclasses.dataclass
class _State:
    roles: dict[str, _ZoneRoles] = dataclasses.field(default_factory=dict)
    target_plan: AOIPlan | None = None
    zones: list[str] = dataclasses.field(default_factory=list)
    #: Canopy heights (uint8 metres, 255 not covered) per plan kind (extent, target), zone and year.
    canopy: dict[str, dict[str, dict[int, np.ndarray]]] | None = None
    canopy_years: list[int] = dataclasses.field(default_factory=list)
    canopy_error: str | None = None


def _check_args(
    *,
    ring_inner_km: float,
    ring_outer_km: float,
    control_pct: float,
    folds: int,
    fold_cell_m: float,
    placebos: int,
    min_post_years: int,
) -> None:
    """Refuse numeric arguments the app cannot use, before anything is loaded.

    Raises:
        AppInputError: Says what to change.
    """
    if not (math.isfinite(ring_inner_km) and math.isfinite(ring_outer_km)):
        raise AppInputError("--ring-inner-km and --ring-outer-km must be numbers.")
    if not 0.0 <= ring_inner_km < ring_outer_km <= MAX_RING_OUTER_KM:
        raise AppInputError(
            f"The ring needs 0 <= --ring-inner-km ({ring_inner_km:g}) < --ring-outer-km ({ring_outer_km:g}) <= "
            f"{MAX_RING_OUTER_KM:g} km."
        )
    if not (math.isfinite(control_pct) and 0.0 < control_pct < 100.0):
        raise AppInputError(f"--control-pct ({control_pct}) must lie strictly between 0 and 100.")
    if int(folds) != folds or folds < 1:
        raise AppInputError(f"--folds ({folds}) must be a whole number, 1 or more (1 sets the threshold in sample).")
    if not (math.isfinite(fold_cell_m) and fold_cell_m >= io.PIXEL_M):
        raise AppInputError(f"--fold-cell-m ({fold_cell_m}) must be at least the {io.PIXEL_M:g} m pixel.")
    if int(placebos) != placebos or placebos < MIN_PLACEBOS:
        raise AppInputError(f"--placebos ({placebos}) must be a whole number, {MIN_PLACEBOS} or more.")
    if int(min_post_years) != min_post_years or min_post_years < 2:
        raise AppInputError(
            f"--min-post-years ({min_post_years}) must be a whole number, 2 or more: the verdict needs two significant "
            "years in a row."
        )


def _planted_from(site: LoadedAOI, planted: int | None) -> int:
    if planted is not None:
        return int(planted)
    if site.demo is not None:
        value = site.properties.get("planted")
        if value is not None:
            return int(value)
        raise AppInputError(f"The demo {site.demo!r} gives no planting year; pass --planted YYYY.")
    raise AppInputError("--planted YYYY is required: the year the site was planted.")


def _sha_record(loaded: LoadedAOI | None) -> dict[str, Any] | None:
    if loaded is None:
        return None
    return {
        "name": loaded.name,
        "source": loaded.source,
        "sha256": loaded.sha256,
        "geometry_sha256": loaded.geometry_sha256,
    }


def run_restore(
    aoi: str | Path | LoadedAOI,
    out_dir: Path | str,
    *,
    planted: int | None = None,
    year_a: int | None = None,
    year_b: int | None = None,
    control: str | Path | LoadedAOI | None = None,
    target: str | Path | LoadedAOI | None = None,
    validate: bool = False,
    canopy: bool = True,
    ring_inner_km: float = DEFAULT_RING_INNER_KM,
    ring_outer_km: float = DEFAULT_RING_OUTER_KM,
    control_pct: float = DEFAULT_CONTROL_PCT,
    folds: int = DEFAULT_FOLDS,
    fold_cell_m: float = DEFAULT_FOLD_CELL_M,
    placebos: int = DEFAULT_PLACEBOS,
    min_post_years: int = DEFAULT_MIN_POST_YEARS,
    tch_url_template: str | None = None,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    max_area_km2: float | None = io.MAX_AOI_KM2,
    max_zones: int = 4,
    snapshot_id: str | None = None,
    reader: Reader | None = None,
    block: tuple[int, int] = io.DEFAULT_BLOCK,
    strict_validate: bool = False,
) -> RestoreResult:
    """Is the site diverging from a matched control the way a working restoration should?

    Writes to ``out_dir``: ``restore_<a>_<b>.tif`` (float32 COG over the site plus ring or control,
    NaN nodata, mosaicked across zones: ``role``, ``baseline_similarity``, ``similarity_to_control_<t>``
    per year and, with a target, ``closer_to_target_<b>``), ``restore_summary.parquet`` and ``.csv``
    (one row per year, the verdict repeated so the csv stands alone), ``restore_placebos.parquet``
    (every placebo's divergence change per year and its verdict), ``restore_preview.png``,
    ``restore_validation.json`` when validating (the null test and the canopy check), and
    ``restore_report.json`` last. A rerun with the same arguments skips unless ``force``; one whose
    canopy check failed is redone once the canopy tiles can be reached.

    Args:
        aoi: The site: a polygon file (any CRS), a demo name, or a loaded AOI.
        out_dir: Output directory.
        planted: The planting year; a demo site's properties may give it.
        year_a: The baseline year, at most ``planted - 1`` (the default).
        year_b: The last year, at least ``planted``; the store's last year by default.
        control: A control polygon file used as given (no filter), ``"demo"`` for the demo's, or None
            for the ring.
        target: A mature reference stand (polygon file or ``"demo"``), or None.
        validate: Write the validation file: the null test and, with ``canopy``, the canopy check.
        canopy: Run the GLAD canopy height check when validating.
        ring_inner_km: Inner edge of the control ring, from the site's edge.
        ring_outer_km: Outer edge of the control ring, at most 10 km.
        control_pct: Percentile of the held-out site similarities that sets the control filter.
        folds: Spatial folds of the site's pixels for the held-out similarities.
        fold_cell_m: Side of the cells that block the site's (and the target's) pixels.
        placebos: Placebo pseudo-sites drawn from the control pool, 20 or more.
        min_post_years: Post-planting years the verdict needs.
        tch_url_template: GLAD canopy height tile URL or local path template;
            ``validate.GLAD_TCH_URL_TEMPLATE`` when None.
        seed: Seed of the fold assignment, the placebos and the target halves.
        force: Run even when the outputs are current.
        max_area_km2: Size limit of the planned extent and of the target; None lifts it.
        max_zones: Most zone groups the site, ring or control, and target may touch together.
        snapshot_id: Icechunk snapshot to pin.
        reader: Reader to use instead of opening the live store.
        block: Block shape of the streaming reads.
        strict_validate: With ``validate`` and ``canopy``, stop before any chunk read when the canopy
            layer cannot be loaded.

    Returns:
        The report, output paths and whether the run was skipped.

    Raises:
        AppInputError: A bad argument, polygon, overlap or output directory, before any chunk read.
        AOIError: The extent or target is too large, touches too many zones, or covers no pixel.
        YearsNotPublishedError: A trajectory year is missing for a zone a role touches (names it).
        StrictValidationError: ``strict_validate`` is set and the canopy layer cannot be loaded.
        RestoreError: Too few valid site, kept control or valid target pixels (before stage 2), or
            too few left after the panel and regime rule.
    """
    _check_args(
        ring_inner_km=ring_inner_km,
        ring_outer_km=ring_outer_km,
        control_pct=control_pct,
        folds=folds,
        fold_cell_m=fold_cell_m,
        placebos=placebos,
        min_post_years=min_post_years,
    )
    site = load_polygon(aoi, "site")
    plant = _planted_from(site, planted)
    year_a, year_b, years = trajectory_years(plant, year_a, year_b)
    ctrl = None if control is None else load_polygon(control, "control", demo_name=site.demo)
    tgt = None if target is None else load_polygon(target, "target", demo_name=site.demo)
    area = geodesic_area_ha(site.geometry)
    if area < MIN_SITE_HA:
        raise AppInputError(
            f"The site covers {area:.3f} ha; the app needs at least {MIN_SITE_HA:g} ha ({MIN_SITE_PIXELS} or more "
            "10 m pixels). Pass the whole planted area."
        )
    zone = main_zone(site.geometry)
    for name, other in (("control", ctrl), ("target", tgt)):
        if other is not None and overlap_m2(site.geometry, other.geometry, zone) > 1.0:
            raise AppInputError(f"The site and the {name} overlap; they must be separate areas.")
    if ctrl is not None and tgt is not None and overlap_m2(ctrl.geometry, tgt.geometry, zone) > 1.0:
        raise AppInputError("The control and the target overlap; they must be separate areas.")
    if ctrl is None:
        ring = ring_geometry(site.geometry, ring_inner_km, ring_outer_km, zone=zone)
        candidates = ring if tgt is None else io._polygonal(ring.difference(tgt.geometry))
        extent = shapely.union_all([site.geometry, ring])
    else:
        ring = None
        candidates = ctrl.geometry
        extent = shapely.union_all([site.geometry, ctrl.geometry])
    if candidates.is_empty:
        raise AppInputError("The control ring is empty once the target is removed; move the target or widen the ring.")
    everything = extent if tgt is None else shapely.union_all([extent, tgt.geometry])
    touched = io.zones_for_geometry(everything)
    if len(touched) > max_zones:
        raise AppInputError(
            f"The site, {'ring' if ctrl is None else 'control'} and target touch {len(touched)} zone groups "
            f"({', '.join(touched)}); the limit is {max_zones}. Move the target closer, or narrow the ring."
        )
    template = _tch_template(tch_url_template)
    opts = _Options(
        site=site,
        control=ctrl,
        target=tgt,
        planted=plant,
        years=years,
        ring_inner_km=float(ring_inner_km),
        ring_outer_km=float(ring_outer_km),
        control_pct=float(control_pct),
        folds=int(folds),
        fold_cell_m=float(fold_cell_m),
        placebos=int(placebos),
        min_post_years=int(min_post_years),
        canopy=bool(canopy),
        tch_template=template,
        strict_validate=bool(strict_validate),
        max_zones=int(max_zones),
        max_area_km2=max_area_km2,
        zone=zone,
        ring=ring,
        candidates=candidates,
    )
    out = Path(out_dir)
    paths = export.output_paths(out, APP, year_a, year_b)
    extras = {"placebos": out / f"{APP}_placebos.parquet"}
    params = {
        "planted": plant,
        "year_a": year_a,
        "year_b": year_b,
        "site": _sha_record(site),
        "control": _sha_record(ctrl),
        "target": _sha_record(tgt),
        "ring_inner_km": None if ctrl is not None else float(ring_inner_km),
        "ring_outer_km": None if ctrl is not None else float(ring_outer_km),
        "control_pct": float(control_pct),
        "folds": int(folds),
        "fold_cell_m": float(fold_cell_m),
        "placebos": int(placebos),
        "min_post_years": int(min_post_years),
        "canopy": bool(canopy),
        "tch_url_template": template,
        "block": list(block),
        "method": {
            "min_site_pixels": MIN_SITE_PIXELS,
            "min_control_pixels": MIN_CONTROL_PIXELS,
            "band_percentiles": list(BAND_PERCENTILES),
            "null_sectors": NULL_SECTORS,
            "canopy_onset_gap": CANOPY_ONSET_GAP,
        },
    }
    if validate and canopy and not force and _canopy_failed_before(paths.report):
        force = _canopy_reachable_now(extent, year_a, template)
    target_px = 0 if tgt is None else int(geodesic_area_ha(tgt.geometry) / HA_PER_PIXEL)
    state = _State()
    loaded = dataclasses.replace(site, geometry=extent)
    try:
        ctx = prepare_run(
            APP,
            loaded,
            years,
            year_a,
            year_b,
            out,
            params,
            validate=validate,
            force=force,
            max_area_km2=max_area_km2,
            max_zones=max_zones,
            snapshot_id=snapshot_id,
            reader=reader,
            seed=seed,
            block=block,
            extra_outputs=extras,
            code_modules=(sys.modules[__name__], wetland),
            memory_estimate=lambda plan: memory_estimate(
                plan, block=block, n_years=len(years), target_pixels=target_px
            ),
            preflight=lambda c: _preflight(c, state, opts),
        )
    except io.YearsNotPublishedError as exc:
        _note_roles(exc, opts, years)
        raise
    except io.AOIError as exc:
        exc.add_note(
            "The planned extent is the site plus its "
            + ("control ring" if ctrl is None else "control")
            + (f" ({ring_inner_km:g} to {ring_outer_km:g} km)" if ctrl is None else "")
            + "; a narrower ring (--ring-outer-km) makes it smaller."
        )
        raise
    if ctx is None:
        report = json.loads(paths.report.read_text(encoding="utf-8"))
        return RestoreResult(
            report=report,
            paths=paths,
            extra_paths=extras,
            skipped=True,
            validation_written="validation" in (report.get("outputs") or {}),
        )
    return _run(ctx, state, opts)


def _tch_template(template: str | None) -> str:
    """The canopy tile template: the given one, else :data:`validate.GLAD_TCH_URL_TEMPLATE` read now."""
    return validate.GLAD_TCH_URL_TEMPLATE if template is None else str(template)


def _roles_touching(zone: str, opts: _Options) -> list[str]:
    roles = []
    for name, geom in (
        ("site", opts.site.geometry),
        ("ring" if opts.control is None else "control", opts.ring if opts.ring is not None else opts.candidates),
        ("target", None if opts.target is None else opts.target.geometry),
    ):
        if geom is not None and zone in io.zones_for_geometry(geom):
            roles.append(name)
    return roles


def _note_roles(exc: io.YearsNotPublishedError, opts: _Options, years: Sequence[int]) -> None:
    for zone, missing in sorted(exc.missing.items()):
        roles = _roles_touching(zone, opts) or ["extent"]
        named = roles[0] if len(roles) == 1 else f"{', '.join(roles[:-1])} and {roles[-1]}"
        exc.add_note(
            f"Zone {zone} is touched by the {named} and does not publish "
            f"{', '.join(str(y) for y in missing)}; restore reads every year from {years[0]} to {years[-1]} there. "
            "Choose other years (--year-a, --year-b) or move the polygons."
        )


def _canopy_failed_before(report_path: Path) -> bool:
    """Whether the report on disk records a canopy check that could not be loaded."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    canopy = ((report.get("summary") or {}).get("canopy")) if isinstance(report, dict) else None
    return isinstance(canopy, dict) and bool(canopy.get("error"))


def _canopy_reachable_now(extent: BaseGeometry, year: int, template: str) -> bool:
    """Whether the canopy tiles of an earlier failed check can be reached now (one HEAD per tile)."""
    try:
        found = validate.glad_tch_reachable(extent.bounds, year, url_template=template)
    except (validate.ReferenceDataError, ValueError) as exc:
        logger.info("%s: the canopy layer still cannot be reached (%s); outputs kept.", APP, exc)
        return False
    if found:
        logger.info("%s: the canopy check failed in the run on disk and the layer can be reached now; rerunning.", APP)
    return found > 0


# --------------------------------------------------------------------------------------------
# Preflight: role masks, the target plan, its years and the canopy layer, before any chunk read
# --------------------------------------------------------------------------------------------


def _preflight(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Role masks on the extent, the target plan and its years, the pixel counts, and the canopy layer."""
    plan = ctx.plan
    site_px = control_px = 0
    for zp in plan.zones:
        site = validate.reference_from_geometries(zp, [opts.site.geometry])
        target = (
            validate.reference_from_geometries(zp, [opts.target.geometry]) & ~site
            if opts.target is not None and opts.control is None
            else np.zeros(zp.shape, dtype=bool)
        )
        control = zp.mask & ~site & ~target
        state.roles[zp.zone] = _ZoneRoles(plan=zp, site=site, control=control, target=target)
        site_px += int(site.sum())
        control_px += int(control.sum())
    if site_px < MIN_SITE_PIXELS:
        raise AppInputError(
            f"The site covers {site_px} pixel centres on the store grid; the app needs at least {MIN_SITE_PIXELS}. "
            "Pass the whole planted area."
        )
    if control_px == 0:
        raise AppInputError("The control covers no pixel centre on the store grid; pass a larger control or ring.")
    zones = list(plan.zone_names)
    if opts.target is not None:
        try:
            tplan = io.plan_aoi(
                ctx.reader, opts.target.geometry, max_zones=opts.max_zones, max_area_km2=opts.max_area_km2
            )
        except io.AOIError as exc:
            raise AppInputError(f"The target cannot be planned: {exc}") from exc
        io.reader_check_years(ctx.reader, tplan.zone_names, opts.years)
        if tplan.n_pixels < MIN_SITE_PIXELS:
            raise AppInputError(
                f"The target covers {tplan.n_pixels} pixel centres; the app needs at least {MIN_SITE_PIXELS}. Pass a "
                "larger mature stand."
            )
        all_zones = sorted({*zones, *tplan.zone_names})
        if len(all_zones) > opts.max_zones:
            raise AppInputError(
                f"The site, control and target touch {len(all_zones)} zone groups ({', '.join(all_zones)}); the "
                f"limit is {opts.max_zones}."
            )
        state.target_plan = tplan
        zones = sorted({*zones, *tplan.zone_names})
        ctx.estimated_bytes += tplan.estimated_bytes(ctx.reader, len(opts.years), block=ctx.block)
    state.zones = sorted(set(zones))
    logger.info(
        "%s: site %d pixels, %s %d pixels%s; planted %d, years %d to %d.",
        APP,
        site_px,
        "ring" if opts.control is None else "control",
        control_px,
        "" if state.target_plan is None else f", target {state.target_plan.n_pixels} pixels",
        opts.planted,
        opts.years[0],
        opts.years[-1],
    )
    if ctx.validate and opts.canopy:
        _load_canopy(ctx, state, opts)


def _load_canopy(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Load GLAD canopy height for every canopy year on the extent and target zone plans.

    Raises:
        StrictValidationError: ``strict_validate`` is set and the layer cannot be loaded.
    """
    years = [y for y in opts.years if y in validate.GLAD_TCH_YEARS]
    state.canopy_years = years
    plans = {"extent": ctx.plan.zones, "target": [] if state.target_plan is None else state.target_plan.zones}
    heights: dict[str, dict[str, dict[int, np.ndarray]]] = {kind: {} for kind in plans}
    try:
        for kind, zps in plans.items():
            for zp in zps:
                per_year: dict[int, np.ndarray] = {}
                for y in years:
                    h = validate.glad_tch(zp, y, url_template=opts.tch_template)
                    per_year[y] = np.where(
                        np.isfinite(h), np.clip(h, 0, validate.GLAD_TCH_NODATA - 1), validate.GLAD_TCH_NODATA
                    ).astype(np.uint8)
                heights[kind][zp.zone] = per_year
    except (validate.ReferenceDataError, rasterio.errors.RasterioError) as exc:
        if opts.strict_validate:
            raise StrictValidationError(
                f"--strict-validate is set and GLAD tree canopy height could not be loaded: {exc}"
            ) from exc
        state.canopy_error = str(exc)
        logger.warning("The canopy check will be skipped, GLAD tree canopy height could not be loaded: %s", exc)
        return
    state.canopy = heights


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


def _stage1(ctx: RunContext, state: _State) -> tuple[_Pixels, _Pixels, _Pixels, dict[str, int]]:
    """Read year_a for every block of the extent and the target: the valid site, control-candidate and target pixels."""
    year_a = ctx.year_a
    zindex = {z: i for i, z in enumerate(state.zones)}
    site, ring, target = _Collector(), _Collector(), _Collector()
    counts = {"site_pixels": 0, "control_pixels": 0, "target_in_extent_pixels": 0}
    for zp in ctx.plan.zones:
        roles = state.roles[zp.zone]
        counts["site_pixels"] += int(roles.site.sum())
        counts["control_pixels"] += int(roles.control.sum())
        counts["target_in_extent_pixels"] += int(roles.target.sum())
        for blk in io.iter_plan_blocks(ctx.reader, zp, [year_a], layers=LAYERS, block=ctx.block):
            win = blk.windows[year_a]
            r = slice(blk.rows.start - zp.rows.start, blk.rows.stop - zp.rows.start)
            c = slice(blk.cols.start - zp.cols.start, blk.cols.stop - zp.cols.start)
            ok = _valid(win)
            site.add(zindex[zp.zone], win, ok & roles.site[r, c])
            ring.add(zindex[zp.zone], win, ok & roles.control[r, c])
    if state.target_plan is not None:
        counts["target_pixels"] = state.target_plan.n_pixels
        for zp in state.target_plan.zones:
            for blk in io.iter_plan_blocks(ctx.reader, zp, [year_a], layers=LAYERS, block=ctx.block):
                win = blk.windows[year_a]
                target.add(zindex[zp.zone], win, _valid(win) & blk.mask)
    # The target's zones join the provenance (and its pipeline commits) when the extent does not touch them.
    ctx.mark_read([year_a], zones=state.zones)
    return site.done(ROLE_SITE), ring.done(ROLE_CONTROL), target.done(ROLE_TARGET), counts


def _stage2(ctx: RunContext, state: _State, pool: _Pixels, years: Sequence[int]) -> int:
    """Read every year after year_a for the chunks holding a pool pixel, filling the pool. Returns windows read."""
    other = [y for y in years if y != ctx.year_a]
    if not other or pool.n == 0:
        return 0
    shapes = {i: ctx.reader.chunk_shape(z) for i, z in enumerate(state.zones)}
    ch0, cw0 = next(iter(shapes.values()))
    max_run = max(1, (ctx.block[0] * ctx.block[1] * 2) // (ch0 * cw0 * len(other)))
    runs = chunk_runs(pool.zone, pool.row, pool.col, shapes, max_run)
    index = {y: years.index(y) for y in other}
    for zi, cr, c0, c1 in runs:
        zone = state.zones[zi]
        grid = ctx.reader.grid(zone)
        ch, cw = shapes[zi]
        rows = slice(cr * ch, min((cr + 1) * ch, grid.height))
        cols = slice(c0 * cw, min((c1 + 1) * cw, grid.width))
        sel = np.flatnonzero(
            (pool.zone == zi)
            & (pool.row >= rows.start)
            & (pool.row < rows.stop)
            & (pool.col >= cols.start)
            & (pool.col < cols.stop)
        )
        wins = ctx.reader.read_windows(zone, other, rows, cols, layers=LAYERS)
        lr, lc = pool.row[sel] - rows.start, pool.col[sel] - cols.start
        for y, win in wins.items():
            assert win.embeddings is not None
            t = index[y]
            pool.emb[sel, t] = win.embeddings[lr, lc]
            pool.scale[sel, t] = win.scales[lr, lc]
            pool.radar[sel, t] = win.radar_present[lr, lc]
    ctx.mark_read(other, zones=state.zones)
    return len(runs)


def majority_pattern(radar: np.ndarray) -> np.ndarray:
    """The most common row of a (n, Y) bool radar pattern array; ties go to the pattern with more radar years.

    Raises:
        ValueError: No rows.
    """
    r = np.asarray(radar, dtype=bool)
    if r.shape[0] == 0:
        raise ValueError("No pixels to take a majority radar pattern from.")
    patterns, counts = np.unique(r, axis=0, return_counts=True)
    order = sorted(range(len(counts)), key=lambda i: (-int(counts[i]), -int(patterns[i].sum()), i))
    return patterns[order[0]].astype(bool)


def _xy_in_zone(state: _State, reader: Reader, pix: _Pixels, zone: str) -> np.ndarray:
    """(n, 2) pixel centres of ``pix`` in one zone's UTM metres."""
    out = np.zeros((pix.n, 2), dtype=np.float64)
    target_epsg = io.zone_epsg(zone)
    for zi in np.unique(pix.zone):
        name = state.zones[int(zi)]
        grid = reader.grid(name)
        sel = pix.zone == zi
        x = grid.left + (pix.col[sel] + 0.5) * grid.pixel
        y = grid.top - (pix.row[sel] + 0.5) * grid.pixel
        if name != zone:
            lon, lat = grid.to_lonlat().transform(x, y)
            x, y = Transformer.from_crs(4326, target_epsg, always_xy=True).transform(lon, lat)
        out[sel, 0] = np.asarray(x)
        out[sel, 1] = np.asarray(y)
    return out


def _site_cells(state: _State, reader: Reader, pix: _Pixels, cell_m: float) -> np.ndarray:
    """Spatial block of every pixel: its ``cell_m`` cell in its own zone's metres (wetland.spatial_blocks)."""
    names = np.array([state.zones[int(z)] for z in pix.zone], dtype=object)
    cell_x = np.zeros(pix.n, dtype=np.int64)
    cell_y = np.zeros(pix.n, dtype=np.int64)
    for zi in np.unique(pix.zone):
        grid = reader.grid(state.zones[int(zi)])
        sel = pix.zone == zi
        cell_x[sel] = np.floor((grid.left + (pix.col[sel] + 0.5) * grid.pixel) / cell_m).astype(np.int64)
        cell_y[sel] = np.floor((grid.top - (pix.row[sel] + 0.5) * grid.pixel) / cell_m).astype(np.int64)
    off = np.full(pix.n, -1, dtype=np.int64)
    return spatial_blocks(np.arange(pix.n, dtype=np.int64), names, cell_x, cell_y, off, off)


def _sums(pix: _Pixels) -> np.ndarray:
    """(Y, D) pooled dequantized sums per year."""
    return np.stack([pooled_sum(pix.emb[:, t], pix.scale[:, t]) for t in range(pix.emb.shape[1])])


def _divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Y,) ``1 - cos`` of two (Y, D) sum series."""
    return np.array([1.0 - cosine(x, y) for x, y in zip(a, b, strict=True)], dtype=np.float64)


def _ratio(num: float, den: float) -> float | None:
    if den == 0 or not (math.isfinite(num) and math.isfinite(den)):
        return None
    return float(num / den)


def _run(ctx: RunContext, state: _State, opts: _Options) -> RestoreResult:
    years = list(opts.years)
    n_years = len(years)
    year_a, year_b, planted = ctx.year_a, ctx.year_b, opts.planted
    supplied = opts.control is not None
    control_word = "control" if supplied else "ring"
    if year_a == FIRST_YEAR:
        ctx.warn(
            f"The baseline year is {FIRST_YEAR}, the store's first: it has few Sentinel-2 observations and sparse "
            "valid pixels in places (survey S16), so the control match and the divergence change rest on a weak year."
        )

    # Stage 1: the baseline.
    t = time.perf_counter()
    site1, ring1, target1, counts = _stage1(ctx, state)
    t = ctx.lap("stage1", t)
    n_site = site1.n
    if n_site < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {n_site} of the site's {counts['site_pixels']} pixels have {year_a} data; the app needs at least "
            f"{MIN_SITE_PIXELS} valid site pixels in the baseline year. Choose another --year-a, or check the site "
            "lies on land with data."
        )
    sum_s = pooled_sum(site1.emb[:, 0], site1.scale[:, 0])
    site_sim = pixel_cosine(site1.emb[:, 0], sum_s)
    groups = _site_cells(state, ctx.reader, site1, opts.fold_cell_m)
    fold, n_folds = assign_folds(groups, opts.folds, ctx.seed)
    held = heldout_similarity(site1.emb[:, 0], site1.scale[:, 0], fold) if n_folds >= 2 else np.full(n_site, np.nan)
    choice = control_threshold(site_sim, held, opts.control_pct, n_folds)
    threshold = 1.0 - choice.threshold
    if choice.method != "heldout":
        reason = (
            f"the site's pixels fall in {np.unique(groups).size} cell(s) of {opts.fold_cell_m:g} m"
            if opts.folds >= 2
            else "--folds is below 2"
        )
        choice = dataclasses.replace(choice, fallback_reason=reason)
        ctx.warn(
            f"The control threshold is the in-sample {opts.control_pct:g}th percentile of the site's baseline "
            f"similarities, not a held-out one: {reason}. In-sample thresholds are too tight (survey S17), so the "
            "filter may reject comparable land; pass a smaller --fold-cell-m for a small site."
        )
    ring_sim = pixel_cosine(ring1.emb[:, 0], sum_s)
    passes = (1.0 - ring_sim) <= choice.threshold
    kept = np.ones(ring1.n, dtype=bool) if supplied else passes
    n_kept = int(kept.sum())
    below_share = float(np.mean(~passes)) if ring1.n else math.nan
    if supplied and ring1.n and below_share > 0.5:
        ctx.warn(
            f"The supplied control is poorly matched: {below_share:.0%} of its valid pixels are less similar to the "
            f"site's {year_a} mean than the held-out threshold ({threshold:.4f}). Divergence from it may be a "
            "difference that predates the planting; consider the ring (omit --control)."
        )
    if n_kept < MIN_CONTROL_PIXELS:
        if supplied:
            hint = "pass a larger control polygon"
        else:
            hint = "widen the ring (--ring-outer-km), lower --control-pct, or pass a --control polygon"
        raise RestoreError(
            f"Only {n_kept} {control_word} pixels are kept ({ring1.n} have {year_a} data, of "
            f"{counts['control_pixels']} in the {control_word}); the app needs at least {MIN_CONTROL_PIXELS} control "
            f"pixels: {hint}."
        )
    if state.target_plan is not None and target1.n < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {target1.n} of the target's {state.target_plan.n_pixels} pixels have {year_a} data; the app needs "
            f"at least {MIN_SITE_PIXELS}. Pass a larger mature stand, or one on land with data."
        )
    t = ctx.lap("control_filter", t)

    # Stage 2: the other years, only for chunks with a kept pixel.
    control1 = ring1.take(kept)
    pool = _expand_years(_concat([site1, control1, target1]), n_years)
    windows_read = _stage2(ctx, state, pool, years)
    t = ctx.lap("stage2", t)

    valid = _valid_years(pool)
    is_site = pool.role == ROLE_SITE
    radar_free = []
    for k in range(n_years):
        v = valid[is_site, k]
        radar_free.append(float(np.mean(~pool.radar[is_site, k][v])) if v.any() else math.nan)

    # Panel and radar regime (survey B5).
    panel = valid.all(axis=1)
    if not (panel & is_site).any():
        raise RestoreError(
            f"No site pixel has data in every year from {year_a} to {year_b}; choose a shorter span (--year-a, "
            "--year-b)."
        )
    pattern = majority_pattern(pool.radar[panel & is_site])
    regime_ok = np.all(pool.radar == pattern[None, :], axis=1)
    use = panel & regime_ok
    dropped: dict[str, dict[str, int]] = {}
    for code, name in ((ROLE_SITE, "site"), (ROLE_CONTROL, "control"), (ROLE_TARGET, "target")):
        r = pool.role == code
        dropped[name] = {
            "read": int(r.sum()),
            "panel": int((r & ~panel).sum()),
            "regime": int((r & panel & ~regime_ok).sum()),
            "used": int((r & use).sum()),
        }
    free_years = [y for y, p in zip(years, pattern, strict=True) if not p]
    if free_years:
        ctx.warn(
            f"The site's majority radar pattern has no Sentinel-1 in {', '.join(str(y) for y in free_years)}: every "
            "kept pixel is optical-only those years, so the within-year comparisons stay fair and the placebo band "
            "sees the same switch, but the trajectory may step at those years (survey B5)."
        )
    for name in ("site", "control", "target"):
        d = dropped[name]
        if d["panel"] or d["regime"]:
            ctx.warn(
                f"{d['panel'] + d['regime']:,} of {d['read']:,} {name} pixels were dropped: {d['panel']:,} lack data "
                f"in some year from {year_a} to {year_b} (the panel rule) and {d['regime']:,} have another radar "
                "pattern than the site's majority (survey B5)."
            )
    site = pool.take(use & is_site)
    ctrl = pool.take(use & (pool.role == ROLE_CONTROL))
    tgt = pool.take(use & (pool.role == ROLE_TARGET))
    if site.n < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {site.n} site pixels have data in every year from {year_a} to {year_b} with the majority radar "
            f"pattern; the app needs at least {MIN_SITE_PIXELS}. Choose a shorter span (--year-a, --year-b)."
        )
    if ctrl.n < MIN_CONTROL_PIXELS:
        raise RestoreError(
            f"Only {ctrl.n} control pixels have data in every year from {year_a} to {year_b} with the site's radar "
            f"pattern; the app needs at least {MIN_CONTROL_PIXELS}. Widen the ring or pass a larger control."
        )
    if state.target_plan is not None and tgt.n < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {tgt.n} target pixels have data in every year from {year_a} to {year_b} with the site's radar "
            f"pattern; the app needs at least {MIN_SITE_PIXELS}. Pass a larger mature stand."
        )
    t = ctx.lap("panel", t)

    # Trajectories.
    s_site = _sums(site)
    s_ctrl = _sums(ctrl)
    div = _divergence(s_site, s_ctrl)
    change = div - div[0]
    n_p = min(site.n, ctrl.n // 3)
    if n_p < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Placebos would hold {n_p} pixels (a third of the {ctrl.n} control pixels); the app needs at least "
            f"{MIN_SITE_PIXELS}. Widen the ring or pass a larger control."
        )
    if n_p < site.n:
        ctx.warn(
            f"Each placebo holds {n_p} pixels, fewer than the site's {site.n} (a third of the control pool): smaller "
            "placebos vary more, so the band is wider and the test conservative. Widen the ring for a tighter band."
        )
    main = opts.zone
    xy_ctrl = _xy_in_zone(state, ctx.reader, ctrl, main)
    members, centres = draw_placebos(xy_ctrl, opts.placebos, n_p, ctx.seed)
    p_div = placebo_divergence(ctrl.emb, ctrl.scale, members)
    p_change = p_div - p_div[:, :1]
    band = placebo_band(p_change)
    p05, p50, p95 = band
    verdict = call_verdict(years, change, p95, planted=planted, min_post_years=opts.min_post_years)
    p_texts, p_diverging = placebo_verdicts(years, p_change, p95, planted=planted, min_post_years=opts.min_post_years)
    fp_rate = false_positive_rate(p_diverging, verdict.verdict == VERDICT_INSUFFICIENT)
    post_mask = np.array([y >= planted for y in years], dtype=bool)
    with np.errstate(invalid="ignore"):
        significant = (change > p95) & np.isfinite(change)
    trend = theil_sen_trend([y for y in years if y >= planted], change[post_mask])
    baseline_rank = float(np.mean(p_div[:, 0] < div[0]))
    t = ctx.lap("placebos", t)

    # The null test: two halves of the control pool by alternating sectors around the site centroid.
    centre = to_zone_metres(opts.site.geometry, main).centroid
    halves = null_halves(xy_ctrl, (float(centre.x), float(centre.y)))
    n_half = (int((halves == 0).sum()), int((halves == 1).sum()))
    null: dict[str, Any]
    null_change = np.full(n_years, np.nan)
    if min(n_half) >= MIN_SITE_PIXELS:
        s0 = _sums(ctrl.take(halves == 0))
        s1 = _sums(ctrl.take(halves == 1))
        null_div = _divergence(s0, s1)
        null_change = null_div - null_div[0]
        nv = call_verdict(years, null_change, p95, planted=planted, min_post_years=opts.min_post_years)
        post_null = null_change[post_mask]
        post_site = change[post_mask]
        null_max = float(np.nanmax(np.abs(post_null))) if post_null.size else math.nan
        site_max = float(np.nanmax(np.abs(post_site))) if post_site.size else math.nan
        with np.errstate(invalid="ignore"):
            null_sig = [y for y, v, h in zip(years, null_change, p95, strict=True) if y >= planted and v > h]
        null = {
            "method": (
                f"The control pool split into two halves by {NULL_SECTORS} alternating angular sectors around the site "
                "centroid (main zone metres), compared exactly as the site is, against the same placebo band."
            ),
            "sectors": NULL_SECTORS,
            "half_pixels": list(n_half),
            "divergence": dict(zip([str(y) for y in years], null_div.tolist(), strict=True)),
            "divergence_change": dict(zip([str(y) for y in years], null_change.tolist(), strict=True)),
            "null_verdict": nv.verdict,
            "null_verdict_detail": nv.detail,
            "significant_years": null_sig,
            "null_max_abs_change": null_max,
            "site_max_abs_change": site_max,
            "site_max_change": float(np.nanmax(post_site)) if post_site.size else math.nan,
            "null_to_site_ratio": _ratio(null_max, site_max),
            "null_near_zero": not null_sig,
            "limitation": LIMITATION,
        }
    else:
        null = {
            "error": (
                f"a half of the control pool has fewer than {MIN_SITE_PIXELS} pixels ({n_half[0]} and {n_half[1]})"
            ),
            "half_pixels": list(n_half),
            "limitation": LIMITATION,
        }
        ctx.warn(f"The null test could not run: {null['error']}.")

    # The target trajectory.
    target_rec: dict[str, Any] | None = None
    s_t, c_t, tau, progress = (np.full(n_years, np.nan) for _ in range(4))
    sum_t: np.ndarray | None = None
    if tgt.n:
        sum_t = _sums(tgt)
        s_t = np.array([cosine(a, b) for a, b in zip(s_site, sum_t, strict=True)])
        c_t = np.array([cosine(a, b) for a, b in zip(s_ctrl, sum_t, strict=True)])
        tgroups = _site_cells(state, ctx.reader, tgt, opts.fold_cell_m)
        tfold, tk = assign_folds(tgroups, 2, ctx.seed)
        smaller = min(int((tfold == 0).sum()), int((tfold == 1).sum())) if tk >= 2 else 0
        split_ok = tk >= 2 and smaller >= MIN_TARGET_HALF_PIXELS
        if split_ok:
            h0, h1 = _sums(tgt.take(tfold == 0)), _sums(tgt.take(tfold == 1))
            tau = np.array([cosine(a, b) for a, b in zip(h0, h1, strict=True)])
            progress = target_progress(s_t, c_t, tau)
        else:
            why = (
                f"its pixels fall in one {opts.fold_cell_m:g} m cell"
                if tk < 2
                else f"one half would hold {smaller} pixels, under {MIN_TARGET_HALF_PIXELS}"
            )
            ctx.warn(
                f"The target cannot be split into two spatial halves for the 100 percent level ({why}): only the raw "
                "similarity to the target is reported, not the progress. Pass a larger target or a smaller "
                "--fold-cell-m."
            )
        target_rec = {
            "pixels": tgt.n,
            "cells": int(np.unique(tgroups).size),
            "halves": None if not split_ok else [int((tfold == 0).sum()), int((tfold == 1).sum())],
            "progress_note": PROGRESS_NOTE,
            "similarity": dict(zip([str(y) for y in years], s_t.tolist(), strict=True)),
            "control_similarity": dict(zip([str(y) for y in years], c_t.tolist(), strict=True)),
            "halves_similarity": dict(zip([str(y) for y in years], tau.tolist(), strict=True)),
            "progress": dict(zip([str(y) for y in years], progress.tolist(), strict=True)),
        }

    # Per pixel.
    site_to_c = np.stack([pixel_cosine(site.emb[:, k], s_ctrl[k]) for k in range(n_years)], axis=1)
    ctrl_to_c = np.stack([pixel_cosine(ctrl.emb[:, k], s_ctrl[k]) for k in range(n_years)], axis=1)
    departed = np.array([departed_fraction(site_to_c[:, k], ctrl_to_c[:, k]) for k in range(n_years)])
    closer = np.full(n_years, np.nan)
    ctrl_closer = np.full(n_years, np.nan)
    site_closer_b: np.ndarray | None = None
    if sum_t is not None:
        site_to_t = np.stack([pixel_cosine(site.emb[:, k], sum_t[k]) for k in range(n_years)], axis=1)
        ctrl_to_t = np.stack([pixel_cosine(ctrl.emb[:, k], sum_t[k]) for k in range(n_years)], axis=1)
        closer = np.array([closer_fraction(site_to_t[:, k], site_to_c[:, k]) for k in range(n_years)])
        ctrl_closer = np.array([closer_fraction(ctrl_to_t[:, k], ctrl_to_c[:, k]) for k in range(n_years)])
        site_closer_b = site_to_t[:, -1] > site_to_c[:, -1]
        del site_to_t, ctrl_to_t
    t = ctx.lap("trajectories", t)

    # The canopy check.
    canopy: dict[str, Any] | None = None
    canopy_heights: dict[str, dict[int, np.ndarray]] | None = None
    if ctx.validate and opts.canopy:
        if state.canopy is None:
            canopy = {"error": state.canopy_error, "years": state.canopy_years}
            ctx.warn(f"{CANOPY_SKIPPED}: {state.canopy_error}")
        else:
            canopy_heights = {
                "site": _heights(state, "extent", site),
                "control": _heights(state, "extent", ctrl),
            }
            if tgt.n:
                canopy_heights["target"] = _heights(state, "target", tgt)
            canopy = {
                "dataset": "GLAD annual tree canopy height (Potapov et al.)",
                "url_template": opts.tch_template,
                **canopy_check(years, change, canopy_heights),
                "caveats": list(CANOPY_CAVEATS),
            }
        t = ctx.lap("canopy", t)

    # Outputs.
    rows = []
    for k, y in enumerate(years):
        row: dict[str, Any] = {
            "year": y,
            "post_planting": bool(y >= planted),
            "divergence": div[k],
            "divergence_change": change[k],
            "placebo_p05": p05[k],
            "placebo_p50": p50[k],
            "placebo_p95": p95[k],
            "significant": bool(significant[k]),
            "null_divergence_change": null_change[k],
            "departed_fraction": departed[k],
            "closer_to_target_fraction": closer[k],
            "control_closer_to_target_fraction": ctrl_closer[k],
            "target_similarity": s_t[k],
            "control_target_similarity": c_t[k],
            "target_halves_similarity": tau[k],
            "target_progress": progress[k],
            "site_radar_free_fraction": radar_free[k],
            "n_site": site.n,
            "n_control": ctrl.n,
            "n_target": tgt.n,
            "verdict": verdict.verdict,
        }
        if canopy_heights is not None and canopy is not None:
            per = canopy["per_year"].get(str(y), {})
            row["site_tch_mean"] = per.get("site_mean_m", math.nan)
            row["control_tch_mean"] = per.get("control_mean_m", math.nan)
            row["site_tall_share"] = per.get("site_tall_share", math.nan)
            row["control_tall_share"] = per.get("control_tall_share", math.nan)
            if tgt.n:
                row["target_tch_mean"] = per.get("target_mean_m", math.nan)
                row["target_tall_share"] = per.get("target_tall_share", math.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in table.columns:
        if col not in ("year", "post_planting", "significant", "n_site", "n_control", "n_target", "verdict"):
            table[col] = table[col].astype(np.float64)
    export.write_table(table, ctx.paths.summary_parquet.with_suffix(""))
    _write_placebos(ctx.extra_outputs["placebos"], years, p_div, p_change, p95, p_texts, xy_ctrl[centres], n_p)

    band_names, mosaic = _write_tif(
        ctx,
        state,
        years=years,
        site1=site1,
        ring1=ring1,
        ring_sim=ring_sim,
        kept=kept,
        site=site,
        ctrl=ctrl,
        target1=target1,
        sum_s=sum_s,
        site_sim=site_sim,
        site_to_c=site_to_c,
        ctrl_to_c=ctrl_to_c,
        site_closer_b=site_closer_b,
    )
    step = stride_for_preview(mosaic.shape[1:])
    _write_preview(
        ctx.paths.preview,
        role=mosaic[0, ::step, ::step],
        years=years,
        planted=planted,
        change=change,
        band=band,
        null_change=null_change,
        progress=progress,
        target_sim=s_t,
        departed=departed,
        closer=closer,
        ctrl_closer=ctrl_closer,
        canopy=canopy if canopy_heights is not None else None,
        site_sim=site_sim,
        ring_sim=ring_sim,
        threshold=threshold,
        verdict=verdict,
        supplied=supplied,
    )
    del mosaic
    t = ctx.lap("write", t)

    control_rec = {
        "source": "file" if supplied else "ring",
        "file": None if opts.control is None else opts.control.source,
        "ring_inner_km": None if supplied else opts.ring_inner_km,
        "ring_outer_km": None if supplied else opts.ring_outer_km,
        "candidate_pixels": counts["control_pixels"],
        "candidate_pixels_valid_year_a": ring1.n,
        "kept_pixels": n_kept,
        "rejected_pixels": 0 if supplied else int((~kept).sum()),
        "filtered": not supplied,
        "threshold_similarity": threshold,
        "threshold_source": choice.method,
        "threshold": choice.record(),
        "threshold_note": (
            "threshold_similarity is the control_pct percentile of the site's baseline cosine similarities to its own "
            "mean (held out by spatial fold unless the fallback ran); the threshold block gives the same cut as a "
            "distance (1 - similarity) at the 100 - control_pct percentile, as wetland.choose_threshold works."
        ),
        "control_pct": opts.control_pct,
        "folds_requested": opts.folds,
        "folds": n_folds,
        "fold_cell_m": opts.fold_cell_m,
        "site_cells": int(np.unique(groups).size),
        "heldout_pixels": int(np.isfinite(held).sum()),
        "share_below_threshold": below_share,
        "balance": {
            "baseline_divergence": div[0],
            "placebo_baseline_divergence_p50": float(np.percentile(p_div[:, 0], 50)),
            "placebo_baseline_divergence_p95": float(np.percentile(p_div[:, 0], 95)),
            "baseline_divergence_rank": baseline_rank,
            "note": (
                "baseline_divergence_rank is the share of placebos whose own baseline divergence from the rest of the "
                "pool is below the site's: near 1 means the site starts less like its control than placebo pieces of "
                "the control do."
            ),
        },
    }
    regime = {
        "majority_pattern": {str(y): ("radar" if p else "optical") for y, p in zip(years, pattern, strict=True)},
        "radar_free_years": free_years,
        "dropped": dropped,
        "site_radar_free_fraction": dict(zip([str(y) for y in years], radar_free, strict=True)),
    }
    placebo_rec = {
        "count": opts.placebos,
        "pixels_each": n_p,
        "band_percentiles": list(BAND_PERCENTILES),
        "false_positive_rate": fp_rate,
        "diverging_placebos": int(p_diverging.sum()),
        "note": PLACEBO_NOTE,
    }
    trajectory = table.drop(columns=["verdict"]).to_dict(orient="records")
    summary: dict[str, Any] = {
        "verdict": verdict.verdict,
        "verdict_detail": verdict.detail,
        "diverging_since": verdict.since,
        "significant_years": verdict.significant_years,
        "planted": planted,
        "year_a": year_a,
        "year_b": year_b,
        "years": years,
        "post_planting_years": verdict.post_years,
        "site": {
            "name": opts.site.name,
            "source": opts.site.source,
            "area_ha": geodesic_area_ha(opts.site.geometry),
            "pixels": counts["site_pixels"],
            "pixels_valid_year_a": n_site,
            "pixels_used": site.n,
            "main_zone": main,
        },
        "extent": {
            "what": "site plus " + ("supplied control" if supplied else "control ring"),
            "area_ha": ctx.plan.area_ha,
            "pixels": ctx.plan.n_pixels,
            "zones": ctx.plan.zone_names,
        },
        "zones": state.zones,
        "control": control_rec,
        "regime": regime,
        "placebos": placebo_rec,
        "placebo_false_positive_rate": fp_rate,
        "departed_fraction_baseline": departed[0],
        "departed_fraction_note": DEPARTED_NOTE,
        "trend": trend,
        "null_test": null,
        "target": target_rec,
        "trajectory": trajectory,
        "limitation": LIMITATION,
        "thresholds": {
            "control_similarity": threshold,
            "control_pct": opts.control_pct,
            "band_percentiles": list(BAND_PERCENTILES),
            "min_post_years": opts.min_post_years,
            "min_site_pixels": MIN_SITE_PIXELS,
            "min_control_pixels": MIN_CONTROL_PIXELS,
        },
        "seed": ctx.seed,
        "chunk_windows_stage2": windows_read,
        "band_names": band_names,
        "canopy": canopy,
        "validation": None,
    }
    validation_written = False
    if ctx.validate:
        _write_validation(ctx, null, canopy, fp_rate)
        validation_written = True
        summary["validation"] = {
            "null_verdict": null.get("null_verdict"),
            "null_near_zero": null.get("null_near_zero"),
            "canopy_onset_year": None if canopy is None else canopy.get("canopy_onset_year"),
            "canopy_error": None if canopy is None else canopy.get("error"),
        }
    ctx.lap("validation", t)
    report = finish_run(ctx, summary=summary, extra=ctx.extra_outputs, validation_written=validation_written)
    return RestoreResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=validation_written,
    )


def _heights(state: _State, kind: str, pix: _Pixels) -> dict[int, np.ndarray]:
    """Canopy height (metres, NaN not covered) of every pixel of ``pix`` per canopy year, from the loaded windows."""
    assert state.canopy is not None
    if kind == "extent":
        plans = {zone: roles.plan for zone, roles in state.roles.items()}
    else:
        plans = {z.zone: z for z in (state.target_plan.zones if state.target_plan is not None else [])}
    out: dict[int, np.ndarray] = {y: np.full(pix.n, np.nan, dtype=np.float32) for y in state.canopy_years}
    for zi in np.unique(pix.zone):
        zone = state.zones[int(zi)]
        if zone not in state.canopy[kind]:
            continue
        zp = plans[zone]
        sel = np.flatnonzero(pix.zone == zi)
        r = pix.row[sel] - zp.rows.start
        c = pix.col[sel] - zp.cols.start
        inside = (r >= 0) & (r < zp.shape[0]) & (c >= 0) & (c < zp.shape[1])
        for y in state.canopy_years:
            h = state.canopy[kind][zone][y][r[inside], c[inside]].astype(np.float32)
            h[h == validate.GLAD_TCH_NODATA] = np.nan
            out[y][sel[inside]] = h
    return out


def _scatter(state: _State, per_zone: dict[str, np.ndarray], pix: _Pixels, values: np.ndarray) -> None:
    """Write per-pixel values onto the extent zone windows (pixels outside a window are skipped)."""
    for zi in np.unique(pix.zone):
        zone = state.zones[int(zi)]
        if zone not in per_zone:
            continue
        zp = state.roles[zone].plan
        sel = np.flatnonzero(pix.zone == zi)
        r = pix.row[sel] - zp.rows.start
        c = pix.col[sel] - zp.cols.start
        inside = (r >= 0) & (r < zp.shape[0]) & (c >= 0) & (c < zp.shape[1])
        per_zone[zone][r[inside], c[inside]] = values[sel[inside]]


def _write_tif(
    ctx: RunContext,
    state: _State,
    *,
    years: Sequence[int],
    site1: _Pixels,
    ring1: _Pixels,
    ring_sim: np.ndarray,
    kept: np.ndarray,
    site: _Pixels,
    ctrl: _Pixels,
    target1: _Pixels,
    sum_s: np.ndarray,
    site_sim: np.ndarray,
    site_to_c: np.ndarray,
    ctrl_to_c: np.ndarray,
    site_closer_b: np.ndarray | None,
) -> tuple[list[str], np.ndarray]:
    """Write the COG; returns the band names and the mosaic (for the preview)."""
    plan = ctx.plan
    names = ["role", "baseline_similarity", *[f"similarity_to_control_{y}" for y in years]]
    if site_closer_b is not None:
        names.append(f"closer_to_target_{years[-1]}")
    dst = max(plan.zones, key=lambda zp: zp.n_pixels).zone
    mosaic = BandMosaic(plan, len(names), dst_zone=dst)
    nan = np.float32(np.nan)

    def blank() -> dict[str, np.ndarray]:
        return {z.zone: np.full(z.shape, nan, dtype=np.float32) for z in plan.zones}

    role = blank()
    for zp in plan.zones:
        roles = state.roles[zp.zone]
        role[zp.zone][roles.site | roles.control] = ROLE_DROPPED
        role[zp.zone][roles.target] = ROLE_TARGET
    _scatter(state, role, ring1.take(~kept), np.full(int((~kept).sum()), ROLE_REJECTED, dtype=np.float32))
    _scatter(state, role, site, np.full(site.n, ROLE_SITE, dtype=np.float32))
    _scatter(state, role, ctrl, np.full(ctrl.n, ROLE_CONTROL, dtype=np.float32))
    mosaic.put(0, role)
    base = blank()
    _scatter(state, base, site1, site_sim.astype(np.float32))
    _scatter(state, base, ring1, ring_sim.astype(np.float32))
    if target1.n:
        _scatter(state, base, target1, pixel_cosine(target1.emb[:, 0], sum_s).astype(np.float32))
    for zp in plan.zones:
        base[zp.zone][~zp.mask] = nan
    mosaic.put(1, base)
    for k in range(len(years)):
        band = blank()
        _scatter(state, band, site, site_to_c[:, k].astype(np.float32))
        _scatter(state, band, ctrl, ctrl_to_c[:, k].astype(np.float32))
        mosaic.put(2 + k, band)
    if site_closer_b is not None:
        band = blank()
        _scatter(state, band, site, site_closer_b.astype(np.float32))
        mosaic.put(len(names) - 1, band)
    assert mosaic.data is not None
    with rasterio.Env(GDAL_CACHEMAX=WRITE_CACHE_BYTES):
        export.write_geotiff(ctx.paths.tif, mosaic.data, mosaic.transform, mosaic.crs, band_names=names)
    return names, mosaic.data


def _write_placebos(
    path: Path,
    years: Sequence[int],
    div: np.ndarray,
    change: np.ndarray,
    p95: np.ndarray,
    verdicts: Sequence[str],
    centres_xy: np.ndarray,
    n_pixels: int,
) -> Path:
    p, n_years = change.shape
    frame = pd.DataFrame(
        {
            "placebo": np.repeat(np.arange(p, dtype=np.int64), n_years),
            "year": np.tile(np.asarray(years, dtype=np.int64), p),
            "divergence": div.reshape(-1),
            "divergence_change": change.reshape(-1),
            "significant": (change > p95[None, :]).reshape(-1),
            "verdict": np.repeat(np.asarray(verdicts, dtype=object), n_years),
            "centre_x": np.repeat(centres_xy[:, 0], n_years),
            "centre_y": np.repeat(centres_xy[:, 1], n_years),
            "pixels": np.full(p * n_years, n_pixels, dtype=np.int64),
        }
    )
    with export.atomic_path(path) as tmp:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
    return path


def _write_validation(
    ctx: RunContext, null: Mapping[str, Any], canopy: Mapping[str, Any] | None, fp_rate: float | None
) -> None:
    caveats = [LIMITATION]
    if canopy is not None:
        if canopy.get("error"):
            caveats.append(f"{CANOPY_SKIPPED}: {canopy['error']}.")
        else:
            caveats += list(CANOPY_CAVEATS)
    reference: dict[str, Any] = {
        "null_test": "the control pool against itself, split into two halves by alternating angular sectors",
        "global_reference": None,
        "placebo_false_positive_rate": fp_rate,
    }
    if canopy is not None:
        reference["canopy"] = {
            "dataset": "GLAD annual tree canopy height (Potapov et al.)",
            "url_template": canopy.get("url_template"),
            "years": canopy.get("years"),
            "error": canopy.get("error"),
        }
    validate.write_validation_json(
        ctx.paths.validation,
        scores={"null_test": dict(null), "canopy": None if canopy is None else dict(canopy)},
        reference=reference,
        caveats=caveats,
    )


# --------------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------------


def _write_preview(
    path: Path,
    *,
    role: np.ndarray,
    years: Sequence[int],
    planted: int,
    change: np.ndarray,
    band: np.ndarray,
    null_change: np.ndarray,
    progress: np.ndarray,
    target_sim: np.ndarray,
    departed: np.ndarray,
    closer: np.ndarray,
    ctrl_closer: np.ndarray,
    canopy: Mapping[str, Any] | None,
    site_sim: np.ndarray,
    ring_sim: np.ndarray,
    threshold: float,
    verdict: Verdict,
    supplied: bool,
) -> Path:
    """Write the six-panel preview with a matplotlib Figure (never pyplot).

    Panels: the role map; the divergence change with the placebo 5 to 95 band, its median and the
    null pair; the target progress (or a note); the per-pixel fractions over time; the canopy gap
    (or a note); and the baseline similarities of the site and the control candidates with the
    control threshold.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    fig = Figure(figsize=(18, 11), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    fig.suptitle(f"Restore: {verdict.verdict} (planted {planted})")
    x = np.asarray(years)
    colours = ["#1a9641", "#2c7bb6", "#fdae61", "#bababa", "#7b3294"]
    ax = axes[0, 0]
    cmap = mpl.colors.ListedColormap(colours).with_extremes(bad=export.PREVIEW_NAN_RGB)
    ax.imshow(np.ma.masked_invalid(role), cmap=cmap, vmin=0.5, vmax=5.5, interpolation="nearest")
    ax.legend(
        handles=[Patch(color=colours[k - 1], label=ROLE_NAMES[k]) for k in ROLE_NAMES],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        fontsize="small",
    )
    ax.set_title("Roles")
    ax.set_axis_off()
    ax = axes[0, 1]
    ax.fill_between(x, band[0], band[2], color="#bdbdbd", alpha=0.6, label="placebo 5 to 95th percentile")
    ax.plot(x, band[1], color="#636363", linestyle="--", linewidth=1, label="placebo median")
    ax.plot(x, change, color="#1a9641", marker="o", linewidth=2, label="site")
    if np.isfinite(null_change).any():
        ax.plot(x, null_change, color="#2166ac", marker="s", linewidth=1, label="null pair (control halves)")
    ax.axvline(planted - 0.5, color="#d7191c", linewidth=1, label="planted")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("year")
    ax.set_ylabel("divergence change since baseline")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_title("Divergence from control (1 - cosine), change since the baseline")
    ax = axes[0, 2]
    if np.isfinite(progress).any():
        ax.plot(x, progress, color="#7b3294", marker="o", label="progress (unclipped)")
        ax.axhline(1.0, color="#7b3294", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_ylabel("share of the way from control to target")
        ax.legend(loc="upper left", fontsize="small")
        ax.set_title("Target progress")
    elif np.isfinite(target_sim).any():
        ax.plot(x, target_sim, color="#7b3294", marker="o", label="similarity to the target")
        ax.legend(loc="upper left", fontsize="small")
        ax.set_title("Similarity to the target (the target could not be split)")
    else:
        ax.text(0.5, 0.5, "no target (--target)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Target progress")
    ax.axvline(planted - 0.5, color="#d7191c", linewidth=1)
    ax = axes[1, 0]
    ax.plot(x, departed, color="#1a9641", marker="o", label="site pixels departed from control")
    ax.axhline(0.05, color="#636363", linestyle="--", linewidth=1, label="expected with no change (0.05)")
    if np.isfinite(closer).any():
        ax.plot(x, closer, color="#7b3294", marker="o", label="site pixels closer to target")
        ax.plot(x, ctrl_closer, color="#2c7bb6", marker="s", label="control pixels closer to target")
    ax.axvline(planted - 0.5, color="#d7191c", linewidth=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("year")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_title("Per-pixel fractions")
    ax = axes[1, 1]
    if canopy is not None and canopy.get("per_year"):
        cy = [int(y) for y in canopy["years"]]
        gap = [canopy["per_year"][str(y)]["gap_mean_m"] for y in cy]
        tall = [canopy["per_year"][str(y)]["gap_tall_share"] for y in cy]
        ax.plot(cy, gap, color="#1a9641", marker="o", label="mean height gap (m), site minus control")
        ax.set_ylabel("metres")
        twin = ax.twinx()
        twin.plot(cy, tall, color="#a6611a", marker="s", label="tall share gap (5 m or more)")
        twin.set_ylabel("share")
        ax.legend(loc="upper left", fontsize="small")
        twin.legend(loc="lower right", fontsize="small")
        ax.axvline(planted - 0.5, color="#d7191c", linewidth=1)
        ax.set_title("GLAD canopy height gap")
    else:
        ax.text(0.5, 0.5, "canopy check not run (--validate)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("GLAD canopy height gap")
    ax = axes[1, 2]
    # The site's spread sets the range; candidates far below it pile up in the first bin.
    site_low = float(np.nanpercentile(site_sim, 0.5)) if site_sim.size else threshold
    low = min(threshold, site_low)
    low = max(-1.0, low - max(0.5 * (1.0 - low), 0.01))
    bins = np.linspace(low, 1.0, 50)
    ax.hist(np.clip(site_sim, low, 1.0), bins=bins, color="#1a9641", alpha=0.6, density=True, label="site")
    if ring_sim.size:
        below = int((ring_sim < low).sum())
        name = "supplied control" if supplied else "ring"
        label = f"{name} ({below:,} below {low:.3f}, shown in the first bin)" if below else name
        ax.hist(np.clip(ring_sim, low, 1.0), bins=bins, color="#2c7bb6", alpha=0.5, density=True, label=label)
    ax.axvline(threshold, color="#d7191c", linewidth=1.5, label=f"threshold {threshold:.4f}")
    ax.set_xlabel("baseline cosine similarity to the site mean")
    ax.legend(loc="upper left", fontsize="small")
    ax.set_title("Control filter")
    final = Path(path)
    with export.atomic_path(final) as tmp, tmp.open("wb") as fh, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(fh, format="png", dpi=100)
    return final


__all__ = [
    "APP",
    "BAND_PERCENTILES",
    "CANOPY_ONSET_GAP",
    "DEFAULT_FOLD_CELL_M",
    "LIMITATION",
    "MIN_CONTROL_PIXELS",
    "MIN_SITE_HA",
    "MIN_SITE_PIXELS",
    "NULL_SECTORS",
    "RestoreError",
    "RestoreResult",
    "Verdict",
    "call_verdict",
    "canopy_check",
    "chunk_runs",
    "closer_fraction",
    "control_threshold",
    "departed_fraction",
    "draw_placebos",
    "false_positive_rate",
    "geodesic_area_ha",
    "heldout_similarity",
    "load_polygon",
    "main_zone",
    "majority_pattern",
    "memory_estimate",
    "null_halves",
    "placebo_band",
    "placebo_divergence",
    "placebo_verdicts",
    "pooled_sum",
    "ring_geometry",
    "run_restore",
    "target_progress",
    "theil_sen_trend",
    "trailing_runs",
    "trajectory_years",
]
