"""Restore: is a restoration site following the trajectory a working project should, against a matched control?

Brief section 4.4, as corrected by the pre-build survey (B5, S2, S16, S17, S18, S19):

- **Inputs** (:func:`run_restore`). The site polygon (any CRS, polygons only; points and lines are an
  error) or a bundled demo (``itapebi``, or one of its features as the site: ``itapebi:negative``),
  the planting year, and optionally a control polygon and a mature reference stand (the target).
  The three must not overlap. ``planted <= 2017`` is refused (no pre-planting year worth having) and
  a 2017 baseline warns (few Sentinel-2 observations, sparse valid pixels in places; survey S16).
  The trajectory covers every year from ``year_a`` (default ``planted - 1``, the baseline) to
  ``year_b`` (default 2025); those years and the match year (``match_year``, below) are checked
  against ``years_complete`` for every zone the site, the ring or control, and the target touch,
  before any chunk is read (survey S2). The error names the zone, the year and the role that touches
  it.
- **Plans.** The site plus a ring (or the site plus the supplied control) is planned as one extent
  with :func:`~tessera_apps.io.plan_aoi` (metadata only), the target separately. The ring is built in
  the site's main UTM zone in metres (outer buffer minus inner buffer, default 1 to 3 km) and less
  the target; its pixels are the control candidates.
- **Stage 1, the match year** (``match_year``: by default ``year_a - 1``, or ``year_a`` itself when
  that is the store's first year; any year from the store's first to ``year_a``; read for the site,
  the ring and the target). The site mean ``m_s`` is the dequantized mean of the site pixels valid
  in the match year (:data:`MIN_SITE_PIXELS` or more). The control filter (survey S17) is the cosine
  of every ring pixel with match-year data to ``m_s`` (exact on int8), kept at or above a threshold
  set on held-out site pixels: the site's pixels are blocked into cells of ``fold_cell_m`` (default
  200 m) with :func:`~tessera_apps.apps.wetland.spatial_blocks`, the cells split into ``folds``
  folds with :func:`~tessera_apps.apps.wetland.assign_folds`, each fold scored against the mean of
  the other folds (:func:`heldout_similarity`), and the threshold is the ``control_pct`` percentile
  of those held-out similarities. A site of one cell falls back to the in-sample percentile with a
  warning (:func:`~tessera_apps.apps.wetland.choose_threshold` models the choice, on distances). By
  default the filter runs on the year before the baseline so that it never selects on the
  baseline's own noise: ring pixels that pass by the luck of one year's noise drift back the next
  year (regression to the mean), and that drift falls between the match year and the baseline,
  outside ``dD``; a match year equal to the baseline is allowed and warns. Filtering the ring is
  the match year's only use: a ring pixel without match-year data cannot be filtered and is left
  out, while site, supplied control and target pixels need no match-year data (the match year may
  be sparse, as 2017 is in places). How many pixels of each role lack it is reported, and the run
  warns when under half the site or the ring has it. A supplied control is used as given, and its
  baseline balance is reported. Too few site pixels with match-year data (:data:`MIN_SITE_PIXELS`)
  or kept control pixels (:data:`MIN_CONTROL_PIXELS`) stop the run here, before stage 2.
- **Stage 2, the other years** (the baseline ``year_a`` to ``year_b``), read only for the store
  chunks that hold a site, kept control or target pixel (:func:`chunk_runs`): a chunk with no kept
  pixel is never fetched again.
- **Panel and radar regime** (survey B5). A pixel is used only if it is valid in every trajectory
  year (``year_a`` to ``year_b``; the match year is not one of them), so the means never change
  composition. Its radar pattern is its radar presence over the trajectory years, and only site,
  control and target pixels with the site's majority pattern are kept, so every within-year
  comparison is between pixels of one regime. When the majority pattern itself has radar-free years
  the run warns: the comparisons stay fair and the placebo band sees the same switch, but the
  trajectory may step at those years.
- **Trajectories** per year: the divergence ``D(t) = 1 - cos(m_s(t), m_c(t))`` and its change since
  the baseline ``dD(t) = D(t) - D(year_a)``; a placebo band from ``placebos`` pseudo-sites of the
  site's footprint, each the kept control pixels within the site's equivalent radius of a random
  control pixel (:func:`draw_placebos`), whose ``dD_p(t)`` is computed exactly as the site's against
  the control pool less the placebo; a year is significant when ``dD(t)`` exceeds both the band's
  95th percentile and zero (:func:`exceeds_band`), so a divergence that falls is never called
  diverging; the verdict (:func:`call_verdict`) and the same rule applied to every placebo (the
  false positive rate); the Theil-Sen slope and Spearman rho of ``dD`` after planting; with a target,
  the site's similarity to it, the control's (the 0 percent level) and the target's own two
  contiguous halves' (the 100 percent level, :func:`axis_halves`), giving an unclipped progress; and
  the per-pixel fractions (:func:`departed_fraction` and its change since the baseline,
  :func:`closer_fraction`).
- **Null test** (always computed, written to the validation file with ``validate``). The control
  pool is split into two halves, by alternating angular sectors around the site centroid for the
  ring (:func:`null_halves`) and by its median along its long axis for a supplied control
  (:func:`axis_halves`), and the same divergence and verdict are computed between the halves. A
  half is usually far larger than the site, which makes this test easy to pass (:func:`null_caveat`
  says when it is not); the placebos are the site-sized null. There is no global reference layer
  for restoration outcomes, so this is a sanity check that the method reports no divergence where
  none is expected, not an accuracy (:data:`LIMITATION`).
- **Canopy height check** (with ``validate`` and ``canopy``): GLAD annual tree canopy height
  (:func:`~tessera_apps.validate.glad_tch`) over the kept pixels, the site minus control gap and its
  change since ``year_a``, the first year the site's tall share leads the control's by
  :data:`CANOPY_ONSET_GAP` and the sign and size of the correlation of the gap change with ``dD``
  (:func:`canopy_check`). It is loaded before any chunk read; when the host cannot be reached the
  run completes with a warning (or stops under ``strict_validate``), and an identical rerun retries
  it once every tile the check needs can be reached.
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
#: The run warns when under this share of the site's or the ring's pixels have match-year data.
MIN_MATCH_SHARE = 0.5
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
#: Angular sectors around the site centroid that split the control ring for the null test.
NULL_SECTORS = 8
#: Fewest pixels in each half of the target for its 100 percent level.
MIN_TARGET_HALF_PIXELS = 10
#: Below this many used target pixels the 100 percent level (two halves of half the pixels each) is
#: biased low by sampling noise against the site's similarity to the whole target, so the progress
#: overstates; the run warns.
MIN_TARGET_UNBIASED_PIXELS = 200
#: Placebo centres tried, per placebo asked for, before the pool counts as too sparse.
PLACEBO_TRIES = 20
#: The run warns when a placebo's median pixel count is below this share of the site's.
PLACEBO_PIXEL_RATIO_WARN = 0.5
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
    "halves, by alternating angular sectors around the site for the ring or by its median along its long axis for a "
    "supplied control, compared exactly as the site is) is a sanity check that the method reports no divergence "
    "where none is expected, not an accuracy. Divergence from the control says the site's embeddings moved away from "
    "comparable land that started like it; whether that movement is the intended vegetation is for the target "
    "trajectory, the canopy check and a site visit to say."
)
#: The smaller control half at least this many times the site's pixels makes the halves test easy to pass.
NULL_EASY_RATIO = 2.0
NULL_CAVEAT = (
    "Each half of the control is far larger than the site (half_to_site_pixels gives the ratio), so its mean "
    "varies far less than a site-sized piece of land does and the halves test is easy to pass; null_to_site_ratio "
    "compares the largest change of the pair with the site's. The site-sized null is the placebo false positive "
    "rate."
)
NULL_CAVEAT_SMALL = (
    "The smaller half of the control is not much larger than the site, or smaller (half_to_site_pixels gives the "
    "ratio), so the halves test is about as demanding as a site-sized one here; null_to_site_ratio compares the "
    "largest change of the pair with the site's. The site-sized null is the placebo false positive rate."
)
#: A site whose baseline divergence from its control is above this share of the placebos' is warned about.
BASELINE_RANK_WARN = 0.95
#: Spearman correlations weaker than this read as no sign (they rest on at most 8 years).
CANOPY_SIGN_MIN_RHO = 0.3
MATCH_NOTE = (
    "The control filter runs on match_year, by default the year before the baseline, so it never selects on the "
    "baseline's own noise: ring pixels that pass by the luck of one year's noise look like their own land again the "
    "next year (regression to the mean), which would show as divergence of a site that did nothing. That drift falls "
    "between the match year and the baseline (selection_drift = baseline_divergence_match_pixels - "
    "match_year_divergence, both over the used site and control pixels that have match-year data, so a sparse match "
    "year does not mix a change of pixels into it; with every used pixel covered, baseline_divergence_match_pixels is "
    "baseline_divergence) and stays out of the divergence change, which is measured from the baseline. When the "
    "match year is the baseline (the baseline is the store's first year, or --match-year says so) the filter and the "
    "baseline share one year, and selection_drift is None."
)
MATCH_DATA_NOTE = (
    "Filtering the ring is the match year's only use. A ring pixel without match-year data cannot be filtered, so it "
    "is left out of the control; site, supplied control and target pixels need data in every trajectory year only "
    "(year_a to year_b), so a sparse match year (2017 in places) does not shrink them. The site's match-year mean, "
    "which sets the filter, is over the site pixels that have match-year data: when they are a minority it matches "
    "the control to that part of the site, and when the ring's are, the control comes from that part of the ring "
    "(the run warns under one half). When the match year is the baseline, a pixel without its data has no baseline "
    "and the panel rule drops it."
)
PROGRESS_NOTE = (
    "target_progress = (s_T - c_T) / (tau - c_T), where s_T is the site's cosine similarity to the target mean, c_T "
    "the control's (the 0 percent level) and tau the cosine between the means of the target's two contiguous halves, "
    "split by a line through its centroid across its long axis (the 100 percent level: as close as one end of the "
    "stand is to the other, so it includes the stand's own spatial variation). It is reported unclipped: below 0 the "
    "site is further from the target than the control is, above 1 it is closer than the target's own halves. With a "
    f"small target (under {MIN_TARGET_UNBIASED_PIXELS} used pixels) each half's mean is noisy, so tau is biased low "
    "and the progress overstates: a site of the target's own kind can read well above 1."
)
FPR_NOTE = (
    "placebo_false_positive_rate is the share of placebos (site-sized pieces of the control, run through the same "
    "verdict rule) that the rule calls diverging. They are judged against a band they are part of, so it cannot "
    "exceed about 5 percent by construction: it shows how much the run requirement of the verdict removes, not "
    "whether the band is calibrated for this site."
)
PLACEBO_NOTE = (
    "Each placebo is a site-sized piece of the control: the kept control pixels within the site's equivalent radius "
    "(sqrt(site area / pi), main zone metres) of a random control pixel, nearest first and at most the site's pixel "
    "count or a third of the control pool, whichever is smaller; a centre whose disk holds fewer than "
    f"{MIN_SITE_PIXELS} kept pixels is redrawn. Its divergence is computed exactly as the site's, against the control "
    "pool less the placebo. The band is the 5th, 50th and 95th percentile of the placebos' divergence change per "
    "year, and a year is significant when the site's change exceeds both the 95th and zero. A placebo with fewer "
    "pixels than the site has a larger sampling term in 1 - cos, so a change in per-pixel spread between years moves "
    "the band rather than only widening it; the zero floor means a divergence that falls is never called diverging. "
    + FPR_NOTE
)
RANK_NOTE = (
    "baseline_divergence_rank is the share of placebos whose own baseline divergence from the rest of the pool is "
    "below the site's: near 1 means the site starts less like its control than placebo pieces of the control do. A "
    "placebo with fewer pixels than the site has a larger sampling term in 1 - cos, so the rank is biased low when "
    "placebos are smaller than the site."
)
DEPARTED_NOTE = (
    "departed_fraction is the share of site pixels whose cosine to the control mean is below the control pixels' own "
    "5th percentile; read departed_fraction_change, its change since the baseline. A site that did nothing does not "
    "sit at 0.05 at the baseline: the control filter drops the ring pixels least like the site in the match year, "
    "which trims the control's low tail, and through noise that persists from year to year part of that trim remains "
    "at the baseline and after, so the level carries an offset the change removes. A supplied control is not "
    "filtered and has no such offset."
)
CANOPY_CAVEATS = (
    "GLAD annual tree canopy height is a Landsat and GEDI model at 30 m that sees only trees of 5 m or more, so a "
    "young planting is invisible to it for its first years and the canopy gap lags the embeddings.",
    "The layer ends in 2024, so 2025 has no canopy value.",
    "The correlations between the divergence change and the canopy gap change rest on at most 8 years (2017 to "
    "2024), and fewer for a later planting: read their sign and the onset year, not their size.",
    "A site picked with this layer (the bundled demo) is not independent of it for site choice, although the "
    "embeddings are independent of it.",
)
CANOPY_SKIPPED = "Canopy check skipped, GLAD tree canopy height could not be loaded"

# Peak memory model (survey S18), bytes. Stage 1 keeps the int8 vector, scale, radar flag and position of
# every valid pixel of the extent and the target, in arrays sized for its role. Stage 2 keeps every kept
# pixel's int8 vectors, scales and radar flags for each year held (the trajectory and the match year) in one
# pool that the panel rule compacts in place, plus the stage 1 positions and similarities, the per-pixel
# cosines, the placebo tree and the pixel centres, and reads windows of whole chunks for every other year
# at once: about two blocks of pixel-years each, wherever the chunks lie (so the block is not clipped to
# the plan), at about 410 bytes per pixel-year with the decode buffers. The write holds the positions, the
# per-pixel cosines, the float32 bands of the mosaic and their per-zone copies and the COG writer's own;
# the canopy check holds one uint8 height window per year and zone. Calibrated on two measured runs (1024
# and 2048 px grids, 3 and 8 km rings, 0.28 and 2.1 million extent pixels, 7 years): peak RSS 1.4 and 3.4
# GB against estimates of 2.4 and 5.5 GB (the earlier model said 1.3 and 4.0 GB for runs that peaked at
# 1.6 and 6.4 GB).
MEMORY_FIXED = 700_000_000
MEMORY_STAGE1_PER_PIXEL = 160
MEMORY_PER_PIXEL_YEAR = 135
MEMORY_STAGE2_PER_PIXEL = 150
MEMORY_PER_BLOCK_PIXEL_YEAR = 420
MEMORY_WRITE_PER_PIXEL = 120
MEMORY_WRITE_PER_WINDOW_PIXEL_BAND = 40
MEMORY_MARGIN = 1.4


class RestoreError(io.TesseraError):
    """The data cannot answer the question as asked (too few valid site, control or target pixels)."""


# --------------------------------------------------------------------------------------------
# Inputs: polygons, the ring and the years
# --------------------------------------------------------------------------------------------


def _demo_role_aoi(name: str, role: str, *, option: str | None = None) -> LoadedAOI:
    """A bundled demo's feature of one role as a loaded polygon, digested with its role.

    For ``--control demo`` and ``--target demo`` (``role`` is the option's) the properties are left
    out; for the site (``--aoi itapebi:negative``, ``option="aoi"``) they are the feature's own, so
    its ``planted`` year is the one used.

    Raises:
        AppInputError: The demo has no feature with that role (names the roles it has).
    """
    try:
        geometry = demo.demo_feature(name, role)
        props = demo.demo_feature_properties(name, role) if option == "aoi" else {}
    except ValueError as exc:
        what = f"--aoi {name}:{role}" if option == "aoi" else f"--{role} demo"
        raise AppInputError(f"{what}: {exc}") from exc
    on_disk = demo.demo_path(name)
    sha = hashlib.sha256(on_disk.read_bytes() + b"\0" + role.encode("utf-8")).hexdigest()
    poly = io._polygonal(geometry if geometry.is_valid else shapely.make_valid(geometry))
    label = f"{name}:{role}" if option == "aoi" else f"{name} {role}"
    return LoadedAOI(
        geometry=poly,
        path=on_disk if on_disk.is_file() else None,
        sha256=sha,
        name=label,
        properties=props,
        demo=name,
        demo_role=role,
    )


def demo_role_spec(text: str) -> tuple[str, str] | None:
    """``(demo, role)`` when ``text`` is ``<demo>:<role>`` with a bundled demo's name, else None.

    Both parts are taken without surrounding spaces and in lower case (``Itapebi:Negative`` is
    ``("itapebi", "negative")``); an empty role is kept, so the loader can refuse it by name.
    """
    name, colon, role = str(text).strip().partition(":")
    key = name.strip().lower()
    if not colon or key not in demo.DEMO_NAMES:
        return None
    return key, role.strip().lower()


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

    For the site a demo name loads the demo's AOI feature (:func:`~tessera_apps.apps._common.load_aoi`),
    and ``<demo>:<role>`` loads the demo's feature of that role instead (``itapebi:negative``, the
    pasture that stayed pasture), with that feature's properties (its ``planted`` year); the site's
    demo stays the demo, so ``"demo"`` for the control and the target still reads that demo's file.
    For the control and the target, ``"demo"`` loads the feature of that role from the site's demo.
    A regular file of the same name wins over a demo name.

    Args:
        spec: Path, demo name or ``<demo>:<role>`` (site), or ``"demo"`` (control, target), or an
            already loaded AOI.
        role: ``site``, ``control`` or ``target``, for the messages.
        demo_name: The site's demo, for ``"demo"``.

    Raises:
        AppInputError: The file is missing or unreadable, holds points or lines, or has no polygon;
            ``"demo"`` is given without a demo site; or the demo has no feature of the role named.
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
    named = demo_role_spec(text) if role == "site" and not path.is_file() else None
    if named is not None:
        name, feature_role = named
        if feature_role == "aoi":
            return load_aoi(name)
        return _demo_role_aoi(name, feature_role, option="aoi")
    if not path.is_file():
        also = (
            f" nor a bundled demo ({', '.join(demo.DEMO_NAMES)}, or <demo>:<role> for one of a demo's features)"
            if role == "site"
            else ""
        )
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


@dataclasses.dataclass(frozen=True)
class Placebos:
    """Placebo pseudo-sites drawn from the control pool (:func:`draw_placebos`).

    Attributes:
        members: (P, k) indices into the control pool, nearest first, padded with -1.
        centres: (P,) centre index of each placebo.
        sizes: (P,) pixels of each placebo.
        reach_m: (P,) distance from each centre to its farthest member, in metres.
        tried: Centres tried to find them.
    """

    members: np.ndarray
    centres: np.ndarray
    sizes: np.ndarray
    reach_m: np.ndarray
    tried: int


def draw_placebos(
    xy: np.ndarray,
    n_placebos: int,
    n_pixels: int,
    seed: int,
    *,
    radius_m: float | None = None,
    min_pixels: int = 1,
) -> Placebos:
    """Placebo pseudo-sites: a random control pixel (seeded) and its nearest control pixels.

    With ``radius_m`` a placebo is the pool pixels within that distance of its centre (at most
    ``n_pixels`` of them, nearest first), so it covers about the site's footprint however sparse the
    pool is; a centre whose disk holds fewer than ``min_pixels`` pixels is passed over. Without it a
    placebo is the ``n_pixels`` nearest pixels, however far they reach. Centres are distinct while
    the pool has enough of them; after :data:`PLACEBO_TRIES` centres per placebo, or once every pool
    pixel has been tried, the placebos found are repeated (seeded) to make up the number.

    Args:
        xy: (N, 2) control pixel centres in metres.
        n_placebos: How many placebos.
        n_pixels: Most pixels per placebo, the centre included.
        seed: Random seed.
        radius_m: Footprint radius in metres, or None for the plain nearest pixels.
        min_pixels: Fewest pixels a placebo may hold.

    Returns:
        The placebos.

    Raises:
        ValueError: ``n_pixels`` is not between 1 and N, or no centre has ``min_pixels`` pixels
            within ``radius_m``.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(xy, dtype=np.float64)
    n = int(pts.shape[0])
    k = int(n_pixels)
    want = int(n_placebos)
    if not 1 <= k <= n:
        raise ValueError(f"Placebos of {n_pixels} pixels need 1 to {n} control pixels.")
    rng = np.random.default_rng(seed)
    tree = cKDTree(pts)
    bound = math.inf if radius_m is None else float(radius_m)
    order = rng.permutation(n)
    budget = min(n, max(want, PLACEBO_TRIES * want))
    found_members: list[np.ndarray] = []
    found_centres: list[int] = []
    found_reach: list[float] = []
    tried = 0
    # Centres are queried in batches whose (batch, k) neighbour arrays stay near 2 million entries.
    batch = max(1, min(max(want, 64), 2_000_000 // k))
    while len(found_centres) < want and tried < budget:
        cand = order[tried : min(tried + batch, budget)]
        tried += cand.size
        dist, idx = tree.query(pts[cand], k=k, distance_upper_bound=bound)
        dist = np.asarray(dist, dtype=np.float64).reshape(cand.size, k)
        idx = np.asarray(idx, dtype=np.int64).reshape(cand.size, k)
        for c, d, i in zip(cand, dist, idx, strict=True):
            inside = i < n
            if int(inside.sum()) < min_pixels:
                continue
            found_members.append(i[inside])
            found_centres.append(int(c))
            found_reach.append(float(d[inside].max()))
            if len(found_centres) == want:
                break
    if not found_centres:
        raise ValueError(
            f"No control pixel has {min_pixels} or more control pixels within {bound:g} m: the pool is too sparse "
            "for placebos of the site's footprint."
        )
    pick = np.arange(len(found_centres))
    if pick.size < want:
        pick = np.concatenate([pick, rng.choice(pick.size, size=want - pick.size, replace=True)])
    members = np.full((want, k), -1, dtype=np.int64)
    sizes = np.zeros(want, dtype=np.int64)
    for row, j in enumerate(pick):
        m = found_members[int(j)]
        members[row, : m.size] = m
        sizes[row] = m.size
    return Placebos(
        members=members,
        centres=np.asarray(found_centres, dtype=np.int64)[pick],
        sizes=sizes,
        reach_m=np.asarray(found_reach, dtype=np.float64)[pick],
        tried=tried,
    )


def placebo_divergence(emb: np.ndarray, scale: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Divergence ``D_p(t)`` of every placebo from the control pool less itself, per year.

    Args:
        emb: (N, Y, D) int8 control vectors.
        scale: (N, Y) control scales.
        members: (P, k) indices of each placebo's pixels; entries below 0 are padding.

    Returns:
        (P, Y) float64 ``1 - cos(sum of the placebo, sum of the rest)``.
    """
    from scipy.sparse import csc_matrix

    n, n_years = scale.shape
    mem = np.asarray(members, dtype=np.int64)
    p, k = mem.shape
    rows = np.repeat(np.arange(p), k)
    cols = mem.reshape(-1)
    ok = cols >= 0
    m = csc_matrix((np.ones(int(ok.sum()), dtype=np.float64), (rows[ok], cols[ok])), shape=(p, n))
    m.sum_duplicates()
    m.data[:] = 1.0
    out = np.empty((p, n_years), dtype=np.float64)
    for t in range(n_years):
        inside = np.zeros((p, emb.shape[-1]), dtype=np.float64)
        total = np.zeros(emb.shape[-1], dtype=np.float64)
        for start in range(0, n, STRIP):
            stop = min(start + STRIP, n)
            x = emb[start:stop, t, :].astype(np.float64)
            x *= scale[start:stop, t, None]
            inside += m[:, start:stop] @ x
            total += x.sum(axis=0)
        out[:, t] = 1.0 - cosine_rows(inside, total[None, :] - inside)
    return out


def placebo_band(change: np.ndarray) -> np.ndarray:
    """(3, Y) 5th, 50th and 95th percentiles of the placebos' (P, Y) divergence change per year."""
    return np.percentile(np.asarray(change, dtype=np.float64), BAND_PERCENTILES, axis=0)


def exceeds_band(change: np.ndarray, p95: np.ndarray) -> np.ndarray:
    """Significant years: the change exceeds both the band's 95th percentile and zero (one-sided).

    The rule of the site, the placebos and the null pair. The zero floor matters when the band sits
    below zero: placebos smaller than the site have a larger sampling term in ``1 - cos``, so a
    year whose per-pixel spread shrinks moves their changes further down than the site's, and a
    divergence that falls would otherwise read as one that grows.

    Args:
        change: (..., Y) divergence changes.
        p95: (Y,) the band's 95th percentile, broadcast against ``change``.

    Returns:
        Bool array of ``change``'s shape; False where either value is not finite.
    """
    ch = np.asarray(change, dtype=np.float64)
    hi = np.asarray(p95, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (ch > np.maximum(hi, 0.0)) & np.isfinite(ch) & np.isfinite(hi)


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
    exceeds the band's 95th percentile and zero, :func:`exceeds_band`), with at least two such
    years: diverging since ``Y``. Otherwise no divergence detected.
    """
    ys = [int(y) for y in years]
    post = np.array([y >= planted for y in ys], dtype=bool)
    post_years = [y for y in ys if y >= planted]
    sig = exceeds_band(change, p95)
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
    if run == 1 and sig_years == [last]:
        detail = f"divergence in {last} only; one year is not enough to call it."
    elif run == 1:
        detail = (
            f"The divergence change exceeds the placebo band in {', '.join(str(y) for y in sig_years)}, but the run "
            f"of significant years ending in {last} is that one year, which is not enough to call it."
        )
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
    sig = exceeds_band(ch, np.asarray(p95, dtype=np.float64)[None, :])
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
    """Half (0 or 1) of every ring pixel: alternating angular sectors around the site centroid.

    This splits a ring around the site into two interleaved halves; a supplied control off to one
    side of the site can fall in one sector, so it is split by :func:`axis_halves` instead.

    Args:
        xy: (N, 2) pixel centres in metres.
        centre: The site centroid in the same metres.
        sectors: Number of equal sectors; even sectors form half 0, odd ones half 1.
    """
    d = np.asarray(xy, dtype=np.float64) - np.asarray(centre, dtype=np.float64)[None, :]
    angle = np.mod(np.arctan2(d[:, 1], d[:, 0]), 2.0 * math.pi)
    sector = np.floor(angle / (2.0 * math.pi / sectors)).astype(np.int64) % sectors
    return (sector % 2).astype(np.int8)


def axis_halves(xy: np.ndarray, *, split: str = "median") -> np.ndarray:
    """Half (0 or 1) of every pixel: the two sides of a line across the pixels' long axis.

    The long axis is the principal axis of the pixel centres. ``split="median"`` cuts at the median
    of the projections on it, so the halves hold equal counts (half 1 gets the odd pixel);
    ``split="centroid"`` cuts at the centroid, so the halves are the two contiguous ends of the area.

    Args:
        xy: (N, 2) pixel centres in metres.
        split: ``"median"`` or ``"centroid"``.

    Raises:
        ValueError: An unknown ``split``.
    """
    if split not in ("median", "centroid"):
        raise ValueError(f"split must be 'median' or 'centroid', not {split!r}.")
    pts = np.asarray(xy, dtype=np.float64)
    out = np.zeros(pts.shape[0], dtype=np.int8)
    if pts.shape[0] < 2:
        return out
    centred = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    proj = centred @ vectors[:, -1]
    if split == "median":
        order = np.argsort(proj, kind="stable")
        out[order[pts.shape[0] // 2 :]] = 1
    else:
        out[proj > 0.0] = 1
    return out


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


def correlation_sign(rho: float | None) -> str | None:
    """The sign of a Spearman correlation, read only when it has some size.

    ``positive`` or ``negative`` for a correlation of at least :data:`CANOPY_SIGN_MIN_RHO` in size,
    ``none`` for a weaker one (at most 8 years cannot tell a weak correlation from none), and None
    without one.
    """
    if rho is None or not math.isfinite(rho):
        return None
    if rho >= CANOPY_SIGN_MIN_RHO:
        return "positive"
    if rho <= -CANOPY_SIGN_MIN_RHO:
        return "negative"
    return "none"


def null_caveat(half_to_site: float | None) -> str:
    """The caveat for the control halves test, by how much larger than the site its smaller half is."""
    if half_to_site is not None and math.isfinite(half_to_site) and half_to_site < NULL_EASY_RATIO:
        return NULL_CAVEAT_SMALL
    return NULL_CAVEAT


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
        ``canopy_onset_year`` and ``correlation_sign`` first (:func:`correlation_sign` of the Spearman
        correlation of ``dD`` with the mean height gap change: what the few years can say), then per-year means,
        tall shares, gaps and gap changes, and Pearson and Spearman correlation of ``dD`` with the
        change in the mean height gap and in the tall share gap over ``correlation_years`` years.
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
    sign = correlation_sign(spearman_m)
    return {
        "canopy_onset_year": onset,
        "correlation_sign": sign,
        "correlation_years": int((np.isfinite(dd) & np.isfinite(gm_change)).sum()),
        "years": ys,
        "tall_m": tall_m,
        "per_year": per_year,
        "pearson_divergence_vs_mean_gap_change": pearson_m,
        "spearman_divergence_vs_mean_gap_change": spearman_m,
        "pearson_divergence_vs_tall_gap_change": pearson_t,
        "spearman_divergence_vs_tall_gap_change": spearman_t,
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
class _Where:
    """Grid positions of pixels: zone index, zone grid row and column."""

    zone: np.ndarray
    row: np.ndarray
    col: np.ndarray

    @property
    def n(self) -> int:
        return int(self.zone.size)


@dataclasses.dataclass
class _Pixels:
    """Pixels of one or more roles: positions and, per year, int8 vectors, scales and radar flags."""

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

    def rows(self, start: int, stop: int) -> _Pixels:
        """Pixels ``start`` to ``stop`` as views (no copy)."""
        return _Pixels(*(getattr(self, f.name)[start:stop] for f in dataclasses.fields(self)))

    def from_year(self, first: int) -> _Pixels:
        """The years from index ``first`` on, as views (no copy)."""
        return dataclasses.replace(
            self, emb=self.emb[:, first:], scale=self.scale[:, first:], radar=self.radar[:, first:]
        )

    def where(self) -> _Where:
        """The positions only, as views: holding them keeps no vector alive."""
        return _Where(self.zone, self.row, self.col)


class _Collector:
    """Stage 1 pixels of one role, gathered block by block into arrays sized for the role (one year).

    A pixel without data that year is kept with a zero vector, a NaN scale and ``ok`` False, so
    it can still join the stage 2 pool (the site, a supplied control and the target need no
    match-year data).
    """

    def __init__(self, capacity: int) -> None:
        cap = int(capacity)
        self.n = 0
        self.zone = np.zeros(cap, dtype=np.int16)
        self.row = np.zeros(cap, dtype=np.int64)
        self.col = np.zeros(cap, dtype=np.int64)
        self.emb = np.zeros((cap, N_BANDS), dtype=np.int8)
        self.scale = np.full(cap, np.nan, dtype=np.float32)
        self.radar = np.zeros(cap, dtype=bool)
        self.ok = np.zeros(cap, dtype=bool)

    def add(self, zone: int, win: io.Window, sel: np.ndarray, ok: np.ndarray) -> None:
        """Add the ``sel`` pixels of a window; ``ok`` marks those with data (the rest get none)."""
        rr, cc = np.nonzero(sel)
        if rr.size == 0:
            return
        assert win.embeddings is not None
        a, b = self.n, self.n + rr.size
        if b > self.zone.size:
            raise AssertionError(f"Stage 1 found {b} pixels of a role sized for {self.zone.size}.")
        good = ok[rr, cc]
        self.zone[a:b] = zone
        self.row[a:b] = rr + win.rows.start
        self.col[a:b] = cc + win.cols.start
        self.emb[a:b] = np.where(good[:, None], win.embeddings[rr, cc], np.int8(0))
        self.scale[a:b] = np.where(good, win.scales[rr, cc], np.float32(np.nan))
        self.radar[a:b] = win.radar_present[rr, cc] & good
        self.ok[a:b] = good
        self.n = b

    def done(self, role: int) -> tuple[_Pixels, np.ndarray]:
        """The pixels gathered, as one-year :class:`_Pixels`, and which of them have data."""
        k = self.n
        pix = _Pixels(
            role=np.full(k, role, dtype=np.int8),
            zone=self.zone[:k],
            row=self.row[:k],
            col=self.col[:k],
            emb=self.emb[:k, None, :],
            scale=self.scale[:k, None],
            radar=self.radar[:k, None],
        )
        return pix, self.ok[:k]


def _build_pool(parts: Sequence[tuple[_Pixels, np.ndarray | None]], n_years: int) -> _Pixels:
    """The stage 2 pool: the selected stage 1 pixels of each part, with year index 0 filled from stage 1.

    Every other year is empty (NaN scale, zero vector) until stage 2 fills it. The vectors are
    copied in strips, so no whole intermediate copy is made.

    Args:
        parts: ``(stage 1 pixels, selection or None for all)`` per role, in role order.
        n_years: Years of the pool.
    """
    sels = [np.arange(p.n) if sel is None else np.flatnonzero(sel) for p, sel in parts]
    n = sum(int(s.size) for s in sels)
    pool = _Pixels(
        role=np.zeros(n, dtype=np.int8),
        zone=np.zeros(n, dtype=np.int16),
        row=np.zeros(n, dtype=np.int64),
        col=np.zeros(n, dtype=np.int64),
        emb=np.zeros((n, n_years, N_BANDS), dtype=np.int8),
        scale=np.full((n, n_years), np.nan, dtype=np.float32),
        radar=np.zeros((n, n_years), dtype=bool),
    )
    at = 0
    for (pix, _), idx in zip(parts, sels, strict=True):
        for start in range(0, idx.size, STRIP):
            i = idx[start : start + STRIP]
            a, b = at + start, at + start + i.size
            pool.role[a:b] = pix.role[i]
            pool.zone[a:b] = pix.zone[i]
            pool.row[a:b] = pix.row[i]
            pool.col[a:b] = pix.col[i]
            pool.emb[a:b, 0] = pix.emb[i, 0]
            pool.scale[a:b, 0] = pix.scale[i, 0]
            pool.radar[a:b, 0] = pix.radar[i, 0]
        at += idx.size
    return pool


def _compact(pix: _Pixels, keep: np.ndarray) -> _Pixels:
    """The kept pixels, moved to the front of ``pix``'s own arrays in strips (order kept), as views.

    The arrays are overwritten in place, so no second copy of the pool is ever held; ``pix`` must not
    be used afterwards.
    """
    idx = np.flatnonzero(keep)
    k = int(idx.size)
    arrays = [getattr(pix, f.name) for f in dataclasses.fields(pix)]
    for start in range(0, k, STRIP):
        i = idx[start : start + STRIP]
        if i.size and i[0] == start and i[-1] == start + i.size - 1:
            continue  # already in place
        for a in arrays:
            a[start : start + i.size] = a[i]
    return _Pixels(*(a[:k] for a in arrays))


def _valid_years(pix: _Pixels) -> np.ndarray:
    """(n, Y) pixels valid each year: finite, positive scale and a vector that is not all zero (in strips)."""
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(pix.scale) & (pix.scale > 0)
    for start in range(0, pix.n, STRIP):
        stop = min(start + STRIP, pix.n)
        ok[start:stop] &= np.any(pix.emb[start:stop] != 0, axis=-1)
    return ok


# --------------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------------


def memory_estimate(plan: AOIPlan, *, block: tuple[int, int], n_years: int, target_pixels: int = 0) -> int:
    """Peak memory of a run in bytes, from the extent plan, the block shape, the years and the target size.

    Args:
        plan: The extent plan.
        block: Block shape of the reads.
        n_years: Years held per pixel: the trajectory years plus the match year when it is separate.
        target_pixels: Pixels of the target, planned apart from the extent.
    """
    window_px = sum(z.shape[0] * z.shape[1] for z in plan.zones)
    pixels = plan.n_pixels + int(target_pixels)
    block_px = max((min(block[0], z.shape[0]) * min(block[1], z.shape[1]) for z in plan.zones), default=0)
    stage1 = MEMORY_STAGE1_PER_PIXEL * pixels + MEMORY_PER_BLOCK_PIXEL_YEAR * block_px
    stage2 = (
        MEMORY_PER_PIXEL_YEAR * n_years * pixels
        + MEMORY_STAGE2_PER_PIXEL * pixels
        + MEMORY_PER_BLOCK_PIXEL_YEAR * int(block[0]) * int(block[1]) * 2
    )
    write = MEMORY_WRITE_PER_PIXEL * pixels + MEMORY_WRITE_PER_WINDOW_PIXEL_BAND * (n_years + 3) * window_px
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
    match_year: int
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

    @property
    def read_years(self) -> list[int]:
        """Every year the run reads: the match year (when it is not the baseline) and the trajectory."""
        return sorted({self.match_year, *self.years})


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
    #: Why the canopy layer could not be loaded: ``unreachable``, ``missing_tile`` or ``bad_tile``.
    canopy_error_kind: str | None = None


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
        what = f"demo {site.demo!r}" if site.demo_role is None else f"demo feature {site.demo}:{site.demo_role}"
        raise AppInputError(f"The {what} gives no planting year; pass --planted YYYY.")
    raise AppInputError("--planted YYYY is required: the year the site was planted.")


def _sha_record(loaded: LoadedAOI | None) -> dict[str, Any] | None:
    """A polygon's digest record: its name and bytes and the loaded geometry, not where it lies.

    The path stays out (it is in the summary), so the same files under another path or spelling do
    not rerun, as wetland's seed digest does.
    """
    if loaded is None:
        return None
    return {"name": loaded.name, "sha256": loaded.sha256, "geometry_sha256": loaded.geometry_sha256}


def match_year_for(year_a: int) -> int:
    """The default match year: the year before the baseline, or the baseline itself when it is the store's first."""
    return int(year_a) - 1 if int(year_a) > FIRST_YEAR else int(year_a)


def resolve_match_year(year_a: int, match_year: int | None) -> int:
    """The year the control filter runs on: ``match_year`` checked, or :func:`match_year_for` when None.

    It must be one of the store's years (:data:`~tessera_apps.io.YEARS`) and at most ``year_a``: the
    control is matched before the trajectory starts. Equal to ``year_a`` it is allowed, and the run
    warns (the filter then selects on the baseline's own noise).

    Raises:
        AppInputError: Says which rule failed and what to change.
    """
    if match_year is None:
        return match_year_for(year_a)
    if int(match_year) != match_year:
        raise AppInputError(f"--match-year ({match_year}) must be a year.")
    m = int(match_year)
    if m not in io.YEARS:
        raise AppInputError(f"--match-year {m} is outside the store's years, {FIRST_YEAR} to {LAST_YEAR}.")
    if m > int(year_a):
        raise AppInputError(
            f"--match-year {m} is after the baseline year ({year_a}): the control is matched before the trajectory "
            f"starts. Use {match_year_for(year_a)} (the default) or another year from {FIRST_YEAR} to {year_a}."
        )
    return m


def run_restore(
    aoi: str | Path | LoadedAOI,
    out_dir: Path | str,
    *,
    planted: int | None = None,
    year_a: int | None = None,
    year_b: int | None = None,
    match_year: int | None = None,
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
    ``restore_validation.json`` when validating (the site-sized null, the null test and the canopy
    check), and ``restore_report.json`` last. A rerun with the same arguments skips unless ``force``;
    one whose canopy check failed because the host could not be reached or a tile was missing is
    redone once every tile the check needs can be reached (a broken tile needs ``force``).

    Args:
        aoi: The site: a polygon file (any CRS), a demo name, ``<demo>:<role>`` for another feature
            of a demo (``itapebi:negative``), or a loaded AOI.
        out_dir: Output directory.
        planted: The planting year; a demo site's (or demo feature's) properties may give it.
        year_a: The baseline year, at most ``planted - 1`` (the default).
        year_b: The last year, at least ``planted``; the store's last year by default.
        match_year: The year the control filter runs on, from the store's first year to ``year_a``;
            None for the year before the baseline (:func:`match_year_for`). Equal to ``year_a`` it
            warns (regression to the mean).
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
        strict_validate: With ``validate``, stop with StrictValidationError when the validation cannot
            be done: before any chunk read when the canopy layer (with ``canopy``) cannot be loaded,
            and before any output is written when the null test cannot run.

    Returns:
        The report, output paths and whether the run was skipped.

    Raises:
        AppInputError: A bad argument, polygon, overlap or output directory, before any chunk read.
        AOIError: The extent or target is too large, touches too many zones, or covers no pixel.
        YearsNotPublishedError: A trajectory year or the match year is missing for a zone a role touches
            (names it).
        StrictValidationError: ``strict_validate`` is set and the canopy layer cannot be loaded or the
            null test cannot run.
        RestoreError: Too few site pixels with match-year data or kept control pixels (before stage
            2), or too few site, control or target pixels left after the panel and regime rule.
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
    match = resolve_match_year(year_a, match_year)
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
        fix = "Move the target closer, or narrow the ring." if ctrl is None else "Move the target or control closer."
        raise AppInputError(
            f"The site, {'ring' if ctrl is None else 'control'} and target touch {len(touched)} zone groups "
            f"({', '.join(touched)}); the limit is {max_zones}. {fix}"
        )
    template = _tch_template(tch_url_template)
    opts = _Options(
        site=site,
        control=ctrl,
        target=tgt,
        planted=plant,
        years=years,
        match_year=match,
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
        "match_year": match,
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
            "match_year_rule": "the ring needs match-year data; site, supplied control and target do not",
            "significance": "change above both the placebo band's 95th percentile and zero",
            "placebo_footprint": "kept control pixels within the site's equivalent radius",
            "target_halves": "two sides of a line through the target's centroid across its long axis",
        },
    }
    if validate and canopy and not force and _canopy_failed_before(paths.report):
        bounds = [extent.bounds] + ([] if tgt is None else [tgt.geometry.bounds])
        force = _canopy_reachable_now(bounds, opts.years, template)
    target_px = 0 if tgt is None else int(geodesic_area_ha(tgt.geometry) / HA_PER_PIXEL)
    state = _State()
    loaded = dataclasses.replace(site, geometry=extent)
    try:
        ctx = prepare_run(
            APP,
            loaded,
            opts.read_years,
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
                plan, block=block, n_years=len(opts.read_years), target_pixels=target_px
            ),
            preflight=lambda c: _preflight(c, state, opts),
        )
    except io.YearsNotPublishedError as exc:
        _note_roles(exc, opts)
        raise
    except io.AOIError as exc:
        if ctrl is None:
            exc.add_note(
                f"The planned extent is the site plus its control ring ({ring_inner_km:g} to {ring_outer_km:g} km); "
                "a narrower ring (--ring-outer-km) makes it smaller."
            )
        else:
            exc.add_note(
                "The planned extent is the site plus the supplied control: choose a smaller control, or one closer "
                "to the site."
            )
        raise
    if ctx is not None:
        # The extent (site plus ring or control) was planned and digested; the report's AOI is the site.
        ctx.aoi = site
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


def _years_read_text(opts: _Options) -> str:
    """The years a run reads, in words: the trajectory and, when it is separate, the match year."""
    a, b, m = opts.years[0], opts.years[-1], opts.match_year
    if m == a - 1:
        return f"every year from {m} to {b} (the control is matched on {m}, the year before the baseline)"
    if m < a:
        return f"{m} (the year the control is matched on, --match-year) and every year from {a} to {b}"
    return f"every year from {a} to {b}"


def _note_roles(exc: io.YearsNotPublishedError, opts: _Options) -> None:
    for zone, missing in sorted(exc.missing.items()):
        roles = _roles_touching(zone, opts) or ["extent"]
        named = roles[0] if len(roles) == 1 else f"{', '.join(roles[:-1])} and {roles[-1]}"
        exc.add_note(
            f"Zone {zone} is touched by the {named} and does not publish "
            f"{', '.join(str(y) for y in missing)}; restore reads {_years_read_text(opts)} there. Choose other "
            "years (--year-a, --year-b, --match-year) or move the polygons."
        )


#: Canopy failures an identical rerun retries by itself, once every tile can be reached: the host was
#: down, or a tile did not exist yet. A tile that exists but cannot be read (a broken local copy, an
#: unexpected grid) stays broken until it is replaced, so it needs ``force``.
CANOPY_RETRYABLE = ("unreachable", "missing_tile")


def _canopy_error_kind(exc: BaseException) -> str:
    """``unreachable`` (the host), ``missing_tile`` (no tile over a window) or ``bad_tile`` (an unreadable tile)."""
    if isinstance(exc, validate.ReferenceUnavailableError):
        return "unreachable"
    if isinstance(exc, validate.GladTileMissingError):
        return "missing_tile"
    return "bad_tile"


def _canopy_failed_before(report_path: Path) -> bool:
    """Whether the report on disk records a canopy check that failed in a way a rerun may retry."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    canopy = ((report.get("summary") or {}).get("canopy")) if isinstance(report, dict) else None
    if not (isinstance(canopy, dict) and canopy.get("error")):
        return False
    kind = canopy.get("error_kind")
    if kind is not None and kind not in CANOPY_RETRYABLE:
        logger.info(
            "%s: the canopy check on disk failed on a tile that cannot be read (%s); pass --force once it is replaced.",
            APP,
            canopy.get("error"),
        )
        return False
    return True


def _canopy_reachable_now(
    bounds: Sequence[tuple[float, float, float, float]], years: Sequence[int], template: str
) -> bool:
    """Whether every canopy year has a tile over every bbox now (one HEAD or file check per tile).

    The loader needs a tile for each canopy year over each window (the extent's and the target's),
    so the retry asks the same of every (year, bbox) pair; one missing pair keeps the outputs.
    """
    canopy_years = [y for y in years if y in validate.GLAD_TCH_YEARS]
    if not canopy_years:
        return False
    try:
        for y in canopy_years:
            for bbox in bounds:
                if validate.glad_tch_reachable(bbox, y, url_template=template) == 0:
                    logger.info("%s: no canopy tile for %d covers %s yet; outputs kept.", APP, y, bbox)
                    return False
    except (validate.ReferenceDataError, ValueError) as exc:
        logger.info("%s: the canopy layer still cannot be reached (%s); outputs kept.", APP, exc)
        return False
    logger.info("%s: the canopy check failed in the run on disk and the layer can be reached now; rerunning.", APP)
    return True


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
    if control_px < MIN_CONTROL_PIXELS:
        what = "ring" if opts.control is None else "control"
        fix = (
            "widen the ring (--ring-outer-km) or move the target out of it"
            if opts.control is None
            else "pass a larger control polygon"
        )
        raise AppInputError(
            f"The {what} covers {control_px} pixel centres on the store grid; the app needs at least "
            f"{MIN_CONTROL_PIXELS} control pixels, so it cannot keep enough: {fix}."
        )
    zones = list(plan.zone_names)
    if opts.target is not None:
        try:
            tplan = io.plan_aoi(
                ctx.reader, opts.target.geometry, max_zones=opts.max_zones, max_area_km2=opts.max_area_km2
            )
        except io.AOIError as exc:
            raise AppInputError(f"The target cannot be planned: {exc}") from exc
        io.reader_check_years(ctx.reader, tplan.zone_names, opts.read_years)
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
        "%s: site %d pixels, %s %d pixels%s; planted %d, years %d to %d, control matched on %d.",
        APP,
        site_px,
        "ring" if opts.control is None else "control",
        control_px,
        "" if state.target_plan is None else f", target {state.target_plan.n_pixels} pixels",
        opts.planted,
        opts.years[0],
        opts.years[-1],
        opts.match_year,
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
        state.canopy_error_kind = _canopy_error_kind(exc)
        logger.warning("The canopy check will be skipped, GLAD tree canopy height could not be loaded: %s", exc)
        return
    state.canopy = heights


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Stage1:
    """Stage 1 (the match year): every site and target pixel, and the ring or control pixels.

    Attributes:
        site: Every site pixel; ``site_ok`` marks those with match-year data.
        ring: The ring pixels with match-year data (the filter needs it), or every pixel of a
            supplied control; ``ring_ok`` marks those with match-year data.
        target: Every target pixel; ``target_ok`` marks those with match-year data.
        counts: Pixel centres per role: ``site_pixels``, ``control_pixels`` (ring candidates or the
            supplied control), ``target_in_extent_pixels`` and, with a target, ``target_pixels``.
        ring_with_data: Ring or control pixels with match-year data.
    """

    site: _Pixels
    site_ok: np.ndarray
    ring: _Pixels
    ring_ok: np.ndarray
    target: _Pixels
    target_ok: np.ndarray
    counts: dict[str, int]
    ring_with_data: int


def _stage1(ctx: RunContext, state: _State, year: int, *, supplied: bool) -> _Stage1:
    """Read one year for every block of the extent and the target.

    Every site and target pixel is kept, with or without data that year; ring pixels are kept only
    with data (a ring pixel without it cannot be filtered), every pixel of a supplied control is.
    """
    zindex = {z: i for i, z in enumerate(state.zones)}
    counts = {"site_pixels": 0, "control_pixels": 0, "target_in_extent_pixels": 0}
    for zp in ctx.plan.zones:
        roles = state.roles[zp.zone]
        counts["site_pixels"] += int(roles.site.sum())
        counts["control_pixels"] += int(roles.control.sum())
        counts["target_in_extent_pixels"] += int(roles.target.sum())
    target_cap = 0 if state.target_plan is None else int(state.target_plan.n_pixels)
    site, ring, target = (
        _Collector(counts["site_pixels"]),
        _Collector(counts["control_pixels"]),
        _Collector(target_cap),
    )
    ring_with_data = 0
    for zp in ctx.plan.zones:
        roles = state.roles[zp.zone]
        for blk in io.iter_plan_blocks(ctx.reader, zp, [year], layers=LAYERS, block=ctx.block):
            win = blk.windows[year]
            r = slice(blk.rows.start - zp.rows.start, blk.rows.stop - zp.rows.start)
            c = slice(blk.cols.start - zp.cols.start, blk.cols.stop - zp.cols.start)
            ok = _valid(win)
            control = roles.control[r, c]
            ring_with_data += int((ok & control).sum())
            site.add(zindex[zp.zone], win, roles.site[r, c], ok)
            ring.add(zindex[zp.zone], win, control if supplied else ok & control, ok)
    if state.target_plan is not None:
        counts["target_pixels"] = state.target_plan.n_pixels
        for zp in state.target_plan.zones:
            for blk in io.iter_plan_blocks(ctx.reader, zp, [year], layers=LAYERS, block=ctx.block):
                win = blk.windows[year]
                target.add(zindex[zp.zone], win, blk.mask, _valid(win))
    # The target's zones join the provenance (and its pipeline commits) when the extent does not touch them.
    ctx.mark_read([year], zones=state.zones)
    site_pix, site_ok = site.done(ROLE_SITE)
    ring_pix, ring_ok = ring.done(ROLE_CONTROL)
    target_pix, target_ok = target.done(ROLE_TARGET)
    return _Stage1(site_pix, site_ok, ring_pix, ring_ok, target_pix, target_ok, counts, ring_with_data)


def _stage2(ctx: RunContext, state: _State, pool: _Pixels, years: Sequence[int]) -> int:
    """Read every pool year but the first (stage 1's) for the chunks holding a pool pixel, filling the pool.

    Args:
        ctx: The run.
        state: The run state (zones).
        pool: The pool; year index 0 already holds stage 1.
        years: The pool's years, stage 1's first.

    Returns:
        Windows read.
    """
    other = list(years[1:])
    if not other or pool.n == 0:
        return 0
    shapes = {i: ctx.reader.chunk_shape(z) for i, z in enumerate(state.zones)}
    ch0, cw0 = next(iter(shapes.values()))
    max_run = max(1, (ctx.block[0] * ctx.block[1] * 2) // (ch0 * cw0 * len(other)))
    runs = chunk_runs(pool.zone, pool.row, pool.col, shapes, max_run)
    index = {y: list(years).index(y) for y in other}
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


def _xy_in_zone(state: _State, reader: Reader, pix: _Where, zone: str) -> np.ndarray:
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


def _site_cells(state: _State, reader: Reader, pix: _Where, cell_m: float) -> np.ndarray:
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


def _sums(pix: _Pixels, mask: np.ndarray | None = None) -> np.ndarray:
    """(Y, D) pooled dequantized sums per year, of every pixel or of the ``mask`` ones (no copy is made)."""
    out = []
    for t in range(pix.emb.shape[1]):
        weight = pix.scale[:, t] if mask is None else np.where(mask, pix.scale[:, t], np.float32(0.0))
        out.append(pooled_sum(pix.emb[:, t], weight))
    return np.stack(out)


def _cosines(pix: _Pixels, refs: np.ndarray) -> np.ndarray:
    """(n, Y) float32 cosine of every pixel's int8 vector to that year's reference vector (exact on int8)."""
    out = np.empty((pix.n, refs.shape[0]), dtype=np.float32)
    if pix.n:
        for k in range(refs.shape[0]):
            out[:, k] = metrics.cosine_similarity(pix.emb[:, k], np.asarray(refs[k], dtype=np.float64))
    return out


def _divergence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Y,) ``1 - cos`` of two (Y, D) sum series."""
    return np.array([1.0 - cosine(x, y) for x, y in zip(a, b, strict=True)], dtype=np.float64)


def _ratio(num: float, den: float) -> float | None:
    if den == 0 or not (math.isfinite(num) and math.isfinite(den)):
        return None
    return float(num / den)


def _by_year(years: Sequence[int], values: np.ndarray) -> dict[str, Any]:
    return dict(zip([str(y) for y in years], np.asarray(values, dtype=np.float64).tolist(), strict=True))


def _no_control_hint(supplied: bool, match: int, planted: int, year_a: int) -> str:
    """What to change when stage 1 keeps too few control pixels (lead decision: name a cleared site)."""
    if supplied:
        return "pass a larger control polygon"
    hint = "widen the ring (--ring-outer-km), lower --control-pct, or pass a --control polygon"
    if match > FIRST_YEAR:
        when = " (two years before planting)" if match == planted - 2 else ""
        fix = (
            f"pass --year-a {match}{when} and leave --match-year at its default, so the control is matched on "
            f"{match - 1}"
            if match < year_a
            else f"pass --match-year {match - 1}, so the control is matched on the year before the baseline"
        )
        return (
            f"{hint}. A site prepared or cleared for planting in {match}, the year the control is matched on, is bare "
            f"ground there that no ring pixel resembles, so it cannot be matched: {fix}, before the site was touched"
        )
    return (
        f"{hint}. A site prepared or cleared for planting in {match} cannot be matched on that year, and the store has "
        "no earlier one: pass a --control polygon"
    )


def _match_year_hint(year_a: int) -> str:
    """Which ``--match-year`` values a run with this baseline could use."""
    if year_a == FIRST_YEAR:
        return f"There is no other year to match on: {FIRST_YEAR} is the store's first; a later --year-a gives one."
    return (
        f"Pass --match-year with a year of fuller data ({FIRST_YEAR} to {year_a}; the default is "
        f"{match_year_for(year_a)})."
    )


def _coverage(total: int, with_data: int) -> dict[str, Any]:
    """How many of a role's pixels have match-year data."""
    return {
        "pixels": int(total),
        "with_data": int(with_data),
        "without_data": int(total) - int(with_data),
        "share_with_data": float(with_data) / float(total) if total else None,
    }


def _year_sum(pix: _Pixels, t: int, mask: np.ndarray) -> np.ndarray:
    """(D,) pooled dequantized sum of year index ``t`` over the ``mask`` pixels only (the rest weigh 0)."""
    return pooled_sum(pix.emb[:, t], np.where(mask, pix.scale[:, t], np.float32(0.0)))


def _divergence_or_none(a: np.ndarray, b: np.ndarray) -> float | None:
    """``1 - cos(a, b)``, or None when either sum is zero (no pixel behind it)."""
    c = cosine(a, b)
    return None if not math.isfinite(c) else float(1.0 - c)


def _run(ctx: RunContext, state: _State, opts: _Options) -> RestoreResult:
    years = list(opts.years)
    n_years = len(years)
    year_a, year_b, planted = ctx.year_a, ctx.year_b, opts.planted
    match = opts.match_year
    separate = match != year_a
    pool_years = [match, *years] if separate else years
    off = len(pool_years) - n_years
    supplied = opts.control is not None
    control_word = "control" if supplied else "ring"
    if year_a == FIRST_YEAR:
        shared = (
            ""
            if supplied
            else " There is no earlier year to match the control on, so the control filter and the baseline share "
            "one year: ring pixels that pass the filter by the luck of that year's noise drift back afterwards "
            "(regression to the mean), which can read as divergence of a site that did nothing. Prefer a --control "
            "polygon chosen without the embeddings, or a later planting."
        )
        ctx.warn(
            f"The baseline year is {FIRST_YEAR}, the store's first: it has few Sentinel-2 observations and sparse "
            "valid pixels in places (survey S16), so the control match and the divergence change rest on a weak "
            f"year.{shared}"
        )
    elif not separate and not supplied:
        ctx.warn(
            f"The control is matched on the baseline year itself (--match-year {match}), so the control filter and "
            "the baseline share one year: ring pixels that pass the filter by the luck of that year's noise drift back "
            "afterwards (regression to the mean), which can read as divergence of a site that did nothing. Prefer the "
            f"default, {match_year_for(year_a)} (the year before the baseline), or a --control polygon chosen without "
            "the embeddings."
        )
    if separate and match == FIRST_YEAR:
        which = f"the year before the baseline {year_a}"
        if match != year_a - 1:
            which = f"--match-year; the baseline is {year_a}"
        ctx.warn(
            f"The control is matched on {FIRST_YEAR}, the store's first year ({which}): it has few Sentinel-2 "
            "observations and sparse valid pixels in places (survey S16), so the control match rests on a weak year."
        )

    # Stage 1: the match year, for the whole extent and the target.
    t = time.perf_counter()
    st1 = _stage1(ctx, state, match, supplied=supplied)
    t = ctx.lap("stage1", t)
    counts = st1.counts
    site1, ring1, target1 = st1.site, st1.ring, st1.target
    n_site = int(st1.site_ok.sum())
    match_words = f"{match}, the year the control is matched on" if separate else f"{match}, the baseline year"
    if n_site < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {n_site} of the site's {counts['site_pixels']} pixels have {match} data; the app needs at least "
            f"{MIN_SITE_PIXELS} valid site pixels in {match_words}, for the site mean the control is matched to. "
            f"{_match_year_hint(year_a)} Or check the site lies on land with data."
        )
    coverage: dict[str, Any] = {
        "year": match,
        "site": _coverage(counts["site_pixels"], n_site),
        control_word: _coverage(counts["control_pixels"], st1.ring_with_data),
        "target": None if state.target_plan is None else _coverage(target1.n, int(st1.target_ok.sum())),
        "note": MATCH_DATA_NOTE,
    }
    logger.info(
        "%s: %d data in %d of %d site, %d of %d %s%s pixels.",
        APP,
        match,
        n_site,
        counts["site_pixels"],
        st1.ring_with_data,
        counts["control_pixels"],
        control_word,
        "" if state.target_plan is None else f" and {int(st1.target_ok.sum())} of {target1.n} target",
    )
    site_share = coverage["site"]["share_with_data"]
    if site_share < MIN_MATCH_SHARE:
        later = (
            f"the trajectories still use every site pixel with data in each year from {year_a} to {year_b}"
            if separate
            else f"a site pixel without {match} data has no baseline, so the panel rule drops it as well"
        )
        ctx.warn(
            f"Only {site_share:.0%} of the site's pixels ({n_site:,} of {counts['site_pixels']:,}) have {match} data, "
            f"the year the control is matched on: the site's {match} mean, which sets the control filter, rests on "
            f"them alone, so the control is matched to that part of the site; {later}. "
            f"{_match_year_hint(year_a)}"
        )
    ring_share = coverage[control_word]["share_with_data"]
    if ring_share is not None and ring_share < MIN_MATCH_SHARE:
        if supplied:
            ctx.warn(
                f"Only {ring_share:.0%} of the supplied control's pixels ({st1.ring_with_data:,} of "
                f"{counts['control_pixels']:,}) have {match} data, the year the control is matched on: its balance "
                "against the site (share_below_threshold and the poorly matched check) rests on them alone. The "
                f"control itself is used as given: every pixel with data in each year from {year_a} to {year_b}."
            )
        else:
            ctx.warn(
                f"Only {ring_share:.0%} of the ring's candidate pixels ({st1.ring_with_data:,} of "
                f"{counts['control_pixels']:,}) have {match} data, the year the control is matched on: a ring pixel "
                f"without it cannot be filtered, so the control is drawn from that {ring_share:.0%} alone and may "
                f"gather where {match} had clear views. {_match_year_hint(year_a)}"
            )

    # The site's match-year mean and the held-out threshold, over the site pixels with match-year data.
    site_m = site1 if bool(st1.site_ok.all()) else site1.take(st1.site_ok)
    sum_s = pooled_sum(site_m.emb[:, 0], site_m.scale[:, 0])
    site_sim = pixel_cosine(site_m.emb[:, 0], sum_s)
    groups = _site_cells(state, ctx.reader, site_m.where(), opts.fold_cell_m)
    fold, n_folds = assign_folds(groups, opts.folds, ctx.seed)
    held = heldout_similarity(site_m.emb[:, 0], site_m.scale[:, 0], fold) if n_folds >= 2 else np.full(n_site, np.nan)
    del site_m
    site_sim_all = np.full(site1.n, np.nan, dtype=np.float64)
    site_sim_all[st1.site_ok] = site_sim
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
            f"The control threshold is the in-sample {opts.control_pct:g}th percentile of the site's {match} "
            f"similarities, not a held-out one: {reason}. In-sample thresholds are too tight (survey S17), so the "
            "filter may reject comparable land; pass a smaller --fold-cell-m for a small site."
        )
    # A pixel without match-year data has a zero vector there, so its similarity is NaN and it never passes.
    ring_sim = pixel_cosine(ring1.emb[:, 0], sum_s)
    with np.errstate(invalid="ignore"):
        passes = (1.0 - ring_sim) <= choice.threshold
    kept = np.ones(ring1.n, dtype=bool) if supplied else passes
    n_kept = int(kept.sum())
    below_share = float(np.mean(~passes[st1.ring_ok])) if st1.ring_ok.any() else math.nan
    if supplied and st1.ring_ok.any() and below_share > 0.5:
        ctx.warn(
            f"The supplied control is poorly matched: {below_share:.0%} of its pixels with {match} data are less "
            f"similar to the site's {match} mean than the held-out threshold ({threshold:.4f}). Divergence from it may "
            "be a difference that predates the planting; consider the ring (omit --control)."
        )
    if n_kept < MIN_CONTROL_PIXELS:
        sparse = (
            f" Only {ring_share:.0%} of the ring has {match} data: {_match_year_hint(year_a)}"
            if not supplied and ring_share is not None and ring_share < MIN_MATCH_SHARE
            else ""
        )
        raise RestoreError(
            f"Only {n_kept} {control_word} pixels are kept ({st1.ring_with_data} have {match} data, of "
            f"{counts['control_pixels']} in the {control_word}); the app needs at least {MIN_CONTROL_PIXELS} control "
            f"pixels: {_no_control_hint(supplied, match, planted, year_a)}.{sparse}"
        )
    target_base = pixel_cosine(target1.emb[:, 0], sum_s)
    ring_with_data = st1.ring_with_data
    t = ctx.lap("control_filter", t)

    # Stage 2: the other years, only for chunks with a kept pixel. The stage 1 vectors go once copied
    # into the pool; their positions stay for the COG.
    pool = _build_pool([(site1, None), (ring1, kept), (target1, None)], len(pool_years))
    site1_w, ring1_w, target1_w = site1.where(), ring1.where(), target1.where()
    del site1, ring1, target1, st1
    windows_read = _stage2(ctx, state, pool, pool_years)
    t = ctx.lap("stage2", t)

    valid = _valid_years(pool)
    is_site = pool.role == ROLE_SITE
    traj_radar = pool.radar[:, off:]
    radar_free = []
    for k in range(n_years):
        v = valid[is_site, off + k]
        radar_free.append(float(np.mean(~traj_radar[is_site, k][v])) if v.any() else math.nan)

    # Panel and radar regime (survey B5), over the trajectory years only: the match year serves the filter alone.
    panel = valid[:, off:].all(axis=1)
    match_ok = valid[:, 0].copy() if separate else None
    del valid
    if not (panel & is_site).any():
        raise RestoreError(
            f"No site pixel has data in every year from {year_a} to {year_b}; choose a shorter span "
            "(--year-a, --year-b)."
        )
    pattern = majority_pattern(traj_radar[panel & is_site])
    regime_ok = np.all(traj_radar == pattern[None, :], axis=1)
    use = panel & regime_ok
    dropped: dict[str, dict[str, int]] = {}
    for code, name in ((ROLE_SITE, "site"), (ROLE_CONTROL, "control"), (ROLE_TARGET, "target")):
        r = pool.role == code
        dropped[name] = {
            "read": int(r.sum()),
            "panel": int((r & ~panel).sum()),
            "regime": int((r & panel & ~use).sum()),
            "used": int((r & use).sum()),
        }
    del panel, regime_ok, is_site
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
                f"in some year from {year_a} to {year_b} (the panel rule) and {d['regime']:,} have another "
                "radar pattern than the site's majority (survey B5)."
            )
    n_s, n_c, n_t = (dropped[name]["used"] for name in ("site", "control", "target"))
    if n_s < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {n_s} site pixels have data in every year from {year_a} to {year_b} with the majority radar "
            f"pattern; the app needs at least {MIN_SITE_PIXELS}. Choose a shorter span (--year-a, --year-b)."
        )
    if n_c < MIN_CONTROL_PIXELS:
        raise RestoreError(
            f"Only {n_c} control pixels have data in every year from {year_a} to {year_b} with the site's "
            f"radar pattern; the app needs at least {MIN_CONTROL_PIXELS}. Widen the ring or pass a larger control."
        )
    if state.target_plan is not None and n_t < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Only {n_t} target pixels have data in every year from {year_a} to {year_b} with the site's "
            f"radar pattern; the app needs at least {MIN_SITE_PIXELS}. Pass a larger mature stand."
        )
    # Match-year data of the used pixels, in the order the compaction keeps.
    match_used = None if match_ok is None else match_ok[use]
    del match_ok
    # The used pixels, moved to the front of the pool in place (roles stay in order): no second copy.
    pool = _compact(pool, use)
    del use, traj_radar
    site_all, ctrl_all, tgt_all = pool.rows(0, n_s), pool.rows(n_s, n_s + n_c), pool.rows(n_s + n_c, pool.n)
    site, ctrl, tgt = site_all.from_year(off), ctrl_all.from_year(off), tgt_all.from_year(off)
    t = ctx.lap("panel", t)

    # Trajectories.
    s_site = _sums(site)
    s_ctrl = _sums(ctrl)
    div = _divergence(s_site, s_ctrl)
    change = div - div[0]
    # The selection drift, over the used pixels with match-year data in both years, so a sparse match year
    # does not mix a change of pixels into it.
    match_div: float | None = None
    base_div_match: float | None = None
    drift_pixels: dict[str, int] | None = None
    if match_used is not None:
        sm, cm = match_used[:n_s], match_used[n_s : n_s + n_c]
        drift_pixels = {"site": int(sm.sum()), "control": int(cm.sum())}
        match_div = _divergence_or_none(_year_sum(site_all, 0, sm), _year_sum(ctrl_all, 0, cm))
        if bool(sm.all()) and bool(cm.all()):
            base_div_match = float(div[0])
        else:
            base_div_match = _divergence_or_none(_year_sum(site_all, off, sm), _year_sum(ctrl_all, off, cm))
    selection_drift = None if match_div is None or base_div_match is None else float(base_div_match - match_div)
    n_cap = min(site.n, ctrl.n // 3)
    if n_cap < MIN_SITE_PIXELS:
        raise RestoreError(
            f"Placebos would hold at most {n_cap} pixels (a third of the {ctrl.n} control pixels); the app needs at "
            f"least {MIN_SITE_PIXELS}. Widen the ring or pass a larger control."
        )
    main = opts.zone
    radius = math.sqrt(counts["site_pixels"] * io.PIXEL_M**2 / math.pi)
    xy_ctrl = _xy_in_zone(state, ctx.reader, ctrl.where(), main)
    try:
        plc = draw_placebos(xy_ctrl, opts.placebos, n_cap, ctx.seed, radius_m=radius, min_pixels=MIN_SITE_PIXELS)
    except ValueError as exc:
        raise RestoreError(
            f"No control pixel has {MIN_SITE_PIXELS} or more kept control pixels within {radius:.0f} m (the site's "
            f"equivalent radius), so no placebo of the site's footprint can be drawn: the kept control is too "
            "sparse. Widen the ring, lower --control-pct, or pass a --control polygon."
        ) from exc
    p_sizes = plc.sizes
    p_median = float(np.median(p_sizes))
    pixel_ratio = p_median / site.n
    footprint_ratio = float(np.median((plc.reach_m / radius) ** 2)) if radius > 0 else math.nan
    distinct = int(np.unique(plc.centres).size)
    if distinct < opts.placebos:
        ctx.warn(
            f"Only {distinct} distinct placebo centres were found for {opts.placebos} placebos (a centre needs "
            f"{MIN_SITE_PIXELS} kept control pixels within {radius:.0f} m), so some placebos repeat and the band rests "
            "on fewer independent pieces of land. Widen the ring or pass a larger control."
        )
    if n_cap < site.n or pixel_ratio < PLACEBO_PIXEL_RATIO_WARN:
        why = (
            f"at most {n_cap}, a third of the control pool"
            if n_cap < site.n
            else f"the kept control is sparse within {radius:.0f} m of their centres"
        )
        ctx.warn(
            f"Placebos hold a median of {p_median:.0f} pixels ({why}), fewer than the site's {site.n}. A smaller "
            "placebo has a larger sampling term in 1 - cos, so a change in per-pixel spread between years moves the "
            "band rather than only widening it; a year counts only when the site's change is above both the band "
            "and zero, so a falling divergence is never called diverging. Widen the ring or pass a larger control "
            "for placebos of the site's size."
        )
    p_div = placebo_divergence(ctrl.emb, ctrl.scale, plc.members)
    p_change = p_div - p_div[:, :1]
    band = placebo_band(p_change)
    p05, p50, p95 = band
    verdict = call_verdict(years, change, p95, planted=planted, min_post_years=opts.min_post_years)
    p_texts, p_diverging = placebo_verdicts(years, p_change, p95, planted=planted, min_post_years=opts.min_post_years)
    fp_rate = false_positive_rate(p_diverging, verdict.verdict == VERDICT_INSUFFICIENT)
    post_mask = np.array([y >= planted for y in years], dtype=bool)
    significant = exceeds_band(change, p95)
    trend = theil_sen_trend([y for y in years if y >= planted], change[post_mask])
    baseline_rank = float(np.mean(p_div[:, 0] < div[0]))
    if baseline_rank >= BASELINE_RANK_WARN:
        moved = (
            f"; it moved {selection_drift:+.4f} further from the control between {opts.match_year} and {year_a}"
            if selection_drift is not None and selection_drift > 0
            else ""
        )
        ctx.warn(
            f"The site already differs from its control at the baseline {year_a}: its baseline divergence is above "
            f"{baseline_rank:.0%} of the placebos'{moved}. Change that began before {year_a} is left out of the "
            f"divergence change, which then understates it; if the site was prepared, planted or began regrowing "
            f"before {year_a}, pass an earlier --year-a (and --match-year)."
        )
    t = ctx.lap("placebos", t)

    # The null test: two halves of the control pool, sectors around the site for the ring, the two
    # sides of the long axis for a supplied control (which may lie in one sector).
    if supplied:
        halves = axis_halves(xy_ctrl, split="median")
        null_method = (
            "The supplied control split at its median along its long axis into two compact halves (main zone "
            "metres), compared exactly as the site is, against the same placebo band."
        )
    else:
        centre = to_zone_metres(opts.site.geometry, main).centroid
        halves = null_halves(xy_ctrl, (float(centre.x), float(centre.y)))
        null_method = (
            f"The control ring split into two halves by {NULL_SECTORS} alternating angular sectors around the site "
            "centroid (main zone metres), compared exactly as the site is, against the same placebo band."
        )
    del xy_ctrl
    n_half = (int((halves == 0).sum()), int((halves == 1).sum()))
    null: dict[str, Any]
    null_change = np.full(n_years, np.nan)
    if min(n_half) >= MIN_SITE_PIXELS:
        null_div = _divergence(_sums(ctrl, halves == 0), _sums(ctrl, halves == 1))
        null_change = null_div - null_div[0]
        nv = call_verdict(years, null_change, p95, planted=planted, min_post_years=opts.min_post_years)
        post_null = null_change[post_mask]
        post_site = change[post_mask]
        null_max = float(np.nanmax(np.abs(post_null))) if post_null.size else math.nan
        site_max = float(np.nanmax(np.abs(post_site))) if post_site.size else math.nan
        null_sig = [y for y, sig in zip(years, exceeds_band(null_change, p95), strict=True) if y >= planted and sig]
        null = {
            "null_verdict": nv.verdict,
            "null_near_zero": not null_sig,
            "null_to_site_ratio": _ratio(null_max, site_max),
            "half_to_site_pixels": _ratio(float(min(n_half)), float(site.n)),
            "caveat": null_caveat(_ratio(float(min(n_half)), float(site.n))),
            "method": null_method,
            "split": "long_axis" if supplied else "sectors",
            "sectors": None if supplied else NULL_SECTORS,
            "half_pixels": list(n_half),
            "site_pixels": site.n,
            "divergence": _by_year(years, null_div),
            "divergence_change": _by_year(years, null_change),
            "null_verdict_detail": nv.detail,
            "significant_years": null_sig,
            "null_max_abs_change": null_max,
            "site_max_abs_change": site_max,
            "site_max_change": float(np.nanmax(post_site)) if post_site.size else math.nan,
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
        if ctx.validate and opts.strict_validate:
            raise StrictValidationError(f"--strict-validate is set and the null test could not run: {null['error']}.")
        ctx.warn(f"The null test could not run: {null['error']}.")

    # The target trajectory.
    target_rec: dict[str, Any] | None = None
    s_t, c_t, tau, progress = (np.full(n_years, np.nan) for _ in range(4))
    sum_t: np.ndarray | None = None
    if tgt.n:
        sum_t = _sums(tgt)
        s_t = np.array([cosine(a, b) for a, b in zip(s_site, sum_t, strict=True)])
        c_t = np.array([cosine(a, b) for a, b in zip(s_ctrl, sum_t, strict=True)])
        thalves = axis_halves(_xy_in_zone(state, ctx.reader, tgt.where(), main), split="centroid")
        sizes = [int((thalves == 0).sum()), int((thalves == 1).sum())]
        split_ok = min(sizes) >= MIN_TARGET_HALF_PIXELS
        if split_ok:
            h0, h1 = _sums(tgt, thalves == 0), _sums(tgt, thalves == 1)
            tau = np.array([cosine(a, b) for a, b in zip(h0, h1, strict=True)])
            progress = target_progress(s_t, c_t, tau)
            if tgt.n < MIN_TARGET_UNBIASED_PIXELS:
                ctx.warn(
                    f"The target holds {tgt.n} used pixels, under {MIN_TARGET_UNBIASED_PIXELS}: each half's mean is "
                    "noisy, so the 100 percent level (the cosine of the two halves) is biased low and the progress "
                    "overstates; a site of the target's own kind can read well above 1. Pass a larger mature stand."
                )
        else:
            ctx.warn(
                "The target cannot be split into two spatial halves for the 100 percent level (one half would hold "
                f"{min(sizes)} pixels, under {MIN_TARGET_HALF_PIXELS}): only the raw similarity to the target is "
                "reported, not the progress. Pass a larger, more compact target."
            )
        target_rec = {
            "pixels": tgt.n,
            "halves": sizes if split_ok else None,
            "split": "two sides of a line through the target's centroid across its long axis",
            "small_target": bool(split_ok and tgt.n < MIN_TARGET_UNBIASED_PIXELS),
            "progress_note": PROGRESS_NOTE,
            "similarity": _by_year(years, s_t),
            "control_similarity": _by_year(years, c_t),
            "halves_similarity": _by_year(years, tau),
            "progress": _by_year(years, progress),
        }

    # Per pixel.
    site_to_c = _cosines(site, s_ctrl)
    ctrl_to_c = _cosines(ctrl, s_ctrl)
    departed = np.array([departed_fraction(site_to_c[:, k], ctrl_to_c[:, k]) for k in range(n_years)])
    departed_change = departed - departed[0]
    closer = np.full(n_years, np.nan)
    ctrl_closer = np.full(n_years, np.nan)
    site_closer_b: np.ndarray | None = None
    if sum_t is not None:
        site_to_t = _cosines(site, sum_t)
        closer = np.array([closer_fraction(site_to_t[:, k], site_to_c[:, k]) for k in range(n_years)])
        site_closer_b = site_to_t[:, -1] > site_to_c[:, -1]
        del site_to_t
        ctrl_to_t = _cosines(ctrl, sum_t)
        ctrl_closer = np.array([closer_fraction(ctrl_to_t[:, k], ctrl_to_c[:, k]) for k in range(n_years)])
        del ctrl_to_t
    t = ctx.lap("trajectories", t)

    # The vectors are no longer needed: keep the positions only, so the write holds no pool.
    site_w, ctrl_w, tgt_w = site.where(), ctrl.where(), tgt.where()
    n_site_used, n_ctrl_used, n_tgt_used = site.n, ctrl.n, tgt.n
    del pool, site_all, ctrl_all, tgt_all, site, ctrl, tgt

    # The canopy check.
    canopy: dict[str, Any] | None = None
    canopy_heights: dict[str, dict[int, np.ndarray]] | None = None
    if ctx.validate and opts.canopy:
        if state.canopy is None:
            canopy = {"error": state.canopy_error, "error_kind": state.canopy_error_kind, "years": state.canopy_years}
            ctx.warn(f"{CANOPY_SKIPPED}: {state.canopy_error}")
        else:
            canopy_heights = {
                "site": _heights(state, "extent", site_w),
                "control": _heights(state, "extent", ctrl_w),
            }
            if n_tgt_used:
                canopy_heights["target"] = _heights(state, "target", tgt_w)
            canopy = {
                **canopy_check(years, change, canopy_heights),
                "dataset": "GLAD annual tree canopy height (Potapov et al.)",
                "url_template": opts.tch_template,
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
            "departed_fraction_change": departed_change[k],
            "closer_to_target_fraction": closer[k],
            "control_closer_to_target_fraction": ctrl_closer[k],
            "target_similarity": s_t[k],
            "control_target_similarity": c_t[k],
            "target_halves_similarity": tau[k],
            "target_progress": progress[k],
            "site_radar_free_fraction": radar_free[k],
            "n_site": n_site_used,
            "n_control": n_ctrl_used,
            "n_target": n_tgt_used,
            "verdict": verdict.verdict,
        }
        if canopy_heights is not None and canopy is not None:
            per = canopy["per_year"].get(str(y), {})
            row["site_tch_mean"] = per.get("site_mean_m", math.nan)
            row["control_tch_mean"] = per.get("control_mean_m", math.nan)
            row["site_tall_share"] = per.get("site_tall_share", math.nan)
            row["control_tall_share"] = per.get("control_tall_share", math.nan)
            if n_tgt_used:
                row["target_tch_mean"] = per.get("target_mean_m", math.nan)
                row["target_tall_share"] = per.get("target_tall_share", math.nan)
        rows.append(row)
    table = pd.DataFrame(rows)
    for col in table.columns:
        if col not in ("year", "post_planting", "significant", "n_site", "n_control", "n_target", "verdict"):
            table[col] = table[col].astype(np.float64)
    export.write_table(table, ctx.paths.summary_parquet.with_suffix(""))
    centres = _Where(ctrl_w.zone[plc.centres], ctrl_w.row[plc.centres], ctrl_w.col[plc.centres])
    _write_placebos(
        ctx.extra_outputs["placebos"],
        years,
        p_div,
        p_change,
        p95,
        p_texts,
        plc,
        _xy_in_zone(state, ctx.reader, centres, main),
    )

    band_names, mosaic = _write_tif(
        ctx,
        state,
        years=years,
        site1=site1_w,
        ring1=ring1_w,
        ring_sim=ring_sim,
        kept=kept,
        site=site_w,
        ctrl=ctrl_w,
        target1=target1_w,
        target_base=target_base,
        site_sim=site_sim_all,
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
        departed_change=departed_change,
        closer=closer,
        ctrl_closer=ctrl_closer,
        canopy=canopy if canopy_heights is not None else None,
        site_sim=site_sim,
        ring_sim=ring_sim[np.isfinite(ring_sim)],
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
        "match_year": match,
        "candidate_pixels": counts["control_pixels"],
        "candidate_pixels_valid_match_year": ring_with_data,
        "kept_pixels": n_kept,
        "rejected_pixels": 0 if supplied else int((~kept).sum()),
        "filtered": not supplied,
        "threshold_similarity": threshold,
        "threshold_source": choice.method,
        "threshold": choice.record(),
        "threshold_note": (
            "threshold_similarity is the control_pct percentile of the site's match-year cosine similarities to its "
            "own mean (held out by spatial fold unless the fallback ran); the threshold block gives the same cut as a "
            "distance (1 - similarity) at the 100 - control_pct percentile, as wetland.choose_threshold works."
        ),
        "control_pct": opts.control_pct,
        "folds_requested": opts.folds,
        "folds": n_folds,
        "fold_cell_m": opts.fold_cell_m,
        "site_cells": int(np.unique(groups).size),
        "heldout_pixels": int(np.isfinite(held).sum()),
        "share_below_threshold": below_share,
        "match_year_divergence": match_div,
        "baseline_divergence": div[0],
        "baseline_divergence_match_pixels": base_div_match,
        "selection_drift": selection_drift,
        "selection_drift_pixels": drift_pixels,
        "match_note": MATCH_NOTE,
        "balance": {
            "baseline_divergence": div[0],
            "placebo_baseline_divergence_p50": float(np.percentile(p_div[:, 0], 50)),
            "placebo_baseline_divergence_p95": float(np.percentile(p_div[:, 0], 95)),
            "baseline_divergence_rank": baseline_rank,
            "note": RANK_NOTE,
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
        "false_positive_rate": fp_rate,
        "false_positive_rate_note": FPR_NOTE,
        "diverging_placebos": int(p_diverging.sum()),
        "radius_m": radius,
        "pixels_cap": n_cap,
        "pixels_min": int(p_sizes.min()),
        "pixels_median": p_median,
        "pixels_max": int(p_sizes.max()),
        "pixel_ratio_median": pixel_ratio,
        "footprint_ratio_median": footprint_ratio,
        "distinct_centres": distinct,
        "centres_tried": plc.tried,
        "band_percentiles": list(BAND_PERCENTILES),
        "significance": "change above both the band's 95th percentile and zero",
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
        "match_year": match,
        "years": years,
        "post_planting_years": verdict.post_years,
        "site": {
            "name": opts.site.name,
            "source": opts.site.source,
            "area_ha": geodesic_area_ha(opts.site.geometry),
            "pixels": counts["site_pixels"],
            "pixels_valid_match_year": n_site,
            "pixels_used": n_site_used,
            "main_zone": main,
        },
        "extent": {
            "what": "site plus " + ("supplied control" if supplied else "control ring"),
            "area_ha": ctx.plan.area_ha,
            "pixels": ctx.plan.n_pixels,
            "zones": ctx.plan.zone_names,
        },
        "zones": state.zones,
        "match_year_data": coverage,
        "control": control_rec,
        "regime": regime,
        "placebos": placebo_rec,
        "placebo_false_positive_rate": fp_rate,
        "placebo_false_positive_rate_note": FPR_NOTE,
        "departed_fraction_change": _by_year(years, departed_change),
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
        _write_validation(ctx, null, canopy, placebo_rec)
        validation_written = True
        summary["validation"] = {
            "placebo_false_positive_rate": fp_rate,
            "null_verdict": null.get("null_verdict"),
            "null_near_zero": null.get("null_near_zero"),
            "null_to_site_ratio": null.get("null_to_site_ratio"),
            "null_error": null.get("error"),
            "canopy_onset_year": None if canopy is None else canopy.get("canopy_onset_year"),
            "canopy_correlation_sign": None if canopy is None else canopy.get("correlation_sign"),
            "canopy_error": None if canopy is None else canopy.get("error"),
        }
    ctx.lap("validation", t)
    aoi_record = {
        "area_ha": geodesic_area_ha(opts.site.geometry),
        "pixels": counts["site_pixels"],
        "zones": io.zones_for_geometry(opts.site.geometry),
        "extent_area_ha": ctx.plan.area_ha,
        "extent_pixels": ctx.plan.n_pixels,
        "extent_zones": ctx.plan.zone_names,
    }
    report = finish_run(
        ctx, summary=summary, extra=ctx.extra_outputs, validation_written=validation_written, aoi_record=aoi_record
    )
    return RestoreResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=validation_written,
    )


def _heights(state: _State, kind: str, pix: _Where) -> dict[int, np.ndarray]:
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


def _scatter(state: _State, per_zone: dict[str, np.ndarray], pix: _Where, values: np.ndarray) -> None:
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
    site1: _Where,
    ring1: _Where,
    ring_sim: np.ndarray,
    kept: np.ndarray,
    site: _Where,
    ctrl: _Where,
    target1: _Where,
    target_base: np.ndarray,
    site_sim: np.ndarray,
    site_to_c: np.ndarray,
    ctrl_to_c: np.ndarray,
    site_closer_b: np.ndarray | None,
) -> tuple[list[str], np.ndarray]:
    """Write the COG; returns the band names and the mosaic (for the preview).

    ``baseline_similarity`` is the cosine to the site's mean in the match year, the similarity the
    control filter used; NaN on a pixel without match-year data.
    """
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
    rejected = ~kept
    _scatter(
        state,
        role,
        _Where(ring1.zone[rejected], ring1.row[rejected], ring1.col[rejected]),
        np.full(int(rejected.sum()), ROLE_REJECTED, dtype=np.float32),
    )
    _scatter(state, role, site, np.full(site.n, ROLE_SITE, dtype=np.float32))
    _scatter(state, role, ctrl, np.full(ctrl.n, ROLE_CONTROL, dtype=np.float32))
    mosaic.put(0, role)
    base = blank()
    _scatter(state, base, site1, site_sim.astype(np.float32))
    _scatter(state, base, ring1, ring_sim.astype(np.float32))
    if target1.n:
        _scatter(state, base, target1, target_base.astype(np.float32))
    for zp in plan.zones:
        base[zp.zone][~zp.mask] = nan
    mosaic.put(1, base)
    for k in range(len(years)):
        band = blank()
        _scatter(state, band, site, site_to_c[:, k])
        _scatter(state, band, ctrl, ctrl_to_c[:, k])
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
    placebos: Placebos,
    centres_xy: np.ndarray,
) -> Path:
    """Write ``restore_placebos.parquet``: per placebo and year its divergence, change, significance and verdict."""
    p, n_years = change.shape
    frame = pd.DataFrame(
        {
            "placebo": np.repeat(np.arange(p, dtype=np.int64), n_years),
            "year": np.tile(np.asarray(years, dtype=np.int64), p),
            "divergence": div.reshape(-1),
            "divergence_change": change.reshape(-1),
            "significant": exceeds_band(change, p95[None, :]).reshape(-1),
            "verdict": np.repeat(np.asarray(verdicts, dtype=object), n_years),
            "centre_x": np.repeat(centres_xy[:, 0], n_years),
            "centre_y": np.repeat(centres_xy[:, 1], n_years),
            "pixels": np.repeat(placebos.sizes, n_years),
            "reach_m": np.repeat(placebos.reach_m, n_years),
        }
    )
    with export.atomic_path(path) as tmp:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
    return path


def _write_validation(
    ctx: RunContext, null: Mapping[str, Any], canopy: Mapping[str, Any] | None, placebos: Mapping[str, Any]
) -> None:
    """Write ``restore_validation.json``: the site-sized null first, then the control halves and the canopy check."""
    caveats = [LIMITATION, FPR_NOTE, null.get("caveat") or NULL_CAVEAT]
    if null.get("error"):
        caveats.append(f"The null test could not run: {null['error']}.")
    if canopy is not None:
        if canopy.get("error"):
            caveats.append(f"{CANOPY_SKIPPED}: {canopy['error']}.")
        else:
            caveats += list(CANOPY_CAVEATS)
    site_sized = {
        "placebo_false_positive_rate": placebos.get("false_positive_rate"),
        "diverging_placebos": placebos.get("diverging_placebos"),
        "placebos": placebos.get("count"),
        "pixels_median": placebos.get("pixels_median"),
        "radius_m": placebos.get("radius_m"),
        "note": FPR_NOTE,
    }
    reference: dict[str, Any] = {
        "site_sized_null": "site-sized pieces of the control (the placebos) through the same verdict rule",
        "null_test": (
            "the control pool against itself, split into two halves (alternating angular sectors around the site for "
            "the ring, the two sides of its long axis for a supplied control)"
        ),
        "global_reference": None,
        "placebo_false_positive_rate": placebos.get("false_positive_rate"),
        "placebo_false_positive_rate_note": FPR_NOTE,
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
        scores={
            "site_sized_null": site_sized,
            "null_test": dict(null),
            "canopy": None if canopy is None else dict(canopy),
        },
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
    departed_change: np.ndarray,
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
    ax.plot(
        x,
        departed_change,
        color="#1a9641",
        marker="o",
        label="site pixels departed from control, change since baseline",
    )
    ax.axhline(0.0, color="#636363", linestyle="--", linewidth=1, label="expected with no change (0)")
    if np.isfinite(closer).any():
        ax.plot(x, closer, color="#7b3294", marker="o", label="site pixels closer to target")
        ax.plot(x, ctrl_closer, color="#2c7bb6", marker="s", label="control pixels closer to target")
    ax.axvline(planted - 0.5, color="#d7191c", linewidth=1)
    ax.set_ylim(-1.02, 1.02)
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
    ax.set_xlabel("match-year cosine similarity to the site mean")
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
    "Placebos",
    "RestoreError",
    "RestoreResult",
    "Verdict",
    "axis_halves",
    "call_verdict",
    "canopy_check",
    "chunk_runs",
    "closer_fraction",
    "control_threshold",
    "correlation_sign",
    "demo_role_spec",
    "departed_fraction",
    "draw_placebos",
    "exceeds_band",
    "false_positive_rate",
    "geodesic_area_ha",
    "heldout_similarity",
    "load_polygon",
    "main_zone",
    "majority_pattern",
    "match_year_for",
    "memory_estimate",
    "null_caveat",
    "null_halves",
    "placebo_band",
    "placebo_divergence",
    "placebo_verdicts",
    "pooled_sum",
    "resolve_match_year",
    "ring_geometry",
    "run_restore",
    "target_progress",
    "theil_sen_trend",
    "trailing_runs",
    "trajectory_years",
]
