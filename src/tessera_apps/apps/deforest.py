"""Deforest: where did forest become something else between two years, and what is the forest baseline now?

Brief section 4.2, as corrected by the pre-build survey (S12, S13, N13, B5) and the checkpoint 3
lessons (validation semantics, memory):

- **Years.** ``year_a`` and ``year_b`` (``year_b > year_a``) are checked against
  ``years_complete`` for every zone the AOI touches before any chunk is read, and ``year_a`` also
  for every zone the label ring or the label points touch (survey S2). The Hansen Global Forest
  Change version (default :data:`~tessera_apps.validate.HANSEN_VERSION`, loss years 2001 to 2025)
  must cover ``year_a`` when it gives the labels and ``year_b`` when it is the validation reference.
- **Labels for year_a** (survey S12, S13). By default they come from Hansen, sampled in a ring
  around the AOI (inner edge at the AOI boundary unless ``label_ring_inner_km`` moves it out, outer
  edge ``label_ring_km`` out), never inside it, so the validation inside the AOI is spatially held
  out. The ring is buffered in metres in the AOI's main UTM zone, turned back into lon/lat and
  planned like an AOI (:func:`plan_label_ring`). Forest is ``treecover2000 >= forest_cover`` with no
  loss in 2001 to ``year_a``; non-forest is ``treecover2000 < nonforest_cover`` or loss in 2001 to
  ``year_a - 2`` (cleared at least two years before ``year_a``, so it reads as non-forest by then).
  Pixels between the two cuts, pixels lost in ``year_a - 1`` or ``year_a``, and pixels that are not
  mapped land (``datamask != 1``) are never labels (:func:`label_classes`). Tree cover is not
  forest: plantations, orchards and tall crops have cover, forest grown since 2000 has none in
  ``treecover2000``, and both cuts are choices. Sampling is clustered to cap the cost
  (:func:`sample_hansen_labels`): up to ``label_chunks`` store chunks of the ring are drawn at
  random (seeded), alternating between draws weighted by each class's pixel count so both classes
  are represented, and up to ``labels_per_class`` pixels per class are drawn from those chunks only
  and read with :func:`~tessera_apps.io.read_points`, which reads each chunk once. ``labels`` (a
  CSV of lon, lat, label and an optional weight, :func:`read_labels_csv`) replaces the Hansen
  labels; it is checked before the store is opened.
- **Radar regime** (survey B5). A sample of AOI blocks is probed (scales and radar counts only) for
  the regime of the pixels valid in both years with the same regime. The classifier is trained only
  on labels whose ``year_a`` regime is that majority regime, and applied only to pixels in it;
  pixels whose regime differs between the two years are not classified for conversion. Every count
  is in the report.
- **Classifier** (:func:`fit_forest_model`). Logistic regression (lbfgs, balanced class weights) on
  standardized dequantized vectors, fitted once on every label for prediction. The held-out
  estimate is spatial: group k-fold (k = :data:`CV_FOLDS`) with each label's store chunk as its
  group, so no chunk is in both the training and the test part of a fold.
- **Transfer to year_b** (:func:`label_transfer`). A classifier fitted on ``year_a`` can transfer
  badly to ``year_b``, and then the ``year_b`` forest map is off while every ``year_a`` score looks
  good. With Hansen labels, ``year_b`` is also read at the label chunks (the same chunks, counted
  in the estimate), and the labels Hansen keeps unchanged through ``year_b`` (non-forest labels,
  and forest labels with no loss through ``year_b``: :func:`hansen_stable_labels`) are scored in
  ``year_b`` next to their ``year_a`` held-out score; the report warns when ``year_b`` falls more
  than :data:`TRANSFER_WARN_DROP` below. The same read gives the ``year_a`` to ``year_b`` distance
  of the ring's Hansen-stable forest labels, whose ``stable_percentile`` percentile is reported
  next to the cross-check cut (:func:`ring_label_cut`).
- **Prediction.** The forest probability of every AOI pixel in both years, block by block
  (:class:`ForestModel` works on the int8 embeddings directly). Forest is ``p >= forest_prob``.
- **Conversion** (:func:`conversion_mask`) = forest in ``year_a``, non-forest in ``year_b`` and a
  cosine distance between the two years above the cross-check cut (:func:`cross_check_cut`): the
  ``stable_percentile`` percentile of the distance among pixels classed forest in both years, in
  the training regime (survey N13: the brief's AOI median flags half of all pixels by definition).
  That set also holds real conversions the classifier still calls forest in ``year_b``, which can
  raise the cut into the change population; so the report gives the same percentile among the
  ring's Hansen-stable forest labels, warns when the cut is more than :data:`CUT_WARN_RATIO` times
  it, and when validating against Hansen splits the flips the cut removed by Hansen loss. The rule
  itself is the one survey N13 fixes.
  A 3 x 3 opening and ``min_area_ha`` then run on the mosaic of every zone
  (:func:`~tessera_apps.apps._common.mosaic_and_clean`), so a patch across a zone seam is judged
  whole. Patch sizes come from the cleaned mosaic (:func:`patch_size_summary`).
- **Validation** (survey S13). Against Hansen loss in ``(year_a, year_b]`` on Hansen land, over every
  classified AOI pixel of every zone, or against ``reference``. With Hansen it also gives the
  score against the lag-aligned window ``[year_a, year_b - 1]`` (what two annual embeddings mostly
  see: clearing late in ``year_b`` can still read as forest, and clearing in ``year_a`` as forest in
  ``year_a``), the brief's narrower comparison against loss in ``year_b`` alone, the agreement of
  both years' forest maps with Hansen's forest proxy, and the share of each Hansen loss year the
  app converted, flipped, and removed with the cross-check (:func:`loss_year_breakdown`). The
  reference is loaded before any chunk read; when Hansen cannot be reached, or holds no land pixel
  over the AOI (its tiles were not found, or the AOI is water), the run completes with a warning
  (or stops early under ``strict_validate``), and an identical validating rerun redoes only the
  validation, from the outputs on disk.
"""

from __future__ import annotations

import dataclasses
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
import shapely
from pyproj import Transformer
from scipy.special import expit
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from tessera_apps import export, io, metrics, validate
from tessera_apps.apps._common import (
    AppInputError,
    BandMosaic,
    LoadedAOI,
    RunContext,
    StrictValidationError,
    aoi_reference,
    finish_run,
    finish_validation_only,
    load_aoi,
    mosaic_and_clean,
    prepare_run,
    spread_indices,
    stride_for_preview,
    zone_values_from_raster,
)
from tessera_apps.export import GridArray, OutputPaths
from tessera_apps.io import FIRST_YEAR, HA_PER_PIXEL, LAST_YEAR, AOIPlan, Reader, ZonePlan

logger = logging.getLogger(__name__)

APP = "deforest"
#: Layers the AOI stream reads: the radar counts give the regime check, s2_obs_count is not needed.
LAYERS = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
#: Layers the radar regime probe reads: about 6 percent of a full chunk.
PROBE_LAYERS = ("scales", "s1_asc_obs_count", "s1_desc_obs_count")
#: Label classes.
FOREST = 1
NONFOREST = 0
NO_LABEL = -1
DEFAULT_SEED = 0
#: Fewest usable labels per class the classifier is trained on.
MIN_LABELS_PER_CLASS = 20
DEFAULT_LABELS_PER_CLASS = 500
DEFAULT_LABEL_CHUNKS = 24
DEFAULT_RING_KM = 5.0
#: Widest label ring accepted (its Hansen layers are read over the ring's whole window).
MAX_RING_KM = 50.0
#: Ring edges are densified to this spacing (metres) before they go back to lon/lat.
RING_DENSIFY_M = 100.0
DEFAULT_FOREST_COVER = 50
DEFAULT_NONFOREST_COVER = 10
DEFAULT_FOREST_PROB = 0.5
DEFAULT_STABLE_PERCENTILE = 95.0
DEFAULT_MIN_AREA_HA = 0.5
#: Folds of the spatially held-out estimate (fewer when there are fewer label chunks).
CV_FOLDS = 5
#: lbfgs iterations; a fit that does not converge in these is reported.
MAX_ITER = 5000
#: Inverse regularization strength of the logistic regression (sklearn's default).
LOGISTIC_C = 1.0
#: AOI blocks, spread evenly, whose scales and radar counts are probed for the training regime.
REGIME_PROBE_BLOCKS = 8
#: The report warns when the cross-check cut rests on fewer stable forest pixels than this.
MIN_STABLE_PIXELS = 100
#: The report warns when the classifier's year_b balanced accuracy on the Hansen-stable labels falls
#: more than this below their year_a held-out balanced accuracy.
TRANSFER_WARN_DROP = 0.05
#: Fewest Hansen-stable labels per class the transfer warning is judged on.
MIN_TRANSFER_LABELS = 10
#: The report warns when the cross-check cut is more than this many times the ring label cut.
CUT_WARN_RATIO = 1.25
#: Fewest Hansen-stable forest labels the ring label cut is set from.
MIN_RING_CUT_LABELS = 20
#: Lower edges (ha) of the patch-size histogram bins; each bin holds its lower edge.
PATCH_BIN_EDGES_HA = (0.5, 1.0, 5.0, 20.0, 100.0)
PATCH_BIN_LABELS = ("<0.5", "0.5-1", "1-5", "5-20", "20-100", ">100")
#: Largest float32 temporary (bytes) of the per-block prediction.
PREDICT_STRIP_BYTES = 64 * 2**20
#: GDAL block cache while the COG is written (see wildfire.WRITE_CACHE_BYTES).
WRITE_CACHE_BYTES = 128 * 2**20
#: Start of the warning a run records when it could not load its validation reference.
VALIDATION_SKIPPED = "Validation skipped, the reference could not be loaded"
#: Peak memory model (bytes): the stream holds three float32 arrays per window pixel and one block
#: of both years (embeddings, scales, radar counts and the cosine and prediction temporaries); the
#: write stage holds the four-band mosaic, its index and the per-zone arrays; the label stage holds
#: the ring's Hansen layers, classes and reprojection buffers, per pixel of the ring's window (the
#: sampling works chunk by chunk, so it adds no per-pixel term: :func:`sample_hansen_labels`). The
#: label stage peaked at 10.1 bytes per ring-window pixel for a 30 km ring around a 1.1 km AOI with
#: every ring pixel labelled (37 million window pixels), measured as resident memory; the sampler
#: before it held about 90 bytes per labelled pixel. The margin covers allocator overhead.
#: The fixed part is the imported libraries and GDAL's caches: a 4 x 4 km live window (195,000 AOI
#: pixels, a 14 x 14 km ring window) peaked at 0.78 GB.
MEMORY_FIXED = 800_000_000
MEMORY_STREAM_PER_PIXEL = 20
MEMORY_WRITE_PER_PIXEL = 60
MEMORY_PER_BLOCK_PIXEL_YEAR = 210
MEMORY_RING_PER_PIXEL = 12
MEMORY_MARGIN = 1.4

HANSEN_MODEL_CAVEAT = (
    "Hansen Global Forest Change is itself a model, at 30 m, resampled here to the 10 m grid by nearest "
    "neighbour, so disagreement along patch edges is expected."
)
NON_CONVERSION_CAVEAT = (
    "Hansen loss records stand-replacing tree cover loss of any cause, including fire, storms and selective or "
    "salvage logging followed by regrowth, while a conversion here needs a non-forest classification in year_b: "
    "loss that still reads as forest in year_b is a miss by construction, and regrowth after early loss in the "
    "window can hide it."
)
SCOPE_CAVEAT = (
    "Scores count AOI pixels classified in both years (valid, and in the training labels' radar regime in both), "
    "in every zone the AOI touches; with Hansen, only pixels Hansen maps as land (datamask 1)."
)
FOREST_AGREEMENT_CAVEAT = (
    "The forest map agreement compares each year's forest map with Hansen's forest proxy for that year "
    "(treecover2000 at or above the forest cover cut and no loss by that year), over the pixels classified that "
    "year. Tree cover is not forest, pixels between the label cover cuts count as non-forest in the proxy, and the "
    "proxy cannot see forest that grew after 2000. The year_b agreement checks how the year_a classifier "
    "transfers: a low year_b score with a good year_a score means the year_b forest baseline is off."
)
FILE_CAVEAT = "A user-supplied reference is scored as given: pixels it marks are taken as conversion."
TIMING_CAVEAT = (
    "An annual embedding sums the whole year, while Hansen dates loss to the year it was detected: forest cleared "
    "late in year_b can still read as forest in the year_b embedding (missed), and forest cleared in year_a can still "
    "read as forest in the year_a embedding (a conversion that Hansen dates outside the window). loss_year_breakdown "
    "gives, for each Hansen loss year, the share of its scored pixels the app converted, the share the classifier "
    "flipped from forest to non-forest before the cross-check and the cleaning, and the share of flips the "
    "cross-check cut removed (below_cut_share)."
)
CROSS_CHECK_CAVEAT = (
    "The cross-check cut is the stable_percentile percentile of the distance among AOI pixels classed forest in both "
    "years (survey N13). That set also holds real conversions the classifier still calls forest in year_b (partial "
    "or edge pixels, degradation, clearing late in year_b), so where much of the AOI changed the cut can rise into "
    "the change population and remove true conversions: cross_check_removed counts the classifier flips the cut "
    "removed by Hansen loss."
)
REDONE_CAVEAT = (
    "The validation was redone from the forest_prob, distance and converted bands of the run's GeoTIFF and the "
    "cross-check cut in its report, without reading the store. For an AOI in one zone the scores equal a full "
    "run's; across zones a few pixels along the seam may differ, where the mosaic moved them."
)


class DeforestError(io.TesseraError):
    """The data cannot answer the question as asked: no usable labels, no comparable pixel."""


# --------------------------------------------------------------------------------------------
# Labels: classes, ring, sampling
# --------------------------------------------------------------------------------------------


def label_classes(
    treecover: np.ndarray,
    lossyear: np.ndarray,
    datamask: np.ndarray,
    year_a: int,
    *,
    forest_cover: float = DEFAULT_FOREST_COVER,
    nonforest_cover: float = DEFAULT_NONFOREST_COVER,
) -> np.ndarray:
    """Hansen label class of each pixel for ``year_a``: :data:`FOREST`, :data:`NONFOREST` or :data:`NO_LABEL`.

    Only mapped land (``datamask == 1``) is labelled. Forest: ``treecover2000 >= forest_cover`` and
    no loss in 2001 to ``year_a``. Non-forest: ``treecover2000 < nonforest_cover``, or loss in 2001 to
    ``year_a - 2`` (cleared at least two years before ``year_a``). Pixels between the two cover cuts
    with no loss, and pixels lost in ``year_a - 1`` or ``year_a`` (whatever their cover: their
    ``year_a`` embedding may show either state) are not labels.

    Args:
        treecover: Hansen ``treecover2000`` (percent).
        lossyear: Hansen ``lossyear`` (0 none, N means 2000 + N).
        datamask: Hansen ``datamask`` (0 no data, 1 land, 2 water).
        year_a: The year the labels describe.
        forest_cover: Lowest cover of a forest label.
        nonforest_cover: Cover below which a pixel is a non-forest label.

    Returns:
        int8 array of the inputs' shape.

    Raises:
        ValueError: The shapes differ or ``nonforest_cover > forest_cover``.
    """
    tc = np.asarray(treecover)
    ly = np.asarray(lossyear)
    dm = np.asarray(datamask)
    if not (tc.shape == ly.shape == dm.shape):
        raise ValueError(f"Shapes differ: treecover {tc.shape}, lossyear {ly.shape}, datamask {dm.shape}.")
    if nonforest_cover > forest_cover:
        raise ValueError(f"nonforest_cover ({nonforest_cover}) must not exceed forest_cover ({forest_cover}).")
    code = int(year_a) - 2000
    # In place, with few temporaries: the label ring can be large (survey S18).
    out = np.full(tc.shape, NO_LABEL, dtype=np.int8)
    land = dm == 1
    lost_by_a = ly >= 1
    lost_by_a &= ly <= code
    recent = ly >= max(code - 1, 1)
    recent &= ly <= code
    nonforest = tc < nonforest_cover
    nonforest |= lost_by_a & ~recent  # cleared in 2001 to year_a - 2
    nonforest &= ~recent
    nonforest &= land
    out[nonforest] = NONFOREST
    del nonforest, recent
    forest = tc >= forest_cover
    forest &= ~lost_by_a
    forest &= land
    out[forest] = FOREST
    return out


@dataclasses.dataclass(frozen=True)
class LabelRing:
    """The ring around an AOI that Hansen labels are sampled from.

    Attributes:
        geometry: The ring in lon/lat, with the AOI removed.
        zone: The AOI's main zone, whose CRS the ring was buffered in.
        epsg: That zone's EPSG code.
        inner_km: Distance of the inner edge from the AOI boundary.
        outer_km: Distance of the outer edge from the AOI boundary.
    """

    geometry: BaseGeometry
    zone: str
    epsg: int
    inner_km: float
    outer_km: float

    def describe(self) -> str:
        """``"0 to 5 km around the AOI"``."""
        return f"{self.inner_km:g} to {self.outer_km:g} km around the AOI"

    def record(self) -> dict[str, Any]:
        """Plain values for the report."""
        return {
            "inner_km": self.inner_km,
            "outer_km": self.outer_km,
            "buffer_zone": self.zone,
            "buffer_epsg": self.epsg,
            "bounds": [float(b) for b in self.geometry.bounds],
        }


def _main_zone(geometry: BaseGeometry) -> str:
    zones = io.zones_for_geometry(geometry)
    if not zones:
        raise ValueError("The AOI touches no zone.")
    return max(zones, key=lambda z: (io._polygonal(geometry.intersection(io.zone_band(z))).area, z))


def label_ring(aoi: BaseGeometry, *, outer_km: float = DEFAULT_RING_KM, inner_km: float = 0.0) -> LabelRing:
    """The label ring of an AOI: ``inner_km`` to ``outer_km`` from its boundary, outside it.

    The AOI is projected to its main UTM zone (the zone holding most of it), buffered there in
    metres, and the ring is densified and turned back into lon/lat, clipped to the store's latitude
    range, and has the AOI itself removed in lon/lat too, so no pixel centre inside the AOI can
    fall in the ring.

    Args:
        aoi: The AOI in lon/lat.
        outer_km: Outer edge, from the AOI boundary.
        inner_km: Inner edge, from the AOI boundary (0: the boundary itself).

    Returns:
        The ring.

    Raises:
        ValueError: The AOI is empty, or the edges are not ``0 <= inner_km < outer_km``.
    """
    if not (math.isfinite(outer_km) and math.isfinite(inner_km) and 0.0 <= inner_km < outer_km):
        raise ValueError(f"The ring needs 0 <= inner_km < outer_km, got {inner_km} and {outer_km}.")
    geometry = io._polygonal(aoi if aoi.is_valid else shapely.make_valid(aoi))
    if geometry.is_empty or geometry.area == 0:
        raise ValueError("The AOI is empty.")
    zone = _main_zone(geometry)
    epsg = io.zone_epsg(zone)
    fwd = Transformer.from_crs(4326, epsg, always_xy=True)
    inv = Transformer.from_crs(epsg, 4326, always_xy=True)
    projected = shapely.transform(shapely.segmentize(geometry, 0.001), fwd.transform, interleaved=False)
    outer = projected.buffer(outer_km * 1000.0, quad_segs=16)
    inner = projected.buffer(inner_km * 1000.0, quad_segs=16) if inner_km > 0 else projected
    ring_m = io._polygonal(outer.difference(inner))
    ring = shapely.transform(shapely.segmentize(ring_m, RING_DENSIFY_M), inv.transform, interleaved=False)
    ring = io._polygonal(ring if ring.is_valid else shapely.make_valid(ring))
    ring = io._polygonal(ring.intersection(box(-180.0, io.SOUTH_LIMIT, 180.0, io.NORTH_LIMIT)))
    ring = io._polygonal(ring.difference(geometry))
    return LabelRing(geometry=ring, zone=zone, epsg=epsg, inner_km=float(inner_km), outer_km=float(outer_km))


def plan_label_ring(reader: Reader, ring: LabelRing, aoi_plan: AOIPlan, *, max_zones: int = 4) -> AOIPlan:
    """Plan the label ring like an AOI (metadata only), with every AOI pixel removed from its masks.

    Raises:
        DeforestError: The ring cannot be planned (outside the store, too many zones, too thin) or
            holds no pixel outside the AOI.
    """
    if ring.geometry.is_empty:
        raise DeforestError(
            f"The label ring ({ring.describe()}) is empty after clipping to the store's latitude range. Pass --labels."
        )
    try:
        planned = io.plan_aoi(reader, ring.geometry, max_zones=max_zones, max_area_km2=None)
    except io.TesseraError as exc:
        raise DeforestError(
            f"The label ring ({ring.describe()}) cannot be planned: {exc} Pass --labels with your own points, or "
            "change --label-ring-km."
        ) from exc
    aoi_zones = {z.zone: z for z in aoi_plan.zones}
    zones = []
    for z in planned.zones:
        mask = z.mask.copy()
        a = aoi_zones.get(z.zone)
        if a is not None:
            r0, r1 = max(z.rows.start, a.rows.start), min(z.rows.stop, a.rows.stop)
            c0, c1 = max(z.cols.start, a.cols.start), min(z.cols.stop, a.cols.stop)
            if r0 < r1 and c0 < c1:
                inside = a.mask[r0 - a.rows.start : r1 - a.rows.start, c0 - a.cols.start : c1 - a.cols.start]
                mask[r0 - z.rows.start : r1 - z.rows.start, c0 - z.cols.start : c1 - z.cols.start] &= ~inside
        if mask.any():
            zones.append(dataclasses.replace(z, mask=mask))
    if not zones:
        raise DeforestError(f"The label ring ({ring.describe()}) holds no pixel outside the AOI. Pass --labels.")
    return AOIPlan(geometry=planned.geometry, zones=zones)


@dataclasses.dataclass(frozen=True)
class ClassMap:
    """Label classes on one ring zone plan's window.

    Attributes:
        zone: The zone group.
        grid: Its grid (pixel centres to lon/lat).
        row0: Grid row of the window's top row.
        col0: Grid column of the window's left column.
        classes: int8 classes (:func:`label_classes`), :data:`NO_LABEL` outside the ring.
        chunk_shape: The store's chunk shape in this zone.
        lossyear: Hansen ``lossyear`` on the same window, or None; sampled labels then carry it in
            ``hansen_lossyear`` (-1 when None).
    """

    zone: str
    grid: io.ZoneGrid
    row0: int
    col0: int
    classes: np.ndarray
    chunk_shape: tuple[int, int]
    lossyear: np.ndarray | None = None


def choose_label_chunks(
    forest_counts: Sequence[int] | np.ndarray,
    nonforest_counts: Sequence[int] | np.ndarray,
    max_chunks: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pick up to ``max_chunks`` chunks at random so both label classes are represented.

    Draws alternate between the classes, the rarer class (fewer pixels) first: each draw picks one
    chunk not yet chosen with probability proportional to that class's pixels in it. When a class
    has no pixel left in the remaining chunks, the other class draws instead. A chunk with only one
    class can still be drawn on the other class's turn: weights, not exclusion, give the balance.

    Args:
        forest_counts: Forest label pixels per chunk.
        nonforest_counts: Non-forest label pixels per chunk, same order.
        max_chunks: Most chunks to pick.
        rng: Random generator (seeded by the caller).

    Returns:
        Indices of the chosen chunks, in the order drawn.
    """
    f = np.asarray(forest_counts, dtype=np.float64)
    n = np.asarray(nonforest_counts, dtype=np.float64)
    if f.shape != n.shape or f.ndim != 1:
        raise ValueError("forest_counts and nonforest_counts must be 1-D and the same length.")
    available = np.ones(f.size, dtype=bool)
    chosen: list[int] = []
    turn = NONFOREST if n.sum() <= f.sum() else FOREST
    while len(chosen) < max_chunks:
        weights = {FOREST: f * available, NONFOREST: n * available}
        w = weights[turn]
        if w.sum() <= 0:
            w = weights[FOREST if turn == NONFOREST else NONFOREST]
            if w.sum() <= 0:
                break
        i = int(rng.choice(f.size, p=w / w.sum()))
        chosen.append(i)
        available[i] = False
        turn = FOREST if turn == NONFOREST else NONFOREST
    return chosen


LABEL_COLUMNS = ("zone", "row", "col", "chunk_row", "chunk_col", "lon", "lat", "label", "weight", "hansen_lossyear")


def _centres_lonlat(grid: io.ZoneGrid, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = grid.left + (cols.astype(np.float64) + 0.5) * grid.pixel
    y = grid.top - (rows.astype(np.float64) + 0.5) * grid.pixel
    lon, lat = grid.to_lonlat().transform(x, y)
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {c: pd.Series(dtype="float64" if c in ("lon", "lat", "weight") else "int64") for c in LABEL_COLUMNS}
    )


def chunk_tiles(m: ClassMap) -> list[tuple[int, int, slice, slice]]:
    """The store chunks a class map's window overlaps, row by row: ``(chunk_row, chunk_col, rows, cols)``.

    ``rows`` and ``cols`` are the chunk's part of the window, in window coordinates (the window need
    not be chunk aligned).
    """
    h, w = np.asarray(m.classes).shape
    ch, cw = m.chunk_shape
    if h <= 0 or w <= 0:
        return []
    out = []
    for kr in range(m.row0 // ch, (m.row0 + h - 1) // ch + 1):
        rows = slice(max(0, kr * ch - m.row0), min(h, (kr + 1) * ch - m.row0))
        for kc in range(m.col0 // cw, (m.col0 + w - 1) // cw + 1):
            out.append((kr, kc, rows, slice(max(0, kc * cw - m.col0), min(w, (kc + 1) * cw - m.col0))))
    return out


def sample_hansen_labels(
    maps: Sequence[ClassMap], *, max_chunks: int, per_class: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cluster-sample label pixels: up to ``max_chunks`` chunks, then up to ``per_class`` pixels per class.

    Chunks are picked by :func:`choose_label_chunks`; pixels are then drawn without replacement,
    uniformly from each class's pixels in the chosen chunks only. Everything is seeded by ``seed``.
    The class maps are counted chunk by chunk (:func:`chunk_tiles`), so beyond the maps themselves
    the memory used is one chunk's worth plus the pixels of the chosen chunks, whatever the ring's
    size (survey S18).

    Args:
        maps: Class maps of the ring's zone plans.
        max_chunks: Most chunks to sample from.
        per_class: Most pixels per class.
        seed: Random seed.

    Returns:
        ``(labels, info)``: one row per sampled pixel (``zone``, grid ``row`` and ``col``,
        ``chunk_row``, ``chunk_col``, centre ``lon`` and ``lat``, ``label``, ``weight`` 1 and
        ``hansen_lossyear``, -1 when the map has no lossyear), sorted by zone, row and column; and
        counts for the report.
    """
    rng = np.random.default_rng(seed)
    chunk_keys: list[tuple[str, int, int]] = []
    forest_counts: list[int] = []
    nonforest_counts: list[int] = []
    members: list[tuple[int, slice, slice]] = []  # (map index, window rows, window cols) of each chunk
    for mi, m in enumerate(maps):
        classes = np.asarray(m.classes)
        for kr, kc, rows, cols in chunk_tiles(m):
            tile = classes[rows, cols]
            f = int(np.count_nonzero(tile == FOREST))
            n = int(np.count_nonzero(tile == NONFOREST))
            if f or n:
                chunk_keys.append((m.zone, kr, kc))
                forest_counts.append(f)
                nonforest_counts.append(n)
                members.append((mi, rows, cols))
    order = sorted(range(len(chunk_keys)), key=lambda i: chunk_keys[i])
    chunk_keys = [chunk_keys[i] for i in order]
    forest_counts = [forest_counts[i] for i in order]
    nonforest_counts = [nonforest_counts[i] for i in order]
    members = [members[i] for i in order]
    chosen = choose_label_chunks(forest_counts, nonforest_counts, max_chunks, rng)
    info: dict[str, Any] = {
        "ring_pixels_forest": int(sum(forest_counts)),
        "ring_pixels_nonforest": int(sum(nonforest_counts)),
        "chunks_with_labels": len(chunk_keys),
        "chunks_with_forest": int(sum(1 for c in forest_counts if c)),
        "chunks_with_nonforest": int(sum(1 for c in nonforest_counts if c)),
        "chunks_used": len(chosen),
        "chunks": [f"{chunk_keys[i][0]}/{chunk_keys[i][1]}/{chunk_keys[i][2]}" for i in sorted(chosen)],
        "pixels_in_chunks_forest": int(sum(forest_counts[i] for i in chosen)),
        "pixels_in_chunks_nonforest": int(sum(nonforest_counts[i] for i in chosen)),
    }
    picks: list[pd.DataFrame] = []
    in_order = sorted(chosen)
    for cls in (FOREST, NONFOREST):
        # The pool is every pixel of the class in the chosen chunks, chunk by chunk in key order and
        # row-major within a chunk. Its size is known from the counts, so the draw comes first and
        # only the chunks holding a picked position are decoded, one at a time.
        counts = forest_counts if cls == FOREST else nonforest_counts
        sizes = np.asarray([counts[i] for i in in_order], dtype=np.int64)
        total = int(sizes.sum())
        k = min(per_class, total)
        if k <= 0:
            continue
        pick = np.sort(rng.choice(total, size=k, replace=False))
        starts = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)])
        where = np.searchsorted(starts, pick, side="right") - 1
        parts_rows: list[np.ndarray] = []
        parts_cols: list[np.ndarray] = []
        parts_maps: list[np.ndarray] = []
        for j in np.unique(where):
            mi, rows, cols = members[in_order[int(j)]]
            rr, cc = np.nonzero(np.asarray(maps[mi].classes)[rows, cols] == cls)
            local = pick[where == j] - starts[j]
            parts_rows.append(rr[local].astype(np.int64) + rows.start)
            parts_cols.append(cc[local].astype(np.int64) + cols.start)
            parts_maps.append(np.full(local.size, mi, dtype=np.int64))
        win_rows = np.concatenate(parts_rows)
        win_cols = np.concatenate(parts_cols)
        picked_maps = np.concatenate(parts_maps)
        for mi in np.unique(picked_maps):
            m = maps[int(mi)]
            here = picked_maps == mi
            wr, wc = win_rows[here], win_cols[here]
            rows, cols = wr + m.row0, wc + m.col0
            lon, lat = _centres_lonlat(m.grid, rows, cols)
            loss = (
                np.full(wr.size, -1, dtype=np.int64)
                if m.lossyear is None
                else np.asarray(m.lossyear)[wr, wc].astype(np.int64)
            )
            picks.append(
                pd.DataFrame(
                    {
                        "zone": m.zone,
                        "row": rows.astype(np.int64),
                        "col": cols.astype(np.int64),
                        "chunk_row": (rows // m.chunk_shape[0]).astype(np.int64),
                        "chunk_col": (cols // m.chunk_shape[1]).astype(np.int64),
                        "lon": lon,
                        "lat": lat,
                        "label": np.full(wr.size, cls, dtype=np.int64),
                        "weight": np.ones(wr.size, dtype=np.float64),
                        "hansen_lossyear": loss,
                    }
                )
            )
    labels = pd.concat(picks, ignore_index=True) if picks else _empty_labels()
    labels = labels.sort_values(["zone", "row", "col"], kind="stable").reset_index(drop=True)
    info["sampled_forest"] = int((labels["label"] == FOREST).sum())
    info["sampled_nonforest"] = int((labels["label"] == NONFOREST).sum())
    return labels, info


# --------------------------------------------------------------------------------------------
# Labels from a CSV
# --------------------------------------------------------------------------------------------


def _rows_text(index: np.ndarray, limit: int = 5) -> str:
    lines = [str(int(i) + 2) for i in index[:limit]]  # 1-based file lines, after the header
    more = f" and {index.size - limit} more" if index.size > limit else ""
    return ", ".join(lines) + more


def read_labels_csv(path: Path | str, *, min_per_class: int | None = None) -> pd.DataFrame:
    """Read and check a labels CSV: columns ``lon``, ``lat``, ``label`` (1 forest, 0 non-forest), optional ``weight``.

    Column names are matched case-insensitively. Every row must have a finite lon in -180 to 180, a
    lat inside the store's UTM range (:data:`~tessera_apps.io.SOUTH_LIMIT` to
    :data:`~tessera_apps.io.NORTH_LIMIT`), a label of 0 or 1 and, when the column is there, a
    finite positive weight. Each class needs at least ``min_per_class`` rows. Nothing is read from
    the store.

    Args:
        path: The CSV.
        min_per_class: Fewest rows per class; :data:`MIN_LABELS_PER_CLASS` when None.

    Returns:
        ``lon``, ``lat`` (float64), ``label`` (int64) and ``weight`` (float64, 1 when absent).

    Raises:
        AppInputError: The file is missing or unreadable, a column is missing, a value is bad, or a
            class has too few rows; the message names the file, the lines and what to change.
    """
    need = MIN_LABELS_PER_CLASS if min_per_class is None else int(min_per_class)
    p = Path(path)
    hint = "Pass a CSV with columns lon, lat, label (1 forest, 0 non-forest) and an optional weight with --labels."
    if not p.is_file():
        raise AppInputError(f"Labels file {p} does not exist. {hint}")
    try:
        frame = pd.read_csv(p)
    except (OSError, ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AppInputError(f"Could not read the labels file {p} as CSV: {exc}. {hint}") from exc
    columns = {str(c).strip().lower(): c for c in frame.columns}
    missing = [c for c in ("lon", "lat", "label") if c not in columns]
    if missing:
        raise AppInputError(
            f"Labels file {p} has no {', '.join(missing)} column (found "
            f"{', '.join(map(str, frame.columns)) or 'none'}). {hint}"
        )
    if not len(frame):
        raise AppInputError(f"Labels file {p} has no rows. {hint}")
    lon = pd.to_numeric(frame[columns["lon"]], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(frame[columns["lat"]], errors="coerce").to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~(np.isfinite(lon) & np.isfinite(lat)))
    if bad.size:
        raise AppInputError(f"Labels file {p}: lon or lat is missing or not a number on line(s) {_rows_text(bad)}.")
    bad = np.flatnonzero((lon < -180.0) | (lon > 180.0) | (lat < io.SOUTH_LIMIT) | (lat > io.NORTH_LIMIT))
    if bad.size:
        raise AppInputError(
            f"Labels file {p}: line(s) {_rows_text(bad)} lie outside lon -180 to 180, lat {io.SOUTH_LIMIT:g} to "
            f"{io.NORTH_LIMIT:g}, the UTM range the store covers. Check the columns are lon/lat in degrees."
        )
    label = pd.to_numeric(frame[columns["label"]], errors="coerce").to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~np.isin(label, (0.0, 1.0)))
    if bad.size:
        shown = sorted({str(frame[columns["label"]].iloc[i]) for i in bad[:5]})
        raise AppInputError(
            f"Labels file {p}: label must be 1 (forest) or 0 (non-forest); line(s) {_rows_text(bad)} hold "
            f"{', '.join(shown)}."
        )
    if "weight" in columns:
        weight = pd.to_numeric(frame[columns["weight"]], errors="coerce").to_numpy(dtype=np.float64)
        bad = np.flatnonzero(~(np.isfinite(weight) & (weight > 0)))
        if bad.size:
            raise AppInputError(f"Labels file {p}: weight must be a positive number; line(s) {_rows_text(bad)}.")
    else:
        weight = np.ones(lon.size, dtype=np.float64)
    n_forest = int((label == 1).sum())
    n_non = int((label == 0).sum())
    if min(n_forest, n_non) < need:
        raise AppInputError(
            f"Labels file {p} has {n_forest} forest and {n_non} non-forest rows; the classifier needs at least {need} "
            "of each. Add labels for the smaller class."
        )
    return pd.DataFrame({"lon": lon, "lat": lat, "label": label.astype(np.int64), "weight": weight})


def csv_label_points(
    reader: Reader, frame: pd.DataFrame, chunk_shapes: Mapping[str, tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Place checked CSV labels on the store grid: zone, grid row and column, chunk (metadata only).

    Each point goes to its nominal zone (the south group on the equator, as
    :func:`~tessera_apps.io.read_points` does). A point outside its zone's grid gets row, column and
    chunk -1; :func:`~tessera_apps.io.read_points` then finds no data for it.
    """
    zones = [io._point_zone(float(x), float(y)) for x, y in zip(frame["lon"], frame["lat"], strict=True)]
    rows = np.full(len(frame), -1, dtype=np.int64)
    cols = np.full(len(frame), -1, dtype=np.int64)
    chunk_r = np.full(len(frame), -1, dtype=np.int64)
    chunk_c = np.full(len(frame), -1, dtype=np.int64)
    zone_arr = np.asarray(zones)
    for zone in sorted(set(zones)):
        idx = np.flatnonzero(zone_arr == zone)
        grid = reader.grid(zone)
        ch, cw = (chunk_shapes or {}).get(zone) or reader.chunk_shape(zone)
        xs, ys = grid.from_lonlat().transform(frame["lon"].to_numpy()[idx], frame["lat"].to_numpy()[idx])
        for i, x, y in zip(idx, np.atleast_1d(xs), np.atleast_1d(ys), strict=True):
            r, c = grid.rowcol(float(x), float(y))
            if 0 <= r < grid.height and 0 <= c < grid.width:
                rows[i], cols[i], chunk_r[i], chunk_c[i] = r, c, r // ch, c // cw
    out = frame.copy()
    out.insert(0, "zone", zones)
    out.insert(1, "row", rows)
    out.insert(2, "col", cols)
    out.insert(3, "chunk_row", chunk_r)
    out.insert(4, "chunk_col", chunk_c)
    out["hansen_lossyear"] = np.full(len(out), -1, dtype=np.int64)
    return out[list(LABEL_COLUMNS)].reset_index(drop=True)


def chunk_ids(labels: pd.DataFrame) -> np.ndarray:
    """``"zone/chunk_row/chunk_col"`` of every label: its source chunk and cross-validation group."""
    return np.asarray(
        [f"{z}/{r}/{c}" for z, r, c in zip(labels["zone"], labels["chunk_row"], labels["chunk_col"], strict=True)],
        dtype=object,
    )


# --------------------------------------------------------------------------------------------
# Radar regime
# --------------------------------------------------------------------------------------------


def training_regime(radar_pixels: int, optical_pixels: int) -> bool | None:
    """The regime the classifier is trained on and applied to: True for radar, False for optical-only.

    The majority of the AOI pixels valid in both years with the same regime in both; ties go to
    radar. None when there is no such pixel.
    """
    if radar_pixels + optical_pixels <= 0:
        return None
    return radar_pixels >= optical_pixels


def regime_name(radar: bool | None) -> str:
    """``"radar"``, ``"optical"`` (Sentinel-2 only) or ``"none"``."""
    return "none" if radar is None else ("radar" if radar else "optical")


def apply_regime(labels: pd.DataFrame, valid: np.ndarray, radar: np.ndarray, train_radar: bool) -> pd.DataFrame:
    """Mark which labels the classifier may use: valid in ``year_a`` and in the training regime (survey B5).

    Adds ``valid``, ``radar``, ``regime`` (``radar``, ``optical`` or ``no_data``), ``used`` and
    ``drop_reason`` (``""``, ``"no_data"`` or ``"regime"``) to a copy of ``labels``.
    """
    v = np.asarray(valid, dtype=bool)
    r = np.asarray(radar, dtype=bool)
    if v.shape != (len(labels),) or r.shape != (len(labels),):
        raise ValueError("valid and radar need one value per label.")
    out = labels.copy()
    out["valid"] = v
    out["radar"] = r & v
    out["regime"] = np.where(v, np.where(r, "radar", "optical"), "no_data")
    used = v & (r == bool(train_radar))
    out["used"] = used
    out["drop_reason"] = np.where(used, "", np.where(v, "regime", "no_data"))
    return out


# --------------------------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------------------------


def _valid(win: io.Window) -> np.ndarray:
    """Pixels with a finite, positive scale and a vector that is not all zero."""
    return win.valid & (win.scales > 0)


@dataclasses.dataclass(frozen=True, eq=False)
class ForestModel:
    """A fitted forest classifier folded into one linear function of the dequantized vector.

    ``standardize then logistic regression`` is ``z = x . weights + intercept`` with ``x`` the
    dequantized vector, so a pixel's forest probability is ``expit(scale * (e . weights) + intercept)``
    with ``e`` its int8 embedding: no dequantized block is ever built.

    Attributes:
        weights: (128,) float64 weights on the dequantized vector.
        intercept: The constant term.
    """

    weights: np.ndarray
    intercept: float

    @classmethod
    def from_fitted(cls, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float) -> ForestModel:
        """Fold a StandardScaler (``mean``, ``scale``) and a logistic regression (``coef``, ``intercept``)."""
        w = np.asarray(coef, dtype=np.float64).ravel() / np.asarray(scale, dtype=np.float64).ravel()
        b = float(intercept) - float(np.dot(np.asarray(mean, dtype=np.float64).ravel(), w))
        return cls(weights=w, intercept=b)

    def probability_vectors(self, x: np.ndarray) -> np.ndarray:
        """Forest probability of dequantized vectors (..., 128), float64."""
        return np.asarray(expit(np.asarray(x, dtype=np.float64) @ self.weights + self.intercept))

    def probability(self, win: io.Window) -> np.ndarray:
        """Forest probability of every pixel of a window, float32 (H, W), NaN where the pixel is not valid."""
        if win.embeddings is None:
            raise ValueError("ForestModel.probability needs the embeddings layer.")
        h, w = win.shape
        out = np.full((h, w), np.nan, dtype=np.float32)
        weights = self.weights.astype(np.float32)
        step = max(1, PREDICT_STRIP_BYTES // max(1, w * win.embeddings.shape[-1] * 4))
        scales = win.scales
        for r0 in range(0, h, step):
            r1 = min(h, r0 + step)
            z = (win.embeddings[r0:r1].astype(np.float32) @ weights) * scales[r0:r1] + np.float32(self.intercept)
            out[r0:r1] = expit(z)
        out[~_valid(win)] = np.nan
        return out


@dataclasses.dataclass(frozen=True, eq=False)
class ClassifierFit:
    """The fitted model and its spatially held-out estimate.

    Attributes:
        model: Fitted on every label.
        folds: Fold of each label in the group k-fold (-1 when there was no held-out estimate).
        heldout_prob: Each label's forest probability from the fold that held it out; NaN when its
            fold could not be fitted (one class only in the training part) or there were no folds.
        record: Diagnostics for the report.
    """

    model: ForestModel
    folds: np.ndarray
    heldout_prob: np.ndarray
    record: dict[str, Any]


def _recalls(prob: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float | int | None]:
    pred = prob >= threshold
    is_forest = truth == FOREST
    rf = float(np.mean(pred[is_forest])) if is_forest.any() else None
    rn = float(np.mean(~pred[~is_forest])) if (~is_forest).any() else None
    bal = None if rf is None or rn is None else (rf + rn) / 2.0
    acc = float(np.mean(pred == is_forest)) if truth.size else None
    return {
        "n": int(truth.size),
        "accuracy": acc,
        "balanced_accuracy": bal,
        "recall_forest": rf,
        "recall_nonforest": rn,
    }


def fit_forest_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | Sequence[Any],
    *,
    weights: np.ndarray | None = None,
    seed: int = DEFAULT_SEED,
    n_folds: int = CV_FOLDS,
    forest_prob: float = DEFAULT_FOREST_PROB,
) -> ClassifierFit:
    """Fit the forest classifier on every label and estimate its skill on held-out chunks.

    StandardScaler then LogisticRegression (lbfgs, ``class_weight="balanced"``, ``random_state=seed``).
    The held-out estimate is group k-fold (sklearn ``GroupKFold``, ``min(n_folds, groups)`` folds)
    with each label's chunk as its group: a fold whose training part has one class only is skipped
    and its labels are not scored. Scores use ``forest_prob`` as the forest cut.

    Args:
        x: (N, 128) dequantized vectors.
        y: (N,) labels, 1 forest and 0 non-forest.
        groups: (N,) group of each label (its store chunk).
        weights: Optional (N,) sample weights for both the scaler and the regression.
        seed: ``random_state`` of the regression.
        n_folds: Most folds.
        forest_prob: Forest cut for the scores.

    Returns:
        The fit.

    Raises:
        ValueError: Shapes differ, or a class has no label.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y).astype(np.int64)
    gs = np.asarray(groups)
    if xs.ndim != 2 or ys.shape != (xs.shape[0],) or gs.shape != (xs.shape[0],):
        raise ValueError(f"x {xs.shape}, y {ys.shape} and groups {gs.shape} do not match.")
    if np.unique(ys).size < 2 or not set(np.unique(ys).tolist()) <= {FOREST, NONFOREST}:
        raise ValueError("The labels need both classes, 1 (forest) and 0 (non-forest).")
    ws = None if weights is None else np.asarray(weights, dtype=np.float64)

    def fit(idx: np.ndarray) -> tuple[ForestModel, bool, int]:
        sw = None if ws is None else ws[idx]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            scaler = StandardScaler().fit(xs[idx], sample_weight=sw)
            lr = LogisticRegression(
                solver="lbfgs", C=LOGISTIC_C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed
            )
            lr.fit(scaler.transform(xs[idx]), ys[idx], sample_weight=sw)
        converged = not any(issubclass(c.category, ConvergenceWarning) for c in caught)
        model = ForestModel.from_fitted(scaler.mean_, scaler.scale_, lr.coef_[0], float(lr.intercept_[0]))
        return model, converged, int(np.max(lr.n_iter_))

    everything = np.arange(xs.shape[0])
    model, converged, n_iter = fit(everything)
    training = _recalls(model.probability_vectors(xs), ys, forest_prob)
    n_groups = int(np.unique(gs).size)
    k = min(int(n_folds), n_groups)
    folds = np.full(xs.shape[0], -1, dtype=np.int64)
    held = np.full(xs.shape[0], np.nan, dtype=np.float64)
    skipped = 0
    folds_converged = True
    if k >= 2:
        for i, (train, test) in enumerate(GroupKFold(n_splits=k).split(xs, ys, gs)):
            folds[test] = i
            if np.unique(ys[train]).size < 2:
                skipped += 1
                continue
            fold_model, fold_converged, _ = fit(train)
            folds_converged &= fold_converged
            held[test] = fold_model.probability_vectors(xs[test])
    scored = np.isfinite(held)
    heldout = _recalls(held[scored], ys[scored], forest_prob)
    record = {
        "method": "StandardScaler + LogisticRegression (lbfgs, class_weight balanced)",
        "c": LOGISTIC_C,
        "max_iter": MAX_ITER,
        "n_iter": n_iter,
        "converged": converged,
        "labels": int(xs.shape[0]),
        "labels_forest": int((ys == FOREST).sum()),
        "labels_nonforest": int((ys == NONFOREST).sum()),
        "weighted": bool(ws is not None and ws.size and not np.all(ws == ws[0])),
        "forest_prob": forest_prob,
        "training": training,
        "heldout": {
            **heldout,
            "method": "GroupKFold by store chunk",
            "groups": n_groups,
            "folds": k if k >= 2 else 0,
            "folds_skipped": skipped,
            "folds_converged": folds_converged,
            "labels_unscored": int((~scored).sum()),
        },
    }
    return ClassifierFit(model=model, folds=folds, heldout_prob=held, record=record)


# --------------------------------------------------------------------------------------------
# Transfer to year_b, at the labels
# --------------------------------------------------------------------------------------------


def hansen_stable_labels(
    label: np.ndarray | Sequence[int], lossyear: np.ndarray | Sequence[int], year_b: int
) -> np.ndarray:
    """Labels whose Hansen class is unchanged through ``year_b``.

    Every non-forest label (low cover, or cleared at least two years before ``year_a``) is taken as
    staying non-forest. A forest label has no loss through ``year_a`` by construction
    (:func:`label_classes`), so it is still forest in ``year_b`` unless Hansen records loss in
    ``(year_a, year_b]``: it is stable when its lossyear is 0 or after ``year_b``. A label with no
    Hansen lossyear (-1, for example from a CSV) is never stable, since nothing says what it became.

    Args:
        label: Label classes, :data:`FOREST` or :data:`NONFOREST`.
        lossyear: Each label's Hansen lossyear code, -1 when unknown.
        year_b: The second year.

    Returns:
        Bool array, one value per label.
    """
    y = np.asarray(label)
    ly = np.asarray(lossyear).astype(np.int64)
    if y.shape != ly.shape:
        raise ValueError(f"label {y.shape} and lossyear {ly.shape} do not match.")
    code = int(year_b) - 2000
    kept_forest = (y == FOREST) & ((ly == 0) | (ly > code))
    return np.asarray((ly >= 0) & ((y == NONFOREST) | kept_forest), dtype=bool)


def vector_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance between matching rows of two (N, 128) arrays; NaN where a row is not finite or zero."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        norms = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
        d = 1.0 - np.sum(x * y, axis=-1) / norms
    return np.where(np.isfinite(d) & (norms > 0), d, np.nan)


def label_transfer(
    y: np.ndarray,
    scored: np.ndarray,
    prob_a: np.ndarray,
    heldout_a: np.ndarray,
    prob_b: np.ndarray,
    *,
    forest_prob: float = DEFAULT_FOREST_PROB,
) -> dict[str, Any]:
    """How the ``year_a`` classifier transfers to ``year_b``, on labels whose class did not change.

    Args:
        y: Label classes (1 forest, 0 non-forest).
        scored: Labels to score: Hansen-stable through ``year_b`` (:func:`hansen_stable_labels`),
            used for training, and valid in ``year_b`` in the training regime.
        prob_a: Each label's ``year_a`` forest probability from the model fitted on every label.
        heldout_a: Each label's ``year_a`` held-out probability (NaN when its fold was not scored).
        prob_b: Each label's ``year_b`` forest probability from the model fitted on every label.
        forest_prob: The forest cut.

    Returns:
        ``labels``, ``labels_forest`` and ``labels_nonforest`` (the scored labels);
        ``year_a_in_sample``, ``year_a_heldout`` (on the scored labels with a held-out
        probability) and ``year_b`` scores (``n``, ``accuracy``, ``balanced_accuracy``,
        ``recall_forest``, ``recall_nonforest``); and ``drop``, the ``year_a`` held-out balanced
        accuracy minus the ``year_b`` one on the same labels (None when either is missing), with
        ``drop_labels`` the labels it is judged on.
    """
    ys = np.asarray(y).astype(np.int64)
    s = np.asarray(scored, dtype=bool)
    pa = np.asarray(prob_a, dtype=np.float64)
    ha = np.asarray(heldout_a, dtype=np.float64)
    pb = np.asarray(prob_b, dtype=np.float64)
    if not (ys.shape == s.shape == pa.shape == ha.shape == pb.shape):
        raise ValueError("y, scored, prob_a, heldout_a and prob_b need one value per label.")
    s = s & np.isfinite(pa) & np.isfinite(pb)
    same = s & np.isfinite(ha)
    heldout = _recalls(ha[same], ys[same], forest_prob)
    year_b_same = _recalls(pb[same], ys[same], forest_prob)
    a_bal, b_bal = heldout["balanced_accuracy"], year_b_same["balanced_accuracy"]
    return {
        "labels": int(s.sum()),
        "labels_forest": int((s & (ys == FOREST)).sum()),
        "labels_nonforest": int((s & (ys == NONFOREST)).sum()),
        "year_a_in_sample": _recalls(pa[s], ys[s], forest_prob),
        "year_a_heldout": heldout,
        "year_b": _recalls(pb[s], ys[s], forest_prob),
        "drop": None if a_bal is None or b_bal is None else float(a_bal) - float(b_bal),
        "drop_labels": int(same.sum()),
        "drop_labels_forest": int((same & (ys == FOREST)).sum()),
        "drop_labels_nonforest": int((same & (ys == NONFOREST)).sum()),
    }


def ring_label_cut(
    distance: np.ndarray, y: np.ndarray, scored: np.ndarray, *, percentile: float = DEFAULT_STABLE_PERCENTILE
) -> tuple[float | None, int]:
    """The cross-check percentile among the ring's Hansen-stable forest labels.

    The same percentile as :func:`cross_check_cut`, of the ``year_a`` to ``year_b`` distance, but
    over forest labels Hansen keeps unchanged through ``year_b`` (``scored``) instead of AOI pixels
    the classifier calls forest in both years, so real conversions cannot raise it. The labels lie
    outside the AOI, so the validation inside it stays independent of this number.

    Returns:
        ``(cut, n)``; ``cut`` is None when no such label has a finite distance.
    """
    d = np.asarray(distance, dtype=np.float64)
    keep = np.asarray(scored, dtype=bool) & (np.asarray(y) == FOREST) & np.isfinite(d)
    if not keep.any():
        return None, 0
    return float(np.percentile(d[keep], percentile)), int(keep.sum())


# --------------------------------------------------------------------------------------------
# Per-block classification, conversion, patches
# --------------------------------------------------------------------------------------------


BLOCK_COUNTS = (
    "valid_a",
    "valid_b",
    "valid_both",
    "regime_mismatch",
    "other_regime",
    "other_regime_a",
    "other_regime_b",
)


def classify_block(
    wa: io.Window, wb: io.Window, mask: np.ndarray, model: ForestModel, train_radar: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Forest probabilities of both years and their distance on one block (survey B5 applied).

    A year's probability is set only on ``mask`` pixels valid that year whose radar regime is the
    training regime. The distance is set on ``mask`` pixels valid in both years with the same
    regime in both (whichever regime). So a pixel whose regime differs between the years, or is not
    the training regime in either year, can never be a conversion.

    Returns:
        ``(prob_a, prob_b, distance, counts)``: float32 arrays (NaN where unset) and the pixel counts
        named in :data:`BLOCK_COUNTS`.
    """
    m = np.asarray(mask, dtype=bool)
    va = _valid(wa) & m
    vb = _valid(wb) & m
    ra = wa.radar_present
    rb = wb.radar_present
    ok_a = va & (ra == train_radar)
    ok_b = vb & (rb == train_radar)
    nan = np.float32(np.nan)
    prob_a = np.where(ok_a, model.probability(wa), nan).astype(np.float32)
    prob_b = np.where(ok_b, model.probability(wb), nan).astype(np.float32)
    both = va & vb
    same = ra == rb
    d = metrics.cosine_distance_map(wa, wb)
    dist = np.where(both & same & np.isfinite(d), d, nan).astype(np.float32)
    counts = {
        "valid_a": int(va.sum()),
        "valid_b": int(vb.sum()),
        "valid_both": int(both.sum()),
        "regime_mismatch": int((both & ~same).sum()),
        "other_regime": int((both & same & (ra != train_radar)).sum()),
        "other_regime_a": int((va & ~ok_a).sum()),
        "other_regime_b": int((vb & ~ok_b).sum()),
    }
    return prob_a, prob_b, dist, counts


def _as_list(a: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(a)] if isinstance(a, np.ndarray) else [np.asarray(v) for v in a]


def cross_check_cut(
    dist: np.ndarray | Sequence[np.ndarray],
    prob_a: np.ndarray | Sequence[np.ndarray],
    prob_b: np.ndarray | Sequence[np.ndarray],
    *,
    forest_prob: float = DEFAULT_FOREST_PROB,
    percentile: float = DEFAULT_STABLE_PERCENTILE,
) -> tuple[float | None, int]:
    """The cross-check cut: the ``percentile`` percentile of the distance among stable forest pixels (survey N13).

    Stable forest is every pixel classed forest in both years (``p >= forest_prob``) with a finite
    distance; NaN anywhere leaves the pixel out. Arrays may be given per zone as sequences, which
    are pooled.

    Returns:
        ``(cut, n_stable)``; ``cut`` is None when there is no stable forest pixel.
    """
    values = []
    for d, pa, pb in zip(_as_list(dist), _as_list(prob_a), _as_list(prob_b), strict=True):
        with np.errstate(invalid="ignore"):
            stable = np.isfinite(d) & (pa >= forest_prob) & (pb >= forest_prob)
        values.append(np.asarray(d[stable], dtype=np.float64))
    v = np.concatenate(values) if values else np.empty(0)
    if not v.size:
        return None, 0
    return float(np.percentile(v, percentile)), int(v.size)


def conversion_mask(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    dist: np.ndarray,
    *,
    forest_prob: float = DEFAULT_FOREST_PROB,
    cut: float | None,
) -> np.ndarray:
    """Raw conversion: forest in ``year_a``, non-forest in ``year_b`` and distance above ``cut``.

    Pixels with a NaN probability or distance are never conversions. ``cut`` None skips the distance
    test (no stable forest to set it).
    """
    pa = np.asarray(prob_a)
    pb = np.asarray(prob_b)
    d = np.asarray(dist)
    with np.errstate(invalid="ignore"):
        m = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(d) & (pa >= forest_prob) & (pb < forest_prob)
        if cut is not None:
            m &= d > cut
    return np.asarray(m, dtype=bool)


def clean_conversion(
    raw: Mapping[str, np.ndarray], plan: AOIPlan, *, min_pixels: int, dst_zone: str | None = None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """3 x 3 opening and minimum patch size on the mosaic of every zone, mapped back per zone.

    Returns:
        ``(cleaned per zone, cleaned uint8 mosaic)`` (1 kept, 0 not, 255 outside the AOI).
    """
    cleaned, mosaic, _, _ = mosaic_and_clean(raw, plan, min_pixels=min_pixels, opening=3, dst_zone=dst_zone)
    return cleaned, mosaic


def patch_size_summary(sizes_px: np.ndarray | Sequence[int]) -> dict[str, Any]:
    """Count, median, 90th percentile, maximum (ha) and histogram of conversion patch sizes.

    Bins (:data:`PATCH_BIN_LABELS`) are counted in whole pixels, each holding its lower edge, so a
    patch of exactly 1 ha falls in ``1-5``.
    """
    s = np.asarray(sizes_px, dtype=np.int64).ravel()
    edges_px = np.asarray([metrics.area_to_pixels(e) for e in PATCH_BIN_EDGES_HA], dtype=np.int64)
    bins = np.searchsorted(edges_px, s, side="right")
    counts = np.bincount(bins, minlength=len(PATCH_BIN_LABELS))
    ha = s * HA_PER_PIXEL
    return {
        "count": int(s.size),
        "total_ha": float(ha.sum()),
        "median_ha": float(np.median(ha)) if s.size else None,
        "p90_ha": float(np.percentile(ha, 90)) if s.size else None,
        "max_ha": float(ha.max()) if s.size else None,
        "histogram": {label: int(c) for label, c in zip(PATCH_BIN_LABELS, counts, strict=True)},
        "bin_edges_ha": list(PATCH_BIN_EDGES_HA),
    }


def memory_estimate(plan: AOIPlan, *, block: tuple[int, int], ring_km: float | None) -> int:
    """Peak memory of a run in bytes, from the plan's windows, the block shape and the label ring (survey S18)."""
    window_px = sum(z.shape[0] * z.shape[1] for z in plan.zones)
    block_px = max((min(block[0], z.shape[0]) * min(block[1], z.shape[1]) for z in plan.zones), default=0)
    stream = MEMORY_STREAM_PER_PIXEL * window_px + MEMORY_PER_BLOCK_PIXEL_YEAR * block_px * 2
    write = MEMORY_WRITE_PER_PIXEL * window_px
    ring = 0
    if ring_km:
        pad = 2 * ring_km * 1000.0 / io.PIXEL_M
        ring = int(MEMORY_RING_PER_PIXEL * sum((z.shape[0] + pad) * (z.shape[1] + pad) for z in plan.zones))
    return int(MEMORY_FIXED + MEMORY_MARGIN * max(stream, write, ring))


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Reference:
    """The validation reference on every AOI zone plan, loaded before any chunk read (or why it was not)."""

    window: dict[str, np.ndarray] | None = None
    year_b_only: dict[str, np.ndarray] | None = None
    land: dict[str, np.ndarray] | None = None
    forest: dict[str, np.ndarray] | None = None
    forest_b: dict[str, np.ndarray] | None = None
    lossyear: dict[str, np.ndarray] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None


def _hansen_template(template: str | None) -> str:
    """The Hansen tile template: the given one, else :data:`validate.HANSEN_URL_TEMPLATE` read now."""
    return validate.HANSEN_URL_TEMPLATE if template is None else template


def _forest_agreement(scores: validate.BinaryScores) -> dict[str, Any]:
    rn = None if scores.tn + scores.fp == 0 else scores.tn / (scores.tn + scores.fp)
    return {
        "accuracy": scores.accuracy,
        "recall_forest": scores.recall,
        "recall_nonforest": rn,
        "precision_forest": scores.precision,
        "pixels": scores.n,
        "scores": scores,
    }


def _labels_caveats(label_info: Mapping[str, Any]) -> list[str]:
    if label_info.get("source") == "hansen":
        ring = label_info.get("ring") or {}
        return [
            "The classifier's labels came from Hansen (treecover2000 and lossyear), so agreement with Hansen is not "
            "independent of the reference. They were sampled outside the AOI only, in a ring "
            f"{ring.get('inner_km', 0):g} to {ring.get('outer_km', 0):g} km from it, so the comparison inside the "
            "AOI is spatially held out."
        ]
    inside = int(label_info.get("inside_aoi") or 0)
    if inside:
        return [
            f"{inside} of the supplied labels lie inside the AOI, so the comparison there is not fully held out from "
            "the training labels."
        ]
    return ["The labels were supplied with --labels and lie outside the AOI."]


# --------------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DeforestResult:
    """What :func:`run_deforest` did.

    Attributes:
        report: The report (read back from disk when the run was skipped).
        paths: The standard outputs.
        extra_paths: Outputs beyond the standard set, by name (``labels``).
        skipped: The outputs were current, so nothing was read or written.
        validation_written: ``deforest_validation.json`` belongs to this report.
        validation_redone: The outputs were current except the validation, which was redone from
            them without reading a chunk.
    """

    report: dict[str, Any]
    paths: OutputPaths
    extra_paths: dict[str, Path]
    skipped: bool
    validation_written: bool
    validation_redone: bool = False

    @property
    def summary(self) -> dict[str, Any]:
        """The report's summary block."""
        return dict(self.report.get("summary") or {})


@dataclasses.dataclass(frozen=True)
class _Options:
    labels_frame: pd.DataFrame | None
    labels_path: Path | None
    labels_per_class: int
    label_chunks: int
    ring_km: float
    ring_inner_km: float
    forest_cover: int
    nonforest_cover: int
    forest_prob: float
    stable_percentile: float
    min_area_ha: float
    hansen_version: str
    hansen_template: str
    reference: Path | None
    strict_validate: bool
    max_zones: int


@dataclasses.dataclass
class _State:
    labels: pd.DataFrame | None = None
    label_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    ref: _Reference = dataclasses.field(default_factory=_Reference)


def _check_args(
    *,
    year_a: int,
    year_b: int,
    labels_per_class: int,
    label_chunks: int,
    ring_km: float,
    ring_inner_km: float,
    forest_cover: float,
    nonforest_cover: float,
    forest_prob: float,
    stable_percentile: float,
    min_area_ha: float,
    hansen_version: str,
    hansen_labels: bool,
    hansen_validation: bool,
    reference: Path | None,
) -> None:
    """Refuse arguments the app cannot use, before the store is opened.

    Raises:
        AppInputError: Says what to change.
    """
    if year_b <= year_a:
        raise AppInputError(f"year_b ({year_b}) must be after year_a ({year_a}).")
    if year_a < FIRST_YEAR or year_b > LAST_YEAR:
        raise AppInputError(f"Years must lie in {FIRST_YEAR} to {LAST_YEAR}; got year_a {year_a}, year_b {year_b}.")
    if labels_per_class < MIN_LABELS_PER_CLASS:
        raise AppInputError(
            f"labels_per_class ({labels_per_class}) must be at least {MIN_LABELS_PER_CLASS}, the fewest labels per "
            "class the classifier is trained on."
        )
    if label_chunks < 1:
        raise AppInputError(f"label_chunks ({label_chunks}) must be at least 1.")
    if not (math.isfinite(ring_km) and math.isfinite(ring_inner_km) and 0.0 <= ring_inner_km < ring_km <= MAX_RING_KM):
        raise AppInputError(
            f"The label ring needs 0 <= inner edge < outer edge <= {MAX_RING_KM:g} km; got {ring_inner_km:g} and "
            f"{ring_km:g} km (--label-ring-km)."
        )
    if not (0 <= nonforest_cover <= forest_cover <= 100):
        raise AppInputError(
            f"Cover cuts must satisfy 0 <= nonforest_cover ({nonforest_cover}) <= forest_cover ({forest_cover}) <= 100."
        )
    if not (math.isfinite(forest_prob) and 0.0 < forest_prob < 1.0):
        raise AppInputError(f"forest_prob ({forest_prob}) must lie strictly between 0 and 1.")
    if not (math.isfinite(stable_percentile) and 0.0 < stable_percentile <= 100.0):
        raise AppInputError(f"stable_percentile ({stable_percentile}) must lie in (0, 100].")
    if not (math.isfinite(min_area_ha) and min_area_ha >= 0.0):
        raise AppInputError(f"min_area_ha ({min_area_ha}) must be zero or more.")
    if reference is not None and not reference.is_file():
        raise AppInputError(
            f"Reference file {reference} does not exist; pass a vector file or a GeoTIFF with --reference."
        )
    try:
        last = validate._hansen_last_year(hansen_version)
    except ValueError as exc:
        raise AppInputError(f"{exc} Pass --hansen-version like {validate.HANSEN_VERSION}.") from exc
    if hansen_labels and year_a > last:
        raise AppInputError(
            f"Hansen {hansen_version} records loss through {last}, so it cannot label year_a {year_a}; pass a newer "
            "--hansen-version or your own --labels."
        )
    if hansen_validation and year_b > last:
        raise AppInputError(
            f"Hansen {hansen_version} records loss through {last}, so it cannot validate year_b {year_b}; pass a newer "
            "--hansen-version or a --reference."
        )


def run_deforest(
    aoi: str | Path | LoadedAOI,
    year_a: int,
    year_b: int,
    out_dir: Path | str,
    *,
    validate: bool = False,
    reference: Path | str | None = None,
    labels: Path | str | None = None,
    labels_per_class: int = DEFAULT_LABELS_PER_CLASS,
    label_chunks: int = DEFAULT_LABEL_CHUNKS,
    label_ring_km: float = DEFAULT_RING_KM,
    label_ring_inner_km: float = 0.0,
    forest_cover: int = DEFAULT_FOREST_COVER,
    nonforest_cover: int = DEFAULT_NONFOREST_COVER,
    forest_prob: float = DEFAULT_FOREST_PROB,
    stable_percentile: float = DEFAULT_STABLE_PERCENTILE,
    min_area_ha: float = DEFAULT_MIN_AREA_HA,
    hansen_version: str = validate.HANSEN_VERSION,
    hansen_url_template: str | None = None,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    max_area_km2: float | None = io.MAX_AOI_KM2,
    max_zones: int = 4,
    snapshot_id: str | None = None,
    reader: Reader | None = None,
    block: tuple[int, int] = io.DEFAULT_BLOCK,
    strict_validate: bool = False,
) -> DeforestResult:
    """Map forest in ``year_a`` and ``year_b`` and the forest that became something else in between.

    Writes to ``out_dir``: ``deforest_<a>_<b>.tif`` (float32 COG, NaN nodata, mosaicked across zones:
    ``forest_prob_<a>``, ``forest_prob_<b>``, ``distance_<a>_<b>`` and ``converted``, which is 1 or 0
    on pixels classified in both years and NaN elsewhere), ``deforest_summary.parquet`` and ``.csv``
    (one row per year, the conversion columns repeated so the csv stands alone),
    ``deforest_labels.parquet`` (every sampled label: where it is, its class, source chunk, radar
    regime, whether it was used and its held-out fold), ``deforest_preview.png``,
    ``deforest_validation.json`` when validating, and ``deforest_report.json`` last. A rerun with
    the same arguments skips (``skipped``) unless ``force``; a validating rerun whose outputs are
    current except ``validation.json`` redoes only the validation, from the outputs on disk.

    Args:
        aoi: AOI file (any CRS), demo name (``rondonia``) or a loaded AOI.
        year_a: First year; the labels describe it.
        year_b: Second year, after ``year_a``.
        out_dir: Output directory.
        validate: Score the conversion against Hansen loss in ``(year_a, year_b]``, or ``reference``.
        reference: A vector file or GeoTIFF of conversion to validate against instead of Hansen.
        labels: A CSV of lon, lat, label (1 forest, 0 non-forest) and an optional weight, used
            instead of the Hansen labels.
        labels_per_class: Most Hansen label pixels per class.
        label_chunks: Most store chunks the Hansen labels are sampled from.
        label_ring_km: Outer edge of the label ring, from the AOI boundary.
        label_ring_inner_km: Inner edge of the label ring (0: the AOI boundary).
        forest_cover: Lowest ``treecover2000`` of a forest label (percent).
        nonforest_cover: ``treecover2000`` below which a pixel is a non-forest label (percent).
        forest_prob: Forest probability at or above which a pixel is forest.
        stable_percentile: Percentile of the stable forest distance that sets the cross-check cut.
        min_area_ha: Conversion patches smaller than this are dropped after a 3 x 3 opening.
        hansen_version: Hansen GFC version for the labels and the validation.
        hansen_url_template: Hansen tile URL or local path template; ``validate.HANSEN_URL_TEMPLATE``
            when None.
        seed: Seed of the label sampling and the classifier.
        force: Run even when the outputs are current.
        max_area_km2: AOI size limit; None lifts it.
        max_zones: Zone group limit (for the AOI and for the label ring).
        snapshot_id: Icechunk snapshot to pin.
        reader: Reader to use instead of opening the live store.
        block: Block shape of the streaming reads.
        strict_validate: With ``validate``, stop before any chunk read when the reference cannot be
            loaded, instead of running without validation.

    Returns:
        The report, output paths and whether the run was skipped.

    Raises:
        AppInputError: A bad argument, AOI file, labels file, reference file or output directory,
            before the store is opened or before any chunk read.
        AOIError: The AOI is too large, touches too many zones, or covers no pixel.
        YearsNotPublishedError: A year the run needs is missing for some zone (before any chunk read).
        StrictValidationError: ``strict_validate`` is set and the reference cannot be loaded (before
            any chunk read).
        DeforestError: No usable label ring or Hansen labels (before any chunk read), too few usable
            labels once read (before the AOI is streamed), or no comparable AOI pixel.
    """
    ref_path = None if reference is None else Path(reference)
    labels_path = None if labels is None else Path(labels)
    template = _hansen_template(hansen_url_template)
    _check_args(
        year_a=year_a,
        year_b=year_b,
        labels_per_class=labels_per_class,
        label_chunks=label_chunks,
        ring_km=label_ring_km,
        ring_inner_km=label_ring_inner_km,
        forest_cover=forest_cover,
        nonforest_cover=nonforest_cover,
        forest_prob=forest_prob,
        stable_percentile=stable_percentile,
        min_area_ha=min_area_ha,
        hansen_version=hansen_version,
        hansen_labels=labels_path is None,
        hansen_validation=validate and ref_path is None,
        reference=ref_path,
    )
    labels_frame = None if labels_path is None else read_labels_csv(labels_path)
    loaded = load_aoi(aoi)
    out = Path(out_dir)
    paths = export.output_paths(out, APP, year_a, year_b)
    extras = {"labels": out / f"{APP}_labels.parquet"}
    params = {
        "year_a": year_a,
        "year_b": year_b,
        "labels": None
        if labels_path is None
        else {"file": labels_path.name, "sha256": export.file_sha256(labels_path)},
        "labels_per_class": labels_per_class,
        "label_chunks": label_chunks,
        "label_ring_km": label_ring_km,
        "label_ring_inner_km": label_ring_inner_km,
        "forest_cover": forest_cover,
        "nonforest_cover": nonforest_cover,
        "forest_prob": forest_prob,
        "stable_percentile": stable_percentile,
        "min_area_ha": min_area_ha,
        "hansen_version": hansen_version,
        "hansen_url_template": template,
        "reference": None if ref_path is None else {"file": ref_path.name, "sha256": export.file_sha256(ref_path)},
        "block": list(block),
        "method": {
            "min_labels_per_class": MIN_LABELS_PER_CLASS,
            "cv_folds": CV_FOLDS,
            "max_iter": MAX_ITER,
            "logistic_c": LOGISTIC_C,
            "regime_probe_blocks": REGIME_PROBE_BLOCKS,
            "opening": 3,
            "patch_bin_edges_ha": list(PATCH_BIN_EDGES_HA),
        },
    }
    opts = _Options(
        labels_frame=labels_frame,
        labels_path=labels_path,
        labels_per_class=labels_per_class,
        label_chunks=label_chunks,
        ring_km=label_ring_km,
        ring_inner_km=label_ring_inner_km,
        forest_cover=forest_cover,
        nonforest_cover=nonforest_cover,
        forest_prob=forest_prob,
        stable_percentile=stable_percentile,
        min_area_ha=min_area_ha,
        hansen_version=hansen_version,
        hansen_template=template,
        reference=ref_path,
        strict_validate=strict_validate,
        max_zones=max_zones,
    )
    state = _State()
    ctx = prepare_run(
        APP,
        loaded,
        [year_a, year_b],
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
        # The stream reads both years of every block; the radar regime probe reads a fraction of a
        # year (scales and radar counts of a sample of blocks). The label chunks are added by the
        # preflight, once they are known.
        year_reads=3,
        code_modules=(sys.modules[__name__],),
        memory_estimate=lambda plan: memory_estimate(
            plan, block=block, ring_km=None if labels_frame is not None else label_ring_km
        ),
        allow_validation_only=True,
        preflight=lambda c: _preflight(c, state, opts),
    )
    if ctx is None:
        report = json.loads(paths.report.read_text(encoding="utf-8"))
        return DeforestResult(
            report=report,
            paths=paths,
            extra_paths=extras,
            skipped=True,
            validation_written="validation" in (report.get("outputs") or {}),
        )
    if ctx.validation_only:
        return _redo_validation(ctx, state, opts)
    return _run(ctx, state, opts)


# --------------------------------------------------------------------------------------------
# Preflight: labels and the validation reference, before any chunk read
# --------------------------------------------------------------------------------------------


def _preflight(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Place the labels and load the validation reference, reading metadata and Hansen only."""
    if not ctx.validation_only:
        if opts.labels_frame is not None:
            _prepare_csv_labels(ctx, state, opts)
        else:
            _prepare_hansen_labels(ctx, state, opts)
        assert state.labels is not None
        located = state.labels[state.labels["chunk_row"] >= 0]
        chunks = located[["zone", "chunk_row", "chunk_col"]].drop_duplicates()
        label_bytes = 0
        for zone, count in chunks["zone"].value_counts().items():
            ch, cw = ctx.reader.chunk_shape(str(zone))
            label_bytes += int(count) * ch * cw * io.BYTES_PER_PIXEL
        # year_a trains the classifier; with Hansen labels year_b is read at the same chunks for the
        # transfer check and the ring label cut.
        years = [ctx.year_a, ctx.year_b] if state.label_info.get("source") == "hansen" else [ctx.year_a]
        ctx.estimated_bytes += label_bytes * len(years)
        state.label_info["label_read_chunks"] = len(chunks)
        state.label_info["label_read_years"] = years
        logger.info(
            "%s: %d labels (%s) in %d chunks, at most %.2f GB of chunks to read for them (%s).",
            APP,
            len(state.labels),
            state.label_info.get("source"),
            len(chunks),
            label_bytes * len(years) / 1e9,
            ", ".join(str(y) for y in years),
        )
    _load_reference(ctx, state.ref, opts)


def _prepare_hansen_labels(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Build the label ring, read its Hansen layers and cluster-sample the labels.

    Raises:
        DeforestError: The ring is empty or unusable, Hansen cannot be loaded or holds no land pixel
            in the ring (its tiles were not found, or the ring is water), or the ring (or the sampled
            chunks) hold too few label pixels of a class.
        YearsNotPublishedError: ``year_a`` is missing for a zone the ring touches; a note names the
            ring as the reason and what to change.
    """
    ring = label_ring(ctx.aoi.geometry, outer_km=opts.ring_km, inner_km=opts.ring_inner_km)
    ring_plan = plan_label_ring(ctx.reader, ring, ctx.plan, max_zones=opts.max_zones)
    try:
        io.reader_check_years(ctx.reader, ring_plan.zone_names, [ctx.year_a])
    except io.YearsNotPublishedError as exc:
        only_ring = sorted(set(exc.missing) - set(ctx.plan.zone_names))
        if only_ring:
            names = ", ".join(only_ring)
            what = f"Zone {names} is" if len(only_ring) == 1 else f"Zones {names} are"
            it = "it" if len(only_ring) == 1 else "them"
            exc.add_note(
                f"{what} read only for the Hansen labels: the label ring ({ring.describe()}) reaches {it} and the "
                f"AOI does not. Narrow --label-ring-km so the ring stays in zones that list {ctx.year_a}, or pass "
                "--labels."
            )
        raise
    maps: list[ClassMap] = []
    land = 0
    kw = {"version": opts.hansen_version, "url_template": opts.hansen_template}
    try:
        for z in ring_plan.zones:
            tc = validate.hansen_layer(z, "treecover2000", **kw)
            ly = validate.hansen_layer(z, "lossyear", **kw)
            dm = validate.hansen_layer(z, "datamask", **kw)
            classes = label_classes(
                tc, ly, dm, ctx.year_a, forest_cover=opts.forest_cover, nonforest_cover=opts.nonforest_cover
            )
            classes[~z.mask] = NO_LABEL
            land += int(np.count_nonzero((dm == 1) & z.mask))
            del tc, dm
            maps.append(
                ClassMap(
                    zone=z.zone,
                    grid=z.grid,
                    row0=z.rows.start,
                    col0=z.cols.start,
                    classes=classes,
                    chunk_shape=ctx.reader.chunk_shape(z.zone),
                    lossyear=ly,
                )
            )
            del ly
    except validate.ReferenceDataError as exc:
        raise DeforestError(
            f"Hansen {opts.hansen_version} could not be loaded for the labels: {exc} Pass --labels with your own "
            "points instead."
        ) from exc
    if land == 0:
        raise DeforestError(
            f"Hansen {opts.hansen_version} has no land pixel (datamask 1) in the label ring ({ring.describe()}) from "
            f"{opts.hansen_template}: its tiles were not found there (check --hansen-version) or the ring is all "
            "water. Pass --labels with your own points instead."
        )
    labels, info = sample_hansen_labels(
        maps, max_chunks=opts.label_chunks, per_class=opts.labels_per_class, seed=ctx.seed
    )
    del maps
    info = {
        "source": "hansen",
        "hansen_version": opts.hansen_version,
        "forest_cover": opts.forest_cover,
        "nonforest_cover": opts.nonforest_cover,
        "labels_per_class": opts.labels_per_class,
        "label_chunks": opts.label_chunks,
        "ring": {
            **ring.record(),
            "zones": ring_plan.zone_names,
            "pixels": ring_plan.n_pixels,
            "area_ha": ring_plan.area_ha,
            "land_pixels": land,
        },
        **info,
    }
    need = MIN_LABELS_PER_CLASS
    if min(info["ring_pixels_forest"], info["ring_pixels_nonforest"]) < need:
        raise DeforestError(
            f"The label ring ({ring.describe()}) holds {info['ring_pixels_forest']:,} forest and "
            f"{info['ring_pixels_nonforest']:,} non-forest Hansen label pixels on {land:,} land pixels; the "
            f"classifier needs at least {need} of each. Widen --label-ring-km, change --forest-cover or "
            "--nonforest-cover, or pass --labels."
        )
    if min(info["sampled_forest"], info["sampled_nonforest"]) < need:
        raise DeforestError(
            f"The {info['chunks_used']} ring chunks sampled hold {info['sampled_forest']} forest and "
            f"{info['sampled_nonforest']} non-forest labels; the classifier needs at least {need} of each. Raise "
            "--label-chunks, or pass --labels."
        )
    state.labels = labels
    state.label_info = info


def _prepare_csv_labels(ctx: RunContext, state: _State, opts: _Options) -> None:
    """Place the CSV labels on the store grid and check ``year_a`` for every zone they fall in.

    Raises:
        YearsNotPublishedError: ``year_a`` is missing for a zone some label falls in; a note per
            zone says how many of the file's labels lie there and what to change.
    """
    assert opts.labels_frame is not None and opts.labels_path is not None
    frame = csv_label_points(ctx.reader, opts.labels_frame)
    zones = sorted(set(frame["zone"]))
    try:
        io.reader_check_years(ctx.reader, zones, [ctx.year_a])
    except io.YearsNotPublishedError as exc:
        for zone in sorted(exc.missing):
            n = int((frame["zone"] == zone).sum())
            if n and zone not in ctx.plan.zone_names:
                exc.add_note(
                    f"Zone {zone} holds {n} of the {len(frame)} labels in {opts.labels_path.name} and does not list "
                    f"{ctx.year_a}; the AOI does not touch it. Drop those labels or choose other years."
                )
        raise
    inside = int(shapely.contains_xy(ctx.aoi.geometry, frame["lon"].to_numpy(), frame["lat"].to_numpy()).sum())
    state.labels = frame
    state.label_info = {
        "source": "csv",
        "file": opts.labels_path.name,
        "sha256": export.file_sha256(opts.labels_path),
        "rows": len(frame),
        "rows_forest": int((frame["label"] == FOREST).sum()),
        "rows_nonforest": int((frame["label"] == NONFOREST).sum()),
        "zones": zones,
        "outside_store_grid": int((frame["row"] < 0).sum()),
        "inside_aoi": inside,
    }
    if inside:
        ctx.warn(
            f"{inside} of the {len(frame)} labels in {opts.labels_path.name} lie inside the AOI, so the validation "
            "there is not fully held out from the training labels."
        )


def _load_reference(ctx: RunContext, ref: _Reference, opts: _Options) -> None:
    """Load the validation reference on every AOI zone plan: the preflight, before any chunk read.

    A Hansen reference with no land pixel over the AOI counts as not loaded: a missing tile reads
    as ocean, so a ``hansen_version`` the bucket does not hold, or a template that points at no
    file, would otherwise score nothing without a word.

    Raises:
        AppInputError: The ``--reference`` file cannot be read.
        StrictValidationError: ``strict_validate`` is set and Hansen cannot be loaded or holds no
            land pixel over the AOI.
    """
    if not ctx.validate:
        return
    plan = ctx.plan
    if opts.reference is not None:
        ref.window = {z.zone: aoi_reference(z, opts.reference) for z in plan.zones}
        ref.info = {
            "source": "file",
            "file": opts.reference.name,
            "sha256": export.file_sha256(opts.reference),
        }
        return
    a, b = ctx.year_a, ctx.year_b
    kw = {"version": opts.hansen_version, "url_template": opts.hansen_template}
    try:
        window, b_only, land, forest, forest_b, lossyears = {}, {}, {}, {}, {}, {}
        for z in plan.zones:
            window[z.zone] = validate.hansen_loss(z, a, b, **kw)
            lossyear = validate.hansen_layer(z, "lossyear", **kw)
            b_only[z.zone] = (lossyear == b - 2000) & z.mask
            lossyears[z.zone] = np.where(z.mask, lossyear, np.uint8(0)).astype(np.uint8)
            del lossyear
            land[z.zone] = validate.hansen_valid(z, **kw)
            forest[z.zone] = validate.hansen_forest(z, a, threshold=opts.forest_cover, **kw)
            forest_b[z.zone] = validate.hansen_forest(z, b, threshold=opts.forest_cover, **kw)
        if not any(m.any() for m in land.values()):
            raise validate.ReferenceDataError(
                f"Hansen {opts.hansen_version} has no land pixel (datamask 1) over the AOI from "
                f"{opts.hansen_template}: its tiles were not found there (check --hansen-version) or the AOI is all "
                "water."
            )
    except validate.ReferenceDataError as exc:
        if opts.strict_validate:
            raise StrictValidationError(
                f"--strict-validate is set and the validation reference could not be loaded: {exc}"
            ) from exc
        ref.error = str(exc)
        logger.warning("Validation will be skipped, the reference could not be loaded: %s", exc)
        return
    ref.window, ref.year_b_only, ref.land, ref.lossyear = window, b_only, land, lossyears
    ref.forest, ref.forest_b = forest, forest_b
    ref.info = {
        "source": "hansen",
        "dataset": "Hansen Global Forest Change",
        "version": opts.hansen_version,
        "url_template": opts.hansen_template,
        "loss_window": {
            "after_year": a,
            "through_year": b,
            "lossyear_codes": list(range(a - 2000 + 1, b - 2000 + 1)),
            "rule": "lossyear > year_a - 2000 and lossyear <= year_b - 2000",
        },
        "lag_aligned_window": {
            "from_year": a,
            "through_year": b - 1,
            "lossyear_codes": list(range(a - 2000, b - 2000)),
            "rule": "lossyear >= year_a - 2000 and lossyear <= year_b - 2001",
        },
        "year_b_only_code": b - 2000,
        "land": "datamask == 1",
        "forest_proxy": {"year": a, "treecover2000_at_least": opts.forest_cover, "no_loss_through": a},
        "forest_proxy_year_b": {"year": b, "treecover2000_at_least": opts.forest_cover, "no_loss_through": b},
    }


# --------------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------------


def _probe_regime(ctx: RunContext) -> dict[str, Any]:
    """The training regime from the scales and radar counts of a sample of AOI blocks (survey B5).

    Up to :data:`REGIME_PROBE_BLOCKS` blocks spread over the AOI are read for both years (scales and
    the two radar counts, about 6 percent of a full read); when none of them has a pixel valid in
    both years with the same regime, the remaining blocks are read until one does.

    Raises:
        DeforestError: No AOI pixel is valid in both years with the same regime.
    """
    a, b = ctx.year_a, ctx.year_b
    jobs = [(z, rs, cs) for z in ctx.plan.zones for rs, cs in z.blocks(ctx.reader.chunk_shape(z.zone), ctx.block)]
    first = spread_indices(len(jobs), REGIME_PROBE_BLOCKS)
    chosen = set(first)
    order = first + [i for i in range(len(jobs)) if i not in chosen]
    radar = optical = mismatch = used = 0
    for count, i in enumerate(order):
        if count >= len(first) and radar + optical > 0:
            break
        z, rs, cs = jobs[i]
        wins = ctx.reader.read_windows(z.zone, [a, b], rs, cs, layers=PROBE_LAYERS)
        ctx.mark_read([a, b])
        m = z.submask(rs, cs)
        wa, wb = wins[a], wins[b]
        both = _valid(wa) & _valid(wb) & m
        ra, rb = wa.radar_present, wb.radar_present
        radar += int((both & ra & rb).sum())
        optical += int((both & ~ra & ~rb).sum())
        mismatch += int((both & (ra != rb)).sum())
        used += 1
    train = training_regime(radar, optical)
    if train is None:
        raise DeforestError(
            f"No AOI pixel has data in both {a} and {b} with the same radar regime ({mismatch:,} switch regime); "
            "choose other years or another AOI."
        )
    return {
        "blocks_read": used,
        "blocks_total": len(jobs),
        "pixels_radar": radar,
        "pixels_optical": optical,
        "pixels_mismatch": mismatch,
        "training_radar": train,
    }


@dataclasses.dataclass
class _Zone:
    """Per-pixel results on one zone plan's grid, filled block by block."""

    plan: ZonePlan
    prob_a: np.ndarray
    prob_b: np.ndarray
    dist: np.ndarray


def _stream(ctx: RunContext, model: ForestModel, train_radar: bool) -> tuple[dict[str, _Zone], dict[str, int]]:
    """One pass over every block of every zone, reading both years at once."""
    a, b = ctx.year_a, ctx.year_b
    totals = dict.fromkeys(BLOCK_COUNTS, 0)
    zones: dict[str, _Zone] = {}
    for zp in ctx.plan.zones:
        shape = zp.shape
        st = _Zone(
            plan=zp,
            prob_a=np.full(shape, np.nan, dtype=np.float32),
            prob_b=np.full(shape, np.nan, dtype=np.float32),
            dist=np.full(shape, np.nan, dtype=np.float32),
        )
        for blk in io.iter_plan_blocks(ctx.reader, zp, [a, b], layers=LAYERS, block=ctx.block):
            ctx.mark_read([a, b])
            r = slice(blk.rows.start - zp.rows.start, blk.rows.stop - zp.rows.start)
            c = slice(blk.cols.start - zp.cols.start, blk.cols.stop - zp.cols.start)
            pa, pb, d, counts = classify_block(blk.windows[a], blk.windows[b], blk.mask, model, train_radar)
            st.prob_a[r, c] = pa
            st.prob_b[r, c] = pb
            st.dist[r, c] = d
            for k, v in counts.items():
                totals[k] += v
        zones[zp.zone] = st
    return zones, totals


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


def _forest_masks(prob: Mapping[str, np.ndarray], forest_prob: float) -> dict[str, np.ndarray]:
    out = {}
    for z, p in prob.items():
        with np.errstate(invalid="ignore"):
            out[z] = np.isfinite(p) & (p >= forest_prob)
    return out


def _window_caveat(
    year_a: int,
    year_b: int,
    codes: Sequence[int],
    lag_codes: Sequence[int],
    breakdown: Mapping[str, Mapping[str, float | int]],
    headline_iou: float | None,
    lag_iou: float | None,
) -> str:
    """Which Hansen loss window this run's embeddings line up with, measured rather than assumed.

    An annual embedding can see loss Hansen dates to ``year_a`` (cleared early that year) or miss
    loss Hansen dates to ``year_b`` (cleared late that year); which effect wins depends on the place
    and the clearing season, so the caveat reports the converted share of each end year and the
    IoU of both windows instead of asserting one.
    """

    def share(year: int) -> str:
        row = breakdown.get(str(year)) or {}
        value = row.get("converted_share")
        return "n/a" if value is None else f"{float(value):.1%}"

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    if headline_iou is not None and lag_iou is not None:
        better = (
            "the headline window" if headline_iou >= lag_iou else f"the lag-aligned window ({year_a} to {year_b - 1})"
        )
        verdict = f"Here {better} scores higher (IoU {fmt(headline_iou)} against {fmt(lag_iou)})."
    else:
        verdict = "The two windows could not both be scored here."
    return (
        f"The headline reference is Hansen loss after {year_a} through {year_b} (lossyear codes {codes[0]} to "
        f"{codes[-1]}), as survey S13 specifies. Annual embeddings and Hansen's detection year need not line up: "
        f"loss Hansen dates to {year_a} may already read as cleared in the {year_a} embedding, and loss it dates to "
        f"{year_b} may still read as forest in the {year_b} embedding. In this run the app converted {share(year_a)} "
        f"of Hansen {year_a} loss pixels and {share(year_b)} of Hansen {year_b} loss pixels. {verdict} "
        f"conversion_lag_aligned scores the same conversion against loss in {year_a} through {year_b - 1} (codes "
        f"{lag_codes[0]} to {lag_codes[-1]}) and loss_year_breakdown gives the share converted for every loss year. "
        f"The brief's comparison with loss in {year_b} alone (conversion_year_b_only) counts conversions in the "
        "earlier years of the window as false positives, so its precision is lower by construction."
    )


def _validate(
    ctx: RunContext,
    state: _State,
    *,
    converted: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
    forest_a: Mapping[str, np.ndarray],
    forest_b: Mapping[str, np.ndarray],
    classified_a: Mapping[str, np.ndarray],
    classified_b: Mapping[str, np.ndarray],
    below_cut: Mapping[str, np.ndarray],
    extra_caveats: Sequence[str] = (),
) -> tuple[dict[str, Any], bool]:
    """Score, write ``validation.json`` and return the summary's validation block and whether it was written.

    Args:
        ctx: The run.
        state: Holds the reference and the label record.
        converted: The cleaned conversion per zone.
        classified: Pixels classified in both years (with a distance) per zone.
        forest_a: The ``year_a`` forest map per zone.
        forest_b: The ``year_b`` forest map per zone.
        classified_a: Pixels classified in ``year_a`` per zone.
        classified_b: Pixels classified in ``year_b`` per zone.
        below_cut: Classifier flips the cross-check cut removed, per zone.
        extra_caveats: Caveats to add.
    """
    ref = state.ref
    if ref.window is None or ref.info is None:
        ctx.warn(f"{VALIDATION_SKIPPED}: {ref.error}")
        return {"error": ref.error}, False
    plan = ctx.plan
    a, b = ctx.year_a, ctx.year_b
    merged, parts = score_against(plan, converted, classified, ref.window, coverage=ref.land)
    scores: dict[str, Any] = {"conversion": merged, "conversion_per_zone": parts}
    n_ref = sum(int((ref.window[z.zone] & z.mask).sum()) for z in plan.zones)
    n_ref_out = sum(
        int((ref.window[z.zone] & z.mask & ~np.asarray(classified[z.zone], dtype=bool)).sum()) for z in plan.zones
    )
    summary: dict[str, Any] = {
        "source": ref.info["source"],
        "window": None,
        "iou": merged.iou,
        "precision": merged.precision,
        "recall": merged.recall,
        "f1": merged.f1,
        "pixels_scored": merged.n,
        "reference_pixels_scored": merged.tp + merged.fn,
        "reference_pixels_in_aoi": n_ref,
        "reference_pixels_unclassified": n_ref_out,
    }
    caveats = [SCOPE_CAVEAT]
    if ref.info["source"] == "hansen":
        assert ref.year_b_only is not None and ref.forest is not None and ref.land is not None
        assert ref.lossyear is not None and ref.forest_b is not None
        codes = ref.info["loss_window"]["lossyear_codes"]
        lag_codes = ref.info["lag_aligned_window"]["lossyear_codes"]
        lag = {
            z.zone: (ref.lossyear[z.zone] >= a - 2000) & (ref.lossyear[z.zone] <= b - 2001) & z.mask for z in plan.zones
        }
        l_merged, l_parts = score_against(plan, converted, classified, lag, coverage=ref.land)
        b_merged, _ = score_against(plan, converted, classified, ref.year_b_only, coverage=ref.land)
        agreement = _forest_agreement(score_against(plan, forest_a, classified_a, ref.forest, coverage=ref.land)[0])
        agreement_b = _forest_agreement(score_against(plan, forest_b, classified_b, ref.forest_b, coverage=ref.land)[0])
        names = [z.zone for z in plan.zones]
        scored = [np.asarray(classified[z], dtype=bool) & ref.land[z] for z in names]
        breakdown = loss_year_breakdown(
            [ref.lossyear[z] for z in names],
            scored,
            [converted[z] for z in names],
            [np.asarray(forest_a[z], dtype=bool) & ~np.asarray(forest_b[z], dtype=bool) for z in names],
            [below_cut[z] for z in names],
        )
        removed = cross_check_removed(
            [ref.lossyear[z] for z in names], scored, [below_cut[z] for z in names], ctx.year_a, ctx.year_b
        )
        scores["conversion_lag_aligned"] = l_merged
        scores["conversion_lag_aligned_per_zone"] = l_parts
        scores["conversion_year_b_only"] = b_merged
        scores["forest_agreement_year_a"] = agreement
        scores["forest_agreement_year_b"] = agreement_b
        scores["loss_year_breakdown"] = breakdown
        scores["cross_check_removed"] = removed
        summary["window"] = {"after_year": a, "through_year": b, "lossyear_codes": codes}
        summary["lag_aligned"] = {
            "window": {"from_year": a, "through_year": b - 1, "lossyear_codes": lag_codes},
            "iou": l_merged.iou,
            "precision": l_merged.precision,
            "recall": l_merged.recall,
            "f1": l_merged.f1,
            "reference_pixels_scored": l_merged.tp + l_merged.fn,
        }
        summary["loss_year_breakdown"] = breakdown
        summary["cross_check_removed"] = removed
        summary["year_b_only"] = {
            "precision": b_merged.precision,
            "recall": b_merged.recall,
            "reference_pixels_scored": b_merged.tp + b_merged.fn,
        }
        keys = ("accuracy", "recall_forest", "recall_nonforest", "pixels")
        summary["forest_agreement_year_a"] = {k: agreement[k] for k in keys}
        summary["forest_agreement_year_b"] = {k: agreement_b[k] for k in keys}
        caveats += [
            HANSEN_MODEL_CAVEAT,
            NON_CONVERSION_CAVEAT,
            _window_caveat(a, b, codes, lag_codes, breakdown, merged.iou, l_merged.iou),
            FOREST_AGREEMENT_CAVEAT,
            TIMING_CAVEAT,
            CROSS_CHECK_CAVEAT,
            f"Reference: Hansen Global Forest Change {ref.info['version']}, tiles from {ref.info['url_template']}.",
        ]
    else:
        caveats += [
            FILE_CAVEAT,
            f"Reference: the file {ref.info['file']} (sha256 {ref.info['sha256']}), supplied with --reference.",
        ]
    caveats += _labels_caveats(state.label_info)
    caveats += list(extra_caveats)
    record = {**ref.info, "label_source": state.label_info.get("source")}
    validate.write_validation_json(ctx.paths.validation, scores=scores, reference=record, caveats=caveats)
    return summary, True


def score_against(
    plan: AOIPlan,
    pred: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    *,
    coverage: Mapping[str, np.ndarray] | None = None,
) -> tuple[validate.BinaryScores, dict[str, validate.BinaryScores]]:
    """Scores of a mask against a reference over ``scored`` AOI pixels (inside ``coverage``), merged over zones.

    Args:
        plan: The AOI plan.
        pred: The predicted mask per zone (conversion, or the ``year_a`` forest map).
        scored: Pixels that count per zone (those the app classified).
        reference: The reference mask per zone.
        coverage: Where the reference is defined (Hansen land); everywhere when None.

    Returns:
        ``(merged, per_zone)``.
    """
    parts: dict[str, validate.BinaryScores] = {}
    for z in plan.zones:
        valid = np.asarray(scored[z.zone], dtype=bool) & z.mask
        if coverage is not None:
            valid &= np.asarray(coverage[z.zone], dtype=bool)
        parts[z.zone] = validate.binary_scores(
            np.asarray(pred[z.zone], dtype=bool), np.asarray(reference[z.zone], dtype=bool), valid=valid
        )
    return validate.merge_scores(list(parts.values())), parts


def loss_year_breakdown(
    lossyear: np.ndarray | Sequence[np.ndarray],
    scored: np.ndarray | Sequence[np.ndarray],
    converted: np.ndarray | Sequence[np.ndarray],
    flips: np.ndarray | Sequence[np.ndarray],
    below_cut: np.ndarray | Sequence[np.ndarray] | None = None,
) -> dict[str, dict[str, float | int]]:
    """For each Hansen loss year, the share of its scored pixels converted, flipped, and removed by the cut.

    A flip is forest in ``year_a`` and non-forest in ``year_b`` before the cross-check and the
    cleaning; ``below_cut`` marks the flips the cross-check cut removed. Arrays may be given per
    zone as sequences, which are pooled.

    Returns:
        ``{"no_loss" or "<year>": {"pixels", "converted_share", "flip_share"}}`` for every loss code
        among the scored pixels, in code order, plus ``below_cut_share`` when ``below_cut`` is given.
    """
    pixels = np.zeros(256, dtype=np.int64)
    conv = np.zeros(256, dtype=np.int64)
    flip = np.zeros(256, dtype=np.int64)
    below = np.zeros(256, dtype=np.int64)
    lys, scs, cvs, fls = _as_list(lossyear), _as_list(scored), _as_list(converted), _as_list(flips)
    bcs = [None] * len(lys) if below_cut is None else _as_list(below_cut)
    for ly, sc, cv, fl, bc in zip(lys, scs, cvs, fls, bcs, strict=True):
        keep = np.asarray(sc, dtype=bool)
        codes = np.asarray(ly)[keep].astype(np.int64)
        pixels += np.bincount(codes, minlength=256)
        conv += np.bincount(codes, weights=np.asarray(cv, dtype=bool)[keep], minlength=256).astype(np.int64)
        flip += np.bincount(codes, weights=np.asarray(fl, dtype=bool)[keep], minlength=256).astype(np.int64)
        if bc is not None:
            below += np.bincount(codes, weights=np.asarray(bc, dtype=bool)[keep], minlength=256).astype(np.int64)
    out: dict[str, dict[str, float | int]] = {}
    for code in np.flatnonzero(pixels):
        n = int(pixels[code])
        row: dict[str, float | int] = {
            "pixels": n,
            "converted_share": int(conv[code]) / n,
            "flip_share": int(flip[code]) / n,
        }
        if below_cut is not None:
            row["below_cut_share"] = int(below[code]) / n
        out["no_loss" if code == 0 else str(2000 + int(code))] = row
    return out


def cross_check_removed(
    lossyear: np.ndarray | Sequence[np.ndarray],
    scored: np.ndarray | Sequence[np.ndarray],
    below_cut: np.ndarray | Sequence[np.ndarray],
    year_a: int,
    year_b: int,
) -> dict[str, int]:
    """The classifier flips the cross-check cut removed, split by Hansen loss (scored pixels only).

    Returns:
        ``flips_below_cut`` and its split into ``loss_in_window`` (lossyear in ``(year_a,
        year_b]``), ``loss_year_a`` (lossyear ``year_a``, inside the lag-aligned window
        ``[year_a, year_b - 1]`` but outside the headline one), ``loss_other_years`` and
        ``no_loss``; plus ``loss_lag_aligned`` (lossyear in ``[year_a, year_b - 1]``).
    """
    a, b = int(year_a) - 2000, int(year_b) - 2000
    out = dict.fromkeys(
        ("flips_below_cut", "loss_in_window", "loss_year_a", "loss_other_years", "no_loss", "loss_lag_aligned"), 0
    )
    for ly, sc, bc in zip(_as_list(lossyear), _as_list(scored), _as_list(below_cut), strict=True):
        codes = np.asarray(ly).astype(np.int64)[np.asarray(sc, dtype=bool) & np.asarray(bc, dtype=bool)]
        window = (codes > a) & (codes <= b)
        out["flips_below_cut"] += int(codes.size)
        out["loss_in_window"] += int(window.sum())
        out["loss_year_a"] += int((codes == a).sum())
        out["no_loss"] += int((codes == 0).sum())
        out["loss_other_years"] += int(((codes > 0) & ~window & (codes != a)).sum())
        out["loss_lag_aligned"] += int(((codes >= a) & (codes <= b - 1)).sum())
    return out


def _transfer_at_labels(
    ctx: RunContext,
    table: pd.DataFrame,
    emb_a: np.ndarray,
    model: ForestModel,
    train_radar: bool,
    label_zones: Sequence[str],
    opts: _Options,
    source: str,
) -> tuple[dict[str, Any], float | None, int]:
    """Read ``year_b`` at the label chunks and check how the classifier transfers (Hansen labels only).

    Adds ``hansen_stable``, ``prob_year_b``, ``distance`` and ``transfer_scored`` to ``table``. The
    read costs the label chunks once more (counted in the estimate by the preflight). Labels from a
    CSV carry no ``year_b`` truth, so nothing is read for them.

    Returns:
        ``(record, ring_cut, n_ring_cut)``: the transfer record for the report, and the ring label
        cut (:func:`ring_label_cut`) with the labels it rests on.
    """
    n = len(table)
    year_a, year_b, fp = ctx.year_a, ctx.year_b, opts.forest_prob
    y = table["label"].to_numpy(dtype=np.int64)
    stable = hansen_stable_labels(y, table["hansen_lossyear"].to_numpy(dtype=np.int64), year_b)
    table["hansen_stable"] = stable
    table["prob_year_b"] = np.full(n, np.nan, dtype=np.float64)
    table["distance"] = np.full(n, np.nan, dtype=np.float64)
    table["transfer_scored"] = np.zeros(n, dtype=bool)
    if source != "hansen":
        reason = "Labels from --labels carry no year_b truth, so the transfer to year_b is not checked."
        return {"available": False, "reason": reason}, None, 0
    points = io.read_points(ctx.reader, table["lon"].to_numpy(), table["lat"].to_numpy(), year_b, with_radar=True)
    ctx.mark_read([year_b], zones=label_zones)
    assert points.radar_present is not None
    used = table["used"].to_numpy(dtype=bool)
    in_regime_b = points.valid & (points.radar_present == bool(train_radar))
    prob_b = np.full(n, np.nan, dtype=np.float64)
    prob_b[in_regime_b] = model.probability_vectors(points.embeddings[in_regime_b])
    both = used & in_regime_b
    distance = np.full(n, np.nan, dtype=np.float64)
    distance[both] = vector_distance(emb_a[both], points.embeddings[both])
    scored = both & stable
    table["prob_year_b"] = prob_b
    table["distance"] = distance
    table["transfer_scored"] = scored
    result = label_transfer(
        y,
        scored,
        table["prob_year_a"].to_numpy(dtype=np.float64),
        table["heldout_prob"].to_numpy(dtype=np.float64),
        prob_b,
        forest_prob=fp,
    )
    ring_cut, n_ring = ring_label_cut(distance, y, scored, percentile=opts.stable_percentile)
    record = {
        "available": True,
        "years": [year_a, year_b],
        "labels_used": int(used.sum()),
        "labels_hansen_stable": int((used & stable).sum()),
        "labels_no_data_year_b": int((used & ~points.valid).sum()),
        "labels_other_regime_year_b": int((used & points.valid & ~in_regime_b).sum()),
        **result,
        "warn_drop": TRANSFER_WARN_DROP,
        "definition": (
            f"The classifier fitted on {year_a} labels scored on {year_b} at the labels Hansen keeps unchanged through "
            f"{year_b} (non-forest labels, and forest labels with no loss through {year_b}), valid in {year_b} in the "
            "training regime; drop is the year_a held-out balanced accuracy minus the year_b one on the same labels."
        ),
    }
    drop = result["drop"]
    enough = min(result["drop_labels_forest"], result["drop_labels_nonforest"]) >= MIN_TRANSFER_LABELS
    if drop is not None and enough and drop > TRANSFER_WARN_DROP:
        held = float(result["year_a_heldout"]["balanced_accuracy"])
        ctx.warn(
            f"The classifier transfers poorly to {year_b}: on {result['drop_labels']} ring labels Hansen keeps "
            f"unchanged through {year_b}, its {year_b} balanced accuracy is {held - drop:.3f}, {drop:.3f} below their "
            f"{year_a} held-out {held:.3f}, so the {year_b} forest map and the conversion may be biased. Raise "
            "--label-chunks or --labels-per-class, widen --label-ring-km, or pass --labels."
        )
    return record, ring_cut, n_ring


def _write_labels(path: Path, table: pd.DataFrame, aoi: BaseGeometry, source: str) -> Path:
    lon = table["lon"].to_numpy(dtype=np.float64)
    lat = table["lat"].to_numpy(dtype=np.float64)
    frame = pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "label": table["label"].to_numpy(dtype=np.int8),
            "weight": table["weight"].to_numpy(dtype=np.float64),
            "source": source,
            "source_chunk": chunk_ids(table).astype(str),
            "zone": table["zone"].astype(str).to_numpy(),
            "row": table["row"].to_numpy(dtype=np.int64),
            "col": table["col"].to_numpy(dtype=np.int64),
            "valid": table["valid"].to_numpy(dtype=bool),
            "regime": table["regime"].astype(str).to_numpy(),
            "used": table["used"].to_numpy(dtype=bool),
            "drop_reason": table["drop_reason"].astype(str).to_numpy(),
            "fold": table["fold"].to_numpy(dtype=np.int16),
            "heldout_prob": table["heldout_prob"].to_numpy(dtype=np.float32),
            "inside_aoi": shapely.contains_xy(aoi, lon, lat) if len(table) else np.zeros(0, dtype=bool),
            "hansen_lossyear": table["hansen_lossyear"].to_numpy(dtype=np.int16),
            "hansen_stable": table["hansen_stable"].to_numpy(dtype=bool),
            "prob_year_a": table["prob_year_a"].to_numpy(dtype=np.float32),
            "prob_year_b": table["prob_year_b"].to_numpy(dtype=np.float32),
            "distance": table["distance"].to_numpy(dtype=np.float32),
            "transfer_scored": table["transfer_scored"].to_numpy(dtype=bool),
        }
    )
    with export.atomic_path(path) as tmp:
        frame.to_parquet(tmp, engine="pyarrow", index=False)
    return path


def _run(ctx: RunContext, state: _State, opts: _Options) -> DeforestResult:
    year_a, year_b = ctx.year_a, ctx.year_b
    plan = ctx.plan
    fp = opts.forest_prob
    assert state.labels is not None
    source = str(state.label_info.get("source"))

    # The training regime first (cheap, and it fails when no AOI pixel is comparable), then year_a
    # at every label (each chunk once), keeping the labels in the training regime.
    t = time.perf_counter()
    probe = _probe_regime(ctx)
    train_radar = bool(probe["training_radar"])
    t = ctx.lap("regime_probe", t)
    lon = state.labels["lon"].to_numpy()
    lat = state.labels["lat"].to_numpy()
    # The zones whose chunks the label reads touch (a label outside its zone's grid reads nothing):
    # they may lie outside the AOI's zones, and the provenance records their zone-years.
    label_zones = sorted(set(state.labels.loc[state.labels["chunk_row"] >= 0, "zone"].astype(str)))
    points = io.read_points(ctx.reader, lon, lat, year_a, with_radar=True)
    ctx.mark_read([year_a], zones=label_zones)
    t = ctx.lap("labels_read", t)
    assert points.radar_present is not None
    table = apply_regime(state.labels, points.valid, points.radar_present, train_radar)
    used = table["used"].to_numpy(dtype=bool)
    y = table["label"].to_numpy(dtype=np.int64)
    used_f = int((used & (y == FOREST)).sum())
    used_n = int((used & (y == NONFOREST)).sum())
    no_data = int((table["drop_reason"] == "no_data").sum())
    off_regime = int((table["drop_reason"] == "regime").sum())
    if min(used_f, used_n) < MIN_LABELS_PER_CLASS:
        raise DeforestError(
            f"Only {used_f} forest and {used_n} non-forest labels are usable ({no_data} have no {year_a} data and "
            f"{off_regime} are not in the AOI's {regime_name(train_radar)} radar regime); the classifier needs at "
            f"least {MIN_LABELS_PER_CLASS} of each. Raise --label-chunks or --labels-per-class, widen "
            "--label-ring-km, or pass --labels."
        )
    if off_regime:
        ctx.warn(
            f"{off_regime} labels were dropped because their {year_a} radar regime is not the AOI's "
            f"({regime_name(train_radar)}, survey B5)."
        )
    groups = chunk_ids(table)
    fit = fit_forest_model(
        points.embeddings[used],
        y[used],
        groups[used],
        weights=table["weight"].to_numpy(dtype=np.float64)[used],
        seed=ctx.seed,
        forest_prob=fp,
    )
    folds = np.full(len(table), -1, dtype=np.int64)
    folds[used] = fit.folds
    held = np.full(len(table), np.nan, dtype=np.float64)
    held[used] = fit.heldout_prob
    table["fold"] = folds
    table["heldout_prob"] = held
    prob_label_a = np.full(len(table), np.nan, dtype=np.float64)
    prob_label_a[used] = fit.model.probability_vectors(points.embeddings[used])
    table["prob_year_a"] = prob_label_a
    clf = fit.record
    if not clf["converged"]:
        ctx.warn(f"The logistic regression did not converge in {MAX_ITER} iterations.")
    if not clf["heldout"]["folds"]:
        ctx.warn("No spatially held-out estimate: the labels come from fewer than two chunks.")
    elif clf["heldout"]["folds_skipped"]:
        ctx.warn(
            f"{clf['heldout']['folds_skipped']} of {clf['heldout']['folds']} held-out folds had one class only in "
            f"their training part and were not scored ({clf['heldout']['labels_unscored']} labels unscored)."
        )
    t = ctx.lap("classifier", t)
    transfer, ring_cut, n_ring_cut = _transfer_at_labels(
        ctx, table, points.embeddings, fit.model, train_radar, label_zones, opts, source
    )
    del points
    t = ctx.lap("transfer", t)

    zones, counts = _stream(ctx, fit.model, train_radar)
    t = ctx.lap("stream", t)

    prob_a = {z: st.prob_a for z, st in zones.items()}
    prob_b = {z: st.prob_b for z, st in zones.items()}
    dist = {z: st.dist for z, st in zones.items()}
    classified_a = {z: np.isfinite(p) for z, p in prob_a.items()}
    classified_b = {z: np.isfinite(p) for z, p in prob_b.items()}
    classified = {z: classified_a[z] & classified_b[z] & np.isfinite(dist[z]) for z in zones}
    forest_a = _forest_masks(prob_a, fp)
    forest_b = _forest_masks(prob_b, fp)
    names = [z.zone for z in plan.zones]
    cut, n_stable = cross_check_cut(
        [dist[z] for z in names],
        [prob_a[z] for z in names],
        [prob_b[z] for z in names],
        forest_prob=fp,
        percentile=opts.stable_percentile,
    )
    if cut is None:
        ctx.warn(
            "No pixel is forest in both years, so the cross-check cut could not be set: the conversion is not "
            "cross-checked against the distance."
        )
    elif n_stable < MIN_STABLE_PIXELS:
        ctx.warn(f"The cross-check cut rests on only {n_stable} stable forest pixels; it is unreliable.")
    cut_ratio = None if cut is None or not ring_cut else cut / ring_cut
    if cut_ratio is not None and n_ring_cut >= MIN_RING_CUT_LABELS and cut_ratio > CUT_WARN_RATIO:
        ctx.warn(
            f"The cross-check cut {cut:.3f} is {cut_ratio:.2f} times the {opts.stable_percentile:g}th percentile of "
            f"the {year_a} to {year_b} distance among {n_ring_cut} forest labels in the ring that Hansen keeps "
            f"unchanged ({ring_cut:.3f}): the AOI pixels classed forest in both years likely include conversions the "
            "classifier missed, so the cut may remove real conversions. The cut keeps the survey N13 rule; see "
            "summary.cross_check (and cross_check_removed when validating against Hansen), or lower "
            "--stable-percentile to keep more."
        )
    raw = {z: conversion_mask(prob_a[z], prob_b[z], dist[z], forest_prob=fp, cut=cut) for z in names}
    below_cut = {z: conversion_mask(prob_a[z], prob_b[z], dist[z], forest_prob=fp, cut=None) & ~raw[z] for z in names}
    n_flips_below = sum(int(m.sum()) for m in below_cut.values())
    dst = max(plan.zones, key=lambda z: z.n_pixels).zone
    min_pixels = metrics.area_to_pixels(opts.min_area_ha)
    converted, conv_mosaic = clean_conversion(raw, plan, min_pixels=min_pixels, dst_zone=dst)
    patches = patch_size_summary(metrics.component_sizes(conv_mosaic == 1))
    del conv_mosaic
    t = ctx.lap("clean", t)

    n_cls_a = sum(int(m.sum()) for m in classified_a.values())
    n_cls_b = sum(int(m.sum()) for m in classified_b.values())
    n_forest_a = sum(int(m.sum()) for m in forest_a.values())
    n_forest_b = sum(int(m.sum()) for m in forest_b.values())
    n_classified = sum(int(m.sum()) for m in classified.values())
    n_forest_a_both = sum(int((forest_a[z] & classified[z]).sum()) for z in names)
    n_raw = sum(int(m.sum()) for m in raw.values())
    n_conv = sum(int(m.sum()) for m in converted.values())
    del raw
    converted_ha = n_conv * HA_PER_PIXEL
    conversion_fraction = _ratio(n_conv, n_forest_a_both)
    if n_conv == 0:
        ctx.warn("No conversion patch survived the cross-check and the minimum area.")

    validation: dict[str, Any] | None = None
    validation_written = False
    if ctx.validate:
        validation, validation_written = _validate(
            ctx,
            state,
            converted=converted,
            classified=classified,
            forest_a=forest_a,
            forest_b=forest_b,
            classified_a=classified_a,
            classified_b=classified_b,
            below_cut=below_cut,
        )
        t = ctx.lap("validation", t)
    del below_cut

    band_names = [f"forest_prob_{year_a}", f"forest_prob_{year_b}", f"distance_{year_a}_{year_b}", "converted"]
    mosaic = BandMosaic(plan, len(band_names), dst_zone=dst)
    mosaic.put(0, prob_a)
    mosaic.put(1, prob_b)
    mosaic.put(2, dist)
    mosaic.put(3, {z: np.where(classified[z], converted[z].astype(np.float32), np.float32(np.nan)) for z in names})
    del prob_a, prob_b, dist, zones, classified_b, forest_b
    assert mosaic.data is not None
    with rasterio.Env(GDAL_CACHEMAX=WRITE_CACHE_BYTES):
        export.write_geotiff(ctx.paths.tif, mosaic.data, mosaic.transform, mosaic.crs, band_names=band_names)
    cut_value = math.nan if cut is None else cut
    common = {
        "year_a": year_a,
        "year_b": year_b,
        "converted_ha": converted_ha,
        "conversion_fraction_of_forest": conversion_fraction,
        "n_patches": patches["count"],
        "cross_check_cut": cut_value,
        "label_source": source,
    }
    rows = [
        {
            "year": year,
            "forest_ha": n_forest * HA_PER_PIXEL,
            "forest_fraction": _ratio(n_forest, n_cls),
            "classified_pixels": n_cls,
            **common,
        }
        for year, n_forest, n_cls in ((year_a, n_forest_a, n_cls_a), (year_b, n_forest_b, n_cls_b))
    ]
    summary_table = pd.DataFrame(rows)
    export.write_table(summary_table, ctx.paths.summary_parquet.with_suffix(""))
    _write_labels(ctx.extra_outputs["labels"], table, ctx.aoi.geometry, source)
    step = stride_for_preview(mosaic.data.shape[1:])
    ref_image = None
    if state.ref.window is not None:
        ref_mosaic, _, _ = export.mosaic_zones(
            [GridArray(state.ref.window[z.zone], z) for z in plan.zones], dst_zone=dst, nodata=export.BOOL_NODATA
        )
        ref_image = ref_mosaic[::step, ::step].copy()
        del ref_mosaic
    _write_preview(
        ctx.paths.preview,
        prob_a=mosaic.data[0, ::step, ::step],
        prob_b=mosaic.data[1, ::step, ::step],
        dist=mosaic.data[2, ::step, ::step],
        converted=mosaic.data[3, ::step, ::step],
        reference=ref_image,
        cut=cut,
        patches=patches,
        classifier=clf,
        year_a=year_a,
        year_b=year_b,
        forest_prob=fp,
    )
    del mosaic
    ctx.lap("write", t)

    radar = {
        "training_regime": regime_name(train_radar),
        "probe": {k: v for k, v in probe.items() if k != "training_radar"},
        "labels_radar": int((table["regime"] == "radar").sum()),
        "labels_optical": int((table["regime"] == "optical").sum()),
        "labels_no_data": no_data,
        "labels_dropped_regime": off_regime,
        "valid_both_pixels": counts["valid_both"],
        "regime_mismatch_pixels": counts["regime_mismatch"],
        "regime_mismatch_fraction": _ratio(counts["regime_mismatch"], counts["valid_both"]),
        "other_regime_pixels": counts["other_regime"],
        "other_regime_fraction": _ratio(counts["other_regime"], counts["valid_both"]),
        "unclassified_other_regime_year_a": counts["other_regime_a"],
        "unclassified_other_regime_year_b": counts["other_regime_b"],
    }
    if counts["regime_mismatch"]:
        ctx.warn(
            f"{counts['regime_mismatch']:,} valid AOI pixels switch radar regime between {year_a} and {year_b} and "
            "were not classified for conversion (survey B5)."
        )
    if counts["other_regime"]:
        ctx.warn(
            f"{counts['other_regime']:,} valid AOI pixels are {regime_name(not train_radar)} in both years, not the "
            f"training regime ({regime_name(train_radar)}), and were not classified."
        )
    label_record = {
        **state.label_info,
        "sampled": len(table),
        "used_forest": used_f,
        "used_nonforest": used_n,
        "dropped_no_data": no_data,
        "dropped_regime": off_regime,
        "class_balance_forest_share": _ratio(used_f, used_f + used_n),
    }
    forest_record = {
        str(year): {
            "forest_ha": n_forest * HA_PER_PIXEL,
            "forest_fraction": _ratio(n_forest, n_cls),
            "forest_pixels": n_forest,
            "classified_pixels": n_cls,
        }
        for year, n_forest, n_cls in ((year_a, n_forest_a, n_cls_a), (year_b, n_forest_b, n_cls_b))
    }
    summary = {
        "aoi_area_ha": plan.area_ha,
        "aoi_pixels": plan.n_pixels,
        "zones": plan.zone_names,
        "year_a": year_a,
        "year_b": year_b,
        "label_source": source,
        "forest": forest_record,
        "forest_ha_year_a": n_forest_a * HA_PER_PIXEL,
        "forest_ha_year_b": n_forest_b * HA_PER_PIXEL,
        "forest_fraction_year_a": _ratio(n_forest_a, n_cls_a),
        "forest_fraction_year_b": _ratio(n_forest_b, n_cls_b),
        "classified_pixels": n_classified,
        "forest_pixels_year_a_classified_both": n_forest_a_both,
        "converted_pixels": n_conv,
        "converted_pixels_before_cleaning": n_raw,
        "converted_ha": converted_ha,
        "conversion_fraction_of_forest": conversion_fraction,
        "conversion_definition": (
            f"Classified in both years, forest probability >= {fp:g} in {year_a} and < {fp:g} in {year_b}, cosine "
            f"distance {year_a} to {year_b} above the cross-check cut; then a 3 x 3 opening and patches of at least "
            f"{opts.min_area_ha:g} ha. The fraction is of {year_a} forest classified in both years."
        ),
        "forest_prob": fp,
        "min_area_ha": opts.min_area_ha,
        "min_pixels": min_pixels,
        "cross_check": {
            "cut": cut,
            "percentile": opts.stable_percentile,
            "stable_forest_pixels": n_stable,
            "classifier_flips_below_cut": n_flips_below,
            "definition": (
                f"The {opts.stable_percentile:g}th percentile of the {year_a} to {year_b} cosine distance among pixels "
                "classed forest in both years (survey N13)."
            ),
            "ring_label_cut": ring_cut,
            "ring_label_cut_labels": n_ring_cut,
            "cut_to_ring_label_cut": cut_ratio,
            "ring_label_cut_definition": (
                f"The same percentile among the forest labels in the ring that Hansen keeps unchanged through "
                f"{year_b} (Hansen labels only): a reference the AOI's own conversions cannot raise."
            ),
            "removed_by_hansen_loss": (validation or {}).get("cross_check_removed"),
        },
        "patches": patches,
        "labels": label_record,
        "classifier": clf,
        "transfer": transfer,
        "radar": radar,
        "hansen": {
            "version": opts.hansen_version,
            "url_template": opts.hansen_template,
            "labels": source == "hansen",
            "validation": bool(ctx.validate and opts.reference is None),
        },
        "validation": validation,
    }
    report = finish_run(ctx, summary=summary, extra=ctx.extra_outputs, validation_written=validation_written)
    return DeforestResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=validation_written,
    )


def _redo_validation(ctx: RunContext, state: _State, opts: _Options) -> DeforestResult:
    """Redo only the validation of a run whose other outputs are current, without reading a chunk.

    The conversion, the classified pixels and the ``year_a`` forest map come back from the run's
    GeoTIFF (:func:`zone_values_from_raster`). When the reference still cannot be loaded, nothing is
    written and the run counts as skipped.
    """
    previous = json.loads(ctx.paths.report.read_text(encoding="utf-8"))
    if state.ref.window is None:
        logger.warning(
            "%s: outputs in %s are current; the validation reference still cannot be loaded (%s), so nothing "
            "was redone.",
            APP,
            ctx.paths.report.parent,
            state.ref.error,
        )
        return DeforestResult(
            report=previous,
            paths=ctx.paths,
            extra_paths=dict(ctx.extra_outputs),
            skipped=True,
            validation_written=False,
        )
    plan = ctx.plan
    summary = dict(previous.get("summary") or {})
    state.label_info = dict(summary.get("labels") or {})
    prob_a = zone_values_from_raster(ctx.paths.tif, plan, band=1)
    prob_b = zone_values_from_raster(ctx.paths.tif, plan, band=2)
    dist = zone_values_from_raster(ctx.paths.tif, plan, band=3)
    flags = zone_values_from_raster(ctx.paths.tif, plan, band=4)
    classified = {z: np.isfinite(v) for z, v in flags.items()}
    converted = {z: v == 1 for z, v in flags.items()}
    cross = dict(summary.get("cross_check") or {})
    cut = cross.get("cut")
    fp = opts.forest_prob
    below_cut = {
        z: conversion_mask(prob_a[z], prob_b[z], dist[z], forest_prob=fp, cut=None)
        & ~conversion_mask(prob_a[z], prob_b[z], dist[z], forest_prob=fp, cut=cut)
        for z in prob_a
    }
    validation, written = _validate(
        ctx,
        state,
        converted=converted,
        classified=classified,
        forest_a=_forest_masks(prob_a, fp),
        forest_b=_forest_masks(prob_b, fp),
        classified_a={z: np.isfinite(v) for z, v in prob_a.items()},
        classified_b={z: np.isfinite(v) for z, v in prob_b.items()},
        below_cut=below_cut,
        extra_caveats=[REDONE_CAVEAT],
    )
    summary["validation"] = validation
    cross["removed_by_hansen_loss"] = validation.get("cross_check_removed")
    summary["cross_check"] = cross
    old = [w for w in previous.get("warnings") or [] if not str(w).startswith(VALIDATION_SKIPPED)]
    report = finish_validation_only(
        ctx, previous, summary=summary, warnings=[*old, *ctx.warnings], validation_written=written
    )
    return DeforestResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=written,
        validation_redone=True,
    )


# --------------------------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------------------------


def _write_preview(
    path: Path,
    *,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    dist: np.ndarray,
    converted: np.ndarray,
    reference: np.ndarray | None,
    cut: float | None,
    patches: Mapping[str, Any],
    classifier: Mapping[str, Any],
    year_a: int,
    year_b: int,
    forest_prob: float,
) -> Path:
    """Write the six-panel preview with a matplotlib Figure (never pyplot).

    Panels: forest probability in both years, the conversion over the reference outline, the
    distance with the cross-check cut, the patch-size histogram and the classifier's scores. Every
    image comes in already strided.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    grey = export.PREVIEW_NAN_RGB
    fig = Figure(figsize=(18, 11), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    forest_cmap = mpl.colormaps["YlGn"].with_extremes(bad=grey)
    for ax, img, year in ((axes[0, 0], prob_a, year_a), (axes[0, 1], prob_b, year_b)):
        shown = ax.imshow(np.ma.masked_invalid(img), cmap=forest_cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        fig.colorbar(shown, ax=ax, label="forest probability", shrink=0.85)
        finite = img[np.isfinite(img)]
        share = float(np.mean(finite >= forest_prob)) if finite.size else math.nan
        ax.set_title(f"Forest {year}: {share:.0%} of classified pixels at p >= {forest_prob:g}")
        ax.set_axis_off()
    cat = np.full(converted.shape, np.nan)
    fin = np.isfinite(converted)
    with np.errstate(invalid="ignore"):
        cat[fin] = np.where(converted[fin] == 1, 2.0, np.where(prob_a[fin] >= forest_prob, 1.0, 0.0))
    cmap = mpl.colors.ListedColormap(["#e8dcc0", "#a6d96a", "#d7191c"]).with_extremes(bad=grey)
    ax = axes[0, 2]
    ax.imshow(np.ma.masked_invalid(cat), cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest")
    title = f"Converted {year_a} to {year_b} (red) over {year_a} forest (green)"
    if reference is not None:
        ref = (reference == 1).astype(np.float64)
        if ref.any():
            ax.contour(ref, levels=[0.5], colors=["#2166ac"], linewidths=0.8)
        title += ",\nreference loss outline in blue"
    ax.set_title(title)
    ax.set_axis_off()
    ax = axes[1, 0]
    finite = dist[np.isfinite(dist)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if cut is not None:
        vmax = max(vmax, cut * 1.5)
    shown = ax.imshow(
        np.ma.masked_invalid(dist),
        cmap=mpl.colormaps["magma"].with_extremes(bad=grey),
        vmin=0.0,
        vmax=max(vmax, 1e-6),
        interpolation="nearest",
    )
    fig.colorbar(shown, ax=ax, label="cosine distance", shrink=0.85)
    ax.set_title(f"Cosine distance {year_a} to {year_b}" + ("" if cut is None else f", cross-check cut {cut:.3f}"))
    ax.set_axis_off()
    ax = axes[1, 1]
    hist = dict(patches.get("histogram") or {})
    ax.bar(list(hist), list(hist.values()), color="#d7191c")
    ax.set_xlabel("patch size (ha)")
    ax.set_ylabel("conversion patches")
    median = patches.get("median_ha")
    ax.set_title(
        f"{patches.get('count', 0)} conversion patches" + ("" if median is None else f", median {median:.2f} ha")
    )
    ax = axes[1, 2]
    held = classifier.get("heldout") or {}
    train = classifier.get("training") or {}
    bars = [
        ("training\naccuracy", train.get("accuracy")),
        ("held-out\nbalanced acc.", held.get("balanced_accuracy")),
        ("held-out\nrecall forest", held.get("recall_forest")),
        ("held-out\nrecall non-forest", held.get("recall_nonforest")),
    ]
    values = [0.0 if v is None else float(v) for _, v in bars]
    ax.bar([n for n, _ in bars], values, color=["#636363", "#2166ac", "#1a9641", "#a6611a"])
    for i, (_, v) in enumerate(bars):
        ax.text(i, (0.0 if v is None else float(v)) + 0.02, "n/a" if v is None else f"{float(v):.2f}", ha="center")
    ax.set_ylim(0.0, 1.1)
    ax.set_title(
        f"Classifier: {classifier.get('labels', 0)} labels, {held.get('folds', 0)} folds held out by store chunk"
    )
    final = Path(path)
    with export.atomic_path(final) as tmp, tmp.open("wb") as fh:
        fig.savefig(fh, format="png", dpi=100)
    return final


__all__ = [
    "APP",
    "CV_FOLDS",
    "FOREST",
    "MIN_LABELS_PER_CLASS",
    "NONFOREST",
    "NO_LABEL",
    "PATCH_BIN_LABELS",
    "ClassMap",
    "ClassifierFit",
    "DeforestError",
    "DeforestResult",
    "ForestModel",
    "LabelRing",
    "apply_regime",
    "choose_label_chunks",
    "chunk_ids",
    "classify_block",
    "clean_conversion",
    "conversion_mask",
    "cross_check_cut",
    "csv_label_points",
    "fit_forest_model",
    "label_classes",
    "label_ring",
    "loss_year_breakdown",
    "memory_estimate",
    "patch_size_summary",
    "plan_label_ring",
    "read_labels_csv",
    "regime_name",
    "run_deforest",
    "sample_hansen_labels",
    "score_against",
    "training_regime",
]
