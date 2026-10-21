"""Wildfire: where did it burn, and how far has each burned pixel recovered since?

Brief section 4.1, as corrected by the pre-build survey, checkpoint 2 and the checkpoint 3 review:

- **Years.** ``year_a`` is the year before the fire and ``year_b`` the year after it (survey N11:
  the annual embedding of the fire year only partly sees a late-season fire). ``fire_year``
  defaults to ``year_b - 1``, or to ``year_b`` when ``year_b`` is ``year_a + 1`` (the year after
  ``year_a`` is the fire year by definition), and the report warns when ``year_b`` is the fire year.
  The app reads ``year_a``, ``year_b`` and every year after ``year_b`` through ``recovery_through``,
  plus ``year_a - 1`` when no ``--threshold`` is given (for the threshold fallback and the recovery
  reference check), and checks all of them against ``years_complete`` for every zone before any
  chunk is read (survey S2).
- **Detection.** Per-pixel cosine distance between ``year_a`` and ``year_b``, on AOI pixels valid in
  both years whose radar regime is the same in both (survey B5: a pixel that gains or loses
  Sentinel-1 shifts by 0.13 to 0.25 of cosine distance with no change on the ground). Pixels whose
  regime differs are not classified and are reported as ``radar_mismatch_fraction``.
- **Threshold.** ``--threshold`` if given, else Otsu over every AOI distance pooled across zones.
  Otsu's own separability flag passes almost any distribution (a single normal scores 0.64), so the
  smoothed histogram must also have a valley between the two class peaks (:func:`valley_ratio`
  measures its floor). When it does not (survey N12), the threshold falls back to the 99th
  percentile of the distance between two pre-fire years, ``year_a - 1`` and ``year_a``, read as an
  extra pass over the blocks not already read. A fallback that flags almost nothing is reported,
  not used silently: either nothing changed or the "stable" pair is not stable.
- **Cleaning.** A 3 x 3 opening and a minimum area, on the mosaic of every zone
  (:func:`~tessera_apps.apps._common.mosaic_and_clean`), so a scar crossing a zone seam is judged
  whole.
- **Recovery** (survey S11 replaces the brief's fixed 0.9). For each year ``t`` from ``year_b``
  through ``recovery_through``, ``s_t`` is each pixel's cosine similarity to its own ``year_a``
  vector, in the same radar regime as ``year_a``. The reference for land that did not burn is the
  classified pixels shown to be unchanged: a ``year_a`` to ``year_b`` distance at most half the
  threshold, and more than :data:`REFERENCE_BUFFER_PX` pixels from any burned pixel.
  Every pixel at or below the threshold would not do: when the AOI lies mostly inside the scar,
  that set is largely burned land just under the threshold, which drags its similarity down and
  inflates recovery. A burned pixel is *recovered* when its ``s_t`` reaches the 10th percentile of
  the reference's, and its recovery index is ``R_t = (s_t - s_post) / (s_ref_t - s_post)`` with
  ``s_post`` its similarity at ``year_b`` and ``s_ref_t`` the reference median. The report records
  the reference's similarity before the fire (``year_a - 1`` to ``year_a``, on a sample of blocks
  or on every block when the fallback ran) and after it, so any contamination is visible. Land that
  did not burn stays about 0.95 similar to itself; the brief's share above 0.9 is kept as
  ``sim90_fraction`` and stays near zero because burned pixels sit near 0.6 for years.
- **Memory** (survey S18). The per-zone results are kept as single arrays, the output COG is
  mosaicked through one index band and gathered band by band instead of from a per-zone stack,
  and arrays are freed before the output stage. The plan log gives a peak memory estimate
  (:func:`memory_estimate`), and the report warns, before any chunk read, when it exceeds the
  memory available.
- **Validation** (survey S10). Against ``--reference`` (any vector file or GeoTIFF) or NIFC
  perimeters, over every classified AOI pixel of every zone, with the caveats that bound what the
  scores can show. The reference is loaded before any chunk read: a bad ``--reference`` file stops
  the run with a usage error, and under ``strict_validate`` an unreachable reference does too. A
  validating rerun whose only missing output is ``validation.json`` redoes the validation from the
  outputs on disk, without reading a chunk.
- **Burn severity** (the owner's checkpoint 3 decision). ``--severity`` scores the burned mask
  against a thematic severity map in MTBS dnbr6 codes, in any CRS, read onto every zone plan with
  nearest resampling and checked in the preflight (:func:`load_severity`). The scores
  (:func:`score_against_severity`) are the detection rate of each class 0 to 5, IoU, precision,
  recall and F1 against class ``--severity-min-class`` or worse, against moderate or worse and
  against low or worse, and where the burned pixels fall; they go into ``validation.json`` as a
  ``severity`` block next to the outline scores (NIFC or ``--reference``; both can be given). The
  scores cover the classified AOI pixels the map covers, background (class 0, outside the mapped
  burn) included as unburned land, so detections outside the burn count against precision. With
  a severity map the brief's IoU 0.6 stop rule reads its IoU against moderate or worse, whatever
  ``--severity-min-class`` is, otherwise the outline IoU together with recall inside the
  reference (:func:`stop_rule`), and the summary's headline (:func:`headline`) puts them in that
  order. The Sentinel-2 dNBR proxy of ``scripts/make_caldor_dnbr.py`` is named as such from its
  ``TESSERA_APPS_SOURCE`` tag. Adding ``--validate`` to a run that had only ``--severity`` redoes
  the validation from the outputs on disk, without reading a chunk.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rasterio.errors
from scipy import ndimage

from tessera_apps import export, io, metrics, validate
from tessera_apps.apps._common import (
    AppInputError,
    LoadedAOI,
    RunContext,
    StrictValidationError,
    aoi_reference,
    finish_run,
    finish_validation_only,
    grow_across_zones,
    load_aoi,
    mosaic_and_clean,
    prepare_run,
    stride_for_preview,
    zone_values_from_raster,
)
from tessera_apps.export import GridArray, OutputPaths
from tessera_apps.io import FIRST_YEAR, HA_PER_PIXEL, LAST_YEAR, AOIPlan, Reader, ZonePlan

logger = logging.getLogger(__name__)

APP = "wildfire"
#: Seed of the PCA pixel sample.
SEED = 0
#: Layers the app reads: the radar counts give the regime check, s2_obs_count is not needed.
LAYERS = ("embeddings", "scales", "s1_asc_obs_count", "s1_desc_obs_count")
#: Largest PCA sample, and the least drawn from each block of the sampling pass.
PCA_SAMPLE_CAP = 200_000
PCA_SAMPLE_PER_BLOCK = 2_000
#: Blocks the PCA sampling pass reads (``year_a`` only), spread evenly over the AOI.
PCA_FIT_BLOCKS = 32
#: Otsu's split is taken as bimodal only when the floor of the smoothed density between the two
#: class peaks is at most this share of the lower peak. On the Caldor fixture the 2020 to 2022
#: burn scores 0.51, 2019 to 2021 0.68 and 2020 to 2021 (the fire year as year_b) 0.76; the
#: no-change pairs 2017 to 2018, 2018 to 2019 and 2017 to 2019 score 1.0.
MAX_VALLEY_RATIO = 0.9
#: Percentile of the stable pre-fire pair's distances used by the fallback threshold.
STABLE_PERCENTILE = 99.0
#: The fallback is reported as suspect when its threshold reaches this percentile of the distances
#: it is applied to: it then flags almost nothing.
FALLBACK_FLAG_PERCENTILE = 99.5
#: A burned pixel is recovered when its similarity reaches this percentile of the reference's.
UNBURNED_PERCENTILE = 10.0
#: Recovery reference: classified pixels whose ``year_a`` to ``year_b`` distance is at most this
#: share of the threshold ...
REFERENCE_DISTANCE_SHARE = 0.5
#: ... and that lie more than this many pixels (8-connected steps) from any burned pixel.
REFERENCE_BUFFER_PX = 5
#: The report warns when the recovery reference has fewer pixels than this.
MIN_REFERENCE_PIXELS = 100
#: The report warns when the recovery reference's median similarity after the fire (``year_a`` to
#: ``year_b``) is lower than before it (``year_a - 1`` to ``year_a``) by more than this: it then
#: likely holds fire-affected land. Two years drift a little more than one, so a clean reference
#: drops a little (0.03 on the Caldor fixture); a 290 ha window inside the live scar drops 0.05.
REFERENCE_DRIFT_WARN = 0.04
#: Blocks, spread evenly over the AOI, whose ``year_a - 1`` is read in the main pass to measure the
#: recovery reference's pre-fire similarity (every block is read when the fallback runs).
PREFIRE_SAMPLE_BLOCKS = 4
#: ``R_t`` is undefined where ``s_ref_t - s_post`` is below this.
MIN_DENOMINATOR = 0.02
#: ``R_t`` is clipped to this range.
RECOVERY_CLIP = (-1.0, 2.0)
#: The brief's similarity level for ``sim90_fraction``.
DEFAULT_RECOVERY_SIM = 0.9
#: Radar-free share of the scar above which the report flags it (brief section 4.1 step 5).
RADAR_FREE_FLAG = 0.2
#: The brief's checkpoint 3 stop-and-report level.
IOU_STOP = 0.6
PCA_BANDS = ("pc1", "pc2", "pc3")
#: Bins of the distance histogram in the preview.
HISTOGRAM_BINS = 128
#: GDAL block cache while the COGs are written. GDAL's default (5 percent of physical memory) fills
#: with the written blocks on top of the array being written; 128 MiB costs about 10 percent more
#: write time and saves up to that 5 percent at the run's peak.
WRITE_CACHE_BYTES = 128 * 2**20
#: Start of the warning a run records when it could not load its validation reference.
VALIDATION_SKIPPED = "Validation skipped, the reference could not be loaded"
#: The caveat in ``validation.json`` when the severity map was scored but the outline reference was not.
OUTLINE_NOT_SCORED = (
    "The outline validation (NIFC, or the --reference file) was requested but not done, so this file holds only the "
    "severity scores; a rerun once the reference loads adds the outline scores without reading the store. "
    "The reference could not be loaded"
)
#: End of the warning a run records when the stop rule trips (:func:`stop_rule`).
STOP_WARNING = "stop and report (brief section 8)."
#: Peak memory model (bytes), from synthetic two-zone AOIs of 4.5 and 10.6 million window pixels:
#: the stream holds the per-pixel arrays (distance, similarities, radar flags, false colour,
#: pre-fire distance) and one block of every year read; the recovery and write stages hold the
#: output mosaic and its index instead of the false colour. Each recovery year after year_b adds a
#: similarity array and an output band. The margin covers allocator overhead.
MEMORY_FIXED = 800_000_000
MEMORY_STREAM_PER_PIXEL = 28
MEMORY_WRITE_PER_PIXEL = 70
MEMORY_PER_LATER_YEAR_STREAM = 4
MEMORY_PER_LATER_YEAR_WRITE = 8
MEMORY_PER_BLOCK_PIXEL_YEAR = 210
MEMORY_MARGIN = 1.4

#: MTBS dnbr6 codes of a ``--severity`` map.
SEVERITY_NAMES = {
    0: "background",
    1: "unburned to low",
    2: "low",
    3: "moderate",
    4: "high",
    5: "increased greenness",
    6: "non-processing",
}
SEVERITY_CODES = tuple(SEVERITY_NAMES)
SEVERITY_BACKGROUND = 0
SEVERITY_LOW = 2
SEVERITY_HIGH = 4
SEVERITY_GREENER = 5
SEVERITY_NON_PROCESSING = 6
#: Code of the per-zone arrays of :func:`load_severity` for pixels the map does not cover (outside
#: its extent or on its nodata) and pixels outside the AOI: never scored.
SEVERITY_UNMAPPED = 255
#: Classes whose classified pixels are scored, each with its own detection rate: background (0,
#: outside the mapped burn) and increased greenness (5) are unburned land in the binary scores.
SEVERITY_SCORED = (0, 1, 2, 3, 4, 5)
#: The order the headline lists the per-class detection rates in.
SEVERITY_HEADLINE_ORDER = (4, 3, 2, 1, 5)
#: Lowest class counted as burned by default (moderate).
DEFAULT_SEVERITY_MIN_CLASS = 3
#: The brief's stop rule reads the IoU against this class or worse (moderate), whatever
#: ``--severity-min-class`` is.
SEVERITY_STOP_CLASS = 3
#: GeoTIFF tag naming where a severity map came from, and the value of the Sentinel-2 proxy
#: (``scripts/make_caldor_dnbr.py``).
SEVERITY_SOURCE_TAG = "TESSERA_APPS_SOURCE"
SEVERITY_PROXY_SOURCE = "sentinel2_dnbr"
#: How the headline names the outline reference, by its source.
OUTLINE_LABELS = {"nifc": "NIFC perimeter", "file": "Reference outline"}

SEVERITY_THRESHOLD_CAVEAT = (
    "MTBS sets its dNBR class thresholds per fire, by analyst judgement; a dNBR proxy with generic thresholds is "
    "cruder, so the severity class boundaries are approximate."
)
SEVERITY_RESOLUTION_CAVEAT = (
    "The severity classes are 30 m (MTBS) or 20 m (Sentinel-2) pixels read onto the 10 m grid by nearest "
    "resampling, so class edges are blocky against the 10 m burned mask and bound the IoU below 1 along every edge."
)
SEVERITY_EXCLUDED_CAVEAT = (
    "Pixels the severity map does not cover (outside its extent or on its nodata) and class 6 (non-processing) "
    "are excluded from every severity score and counted under excluded. Background (class 0, outside the mapped "
    "burn) and increased greenness (class 5) are never burned classes: they are scored as unburned land in the "
    "binary scores, so detections there count as false positives, and each has its own detection rate."
)
SEVERITY_NODATA_ZERO_CAVEAT = (
    "The severity file declares 0 as its nodata, so its background cannot be told apart from pixels it does not "
    "map: background pixels are excluded as unmapped, and detections outside the mapped burn do not count "
    "against precision or IoU."
)
SEVERITY_PROXY_CAVEAT = (
    "The severity map carries TESSERA_APPS_SOURCE=sentinel2_dnbr: it is the Sentinel-2 dNBR proxy written by "
    "scripts/make_caldor_dnbr.py, not MTBS. Its classes come from generic Key and Benson thresholds on a "
    "median-offset dNBR of 20 m Sentinel-2 composites, not from an analyst's per-fire thresholds."
)

NIFC_CAVEAT = (
    "NIFC perimeters are outer fire boundaries, not adjusted for unburned islands inside them, so a "
    "pixel-level burned mask has an IoU ceiling below 1 against them; recall_inside_reference and "
    "burned_outside_reference_fraction separate missed burn from detections outside the perimeter."
)
OUTLINE_CAVEAT = (
    "A filled outline or perimeter reference includes unburned islands, so a pixel-level burned mask "
    "has an IoU ceiling below 1 against it."
)
HANSEN_CAVEAT = (
    "This reference appears to be derived from Hansen Global Forest Change loss, which records "
    "stand-replacing forest loss, not burn severity: low-severity burn and burned non-forest are not "
    "in it, and Hansen is itself a 30 m model."
)
SCOPE_CAVEAT = (
    "Scores count AOI pixels valid in both years with the same radar regime, in every zone the AOI "
    "touches; pixels outside the AOI are never classified, so the AOI should extend past the scar."
)
REDONE_CAVEAT = (
    "The validation was redone from the burned band of the run's GeoTIFF, without reading the store. "
    "For an AOI in one zone the scores equal a full run's; across zones a few pixels along the seam "
    "may differ, where the mosaic moved them."
)


class WildfireError(io.TesseraError):
    """The data cannot answer the question as asked, for example no pixel is comparable."""


# --------------------------------------------------------------------------------------------
# Threshold
# --------------------------------------------------------------------------------------------


def valley_ratio(values: np.ndarray, threshold: float, *, sigma_bins: float = 1.0) -> float:
    """Floor of the smoothed histogram between the two class peaks, over the lower peak.

    ``threshold`` splits the values into two classes; each class's peak is the highest smoothed
    bin on its side. Two real modes have a valley between their peaks and score well below 1,
    wherever in that valley (or on the shoulder of a broad mode) the threshold falls. A threshold
    that cuts through one population (Otsu on a unimodal or skewed distribution, or Otsu pulled
    into the stable tail by a small burn, survey N12) has no valley between the peaks, since one of
    them is the bin next to the cut, and scores 1.

    The histogram spans the 0.5th to 99.5th percentile of the finite values in
    ``clip(sqrt(n) / 2, 16, 128)`` bins and is smoothed with a Gaussian of ``sigma_bins`` bins.

    Args:
        values: Distances; NaN is ignored.
        threshold: The split to judge.
        sigma_bins: Smoothing width in bins.

    Returns:
        The ratio, 1.0 when it cannot be computed (fewer than two distinct values).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 1.0
    lo, hi = np.percentile(v, [0.5, 99.5])
    if not hi > lo:
        return 1.0
    bins = int(np.clip(round(math.sqrt(v.size) / 2), 16, 128))
    counts, edges = np.histogram(v, bins=bins, range=(float(lo), float(hi)))
    smooth = ndimage.gaussian_filter1d(counts.astype(np.float64), sigma_bins, mode="nearest")
    k = int(np.clip(np.searchsorted(edges, threshold, side="right") - 1, 0, bins - 1))
    low_peak = int(np.argmax(smooth[: k + 1]))
    high_peak = k + int(np.argmax(smooth[k:]))
    peak = min(float(smooth[low_peak]), float(smooth[high_peak]))
    floor = float(smooth[low_peak : high_peak + 1].min())
    return float(floor / peak) if peak > 0 else 1.0


@dataclasses.dataclass(frozen=True)
class ThresholdChoice:
    """The burn threshold and how it was chosen.

    Attributes:
        threshold: Distances strictly above this are burned.
        method: ``"override"`` (``--threshold``), ``"otsu"`` or ``"stable_p99"`` (the fallback).
        otsu: Otsu's result on the pooled distances, None when it could not be computed.
        valley_ratio: :func:`valley_ratio` at Otsu's threshold.
        bimodal: Otsu's flag and the valley check together; None without Otsu.
        fallback_ran: The stable-pair pass ran.
        fallback_available: A stable pair could be read (``year_a - 1 >= 2017``, no override).
        stable_pair: The pre-fire pair of the fallback.
        stable_p99: The fallback's percentile, when it ran.
        n_values: Distances Otsu saw.
        n_stable_values: Distances the fallback saw.
        note: Why the method was chosen, when it is not the plain case.
        fallback_flagged_share: Share of the distances above the fallback threshold.
        fallback_suspect: The fallback threshold reaches the :data:`FALLBACK_FLAG_PERCENTILE`
            percentile of the distances, so it flags almost nothing: either nothing changed, or the
            stable pair is not stable.
    """

    threshold: float
    method: str
    otsu: metrics.OtsuResult | None
    valley_ratio: float | None
    bimodal: bool | None
    fallback_ran: bool
    fallback_available: bool
    stable_pair: tuple[int, int] | None
    stable_p99: float | None
    n_values: int
    n_stable_values: int | None
    note: str | None
    fallback_flagged_share: float | None = None
    fallback_suspect: bool | None = None

    def record(self) -> dict[str, Any]:
        """Plain values for the report."""
        return {
            "threshold": self.threshold,
            "method": self.method,
            "otsu": None if self.otsu is None else dataclasses.asdict(self.otsu),
            "valley_ratio": self.valley_ratio,
            "max_valley_ratio": MAX_VALLEY_RATIO,
            "bimodal": self.bimodal,
            "fallback_ran": self.fallback_ran,
            "fallback_available": self.fallback_available,
            "stable_pair": None if self.stable_pair is None else list(self.stable_pair),
            "stable_percentile": STABLE_PERCENTILE,
            "stable_p99": self.stable_p99,
            "fallback_flagged_share": self.fallback_flagged_share,
            "fallback_suspect": self.fallback_suspect,
            "n_values": self.n_values,
            "n_stable_values": self.n_stable_values,
            "note": self.note,
        }


def select_threshold(
    values: np.ndarray,
    *,
    override: float | None = None,
    stable_pair: tuple[int, int] | None = None,
    stable_values: Callable[[], np.ndarray] | None = None,
) -> ThresholdChoice:
    """Choose the burn threshold: the override, Otsu when bimodal, else the stable-pair fallback.

    Otsu's diagnostics are computed in every case so the report can show them. ``stable_values`` is
    called only when the fallback is needed: it streams the pre-fire pair, which costs two more
    years of reads. A fallback threshold at or above the :data:`FALLBACK_FLAG_PERCENTILE`
    percentile of ``values`` is still used (when nothing changed, flagging almost nothing is right)
    but marked ``fallback_suspect``, so the app can say so.

    Args:
        values: Pooled distances (NaN ignored).
        override: A fixed threshold.
        stable_pair: The pre-fire years ``stable_values`` compares, for the record.
        stable_values: Returns the stable pair's distances; None when no pair can be read.

    Returns:
        The choice.

    Raises:
        WildfireError: No override, fewer than two distinct distances, and no usable fallback.
    """
    v = np.asarray(values).ravel()
    v = v[np.isfinite(v)]
    otsu: metrics.OtsuResult | None
    try:
        otsu = metrics.otsu_threshold(v)
    except ValueError:
        otsu = None
    ratio = None if otsu is None else valley_ratio(v, otsu.threshold)
    bimodal = None if otsu is None else bool(otsu.bimodal and ratio is not None and ratio <= MAX_VALLEY_RATIO)
    available = stable_values is not None
    base = ThresholdChoice(
        threshold=math.nan,
        method="otsu",
        otsu=otsu,
        valley_ratio=ratio,
        bimodal=bimodal,
        fallback_ran=False,
        fallback_available=available,
        stable_pair=None,
        stable_p99=None,
        n_values=int(v.size),
        n_stable_values=None,
        note=None,
    )
    if override is not None:
        return dataclasses.replace(base, threshold=float(override), method="override")
    if otsu is not None and bimodal:
        return dataclasses.replace(base, threshold=otsu.threshold)
    note = (
        "fewer than two distinct distances"
        if otsu is None
        else f"Otsu split not bimodal (eta {otsu.eta:.3f}, valley ratio {ratio:.2f})"
    )
    if stable_values is not None:
        stable = np.asarray(stable_values()).ravel()
        stable = stable[np.isfinite(stable)]
        if stable.size:
            p = float(np.percentile(stable, STABLE_PERCENTILE))
            flagged = float(np.mean(v > p)) if v.size else None
            suspect = bool(v.size and p >= float(np.percentile(v, FALLBACK_FLAG_PERCENTILE)))
            return dataclasses.replace(
                base,
                threshold=p,
                method="stable_p99",
                fallback_ran=True,
                stable_pair=stable_pair,
                stable_p99=p,
                n_stable_values=int(stable.size),
                note=f"{note}; fell back to the {STABLE_PERCENTILE:g}th percentile of the stable pair",
                fallback_flagged_share=flagged,
                fallback_suspect=suspect,
            )
        tail = "the stable pair has no comparable pixel, so Otsu's threshold is used and may flag stable land"
    else:
        tail = "no stable-pair fallback, so Otsu's threshold is used and may flag stable land"
    if otsu is None:
        raise WildfireError(f"Cannot choose a burn threshold ({note}; {tail}). Pass --threshold.")
    return dataclasses.replace(
        base,
        threshold=otsu.threshold,
        fallback_ran=stable_values is not None,
        stable_pair=stable_pair,
        note=f"{note}; {tail}",
    )


# --------------------------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------------------------


def recovery_index(
    s_t: np.ndarray,
    s_post: np.ndarray,
    s_ref: float,
    *,
    min_denominator: float = MIN_DENOMINATOR,
    clip: tuple[float, float] = RECOVERY_CLIP,
) -> np.ndarray:
    """Per-pixel recovery index ``R_t = (s_t - s_post) / (s_ref - s_post)``, clipped (survey S11).

    0 means no change since just after the fire, 1 means as similar to the pre-fire vector as a
    typical unburned pixel. Undefined (NaN) where the denominator is under ``min_denominator``
    (the pixel was already as similar as unburned land right after the fire, so there is nothing to
    recover) or any input is NaN.

    Args:
        s_t: Similarity to the pixel's own ``year_a`` vector in year t.
        s_post: The same at ``year_b``.
        s_ref: Median ``s_t`` of the unburned reference that year.
        min_denominator: Smallest denominator for a defined index.
        clip: Range the index is clipped to.

    Returns:
        Float32 array of the broadcast shape.
    """
    st = np.asarray(s_t, dtype=np.float64)
    sp = np.asarray(s_post, dtype=np.float64)
    denom = np.float64(s_ref) - sp
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (st - sp) / denom
    r = np.clip(r, clip[0], clip[1])
    r = np.where(denom >= min_denominator, r, np.nan)
    return np.asarray(r, dtype=np.float32)


def recovery_metrics(
    s_burned: np.ndarray,
    s_post_burned: np.ndarray,
    s_unburned: np.ndarray,
    *,
    recovery_sim: float = DEFAULT_RECOVERY_SIM,
    unburned_percentile: float = UNBURNED_PERCENTILE,
    min_denominator: float = MIN_DENOMINATOR,
) -> dict[str, float | int]:
    """One year's recovery figures from burned and unburned similarities to ``year_a``.

    Args:
        s_burned: ``s_t`` of burned pixels.
        s_post_burned: ``s_post`` of the same pixels, in the same order.
        s_unburned: ``s_t`` of the unburned reference pixels.
        recovery_sim: Level for the brief's ``sim90_fraction``.
        unburned_percentile: Percentile of unburned ``s_t`` a burned pixel must reach to count as
            recovered.
        min_denominator: Passed to :func:`recovery_index`.

    Returns:
        ``n_burned`` (burned pixels with a finite ``s_t``), ``n_unburned``, ``median_sim_burned``,
        ``median_sim_unburned``, ``unburned_sim_p10``, ``recovered_fraction``,
        ``median_recovery_index``, ``n_recovery_index_undefined`` and ``sim90_fraction``. Ratios
        and medians are NaN when their pixels are missing.
    """
    sb = np.asarray(s_burned, dtype=np.float64).ravel()
    sp = np.asarray(s_post_burned, dtype=np.float64).ravel()
    if sb.shape != sp.shape:
        raise ValueError(f"s_burned has {sb.size} values but s_post_burned has {sp.size}.")
    keep = np.isfinite(sb) & np.isfinite(sp)
    sb, sp = sb[keep], sp[keep]
    su = np.asarray(s_unburned, dtype=np.float64).ravel()
    su = su[np.isfinite(su)]
    nan = math.nan
    median_b = float(np.median(sb)) if sb.size else nan
    median_u = float(np.median(su)) if su.size else nan
    p_u = float(np.percentile(su, unburned_percentile)) if su.size else nan
    recovered = float(np.mean(sb >= p_u)) if sb.size and su.size else nan
    r = recovery_index(sb, sp, median_u, min_denominator=min_denominator)
    defined = r[np.isfinite(r)]
    return {
        "n_burned": int(sb.size),
        "n_unburned": int(su.size),
        "median_sim_burned": median_b,
        "median_sim_unburned": median_u,
        "unburned_sim_p10": p_u,
        "recovered_fraction": recovered,
        "median_recovery_index": float(np.median(defined)) if defined.size else nan,
        "n_recovery_index_undefined": int(r.size - defined.size),
        "sim90_fraction": float(np.mean(sb > recovery_sim)) if sb.size else nan,
    }


def recovery_reference(
    plan: AOIPlan,
    dist: Mapping[str, np.ndarray],
    threshold: float,
    burned: Mapping[str, np.ndarray],
    *,
    distance_share: float = REFERENCE_DISTANCE_SHARE,
    buffer_px: int = REFERENCE_BUFFER_PX,
) -> dict[str, np.ndarray]:
    """The recovery reference: classified pixels shown to be unchanged between ``year_a`` and ``year_b``.

    A pixel qualifies when its distance is at most ``distance_share`` of the threshold and it lies
    more than ``buffer_px`` pixels (8-connected steps, across zone seams) from every burned pixel.
    Pixels just under the threshold, scar edges and unburned islands are left out: when the AOI lies
    mostly inside a scar, they are largely fire-affected land, and a reference made of them is less
    similar to its pre-fire self than land that did not burn, which inflates recovery. The buffer is
    drawn around the cleaned burn mask, not around every pixel over the threshold, so scattered
    speckle across a large AOI does not carve the reference away.

    Args:
        plan: The AOI plan.
        dist: ``year_a`` to ``year_b`` distance per zone, NaN where not classified.
        threshold: The burn threshold.
        burned: The cleaned burn mask per zone.
        distance_share: Largest distance, as a share of the threshold.
        buffer_px: Pixels kept clear around every burned pixel.

    Returns:
        Bool mask per zone.
    """
    near = grow_across_zones(burned, plan, pixels=buffer_px)
    cut = distance_share * threshold
    out = {}
    for z in plan.zones:
        d = dist[z.zone]
        with np.errstate(invalid="ignore"):
            out[z.zone] = np.isfinite(d) & (d <= cut) & ~near[z.zone] & z.mask
    return out


def _similarity_stats(values: np.ndarray) -> dict[str, float | int | None]:
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return {"pixels": 0, "median": None, "p10": None, "share_above_0_9": None}
    return {
        "pixels": int(v.size),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, UNBURNED_PERCENTILE)),
        "share_above_0_9": float(np.mean(v > DEFAULT_RECOVERY_SIM)),
    }


# --------------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------------


def score_against_reference(
    plan: io.AOIPlan,
    burned: Mapping[str, np.ndarray],
    reference: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Score the burned mask against a reference over every classified AOI pixel of every zone.

    Args:
        plan: The AOI plan.
        burned: Burned mask per zone.
        reference: Reference mask per zone.
        classified: Pixels the app classified (valid in both years, same radar regime) per zone.

    Returns:
        ``burned`` (:class:`validate.BinaryScores` merged over zones), ``per_zone`` scores,
        ``recall_inside_reference`` (share of all AOI reference pixels, classified or not, that are
        burned), ``burned_outside_reference_fraction`` (share of burned pixels outside the
        reference), ``iou_below_0_6``, ``reference_pixels`` and ``reference_pixels_unclassified``.
    """
    parts: dict[str, validate.BinaryScores] = {}
    ref_px = ref_hit = ref_unclassified = burned_px = burned_out = 0
    for z in plan.zones:
        b = np.asarray(burned[z.zone], dtype=bool) & z.mask
        r = np.asarray(reference[z.zone], dtype=bool) & z.mask
        v = np.asarray(classified[z.zone], dtype=bool) & z.mask
        parts[z.zone] = validate.binary_scores(b, r, valid=v)
        ref_px += int(r.sum())
        ref_hit += int((b & r).sum())
        ref_unclassified += int((r & ~v).sum())
        burned_px += int(b.sum())
        burned_out += int((b & ~r).sum())
    merged = validate.merge_scores(list(parts.values()))
    return {
        "burned": merged,
        "per_zone": parts,
        "recall_inside_reference": ref_hit / ref_px if ref_px else None,
        "burned_outside_reference_fraction": burned_out / burned_px if burned_px else None,
        "iou_below_0_6": merged.iou is not None and merged.iou < IOU_STOP,
        "reference_pixels": ref_px,
        "reference_pixels_unclassified": ref_unclassified,
    }


def severity_burned(severity: np.ndarray, min_class: int) -> np.ndarray:
    """Pixels of an MTBS dnbr6 map counted as burned: class ``min_class`` to 4 (high).

    The codes are ordinal from 1 (unburned to low) to 4 (high) only: 0 (background, outside the
    mapped burn) and 5 (increased greenness) are unburned land, 6 (non-processing) is not mapped and
    :data:`SEVERITY_UNMAPPED` is not covered, so ``class >= min_class`` stops at 4.
    """
    s = np.asarray(severity)
    return (s >= min_class) & (s <= SEVERITY_HIGH)


def _check_min_class(min_class: int) -> int:
    if isinstance(min_class, bool) or not isinstance(min_class, int | np.integer) or not 1 <= min_class <= 4:
        raise AppInputError(
            f"severity_min_class ({min_class!r}) must be a burn severity class from 1 to 4: 1 unburned to low, 2 low, "
            "3 moderate (the default), 4 high."
        )
    return int(min_class)


def load_severity(plan: io.AOIPlan, path: Path | str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """A burn severity GeoTIFF in MTBS dnbr6 codes, read onto every zone plan with nearest resampling.

    Any CRS: only the window around each plan is read and reprojected
    (:func:`validate._raster_on_plan`). AOI pixels the raster covers keep their code, 0
    (background) included; pixels outside the raster, on its nodata or outside the AOI are
    :data:`SEVERITY_UNMAPPED`. The file is refused, naming it, when it cannot be read, when a value
    it holds over the AOI is not one of the codes 0 to 6 (values between the AOI polygon and its
    bounding window are never used, so they are not checked), when it does not overlap the AOI, or
    when no AOI pixel holds a class from 1 to 5.

    Args:
        plan: The AOI plan.
        path: The GeoTIFF (band 1).

    Returns:
        ``(classes, info)``: uint8 codes per zone, and a record of the file (name, sha256, CRS,
        resolution, ``nodata``, the ``TESSERA_APPS_*`` tags, ``source_tag`` and ``proxy``, true
        when the tag ``TESSERA_APPS_SOURCE`` is ``sentinel2_dnbr``).

    Raises:
        AppInputError: The file cannot be used, with a message that names it.
    """
    src = Path(path)
    hint = "Pass a GeoTIFF of MTBS dnbr6 codes (0 to 6) with --severity."
    raw: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        with rasterio.open(src) as ds:
            tags = {str(k): str(v) for k, v in ds.tags().items()}
            crs = None if ds.crs is None else ds.crs.to_string()
            res = [float(abs(r)) for r in ds.res]
            nodata = None if ds.nodata is None or math.isnan(ds.nodata) else float(ds.nodata)
        for z in plan.zones:
            raw[z.zone] = validate._raster_on_plan(src, z)
    except (OSError, ValueError, RuntimeError, rasterio.errors.RasterioError) as exc:
        raise AppInputError(f"Could not read the severity file {src}: {exc}. {hint}") from exc
    bad: set[float] = set()
    for z in plan.zones:
        values, covered = raw[z.zone]
        v = values[covered & z.mask]
        wrong = ~np.isin(v, SEVERITY_CODES)
        if wrong.any():
            bad.update(float(x) for x in np.unique(v[wrong])[:5])
    if bad:
        shown = ", ".join(f"{x:g}" for x in sorted(bad)[:5])
        raise AppInputError(
            f"The severity file {src} holds values outside the MTBS dnbr6 codes 0 to 6 ({shown}) over the AOI; it may "
            f"be a continuous dNBR rather than its classes, or a fill value its nodata tag does not declare. {hint}"
        )
    classes = {
        z.zone: np.where(raw[z.zone][1] & z.mask, raw[z.zone][0], SEVERITY_UNMAPPED).astype(np.uint8)
        for z in plan.zones
    }
    overlap = sum(int((raw[z.zone][1] & z.mask).sum()) for z in plan.zones)
    if overlap == 0:
        raise AppInputError(f"The severity file {src} does not overlap the AOI. {hint}")
    mapped = sum(int(((c >= 1) & (c <= SEVERITY_GREENER)).sum()) for c in classes.values())
    if mapped == 0:
        raise AppInputError(
            f"The severity file {src} has no pixel in classes 1 to 5 inside the AOI (only background or "
            f"non-processing), so there is nothing to score. {hint}"
        )
    source = tags.get(SEVERITY_SOURCE_TAG)
    info = {
        "file": src.name,
        "sha256": export.file_sha256(src),
        "crs": crs,
        "resolution": res,
        "nodata": nodata,
        "source_tag": source,
        "proxy": source == SEVERITY_PROXY_SOURCE,
        "tags": {k: v for k, v in sorted(tags.items()) if k.startswith("TESSERA_APPS_")},
        "codes": {str(k): v for k, v in SEVERITY_NAMES.items()},
    }
    return classes, info


def score_against_severity(
    plan: io.AOIPlan,
    burned: Mapping[str, np.ndarray],
    severity: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
    *,
    min_class: int = DEFAULT_SEVERITY_MIN_CLASS,
) -> dict[str, Any]:
    """Score the burned mask against a burn severity map, over classified AOI pixels of every zone.

    Classified AOI pixels the map covers in classes 0 to 5 are scored; those it does not cover
    (:data:`SEVERITY_UNMAPPED`: outside its extent or on its nodata) and those in class 6
    (non-processing) are left out and counted. The binary scores take burned as class
    ``min_class`` to 4 (:func:`severity_burned`), and also class 3 to 4 (moderate or worse, which
    the stop rule reads whatever ``min_class`` is) and class 2 to 4 (low or worse). Background (0,
    outside the mapped burn) and increased greenness (5) are unburned land there, so detections in
    them count as false positives. Zones are merged with :func:`validate.merge_scores`.

    Args:
        plan: The AOI plan.
        burned: The app's burned mask per zone.
        severity: dnbr6 codes per zone, :data:`SEVERITY_UNMAPPED` where the map has no data
            (:func:`load_severity`).
        classified: Pixels the app classified per zone.
        min_class: Lowest class counted as burned, 1 to 4.

    Returns:
        ``min_class`` and its name; ``classes``: per class 0 to 5 the classified ``pixels`` and
        ``hectares``, the ``detected_pixels`` and ``detected_hectares`` the app called burned and
        their share, ``detection_rate``; ``min_class_or_worse``, ``moderate_or_worse`` and
        ``low_or_worse`` (:class:`validate.BinaryScores` merged over zones) with ``per_zone``
        scores of each; ``detections_by_class``: for every code 0 to 6, and ``unmapped``, the
        ``pixels`` and ``share`` of the app's burned pixels in it; ``excluded``: classified pixels
        the map does not cover (``unmapped_pixels``) and in class 6 (``non_processing_pixels``);
        ``unclassified_pixels_in_classes_1_to_5``; ``pixels_scored``; ``stop_rule_min_class`` and
        ``iou_below_0_6`` (of ``moderate_or_worse``, the IoU the stop rule reads).

    Raises:
        AppInputError: ``min_class`` is not 1 to 4.
    """
    k = _check_min_class(min_class)
    bases = {"min_class_or_worse": k, "moderate_or_worse": SEVERITY_STOP_CLASS, "low_or_worse": SEVERITY_LOW}
    per: dict[int, dict[str, validate.BinaryScores]] = {c: {} for c in sorted(set(bases.values()))}
    n_codes = len(SEVERITY_CODES)
    class_px = np.zeros(n_codes, dtype=np.int64)
    burned_by_class = np.zeros(n_codes, dtype=np.int64)
    unclassified = np.zeros(n_codes, dtype=np.int64)
    unmapped_px = unmapped_burned = 0
    for z in plan.zones:
        s = np.where(z.mask, np.asarray(severity[z.zone]), SEVERITY_UNMAPPED).astype(np.int64)
        mapped = s != SEVERITY_UNMAPPED
        if ((s < 0) | (mapped & (s >= n_codes))).any():
            raise ValueError(
                f"Severity codes for {z.zone} must lie in 0 to {n_codes - 1}, or be {SEVERITY_UNMAPPED} where unmapped."
            )
        v = np.asarray(classified[z.zone], dtype=bool) & z.mask
        b = np.asarray(burned[z.zone], dtype=bool) & v
        scored = v & mapped & (s <= SEVERITY_GREENER)
        for c, parts in per.items():
            parts[z.zone] = validate.binary_scores(b, severity_burned(s, c), valid=scored)
        class_px += np.bincount(s[v & mapped], minlength=n_codes)
        burned_by_class += np.bincount(s[b & mapped], minlength=n_codes)
        unclassified += np.bincount(s[z.mask & ~v & mapped], minlength=n_codes)
        unmapped_px += int((v & ~mapped).sum())
        unmapped_burned += int((b & ~mapped).sum())
    merged = {name: validate.merge_scores(list(per[c].values())) for name, c in bases.items()}
    stop: validate.BinaryScores = merged["moderate_or_worse"]
    n_burned = int(burned_by_class.sum()) + unmapped_burned
    classes = {}
    for c in SEVERITY_SCORED:
        px, hit = int(class_px[c]), int(burned_by_class[c])
        classes[str(c)] = {
            "name": SEVERITY_NAMES[c],
            "pixels": px,
            "hectares": px * HA_PER_PIXEL,
            "detected_pixels": hit,
            "detected_hectares": hit * HA_PER_PIXEL,
            "detection_rate": hit / px if px else None,
        }
    where = {str(c): int(burned_by_class[c]) for c in SEVERITY_CODES}
    where["unmapped"] = unmapped_burned
    return {
        "min_class": k,
        "min_class_name": SEVERITY_NAMES[k],
        "classes": classes,
        **merged,
        "per_zone": {name: dict(per[c]) for name, c in bases.items()},
        "detections_by_class": {
            c: {"pixels": n, "share": n / n_burned if n_burned else None} for c, n in where.items()
        },
        "burned_pixels": n_burned,
        "excluded": {
            "unmapped_pixels": unmapped_px,
            "non_processing_pixels": int(class_px[SEVERITY_NON_PROCESSING]),
        },
        "unclassified_pixels_in_classes_1_to_5": int(unclassified[1 : SEVERITY_GREENER + 1].sum()),
        "pixels_scored": merged["min_class_or_worse"].n,
        "stop_rule_min_class": SEVERITY_STOP_CLASS,
        "iou_below_0_6": stop.iou is not None and stop.iou < IOU_STOP,
    }


def _severity_caveats(info: Mapping[str, Any], scores: Mapping[str, Any]) -> list[str]:
    out = [SEVERITY_THRESHOLD_CAVEAT, SEVERITY_RESOLUTION_CAVEAT, SEVERITY_EXCLUDED_CAVEAT]
    out.append(f"Severity reference: the file {info['file']} (sha256 {info['sha256']}), supplied with --severity.")
    if info.get("nodata") == 0:
        out.append(SEVERITY_NODATA_ZERO_CAVEAT)
    if info.get("proxy"):
        out.append(SEVERITY_PROXY_CAVEAT)
    if scores.get("iou_below_0_6"):
        out.append(
            f"IoU against severity class {SEVERITY_STOP_CLASS} ({SEVERITY_NAMES[SEVERITY_STOP_CLASS]}) or worse is "
            f"below {IOU_STOP}, the brief's stop-and-report level: read it with the per-class detection and the "
            "caveats above rather than tuning the threshold to the reference."
        )
    return out


def _scores_record(scores: validate.BinaryScores) -> dict[str, float | None]:
    return {"iou": scores.iou, "precision": scores.precision, "recall": scores.recall, "f1": scores.f1}


def _severity_summary(
    scores: Mapping[str, Any],
    info: Mapping[str, Any],
    severity: Mapping[str, np.ndarray],
    reference_mask: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """The summary's compact view of :func:`score_against_severity`.

    ``iou`` and its neighbours are the scores against ``min_class`` or worse; ``moderate_or_worse``
    holds the scores the stop rule reads, and ``iou_below_0_6`` is its verdict.
    """
    merged: validate.BinaryScores = scores["min_class_or_worse"]
    k = int(scores["min_class"])
    n_ref = sum(int(m.sum()) for m in reference_mask.values())
    n_ref_burned = sum(int((reference_mask[z] & severity_burned(severity[z], k)).sum()) for z in reference_mask)
    return {
        "file": info["file"],
        "proxy": bool(info.get("proxy")),
        "source_tag": info.get("source_tag"),
        "min_class": k,
        "min_class_name": scores["min_class_name"],
        **_scores_record(merged),
        "pixels_scored": merged.n,
        "iou_below_0_6": scores["iou_below_0_6"],
        "moderate_or_worse": _scores_record(scores["moderate_or_worse"]),
        "low_or_worse": _scores_record(scores["low_or_worse"]),
        "detection_rate_by_class": {c: row["detection_rate"] for c, row in scores["classes"].items()},
        "detections_by_class": {c: row["share"] for c, row in scores["detections_by_class"].items()},
        "excluded": dict(scores["excluded"]),
        # Share of the recovery reference (land taken as unburned) in the severity map's burned classes.
        "recovery_reference_burned_fraction": n_ref_burned / n_ref if n_ref else None,
    }


def stop_rule(validation: Mapping[str, Any]) -> dict[str, Any] | None:
    """The brief's IoU 0.6 stop rule (section 8), as the survey and the owner's decision amended it.

    With a severity map, it applies to the IoU against moderate or worse
    (:data:`SEVERITY_STOP_CLASS`), whatever ``--severity-min-class`` is, so the verdict cannot be
    moved by choosing the class; otherwise to the outline IoU, read together with recall inside the
    reference.

    Returns:
        ``basis`` (``"severity"`` or ``"reference"``), ``iou``, ``below_0_6``, for the severity map
        ``min_class`` (always 3) and for the outline ``recall_inside_reference``; None when nothing
        was scored.
    """
    sev = validation.get("severity")
    if isinstance(sev, Mapping):
        iou = (sev.get("moderate_or_worse") or {}).get("iou")
        return {
            "basis": "severity",
            "min_class": SEVERITY_STOP_CLASS,
            "iou": iou,
            "below_0_6": iou is not None and iou < IOU_STOP,
        }
    if "iou" in validation:
        iou = validation.get("iou")
        return {
            "basis": "reference",
            "iou": iou,
            "recall_inside_reference": validation.get("recall_inside_reference"),
            "below_0_6": iou is not None and iou < IOU_STOP,
        }
    return None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def headline(validation: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The validation headline, in the order the owner set: severity first, then the outline.

    With a severity map: its IoU against ``min_class`` or worse (and, when ``min_class`` is not 3,
    the IoU against moderate or worse that the stop rule reads) and the detection rate per class
    (high first), then, when an outline reference was scored, recall inside it and its IoU. Without
    one: recall inside the reference, then IoU.

    Returns:
        ``basis``, ``order`` (the keys in reading order), those values and ``text``; None when
        nothing was scored.
    """
    if not validation:
        return None
    out: dict[str, Any] = {}
    order: list[str] = []
    parts: list[str] = []
    sev = validation.get("severity")
    if isinstance(sev, Mapping):
        out["basis"] = "severity"
        rates = sev.get("detection_rate_by_class") or {}
        out["severity_min_class"] = sev.get("min_class")
        out["severity_iou"] = sev.get("iou")
        out["detection_rate_by_class"] = dict(rates)
        order += ["severity_iou", "detection_rate_by_class"]
        what = "Sentinel-2 dNBR proxy" if sev.get("proxy") else "severity map"
        stop = ""
        if sev.get("min_class") != SEVERITY_STOP_CLASS:
            stop_iou = (sev.get("moderate_or_worse") or {}).get("iou")
            stop = f" (the stop rule reads IoU {_num(stop_iou)} against moderate or worse)"
        detected = ", ".join(f"{SEVERITY_NAMES[c]} {_pct(rates.get(str(c)))}" for c in SEVERITY_HEADLINE_ORDER)
        parts.append(
            f"Severity ({what}): IoU {_num(sev.get('iou'))} against {sev.get('min_class_name')} or worse{stop}; "
            f"detected {detected}"
        )
    if "iou" in validation:
        out.setdefault("basis", "reference")
        out["recall_inside_reference"] = validation.get("recall_inside_reference")
        out["iou"] = validation.get("iou")
        order += ["recall_inside_reference", "iou"]
        recall = _num(validation.get("recall_inside_reference"))
        label = OUTLINE_LABELS.get(str(validation.get("source")), "Reference")
        parts.append(f"{label}: recall inside it {recall}, IoU {_num(validation.get('iou'))}")
    if not parts:
        return None
    out["order"] = order
    out["text"] = ". ".join(parts) + "."
    return out


def _looks_like_hansen(path: Path) -> bool:
    """Whether a reference file names Hansen in its file name or (for small files) its contents."""
    if "hansen" in path.name.lower():
        return True
    try:
        if path.stat().st_size <= 50_000_000:
            return b"hansen" in path.read_bytes().lower()
    except OSError:
        return False
    return False


@dataclasses.dataclass
class _Reference:
    """The validation references, loaded before any chunk read (or why the outline could not be).

    Attributes:
        outline: An outline reference (``--reference`` or NIFC) was requested.
        masks: The outline reference per zone.
        info: Where the outline came from.
        error: Why the outline could not be loaded.
        severity: The ``--severity`` dnbr6 codes per zone.
        severity_info: Where the severity map came from.
        severity_min_class: The lowest severity class counted as burned.
    """

    outline: bool = False
    masks: dict[str, np.ndarray] | None = None
    info: dict[str, Any] | None = None
    error: str | None = None
    severity: dict[str, np.ndarray] | None = None
    severity_info: dict[str, Any] | None = None
    severity_min_class: int = DEFAULT_SEVERITY_MIN_CLASS


def _reference_masks(
    ctx: RunContext, reference: Path | None, fire_year: int, fire_name: str | None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The reference on every zone plan and a record of where it came from.

    Raises:
        AppInputError: A ``--reference`` file cannot be read.
        validate.ReferenceDataError: NIFC cannot be reached, or has no perimeter (Unavailable included).
    """
    if reference is not None:
        refs = {z.zone: aoi_reference(z, reference) for z in ctx.plan.zones}
        info = {
            "source": "file",
            "file": reference.name,
            "sha256": export.file_sha256(reference),
            "hansen_derived": _looks_like_hansen(reference),
        }
        return refs, info
    west, south, east, north = (float(b) for b in ctx.aoi.geometry.bounds)
    bounds = (west, south, east, north)
    geoms = validate.nifc_perimeters(bounds, year=fire_year, name=fire_name)
    if not geoms:
        raise validate.ReferenceDataError(
            f"NIFC returned no perimeter for {fire_name or 'any fire'} in {fire_year} within {bounds}; check "
            "--fire-year and --fire-name, or pass --reference."
        )
    refs = {z.zone: validate.reference_from_geometries(z, geoms) for z in ctx.plan.zones}
    info = {
        "source": "nifc",
        "dataset": "NIFC InterAgency Fire Perimeter History",
        "url": validate.NIFC_HISTORY_URL,
        "fire_year": fire_year,
        "fire_name": fire_name,
        "perimeters": len(geoms),
        "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "hansen_derived": False,
    }
    return refs, info


def _load_reference(
    ctx: RunContext,
    state: _Reference,
    *,
    reference: Path | None,
    fire_year: int,
    fire_name: str | None,
    strict: bool,
    outline: bool = True,
    severity: Path | None = None,
    severity_min_class: int = DEFAULT_SEVERITY_MIN_CLASS,
) -> None:
    """Load the validation references into ``state``: the preflight, before any chunk read.

    The ``--severity`` map first (:func:`load_severity`; any problem with it is a usage error), then,
    with ``outline``, the ``--reference`` file or NIFC.

    Raises:
        AppInputError: The ``--severity`` or ``--reference`` file cannot be used.
        StrictValidationError: ``strict`` is set and NIFC cannot be reached or has no perimeter.
    """
    if severity is not None:
        state.severity, state.severity_info = load_severity(ctx.plan, severity)
        state.severity_min_class = severity_min_class
    state.outline = bool(outline and ctx.validate)
    if not state.outline:
        return
    try:
        state.masks, state.info = _reference_masks(ctx, reference, fire_year, fire_name)
    except validate.ReferenceDataError as exc:
        if strict:
            raise StrictValidationError(
                f"--strict-validate is set and the validation reference could not be loaded: {exc}"
            ) from exc
        state.error = str(exc)
        logger.warning("Validation will be skipped, the reference could not be loaded: %s", exc)


def _caveats(info: Mapping[str, Any], scores: Mapping[str, Any], *, stop_rule: bool = True) -> list[str]:
    """The outline reference's caveats; ``stop_rule`` when its IoU is the one the stop rule reads."""
    out = []
    if info["source"] == "nifc":
        out.append(NIFC_CAVEAT)
        out.append(f"Reference: {info['dataset']}, {info['url']}, retrieved {info['retrieved_utc']}.")
    else:
        out.append(OUTLINE_CAVEAT)
        out.append(f"Reference: the file {info['file']} (sha256 {info['sha256']}), supplied with --reference.")
    if info.get("hansen_derived"):
        out.append(HANSEN_CAVEAT)
    if stop_rule and scores.get("iou_below_0_6"):
        out.append(
            f"IoU is below {IOU_STOP}, the brief's stop-and-report level: read it with recall inside the reference "
            "and the caveats above rather than tuning the threshold to the reference."
        )
    return out


def _validate(
    ctx: RunContext,
    state: _Reference,
    *,
    burned: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
    reference_mask: Mapping[str, np.ndarray],
    extra_caveats: Sequence[str] = (),
) -> tuple[dict[str, Any], bool]:
    """Score against the outline reference and the severity map, and write ``validation.json``.

    ``validation.json`` holds the outline scores (``--reference`` or NIFC) and, with ``--severity``,
    a ``severity`` block next to them (:func:`score_against_severity`). It is written whenever
    something was scored; the report vouches for it (the second value) only when every requested
    reference was scored, so when the outline could not be loaded the severity scores are still
    written and a rerun redoes the validation. The stop rule (:func:`stop_rule`) warns on the
    severity IoU against moderate or worse when a severity map was given, else on the outline IoU;
    the warning ends with :data:`STOP_WARNING`.

    Returns:
        The summary's validation block, and whether the report vouches for ``validation.json``.
    """
    plan = ctx.plan
    validation: dict[str, Any] = {}
    scores: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    caveats = [SCOPE_CAVEAT]
    complete = True
    severity_scores = None
    if state.severity is not None and state.severity_info is not None:
        severity_scores = score_against_severity(
            plan, burned, state.severity, classified, min_class=state.severity_min_class
        )
    if state.outline and (state.masks is None or state.info is None):
        ctx.warn(f"{VALIDATION_SKIPPED}: {state.error}")
        validation["error"] = state.error
        # When the severity map is scored, validation.json is written, and it must say on its own
        # that the outline was asked for and not scored, not leave that to the report.
        reference["outline"] = {"scored": False, "error": state.error}
        caveats.append(f"{OUTLINE_NOT_SCORED}: {state.error}")
        complete = False
    elif state.outline and state.masks is not None and state.info is not None:
        outline = score_against_reference(plan, burned, state.masks, classified)
        scores.update(outline)
        reference.update(state.info)
        caveats += _caveats(state.info, outline, stop_rule=severity_scores is None)
        merged: validate.BinaryScores = outline["burned"]
        n_ref = sum(int(reference_mask[z.zone].sum()) for z in plan.zones)
        n_ref_inside = sum(int((reference_mask[z.zone] & state.masks[z.zone]).sum()) for z in plan.zones)
        validation.update(
            {
                "source": state.info["source"],
                "iou": merged.iou,
                "precision": merged.precision,
                "recall": merged.recall,
                "f1": merged.f1,
                "recall_inside_reference": outline["recall_inside_reference"],
                "burned_outside_reference_fraction": outline["burned_outside_reference_fraction"],
                "iou_below_0_6": outline["iou_below_0_6"],
                "pixels_scored": merged.n,
                # Share of the recovery reference (land taken as unburned) inside the validation reference.
                "recovery_reference_inside_fraction": n_ref_inside / n_ref if n_ref else None,
            }
        )
    if severity_scores is not None and state.severity is not None and state.severity_info is not None:
        scores["severity"] = severity_scores
        reference["severity"] = state.severity_info
        caveats += _severity_caveats(state.severity_info, severity_scores)
        validation["severity"] = _severity_summary(severity_scores, state.severity_info, state.severity, reference_mask)
    if not scores:
        return validation, False
    validate.write_validation_json(
        ctx.paths.validation, scores=scores, reference=reference, caveats=[*caveats, *extra_caveats]
    )
    rule = stop_rule(validation)
    validation["stop_rule"] = rule
    if rule is not None and rule["below_0_6"]:
        if rule["basis"] == "severity":
            ctx.warn(
                f"IoU {rule['iou']:.3f} against severity class {rule['min_class']} "
                f"({SEVERITY_NAMES[rule['min_class']]}) or worse is below {IOU_STOP}: {STOP_WARNING}"
            )
        else:
            ctx.warn(
                f"IoU {rule['iou']:.3f} is below {IOU_STOP} (recall inside the reference "
                f"{_num(rule['recall_inside_reference'])}): {STOP_WARNING}"
            )
    return validation, complete


# --------------------------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass
class _Zone:
    """Per-pixel results on one zone plan's grid, filled block by block."""

    plan: ZonePlan
    dist: np.ndarray
    sims: dict[int, np.ndarray]
    radar_free_a: np.ndarray
    radar_free_b: np.ndarray
    rgb_a: np.ndarray | None
    rgb_b: np.ndarray | None
    pre: np.ndarray | None
    valid_a: int = 0

    @classmethod
    def empty(cls, plan: ZonePlan, later_years: Sequence[int], *, prefire: bool) -> _Zone:
        h, w = plan.shape
        return cls(
            plan=plan,
            dist=np.full((h, w), np.nan, dtype=np.float32),
            sims={t: np.full((h, w), np.nan, dtype=np.float32) for t in later_years},
            radar_free_a=np.zeros((h, w), dtype=bool),
            radar_free_b=np.zeros((h, w), dtype=bool),
            rgb_a=np.zeros((3, h, w), dtype=np.uint8),
            rgb_b=np.zeros((3, h, w), dtype=np.uint8),
            pre=np.full((h, w), np.nan, dtype=np.float32) if prefire else None,
        )


def _all_blocks(ctx: RunContext) -> list[tuple[ZonePlan, slice, slice]]:
    return [(z, rs, cs) for z in ctx.plan.zones for rs, cs in z.blocks(ctx.reader.chunk_shape(z.zone), ctx.block)]


def _spread(n: int, k: int) -> list[int]:
    """Up to ``k`` indices spread evenly over ``range(n)``."""
    if n <= 0 or k <= 0:
        return []
    return sorted({int(i) for i in np.linspace(0, n - 1, min(n, k)).round()})


def _sample_vectors(win: io.Window, mask: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray | None:
    """Up to ``k`` dequantized vectors of valid ``mask`` pixels, drawn without replacement."""
    assert win.embeddings is not None
    sel = win.valid & (win.scales > 0) & mask
    idx = np.flatnonzero(sel)
    k = min(k, idx.size)
    if k <= 0:
        return None
    pick = np.sort(rng.choice(idx.size, size=k, replace=False))
    rr, cc = np.unravel_index(idx[pick], sel.shape)
    return win.embeddings[rr, cc].astype(np.float32) * win.scales[rr, cc][:, None]


def _fit_basis(ctx: RunContext) -> tuple[metrics.PCABasis, dict[str, Any]]:
    """Fit one PCA basis on ``year_a`` pixels sampled from up to :data:`PCA_FIT_BLOCKS` blocks.

    The blocks are spread evenly over the AOI's block list and read for ``year_a`` only (embeddings
    and scales), so the basis exists before the main pass colours every block of both years.

    Raises:
        WildfireError: No valid ``year_a`` pixel in the AOI.
    """
    jobs = _all_blocks(ctx)
    first = _spread(len(jobs), PCA_FIT_BLOCKS)
    per_block = max(PCA_SAMPLE_PER_BLOCK, PCA_SAMPLE_CAP // max(len(first), 1))
    rng = np.random.default_rng(SEED)
    samples: list[np.ndarray] = []
    total = 0
    used = 0
    # Blocks with no valid pixel give nothing; then keep going through the rest in order.
    chosen = set(first)
    order = first + [i for i in range(len(jobs)) if i not in chosen]
    for count, i in enumerate(order):
        if total >= PCA_SAMPLE_CAP or (count >= len(first) and total > 0):
            break
        z, rs, cs = jobs[i]
        win = ctx.reader.read_window(z.zone, ctx.year_a, rs, cs, layers=("embeddings", "scales"))
        ctx.mark_read([ctx.year_a])
        got = _sample_vectors(win, z.submask(rs, cs), min(per_block, PCA_SAMPLE_CAP - total), rng)
        used += 1
        if got is not None:
            samples.append(got)
            total += got.shape[0]
    if not samples:
        raise WildfireError(f"No valid {ctx.year_a} pixel inside the AOI; the store has no data there.")
    basis = metrics.fit_pca(np.concatenate(samples), sample=PCA_SAMPLE_CAP, seed=SEED)
    return basis, {"fit_year": ctx.year_a, "blocks_read": used, "blocks_total": len(jobs), "pixels": total}


def _rgb(win: io.Window, basis: metrics.PCABasis, mask: np.ndarray) -> np.ndarray:
    """(3, H, W) false colour with 0 only where there is no data, so 0 can be the COG nodata."""
    rgb = metrics.pca_rgb(win, basis, mask=mask)
    sel = win.valid & (win.scales > 0) & mask
    out = np.where(sel[..., None], np.maximum(rgb, 1), 0).astype(np.uint8)
    return np.moveaxis(out, -1, 0)


def _prefire_distance(w_prev: io.Window, w_a: io.Window, mask: np.ndarray) -> np.ndarray:
    """``year_a - 1`` to ``year_a`` distance on ``mask`` pixels valid in both, same regime; NaN elsewhere."""
    d = metrics.cosine_distance_map(w_prev, w_a)
    keep = mask & np.isfinite(d) & metrics.same_radar_regime(w_prev, w_a)
    return np.where(keep, d, np.float32(np.nan)).astype(np.float32)


def _stream(
    ctx: RunContext,
    basis: metrics.PCABasis,
    recovery_years: Sequence[int],
    prefire_year: int | None,
) -> tuple[dict[str, _Zone], dict[int, tuple[int, int]], set[tuple[str, int, int]]]:
    """One pass over every block of every zone, reading ``year_a`` and every recovery year at once.

    With ``prefire_year``, the :data:`PREFIRE_SAMPLE_BLOCKS` blocks spread over the AOI also read that
    year, for the pre-fire distance of the recovery reference check (and of the fallback, which then
    reads only the other blocks).

    Returns:
        The per-zone results; per recovery year t, (AOI pixels valid in ``year_a`` and t, those whose
        radar regime differs); and the blocks whose pre-fire distance is filled, as
        ``(zone, row start, col start)``.
    """
    year_a, year_b = ctx.year_a, ctx.year_b
    later = [t for t in recovery_years if t != year_b]
    pairs = {t: [0, 0] for t in recovery_years}
    zones: dict[str, _Zone] = {}
    sampled = set(_spread(len(_all_blocks(ctx)), PREFIRE_SAMPLE_BLOCKS)) if prefire_year is not None else set()
    done: set[tuple[str, int, int]] = set()
    index = 0
    for zp in ctx.plan.zones:
        st = _Zone.empty(zp, later, prefire=prefire_year is not None)
        assert st.rgb_a is not None and st.rgb_b is not None
        for blk in io.iter_plan_blocks(ctx.reader, zp, [year_a, *recovery_years], layers=LAYERS, block=ctx.block):
            ctx.mark_read([year_a, *recovery_years])
            r = slice(blk.rows.start - zp.rows.start, blk.rows.stop - zp.rows.start)
            c = slice(blk.cols.start - zp.cols.start, blk.cols.stop - zp.cols.start)
            m = blk.mask
            wa, wb = blk.windows[year_a], blk.windows[year_b]
            dist = metrics.cosine_distance_map(wa, wb)
            both = m & np.isfinite(dist)
            same = metrics.same_radar_regime(wa, wb)
            pairs[year_b][0] += int(both.sum())
            pairs[year_b][1] += int((both & ~same).sum())
            st.dist[r, c] = np.where(both & same, dist, np.float32(np.nan))
            for t in later:
                wt = blk.windows[t]
                sim = metrics.cosine_similarity_map(wa, wt)
                both_t = m & np.isfinite(sim)
                same_t = metrics.same_radar_regime(wa, wt)
                pairs[t][0] += int(both_t.sum())
                pairs[t][1] += int((both_t & ~same_t).sum())
                st.sims[t][r, c] = np.where(both_t & same_t, sim, np.float32(np.nan))
            st.radar_free_a[r, c] = wa.radar_free & m
            st.radar_free_b[r, c] = wb.radar_free & m
            st.valid_a += int((wa.valid & m).sum())
            st.rgb_a[:, r, c] = _rgb(wa, basis, m)
            st.rgb_b[:, r, c] = _rgb(wb, basis, m)
            if prefire_year is not None and st.pre is not None and index in sampled:
                w_prev = ctx.reader.read_window(zp.zone, prefire_year, blk.rows, blk.cols, layers=LAYERS)
                ctx.mark_read([prefire_year])
                st.pre[r, c] = _prefire_distance(w_prev, wa, m)
                done.add((zp.zone, blk.rows.start, blk.cols.start))
            index += 1
        zones[zp.zone] = st
    return zones, {t: (v[0], v[1]) for t, v in pairs.items()}, done


def _stable_distances(
    ctx: RunContext, zones: Mapping[str, _Zone], year_prev: int, done: set[tuple[str, int, int]]
) -> np.ndarray:
    """Distances between ``year_prev`` and ``year_a`` on AOI pixels valid in both, same regime.

    Blocks the main pass already compared (``done``) are not read again; the others are read for
    both years. Fills each zone's pre-fire distance array and returns every finite value.
    """
    for zp in ctx.plan.zones:
        st = zones[zp.zone]
        assert st.pre is not None
        for rs, cs in zp.blocks(ctx.reader.chunk_shape(zp.zone), ctx.block):
            key = (zp.zone, rs.start, cs.start)
            if key in done:
                continue
            wins = ctx.reader.read_windows(zp.zone, [year_prev, ctx.year_a], rs, cs, layers=LAYERS)
            ctx.mark_read([year_prev, ctx.year_a])
            sub = zp.submask(rs, cs)
            for w in wins.values():
                w.mask = sub
            r = slice(rs.start - zp.rows.start, rs.stop - zp.rows.start)
            c = slice(cs.start - zp.cols.start, cs.stop - zp.cols.start)
            st.pre[r, c] = _prefire_distance(wins[year_prev], wins[ctx.year_a], sub)
            done.add(key)
    vals = [st.pre[np.isfinite(st.pre)] for st in zones.values() if st.pre is not None]
    return np.concatenate(vals) if vals else np.empty(0, dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float:
    return num / den if den else math.nan


class _BandMosaic:
    """The main COG's bands, mosaicked across zones without holding the per-zone stack.

    :func:`export.mosaic_zones` runs once, on a band that holds each classified pixel's position in
    the concatenated zone arrays (and nodata elsewhere). Every band is then gathered through that
    index into one preallocated array, so only one band per zone is alive at a time (survey S18, the
    review's memory finding) and the reprojection runs once instead of once per band. The result is
    the mosaic of the whole stack: a pixel of the stack is empty exactly when it is unclassified, as
    in the index band, and each mosaic pixel (seam fills included) takes every band from the one
    source pixel whose index it holds.
    """

    def __init__(self, plan: AOIPlan, dst: str, classified: Mapping[str, np.ndarray], n_bands: int) -> None:
        self.plan = plan
        self.n_bands = n_bands
        self.sizes = [z.shape[0] * z.shape[1] for z in plan.zones]
        self.take: np.ndarray | None = None
        self.source: np.ndarray | None = None
        if len(plan.zones) == 1:
            z = plan.zones[0]
            shape, self.transform, self.crs = z.shape, z.transform, z.grid.crs
        else:
            if sum(self.sizes) >= np.iinfo(np.int32).max:
                raise WildfireError("The AOI windows hold too many pixels to mosaic; split the AOI.")
            pieces = []
            offset = 0
            for z, size in zip(plan.zones, self.sizes, strict=True):
                index = np.arange(offset, offset + size, dtype=np.int32).reshape(z.shape)
                pieces.append(GridArray(np.where(classified[z.zone] & z.mask, index, np.int32(-1)), z))
                offset += size
            mosaic, self.transform, self.crs = export.mosaic_zones(pieces, dst_zone=dst, nodata=-1)
            del pieces
            self.take = mosaic >= 0
            self.source = mosaic[self.take]
            shape = mosaic.shape
            del mosaic
        self.data: np.ndarray | None = np.full((n_bands, *shape), np.nan, dtype=np.float32)

    def put(self, i: int, per_zone: Mapping[str, np.ndarray]) -> None:
        """Mosaic band ``i`` from one array per zone plan (NaN where there is no value)."""
        assert self.data is not None
        if self.take is None or self.source is None:
            z = self.plan.zones[0]
            self.data[i] = np.where(z.mask, per_zone[z.zone], np.float32(np.nan))
            return
        flat = np.concatenate([np.asarray(per_zone[z.zone], dtype=np.float32).ravel() for z in self.plan.zones])
        self.data[i][self.take] = flat[self.source]


def _image(rgb: np.ndarray, step: int) -> np.ndarray:
    """(3, H, W) uint8 to a strided (h, w, 3) image with no-data pixels light grey."""
    img = np.moveaxis(rgb[:, ::step, ::step], 0, -1).copy()
    missing = (img == 0).all(axis=-1)
    img[missing] = np.round(np.array(export.PREVIEW_NAN_RGB) * 255).astype(np.uint8)
    return img


def _write_preview(
    path: Path,
    *,
    pca_a: np.ndarray,
    pca_b: np.ndarray,
    dist: np.ndarray,
    burned: np.ndarray,
    reference: np.ndarray | None,
    histogram: tuple[np.ndarray, np.ndarray] | None,
    choice: ThresholdChoice,
    table: pd.DataFrame,
    year_a: int,
    year_b: int,
) -> Path:
    """Write the six-panel preview with a matplotlib Figure (never pyplot).

    Panels: PCA before and after, distance, burned mask with the reference outline, the distance
    histogram with the threshold, and the recovery curve. Every image comes in already strided.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    grey = export.PREVIEW_NAN_RGB
    fig = Figure(figsize=(18, 11), dpi=100, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3)
    for ax, img, title in (
        (axes[0, 0], pca_a, f"PCA false colour {year_a} (before)"),
        (axes[0, 1], pca_b, f"PCA false colour {year_b} (after, same basis)"),
    ):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    finite = dist[np.isfinite(dist)]
    vmax = float(max(np.percentile(finite, 99.5), choice.threshold * 1.5)) if finite.size else 1.0
    shown = axes[0, 2].imshow(
        np.ma.masked_invalid(dist),
        cmap=mpl.colormaps["magma"].with_extremes(bad=grey),
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(shown, ax=axes[0, 2], label="cosine distance", shrink=0.85)
    axes[0, 2].set_title(f"Cosine distance {year_a} to {year_b}")
    axes[0, 2].set_axis_off()
    cmap = mpl.colors.ListedColormap(["#f2efe6", "#b2182b"]).with_extremes(bad=grey)
    axes[1, 0].imshow(np.ma.masked_invalid(burned), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    title = f"Burned (distance > {choice.threshold:.3f}, {choice.method})"
    if reference is not None:
        ref = (reference == 1).astype(np.float64)
        if ref.any():
            axes[1, 0].contour(ref, levels=[0.5], colors=["#2166ac"], linewidths=1.0)
        title += ", reference outline in blue"
    axes[1, 0].set_title(title)
    axes[1, 0].set_axis_off()
    ax = axes[1, 1]
    if histogram is not None:
        counts, edges = histogram
        ax.stairs(counts, edges, fill=True, color="#636363")
    ax.axvline(choice.threshold, color="#b2182b", label=f"threshold {choice.threshold:.3f} ({choice.method})")
    if choice.otsu is not None and choice.method != "otsu":
        ax.axvline(choice.otsu.threshold, color="#2166ac", linestyle="--", label=f"Otsu {choice.otsu.threshold:.3f}")
    ax.set_xlabel("cosine distance")
    ax.set_ylabel("pixels")
    ax.set_title("Distance histogram over the AOI")
    ax.legend(loc="upper right", fontsize="small")
    ax = axes[1, 2]
    low, high = -0.05, 1.05
    if len(table):
        ax.plot(table["year"], table["recovered_fraction"], marker="o", label="recovered fraction (p10 of reference)")
        ax.plot(table["year"], table["median_recovery_index"], marker="s", label="median recovery index R")
        ax.plot(table["year"], table["sim90_fraction"], marker="^", linestyle="--", label="share above 0.9 (brief)")
        ax.set_xticks(list(table["year"]))
        r = table["median_recovery_index"].to_numpy(dtype=np.float64)
        if np.isfinite(r).any():
            low = min(low, float(np.nanmin(r)) - 0.05)
            high = max(high, float(np.nanmax(r)) + 0.05)
    ax.axhline(0.0, color="#bdbdbd", linewidth=0.8)
    ax.set_ylim(low, high)
    ax.set_xlabel("year")
    ax.set_title(f"Recovery of burned pixels since {year_b}")
    ax.legend(loc="upper left", fontsize="small")
    final = Path(path)
    with export.atomic_path(final) as tmp, tmp.open("wb") as fh:
        fig.savefig(fh, format="png", dpi=100)
    return final


# --------------------------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WildfireResult:
    """What :func:`run_wildfire` did.

    Attributes:
        report: The report (read back from disk when the run was skipped).
        paths: The standard outputs.
        extra_paths: The PCA COGs, by name (``pca_a``, ``pca_b``).
        skipped: The outputs were current, so nothing was read or written.
        validation_written: ``wildfire_validation.json`` belongs to this report.
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


def default_fire_year(year_a: int, year_b: int) -> int:
    """The fire year when none is given: ``year_b - 1``, but never ``year_a`` (survey N11).

    ``year_a`` is the year before the fire, so when ``year_b`` is ``year_a + 1`` the fire year is
    ``year_b`` (and the report warns that ``year_b`` is the fire year).
    """
    return max(year_a + 1, year_b - 1)


def memory_estimate(plan: AOIPlan, *, n_recovery_years: int, years_per_block: int, block: tuple[int, int]) -> int:
    """Peak memory of a run in bytes, from the plan's windows and the block shape (survey S18).

    The larger of the streaming stage (per-pixel arrays plus one block of every year read) and the
    recovery and write stage (per-pixel arrays, the output mosaic and its index), with a margin.
    """
    window_px = sum(z.shape[0] * z.shape[1] for z in plan.zones)
    block_px = max((min(block[0], z.shape[0]) * min(block[1], z.shape[1]) for z in plan.zones), default=0)
    later = max(n_recovery_years - 1, 0)
    stream = (MEMORY_STREAM_PER_PIXEL + MEMORY_PER_LATER_YEAR_STREAM * later) * window_px
    stream += MEMORY_PER_BLOCK_PIXEL_YEAR * block_px * years_per_block
    write = (MEMORY_WRITE_PER_PIXEL + MEMORY_PER_LATER_YEAR_WRITE * later) * window_px
    return int(MEMORY_FIXED + MEMORY_MARGIN * max(stream, write))


def _check_args(
    *,
    year_a: int,
    year_b: int,
    fire_year: int,
    recovery_through: int,
    threshold: float | None,
    min_area_ha: float,
    recovery_sim: float,
    reference: Path | None,
    severity: Path | None = None,
    severity_min_class: int = DEFAULT_SEVERITY_MIN_CLASS,
) -> None:
    """Refuse arguments the app cannot use, before the store is opened.

    Raises:
        AppInputError: Says what to change.
    """
    _check_min_class(severity_min_class)
    if severity is not None and not severity.is_file():
        raise AppInputError(
            f"Severity file {severity} does not exist; pass a GeoTIFF of MTBS dnbr6 codes (0 to 6) with --severity."
        )
    if year_b <= year_a:
        raise AppInputError(
            f"year_b ({year_b}) must be after year_a ({year_a}): year_a is the year before the fire and year_b "
            "the year after it."
        )
    if year_a < FIRST_YEAR or year_b > LAST_YEAR:
        raise AppInputError(f"Years must lie in {FIRST_YEAR} to {LAST_YEAR}; got year_a {year_a}, year_b {year_b}.")
    if not year_b <= recovery_through <= LAST_YEAR:
        raise AppInputError(
            f"recovery_through ({recovery_through}) must lie between year_b ({year_b}) and {LAST_YEAR}, the last "
            "year in the store."
        )
    if not year_a <= fire_year <= year_b:
        raise AppInputError(
            f"fire_year ({fire_year}) must lie between year_a ({year_a}) and year_b ({year_b}): year_a is before "
            "the fire and year_b after it."
        )
    if threshold is not None and not (math.isfinite(threshold) and 0.0 < threshold < 2.0):
        raise AppInputError(f"threshold ({threshold}) must be a cosine distance between 0 and 2.")
    if not (math.isfinite(min_area_ha) and min_area_ha >= 0.0):
        raise AppInputError(f"min_area_ha ({min_area_ha}) must be zero or more.")
    if not (math.isfinite(recovery_sim) and 0.0 < recovery_sim <= 1.0):
        raise AppInputError(f"recovery_sim ({recovery_sim}) must lie in (0, 1].")
    if reference is not None and not reference.is_file():
        raise AppInputError(
            f"Reference file {reference} does not exist; pass a vector file or a GeoTIFF with --reference."
        )


def run_wildfire(
    aoi: str | Path | LoadedAOI,
    year_a: int,
    year_b: int,
    out_dir: Path | str,
    *,
    threshold: float | None = None,
    min_area_ha: float = 1.0,
    fire_year: int | None = None,
    fire_name: str | None = None,
    validate: bool = False,
    reference: Path | str | None = None,
    recovery_through: int = LAST_YEAR,
    recovery_sim: float = DEFAULT_RECOVERY_SIM,
    force: bool = False,
    max_area_km2: float | None = io.MAX_AOI_KM2,
    max_zones: int = 4,
    snapshot_id: str | None = None,
    reader: Reader | None = None,
    block: tuple[int, int] = io.DEFAULT_BLOCK,
    strict_validate: bool = False,
    severity: Path | str | None = None,
    severity_min_class: int = DEFAULT_SEVERITY_MIN_CLASS,
) -> WildfireResult:
    """Map the burn scar between ``year_a`` and ``year_b`` and track its recovery through ``recovery_through``.

    Writes to ``out_dir``: ``wildfire_<a>_<b>.tif`` (float32 COG: ``distance_<a>_<b>``, ``burned``
    (1, 0, NaN where not classified) and ``recovery_index_<t>`` per recovery year),
    ``wildfire_pca_<a>.tif`` and ``wildfire_pca_<b>.tif`` (uint8 false colour, one shared basis, 0 is
    no data), ``wildfire_summary.parquet`` and ``.csv`` (one row per recovery year),
    ``wildfire_preview.png``, ``wildfire_validation.json`` when validating or given a severity map,
    and ``wildfire_report.json`` last. A rerun with the same arguments skips (``skipped`` is True)
    unless ``force``; a validating rerun whose outputs are current except ``validation.json`` redoes
    only the validation, from the outputs on disk (``validation_redone``).

    With ``severity`` (an MTBS dnbr6 GeoTIFF, or the Sentinel-2 proxy of
    ``scripts/make_caldor_dnbr.py``), ``validation.json`` gets a ``severity`` block
    (:func:`score_against_severity`) next to the outline scores, whether or not ``validate`` asks for
    the outline too, and the stop rule reads the IoU against moderate or worse, whatever
    ``severity_min_class`` is. When the outline reference cannot be loaded, the severity scores are
    still written, and the report does not vouch for the file, so a rerun tries the outline again.
    Whether the outline was asked for is not in the arguments digest (adding ``validate`` must not
    re-read the store), so a ``validate`` rerun of a run whose ``validation.json`` has no outline
    scores (it had only ``severity``) redoes the validation from the outputs on disk too
    (:func:`_withdraw_validation`); when the outline still cannot be loaded, the report is put
    back as it was.

    Args:
        aoi: AOI file (any CRS), demo name (``caldor``) or a loaded AOI. Make it extend past the
            scar: nothing outside it is classified.
        year_a: The year before the fire.
        year_b: The year after the fire (survey N11).
        out_dir: Output directory.
        threshold: Fixed distance threshold; Otsu (with the stable-pair fallback) when None.
        min_area_ha: Burned patches smaller than this are dropped after a 3 x 3 opening.
        fire_year: The fire's year, for NIFC and the year check; :func:`default_fire_year` when None.
        fire_name: Part of the NIFC incident name; every perimeter of the fire year when None.
        validate: Score the burned mask against ``reference`` or NIFC.
        reference: A vector file or GeoTIFF of burned area to validate against instead of NIFC.
        recovery_through: Last recovery year.
        recovery_sim: Similarity level of the brief's ``sim90_fraction``.
        force: Run even when the outputs are current.
        max_area_km2: AOI size limit; None lifts it.
        max_zones: Zone group limit.
        snapshot_id: Icechunk snapshot to pin.
        reader: Reader to use instead of opening the live store.
        block: Block shape of the streaming reads.
        strict_validate: With ``validate``, stop before any chunk read when the reference cannot be
            loaded, instead of running without validation.
        severity: A burn severity GeoTIFF in MTBS dnbr6 codes (any CRS) to score against.
        severity_min_class: The lowest severity class counted as burned (1 to 4; 3, moderate, by
            default).

    Returns:
        The report, output paths and whether the run was skipped.

    Raises:
        AppInputError: A bad argument, AOI file, reference or severity file, or output directory,
            before any chunk read.
        AOIError: The AOI is too large, touches too many zones, or covers no pixel.
        YearsNotPublishedError: A year the run needs is missing for some zone (before any chunk read).
        StrictValidationError: ``strict_validate`` is set and the reference cannot be loaded (before
            any chunk read).
        WildfireError: No comparable pixel to threshold.
    """
    fire = int(default_fire_year(year_a, year_b) if fire_year is None else fire_year)
    ref_path = None if reference is None else Path(reference)
    sev_path = None if severity is None else Path(severity)
    _check_args(
        year_a=year_a,
        year_b=year_b,
        fire_year=fire,
        recovery_through=recovery_through,
        threshold=threshold,
        min_area_ha=min_area_ha,
        recovery_sim=recovery_sim,
        reference=ref_path,
        severity=sev_path,
        severity_min_class=severity_min_class,
    )
    loaded = load_aoi(aoi)
    recovery_years = list(range(year_b, recovery_through + 1))
    fallback_year = year_a - 1 if threshold is None and year_a - 1 >= FIRST_YEAR else None
    years_read = sorted({year_a, *recovery_years, *([fallback_year] if fallback_year is not None else [])})
    out = Path(out_dir)
    paths = export.output_paths(out, APP, year_a, year_b)
    extras = {"pca_a": out / f"{APP}_pca_{year_a}.tif", "pca_b": out / f"{APP}_pca_{year_b}.tif"}
    params = {
        "year_a": year_a,
        "year_b": year_b,
        "fire_year": fire,
        "fire_name": fire_name,
        "threshold": threshold,
        "min_area_ha": min_area_ha,
        "recovery_through": recovery_through,
        "recovery_sim": recovery_sim,
        "reference": None if ref_path is None else {"file": ref_path.name, "sha256": export.file_sha256(ref_path)},
        "severity": None
        if sev_path is None
        else {"file": sev_path.name, "sha256": export.file_sha256(sev_path), "min_class": int(severity_min_class)},
        "block": list(block),
        "method": {
            "max_valley_ratio": MAX_VALLEY_RATIO,
            "stable_percentile": STABLE_PERCENTILE,
            "fallback_flag_percentile": FALLBACK_FLAG_PERCENTILE,
            "unburned_percentile": UNBURNED_PERCENTILE,
            "reference_distance_share": REFERENCE_DISTANCE_SHARE,
            "reference_buffer_px": REFERENCE_BUFFER_PX,
            "prefire_sample_blocks": PREFIRE_SAMPLE_BLOCKS,
            "reference_drift_warn": REFERENCE_DRIFT_WARN,
            "min_denominator": MIN_DENOMINATOR,
            "recovery_clip": list(RECOVERY_CLIP),
            "pca_sample_cap": PCA_SAMPLE_CAP,
            "pca_sample_per_block": PCA_SAMPLE_PER_BLOCK,
            "pca_fit_blocks": PCA_FIT_BLOCKS,
        },
    }
    state = _Reference()
    years_per_block = 1 + len(recovery_years) + (1 if fallback_year is not None else 0)

    def prepare() -> RunContext | None:
        try:
            return prepare_run(
                APP,
                loaded,
                years_read,
                year_a,
                year_b,
                out,
                params,
                validate=validate or sev_path is not None,
                force=force,
                max_area_km2=max_area_km2,
                max_zones=max_zones,
                snapshot_id=snapshot_id,
                reader=reader,
                seed=SEED,
                block=block,
                extra_outputs=extras,
                # The main pass reads year_a and every recovery year; the PCA pass may read year_a once
                # more; the fallback reads year_a - 1 and year_a (and covers the pre-fire sample).
                year_reads=1 + len(recovery_years) + 1 + (2 if fallback_year is not None else 0),
                code_modules=(sys.modules[__name__],),
                memory_estimate=lambda plan: memory_estimate(
                    plan, n_recovery_years=len(recovery_years), years_per_block=years_per_block, block=block
                ),
                allow_validation_only=True,
                preflight=lambda c: _load_reference(
                    c,
                    state,
                    reference=ref_path,
                    fire_year=fire,
                    fire_name=fire_name,
                    strict=strict_validate and validate,
                    outline=validate,
                    severity=sev_path,
                    severity_min_class=severity_min_class,
                ),
            )
        except io.YearsNotPublishedError as exc:
            if fallback_year is not None and any(fallback_year in years for years in exc.missing.values()):
                exc.add_note(
                    f"Year {fallback_year} (year_a - 1) is read only for the stable-pair threshold fallback and the "
                    "recovery reference check; pass --threshold to run without it."
                )
            raise

    ctx = prepare()
    # The outputs are current, but the validation.json the report vouches for may hold no outline
    # scores (the earlier run had only --severity): stop vouching for it and prepare again, which
    # then takes the validation-only path. The report is put back when nothing is redone.
    vouched = _withdraw_validation(paths.report) if ctx is None and validate else None
    if vouched is not None:
        try:
            ctx = prepare()
        except BaseException:
            _restore_report(paths.report, vouched)
            raise
        if ctx is None:
            _restore_report(paths.report, vouched)
    if ctx is None:
        report = json.loads(paths.report.read_text(encoding="utf-8"))
        return WildfireResult(
            report=report,
            paths=paths,
            extra_paths=extras,
            skipped=True,
            validation_written="validation" in (report.get("outputs") or {}),
        )
    if ctx.validation_only:
        try:
            result = _redo_validation(ctx, state)
        except BaseException:
            if vouched is not None:
                _restore_report(paths.report, vouched)
            raise
        if vouched is not None and not result.validation_redone:
            # The outline still cannot be loaded: the earlier run's report stands, and the outline
            # this run asked for was not written.
            _restore_report(paths.report, vouched)
            result = dataclasses.replace(result, report=json.loads(vouched), validation_written=False)
        return result
    return _run(
        ctx,
        recovery_years=recovery_years,
        fire_year=fire,
        threshold=threshold,
        fallback_year=fallback_year,
        min_area_ha=min_area_ha,
        recovery_sim=recovery_sim,
        reference=state,
    )


def _withdraw_validation(report_path: Path) -> str | None:
    """Make a current report stop vouching for a ``validation.json`` that has no outline scores.

    A run with ``--severity`` and no ``--validate`` writes ``validation.json`` with the severity
    block only, and its report vouches for it. Whether the outline was asked for is not in the
    arguments digest, so :func:`export.outputs_current` takes those outputs as current for a later
    ``--validate`` run with the same other arguments, which would then skip without scoring the
    outline. Dropping ``validation`` from the report's ``outputs`` (atomically) sends that run down
    :func:`prepare_run`'s validation-only path instead: the validation is redone from the outputs
    on disk, without reading a chunk.

    Args:
        report_path: The report of outputs :func:`prepare_run` found current.

    Returns:
        The report's previous text when it was rewritten (to put back with :func:`_restore_report`
        when nothing is redone), None when its validation already has outline scores (``iou``) or
        it vouches for no validation.
    """
    text = report_path.read_text(encoding="utf-8")
    report = json.loads(text)
    outputs = dict(report.get("outputs") or {})
    validation = (report.get("summary") or {}).get("validation") or {}
    if "validation" not in outputs or "iou" in validation:
        return None
    outputs.pop("validation")
    report["outputs"] = outputs
    export.write_json(report_path, report)
    logger.info(
        "%s: %s vouches for a validation without outline scores, which --validate asks for; the validation is "
        "redone from the outputs on disk.",
        APP,
        report_path,
    )
    return text


def _restore_report(report_path: Path, text: str) -> None:
    """Put back a report :func:`_withdraw_validation` rewrote, atomically."""
    with export.atomic_path(report_path) as tmp:
        tmp.write_text(text, encoding="utf-8")


def _redo_validation(ctx: RunContext, state: _Reference) -> WildfireResult:
    """Redo only the validation of a run whose other outputs are current, without reading a chunk.

    The burned and classified masks and the distances come back from the run's GeoTIFF
    (:func:`zone_values_from_raster`). When the outline reference still cannot be loaded, nothing is
    written and the run counts as skipped (a severity block written by the earlier run stays). The
    earlier run's validation warnings (:data:`VALIDATION_SKIPPED`, the stop rule's) are replaced by
    this validation's; its other warnings stay.
    """
    previous = json.loads(ctx.paths.report.read_text(encoding="utf-8"))
    if state.outline and state.masks is None:
        logger.warning(
            "%s: outputs in %s are current; the validation reference still cannot be loaded (%s), so nothing "
            "was redone.",
            APP,
            ctx.paths.report.parent,
            state.error,
        )
        return WildfireResult(
            report=previous,
            paths=ctx.paths,
            extra_paths=dict(ctx.extra_outputs),
            skipped=True,
            validation_written=False,
        )
    plan = ctx.plan
    summary = dict(previous.get("summary") or {})
    threshold = float(summary["threshold"]["threshold"])
    dist = zone_values_from_raster(ctx.paths.tif, plan, band=1)
    flags = zone_values_from_raster(ctx.paths.tif, plan, band=2)
    classified = {z: np.isfinite(v) for z, v in flags.items()}
    burned = {z: v == 1 for z, v in flags.items()}
    reference_mask = recovery_reference(plan, dist, threshold, burned)
    validation, written = _validate(
        ctx, state, burned=burned, classified=classified, reference_mask=reference_mask, extra_caveats=[REDONE_CAVEAT]
    )
    summary["validation"] = validation
    summary["headline"] = headline(validation)
    # The validation's own warnings are rebuilt by _validate; every other warning of the run stays.
    old = [
        w
        for w in previous.get("warnings") or []
        if not (str(w).startswith(VALIDATION_SKIPPED) or str(w).endswith(STOP_WARNING))
    ]
    report = finish_validation_only(
        ctx, previous, summary=summary, warnings=[*old, *ctx.warnings], validation_written=written
    )
    return WildfireResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=written,
        validation_redone=True,
    )


def _write_pca(ctx: RunContext, zones: Mapping[str, _Zone], dst: str) -> dict[str, np.ndarray]:
    """Write both PCA COGs and free the per-zone colours; returns the strided preview images."""
    images = {}
    for name, attr in (("pca_a", "rgb_a"), ("pca_b", "rgb_b")):
        pieces = [GridArray(getattr(zones[z.zone], attr), z) for z in ctx.plan.zones]
        rgb, rgb_t, rgb_crs = export.mosaic_zones(pieces, dst_zone=dst, nodata=0)
        del pieces
        for st in zones.values():
            setattr(st, attr, None)
        with rasterio.Env(GDAL_CACHEMAX=WRITE_CACHE_BYTES):
            export.write_geotiff(ctx.extra_outputs[name], rgb, rgb_t, rgb_crs, nodata=0, band_names=PCA_BANDS)
        images[name] = _image(rgb, stride_for_preview(rgb.shape[1:]))
        del rgb
    return images


def _run(
    ctx: RunContext,
    *,
    recovery_years: list[int],
    fire_year: int,
    threshold: float | None,
    fallback_year: int | None,
    min_area_ha: float,
    recovery_sim: float,
    reference: _Reference,
) -> WildfireResult:
    year_a, year_b = ctx.year_a, ctx.year_b
    plan = ctx.plan
    if fire_year == year_b:
        ctx.warn(
            f"year_b ({year_b}) is the fire year: its annual embedding only partly sees a late-season fire. "
            f"Use year_b = {fire_year + 1} (survey N11)."
        )
    if fire_year == year_a:
        ctx.warn(f"year_a ({year_a}) is the fire year, so the before image may already show the fire.")

    t = time.perf_counter()
    basis, pca_info = _fit_basis(ctx)
    t = ctx.lap("pca_fit", t)
    zones, pairs, prefire_done = _stream(ctx, basis, recovery_years, fallback_year)
    t = ctx.lap("stream", t)

    pooled = np.concatenate([st.dist[np.isfinite(st.dist)] for st in zones.values()])
    if pooled.size == 0 and threshold is None:
        raise WildfireError(
            f"No AOI pixel is valid in both {year_a} and {year_b} with the same radar regime "
            f"({pairs[year_b][1]:,} of {pairs[year_b][0]:,} valid pixels switch regime). Choose other years."
        )
    histogram = np.histogram(pooled, bins=HISTOGRAM_BINS) if pooled.size else None
    dst = max(plan.zones, key=lambda z: z.n_pixels).zone
    pca_images = _write_pca(ctx, zones, dst)
    t = ctx.lap("write", t)

    stable_fn = None if fallback_year is None else (lambda: _stable_distances(ctx, zones, fallback_year, prefire_done))
    choice = select_threshold(
        pooled,
        override=threshold,
        stable_pair=None if fallback_year is None else (fallback_year, year_a),
        stable_values=stable_fn,
    )
    del pooled
    if choice.note and choice.method == "otsu":
        ctx.warn(f"Threshold: {choice.note}. Pass --threshold to set it by hand.")
    if choice.fallback_suspect:
        otsu_text = "" if choice.otsu is None else f" (Otsu would put it at {choice.otsu.threshold:.3f})"
        ctx.warn(
            f"Threshold: the stable pair {choice.stable_pair} has a 99th percentile distance of "
            f"{choice.threshold:.3f}, at or above the {FALLBACK_FLAG_PERCENTILE:g}th percentile of the {year_a} to "
            f"{year_b} distances, so the fallback flags only {choice.fallback_flagged_share:.1%} of pixels. Either "
            f"nothing changed between {year_a} and {year_b}, or the stable pair is not stable{otsu_text}. Check the "
            "distance histogram in the preview and pass --threshold to set it by hand."
        )
    t = ctx.lap("threshold", t)

    raw = {z: np.isfinite(st.dist) & (st.dist > choice.threshold) for z, st in zones.items()}
    classified = {z: np.isfinite(st.dist) for z, st in zones.items()}
    min_pixels = metrics.area_to_pixels(min_area_ha)
    burned, _, _, _ = mosaic_and_clean(raw, plan, min_pixels=min_pixels, dst_zone=dst)
    t = ctx.lap("clean", t)

    n_classified = sum(int(v.sum()) for v in classified.values())
    n_burned = sum(int(b.sum()) for b in burned.values())
    n_raw = sum(int(r.sum()) for r in raw.values())
    del raw
    burned_area_ha = n_burned * HA_PER_PIXEL
    burned_fraction = _ratio(n_burned, n_classified)
    if n_burned == 0:
        ctx.warn("No burned pixel survived the threshold and the minimum area.")

    dist = {z: st.dist for z, st in zones.items()}
    ref = recovery_reference(plan, dist, choice.threshold, burned)
    n_ref = sum(int(m.sum()) for m in ref.values())
    if n_ref < MIN_REFERENCE_PIXELS:
        ctx.warn(
            f"The recovery reference (land shown unchanged between {year_a} and {year_b}) has only {n_ref} pixels; "
            "the recovery figures are unreliable. Extend the AOI past the scar."
        )
    reference_record = _reference_record(
        zones,
        ref,
        classified,
        choice.threshold,
        prefire_year=fallback_year,
        prefire_complete=len(prefire_done) == len(_all_blocks(ctx)),
        year_a=year_a,
        year_b=year_b,
    )
    drift = reference_record["pre_minus_post_median"]
    if drift is not None and drift > REFERENCE_DRIFT_WARN:
        ctx.warn(
            f"The recovery reference is less similar to itself after the fire (median "
            f"{reference_record['post_fire_similarity']['median']:.3f}, {year_a} to {year_b}) than before it "
            f"({reference_record['pre_fire_similarity']['median']:.3f}, {year_a - 1} to {year_a}): it may still "
            "hold fire-affected land, which inflates recovery. Extend the AOI past the scar."
        )
    # Free what the recovery and the output mosaic do not need before they allocate.
    radar_free_scar = [
        (int((st.radar_free_a & burned[z]).sum()), int((st.radar_free_b & burned[z]).sum())) for z, st in zones.items()
    ]
    for st in zones.values():
        st.pre = None
        st.radar_free_a = st.radar_free_b = np.zeros((0, 0), dtype=bool)
    band_names = [f"distance_{year_a}_{year_b}", "burned", *[f"recovery_index_{y}" for y in recovery_years]]
    mosaic = _BandMosaic(plan, dst, classified, len(band_names))
    rows: list[dict[str, Any]] = []
    for i, y in enumerate(recovery_years):
        s_by_zone = {z: (np.float32(1.0) - st.dist) if y == year_b else st.sims[y] for z, st in zones.items()}
        s_b, s_p, s_u = [], [], []
        for z, st in zones.items():
            s_y = s_by_zone[z]
            fin = np.isfinite(s_y)
            in_burn = burned[z] & fin
            s_b.append(s_y[in_burn])
            s_p.append(np.float32(1.0) - st.dist[in_burn])
            s_u.append(s_y[ref[z] & fin])
        figures = recovery_metrics(
            np.concatenate(s_b), np.concatenate(s_p), np.concatenate(s_u), recovery_sim=recovery_sim
        )
        bands: dict[str, np.ndarray] = {}
        for z, st in zones.items():
            s_y = s_by_zone[z]
            in_burn = burned[z] & np.isfinite(s_y)
            band = np.full(st.plan.shape, np.nan, dtype=np.float32)
            band[in_burn] = recovery_index(
                s_y[in_burn], np.float32(1.0) - st.dist[in_burn], figures["median_sim_unburned"]
            )
            bands[z] = band
        del s_by_zone
        mosaic.put(2 + i, bands)
        del bands
        for st in zones.values():
            st.sims.pop(y, None)
        n_pair, n_mismatch = pairs[y]
        rows.append(
            {
                "year": y,
                **figures,
                "radar_mismatch_fraction": _ratio(n_mismatch, n_pair),
                "burned_area_ha": burned_area_ha,
                "burned_fraction": burned_fraction,
                "threshold": choice.threshold,
                "threshold_method": choice.method,
            }
        )
    table = pd.DataFrame(rows)
    mosaic.put(1, {z: np.where(classified[z], burned[z].astype(np.float32), np.float32(np.nan)) for z in zones})
    mosaic.put(0, dist)
    t = ctx.lap("recovery", t)

    validation: dict[str, Any] | None = None
    validation_written = False
    if ctx.validate:
        validation, validation_written = _validate(
            ctx, reference, burned=burned, classified=classified, reference_mask=ref
        )
        t = ctx.lap("validation", t)

    assert mosaic.data is not None
    with rasterio.Env(GDAL_CACHEMAX=WRITE_CACHE_BYTES):
        export.write_geotiff(ctx.paths.tif, mosaic.data, mosaic.transform, mosaic.crs, band_names=band_names)
    export.write_table(table, ctx.paths.summary_parquet.with_suffix(""))
    step = stride_for_preview(mosaic.data.shape[1:])
    ref_image = None
    if reference.masks is not None:
        ref_mosaic, _, _ = export.mosaic_zones(
            [GridArray(reference.masks[z.zone], z) for z in plan.zones], dst_zone=dst, nodata=export.BOOL_NODATA
        )
        ref_image = ref_mosaic[::step, ::step].copy()
        del ref_mosaic
    _write_preview(
        ctx.paths.preview,
        pca_a=pca_images["pca_a"],
        pca_b=pca_images["pca_b"],
        dist=mosaic.data[0, ::step, ::step],
        burned=mosaic.data[1, ::step, ::step],
        reference=ref_image,
        histogram=histogram,
        choice=choice,
        table=table,
        year_a=year_a,
        year_b=year_b,
    )
    del mosaic
    ctx.lap("write", t)

    rf_a = _ratio(sum(a for a, _ in radar_free_scar), n_burned)
    rf_b = _ratio(sum(b for _, b in radar_free_scar), n_burned)
    radar = {
        "radar_free_fraction_in_scar_year_a": rf_a,
        "radar_free_fraction_in_scar_year_b": rf_b,
        "radar_free_flag": bool(max(np.nan_to_num(rf_a), np.nan_to_num(rf_b)) > RADAR_FREE_FLAG),
        "radar_mismatch_fraction": _ratio(pairs[year_b][1], pairs[year_b][0]),
        "radar_mismatch_pixels": pairs[year_b][1],
        "radar_mismatch_fraction_by_year": {y: _ratio(m, n) for y, (n, m) in pairs.items()},
    }
    if radar["radar_free_flag"]:
        ctx.warn(f"More than {RADAR_FREE_FLAG:.0%} of the scar is radar-free (optical only); expect more noise.")
    if radar["radar_mismatch_pixels"]:
        ctx.warn(
            f"{radar['radar_mismatch_pixels']:,} valid AOI pixels switch radar regime between {year_a} and {year_b} "
            "and were not classified (survey B5)."
        )
    summary = {
        "aoi_area_ha": plan.area_ha,
        "aoi_pixels": plan.n_pixels,
        "zones": plan.zone_names,
        "year_a": year_a,
        "year_b": year_b,
        "fire_year": fire_year,
        "recovery_years": recovery_years,
        "valid_pixels_year_a": sum(st.valid_a for st in zones.values()),
        "classified_pixels": n_classified,
        "burned_pixels": n_burned,
        "burned_pixels_before_cleaning": n_raw,
        "burned_area_ha": burned_area_ha,
        "burned_fraction": burned_fraction,
        "min_area_ha": min_area_ha,
        "min_pixels": min_pixels,
        "threshold": choice.record(),
        "fallback_ran": choice.fallback_ran,
        "radar": radar,
        "recovery": table.to_dict(orient="records"),
        "recovery_reference": reference_record,
        "pca": pca_info,
        "validation": validation,
        "headline": headline(validation),
    }
    report = finish_run(ctx, summary=summary, extra=ctx.extra_outputs, validation_written=validation_written)
    return WildfireResult(
        report=report,
        paths=ctx.paths,
        extra_paths=dict(ctx.extra_outputs),
        skipped=False,
        validation_written=validation_written,
    )


def _reference_record(
    zones: Mapping[str, _Zone],
    ref: Mapping[str, np.ndarray],
    classified: Mapping[str, np.ndarray],
    threshold: float,
    *,
    prefire_year: int | None,
    prefire_complete: bool,
    year_a: int,
    year_b: int,
) -> dict[str, Any]:
    """How the recovery reference was defined, and its similarity before and after the fire.

    A clean reference stays about as similar to itself after the fire (``year_a`` to ``year_b``) as
    before it (``year_a - 1`` to ``year_a``); one that holds fire-affected land drops.
    """
    post = np.concatenate([np.float32(1.0) - st.dist[ref[z]] for z, st in zones.items()])
    candidates = sum(int((classified[z] & ~(st.dist > threshold)).sum()) for z, st in zones.items())
    pre: dict[str, Any] | None = None
    note = None
    if prefire_year is None:
        note = f"{year_a - 1} was not read (--threshold was given, or {year_a - 1} is before {FIRST_YEAR})."
    else:
        pre_values = np.concatenate([np.float32(1.0) - st.pre[ref[z]] for z, st in zones.items() if st.pre is not None])
        pre = {
            "years": [prefire_year, year_a],
            **_similarity_stats(pre_values),
            "coverage": "every block" if prefire_complete else f"a sample of {PREFIRE_SAMPLE_BLOCKS} blocks",
        }
    post_stats = _similarity_stats(post)
    drift = None
    if pre is not None and pre["median"] is not None and post_stats["median"] is not None:
        drift = float(pre["median"]) - float(post_stats["median"])
    return {
        "definition": (
            f"Classified pixels whose {year_a} to {year_b} distance is at most {REFERENCE_DISTANCE_SHARE:g} x the "
            f"threshold and that lie more than {REFERENCE_BUFFER_PX} pixels from any burned pixel."
        ),
        "max_distance": REFERENCE_DISTANCE_SHARE * threshold,
        "buffer_pixels": REFERENCE_BUFFER_PX,
        "pixels": int(sum(int(m.sum()) for m in ref.values())),
        "unburned_candidates": candidates,
        "post_fire_similarity": {"years": [year_a, year_b], **post_stats},
        "pre_fire_similarity": pre,
        "pre_fire_note": note,
        "pre_minus_post_median": drift,
    }


__all__ = [
    "APP",
    "ThresholdChoice",
    "WildfireError",
    "WildfireResult",
    "default_fire_year",
    "headline",
    "load_severity",
    "memory_estimate",
    "recovery_index",
    "recovery_metrics",
    "recovery_reference",
    "run_wildfire",
    "score_against_reference",
    "score_against_severity",
    "select_threshold",
    "severity_burned",
    "stop_rule",
    "valley_ratio",
]
