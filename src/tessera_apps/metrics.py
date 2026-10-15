"""Per-pixel metrics on TESSERA embeddings: cosine similarity, means, trajectories, Otsu, PCA colour.

Everything here is pure numpy and scipy: no network, no file I/O. Inputs are either plain arrays of
shape (..., D) or :class:`tessera_apps.io.Window` objects (one year of one window of a zone group).

Conventions:

- **Cosine on raw int8.** True vectors are ``embeddings * scales`` with one strictly positive scale
  per pixel, so the scale cancels in a cosine. Cosines are computed on the raw int8 values with
  exact integer dot products; nothing is dequantized first (survey S18).
- **Means and PCA on dequantized vectors.** Averages and principal components mix pixels with
  different scales, so they use ``embeddings * scales``.
- **Validity.** A window pixel is valid when its scale is finite and positive and its vector is not
  all zero (the fill value). Every result is NaN (or 0 for colour images) where a pixel is not valid.
- **Masks.** A window's own ``Window.mask`` (the AOI pixels of a planned read that lie in the zone's
  nominal band) always applies in :func:`mean_embedding`, :func:`trajectory`, :func:`fit_pca` and
  :func:`pca_rgb`, and an explicit ``mask`` argument is combined with it by logical and, never used
  in its place. A multi-zone AOI therefore never counts a pixel twice and never pulls in pixels
  outside the AOI, even when the explicit mask comes from a cosine map such as
  ``cosine_distance_map(a, b) > threshold``, which does not apply ``Window.mask``. A plain
  rectangular read has ``mask`` None, so only the explicit mask applies. To select pixels outside
  ``Window.mask`` on purpose, pass ``dataclasses.replace(win, mask=None)``.
- **Memory.** Large inputs are processed in strips of at most ``_STRIP_ELEMENTS`` array elements,
  so a 512 x 2048 x 128 block never needs a full float64 copy.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from scipy import ndimage

from tessera_apps.io import HA_PER_PIXEL, Window

# Largest number of array elements converted or compared at once. 2**22 int8 elements is 4 MB, and
# its float64 copy 32 MB.
_STRIP_ELEMENTS = 2**22

# Values per chunk in otsu_threshold's passes. Each chunk needs a few float64 temporaries, so 2**18
# values keep the working memory near 10 MB whatever the input size.
_OTSU_CHUNK = 2**18

# 8-connectivity for component labelling.
_EIGHT = np.ones((3, 3), dtype=bool)


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _pieces(lead: tuple[int, ...], inner: int, budget: int) -> Iterator[tuple[int | slice, ...]]:
    """Index tuples that cover ``lead`` in pieces of at most ``budget`` elements.

    Each piece is ``lead``-shaped indexing (leading integers, then one slice) over an array whose
    trailing dimensions hold ``inner`` elements per index. A piece always holds at least one index.

    Args:
        lead: Shape of the leading dimensions to cover.
        inner: Elements per leading index (for example the embedding dimension D).
        budget: Largest number of elements per piece.

    Yields:
        Index tuples to apply to the leading dimensions.
    """
    n = len(lead)
    if n == 0:
        yield ()
        return
    k = 0
    tail = math.prod(lead[1:]) * inner
    while tail > budget and k < n - 1:
        k += 1
        tail = math.prod(lead[k + 1 :]) * inner
    step = max(1, budget // max(tail, 1))
    for head in np.ndindex(*lead[:k]):
        for start in range(0, lead[k], step):
            yield (*head, slice(start, min(start + step, lead[k])))


def _is_small_int(dtype: np.dtype) -> bool:
    """True for bool and integer dtypes of at most 2 bytes, whose dot products fit int64 exactly."""
    return (dtype == np.bool_ or np.issubdtype(dtype, np.integer)) and dtype.itemsize <= 2


def _accumulator(itemsize: int, d: int) -> type[np.signedinteger]:
    """Integer type that holds a dot product of ``d`` components of ``itemsize`` bytes without overflow.

    One-byte components are at most 2**8 in magnitude, so int32 holds ``d * 2**16`` for any
    ``d < 2**15``. Wider ones use int64.
    """
    return np.int32 if itemsize == 1 and d < 2**15 else np.int64


def _cos_int(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine of integer vectors with exact integer dot products. Float64 result, NaN for zero norm."""
    acc = _accumulator(max(x.dtype.itemsize, y.dtype.itemsize), x.shape[-1])
    dot = np.einsum("...d,...d->...", x, y, dtype=acc).astype(np.float64)
    nx = np.einsum("...d,...d->...", x, x, dtype=acc).astype(np.float64)
    ny = np.einsum("...d,...d->...", y, y, dtype=acc).astype(np.float64)
    return _finish(dot, nx, ny, None)


def _scaled(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Float64 copy of (..., D) vectors, each divided by its largest magnitude, and a finite flag.

    The division keeps squares from overflowing or underflowing. Non-finite vectors become zero.
    """
    xf = x.astype(np.float64)
    ok = np.isfinite(xf).all(axis=-1)
    xf[~ok] = 0.0
    top = np.abs(xf).max(axis=-1, initial=0.0)
    xf /= np.where(top > 0, top, 1.0)[..., None]
    return xf, ok


def _cos_float(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cosine of float vectors in float64. NaN for zero norm or any non-finite component."""
    xf, okx = _scaled(x)
    yf, oky = _scaled(y)
    dot = np.einsum("...d,...d->...", xf, yf)
    nx = np.einsum("...d,...d->...", xf, xf)
    ny = np.einsum("...d,...d->...", yf, yf)
    return _finish(dot, nx, ny, okx & oky)


def _cos_ref(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Cosine of (..., D) vectors with one finite float64 vector ``ref`` scaled to a largest magnitude of 1."""
    nr = np.asarray(ref @ ref)
    if _is_small_int(x.dtype):
        acc = _accumulator(x.dtype.itemsize, x.shape[-1])
        dot = np.asarray(x @ ref, dtype=np.float64)
        nx = np.einsum("...d,...d->...", x, x, dtype=acc).astype(np.float64)
        return _finish(dot, nx, nr, None)
    xf, ok = _scaled(x)
    return _finish(np.asarray(xf @ ref), np.einsum("...d,...d->...", xf, xf), nr, ok)


def _finish(dot: np.ndarray, nx: np.ndarray, ny: np.ndarray, ok: np.ndarray | None) -> np.ndarray:
    """Divide dot products by the norms, clip to [-1, 1], and set NaN where undefined."""
    bad = (nx <= 0) | (ny <= 0)
    if ok is not None:
        bad |= ~ok
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = dot / (np.sqrt(nx) * np.sqrt(ny))
    cos = np.clip(cos, -1.0, 1.0)
    return np.where(bad, np.nan, cos)


def _embeddings(win: Window) -> np.ndarray:
    """The window's (H, W, D) embeddings, checked against its scales.

    Raises:
        ValueError: The embeddings layer was not read, or its shape does not match the scales.
    """
    if win.embeddings is None:
        raise ValueError(f"Window {win.zone} {win.year} has no embeddings; read the 'embeddings' layer.")
    emb = win.embeddings
    if emb.ndim != 3 or emb.shape[:2] != win.scales.shape:
        raise ValueError(
            f"Window {win.zone} {win.year}: embeddings {emb.shape} do not match scales {win.scales.shape}."
        )
    return emb


def _any_nonzero(arr: np.ndarray) -> np.ndarray:
    """``np.any(arr != 0, axis=-1)``, computed in strips."""
    lead = arr.shape[:-1]
    out = np.empty(lead, dtype=bool)
    for idx in _pieces(lead, arr.shape[-1], _STRIP_ELEMENTS):
        out[idx] = np.any(arr[idx] != 0, axis=-1)
    return out


def _window_valid(win: Window) -> np.ndarray:
    """Pixels with a finite, positive scale and a vector that is not all zero."""
    emb = _embeddings(win)
    scales = win.scales
    return np.isfinite(scales) & (scales > 0) & _any_nonzero(emb)


def _array_valid(arr: np.ndarray) -> np.ndarray:
    """Vectors of a plain (..., D) array that are finite and not all zero."""
    lead = arr.shape[:-1]
    out = _any_nonzero(arr)
    if not (arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer)):
        for idx in _pieces(lead, arr.shape[-1], _STRIP_ELEMENTS):
            out[idx] &= np.isfinite(arr[idx]).all(axis=-1)
    return out


def _as_mask(mask: np.ndarray | None, shape: tuple[int, ...], what: str) -> np.ndarray | None:
    """``mask`` as a bool array of ``shape``, or None.

    Raises:
        ValueError: The mask has a different shape.
    """
    if mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    if m.shape != shape:
        raise ValueError(f"{what}: mask shape {m.shape} does not match {shape}.")
    return m


def _selection(win: Window, mask: np.ndarray | None) -> np.ndarray:
    """Valid pixels of ``win`` inside both ``win.mask`` and ``mask``; a mask that is None is skipped."""
    sel = _window_valid(win)
    for m in (win.mask, mask):
        mm = _as_mask(m, win.shape, f"Window {win.zone} {win.year}")
        if mm is not None:
            sel &= mm
    return sel


def _check_same_window(a: Window, b: Window) -> None:
    """Raise ValueError unless ``a`` and ``b`` cover the same zone, rows and columns."""
    same = (
        a.zone == b.zone
        and (a.rows.start, a.rows.stop) == (b.rows.start, b.rows.stop)
        and (a.cols.start, a.cols.stop) == (b.cols.start, b.cols.stop)
        and a.shape == b.shape
    )
    if not same:
        raise ValueError(
            f"Windows differ: {a.zone} rows {a.rows.start}:{a.rows.stop} cols {a.cols.start}:{a.cols.stop} "
            f"against {b.zone} rows {b.rows.start}:{b.rows.stop} cols {b.cols.start}:{b.cols.stop}."
        )


def _windows_and_masks(
    win: Window | Sequence[Window], mask: np.ndarray | Sequence[np.ndarray] | None
) -> tuple[list[Window], list[np.ndarray | None]]:
    """Normalise one window or several (blocks) with their masks into two equal-length lists.

    Raises:
        ValueError: No windows, or the masks do not pair one to one with the windows.
        TypeError: An item is not a Window.
    """
    if isinstance(win, Window):
        if mask is not None and not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        return [win], [mask]
    wins = list(win)
    if not wins:
        raise ValueError("No windows given.")
    for w in wins:
        if not isinstance(w, Window):
            raise TypeError(f"Expected Window objects, got {type(w).__name__}.")
    if mask is None:
        return wins, [None] * len(wins)
    if isinstance(mask, np.ndarray):
        raise ValueError("Several windows need a sequence of masks, one per window.")
    masks: list[np.ndarray | None] = list(mask)
    if len(masks) != len(wins):
        raise ValueError(f"Got {len(masks)} masks for {len(wins)} windows.")
    return wins, masks


# --------------------------------------------------------------------------------------------
# Cosine similarity
# --------------------------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of vectors along the last axis.

    Integer inputs of at most 2 bytes (int8 embeddings) use exact integer dot products, so int8
    extremes such as -128 never overflow. Anything else is computed in float64. Per-pixel scales are
    positive, so pass raw int8 embeddings: dequantizing first gives the same answer at more cost.

    Args:
        a: Array of shape (..., D), int8 or float.
        b: Array of shape (..., D) broadcastable against ``a``, for example a single (D,) vector.

    Returns:
        Float32 array of the broadcast leading shape, in [-1, 1]. NaN where either vector has zero
        norm or any non-finite component.

    Raises:
        ValueError: An input is a scalar, the last dimensions differ or are empty, or the leading
            shapes do not broadcast.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 0 or b.ndim == 0:
        raise ValueError("cosine_similarity needs arrays of shape (..., D), not scalars.")
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"Vector lengths differ: {a.shape[-1]} and {b.shape[-1]}.")
    if a.shape[-1] == 0:
        raise ValueError("Vectors have length 0.")
    try:
        shape = np.broadcast_shapes(a.shape, b.shape)
    except ValueError as exc:
        raise ValueError(f"Shapes {a.shape} and {b.shape} do not broadcast.") from exc
    lead, d = shape[:-1], shape[-1]
    out = np.empty(lead, dtype=np.float32)
    if a.ndim == 1 or b.ndim == 1:
        # One reference vector: a matrix-vector product per strip, without broadcasting it out.
        vec, other = (b, a) if b.ndim == 1 else (a, b)
        r = vec.astype(np.float64)
        if not np.all(np.isfinite(r)) or not np.any(r != 0):
            out.fill(np.nan)
            return out
        r /= np.abs(r).max()
        for idx in _pieces(lead, d, _STRIP_ELEMENTS):
            out[idx] = _cos_ref(other[idx], r)
        return out
    kernel = _cos_int if _is_small_int(a.dtype) and _is_small_int(b.dtype) else _cos_float
    ab = np.broadcast_to(a, shape)
    bb = np.broadcast_to(b, shape)
    for idx in _pieces(lead, d, _STRIP_ELEMENTS):
        out[idx] = kernel(ab[idx], bb[idx])
    return out


def cosine_similarity_map(a: Window, b: Window) -> np.ndarray:
    """Per-pixel cosine similarity between two windows of the same zone, rows and columns.

    Uses the raw int8 embeddings (scales cancel). The windows' own ``mask`` is not applied here, so
    a mask derived from the result covers the whole window; the functions that take a ``mask``
    combine it with ``Window.mask`` again (see the module notes).

    Args:
        a: First window, usually the earlier year.
        b: Second window over the same pixels.

    Returns:
        Float32 (H, W), NaN where either window's pixel is not valid.

    Raises:
        ValueError: The windows cover different pixels or lack embeddings.
    """
    _check_same_window(a, b)
    sim = cosine_similarity(_embeddings(a), _embeddings(b))
    sim[~(_window_valid(a) & _window_valid(b))] = np.nan
    return sim


def cosine_distance_map(a: Window, b: Window) -> np.ndarray:
    """Per-pixel cosine distance, ``1 - cosine_similarity_map(a, b)``, in [0, 2].

    Args:
        a: First window, usually the earlier year.
        b: Second window over the same pixels.

    Returns:
        Float32 (H, W), NaN where either window's pixel is not valid.

    Raises:
        ValueError: The windows cover different pixels or lack embeddings.
    """
    return np.float32(1.0) - cosine_similarity_map(a, b)


def cosine_to_vector(emb: np.ndarray | Window, ref: np.ndarray) -> np.ndarray:
    """Cosine similarity of every pixel to one reference vector, such as a dequantized mean.

    Args:
        emb: A (H, W, D) array (any (..., D) works) or a Window, whose raw int8 embeddings are used.
        ref: Reference vector of shape (D,).

    Returns:
        Float32 array of the leading shape. NaN where a pixel vector is zero or non-finite, and for
        a Window, where the pixel is not valid.

    Raises:
        ValueError: ``ref`` is not a finite, non-zero (D,) vector matching the embeddings.
    """
    r = np.asarray(ref, dtype=np.float64)
    if r.ndim != 1:
        raise ValueError(f"Reference must be a (D,) vector, got shape {r.shape}.")
    if not np.all(np.isfinite(r)) or not np.any(r != 0):
        raise ValueError("Reference vector must be finite and not all zero.")
    if isinstance(emb, Window):
        out = cosine_similarity(_embeddings(emb), r)
        out[~_window_valid(emb)] = np.nan
        return out
    return cosine_similarity(np.asarray(emb), r)


# --------------------------------------------------------------------------------------------
# Means, trajectories and radar regime
# --------------------------------------------------------------------------------------------


def mean_embedding(
    win: Window | Sequence[Window], mask: np.ndarray | Sequence[np.ndarray] | None = None
) -> tuple[np.ndarray, int]:
    """Mean dequantized vector (``embeddings * scales``) over valid pixels inside a mask.

    Several windows (for example the blocks of a streamed AOI) are pooled by summing, so the result
    is the mean over every selected pixel of every window, not a mean of block means.

    Args:
        win: One window or a sequence of windows.
        mask: For one window, a bool (H, W) mask. For several, a sequence with one mask per window
            (None entries allowed). It is combined (logical and) with each window's own ``mask``
            when the window has one; when None, the window's own ``mask`` alone applies, or every
            valid pixel for a plain rectangular read.

    Returns:
        The float64 (D,) mean and the number of pixels averaged.

    Raises:
        ValueError: No pixel is selected, masks do not match the windows, embedding lengths differ,
            or a window lacks embeddings.
    """
    wins, masks = _windows_and_masks(win, mask)
    total: np.ndarray | None = None
    count = 0
    for w, m in zip(wins, masks, strict=True):
        emb = _embeddings(w)
        if total is None:
            total = np.zeros(emb.shape[-1], dtype=np.float64)
        elif emb.shape[-1] != total.size:
            raise ValueError(f"Embedding lengths differ: {emb.shape[-1]} and {total.size}.")
        sel = _selection(w, m)
        d = emb.shape[-1]
        for idx in _pieces(w.shape, d, _STRIP_ELEMENTS):
            s = sel[idx]
            k = int(np.count_nonzero(s))
            if k == 0:
                continue
            # Unselected pixels get weight 0, so one matrix-vector product sums the strip.
            weights = np.where(s, w.scales[idx], 0.0).astype(np.float64)
            total += weights.reshape(-1) @ emb[idx].reshape(-1, d)
            count += k
    if total is None or count == 0:
        raise ValueError("No valid pixels inside the mask; cannot take a mean.")
    return total / count, count


def trajectory(
    windows: Mapping[int, Window], ref_year: int, *, mask: np.ndarray | None = None
) -> dict[int, np.ndarray]:
    """Per-pixel cosine similarity of each year's vector to the same pixel's ``ref_year`` vector.

    Check :func:`same_radar_regime` for each pair and fold it into ``mask`` when the radar regime
    may change between years (survey B5).

    Args:
        windows: Windows of the same zone, rows and columns, keyed by year.
        ref_year: The year every other year is compared with; must be a key of ``windows``.
        mask: Bool (H, W) pixels to keep. It is combined (logical and) with every window's own
            ``mask``; when None, the windows' own masks alone apply, if any has one.

    Returns:
        Float32 (H, W) per year, in ascending year order. The reference year is exactly 1.0 where
        valid. NaN where either year's pixel is invalid or the pixel is outside the mask.

    Raises:
        ValueError: ``ref_year`` is missing, windows cover different pixels, or the mask shape differs.
    """
    if ref_year not in windows:
        raise ValueError(f"Reference year {ref_year} is not among the windows {sorted(windows)}.")
    ref = windows[ref_year]
    for w in windows.values():
        _check_same_window(ref, w)
    keep = _as_mask(mask, ref.shape, "trajectory")
    for w in windows.values():
        wm = _as_mask(w.mask, ref.shape, f"Window {w.zone} {w.year}")
        if wm is not None:
            keep = wm if keep is None else keep & wm
    base = _window_valid(ref)
    if keep is not None:
        base &= keep
    ref_emb = _embeddings(ref)
    out: dict[int, np.ndarray] = {}
    for year in sorted(windows):
        if year == ref_year:
            out[year] = np.where(base, np.float32(1.0), np.float32(np.nan)).astype(np.float32)
            continue
        w = windows[year]
        sim = cosine_similarity(_embeddings(w), ref_emb)
        sim[~(base & _window_valid(w))] = np.nan
        out[year] = sim
    return out


def same_radar_regime(a: Window, b: Window) -> np.ndarray:
    """True where both pixels have Sentinel-1 data or both lack it (survey B5).

    A pixel that switches between radar and optical-only embedding shifts by 0.13 to 0.25 of cosine
    distance with no change on the ground, so compared years must share a regime. Validity is not
    checked here; combine with the windows' validity yourself.

    Args:
        a: First window, read with the ``s1_asc_obs_count`` and ``s1_desc_obs_count`` layers.
        b: Second window over the same pixels, with the same layers.

    Returns:
        Bool (H, W).

    Raises:
        ValueError: The windows cover different pixels.
        TesseraError: A window lacks the radar count layers.
    """
    _check_same_window(a, b)
    return np.asarray(a.radar_present == b.radar_present, dtype=bool)


# --------------------------------------------------------------------------------------------
# Thresholds and masks
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OtsuResult:
    """An Otsu threshold with its separability, so callers can fall back when it is weak (survey N12).

    Attributes:
        threshold: Values strictly above this are the high class.
        eta: Between-class variance over total variance for the split, 0 to 1.
        low_mode_share: Fraction of finite values at or below the threshold.
        high_mode_share: Fraction of finite values above the threshold.
        bimodal: ``eta >= min_eta`` and the smaller class holds at least ``min_share``.
    """

    threshold: float
    eta: float
    low_mode_share: float
    high_mode_share: float
    bimodal: bool


def _finite_chunks(arr: np.ndarray) -> Iterator[np.ndarray]:
    """The finite values of ``arr`` as float64 1-D chunks of at most ``_OTSU_CHUNK`` values.

    Chunks follow the array's row-major order, never copy the whole input, and are never empty.
    """
    for idx in _pieces(arr.shape, 1, _OTSU_CHUNK):
        c = np.asarray(arr[idx], dtype=np.float64).reshape(-1)
        fin = np.isfinite(c)
        if not fin.all():
            c = c[fin]
        if c.size:
            yield c


def otsu_threshold(
    values: np.ndarray, *, bins: int = 256, min_eta: float = 0.5, min_share: float = 0.005
) -> OtsuResult:
    """Otsu's threshold over the finite values, with a separability check.

    The split maximises between-class variance over the ``bins`` histogram edges, using the exact
    sum of the values in each bin. The returned threshold is the midpoint of the gap between the
    largest low-class value and the smallest high-class value, so ``values > threshold`` reproduces
    the split exactly. ``eta`` is computed from the values themselves.

    The input is streamed in chunks over three passes (range, histogram, split), so the working
    memory stays at a few MB whatever the input size, and a float32 input is never copied whole.

    ``eta`` measures how well the split separates the values, not whether there are two modes: a
    single normal distribution gives ``2 / pi`` (about 0.64) and a uniform one 0.75. Treat
    ``bimodal`` as a necessary check, not a sufficient one.

    Args:
        values: Any array; NaN and infinite values are ignored.
        bins: Histogram bins between the smallest and largest finite value.
        min_eta: Smallest ``eta`` for ``bimodal``.
        min_share: Smallest share of the smaller class for ``bimodal``.

    Returns:
        The threshold, ``eta``, class shares and the ``bimodal`` flag.

    Raises:
        ValueError: Fewer than two distinct finite values, or ``bins`` below 2.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}.")
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    # Pass 1: count, smallest and largest finite value.
    n = 0
    lo_v, hi_v = math.inf, -math.inf
    for c in _finite_chunks(arr):
        n += c.size
        lo_v = min(lo_v, float(c.min()))
        hi_v = max(hi_v, float(c.max()))
    if n < 2 or lo_v == hi_v:
        raise ValueError("Otsu needs at least two distinct finite values.")
    # Work on values rescaled to exactly [0, 1]: Otsu is unchanged by an affine map, and the rescale
    # keeps histogram edges distinct and squares finite whatever the range of the input. The rescale
    # is monotone, so a split of the rescaled values is a split of the values.
    span = hi_v - lo_v
    big = not math.isfinite(span)

    def rescale(c: np.ndarray) -> np.ndarray:
        if big:
            return (c * 0.5 - lo_v * 0.5) / (hi_v * 0.5 - lo_v * 0.5)
        return (c - lo_v) / span

    # Pass 2: histogram counts and the exact sum of the values in each bin. np.histogram with a
    # fixed range gives every chunk the same edges, so the chunk histograms add up.
    counts = np.zeros(bins, dtype=np.int64)
    sums = np.zeros(bins, dtype=np.float64)
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for c in _finite_chunks(arr):
        u = rescale(c)
        cnt, edges = np.histogram(u, bins=bins, range=(0.0, 1.0))
        sm, _ = np.histogram(u, bins=bins, range=(0.0, 1.0), weights=u)
        counts += cnt
        sums += sm
        total += float(u.sum())
    # Split after bin k, for k = 0 .. bins - 2. Both classes are non-empty for every k, since the
    # minimum (exactly 0) lies in the first bin and the maximum (exactly 1) in the last.
    c0 = np.cumsum(counts)[:-1].astype(np.float64)
    s0 = np.cumsum(sums)[:-1]
    w0 = c0 / n
    w1 = 1.0 - w0
    mu_t = total / n
    between = (mu_t * w0 - s0 / n) ** 2 / (w0 * w1)
    k = int(np.argmax(between))
    # Pass 3: the low class is u < edges[k + 1], the same half-open bins np.histogram uses. Find
    # the largest low value, the smallest high value, the class size and the total variance.
    cut = float(edges[k + 1])
    a, b = -math.inf, math.inf
    n_low = 0
    sq = 0.0
    for c in _finite_chunks(arr):
        u = rescale(c)
        is_low = u < cut
        k_low = int(np.count_nonzero(is_low))
        n_low += k_low
        if k_low:
            a = max(a, float(c[is_low].max()))
        if k_low < c.size:
            b = min(b, float(c[~is_low].min()))
        u -= mu_t
        sq += float(np.dot(u, u))
    threshold = a + (b - a) / 2
    if not threshold < b:
        threshold = a
    share_low = n_low / n
    share_high = (n - n_low) / n
    # Class means from the bin sums: every bin up to k is low, every later bin is high.
    gap = float(sums[k + 1 :].sum()) / (n - n_low) - float(sums[: k + 1].sum()) / n_low
    var = sq / n
    eta = float(np.clip(share_low * share_high * gap**2 / var, 0.0, 1.0))
    bimodal = eta >= min_eta and min(share_low, share_high) >= min_share
    return OtsuResult(
        threshold=float(threshold),
        eta=eta,
        low_mode_share=float(share_low),
        high_mode_share=float(share_high),
        bimodal=bool(bimodal),
    )


def area_to_pixels(area_ha: float) -> int:
    """Smallest whole number of 10 m pixels whose area reaches ``area_ha`` hectares.

    Rounds away float noise first, so 0.07 ha is 7 pixels, not 8.

    Args:
        area_ha: Area in hectares, zero or more.

    Returns:
        Pixel count, for ``clean_mask(min_pixels=...)``.

    Raises:
        ValueError: The area is negative or not finite.
    """
    if not math.isfinite(area_ha) or area_ha < 0:
        raise ValueError(f"Area must be a finite number of hectares, zero or more, got {area_ha}.")
    return math.ceil(round(area_ha / HA_PER_PIXEL, 9))


def clean_mask(mask: np.ndarray, *, min_pixels: int, opening: int = 3) -> np.ndarray:
    """Remove speckle from a binary mask: a square opening, then drop small 8-connected components.

    Run it on one grid that holds the whole AOI, never piece by piece. A component cut by a block
    edge, or by a zone seam (the nominal-band edge between the ZonePlans of a multi-zone AOI, such
    as 120 W for Caldor's 10N and 11N), is split, and each part is tested against ``min_pixels`` on
    its own, so patches that straddle the cut shrink or vanish. Clean a single-zone AOI on the whole
    zone's mask. Clean a multi-zone AOI on its single-grid mosaic from ``export.mosaic_zones`` with
    nearest resampling (which moves pixel counts along the seam slightly), or merge component sizes
    across the seam before filtering.

    Args:
        mask: 2-D mask; non-zero is True.
        min_pixels: Components with fewer pixels are dropped. 0 or 1 keeps every component.
        opening: Side of the square structuring element. 1 (or 0) skips the opening.

    Returns:
        A new bool mask of the same shape.

    Raises:
        ValueError: The mask is not 2-D, or ``min_pixels`` or ``opening`` is negative.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"clean_mask needs a 2-D mask, got shape {m.shape}.")
    if min_pixels < 0 or opening < 0:
        raise ValueError(f"min_pixels and opening must not be negative, got {min_pixels} and {opening}.")
    if opening > 1:
        m = ndimage.binary_opening(m, structure=np.ones((opening, opening), dtype=bool))
    else:
        m = m.copy()
    if min_pixels <= 1:
        return m
    labels, n = ndimage.label(m, structure=_EIGHT)
    if n == 0:
        return m
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    return np.asarray(keep[labels], dtype=bool)


def component_sizes(mask: np.ndarray) -> np.ndarray:
    """Pixel counts of the 8-connected components of a 2-D mask, for patch-size distributions.

    Args:
        mask: 2-D mask; non-zero is True.

    Returns:
        Int64 array with one count per component, in label (raster scan) order; empty if none.

    Raises:
        ValueError: The mask is not 2-D.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"component_sizes needs a 2-D mask, got shape {m.shape}.")
    labels, n = ndimage.label(m, structure=_EIGHT)
    return np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)


# --------------------------------------------------------------------------------------------
# PCA false colour
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, eq=False)
class PCABasis:
    """Three principal components of dequantized embeddings and their display stretch.

    Arrays are read-only. Reuse one basis for several years so their colours are comparable.

    Attributes:
        mean: (D,) mean vector the projections are centred on.
        components: (3, D) unit components, each signed so its largest-magnitude loading is positive.
        lo: (3,) 2nd percentile of the fitted pixels' projections, shown as 0.
        hi: (3,) 98th percentile, shown as 255.
    """

    mean: np.ndarray
    components: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def _pca_sources(
    emb: np.ndarray | Window | Sequence[Window], mask: np.ndarray | Sequence[np.ndarray] | None
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """(vectors, scales or None, selection) for each input, with selections already masked."""
    if isinstance(emb, np.ndarray):
        if emb.ndim < 2:
            raise ValueError(f"fit_pca needs an array of shape (N, D) or (H, W, D), got {emb.shape}.")
        sel = _array_valid(emb)
        m = _as_mask(None if mask is None else np.asarray(mask), sel.shape, "fit_pca")
        if m is not None:
            sel &= m
        return [(emb, None, sel)]
    wins, masks = _windows_and_masks(emb, mask)
    return [(_embeddings(w), w.scales, _selection(w, m)) for w, m in zip(wins, masks, strict=True)]


def fit_pca(
    emb: np.ndarray | Window | Sequence[Window],
    mask: np.ndarray | Sequence[np.ndarray] | None = None,
    *,
    sample: int = 200_000,
    seed: int = 0,
) -> PCABasis:
    """Fit three principal components on dequantized valid pixels.

    At most ``sample`` pixels are used, drawn uniformly without replacement with a seeded generator,
    so the same inputs and seed always give the same basis. Components come from a numpy SVD of the
    sample covariance, and each is signed so its largest-magnitude loading is positive, so colours
    do not flip between runs.

    Args:
        emb: A Window, a sequence of windows (blocks or years), or a plain (..., D) array. Windows are
            dequantized (``embeddings * scales``); an array is used as given, so dequantize it first.
        mask: Pixels to fit on, shaped like the leading dimensions; one per window for a sequence.
            For windows it is combined (logical and) with each window's own ``mask`` when set; when
            None, the windows' own masks alone apply.
        sample: Largest number of pixels to fit on.
        seed: Seed for the pixel sample.

    Returns:
        The fitted basis.

    Raises:
        ValueError: No valid pixel, fewer than 3 dimensions, ``sample`` below 1, or mismatched masks.
    """
    if sample < 1:
        raise ValueError(f"sample must be at least 1, got {sample}.")
    sources = _pca_sources(emb, mask)
    d = int(sources[0][0].shape[-1])
    if d < 3:
        raise ValueError(f"PCA to three components needs at least 3 dimensions, got {d}.")
    if any(int(src.shape[-1]) != d for src, _, _ in sources):
        raise ValueError("Embedding lengths differ between windows.")
    counts = [int(sel.sum()) for _, _, sel in sources]
    total = sum(counts)
    if total == 0:
        raise ValueError("No valid pixels to fit PCA on.")
    if total > sample:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(total, size=sample, replace=False))
    else:
        pick = np.arange(total)
    # The sample is held as float32 (100 MB at the default size); sums and products are float64.
    rows = max(1, _STRIP_ELEMENTS // d)
    xs = np.empty((pick.size, d), dtype=np.float32)
    filled = 0
    offset = 0
    for (src, scales, sel), n in zip(sources, counts, strict=True):
        flat = np.flatnonzero(sel)[pick[(pick >= offset) & (pick < offset + n)] - offset]
        offset += n
        for i in range(0, flat.size, rows):
            where = np.unravel_index(flat[i : i + rows], sel.shape)
            x = src[where].astype(np.float64)
            if scales is not None:
                x *= scales[where].astype(np.float64)[:, None]
            xs[filled : filled + x.shape[0]] = x
            filled += x.shape[0]
    mean = xs.sum(axis=0, dtype=np.float64) / xs.shape[0]
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(0, xs.shape[0], rows):
        xc = xs[i : i + rows].astype(np.float64) - mean
        cov += xc.T @ xc
    _, _, vt = np.linalg.svd(cov)
    comps = vt[:3].copy()
    for c in comps:
        if c[int(np.argmax(np.abs(c)))] < 0:
            c *= -1.0
    proj = np.empty((xs.shape[0], 3), dtype=np.float64)
    for i in range(0, xs.shape[0], rows):
        proj[i : i + rows] = (xs[i : i + rows].astype(np.float64) - mean) @ comps.T
    lo = np.percentile(proj, 2, axis=0)
    hi = np.percentile(proj, 98, axis=0)
    for arr in (mean, comps, lo, hi):
        arr.setflags(write=False)
    return PCABasis(mean=mean, components=comps, lo=lo, hi=hi)


def pca_rgb(emb: np.ndarray | Window, basis: PCABasis | None = None, *, mask: np.ndarray | None = None) -> np.ndarray:
    """False-colour image: dequantized vectors projected on three components, stretched to 0..255.

    Each channel maps the basis' ``lo`` to 0 and ``hi`` to 255, clipped. A channel whose ``hi`` does
    not exceed ``lo`` (a degenerate fit, such as a single pixel) is shown as 128.

    Args:
        emb: A Window (dequantized) or a plain (..., D) array (used as given).
        basis: Basis to project on. When None, one is fitted on this input with :func:`fit_pca`.
            Pass the same basis for several years to make their colours comparable.
        mask: Bool pixels to colour, shaped like the leading dimensions. For a Window it is combined
            (logical and) with the window's own ``mask`` when set; when None, that mask alone
            applies. An automatic basis is fitted on the same pixels.

    Returns:
        Uint8 array of shape (..., 3). Invalid and out-of-mask pixels are 0.

    Raises:
        ValueError: The embedding length differs from the basis, or the mask shape differs.
    """
    if basis is None:
        basis = fit_pca(emb, mask)
    if isinstance(emb, Window):
        src = _embeddings(emb)
        scales: np.ndarray | None = emb.scales
        sel = _selection(emb, mask)
    else:
        src = np.asarray(emb)
        scales = None
        sel = _array_valid(src)
        m = _as_mask(mask, sel.shape, "pca_rgb")
        if m is not None:
            sel &= m
    d = src.shape[-1]
    if basis.mean.shape != (d,) or basis.components.shape != (3, d):
        raise ValueError(f"Basis is for {basis.mean.shape[0]} dimensions, embeddings have {d}.")
    comps_t = np.asarray(basis.components, dtype=np.float64).T
    offset = np.asarray(basis.mean, dtype=np.float64) @ comps_t
    lo = np.asarray(basis.lo, dtype=np.float64)
    span = np.asarray(basis.hi, dtype=np.float64) - lo
    good = span > 0
    span = np.where(good, span, 1.0)
    lead = src.shape[:-1]
    out = np.zeros((*lead, 3), dtype=np.uint8)
    for idx in _pieces(lead, d, _STRIP_ELEMENTS):
        s = sel[idx]
        if not s.any():
            continue
        x = src[idx][s].astype(np.float64)
        if scales is not None:
            x *= scales[idx][s].astype(np.float64)[:, None]
        p = x @ comps_t - offset
        v = np.where(good, (p - lo) / span * 255.0, 127.5)
        piece = out[idx]
        piece[s] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
    return out
