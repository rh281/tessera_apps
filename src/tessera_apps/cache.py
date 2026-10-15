"""Optional on-disk cache of whole store chunks.

Entries are keyed by (zone, year, chunk row, chunk col) and layer, under a namespace that names
the data version. :class:`~tessera_apps.io.TesseraReader` sets it to the Icechunk snapshot id, so a
new snapshot never serves stale chunks, and refuses a cache whose namespace says nothing about the
data. One embeddings chunk-year is about 8.4 MB on disk, so a Caldor-sized AOI is a few GB per year.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path

import numpy as np


def cache_dir() -> Path:
    """Root of every on-disk cache: ``$TESSERA_APPS_CACHE`` if set and not empty, else ``~/.cache/tessera_apps``.

    Read when called, so chunk caches and reference downloads always agree on the root.
    """
    root = os.environ.get("TESSERA_APPS_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "tessera_apps"


@dataclasses.dataclass(frozen=True)
class ChunkKey:
    """Address of one inner chunk of one year in one zone group."""

    zone: str
    year: int
    chunk_row: int
    chunk_col: int


@dataclasses.dataclass
class CacheStats:
    """Hit and miss counts, shared by a cache and every namespace derived from it."""

    hits: int = 0
    misses: int = 0


class ChunkCache:
    """Directory of ``.npy`` files, one per (namespace, zone, year, chunk, layer).

    Writes go to a temporary file in the same directory and are renamed into place, so a crash never
    leaves a partial entry behind. An entry whose shape or dtype does not match what the caller
    expects is treated as a miss and overwritten.
    """

    UNPINNED = "unpinned"

    def __init__(
        self, root: Path | str | None = None, namespace: str = UNPINNED, *, stats: CacheStats | None = None
    ) -> None:
        self.root = Path(root) if root is not None else cache_dir() / "chunks"
        self.namespace = namespace
        self.stats = stats if stats is not None else CacheStats()

    @property
    def hits(self) -> int:
        """Cache hits so far, across this cache and its derived namespaces."""
        return self.stats.hits

    @property
    def misses(self) -> int:
        """Cache misses so far, across this cache and its derived namespaces."""
        return self.stats.misses

    def for_namespace(self, namespace: str) -> ChunkCache:
        """A cache on the same root and with the same stats, under a different namespace."""
        return ChunkCache(self.root, namespace, stats=self.stats)

    def path(self, key: ChunkKey, layer: str) -> Path:
        """File holding one layer of one chunk."""
        return (
            self.root / self.namespace / key.zone / str(key.year) / f"{key.chunk_row}_{key.chunk_col}" / f"{layer}.npy"
        )

    def get(
        self, key: ChunkKey, layer: str, *, shape: tuple[int, ...] | None = None, dtype: np.dtype | None = None
    ) -> np.ndarray | None:
        """Cached array, or None on a miss or when it does not have the expected shape and dtype."""
        p = self.path(key, layer)
        if not p.exists():
            self.stats.misses += 1
            return None
        try:
            arr = np.load(p, allow_pickle=False)
        except (OSError, ValueError):
            self.stats.misses += 1
            return None
        if (shape is not None and arr.shape != tuple(shape)) or (dtype is not None and arr.dtype != np.dtype(dtype)):
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return arr

    def put(self, key: ChunkKey, layer: str, array: np.ndarray) -> None:
        """Store one array atomically."""
        p = self.path(key, layer)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.save(fh, array, allow_pickle=False)
            Path(tmp).replace(p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
