"""State contract surfaces for persistence and runtime state.

Architecture source of truth:
- ARCHITECTURE.md#2.10 (module responsibility, public interface, storage layout)
- PROTOTYPING.md#8 (embedding matching context and cache usage)

This module intentionally provides contract definitions only for the first
state milestone. It does not implement disk IO, mutation behavior, or any
runtime side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np

if TYPE_CHECKING:
    from sfumato.layout_ai import LayoutParams
    from sfumato.news import CurationResult, Story


DEFAULT_STATE_DIR = Path("~/.sfumato/state")

NEWS_QUEUE_JSON = "news_queue.json"
USED_PAINTINGS_JSON = "used_paintings.json"
LAYOUT_CACHE_JSON = "layout_cache.json"
EMBEDDING_CACHE_NPZ = "embedding_cache.npz"


class StoryJson(TypedDict):
    """JSON boundary for ``sfumato.news.Story`` inside queued batches.

    Boundary notes:
    - ``published_at`` uses ISO-8601 text.
    - ``featured`` is required to preserve queue payload fidelity.
    """

    headline: str
    summary: str
    source: str
    category: str
    url: str
    published_at: str
    featured: bool


class QueuedBatchJson(TypedDict):
    """JSON boundary for ``QueuedBatch`` in ``news_queue.json``."""

    stories: list[StoryJson]
    tone_description: str
    enqueued_at: str


class NewsQueueFileJson(TypedDict):
    """Versioned JSON schema boundary for queue persistence."""

    version: int
    batches: list[QueuedBatchJson]


class UsedPaintingsFileJson(TypedDict):
    """Versioned JSON schema boundary for used painting hashes."""

    version: int
    content_hashes: list[str]


class LayoutCacheEntryJson(TypedDict):
    """JSON-serializable ``LayoutParams`` projection boundary.

    Contract note:
    - Value is a plain JSON object produced from ``LayoutParams`` fields.
    - ``None`` values are explicit and must overwrite previous persisted values.
    """


class LayoutCacheFileJson(TypedDict):
    """Versioned JSON schema boundary for layout cache persistence."""

    version: int
    layouts: dict[str, LayoutCacheEntryJson]


@dataclass(frozen=True)
class EmbeddingCacheNpzBoundary:
    """Boundary contract for ``embedding_cache.npz``.

    Contract:
    - Format is NumPy ``.npz`` under ``~/.sfumato/state/embedding_cache.npz``.
    - Each array key is a stable string key used by ``EmbeddingCache``.
    - Each array value is a 1-D ``numpy.ndarray`` numeric embedding vector.
    - Missing key => cache miss; no synthetic fallback vector is allowed.
    """

    path: Path
    allow_pickle: bool = False
    required_ndim: int = 1


@dataclass(frozen=True)
class StateLoadPolicy:
    """Contract notes for load-time precedence and fallback behavior.

    Attributes:
        missing_file: Missing state file behavior. Contract: return defaults.
        legacy_layout: Legacy JSON shape fallback behavior.
        stale_vs_new: Merge precedence between disk snapshot and in-memory data.
        partial_write_or_conflict: Behavior when partial write or CAS conflict is
            detected while full CAS is not implemented.
    """

    missing_file: str = "default-empty"
    legacy_layout: str = "best-effort-upgrade"
    stale_vs_new: str = "disk-wins-on-load-memory-wins-after-load"
    partial_write_or_conflict: str = "last-successful-snapshot-no-raise"


LOAD_POLICY = StateLoadPolicy()


def resolve_state_dir(
    state_dir: Path | str | None,
    *,
    cwd: Path | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve caller-supplied ``state_dir`` to an absolute directory path.

    Path resolution rules (contract):
    - ``None`` resolves to ``~/.sfumato/state``.
    - ``~``-prefixed paths expand against ``home`` or the process home.
    - Absolute paths remain absolute.
    - Relative paths resolve against ``cwd`` or process current directory.
    - Returned path is normalized/absolute; this function does not create it.

    Args:
        state_dir: Caller-provided directory, or ``None`` for default path.
        cwd: Optional base path for relative inputs.
        home: Optional home path for ``~`` expansion.

    Returns:
        Absolute path to the state directory.

    Raises:
        NotImplementedError: This module currently defines contract only.
    """
    raise NotImplementedError(
        "Contract-only stub: path resolution implementation pending."
    )


@dataclass
class QueuedBatch:
    """A queued news batch persisted in FIFO order.

    Serialization boundary:
    - Stored inside ``news_queue.json`` as ``QueuedBatchJson``.
    - ``enqueued_at`` is serialized as ISO-8601 text.
    """

    stories: list[Story]
    tone_description: str
    enqueued_at: datetime


class NewsQueue:
    """FIFO queue of news story batches, persisted to ``news_queue.json``.

    Save/load contract:
    - ``load()`` missing file => empty queue.
    - Legacy file layouts must be accepted when unambiguous; unknown fields are
      ignored for backward compatibility.
    - During ``load()``, disk snapshot is authoritative.
    - During ``save()``, current in-memory queue is authoritative.
    - Null/empty overwrite is explicit: saving an empty queue writes an empty
      persisted queue payload, replacing prior non-empty content.
    - Partial-write/CAS-conflict in this milestone: fall back to last complete
      valid snapshot behavior; no CAS guarantees yet.
    """

    def __init__(self, state_dir: Path | str | None) -> None:
        """Initialize queue with caller-supplied state directory contract.

        Args:
            state_dir: Absolute/relative/``~``/``None`` path input resolved by
                ``resolve_state_dir``.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: NewsQueue.__init__ not implemented."
        )

    def enqueue(self, result: CurationResult, batch_size: int) -> int:
        """Split curation result into batches and append to queue.

        Returns:
            Number of batches enqueued.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: NewsQueue.enqueue not implemented."
        )

    def dequeue(self) -> QueuedBatch | None:
        """Remove and return the next batch.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: NewsQueue.dequeue not implemented."
        )

    def peek(self) -> QueuedBatch | None:
        """Return the next batch without removal.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError("Contract-only stub: NewsQueue.peek not implemented.")

    def expire(self, expire_days: int) -> int:
        """Drop batches older than ``expire_days`` and return removed count.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: NewsQueue.expire not implemented."
        )

    @property
    def size(self) -> int:
        """Return current batch count.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError("Contract-only stub: NewsQueue.size not implemented.")

    def save(self) -> None:
        """Persist queue snapshot to JSON boundary.

        Clear/null overwrite semantics:
        - Persisted queue content is replaced by in-memory queue content.
        - Empty in-memory queue overwrites persisted queue to empty.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError("Contract-only stub: NewsQueue.save not implemented.")

    def load(self) -> None:
        """Load queue snapshot from disk according to ``LOAD_POLICY``.

        Missing file and legacy fallback:
        - Missing file => queue remains default-empty.
        - Legacy schema => best-effort upgrade when structural intent is clear.
        - Corrupt/partial content => keep last complete valid snapshot behavior.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError("Contract-only stub: NewsQueue.load not implemented.")


class UsedPaintings:
    """Set of displayed painting content hashes in ``used_paintings.json``.

    Save/load contract:
    - Missing file => empty set.
    - Persisted set is fully overwritten by current in-memory set on ``save()``.
    - ``reset()`` plus ``save()`` is explicit persisted clear.
    """

    def __init__(self, state_dir: Path | str | None) -> None:
        """Initialize used-paintings tracker with state directory contract.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.__init__ not implemented."
        )

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting hash as used.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.mark_used not implemented."
        )

    def is_used(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` is in the used set.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.is_used not implemented."
        )

    def reset(self) -> None:
        """Clear in-memory used-hash set.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.reset not implemented."
        )

    @property
    def count(self) -> int:
        """Return number of tracked used hashes.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.count not implemented."
        )

    def save(self) -> None:
        """Persist used-hash snapshot to JSON boundary.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.save not implemented."
        )

    def load(self) -> None:
        """Load used-hash snapshot from disk according to ``LOAD_POLICY``.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: UsedPaintings.load not implemented."
        )


class LayoutCache:
    """Cache of ``LayoutParams`` keyed by content hash in ``layout_cache.json``.

    Clear/null overwrite semantics:
    - Saving a key with value null-equivalent in the JSON boundary is an
      explicit overwrite for that key (no merge with old value).
    - Saving the full cache snapshot overwrites the persisted map.
    """

    def __init__(self, state_dir: Path | str | None) -> None:
        """Initialize layout cache with state directory contract.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.__init__ not implemented."
        )

    def get(self, content_hash: str) -> LayoutParams | None:
        """Return cached layout parameters for ``content_hash`` if present.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.get not implemented."
        )

    def put(self, content_hash: str, layout: LayoutParams) -> None:
        """Insert/replace cached ``LayoutParams`` for ``content_hash``.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.put not implemented."
        )

    def has(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` exists in cache.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.has not implemented."
        )

    @property
    def size(self) -> int:
        """Return number of cached layout entries.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.size not implemented."
        )

    def save(self) -> None:
        """Persist layout cache snapshot to JSON boundary.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.save not implemented."
        )

    def load(self) -> None:
        """Load layout cache snapshot with missing-file and legacy fallback.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: LayoutCache.load not implemented."
        )


class EmbeddingCache:
    """Cache of embedding vectors keyed by semantic lookup key.

    Persistence boundary:
    - ``embedding_cache.npz`` stores one 1-D array per key.
    - Missing file => empty cache on ``load()``.
    - On ``save()``, persisted archive is replaced by current in-memory cache.
    - Partial archive writes and CAS conflicts currently degrade to last complete
      valid snapshot behavior for backward compatibility.
    """

    def __init__(self, state_dir: Path | str | None) -> None:
        """Initialize embedding cache with state directory contract.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.__init__ not implemented."
        )

    def get(self, key: str) -> np.ndarray | None:
        """Return embedding vector by key.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.get not implemented."
        )

    def put(self, key: str, vector: np.ndarray) -> None:
        """Insert or replace embedding vector for key.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.put not implemented."
        )

    def has(self, key: str) -> bool:
        """Return whether ``key`` exists in cache.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.has not implemented."
        )

    @property
    def size(self) -> int:
        """Return number of cached embedding vectors.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.size not implemented."
        )

    def save(self) -> None:
        """Persist embedding cache snapshot to NPZ boundary.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.save not implemented."
        )

    def load(self) -> None:
        """Load embedding cache snapshot with fallback behavior.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: EmbeddingCache.load not implemented."
        )


@dataclass
class AppState:
    """Aggregate root for daemon state surfaces.

    Main-path semantics:
    - ``AppState.load()`` resolves path, constructs all components, runs
      component ``load()`` in deterministic order, and returns the aggregate.
    - Missing files across any component produce default-empty component state.
    - During ``load()``, persisted disk snapshots take precedence over any
      ephemeral in-memory defaults created at construction time.
    - ``save_all()`` persists all in-memory components as the new authoritative
      snapshot.
    """

    news_queue: NewsQueue
    used_paintings: UsedPaintings
    layout_cache: LayoutCache
    embedding_cache: EmbeddingCache

    @classmethod
    def load(cls, state_dir: Path | str | None) -> AppState:
        """Load all state components from ``state_dir`` using ``LOAD_POLICY``.

        Legacy fallback contract:
        - If current file schema is absent but compatible legacy keys exist,
          component loaders should best-effort map legacy payloads to current
          in-memory models.
        - If mapping is ambiguous, keep defaults and preserve backward
          compatibility by not raising during daemon startup.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError("Contract-only stub: AppState.load not implemented.")

    def save_all(self) -> None:
        """Persist all state components in a single logical save flow.

        Partial-write/CAS-conflict in-scope behavior (pre-CAS milestone):
        - Full compare-and-swap guarantees are not implemented in this module.
        - Implementations must remain backward-compatible by favoring last
          complete valid snapshot semantics over raising startup-fatal errors.

        Raises:
            NotImplementedError: This module currently defines contract only.
        """
        raise NotImplementedError(
            "Contract-only stub: AppState.save_all not implemented."
        )


__all__ = [
    "DEFAULT_STATE_DIR",
    "EMBEDDING_CACHE_NPZ",
    "EmbeddingCache",
    "EmbeddingCacheNpzBoundary",
    "AppState",
    "LAYOUT_CACHE_JSON",
    "LOAD_POLICY",
    "LayoutCache",
    "LayoutCacheEntryJson",
    "LayoutCacheFileJson",
    "NEWS_QUEUE_JSON",
    "NewsQueue",
    "NewsQueueFileJson",
    "QueuedBatch",
    "QueuedBatchJson",
    "StateLoadPolicy",
    "StoryJson",
    "USED_PAINTINGS_JSON",
    "UsedPaintings",
    "UsedPaintingsFileJson",
    "resolve_state_dir",
]
