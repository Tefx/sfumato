"""State contract surfaces for persistence and runtime state.

Architecture source of truth:
- ARCHITECTURE.md#2.10 (module responsibility, public interface, storage layout)
- PROTOTYPING.md#8 (embedding matching context and cache usage)

This module intentionally provides contract definitions only for the first
state milestone. It does not implement disk IO, mutation behavior, or any
runtime side effects.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
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

    orientation: str
    painting_title: str
    painting_artist: str
    painting_description: str
    text_zone: dict[str, str]
    colors: dict[str, str]
    scrim: dict[str, str]
    recommended_stories: int
    template_hint: str
    portrait_layout: dict[str, object] | None


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


def _to_story_json(story: Story) -> StoryJson:
    return {
        "headline": story.headline,
        "summary": story.summary,
        "source": story.source,
        "category": story.category,
        "url": story.url,
        "published_at": story.published_at.isoformat(),
        "featured": story.featured,
    }


def _from_story_json(payload: dict[str, object]) -> Story:
    from sfumato.news import Story

    published_raw = payload.get("published_at")
    if not isinstance(published_raw, str):
        raise ValueError("Story payload missing published_at")

    return Story(
        headline=str(payload.get("headline", "")),
        summary=str(payload.get("summary", "")),
        source=str(payload.get("source", "")),
        category=str(payload.get("category", "")),
        url=str(payload.get("url", "")),
        published_at=datetime.fromisoformat(published_raw),
        featured=bool(payload.get("featured", False)),
    )


def _to_layout_json(layout: LayoutParams) -> LayoutCacheEntryJson:
    portrait_layout: dict[str, object] | None = None
    if layout.portrait_layout is not None:
        portrait_layout = {
            "painting_width_percent": layout.portrait_layout.painting_width_percent,
            "left_panel_color": layout.portrait_layout.left_panel_color,
            "right_panel_color": layout.portrait_layout.right_panel_color,
            "info_side": layout.portrait_layout.info_side,
        }

    return {
        "orientation": layout.orientation,
        "painting_title": layout.painting_title,
        "painting_artist": layout.painting_artist,
        "painting_description": layout.painting_description,
        "text_zone": {
            "position": layout.text_zone.position,
            "reason": layout.text_zone.reason,
        },
        "colors": {
            "text_primary": layout.colors.text_primary,
            "text_secondary": layout.colors.text_secondary,
            "text_dim": layout.colors.text_dim,
            "text_shadow": layout.colors.text_shadow,
            "scrim_color": layout.colors.scrim_color,
            "panel_bg": layout.colors.panel_bg,
            "border": layout.colors.border,
            "accent": layout.colors.accent,
        },
        "scrim": {
            "position_css": layout.scrim.position_css,
            "size_css": layout.scrim.size_css,
            "gradient_css": layout.scrim.gradient_css,
        },
        "recommended_stories": layout.recommended_stories,
        "template_hint": layout.template_hint,
        "portrait_layout": portrait_layout,
    }


def _from_layout_json(payload: dict[str, object]) -> LayoutParams:
    from sfumato.layout_ai import (
        LayoutColors,
        LayoutParams,
        PortraitLayout,
        ScrimParams,
        TextZone,
    )

    text_zone_raw = payload.get("text_zone")
    colors_raw = payload.get("colors")
    scrim_raw = payload.get("scrim")

    if not isinstance(text_zone_raw, dict):
        raise ValueError("Layout payload missing text_zone")
    if not isinstance(colors_raw, dict):
        raise ValueError("Layout payload missing colors")
    if not isinstance(scrim_raw, dict):
        raise ValueError("Layout payload missing scrim")

    portrait_payload = payload.get("portrait_layout")
    portrait_layout: PortraitLayout | None = None
    if isinstance(portrait_payload, dict):
        portrait_layout = PortraitLayout(
            painting_width_percent=int(
                portrait_payload.get("painting_width_percent", 50)
            ),
            left_panel_color=str(portrait_payload.get("left_panel_color", "#000000")),
            right_panel_color=str(portrait_payload.get("right_panel_color", "#000000")),
            info_side=str(portrait_payload.get("info_side", "left")),
        )

    return LayoutParams(
        orientation=str(payload.get("orientation", "landscape")),
        painting_title=str(payload.get("painting_title", "")),
        painting_artist=str(payload.get("painting_artist", "")),
        painting_description=str(payload.get("painting_description", "")),
        text_zone=TextZone(
            position=str(text_zone_raw.get("position", "top-right")),
            reason=str(text_zone_raw.get("reason", "")),
        ),
        colors=LayoutColors(
            text_primary=str(colors_raw.get("text_primary", "#ffffff")),
            text_secondary=str(colors_raw.get("text_secondary", "#dddddd")),
            text_dim=str(colors_raw.get("text_dim", "#999999")),
            text_shadow=str(colors_raw.get("text_shadow", "0 1px 2px rgba(0,0,0,0.5)")),
            scrim_color=str(colors_raw.get("scrim_color", "rgba(0,0,0,0.35)")),
            panel_bg=str(colors_raw.get("panel_bg", "#111111")),
            border=str(colors_raw.get("border", "#222222")),
            accent=str(colors_raw.get("accent", "#ff7a3d")),
        ),
        scrim=ScrimParams(
            position_css=str(scrim_raw.get("position_css", "")),
            size_css=str(scrim_raw.get("size_css", "")),
            gradient_css=str(scrim_raw.get("gradient_css", "")),
        ),
        recommended_stories=_coerce_int(payload.get("recommended_stories"), default=3),
        template_hint=str(payload.get("template_hint", "painting_text")),
        portrait_layout=portrait_layout,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        temp_file.write(content)
        temp_name = temp_file.name
    os.replace(temp_name, path)


def _coerce_int(value: object, *, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


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
        ValueError: If ``state_dir`` is an unsupported type.
    """
    base_cwd = cwd.resolve() if cwd is not None else Path.cwd().resolve()
    base_home = home.resolve() if home is not None else Path.home().resolve()

    if state_dir is None:
        candidate = base_home / ".sfumato" / "state"
    elif isinstance(state_dir, Path):
        candidate = state_dir
    elif isinstance(state_dir, str):
        if state_dir.startswith("~"):
            candidate = (
                base_home / state_dir[2:] if state_dir.startswith("~/") else base_home
            )
        else:
            candidate = Path(state_dir)
    else:
        raise ValueError(f"Unsupported state_dir type: {type(state_dir)!r}")

    if not candidate.is_absolute():
        candidate = base_cwd / candidate
    return candidate.resolve()


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
            ValueError: If ``state_dir`` cannot be resolved.
        """
        self._state_dir = resolve_state_dir(state_dir)
        self._path = self._state_dir / NEWS_QUEUE_JSON
        self._batches: list[QueuedBatch] = []

    def enqueue(self, result: CurationResult, batch_size: int) -> int:
        """Split curation result into batches and append to queue.

        Returns:
            Number of batches enqueued.

        Raises:
            ValueError: If ``batch_size`` is less than 1.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if not result.stories:
            return 0

        enqueued = 0
        for idx in range(0, len(result.stories), batch_size):
            self._batches.append(
                QueuedBatch(
                    stories=result.stories[idx : idx + batch_size],
                    tone_description=result.tone_description,
                    enqueued_at=datetime.now().astimezone(),
                )
            )
            enqueued += 1
        return enqueued

    def dequeue(self) -> QueuedBatch | None:
        """Remove and return the next batch.

        Raises:
            Nothing.
        """
        if not self._batches:
            return None
        return self._batches.pop(0)

    def peek(self) -> QueuedBatch | None:
        """Return the next batch without removal.

        Raises:
            Nothing.
        """
        if not self._batches:
            return None
        return self._batches[0]

    def expire(self, expire_days: int) -> int:
        """Drop batches older than ``expire_days`` and return removed count.

        Raises:
            ValueError: If ``expire_days`` is negative.
        """
        if expire_days < 0:
            raise ValueError("expire_days must be >= 0")

        cutoff = datetime.now().astimezone() - timedelta(days=expire_days)
        kept: list[QueuedBatch] = []
        removed = 0
        for batch in self._batches:
            if batch.enqueued_at < cutoff:
                removed += 1
            else:
                kept.append(batch)
        self._batches = kept
        return removed

    @property
    def size(self) -> int:
        """Return current batch count.

        Raises:
            Nothing.
        """
        return len(self._batches)

    def save(self) -> None:
        """Persist queue snapshot to JSON boundary.

        Clear/null overwrite semantics:
        - Persisted queue content is replaced by in-memory queue content.
        - Empty in-memory queue overwrites persisted queue to empty.

        Raises:
            OSError: If file write fails.
        """
        payload: NewsQueueFileJson = {
            "version": 1,
            "batches": [
                {
                    "stories": [_to_story_json(story) for story in batch.stories],
                    "tone_description": batch.tone_description,
                    "enqueued_at": batch.enqueued_at.isoformat(),
                }
                for batch in self._batches
            ],
        }
        _atomic_write_text(self._path, json.dumps(payload, ensure_ascii=False))

    def load(self) -> None:
        """Load queue snapshot from disk according to ``LOAD_POLICY``.

        Missing file and legacy fallback:
        - Missing file => queue remains default-empty.
        - Legacy schema => best-effort upgrade when structural intent is clear.
        - Corrupt/partial content => keep last complete valid snapshot behavior.

        Raises:
            Nothing.
        """
        if not self._path.exists():
            self._batches = []
            return

        try:
            raw_text = self._path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        batches_raw = payload.get("batches")
        if not isinstance(batches_raw, list):
            return

        loaded: list[QueuedBatch] = []
        for item in batches_raw:
            if not isinstance(item, dict):
                continue
            stories_raw = item.get("stories")
            enqueued_raw = item.get("enqueued_at")
            if not isinstance(stories_raw, list) or not isinstance(enqueued_raw, str):
                continue
            try:
                stories = [
                    _from_story_json(story_payload)
                    for story_payload in stories_raw
                    if isinstance(story_payload, dict)
                ]
                loaded.append(
                    QueuedBatch(
                        stories=stories,
                        tone_description=str(item.get("tone_description", "")),
                        enqueued_at=datetime.fromisoformat(enqueued_raw),
                    )
                )
            except (TypeError, ValueError):
                continue

        self._batches = loaded


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
            ValueError: If ``state_dir`` cannot be resolved.
        """
        self._state_dir = resolve_state_dir(state_dir)
        self._path = self._state_dir / USED_PAINTINGS_JSON
        self._content_hashes: set[str] = set()

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting hash as used.

        Raises:
            Nothing.
        """
        self._content_hashes.add(content_hash)

    def is_used(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` is in the used set.

        Raises:
            Nothing.
        """
        return content_hash in self._content_hashes

    def reset(self) -> None:
        """Clear in-memory used-hash set.

        Raises:
            Nothing.
        """
        self._content_hashes.clear()

    @property
    def count(self) -> int:
        """Return number of tracked used hashes.

        Raises:
            Nothing.
        """
        return len(self._content_hashes)

    def save(self) -> None:
        """Persist used-hash snapshot to JSON boundary.

        Raises:
            OSError: If file write fails.
        """
        payload: UsedPaintingsFileJson = {
            "version": 1,
            "content_hashes": sorted(self._content_hashes),
        }
        _atomic_write_text(self._path, json.dumps(payload, ensure_ascii=False))

    def load(self) -> None:
        """Load used-hash snapshot from disk according to ``LOAD_POLICY``.

        Raises:
            Nothing.
        """
        if not self._path.exists():
            self._content_hashes = set()
            return

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        hashes_raw = payload.get("content_hashes")
        if not isinstance(hashes_raw, list):
            legacy_hashes_raw = payload.get("hashes")
            if isinstance(legacy_hashes_raw, list):
                hashes_raw = legacy_hashes_raw
            else:
                return

        self._content_hashes = {str(item) for item in hashes_raw}


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
            ValueError: If ``state_dir`` cannot be resolved.
        """
        self._state_dir = resolve_state_dir(state_dir)
        self._path = self._state_dir / LAYOUT_CACHE_JSON
        self._layouts: dict[str, LayoutParams] = {}

    def get(self, content_hash: str) -> LayoutParams | None:
        """Return cached layout parameters for ``content_hash`` if present.

        Raises:
            Nothing.
        """
        return self._layouts.get(content_hash)

    def put(self, content_hash: str, layout: LayoutParams) -> None:
        """Insert/replace cached ``LayoutParams`` for ``content_hash``.

        Raises:
            Nothing.
        """
        self._layouts[content_hash] = layout

    def has(self, content_hash: str) -> bool:
        """Return whether ``content_hash`` exists in cache.

        Raises:
            Nothing.
        """
        return content_hash in self._layouts

    @property
    def size(self) -> int:
        """Return number of cached layout entries.

        Raises:
            Nothing.
        """
        return len(self._layouts)

    def save(self) -> None:
        """Persist layout cache snapshot to JSON boundary.

        Raises:
            OSError: If file write fails.
        """
        payload: LayoutCacheFileJson = {
            "version": 1,
            "layouts": {
                key: _to_layout_json(layout) for key, layout in self._layouts.items()
            },
        }
        _atomic_write_text(self._path, json.dumps(payload, ensure_ascii=False))

    def load(self) -> None:
        """Load layout cache snapshot with missing-file and legacy fallback.

        Raises:
            Nothing.
        """
        if not self._path.exists():
            self._layouts = {}
            return

        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        layouts_raw = payload.get("layouts")
        if not isinstance(layouts_raw, dict):
            return

        loaded: dict[str, LayoutParams] = {}
        for key, value in layouts_raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            try:
                loaded[key] = _from_layout_json(value)
            except (TypeError, ValueError):
                continue

        self._layouts = loaded


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
            ValueError: If ``state_dir`` cannot be resolved.
        """
        self._state_dir = resolve_state_dir(state_dir)
        self._path = self._state_dir / EMBEDDING_CACHE_NPZ
        self._embeddings: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        """Return embedding vector by key.

        Raises:
            Nothing.
        """
        return self._embeddings.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        """Insert or replace embedding vector for key.

        Raises:
            ValueError: If vector is not 1-dimensional.
        """
        if vector.ndim != 1:
            raise ValueError("Embedding vectors must be 1-dimensional")
        self._embeddings[key] = vector

    def has(self, key: str) -> bool:
        """Return whether ``key`` exists in cache.

        Raises:
            Nothing.
        """
        return key in self._embeddings

    @property
    def size(self) -> int:
        """Return number of cached embedding vectors.

        Raises:
            Nothing.
        """
        return len(self._embeddings)

    def save(self) -> None:
        """Persist embedding cache snapshot to NPZ boundary.

        Raises:
            OSError: If file write fails.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            dir=self._path.parent,
            delete=False,
        ) as temp_file:
            temp_name = temp_file.name

        try:
            np.savez(temp_name, **self._embeddings)
            os.replace(temp_name, self._path)
        finally:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass

    def load(self) -> None:
        """Load embedding cache snapshot with fallback behavior.

        Raises:
            Nothing.
        """
        if not self._path.exists():
            self._embeddings = {}
            return

        try:
            with np.load(self._path, allow_pickle=False) as archive:
                loaded: dict[str, np.ndarray] = {}
                for key in archive.files:
                    vec = archive[key]
                    if isinstance(vec, np.ndarray) and vec.ndim == 1:
                        loaded[key] = vec
        except (OSError, ValueError):
            return

        self._embeddings = loaded


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
            Nothing.
        """
        resolved = resolve_state_dir(state_dir)

        news_queue = NewsQueue(resolved)
        used_paintings = UsedPaintings(resolved)
        layout_cache = LayoutCache(resolved)
        embedding_cache = EmbeddingCache(resolved)

        news_queue.load()
        used_paintings.load()
        layout_cache.load()
        embedding_cache.load()

        return cls(
            news_queue=news_queue,
            used_paintings=used_paintings,
            layout_cache=layout_cache,
            embedding_cache=embedding_cache,
        )

    def save_all(self) -> None:
        """Persist all state components in a single logical save flow.

        Partial-write/CAS-conflict in-scope behavior (pre-CAS milestone):
        - Full compare-and-swap guarantees are not implemented in this module.
        - Implementations must remain backward-compatible by favoring last
          complete valid snapshot semantics over raising startup-fatal errors.

        Raises:
            OSError: If any component save fails.
        """
        self.news_queue.save()
        self.used_paintings.save()
        self.layout_cache.save()
        self.embedding_cache.save()


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
