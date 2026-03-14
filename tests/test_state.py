"""Contract-driven tests for state persistence and load semantics.

Sources:
- ARCHITECTURE.md#2.10 (state module boundaries and APIs)
- ARCHITECTURE.md#5 (state lifecycle and persistence strategy)
- PROTOTYPING.md#8 (embedding cache usage context)
- src/sfumato/state.py (load policy and overwrite contracts)

These tests are written ahead of full implementation and therefore tolerate
current ``NotImplementedError`` stubs by skipping those paths until dispatch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypeVar

import numpy as np
import pytest

from sfumato.layout_ai import LayoutColors, LayoutParams, ScrimParams, TextZone
from sfumato.news import CurationResult, Story
from sfumato.state import (
    EMBEDDING_CACHE_NPZ,
    LAYOUT_CACHE_JSON,
    NEWS_QUEUE_JSON,
    USED_PAINTINGS_JSON,
    AppState,
    EmbeddingCache,
    LayoutCache,
    NewsQueue,
    UsedPaintings,
    resolve_state_dir,
)

T = TypeVar("T")


def _or_skip_not_implemented(label: str, call: Callable[[], T]) -> T:
    """Run ``call`` or skip this test if state implementation is pending."""
    try:
        return call()
    except NotImplementedError as exc:
        pytest.skip(f"{label} pending implementation: {exc}")


def _story(index: int, *, published_at: datetime | None = None) -> Story:
    if published_at is None:
        published_at = datetime.now(timezone.utc)
    return Story(
        headline=f"story-{index}",
        summary=f"summary-{index}",
        source="test-source",
        category="tech",
        url=f"https://example.com/{index}",
        published_at=published_at,
        featured=index == 0,
    )


def _curation_result(count: int, *, tone: str = "test tone") -> CurationResult:
    stories = [_story(i) for i in range(count)]
    return CurationResult(
        stories=stories,
        tone_description=tone,
        curated_at=datetime.now(timezone.utc),
        feed_count=1,
        entry_count=count,
    )


def _layout_params(seed: str) -> LayoutParams:
    return LayoutParams(
        orientation="landscape",
        painting_title=f"title-{seed}",
        painting_artist=f"artist-{seed}",
        painting_description=f"description-{seed}",
        text_zone=TextZone(position="top-right", reason="quiet corner"),
        colors=LayoutColors(
            text_primary="#ffffff",
            text_secondary="#dddddd",
            text_dim="#999999",
            text_shadow="0 1px 2px rgba(0,0,0,0.5)",
            scrim_color="rgba(0,0,0,0.35)",
            panel_bg="#111111",
            border="#222222",
            accent="#ff7a3d",
        ),
        scrim=ScrimParams(
            position_css="top: 0; right: 0;",
            size_css="width: 1200px; height: 900px;",
            gradient_css="radial-gradient(circle, rgba(0,0,0,0.4), transparent 70%)",
        ),
        recommended_stories=3,
        template_hint="painting_text",
        portrait_layout=None,
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class TestResolveStateDirContract:
    """Path resolution for default, relative, and absolute ``state_dir``."""

    def test_resolve_state_dir_keeps_absolute_input(self, tmp_path: Path) -> None:
        resolved = _or_skip_not_implemented(
            "resolve_state_dir absolute",
            lambda: resolve_state_dir(tmp_path / "state", cwd=tmp_path, home=tmp_path),
        )
        assert resolved == (tmp_path / "state").resolve()

    def test_resolve_state_dir_resolves_relative_against_cwd(self, tmp_path: Path) -> None:
        resolved = _or_skip_not_implemented(
            "resolve_state_dir relative",
            lambda: resolve_state_dir("relative/state", cwd=tmp_path, home=tmp_path),
        )
        assert resolved == (tmp_path / "relative/state").resolve()

    def test_resolve_state_dir_expands_default_against_home(self, tmp_path: Path) -> None:
        fake_home = tmp_path / "home"
        resolved = _or_skip_not_implemented(
            "resolve_state_dir default",
            lambda: resolve_state_dir(None, cwd=tmp_path, home=fake_home),
        )
        assert resolved == (fake_home / ".sfumato/state").resolve()


class TestNewsQueueContract:
    """FIFO queue behavior, expiration, persistence, and fallback contracts."""

    def test_news_queue_fifo_peek_dequeue_contract(self, tmp_path: Path) -> None:
        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        enqueued = _or_skip_not_implemented(
            "NewsQueue.enqueue",
            lambda: queue.enqueue(_curation_result(5, tone="fifo"), batch_size=2),
        )

        assert enqueued == 3
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 3

        first_peek = _or_skip_not_implemented("NewsQueue.peek", queue.peek)
        assert first_peek is not None
        assert first_peek.stories[0].headline == "story-0"
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 3

        first = _or_skip_not_implemented("NewsQueue.dequeue", queue.dequeue)
        second = _or_skip_not_implemented("NewsQueue.dequeue", queue.dequeue)
        assert first is not None
        assert second is not None
        assert first.stories[0].headline == "story-0"
        assert second.stories[0].headline == "story-2"

    def test_news_queue_expire_drops_old_batches_only(self, tmp_path: Path) -> None:
        queue_path = tmp_path / NEWS_QUEUE_JSON
        old_dt = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        new_dt = datetime.now(timezone.utc).isoformat()
        queue_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "batches": [
                        {
                            "stories": [_story(0, published_at=datetime.now(timezone.utc)).__dict__ | {"published_at": new_dt}],
                            "tone_description": "old",
                            "enqueued_at": old_dt,
                        },
                        {
                            "stories": [_story(1, published_at=datetime.now(timezone.utc)).__dict__ | {"published_at": new_dt}],
                            "tone_description": "new",
                            "enqueued_at": new_dt,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        _or_skip_not_implemented("NewsQueue.load", queue.load)
        removed = _or_skip_not_implemented("NewsQueue.expire", lambda: queue.expire(7))
        assert removed == 1
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 1

    def test_news_queue_save_then_load_round_trip(self, tmp_path: Path) -> None:
        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        _or_skip_not_implemented(
            "NewsQueue.enqueue",
            lambda: queue.enqueue(_curation_result(4, tone="roundtrip"), batch_size=2),
        )
        _or_skip_not_implemented("NewsQueue.save", queue.save)

        reloaded = _or_skip_not_implemented(
            "NewsQueue.__init__ reload", lambda: NewsQueue(tmp_path)
        )
        _or_skip_not_implemented("NewsQueue.load", reloaded.load)

        assert _or_skip_not_implemented("NewsQueue.size", lambda: reloaded.size) == 2
        next_batch = _or_skip_not_implemented("NewsQueue.peek", reloaded.peek)
        assert next_batch is not None
        assert next_batch.tone_description == "roundtrip"

    def test_news_queue_missing_file_and_legacy_layout_fallback(self, tmp_path: Path) -> None:
        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        _or_skip_not_implemented("NewsQueue.load missing", queue.load)
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 0

        legacy_payload = {
            "batches": [
                {
                    "stories": [
                        {
                            "headline": "legacy",
                            "summary": "legacy summary",
                            "source": "legacy source",
                            "category": "legacy",
                            "url": "https://example.com/legacy",
                            "published_at": datetime.now(timezone.utc).isoformat(),
                            "featured": False,
                        }
                    ],
                    "tone_description": "legacy-tone",
                    "enqueued_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        }
        (tmp_path / NEWS_QUEUE_JSON).write_text(
            json.dumps(legacy_payload),
            encoding="utf-8",
        )

        _or_skip_not_implemented("NewsQueue.load legacy", queue.load)
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 1

    def test_news_queue_empty_save_overwrites_prior_file(self, tmp_path: Path) -> None:
        preexisting = {
            "version": 1,
            "batches": [{"stories": [], "tone_description": "stale", "enqueued_at": "2026-01-01T00:00:00+00:00"}],
        }
        (tmp_path / NEWS_QUEUE_JSON).write_text(json.dumps(preexisting), encoding="utf-8")

        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        _or_skip_not_implemented("NewsQueue.save", queue.save)

        payload = _read_json(tmp_path / NEWS_QUEUE_JSON)
        assert isinstance(payload.get("batches"), list)
        assert payload.get("batches") == []

    def test_news_queue_corrupt_partial_file_does_not_raise_on_load(self, tmp_path: Path) -> None:
        (tmp_path / NEWS_QUEUE_JSON).write_text("{\n  \"version\": 1,\n  \"batches\": [", encoding="utf-8")

        queue = _or_skip_not_implemented("NewsQueue.__init__", lambda: NewsQueue(tmp_path))
        _or_skip_not_implemented("NewsQueue.load partial", queue.load)
        assert _or_skip_not_implemented("NewsQueue.size", lambda: queue.size) == 0


class TestUsedPaintingsContract:
    """Used painting marks, reset semantics, and persistence contracts."""

    def test_used_paintings_mark_reset_and_persist_round_trip(self, tmp_path: Path) -> None:
        used = _or_skip_not_implemented("UsedPaintings.__init__", lambda: UsedPaintings(tmp_path))
        _or_skip_not_implemented("UsedPaintings.mark_used", lambda: used.mark_used("hash-a"))
        _or_skip_not_implemented("UsedPaintings.mark_used", lambda: used.mark_used("hash-b"))

        assert _or_skip_not_implemented("UsedPaintings.is_used", lambda: used.is_used("hash-a"))
        assert _or_skip_not_implemented("UsedPaintings.count", lambda: used.count) == 2

        _or_skip_not_implemented("UsedPaintings.save", used.save)

        reloaded = _or_skip_not_implemented(
            "UsedPaintings.__init__ reload", lambda: UsedPaintings(tmp_path)
        )
        _or_skip_not_implemented("UsedPaintings.load", reloaded.load)
        assert _or_skip_not_implemented("UsedPaintings.is_used", lambda: reloaded.is_used("hash-b"))

        _or_skip_not_implemented("UsedPaintings.reset", reloaded.reset)
        assert _or_skip_not_implemented("UsedPaintings.count", lambda: reloaded.count) == 0

    def test_used_paintings_load_disk_snapshot_precedes_in_memory(self, tmp_path: Path) -> None:
        payload = {"version": 1, "content_hashes": ["disk-hash"]}
        (tmp_path / USED_PAINTINGS_JSON).write_text(json.dumps(payload), encoding="utf-8")

        used = _or_skip_not_implemented("UsedPaintings.__init__", lambda: UsedPaintings(tmp_path))
        _or_skip_not_implemented(
            "UsedPaintings.mark_used",
            lambda: used.mark_used("in-memory-hash"),
        )

        _or_skip_not_implemented("UsedPaintings.load", used.load)
        assert _or_skip_not_implemented("UsedPaintings.is_used", lambda: used.is_used("disk-hash"))
        assert not _or_skip_not_implemented(
            "UsedPaintings.is_used",
            lambda: used.is_used("in-memory-hash"),
        )


class TestLayoutCacheContract:
    """Layout cache CRUD and overwrite/fallback contract tests."""

    def test_layout_cache_put_get_has_and_persist_round_trip(self, tmp_path: Path) -> None:
        cache = _or_skip_not_implemented("LayoutCache.__init__", lambda: LayoutCache(tmp_path))
        layout = _layout_params("one")

        _or_skip_not_implemented("LayoutCache.put", lambda: cache.put("hash-1", layout))
        assert _or_skip_not_implemented("LayoutCache.has", lambda: cache.has("hash-1"))
        loaded = _or_skip_not_implemented("LayoutCache.get", lambda: cache.get("hash-1"))
        assert loaded is not None
        assert loaded.painting_title == layout.painting_title

        _or_skip_not_implemented("LayoutCache.save", cache.save)

        reloaded = _or_skip_not_implemented("LayoutCache.__init__ reload", lambda: LayoutCache(tmp_path))
        _or_skip_not_implemented("LayoutCache.load", reloaded.load)
        loaded_again = _or_skip_not_implemented("LayoutCache.get", lambda: reloaded.get("hash-1"))
        assert loaded_again is not None
        assert loaded_again.painting_artist == layout.painting_artist

    def test_layout_cache_legacy_layout_and_missing_file_fallback(self, tmp_path: Path) -> None:
        cache = _or_skip_not_implemented("LayoutCache.__init__", lambda: LayoutCache(tmp_path))
        _or_skip_not_implemented("LayoutCache.load missing", cache.load)
        assert _or_skip_not_implemented("LayoutCache.size", lambda: cache.size) == 0

        legacy = {
            "layouts": {
                "legacy-hash": {
                    "orientation": "landscape",
                    "painting_title": "legacy-title",
                    "painting_artist": "legacy-artist",
                    "painting_description": "legacy-description",
                    "text_zone": {"position": "top-right", "reason": "legacy"},
                    "colors": {
                        "text_primary": "#ffffff",
                        "text_secondary": "#dddddd",
                        "text_dim": "#999999",
                        "text_shadow": "0 1px 2px rgba(0,0,0,0.5)",
                        "scrim_color": "rgba(0,0,0,0.35)",
                        "panel_bg": "#111111",
                        "border": "#222222",
                        "accent": "#ff7a3d",
                    },
                    "scrim": {
                        "position_css": "top: 0; right: 0;",
                        "size_css": "width: 1000px; height: 800px;",
                        "gradient_css": "radial-gradient(circle, rgba(0,0,0,0.4), transparent 70%)",
                    },
                    "recommended_stories": 3,
                    "template_hint": "painting_text",
                    "portrait_layout": None,
                }
            }
        }
        (tmp_path / LAYOUT_CACHE_JSON).write_text(json.dumps(legacy), encoding="utf-8")
        _or_skip_not_implemented("LayoutCache.load legacy", cache.load)
        assert _or_skip_not_implemented("LayoutCache.has", lambda: cache.has("legacy-hash"))

    def test_layout_cache_save_overwrites_stale_snapshot_not_merges(self, tmp_path: Path) -> None:
        stale = {
            "version": 1,
            "layouts": {
                "stale-hash": {
                    "orientation": "landscape",
                    "painting_title": "stale",
                    "painting_artist": "stale",
                    "painting_description": "stale",
                    "text_zone": {"position": "top-right", "reason": "stale"},
                    "colors": {
                        "text_primary": "#ffffff",
                        "text_secondary": "#dddddd",
                        "text_dim": "#999999",
                        "text_shadow": "0 1px 2px rgba(0,0,0,0.5)",
                        "scrim_color": "rgba(0,0,0,0.35)",
                        "panel_bg": "#111111",
                        "border": "#222222",
                        "accent": "#ff7a3d",
                    },
                    "scrim": {
                        "position_css": "top: 0; right: 0;",
                        "size_css": "width: 1000px; height: 800px;",
                        "gradient_css": "radial-gradient(circle, rgba(0,0,0,0.4), transparent 70%)",
                    },
                    "recommended_stories": 3,
                    "template_hint": "painting_text",
                    "portrait_layout": None,
                }
            },
        }
        (tmp_path / LAYOUT_CACHE_JSON).write_text(json.dumps(stale), encoding="utf-8")

        cache = _or_skip_not_implemented("LayoutCache.__init__", lambda: LayoutCache(tmp_path))
        _or_skip_not_implemented("LayoutCache.put", lambda: cache.put("fresh-hash", _layout_params("fresh")))
        _or_skip_not_implemented("LayoutCache.save", cache.save)

        payload = _read_json(tmp_path / LAYOUT_CACHE_JSON)
        assert isinstance(payload.get("layouts"), dict)
        assert "fresh-hash" in payload["layouts"]
        assert "stale-hash" not in payload["layouts"]


class TestEmbeddingCacheContract:
    """Embedding cache CRUD and NPZ persistence/load fallback contracts."""

    def test_embedding_cache_put_get_has_save_load_npz_round_trip(self, tmp_path: Path) -> None:
        cache = _or_skip_not_implemented(
            "EmbeddingCache.__init__", lambda: EmbeddingCache(tmp_path)
        )
        vector = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        _or_skip_not_implemented("EmbeddingCache.put", lambda: cache.put("key-1", vector))
        assert _or_skip_not_implemented("EmbeddingCache.has", lambda: cache.has("key-1"))

        loaded_vec = _or_skip_not_implemented("EmbeddingCache.get", lambda: cache.get("key-1"))
        assert loaded_vec is not None
        np.testing.assert_allclose(loaded_vec, vector)

        _or_skip_not_implemented("EmbeddingCache.save", cache.save)
        npz_path = tmp_path / EMBEDDING_CACHE_NPZ
        assert npz_path.exists()

        with np.load(npz_path, allow_pickle=False) as archive:
            assert "key-1" in archive.files
            np.testing.assert_allclose(archive["key-1"], vector)

        reloaded = _or_skip_not_implemented(
            "EmbeddingCache.__init__ reload", lambda: EmbeddingCache(tmp_path)
        )
        _or_skip_not_implemented("EmbeddingCache.load", reloaded.load)
        reloaded_vec = _or_skip_not_implemented(
            "EmbeddingCache.get reload", lambda: reloaded.get("key-1")
        )
        assert reloaded_vec is not None
        np.testing.assert_allclose(reloaded_vec, vector)

    def test_embedding_cache_missing_file_and_partial_archive_fallback(self, tmp_path: Path) -> None:
        cache = _or_skip_not_implemented(
            "EmbeddingCache.__init__", lambda: EmbeddingCache(tmp_path)
        )
        _or_skip_not_implemented("EmbeddingCache.load missing", cache.load)
        assert _or_skip_not_implemented("EmbeddingCache.size", lambda: cache.size) == 0

        (tmp_path / EMBEDDING_CACHE_NPZ).write_bytes(b"not a valid npz archive")
        _or_skip_not_implemented("EmbeddingCache.load partial", cache.load)
        assert _or_skip_not_implemented("EmbeddingCache.size", lambda: cache.size) == 0

    def test_embedding_cache_load_disk_snapshot_precedes_stale_memory(self, tmp_path: Path) -> None:
        disk_vector = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.savez(tmp_path / EMBEDDING_CACHE_NPZ, shared_key=disk_vector)

        cache = _or_skip_not_implemented(
            "EmbeddingCache.__init__", lambda: EmbeddingCache(tmp_path)
        )
        _or_skip_not_implemented(
            "EmbeddingCache.put",
            lambda: cache.put("shared_key", np.array([-1.0, -2.0, -3.0], dtype=np.float32)),
        )

        _or_skip_not_implemented("EmbeddingCache.load", cache.load)
        loaded = _or_skip_not_implemented("EmbeddingCache.get", lambda: cache.get("shared_key"))
        assert loaded is not None
        np.testing.assert_allclose(loaded, disk_vector)


class TestAppStateLoadContract:
    """Aggregate load behavior for missing files and path resolution contracts."""

    def test_app_state_load_creates_defaults_when_all_files_missing(self, tmp_path: Path) -> None:
        state = _or_skip_not_implemented("AppState.load", lambda: AppState.load(tmp_path))

        assert isinstance(state.news_queue, NewsQueue)
        assert isinstance(state.used_paintings, UsedPaintings)
        assert isinstance(state.layout_cache, LayoutCache)
        assert isinstance(state.embedding_cache, EmbeddingCache)

        assert _or_skip_not_implemented("NewsQueue.size", lambda: state.news_queue.size) == 0
        assert _or_skip_not_implemented(
            "UsedPaintings.count", lambda: state.used_paintings.count
        ) == 0
        assert _or_skip_not_implemented("LayoutCache.size", lambda: state.layout_cache.size) == 0
        assert _or_skip_not_implemented(
            "EmbeddingCache.size", lambda: state.embedding_cache.size
        ) == 0

    def test_app_state_load_handles_legacy_files_without_startup_failure(self, tmp_path: Path) -> None:
        (tmp_path / NEWS_QUEUE_JSON).write_text(
            json.dumps(
                {
                    "batches": [
                        {
                            "stories": [
                                {
                                    "headline": "legacy",
                                    "summary": "legacy summary",
                                    "source": "legacy source",
                                    "category": "legacy",
                                    "url": "https://example.com/legacy",
                                    "published_at": datetime.now(timezone.utc).isoformat(),
                                    "featured": True,
                                }
                            ],
                            "tone_description": "legacy tone",
                            "enqueued_at": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        (tmp_path / USED_PAINTINGS_JSON).write_text(
            json.dumps({"hashes": ["legacy-hash"]}),
            encoding="utf-8",
        )
        (tmp_path / LAYOUT_CACHE_JSON).write_text(
            json.dumps({"layouts": {}}),
            encoding="utf-8",
        )
        np.savez(tmp_path / EMBEDDING_CACHE_NPZ, legacy_vec=np.array([0.0, 1.0]))

        state = _or_skip_not_implemented("AppState.load", lambda: AppState.load(tmp_path))
        assert isinstance(state, AppState)
        assert _or_skip_not_implemented("NewsQueue.size", lambda: state.news_queue.size) >= 0
        assert _or_skip_not_implemented(
            "UsedPaintings.count", lambda: state.used_paintings.count
        ) >= 0

    def test_app_state_load_relative_state_dir_resolves_from_cwd(self, tmp_path: Path) -> None:
        relative_dir = Path("state-data")
        expected = (tmp_path / relative_dir).resolve()

        resolved = _or_skip_not_implemented(
            "resolve_state_dir relative for AppState",
            lambda: resolve_state_dir(relative_dir, cwd=tmp_path, home=tmp_path),
        )
        assert resolved == expected
