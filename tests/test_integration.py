"""Integration tests for run_once using real config/state/render.

These tests exercise the runtime path with real config loading, real state
objects, and real rendering. Only LLM calls are mocked.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from sfumato.config import AppConfig, load_config
from sfumato.llm import LlmResponse
from sfumato.news import CurationResult, Story
from sfumato.orchestrator import RunOptions, run_news_refresh, run_once
from sfumato.paintings import content_hash, detect_orientation
from sfumato.state import (
    AppState,
    LayoutCache,
    NewsQueue,
    UsedPaintings,
)


def _write_config(tmp_path: Path, feeds_toml: str) -> AppConfig:
    """Create and load a real TOML config rooted in tmp_path."""
    config_path = tmp_path / "config.toml"
    data_dir = tmp_path / "data"
    paintings_dir = data_dir / "paintings"
    config_path.write_text(
        "\n".join(
            [
                f'data_dir = "{data_dir}"',
                "",
                "[tv]",
                'ip = ""',
                "port = 8002",
                "max_uploads = 5",
                "",
                "[schedule]",
                "news_interval_hours = 6",
                "rotate_interval_minutes = 15",
                "quiet_hours = [0, 6]",
                "active_hours = [7, 23]",
                "",
                "[news]",
                'language = "en"',
                "max_age_days = 3",
                "expire_days = 7",
                feeds_toml,
                "",
                "[paintings]",
                f'cache_dir = "{paintings_dir}"',
                "seed_size = 1",
                "pool_size = 1",
                'sources = ["met"]',
                'match_strategy = "random"',
                "",
                "[ai]",
                'cli = "gemini"',
                'model = "test-model"',
            ]
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


def _create_cached_painting(cache_dir: Path, source_id: str = "seed-1") -> Path:
    """Create a real cached painting (jpg + sidecar) for pool selection."""
    source_dir = cache_dir / "met"
    source_dir.mkdir(parents=True, exist_ok=True)
    image_path = source_dir / f"{source_id}.jpg"
    Image.new("RGB", (1600, 1000), color="#223366").save(image_path)

    sidecar = {
        "content_hash": content_hash(image_path),
        "title": "Integration Painting",
        "artist": "Integration Artist",
        "year": "2026",
        "source": "met",
        "source_id": source_id,
        "source_url": "https://example.test/painting",
        "orientation": detect_orientation(image_path).value,
        "width": 1600,
        "height": 1000,
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    return image_path


def _make_story(index: int) -> Story:
    """Build deterministic story payload for queue and render assertions."""
    return Story(
        headline=f"Headline {index}",
        summary=f"Summary {index}",
        source="Local Feed",
        category="Tech",
        url=f"https://example.test/story-{index}",
        published_at=datetime(2026, 3, 15, 8, index, tzinfo=timezone.utc),
        featured=index == 1,
    )


def _layout_json(*, recommended_stories: int) -> str:
    """Deterministic LLM JSON response for layout analysis."""
    return json.dumps(
        {
            "orientation": "landscape",
            "painting_title": "Integration Painting",
            "painting_artist": "Integration Artist",
            "painting_description": "Calm dusk scene with open sky and distant hills.",
            "text_zone": {
                "position": "top-right",
                "reason": "Quiet sky area suitable for text.",
            },
            "subject_zone": {
                "position": "bottom-left",
                "reason": "Main subject mass in lower-left foreground.",
            },
            "whisper_zone": {
                "position": "bottom-right",
                "reason": "Low-emphasis corner for art facts.",
                "max_width_percent": 18,
                "readability_notes": "Dark corner with minimal visual competition, readable at TV distance.",
            },
            "art_facts": [
                "Painted in the style of American Impressionism.",
                "Features a characteristic warm color palette.",
            ],
            "colors": {
                "text_primary": "#f0f0f0",
                "text_secondary": "#d0d0d0",
                "text_dim": "#a0a0a0",
                "text_shadow": "0 1px 3px rgba(0,0,0,0.7)",
                "scrim_color": "rgba(0,0,0,0.35)",
                "panel_bg": "#1a1a1a",
                "border": "#2f2f2f",
                "accent": "#ffcc66",
            },
            "scrim": {
                "position_css": "top: 0; right: 0;",
                "size_css": "width: 40%; height: 45%;",
                "gradient_css": "radial-gradient(ellipse at top right, rgba(0,0,0,0.4) 0%, transparent 70%)",
            },
            "recommended_stories": recommended_stories,
            "template_hint": "painting_text",
            "portrait_layout": None,
        }
    )


def _mock_layout_llm(*, recommended_stories: int) -> AsyncMock:
    """Create AsyncMock return value for layout_ai.invoke_vision."""
    return AsyncMock(
        return_value=LlmResponse(
            text=_layout_json(recommended_stories=recommended_stories),
            model="test-model",
            cli="gemini",
            usage=None,
        )
    )


@contextmanager
def _rss_server(rss_body: str) -> Iterator[str]:
    """Serve a deterministic RSS document on localhost for tests."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml; charset=utf-8")
            self.end_headers()
            self.wfile.write(rss_body.encode("utf-8"))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/rss.xml"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.asyncio
async def test_run_once_happy_path(tmp_path: Path) -> None:
    """Full run_once pipeline produces a PNG with real config/state/render."""
    config = _write_config(tmp_path, "feeds = []")
    _create_cached_painting(config.paintings.cache_dir)

    state = AppState.load(config.data_dir / "state")
    assert isinstance(state.news_queue, NewsQueue)
    assert isinstance(state.used_paintings, UsedPaintings)
    assert isinstance(state.layout_cache, LayoutCache)

    stories = [_make_story(1), _make_story(2)]
    state.news_queue.enqueue(
        CurationResult(
            stories=stories,
            tone_description="Steady and optimistic",
            curated_at=datetime(2026, 3, 15, 8, 0, tzinfo=timezone.utc),
            feed_count=1,
            entry_count=2,
        ),
        batch_size=4,
    )

    with patch(
        "sfumato.layout_ai.invoke_vision", _mock_layout_llm(recommended_stories=3)
    ):
        result = await run_once(
            config=config,
            state=state,
            options=RunOptions(no_upload=True, no_news=False),
        )

    assert result.render_result is not None
    assert result.render_result.png_path.exists()
    assert result.render_result.png_path.suffix == ".png"
    assert result.story_count == 2
    assert result.painting is not None
    assert state.used_paintings.is_used(result.painting.content_hash)
    assert state.layout_cache.has(result.painting.content_hash)


@pytest.mark.asyncio
async def test_run_once_empty_queue(tmp_path: Path) -> None:
    """run_once handles an empty queue by rendering without stories."""
    config = _write_config(tmp_path, "feeds = []")
    _create_cached_painting(config.paintings.cache_dir)
    state = AppState.load(config.data_dir / "state")

    with patch(
        "sfumato.layout_ai.invoke_vision", _mock_layout_llm(recommended_stories=3)
    ):
        result = await run_once(
            config=config,
            state=state,
            options=RunOptions(no_upload=True, no_news=False),
        )

    assert result.render_result is not None
    assert result.render_result.png_path.exists()
    assert result.story_count == 0
    assert state.news_queue.size == 0


@pytest.mark.asyncio
async def test_run_once_respects_recommended_stories(tmp_path: Path) -> None:
    """run_once limits rendered stories to layout.recommended_stories."""
    config = _write_config(tmp_path, "feeds = []")
    _create_cached_painting(config.paintings.cache_dir)
    state = AppState.load(config.data_dir / "state")
    stories = [_make_story(1), _make_story(2), _make_story(3)]
    state.news_queue.enqueue(
        CurationResult(
            stories=stories,
            tone_description="Urgent but balanced",
            curated_at=datetime(2026, 3, 15, 8, 5, tzinfo=timezone.utc),
            feed_count=1,
            entry_count=3,
        ),
        batch_size=4,
    )

    with patch(
        "sfumato.layout_ai.invoke_vision", _mock_layout_llm(recommended_stories=1)
    ):
        result = await run_once(
            config=config,
            state=state,
            options=RunOptions(no_upload=True, no_news=False),
        )

    assert result.render_result is not None
    assert result.story_count == 1
    assert result.render_result.story_count == 1
    html_content = result.render_result.html_path.read_text(encoding="utf-8")
    assert "Headline 1" in html_content
    assert "Headline 2" not in html_content


@pytest.mark.asyncio
async def test_enqueue_dequeue_roundtrip(tmp_path: Path) -> None:
    """run_news_refresh enqueue path round-trips through real NewsQueue."""
    rss = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<rss version=\"2.0\"><channel>
  <title>Local Feed</title>
  <item><title>Raw A</title><link>https://example.test/story-a</link><description>Alpha</description><pubDate>Sun, 15 Mar 2026 08:00:00 GMT</pubDate></item>
  <item><title>Raw B</title><link>https://example.test/story-b</link><description>Beta</description><pubDate>Sun, 15 Mar 2026 08:01:00 GMT</pubDate></item>
  <item><title>Raw C</title><link>https://example.test/story-c</link><description>Gamma</description><pubDate>Sun, 15 Mar 2026 08:02:00 GMT</pubDate></item>
</channel></rss>
"""
    with _rss_server(rss) as feed_url:
        config = _write_config(
            tmp_path,
            "\n".join(
                [
                    "",
                    "[[news.feeds]]",
                    'name = "Local"',
                    f'url = "{feed_url}"',
                    'category = "Tech"',
                ]
            ),
        )
        state = AppState.load(config.data_dir / "state")

        curation_json = json.dumps(
            {
                "stories": [
                    {
                        "headline": "Curated A",
                        "summary": "Summary A",
                        "source": "Local Feed",
                        "category": "Tech",
                        "url": "https://example.test/story-a",
                        "published_at": "2026-03-15T08:00:00+00:00",
                        "featured": True,
                    },
                    {
                        "headline": "Curated B",
                        "summary": "Summary B",
                        "source": "Local Feed",
                        "category": "Tech",
                        "url": "https://example.test/story-b",
                        "published_at": "2026-03-15T08:01:00+00:00",
                        "featured": False,
                    },
                    {
                        "headline": "Curated C",
                        "summary": "Summary C",
                        "source": "Local Feed",
                        "category": "Tech",
                        "url": "https://example.test/story-c",
                        "published_at": "2026-03-15T08:02:00+00:00",
                        "featured": False,
                    },
                ],
                "tone_description": "Measured and practical",
                "feed_count": 1,
                "entry_count": 3,
            }
        )

        with patch(
            "sfumato.news.invoke_text",
            AsyncMock(
                return_value=LlmResponse(
                    text=curation_json,
                    model="test-model",
                    cli="gemini",
                    usage=None,
                )
            ),
        ):
            enqueued = await run_news_refresh(config=config, state=state)

    assert enqueued == 1
    dequeued = state.news_queue.dequeue()
    assert dequeued is not None
    assert [story.headline for story in dequeued.stories] == [
        "Curated A",
        "Curated B",
        "Curated C",
    ]
    assert dequeued.tone_description == "Measured and practical"
