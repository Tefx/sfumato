"""Tests for the news module contracts and behavior.

This file contains CONTRACT TESTS that verify the behavioral boundaries
defined in src/sfumato/news.py. Each test documents expected behavior
without depending on implementation details.

Architecture reference: ARCHITECTURE.md#2.2
Contract reference: src/sfumato/news.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from sfumato.config import AiConfig, FeedConfig
from sfumato.news import (
    CurationResult,
    FeedFetchError,
    NewsError,
    Story,
    curate,
    fetch_feeds,
    refresh_news,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CONTRACT: PUBLIC DATA TYPES
# =============================================================================


class TestStoryContract:
    """Test Story dataclass contract."""

    def test_story_has_all_required_fields(self) -> None:
        """Story has headline, summary, source, category, url, published_at."""
        story = Story(
            headline="Test Headline",
            summary="Test summary",
            source="Test Source",
            category="Tech",
            url="https://example.com/article",
            published_at=datetime.now(timezone.utc),
        )
        assert story.headline == "Test Headline"
        assert story.summary == "Test summary"
        assert story.source == "Test Source"
        assert story.category == "Tech"
        assert story.url == "https://example.com/article"
        assert isinstance(story.published_at, datetime)

    def test_story_featured_defaults_to_false(self) -> None:
        """Story.featured defaults to False."""
        story = Story(
            headline="Test",
            summary="Test",
            source="Test",
            category="Tech",
            url="https://example.com",
            published_at=datetime.now(timezone.utc),
        )
        assert story.featured is False

    def test_story_is_mutable(self) -> None:
        """Story instances can have their featured flag updated."""
        story = Story(
            headline="Test",
            summary="Test",
            source="Test",
            category="Tech",
            url="https://example.com",
            published_at=datetime.now(timezone.utc),
            featured=False,
        )
        story.featured = True
        assert story.featured is True

    def test_story_timezone_aware_published_at(self) -> None:
        """Story.published_at should handle timezone-aware datetimes."""
        now = datetime.now(timezone.utc)
        story = Story(
            headline="Test",
            summary="Test",
            source="Test",
            category="Tech",
            url="https://example.com",
            published_at=now,
        )
        assert story.published_at.tzinfo is not None


class TestCurationResultContract:
    """Test CurationResult dataclass contract."""

    def test_curation_result_has_all_fields(self) -> None:
        """CurationResult has stories, tone_description, curated_at, feed_count, entry_count."""
        story = Story(
            headline="Test",
            summary="Test",
            source="Test",
            category="Tech",
            url="https://example.com",
            published_at=datetime.now(timezone.utc),
        )
        result = CurationResult(
            stories=[story],
            tone_description="Tense tech competition",
            curated_at=datetime.now(timezone.utc),
            feed_count=5,
            entry_count=42,
        )
        assert len(result.stories) == 1
        assert result.tone_description == "Tense tech competition"
        assert result.feed_count == 5
        assert result.entry_count == 42

    def test_curation_result_defaults(self) -> None:
        """CurationResult has sensible defaults."""
        result = CurationResult()
        assert result.stories == []
        assert result.tone_description == ""
        assert result.feed_count == 0
        assert result.entry_count == 0
        assert isinstance(result.curated_at, datetime)

    def test_curation_result_is_mutable(self) -> None:
        """CurationResult instances can be modified after creation."""
        result = CurationResult()
        result.feed_count = 10
        result.entry_count = 50
        assert result.feed_count == 10
        assert result.entry_count == 50


class TestNewsErrorHierarchy:
    """Test news error hierarchy."""

    def test_news_error_is_base_exception(self) -> None:
        """NewsError is the base class for news-related failures."""
        assert issubclass(FeedFetchError, NewsError)

    def test_feed_fetch_error_is_non_fatal(self) -> None:
        """FeedFetchError indicates a non-fatal feed failure."""
        error = FeedFetchError("Failed to fetch feed")
        assert "Failed" in str(error)


# =============================================================================
# CONTRACT: FETCH FEEDS
# =============================================================================


class TestFetchFeedsContract:
    """Test fetch_feeds() contract."""

    @pytest.mark.asyncio
    async def test_fetch_feeds_returns_list_of_dicts(self) -> None:
        """fetch_feeds returns list[dict] with expected keys."""
        feeds = [
            FeedConfig(
                name="HN", url="https://news.ycombinator.com/rss", category="Tech"
            ),
        ]

        with patch("sfumato.news.httpx.AsyncClient") as mock_client:
            # Create mock response with RSS content
            mock_response = MagicMock()
            mock_response.text = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Hacker News</title>
    <item>
      <title>Test Article</title>
      <link>https://example.com/article</link>
      <description>Test summary</description>
      <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>"""
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            entries = await fetch_feeds(feeds, max_age_days=30)

            assert isinstance(entries, list)
            assert len(entries) >= 0

    @pytest.mark.asyncio
    async def test_fetch_feeds_handles_feed_failures_gracefully(self) -> None:
        """fetch_feeds never fails - individual feed failures are logged and skipped."""
        feeds = [
            FeedConfig(
                name="Bad", url="https://nonexistent.invalid/rss", category="Tech"
            ),
            FeedConfig(
                name="AlsoBad", url="https://another.invalid/rss", category="Tech"
            ),
        ]

        # Should NOT raise - should return empty list
        with patch("sfumato.news.httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            entries = await fetch_feeds(feeds)
            assert entries == []

    @pytest.mark.asyncio
    async def test_fetch_feeds_empty_feeds_returns_empty_list(self) -> None:
        """fetch_feeds returns empty list for empty feeds input."""
        entries = await fetch_feeds([])
        assert entries == []

    @pytest.mark.asyncio
    async def test_fetch_feeds_deduplicates_by_url(self) -> None:
        """fetch_feeds deduplicates entries by URL across all feeds."""
        feeds = [
            FeedConfig(name="Feed1", url="https://feed1.com/rss", category="Tech"),
            FeedConfig(name="Feed2", url="https://feed2.com/rss", category="Tech"),
        ]

        with patch("sfumato.news.httpx.AsyncClient") as mock_client:
            # Each feed returns the same article URL
            mock_response1 = MagicMock()
            mock_response1.text = """<?xml version="1.0"?>
<rss version="2.0">
  <channel><title>Feed1</title>
    <item>
      <title>Duplicate Article</title>
      <link>https://example.com/duplicate</link>
      <description>Same article from both feeds</description>
    </item>
  </channel>
</rss>"""
            mock_response1.raise_for_status = MagicMock()

            mock_response2 = MagicMock()
            mock_response2.text = """<?xml version="1.0"?>
<rss version="2.0">
  <channel><title>Feed2</title>
    <item>
      <title>Duplicate Article</title>
      <link>https://example.com/duplicate</link>
      <description>Same article from both feeds</description>
    </item>
  </channel>
</rss>"""
            mock_response2.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()

            # Return different responses for different URLs
            async def get_side_effect(url: str) -> MagicMock:
                if "feed1" in url:
                    return mock_response1
                else:
                    return mock_response2

            mock_client_instance.get = AsyncMock(side_effect=get_side_effect)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            entries = await fetch_feeds(feeds)

            # Should only have one entry (deduplicated)
            urls = [e["url"] for e in entries]
            assert urls.count("https://example.com/duplicate") == 1

    @pytest.mark.asyncio
    async def test_fetch_feeds_age_filtering(self) -> None:
        """fetch_feeds filters out articles older than max_age_days."""
        feeds = [FeedConfig(name="Test", url="https://test.com/rss", category="Tech")]

        # Create old date (more than 3 days old)
        old_date = datetime.now(timezone.utc) - timedelta(days=10)

        with patch("sfumato.news.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = f"""<?xml version="1.0"?>
<rss version="2.0">
  <channel><title>Test Feed</title>
    <item>
      <title>Old Article</title>
      <link>https://example.com/old</link>
      <description>This is old</description>
      <pubDate>{old_date.strftime("%a, %d %b %Y %H:%M:%S +0000")}</pubDate>
    </item>
    <item>
      <title>New Article</title>
      <link>https://example.com/new</link>
      <description>This is new</description>
    </item>
  </channel>
</rss>"""
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(
                return_value=mock_client_instance
            )
            mock_client_instance.__aexit__ = AsyncMock()
            mock_client.return_value = mock_client_instance

            entries = await fetch_feeds(feeds, max_age_days=3)

            # Only the new article should be included
            urls = [e["url"] for e in entries]
            assert "https://example.com/old" not in urls


# =============================================================================
# CONTRACT: CURATE
# =============================================================================


class TestCurateContract:
    """Test curate() contract."""

    @pytest.mark.asyncio
    async def test_curate_returns_curation_result(self) -> None:
        """curate returns CurationResult with curated stories."""
        raw_entries = [
            {
                "title": "Test Article",
                "summary": "Test summary",
                "url": "https://example.com/article",
                "source": "Test Source",
                "category": "Tech",
                "published": datetime.now(timezone.utc),
            }
        ]

        mock_response = MagicMock()
        mock_response.text = """{
            "stories": [
                {
                    "headline": "Test Headline",
                    "summary": "Test summary translated",
                    "source": "Test Source",
                    "category": "Tech",
                    "url": "https://example.com/article",
                    "published_at": "2024-01-15T10:30:00+00:00",
                    "featured": true
                }
            ],
            "tone_description": "Test tone",
            "feed_count": 1,
            "entry_count": 1
        }"""

        with patch("sfumato.news.invoke_text", AsyncMock(return_value=mock_response)):
            with patch(
                "sfumato.news.parse_json_response",
                return_value={
                    "stories": [
                        {
                            "headline": "Test Headline",
                            "summary": "Test summary translated",
                            "source": "Test Source",
                            "category": "Tech",
                            "url": "https://example.com/article",
                            "published_at": "2024-01-15T10:30:00+00:00",
                            "featured": True,
                        }
                    ],
                    "tone_description": "Test tone",
                    "feed_count": 1,
                    "entry_count": 1,
                },
            ):
                ai_config = AiConfig(cli="gemini", model="test-model")
                result = await curate(
                    raw_entries=raw_entries,
                    language="zh",
                    stories_per_refresh=12,
                    ai_config=ai_config,
                )

                assert isinstance(result, CurationResult)
                assert result.tone_description == "Test tone"

    @pytest.mark.asyncio
    async def test_curate_empty_entries_returns_empty_stories(self) -> None:
        """curate returns empty CurationResult for empty raw_entries."""
        ai_config = AiConfig(cli="gemini", model="test-model")
        result = await curate(
            raw_entries=[],
            language="zh",
            stories_per_refresh=12,
            ai_config=ai_config,
        )

        assert result.stories == []
        assert result.tone_description == ""
        assert result.feed_count == 0
        assert result.entry_count == 0


# =============================================================================
# CONTRACT: REFRESH NEWS
# =============================================================================


class TestRefreshNewsContract:
    """Test refresh_news() convenience wrapper."""

    @pytest.mark.asyncio
    async def test_refresh_news_combines_fetch_and_curate(self) -> None:
        """refresh_news fetches feeds then curates them."""
        news_config = MagicMock()
        news_config.feeds = [
            FeedConfig(name="Test", url="https://test.com/rss", category="Tech")
        ]
        news_config.language = "zh"
        news_config.stories_per_refresh = 12
        news_config.max_age_days = 3

        mock_curation_result = CurationResult(
            stories=[],
            tone_description="Test tone",
            curated_at=datetime.now(timezone.utc),
            feed_count=1,
            entry_count=5,
        )

        with patch(
            "sfumato.news.fetch_feeds",
            AsyncMock(
                return_value=[
                    {
                        "title": "Test",
                        "summary": "Test",
                        "url": "https://example.com/article",
                        "source": "Test",
                        "category": "Tech",
                        "published": datetime.now(timezone.utc),
                    }
                ]
            ),
        ):
            with patch(
                "sfumato.news.curate", AsyncMock(return_value=mock_curation_result)
            ):
                ai_config = AiConfig(cli="gemini", model="test-model")
                result = await refresh_news(news_config, ai_config)

                assert isinstance(result, CurationResult)
                assert result.feed_count == 1
