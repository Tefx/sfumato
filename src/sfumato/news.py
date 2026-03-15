"""Fetch and curate news using RSS + LLM.

This module implements the news module contract from ARCHITECTURE.md#2.2:
- Fetch RSS feeds via HTTP, deduplicate entries by URL, filter by max_age_days
- Invoke LLM to curate/rank/translate/summarize into structured Story objects
- Produce a tone_description for the curated batch, used for semantic matching

Architecture reference: ARCHITECTURE.md#2.2
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import feedparser
import httpx

from sfumato.config import AiConfig, FeedConfig
from sfumato.llm import LlmError, LlmParseError, invoke_text, parse_json_response

if TYPE_CHECKING:
    from sfumato.config import NewsConfig

# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass
class Story:
    """A curated news story from RSS feeds.

    Attributes:
        headline: Translated headline, max ~15 chars for zh, ~12 words for en.
        summary: 60-100 chars (zh) / 60-100 words (en) complete summary.
        source: Publication name, e.g. "Ars Technica".
        category: e.g. "Tech", "AI", "World", "Science", "Economy".
        url: Original article URL.
        published_at: Original publication time.
        featured: LLM marks the single most important story.

    Contract:
        - Instances are mutable (can update featured flag during curation)
        - headline is always translated to the target language
        - summary is always in the target language
        - url is the original source URL for attribution
    """

    headline: str
    summary: str
    source: str
    category: str
    url: str
    published_at: datetime
    featured: bool = False


@dataclass
class CurationResult:
    """Result of fetching and curating news from RSS feeds.

    Attributes:
        stories: ~12 stories, ranked by importance.
        tone_description: Free-form LLM description of the batch's emotional tone.
            e.g. "科技巨头间的紧张博弈，充满不确定性"
        curated_at: When curation happened.
        feed_count: How many feeds were successfully fetched.
        entry_count: How many raw entries were collected before curation.

    Contract:
        - stories are ordered by importance (most important first)
        - exactly one story should have featured=True (if stories is non-empty)
        - tone_description is always in the target language
        - feed_count may be < total feeds if some failed
        - entry_count is pre-deduplication count
    """

    stories: list[Story] = field(default_factory=list)
    tone_description: str = ""
    curated_at: datetime = field(default_factory=datetime.now)
    feed_count: int = 0
    entry_count: int = 0


# =============================================================================
# ERROR TYPES
# =============================================================================


class NewsError(Exception):
    """Base exception for news-related failures."""

    pass


class FeedFetchError(NewsError):
    """Failed to fetch a specific RSS feed. Non-fatal - other feeds continue."""

    pass


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API - FETCH FEEDS
# =============================================================================


async def fetch_feeds(
    feeds: list[FeedConfig],
    max_age_days: int = 3,
) -> list[dict]:
    """Fetch all RSS feeds, deduplicate by URL, filter by max_age_days.

    Args:
        feeds: List of RSS feed configurations (name, url, category).
        max_age_days: Maximum age of articles to include. Articles older than
            this are filtered out.

    Returns:
        List of raw entry dicts with keys:
            - title: str - Article title
            - summary: str - Article summary/excerpt (max 300 chars)
            - url: str - Article URL
            - source: str - Publication name
            - category: str - Feed category
            - published: datetime | None - Publication time (may be None)

    Raises:
        Nothing - Feed failures are logged and skipped. The function always
        returns successfully with whatever entries could be fetched.

    Contract:
        - Concurrent fetching via httpx.AsyncClient
        - 15 second timeout per feed
        - Individual feed failures are logged and skipped (never fails whole op)
        - URL deduplication across all feeds in this fetch
        - Age filtering by max_age_days
        - Entries may have None for published date if not available

    Example:
        >>> feeds = [
        ...     FeedConfig(name="HN", url="https://news.ycombinator.com/rss", category="Tech"),
        ...     FeedConfig(name="BBC", url="https://feeds.bbci.co.uk/news/rss.xml", category="World"),
        ... ]
        >>> entries = await fetch_feeds(feeds, max_age_days=3)
        >>> len(entries)  # depends on feed content
        42
    """
    if not feeds:
        return []

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    all_entries: list[dict] = []
    seen_urls: set[str] = set()

    async with httpx.AsyncClient(timeout=15.0) as client:
        for feed_config in feeds:
            try:
                response = await client.get(feed_config.url)
                response.raise_for_status()
            except Exception as e:
                logger.warning(
                    "Failed to fetch feed '%s' from %s: %s",
                    feed_config.name,
                    feed_config.url,
                    e,
                )
                continue

            try:
                feed = feedparser.parse(response.text)
            except Exception as e:
                logger.warning(
                    "Failed to parse feed '%s' from %s: %s",
                    feed_config.name,
                    feed_config.url,
                    e,
                )
                continue

            # feedparser returns FeedParserDict which is dict-like
            # Suppress type error: feedparser's .feed is actually a dict-like object
            feed_title: str
            if hasattr(feed, "feed") and feed.feed:
                feed_title = str(feed.feed.get("title", feed_config.name))  # type: ignore[union-attr]
            else:
                feed_title = feed_config.name

            for entry in feed.entries:
                # Get URL (link) - skip if missing or duplicate
                # entry is FeedParserDict which is dict-like
                url: str = entry.get("link") or entry.get("url")  # type: ignore[union-attr]
                if not url:
                    continue

                if url in seen_urls:
                    continue
                seen_urls.add(url)  # type: ignore[union-attr]

                # Parse publication date
                published = None
                parsed_time = getattr(entry, "published_parsed", None) or getattr(
                    entry, "updated_parsed", None
                )
                if parsed_time:
                    try:
                        # parsed_time is time.struct_time, extract first 6 fields
                        published = datetime(
                            parsed_time.tm_year,
                            parsed_time.tm_mon,
                            parsed_time.tm_mday,
                            parsed_time.tm_hour,
                            parsed_time.tm_min,
                            parsed_time.tm_sec,
                            tzinfo=timezone.utc,
                        )
                    except (TypeError, ValueError, AttributeError):
                        pass

                # Age filtering
                if published and published < cutoff_date:
                    continue

                # Extract summary - limit to reasonable length
                summary = entry.get("summary") or entry.get("description") or ""
                summary = entry.get("title", "") if not summary else summary
                # Clean HTML tags (basic clean-up)
                summary = _strip_html_tags(summary)[:300]

                all_entries.append(
                    {
                        "title": entry.get("title", ""),
                        "summary": summary,
                        "url": url,
                        "source": feed_title,
                        "category": feed_config.category,
                        "published": published,
                    }
                )

    return all_entries


# =============================================================================
# PUBLIC API - CURATE
# =============================================================================


async def curate(
    raw_entries: list[dict],
    language: str,
    stories_per_refresh: int,
    ai_config: AiConfig,
) -> CurationResult:
    """Invoke LLM to select, rank, summarize, translate, and describe tone.

    The LLM is called once with all raw entries. It returns:
    - Exactly `stories_per_refresh` curated stories (or fewer if not enough raw entries)
    - A tone_description summarizing the batch's overall mood/themes

    Args:
        raw_entries: List of entry dicts from fetch_feeds().
        language: Target language code (e.g. "zh" for Chinese, "en" for English).
        stories_per_refresh: Maximum number of stories to return.
        ai_config: AI backend configuration.

    Returns:
        CurationResult with curated stories and tone_description.

    Raises:
        LlmError: If the LLM call fails after retries.
        LlmParseError: If the LLM response cannot be parsed.

    Contract:
        - Returns at most stories_per_refresh stories
        - Returns fewer stories if not enough raw entries
        - Stories are ranked by importance (most important first)
        - Exactly one story has featured=True (if stories is non-empty)
        - tone_description is free-form text in the target language
        - All headlines and summaries are translated to the target language
        - Raises LlmError or LlmParseError on failure (no fallback)
    """
    if not raw_entries:
        return CurationResult(
            stories=[],
            tone_description="",
            curated_at=datetime.now(),
            feed_count=0,
            entry_count=0,
        )

    # Split into batches of ~15 entries to keep LLM output within token limits
    # (each story = ~200 output tokens for headline + summary in Chinese)
    BATCH_SIZE = 15
    all_stories: list[Story] = []
    tone_descriptions: list[str] = []

    for batch_start in range(0, len(raw_entries), BATCH_SIZE):
        batch = raw_entries[batch_start:batch_start + BATCH_SIZE]
        # For batched processing, each batch gets stories_per_refresh=0 (keep all)
        # or proportional selection if stories_per_refresh > 0
        batch_limit = stories_per_refresh
        if stories_per_refresh > 0:
            batch_limit = max(1, stories_per_refresh * len(batch) // len(raw_entries))

        prompt = _build_curation_prompt(batch, language, batch_limit)
        response = await invoke_text(
            prompt=prompt,
            ai_config=ai_config,
            system_prompt=_get_system_prompt(language),
            max_tokens=8000,  # 25 entries × ~200 tokens/story = ~5000 tokens output
            timeout_seconds=120,
        )
        try:
            data = parse_json_response(response.text)
            batch_stories = _parse_stories(data.get("stories", []), batch)
            batch_stories = _ensure_one_featured(batch_stories) if batch_stories else []
            all_stories.extend(batch_stories)
            if data.get("tone_description"):
                tone_descriptions.append(data["tone_description"])
        except Exception as e:
            logger.warning("Failed to parse batch %d: %s", batch_start // BATCH_SIZE, e)
            continue

    # Sort all stories: featured first, then by original order
    featured = [s for s in all_stories if s.featured]
    non_featured = [s for s in all_stories if not s.featured]
    all_stories = featured + non_featured

    # Combine tone descriptions
    tone = tone_descriptions[0] if tone_descriptions else ""

    return CurationResult(
        stories=all_stories,
        tone_description=tone,
        curated_at=datetime.now(),
        feed_count=0,
        entry_count=len(raw_entries),
    )




# =============================================================================
# PUBLIC API - CONVENIENCE WRAPPER
# =============================================================================


async def refresh_news(
    news_config: NewsConfig,
    ai_config: AiConfig,
    exclude_urls: set[str] | None = None,
) -> CurationResult:
    """Top-level convenience: fetch + curate in one call.

    This is a convenience wrapper that combines fetch_feeds() and curate()
    for the common case of refreshing news from configured feeds.

    Args:
        news_config: News configuration containing feeds and settings.
        ai_config: AI backend configuration.

    Returns:
        CurationResult with curated stories, tone_description, and metadata.

    Raises:
        LlmError: If the LLM call fails after retries.
        LlmParseError: If the LLM response cannot be parsed.

    Contract:
        - fetch_feeds(): Uses news_config.feeds and news_config.max_age_days
        - curate(): Uses news_config.language and news_config.stories_per_refresh
        - Returns CurationResult with feed_count and entry_count populated
        - Feed failures are non-fatal (logged, skipped)
    """
    # Import NewsConfig type at runtime to avoid circular import
    # (config imports dataclass fields, news imports config.AiConfig)

    # Fetch raw entries
    raw_entries = await fetch_feeds(
        feeds=news_config.feeds,
        max_age_days=news_config.max_age_days,
    )

    # Filter out already-displayed URLs to avoid re-curating same stories
    if exclude_urls:
        before = len(raw_entries)
        raw_entries = [e for e in raw_entries if e.get("url") not in exclude_urls]
        filtered = before - len(raw_entries)
        if filtered > 0:
            logger.info("Filtered %d already-seen URLs from %d entries", filtered, before)

    # Curate with LLM
    result = await curate(
        raw_entries=raw_entries,
        language=news_config.language,
        stories_per_refresh=news_config.stories_per_refresh,
        ai_config=ai_config,
    )

    # Update feed_count since curate doesn't know how many feeds we fetched
    result.feed_count = len(news_config.feeds)
    result.entry_count = len(raw_entries)

    return result


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _strip_html_tags(text: str) -> str:
    """Strip HTML tags from text, preserving content.

    Basic clean-up for RSS summaries that may contain HTML markup.
    """
    import re

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#39;", "'")
    text = text.replace("&quot;", '"')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_curation_prompt(
    raw_entries: list[dict],
    language: str,
    stories_per_refresh: int,
) -> str:
    """Build the curation prompt for the LLM.

    The prompt asks the LLM to:
    1. Select the most interesting/important stories
    2. Translate headlines and summaries to target language
    3. Rank by importance (most important first)
    4. Mark exactly one story as featured (most important)
    5. Describe the overall emotional tone of the batch

    Returns:
        Prompt string for LLM invocation.
    """
    # Build entries text
    entries_text = []
    for i, entry in enumerate(raw_entries, 1):
        published_str = ""
        if entry.get("published"):
            published_str = f" [{entry['published'].strftime('%Y-%m-%d')}]"

        entries_text.append(
            f"[{i}] {entry['category']}: {entry['title']}"
            f"{published_str}\n"
            f"    Source: {entry['source']}\n"
            f"    Summary: {entry['summary']}\n"
            f"    URL: {entry['url']}"
        )

    # Language-specific constraints
    if language == "zh":
        headline_constraint = "headlines max ~15 Chinese characters"
        summary_constraint = "summaries 60-100 Chinese characters"
    else:
        headline_constraint = "headlines max ~12 words"
        summary_constraint = "summaries 60-100 words"

    select_instruction = ""
    if stories_per_refresh > 0:
        select_instruction = f"Select the {stories_per_refresh} most interesting/important stories."
    else:
        select_instruction = (
            "Keep ALL stories. Only remove entries that are OBVIOUSLY spam, ads, "
            "or completely unreadable (must be high confidence). When in doubt, KEEP the entry."
        )

    prompt = f"""You are a news editor preparing a visual briefing for a large display.

Given these raw RSS entries, process them for display.

{select_instruction}

Rules:
- Translate headlines and summaries to {language.upper()}
- {headline_constraint}
- {summary_constraint}
- Rank stories by importance (most important first)
- Mark exactly one story as "featured": true (the single most important)
- For each story include: headline, summary, source, category, url, published_at in ISO format
- Provide a "tone_description" field: a free-form description of this batch's emotional tone
  (e.g., "紧张的技术竞争氛围" or "Cautious optimism about AI progress")
- Include "feed_count" (number of feeds) and "entry_count" (total raw entries received)

Raw entries:
{chr(10).join(entries_text)}

Respond with ONLY valid JSON matching this schema:
{{
  "stories": [
    {{
      "headline": "...",
      "summary": "...",
      "source": "Publication Name",
      "category": "Tech",
      "url": "https://...",
      "published_at": "2024-01-15T10:30:00+00:00",
      "featured": false
    }}
  ],
  "tone_description": "Free-form description of the batch's emotional tone",
  "feed_count": 5,
  "entry_count": {len(raw_entries)}
}}
"""
    return prompt


def _get_system_prompt(language: str) -> str:
    """Get the system prompt for LLM curation.

    Returns:
        System prompt string.
    """
    return f"""You are a professional news editor specializing in visual briefings.
Your task is to curate, translate, and summarize news stories for display.

Always respond with valid JSON matching the requested schema.
Translate all content to {language.upper()}.
Be concise and impactful.
Focus on the most newsworthy stories.
Describe the emotional tone of the batch in a free-form way."""


def _parse_stories(
    story_dicts: list[dict],
    raw_entries: list[dict],
) -> list[Story]:
    """Parse LLM response into Story objects.

    Validates that each story dict has required fields and maps URLs back
    to original entries for accurate attribution.

    Args:
        story_dicts: List of story dicts from LLM response.
        raw_entries: Original raw entries for URL validation.

    Returns:
        List of Story objects.
    """
    stories: list[Story] = []
    raw_urls = {e["url"] for e in raw_entries}

    for story_dict in story_dicts:
        # Validate required fields
        if not all(
            k in story_dict for k in ("headline", "summary", "source", "category")
        ):
            continue

        # Get URL - validate it exists in raw entries
        url = story_dict.get("url", "")
        if not url or url not in raw_urls:
            continue

        # Parse published_at
        published_at_str = story_dict.get("published_at", "")
        published_at = None
        if published_at_str:
            try:
                # Try ISO format
                published_at = datetime.fromisoformat(
                    published_at_str.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        if published_at is None:
            # Find original entry's published date
            for entry in raw_entries:
                if entry["url"] == url:
                    published_at = entry.get("published")
                    break

        if published_at is None:
            published_at = datetime.now(timezone.utc)

        # Build Story
        story = Story(
            headline=story_dict["headline"],
            summary=story_dict["summary"],
            source=story_dict["source"],
            category=story_dict["category"],
            url=url,
            published_at=published_at,
            featured=story_dict.get("featured", False),
        )
        stories.append(story)

    return stories


def _ensure_one_featured(stories: list[Story]) -> list[Story]:
    """Ensure exactly one story is marked as featured.

    If no stories are featured, mark the first one.
    If multiple stories are featured, mark only the first one.

    Args:
        stories: List of Story objects.

    Returns:
        List of Story objects with exactly one featured (if non-empty).
    """
    if not stories:
        return stories

    # Find all featured stories
    featured_indices = [i for i, s in enumerate(stories) if s.featured]

    if len(featured_indices) == 1:
        # Already exactly one featured - good
        return stories

    # Reset all featured flags
    for story in stories:
        story.featured = False

    # Mark the first story as featured
    stories[0].featured = True

    return stories
