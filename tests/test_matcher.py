"""Tests for matcher.py LLM-based painting selection.

This test module covers:
- Random strategy (returns score=0.0, ignores tone)
- Semantic strategy (LLM-based matching via invoke_text)
- Error/fallback semantics (empty paintings, LLM failures)
- Backward compatibility (**kwargs absorbs legacy embedding_cache)

These tests are CONTRACT-DRIVEN: they verify expected behavior of the
LLM-based matcher that replaced the embedding-based matcher.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from sfumato.matcher import (
    MatcherError,
    select_painting,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# LOCAL TEST DOUBLES (avoid external module dependencies)
# =============================================================================


class Orientation(Enum):
    """Test double for paintings.Orientation."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class ArtSource(Enum):
    """Test double for paintings.ArtSource."""

    MET = "met"
    WIKIMEDIA = "wikimedia"


@dataclass(frozen=True)
class MockPaintingInfo:
    """Test double for PaintingInfo - minimal fields for matcher tests."""

    image_path: Path
    content_hash: str
    title: str
    artist: str
    year: str
    source: ArtSource
    source_id: str
    source_url: str
    orientation: Orientation
    width: int
    height: int


@dataclass(frozen=True)
class MockAiConfig:
    """Test double for AiConfig."""

    cli: str = "gemini"
    model: str = "test-model"
    backend: str = "sdk"
    sdk_provider: str = "openrouter"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def make_painting(content_hash: str) -> MockPaintingInfo:
    """Create a test painting with minimal required fields."""
    return MockPaintingInfo(
        image_path=Path(f"/tmp/{content_hash}.jpg"),
        content_hash=content_hash,
        title=f"Painting {content_hash}",
        artist="Test Artist",
        year="2024",
        source=ArtSource.WIKIMEDIA,
        source_id=content_hash,
        source_url=f"https://example.com/{content_hash}",
        orientation=Orientation.LANDSCAPE,
        width=100,
        height=100,
    )


def make_ai_config() -> MockAiConfig:
    """Create a test AiConfig."""
    return MockAiConfig()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def ai_config() -> MockAiConfig:
    """Return a test AiConfig."""
    return MockAiConfig()


@pytest.fixture
def sample_paintings_list() -> list[MockPaintingInfo]:
    """Return a list of sample paintings."""
    return [
        make_painting("hash_1"),
        make_painting("hash_2"),
        make_painting("hash_3"),
    ]


# =============================================================================
# RANDOM STRATEGY TESTS
# =============================================================================


class TestSelectPaintingRandomStrategy:
    """Tests for random strategy behavior per contract.

    Strategy "random" MUST:
    - Ignore news_tone entirely
    - Select uniformly at random from paintings
    - Return score = 0.0 EXACTLY
    - No LLM invocation
    """

    @pytest.mark.asyncio
    async def test_random_returns_score_zero(self, ai_config: MockAiConfig) -> None:
        """Random strategy must return score exactly 0.0."""
        paintings = [
            make_painting("hash_a"),
            make_painting("hash_b"),
            make_painting("hash_c"),
        ]

        painting, score = await select_painting(
            news_tone="energetic tech tone",
            paintings=paintings,
            painting_descriptions={
                "hash_a": "desc",
                "hash_b": "desc",
                "hash_c": "desc",
            },
            ai_config=ai_config,
            strategy="random",
        )

        assert score == 0.0

    @pytest.mark.asyncio
    async def test_random_ignores_news_tone(self, ai_config: MockAiConfig) -> None:
        """Random strategy must ignore news_tone parameter entirely."""
        paintings = [make_painting(f"hash_{i}") for i in range(5)]

        for tone in ["energetic", "somber", "mysterious", "chaotic"]:
            painting, score = await select_painting(
                news_tone=tone,
                paintings=paintings,
                painting_descriptions={f"hash_{i}": f"desc_{i}" for i in range(5)},
                ai_config=ai_config,
                strategy="random",
            )
            assert score == 0.0
            assert painting.content_hash.startswith("hash_")

    @pytest.mark.asyncio
    async def test_random_with_single_painting_returns_that_painting(
        self, ai_config: MockAiConfig
    ) -> None:
        """Random with one painting must return that painting."""
        painting = make_painting("only_one")

        result, score = await select_painting(
            news_tone="any",
            paintings=[painting],
            painting_descriptions={"only_one": "the only one"},
            ai_config=ai_config,
            strategy="random",
        )

        assert result.content_hash == "only_one"
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_random_accepts_legacy_embedding_cache_kwarg(
        self, ai_config: MockAiConfig
    ) -> None:
        """Random strategy must accept embedding_cache kwarg for backward compat."""
        paintings = [make_painting("hash_a")]

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={"hash_a": "desc"},
            embedding_cache={},
            ai_config=ai_config,
            strategy="random",
        )

        assert score == 0.0


# =============================================================================
# SEMANTIC STRATEGY TESTS (LLM-based)
# =============================================================================


class TestSelectPaintingSemanticStrategy:
    """Tests for semantic (LLM-based) strategy behavior per contract.

    Strategy "semantic" MUST:
    - Invoke LLM to compare paintings with news tone
    - Return painting selected by LLM
    - Return score 1.0 for LLM-matched selection
    - Fall back to random (score 0.0) on LLM failure
    """

    @pytest.mark.asyncio
    async def test_semantic_returns_llm_selected_painting(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic must return painting selected by LLM."""
        paintings = [
            make_painting("hash_1"),
            make_painting("hash_2"),
            make_painting("hash_3"),
        ]

        mock_response = AsyncMock()
        mock_response.return_value = AsyncMock(
            text='{"selected_index": 1, "reason": "test"}'
        )

        with patch("sfumato.matcher.invoke_text", mock_response):
            painting, score = await select_painting(
                news_tone="energetic tech",
                paintings=paintings,
                painting_descriptions={
                    "hash_1": "energetic scene",
                    "hash_2": "calm landscape",
                    "hash_3": "abstract art",
                },
                ai_config=ai_config,
                strategy="semantic",
            )

        assert painting.content_hash == "hash_2"
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_semantic_falls_back_to_random_on_llm_error(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic falls back to random if LLM fails."""
        from sfumato.llm import LlmError

        paintings = [make_painting("hash_1"), make_painting("hash_2")]

        mock_response = AsyncMock(side_effect=LlmError("API error"))

        with patch("sfumato.matcher.invoke_text", mock_response):
            painting, score = await select_painting(
                news_tone="test tone",
                paintings=paintings,
                painting_descriptions={
                    "hash_1": "desc1",
                    "hash_2": "desc2",
                },
                ai_config=ai_config,
                strategy="semantic",
            )

        assert painting in paintings
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_semantic_falls_back_when_no_descriptions(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic falls back to random when no descriptions available."""
        paintings = [make_painting("hash_1"), make_painting("hash_2")]

        painting, score = await select_painting(
            news_tone="test tone",
            paintings=paintings,
            painting_descriptions={},
            ai_config=ai_config,
            strategy="semantic",
        )

        assert painting in paintings
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_semantic_handles_out_of_range_index(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic handles out-of-range index from LLM gracefully."""
        paintings = [make_painting("hash_1"), make_painting("hash_2")]

        mock_response = AsyncMock()
        mock_response.return_value = AsyncMock(
            text='{"selected_index": 99, "reason": "test"}'
        )

        with patch("sfumato.matcher.invoke_text", mock_response):
            painting, score = await select_painting(
                news_tone="test",
                paintings=paintings,
                painting_descriptions={
                    "hash_1": "desc1",
                    "hash_2": "desc2",
                },
                ai_config=ai_config,
                strategy="semantic",
            )

        # Should fall back to first candidate
        assert painting.content_hash == "hash_1"
        assert score == 1.0


# =============================================================================
# ERROR/FALLBACK SEMANTICS TESTS
# =============================================================================


class TestSelectPaintingErrorSemantics:
    """Tests for error and fallback behavior per contract."""

    @pytest.mark.asyncio
    async def test_empty_paintings_raises_matcher_error(
        self, ai_config: MockAiConfig
    ) -> None:
        """Empty paintings list must raise MatcherError."""
        with pytest.raises(MatcherError, match="empty|no paintings|cannot select"):
            await select_painting(
                news_tone="test",
                paintings=[],
                painting_descriptions={},
                ai_config=ai_config,
                strategy="semantic",
            )

    @pytest.mark.asyncio
    async def test_empty_paintings_raises_for_random(
        self, ai_config: MockAiConfig
    ) -> None:
        """Empty paintings list must raise MatcherError even for random strategy."""
        with pytest.raises(MatcherError, match="empty|no paintings|cannot select"):
            await select_painting(
                news_tone="test",
                paintings=[],
                painting_descriptions={},
                ai_config=ai_config,
                strategy="random",
            )
