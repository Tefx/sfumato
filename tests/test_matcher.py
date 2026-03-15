"""Comprehensive tests for matcher.py implementation contract.

This test module covers:
- Cosine similarity correctness including zero-vector semantics and range behavior
- Random strategy precedence (bypasses semantic scoring, returns score=0.0)
- Semantic strategy selection with highest-valid-similarity behavior
- Fallback/error semantics for empty paintings, missing embeddings, compute failures
- Cache key behavior for tone hashes and painting content-hash keyed embeddings
- Targeted regression tests for highest-risk failure paths

These tests are CONTRACT-DRIVEN: they define expected behavior that the
implementation MUST fulfill. The stubs currently raise NotImplementedError.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pytest

from sfumato.matcher import (
    MatcherError,
    EmbeddingResult,
    cosine_similarity,
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
    backend: str = "cli"
    sdk_provider: str = "openrouter"
    embedding_provider: str = "google"
    embedding_model: str = "text-embedding-004"


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


def make_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    """Create a deterministic embedding from text for testing."""
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    vec = np.random.randn(dimensions).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


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
def sample_painting() -> MockPaintingInfo:
    """Return a sample painting for testing."""
    return make_painting("hash_starry_night_001")


@pytest.fixture
def sample_paintings_list() -> list[MockPaintingInfo]:
    """Return a list of sample paintings."""
    return [
        make_painting("hash_1"),
        make_painting("hash_2"),
        make_painting("hash_3"),
    ]


# =============================================================================
# COSINE SIMILARITY TESTS
# =============================================================================


class TestCosineSimilarityCorrectness:
    """Tests for cosine_similarity correctness and mathematical properties."""

    def test_identical_vectors_returns_one(self) -> None:
        """Identical vectors should return similarity of 1.0."""
        vec = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        result = cosine_similarity(vec, vec.copy())
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors_returns_minus_one(self) -> None:
        """Opposite vectors (a = -b) should return similarity of -1.0."""
        vec = np.array([1.0, 0.0, 0.5, 0.25], dtype=np.float32)
        opposite = -vec
        result = cosine_similarity(vec, opposite)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_vectors_returns_zero(self) -> None:
        """Orthogonal vectors should return similarity near 0.0."""
        a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_45_degree_vectors(self) -> None:
        """Vectors at 45 degrees should return similarity of cos(45°) ≈ 0.707."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        # Normalize b for proper cosine similarity
        b = b / np.linalg.norm(b)
        result = cosine_similarity(a, b)
        # cos(45°) = 0.70710678...
        assert result == pytest.approx(0.70710678, abs=1e-5)

    def test_result_always_in_valid_range(self) -> None:
        """Result must always be in [-1.0, 1.0] for any valid input."""
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randn(384).astype(np.float32)
            b = np.random.randn(384).astype(np.float32)
            result = cosine_similarity(a, b)
            assert -1.0 <= result <= 1.0, f"Result {result} outside valid range"

    def test_different_dimensions_raises_error(self) -> None:
        """Vectors with different dimensions must raise ValueError."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # 3 dimensions
        b = np.array([1.0, 0.0], dtype=np.float32)  # 2 dimensions
        with pytest.raises(ValueError, match="dimension"):
            cosine_similarity(a, b)

    def test_non_1d_vectors_raise_error(self) -> None:
        """2D vectors must raise ValueError."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(a, b)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(b, a)


class TestCosineSimilarityZeroVectorSemantics:
    """Tests for zero-vector edge case handling per contract."""

    def test_both_zero_vectors_returns_zero(self) -> None:
        """When BOTH vectors are zero, return 0.0 (not undefined)."""
        a = np.zeros(384, dtype=np.float32)
        b = np.zeros(384, dtype=np.float32)
        result = cosine_similarity(a, b)
        assert result == 0.0

    def test_one_zero_one_nonzero_returns_zero(self) -> None:
        """When ONE vector is zero and other is non-zero, return 0.0."""
        zero = np.zeros(384, dtype=np.float32)
        nonzero = np.array([1.0, 0.5, 0.25] + [0.0] * 381, dtype=np.float32)

        # Zero vector first
        result = cosine_similarity(zero, nonzero)
        assert result == 0.0

        # Zero vector second
        result = cosine_similarity(nonzero, zero)
        assert result == 0.0

    def test_near_zero_vectors_not_treated_as_zero(self) -> None:
        """Very small but non-zero vectors should compute normally."""
        near_zero_a = np.ones(384, dtype=np.float32) * 1e-10
        near_zero_b = np.ones(384, dtype=np.float32) * 1e-10
        # These are identical, so similarity should be 1.0
        result = cosine_similarity(near_zero_a, near_zero_b)
        assert result == pytest.approx(1.0, abs=1e-5)


class TestCosineSimilarityNaNHandling:
    """Tests for NaN handling per contract."""

    def test_nan_in_first_vector_raises_error(self) -> None:
        """NaN in first vector must raise ValueError."""
        clean = np.ones(384, dtype=np.float32)
        with_nan = np.ones(384, dtype=np.float32)
        with_nan[0] = np.nan

        with pytest.raises(ValueError, match="NaN|nan"):
            cosine_similarity(with_nan, clean)

    def test_nan_in_second_vector_raises_error(self) -> None:
        """NaN in second vector must raise ValueError."""
        clean = np.ones(384, dtype=np.float32)
        with_nan = np.ones(384, dtype=np.float32)
        with_nan[50] = np.nan

        with pytest.raises(ValueError, match="NaN|nan"):
            cosine_similarity(clean, with_nan)

    def test_nan_in_both_vectors_raises_error(self) -> None:
        """NaN in both vectors must raise ValueError."""
        nan_a = np.ones(384, dtype=np.float32)
        nan_a[100] = np.nan
        nan_b = np.ones(384, dtype=np.float32)
        nan_b[200] = np.nan

        with pytest.raises(ValueError, match="NaN|nan"):
            cosine_similarity(nan_a, nan_b)


class TestCosineSimilarityFloatTypes:
    """Tests for float type handling."""

    def test_float32_vectors_work(self) -> None:
        """float32 vectors must work correctly."""
        a = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert isinstance(result, float)

    def test_float64_vectors_work(self) -> None:
        """float64 vectors must work correctly."""
        a = np.array([1.0, 0.0, 0.5], dtype=np.float64)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        result = cosine_similarity(a, b)
        assert isinstance(result, float)


# =============================================================================
# RANDOM STRATEGY TESTS
# =============================================================================


class TestSelectPaintingRandomStrategy:
    """Tests for random strategy behavior per contract.

    Strategy "random" MUST:
    - Ignore news_tone entirely
    - Select uniformly at random from paintings
    - Return score = 0.0 EXACTLY
    - No embedding computation or cache lookup
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
            embedding_cache={},
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
                embedding_cache={},
                ai_config=ai_config,
                strategy="random",
            )
            assert score == 0.0
            assert painting.content_hash.startswith("hash_")

    @pytest.mark.asyncio
    async def test_random_works_with_empty_embedding_cache(
        self, ai_config: MockAiConfig
    ) -> None:
        """Random strategy must work with empty embedding_cache."""
        paintings = [make_painting(f"hash_{i}") for i in range(3)]

        painting, score = await select_painting(
            news_tone="any tone",
            paintings=paintings,
            painting_descriptions={f"hash_{i}": f"desc_{i}" for i in range(3)},
            embedding_cache={},
            ai_config=ai_config,
            strategy="random",
        )

        assert painting in paintings
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_random_distributes_uniformly_over_many_calls(
        self, ai_config: MockAiConfig
    ) -> None:
        """Random strategy should distribute selections uniformly (statistical test)."""
        paintings = [make_painting(f"hash_{i}") for i in range(10)]
        counts = {f"hash_{i}": 0 for i in range(10)}

        for _ in range(1000):
            painting, score = await select_painting(
                news_tone="test",
                paintings=paintings,
                painting_descriptions={f"hash_{i}": f"desc_{i}" for i in range(10)},
                embedding_cache={},
                ai_config=ai_config,
                strategy="random",
            )
            counts[painting.content_hash] += 1

        # With 1000 trials and 10 options, expect ~100 per option
        # Allow wide variance for flakiness: 30-200 is reasonable
        for hash_key, count in counts.items():
            assert 30 <= count <= 200, (
                f"Count {count} for {hash_key} outside expected range"
            )

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
            embedding_cache={},
            ai_config=ai_config,
            strategy="random",
        )

        assert result.content_hash == "only_one"
        assert score == 0.0


# =============================================================================
# SEMANTIC STRATEGY TESTS
# =============================================================================


class TestSelectPaintingSemanticStrategy:
    """Tests for semantic strategy behavior per contract.

    Strategy "semantic" MUST:
    - Compute or load embedding for news_tone
    - Look up painting embeddings by content_hash in cache
    - Skip paintings without cached embeddings
    - Return painting with HIGHEST similarity among valid ones
    - Return similarity score (typically 0.0 to 1.0)
    """

    @pytest.mark.asyncio
    async def test_semantic_returns_highest_similarity_painting(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic must return painting with highest similarity score."""
        # Create embeddings where hash_1 is most similar to tone
        tone_vec = make_embedding("energetic vibrant")
        painting_1_vec = make_embedding("energetic vibrant exciting")  # Similar
        painting_2_vec = make_embedding("dark gloomy sad")  # Different
        painting_3_vec = make_embedding("neutral calm")  # Different

        paintings = [
            make_painting("hash_1"),
            make_painting("hash_2"),
            make_painting("hash_3"),
        ]

        embedding_cache = {
            "hash_1": painting_1_vec,
            "hash_2": painting_2_vec,
            "hash_3": painting_3_vec,
        }

        # Pre-compute tone key for cache
        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("energetic tech")
        embedding_cache[tone_key] = tone_vec

        painting, score = await select_painting(
            news_tone="energetic tech",
            paintings=paintings,
            painting_descriptions={
                "hash_1": "energetic",
                "hash_2": "gloomy",
                "hash_3": "neutral",
            },
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        # hash_1 should have highest similarity since embeddings are similar
        assert painting.content_hash == "hash_1"

    @pytest.mark.asyncio
    async def test_semantic_skips_paintings_without_embeddings(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic must skip paintings that don't have cached embeddings."""
        paintings = [
            make_painting("has_embedding"),
            make_painting("no_embedding_1"),
            make_painting("no_embedding_2"),
        ]

        tone_vec = make_embedding("test tone")
        painting_vec = make_embedding("test tone")

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        embedding_cache = {
            tone_key: tone_vec,
            "has_embedding": painting_vec,
            # no_embedding_1 and no_embedding_2 are NOT in cache
        }

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={
                "has_embedding": "desc",
                "no_embedding_1": "desc2",
                "no_embedding_2": "desc3",
            },
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        # Should select the only painting with embedding
        assert painting.content_hash == "has_embedding"

    @pytest.mark.asyncio
    async def test_semantic_score_in_valid_range(self, ai_config: MockAiConfig) -> None:
        """Semantic similarity score must be in valid range."""
        paintings = [make_painting(f"hash_{i}") for i in range(3)]

        tone_vec = make_embedding("test tone")

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test tone")

        embedding_cache = {
            tone_key: tone_vec,
            **{f"hash_{i}": make_embedding(f"desc_{i}") for i in range(3)},
        }

        for _ in range(10):
            painting, score = await select_painting(
                news_tone="test tone",
                paintings=paintings,
                painting_descriptions={f"hash_{i}": f"desc_{i}" for i in range(3)},
                embedding_cache=embedding_cache,
                ai_config=ai_config,
                strategy="semantic",
            )
            assert -1.0 <= score <= 1.0, f"Score {score} outside valid range"


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
                embedding_cache={},
                ai_config=ai_config,
                strategy="semantic",
            )

    @pytest.mark.asyncio
    async def test_semantic_no_cached_embeddings_raises_error(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic strategy with NO cached embeddings must raise MatcherError."""
        paintings = [
            make_painting("hash_1"),
            make_painting("hash_2"),
        ]

        with pytest.raises(MatcherError, match="embedding|cache"):
            await select_painting(
                news_tone="test tone",
                paintings=paintings,
                painting_descriptions={"hash_1": "desc1", "hash_2": "desc2"},
                embedding_cache={},
                ai_config=ai_config,
                strategy="semantic",
            )

    @pytest.mark.asyncio
    async def test_semantic_all_zero_embeddings_raises_error(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic with all zero-vector embeddings must raise MatcherError."""
        paintings = [
            make_painting("hash_1"),
            make_painting("hash_2"),
        ]

        zero_emb = np.zeros(384, dtype=np.float32)

        with pytest.raises(MatcherError, match="zero|invalid|embedding"):
            await select_painting(
                news_tone="test",
                paintings=paintings,
                painting_descriptions={"hash_1": "desc1", "hash_2": "desc2"},
                embedding_cache={
                    "hash_1": zero_emb,
                    "hash_2": zero_emb,
                },
                ai_config=ai_config,
                strategy="semantic",
            )


class TestSelectPaintingFallbackBehavior:
    """Tests for fallback/degradation behavior."""

    @pytest.mark.asyncio
    async def test_semantic_skips_individual_zero_embeddings(
        self, ai_config: MockAiConfig
    ) -> None:
        """Semantic must skip individual zero embeddings, not fail entirely."""
        paintings = [
            make_painting("valid_embedding"),
            make_painting("zero_embedding"),
        ]

        valid_emb = make_embedding("valid")
        zero_emb = np.zeros(384, dtype=np.float32)

        tone_vec = make_embedding("tone")

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={
                "valid_embedding": "valid",
                "zero_embedding": "zero",
            },
            embedding_cache={
                tone_key: tone_vec,
                "valid_embedding": valid_emb,
                "zero_embedding": zero_emb,
            },
            ai_config=ai_config,
            strategy="semantic",
        )

        # Should select the valid one, not fail
        assert painting.content_hash == "valid_embedding"


# =============================================================================
# CACHE KEY TESTS
# =============================================================================


class TestComputeToneCacheKey:
    """Tests for _compute_tone_cache_key behavior."""

    def test_deterministic_for_same_input(self) -> None:
        """Same input must produce same key."""
        from sfumato.matcher import _compute_tone_cache_key

        key1 = _compute_tone_cache_key("energetic tech vibe")
        key2 = _compute_tone_cache_key("energetic tech vibe")
        assert key1 == key2

    def test_different_for_different_input(self) -> None:
        """Different inputs should produce different keys."""
        from sfumato.matcher import _compute_tone_cache_key

        key1 = _compute_tone_cache_key("energetic tech vibe")
        key2 = _compute_tone_cache_key("somber artistic mood")
        assert key1 != key2

    def test_includes_tone_prefix(self) -> None:
        """Key should include 'tone:' prefix to distinguish from painting hashes."""
        from sfumato.matcher import _compute_tone_cache_key

        key = _compute_tone_cache_key("test tone")
        assert key.startswith("tone:"), "Tone cache key should start with 'tone:'"


class TestPaintingCacheKeySemantics:
    """Tests for painting cache key usage in select_painting."""

    @pytest.mark.asyncio
    async def test_uses_content_hash_as_key(self, ai_config: MockAiConfig) -> None:
        """select_painting must use PaintingInfo.content_hash as cache key."""
        paintings = [make_painting("specific_hash_12345")]

        emb = make_embedding("test")

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        # Embedding must be keyed by content_hash
        embedding_cache = {
            tone_key: emb,
            "specific_hash_12345": emb,
        }

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={"specific_hash_12345": "desc"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        # If cache key was correct, selection should work
        assert painting.content_hash == "specific_hash_12345"


# =============================================================================
# TARGETED REGRESSION TESTS
# =============================================================================


class TestRegressionHighRiskPaths:
    """Targeted tests for highest-risk failure paths."""

    @pytest.mark.asyncio
    async def test_regression_very_similar_embeddings(
        self, ai_config: MockAiConfig
    ) -> None:
        """Regression: very similar embeddings should not cause numerical instability."""
        base = make_embedding("base")
        similar = base.copy()
        similar[0] += 1e-10

        paintings = [
            make_painting("hash_1"),
            make_painting("hash_2"),
        ]

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        embedding_cache = {
            tone_key: base,
            "hash_1": base,
            "hash_2": similar,
        }

        # Should not raise or produce NaN/Inf
        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={"hash_1": "a", "hash_2": "b"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        assert math.isfinite(score), "Score must be finite"

    @pytest.mark.asyncio
    async def test_regression_all_negative_embeddings(
        self, ai_config: MockAiConfig
    ) -> None:
        """Regression: all-negative embeddings should still produce valid similarity."""
        # All negative components
        neg_emb = -np.abs(make_embedding("negative"))

        paintings = [make_painting("hash_neg")]

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        embedding_cache = {
            tone_key: neg_emb,
            "hash_neg": neg_emb,
        }

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={"hash_neg": "negative"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        # Similar identical embeddings should be ~1.0
        assert score == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_regression_large_embedding_dimensions(
        self, ai_config: MockAiConfig
    ) -> None:
        """Regression: large embeddings (e.g., 1536 for OpenAI) should work."""
        # Simulate OpenAI embedding size
        large_dim = 1536
        np.random.seed(42)
        tone_vec = np.random.randn(large_dim).astype(np.float32)
        tone_vec = tone_vec / np.linalg.norm(tone_vec)

        painting_vec = tone_vec.copy()

        paintings = [make_painting("large")]

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("test")

        embedding_cache = {
            tone_key: tone_vec,
            "large": painting_vec,
        }

        painting, score = await select_painting(
            news_tone="test",
            paintings=paintings,
            painting_descriptions={"large": "large embedding"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        assert score == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_regression_empty_tone_description(
        self, ai_config: MockAiConfig
    ) -> None:
        """Regression: empty tone description should still compute embedding."""
        paintings = [make_painting("hash_1")]
        emb = make_embedding("any")

        from sfumato.matcher import _compute_tone_cache_key

        tone_key = _compute_tone_cache_key("")

        embedding_cache = {
            tone_key: emb,
            "hash_1": emb,
        }

        # Should not crash on empty tone
        painting, score = await select_painting(
            news_tone="",
            paintings=paintings,
            painting_descriptions={"hash_1": "desc"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        assert painting is not None


# =============================================================================
# TONE CACHE KEY INTEGRATION TESTS
# =============================================================================


class TestToneCacheKeyIntegration:
    """Tests for tone cache key integration with select_painting."""

    @pytest.mark.asyncio
    async def test_semantic_uses_tone_cache_key(self, ai_config: MockAiConfig) -> None:
        """Semantic should look up tone embedding using _compute_tone_cache_key."""
        paintings = [make_painting("hash_1")]

        tone_vec = make_embedding("test_tone_embedding")
        painting_vec = make_embedding("painting_embedding")

        # Tone key should match what _compute_tone_cache_key produces
        from sfumato.matcher import _compute_tone_cache_key

        expected_tone_key = _compute_tone_cache_key("my test tone")

        embedding_cache = {
            expected_tone_key: tone_vec,
            "hash_1": painting_vec,
        }

        # This should work if select_painting uses the same key derivation
        painting, score = await select_painting(
            news_tone="my test tone",
            paintings=paintings,
            painting_descriptions={"hash_1": "test painting"},
            embedding_cache=embedding_cache,
            ai_config=ai_config,
            strategy="semantic",
        )

        assert painting.content_hash == "hash_1"

    @pytest.mark.asyncio
    async def test_semantic_can_share_tone_across_calls(
        self, ai_config: MockAiConfig
    ) -> None:
        """Multiple calls with same tone can share cached embedding."""
        # This tests that the cache key is deterministic
        from sfumato.matcher import _compute_tone_cache_key

        tone = "repeated tone"
        tone_key = _compute_tone_cache_key(tone)

        # Same key should be produced every time
        key1 = _compute_tone_cache_key(tone)
        key2 = _compute_tone_cache_key(tone)
        assert key1 == key2
        assert key1 == tone_key
