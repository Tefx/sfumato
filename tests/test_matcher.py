"""Tests for matcher.py contract definitions.

This test module verifies:
- EmbeddingResult dataclass fields and validation
- Error class hierarchy
- Function signatures and stub behaviors
- Docstring contracts are correct and complete

Implementation tests will be added by future implementation steps.
"""

from __future__ import annotations

import pytest
import numpy as np

from sfumato.matcher import (
    MatcherError,
    EmbeddingResult,
    compute_embedding,
    cosine_similarity,
    select_painting,
)


class TestEmbeddingResultContract:
    """Test EmbeddingResult dataclass contract per ARCHITECTURE.md#2.6."""

    def test_fields_match_architecture(self) -> None:
        """EmbeddingResult has text, vector, model fields as specified."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EmbeddingResult(text="test text", vector=vector, model="test-model")

        assert result.text == "test text"
        assert result.model == "test-model"
        assert isinstance(result.vector, np.ndarray)
        assert result.vector.dtype == np.float32
        assert np.array_equal(result.vector, vector)

    def test_frozen_dataclass_immutable(self) -> None:
        """EmbeddingResult is frozen (immutable)."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = EmbeddingResult(text="test", vector=vector, model="model")

        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]

    def test_vector_must_be_numpy_array(self) -> None:
        """vector must be a numpy array."""
        with pytest.raises(ValueError, match="vector must be a numpy.ndarray"):
            EmbeddingResult(text="test", vector=[0.1, 0.2, 0.3], model="model")  # type: ignore[arg-type]

    def test_vector_must_be_float32(self) -> None:
        """vector must have dtype float32."""
        vector_f64 = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        with pytest.raises(ValueError, match="vector must be float32"):
            EmbeddingResult(text="test", vector=vector_f64, model="model")

    def test_vector_must_be_1d(self) -> None:
        """vector must be 1-dimensional."""
        vector_2d = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        with pytest.raises(ValueError, match="vector must be 1-D"):
            EmbeddingResult(text="test", vector=vector_2d, model="model")

    def test_vector_must_not_be_empty(self) -> None:
        """vector must have at least one element."""
        empty_vector = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="vector must not be empty"):
            EmbeddingResult(text="test", vector=empty_vector, model="model")

    def test_model_must_be_non_empty(self) -> None:
        """model must be a non-empty string."""
        vector = np.array([0.1], dtype=np.float32)
        with pytest.raises(ValueError, match="model must be non-empty"):
            EmbeddingResult(text="test", vector=vector, model="")

    def test_preserves_original_text(self) -> None:
        """text is preserved exactly as provided (no truncation)."""
        long_text = "A" * 10000
        vector = np.array([0.1], dtype=np.float32)
        result = EmbeddingResult(text=long_text, vector=vector, model="model")

        assert result.text == long_text
        assert len(result.text) == 10000


class TestMatcherErrorContract:
    """Test MatcherError exception contract."""

    def test_is_exception_subclass(self) -> None:
        """MatcherError is an Exception subclass."""
        assert issubclass(MatcherError, Exception)

    def test_can_raise_with_message(self) -> None:
        """MatcherError can be raised with a custom message."""
        with pytest.raises(MatcherError, match="no paintings available"):
            raise MatcherError("no paintings available")


class TestComputeEmbeddingStub:
    """Test compute_embedding stub raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_stub_raises_not_implemented(self) -> None:
        """compute_embedding stub raises NotImplementedError."""
        from sfumato.config import AiConfig

        ai_config = AiConfig()
        with pytest.raises(NotImplementedError, match="contract stub"):
            await compute_embedding("test text", ai_config)


class TestCosineSimilarityStub:
    """Test cosine_similarity stub raises NotImplementedError."""

    def test_stub_raises_not_implemented(self) -> None:
        """cosine_similarity stub raises NotImplementedError."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        with pytest.raises(NotImplementedError, match="contract stub"):
            cosine_similarity(a, b)


class TestSelectPaintingStub:
    """Test select_painting stub raises NotImplementedError."""

    @pytest.mark.asyncio
    async def test_stub_raises_not_implemented(self) -> None:
        """select_painting stub raises NotImplementedError."""
        from sfumato.config import AiConfig
        from sfumato.paintings import PaintingInfo, ArtSource, Orientation
        from pathlib import Path

        ai_config = AiConfig()
        painting = PaintingInfo(
            image_path=Path("/tmp/test.jpg"),
            content_hash="abc123",
            title="Test",
            artist="Artist",
            year="2024",
            source=ArtSource.WIKIMEDIA,
            source_id="test-1",
            source_url="https://example.com",
            orientation=Orientation.LANDSCAPE,
            width=100,
            height=100,
        )

        with pytest.raises(NotImplementedError, match="contract stub"):
            await select_painting(
                news_tone="energetic tech vibe",
                paintings=[painting],
                painting_descriptions={"abc123": "swirling night sky"},
                embedding_cache={},
                ai_config=ai_config,
                strategy="random",
            )


class TestCosineSimilarityContractSpec:
    """Tests verifying cosine_similarity contract semantics in docstring.

    These tests verify the CONTRACT SPECIFICATION (not implementation) is correct.
    The docstring specifies:

    1. Result is ALWAYS in [-1.0, 1.0]
    2. Zero-vector behavior:
       - BOTH zero: return 0.0
       - ONE zero, other non-zero: return 0.0
    3. NaN handling: raise ValueError
    4. Different dimensions: raise ValueError
    5. Not 1-D: raise ValueError
    6. Identical vectors: return ~1.0 (or computed similarity)

    These tests serve as contract validators for future implementation.
    """

    def test_contract_spec_zero_vector_behavior_documented(self) -> None:
        """Contract specifies zero-vector behavior clearly."""
        import inspect

        doc = cosine_similarity.__doc__ or ""
        assert "[-1.0, 1.0]" in doc, "Contract must specify result range"
        assert "Zero-vector behavior" in doc or "zero" in doc.lower(), (
            "Contract must specify zero-vector behavior"
        )

    def test_contract_spec_nan_handling_documented(self) -> None:
        """Contract specifies NaN handling."""
        import inspect

        doc = cosine_similarity.__doc__ or ""
        assert "NaN" in doc, "Contract must specify NaN handling"


class TestSelectPaintingContractSpec:
    """Tests verifying select_painting contract semantics in docstring.

    Verifies the contract spec for:
    1. Strategy precedence (random vs semantic)
    2. Random mode: score=0.0, no embeddings
    3. Semantic mode: computing/loading embeddings
    4. Fallback/missing embedding behavior
    5. Empty paintings error
    """

    def test_contract_spec_strategy_precedence_documented(self) -> None:
        """Contract specifies strategy precedence."""
        import inspect

        doc = select_painting.__doc__ or ""
        assert "semantic" in doc.lower()
        assert "random" in doc.lower()
        assert "precedence" in doc.lower() or ("semantic" in doc and "random" in doc)

    def test_contract_spec_random_returns_zero_score_documented(self) -> None:
        """Contract specifies random strategy returns score=0.0."""
        import inspect

        doc = select_painting.__doc__ or ""
        assert "0.0" in doc, "Contract must specify random returns score=0.0"
        assert "random" in doc.lower()

    def test_contract_spec_missing_embedding_behavior_documented(self) -> None:
        """Contract specifies behavior for missing/failed embeddings."""
        import inspect

        doc = select_painting.__doc__ or ""
        assert "MatcherError" in doc, "Contract must specify error behavior"
        # Should mention what happens when embeddings missing/failed
        assert "embedding" in doc.lower()

    def test_contract_spec_empty_paintings_error_documented(self) -> None:
        """Contract specifies error for empty paintings list."""
        import inspect

        doc = select_painting.__doc__ or ""
        assert "empty" in doc.lower() or "no paintings" in doc.lower()

    def test_contract_spec_cache_keys_documented(self) -> None:
        """Contract specifies cache key surface for paintings and tone."""
        import inspect

        doc = select_painting.__doc__ or ""
        # Should mention content_hash for paintings
        assert "content_hash" in doc.lower() or "hash" in doc.lower()


class TestEmbeddingResultCacheKeySemantics:
    """Tests verifying EmbeddingResult cache key semantics documented."""

    def test_contract_spec_painting_cache_key_documented(self) -> None:
        """Contract specifies painting embeddings keyed by content_hash."""
        import inspect

        doc = EmbeddingResult.__doc__ or ""
        # Should mention content_hash for painting cache keys
        assert "content_hash" in doc.lower()


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports_match_architecture(self) -> None:
        """Exports match ARCHITECTURE.md#2.6 public interface."""
        from sfumato import matcher

        expected_exports = {
            "MatcherError",
            "EmbeddingResult",
            "compute_embedding",
            "cosine_similarity",
            "select_painting",
        }

        actual_exports = set(matcher.__all__)
        assert expected_exports == actual_exports, (
            f"Expected __all__={expected_exports}, got {actual_exports}"
        )
