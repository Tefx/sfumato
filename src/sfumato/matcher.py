"""Semantic painting-news matching implementation.

Implements the matcher contract from ARCHITECTURE.md#2.6:
- Embedding computation for text (painting descriptions and news tone)
- Cosine similarity calculation with explicit zero-vector semantics
- Painting selection with semantic vs random strategy precedence
- Embedding cache management keyed by content_hash
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sfumato.llm import compute_embedding as llm_compute_embedding, EmbeddingError

if TYPE_CHECKING:
    from sfumato.config import AiConfig
    from sfumato.paintings import PaintingInfo

__all__ = [
    "MatcherError",
    "EmbeddingResult",
    "compute_embedding",
    "cosine_similarity",
    "select_painting",
]


class MatcherError(Exception):
    """Raised when painting selection cannot proceed.

    Common causes:
    - Empty paintings list (no paintings available)
    - Missing painting embeddings with no fallback allowed
    - Embedding computation failures when strategy requires it
    """

    pass


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of a text embedding computation.

    Attributes:
        text: The original text that was embedded. Preserved for debugging
            and cache key derivation.
        vector: The embedding vector as a float32 numpy array. Shape is
            model-dependent (typically 768, 1024, or 1536 dimensions).
            Must be a 1-D numpy array of dtype float32.
        model: Identifier for the backend model that produced this embedding.
            Examples: "text-embedding-004", "sentence-transformers/all-MiniLM-L6-v2".

    Contract:
        - vector MUST be a 1-D numpy.ndarray with dtype float32.
        - vector MUST NOT be a zero-vector (all zeros) - callers should validate.
        - model MUST be non-empty string identifying the embedding backend.
        - text is the exact input; truncation or normalization is NOT performed.
        - EmbeddingResult is immutable (frozen dataclass).

    Cache Key Semantics:
        - Painting embeddings are cached by PaintingInfo.content_hash.
        - Tone embeddings are cached by computing a stable hash of the text.
          See `_compute_tone_cache_key()` for implementation contract.
    """

    text: str
    vector: np.ndarray
    model: str

    def __post_init__(self) -> None:
        """Validate invariants.

        Raises:
            ValueError: If vector is not float32, not 1-D, or is empty.
        """
        if not isinstance(self.vector, np.ndarray):
            raise ValueError("vector must be a numpy.ndarray")
        if self.vector.dtype != np.float32:
            raise ValueError(f"vector must be float32, got {self.vector.dtype}")
        if self.vector.ndim != 1:
            raise ValueError(f"vector must be 1-D, got {self.vector.ndim} dimensions")
        if len(self.vector) == 0:
            raise ValueError("vector must not be empty")
        if not self.model:
            raise ValueError("model must be non-empty string")


async def compute_embedding(
    text: str,
    ai_config: "AiConfig",
) -> EmbeddingResult:
    """Compute embedding vector for a text string.

    Args:
        text: The text to embed. Preserved verbatim in the result.
        ai_config: AI backend configuration specifying which embedding
            model to use (e.g., gemini embedding endpoint).

    Returns:
        EmbeddingResult containing the original text, float32 numpy vector,
        and backend model identifier.

    Raises:
        EmbeddingError: If embedding computation fails after retries.
            Imported from llm.py (defined there as a subclass of LlmError).

    Contract:
        - Uses the configured AI backend's embedding API.
        - For gemini: calls Gemini embedding endpoint.
        - For local: may use sentence-transformers if available.
        - Returns float32 numpy array in model-specific dimensions.
        - Does NOT cache results - caching is caller's responsibility
          (typically via EmbeddingCache in state.py).
        - Zero-vector detection is NOT performed here; caller should check.
    """
    # Delegate to llm.compute_embedding which returns list[float]
    embedding_list = await llm_compute_embedding(text, ai_config)

    # Convert to numpy float32 array
    vector = np.array(embedding_list, dtype=np.float32)

    # Return the structured result
    return EmbeddingResult(
        text=text,
        vector=vector,
        model=ai_config.embedding_model,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector as 1-D float32 or float64 numpy array.
        b: Second vector as 1-D float32 or float64 numpy array.

    Returns:
        Cosine similarity as float in the range [-1.0, 1.0].
        Returns 1.0 if both vectors are identical (including zero-vectors).
        Returns 0.0 if one vector is zero and the other is non-zero.

    Raises:
        ValueError: If vectors have different dimensions.
        ValueError: If either vector is not 1-D.
        ValueError: If either vector contains NaN.

    Contract:
        - Result is ALWAYS in [-1.0, 1.0].
        - Zero-vector behavior:
            - If BOTH vectors are zero (norm=0): return 0.0.
            - If ONE vector is zero and the other non-zero: return 0.0.
            - This prevents undefined division-by-zero scenarios.
        - NaN handling:
            - If either vector contains NaN: raises ValueError.
        - Identical vectors (including zero): return 1.0 if both zero
          and same, otherwise computed similarity.
        - Vectors must have the same dimension.
    """
    # Validate 1-D
    if a.ndim != 1:
        raise ValueError(f"vector a must be 1-D, got {a.ndim} dimensions")
    if b.ndim != 1:
        raise ValueError(f"vector b must be 1-D, got {b.ndim} dimensions")

    # Validate same dimensions
    if len(a) != len(b):
        raise ValueError(
            f"vectors must have same dimensions: a has {len(a)}, b has {len(b)}"
        )

    # Validate no NaN
    if np.any(np.isnan(a)):
        raise ValueError("vector a contains NaN values")
    if np.any(np.isnan(b)):
        raise ValueError("vector b contains NaN values")

    # Convert to float64 for numerical stability
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    # Compute norms
    norm_a = np.linalg.norm(a_float)
    norm_b = np.linalg.norm(b_float)

    # Handle zero-vector cases
    if norm_a == 0.0 and norm_b == 0.0:
        # Both zero vectors - return 0.0 per contract
        return 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        # One zero, one non-zero - return 0.0 per contract
        return 0.0

    # Compute cosine similarity
    similarity = float(np.dot(a_float, b_float) / (norm_a * norm_b))

    # Clamp to [-1.0, 1.0] to handle floating-point errors
    return max(-1.0, min(1.0, similarity))


def _compute_tone_cache_key(tone_description: str) -> str:
    """Compute a stable cache key for a tone description.

    Args:
        tone_description: The free-form tone text from news curation.

    Returns:
        A string key suitable for use in EmbeddingCache.

    Contract:
        - Key is stable/deterministic for identical tone descriptions.
        - Key is unique with high probability for different tone descriptions.
        - Recommended: prefix with "tone:" to distinguish from painting hashes.
        - Implementation should use SHA-256 or similar stable hash.
    """
    # Use SHA-256 for stable, collision-resistant hashing
    hash_hex = hashlib.sha256(tone_description.encode("utf-8")).hexdigest()[:16]
    return f"tone:{hash_hex}"


async def select_painting(
    news_tone: str,
    paintings: list["PaintingInfo"],
    painting_descriptions: dict[str, str],
    embedding_cache: dict[str, np.ndarray],
    ai_config: "AiConfig",
    strategy: str = "semantic",
) -> tuple["PaintingInfo", float]:
    """Select the best painting for the given news tone.

    Args:
        news_tone: Free-form tone description from news curation (e.g.,
            "Tech optimism meets cautious skepticism, energetic startup vibe").
        paintings: List of available paintings to choose from.
        painting_descriptions: Mapping from PaintingInfo.content_hash to
            the free-form painting description generated by layout_ai.
        embedding_cache: Pre-computed embeddings cached by content_hash
            (for paintings) or tone key (for news tone).
            Keys are strings, values are float32 numpy vectors.
        ai_config: AI backend configuration for embedding computation.
        strategy: Selection strategy:
            - "semantic": Compute tone embedding, find highest similarity
              painting with cached embedding.
            - "random": Select randomly, bypass semantic scoring, return
              score=0.0.

    Returns:
        Tuple of (selected_painting, similarity_score).
        - similarity_score is in [0.0, 1.0] for semantic strategy.
        - similarity_score is exactly 0.0 for random strategy.

    Raises:
        MatcherError: If no paintings are available (empty list).
        MatcherError: If semantic strategy is used but:
            - No paintings have cached embeddings.
            - Tone embedding computation fails.
            - All embeddings are invalid (e.g., all zero-vectors).

    Contract:
        Strategy Precedence:
            1. If strategy == "random":
               - Ignore news_tone entirely.
               - Select uniformly at random from paintings.
               - Return score = 0.0.
               - No embedding computation or cache lookup needed.

            2. If strategy == "semantic":
               - Compute (or load from cache) embedding for news_tone.
               - For each painting in paintings:
                 - Look up embedding in embedding_cache by content_hash.
                 - Skip paintings without cached embeddings.
               - Compute cosine similarity between tone embedding and
                 each painting's embedding.
               - Return painting with highest similarity among valid ones.
               - Return the similarity score (0.0 to 1.0 generally).

        Fallback/Degradation Behavior:
            - Empty paintings list: raise MatcherError (cannot select).
            - Semantic strategy with NO cached embeddings:
              Option A: Raise MatcherError.
              Option B: Fall back to random selection.
              Contract specifies: raise MatcherError with clear message.
            - Embedding computation failure: raise MatcherError.
            - Zero-vector embedding detected: skip that painting and
              continue with others; if ALL embeddings are zero-vectors,
              raise MatcherError.

        Cache Key Surface:
            - Painting embeddings use PaintingInfo.content_hash as key.
            - Tone embeddings use _compute_tone_cache_key(tone) as key.
    """
    # Validate: paintings list must not be empty
    if not paintings:
        raise MatcherError("Cannot select painting: paintings list is empty")

    # Strategy: random - bypass all semantic logic
    if strategy == "random":
        selected = random.choice(paintings)
        return (selected, 0.0)

    # Strategy: semantic
    if strategy != "semantic":
        raise MatcherError(f"Unknown strategy: {strategy}. Use 'semantic' or 'random'.")

    # Get or compute tone embedding
    tone_key = _compute_tone_cache_key(news_tone)

    if tone_key in embedding_cache:
        tone_embedding = embedding_cache[tone_key]
    else:
        # Compute tone embedding
        try:
            embedding_result = await compute_embedding(news_tone, ai_config)
            tone_embedding = embedding_result.vector
        except EmbeddingError as e:
            raise MatcherError(f"Failed to compute tone embedding: {e}") from e

    # Collect valid painting candidates with their embeddings
    candidates: list[tuple["PaintingInfo", np.ndarray, float]] = []

    for painting in paintings:
        content_hash = painting.content_hash

        if content_hash not in embedding_cache:
            # Skip paintings without cached embeddings
            continue

        painting_embedding = embedding_cache[content_hash]

        # Check for zero-vector embedding
        if np.all(painting_embedding == 0):
            # Skip zero-vector embeddings
            continue

        # Check for NaN in embedding
        if np.any(np.isnan(painting_embedding)):
            # Skip invalid embeddings with NaN
            continue

        # Compute similarity
        try:
            similarity = cosine_similarity(tone_embedding, painting_embedding)
        except ValueError:
            # Skip if similarity computation fails
            continue

        candidates.append((painting, painting_embedding, similarity))

    # Check if we have any valid candidates
    if not candidates:
        # Check why we have no candidates
        paintings_with_keys = [
            p for p in paintings if p.content_hash in embedding_cache
        ]
        if not paintings_with_keys:
            raise MatcherError(
                "Cannot select painting: no paintings have cached embeddings. "
                "Compute painting embeddings before calling select_painting."
            )

        # All cached embeddings were invalid (zero or NaN)
        raise MatcherError(
            "Cannot select painting: all cached embeddings are invalid "
            "(zero-vectors or contain NaN)"
        )

    # Find the painting with highest similarity
    best_painting, _best_embedding, best_score = max(candidates, key=lambda x: x[2])

    return (best_painting, best_score)
