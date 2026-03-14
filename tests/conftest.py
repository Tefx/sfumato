"""Shared fixtures for matcher module tests.

Provides minimal test fixtures and mocks for matcher testing.
No product implementation code - only test data and mock objects.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest


class Orientation(Enum):
    """Test double for paintings.Orientation."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class ArtSource(Enum):
    """Test double for paintings.ArtSource."""

    MET = "met"
    WIKIMEDIA = "wikimedia"


@dataclass(frozen=True)
class PaintingInfo:
    """Test double for paintings.PaintingInfo.

    Minimal implementation for matcher tests - contains only fields
    needed for matching logic (content_hash for cache key).
    """

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
class AiConfig:
    """Test double for config.AiConfig."""

    cli: str = "gemini"
    model: str = "test-model"


# ---------------------------------------------------------------------------
# Fixtures for painting data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embedding_vector() -> np.ndarray:
    """Return a sample embedding vector (384 dimensions like MiniLM)."""
    # Create a deterministic "unit-like" vector for testing
    np.random.seed(42)
    vec = np.random.randn(384).astype(np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


@pytest.fixture
def sample_embedding_vector_2() -> np.ndarray:
    """Return a second sample embedding vector, orthogonal to the first."""
    np.random.seed(123)
    vec = np.random.randn(384).astype(np.float32)
    # Normalize to unit length
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


@pytest.fixture
def zero_vector() -> np.ndarray:
    """Return a zero vector of same shape as sample_embedding_vector."""
    return np.zeros(384, dtype=np.float32)


@pytest.fixture
def nan_vector() -> np.ndarray:
    """Return a vector containing NaN values."""
    vec = np.zeros(384, dtype=np.float32)
    vec[0] = np.nan
    vec[100] = np.nan
    return vec


@pytest.fixture
def sample_painting() -> PaintingInfo:
    """Return a sample PaintingInfo for testing."""
    return PaintingInfo(
        image_path=Path("/tmp/test/starry_night.jpg"),
        content_hash="hash_starry_night_001",
        title="The Starry Night",
        artist="Vincent van Gogh",
        year="1889",
        source=ArtSource.WIKIMEDIA,
        source_id="Starry_Night",
        source_url="https://example.com/starry_night",
        orientation=Orientation.LANDSCAPE,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_painting_2() -> PaintingInfo:
    """Return a second sample PaintingInfo for testing."""
    return PaintingInfo(
        image_path=Path("/tmp/test/great_wave.jpg"),
        content_hash="hash_great_wave_002",
        title="The Great Wave off Kanagawa",
        artist="Katsushika Hokusai",
        year="c. 1831",
        source=ArtSource.WIKIMEDIA,
        source_id="Great_Wave",
        source_url="https://example.com/great_wave",
        orientation=Orientation.LANDSCAPE,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_painting_3() -> PaintingInfo:
    """Return a third sample PaintingInfo for testing."""
    return PaintingInfo(
        image_path=Path("/tmp/test/wanderer.jpg"),
        content_hash="hash_wanderer_003",
        title="Wanderer above the Sea of Fog",
        artist="Caspar David Friedrich",
        year="c. 1818",
        source=ArtSource.WIKIMEDIA,
        source_id="Wanderer",
        source_url="https://example.com/wanderer",
        orientation=Orientation.PORTRAIT,
        width=1080,
        height=1920,
    )


@pytest.fixture
def sample_paintings_list(
    sample_painting: PaintingInfo,
    sample_painting_2: PaintingInfo,
    sample_painting_3: PaintingInfo,
) -> list[PaintingInfo]:
    """Return a list of 3 sample paintings."""
    return [sample_painting, sample_painting_2, sample_painting_3]


@pytest.fixture
def sample_painting_descriptions() -> dict[str, str]:
    """Return sample painting descriptions keyed by content_hash."""
    return {
        "hash_starry_night_001": "Swirling night sky with cypress tree, energetic brushstrokes, deep blues and yellows",
        "hash_great_wave_002": "Dramatic ocean wave with Mount Fuji, tense composition, blue and white",
        "hash_wanderer_003": "Solitary figure on rocky cliff, contemplative atmosphere, fog and mist",
    }


@pytest.fixture
def sample_painting_embeddings(
    sample_embedding_vector: np.ndarray,
    sample_embedding_vector_2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return sample painting embeddings keyed by content_hash."""
    # Use same base vectors but with slight variations for each painting
    vec1 = sample_embedding_vector.copy()
    vec2 = sample_embedding_vector_2.copy()
    vec3 = (sample_embedding_vector + sample_embedding_vector_2) / 2
    # Normalize vec3
    norm = np.linalg.norm(vec3)
    if norm > 0:
        vec3 = vec3 / norm

    return {
        "hash_starry_night_001": vec1.astype(np.float32),
        "hash_great_wave_002": vec2.astype(np.float32),
        "hash_wanderer_003": vec3.astype(np.float32),
    }


@pytest.fixture
def sample_tone_embedding() -> np.ndarray:
    """Return a sample tone embedding similar to starry night."""
    np.random.seed(42)
    vec = np.random.randn(384).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


@pytest.fixture
def ai_config() -> AiConfig:
    """Return a default AiConfig for testing."""
    return AiConfig()


# ---------------------------------------------------------------------------
# Helper functions for tests
# ---------------------------------------------------------------------------


def make_embedding(text: str, dimensions: int = 384) -> np.ndarray:
    """Create a deterministic embedding vector from text.

    Used for generating consistent test embeddings. NOT a real embedding algorithm.
    """
    # Use hash to seed random for deterministic output
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    vec = np.random.randn(dimensions).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def make_painting(
    content_hash: str,
    title: str = "Test Painting",
    artist: str = "Test Artist",
) -> PaintingInfo:
    """Create a PaintingInfo with minimal required fields."""
    return PaintingInfo(
        image_path=Path(f"/tmp/{content_hash}.jpg"),
        content_hash=content_hash,
        title=title,
        artist=artist,
        year="2024",
        source=ArtSource.WIKIMEDIA,
        source_id=content_hash,
        source_url=f"https://example.com/{content_hash}",
        orientation=Orientation.LANDSCAPE,
        width=100,
        height=100,
    )
