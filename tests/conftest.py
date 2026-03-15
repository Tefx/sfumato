"""Shared fixtures for matcher module tests.

Provides minimal test fixtures and mocks for matcher testing.
No product implementation code - only test data and mock objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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
def ai_config() -> AiConfig:
    """Return a default AiConfig for testing."""
    return AiConfig()


# ---------------------------------------------------------------------------
# Helper functions for tests
# ---------------------------------------------------------------------------




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
