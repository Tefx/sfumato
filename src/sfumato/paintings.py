"""Fetch and cache paintings from cloud art APIs.

This module implements the paintings module contract from ARCHITECTURE.md#2.3:
- Fetch paintings from Rijksmuseum, Met Museum, Wikimedia Commons APIs
- Download high-resolution images to local cache with metadata sidecar
- Manage painting pool and track which paintings have been used
- Provide painting metadata for semantic matching

Architecture reference: ARCHITECTURE.md#2.3
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# =============================================================================
# PUBLIC ENUMS
# =============================================================================


class Orientation(Enum):
    """Image orientation determined by dimensions.

    Contract:
        - LANDSCAPE: width > height (wider than tall)
        - PORTRAIT: width <= height (taller than wide or square)
        - Square images (width == height) are classified as PORTRAIT
    """

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class ArtSource(Enum):
    """Art API source identifiers.

    Contract:
        - RIJKSMUSEUM: Rijksmuseum API (requires RIJKSMUSEUM_API_KEY env var)
        - MET: Metropolitan Museum of Art API (no key required)
        - WIKIMEDIA: Wikimedia Commons API (no key required)
    """

    RIJKSMUSEUM = "rijksmuseum"
    MET = "met"
    WIKIMEDIA = "wikimedia"


# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass
class PaintingInfo:
    """Metadata for a cached painting.

    Attributes:
        image_path: Absolute path to cached image file.
        content_hash: SHA-256 hash of image bytes, used as cache key everywhere.
        title: Painting title (original language).
        artist: Artist name.
        year: Year or period string, e.g. "1889" or "c. 1665".
        source: Which API this painting came from.
        source_id: ID in the source API (for deduplication).
        source_url: URL to the painting's page on the source website.
        orientation: Detected from image dimensions.
        width: Image width in pixels.
        height: Image height in pixels.

    Contract:
        - image_path is always absolute and points to an existing file
        - content_hash is SHA-256 hex digest of raw image file bytes
        - source_id is unique within its source (rijksmuseum, met, wikimedia)
        - content_hash is used as the global deduplication key across all sources
        - orientation is PORTRAIT for square images (width == height)
        - All fields are populated before caching (no None values)
        - JSON sidecar contains all fields except image_path (computed from cache_dir)
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


# =============================================================================
# ERROR TYPES
# =============================================================================


class PaintingsError(Exception):
    """Base exception for paintings-related failures."""

    pass


class SourceAuthError(PaintingsError):
    """A source requires authentication that is not configured.

    Contract:
        - Raised when a source requires an API key but the key is missing
        - For Rijksmuseum: RIJKSMUSEUM_API_KEY env var not set
        - This is a SKIP scenario - the source should be excluded gracefully
    """

    pass


class ImageDownloadError(PaintingsError):
    """Failed to download an image after retries.

    Contract:
        - Individual download failures are logged and skipped
        - Does NOT fail the entire fetch operation
        - Contains source_id in error message for debugging
    """

    pass


# =============================================================================
# LOGGING
# =============================================================================


logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API - FETCH PAINTINGS
# =============================================================================


async def fetch_paintings(
    sources: list[str],
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch `count` paintings from the specified sources.

    Downloads images to cache_dir/{source}/{source_id}.jpg.
    Skips paintings whose source_id is in exclude_ids (PRE-DOWNLOAD filter).
    Returns PaintingInfo for each successfully downloaded painting.

    Args:
        sources: List of source names: "rijksmuseum", "met", "wikimedia".
        count: Number of paintings to fetch per source.
        cache_dir: Base cache directory, e.g. ~/.sfumato/paintings.
        exclude_ids: Set of source_id values to skip (already in pool/used).
            This is a PRE-DOWNLOAD filter - checked BEFORE API calls.

    Returns:
        List of PaintingInfo objects for successfully downloaded paintings.
        Individual download failures are logged and skipped.

    Raises:
        SourceAuthError: If Rijksmuseum source is requested but
            RIJKSMUSEUM_API_KEY env var is not set.
            For Met and Wikimedia, no auth is required.

    Contract:
        AGGREGATE SUCCESS:
        - Returns list of successfully downloaded paintings
        - Empty list if all downloads fail (logs warnings)
        - Individual source failures are logged and that source is skipped
        - Never raises - always returns a list (possibly empty)

        EXCLUDE_ID FILTERING (PRE-DOWNLOAD):
        - exclude_ids is checked BEFORE making API calls
        - If exclude_ids contains source_id "SK-A-1234", don't fetch that painting
        - This prevents downloading paintings already in the pool
        - Format: "{source}:{source_id}" e.g. "rijksmuseum:SK-A-1234"

        CACHE LAYOUT:
        - Images: cache_dir/{source}/{source_id}.jpg
        - Metadata: cache_dir/{source}/{source_id}.json (sidecar)
        - Both files are created atomically (download image first, then write JSON)

        CONTENT-HASH DEDUP (POST-CACHE):
        - After download, compute SHA-256 of raw image bytes
        - If content_hash matches an existing cached painting, skip it
        - This handles the case where same image has different source_ids

        DEFERRED SOURCE-AUTH BEHAVIOR:
        - Rijksmuseum requires RIJKSMUSEUM_API_KEY env var
        - If key is missing, log warning and SKIP that source (don't fail)
        - Met and Wikimedia do not require auth
        - SourceAuthError is raised only if ALL sources require auth and ALL fail

        ORIENTATION SEMANTICS:
        - LANDSCAPE: width > height
        - PORTRAIT: width <= height (includes square images)

    Example:
        >>> paintings = await fetch_paintings(
        ...     sources=["rijksmuseum", "met"],
        ...     count=10,
        ...     cache_dir=Path("~/.sfumato/paintings"),
        ...     exclude_ids={"rijksmuseum:SK-A-3262", "met:436535"},
        ... )
        >>> len(paintings)  # depends on API responses
        18
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "fetch_paintings contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


async def fetch_from_rijksmuseum(
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch paintings from Rijksmuseum API.

    Requires RIJKSMUSEUM_API_KEY env var.
    Filters for paintings only (type=painting), high resolution available.

    Args:
        count: Number of paintings to fetch.
        cache_dir: Base cache directory for downloads.
        exclude_ids: Set of source_id values to skip (PRE-DOWNLOAD filter).

    Returns:
        List of PaintingInfo objects for successfully downloaded paintings.

    Raises:
        SourceAuthError: If RIJKSMUSEUM_API_KEY env var is not set.
            This is a FAIL scenario for THIS source - but aggregate fetch_paintings
            handles it gracefully by skipping this source.

    Contract:
        AUTH REQUIREMENT:
        - Requires RIJKSMUSEUM_API_KEY environment variable
        - If missing, raise SourceAuthError (callers should skip this source)

        API FILTERING:
        - Only fetches objects where artObject.hasImage is true
        - Only fetches objects where artObject.objectTypes includes "painting"
        - Prefers high-resolution images when available

        EXCLUDE_ID FILTERING:
        - exclude_ids is checked BEFORE making API calls
        - Format: "rijksmuseum:{objectNumber}" e.g. "rijksmuseum:SK-A-3262"

        CACHE BEHAVIOR:
        - Downloads to cache_dir/rijksmuseum/{objectNumber}.jpg
        - Writes metadata sidecar to cache_dir/rijksmuseum/{objectNumber}.json
        - Skips if file already exists and content_hash matches

    Example:
        >>> import os
        >>> os.environ["RIJKSMUSEUM_API_KEY"] = "your-key"
        >>> paintings = await fetch_from_rijksmuseum(count=5, cache_dir=Path("~/.sfumato/paintings"))
        >>> len(paintings)
        5
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "fetch_from_rijksmuseum contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


async def fetch_from_met(
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch paintings from Met Museum API.

    No API key required.
    Filters for paintings with isPublicDomain=true.

    Args:
        count: Number of paintings to fetch.
        cache_dir: Base cache directory for downloads.
        exclude_ids: Set of source_id values to skip (PRE-DOWNLOAD filter).

    Returns:
        List of PaintingInfo objects for successfully downloaded paintings.

    Contract:
        AUTH REQUIREMENT:
        - No API key required
        - Never raises SourceAuthError

        API FILTERING:
        - Only fetches objects where isPublicDomain is true
        - Only fetches objects where objectID relates to paintings
        - Uses the `/objects` endpoint with department filter for "European Paintings"

        EXCLUDE_ID FILTERING:
        - exclude_ids is checked BEFORE making API calls
        - Format: "met:{objectID}" e.g. "met:436535"

        CACHE BEHAVIOR:
        - Downloads to cache_dir/met/{objectID}.jpg
        - Writes metadata sidecar to cache_dir/met/{objectID}.json
        - Skips if file already exists and content_hash matches

    Example:
        >>> paintings = await fetch_from_met(count=5, cache_dir=Path("~/.sfumato/paintings"))
        >>> len(paintings)
        5
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "fetch_from_met contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


async def fetch_from_wikimedia(
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None = None,
) -> list[PaintingInfo]:
    """Fetch paintings from Wikimedia Commons.

    No API key required.
    Searches Category:Featured_pictures_of_paintings.

    Args:
        count: Number of paintings to fetch.
        cache_dir: Base cache directory for downloads.
        exclude_ids: Set of source_id values to skip (PRE-DOWNLOAD filter).

    Returns:
        List of PaintingInfo objects for successfully downloaded paintings.

    Contract:
        AUTH REQUIREMENT:
        - No API key required
        - Never raises SourceAuthError

        API FILTERING:
        - Searches Category:Featured_pictures_of_paintings
        - Prefers high-resolution images
        - Handles rate limiting with delays between requests

        EXCLUDE_ID FILTERING:
        - exclude_ids is checked BEFORE making API calls
        - Format: "wikimedia:{filename}" e.g. "wikimedia:Mona_Lisa"

        CACHE BEHAVIOR:
        - Downloads to cache_dir/wikimedia/{filename}.jpg
        - Writes metadata sidecar to cache_dir/wikimedia/{filename}.json
        - Skips if file already exists and content_hash matches

        RATE LIMITING:
        - Wikimedia may return 429 if rate limited
        - Add 2-3 second delays between requests
        - Use browser-like User-Agent header

    Example:
        >>> paintings = await fetch_from_wikimedia(count=5, cache_dir=Path("~/.sfumato/paintings"))
        >>> len(paintings)
        5
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "fetch_from_wikimedia contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


# =============================================================================
# PUBLIC API - ORIENTATION
# =============================================================================


def detect_orientation(image_path: Path) -> Orientation:
    """Determine orientation from image dimensions using PIL.

    Args:
        image_path: Path to the image file.

    Returns:
        Orientation enum value.

    Raises:
        FileNotFoundError: If image file does not exist.
        PIL.UnidentifiedImageError: If file is not a valid image.

    Contract:
        ORIENTATION RULES:
        - LANDSCAPE: width > height (wider than tall)
        - PORTRAIT: width <= height (taller than wide OR square)
        - SQUARE: width == height → PORTRAIT (not a separate category)

        IMPLEMENTATION:
        - Uses PIL.Image.open() to get dimensions
        - No image processing, just reads header metadata

    Example:
        >>> from PIL import Image
        >>> # Create a landscape image
        >>> img = Image.new("RGB", (1920, 1080))
        >>> img.save("/tmp/test_landscape.jpg")
        >>> detect_orientation(Path("/tmp/test_landscape.jpg"))
        <Orientation.LANDSCAPE: 'landscape'>
        >>> # Create a portrait image
        >>> img = Image.new("RGB", (1080, 1920))
        >>> img.save("/tmp/test_portrait.jpg")
        >>> detect_orientation(Path("/tmp/test_portrait.jpg"))
        <Orientation.PORTRAIT: 'portrait'>
        >>> # Create a square image
        >>> img = Image.new("RGB", (1000, 1000))
        >>> img.save("/tmp/test_square.jpg")
        >>> detect_orientation(Path("/tmp/test_square.jpg"))
        <Orientation.PORTRAIT: 'portrait'>
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "detect_orientation contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


# =============================================================================
# PUBLIC API - CACHE OPERATIONS
# =============================================================================


def list_cached_paintings(cache_dir: Path) -> list[PaintingInfo]:
    """List all paintings in the local cache.

    Reads metadata from sidecar JSON files (same name, .json extension).

    Args:
        cache_dir: Base cache directory, e.g. ~/.sfumato/paintings.

    Returns:
        List of PaintingInfo objects for all cached paintings.

    Raises:
        FileNotFoundError: If cache_dir does not exist.

    Contract:
        DISCOVERY:
        - Walks cache_dir/{source}/ directories
        - For each .jpg file, looks for corresponding .json sidecar
        - Returns PaintingInfo for each valid image+sidecar pair

        SIDEcar FORMAT:
        - JSON file contains all PaintingInfo fields except image_path
        - image_path is computed from source, source_id, and cache_dir
        - Missing sidecar files are logged as warnings and skipped

        PATH RESOLUTION:
        - cache_dir may contain ~ which is expanded
        - All paths in returned PaintingInfo are absolute

    Example:
        >>> paintings = list_cached_paintings(Path("~/.sfumato/paintings"))
        >>> len(paintings)
        150
        >>> paintings[0].source
        <ArtSource.RIJKSMUSEUM: 'rijksmuseum'>
    """
    # Contract-only stub - implementation deferred to subsequent step
    raise NotImplementedError(
        "list_cached_paintings contract defined in src/sfumato/paintings.py - "
        "implementation deferred to subsequent step"
    )


def content_hash(image_path: Path) -> str:
    """Compute SHA-256 hash of image file bytes.

    This is the POST-CACHE deduplication key.
    Two images with identical content but different source_ids will have
    the same content_hash.

    Args:
        image_path: Path to the image file.

    Returns:
        SHA-256 hex digest string (64 characters).

    Raises:
        FileNotFoundError: If image file does not exist.

    Contract:
        HASH ALGORITHM:
        - SHA-256 of raw image file bytes
        - Hex digest output (64 character string)
        - lowercase letters (standard hex)

        BYTE-IDENTITY:
        - Hash is computed on EXACT file bytes
        - No image decoding or re-encoding
        - Same image content from different sources → same hash
        - Different encodings of same visual content → different hash

        USAGE FOR DEDUP:
        - content_hash is the global deduplication key across all sources
        - Check this AFTER downloading to prevent duplicates
        - If hash matches existing cached painting, skip it

    Example:
        >>> # Two identical files have the same hash
        >>> h1 = content_hash(Path("/tmp/image1.jpg"))
        >>> h2 = content_hash(Path("/tmp/copy_of_image1.jpg"))
        >>> h1 == h2
        True
        >>> len(h1)
        64
        >>> h1[:8]
        'a1b2c3d4'
    """
    # Implementation: simple and stable enough to include even in contract step
    # This is a pure function with no external dependencies
    _validate_image_path(image_path)
    return _compute_sha256(image_path)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _validate_image_path(image_path: Path) -> None:
    """Validate that image_path exists and is a file.

    Args:
        image_path: Path to validate.

    Raises:
        FileNotFoundError: If path does not exist or is not a file.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {image_path}")


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of file bytes.

    Args:
        file_path: Path to the file.

    Returns:
        Lowercase hex digest string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()
