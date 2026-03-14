"""Fetch and cache paintings from cloud art APIs.

This module implements the paintings module contract from ARCHITECTURE.md#2.3:
- Fetch paintings from Rijksmuseum, Met Museum, Wikimedia Commons APIs
- Download high-resolution images to local cache with metadata sidecar
- Manage painting pool and track which paintings have been used
- Provide painting metadata for semantic matching

Architecture reference: ARCHITECTURE.md#2.3
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import httpx
from PIL import Image

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


class SourceAuthError(PaintingsError):
    """A source requires authentication that is not configured.

    Contract:
        - Raised when a source requires an API key but the key is missing
        - For Rijksmuseum: RIJKSMUSEUM_API_KEY env var not set
        - This is a SKIP scenario - the source should be excluded gracefully
    """


class ImageDownloadError(PaintingsError):
    """Failed to download an image after retries.

    Contract:
        - Individual download failures are logged and skipped
        - Does NOT fail the entire fetch operation
        - Contains source_id in error message for debugging
    """


# =============================================================================
# LOGGING
# =============================================================================


logger = logging.getLogger(__name__)


@dataclass
class _SourceCandidate:
    """Internal candidate representation before download."""

    source_id: str
    title: str
    artist: str
    year: str
    source_url: str
    image_url: str


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
    if count <= 0 or not sources:
        return []

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    dispatch: dict[str, _SourceFetcher] = {
        ArtSource.RIJKSMUSEUM.value: fetch_from_rijksmuseum,
        ArtSource.MET.value: fetch_from_met,
        ArtSource.WIKIMEDIA.value: fetch_from_wikimedia,
    }

    requested_sources = [source.strip().lower() for source in sources]
    known_sources = [source for source in requested_sources if source in dispatch]
    auth_errors: list[SourceAuthError] = []
    paintings: list[PaintingInfo] = []

    for source in requested_sources:
        fetcher = dispatch.get(source)
        if fetcher is None:
            logger.warning("Unknown painting source requested: %s", source)
            continue

        try:
            paintings.extend(await fetcher(count, resolved_cache_dir, exclude_ids))
        except SourceAuthError as exc:
            auth_errors.append(exc)
            logger.warning("Skipping source '%s' due to missing auth: %s", source, exc)
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning("Skipping source '%s' due to fetch failure: %s", source, exc)

    if paintings:
        return paintings

    if (
        auth_errors
        and known_sources
        and all(_source_requires_auth(source_name) for source_name in known_sources)
    ):
        raise auth_errors[0]

    return []


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
    api_key = os.getenv("RIJKSMUSEUM_API_KEY")
    if not api_key:
        raise SourceAuthError("RIJKSMUSEUM_API_KEY env var is not set")

    candidates = await _discover_rijksmuseum_candidates(count=count, api_key=api_key)
    return await _download_candidates(
        source=ArtSource.RIJKSMUSEUM,
        candidates=candidates,
        count=count,
        cache_dir=cache_dir,
        exclude_ids=exclude_ids,
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
    candidates = await _discover_met_candidates(count=count)
    return await _download_candidates(
        source=ArtSource.MET,
        candidates=candidates,
        count=count,
        cache_dir=cache_dir,
        exclude_ids=exclude_ids,
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
    candidates = await _discover_wikimedia_candidates(count=count)
    return await _download_candidates(
        source=ArtSource.WIKIMEDIA,
        candidates=candidates,
        count=count,
        cache_dir=cache_dir,
        exclude_ids=exclude_ids,
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
    _validate_image_path(image_path)
    with Image.open(image_path) as image:
        width, height = image.size
    if width > height:
        return Orientation.LANDSCAPE
    return Orientation.PORTRAIT


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
    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    if not resolved_cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {resolved_cache_dir}")

    paintings: list[PaintingInfo] = []
    for source in ArtSource:
        source_dir = resolved_cache_dir / source.value
        if not source_dir.exists() or not source_dir.is_dir():
            continue

        for image_path in sorted(source_dir.glob("*.jpg")):
            sidecar_path = image_path.with_suffix(".json")
            if not sidecar_path.exists():
                logger.warning("Skipping cached image without sidecar: %s", image_path)
                continue

            try:
                sidecar_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping malformed sidecar %s: %s", sidecar_path, exc)
                continue

            try:
                paintings.append(
                    _painting_info_from_sidecar(
                        cache_dir=resolved_cache_dir,
                        sidecar_data=sidecar_data,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid sidecar payload %s: %s", sidecar_path, exc
                )
                continue

    return paintings


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


type _SourceFetcher = Callable[
    [int, Path, set[str] | None],
    Awaitable[list[PaintingInfo]],
]


def _resolve_cache_dir(cache_dir: Path) -> Path:
    """Resolve cache directory into a stable absolute path."""
    return cache_dir.expanduser().resolve()


def _source_requires_auth(source_name: str) -> bool:
    """Return whether a source currently requires credentials."""
    return source_name == ArtSource.RIJKSMUSEUM.value


async def _download_candidates(
    source: ArtSource,
    candidates: list[_SourceCandidate],
    count: int,
    cache_dir: Path,
    exclude_ids: set[str] | None,
) -> list[PaintingInfo]:
    """Download source candidates, write cache entries, and return metadata."""
    if count <= 0:
        return []

    resolved_cache_dir = _resolve_cache_dir(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    existing = (
        list_cached_paintings(resolved_cache_dir) if resolved_cache_dir.exists() else []
    )
    existing_hashes = {painting.content_hash for painting in existing}

    filtered_candidates = [
        candidate
        for candidate in candidates
        if not _is_excluded(
            source=source, source_id=candidate.source_id, exclude_ids=exclude_ids
        )
    ]

    downloaded: list[PaintingInfo] = []
    for candidate in filtered_candidates:
        if len(downloaded) >= count:
            break

        try:
            painting = await _download_one(
                source=source,
                candidate=candidate,
                cache_dir=resolved_cache_dir,
            )
        except ImageDownloadError as exc:
            logger.warning(
                "Skipping failed download for %s:%s: %s",
                source.value,
                candidate.source_id,
                exc,
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive boundary
            logger.warning(
                "Skipping unusable metadata for %s:%s: %s",
                source.value,
                candidate.source_id,
                exc,
            )
            continue

        if painting.content_hash in existing_hashes and not _same_cache_entry_exists(
            existing=existing,
            candidate=painting,
        ):
            _cleanup_cache_entry(painting.image_path)
            continue

        existing_hashes.add(painting.content_hash)
        existing.append(painting)
        downloaded.append(painting)

    return downloaded


def _cleanup_cache_entry(image_path: Path) -> None:
    """Remove image and sidecar files for a duplicate cache entry."""
    image_path.unlink(missing_ok=True)
    image_path.with_suffix(".json").unlink(missing_ok=True)


def _is_excluded(
    source: ArtSource, source_id: str, exclude_ids: set[str] | None
) -> bool:
    """Return True when source/source_id is in caller exclusion set."""
    if not exclude_ids:
        return False
    return f"{source.value}:{source_id}" in exclude_ids


def _same_cache_entry_exists(
    existing: list[PaintingInfo], candidate: PaintingInfo
) -> bool:
    """Return True if the existing list already includes this source/source_id pair."""
    return any(
        entry.source == candidate.source and entry.source_id == candidate.source_id
        for entry in existing
    )


async def _download_one(
    source: ArtSource,
    candidate: _SourceCandidate,
    cache_dir: Path,
) -> PaintingInfo:
    """Download one candidate and write image + sidecar."""
    source_dir = cache_dir / source.value
    source_dir.mkdir(parents=True, exist_ok=True)

    image_path = source_dir / f"{candidate.source_id}.jpg"

    image_bytes = await _download_image_bytes(candidate.image_url)
    image_path.write_bytes(image_bytes)

    width, height = _image_dimensions(image_path)
    painting = PaintingInfo(
        image_path=image_path.resolve(),
        content_hash=content_hash(image_path),
        title=candidate.title,
        artist=candidate.artist,
        year=candidate.year,
        source=source,
        source_id=candidate.source_id,
        source_url=candidate.source_url,
        orientation=detect_orientation(image_path),
        width=width,
        height=height,
    )

    sidecar_path = image_path.with_suffix(".json")
    sidecar_path.write_text(
        json.dumps(
            _painting_to_sidecar_dict(painting), ensure_ascii=True, sort_keys=True
        ),
        encoding="utf-8",
    )
    return painting


def _painting_to_sidecar_dict(painting: PaintingInfo) -> dict[str, str | int]:
    """Serialize cache metadata payload without image_path field."""
    return {
        "content_hash": painting.content_hash,
        "title": painting.title,
        "artist": painting.artist,
        "year": painting.year,
        "source": painting.source.value,
        "source_id": painting.source_id,
        "source_url": painting.source_url,
        "orientation": painting.orientation.value,
        "width": painting.width,
        "height": painting.height,
    }


def _painting_info_from_sidecar(
    cache_dir: Path, sidecar_data: dict[str, object]
) -> PaintingInfo:
    """Reconstruct PaintingInfo from sidecar metadata and cache root."""
    source = ArtSource(str(sidecar_data["source"]))
    source_id = str(sidecar_data["source_id"])
    image_path = (cache_dir / source.value / f"{source_id}.jpg").resolve()
    return PaintingInfo(
        image_path=image_path,
        content_hash=str(sidecar_data["content_hash"]),
        title=str(sidecar_data["title"]),
        artist=str(sidecar_data["artist"]),
        year=str(sidecar_data["year"]),
        source=source,
        source_id=source_id,
        source_url=str(sidecar_data["source_url"]),
        orientation=Orientation(str(sidecar_data["orientation"])),
        width=int(str(sidecar_data["width"])),
        height=int(str(sidecar_data["height"])),
    )


def _image_dimensions(image_path: Path) -> tuple[int, int]:
    """Read image width/height from file metadata."""
    with Image.open(image_path) as image:
        return image.size


# Browser-like headers to avoid 403 from Wikimedia/Met (see PROTOTYPING.md#5)
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
}


async def _download_image_bytes(image_url: str) -> bytes:
    """Download raw image bytes from URL."""
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True, headers=_BROWSER_HEADERS
    ) as client:
        try:
            response = await client.get(image_url)
            response.raise_for_status()
        except Exception as exc:
            raise ImageDownloadError(
                f"Failed to download image from {image_url}: {exc}"
            ) from exc
    return response.content


async def _discover_rijksmuseum_candidates(
    count: int,
    api_key: str,
) -> list[_SourceCandidate]:
    """Discover candidate paintings from the Rijksmuseum API."""
    params = {
        "key": api_key,
        "format": "json",
        "imgonly": "True",
        "type": "painting",
        "ps": str(max(count * 4, 20)),
        "p": "1",
    }

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(
            "https://www.rijksmuseum.nl/api/en/collection",
            params=params,
        )
        response.raise_for_status()
        payload = response.json()

    candidates: list[_SourceCandidate] = []
    for item in payload.get("artObjects", []):
        object_number = str(item.get("objectNumber", "")).strip()
        web_image = item.get("webImage") or {}
        image_url = str(web_image.get("url", "")).strip()
        if not object_number or not image_url:
            continue

        candidates.append(
            _SourceCandidate(
                source_id=object_number,
                title=str(item.get("title", "Untitled")),
                artist=str(item.get("principalOrFirstMaker", "Unknown")),
                year=str((item.get("dating") or {}).get("presentingDate", "")),
                source_url=str((item.get("links") or {}).get("web", "")),
                image_url=image_url,
            )
        )
    return candidates


async def _discover_met_candidates(count: int) -> list[_SourceCandidate]:
    """Discover candidate paintings from the Met Museum API."""
    import asyncio

    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(
        timeout=timeout, headers=_BROWSER_HEADERS
    ) as client:
        search_response = await client.get(
            "https://collectionapi.metmuseum.org/public/collection/v1/search",
            params={
                "hasImages": "true",
                "q": "painting landscape",
                "isPublicDomain": "true",
            },
        )
        search_response.raise_for_status()
        search_payload = search_response.json()

        object_ids = search_payload.get("objectIDs") or []
        candidates: list[_SourceCandidate] = []
        for object_id in object_ids[: max(count * 5, 20)]:
            await asyncio.sleep(0.5)  # Rate limit: ~2 req/s
            object_response = await client.get(
                f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
            )
            object_response.raise_for_status()
            item = object_response.json()

            image_url = str(item.get("primaryImage", "")).strip()
            if not image_url or not item.get("isPublicDomain", False):
                continue

            candidates.append(
                _SourceCandidate(
                    source_id=str(item.get("objectID", object_id)),
                    title=str(item.get("title", "Untitled")),
                    artist=str(item.get("artistDisplayName", "Unknown")),
                    year=str(item.get("objectDate", "")),
                    source_url=str(item.get("objectURL", "")),
                    image_url=image_url,
                )
            )
            if len(candidates) >= max(count * 3, count):
                break

    return candidates


async def _discover_wikimedia_candidates(count: int) -> list[_SourceCandidate]:
    """Discover candidate paintings from Wikimedia Commons."""
    import asyncio

    timeout = httpx.Timeout(20.0)
    # Wikimedia "Featured pictures of paintings" has only subcategories,
    # not files directly. We query several country subcategories for actual files.
    subcategories = [
        "Category:Featured_pictures_of_paintings_from_France",
        "Category:Featured_pictures_of_paintings_from_the_Netherlands",
        "Category:Featured_pictures_of_paintings_from_Italy",
        "Category:Featured_pictures_of_paintings_from_Spain",
        "Category:Featured_pictures_of_paintings_from_Germany",
        "Category:Featured_pictures_of_paintings_from_the_United_Kingdom",
        "Category:Featured_pictures_of_paintings_from_the_United_States",
        "Category:Featured_pictures_of_paintings_from_Russia",
        "Category:Featured_pictures_of_paintings_from_Austria",
    ]

    async with httpx.AsyncClient(
        timeout=timeout, headers=_BROWSER_HEADERS
    ) as client:
        all_members: list[dict] = []
        for subcat in subcategories:
            if len(all_members) >= max(count * 5, 20):
                break
            await asyncio.sleep(1)
            members_response = await client.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "list": "categorymembers",
                    "cmtitle": subcat,
                    "cmtype": "file",
                    "cmlimit": str(max(count * 2, 10)),
                },
            )
            members_response.raise_for_status()
            members_payload = members_response.json()
            all_members.extend(
                members_payload.get("query", {}).get("categorymembers", [])
            )

        candidates: list[_SourceCandidate] = []
        for member in all_members:
            title = str(member.get("title", "")).strip()
            if not title.startswith("File:"):
                continue

            await asyncio.sleep(2)  # Wikimedia rate limit: ~1 req/2s
            image_response = await client.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata",
                    "titles": title,
                },
            )
            image_response.raise_for_status()
            image_payload = image_response.json()
            page_items = (
                image_payload.get("query", {}).get("pages", {}) or {}
            ).values()
            page = next(iter(page_items), {})
            image_info = (page.get("imageinfo") or [{}])[0]
            image_url = str(image_info.get("url", "")).strip()
            if not image_url:
                continue

            ext_metadata = image_info.get("extmetadata") or {}
            artist_raw = str((ext_metadata.get("Artist") or {}).get("value", "Unknown"))
            year_raw = str(
                (ext_metadata.get("DateTimeOriginal") or {}).get("value", "")
            )
            normalized_title = title.replace(" ", "_")
            candidates.append(
                _SourceCandidate(
                    source_id=normalized_title.removeprefix("File:"),
                    title=normalized_title.removeprefix("File:"),
                    artist=_strip_html(artist_raw),
                    year=_strip_html(year_raw),
                    source_url=f"https://commons.wikimedia.org/wiki/{normalized_title}",
                    image_url=image_url,
                )
            )
            if len(candidates) >= max(count * 3, count):
                break

    return candidates


def _strip_html(raw: str) -> str:
    """Strip simple HTML tags from metadata values."""
    result = []
    inside_tag = False
    for char in raw:
        if char == "<":
            inside_tag = True
            continue
        if char == ">":
            inside_tag = False
            continue
        if not inside_tag:
            result.append(char)
    return "".join(result).strip()
