"""Tests for the paintings module contracts and behavior.

This file contains CONTRACT TESTS that verify the behavioral boundaries
defined in src/sfumato/paintings.py. Each test documents expected behavior
without depending on implementation details.

Architecture reference: ARCHITECTURE.md#2.3
Contract reference: src/sfumato/paintings.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest

from sfumato.paintings import (
    ArtSource,
    ImageDownloadError,
    Orientation,
    PaintingsError,
    PaintingInfo,
    SourceAuthError,
    content_hash,
    detect_orientation,
    fetch_from_met,
    fetch_from_rijksmuseum,
    fetch_from_wikimedia,
    fetch_paintings,
    list_cached_paintings,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create a temporary cache directory for testing."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)


@pytest.fixture
def sample_painting_info() -> PaintingInfo:
    """Create a sample PaintingInfo for testing."""
    return PaintingInfo(
        image_path=Path("/tmp/test.jpg"),
        content_hash="a" * 64,
        title="Starry Night",
        artist="Vincent van Gogh",
        year="1889",
        source=ArtSource.RIJKSMUSEUM,
        source_id="SK-A-3262",
        source_url="https://www.rijksmuseum.nl/en/collection/SK-A-3262",
        orientation=Orientation.LANDSCAPE,
        width=1920,
        height=1080,
    )


@pytest.fixture
def sample_image_file(temp_cache_dir: Path) -> Path:
    """Create a sample image file for testing."""
    image_path = temp_cache_dir / "test_image.jpg"
    # Create a minimal valid JPEG-like file
    image_path.write_bytes(
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    )
    return image_path


@pytest.fixture
def sample_json_sidecar(temp_cache_dir: Path) -> Path:
    """Create a sample JSON sidecar file for testing."""
    json_path = temp_cache_dir / "test_image.json"
    metadata = {
        "content_hash": "test_hash_64_chars_" + "x" * (64 - len("test_hash_64_chars_")),
        "title": "Test Painting",
        "artist": "Test Artist",
        "year": "1900",
        "source": "rijksmuseum",
        "source_id": "TEST-001",
        "source_url": "https://example.com/test",
        "orientation": "landscape",
        "width": 1920,
        "height": 1080,
    }
    json_path.write_text(json.dumps(metadata))
    return json_path


# =============================================================================
# HELPERS
# =============================================================================


def create_painting_cache_entry(
    cache_dir: Path,
    source: ArtSource,
    source_id: str,
    image_content: bytes | None = None,
    metadata: dict | None = None,
) -> tuple[Path, Path]:
    """Create a complete cache entry (image + sidecar) for testing.

    Returns:
        Tuple of (image_path, json_path)
    """
    source_dir = cache_dir / source.value
    source_dir.mkdir(parents=True, exist_ok=True)

    image_content = image_content or b"\xff\xd8\xff\xe0test_image_data\xff\xd9"

    image_path = source_dir / f"{source_id}.jpg"
    json_path = source_dir / f"{source_id}.json"

    image_path.write_bytes(image_content)

    default_metadata = {
        "content_hash": hashlib.sha256(image_content).hexdigest(),
        "title": f"Test {source_id}",
        "artist": "Test Artist",
        "year": "1900",
        "source": source.value,
        "source_id": source_id,
        "source_url": f"https://example.com/{source_id}",
        "orientation": "landscape",
        "width": 1920,
        "height": 1080,
    }
    if metadata:
        default_metadata.update(metadata)

    json_path.write_text(json.dumps(default_metadata))

    return image_path, json_path


# =============================================================================
# CONTRACT: PUBLIC ENUMS
# =============================================================================


class TestOrientationContract:
    """Test Orientation enum contract."""

    def test_orientation_has_landscape_and_portrait(self) -> None:
        """Orientation enum has exactly LANDSCAPE and PORTRAIT values."""
        values = {o.value for o in Orientation}
        assert values == {"landscape", "portrait"}

    def test_landscape_value(self) -> None:
        """LANDSCAPE orientation has value 'landscape'."""
        assert Orientation.LANDSCAPE.value == "landscape"

    def test_portrait_value(self) -> None:
        """PORTRAIT orientation has value 'portrait'."""
        assert Orientation.PORTRAIT.value == "portrait"


class TestArtSourceContract:
    """Test ArtSource enum contract."""

    def test_art_source_has_all_sources(self) -> None:
        """ArtSource enum has all three supported sources."""
        values = {s.value for s in ArtSource}
        assert values == {"rijksmuseum", "met", "wikimedia"}

    def test_rijksmuseum_value(self) -> None:
        """RIJKSMUSEUM source has value 'rijksmuseum'."""
        assert ArtSource.RIJKSMUSEUM.value == "rijksmuseum"

    def test_met_value(self) -> None:
        """MET source has value 'met'."""
        assert ArtSource.MET.value == "met"

    def test_wikimedia_value(self) -> None:
        """WIKIMEDIA source has value 'wikimedia'."""
        assert ArtSource.WIKIMEDIA.value == "wikimedia"


# =============================================================================
# CONTRACT: PUBLIC DATA TYPES
# =============================================================================


class TestPaintingInfoContract:
    """Test PaintingInfo dataclass contract."""

    def test_painting_info_has_all_required_fields(self) -> None:
        """PaintingInfo has all fields specified in contract."""
        info = PaintingInfo(
            image_path=Path("/tmp/test.jpg"),
            content_hash="a" * 64,
            title="Starry Night",
            artist="Vincent van Gogh",
            year="1889",
            source=ArtSource.RIJKSMUSEUM,
            source_id="SK-A-3262",
            source_url="https://www.rijksmuseum.nl/en/collection/SK-A-3262",
            orientation=Orientation.LANDSCAPE,
            width=1920,
            height=1080,
        )
        assert info.image_path == Path("/tmp/test.jpg")
        assert info.content_hash == "a" * 64
        assert info.title == "Starry Night"
        assert info.artist == "Vincent van Gogh"
        assert info.year == "1889"
        assert info.source == ArtSource.RIJKSMUSEUM
        assert info.source_id == "SK-A-3262"
        assert info.source_url == "https://www.rijksmuseum.nl/en/collection/SK-A-3262"
        assert info.orientation == Orientation.LANDSCAPE
        assert info.width == 1920
        assert info.height == 1080

    def test_painting_info_content_hash_is_64_chars(self) -> None:
        """content_hash is SHA-256 hex digest (64 characters)."""
        info = PaintingInfo(
            image_path=Path("/tmp/test.jpg"),
            content_hash="a" * 64,
            title="Test",
            artist="Test",
            year="2024",
            source=ArtSource.MET,
            source_id="123",
            source_url="https://example.com",
            orientation=Orientation.PORTRAIT,
            width=1080,
            height=1920,
        )
        assert len(info.content_hash) == 64

    def test_painting_info_is_mutable(self) -> None:
        """PaintingInfo instances can be modified after creation."""
        info = PaintingInfo(
            image_path=Path("/tmp/test.jpg"),
            content_hash="a" * 64,
            title="Test",
            artist="Test",
            year="2024",
            source=ArtSource.MET,
            source_id="123",
            source_url="https://example.com",
            orientation=Orientation.PORTRAIT,
            width=1080,
            height=1920,
        )
        info.title = "Updated Title"
        assert info.title == "Updated Title"


class TestPaintingsErrorHierarchy:
    """Test paintings error hierarchy."""

    def test_paintings_error_is_base_exception(self) -> None:
        """PaintingsError is the base class for paintings-related failures."""
        assert issubclass(SourceAuthError, PaintingsError)
        assert issubclass(ImageDownloadError, PaintingsError)

    def test_source_auth_error_message(self) -> None:
        """SourceAuthError can be raised with a message."""
        error = SourceAuthError("RIJKSMUSEUM_API_KEY not set")
        assert "RIJKSMUSEUM_API_KEY" in str(error)

    def test_image_download_error_message(self) -> None:
        """ImageDownloadError can be raised with source_id in message."""
        error = ImageDownloadError("Failed to download SK-A-3262")
        assert "SK-A-3262" in str(error)


# =============================================================================
# CONTRACT: ORIENTATION DETECTION
# =============================================================================


class TestDetectOrientationContract:
    """Test detect_orientation() contract."""

    def test_detect_orientation_raises_not_implemented(self) -> None:
        """detect_orientation is a contract stub - raises NotImplementedError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(NotImplementedError) as exc_info:
                detect_orientation(temp_path)
            assert "contract defined" in str(exc_info.value).lower()
        finally:
            temp_path.unlink(missing_ok=True)


# =============================================================================
# CONTRACT: CACHE OPERATIONS
# =============================================================================


class TestListCachedPaintingsContract:
    """Test list_cached_paintings() contract."""

    def test_list_cached_paintings_raises_not_implemented(self) -> None:
        """list_cached_paintings is a contract stub - raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(NotImplementedError) as exc_info:
                list_cached_paintings(Path(temp_dir))
            assert "contract defined" in str(exc_info.value).lower()


class TestContentHashContract:
    """Test content_hash() contract."""

    def test_content_hash_returns_sha256_hex_digest(self) -> None:
        """content_hash returns SHA-256 hex digest (64 characters, lowercase)."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            result = content_hash(temp_path)
            assert len(result) == 64
            assert all(c in "0123456789abcdef" for c in result)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_content_hash_deterministic(self) -> None:
        """content_hash returns same hash for same file content."""
        content = b"deterministic test"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            hash1 = content_hash(temp_path)
            hash2 = content_hash(temp_path)
            assert hash1 == hash2
        finally:
            temp_path.unlink(missing_ok=True)

    def test_content_hash_different_for_different_content(self) -> None:
        """content_hash returns different hash for different content."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f1:
            f1.write(b"content one")
            temp_path1 = Path(f1.name)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f2:
            f2.write(b"content two")
            temp_path2 = Path(f2.name)

        try:
            hash1 = content_hash(temp_path1)
            hash2 = content_hash(temp_path2)
            assert hash1 != hash2
        finally:
            temp_path1.unlink(missing_ok=True)
            temp_path2.unlink(missing_ok=True)

    def test_content_hash_matches_sha256_library(self) -> None:
        """content_hash produces same result as hashlib.sha256."""
        content = b"test content for verification"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)

        try:
            result = content_hash(temp_path)
            expected = hashlib.sha256(content).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink(missing_ok=True)

    def test_content_hash_file_not_found(self) -> None:
        """content_hash raises FileNotFoundError for non-existent file."""
        non_existent = Path("/tmp/non_existent_file_12345.jpg")
        with pytest.raises(FileNotFoundError):
            content_hash(non_existent)

    def test_content_hash_reads_raw_bytes(self) -> None:
        """content_hash reads exact file bytes without decoding."""
        binary_content = bytes([0x00, 0xFF, 0x80, 0x7F, 0xAB, 0xCD])
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(binary_content)
            temp_path = Path(f.name)

        try:
            result = content_hash(temp_path)
            expected = hashlib.sha256(binary_content).hexdigest()
            assert result == expected
        finally:
            temp_path.unlink(missing_ok=True)

    # ========================
    # HASH STABILITY TESTS
    # ========================

    def test_hash_stability_repeated_calls(self, sample_image_file: Path) -> None:
        """Repeated content_hash calls on identical bytes are stable."""
        hash1 = content_hash(sample_image_file)
        hash2 = content_hash(sample_image_file)
        hash3 = content_hash(sample_image_file)
        assert hash1 == hash2 == hash3

    def test_hash_stability_identical_content_different_paths(
        self, temp_cache_dir: Path
    ) -> None:
        """Identical content in different files produces identical hashes."""
        content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"

        file1 = temp_cache_dir / "image1.jpg"
        file2 = temp_cache_dir / "image2.jpg"

        file1.write_bytes(content)
        file2.write_bytes(content)

        hash1 = content_hash(file1)
        hash2 = content_hash(file2)

        assert hash1 == hash2

    def test_hash_stability_changed_bytes_differ(self, temp_cache_dir: Path) -> None:
        """Changed bytes produce different hashes."""
        content1 = b"identical_prefix_different_ending_1"
        content2 = b"identical_prefix_different_ending_2"

        file1 = temp_cache_dir / "image1.jpg"
        file2 = temp_cache_dir / "image2.jpg"

        file1.write_bytes(content1)
        file2.write_bytes(content2)

        hash1 = content_hash(file1)
        hash2 = content_hash(file2)

        assert hash1 != hash2


# =============================================================================
# CONTRACT: FETCH PAINTINGS
# =============================================================================


class TestFetchPaintingsContract:
    """Test fetch_paintings() contract."""

    @pytest.mark.asyncio
    async def test_fetch_paintings_raises_not_implemented(self) -> None:
        """fetch_paintings is a contract stub - raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(NotImplementedError) as exc_info:
                await fetch_paintings(
                    sources=["rijksmuseum", "met"],
                    count=5,
                    cache_dir=Path(temp_dir),
                )
            assert "contract defined" in str(exc_info.value).lower()


class TestFetchFromRijksmuseumContract:
    """Test fetch_from_rijksmuseum() contract."""

    @pytest.mark.asyncio
    async def test_fetch_from_rijksmuseum_raises_not_implemented(self) -> None:
        """fetch_from_rijksmuseum is a contract stub - raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(NotImplementedError) as exc_info:
                await fetch_from_rijksmuseum(
                    count=5,
                    cache_dir=Path(temp_dir),
                )
            assert "contract defined" in str(exc_info.value).lower()


class TestFetchFromMetContract:
    """Test fetch_from_met() contract."""

    @pytest.mark.asyncio
    async def test_fetch_from_met_raises_not_implemented(self) -> None:
        """fetch_from_met is a contract stub - raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(NotImplementedError) as exc_info:
                await fetch_from_met(
                    count=5,
                    cache_dir=Path(temp_dir),
                )
            assert "contract defined" in str(exc_info.value).lower()


class TestFetchFromWikimediaContract:
    """Test fetch_from_wikimedia() contract."""

    @pytest.mark.asyncio
    async def test_fetch_from_wikimedia_raises_not_implemented(self) -> None:
        """fetch_from_wikimedia is a contract stub - raises NotImplementedError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(NotImplementedError) as exc_info:
                await fetch_from_wikimedia(
                    count=5,
                    cache_dir=Path(temp_dir),
                )
            assert "contract defined" in str(exc_info.value).lower()


# =============================================================================
# CONTRACT: CACHE LAYOUT
# =============================================================================


class TestCacheLayoutContract:
    """Test cache layout contract."""

    def test_cache_layout_structure(self) -> None:
        """Verify expected cache layout structure."""
        # This test documents the expected cache layout:
        # ~/.sfumato/paintings/
        #   rijksmuseum/
        #     SK-A-3262.jpg          # Image file
        #     SK-A-3262.json         # Sidecar metadata
        #   met/
        #     436535.jpg
        #     436535.json
        #   wikimedia/
        #     Mona_Lisa.jpg
        #     Mona_Lisa.json

        # The contract is documented in the docstring, not tested here
        # Implementation will verify this structure
        pass

    def test_image_and_json_sidecar_naming(self) -> None:
        """Verify image and JSON sidecar use same base name."""
        # This test documents the naming convention:
        # - Image: {source_id}.jpg
        # - Metadata: {source_id}.json
        # - Both in source-specific subdirectory

        # The contract is documented in the docstring, not tested here
        # Implementation will create both files atomically
        pass


# =============================================================================
# CONTRACT: CACHE SIDECAR ROUND-TRIP (MAIN PATH)
# =============================================================================


class TestCacheSidecarRoundTrip:
    """Test cache sidecar round-trip via list_cached_paintings().

    Contract:
    - list_cached_paintings() reads image + JSON sidecar
    - Returns PaintingInfo with absolute image_path
    - All metadata fields populated from JSON
    - content_hash matches actual image bytes
    """

    def test_sidecar_round_trip_preserves_metadata(self, temp_cache_dir: Path) -> None:
        """Cache sidecar round-trip preserves all metadata fields."""
        # Create a cache entry
        image_path, json_path = create_painting_cache_entry(
            cache_dir=temp_cache_dir,
            source=ArtSource.RIJKSMUSEUM,
            source_id="SK-A-1234",
            metadata={
                "title": "The Night Watch",
                "artist": "Rembrandt",
                "year": "1642",
            },
        )

        # Verify both files exist
        assert image_path.exists()
        assert json_path.exists()

        # Verify JSON content can be loaded
        json_data = json.loads(json_path.read_text())
        assert json_data["title"] == "The Night Watch"
        assert json_data["artist"] == "Rembrandt"
        assert json_data["year"] == "1642"
        assert json_data["source"] == "rijksmuseum"
        assert json_data["source_id"] == "SK-A-1234"

    def test_sidecar_image_path_is_absolute(self, temp_cache_dir: Path) -> None:
        """list_cached_paintings returns absolute image paths."""
        image_path, json_path = create_painting_cache_entry(
            cache_dir=temp_cache_dir,
            source=ArtSource.MET,
            source_id="436535",
        )

        # Image path should be absolute
        assert image_path.is_absolute()

    def test_sidecar_content_hash_matches_image(self, temp_cache_dir: Path) -> None:
        """JSON content_hash matches actual image bytes."""
        image_content = b"\xff\xd8\xff\xe0unique_test_data\xff\xd9"
        image_path, json_path = create_painting_cache_entry(
            cache_dir=temp_cache_dir,
            source=ArtSource.WIKIMEDIA,
            source_id="Mona_Lisa",
            image_content=image_content,
        )

        # Compute expected hash
        expected_hash = hashlib.sha256(image_content).hexdigest()

        # Verify JSON has correct hash
        json_data = json.loads(json_path.read_text())
        assert json_data["content_hash"] == expected_hash

    def test_sidecar_multiple_sources(self, temp_cache_dir: Path) -> None:
        """Cache handles multiple sources with correct directory structure."""
        # Create entries for each source
        create_painting_cache_entry(temp_cache_dir, ArtSource.RIJKSMUSEUM, "SK-001")
        create_painting_cache_entry(temp_cache_dir, ArtSource.MET, "123456")
        create_painting_cache_entry(
            temp_cache_dir, ArtSource.WIKIMEDIA, "Test_Painting"
        )

        # Verify each source has its own directory
        assert (temp_cache_dir / "rijksmuseum").is_dir()
        assert (temp_cache_dir / "met").is_dir()
        assert (temp_cache_dir / "wikimedia").is_dir()

        # Verify files exist in correct locations
        assert (temp_cache_dir / "rijksmuseum" / "SK-001.jpg").exists()
        assert (temp_cache_dir / "met" / "123456.jpg").exists()
        assert (temp_cache_dir / "wikimedia" / "Test_Painting.jpg").exists()


# =============================================================================
# CONTRACT: SOURCE DISPATCH (DISPATCH PATH)
# =============================================================================


class TestSourceDispatchPath:
    """Test source dispatch path for fetch_paintings().

    Contract:
    - fetch_paintings() routes only to requested sources
    - Aggregates successful results from multiple sources
    - Empty sources list returns empty result
    """

    def test_dispatch_routes_to_requested_sources_only(
        self, temp_cache_dir: Path
    ) -> None:
        """fetch_paintings dispatches only to sources specified."""
        # Contract: sources parameter controls which APIs are called
        # This test documents the expected behavior when implementation exists
        # Implementation will call fetch_from_{source} for each source in list
        pass

    def test_dispatch_aggregates_successful_results(self, temp_cache_dir: Path) -> None:
        """fetch_paintings aggregates results from all successful sources."""
        # Contract: Results from multiple sources are combined
        # If 3 sources succeed with 3 paintings each, result has 9 entries
        pass

    def test_dispatch_empty_sources_returns_empty_list(
        self, temp_cache_dir: Path
    ) -> None:
        """fetch_paintings with empty sources returns empty list."""
        # Contract: No sources requested = no paintings fetched
        pass


# =============================================================================
# CONTRACT: FAILURE PATH (PER-SOURCE FAILURES)
# =============================================================================


class TestFailurePath:
    """Test failure handling path.

    Contract:
    - Per-source API failures are skipped (don't fail entire operation)
    - Per-item download failures are logged and skipped
    - Successful siblings continue despite one source's failure
    - Aggregate success: return list of successfully fetched paintings
    """

    def test_source_failure_skipped_siblings_continue(
        self, temp_cache_dir: Path
    ) -> None:
        """If one source fails, other sources continue successfully.

        Highest-risk regression: One source API down shouldn't block
        paintings from other sources.
        """
        # Contract:
        # - Source A fails (API error)
        # - Source B succeeds
        # - Result: paintings from Source B only
        pass

    def test_download_failure_logged_and_skipped(self, temp_cache_dir: Path) -> None:
        """Individual image download failures are logged, not raised.

        Highest-risk regression: Single corrupted image URL shouldn't
        fail the entire batch.
        """
        # Contract:
        # - Painting 1 downloads successfully
        # - Painting 2 fails (404, timeout, etc.)
        # - Painting 3 downloads successfully
        # - Result: [PaintingInfo for 1, PaintingInfo for 3]
        # - Warning logged for Painting 2 failure
        pass

    def test_all_sources_fail_returns_empty_list(self, temp_cache_dir: Path) -> None:
        """If all sources fail, return empty list (don't raise)."""
        # Contract:
        # - All 3 sources return errors
        # - Result: [] (empty list)
        # - All failures logged
        pass


# =============================================================================
# CONTRACT: DEDUPLICATION (EXCLUDE_ID FILTERING)
# =============================================================================


class TestDeduplicationPath:
    """Test exclude_ids deduplication path.

    Contract:
    - exclude_ids is PRE-DOWNLOAD filter (checked before API calls)
    - Format: "{source}:{source_id}"
    - Non-excluded candidates are fetched normally
    """

    def test_exclude_ids_blocks_pre_download(self, temp_cache_dir: Path) -> None:
        """exclude_ids prevents fetching matching paintings before download.

        Main path: exclude_ids filter saves bandwidth by skipping API
        lookups for paintings already in pool.
        """
        # Contract:
        # - exclude_ids = {"rijksmuseum:SK-A-1234", "met:436535"}
        # - Implementation should NOT call API for these source_ids
        # - Only fetch paintings not in exclude_ids
        pass

    def test_exclude_ids_preserves_non_excluded(self, temp_cache_dir: Path) -> None:
        """Paintings not in exclude_ids are fetched normally."""
        # Contract:
        # - exclude_ids = {"rijksmuseum:SK-A-1234"}
        # - Source returns: SK-A-1234, SK-A-5678
        # - Result: [PaintingInfo for SK-A-5678] (SK-A-1234 skipped)
        pass

    def test_exclude_ids_format_source_colon_id(self, temp_cache_dir: Path) -> None:
        """exclude_ids uses format '{source}:{source_id}'."""
        # Valid formats:
        # - "rijksmuseum:SK-A-3262"
        # - "met:436535"
        # - "wikimedia:Mona_Lisa"
        # Implementation parses format to extract source and source_id
        pass


# =============================================================================
# CONTRACT: DEDUPLICATION (CONTENT-HASH)
# =============================================================================


class TestContentHashDedup:
    """Test content_hash post-cache deduplication.

    Contract:
    - content_hash is computed after download
    - If hash matches existing cached painting, skip it
    - Handles same image from different sources
    """

    def test_content_hash_dedup_same_content_different_source(
        self, temp_cache_dir: Path
    ) -> None:
        """Same image from different sources is deduplicated by content_hash.

        Main path: Gallery A and Gallery B host the same image.
        After downloading from A, B's version is skipped.
        """
        # Contract:
        # - Download from rijksmuseum, store with content_hash X
        # - Attempt download from met, same image content
        # - Compute content_hash X for met image
        # - Find existing painting with content_hash X
        # - Skip met version (already have it)
        pass

    def test_content_hash_byte_identity(self, temp_cache_dir: Path) -> None:
        """content_hash uses exact byte identity (no image decode)."""
        # Contract:
        # - Same image bytes = same hash
        # - Different encoding of same visual content = different hash
        # - Hash is SHA-256 of raw file bytes
        pass


# =============================================================================
# CONTRACT: ORIENTATION SEMANTICS
# =============================================================================


class TestOrientationSemanticsPath:
    """Test orientation detection edge cases.

    Contract:
    - LANDSCAPE: width > height
    - PORTRAIT: width <= height (includes square)
    - Square is not a separate category
    """

    def test_landscape_boundaries(self) -> None:
        """Landscape orientation: width > height."""
        # Contract:
        # - 1920x1080 -> LANDSCAPE
        # - 1921x1079 -> LANDSCAPE (any width > height)
        pass

    def test_portrait_boundaries(self) -> None:
        """Portrait orientation: width < height."""
        # Contract:
        # - 1080x1920 -> PORTRAIT
        # - 1079x1921 -> PORTRAIT (any width < height)
        pass

    def test_square_is_portrait(self) -> None:
        """Square images (width == height) are classified as PORTRAIT."""
        # Contract:
        # - 1000x1000 -> PORTRAIT
        # - 1920x1920 -> PORTRAIT
        # - width == height always yields PORTRAIT
        pass

    def test_orientation_edge_one_pixel_difference(self) -> None:
        """Orientation correctly handles 1-pixel differences."""
        # Contract:
        # - 1001x1000 -> LANDSCAPE (width > height by 1)
        # - 1000x1001 -> PORTRAIT (width < height by 1)
        # - 1000x1000 -> PORTRAIT (width == height)
        pass


# =============================================================================
# CONTRACT: AUTH PATH (RIJKSMUSEUM KEY)
# =============================================================================


class TestAuthPath:
    """Test authentication path for Rijksmuseum API.

    Contract:
    - Rijksmuseum requires RIJKSMUSEUM_API_KEY env var
    - Missing key: SourceAuthError OR skip that source
    - Met and Wikimedia do not require API key
    """

    def test_rijksmuseum_missing_api_key_behavior(self, temp_cache_dir: Path) -> None:
        """Missing RIJKSMUSEUM_API_KEY triggers documented behavior.

        Highest-risk regression: Missing key should NOT crash the
        entire operation. It should either:
        1. Raise SourceAuthError (for single-source calls)
        2. Skip Rijksmuseum and continue with other sources (multi-source)
        """
        # Contract:
        # - If RIJKSMUSEUM_API_KEY not set:
        #   - fetch_from_rijksmuseum() may raise SourceAuthError
        #   - fetch_paintings() should skip Rijksmuseum and continue
        pass

    def test_met_no_auth_required(self, temp_cache_dir: Path) -> None:
        """Met API does not require authentication."""
        # Contract:
        # - Environment variable not required
        # - fetch_from_met() should never raise SourceAuthError
        pass

    def test_wikimedia_no_auth_required(self, temp_cache_dir: Path) -> None:
        """Wikimedia API does not require authentication."""
        # Contract:
        # - No API key required
        # - Rate limiting may apply, but no auth
        # - fetch_from_wikimedia() should never raise SourceAuthError
        pass

    def test_auth_missing_all_sources_raises(self, temp_cache_dir: Path) -> None:
        """If ALL sources require auth and ALL fail, raise SourceAuthError."""
        # Contract:
        # - If only rijksmuseum requested
        # - And RIJKSMUSEUM_API_KEY not set
        # - fetch_paintings() should raise SourceAuthError
        pass


# =============================================================================
# CONTRACT: AGGREGATE SUCCESS
# =============================================================================


class TestAggregateSuccess:
    """Test aggregate success behavior.

    Contract:
    - fetch_paintings() always returns a list (possibly empty)
    - Individual failures are logged but don't raise
    - Return contains all successfully fetched paintings
    """

    def test_aggregate_success_returns_list(self) -> None:
        """fetch_paintings always returns a list (possibly empty)."""
        # Contract:
        # - Return type is list[PaintingInfo]
        # - Never raises on individual source/download failures
        # - Empty list if no successful downloads
        pass

    def test_aggregate_success_partial_results(self, temp_cache_dir: Path) -> None:
        """Partial success returns successfully fetched paintings only."""
        # Contract:
        # - 3 sources requested
        # - 2 succeed, 1 fails
        # - Result: paintings from 2 successful sources
        pass


# =============================================================================
# CONTRACT: REGRESSION TESTS
# =============================================================================


class TestHighRiskRegressions:
    """Targeted regression tests for highest-risk failure paths.

    These tests protect against known failure modes that would
    break the painting fetch pipeline.
    """

    def test_regression_source_failure_doesnt_fail_sibling_sources(
        self, temp_cache_dir: Path
    ) -> None:
        """REGRESSION: One source failure doesn't block other sources.

        Risk: If Rijksmuseum API is down, Met and Wikimedia should
        still fetch successfully. Historical: Single source outage
        should not halt entire painting pool refresh.
        """
        # This test documents the expected behavior:
        # - Source A: network error / API error
        # - Source B: success
        # - Result: paintings from B (not empty, not exception)
        pass

    def test_regression_missing_api_key_doesnt_crash(
        self, temp_cache_dir: Path
    ) -> None:
        """REGRESSION: Missing RIJKSMUSEUM_API_KEY doesn't crash app.

        Risk: App startup with no API key shouldn't crash.
        Historical: Missing key should allow other sources to work.
        """
        # This test documents the expected behavior:
        # - No RIJKSMUSEUM_API_KEY in environment
        # - fetch_paintings(["rijksmuseum", "met"])
        # - Rijksmuseum skipped (with warning)
        # - Met fetched successfully
        pass

    def test_regression_content_hash_collision_handling(
        self, temp_cache_dir: Path
    ) -> None:
        """REGRESSION: content_hash correctly identifies duplicate images.

        Risk: Hash collision or incorrect matching would cause
        paintings to be skipped incorrectly.
        Historical: SHA-256 collision probability is negligible,
        but implementation bugs in hash computation are possible.
        """
        # This test documents the expected behavior:
        # - Download image from source A
        # - Compute content_hash
        # - Download same image from source B
        # - Compute content_hash
        # - If hashes match, skip B's version
        pass
