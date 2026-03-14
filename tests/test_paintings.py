"""Tests for the paintings module contracts and behavior.

This file contains CONTRACT TESTS that verify the behavioral boundaries
defined in src/sfumato/paintings.py. Each test documents expected behavior
without depending on implementation details.

Architecture reference: ARCHITECTURE.md#2.3
Contract reference: src/sfumato/paintings.py
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

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
        # Create a temporary file with known content
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            result = content_hash(temp_path)
            # SHA-256 hex digest is always 64 characters
            assert len(result) == 64
            # All lowercase hex
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
        # Create two files with different content
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
# CONTRACT: DEDUPLICATION
# =============================================================================


class TestDedupContract:
    """Test deduplication contract."""

    def test_exclude_ids_format(self) -> None:
        """exclude_ids format is '{source}:{source_id}'."""
        # This test documents the exclude_ids format:
        # Examples:
        # - "rijksmuseum:SK-A-3262"
        # - "met:436535"
        # - "wikimedia:Mona_Lisa"

        # The contract is documented in the docstring
        # Implementation will filter by this prefix:id format
        pass

    def test_content_hash_as_dedup_key(self) -> None:
        """content_hash is used as POST-CACHE deduplication key."""
        # This test documents the post-cache dedup behavior:
        # 1. Image is downloaded
        # 2. content_hash is computed
        # 3. If content_hash matches existing cached painting, skip it
        # 4. This handles same image from different sources

        # The contract is implemented in content_hash()
        # Implementation will use this for deduplication
        pass


# =============================================================================
# CONTRACT: ORIENTATION SEMANTICS
# =============================================================================


class TestOrientationSemanticsContract:
    """Test orientation semantics contract."""

    def test_square_is_portrait(self) -> None:
        """Square images (width == height) are classified as PORTRAIT."""
        # This test documents the orientation semantics:
        # - LANDSCAPE: width > height
        # - PORTRAIT: width <= height (includes square)
        # - Square is NOT a separate category

        # The contract is documented in the Orientation enum docstring
        # Implementation will use detect_orientation() at runtime
        pass


# =============================================================================
# CONTRACT: DEFERRED SOURCE-AUTH BEHAVIOR
# =============================================================================


class TestDeferredSourceAuthContract:
    """Test deferred source-auth behavior contract."""

    def test_rijksmuseum_requires_api_key(self) -> None:
        """Rijksmuseum source requires RIJKSMUSEUM_API_KEY env var."""
        # This test documents the auth requirement:
        # - Rijksmuseum: requires RIJKSMUSEUM_API_KEY
        # - If missing: raise SourceAuthError (or skip that source)
        # - Met: no key required
        # - Wikimedia: no key required

        # The contract is documented in fetch_from_rijksmuseum() docstring
        pass

    def test_met_no_key_required(self) -> None:
        """Met Museum source does not require API key."""
        # Met API is free and open
        # Implementation should never raise SourceAuthError for Met
        pass

    def test_wikimedia_no_key_required(self) -> None:
        """Wikimedia source does not require API key."""
        # Wikimedia API is free (with rate limiting)
        # Implementation should never raise SourceAuthError for Wikimedia
        pass


# =============================================================================
# CONTRACT: SOURCE FAILURE HANDLING
# =============================================================================


class TestSourceFailureContract:
    """Test source failure handling contract."""

    def test_individual_download_failures_are_logged_and_skipped(self) -> None:
        """ImageDownloadError is non-fatal - logged and skipped."""
        # This test documents the failure handling:
        # - Individual image download failures are logged
        # - The failed painting is skipped (not added to result)
        # - The overall fetch operation continues

        # The contract is documented in fetch_paintings() docstring
        pass

    def test_source_auth_error_is_skip_for_individual_source(self) -> None:
        """SourceAuthError causes that source to be skipped."""
        # For Rijksmuseum: missing API key = skip that source
        # For fetch_paintings(): if ALL sources fail auth, raise SourceAuthError
        pass

    def test_aggregate_success_returns_list(self) -> None:
        """fetch_paintings always returns a list (possibly empty)."""
        # Never raises on individual source failures
        # Returns empty list if all sources fail
        # Returns empty list if no paintings match criteria
        pass
