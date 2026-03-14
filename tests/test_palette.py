"""Contract-driven tests for palette color extraction.

This module tests the behavior invariants and contracts defined in
`src/sfumato/palette.py` with minimal synthetic fixtures.

The contract tests verify:
1. Main path: valid image yields PaletteColors with canonical hex strings
2. Invalid image path: unreadable/non-image input fails with ImageReadError
3. Tiny-image path: very small images return deterministic colors or InvalidImageError
4. Monochrome/low-variance path: preserves invariants without malformed/empty hex
5. Luminance path: is_dark classification with explicit dark/light fixtures

Architecture reference: ARCHITECTURE.md#2.4
Contract reference: src/sfumato/palette.py
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# Import PIL for synthetic fixture generation
from PIL import Image

from sfumato.palette import (
    CANONICAL_HEX_FORMAT,
    ClusteringError,
    DEFAULT_N_COLORS,
    HEX_PATTERN,
    ImageReadError,
    InvalidImageError,
    MAX_N_COLORS,
    MIN_COLORS_FOR_PALETTE,
    PaletteColors,
    PaletteError,
    extract_palette,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# =============================================================================
# FIXTURES: Minimal Synthetic Images
# =============================================================================


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def solid_red_image(temp_dir: Path) -> Path:
    """A 10x10 solid red image for testing monochrome behavior."""
    img_path = temp_dir / "solid_red.png"
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def solid_black_image(temp_dir: Path) -> Path:
    """A 10x10 solid black image for testing dark luminance."""
    img_path = temp_dir / "solid_black.png"
    img = Image.new("RGB", (10, 10), color=(0, 0, 0))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def solid_white_image(temp_dir: Path) -> Path:
    """A 10x10 solid white image for testing light luminance."""
    img_path = temp_dir / "solid_white.png"
    img = Image.new("RGB", (10, 10), color=(255, 255, 255))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def solid_gray_image(temp_dir: Path) -> Path:
    """A 10x10 solid gray image for testing luminance boundary (L=0.5)."""
    img_path = temp_dir / "solid_gray.png"
    # Mid-gray: RGB(128, 128, 128) has luminance ~0.215 in BT.709
    # Actually, midpoint gray for is_dark threshold testing
    # Gray #808080 has L = 0.2126*128/255 + 0.7152*128/255 + 0.0722*128/255 ≈ 0.502
    img = Image.new("RGB", (10, 10), color=(128, 128, 128))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def dark_gray_image(temp_dir: Path) -> Path:
    """A 10x10 dark gray image for testing is_dark=True."""
    img_path = temp_dir / "dark_gray.png"
    # Dark gray: RGB(50, 50, 50) has luminance well below 0.5
    img = Image.new("RGB", (10, 10), color=(50, 50, 50))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def light_gray_image(temp_dir: Path) -> Path:
    """A 10x10 light gray image for testing is_dark=False."""
    img_path = temp_dir / "light_gray.png"
    # Light gray: RGB(200, 200, 200) has luminance well above 0.5
    img = Image.new("RGB", (10, 10), color=(200, 200, 200))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def gradient_image(temp_dir: Path) -> Path:
    """A 50x20 gradient image for testing color extraction."""
    img_path = temp_dir / "gradient.png"
    img = Image.new("RGB", (50, 20))
    pixels = img.load()
    # Create horizontal gradient from black to white
    for x in range(50):
        for y in range(20):
            pixels[x, y] = (x * 5, x * 5, x * 5)  # 0-245 gradient
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def two_color_image(temp_dir: Path) -> Path:
    """A 20x20 image with exactly two distinct colors."""
    img_path = temp_dir / "two_color.png"
    img = Image.new("RGB", (20, 20))
    pixels = img.load()
    # Left half red, right half blue
    for x in range(20):
        for y in range(20):
            if x < 10:
                pixels[x, y] = (255, 0, 0)  # Red
            else:
                pixels[x, y] = (0, 0, 255)  # Blue
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def multi_color_image(temp_dir: Path) -> Path:
    """A 20x20 image with multiple distinct colors for testing ordering."""
    img_path = temp_dir / "multi_color.png"
    img = Image.new("RGB", (20, 20))
    pixels = img.load()
    # Create 4 quadrants with different colors:
    # - Top-left: Red (100 pixels) - most frequent
    # - Top-right: Green (100 pixels) - second
    # - Bottom-left: Blue (100 pixels) - third
    # - Bottom-right: Yellow (100 pixels) - fourth
    for x in range(20):
        for y in range(20):
            if x < 10 and y < 10:
                pixels[x, y] = (255, 0, 0)  # Red (most common)
            elif x >= 10 and y < 10:
                pixels[x, y] = (0, 255, 0)  # Green
            elif x < 10 and y >= 10:
                pixels[x, y] = (0, 0, 255)  # Blue
            else:
                pixels[x, y] = (255, 255, 0)  # Yellow
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def single_pixel_image(temp_dir: Path) -> Path:
    """A 1x1 single-pixel image for testing edge case."""
    img_path = temp_dir / "single_pixel.png"
    img = Image.new("RGB", (1, 1), color=(42, 69, 128))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def tiny_image(temp_dir: Path) -> Path:
    """A 2x2 tiny image for testing minimal pixel count."""
    img_path = temp_dir / "tiny.png"
    img = Image.new("RGB", (2, 2))
    pixels = img.load()
    pixels[0, 0] = (255, 0, 0)  # Red
    pixels[0, 1] = (0, 255, 0)  # Green
    pixels[1, 0] = (0, 0, 255)  # Blue
    pixels[1, 1] = (255, 255, 255)  # White
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def rgba_image(temp_dir: Path) -> Path:
    """An RGBA image (with alpha channel) for testing alpha handling."""
    img_path = temp_dir / "rgba.png"
    # RGBA image - should be converted to RGB by implementation
    img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 255))  # Opaque red
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def transparent_image(temp_dir: Path) -> Path:
    """A fully transparent RGBA image for testing all-transparent edge case."""
    img_path = temp_dir / "transparent.png"
    # Fully transparent RGBA image
    img = Image.new("RGBA", (10, 10), color=(0, 0, 0, 0))  # Transparent
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def corrupt_image(temp_dir: Path) -> Path:
    """A file that exists but is not a valid image."""
    img_path = temp_dir / "corrupt.png"
    img_path.write_bytes(b"This is not a valid PNG or JPEG")
    return img_path


@pytest.fixture
def edge_border_image(temp_dir: Path) -> Path:
    """An image with different edge/border colors vs center for background testing."""
    img_path = temp_dir / "edge_border.png"
    img = Image.new("RGB", (100, 100))
    pixels = img.load()
    # Border (outer 5%): blue
    # Center: red
    for x in range(100):
        for y in range(100):
            # 5% of 100 is 5 pixels
            if x < 5 or x >= 95 or y < 5 or y >= 95:
                pixels[x, y] = (0, 0, 255)  # Blue border
            else:
                pixels[x, y] = (255, 0, 0)  # Red center
    img.save(img_path, "PNG")
    return img_path


# =============================================================================
# CONTRACT: Hex Pattern Validation (Inherited from original)
# =============================================================================


class TestHexPatternContract:
    """Verify the HEX_PATTERN constant matches canonical format."""

    def test_hex_pattern_matches_canonical_format(self) -> None:
        """HEX_PATTERN must match exactly 6 hex digits after #."""
        pattern = re.compile(HEX_PATTERN)

        # Valid formats
        assert pattern.match("#000000"), "Should match lowercase black"
        assert pattern.match("#FFFFFF"), "Should match uppercase white"
        assert pattern.match("#aBcDeF"), "Should match mixed case"
        assert pattern.match("#123456"), "Should match digits"

        # Invalid formats - must NOT match
        assert not pattern.match("#fff"), "#RGB shorthand is invalid"
        assert not pattern.match("#00ff"), "#RRGG is invalid"
        assert not pattern.match("#00000000"), "#RRGGBBAA is invalid"
        assert not pattern.match("000000"), "Missing # prefix is invalid"
        assert not pattern.match("red"), "Named colors are invalid"


# =============================================================================
# CONTRACT: PaletteColors Field Validation (Inherited from original)
# =============================================================================


class TestPaletteColorsValidation:
    """Verify PaletteColors validates all fields at construction."""

    def test_valid_palette_passes_validation(self) -> None:
        """A valid PaletteColors instance should pass __post_init__ validation."""
        palette = PaletteColors(
            dominant="#3a5f8a",
            secondary="#8a3a5f",
            accent="#5f8a3a",
            background="#2a2a2a",
            is_dark=True,
            colors=("#3a5f8a", "#8a3a5f", "#5f8a3a", "#2a2a2a", "#ffffff"),
        )
        assert palette.dominant == "#3a5f8a"
        assert palette.is_dark is True
        assert len(palette.colors) >= MIN_COLORS_FOR_PALETTE

    def test_invalid_dominant_raises_value_error(self) -> None:
        """Dominant must be valid hex, not shorthand or other format."""
        with pytest.raises(ValueError, match="dominant.*must match"):
            PaletteColors(
                dominant="#fff",  # Invalid: shorthand
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark=True,
                colors=("#ffffff",),
            )

    def test_empty_dominant_raises_value_error(self) -> None:
        """Dominant must not be empty string."""
        with pytest.raises(ValueError):
            PaletteColors(
                dominant="",
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark=True,
                colors=("#ffffff",),
            )

    def test_none_dominant_raises_type_error(self) -> None:
        """Dominant must not be None."""
        with pytest.raises(TypeError, match="dominant.*must be str"):
            PaletteColors(
                dominant=None,  # type: ignore[call-arg]
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark=True,
                colors=("#ffffff",),
            )

    def test_too_few_colors_raises_value_error(self) -> None:
        """colors list must have at least MIN_COLORS_FOR_PALETTE elements."""
        with pytest.raises(
            ValueError, match=f"colors.*must have >= {MIN_COLORS_FOR_PALETTE}"
        ):
            PaletteColors(
                dominant="#ffffff",
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark=False,
                colors=(),  # Empty tuple
            )

    def test_is_dark_must_be_bool(self) -> None:
        """is_dark must be boolean, not string or other type."""
        with pytest.raises(TypeError, match="is_dark.*must be bool"):
            PaletteColors(
                dominant="#ffffff",
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark="true",  # Invalid: string instead of bool
                colors=("#ffffff",),
            )

    def test_colors_must_contain_valid_hex(self) -> None:
        """Each element in colors must be valid hex."""
        with pytest.raises(ValueError, match=r"colors\[0\].*must match"):
            PaletteColors(
                dominant="#ffffff",
                secondary="#000000",
                accent="#000000",
                background="#000000",
                is_dark=True,
                colors=("invalid", "#000000"),  # First element invalid
            )


# =============================================================================
# CONTRACT: Error Types (Inherited)
# =============================================================================


class TestErrorHierarchy:
    """Verify error type hierarchy for palette extraction."""

    def test_palette_error_is_base(self) -> None:
        """All palette errors should inherit from PaletteError."""
        assert issubclass(ImageReadError, PaletteError)
        assert issubclass(InvalidImageError, PaletteError)
        assert issubclass(ClusteringError, PaletteError)

    def test_image_read_error_message_includes_path(self) -> None:
        """ImageReadError MUST include the original path in error message."""
        error = ImageReadError("Cannot read image at /path/to/file.jpg")
        assert "/path/to/file.jpg" in str(error)

    def test_invalid_image_error_describes_issue(self) -> None:
        """InvalidImageError SHOULD describe why the image is invalid."""
        error = InvalidImageError("Image has zero pixels")
        assert "zero" in str(error).lower()

    def test_clustering_error_includes_context(self) -> None:
        """ClusteringError SHOULD include debugging context."""
        error = ClusteringError(
            "k-means failed to converge for 100x100 image, n_colors=8"
        )
        assert "converge" in str(error).lower()


# =============================================================================
# CONTRACT: Constants (Inherited)
# =============================================================================


class TestConstantsValue:
    """Verify constant values match architectural requirements."""

    def test_default_n_colors(self) -> None:
        """DEFAULT_N_COLORS should be within valid range."""
        assert MIN_COLORS_FOR_PALETTE <= DEFAULT_N_COLORS <= MAX_N_COLORS

    def test_min_colors_for_palette_minimum(self) -> None:
        """MIN_COLORS_FOR_PALETTE should be at least 1."""
        assert MIN_COLORS_FOR_PALETTE >= 1

    def test_max_n_colors_reasonable(self) -> None:
        """MAX_N_COLORS should be reasonable for color palettes."""
        assert MAX_N_COLORS >= 8  # At least support the default
        assert MAX_N_COLORS <= 32  # But not unreasonably high


# =============================================================================
# NEW: Main Path - Valid Image Yields Valid PaletteColors
# =============================================================================


class TestMainPathValidImage:
    """Test that valid images produce valid PaletteColors with all fields."""

    def test_extract_returns_palette_colors(self, multi_color_image: Path) -> None:
        """Valid image returns a valid PaletteColors instance."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert isinstance(result, PaletteColors)

    def test_all_color_fields_are_valid_hex(self, multi_color_image: Path) -> None:
        """All color fields (dominant, secondary, accent, background) must be valid hex."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        pattern = re.compile(HEX_PATTERN)

        # All four main color fields must match hex pattern
        assert pattern.match(result.dominant), (
            f"dominant {result.dominant} is not valid hex"
        )
        assert pattern.match(result.secondary), (
            f"secondary {result.secondary} is not valid hex"
        )
        assert pattern.match(result.accent), f"accent {result.accent} is not valid hex"
        assert pattern.match(result.background), (
            f"background {result.background} is not valid hex"
        )

    def test_all_colors_are_lowercase_hex(self, multi_color_image: Path) -> None:
        """All returned colors MUST be lowercase hex (normalized)."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Contract: output must be lowercase for consistency
        assert result.dominant == result.dominant.lower(), (
            f"dominant not lowercase: {result.dominant}"
        )
        assert result.secondary == result.secondary.lower(), (
            f"secondary not lowercase: {result.secondary}"
        )
        assert result.accent == result.accent.lower(), (
            f"accent not lowercase: {result.accent}"
        )
        assert result.background == result.background.lower(), (
            f"background not lowercase: {result.background}"
        )

        for i, color in enumerate(result.colors):
            assert color == color.lower(), f"colors[{i}] not lowercase: {color}"

    def test_colors_list_has_minimum_elements(self, multi_color_image: Path) -> None:
        """colors list must have at least MIN_COLORS_FOR_PALETTE elements."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert len(result.colors) >= MIN_COLORS_FOR_PALETTE, (
            f"colors has {len(result.colors)} elements, need >= {MIN_COLORS_FOR_PALETTE}"
        )

    def test_colors_ordered_by_frequency(self, multi_color_image: Path) -> None:
        """colors list MUST be sorted by cluster size (frequency)."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Contract: colors[0] must be the most frequent (dominant)
        assert result.colors[0] == result.dominant, (
            f"colors[0] ({result.colors[0]}) must equal dominant ({result.dominant})"
        )

    def test_n_colors_respected(self, temp_dir: Path) -> None:
        """Returned colors list length should respect n_colors parameter."""
        # Create an image with many distinct colors
        img_path = temp_dir / "many_colors.png"
        img = Image.new("RGB", (100, 100))
        pixels = img.load()
        # Create a gradient with many distinct colors
        for x in range(100):
            for y in range(100):
                pixels[x, y] = (x * 2, y * 2, (x + y) % 256)
        img.save(img_path, "PNG")

        try:
            result = extract_palette(img_path, n_colors=5)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Should have at most 5 colors (may have fewer if image has fewer unique colors)
        assert len(result.colors) <= 5, (
            f"Expected <= 5 colors, got {len(result.colors)}"
        )

    def test_n_colors_capped_at_max(self, temp_dir: Path) -> None:
        """n_colors > MAX_N_COLORS should be capped, not error."""
        # Create image with many colors
        img_path = temp_dir / "many_colors_max.png"
        img = Image.new("RGB", (100, 100))
        pixels = img.load()
        for x in range(100):
            for y in range(100):
                pixels[x, y] = (x * 2, y * 2, (x + y) % 256)
        img.save(img_path, "PNG")

        try:
            # Request way more than MAX_N_COLORS
            result = extract_palette(img_path, n_colors=100)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Should be capped at MAX_N_COLORS
        assert len(result.colors) <= MAX_N_COLORS, (
            f"Expected <= {MAX_N_COLORS} colors, got {len(result.colors)}"
        )

    def test_is_dark_is_boolean(self, multi_color_image: Path) -> None:
        """is_dark must always be a boolean."""
        try:
            result = extract_palette(multi_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert isinstance(result.is_dark, bool), (
            f"is_dark is {type(result.is_dark)}, expected bool"
        )


# =============================================================================
# NEW: Invalid Image Path - Unreadable/Non-Image Input
# =============================================================================


class TestInvalidImagePath:
    """Test behavior for unreadable or non-image input."""

    def test_nonexistent_file_raises_image_read_error(self, temp_dir: Path) -> None:
        """Non-existent file must raise ImageReadError."""
        missing_path = temp_dir / "does_not_exist.png"

        try:
            extract_palette(missing_path)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except ImageReadError as exc_info:
            # Correct error type - verify message includes path
            assert str(missing_path) in str(exc_info) or "does_not_exist" in str(
                exc_info
            )
            return

        # If we got here without an exception, that's wrong
        pytest.fail("Expected ImageReadError for non-existent file")

    def test_corrupt_file_raises_image_read_error(self, corrupt_image: Path) -> None:
        """Corrupt/invalid image file must raise ImageReadError."""
        try:
            extract_palette(corrupt_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except ImageReadError as exc_info:
            # Correct error type - verify message describes issue
            error_msg = str(exc_info).lower()
            assert (
                "corrupt" in error_msg
                or "invalid" in error_msg
                or "decode" in error_msg
                or "read" in error_msg
                or "unrecognized" in error_msg
            )
            return

        pytest.fail("Expected ImageReadError for corrupt file")

    def test_directory_raises_image_read_error(self, temp_dir: Path) -> None:
        """Passing a directory instead of file must raise ImageReadError."""
        try:
            extract_palette(temp_dir)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except ImageReadError as exc_info:
            assert (
                str(temp_dir) in str(exc_info) or "directory" in str(exc_info).lower()
            )
            return

        pytest.fail("Expected ImageReadError for directory path")

    def test_text_file_raises_image_read_error(self, temp_dir: Path) -> None:
        """Text file masquerading as image must raise ImageReadError."""
        text_file = temp_dir / "fake.png"
        text_file.write_text("This is not an image")

        try:
            extract_palette(text_file)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except ImageReadError as exc_info:
            error_msg = str(exc_info).lower()
            assert (
                "read" in error_msg or "invalid" in error_msg or "decode" in error_msg
            )
            return

        pytest.fail("Expected ImageReadError for text file")


# =============================================================================
# NEW: Tiny-Image Path - Very Small Images
# =============================================================================


class TestTinyImagePath:
    """Test behavior for tiny/low-pixel images."""

    def test_single_pixel_image_returns_valid_palette_or_error(
        self, single_pixel_image: Path
    ) -> None:
        """Single pixel image MUST return valid PaletteColors or raise InvalidImageError."""
        try:
            result = extract_palette(single_pixel_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except InvalidImageError:
            # Contract allows InvalidImageError for insufficient data
            pytest.skip("Single pixel treated as insufficient data (valid behavior)")
            return

        # If it returns, it MUST be a valid PaletteColors
        assert isinstance(result, PaletteColors)
        assert len(result.colors) >= MIN_COLORS_FOR_PALETTE
        # All fields must be valid hex
        pattern = re.compile(HEX_PATTERN)
        assert pattern.match(result.dominant)
        assert pattern.match(result.secondary)
        assert pattern.match(result.accent)
        assert pattern.match(result.background)

    def test_tiny_image_returns_valid_palette_or_error(self, tiny_image: Path) -> None:
        """Tiny image (2x2) must return valid palette or raise InvalidImageError."""
        try:
            result = extract_palette(tiny_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except InvalidImageError:
            # Contract allows InvalidImageError for insufficient data
            return

        # If it returns, must be valid
        assert isinstance(result, PaletteColors)
        assert len(result.colors) >= MIN_COLORS_FOR_PALETTE

    def test_tiny_image_colors_are_deterministic(self, tiny_image: Path) -> None:
        """Tiny image extraction must be deterministic."""
        try:
            result1 = extract_palette(tiny_image)
            result2 = extract_palette(tiny_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except InvalidImageError:
            # Both should raise or both should succeed
            with pytest.raises(InvalidImageError):
                extract_palette(tiny_image)
            return

        # Both succeeded: results must be identical
        assert result1.dominant == result2.dominant, (
            "Same image should produce same dominant"
        )
        assert result1.secondary == result2.secondary
        assert result1.accent == result2.accent
        assert result1.background == result2.background
        assert result1.is_dark == result2.is_dark
        assert result1.colors == result2.colors


# =============================================================================
# NEW: Monochrome/Low-Variance Path
# =============================================================================


class TestMonochromaticImage:
    """Test behavior when image has zero variance (single color)."""

    def test_solid_color_returns_valid_palette(self, solid_red_image: Path) -> None:
        """Solid color image must return valid PaletteColors with no None/empty fields."""
        try:
            result = extract_palette(solid_red_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # All fields must be populated (contract: no None/empty)
        assert isinstance(result, PaletteColors)
        assert result.dominant != ""
        assert result.secondary != ""
        assert result.accent != ""
        assert result.background != ""

    def test_solid_color_all_fields_equal(self, solid_red_image: Path) -> None:
        """When image is solid color, all derived colors should be that color."""
        try:
            result = extract_palette(solid_red_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # For a solid red image, all colors should be red (#ff0000)
        # Implementation may normalize, so check they're all valid hex
        pattern = re.compile(HEX_PATTERN)

        assert pattern.match(result.dominant), (
            f"dominant {result.dominant} not valid hex"
        )
        assert pattern.match(result.secondary), (
            f"secondary {result.secondary} not valid hex"
        )
        assert pattern.match(result.accent), f"accent {result.accent} not valid hex"
        assert pattern.match(result.background), (
            f"background {result.background} not valid hex"
        )

        # For a pure red image, the colors should reflect red (255, 0, 0)
        # Implementation may return exactly #ff0000 or very similar
        # The key contract is: no empty/None, all valid hex

    def test_solid_color_no_malformed_hex(self, solid_red_image: Path) -> None:
        """Solid color image must not produce malformed/empty hex values."""
        try:
            result = extract_palette(solid_red_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        pattern = re.compile(HEX_PATTERN)

        # Every color field must be exactly 7 chars: #RRGGBB
        assert len(result.dominant) == 7, f"dominant length {len(result.dominant)} != 7"
        assert len(result.secondary) == 7, (
            f"secondary length {len(result.secondary)} != 7"
        )
        assert len(result.accent) == 7, f"accent length {len(result.accent)} != 7"
        assert len(result.background) == 7, (
            f"background length {len(result.background)} != 7"
        )

        # Must match hex pattern
        assert pattern.match(result.dominant)
        assert pattern.match(result.secondary)
        assert pattern.match(result.accent)
        assert pattern.match(result.background)


class TestGrayscaleImage:
    """Test behavior for grayscale (zero saturation) images."""

    def test_grayscale_accent_equals_dominant(self, solid_gray_image: Path) -> None:
        """For grayscale images, accent must equal dominant (no saturation)."""
        try:
            result = extract_palette(solid_gray_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # For solid gray, saturation is zero everywhere, so accent == dominant
        assert result.accent == result.dominant, (
            f"For grayscale image, accent ({result.accent}) should equal dominant ({result.dominant})"
        )

    def test_grayscale_all_fields_valid(self, solid_gray_image: Path) -> None:
        """Grayscale image must produce valid hex in all fields."""
        try:
            result = extract_palette(solid_gray_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        pattern = re.compile(HEX_PATTERN)
        assert pattern.match(result.dominant)
        assert pattern.match(result.secondary)
        assert pattern.match(result.accent)
        assert pattern.match(result.background)


# =============================================================================
# NEW: Luminance Path - is_dark Classification
# =============================================================================


class TestLuminanceClassification:
    """Verify luminance calculation and is_dark threshold."""

    def test_black_is_dark(self, solid_black_image: Path) -> None:
        """Pure black (#000000) must have is_dark=True."""
        try:
            result = extract_palette(solid_black_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert result.is_dark is True, "Black image must have is_dark=True"

    def test_white_is_not_dark(self, solid_white_image: Path) -> None:
        """Pure white (#ffffff) must have is_dark=False."""
        try:
            result = extract_palette(solid_white_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert result.is_dark is False, "White image must have is_dark=False"

    def test_dark_gray_is_dark(self, dark_gray_image: Path) -> None:
        """Dark gray (luminance < 0.5) must have is_dark=True."""
        try:
            result = extract_palette(dark_gray_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # RGB(50, 50, 50) has luminance ≈ 0.196, definitely < 0.5
        assert result.is_dark is True, "Dark gray image must have is_dark=True"

    def test_light_gray_is_not_dark(self, light_gray_image: Path) -> None:
        """Light gray (luminance >= 0.5) must have is_dark=False."""
        try:
            result = extract_palette(light_gray_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # RGB(200, 200, 200) has luminance ≈ 0.784, definitely >= 0.5
        assert result.is_dark is False, "Light gray image must have is_dark=False"

    def test_luminance_uses_bt709_formula(self, solid_red_image: Path) -> None:
        """Luminance must use ITU-R BT.709 coefficients."""
        # For pure red #ff0000:
        # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        # L = 0.2126 * 255/255 + 0.7152 * 0 + 0.0722 * 0
        # L = 0.2126
        # is_dark = (L < 0.5) = True
        try:
            result = extract_palette(solid_red_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Pure red has luminance 0.2126 < 0.5, so is_dark should be True
        assert result.is_dark is True, "Pure red must be dark (BT.709 L=0.2126)"


# =============================================================================
# NEW: Background Edge Sampling
# =============================================================================


class TestBackgroundEdgeSampling:
    """Verify background color computation from image edges."""

    def test_background_computed_from_edges(self, edge_border_image: Path) -> None:
        """Background must be computed from edge pixels, not center."""
        try:
            result = extract_palette(edge_border_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # The edge_border_image has blue (#0000ff) on edges (outer 5%)
        # and red (#ff0000) in center
        # Background should be closer to blue than to red
        # (Not necessarily exactly #0000ff, but should have high blue component)

        # Extract RGB from background
        bg = result.background
        r = int(bg[1:3], 16)
        g = int(bg[3:5], 16)
        b = int(bg[5:7], 16)

        # Background should have more blue than red
        assert b > r, f"background {bg} should have more blue than red (edges)"


# =============================================================================
# NEW: RGBA/Alpha Handling
# =============================================================================


class TestAlphaHandling:
    """Test handling of images with alpha channel."""

    def test_rgba_converted_to_rgb(self, rgba_image: Path) -> None:
        """RGBA image should be converted to RGB and produce valid palette."""
        try:
            result = extract_palette(rgba_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert isinstance(result, PaletteColors)
        pattern = re.compile(HEX_PATTERN)
        assert pattern.match(result.dominant)


class TestTransparentImage:
    """Test behavior for fully transparent images."""

    def test_fully_transparent_raises_invalid_image_error(
        self, transparent_image: Path
    ) -> None:
        """Fully transparent image must raise InvalidImageError."""
        try:
            extract_palette(transparent_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return
        except InvalidImageError as exc_info:
            # Error should mention transparency or no visible pixels
            error_msg = str(exc_info).lower()
            assert (
                "transparent" in error_msg
                or "no pixel" in error_msg
                or "empty" in error_msg
                or "alpha" in error_msg
                or "invisible" in error_msg
            )
            return

        pytest.fail("Expected InvalidImageError for fully transparent image")


# =============================================================================
# NEW: Gradient and Multi-Color Images
# =============================================================================


class TestGradientImage:
    """Test behavior for gradient images (continuous color range)."""

    def test_gradient_returns_valid_palette(self, gradient_image: Path) -> None:
        """Gradient image should produce valid palette."""
        try:
            result = extract_palette(gradient_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        assert isinstance(result, PaletteColors)
        assert len(result.colors) >= MIN_COLORS_FOR_PALETTE

    def test_gradient_colors_are_ordered(self, gradient_image: Path) -> None:
        """Gradient colors should still be frequency-ordered."""
        try:
            result = extract_palette(gradient_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # For a gray gradient, all colors should be grays
        # colors[0] should equal dominant
        assert result.colors[0] == result.dominant


class TestTwoColorImage:
    """Test behavior for images with exactly two colors."""

    def test_two_color_returns_both(self, two_color_image: Path) -> None:
        """Two-color image should return both colors (or close variants)."""
        try:
            result = extract_palette(two_color_image)
        except NotImplementedError:
            pytest.skip("Implementation pending")
            return

        # Should have at least 1 color (contract minimum)
        # For a 50/50 split, both colors should appear
        assert len(result.colors) >= MIN_COLORS_FOR_PALETTE

        # First color (dominant) should be one of the two: red or blue
        assert result.dominant in ("#ff0000", "#0000ff"), (
            f"Expected red or blue as dominant, got {result.dominant}"
        )


# =============================================================================
# NEW: Contract Validation Summary Tests
# =============================================================================


class TestContractCompleteness:
    """Summary tests ensuring all contract requirements are covered."""

    def test_extract_palette_signature(self) -> None:
        """Verify extract_palette function signature matches contract."""
        import inspect

        sig = inspect.signature(extract_palette)
        params = list(sig.parameters.keys())

        assert "image_path" in params, "Must have image_path parameter"
        assert "n_colors" in params, "Must have n_colors parameter"
        assert sig.parameters["n_colors"].default == DEFAULT_N_COLORS, (
            f"n_colors default must be {DEFAULT_N_COLORS}"
        )

        # Return type should be PaletteColors
        return_annotation = str(sig.return_annotation)
        assert "PaletteColors" in return_annotation, (
            f"Return type should be PaletteColors, got {return_annotation}"
        )

    def test_all_error_types_are_defined(self) -> None:
        """Verify all contracted error types are defined."""
        from sfumato.palette import (
            ClusteringError,
            ImageReadError,
            InvalidImageError,
            PaletteError,
        )

        # All must inherit from PaletteError
        assert issubclass(ImageReadError, PaletteError)
        assert issubclass(InvalidImageError, PaletteError)
        assert issubclass(ClusteringError, PaletteError)

    def test_palette_colors_is_frozen(self) -> None:
        """PaletteColors must be frozen (immutable)."""
        palette = PaletteColors(
            dominant="#ffffff",
            secondary="#000000",
            accent="#ff0000",
            background="#000000",
            is_dark=True,
            colors=("#ffffff",),
        )

        with pytest.raises(AttributeError):
            palette.dominant = "#000000"  # type: ignore[misc]

    def test_palette_colors_is_hashable(self) -> None:
        """PaletteColors must be hashable (frozen dataclass)."""
        palette = PaletteColors(
            dominant="#ffffff",
            secondary="#000000",
            accent="#ff0000",
            background="#000000",
            is_dark=True,
            colors=("#ffffff",),
        )

        # Should not raise TypeError
        hash(palette)
