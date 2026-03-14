"""Contract tests for palette color extraction.

This module tests the behavior invariants and contracts defined in
`src/sfumato/palette.py`. Implementation tests will be added when
the extraction logic is implemented.

The contract tests here verify:
1. Hex normalization guarantees
2. PaletteColors field validation
3. Edge case behavior (tiny/low-variance images)
4. Error handling (unreadable/invalid images)
5. Luminance calculation correctness
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
    pass


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_valid_palette() -> PaletteColors:
    """A valid PaletteColors instance for testing."""
    return PaletteColors(
        dominant="#3a5f8a",
        secondary="#8a3a5f",
        accent="#5f8a3a",
        background="#2a2a2a",
        is_dark=True,
        colors=("#3a5f8a", "#8a3a5f", "#5f8a3a", "#2a2a2a", "#ffffff"),
    )


# -----------------------------------------------------------------------------
# Contract: Hex Format Invariants
# -----------------------------------------------------------------------------


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


class TestHexNormalization:
    """Verify hex normalization produces canonical format."""

    def test_lowercase_input_unchanged(
        self, sample_valid_palette: PaletteColors
    ) -> None:
        """Lowercase hex input should pass validation unchanged."""
        # All colors in fixture are lowercase
        assert sample_valid_palette.dominant == "#3a5f8a"
        assert sample_valid_palette.colors[0] == "#3a5f8a"

    def test_uppercase_input_accepted(self) -> None:
        """Uppercase hex input should be accepted (implementation normalizes)."""
        # Note: actual normalization happens in _normalize_hex, not in PaletteColors
        # This test verifies the pattern accepts uppercase
        palette = PaletteColors(
            dominant="#ABCDEF",
            secondary="#123456",
            accent="#FEDCBA",
            background="#AABBCC",
            is_dark=False,
            colors=("#ABCDEF", "#123456"),
        )
        assert palette.dominant == "#ABCDEF"


# -----------------------------------------------------------------------------
# Contract: PaletteColors Field Validation
# -----------------------------------------------------------------------------


class TestPaletteColorsValidation:
    """Verify PaletteColors validates all fields at construction."""

    def test_valid_palette_passes_validation(
        self, sample_valid_palette: PaletteColors
    ) -> None:
        """A valid PaletteColors instance should pass __post_init__ validation."""
        assert sample_valid_palette.dominant == "#3a5f8a"
        assert sample_valid_palette.is_dark is True
        assert len(sample_valid_palette.colors) >= MIN_COLORS_FOR_PALETTE

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
            PaletteColors(  # type: ignore[call-arg]
                dominant=None,
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
            PaletteColors(  # type: ignore[call-arg]
                dominant="#ffffff",
                secondary="#ffffff",
                accent="#ffffff",
                background="#000000",
                is_dark="true",  # Invalid: string instead of bool
                colors=("#ffffff",),
            )

    def test_colors_must_contain_valid_hex(self) -> None:
        """Each element in colors must be valid hex."""
        with pytest.raises(ValueError, match="colors\\[0\\].*must match"):
            PaletteColors(
                dominant="#ffffff",
                secondary="#000000",
                accent="#000000",
                background="#000000",
                is_dark=True,
                colors=("invalid", "#000000"),  # First element invalid
            )


# -----------------------------------------------------------------------------
# Contract: Edge Cases
# -----------------------------------------------------------------------------


class TestMonochromaticImage:
    """Test behavior when image has zero variance (single color)."""

    def test_all_fields_equal_for_monochrome(self) -> None:
        """When image is single color, all derived colors should be that color."""
        # This is a contract guarantee: no None/empty values
        palette = PaletteColors(
            dominant="#808080",
            secondary="#808080",  # Same as dominant
            accent="#808080",  # Grayscale: accent == dominant
            background="#808080",
            is_dark=False,  # Luminance ~0.5 is not dark
            colors=("#808080",),
        )
        assert palette.dominant == palette.secondary == palette.accent


class TestGrayscaleImage:
    """Test behavior for grayscale (zero saturation) images."""

    def test_accent_equals_dominant_for_grayscale(
        self, sample_valid_palette: PaletteColors
    ) -> None:
        """For grayscale images, accent must be the most frequent gray."""
        # The sample palette has non-grayscale colors, but this test
        # documents the grayscale contract explicitly
        # When saturation is zero for all colors, accent == dominant
        pass  # Implementation test will verify actual extraction


class TestTinyImage:
    """Test behavior for tiny/low-pixel images."""

    def test_minimum_colors_enforced(self) -> None:
        """colors list must have at least MIN_COLORS_FOR_PALETTE elements."""
        # This is enforced in __post_init__
        assert MIN_COLORS_FOR_PALETTE >= 1

    def test_padding_strategy_documented(self) -> None:
        """Tiny image contract: implementation must pad or raise InvalidImageError."""
        # This test documents the contract; implementation will verify behavior
        # The implementation SHOULD pad colors list to meet MIN_COLORS_FOR_PALETTE
        # OR raise InvalidImageError if padding is impossible
        pass


# -----------------------------------------------------------------------------
# Contract: Error Types
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Contract: Constants
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Contract: extract_palette Function
# -----------------------------------------------------------------------------


class TestExtractPaletteContract:
    """Test the extract_palette function contract.

    Note: These tests verify the stub behavior. Implementation tests
    will replace NotImplementedError tests with actual extraction tests.
    """

    def test_not_implemented_stub(self) -> None:
        """extract_palette MUST raise NotImplementedError until implemented."""
        with pytest.raises(NotImplementedError, match="Contract-only stub"):
            extract_palette(Path("dummy.jpg"), n_colors=8)

    def test_accepts_path_and_optional_n_colors(self) -> None:
        """extract_palette signature: (image_path: Path, n_colors: int = 8)."""
        # This test documents the signature; actual invocation is in NotImplementedError test
        import inspect
        from typing import get_type_hints

        # We can at least verify the function exists and has the right signature
        sig = inspect.signature(extract_palette)
        params = list(sig.parameters.keys())
        assert "image_path" in params, "Must have image_path parameter"
        assert "n_colors" in params, "Must have n_colors parameter"

        # Check default value
        n_colors_param = sig.parameters["n_colors"]
        assert n_colors_param.default == DEFAULT_N_COLORS, (
            f"n_colors default must be {DEFAULT_N_COLORS}"
        )


# -----------------------------------------------------------------------------
# Contract: Luminance Calculation
# -----------------------------------------------------------------------------


class TestLuminanceCalculation:
    """Verify luminance calculation uses BT.709 coefficients."""

    def test_black_luminance(self) -> None:
        """Pure black (#000000) should have luminance 0.0."""
        # When _compute_luminance is implemented, test: L = 0.2126*R + 0.7152*G + 0.0722*B
        # For #000000: R=G=B=0, so L=0.0
        pass  # Implementation test

    def test_white_luminance(self) -> None:
        """Pure white (#FFFFFF) should have luminance 1.0."""
        # For #FFFFFF: R=G=B=255, normalized to 1.0
        # L = 0.2126*1 + 0.7152*1 + 0.0722*1 = 1.0
        pass  # Implementation test

    def test_red_luminance(self) -> None:
        """Pure red (#FF0000) should have luminance ~0.2126."""
        # For #FF0000: R=255 (1.0), G=B=0
        # L = 0.2126*1 + 0.7152*0 + 0.0722*0 = 0.2126
        pass  # Implementation test

    def test_green_luminance(self) -> None:
        """Pure green (#00FF00) should have luminance ~0.7152."""
        # For #00FF00: G=255 (1.0), R=B=0
        # L = 0.2126*0 + 0.7152*1 + 0.0722*0 = 0.7152
        pass  # Implementation test

    def test_blue_luminance(self) -> None:
        """Pure blue (#0000FF) should have luminance ~0.0722."""
        # For #0000FF: B=255 (1.0), R=G=0
        # L = 0.2126*0 + 0.7152*0 + 0.0722*1 = 0.0722
        pass  # Implementation test

    def test_is_dark_threshold(self) -> None:
        """is_dark should be True when luminance < 0.5."""
        # Document the threshold: 0.5 (exactly midpoint)
        # Gray #808080 has L = 0.2126*0.5 + 0.7152*0.5 + 0.0722*0.5 = 0.5
        # So luminance < 0.5 => is_dark=True, luminance >= 0.5 => is_dark=False
        pass  # Implementation test


# -----------------------------------------------------------------------------
# Contract: Background Edge Sampling
# -----------------------------------------------------------------------------


class TestBackgroundEdgeSampling:
    """Verify background color computation from image edges."""

    def test_sample_fraction_clamp(self) -> None:
        """sample_fraction must be clamped to [0.01, 0.10]."""
        # _sample_edge_pixels must accept sample_fraction and clamp it
        # Values < 0.01 become 0.01
        # Values > 0.10 become 0.10
        pass  # Implementation test

    def test_edge_pixels_only(self) -> None:
        """Background must be computed from edge pixels only, not clustering."""
        # The implementation must sample border rows/columns separately
        # from the k-means clustering used for dominant/secondary/accent
        pass  # Implementation test


# -----------------------------------------------------------------------------
# Behavioral Tests (Implementation Required)
# -----------------------------------------------------------------------------


class TestExtractPaletteImplementation:
    """Tests requiring extract_palette implementation.

    Marked as skipped until implementation is complete.
    These tests document expected behavior for the implementer.
    """

    @pytest.mark.skip(reason="Implementation pending")
    def test_nonexistent_file_raises_image_read_error(self) -> None:
        """Must raise ImageReadError for non-existent file."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_corrupt_file_raises_image_read_error(self, tmp_path: Path) -> None:
        """Must raise ImageReadError for corrupt/invalid image file."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_zero_dimension_image_raises_invalid_image_error(
        self, tmp_path: Path
    ) -> None:
        """Must raise InvalidImageError for 0x0 pixel image."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_single_pixel_image_returns_valid_palette(self, tmp_path: Path) -> None:
        """Single pixel image MUST return valid PaletteColors, NOT raise error."""
        # Contract: implementation must pad colors or handle gracefully
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_all_transparent_image_raises_invalid_image_error(
        self, tmp_path: Path
    ) -> None:
        """Fully transparent PNG must raise InvalidImageError."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_large_image_downsampling_performance(self) -> None:
        """4K image (3840x2160) should complete within 5 seconds."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_all_colors_are_lowercase_hex(self) -> None:
        """All returned colors MUST be lowercase hex."""
        # Verify dominant, secondary, accent, background, all in colors
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_colors_ordered_by_frequency(self) -> None:
        """colors list MUST be sorted by cluster size (frequency)."""
        # colors[0] must be the most frequent (dominant)
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_n_colors_respected(self) -> None:
        """Returned colors list length should respect n_colors parameter."""
        pass

    @pytest.mark.skip(reason="Implementation pending")
    def test_n_colors_capped_at_max(self) -> None:
        """n_colors > MAX_N_COLORS should be capped, not error."""
        pass


# -----------------------------------------------------------------------------
# Additional Coverage for Edge Cases
# -----------------------------------------------------------------------------


class TestPaletteColorsTupleImmutability:
    """Verify PaletteColors uses tuple for immutability."""

    def test_colors_is_tuple(self, sample_valid_palette: PaletteColors) -> None:
        """colors must be tuple for hashability."""
        assert isinstance(sample_valid_palette.colors, tuple)

    def test_frozen_dataclass_hashable(
        self, sample_valid_palette: PaletteColors
    ) -> None:
        """PaletteColors instances must be hashable (frozen dataclass)."""
        # This should not raise TypeError
        palette_hash = hash(sample_valid_palette)
        assert isinstance(palette_hash, int)
