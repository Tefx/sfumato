"""Palette extraction contracts for sfumato core.

This module intentionally pins API and behavior contracts from
`ARCHITECTURE.md#2.4` without implementing clustering logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------------
# Normalization Invariants
# -----------------------------------------------------------------------------

HEX_PATTERN = r"^#[0-9A-Fa-f]{6}$"
"""Regex pattern for canonical hex color format: exactly 6 hex digits after '#'."""

CANONICAL_HEX_FORMAT = "#RRGGBB"
"""Canonical format guarantee: uppercase or lowercase hex digits are both valid input,
but all colors returned by this module MUST match this format with exactly 6 hex digits.
No shorthand (#RGB), no 8-digit alpha (#RRGGBBAA), no named colors.
"""

MIN_COLORS_FOR_PALETTE = 1
"""Minimum number of distinct colors required to form a valid palette.

If k-means clustering yields fewer than this many clusters, the implementation
must synthesize placeholder colors or fail cleanly (see `PaletteExtractionError`).
"""

DEFAULT_N_COLORS = 8
"""Default number of colors to extract from an image."""

MAX_N_COLORS = 16
"""Maximum permitted colors to extract. Values above this are capped."""


# -----------------------------------------------------------------------------
# Error Model
# -----------------------------------------------------------------------------


class PaletteError(Exception):
    """Base exception for palette extraction failures."""


class ImageReadError(PaletteError):
    """Raised when the image file cannot be read or decoded.

    Causes:
        - File does not exist at the given path
        - File exists but is not a valid image format (corrupt, unsupported)
        - PIL cannot open or decode the image

    Contract:
        - MUST include the original path in error message
        - MUST NOT attempt to recover from read failures
        - Caller should handle this gracefully (skip painting, log, continue)
    """


class InvalidImageError(PaletteError):
    """Raised when the image is readable but cannot yield a valid palette.

    Causes:
        - Image has zero pixels (empty data)
        - Image is entirely a single pixel (no variance)
        - Image dimensions produce fewer pixels than MIN_COLORS_FOR_PALETTE after downsampling
        - All pixels are transparent/fully-missing alpha channel

    Contract:
        - MUST check for these conditions AFTER read succeeds
        - MUST NOT return an incomplete PaletteColors with missing/None fields
        - Caller should handle this gracefully (use fallback palette, log, continue)
    """


class ClusteringError(PaletteError):
    """Raised when k-means clustering fails to converge.

    This is an internal implementation error indicating that the clustering
    algorithm failed to produce stable cluster centers.

    Contract:
        - SHOULD include debugging context (image dimensions, pixel count, n_colors)
        - Implementation SHOULD retry with adjusted parameters beforeraising
        - Final resort: raise this error to caller
    """


# -----------------------------------------------------------------------------
# Data Model
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PaletteColors:
    """Structural output from palette extraction.

    All color fields are guaranteed to be in canonical hex format: #RRGGBB
    where R, G, B are hexadecimal digits (uppercase or lowercase input accepted,
    normalized to lowercase on output for consistency).

    Fields:
        dominant: Most frequent color in the image, determined by k-means cluster
            sizes after weighting. The cluster with the largest weighted size wins.
            MUST be in #RRGGBB format. MUST NOT be empty or None.

        secondary: Second most frequent color. If only one unique color exists,
            this MUST be identical to `dominant` (no empty/None fields allowed).
            MUST be in #RRGGBB format.

        accent: Most saturated color among all extracted colors. Saturation is
            computed from HSL/HSV color space. If all colors have zero saturation
            (grayscale image), this MUST be the most frequent gray (i.e., dominant).
            MUST be in #RRGGBB format.

        background: Average color of image edge pixels (typically a border sample
            from each edge, or a simple average of all pixels in border rows/columns).
            This provides a good "background" color for text overlays.
            MUST be in #RRGGBB format.

        is_dark: True if the perceived luminance of the dominant color is < 0.5.
            Luminance MUST be computed using perceived luminance formula:
            L = 0.2126 * R + 0.7152 * G + 0.0722 * B (ITU-R BT.709).
            This determines whether light or dark text should be used on this palette.

        colors: Ordered list of top N extracted colors (5 to MAX_N_COLORS, capped).
            MUST be sorted by frequency (most frequent first).
            Each element MUST be in #RRGGBB format.
            MUST contain at least MIN_COLORS_FOR_PALETTE elements.
            MAY contain duplicates if the image has low color variance (the implementation
            should dedupe, but contract does not require it).

    Invariants:
        - All color strings match HEX_PATTERN (exactly 7 chars, # prefix + 6 hex digits)
        - dominant, secondary, accent, background are ALWAYS populated (no None/empty)
        - colors list ALWAYS has >= MIN_COLORS_FOR_PALETTE elements
        - is_dark is ALWAYS a boolean
        - All hex digits are normalized to lowercase (e.g., "#ff00aa" not "#FF00AA")

    Edge Cases:
        - Monochromatic image: dominant == secondary == accent (all same color)
        - Grayscale image: accent is the most saturated (i.e., dominant gray)
        - Tiny image (< MIN_COLORS_FOR_PALETTE unique colors): synthetically pad colors
          by duplicating the most dominant color, OR raise InvalidImageError
        - Low-variance image (clusters collapse to same centroid): treat as monochromatic

    Serialization:
        This dataclass is JSON-serializable for cache storage.
        Path serialization handles the frozen dataclass correctly.
    """

    dominant: str
    secondary: str
    accent: str
    background: str
    is_dark: bool
    colors: tuple[str, ...]  # Use tuple for hashability/frozen dataclass

    def __post_init__(self) -> None:
        """Validate all color fields match canonical hex format."""
        import re

        pattern = re.compile(HEX_PATTERN)

        for field_name in ("dominant", "secondary", "accent", "background"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be str, got {type(value).__name__}")
            if not pattern.match(value):
                raise ValueError(
                    f"{field_name} must match {CANONICAL_HEX_FORMAT}, got {value!r}"
                )

        if len(self.colors) < MIN_COLORS_FOR_PALETTE:
            raise ValueError(
                f"colors must have >= {MIN_COLORS_FOR_PALETTE} elements, "
                f"got {len(self.colors)}"
            )

        for i, color in enumerate(self.colors):
            if not isinstance(color, str):
                raise TypeError(f"colors[{i}] must be str, got {type(color).__name__}")
            if not pattern.match(color):
                raise ValueError(
                    f"colors[{i}] must match {CANONICAL_HEX_FORMAT}, got {color!r}"
                )

        if not isinstance(self.is_dark, bool):
            raise TypeError(f"is_dark must be bool, got {type(self.is_dark).__name__}")


# -----------------------------------------------------------------------------
# Extraction Contract
# -----------------------------------------------------------------------------


def extract_palette(
    image_path: Path,
    n_colors: int = DEFAULT_N_COLORS,
) -> PaletteColors:
    """Extract dominant color palette from a painting image.

    This is a contract-only stub. Implementation deferred to a later step.

    Args:
        image_path: Absolute path to the image file. MUST be a valid image format
            supported by PIL (Pillow). Common formats: JPEG, PNG, WebP.
        n_colors: Number of colors to extract. Capped at MAX_N_COLORS.
            Defaults to DEFAULT_N_COLORS. Values < MIN_COLORS_FOR_PALETTE are
            normalized to MIN_COLORS_FOR_PALETTE.

    Returns:
        PaletteColors with canonical hex colors, luminance classification, and
        ordered color list.

    Raises:
        ImageReadError: If the file cannot be read, does not exist, or is not
            a valid image format that PIL can decode.
        InvalidImageError: If the image is valid but cannot produce a meaningful
            palette (empty, single-pixel, zero-variance, all-transparent).
        ClusteringError: If k-means clustering fails to converge after retries.

    Contract Invariants:
        1. Hex Normalization:
           All returned colors MUST match the pattern #[0-9A-Fa-f]{6}.
           Input may be lowercase (#ff0000) or uppercase (#FF0000).
           Output MUST be lowercase for consistency.

        2. No Missing Values:
           dominant, secondary, accent, background MUST ALWAYS be valid hex strings.
           The implementation MUST NOT return None or empty string for these fields.
           If the image has only one color, all four may be identical.

        3. Background Color:
           MUST be computed from image edge pixels, not from clustering.
           Sample strategy: take 1-5% of border width from all four edges,
           compute average RGB of those pixels.
           This provides a background-matching color for text overlays.

        4. is_dark Classification:
           MUST use ITU-R BT.709 luminance coefficients:
           L = 0.2126 * R + 0.7152 * G + 0.0722 * B
           Threshold: is_dark = (L < 0.5)
           The dominant color's luminance is used for this calculation.

        5. Accent Color:
           MUST be the most saturated color among all extracted colors.
           Saturation S = max(R, G, B) - min(R, G, B) in RGB space,
           OR S from HSV/HSL color space (implementation choice).
           For grayscale images (S=0 everywhere), accent == dominant.

        6. Color Ordering:
           colors list MUST be sorted by frequency (cluster size).
           Most frequent color is colors[0], which MUST equal dominant.
           The implementation MAY dedupe colors, but is NOT REQUIRED to.

        7. Tiny Image Handling:
           If image has fewer unique pixels than requested n_colors,
           the implementation MUST either:
           (a) Pad colors list with duplicates of dominant to reach MIN_COLORS_FOR_PALETTE, OR
           (b) Raise InvalidImageError if padding is impossible/undesirable.

        8. Downsampling:
           The implementation SHOULD downsample large images before clustering
           to improve performance. Typical: resize to max 200x200 pixels.
           Downsampling MUST NOT affect color accuracy beyond acceptable tolerance.

        9. Caching:
           This function does NOT cache results. The caller (orchestrator) is
           responsible for caching PaletteColors by painting content_hash.
           See ARCHITECTURE.md#2.10 (state.LayoutCache pattern).

        10. Thread Safety:
           The implementation MUST be safe to call concurrently from multiple
           threads (no shared mutable state without synchronization).

    Performance Requirements:
        - SHOULD complete within 5 seconds for a 4K image (3840x2160).
        - MUST complete within 30 seconds for any supported image size.
        - Memory usage SHOULD stay under 500MB during extraction.

    Implementation Notes (for future implementer):
        - Use PIL.Image.open() with context manager
        - Convert to RGB if necessary (discard alpha channel)
        - Downsample to ~200-500 pixels max dimension
        - Reshape to (pixels, 3) for k-means
        - Use sklearn.cluster.MiniBatchKMeans or scipy.cluster.vq.kmeans
        - Count cluster sizes to determine frequency ordering
        - Sample edge pixels for background (not in clustering)
        - Compute saturation for accent selection
        - Luminance check per BT.709 coefficients

    Example:
        >>> from pathlib import Path
        >>> colors = extract_palette(Path("painting.jpg"))
        >>> colors.dominant
        '#3a5f8a'
        >>> colors.is_dark
        True
        >>> len(colors.colors)
        8
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


# -----------------------------------------------------------------------------
# Helper Function Contracts (for implementation use)
# -----------------------------------------------------------------------------


def _normalize_hex(color: str) -> str:
    """Normalize a hex color string to canonical lowercase format.

    Args:
        color: Hex color string, possibly with shorthand (#RGB), uppercase,
            or whitespace prefix.

    Returns:
        Canonical lowercase #rrggbb format.

    Raises:
        ValueError: If the color cannot be parsed as a valid hex color.

    Contract:
        - Accepts #RGB (expands to #RRGGBB)
        - Accepts #RRGGBB (normalizes to lowercase)
        - Accepts RRGGBB or RGB without # prefix (adds # prefix)
        - Rejects named colors ("red"), rgba(), hsl(), etc.
        - Strips leading/trailing whitespace.

    Example:
        >>> _normalize_hex("#FF0000")
        '#ff0000'
        >>> _normalize_hex("F00")
        '#ff0000'
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def _compute_luminance(hex_color: str) -> float:
    """Compute perceived luminance of a hex color.

    Args:
        hex_color: Color in canonical #rrggbb format.

    Returns:
        Luminance value in range [0.0, 1.0], where 0 is darkest.

    Raises:
        ValueError: If hex_color does not match HEX_PATTERN.

    Contract:
        Uses ITU-R BT.709 coefficients: L = 0.2126*R + 0.7152*G + 0.0722*B

    Example:
        >>> _compute_luminance("#000000")  # Black
        0.0
        >>> _compute_luminance("#ffffff")  # White
        1.0
        >>> _compute_luminance("#ff0000")  # Pure red
        0.2126
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def _sample_edge_pixels(image_path: Path, sample_fraction: float = 0.05) -> str:
    """Sample edge pixels from an image and compute their average color.

    Args:
        image_path: Path to the image file.
        sample_fraction: Fraction of edge width/height to sample (0.01 to 0.10).

    Returns:
        Average edge color in canonical #rrggbb format.

    Raises:
        ImageReadError: If image cannot be read.
        InvalidImageError: If image has no edge pixels (zero dimension).

    Contract:
        - Samples pixels from all four edges (top, bottom, left, right).
        - sample_fraction is clamped to [0.01, 0.10].
        - For each edge, take strip of width = int(sample_fraction * dimension).
        - Computes mean RGB of all edge pixels.
        - Returns normalized hex color.

    Example:
        >>> _sample_edge_pixels(Path("painting.jpg"), 0.05)
        '#2a4a6a'
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )
