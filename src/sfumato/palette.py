"""Palette extraction for sfumato core.

Implementation of color extraction using PIL + numpy k-means clustering.
Contract reference: ARCHITECTURE.md#2.4
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence


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

# BT.709 luminance coefficients
BT709_R_COEFF = 0.2126
BT709_G_COEFF = 0.7152
BT709_B_COEFF = 0.0722

# Downsampling target size
DOWNSAMPLE_MAX_DIMENSION = 200


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
        - Implementation SHOULD retry with adjusted parameters before raising
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

    Uses k-means clustering on downsampled image pixels.
    Classifies is_dark based on perceived luminance of dominant color.

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
    """
    # Validate and normalize n_colors
    n_colors = max(MIN_COLORS_FOR_PALETTE, min(n_colors, MAX_N_COLORS))

    # Step 1: Load and validate image
    try:
        img = _load_image(image_path)
    except ImageReadError:
        raise
    except Exception as e:
        raise ImageReadError(f"Cannot read image at {image_path}: {e}") from e

    # Step 2: Check for fully transparent images (RGBA mode)
    if img.mode == "RGBA":
        # Check if all pixels are fully transparent (alpha = 0)
        alpha_band = np.array(img)[:, :, 3]
        if np.all(alpha_band == 0):
            raise InvalidImageError("Image is fully transparent - no visible pixels")

    # Step 3: Convert to RGB if necessary (discard alpha)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Step 4: Check for invalid images
    width, height = img.size
    if width == 0 or height == 0:
        raise InvalidImageError(f"Image has zero dimensions: {width}x{height}")

    # Step 5: Downsample for performance
    img = _downsample_image(img, DOWNSAMPLE_MAX_DIMENSION)

    # Step 6: Extract pixel data as numpy array
    pixels = np.array(img)
    if pixels.size == 0:
        raise InvalidImageError("Image has zero pixels after processing")

    # Reshape to (n_pixels, 3)
    pixel_count = pixels.shape[0] * pixels.shape[1]
    pixels_flat = pixels.reshape(pixel_count, 3)

    # Check for all-transparent edge case (should not happen after RGB conversion, but be safe)
    if pixel_count == 0:
        raise InvalidImageError("Image has no pixels")

    # Step 7: Perform k-means clustering
    try:
        cluster_centers, cluster_labels = _kmeans_cluster(pixels_flat, n_colors)
    except ClusteringError:
        raise
    except Exception as e:
        raise ClusteringError(f"k-means clustering failed for {image_path}: {e}") from e

    # Step 8: Compute cluster sizes and order by frequency
    cluster_sizes = np.bincount(cluster_labels, minlength=len(cluster_centers))
    sorted_indices = np.argsort(cluster_sizes)[::-1]  # Descending order

    # Get counts of unique colors for ordering
    unique_colors = cluster_centers[sorted_indices]
    unique_sizes = cluster_sizes[sorted_indices]

    # Step 9: Normalize colors to canonical hex
    colors_hex = [_rgb_to_hex(int(r), int(g), int(b)) for r, g, b in unique_colors]

    # Ensure we have at least MIN_COLORS_FOR_PALETTE
    while len(colors_hex) < MIN_COLORS_FOR_PALETTE:
        colors_hex.append(colors_hex[0])  # Pad with dominant

    colors_tuple = tuple(colors_hex)

    # Step 10: Assign dominant and secondary
    dominant = colors_hex[0]
    secondary = colors_hex[1] if len(colors_hex) > 1 else dominant

    # Step 11: Find accent color (most saturated)
    accent = _find_accent_color(cluster_centers, cluster_sizes)

    # Step 12: Compute background from edge pixels
    background = _sample_edge_pixels(image_path)

    # Step 13: Compute is_dark from dominant luminance
    is_dark = _compute_luminance(dominant) < 0.5

    return PaletteColors(
        dominant=dominant,
        secondary=secondary,
        accent=accent,
        background=background,
        is_dark=is_dark,
        colors=colors_tuple,
    )


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _load_image(image_path: Path) -> Image.Image:
    """Load an image file with PIL.

    Args:
        image_path: Path to the image file.

    Returns:
        PIL Image object.

    Raises:
        ImageReadError: If file does not exist, is not readable, or PIL cannot decode.
    """
    if not image_path.exists():
        raise ImageReadError(f"Image file does not exist: {image_path}")

    if not image_path.is_file():
        raise ImageReadError(f"Path is not a file: {image_path}")

    try:
        img = Image.open(image_path)
        # Force load to catch corrupt images
        img.load()
        return img
    except Image.UnidentifiedImageError as e:
        raise ImageReadError(f"Cannot decode image at {image_path}: {e}") from e
    except Image.ImageFile.IOError as e:
        raise ImageReadError(f"Cannot read image at {image_path}: {e}") from e
    except OSError as e:
        raise ImageReadError(f"Cannot read image at {image_path}: {e}") from e


def _downsample_image(img: Image.Image, max_dimension: int) -> Image.Image:
    """Downsample image to max dimension while preserving aspect ratio.

    Args:
        img: PIL Image to downsample.
        max_dimension: Maximum width or height.

    Returns:
        Downsampled image (or original if already small enough).
    """
    width, height = img.size

    if width <= max_dimension and height <= max_dimension:
        return img

    # Calculate new dimensions preserving aspect ratio
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    # LANCZOS is high-quality for downscaling
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _kmeans_cluster(
    pixels: np.ndarray,
    n_clusters: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    n_init: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform k-means clustering on pixel data.

    Uses numpy-only implementation (no sklearn/scipy dependency).

    Args:
        pixels: Array of shape (n_pixels, 3) with RGB values in 0-255.
        n_clusters: Number of clusters to form.
        max_iterations: Maximum iterations per initialization.
        tolerance: Convergence tolerance for centroid movement.
        n_init: Number of random initializations to try.

    Returns:
        Tuple of (cluster_centers, labels) where:
        - cluster_centers: Array of shape (n_clusters, 3) with RGB centroids
        - labels: Array of shape (n_pixels,) with cluster assignments

    Raises:
        ClusteringError: If clustering fails to converge.
    """
    if len(pixels) == 0:
        raise ClusteringError("No pixels to cluster")

    # Handle case where we have fewer pixels than clusters
    if len(pixels) < n_clusters:
        # Each pixel becomes its own cluster, pad with duplicates
        unique_pixels = np.unique(pixels, axis=0)
        if len(unique_pixels) == 1:
            # Monochromatic image - all pixels same color
            centers = np.tile(unique_pixels[0], (n_clusters, 1))
            labels = np.zeros(len(pixels), dtype=int)
            return centers, labels

        # Use available unique colors, pad with most common
        centers = np.zeros((n_clusters, 3), dtype=np.float64)
        for i in range(min(len(unique_pixels), n_clusters)):
            centers[i] = unique_pixels[i]
        # Pad remaining with first color (most common will be first after unique)
        for i in range(len(unique_pixels), n_clusters):
            centers[i] = unique_pixels[0]

        # Assign labels
        labels = np.zeros(len(pixels), dtype=int)
        for i, pixel in enumerate(pixels):
            distances = np.sqrt(np.sum((centers - pixel) ** 2, axis=1))
            labels[i] = int(np.argmin(distances))

        return centers, labels

    best_centers = None
    best_inertia = float("inf")
    best_labels = None

    for init_idx in range(n_init):
        # Random initialization: choose k random pixels as initial centers
        indices = np.random.choice(len(pixels), n_clusters, replace=False)
        centers = pixels[indices].astype(np.float64)

        labels = np.zeros(len(pixels), dtype=int)

        for iteration in range(max_iterations):
            # Assignment step: each pixel to nearest center
            distances = np.sqrt(
                np.sum(
                    (pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2
                )
            )
            labels = np.argmin(distances, axis=1)

            # Update step: compute new centers
            new_centers = np.zeros_like(centers)
            for k in range(n_clusters):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_centers[k] = pixels[cluster_mask].mean(axis=0)
                else:
                    # Empty cluster: reinitialize randomly
                    new_centers[k] = pixels[np.random.randint(len(pixels))]

            # Check convergence
            center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
            centers = new_centers

            if center_shift < tolerance:
                break

        # Compute inertia (sum of squared distances to assigned center)
        inertia = 0.0
        for k in range(n_clusters):
            cluster_mask = labels == k
            if np.any(cluster_mask):
                inertia += np.sum((pixels[cluster_mask] - centers[k]) ** 2)

        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    if best_centers is None or best_labels is None:
        raise ClusteringError(
            f"k-means failed to converge for {len(pixels)} pixels, n_clusters={n_clusters}"
        )

    return best_centers, best_labels


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values to canonical lowercase hex string.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        Hex color string in #rrggbb format (lowercase).
    """
    # Clamp to valid range
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color in #RRGGBB format.

    Returns:
        Tuple of (r, g, b) values (0-255).
    """
    # Remove # prefix if present
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


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
    """
    import re

    color = color.strip()

    # Add # prefix if missing
    if not color.startswith("#"):
        color = "#" + color

    # Handle shorthand #RGB -> #RRGGBB
    if re.match(r"^#[0-9a-fA-F]{3}$", color):
        r, g, b = color[1], color[2], color[3]
        color = f"#{r}{r}{g}{g}{b}{b}"

    # Validate format
    if not re.match(r"^#[0-9a-fA-F]{6}$", color):
        raise ValueError(f"Invalid hex color: {color!r}")

    return color.lower()


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
    """
    import re

    if not re.match(HEX_PATTERN, hex_color):
        raise ValueError(f"Invalid hex color format: {hex_color!r}")

    r, g, b = _hex_to_rgb(hex_color)
    return (
        BT709_R_COEFF * (r / 255.0)
        + BT709_G_COEFF * (g / 255.0)
        + BT709_B_COEFF * (b / 255.0)
    )


def _find_accent_color(
    cluster_centers: np.ndarray,
    cluster_sizes: np.ndarray,
) -> str:
    """Find the most saturated color among cluster centers.

    Args:
        cluster_centers: Array of shape (n_clusters, 3) with RGB values.
        cluster_sizes: Array of cluster sizes (weights for priority).

    Returns:
        The most saturated color as #rrggbb hex string.
        For grayscale images, returns the dominant (most frequent) color.
    """
    max_saturation = -1.0
    accent_idx = 0

    for i, (r, g, b) in enumerate(cluster_centers):
        # Saturation in RGB space: S = max(R,G,B) - min(R,G,B) / 255
        r_int, g_int, b_int = int(r), int(g), int(b)
        color_max = max(r_int, g_int, b_int)
        color_min = min(r_int, g_int, b_int)
        saturation = (color_max - color_min) / 255.0

        # Prefer higher saturation, weighted by cluster size for ties
        if saturation > max_saturation:
            max_saturation = saturation
            accent_idx = i
        elif saturation == max_saturation and saturation > 0:
            # Tie-break by cluster size
            if cluster_sizes[i] > cluster_sizes[accent_idx]:
                accent_idx = i

    # If all colors have zero saturation (grayscale), return the most frequent
    if max_saturation == 0.0:
        # Find the largest cluster
        accent_idx = int(np.argmax(cluster_sizes))

    r, g, b = cluster_centers[accent_idx]
    return _rgb_to_hex(int(r), int(g), int(b))


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
    """
    # Clamp sample_fraction
    sample_fraction = max(0.01, min(0.10, sample_fraction))

    # Load original image (not downsampled)
    img = _load_image(image_path)

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size

    if width == 0 or height == 0:
        raise InvalidImageError(f"Image has zero dimensions: {width}x{height}")

    # Calculate edge strip width (at least 1 pixel)
    edge_w = max(1, int(sample_fraction * width))
    edge_h = max(1, int(sample_fraction * height))

    pixels = np.array(img)

    # Sample pixels from all four edges
    edge_pixels = []

    # Top edge
    if edge_h > 0:
        edge_pixels.append(pixels[0:edge_h, :, :].reshape(-1, 3))

    # Bottom edge
    if edge_h > 0:
        edge_pixels.append(pixels[-edge_h:, :, :].reshape(-1, 3))

    # Left edge (excluding corners already counted)
    if edge_w > 0:
        edge_pixels.append(
            pixels[
                edge_h : -edge_h if edge_h < height else height, 0:edge_w, :
            ].reshape(-1, 3)
        )

    # Right edge (excluding corners already counted)
    if edge_w > 0:
        edge_pixels.append(
            pixels[
                edge_h : -edge_h if edge_h < height else height, -edge_w:, :
            ].reshape(-1, 3)
        )

    if not edge_pixels:
        # Fallback: use entire image
        edge_pixels.append(pixels.reshape(-1, 3))

    # Combine all edge pixels
    all_edge_pixels = np.concatenate(edge_pixels, axis=0)

    # Compute average RGB
    mean_r, mean_g, mean_b = np.mean(all_edge_pixels, axis=0)

    return _rgb_to_hex(int(mean_r), int(mean_g), int(mean_b))
