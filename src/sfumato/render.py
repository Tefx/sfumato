"""Render news data into a 4K poster image for Samsung The Frame TV.

This module implements the render module contract from ARCHITECTURE.md#2.7.
It assembles HTML from selected templates using painting, news, layout, and
palette data, then uses Playwright to screenshot the HTML at 3840x2160.

Architecture reference: ARCHITECTURE.md#2.7
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.async_api import async_playwright

if TYPE_CHECKING:
    from sfumato.layout_ai import LayoutColors, LayoutParams
    from sfumato.news import Story
    from sfumato.palette import PaletteColors


# =============================================================================
# CONSTANTS
# =============================================================================

TEMPLATE_DIR = Path(__file__).parent.parent.parent / "templates"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"
VIEWPORT_WIDTH = 3840
VIEWPORT_HEIGHT = 2160

SUPPORTED_TEMPLATES = {
    "painting_text",
    "magazine",
    "portrait",
    "art_overlay",
    "art_minimal",
}


# =============================================================================
# ERROR TYPES
# =============================================================================


class RenderError(Exception):
    """Base exception for render failures."""

    pass


class TemplateNotFoundError(RenderError):
    """Raised when template file is not found."""

    pass


class PlaywrightError(RenderError):
    """Raised when Playwright screenshot fails."""

    pass


# =============================================================================
# DATA TYPES (from ARCHITECTURE.md spec)
# =============================================================================


class Orientation(Enum):
    """Painting orientation determined from image dimensions."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


@dataclass(frozen=True)
class PaintingInfo:
    """Painting metadata and image path.

    Note: This is defined in ARCHITECTURE.md#2.3 (paintings module).
    Included here as a minimal definition since paintings.py may not exist yet.
    """

    image_path: Path
    content_hash: str
    title: str
    artist: str
    year: str
    orientation: Orientation
    width: int
    height: int
    source: str
    source_id: str
    source_url: str


@dataclass
class RenderContext:
    """Render context containing all data needed to generate a 4K PNG.

    Attributes:
        painting: Painting metadata and image path.
        stories: News stories for this rotation.
        layout: Layout analysis results.
        palette: Extracted color palette.
        template_name: Template to use (painting_text, magazine, portrait, etc).
        language: Display language code.
        date_str: Formatted date string.
        time_str: Formatted time string.

    Contract:
        - painting.image_path must exist and be a valid image file
        - stories is a list (may be empty for pure art mode)
        - layout contains all CSS positioning computed by layout_ai
        - palette contains all color values for the template
        - template_name must be one of SUPPORTED_TEMPLATES
    """

    painting: "PaintingInfo"
    stories: list["Story"]
    layout: "LayoutParams"
    palette: "PaletteColors"
    template_name: str
    language: str
    date_str: str
    time_str: str


@dataclass
class RenderResult:
    """Result of rendering a 4K PNG.

    Attributes:
        png_path: Path to rendered 4K PNG file.
        html_path: Path to rendered HTML file (for debugging).
        template_used: Which template was used.
        story_count: How many stories were rendered.
        painting_hash: Content hash of the painting used.

    Contract:
        - png_path and html_path are absolute paths
        - png_path points to a valid 4K PNG file
        - html_path points to the HTML used for rendering
    """

    png_path: Path
    html_path: Path
    template_used: str
    story_count: int
    painting_hash: str


# =============================================================================
# PUBLIC API
# =============================================================================


def build_template_variables(ctx: RenderContext) -> dict[str, str]:
    """Convert RenderContext into a flat dict of template variables.

    Each key corresponds to a {{PLACEHOLDER}} in the HTML template.
    Builds story HTML blocks, handles orientation-specific variables,
    and resolves the painting image path to a file:// URL.

    Args:
        ctx: RenderContext containing all rendering data.

    Returns:
        Dictionary mapping placeholder names to their values.
        All values are strings suitable for HTML template substitution.

    Template Variable Mapping:
        painting_text.html:
            - BG_IMAGE: file:// URL to painting
            - SCRIM_POSITION, SCRIM_GRADIENT, TEXT_POSITION, TEXT_WIDTH
            - TEXT_COLOR, TEXT_COLOR_SEC, TEXT_COLOR_DIM, TEXT_SHADOW
            - DATELINE, NEWS_BLOCKS

        portrait.html:
            - BG_COLOR, LEFT_WIDTH, DIM, TEXT_SEC
            - PAINTING_WIDTH, PAINTING_SRC, ACCENT, BORDER, TEXT
            - DATE, SOURCE, ARTIST, PAINTING_TITLE, PAINTING_YEAR
            - SECTION_LABEL, STORIES

        magazine.html:
            - BG_IMAGE, PANEL_BG, DIM, TEXT, TEXT_SEC, BORDER
            - DATE, SOURCE, STORIES, ARTIST, PAINTING_TITLE

        art_overlay.html:
            - BG_IMAGE, TITLE, DATE, STORIES, UPDATE_TIME, ART_CREDIT

        art_minimal.html:
            - BG_IMAGE, TITLE, DATE, UPDATE_TIME, STORIES, ART_CREDIT
    """
    def _position_to_css(position: str) -> str:
        """Convert semantic position name to CSS absolute positioning."""
        mapping = {
            "top-left": "top: 120px; left: 160px;",
            "top-right": "top: 120px; right: 160px;",
            "bottom-left": "bottom: 120px; left: 160px;",
            "bottom-right": "bottom: 120px; right: 160px;",
            "left-side": "top: 120px; left: 160px;",
            "right-side": "top: 120px; right: 160px;",
        }
        # If it already looks like CSS (contains ":"), pass through
        if ":" in position and ";" in position:
            return position
        return mapping.get(position, "top: 120px; right: 160px;")

    # Resolve painting image to file:// URL
    bg_image = ctx.painting.image_path.as_uri()

    # Base variables for all templates
    variables: dict[str, str] = {
        "BG_IMAGE": bg_image,
        "DATE": ctx.date_str,
        "UPDATE_TIME": ctx.time_str,
        "ARTIST": ctx.painting.artist,
        "PAINTING_TITLE": ctx.painting.title,
        "PAINTING_YEAR": ctx.painting.year,
    }

    # Add palette-based colors
    variables.update(
        {
            "TEXT_COLOR": ctx.layout.colors.text_primary,
            "TEXT_COLOR_SEC": ctx.layout.colors.text_secondary,
            "TEXT_COLOR_DIM": ctx.layout.colors.text_dim,
            "TEXT_SHADOW": ctx.layout.colors.text_shadow,
        }
    )

    # Add layout-based CSS parameters (for painting_text template)
    # Convert semantic position ("top-right") to CSS absolute positioning
    text_position_css = _position_to_css(ctx.layout.text_zone.position)
    scrim_position_css = ctx.layout.scrim.position_css
    scrim_size_css = ctx.layout.scrim.size_css if hasattr(ctx.layout.scrim, 'size_css') and ctx.layout.scrim.size_css else "width: 50%; height: 50%;"

    variables.update(
        {
            "SCRIM_POSITION": f"{scrim_position_css} {scrim_size_css}",
            "SCRIM_GRADIENT": ctx.layout.scrim.gradient_css,
            "TEXT_POSITION": text_position_css,
            "TEXT_WIDTH": "45%",  # Default max width for text zone
        }
    )

    # Build news blocks based on template
    if ctx.template_name == "painting_text":
        variables["DATELINE"] = f"{ctx.date_str} · {ctx.time_str}"
        variables["NEWS_BLOCKS"] = _build_news_blocks_painting_text(
            ctx.stories, ctx.layout.colors
        )

    elif ctx.template_name == "portrait":
        # Portrait template uses panels with specific colors
        if ctx.layout.portrait_layout:
            variables["BG_COLOR"] = ctx.layout.portrait_layout.left_panel_color
            variables["LEFT_WIDTH"] = (
                f"{ctx.layout.portrait_layout.painting_width_percent // 2}%"
            )
            variables["PAINTING_WIDTH"] = (
                f"{ctx.layout.portrait_layout.painting_width_percent}%"
            )
            variables["PAINTING_SRC"] = bg_image
        else:
            # Fallback if portrait_layout not set
            variables["BG_COLOR"] = ctx.layout.colors.panel_bg
            variables["LEFT_WIDTH"] = "20%"
            variables["PAINTING_WIDTH"] = "50%"
            variables["PAINTING_SRC"] = bg_image

        variables["DIM"] = ctx.layout.colors.text_dim
        variables["TEXT_SEC"] = ctx.layout.colors.text_secondary
        variables["TEXT"] = ctx.layout.colors.text_primary
        variables["ACCENT"] = ctx.layout.colors.accent
        variables["BORDER"] = ctx.layout.colors.border
        variables["SOURCE"] = ctx.stories[0].source if ctx.stories else ""
        variables["SECTION_LABEL"] = f"{ctx.language.upper()} NEWS"
        variables["STORIES"] = _build_stories_portrait(ctx.stories, ctx.layout.colors)

    elif ctx.template_name == "magazine":
        variables["PANEL_BG"] = ctx.layout.colors.panel_bg
        variables["DIM"] = ctx.layout.colors.text_dim
        variables["TEXT"] = ctx.layout.colors.text_primary
        variables["TEXT_SEC"] = ctx.layout.colors.text_secondary
        variables["BORDER"] = ctx.layout.colors.border
        variables["SOURCE"] = ctx.stories[0].source if ctx.stories else ""
        variables["STORIES"] = _build_stories_magazine(ctx.stories, ctx.layout.colors)

    elif ctx.template_name in ("art_overlay", "art_minimal"):
        variables["TITLE"] = "DAILY BRIEF" if ctx.language == "en" else "每日简报"
        variables["ART_CREDIT"] = f"{ctx.painting.artist} · {ctx.painting.title}"
        variables["STORIES"] = _build_stories_overlay(ctx.stories, ctx.layout.colors)

    else:
        # Generic fallback for unknown templates - use painting_text style
        variables["DATELINE"] = f"{ctx.date_str} · {ctx.time_str}"
        variables["NEWS_BLOCKS"] = _build_news_blocks_painting_text(
            ctx.stories, ctx.layout.colors
        )

    return variables


async def render_to_png(
    ctx: RenderContext,
    output_dir: Path | None = None,
) -> RenderResult:
    """Render a 4K PNG from the given context.

    Steps:
    1. Load template HTML from templates/{template_name}.html
    2. Substitute template variables
    3. Write HTML to output_dir/
    4. Launch Playwright, set viewport to 3840x2160
    5. Screenshot to PNG
    6. Close browser
    7. Return RenderResult

    Args:
        ctx: RenderContext containing all rendering data.
        output_dir: Directory to write output files. Defaults to OUTPUT_DIR.

    Returns:
        RenderResult with paths and metadata.

    Raises:
        TemplateNotFoundError: If template file is not found.
        PlaywrightError: If Playwright fails to render.

    Contract:
        - Creates output_dir if it doesn't exist
        - Generates unique filenames with timestamp
        - Template must exist in TEMPLATE_DIR/{template_name}.html
        - PNG is exactly 3840x2160 pixels
    """
    # Resolve output directory
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"render_{timestamp}"
    html_path = output_dir / f"{base_name}.html"
    png_path = output_dir / f"{base_name}.png"

    # Validate and load template
    template_path = TEMPLATE_DIR / f"{ctx.template_name}.html"

    if not template_path.exists():
        raise TemplateNotFoundError(
            f"Template '{ctx.template_name}' not found at {template_path}"
        )

    template_content = template_path.read_text(encoding="utf-8")

    # Build variables and substitute
    variables = build_template_variables(ctx)
    html_content = _substitute_template(template_content, variables)

    # Write HTML for debugging
    html_path.write_text(html_content, encoding="utf-8")

    # Render with Playwright
    try:
        await _playwright_screenshot(html_content, png_path)
    except Exception as e:
        raise PlaywrightError(f"Failed to render PNG: {e}") from e

    return RenderResult(
        png_path=png_path.resolve(),
        html_path=html_path.resolve(),
        template_used=ctx.template_name,
        story_count=len(ctx.stories),
        painting_hash=ctx.painting.content_hash,
    )


def render_to_png_sync(
    ctx: RenderContext,
    output_dir: Path | None = None,
) -> RenderResult:
    """Synchronous wrapper around render_to_png.

    Args:
        ctx: RenderContext containing all rendering data.
        output_dir: Directory to write output files.

    Returns:
        RenderResult with paths and metadata.

    Contract:
        - Runs the async render_to_png in a synchronous context
        - Uses asyncio.run() internally
        - Not suitable for use in async contexts (will raise)
    """
    return asyncio.run(render_to_png(ctx, output_dir))


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _substitute_template(template: str, variables: dict[str, str]) -> str:
    """Substitute {{PLACEHOLDER}} variables in template.

    Args:
        template: HTML template content with {{PLACEHOLDER}} markers.
        variables: Dictionary mapping placeholder names to values.

    Returns:
        Template with all placeholders replaced.
    """
    result = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        result = result.replace(placeholder, value)
    return result


def _build_news_blocks_painting_text(
    stories: list["Story"],
    colors: "LayoutColors",  # type: ignore[name-defined]
) -> str:
    """Build news blocks for painting_text template.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.

    Returns:
        HTML string for news blocks.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        featured_class = " featured" if story.featured else ""
        parts.append(f'<div class="news-block{featured_class}">')
        parts.append(
            f'  <div class="title" style="color: {colors.text_primary}">{story.headline}</div>'
        )
        parts.append(
            f'  <div class="body" style="color: {colors.text_secondary}">{story.summary}</div>'
        )
        parts.append("</div>")

    return "\n".join(parts)


def _build_stories_portrait(
    stories: list["Story"],
    _colors: "LayoutColors",  # type: ignore[name-defined]
) -> str:
    """Build stories HTML for portrait template.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.

    Returns:
        HTML string for stories.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        parts.append('<div class="story">')
        parts.append(f'  <div class="headline">{story.headline}</div>')
        parts.append(f'  <div class="body">{story.summary}</div>')
        parts.append("</div>")

    return "\n".join(parts)


def _build_stories_magazine(
    stories: list["Story"],
    _colors: "LayoutColors",  # type: ignore[name-defined]
) -> str:
    """Build stories HTML for magazine template.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.

    Returns:
        HTML string for stories.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        parts.append('<div class="story">')
        parts.append(f'  <div class="headline">{story.headline}</div>')
        parts.append(f'  <div class="body">{story.summary}</div>')
        parts.append("</div>")

    return "\n".join(parts)


def _build_stories_overlay(
    stories: list["Story"],
    _colors: "LayoutColors",  # type: ignore[name-defined]
) -> str:
    """Build stories HTML for art_overlay and art_minimal templates.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.

    Returns:
        HTML string for stories.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        featured_class = " featured" if story.featured else ""
        parts.append(f'<div class="story{featured_class}">')
        if story.category:
            parts.append(f'  <div class="category">{story.category}</div>')
        parts.append(f'  <div class="headline">{story.headline}</div>')
        parts.append(f'  <div class="summary">{story.summary}</div>')
        if story.source:
            parts.append(f'  <div class="source">{story.source}</div>')
        parts.append("</div>")

    return "\n".join(parts)


async def _playwright_screenshot(html_content: str, output_path: Path) -> None:
    """Use Playwright to render HTML and take a screenshot.

    Args:
        html_content: Full HTML document content.
        output_path: Path to write PNG screenshot.

    Raises:
        PlaywrightError: If rendering fails.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
            )
            await page.set_content(html_content, wait_until="networkidle")
            # Wait for fonts to load
            await page.wait_for_timeout(1000)
            await page.screenshot(path=str(output_path), type="png")
        finally:
            await browser.close()
