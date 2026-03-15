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
from typing import TYPE_CHECKING, TypedDict, TypeAlias

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


WhisperFactIndex: TypeAlias = int | None
"""Selected art-fact index for whisper rendering.

Contract:
    - ``None`` means the caller intentionally disables whisper copy for this frame.
    - Integer values are zero-based indexes into ``RenderContext.layout.art_facts``.
    - The caller owns rotation; render consumes the selected index without mutating it.
"""


class WhisperTemplateVariables(TypedDict):
    """Flat whisper placeholder contract returned inside template variables.

    Contract:
        - ``WHISPER_POSITION`` is a CSS declaration fragment derived from
          ``layout.whisper_zone``. It carries absolute placement and width limits,
          remaining additive to existing template positioning placeholders.
        - ``WHISPER_COLOR`` is the subordinate whisper text color derived from
          ``layout.colors.text_dim``.
        - ``WHISPER_SHADOW`` is the whisper readability shadow derived from
          ``layout.colors.text_shadow``.
        - ``WHISPER_TEXT`` is the selected art-fact payload from
          ``layout.art_facts[whisper_fact_index].text``.
    """

    WHISPER_POSITION: str
    WHISPER_COLOR: str
    WHISPER_SHADOW: str
    WHISPER_TEXT: str


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
        whisper_fact_index: Caller-selected art-fact index for whisper rendering.

    Contract:
        - painting.image_path must exist and be a valid image file
        - stories is a list (may be empty for pure art mode)
        - layout contains all CSS positioning computed by layout_ai
        - palette contains all color values for the template
        - template_name must be one of SUPPORTED_TEMPLATES
        - whisper_fact_index is None or a valid zero-based index into layout.art_facts
        - render does not choose or rotate whisper art facts; caller provides the selection
    """

    painting: "PaintingInfo"
    stories: list["Story"]
    layout: "LayoutParams"
    palette: "PaletteColors"
    template_name: str
    language: str
    date_str: str
    time_str: str
    qr_size: int = 60
    whisper_fact_index: WhisperFactIndex = None


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

        Whisper wiring contract (additive to all existing mappings):
            - WHISPER_POSITION: derived from ctx.layout.whisper_zone.position and
              ctx.layout.whisper_zone.max_width_percent. The value is a CSS declaration
              fragment that preserves existing template positioning semantics.
            - WHISPER_COLOR: derived from ctx.layout.colors.text_dim so whisper copy
              stays subordinate to the primary news block.
            - WHISPER_SHADOW: derived from ctx.layout.colors.text_shadow so whisper copy
              inherits the same readability envelope as other overlay text.
            - WHISPER_TEXT: derived from ctx.layout.art_facts[ctx.whisper_fact_index].text.

        Whisper index semantics:
            - ctx.whisper_fact_index is owned by the caller, not by render.
            - None disables whisper copy selection for the frame.
            - Integers are zero-based indexes into ctx.layout.art_facts.
            - Rotation is external to render: repeated calls with the same context must
              resolve the same whisper fact.

        Compatibility contract:
            - The return type remains dict[str, str].
            - Existing non-whisper keys keep their current meaning.
            - Whisper keys are additive and must not rename or replace existing keys.
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

    # === LLM OUTPUT SANITIZATION ===
    # LLMs generate CSS parameters that often need correction.
    # Validated against prototype results in PROTOTYPING.md.

    # 1. Text-shadow: constrain blur radius (LLMs return 10px+ creating halo boxes)
    text_shadow = ctx.layout.colors.text_shadow
    if any(f"{n}px" in text_shadow for n in range(10, 30)):
        text_shadow = "0 1px 3px rgba(0,0,0,0.7), 0 0 8px rgba(0,0,0,0.3)"

    # 2. Text color: trust LLM (prompt now includes brightness data from PIL analysis)
    text_primary = ctx.layout.colors.text_primary
    text_secondary = ctx.layout.colors.text_secondary
    text_dim = ctx.layout.colors.text_dim

    variables.update(
        {
            "TEXT_COLOR": text_primary,
            "TEXT_COLOR_SEC": text_secondary,
            "TEXT_COLOR_DIM": text_dim,
            "TEXT_SHADOW": text_shadow,
        }
    )

    # Add layout-based CSS parameters (for painting_text template)
    # Convert semantic position ("top-right") to CSS absolute positioning
    text_position_css = _position_to_css(ctx.layout.text_zone.position)

    # Scrim is now integrated as .text-zone background (auto-sizes with content).
    # LLM provides gradient style, template applies it directly on text-zone.
    # No separate SCRIM_POSITION needed — scrim follows text content automatically.
    variables.update(
        {
            "SCRIM_GRADIENT": ctx.layout.scrim.gradient_css,
            "TEXT_POSITION": text_position_css,
            "TEXT_WIDTH": "38%",
        }
    )

    # Build news blocks based on template
    if ctx.template_name == "painting_text":
        variables["DATELINE"] = f"{ctx.date_str} · {ctx.time_str}"
        variables["NEWS_BLOCKS"] = _build_news_blocks_painting_text(
            ctx.stories, ctx.layout.colors, qr_size=ctx.qr_size
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
        variables["STORIES"] = _build_stories_portrait(ctx.stories, ctx.layout.colors, qr_size=ctx.qr_size)

    elif ctx.template_name == "magazine":
        variables["PANEL_BG"] = ctx.layout.colors.panel_bg
        variables["DIM"] = ctx.layout.colors.text_dim
        variables["TEXT"] = ctx.layout.colors.text_primary
        variables["TEXT_SEC"] = ctx.layout.colors.text_secondary
        variables["BORDER"] = ctx.layout.colors.border
        variables["SOURCE"] = ctx.stories[0].source if ctx.stories else ""
        variables["STORIES"] = _build_stories_magazine(ctx.stories, ctx.layout.colors, qr_size=ctx.qr_size)

    elif ctx.template_name in ("art_overlay", "art_minimal"):
        variables["TITLE"] = "DAILY BRIEF" if ctx.language == "en" else "每日简报"
        variables["ART_CREDIT"] = f"{ctx.painting.artist} · {ctx.painting.title}"
        variables["STORIES"] = _build_stories_overlay(ctx.stories, ctx.layout.colors)

    else:
        # Generic fallback for unknown templates - use painting_text style
        variables["DATELINE"] = f"{ctx.date_str} · {ctx.time_str}"
        variables["NEWS_BLOCKS"] = _build_news_blocks_painting_text(
            ctx.stories, ctx.layout.colors, qr_size=ctx.qr_size
        )

    # === WHISPER VARIABLE POPULATION (minimal fixture) ===
    # Populate WHISPER_* variables for the whisper template contract.
    # These are derived from layout analysis and caller-provided index.

    # WHISPER_POSITION: CSS positioning derived from whisper_zone.position
    # Collision avoidance: portrait templates show painting credit in bottom-left.
    # If whisper also targets bottom-left, shift it to avoid overlap.
    whisper_zone = ctx.layout.whisper_zone
    whisper_pos = whisper_zone.position

    # Safety net: if LLM placed whisper in same zone as news (prompt says not to,
    # but LLMs aren't perfect), move whisper to an unused zone.
    news_pos = ctx.layout.text_zone.position
    subject_pos = ctx.layout.subject_zone.position
    occupied = {news_pos, subject_pos}

    # Portrait template: right side = news panel, bottom-left = credit
    if ctx.template_name == "portrait":
        occupied.add("bottom-left")
        occupied.add("top-right")
        occupied.add("bottom-right")
        occupied.add("right-side")

    # Magazine template: right 28% = news panel. Whisper goes on painting area (left side)
    if ctx.template_name == "magazine":
        occupied.add("top-right")
        occupied.add("bottom-right")
        occupied.add("right-side")

    if whisper_pos in occupied:
        # Find first unoccupied zone
        all_zones = ["bottom-right", "bottom-left", "top-left", "top-right"]
        for z in all_zones:
            if z not in occupied:
                whisper_pos = z
                break

    whisper_position_css = _position_to_css(whisper_pos)
    whisper_max_width = f"max-width: {whisper_zone.max_width_percent}%;"
    variables["WHISPER_POSITION"] = f"{whisper_position_css} {whisper_max_width}"

    # WHISPER_COLOR: subordinate text color from text_dim
    variables["WHISPER_COLOR"] = ctx.layout.colors.text_dim

    # WHISPER_SHADOW: readability shadow (same as text_shadow)
    variables["WHISPER_SHADOW"] = text_shadow

    # WHISPER_TEXT: selected art-fact or empty (caller controls via whisper_fact_index)
    if ctx.whisper_fact_index is not None and ctx.layout.art_facts:
        # Zero-based index into art_facts list
        if 0 <= ctx.whisper_fact_index < len(ctx.layout.art_facts):
            variables["WHISPER_TEXT"] = ctx.layout.art_facts[
                ctx.whisper_fact_index
            ].text
        else:
            # Index out of range: use empty to satisfy contract
            variables["WHISPER_TEXT"] = ""
    else:
        # whisper disabled or no art facts
        variables["WHISPER_TEXT"] = ""

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


def _rgba_to_hex(color: str) -> str:
    """Convert rgba(...) or any color to hex format for segno."""
    import re as _re
    # Already hex
    if color.startswith('#'):
        return color[:7]  # Strip alpha if #RRGGBBAA
    # rgba(r, g, b, a) or rgb(r, g, b)
    m = _re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', color)
    if m:
        r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f'#{r:02x}{g:02x}{b:02x}'
    return '#888888'  # fallback


def _make_qr_svg(url: str, display_size: int = 60, color: str = "#888888") -> str:
    """Generate inline SVG QR code for a URL."""
    import segno
    import io
    import re

    # segno only accepts hex colors, not rgba
    hex_color = _rgba_to_hex(color)

    qr = segno.make(url, error='m')
    buf = io.BytesIO()
    qr.save(buf, kind='svg', scale=4, border=1, dark=hex_color, light=None)
    svg = buf.getvalue().decode()
    if '<?xml' in svg:
        svg = svg[svg.index('<svg'):]

    # Add viewBox for proper CSS scaling (without this, browser crops instead of scales)
    w_match = re.search(r'width="(\d+)"', svg)
    h_match = re.search(r'height="(\d+)"', svg)
    if w_match and h_match:
        orig_w, orig_h = w_match.group(1), h_match.group(1)
        svg = svg.replace(f'width="{orig_w}"', f'width="{display_size}"')
        svg = svg.replace(f'height="{orig_h}"', f'height="{display_size}"')
        svg = svg.replace('<svg ', f'<svg viewBox="0 0 {orig_w} {orig_h}" ', 1)

    return svg


def _build_news_blocks_painting_text(
    stories: list["Story"],
    colors: "LayoutColors",  # type: ignore[name-defined]
    qr_size: int = 60,
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
        # Build headline row with optional QR code
        qr_html = ""
        if story.url:
            qr_svg = _make_qr_svg(story.url, display_size=qr_size, color=colors.text_dim)
            qr_html = f'  <div class="qr-code" style="flex: 0 0 {qr_size}px;">{qr_svg}</div>'
        parts.append('  <div class="headline-row">')
        parts.append(
            f'    <div class="title" style="color: {colors.text_primary}">{story.headline}</div>'
        )
        if qr_html:
            parts.append(qr_html)
        parts.append('  </div>')
        parts.append(
            f'  <div class="body" style="color: {colors.text_secondary}">{story.summary}</div>'
        )
        parts.append("</div>")

    return "\n".join(parts)


def _build_stories_portrait(
    stories: list["Story"],
    colors: "LayoutColors",  # type: ignore[name-defined]
    qr_size: int = 60,
) -> str:
    """Build stories HTML for portrait template.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.
        qr_size: Display size in pixels for QR codes.

    Returns:
        HTML string for stories.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        parts.append('<div class="story">')
        qr_html = ""
        if story.url:
            qr_svg = _make_qr_svg(story.url, display_size=qr_size, color=colors.text_dim)
            qr_html = f'  <div class="qr-code" style="flex: 0 0 {qr_size}px;">{qr_svg}</div>'
        parts.append('  <div class="headline-row">')
        parts.append(f'    <div class="headline" style="flex: 1;">{story.headline}</div>')
        if qr_html:
            parts.append(qr_html)
        parts.append('  </div>')
        parts.append(f'  <div class="body">{story.summary}</div>')
        parts.append("</div>")

    return "\n".join(parts)


def _build_stories_magazine(
    stories: list["Story"],
    colors: "LayoutColors",  # type: ignore[name-defined]
    qr_size: int = 60,
) -> str:
    """Build stories HTML for magazine template.

    Args:
        stories: List of Story objects.
        colors: LayoutColors for text styling.
        qr_size: Display size in pixels for QR codes.

    Returns:
        HTML string for stories.
    """
    if not stories:
        return ""

    parts = []
    for story in stories:
        parts.append('<div class="story">')
        qr_html = ""
        if story.url:
            qr_svg = _make_qr_svg(story.url, display_size=qr_size, color=colors.text_dim)
            qr_html = f'  <div class="qr-code" style="flex: 0 0 {qr_size}px;">{qr_svg}</div>'
        parts.append('  <div class="headline-row">')
        parts.append(f'    <div class="headline" style="flex: 1;">{story.headline}</div>')
        if qr_html:
            parts.append(qr_html)
        parts.append('  </div>')
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
    # Write HTML to a temp file next to output so file:// image refs work
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html_content, encoding="utf-8")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page(
                viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
            )
            # Use goto with file:// URL instead of set_content,
            # because set_content uses about:blank as base URL which
            # blocks file:// image references due to security policy.
            await page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            # Wait for fonts and images to load
            await page.wait_for_timeout(1500)
            await page.screenshot(path=str(output_path), type="png")
        finally:
            await browser.close()
