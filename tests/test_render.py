"""Tests for render module.

Contract verification for ARCHITECTURE.md#2.7.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sfumato.render import (
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    SUPPORTED_TEMPLATES,
    Orientation,
    PaintingInfo,
    RenderContext,
    RenderError,
    RenderResult,
    TemplateNotFoundError,
    PlaywrightError,
    build_template_variables,
    render_to_png,
    render_to_png_sync,
)
from sfumato.layout_ai import LayoutColors, LayoutParams, ScrimParams, TextZone
from sfumato.news import Story
from sfumato.palette import PaletteColors


WHISPER_CONTRACT_TEMPLATE = """<template id=\"whisper-contract\">\n    <div class=\"whisper-contract\" data-whisper-position=\"{{WHISPER_POSITION}}\" data-whisper-color=\"{{WHISPER_COLOR}}\" data-whisper-shadow=\"{{WHISPER_SHADOW}}\">{{WHISPER_TEXT}}</div>\n  </template>"""


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_painting_info(tmp_path: Path) -> PaintingInfo:
    """Create a sample PaintingInfo for testing."""
    image_path = tmp_path / "test_painting.jpg"
    image_path.touch()  # Create empty file
    return PaintingInfo(
        image_path=image_path,
        content_hash="abc123",
        title="Starry Night",
        artist="Vincent van Gogh",
        year="1889",
        orientation=Orientation.LANDSCAPE,
        width=3840,
        height=2160,
        source="met",
        source_id="436535",
        source_url="https://example.com/painting",
    )


@pytest.fixture
def sample_story() -> Story:
    """Create a sample Story for testing."""
    return Story(
        headline="AI Revolution",
        summary="Large language models continue to advance rapidly.",
        source="Tech News",
        category="AI",
        url="https://example.com/article",
        published_at=datetime.now(),
        featured=True,
    )


@pytest.fixture
def sample_layout_params() -> LayoutParams:
    """Create a sample LayoutParams for testing."""
    return LayoutParams(
        orientation="landscape",
        painting_title="Starry Night",
        painting_artist="Vincent van Gogh",
        painting_description="A swirling night sky with bright stars",
        text_zone=TextZone(position="top-right", reason="Dark sky area"),
        colors=LayoutColors(
            text_primary="#ffffff",
            text_secondary="#e0e0e0",
            text_dim="#a0a0a0",
            text_shadow="0 2px 4px rgba(0,0,0,0.5)",
            scrim_color="rgba(0,0,0,0.4)",
            panel_bg="#1a1a1a",
            border="#333333",
            accent="#ffd700",
        ),
        scrim=ScrimParams(
            position_css="top: 0; right: 0;",
            size_css="width: 1800px; height: 1400px;",
            gradient_css="radial-gradient(ellipse at top right, rgba(0,0,0,0.4) 0%, transparent 70%)",
        ),
        recommended_stories=3,
        template_hint="painting_text",
        portrait_layout=None,
    )


@pytest.fixture
def sample_layout_colors() -> LayoutColors:
    """Create a sample LayoutColors for testing."""
    return LayoutColors(
        text_primary="#ffffff",
        text_secondary="#e0e0e0",
        text_dim="#a0a0a0",
        text_shadow="0 2px 4px rgba(0,0,0,0.5)",
        scrim_color="rgba(0,0,0,0.4)",
        panel_bg="#1a1a1a",
        border="#333333",
        accent="#ffd700",
    )


@pytest.fixture
def sample_palette_colors() -> PaletteColors:
    """Create a sample PaletteColors for testing."""
    return PaletteColors(
        dominant="#1a237e",
        secondary="#283593",
        accent="#ffd700",
        background="#0d1b2a",
        is_dark=True,
        colors=("#1a237e", "#283593", "#ffd700", "#0d1b2a", "#424242"),
    )


@pytest.fixture
def sample_render_context(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
) -> RenderContext:
    """Create a sample RenderContext for testing."""
    return RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
    )


# =============================================================================
# DATA MODEL TESTS
# =============================================================================


def test_painting_info_is_frozen(sample_painting_info: PaintingInfo):
    """PaintingInfo should be immutable (frozen dataclass)."""
    # FrozenInstanceError is raised when trying to modify a frozen dataclass
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_painting_info.title = "New Title"  # type: ignore[misc]


def test_render_context_is_mutable(sample_render_context: RenderContext):
    """RenderContext should be mutable (regular dataclass)."""
    sample_render_context.language = "zh"
    assert sample_render_context.language == "zh"


def test_render_result_is_mutable():
    """RenderResult should be mutable (regular dataclass)."""
    result = RenderResult(
        png_path=Path("/tmp/test.png"),
        html_path=Path("/tmp/test.html"),
        template_used="painting_text",
        story_count=3,
        painting_hash="abc123",
    )
    result.story_count = 5
    assert result.story_count == 5


def test_supported_templates():
    """SUPPORTED_TEMPLATES should contain all required templates."""
    expected = {"painting_text", "magazine", "portrait", "art_overlay", "art_minimal"}
    assert SUPPORTED_TEMPLATES == expected


@pytest.mark.parametrize(
    ("template_name", "required_anchor"),
    [
        ("painting_text.html", '<div class="text-zone">'),
        ("magazine.html", '<div class="text-zone">'),
        ("portrait.html", '<div class="right-panel">'),
    ],
)
def test_templates_include_inert_whisper_contract_placeholder(
    template_name: str,
    required_anchor: str,
):
    """Production templates should expose the shared whisper placeholder shape."""
    template_path = Path("templates") / template_name
    template_content = template_path.read_text()

    assert WHISPER_CONTRACT_TEMPLATE in template_content
    assert required_anchor in template_content


def test_whisper_contract_specifies_variables_and_constraints():
    """Whisper contract doc should define variables and shared constraints."""
    contract = Path("WHISPER_TEMPLATE_CONTRACT.md").read_text()

    for variable in (
        "WHISPER_POSITION",
        "WHISPER_COLOR",
        "WHISPER_SHADOW",
        "WHISPER_TEXT",
    ):
        assert variable in contract

    assert "Shared Placeholder Shape" in contract
    assert "Shared Typography Constraints" in contract
    assert "Shared Positioning Constraints" in contract
    assert "Compatibility Expectations" in contract


@pytest.mark.parametrize(
    "template_name",
    [
        "painting_text.html",
        "magazine.html",
        "portrait.html",
    ],
)
def test_templates_expose_all_required_whisper_variables(template_name: str):
    """All three templates must expose required WHISPER_* placeholders."""
    template_path = Path("templates") / template_name
    template_content = template_path.read_text()

    required_whisper_vars = [
        "WHISPER_POSITION",
        "WHISPER_COLOR",
        "WHISPER_SHADOW",
        "WHISPER_TEXT",
    ]

    for var in required_whisper_vars:
        assert f"{{{{{var}}}}}" in template_content, (
            f"Template {template_name} missing required WHISPER_* variable: {var}"
        )


def test_whisper_contract_uniform_structure_across_templates():
    """Whisper contract structure should be uniform in all three templates."""
    templates = ["painting_text.html", "magazine.html", "portrait.html"]

    for template_name in templates:
        template_path = Path("templates") / template_name
        content = template_path.read_text()

        # All must have the whisper-contract template element
        assert '<template id="whisper-contract">' in content, (
            f"Template {template_name} missing whisper-contract template element"
        )
        assert "</template>" in content, (
            f"Template {template_name} has unclosed template element"
        )

        # All must have consistent data attributes
        assert 'data-whisper-position="{{WHISPER_POSITION}}"' in content, (
            f"Template {template_name} missing whisper-position data attribute"
        )
        assert 'data-whisper-color="{{WHISPER_COLOR}}"' in content, (
            f"Template {template_name} missing whisper-color data attribute"
        )
        assert 'data-whisper-shadow="{{WHISPER_SHADOW}}"' in content, (
            f"Template {template_name} missing whisper-shadow data attribute"
        )
        assert "{{WHISPER_TEXT}}</div>" in content, (
            f"Template {template_name} missing whisper-text content"
        )


def test_whisper_template_syntax_valid():
    """Smoke test: whisper placeholders have valid syntax in all templates."""
    templates = ["painting_text.html", "magazine.html", "portrait.html"]

    for template_name in templates:
        template_path = Path("templates") / template_name
        content = template_path.read_text()

        # Verify placeholder syntax: {{VARIABLE_NAME}} with valid chars
        import re

        # Find all whisper-related placeholders
        whisper_placeholders = re.findall(r"\{\{WHISPER_[A-Z_]+\}\}", content)

        # Should have exactly 4 whisper placeholders (POSITION, COLOR, SHADOW, TEXT)
        assert len(whisper_placeholders) == 4, (
            f"Template {template_name} has {len(whisper_placeholders)} whisper placeholders, expected 4"
        )

        # Verify valid HTML structure around whisper contract
        assert content.count("<template") == content.count("</template>"), (
            f"Template {template_name} has mismatched template tags"
        )


# =============================================================================
# build_template_variables TESTS
# =============================================================================


def test_build_template_variables_painting_text(sample_render_context: RenderContext):
    """build_template_variables should correctly map painting_text template variables."""
    variables = build_template_variables(sample_render_context)

    # Required variables for painting_text template
    assert "BG_IMAGE" in variables
    assert "SCRIM_POSITION" in variables
    assert "SCRIM_GRADIENT" in variables
    assert "TEXT_POSITION" in variables
    assert "TEXT_WIDTH" in variables
    assert "TEXT_COLOR" in variables
    assert "TEXT_COLOR_SEC" in variables
    assert "TEXT_COLOR_DIM" in variables
    assert "TEXT_SHADOW" in variables
    assert "DATELINE" in variables
    assert "NEWS_BLOCKS" in variables

    # Verify values
    assert variables["BG_IMAGE"].startswith("file://")
    assert variables["TEXT_COLOR"] == "#ffffff"
    assert "08:00" in variables["DATELINE"]


def test_build_template_variables_with_date_time(sample_render_context: RenderContext):
    """build_template_variables should include date and time strings."""
    variables = build_template_variables(sample_render_context)

    assert "DATE" in variables
    assert "UPDATE_TIME" in variables
    assert variables["DATE"] == "2024-03-14"
    assert variables["UPDATE_TIME"] == "08:00"


def test_build_template_variables_empty_stories(
    sample_painting_info: PaintingInfo,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """build_template_variables should handle empty stories list."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
    )
    variables = build_template_variables(ctx)

    assert variables["NEWS_BLOCKS"] == ""
    assert variables["ARTIST"]  # Painting info should still be present


def test_build_template_variables_portrait(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_colors: LayoutColors,
    sample_palette_colors: PaletteColors,
):
    """build_template_variables should correctly handle portrait template."""
    from sfumato.layout_ai import PortraitLayout

    # Create portrait-specific layout
    layout = LayoutParams(
        orientation="portrait",
        painting_title="Wanderer above the Sea of Fog",
        painting_artist="Caspar David Friedrich",
        painting_description="Contemplative figure overlooking misty landscape",
        text_zone=TextZone(position="right-side", reason="Right panel for news"),
        colors=sample_layout_colors,
        scrim=ScrimParams(
            position_css="right: 0; top: 0;",
            size_css="width: 800px; height: 2160px;",
            gradient_css="linear-gradient(to right, transparent, rgba(0,0,0,0.3))",
        ),
        recommended_stories=3,
        template_hint="portrait",
        portrait_layout=PortraitLayout(
            painting_width_percent=50,
            left_panel_color="#1c1e20",
            right_panel_color="#1c1e20",
            info_side="left",
        ),
    )

    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=layout,
        palette=sample_palette_colors,
        template_name="portrait",
        language="zh",
        date_str="2024-03-14",
        time_str="08:00",
    )

    variables = build_template_variables(ctx)

    # Portrait-specific variables
    assert "BG_COLOR" in variables
    assert "LEFT_WIDTH" in variables
    assert "PAINTING_WIDTH" in variables
    assert "PAINTING_SRC" in variables
    assert "STORIES" in variables


def test_build_template_variables_magazine(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """build_template_variables should correctly handle magazine template."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="magazine",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
    )

    variables = build_template_variables(ctx)

    # Magazine-specific variables
    assert "PANEL_BG" in variables
    assert "STORIES" in variables
    assert "ARTIST" in variables


def test_build_template_variables_art_overlay(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """build_template_variables should correctly handle art_overlay template."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="art_overlay",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
    )

    variables = build_template_variables(ctx)

    # art_overlay-specific variables
    assert "TITLE" in variables
    assert "STORIES" in variables
    assert "ART_CREDIT" in variables


# =============================================================================
# render_to_png TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_render_to_png_template_not_found(sample_render_context: RenderContext):
    """render_to_png should raise TemplateNotFoundError for missing template."""
    sample_render_context.template_name = "nonexistent_template"

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(TemplateNotFoundError):
            await render_to_png(sample_render_context, Path(tmpdir))


@pytest.mark.asyncio
async def test_render_to_png_creates_output(
    sample_render_context: RenderContext,
):
    """render_to_png should create PNG and HTML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Mock Playwright to avoid actual rendering
        with patch("sfumato.render._playwright_screenshot") as mock_screenshot:
            mock_screenshot.return_value = None

            result = await render_to_png(sample_render_context, output_dir)

            # Verify result structure
            assert isinstance(result, RenderResult)
            assert result.template_used == "painting_text"
            assert result.story_count == 1
            assert result.painting_hash == "abc123"

            # Verify files were created
            assert result.html_path.exists()
            assert result.html_path.suffix == ".html"

            # HTML should have variables substituted
            html_content = result.html_path.read_text()
            assert "2024-03-14" in html_content
            assert "AI Revolution" in html_content


def test_render_to_png_sync_wrapper(sample_render_context: RenderContext):
    """render_to_png_sync should call render_to_png."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        with patch("sfumato.render._playwright_screenshot") as mock_screenshot:
            mock_screenshot.return_value = None

            result = render_to_png_sync(sample_render_context, output_dir)

            assert isinstance(result, RenderResult)
            assert result.template_used == "painting_text"


# =============================================================================
# ERROR TYPES TESTS
# =============================================================================


def test_render_error_hierarchy():
    """RenderError should be the base exception for render failures."""
    assert issubclass(TemplateNotFoundError, RenderError)
    assert issubclass(PlaywrightError, RenderError)
    assert issubclass(RenderError, Exception)


# =============================================================================
# VIEWPORT TESTS
# =============================================================================


def test_viewport_dimensions():
    """Viewport should be 4K resolution (3840x2160)."""
    assert VIEWPORT_WIDTH == 3840
    assert VIEWPORT_HEIGHT == 2160
