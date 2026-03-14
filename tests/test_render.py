"""Tests for render module.

Contract verification for ARCHITECTURE.md#2.7.
"""

import dataclasses
import inspect
import tempfile
import typing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

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
    WhisperTemplateVariables,
    PlaywrightError,
    build_template_variables,
    render_to_png,
    render_to_png_sync,
)
from sfumato.layout_ai import (
    ArtFact,
    LayoutColors,
    LayoutParams,
    ScrimParams,
    SubjectZone,
    TextZone,
    WhisperZone,
)
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
        subject_zone=SubjectZone(
            position="bottom-left",
            reason="Village and cypress dominate the lower-left mass.",
        ),
        whisper_zone=WhisperZone(
            position="top-left",
            reason="Quiet corner opposite the main news block.",
            max_width_percent=18,
            readability_notes="Dark, low-detail sky remains readable at TV distance.",
        ),
        art_facts=[
            ArtFact(text="Painted in Saint-Remy during van Gogh's asylum stay."),
            ArtFact(text="The village is imagined rather than directly observed."),
        ],
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
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_painting_info.title = "New Title"  # type: ignore[misc]


def test_render_context_is_mutable(sample_render_context: RenderContext):
    """RenderContext should be mutable (regular dataclass)."""
    sample_render_context.language = "zh"
    assert sample_render_context.language == "zh"


def test_render_context_exposes_whisper_fact_index_contract():
    """RenderContext should expose caller-owned whisper_fact_index metadata."""
    field = next(
        candidate
        for candidate in dataclasses.fields(RenderContext)
        if candidate.name == "whisper_fact_index"
    )

    assert field.default is None


def test_whisper_template_variables_type_defines_required_keys():
    """WhisperTemplateVariables should define the four additive whisper placeholders."""
    assert typing.get_type_hints(WhisperTemplateVariables) == {
        "WHISPER_POSITION": str,
        "WHISPER_COLOR": str,
        "WHISPER_SHADOW": str,
        "WHISPER_TEXT": str,
    }


def test_build_template_variables_docstring_defines_whisper_contract():
    """build_template_variables docstring should define whisper mapping and index semantics."""
    doc = inspect.getdoc(build_template_variables)
    assert doc is not None

    for token in (
        "WHISPER_POSITION",
        "WHISPER_COLOR",
        "WHISPER_SHADOW",
        "WHISPER_TEXT",
        "ctx.layout.whisper_zone.position",
        "ctx.layout.whisper_zone.max_width_percent",
        "ctx.layout.colors.text_dim",
        "ctx.layout.colors.text_shadow",
        "ctx.layout.art_facts[ctx.whisper_fact_index].text",
        "ctx.whisper_fact_index is owned by the caller",
        "Rotation is external to render",
        "The return type remains dict[str, str].",
    ):
        assert token in doc


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

        # Should have 8 whisper placeholders total:
        # 4 in <template> contract (POSITION, COLOR, SHADOW, TEXT) +
        # 4 in visible .whisper-zone (POSITION, COLOR, SHADOW, TEXT)
        assert len(whisper_placeholders) == 8, (
            f"Template {template_name} has {len(whisper_placeholders)} whisper placeholders, expected 8"
        )

        # Verify valid HTML structure around whisper contract
        assert content.count("<template") == content.count("</template>"), (
            f"Template {template_name} has mismatched template tags"
        )


def test_whisper_zone_visible_element_present():
    """All three templates must have visible .whisper-zone element for rendering."""
    templates = ["painting_text.html", "magazine.html", "portrait.html"]

    for template_name in templates:
        template_path = Path("templates") / template_name
        content = template_path.read_text()

        # All must have the visible whisper-zone div
        assert '<div class="whisper-zone">{{WHISPER_TEXT}}</div>' in content, (
            f"Template {template_name} missing visible whisper-zone div element"
        )

        # Verify CSS styling for whisper-zone exists
        assert ".whisper-zone {" in content, (
            f"Template {template_name} missing whisper-zone CSS styling"
        )

        # Verify key typography constraints from WHISPER_TEMPLATE_CONTRACT.md
        assert "font-size: 28px" in content, (
            f"Template {template_name} missing 28px font-size for whisper readability"
        )
        assert "text-shadow:" in content.lower() or "{{WHISPER_SHADOW}}" in content, (
            f"Template {template_name} missing text-shadow for whisper readability"
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
        subject_zone=SubjectZone(
            position="bottom-left",
            reason="The wanderer and ridge dominate the lower-left subject area.",
        ),
        whisper_zone=WhisperZone(
            position="top-right",
            reason="Upper-right panel space stays separate from the subject and news.",
            max_width_percent=16,
            readability_notes="The upper-right panel area remains quiet enough for small facts.",
        ),
        art_facts=[
            ArtFact(text="Painted around 1818 during German Romanticism."),
        ],
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


# =============================================================================
# WHISPER CONTRACT TESTS
# =============================================================================


def test_whisper_fact_index_defaults_to_none(sample_render_context: RenderContext):
    """RenderContext.whisper_fact_index should default to None (disabled state)."""
    # When not specified, whisper_fact_index should be None
    ctx = RenderContext(
        painting=sample_render_context.painting,
        stories=sample_render_context.stories,
        layout=sample_render_context.layout,
        palette=sample_render_context.palette,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
    )
    assert ctx.whisper_fact_index is None


def test_whisper_fact_index_accepts_valid_integer(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """RenderContext.whisper_fact_index should accept integer values."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=0,  # Valid: first art-fact
    )
    assert ctx.whisper_fact_index == 0


# =============================================================================
# WHISPER VARIABLE POPULATION CONTRACT TESTS
# =============================================================================


def test_build_template_variables_includes_whisper_position(
    sample_render_context: RenderContext,
):
    """WHISPER_POSITION should be derived from layout.whisper_zone."""
    variables = build_template_variables(sample_render_context)

    # WHISPER_POSITION must be in template variables
    assert "WHISPER_POSITION" in variables, (
        "build_template_variables must populate WHISPER_POSITION"
    )

    # WHISPER_POSITION should be a CSS declaration fragment
    position = variables["WHISPER_POSITION"]
    # Valid CSS position contains positioning keywords
    assert "top" in position or "bottom" in position, (
        "WHISPER_POSITION should contain vertical positioning"
    )
    assert "left" in position or "right" in position, (
        "WHISPER_POSITION should contain horizontal positioning"
    )


def test_build_template_variables_whisper_position_derives_from_zone(
    sample_render_context: RenderContext,
):
    """WHISPER_POSITION should derive from layout.whisper_zone.position and max_width_percent."""
    variables = build_template_variables(sample_render_context)

    position = variables["WHISPER_POSITION"]

    # The whisper zone from fixture has position="top-left" and max_width_percent=18
    # The rendered CSS should reflect these
    expected_zone = sample_render_context.layout.whisper_zone

    # Position mapping: "top-left" -> "top: ...; left: ...;"
    # The exact values depend on implementation, but semantics must align
    if expected_zone.position == "top-left":
        assert "top" in position, "top-left zone should have top positioning"
        assert "left" in position, "top-left zone should have left positioning"
    elif expected_zone.position == "top-right":
        assert "top" in position
        assert "right" in position
    elif expected_zone.position == "bottom-left":
        assert "bottom" in position
        assert "left" in position
    elif expected_zone.position == "bottom-right":
        assert "bottom" in position
        assert "right" in position


def test_build_template_variables_includes_whisper_color(
    sample_render_context: RenderContext,
):
    """WHISPER_COLOR should be derived from layout.colors.text_dim."""
    variables = build_template_variables(sample_render_context)

    assert "WHISPER_COLOR" in variables, (
        "build_template_variables must populate WHISPER_COLOR"
    )

    # WHISPER_COLOR should match text_dim (subordinate color)
    expected_color = sample_render_context.layout.colors.text_dim
    assert variables["WHISPER_COLOR"] == expected_color, (
        f"WHISPER_COLOR should be text_dim ({expected_color}), got {variables['WHISPER_COLOR']}"
    )


def test_build_template_variables_includes_whisper_shadow(
    sample_render_context: RenderContext,
):
    """WHISPER_SHADOW should be derived from layout.colors.text_shadow."""
    variables = build_template_variables(sample_render_context)

    assert "WHISPER_SHADOW" in variables, (
        "build_template_variables must populate WHISPER_SHADOW"
    )

    # WHISPER_SHADOW should match text_shadow (readability shadow)
    expected_shadow = sample_render_context.layout.colors.text_shadow
    assert variables["WHISPER_SHADOW"] == expected_shadow, (
        f"WHISPER_SHADOW should be text_shadow ({expected_shadow}), "
        f"got {variables['WHISPER_SHADOW']}"
    )


def test_build_template_variables_whisper_text_none_disables_whisper(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """WHISPER_TEXT should be empty string when whisper_fact_index is None."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=None,  # Disabled
    )

    variables = build_template_variables(ctx)

    assert "WHISPER_TEXT" in variables, (
        "build_template_variables must populate WHISPER_TEXT (even if empty)"
    )
    assert variables["WHISPER_TEXT"] == "", (
        "WHISPER_TEXT should be empty string when whisper_fact_index is None"
    )


def test_build_template_variables_whisper_text_selects_art_fact_index_zero(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """WHISPER_TEXT should select the first art-fact when whisper_fact_index=0."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=0,  # First art-fact
    )

    variables = build_template_variables(ctx)

    assert "WHISPER_TEXT" in variables

    # First art-fact from fixture
    expected_text = sample_layout_params.art_facts[0].text
    assert variables["WHISPER_TEXT"] == expected_text, (
        f"WHISPER_TEXT should be first art-fact text, got {variables['WHISPER_TEXT']}"
    )


def test_build_template_variables_whisper_text_selects_art_fact_index_one(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """WHISPER_TEXT should select the second art-fact when whisper_fact_index=1."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=1,  # Second art-fact
    )

    variables = build_template_variables(ctx)

    assert "WHISPER_TEXT" in variables

    # Second art-fact from fixture
    expected_text = sample_layout_params.art_facts[1].text
    assert variables["WHISPER_TEXT"] == expected_text, (
        f"WHISPER_TEXT should be second art-fact text, got {variables['WHISPER_TEXT']}"
    )


def test_build_template_variables_whisper_text_preserves_caller_index(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """RenderContext preserves the caller-provided whisper_fact_index without modification."""
    # Two contexts with same layout but different indices should produce different WHISPER_TEXT
    ctx_0 = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=0,
    )

    ctx_1 = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name="painting_text",
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=1,
    )

    vars_0 = build_template_variables(ctx_0)
    vars_1 = build_template_variables(ctx_1)

    # Different indices produce different whisper text (no rotation in render)
    assert vars_0["WHISPER_TEXT"] != vars_1["WHISPER_TEXT"], (
        "Different whisper_fact_index values should produce different WHISPER_TEXT"
    )


# =============================================================================
# ART-FACT SELECTION INDEXING CONTRACT TESTS
# =============================================================================


def test_art_facts_index_valid_range(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
):
    """Valid whisper_fact_index must be within range [0, len(art_facts)-1]."""
    n_facts = len(sample_layout_params.art_facts)

    # All valid indices should work
    for i in range(n_facts):
        ctx = RenderContext(
            painting=sample_painting_info,
            stories=[sample_story],
            layout=sample_layout_params,
            palette=sample_palette_colors,
            template_name="painting_text",
            language="en",
            date_str="2024-03-14",
            time_str="08:00",
            whisper_fact_index=i,
        )
        variables = build_template_variables(ctx)
        assert variables["WHISPER_TEXT"] == sample_layout_params.art_facts[i].text


def test_art_facts_list_has_expected_count(
    sample_layout_params: LayoutParams,
):
    """Layout should contain art_facts list with 1-3 items (per ARCHITECTURE.md contract)."""
    art_facts = sample_layout_params.art_facts
    assert isinstance(art_facts, list), "art_facts should be a list"
    assert 1 <= len(art_facts) <= 3, "art_facts should contain 1-3 items"


def test_art_fact_text_is_string(sample_layout_params: LayoutParams):
    """Each ArtFact should have a string text field."""
    for i, fact in enumerate(sample_layout_params.art_facts):
        assert isinstance(fact.text, str), f"art_facts[{i}].text should be a string"
        assert len(fact.text) > 0, f"art_facts[{i}].text should not be empty"


# =============================================================================
# WHISPER ZONE CONTRACT TESTS
# =============================================================================


def test_whisper_zone_exists_in_layout(sample_layout_params: LayoutParams):
    """LayoutParams should include whisper_zone for secondary text placement."""
    assert hasattr(sample_layout_params, "whisper_zone")
    assert sample_layout_params.whisper_zone is not None


def test_whisper_zone_has_required_fields(sample_layout_params: LayoutParams):
    """WhisperZone should have position, reason, max_width_percent, readability_notes."""
    wz = sample_layout_params.whisper_zone

    assert hasattr(wz, "position")
    assert hasattr(wz, "reason")
    assert hasattr(wz, "max_width_percent")
    assert hasattr(wz, "readability_notes")


def test_whisper_zone_max_width_percent_range(sample_layout_params: LayoutParams):
    """WhisperZone.max_width_percent must be between 12 and 24 (per ARCHITECTURE.md contract)."""
    wz = sample_layout_params.whisper_zone

    assert 12 <= wz.max_width_percent <= 24, (
        f"whisper_zone.max_width_percent must be in [12, 24], got {wz.max_width_percent}"
    )


# =============================================================================
# WHISPER VARIABLE ADDITIVITY CONTRACT TESTS
# =============================================================================


def test_whisper_variables_are_additive(sample_render_context: RenderContext):
    """WHISPER_* variables should be additive to existing template variables."""
    variables = build_template_variables(sample_render_context)

    # Existing variables should still be present
    existing_vars = [
        "BG_IMAGE",
        "DATE",
        "UPDATE_TIME",
        "ARTIST",
        "PAINTING_TITLE",
        "TEXT_COLOR",
        "TEXT_COLOR_SEC",
        "TEXT_COLOR_DIM",
        "TEXT_SHADOW",
    ]

    for var in existing_vars:
        assert var in variables, f"Existing {var} should still be present"

    # Whisper variables should also be present
    whisper_vars = [
        "WHISPER_POSITION",
        "WHISPER_COLOR",
        "WHISPER_SHADOW",
        "WHISPER_TEXT",
    ]

    for var in whisper_vars:
        assert var in variables, f"WHISPER variable {var} should be present"


def test_whisper_variables_do_not_replace_existing(
    sample_render_context: RenderContext,
):
    """WHISPER_* variables should not rename or replace existing keys."""
    variables = build_template_variables(sample_render_context)

    # Verify no KEY collisions (different semantic purposes)
    # WHISPER_COLOR should be separate from TEXT_COLOR_DIM even if values match
    assert "WHISPER_COLOR" in variables
    assert "TEXT_COLOR_DIM" in variables
    # These may have the same value (both derived from text_dim)
    # but they must both exist for their respective template roles

    # Similarly for WHISPER_SHADOW vs TEXT_SHADOW
    assert "WHISPER_SHADOW" in variables
    assert "TEXT_SHADOW" in variables


# =============================================================================
# TEMPLATE COMPATIBILITY TESTS
# =============================================================================


@pytest.mark.parametrize(
    "template_name",
    ["painting_text", "magazine", "portrait"],
)
def test_whisper_variables_available_for_all_templates(
    sample_painting_info: PaintingInfo,
    sample_story: Story,
    sample_layout_params: LayoutParams,
    sample_palette_colors: PaletteColors,
    template_name: str,
):
    """WHISPER_* variables should be populated for all three templates."""
    ctx = RenderContext(
        painting=sample_painting_info,
        stories=[sample_story],
        layout=sample_layout_params,
        palette=sample_palette_colors,
        template_name=template_name,
        language="en",
        date_str="2024-03-14",
        time_str="08:00",
        whisper_fact_index=0,
    )

    variables = build_template_variables(ctx)

    whisper_vars = [
        "WHISPER_POSITION",
        "WHISPER_COLOR",
        "WHISPER_SHADOW",
        "WHISPER_TEXT",
    ]

    for var in whisper_vars:
        assert var in variables, f"{var} should be present for template {template_name}"


def test_render_context_whisper_fact_index_is_caller_owned(
    sample_render_context: RenderContext,
):
    """whisper_fact_index is caller-owned metadata, never modified by render."""
    # Store original index
    original_index = sample_render_context.whisper_fact_index

    # Call build_template_variables (render should not mutate context)
    _ = build_template_variables(sample_render_context)

    # Verify index wasn't modified
    assert sample_render_context.whisper_fact_index == original_index, (
        "render should not modify whisper_fact_index"
    )
