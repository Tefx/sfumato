"""LLM-powered painting composition analysis for layout design.

This module analyzes paintings using vision-capable LLMs to determine
optimal text placement, color schemes, and layout parameters for
overlaying news content on artwork displayed on Samsung The Frame TV.

Architecture reference: ARCHITECTURE.md#2.5
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, TypeAlias

from sfumato.config import AiConfig
from sfumato.llm import LlmError, LlmParseError, invoke_vision, parse_json_response

if TYPE_CHECKING:
    pass


OrientationName: TypeAlias = Literal["landscape", "portrait"]
ZonePosition: TypeAlias = Literal[
    "top-left",
    "top-right",
    "bottom-left",
    "bottom-right",
    "left-side",
    "right-side",
]
TemplateHint: TypeAlias = Literal[
    "painting_text", "magazine", "portrait", "art_overlay"
]
InfoSide: TypeAlias = Literal["left", "right", "both"]


class TextZoneJson(TypedDict):
    """LLM JSON contract for the primary news text zone."""

    position: ZonePosition
    reason: str


class SubjectZoneJson(TypedDict):
    """LLM JSON contract for the painting subject zone to preserve."""

    position: ZonePosition
    reason: str


class WhisperZoneJson(TypedDict):
    """LLM JSON contract for low-emphasis whisper text placement.

    Contract:
        - max_width_percent defines the allowed whisper text block width.
        - readability_notes must explain why the zone remains legible at TV distance.
    """

    position: ZonePosition
    reason: str
    max_width_percent: int
    readability_notes: str


class LayoutColorsJson(TypedDict):
    """LLM JSON contract for overlay color recommendations."""

    text_primary: str
    text_secondary: str
    text_dim: str
    text_shadow: str
    scrim_color: str
    panel_bg: str
    border: str
    accent: str


class ScrimParamsJson(TypedDict):
    """LLM JSON contract for the news scrim CSS payload."""

    position_css: str
    size_css: str
    gradient_css: str


class PortraitLayoutJson(TypedDict):
    """LLM JSON contract for portrait-only layout parameters."""

    painting_width_percent: int
    left_panel_color: str
    right_panel_color: str
    info_side: InfoSide


class LayoutAnalysisResponseJson(TypedDict):
    """Structured LLM response contract for layout analysis.

    Top-level output items requested from the LLM:
        1. orientation
        2. painting_title
        3. painting_artist
        4. text_zone
        5. subject_zone
        6. whisper_zone
        7. art_facts
        8. colors
        9. scrim
        10. recommended_stories

    Additional required fields:
        - painting_description
        - template_hint
        - portrait_layout
    """

    orientation: OrientationName
    painting_title: str
    painting_artist: str
    text_zone: TextZoneJson
    subject_zone: SubjectZoneJson
    whisper_zone: WhisperZoneJson
    art_facts: list[str]
    colors: LayoutColorsJson
    scrim: ScrimParamsJson
    recommended_stories: int
    painting_description: str
    template_hint: TemplateHint
    portrait_layout: PortraitLayoutJson | None


# =============================================================================
# PUBLIC ERROR TYPES
# =============================================================================


class LayoutAnalysisError(LlmError):
    """Raised when painting layout analysis fails.

    This wraps LLM invocation and parsing errors with additional context
    specific to layout analysis failures.

    Contract:
        - Subclass of LlmError (analysis is a category of LLM failure)
        - Message includes the image path and failure reason
    """

    pass


# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass(frozen=True)
class TextZone:
    """Identifies where text should be placed on a painting.

    The position is chosen based on finding the quietest visual area
    (lowest visual density) where text can be overlaid without obscuring
    important elements of the artwork.

    Attributes:
        position: One of six predefined zones:
            - "top-left": Upper left corner area
            - "top-right": Upper right corner area
            - "bottom-left": Lower left corner area
            - "bottom-right": Lower right corner area
            - "left-side": Left edge of the painting (vertical strip)
            - "right-side": Right edge of the painting (vertical strip)
        reason: LLM's natural language explanation for choosing this zone.

    Contract:
        - position must be one of the six valid values
        - reason is free-form text for debugging/display
    """

    position: ZonePosition
    reason: str


@dataclass(frozen=True)
class SubjectZone:
    """Identifies the dominant subject area that overlays must avoid.

    Attributes:
        position: Coarse zone containing the primary painted subject mass.
        reason: LLM explanation describing the protected visual subject.

    Contract:
        - position must be one of the six valid values
        - Must remain mutually exclusive with ``text_zone`` and ``whisper_zone``
    """

    position: ZonePosition
    reason: str


@dataclass(frozen=True)
class WhisperZone:
    """Placement contract for secondary whisper text carrying art facts.

    Attributes:
        position: Coarse zone reserved for low-emphasis art facts.
        reason: LLM explanation for choosing this zone.
        max_width_percent: Maximum whisper block width as percent of screen width.
        readability_notes: Why the zone remains readable at TV distance.

    Contract:
        - position must be one of the six valid values
        - Must remain mutually exclusive with ``text_zone`` and ``subject_zone``
        - max_width_percent must be between 12 and 24 inclusive
        - readability_notes must justify TV-distance readability for whisper text
    """

    position: ZonePosition
    reason: str
    max_width_percent: int
    readability_notes: str


@dataclass(frozen=True)
class ArtFact:
    """Single art-fact line for whisper text rendering.

    Contract:
        - text is presentation-ready whisper copy, not markup
        - Facts should stand alone when rendered independently
    """

    text: str


@dataclass(frozen=True)
class LayoutColors:
    """Color scheme for text overlays based on painting palette.

    Colors are chosen to harmonize with the painting while ensuring
    sufficient contrast for readability. The scrim_color and panel_bg
    provide subtle backgrounds to improve text legibility.

    Attributes:
        text_primary: Primary text color for headlines and main content.
            Format: "#RRGGBB" hex color.
        text_secondary: Secondary text color for subheadings.
            Format: "#RRGGBB" hex color.
        text_dim: Dimmed text color for metadata and less important info.
            Format: "#RRGGBB" or "rgba(R,G,B,A)" for transparency.
        text_shadow: CSS text-shadow value for improving contrast.
            Format: CSS text-shadow syntax (e.g., "0 2px 4px rgba(0,0,0,0.5)").
        scrim_color: Semi-transparent background for text zone.
            Format: "rgba(R,G,B,A)" for subtle darkening/lightening.
        panel_bg: Background color for magazine/portrait side panels.
            Format: "#RRGGBB" hex color.
        border: Divider/border color for structured layouts.
            Format: "#RRGGBB" hex color.
        accent: Highlight color for category labels and emphasis.
            Format: "#RRGGBB" hex color.

    Contract:
        - All colors must be valid CSS color values
        - Colors should complement the painting's palette
        - scrim_color should provide enough contrast for text readability
    """

    text_primary: str
    text_secondary: str
    text_dim: str
    text_shadow: str
    scrim_color: str
    panel_bg: str
    border: str
    accent: str


@dataclass(frozen=True)
class ScrimParams:
    """CSS styling for the semi-transparent text backdrop.

    The scrim is a gradient overlay placed behind text to improve
    readability without fully obscuring the painting. It uses radial
    or linear gradients to create a smooth transition.

    Attributes:
        position_css: CSS position properties for the scrim div.
            Format: CSS properties string (e.g., "top: 0; right: 0;").
        size_css: CSS width and height for the scrim div.
            Format: CSS dimension string (e.g., "width: 1800px; height: 1200px;").
        gradient_css: CSS gradient value for the scrim.
            Format: CSS gradient (e.g., "radial-gradient(ellipse at top right, ...)").

    Contract:
        - All values must be valid CSS property strings
        - Gradient should create subtle contrast, not block the painting
        - Position should align with text_zone position
    """

    position_css: str
    size_css: str
    gradient_css: str


@dataclass(frozen=True)
class PortraitLayout:
    """Layout parameters specific to portrait-mode orientation.

    When a painting is in portrait orientation, it's displayed between
    two side panels. The painting occupies the center, with information
    and news on the sides.

    Attributes:
        painting_width_percent: How much of screen width the painting occupies.
            Range: 45-55 (typically 50 for balance).
        left_panel_color: Background color for the left panel.
            Format: "#RRGGBB" hex color.
        right_panel_color: Background color for the right panel.
            Format: "#RRGGBB" hex color.
        info_side: Which side displays painting information.
            Values: "left", "right", or "both".

    Contract:
        - painting_width_percent must be between 45 and 55
        - Panel colors should be derived from painting edge colors
        - If info_side is "both", split layout with artist info on one side
    """

    painting_width_percent: int
    left_panel_color: str
    right_panel_color: str
    info_side: InfoSide


@dataclass(frozen=True)
class LayoutParams:
    """Complete layout parameters for rendering news over a painting.

    This is the main result of layout analysis, containing all the
    information needed by templates to render text overlays.

    Attributes:
        orientation: Detected painting orientation.
            Values: "landscape" or "portrait".
        painting_title: LLM-identified title of the painting.
            May be "Unknown" if not identifiable.
        painting_artist: LLM-identified artist name.
            May be "Unknown" if not identifiable.
        painting_description: Free-form description for semantic matching.
            Rich natural language describing mood, atmosphere, colors,
            and subject matter. Used for embedding-based matching with
            news tone descriptions.
        text_zone: Where to place text overlays.
        subject_zone: Where the painting's primary subject mass sits.
        whisper_zone: Reserved low-emphasis area for art facts.
        art_facts: Short whisper-copy facts about the artwork.
        colors: Color scheme for text and overlays.
        scrim: CSS parameters for the text backdrop.
        recommended_stories: LLM's suggestion for story count.
            Range: 2-5 stories that fit comfortably in the layout.
        template_hint: LLM's recommendation for template selection.
            Values: "painting_text", "magazine", "portrait", "art_overlay".
            Final selection may differ based on orchestrator logic.
        portrait_layout: Portrait-specific parameters.
            Only populated when orientation is "portrait".
            None for landscape orientations.

    Contract:
        - orientation matches the actual painting dimensions
        - painting_description is rich enough for embedding-based matching
        - text_zone, subject_zone, and whisper_zone are mutually exclusive
        - whisper_zone.max_width_percent is between 12 and 24 inclusive
        - art_facts contains 1 to 3 items suitable for whisper presentation
        - recommended_stories is between 2 and 5
        - portrait_layout is set when orientation is "portrait"
        - All nested dataclass fields are populated (no None within sub-objects)
    """

    orientation: OrientationName
    painting_title: str
    painting_artist: str
    painting_description: str
    text_zone: TextZone
    subject_zone: SubjectZone
    whisper_zone: WhisperZone
    art_facts: list[ArtFact]
    colors: LayoutColors
    scrim: ScrimParams
    recommended_stories: int
    template_hint: TemplateHint
    portrait_layout: PortraitLayout | None


# =============================================================================
# LLM PROMPT
# =============================================================================


LAYOUT_ANALYSIS_PROMPT = """\
You are a visual layout designer for Samsung The Frame TV. Analyze this painting and recommend how to overlay news text without obscuring important elements.

TV resolution: 3840x2160px (4K). The painting fills the entire screen.

Analyze and provide exactly these 10 items:

1. **Orientation**: Is this painting landscape or portrait composition?

2. **Painting Identity**: Identify the title and artist if known, or "Unknown" otherwise.

3. **News Zone**: Find the area of lowest visual density where news text can be placed. Consider:
   - Large uniform areas (skies, water, fog, dark corners)
   - Areas with simple color fields rather than complex details
   - Edges where the eye naturally rests less
   Choose from: "top-left", "top-right", "bottom-left", "bottom-right", "left-side", "right-side"

4. **Subject Zone**: Identify the coarse zone containing the painting's primary subject mass.
   This protected zone must not overlap the news zone or whisper zone.

5. **Whisper Zone**: Choose a secondary zone for art-fact whisper text.
   - Must not overlap the news zone (text_zone) — MUST be in a DIFFERENT zone
   - Must not overlap the subject zone
   - Must not overlap the painting credit area:
     * Portrait templates: bottom-left is reserved for painting credit (artist, title, year)
     * Landscape templates: dateline appears above the news zone
   - Must remain readable at TV distance
   - Width must stay between 12% and 24% of screen width
   - Keep it visually subordinate to the main news block
   - CRITICAL: whisper_zone position must differ from BOTH text_zone AND subject_zone. All three must be in different zones.

6. **Art Facts**: Produce 1-3 whisper-ready art facts.
   - Short, factual, and display-ready
   - Each item should fit comfortably inside the whisper zone
   - IMPORTANT: Write art facts in {{LANGUAGE}} language

7. **Color Harmony**: Design a text color scheme that:
   - Harmonizes with the painting's palette
   - Has sufficient contrast for readability
   - text_shadow must use a SMALL blur radius (2-4px), not large halos.
      Good: "0 1px 3px rgba(0,0,0,0.7), 0 0 8px rgba(0,0,0,0.3)"
      Bad:  "0 2px 10px rgba(0,0,0,0.5)" (too blurry, creates visible boxes)
   Text colors should feel like they belong to the painting, not fight with it.

8. **Scrim Design**: Create a subtle gradient overlay for the news zone:
   - Use radial-gradient or linear-gradient
   - Should be barely visible - just enough to improve text contrast
   - Position and size to protect the news zone only

9. **Story Count**: How many news stories (2-5) fit comfortably? Consider:
   - Visual complexity of the painting
   - Size of the available news zone
   - Balance between art and information

10. **Composition Notes**: Provide the remaining structured guidance:
   - painting_description: rich, evocative description of the painting's mood, atmosphere, colors, and subject matter for semantic matching. Write in the language most appropriate to the painting's origin or style.
   - template_hint: Which template works best? Choose based on composition density:
    - "painting_text": Full painting with text in quiet zone. ONLY use when the painting has clear, large quiet zones (>25% of area) with low visual complexity. NOT suitable for densely composed paintings.
    - "portrait": Three-column layout with painting centered (for portrait orientation)
    - "magazine": 72/28 split with painting left, text panel right. USE THIS when the painting is densely composed with no clear quiet zones — figures, details, or action fill most of the canvas. This is the SAFE choice for busy paintings.
    - "art_overlay": Frosted glass cards over painting (rare)
    IMPORTANT: If the BRIGHTNESS DATA shows all quadrants have high variance (>2000), the painting is too busy for "painting_text". Use "magazine" instead.
   - portrait_layout (if orientation is "portrait"):
     - What percentage of screen should the painting occupy? (45-55)
     - Panel colors derived from painting edge colors
     - Which side for painting info? (left/right/both)

Output strict JSON (no markdown fence, no commentary):

{
  "orientation": "landscape" or "portrait",
  "painting_title": "...",
  "painting_artist": "...",
  "text_zone": {
    "position": "top-left" or "top-right" or "bottom-left" or "bottom-right" or "left-side" or "right-side",
    "reason": "Brief explanation for choosing this zone"
  },
  "subject_zone": {
    "position": "top-left" or "top-right" or "bottom-left" or "bottom-right" or "left-side" or "right-side",
    "reason": "Brief explanation of the protected subject area"
  },
  "whisper_zone": {
    "position": "top-left" or "top-right" or "bottom-left" or "bottom-right" or "left-side" or "right-side",
    "reason": "Brief explanation for choosing this whisper zone",
    "max_width_percent": 18,
    "readability_notes": "Why the whisper text remains readable at TV distance"
  },
  "art_facts": [
    "Short art fact 1",
    "Short art fact 2"
  ],
  "colors": {
    "text_primary": "#RRGGBB",
    "text_secondary": "#RRGGBB",
    "text_dim": "#RRGGBB or rgba(...)",
    "text_shadow": "CSS text-shadow value",
    "scrim_color": "rgba(R,G,B,A)",
    "panel_bg": "#RRGGBB",
    "border": "#RRGGBB",
    "accent": "#RRGGBB"
  },
  "scrim": {
    "position_css": "top: 100px; right: 160px;",
    "size_css": "width: 1500px; height: 1400px;",
    "gradient_css": "radial-gradient(ellipse at center, rgba(0,0,0,0.4) 0%, transparent 70%)"
  },
  "recommended_stories": 3,
  "painting_description": "Free-form evocative description in appropriate language...",
  "template_hint": "painting_text" or "portrait" or "magazine" or "art_overlay",
  "portrait_layout": null or {
    "painting_width_percent": 50,
    "left_panel_color": "#RRGGBB",
    "right_panel_color": "#RRGGBB",
    "info_side": "left" or "right" or "both"
  }
}

Note: Some museum photographs may include the physical painting frame (ornate border). If you detect a frame in the image, mention it in your analysis and adjust text placement to avoid the frame border area. The scrim gradient should also account for the frame.

CRITICAL RULES:
- Colors MUST harmonize with the painting's actual palette
- text_zone, subject_zone, and whisper_zone MUST be mutually exclusive
- Use the BRIGHTNESS DATA below to choose text colors:
  - If the chosen text zone is BRIGHT (>150), use DARK text (#1a1a1a to #3a3a3a range) with light text-shadow
  - If the chosen text zone is DARK (<100), use LIGHT text (#e0e0e0 to #f0f0f0 range) with dark text-shadow
  - NEVER put light text on light areas or dark text on dark areas
- text_shadow blur radius MUST be 2-4px, never 10px+
- Scrim should be subtle, not create a visible box
- Position text at least 120-160px from screen edges
- Text zone width MUST NOT exceed 38% of screen width (max ~1460px) for landscape
- Whisper zone width MUST stay between 12% and 24% of screen width
- Whisper text must remain readable at TV distance while staying visually subordinate to the news block
- scrim size_css should cover the text zone area (e.g. "width: 40%; height: 45%;")
- Portrait paintings MUST have portrait_layout populated
"""
"""Structured prompt for layout analysis covering all 10 requested items."""


# =============================================================================
# PUBLIC API
# =============================================================================


async def analyze_painting(
    image_path: Path,
    ai_config: AiConfig,
    language: str = "zh",
) -> LayoutParams:
    """Send painting image to LLM for composition analysis.

    Invokes a vision-capable LLM with the painting image and a structured
    prompt requesting layout parameters for text overlay.

    Args:
        image_path: Path to the painting image file (PNG or JPEG).
        ai_config: Configuration specifying the LLM backend and model.
        language: Display language for art facts (e.g. "zh", "en", "ja").

    Returns:
        LayoutParams with all layout decisions needed for rendering.

    Raises:
        LayoutAnalysisError: If LLM invocation fails after retries.
        LayoutAnalysisError: If LLM response cannot be parsed.
        LayoutAnalysisError: If response is missing required fields.

    Contract:
        1. Calls llm.invoke_vision() with the painting image and prompt.
        2. Parses the JSON response using llm.parse_json_response().
        3. Validates all required fields are present.
        4. Constructs and returns a complete LayoutParams object.
        5. Results should be cached by caller using painting content_hash.

    Non-goals:
        - No image preprocessing (caller handles format conversion)
        - No caching (caller manages result caching)
        - No fallback strategies (failure is propagated to caller)
    """
    # Pre-analyze painting brightness per quadrant to help LLM make informed choices
    brightness_info = _analyze_brightness(image_path)

    # Inject display language into prompt template
    language_map = {"zh": "Chinese (中文)", "en": "English", "ja": "Japanese (日本語)"}
    lang_label = language_map.get(language, language)
    prompt = LAYOUT_ANALYSIS_PROMPT.replace("{{LANGUAGE}}", lang_label)
    enriched_prompt = prompt + "\n\n" + brightness_info

    # Invoke LLM with vision analysis + brightness data
    try:
        response = await invoke_vision(
            prompt=enriched_prompt,
            image_path=image_path,
            ai_config=ai_config,
        )
    except LlmError as e:
        raise LayoutAnalysisError(
            f"Layout analysis failed for '{image_path}': LLM invocation error: {e}"
        ) from e

    # Parse JSON response
    try:
        data = parse_json_response(response.text)
    except LlmParseError as e:
        raise LayoutAnalysisError(
            f"Layout analysis failed for '{image_path}': Failed to parse LLM response: {e}"
        ) from e

    # Build LayoutParams from parsed JSON
    try:
        return _build_layout_params(data)
    except (KeyError, TypeError, ValueError) as e:
        raise LayoutAnalysisError(
            f"Layout analysis failed for '{image_path}': Invalid response structure: {e}"
        ) from e


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _analyze_brightness(image_path: Path) -> str:
    """Analyze painting brightness per quadrant and return as text for the prompt."""
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("L")
        arr = np.array(img)
        h, w = arr.shape

        quadrants = {
            "top-left": arr[: h // 2, : w // 2],
            "top-right": arr[: h // 2, w // 2 :],
            "bottom-left": arr[h // 2 :, : w // 2],
            "bottom-right": arr[h // 2 :, w // 2 :],
        }

        lines = ["MEASURED BRIGHTNESS DATA (0=black, 255=white):"]
        for name, zone in quadrants.items():
            avg = int(np.mean(zone))
            var = int(np.var(zone))
            label = "BRIGHT" if avg > 150 else ("DARK" if avg < 100 else "MID-TONE")
            flatness = "flat/uniform" if var < 1000 else "complex/detailed"
            lines.append(
                f"  {name}: brightness={avg} ({label}), variance={var} ({flatness})"
            )

        # Recommend best zone
        best_zone = min(quadrants, key=lambda k: int(np.var(quadrants[k])))
        best_brightness = int(np.mean(quadrants[best_zone]))
        lines.append(f"\nRECOMMENDED text zone: {best_zone} (lowest visual complexity)")
        lines.append(
            f"Since {best_zone} is {'BRIGHT' if best_brightness > 150 else 'DARK'} "
            f"(brightness={best_brightness}), use {'DARK' if best_brightness > 150 else 'LIGHT'} text colors."
        )

        return "\n".join(lines)
    except Exception:
        return "BRIGHTNESS DATA: unavailable (analysis failed)"


def _build_layout_params(data: dict) -> LayoutParams:
    """Construct LayoutParams from parsed LLM response.

    This is an internal helper that validates and constructs all nested
    dataclass objects from the raw JSON dict.

    Args:
        data: Parsed JSON dict from LLM response.

    Returns:
        Fully constructed LayoutParams object.

    Raises:
        KeyError: If required fields are missing.
        ValueError: If field values are invalid.
    """
    # Build TextZone
    text_zone_data = data["text_zone"]
    text_zone = TextZone(
        position=text_zone_data["position"],
        reason=text_zone_data["reason"],
    )

    # Build SubjectZone
    subject_zone_data = data["subject_zone"]
    subject_zone = SubjectZone(
        position=subject_zone_data["position"],
        reason=subject_zone_data["reason"],
    )

    # Build WhisperZone
    whisper_zone_data = data["whisper_zone"]
    whisper_zone = WhisperZone(
        position=whisper_zone_data["position"],
        reason=whisper_zone_data["reason"],
        max_width_percent=whisper_zone_data["max_width_percent"],
        readability_notes=whisper_zone_data["readability_notes"],
    )

    if not 12 <= whisper_zone.max_width_percent <= 24:
        raise ValueError(
            "whisper_zone.max_width_percent must be between 12 and 24 inclusive"
        )
    if not whisper_zone.readability_notes.strip():
        raise ValueError("whisper_zone.readability_notes must be non-empty")

    if (
        len(
            {
                text_zone.position,
                subject_zone.position,
                whisper_zone.position,
            }
        )
        != 3
    ):
        raise ValueError(
            "text_zone, subject_zone, and whisper_zone must be mutually exclusive"
        )

    art_facts_data = data["art_facts"]
    if not isinstance(art_facts_data, list):
        raise TypeError("art_facts must be a list")
    if not 1 <= len(art_facts_data) <= 3:
        raise ValueError("art_facts must contain between 1 and 3 items")
    art_facts = [ArtFact(text=fact) for fact in art_facts_data]

    # Build LayoutColors
    colors_data = data["colors"]
    colors = LayoutColors(
        text_primary=colors_data["text_primary"],
        text_secondary=colors_data["text_secondary"],
        text_dim=colors_data["text_dim"],
        text_shadow=colors_data["text_shadow"],
        scrim_color=colors_data["scrim_color"],
        panel_bg=colors_data["panel_bg"],
        border=colors_data["border"],
        accent=colors_data["accent"],
    )

    # Build ScrimParams
    scrim_data = data["scrim"]
    scrim = ScrimParams(
        position_css=scrim_data["position_css"],
        size_css=scrim_data["size_css"],
        gradient_css=scrim_data["gradient_css"],
    )

    # Build PortraitLayout if present
    portrait_layout: PortraitLayout | None = None
    if data.get("portrait_layout") is not None:
        pl_data = data["portrait_layout"]
        portrait_layout = PortraitLayout(
            painting_width_percent=pl_data["painting_width_percent"],
            left_panel_color=pl_data["left_panel_color"],
            right_panel_color=pl_data["right_panel_color"],
            info_side=pl_data["info_side"],
        )

    # Build main LayoutParams
    return LayoutParams(
        orientation=data["orientation"],
        painting_title=data["painting_title"],
        painting_artist=data["painting_artist"],
        painting_description=data["painting_description"],
        text_zone=text_zone,
        subject_zone=subject_zone,
        whisper_zone=whisper_zone,
        art_facts=art_facts,
        colors=colors,
        scrim=scrim,
        recommended_stories=data["recommended_stories"],
        template_hint=data["template_hint"],
        portrait_layout=portrait_layout,
    )
