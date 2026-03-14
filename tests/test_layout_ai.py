"""Contract tests for layout-analysis prompt and type surfaces.

Architecture reference: ARCHITECTURE.md#2.5
Prototype reference: PROTOTYPING.md#11
Contract reference: src/sfumato/layout_ai.py
"""

from __future__ import annotations

from dataclasses import fields
from typing import get_type_hints

import pytest

from sfumato.layout_ai import (
    ArtFact,
    LAYOUT_ANALYSIS_PROMPT,
    LayoutAnalysisResponseJson,
    LayoutParams,
    SubjectZone,
    WhisperZone,
    _build_layout_params,
)


class TestLayoutParamsContract:
    """Verify contract-only field surfaces for layout analysis."""

    def test_layout_params_adds_subject_whisper_and_art_facts(self) -> None:
        names = [field.name for field in fields(LayoutParams)]

        assert "subject_zone" in names
        assert "whisper_zone" in names
        assert "art_facts" in names

    def test_subject_zone_carries_position_and_reason(self) -> None:
        """SubjectZone must have position (ZonePosition) and reason (str)."""
        hints = get_type_hints(SubjectZone)

        assert "position" in hints
        assert "reason" in hints

    def test_whisper_zone_carries_width_and_readability_contract(self) -> None:
        hints = get_type_hints(WhisperZone)

        assert hints["max_width_percent"] is int
        assert hints["readability_notes"] is str

    def test_art_fact_carries_text(self) -> None:
        """ArtFact must carry presentation-ready whisper text."""
        hints = get_type_hints(ArtFact)

        assert "text" in hints


class TestLlmResponseContract:
    """Verify the structured JSON contract presented to the LLM."""

    def test_llm_contract_exposes_required_top_level_fields(self) -> None:
        hints = get_type_hints(LayoutAnalysisResponseJson)

        assert hints["subject_zone"]
        assert hints["whisper_zone"]
        assert hints["art_facts"]
        assert hints["painting_description"]
        assert hints["template_hint"]
        assert hints["portrait_layout"]

    def test_prompt_enumerates_all_ten_requested_items(self) -> None:
        for index in range(1, 11):
            assert f"{index}. **" in LAYOUT_ANALYSIS_PROMPT

    def test_llm_contract_response_fields_match_ten_prompt_items(self) -> None:
        """Regressions: ensure response JSON fields correspond to 10 prompt items."""
        hints = get_type_hints(LayoutAnalysisResponseJson)

        # Prompt items 1-10 map to response fields:
        # 1. Orientation -> orientation
        # 2. Painting Identity -> painting_title, painting_artist
        # 3. News Zone -> text_zone
        # 4. Subject Zone -> subject_zone
        # 5. Whisper Zone -> whisper_zone
        # 6. Art Facts -> art_facts
        # 7. Color Harmony -> colors
        # 8. Scrim Design -> scrim
        # 9. Story Count -> recommended_stories
        # 10. Composition Notes -> painting_description, template_hint, portrait_layout
        required_fields = [
            "orientation",  # 1
            "painting_title",  # 2
            "painting_artist",  # 2
            "text_zone",  # 3
            "subject_zone",  # 4
            "whisper_zone",  # 5
            "art_facts",  # 6
            "colors",  # 7
            "scrim",  # 8
            "recommended_stories",  # 9
            "painting_description",  # 10
            "template_hint",  # 10
            "portrait_layout",  # 10
        ]
        for field_name in required_fields:
            assert field_name in hints, f"Missing field: {field_name}"

    def test_prompt_requires_mutual_exclusion_and_whisper_constraints(self) -> None:
        assert (
            "text_zone, subject_zone, and whisper_zone MUST be mutually exclusive"
            in LAYOUT_ANALYSIS_PROMPT
        )
        assert (
            "Whisper zone width MUST stay between 12% and 24% of screen width"
            in LAYOUT_ANALYSIS_PROMPT
        )
        assert (
            "Whisper text must remain readable at TV distance" in LAYOUT_ANALYSIS_PROMPT
        )


class TestZoneExclusivityContract:
    """Overlap-validation regressions for news/subject/whisper zone exclusivity."""

    def test_prompt_declares_three_exclusive_zones(self) -> None:
        """Prompt must enumerate text_zone, subject_zone, whisper_zone as exclusive."""
        # Prompt defines these three zones as mutually exclusive
        assert "text_zone" in LAYOUT_ANALYSIS_PROMPT
        assert "subject_zone" in LAYOUT_ANALYSIS_PROMPT
        assert "whisper_zone" in LAYOUT_ANALYSIS_PROMPT
        assert (
            "text_zone, subject_zone, and whisper_zone MUST be mutually exclusive"
            in LAYOUT_ANALYSIS_PROMPT
        )

    def test_response_zone_types_define_position_and_reason(self) -> None:
        """All zone response types must have position (ZonePosition) and reason."""
        from sfumato.layout_ai import TextZoneJson, SubjectZoneJson, WhisperZoneJson

        text_hints = get_type_hints(TextZoneJson)
        subject_hints = get_type_hints(SubjectZoneJson)
        whisper_hints = get_type_hints(WhisperZoneJson)

        # All zones share position and reason
        assert "position" in text_hints
        assert "reason" in text_hints
        assert "position" in subject_hints
        assert "reason" in subject_hints
        assert "position" in whisper_hints
        assert "reason" in whisper_hints

    def test_whisper_zone_has_width_constraint_fields(self) -> None:
        """Whisper zone must include max_width_percent and readability_notes."""
        from sfumato.layout_ai import WhisperZoneJson

        hints = get_type_hints(WhisperZoneJson)

        assert "max_width_percent" in hints
        assert "readability_notes" in hints

    def test_prompt_constrains_whisper_width_range(self) -> None:
        """Width constraint 12%-24% must be stated in prompt."""
        assert "12%" in LAYOUT_ANALYSIS_PROMPT
        assert "24%" in LAYOUT_ANALYSIS_PROMPT

    def test_response_json_subject_zone_type_exists(self) -> None:
        """LayoutAnalysisResponseJson must include subject_zone as SubjectZoneJson."""
        from sfumato.layout_ai import SubjectZoneJson

        hints = get_type_hints(LayoutAnalysisResponseJson)
        assert hints["subject_zone"] == SubjectZoneJson

    def test_response_json_whisper_zone_type_exists(self) -> None:
        """LayoutAnalysisResponseJson must include whisper_zone as WhisperZoneJson."""
        from sfumato.layout_ai import WhisperZoneJson

        hints = get_type_hints(LayoutAnalysisResponseJson)
        assert hints["whisper_zone"] == WhisperZoneJson

    def test_response_json_art_facts_is_list_of_strings(self) -> None:
        """art_facts must be list[str] for whisper rendering."""
        hints = get_type_hints(LayoutAnalysisResponseJson)

        assert hints["art_facts"] == list[str]


def _base_layout_data() -> dict:
    return {
        "orientation": "landscape",
        "painting_title": "Test Title",
        "painting_artist": "Test Artist",
        "painting_description": "Rich composition with layered brushwork.",
        "text_zone": {
            "position": "top-left",
            "reason": "Low detail and quiet tonal field.",
        },
        "subject_zone": {
            "position": "bottom-right",
            "reason": "Primary subject cluster occupies this quadrant.",
        },
        "whisper_zone": {
            "position": "top-right",
            "reason": "Secondary quiet zone for whisper facts.",
            "max_width_percent": 18,
            "readability_notes": "Strong local contrast keeps small text readable.",
        },
        "art_facts": ["Fact A", "Fact B"],
        "colors": {
            "text_primary": "#EAEAEA",
            "text_secondary": "#DADADA",
            "text_dim": "rgba(220,220,220,0.7)",
            "text_shadow": "0 1px 3px rgba(0,0,0,0.7)",
            "scrim_color": "rgba(0,0,0,0.3)",
            "panel_bg": "#202020",
            "border": "#404040",
            "accent": "#C0A050",
        },
        "scrim": {
            "position_css": "top: 120px; right: 160px;",
            "size_css": "width: 38%; height: 45%;",
            "gradient_css": "radial-gradient(ellipse at center, rgba(0,0,0,0.4) 0%, transparent 70%)",
        },
        "recommended_stories": 3,
        "template_hint": "painting_text",
        "portrait_layout": None,
    }


class TestBuildLayoutParamsValidation:
    def test_build_layout_params_populates_subject_whisper_and_art_facts(self) -> None:
        params = _build_layout_params(_base_layout_data())

        assert params.subject_zone.position == "bottom-right"
        assert params.whisper_zone.max_width_percent == 18
        assert params.art_facts == [ArtFact(text="Fact A"), ArtFact(text="Fact B")]

    def test_build_layout_params_rejects_overlapping_zones(self) -> None:
        data = _base_layout_data()
        data["whisper_zone"]["position"] = "top-left"

        with pytest.raises(ValueError, match="mutually exclusive"):
            _build_layout_params(data)

    def test_build_layout_params_rejects_invalid_whisper_width(self) -> None:
        data = _base_layout_data()
        data["whisper_zone"]["max_width_percent"] = 30

        with pytest.raises(ValueError, match="between 12 and 24"):
            _build_layout_params(data)

    def test_build_layout_params_rejects_empty_whisper_readability_notes(self) -> None:
        data = _base_layout_data()
        data["whisper_zone"]["readability_notes"] = "   "

        with pytest.raises(ValueError, match="readability_notes"):
            _build_layout_params(data)
