"""Contract tests for layout-analysis prompt and type surfaces.

Architecture reference: ARCHITECTURE.md#2.5
Prototype reference: PROTOTYPING.md#11
Contract reference: src/sfumato/layout_ai.py
"""

from __future__ import annotations

from dataclasses import fields
from typing import get_type_hints

from sfumato.layout_ai import (
    LAYOUT_ANALYSIS_PROMPT,
    LayoutAnalysisResponseJson,
    LayoutParams,
    WhisperZone,
)


class TestLayoutParamsContract:
    """Verify contract-only field surfaces for layout analysis."""

    def test_layout_params_adds_subject_whisper_and_art_facts(self) -> None:
        names = [field.name for field in fields(LayoutParams)]

        assert "subject_zone" in names
        assert "whisper_zone" in names
        assert "art_facts" in names

    def test_whisper_zone_carries_width_and_readability_contract(self) -> None:
        hints = get_type_hints(WhisperZone)

        assert hints["max_width_percent"] is int
        assert hints["readability_notes"] is str


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
