"""Contract tests for orchestrator.run_once.

Spec sources:
- ARCHITECTURE.md#2.12
- ARCHITECTURE.md#2.13
- README.md
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from sfumato.config import AppConfig
from sfumato.orchestrator import (
    RUN_ONCE_ERROR_SURFACE_STAGES,
    RUN_ONCE_FLAG_SEMANTICS,
    RUN_ONCE_OUTPUT_PATH_GUARANTEE,
    RUN_ONCE_STAGE_ORDER,
    RUN_ONCE_TV_DOWNGRADE_SEMANTICS,
    RunOptions,
    RunResult,
    run_once,
)


def test_run_options_public_fields_and_defaults() -> None:
    """RunOptions shape is pinned for CLI callers."""
    assert [field.name for field in fields(RunOptions)] == [
        "no_upload",
        "no_news",
        "painting_path",
        "preview",
    ]

    options = RunOptions()
    assert options.no_upload is False
    assert options.no_news is False
    assert options.painting_path is None
    assert options.preview is False


def test_run_result_public_fields_are_pinned() -> None:
    """RunResult shape is pinned for CLI result consumption."""
    assert [field.name for field in fields(RunResult)] == [
        "render_result",
        "painting",
        "story_count",
        "uploaded",
        "match_score",
        "action",
    ]


def test_run_once_stage_order_is_contracted_and_stable() -> None:
    """run_once stage order matches the architecture and step contract."""
    assert RUN_ONCE_STAGE_ORDER == (
        "news_dequeue_or_refresh",
        "painting_selection",
        "layout_analysis",
        "palette_extraction",
        "template_selection",
        "render_4k_png",
        "tv_upload_and_display_optional",
        "mark_painting_used",
        "preview_optional",
        "state_save",
    )


def test_run_once_flag_semantics_are_explicit() -> None:
    """--no-news and --no-upload behavior is pinned by contract text."""
    assert sorted(RUN_ONCE_FLAG_SEMANTICS.keys()) == ["no_news", "no_upload"]

    no_news = RUN_ONCE_FLAG_SEMANTICS["no_news"]
    assert "Skip the dequeue/refresh" in no_news
    assert "pure-art" in no_news

    no_upload = RUN_ONCE_FLAG_SEMANTICS["no_upload"]
    assert "Render locally" in no_upload
    assert "skip TV availability checks" in no_upload
    assert "cleanup side effects" in no_upload


def test_tv_downgrade_and_output_path_guarantees_are_pinned() -> None:
    """TV-unavailable branch degrades to local success with png path preserved."""
    assert "uploaded=False" in RUN_ONCE_TV_DOWNGRADE_SEMANTICS
    assert "render_result.png_path" in RUN_ONCE_TV_DOWNGRADE_SEMANTICS
    assert "local 4K PNG" in RUN_ONCE_OUTPUT_PATH_GUARANTEE


def test_error_propagation_boundary_is_explicit() -> None:
    """Only TV branch may degrade; core stage errors must surface."""
    assert RUN_ONCE_ERROR_SURFACE_STAGES == {
        "news_dequeue_or_refresh",
        "layout_analysis",
        "palette_extraction",
        "render_4k_png",
    }


@pytest.mark.asyncio
async def test_run_once_is_contract_stub_until_implementation_dispatch() -> None:
    """run_once intentionally raises until implementation step."""
    with pytest.raises(NotImplementedError) as exc_info:
        await run_once(
            config=AppConfig(),
            state=object(),  # type: ignore[arg-type]
            options=RunOptions(painting_path=Path("painting.jpg")),
        )

    assert "contract" in str(exc_info.value)
