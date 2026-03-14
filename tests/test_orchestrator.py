"""Contract tests for orchestrator.run_once.

Spec sources:
- ARCHITECTURE.md#2.12 (orchestrator module contract)
- ARCHITECTURE.md#2.13 (CLI flags and dispatch)
- README.md (end-to-end behavior)
- tests/test_orchestrator.py (existing contract shape tests)

These tests verify the contracted behavior of run_once before
implementation dispatch. They use seam-based test doubles to
islate orchestrator logic from external dependencies.

Required Coverage:
- Success path: full pipeline runs in contracted order, returns RunResult
  with local 4K png_path, and marks uploaded=True only after successful TV push
- Flag path --no-upload: local render completes, uploaded=False, no TV calls
- Flag path --no-news: news dequeue/refresh skipped, pure-art selection, valid PNG
- TV-unavailable branch: availability false yields uploaded=False, png_path preserved
- Error propagation: refresh/dequeue/layout/palette/render failures surface
- Output-path guarantee: successful render always provides png_path
"""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sfumato.config import (
    AppConfig,
    AiConfig,
    FeedConfig,
    NewsConfig,
    PaintingsConfig,
    ScheduleConfig,
    TvConfig,
)
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

if TYPE_CHECKING:
    from sfumato.layout_ai import LayoutParams
    from sfumato.news import CurationResult, Story
    from sfumato.palette import PaletteColors
    from sfumato.render import Orientation, PaintingInfo, RenderResult
    from sfumato.state import AppState, QueuedBatch

# =============================================================================
# EXISTING CONTRACT SHAPE TESTS (from original file)
# =============================================================================


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
async def test_run_once_no_news_with_painting_path_succeeds(tmp_path: Path) -> None:
    """run_once with painting_path and no_news succeeds.

    This is the basic success path for pure art mode with a specific painting.

    Verification:
    - Returns RunResult with render_result.png_path existing
    - story_count is 0 (no_news)
    - uploaded is False (no_upload)
    - match_score is None (Phase 1)
    - action is "pure_art"
    - painting is marked used in state
    - layout is cached
    """
    # Create a test painting file
    from PIL import Image
    from unittest.mock import AsyncMock, patch, MagicMock

    painting_path = tmp_path / "test_painting.jpg"
    img = Image.new("RGB", (3840, 2160), color="#1a237e")
    img.save(painting_path)

    # Create mock state
    mock_state = MockAppState()

    # Create config with data_dir pointing to tmp
    from sfumato.config import (
        AppConfig,
        AiConfig,
        NewsConfig,
        PaintingsConfig,
        ScheduleConfig,
        TvConfig,
    )

    config = AppConfig(
        tv=TvConfig(ip="192.168.1.100", port=8002, max_uploads=5),
        schedule=ScheduleConfig(),
        news=NewsConfig(language="en"),
        paintings=PaintingsConfig(cache_dir=tmp_path / "paintings"),
        ai=AiConfig(cli="gemini", model="test-model"),
        data_dir=tmp_path,
    )

    # Create mock layout params
    mock_layout = create_mock_layout_params()

    # Mock the external dependencies
    with patch(
        "sfumato.orchestrator.analyze_painting", new_callable=AsyncMock
    ) as mock_analyze:
        mock_analyze.return_value = mock_layout

        result = await run_once(
            config=config,
            state=mock_state,  # type: ignore[arg-type]
            options=RunOptions(
                no_news=True, no_upload=True, painting_path=painting_path
            ),
        )

    # Verify result
    assert result.render_result is not None
    assert result.render_result.png_path.exists()
    assert result.painting is not None
    assert result.story_count == 0  # no_news means no stories
    assert result.uploaded is False  # no_upload means not uploaded
    assert result.match_score is None  # Phase 1 has no semantic matching
    assert result.action == "pure_art"

    # Verify painting was marked used
    assert mock_state.used_paintings.is_used(result.painting.content_hash)

    # Verify layout was cached
    assert mock_state.layout_cache.has(result.painting.content_hash)


# =============================================================================
# SEAM DOUBLES: Test helpers for dependency injection
# =============================================================================


class MockNewsQueue:
    """Seam double for state.NewsQueue."""

    def __init__(self) -> None:
        self.dequeue_calls: int = 0
        self.enqueue_calls: int = 0
        self._next_batch: "QueuedBatch | None" = None

    def set_next_batch(self, batch: "QueuedBatch | None") -> None:
        """Configure the next batch to return from dequeue."""
        self._next_batch = batch

    def dequeue(self) -> "QueuedBatch | None":
        """Simulate dequeue behavior for testing."""
        self.dequeue_calls += 1
        return self._next_batch


class MockQueuedBatch:
    """Seam double for state.QueuedBatch."""

    def __init__(
        self,
        stories: list["Story"] | None = None,
        tone_description: str = "A thoughtful mood",
    ) -> None:
        from sfumato.news import Story

        self.stories = stories or [
            Story(
                headline="Test Story",
                summary="Test summary content",
                source="Test Source",
                category="Tech",
                url="https://example.com/test",
                published_at=datetime.now(),
                featured=True,
            )
        ]
        self.tone_description = tone_description
        self.enqueued_at = datetime.now()


class MockLayoutCache:
    """Seam double for state.LayoutCache."""

    def __init__(self) -> None:
        self._cache: dict[str, "LayoutParams"] = {}

    def get(self, content_hash: str) -> "LayoutParams | None":
        return self._cache.get(content_hash)

    def put(self, content_hash: str, layout: "LayoutParams") -> None:
        self._cache[content_hash] = layout

    def has(self, content_hash: str) -> bool:
        return content_hash in self._cache


class MockEmbeddingCache:
    """Seam double for state.EmbeddingCache."""

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def get(self, key: str) -> list[float] | None:
        return self._cache.get(key)

    def put(self, key: str, vector: list[float]) -> None:
        self._cache[key] = vector


class MockUsedPaintings:
    """Seam double for state.UsedPaintings."""

    def __init__(self) -> None:
        self._used: set[str] = set()

    def mark_used(self, content_hash: str) -> None:
        self._used.add(content_hash)

    def is_used(self, content_hash: str) -> bool:
        return content_hash in self._used

    def reset(self) -> None:
        self._used.clear()

    @property
    def count(self) -> int:
        return len(self._used)


class MockAppState:
    """Seam double for state.AppState."""

    def __init__(self) -> None:
        self.news_queue = MockNewsQueue()
        self.used_paintings = MockUsedPaintings()
        self.layout_cache = MockLayoutCache()
        self.embedding_cache = MockEmbeddingCache()

    def save_all(self) -> None:
        """Simulate state persistence."""
        pass


def create_minimal_app_config() -> AppConfig:
    """Create a minimal valid AppConfig for testing."""
    return AppConfig(
        tv=TvConfig(ip="192.168.1.100", port=8002, max_uploads=5),
        schedule=ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            quiet_hours=(0, 6),
            active_hours=(7, 23),
        ),
        news=NewsConfig(
            language="en",
            stories_per_refresh=12,
            max_age_days=3,
            expire_days=7,
            feeds=[],
        ),
        paintings=PaintingsConfig(
            cache_dir=Path("/tmp/sfumato-test/paintings"),
            seed_size=50,
            pool_size=200,
            sources=["test"],
            match_strategy="random",
        ),
        ai=AiConfig(cli="test", model="test-model"),
        data_dir=Path("/tmp/sfumato-test"),
    )


def create_mock_painting_info(tmp_path: Path) -> "PaintingInfo":
    """Create a mock PaintingInfo for testing."""
    from sfumato.render import Orientation, PaintingInfo

    image_path = tmp_path / "test_painting.jpg"
    image_path.touch()

    return PaintingInfo(
        image_path=image_path,
        content_hash="test_hash_abc123",
        title="Test Painting",
        artist="Test Artist",
        year="2024",
        orientation=Orientation.LANDSCAPE,
        width=3840,
        height=2160,
        source="test",
        source_id="test_001",
        source_url="https://example.com/test_painting",
    )


def create_mock_layout_params() -> "LayoutParams":
    """Create a mock LayoutParams for testing."""
    from sfumato.layout_ai import LayoutColors, LayoutParams, ScrimParams, TextZone

    return LayoutParams(
        orientation="landscape",
        painting_title="Test Painting",
        painting_artist="Test Artist",
        painting_description="A test painting description",
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


def create_mock_palette_colors() -> "PaletteColors":
    """Create a mock PaletteColors for testing."""
    from sfumato.palette import PaletteColors

    return PaletteColors(
        dominant="#1a237e",
        secondary="#283593",
        accent="#ffd700",
        background="#0d1b2a",
        is_dark=True,
        colors=("#1a237e", "#283593", "#ffd700", "#0d1b2a", "#424242"),
    )


def create_mock_render_result(tmp_path: Path) -> "RenderResult":
    """Create a mock RenderResult for testing."""
    from sfumato.render import RenderResult

    png_path = tmp_path / "test_output.png"
    png_path.touch()
    html_path = tmp_path / "test_output.html"
    html_path.touch()

    return RenderResult(
        png_path=png_path.resolve(),
        html_path=html_path.resolve(),
        template_used="painting_text",
        story_count=3,
        painting_hash="test_hash_abc123",
    )


# =============================================================================
# SUCCESS PATH TESTS
# =============================================================================


class TestRunOnceSuccessPath:
    """Tests for the main success path through run_once.

    Contract:
    - Full pipeline runs in contracted order (RUN_ONCE_STAGE_ORDER)
    - Returns RunResult with local 4K png_path
    - Marks uploaded=True only after successful TV push
    - Output path is preserved regardless of TV operations
    """

    @pytest.mark.asyncio
    async def test_success_path_full_pipeline_order(self, tmp_path: Path) -> None:
        """Pipeline stages execute in the contracted order.

        When run_once completes successfully, each stage should have been
        called exactly once in the order defined by RUN_ONCE_STAGE_ORDER.
        """
        # This test will pass once implementation is dispatched.
        # For now, it documents the contract.
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_success_path_returns_png_path(self, tmp_path: Path) -> None:
        """Successful render produces local 4K PNG path in RunResult.

        Contract: RunResult.render_result.png_path points to a valid
        local file after successful render, regardless of TV operations.
        """
        # This test will pass once implementation is dispatched.
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_success_path_uploaded_true_after_tv_push(
        self, tmp_path: Path
    ) -> None:
        """uploaded=True only after successful TV upload and display.

        Contract: RunResult.uploaded is False until TV upload+display
        completes successfully, then becomes True.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_success_path_stage_sequence_no_skip(self, tmp_path: Path) -> None:
        """No stages are skipped in the success path.

        Contract: All stages in RUN_ONCE_STAGE_ORDER execute,
        including optional TV upload when TV is available.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# FLAG PATH: --no-upload
# =============================================================================


class TestRunOnceNoUploadFlag:
    """Tests for the --no-upload flag path.

    Contract (RUN_ONCE_FLAG_SEMANTICS['no_upload']):
    - Render locally, skip TV availability checks
    - Skip TV upload, display switching, cleanup side effects
    - uploaded=False
    - png_path preserved from local render
    """

    @pytest.mark.asyncio
    async def test_no_upload_skips_tv_availability_check(self, tmp_path: Path) -> None:
        """--no-upload must not call TV availability check.

        The TV module's check_status and is_available_for_push must NOT
        be called when no_upload=True.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_skips_tv_upload_call(self, tmp_path: Path) -> None:
        """--no-upload must not call TV upload_image.

        The TV upload_image function must NOT be called when no_upload=True.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_skips_tv_display_call(self, tmp_path: Path) -> None:
        """--no-upload must not call TV set_displayed.

        The TV set_displayed function must NOT be called when no_upload=True.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_marks_uploaded_false(self, tmp_path: Path) -> None:
        """--no-upload must set RunResult.uploaded=False.

        Even if rendering succeeds, uploaded must be False because
        no TV operations were attempted.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_preserves_png_path(self, tmp_path: Path) -> None:
        """--no-upload preserves local PNG path in RunResult.

        Contract (RUN_ONCE_OUTPUT_PATH_GUARANTEE): The png_path must
        be set correctly even when TV operations are skipped.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_render_still_produces_4k_output(
        self, tmp_path: Path
    ) -> None:
        """--no-upload does not affect local render quality.

        The render stage must still produce a 4K PNG as if TV upload
        were enabled.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# FLAG PATH: --no-news
# =============================================================================


class TestRunOnceNoNewsFlag:
    """Tests for the --no-news flag path (pure art mode).

    Contract (RUN_ONCE_FLAG_SEMANTICS['no_news']):
    - Skip the dequeue/refresh branch entirely
    - Use pure-art painting selection (random or without news tone)
    - Render still produces valid local PNG
    - story_count = 0 in result
    """

    @pytest.mark.asyncio
    async def test_no_news_skips_dequeue(self, tmp_path: Path) -> None:
        """--no-news must not call news_queue.dequeue().

        The news queue dequeue operation must be skipped entirely.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_skips_news_refresh(self, tmp_path: Path) -> None:
        """--no-news must not trigger on-demand news refresh.

        Even if the queue is empty, no refresh should be triggered.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_uses_pure_art_selection(self, tmp_path: Path) -> None:
        """--no-news uses random or pool-based selection without news tone.

        The painting selection must NOT use semantic matching with news
        tone when no_news=True.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_render_produces_valid_png(self, tmp_path: Path) -> None:
        """--no-news render must produce valid local PNG.

        The render pipeline must still complete successfully.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_story_count_zero(self, tmp_path: Path) -> None:
        """--no-news must set RunResult.story_count=0.

        No news stories are rendered in pure art mode.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_match_score_is_none(self, tmp_path: Path) -> None:
        """--no-news must set RunResult.match_score=None.

        No semantic matching occurs when news is skipped.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# TV-UNAVAILABLE BRANCH
# =============================================================================


class TestRunOnceTvUnavailable:
    """Tests for TV-unavailable downgrade path.

    Contract (RUN_ONCE_TV_DOWNGRADE_SEMANTICS):
    - TV availability check returns False or TV refuses upload
    - Local render success remains
    - RunResult.uploaded=False
    - RunResult.render_result.png_path preserved
    - Non-fatal: no exceptions raised
    """

    @pytest.mark.asyncio
    async def test_tv_unavailable_availability_false(self, tmp_path: Path) -> None:
        """TV unavailable (check_status.reachable=False) yields uploaded=False.

        When TV is unreachable, the pipeline should complete successfully
        with uploaded=False and png_path preserved.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_unavailable_art_mode_inactive(self, tmp_path: Path) -> None:
        """TV not in Art Mode yields uploaded=False.

        When TV is reachable but art_mode_active=False,
        the pipeline should still succeed locally.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_upload_refusal_non_fatal(self, tmp_path: Path) -> None:
        """TV upload refusal (TvUploadError) yields uploaded=False.

        When TV accepts connection but refuses upload, the pipeline
        should complete with uploaded=False, not crash.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_connection_error_non_fatal(self, tmp_path: Path) -> None:
        """TV TvConnectionError yields uploaded=False.

        When TV connection fails, the pipeline should complete
        with uploaded=False rather than propagate the exception.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_unavailable_png_path_preserved(self, tmp_path: Path) -> None:
        """TV-unavailable branch must preserve png_path.

        Contract (RUN_ONCE_OUTPUT_PATH_GUARANTEE): Even when TV
        operations fail, the local PNG path must be valid.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_unavailable_no_display_call(self, tmp_path: Path) -> None:
        """TV set_displayed must not be called after upload failure.

        If upload fails, the display switch must also be skipped.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# ERROR PROPAGATION BOUNDARIES
# =============================================================================


class TestRunOnceErrorPropagation:
    """Tests for error propagation boundaries.

    Contract (RUN_ONCE_ERROR_SURFACE_STAGES):
    - news_dequeue_or_refresh failure must surface (not swallowed)
    - layout_analysis failure must surface
    - palette_extraction failure must surface
    - render_4k_png failure must surface
    - No partial success result fabricated
    """

    @pytest.mark.asyncio
    async def test_news_dequeue_failure_propagates(self, tmp_path: Path) -> None:
        """News dequeue failure must propagate to caller.

        If news_queue.dequeue() raises an exception, run_once must
        propagate it, not fabricate a partial RunResult.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_news_refresh_failure_propagates(self, tmp_path: Path) -> None:
        """News refresh failure must propagate to caller.

        If on-demand news refresh fails, run_once must propagate
        the exception, not continue with empty queue.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_layout_analysis_failure_propagates(self, tmp_path: Path) -> None:
        """Layout analysis failure must propagate to caller.

        If layout_ai.analyze_painting() raises LayoutAnalysisError,
        run_once must propagate it.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_palette_extraction_failure_propagates(self, tmp_path: Path) -> None:
        """Palette extraction failure must propagate to caller.

        If palette.extract_palette() raises PaletteError,
        run_once must propagate it.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_render_failure_propagates(self, tmp_path: Path) -> None:
        """Render failure must propagate to caller.

        If render.render_to_png() raises RenderError,
        run_once must propagate it.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_partial_result_on_core_stage_failure(
        self, tmp_path: Path
    ) -> None:
        """Core stage failure must not return partial RunResult.

        If any stage in RUN_ONCE_ERROR_SURFACE_STAGES fails,
        the function must raise an exception, not return a
        partially populated RunResult.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_error_does_not_mask_core_success(self, tmp_path: Path) -> None:
        """TV errors must not override core stage results.

        If all core stages succeed but TV fails, the png_path
        must still be available in the result.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# OUTPUT-PATH GUARANTEE
# =============================================================================


class TestRunOnceOutputPathGuarantee:
    """Tests for output-path guarantee.

    Contract (RUN_ONCE_OUTPUT_PATH_GUARANTEE):
    - Every successful render provides valid png_path
    - Path is absolute and points to existing file
    - Preserved regardless of TV/upload/preview operations
    """

    @pytest.mark.asyncio
    async def test_success_path_png_path_exists(self, tmp_path: Path) -> None:
        """Successful render produces existing PNG file.

        After successful render, render_result.png_path must point
        to an existing file.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_png_path_exists(self, tmp_path: Path) -> None:
        """--no-upload still produces existing PNG file.

        The png_path must exist even when TV operations are skipped.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_tv_failure_png_path_exists(self, tmp_path: Path) -> None:
        """TV failure preserves existing PNG file.

        Even when TV operations fail, the local PNG must exist.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_png_path_is_absolute(self, tmp_path: Path) -> None:
        """png_path must be an absolute path.

        All paths returned in RunResult must be absolute.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_render_failure_no_png_path(self, tmp_path: Path) -> None:
        """Render failure returns render_result=None.

        When render fails, RunResult.render_result should be None
        rather than a path to a non-existent file.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# STAGE ORDERING CONTRACT
# =============================================================================


class TestRunOnceStageOrdering:
    """Tests for pipeline stage ordering contract.

    Contract (RUN_ONCE_STAGE_ORDER):
    - Stages execute in exact order specified
    - Flag branches do not reorder stages
    - Early exit (error) stops pipeline
    """

    @pytest.mark.asyncio
    async def test_stage_order_matches_constants(self, tmp_path: Path) -> None:
        """Actual stage order must match RUN_ONCE_STAGE_ORDER constants."""
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_upload_preserves_core_stage_order(self, tmp_path: Path) -> None:
        """--no-upload does not change core stage order.

        The news/layout/palette/render stages must still execute
        in order even when TV stages are skipped.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_no_news_preserves_remaining_stage_order(
        self, tmp_path: Path
    ) -> None:
        """--no-news does not change remaining stage order.

        Painting selection through render must still execute in order.
        """
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_early_error_stops_pipeline(self, tmp_path: Path) -> None:
        """Early stage failure stops subsequent stages.

        If layout_analysis fails, palette/render must not be called.
        """
        pytest.skip("Implementation dispatch pending")


# =============================================================================
# INTEGRATION CONTRACT TESTS
# =============================================================================


class TestRunOnceIntegrationContracts:
    """Tests for integration contracts between stages.

    These tests verify the handoff between stages matches
    the contracted data shapes.
    """

    @pytest.mark.asyncio
    async def test_news_to_painting_integration(self, tmp_path: Path) -> None:
        """News batch tone is passed to painting selection correctly."""
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_painting_to_layout_integration(self, tmp_path: Path) -> None:
        """Painting image path is passed to layout analysis correctly."""
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_layout_to_render_integration(self, tmp_path: Path) -> None:
        """Layout params are passed to render with all required fields."""
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_render_to_tv_integration(self, tmp_path: Path) -> None:
        """Render PNG path is passed to TV upload correctly."""
        pytest.skip("Implementation dispatch pending")

    @pytest.mark.asyncio
    async def test_state_persistence_after_success(self, tmp_path: Path) -> None:
        """State is saved after successful pipeline completion.

        The used_paintings mark and any caches must be persisted.
        """
        pytest.skip("Implementation dispatch pending")
