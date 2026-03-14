"""Tests for orchestrator batching behavior and recommended_stories enforcement.

Bug #7: stories_per_refresh=12 must produce multiple batches (not one giant batch)
Bug #8: recommended_stories from layout must limit stories rendered

Spec sources:
- ARCHITECTURE.md#5.2 (Dynamic batch sizing)
- ARCHITECTURE.md A.4 (Batch sizing is determined by selected painting's layout)
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from sfumato.config import (
    AppConfig,
    AiConfig,
    NewsConfig,
    PaintingsConfig,
    ScheduleConfig,
    TvConfig,
)
from sfumato.orchestrator import (
    RunOptions,
    run_news_refresh,
    run_once,
)
from sfumato.news import Story, CurationResult

# Import from sibling test file
sys.path.insert(0, str(Path(__file__).parent))

from test_orchestrator import (  # noqa: E402
    MockAppState,
    MockQueuedBatch,
    create_minimal_app_config,
    create_mock_layout_params,
    create_mock_palette_colors,
    create_mock_render_result,
)


# =============================================================================
# BUG #7: stories_per_refresh batching behavior
# =============================================================================


class TestStoriesPerRefreshBatching:
    """Tests for correct batching when stories_per_refresh is large.

    Contract:
    - stories_per_refresh (~12) stories are curated per refresh
    - They are split into batches of recommended_stories (3-4)
    - Multiple batches should be created when stories_per_refresh > batch_size
    - The batch_size is min(4, stories_per_refresh) per current implementation

    Bug #7: When stories_per_refresh=12, should create multiple batches,
    not one giant batch of 12 stories.
    """

    @pytest.mark.asyncio
    async def test_stories_per_refresh_12_creates_multiple_batches(self) -> None:
        """stories_per_refresh=12 creates 3 batches (not 1 giant batch).

        Default batch_size = min(4, 12) = 4
        12 stories / 4 per batch = 3 batches
        """
        mock_state = MockAppState()
        config = create_minimal_app_config()

        # Create 12 stories (matching stories_per_refresh default)
        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(12)
        ]

        mock_result = CurationResult(
            stories=stories,
            tone_description="test tone",
            curated_at=datetime.now(),
            feed_count=3,
            entry_count=15,
        )

        with patch(
            "sfumato.orchestrator.refresh_news", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = mock_result

            batches_enqueued = await run_news_refresh(config, mock_state)

        # 12 stories / batch_size 4 = 3 batches
        assert batches_enqueued == 3
        # Verify queue contains 3 batches
        assert mock_state.news_queue.size == 3

    @pytest.mark.asyncio
    async def test_stories_per_refresh_15_creates_four_batches(self) -> None:
        """stories_per_refresh=15 creates 4 batches (not 1 giant batch).

        Default batch_size = min(4, 15) = 4
        15 stories / 4 per batch = 4 batches (3 full + 1 partial)
        """
        mock_state = MockAppState()
        config = create_minimal_app_config()

        # Create 15 stories
        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(15)
        ]

        mock_result = CurationResult(
            stories=stories,
            tone_description="test tone",
            curated_at=datetime.now(),
            feed_count=3,
            entry_count=15,
        )

        with patch(
            "sfumato.orchestrator.refresh_news", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = mock_result

            batches_enqueued = await run_news_refresh(config, mock_state)

        # 15 stories / batch_size 4 = 4 batches (12 + 3)
        assert batches_enqueued == 4

    @pytest.mark.asyncio
    async def test_small_stories_count_creates_one_batch(self) -> None:
        """stories_count < batch_size creates 1 batch.

        3 stories / batch_size 4 = 1 batch
        """
        mock_state = MockAppState()
        config = create_minimal_app_config()

        # Create only 3 stories
        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(3)
        ]

        mock_result = CurationResult(
            stories=stories,
            tone_description="test tone",
            curated_at=datetime.now(),
            feed_count=1,
            entry_count=3,
        )

        with patch(
            "sfumato.orchestrator.refresh_news", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = mock_result

            batches_enqueued = await run_news_refresh(config, mock_state)

        # 3 stories / batch_size 4 = 1 batch (partial)
        assert batches_enqueued == 1

    @pytest.mark.asyncio
    async def test_batch_size_never_exceeds_stories_per_refresh(self) -> None:
        """Batch size is capped at stories_per_refresh even if internal logic tries larger."""
        mock_state = MockAppState()

        # Create config with low stories_per_refresh
        config = AppConfig(
            tv=TvConfig(ip="192.168.1.100"),
            schedule=ScheduleConfig(),
            news=NewsConfig(
                language="en",
                stories_per_refresh=2,  # Low limit
                max_age_days=3,
                expire_days=7,
                feeds=[],
            ),
            paintings=PaintingsConfig(
                cache_dir=Path("/tmp/sfumato-test/paintings"),
            ),
            ai=AiConfig(cli="test", model="test-model"),
            data_dir=Path("/tmp/sfumato-test"),
        )

        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(5)  # More than stories_per_refresh
        ]

        mock_result = CurationResult(
            stories=stories,
            tone_description="test tone",
            curated_at=datetime.now(),
            feed_count=1,
            entry_count=5,
        )

        with patch(
            "sfumato.orchestrator.refresh_news", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = mock_result

            batches_enqueued = await run_news_refresh(config, mock_state)

        # batch_size = min(4, 2) = 2
        # 5 stories / 2 per batch = 3 batches (2, 2, 1)
        assert batches_enqueued == 3


# =============================================================================
# BUG #8: recommended_stories limits rendered stories
# =============================================================================


class TestRecommendedStoriesLimitsRenderedStories:
    """Tests for recommended_stories limiting stories passed to render.

    Contract (ARCHITECTURE.md A.4):
    - recommended_stories from layout determines how many stories to render
    - A painting with large quiet zone can fit 4-5 stories
    - A painting with limited space should show only 2-3
    - The LLM (layout analysis) recommends this based on composition

    Bug #8: recommended_stories must be used to slice stories before rendering.
    """

    @pytest.mark.asyncio
    async def test_recommended_stories_limits_rendered_stories(
        self, tmp_path: Path
    ) -> None:
        """recommended_stories from layout limits stories passed to render.

        When batch has 5 stories but layout.recommended_stories=3,
        only 3 stories should be rendered.
        """
        from PIL import Image

        painting_path = tmp_path / "test_painting.jpg"
        img = Image.new("RGB", (3840, 2160), color="#1a237e")
        img.save(painting_path)

        mock_state = MockAppState()
        config = create_minimal_app_config()
        config = AppConfig(
            tv=config.tv,
            schedule=config.schedule,
            news=config.news,
            paintings=config.paintings,
            ai=config.ai,
            data_dir=tmp_path,
        )

        # Layout with recommended_stories=3
        mock_layout = create_mock_layout_params()
        mock_layout = MagicMock()
        mock_layout.recommended_stories = 3
        mock_layout.template_hint = "painting_text"
        mock_layout.text_zone = MagicMock()
        mock_layout.text_zone.position = "top-right"
        mock_layout.colors = create_mock_palette_colors()
        mock_layout.scrim = MagicMock()
        mock_layout.scrim.position_css = "top: 120px; right: 160px;"
        mock_layout.scrim.size_css = "width: 1500px; height: 1400px;"
        mock_layout.scrim.gradient_css = (
            "radial-gradient(ellipse at top right, rgba(0,0,0,0.4) 0%, transparent 70%)"
        )
        mock_layout.portrait_layout = None

        # Create a batch with 5 stories
        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(5)
        ]
        mock_batch = MockQueuedBatch(stories=stories, tone_description="test")
        mock_state.news_queue.set_next_batch(mock_batch)

        render_call_args = []

        async def capture_render(*args, **kwargs):
            render_call_args.append((args, kwargs))
            return create_mock_render_result(tmp_path)

        with (
            patch(
                "sfumato.orchestrator.analyze_painting", new_callable=AsyncMock
            ) as mock_analyze,
            patch("sfumato.orchestrator.extract_palette") as mock_palette,
            patch(
                "sfumato.orchestrator.render_to_png", new_callable=AsyncMock
            ) as mock_render,
            patch(
                "sfumato.orchestrator._try_tv_upload", new_callable=AsyncMock
            ) as mock_tv_upload,
        ):
            mock_analyze.return_value = mock_layout
            mock_palette.return_value = create_mock_palette_colors()
            mock_render.side_effect = capture_render
            mock_tv_upload.return_value = False

            result = await run_once(
                config=config,
                state=mock_state,
                options=RunOptions(no_upload=True, painting_path=painting_path),
            )

        # Verify render was called with limited stories
        assert len(render_call_args) == 1
        call_args = render_call_args[0]
        render_context = call_args[0][0]  # First positional arg is RenderContext

        # CRITICAL: Only recommended_stories (3) should be passed, not all 5
        assert len(render_context.stories) == 3
        assert result.story_count == 3

    @pytest.mark.asyncio
    async def test_recommended_stories_two_limits_to_two(self, tmp_path: Path) -> None:
        """recommended_stories=2 limits to 2 stories even when batch has 5."""
        from PIL import Image

        painting_path = tmp_path / "test_painting.jpg"
        img = Image.new("RGB", (3840, 2160), color="#1a237e")
        img.save(painting_path)

        mock_state = MockAppState()
        config = create_minimal_app_config()
        config = AppConfig(
            tv=config.tv,
            schedule=config.schedule,
            news=config.news,
            paintings=config.paintings,
            ai=config.ai,
            data_dir=tmp_path,
        )

        # Layout with recommended_stories=2 (frozen dataclass, use replace)
        mock_layout = replace(create_mock_layout_params(), recommended_stories=2)

        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(5)
        ]
        mock_batch = MockQueuedBatch(stories=stories, tone_description="test")
        mock_state.news_queue.set_next_batch(mock_batch)

        render_call_args = []

        async def capture_render(*args, **kwargs):
            render_call_args.append((args, kwargs))
            return create_mock_render_result(tmp_path)

        with (
            patch(
                "sfumato.orchestrator.analyze_painting", new_callable=AsyncMock
            ) as mock_analyze,
            patch("sfumato.orchestrator.extract_palette") as mock_palette,
            patch(
                "sfumato.orchestrator.render_to_png", new_callable=AsyncMock
            ) as mock_render,
            patch(
                "sfumato.orchestrator._try_tv_upload", new_callable=AsyncMock
            ) as mock_tv_upload,
        ):
            mock_analyze.return_value = mock_layout
            mock_palette.return_value = create_mock_palette_colors()
            mock_render.side_effect = capture_render
            mock_tv_upload.return_value = False

            result = await run_once(
                config=config,
                state=mock_state,
                options=RunOptions(no_upload=True, painting_path=painting_path),
            )

        # Only 2 stories should be rendered
        assert len(render_call_args) == 1
        call_args = render_call_args[0]
        render_context = call_args[0][0]

        assert len(render_context.stories) == 2
        assert result.story_count == 2

    @pytest.mark.asyncio
    async def test_recommended_stories_all_when_batch_smaller(
        self, tmp_path: Path
    ) -> None:
        """When batch has fewer stories than recommended_stories, render all batch stories."""
        from PIL import Image

        painting_path = tmp_path / "test_painting.jpg"
        img = Image.new("RGB", (3840, 2160), color="#1a237e")
        img.save(painting_path)

        mock_state = MockAppState()
        config = create_minimal_app_config()
        config = AppConfig(
            tv=config.tv,
            schedule=config.schedule,
            news=config.news,
            paintings=config.paintings,
            ai=config.ai,
            data_dir=tmp_path,
        )

        # Layout recommends 5 stories (frozen dataclass, use replace)
        mock_layout = replace(create_mock_layout_params(), recommended_stories=5)

        # But batch only has 2 stories
        stories = [
            Story(
                headline=f"Story {i}",
                summary=f"Summary {i}",
                source="Test",
                category="Tech",
                url=f"https://example.com/{i}",
                published_at=datetime.now(),
                featured=(i == 0),
            )
            for i in range(2)
        ]
        mock_batch = MockQueuedBatch(stories=stories, tone_description="test")
        mock_state.news_queue.set_next_batch(mock_batch)

        render_call_args = []

        async def capture_render(*args, **kwargs):
            render_call_args.append((args, kwargs))
            return create_mock_render_result(tmp_path)

        with (
            patch(
                "sfumato.orchestrator.analyze_painting", new_callable=AsyncMock
            ) as mock_analyze,
            patch("sfumato.orchestrator.extract_palette") as mock_palette,
            patch(
                "sfumato.orchestrator.render_to_png", new_callable=AsyncMock
            ) as mock_render,
            patch(
                "sfumato.orchestrator._try_tv_upload", new_callable=AsyncMock
            ) as mock_tv_upload,
        ):
            mock_analyze.return_value = mock_layout
            mock_palette.return_value = create_mock_palette_colors()
            mock_render.side_effect = capture_render
            mock_tv_upload.return_value = False

            result = await run_once(
                config=config,
                state=mock_state,
                options=RunOptions(no_upload=True, painting_path=painting_path),
            )

        # All 2 stories (batch size) should be rendered, not clipped to recommended_stories
        assert len(render_call_args) == 1
        call_args = render_call_args[0]
        render_context = call_args[0][0]

        assert len(render_context.stories) == 2
        assert result.story_count == 2

    @pytest.mark.asyncio
    async def test_no_news_uses_empty_stories_not_recommended(
        self, tmp_path: Path
    ) -> None:
        """When no_news=True, story list is empty regardless of recommended_stories."""
        from PIL import Image

        painting_path = tmp_path / "test_painting.jpg"
        img = Image.new("RGB", (3840, 2160), color="#1a237e")
        img.save(painting_path)

        mock_state = MockAppState()
        config = create_minimal_app_config()
        config = AppConfig(
            tv=config.tv,
            schedule=config.schedule,
            news=config.news,
            paintings=config.paintings,
            ai=config.ai,
            data_dir=tmp_path,
        )

        # Layout recommends 4 stories (but won't be used in no_news mode)
        # Frozen dataclass, use replace
        mock_layout = replace(create_mock_layout_params(), recommended_stories=4)

        render_call_args = []

        async def capture_render(*args, **kwargs):
            render_call_args.append((args, kwargs))
            return create_mock_render_result(tmp_path)

        with (
            patch(
                "sfumato.orchestrator.analyze_painting", new_callable=AsyncMock
            ) as mock_analyze,
            patch("sfumato.orchestrator.extract_palette") as mock_palette,
            patch(
                "sfumato.orchestrator.render_to_png", new_callable=AsyncMock
            ) as mock_render,
        ):
            mock_analyze.return_value = mock_layout
            mock_palette.return_value = create_mock_palette_colors()
            mock_render.side_effect = capture_render

            result = await run_once(
                config=config,
                state=mock_state,
                options=RunOptions(
                    no_news=True, no_upload=True, painting_path=painting_path
                ),
            )

        # No stories in no_news mode
        assert len(render_call_args) == 1
        call_args = render_call_args[0]
        render_context = call_args[0][0]

        assert len(render_context.stories) == 0
        assert result.story_count == 0
