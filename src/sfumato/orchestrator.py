"""Pipeline composition contracts for orchestrator entry points.

This module pins the public contract for ``run_once`` before implementation
dispatch. It defines the data shape consumed by CLI callers, ordered stage
boundaries, flag semantics, and downgrade/error guarantees.

Spec references:
- ARCHITECTURE.md#2.12
- ARCHITECTURE.md#2.13
- README.md
"""

from __future__ import annotations

import datetime
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from sfumato.config import AppConfig
from sfumato.layout_ai import LayoutParams, analyze_painting
from sfumato.news import CurationResult, Story
from sfumato.palette import PaletteColors, extract_palette
from sfumato.render import Orientation, PaintingInfo, RenderContext, render_to_png

if TYPE_CHECKING:
    from sfumato.render import RenderResult

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC CONTRACT CONSTANTS
# =============================================================================


RUN_ONCE_STAGE_ORDER: Final[tuple[str, ...]] = (
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
"""Contracted stage order for ``run_once``.

Source: ARCHITECTURE.md#2.12 pipeline order (steps 1-10), expanded to include
the dequeue/refresh branch and TV branch wording from the step contract.
"""


RUN_ONCE_ERROR_SURFACE_STAGES: Final[frozenset[str]] = frozenset(
    {
        "news_dequeue_or_refresh",
        "layout_analysis",
        "palette_extraction",
        "render_4k_png",
    }
)
"""Stages that must propagate errors to the caller.

Source: step contract "Pin error propagation boundaries" and
ARCHITECTURE.md#2.12 (orchestrator composes these stages without silent
downgrades).
"""


RUN_ONCE_FLAG_SEMANTICS: Final[dict[str, str]] = {
    "no_news": (
        "Skip the dequeue/refresh branch and run pure-art selection/rendering. "
        "Downstream stages still run in the contracted order."
    ),
    "no_upload": (
        "Render locally but skip TV availability checks, upload, display "
        "switching, and TV cleanup side effects."
    ),
}
"""Flag semantics consumed by CLI ``run`` and ``preview``.

Source: step contract "Pin flag semantics" and ARCHITECTURE.md#2.13 CLI flags.
"""


RUN_ONCE_TV_DOWNGRADE_SEMANTICS: Final[str] = (
    "If TV push is unavailable, local render success remains successful: "
    "RunResult.uploaded=False and RunResult.render_result.png_path remains the "
    "generated local 4K PNG path."
)
"""Non-fatal TV-unavailable branch semantics.

Source: step contract "Pin the TV-unavailable branch as non-fatal".
"""


RUN_ONCE_OUTPUT_PATH_GUARANTEE: Final[str] = (
    "Whenever rendering succeeds, RunResult.render_result.png_path points to the "
    "generated local 4K PNG regardless of upload outcome."
)
"""Output-path guarantee for successful renders.

Source: step contract "Pin output-path guarantees" and ARCHITECTURE.md#2.12.
"""


# =============================================================================
# DATA TYPES
# =============================================================================


@dataclass
class RunOptions:
    """Options contract for a single pipeline execution.

    Source: ARCHITECTURE.md#2.12 ``RunOptions`` + step contract flag semantics.
    """

    no_upload: bool = False
    no_news: bool = False
    painting_path: Path | None = None
    preview: bool = False


@dataclass
class RunResult:
    """Result contract returned by ``run_once``.

    Source: ARCHITECTURE.md#2.12 ``RunResult`` and step contract guarantees.

    Contract notes:
    - ``uploaded`` is ``False`` for TV-unavailable downgrade success.
    - ``render_result`` remains non-``None`` on successful local render even when
      upload/display is skipped or unavailable.
    - ``render_result.png_path`` is the authoritative local output artifact when
      rendering succeeds.
    """

    render_result: RenderResult | None
    painting: PaintingInfo | None
    story_count: int
    uploaded: bool
    match_score: float | None
    action: str


# =============================================================================
# STATE PROTOCOL (seam for testing)
# =============================================================================


class QueuedBatch(Protocol):
    """Protocol for news queue batch."""

    stories: list[Story]
    tone_description: str
    enqueued_at: datetime.datetime


class NewsQueueProtocol(Protocol):
    """Protocol for news queue operations."""

    def dequeue(self) -> QueuedBatch | None:
        """Remove and return the next batch."""
        ...

    @property
    def size(self) -> int:
        """Number of batches currently in queue."""
        ...


class LayoutCacheProtocol(Protocol):
    """Protocol for layout cache operations."""

    def get(self, content_hash: str) -> LayoutParams | None:
        """Get cached layout params for a painting."""
        ...

    def put(self, content_hash: str, layout: LayoutParams) -> None:
        """Cache layout params for a painting."""
        ...

    def has(self, content_hash: str) -> bool:
        """Check if layout is cached."""
        ...


class UsedPaintingsProtocol(Protocol):
    """Protocol for used paintings tracking."""

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting as used."""
        ...

    def is_used(self, content_hash: str) -> bool:
        """Check if a painting has been used."""
        ...


class AppStateProtocol(Protocol):
    """Protocol for application state."""

    news_queue: NewsQueueProtocol
    used_paintings: UsedPaintingsProtocol
    layout_cache: LayoutCacheProtocol

    def save_all(self) -> None:
        """Persist all state components."""
        ...


# =============================================================================
# PUBLIC API
# =============================================================================


async def run_once(
    config: AppConfig,
    state: AppStateProtocol,
    options: RunOptions,
) -> RunResult:
    """Execute one orchestrated rotation cycle.

    Ordered stage boundary (must not be reordered by flag branches):
    1. news dequeue/refresh branch
    2. painting selection
    3. layout analysis
    4. palette extraction
    5. template selection
    6. 4K render
    7. optional TV upload/display
    8. mark used
    9. optional preview
    10. state save

    Error propagation boundary:
    - news/layout/palette/render failures surface to caller
    - only TV availability/upload branch may degrade to local-render success

    Args:
        config: Application configuration.
        state: Application state (news queue, used paintings, caches).
        options: Run options (no_upload, no_news, painting_path, preview).

    Returns:
        RunResult with render_result, painting, story_count, uploaded, match_score.

    Raises:
        ValueError: If painting_path is specified but file doesn't exist.
        OSError: If required files cannot be read.
        Exception: Propagated from news/layout/palette/render stages.
    """
    # Stage 1: news_dequeue_or_refresh
    # CONTRACT: Skip if no_news, propagate errors
    batch: QueuedBatch | None = None
    if not options.no_news:
        batch = state.news_queue.dequeue()
        # TODO: If batch is None, trigger on-demand refresh (run_news_refresh)
        # For now, we'll continue with no stories if queue is empty
        # This follows the Phase 1 contract where pure-art mode is valid

    # Stage 2: painting_selection
    # CONTRACT: Use painting_path if specified, random selection for Phase 1
    painting = await _select_painting(
        config=config,
        state=state,
        painting_path=options.painting_path,
        batch=batch,
    )

    # Stage 3: layout_analysis
    # CONTRACT: Check cache first, analyze if not cached, propagate errors
    layout = await _analyze_layout(
        painting=painting,
        state=state,
        config=config,
    )

    # Stage 4: palette_extraction
    # CONTRACT: Always extract (no caching in palette module), propagate errors
    palette = extract_palette(painting.image_path)

    # Stage 5: template_selection
    # CONTRACT: Use layout.template_hint with orientation fallback
    template_name = _select_template(layout, painting)

    # Stage 6: render_4k_png
    # CONTRACT: Render to 4K PNG, propagate errors
    stories = batch.stories if batch else []
    render_result = await _render_4k(
        painting=painting,
        stories=stories,
        layout=layout,
        palette=palette,
        template_name=template_name,
        config=config,
    )

    # Stage 7: tv_upload_and_display_optional
    # CONTRACT: Skip if no_upload, degrade gracefully if TV unavailable
    uploaded = False
    if not options.no_upload:
        uploaded = await _try_tv_upload(
            config=config,
            png_path=render_result.png_path,
        )

    # Stage 8: mark_painting_used
    # CONTRACT: Mark after successful render
    state.used_paintings.mark_used(painting.content_hash)

    # Stage 9: preview_optional
    # CONTRACT: Open in system viewer if preview=True
    if options.preview:
        _open_preview(render_result.png_path)

    # Stage 10: state_save
    # CONTRACT: Persist state changes
    state.save_all()

    # Build result
    story_count = len(stories)
    action = "pure_art" if options.no_news else "news_rotation"

    return RunResult(
        render_result=render_result,
        painting=painting,
        story_count=story_count,
        uploaded=uploaded,
        match_score=None,  # Phase 1: no semantic matching
        action=action,
    )


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


async def _select_painting(
    config: AppConfig,
    state: AppStateProtocol,
    painting_path: Path | None,
    batch: QueuedBatch | None,
) -> PaintingInfo:
    """Select a painting for this rotation.

    Selection order:
    1. If painting_path is specified, use it directly
    2. If batch is None (no_news), use random unused painting from pool
    3. Otherwise, use random selection (Phase 1: semantic matching deferred)

    Args:
        config: Application configuration.
        state: Application state.
        painting_path: Optional path to specific painting.
        batch: Current news batch (None if no_news).

    Returns:
        PaintingInfo for the selected painting.

    Raises:
        ValueError: If painting_path specified but file doesn't exist.
        RuntimeError: If no painting is available.
    """
    # If specific painting is requested, create PaintingInfo from file
    if painting_path is not None:
        if not painting_path.exists():
            raise ValueError(f"Painting file not found: {painting_path}")

        return _create_painting_info_from_path(painting_path)

    # Phase 1: Random selection
    # TODO: In intelligence phase, use semantic matching with batch.tone_description
    # For now, we need a pool of paintings. Since paintings module doesn't exist yet,
    # we'll raise NotImplementedError for the pool case.
    # This should be implemented by the caller providing a pool in state or config.

    # For now, we'll use a placeholder that works with the test doubles
    # In practice, the CLI would need to initialize a painting pool
    raise NotImplementedError(
        "Painting pool selection not implemented. "
        "Provide a painting_path in RunOptions, or implement painting pool management."
    )


def _create_painting_info_from_path(painting_path: Path) -> PaintingInfo:
    """Create PaintingInfo from an image file path.

    Args:
        painting_path: Path to the painting image file.

    Returns:
        PaintingInfo with detected orientation and generated hash.

    Raises:
        OSError: If file cannot be read.
    """
    from PIL import Image

    # Load image to get dimensions
    img = Image.open(painting_path)
    width, height = img.size

    # Detect orientation
    orientation = Orientation.LANDSCAPE if width >= height else Orientation.PORTRAIT

    # Compute content hash
    import hashlib

    content = painting_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()

    return PaintingInfo(
        image_path=painting_path.resolve(),
        content_hash=content_hash,
        title="Unknown",
        artist="Unknown",
        year="Unknown",
        orientation=orientation,
        width=width,
        height=height,
        source="local",
        source_id=content_hash[:12],
        source_url="",
    )


async def _analyze_layout(
    painting: PaintingInfo,
    state: AppStateProtocol,
    config: AppConfig,
) -> LayoutParams:
    """Analyze painting layout, using cache if available.

    Args:
        painting: Painting to analyze.
        state: Application state (with layout_cache).
        config: Application configuration.

    Returns:
        LayoutParams for the painting.

    Raises:
        Exception: Propagated from layout_ai.analyze_painting.
    """
    # Check cache first
    content_hash = painting.content_hash
    if state.layout_cache.has(content_hash):
        cached = state.layout_cache.get(content_hash)
        if cached is not None:
            return cached

    # Analyze with LLM
    layout = await analyze_painting(painting.image_path, config.ai)

    # Cache result
    state.layout_cache.put(content_hash, layout)

    return layout


def _select_template(layout: LayoutParams, painting: PaintingInfo) -> str:
    """Select template based on layout and painting orientation.

    Template selection order:
    1. Use layout.template_hint if valid
    2. Fallback based on orientation: portrait -> "portrait", landscape -> "painting_text"

    Args:
        layout: Layout parameters from analysis.
        painting: Painting info with orientation.

    Returns:
        Template name to use for rendering.
    """
    valid_templates = {"painting_text", "magazine", "portrait", "art_overlay"}

    # Use hint if valid
    if layout.template_hint in valid_templates:
        return layout.template_hint

    # Fallback based on orientation
    if painting.orientation == Orientation.PORTRAIT:
        return "portrait"

    return "painting_text"


async def _render_4k(
    painting: PaintingInfo,
    stories: list[Story],
    layout: LayoutParams,
    palette: PaletteColors,
    template_name: str,
    config: AppConfig,
) -> "RenderResult":
    """Render the 4K PNG.

    Args:
        painting: Painting info.
        stories: News stories to render.
        layout: Layout parameters.
        palette: Extracted color palette.
        template_name: Template to use.
        config: Application configuration.

    Returns:
        RenderResult with png_path and metadata.

    Raises:
        Exception: Propagated from render_to_png.
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%A, %B %d, %Y")
    time_str = now.strftime("%H:%M")

    context = RenderContext(
        painting=painting,
        stories=stories,
        layout=layout,
        palette=palette,
        template_name=template_name,
        language=config.news.language,
        date_str=date_str,
        time_str=time_str,
    )

    # Use default output directory
    output_dir = config.data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return await render_to_png(context, output_dir)


async def _try_tv_upload(
    config: AppConfig,
    png_path: Path,
) -> bool:
    """Attempt to upload to TV, returning success status.

    This is the TV-unavailable branch that degrades gracefully.
    If TV is unavailable or upload fails, return False without raising.

    Args:
        config: Application configuration with TV settings.
        png_path: Path to the rendered PNG.

    Returns:
        True if upload succeeded, False if TV unavailable or upload failed.
    """
    # Import here to avoid dependency issues when TV module isn't needed
    from sfumato.tv import (
        TvConnectionError,
        TvError,
        TvUploadError,
        is_available_for_push,
        set_displayed,
        upload_image,
    )

    # Check if TV is available
    if not config.tv.ip:
        # No TV configured
        return False

    try:
        # Check availability (non-throwing)
        if not is_available_for_push(config.tv):
            logger.info("TV not available for push (unreachable or not in Art Mode)")
            return False

        # Upload image
        content_id = upload_image(config.tv, png_path)

        # Set displayed
        set_displayed(config.tv, content_id)

        return True

    except TvConnectionError as e:
        logger.warning(f"TV connection failed: {e}")
        return False
    except TvUploadError as e:
        logger.warning(f"TV upload failed: {e}")
        return False
    except TvError as e:
        logger.warning(f"TV operation failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected TV error: {e}")
        return False


def _open_preview(png_path: Path) -> None:
    """Open PNG in system viewer for preview.

    Args:
        png_path: Path to the PNG file to open.
    """
    import platform
    import subprocess

    system = platform.system()

    try:
        if system == "Darwin":
            subprocess.run(["open", str(png_path)], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", str(png_path)], check=False)
        elif system == "Windows":
            subprocess.run(["start", str(png_path)], check=False, shell=True)
        else:
            logger.warning(f"Unknown platform {system}, cannot open preview")
    except Exception as e:
        logger.warning(f"Failed to open preview: {e}")
