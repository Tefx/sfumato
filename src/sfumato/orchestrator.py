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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sfumato.config import AppConfig

if TYPE_CHECKING:
    from sfumato.render import PaintingInfo, RenderResult
    from sfumato.state import AppState


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


async def run_once(
    config: AppConfig,
    state: AppState,
    options: RunOptions,
) -> RunResult:
    """Execute one orchestrated rotation cycle (contract-only stub).

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

    Raises:
        NotImplementedError: This contract step intentionally does not implement
            pipeline logic.
    """
    raise NotImplementedError(
        "run_once contract is defined; implementation is dispatched in a later step"
    )
