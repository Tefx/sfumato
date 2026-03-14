"""Pipeline composition contracts for orchestrator entry points.

This module pins the public contract for ``run_once`` and daemon ``watch``
before implementation dispatch. It defines the data shape consumed by CLI
callers, ordered stage boundaries, scheduler-action mapping, shutdown semantics,
state-save guarantees, and downgrade/error boundaries.

Spec references:
- ARCHITECTURE.md#2.12
- ARCHITECTURE.md#2.13
- README.md
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import numpy as np

from sfumato.config import AppConfig
from sfumato.layout_ai import LayoutParams, analyze_painting
from sfumato.matcher import MatcherError, compute_embedding, select_painting
from sfumato.news import CurationResult, Story, refresh_news
from sfumato.palette import PaletteColors, extract_palette
from sfumato.paintings import (
    ArtSource,
    Orientation as PaintingsOrientation,
    PaintingInfo as PaintingsPaintingInfo,
    fetch_paintings,
    list_cached_paintings,
)
from sfumato.render import (
    Orientation,
    PaintingInfo,
    RenderContext,
    WhisperFactIndex,
    render_to_png,
)
from sfumato.state import AppState

if TYPE_CHECKING:
    from sfumato.render import RenderResult

logger = logging.getLogger(__name__)

WATCH_HEALTH_FILE: Final[str] = "last_action.json"


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


RUN_ONCE_ART_FACT_FLOW: Final[tuple[str, ...]] = (
    "layout_analysis yields LayoutParams with whisper_zone and art_facts from cache or fresh analysis.",
    "layout_cache is the persisted source for art_facts keyed by painting.content_hash; orchestrator must not synthesize replacement facts.",
    "orchestrator resolves RenderContext.whisper_fact_index from caller-owned rotation state after layout selection and before render.",
    "RenderContext receives the selected LayoutParams unchanged, including layout.art_facts ordering.",
    "If layout.art_facts is empty, RenderContext.whisper_fact_index must be None.",
)
"""Art-fact data flow contract for ``run_once``.

Source: task contract for art-fact wiring across layout cache, orchestrator, and
render context.
"""


ART_FACT_INDEX_STATE_OWNERSHIP: Final[dict[str, str]] = {
    "owner": "Orchestrator-owned rotation state; render consumes the chosen index but never persists or mutates it.",
    "key": "Indexes are tracked per painting.content_hash.",
    "default": "Missing state resolves to whisper_fact_index=0 when layout.art_facts is non-empty.",
    "disabled": "Empty art_facts resolves to whisper_fact_index=None and does not require a stored cursor.",
    "advance": "Successful rotate advances the next index by one modulo the current art_fact_count.",
    "rebase": "If cached/fresh layout changes the art_fact_count, any stored cursor is rebased modulo the current count before use.",
    "failure_boundary": "Pre-render failures and render failures do not commit cursor advancement.",
}
"""Ownership and rotation semantics for whisper art-fact indexes.

Source: task contract for orchestrator-owned art-fact rotation state.
"""


ROTATE_ART_FACT_ACCEPTANCE_CRITERIA: Final[tuple[str, ...]] = (
    "First successful rotate for a painting with art_facts passes whisper_fact_index=0 into RenderContext.",
    "Subsequent successful rotates for the same painting.content_hash advance modulo len(layout.art_facts).",
    "A cached layout and a freshly analyzed layout are equivalent inputs if they expose the same ordered art_facts list.",
    "When layout.art_facts is empty, rotate passes whisper_fact_index=None and emits no art-fact selection.",
    "A failed rotate attempt must not consume the next art-fact index.",
)
"""Acceptance criteria for rotate-mode art-fact behavior.

Source: task contract for rotate behavior and render-context propagation.
"""


RUN_ONCE_QUEUE_SOURCE_PRECEDENCE: Final[tuple[str, ...]] = (
    "primary_news_queue",
    "replay_queue",
)
"""Ordered batch-source precedence for ``run_once`` news input.

Primary queue content must be consumed before replay is considered. Replay is a
fallback source, not a peer with equal priority.
"""


RUN_ONCE_PRIMARY_BATCH_REPLAY_TRANSFER_GUARANTEE: Final[str] = (
    "When run_once consumes a primary NewsQueue batch, it must attempt "
    "ReplayQueue.transfer_from_news_queue(batch) before downstream painting "
    "selection so consumed primary batches become replay-eligible."
)
"""Replay-transfer guarantee for consumed primary batches.

This pins the ownership boundary: primary queue remains the source of fresh
content while replay queue retains already-consumed primary batches for later
fallback use.
"""


RUN_ONCE_REPLAY_FALLBACK_GUARANTEE: Final[str] = (
    "ReplayQueue.next() may be consulted only after the primary NewsQueue stays "
    "empty following the contracted dequeue/refresh attempt."
)
"""Fallback rule for replay consumption inside ``run_once``.

This preserves primary-first semantics while allowing rotations to continue when
fresh primary batches are unavailable.
"""


RUN_NEWS_REFRESH_REPLAY_EXPIRY_GUARANTEE: Final[str] = (
    "Each run_news_refresh cycle must trigger ReplayQueue.expire("
    "config.news.replay_expire_days) during refresh before replay-aware enqueue "
    "decisions are finalized."
)
"""Replay expiry trigger pinned to refresh cycles.

This keeps replay backlog freshness aligned to ``config.news.replay_expire_days``
instead of piggybacking on the primary queue expiry horizon.
"""


RUN_NEWS_REFRESH_REPLAY_CURATION_SKIP_OVERLAP_RATIO: Final[float] = 0.8
"""Strict overlap threshold for skipping replay-duplicate curation enqueue."""


RUN_NEWS_REFRESH_REPLAY_CURATION_SKIP_GUARANTEE: Final[str] = (
    "If curated refresh output overlaps the replay backlog by more than 80%, the "
    "refresh cycle must skip primary curation enqueue for that payload to avoid "
    "reintroducing near-duplicate replay content."
)
"""Replay-aware dedup contract for refresh cycles.

This is intentionally stricter than replay-transfer acceptance thresholds because
the refresh path decides whether to introduce another primary payload at all.
"""


WATCH_LOOP_STAGE_ORDER: Final[tuple[str, ...]] = (
    "load_state_once",
    "scheduler_decision",
    "action_dispatch",
    "state_save",
    "sleep_until_next_action",
)
"""Contracted daemon watch loop stage order.

Source:
- ARCHITECTURE.md#2.12 ``watch`` loop contract (load -> loop -> save -> sleep)
- ARCHITECTURE.md lines 1494-1520 pseudocode ordering.
"""


WATCH_SCHEDULER_ACTION_MAPPING: Final[dict[str, str]] = {
    "REFRESH_NEWS": "run_news_refresh(config, state)",
    "ROTATE": "run_once(config, state, RunOptions())",
    "BACKFILL": "run_backfill(config, state)",
    "QUIET_ART": "run_once(config, state, RunOptions(no_news=True))",
    "IDLE": "no-op (only state save + sleep)",
}
"""Scheduler action to orchestrator operation mapping.

Source:
- ARCHITECTURE.md#2.11 ``Action`` definitions.
- ARCHITECTURE.md lines 1502-1516 watch-loop pseudocode dispatch.
"""


WATCH_ACTION_DISPATCH_ORDER: Final[tuple[str, ...]] = (
    "REFRESH_NEWS",
    "ROTATE",
    "BACKFILL",
    "QUIET_ART",
)
"""Deterministic dispatch order for combined scheduler actions.

Source: ARCHITECTURE.md lines 1502-1516 pseudocode branch ordering.
"""


WATCH_SHUTDOWN_SIGNALS: Final[frozenset[str]] = frozenset({"SIGINT", "SIGTERM"})
"""Signals that trigger graceful daemon shutdown.

Source:
- ARCHITECTURE.md#2.12 (watch handles SIGINT/SIGTERM)
- ARCHITECTURE.md lines 1946-1949 container process management.
"""


WATCH_SHUTDOWN_STATE_SAVE_GUARANTEE: Final[str] = (
    "On SIGINT/SIGTERM, finish the current in-flight action boundary, persist "
    "state, and then exit without starting another scheduler cycle."
)
"""Graceful shutdown guarantee for daemon mode.

Source:
- ARCHITECTURE.md lines 1947-1948 (finish action, save state, exit)
- ARCHITECTURE.md line 1472 (state saved after every action).
"""


WATCH_STATE_SAVE_GUARANTEES: Final[dict[str, str]] = {
    "after_action_cycle": "Call state.save_all() after each action cycle and before sleep.",
    "startup_load": "Load state exactly once on daemon startup before entering the loop.",
    "signal_exit": WATCH_SHUTDOWN_STATE_SAVE_GUARANTEE,
}
"""State persistence guarantees for daemon watch mode.

Source:
- ARCHITECTURE.md#2.12 watch contract (load state then loop)
- ARCHITECTURE.md lines 1472-1474 and 1518-1520.
"""


WATCH_ERROR_PROPAGATION_BOUNDARIES: Final[dict[str, str]] = {
    "run_news_refresh": "Recoverable failures end the current refresh action; retry is deferred to the next scheduler interval.",
    "run_once": "Core rotation failures end the current rotation action; daemon loop continues on next scheduled tick.",
    "run_backfill": "Backfill failures end only the current backfill action; later cycles may retry.",
    "watch_loop": "Fatal bootstrap/persistence failures propagate and terminate the daemon.",
}
"""Watch-loop error propagation and retry boundary contract.

Source:
- ARCHITECTURE.md lines 1962-1964 (refresh/LLM retry then skip to next interval)
- ARCHITECTURE.md lines 1969-1971 (rotation/upload failures skip current action)
- ARCHITECTURE.md lines 1965-1966 (painting source failures degrade to cached/backfill retry)
- ARCHITECTURE.md line 1974 (disk full is fatal).
"""


RUN_BACKFILL_STAGE_ORDER: Final[tuple[str, ...]] = (
    "measure_pool_deficit",
    "fetch_new_paintings_if_needed",
    "analyze_layout_for_new_paintings",
    "compute_embeddings_for_new_paintings",
    "state_save",
)
"""Contracted stage order for ``run_backfill``.

Source:
- ARCHITECTURE.md#2.12 ``run_backfill`` contract.
- README.md lines 38 and 171 (daemon backfill toward pool_size target).
"""


RUN_BACKFILL_BOUNDED_BEHAVIOR: Final[tuple[str, ...]] = (
    "Never add more than max(0, config.paintings.pool_size - current_pool_count) paintings in one call.",
    "Return value is bounded to [0, requested_deficit] and counts only successfully added paintings.",
    "If current_pool_count >= config.paintings.pool_size, perform no fetch/analyze work and return 0.",
)
"""Bounded-behavior contract for ``run_backfill``.

Source:
- ARCHITECTURE.md#2.12 (expand pool toward pool_size)
- README.md line 171 (pool_size is the background backfill target).
"""


RUN_BACKFILL_ERROR_BOUNDARIES: Final[dict[str, str]] = {
    "item_level": "Individual painting download/analyze failures are non-fatal and skipped.",
    "source_level": "If all external painting sources fail, fallback is cached pool only and return 0 additions.",
    "fatal": "State persistence failures propagate to caller.",
}
"""Error propagation and retry boundaries for ``run_backfill``.

Source:
- ARCHITECTURE.md lines 1965-1966 (painting failures are warning/degradation)
- ARCHITECTURE.md line 1974 (disk full is fatal).
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

    def enqueue(self, result: CurationResult, batch_size: int) -> int:
        """Split curation result into batches and append to queue.

        Returns:
            Number of batches enqueued.
        """
        ...

    def expire(self, expire_days: int) -> int:
        """Drop batches older than ``expire_days`` and return removed count."""
        ...

    def peek(self) -> QueuedBatch | None:
        """Return next batch without removal."""
        ...

    @property
    def size(self) -> int:
        """Number of batches currently in queue."""
        ...

    def save(self) -> None:
        """Persist queue state."""
        ...

    def load(self) -> None:
        """Load queue state."""
        ...


class ReplayTransferResultProtocol(Protocol):
    """Protocol for replay-transfer outcome data."""

    accepted: bool
    reason: str
    overlap_ratio: float
    matched_batch_index: int | None


class ReplayBatchProtocol(Protocol):
    """Protocol for replay queue batch payloads."""

    stories: list[Story]
    tone_description: str
    source_enqueued_at: datetime.datetime
    transferred_at: datetime.datetime
    replay_count: int
    last_replayed_at: datetime.datetime | None


class ReplayQueueProtocol(Protocol):
    """Protocol for replay queue operations used by orchestrator wiring."""

    def next(self) -> ReplayBatchProtocol | None:
        """Return the next replay batch without destructive removal."""
        ...

    def expire(self, expire_days: int) -> int:
        """Drop replay batches older than ``expire_days`` and return removed count."""
        ...

    def transfer_from_news_queue(
        self,
        batch: QueuedBatch,
    ) -> ReplayTransferResultProtocol:
        """Attempt to transfer a consumed primary batch into replay storage."""
        ...

    @property
    def size(self) -> int:
        """Number of replay batches currently tracked."""
        ...

    def persist(self) -> None:
        """Persist replay queue state."""
        ...

    def load(self) -> None:
        """Load replay queue state."""
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

    @property
    def size(self) -> int:
        """Number of cached layouts."""
        ...

    def save(self) -> None:
        """Persist layout cache state."""
        ...

    def load(self) -> None:
        """Load layout cache state."""
        ...


class EmbeddingCacheProtocol(Protocol):
    """Protocol for embedding cache operations."""

    def get(self, key: str) -> np.ndarray | None:
        """Get cached embedding vector by key (content_hash or tone key)."""
        ...

    def put(self, key: str, vector: np.ndarray) -> None:
        """Cache embedding vector."""
        ...

    def has(self, key: str) -> bool:
        """Check if embedding is cached."""
        ...

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        ...

    def save(self) -> None:
        """Persist embedding cache state."""
        ...

    def load(self) -> None:
        """Load embedding cache state."""
        ...


class UsedPaintingsProtocol(Protocol):
    """Protocol for used paintings tracking."""

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting as used."""
        ...

    def is_used(self, content_hash: str) -> bool:
        """Check if a painting has been used."""
        ...

    def reset(self) -> None:
        """Reset used paintings tracking (allow re-use of all paintings)."""
        ...

    @property
    def count(self) -> int:
        """Number of used paintings tracked."""
        ...

    def save(self) -> None:
        """Persist used paintings state."""
        ...

    def load(self) -> None:
        """Load used paintings state."""
        ...


class ArtFactRotationStateProtocol(Protocol):
    """Protocol for orchestrator-owned art-fact rotation state.

    Contract:
        - State is keyed by ``painting.content_hash``.
        - Values represent the next zero-based ``whisper_fact_index`` to emit for
          that painting.
        - Missing state means "start at index 0" when ``art_fact_count > 0``.
        - Callers must rebase stored indexes modulo ``art_fact_count`` before use.
        - Empty art-fact sets map to ``None`` and should not require persisted
          cursor state.
    """

    def get_next_index(
        self, content_hash: str, art_fact_count: int
    ) -> WhisperFactIndex:
        """Return the next caller-owned whisper index for ``content_hash``."""
        ...

    def commit_rotation(
        self, content_hash: str, art_fact_count: int
    ) -> WhisperFactIndex:
        """Advance and persist the next whisper index after a successful rotate."""
        ...

    def clear(self, content_hash: str) -> None:
        """Drop any stored cursor for ``content_hash``."""
        ...


class AppStateProtocol(Protocol):
    """Protocol for application state."""

    @property
    def news_queue(self) -> NewsQueueProtocol:
        """News queue state component."""
        ...

    @property
    def used_paintings(self) -> UsedPaintingsProtocol:
        """Used paintings state component."""
        ...

    @property
    def replay_queue(self) -> ReplayQueueProtocol:
        """Replay queue state component."""
        ...

    @property
    def layout_cache(self) -> LayoutCacheProtocol:
        """Layout cache state component."""
        ...

    @property
    def embedding_cache(self) -> EmbeddingCacheProtocol:
        """Embedding cache state component."""
        ...

    @property
    def art_fact_rotation(self) -> ArtFactRotationStateProtocol:
        """Art-fact rotation state component."""
        ...

    def save_all(self) -> None:
        """Persist all state components."""
        ...


# =============================================================================
# PUBLIC API
# =============================================================================


async def run_news_refresh(
    config: AppConfig,
    state: AppStateProtocol,
) -> int:
    """Fetch, curate, and enqueue news batches.

    This is the news refresh operation triggered by the scheduler every
    ``news_interval_hours`` or on-demand when the queue is empty.

    Pipeline:
    1. Call ``news.refresh_news()`` to fetch and curate stories
    2. Expire old batches from the queue (using config.news.expire_days)
    3. Split curated stories into batches (using recommended_stories from layout)
    4. Enqueue batches into state.news_queue

    Args:
        config: Application configuration with news settings.
        state: Application state (news queue for enqueue, caches for layout).

    Returns:
        Number of batches enqueued (0 if no stories were curated).

    Raises:
        LlmError: If the LLM call fails after retries.
        LlmParseError: If the LLM response cannot be parsed.

    Contract:
        - Uses config.news.max_age_days for filtering old articles
        - Uses config.news.expire_days for expiring old queue batches
        - Uses config.news.replay_expire_days for expiring replay batches during refresh
        - Uses config.news.stories_per_refresh as the target batch size default
        - Skips primary curation enqueue when replay overlap exceeds
          RUN_NEWS_REFRESH_REPLAY_CURATION_SKIP_OVERLAP_RATIO
        - Individual feed fetch failures are non-fatal (logged, skipped)
        - State is NOT saved; caller must call state.save_all() if persistence needed
    """
    # Batch size = how many stories per rotation display (not total curated).
    # stories_per_refresh is total curated (~12), we split into batches of 3-4.
    batch_size = min(4, config.news.stories_per_refresh)

    # Fetch and curate news
    result = await refresh_news(
        news_config=config.news,
        ai_config=config.ai,
    )

    if not result.stories:
        logger.info("No stories curated from feeds")
        return 0

    # Expire old batches before adding new ones
    expired_count = state.news_queue.expire(config.news.expire_days)
    if expired_count > 0:
        logger.info("Expired %d old batches from queue", expired_count)

    # Enqueue in batches
    enqueued_count = state.news_queue.enqueue(result, batch_size)
    logger.info(
        "Enqueued %d batches (%d stories total, %d feeds fetched)",
        enqueued_count,
        len(result.stories),
        result.feed_count,
    )

    return enqueued_count


async def run_backfill(
    config: AppConfig,
    state: AppStateProtocol,
) -> int:
    """Expand the local painting pool toward ``config.paintings.pool_size``.

    Contracted stage order:
    1. Measure current pool deficit relative to ``pool_size``
    2. Fetch new paintings only when deficit > 0
    3. Analyze layout for each newly fetched painting
    4. Compute and cache embeddings for each newly fetched painting
    5. Persist state

    Bounded behavior contract:
    - Never fetch/process more than the current deficit in one call.
    - Return count is bounded by ``[0, deficit]``.
    - If pool already meets/exceeds target size, return ``0``.

    Error boundary contract:
    - Individual painting failures are skipped and do not abort the full backfill.
    - If all sources fail, the call returns ``0`` (cached pool remains usable).
    - Persistence failures propagate to the caller.

    Source:
    - ARCHITECTURE.md#2.12 ``run_backfill``
    - ARCHITECTURE.md#10.1 error strategy table
    - README.md backfill target notes.

    Args:
        config: Application configuration.
        state: Application state for cache and persistence updates.

    Returns:
        Number of paintings successfully added to the pool.
    """
    # Stage 1: measure_pool_deficit
    current_paintings = list_cached_paintings(config.paintings.cache_dir)
    current_count = len(current_paintings)
    target_size = config.paintings.pool_size

    # Bounded behavior: if pool already meets target, return 0 immediately
    if current_count >= target_size:
        logger.info(
            "Pool already at target size (%d >= %d), skipping backfill",
            current_count,
            target_size,
        )
        return 0

    # Calculate how many paintings we need
    deficit = target_size - current_count
    logger.info(
        "Pool deficit: %d paintings needed (current: %d, target: %d)",
        deficit,
        current_count,
        target_size,
    )

    # Stage 2: fetch_new_paintings_if_needed
    # Get existing content hashes to exclude
    existing_hashes = {p.content_hash for p in current_paintings}

    try:
        new_paintings = await fetch_paintings(
            sources=config.paintings.sources,
            count=deficit,
            cache_dir=config.paintings.cache_dir,
            exclude_ids=existing_hashes,
        )
    except Exception as e:
        logger.warning("All painting sources failed during backfill: %s", e)
        # Source-level failure: return 0, cached pool remains usable
        return 0

    if not new_paintings:
        logger.info("No new paintings fetched during backfill")
        return 0

    logger.info("Fetched %d new paintings for analysis", len(new_paintings))

    # Stages 3-4: analyze_layout and compute_embeddings for each painting
    # Track successfully processed paintings (bounded by deficit)
    added_count = 0

    for painting in new_paintings:
        # Don't exceed the deficit
        if added_count >= deficit:
            break

        try:
            # Stage 3: analyze_layout_for_new_paintings
            if state.layout_cache.has(painting.content_hash):
                logger.debug(
                    "Layout already cached for %s, skipping analysis",
                    painting.content_hash[:8],
                )
            else:
                layout = await analyze_painting(painting.image_path, config.ai)
                state.layout_cache.put(painting.content_hash, layout)
                logger.debug(
                    "Analyzed layout for %s: %s",
                    painting.content_hash[:8],
                    layout.template_hint,
                )

            # Stage 4: compute_embeddings_for_new_paintings
            if state.embedding_cache.has(painting.content_hash):
                logger.debug(
                    "Embedding already cached for %s, skipping computation",
                    painting.content_hash[:8],
                )
            else:
                # Get description from layout cache
                layout = state.layout_cache.get(painting.content_hash)
                if layout is None or not layout.painting_description:
                    logger.warning(
                        "No layout/description for %s, skipping embedding",
                        painting.content_hash[:8],
                    )
                    continue

                embedding_result = await compute_embedding(
                    layout.painting_description, config.ai
                )
                state.embedding_cache.put(
                    painting.content_hash, embedding_result.vector
                )
                logger.debug(
                    "Computed embedding for %s (%d dimensions)",
                    painting.content_hash[:8],
                    len(embedding_result.vector),
                )

            added_count += 1

        except Exception as e:
            # Item-level failure: non-fatal, log and continue
            logger.warning(
                "Failed to process painting %s during backfill: %s",
                painting.content_hash[:8],
                e,
            )
            continue

    # Stage 5: state_save
    # Persistence failures propagate (fatal error boundary)
    state.save_all()

    logger.info(
        "Backfill complete: added %d paintings (target deficit was %d)",
        added_count,
        deficit,
    )

    return added_count


async def watch(config: AppConfig) -> None:
    """Run daemon watch mode with scheduler-driven action dispatch.

    Contracted loop stage order:
    1. Load state once at startup
    2. Ask scheduler ``what_to_do(now, scheduler_state)`` each cycle
    3. Dispatch actions using ``WATCH_ACTION_DISPATCH_ORDER``
    4. Persist state after the action cycle
    5. Sleep until ``scheduler.seconds_until_next_action(...)``

    Scheduler action mapping:
    - ``REFRESH_NEWS`` -> ``run_news_refresh``
    - ``ROTATE`` -> ``run_once(..., RunOptions())``
    - ``BACKFILL`` -> ``run_backfill``
    - ``QUIET_ART`` -> ``run_once(..., RunOptions(no_news=True))``
    - ``IDLE`` -> no pipeline action

    Graceful shutdown semantics:
    - Handle ``SIGINT`` and ``SIGTERM``.
    - Finish the current in-flight action boundary.
    - Persist state before exit.
    - Do not start another scheduler cycle after shutdown is requested.

    Error/retry boundary contract:
    - Per-action recoverable errors are scoped to the current action and retried by future scheduler cycles.
    - Fatal bootstrap/persistence failures propagate and terminate daemon startup/runtime.

    Source:
    - ARCHITECTURE.md#2.11 scheduler contract
    - ARCHITECTURE.md#2.12 watch contract
    - ARCHITECTURE.md#6.1 loop pseudocode
    - ARCHITECTURE.md#9.5 shutdown semantics
    - ARCHITECTURE.md#10.1 error handling table.

    Args:
        config: Application configuration.
    """
    import asyncio
    import signal
    from datetime import datetime, timezone

    from sfumato.scheduler import Action, Scheduler, SchedulerState

    # Stage 1: load_state_once
    state_dir = config.data_dir / "state"
    state = AppState.load(state_dir)
    typed_state = cast(AppStateProtocol, state)

    def write_health(
        status: str, action_names: list[str], error: str | None = None
    ) -> None:
        health_path = state_dir / WATCH_HEALTH_FILE
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "actions": action_names,
            "error": error,
        }
        temp_path = health_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temp_path.replace(health_path)

    # Initialize scheduler and scheduler state
    scheduler = Scheduler(config.schedule)
    scheduler_state = SchedulerState(
        last_news_refresh=None,
        last_rotation=None,
        last_backfill=None,
    )

    # Shutdown coordination
    shutdown_requested = False

    def handle_shutdown(signum: int, frame: object) -> None:
        nonlocal shutdown_requested
        logger.info("Received shutdown signal %s, finishing current action...", signum)
        shutdown_requested = True

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    logger.info("Watch daemon started, entering main loop")
    write_health("starting", [])

    # Main daemon loop
    while not shutdown_requested:
        # Stage 2: scheduler_decision
        now = datetime.now()
        action = scheduler.what_to_do(now, scheduler_state)

        # Stage 3: action_dispatch
        # Handle combined actions in DISPATCH_ORDER
        actions_to_dispatch = []

        # Check for combined actions (news refresh + rotate)
        if Action.REFRESH_NEWS in action and Action.ROTATE in action:
            actions_to_dispatch = [Action.REFRESH_NEWS, Action.ROTATE]
        elif Action.REFRESH_NEWS in action:
            actions_to_dispatch = [Action.REFRESH_NEWS]
        elif Action.ROTATE in action:
            actions_to_dispatch = [Action.ROTATE]
        elif Action.BACKFILL in action:
            actions_to_dispatch = [Action.BACKFILL]
        elif Action.QUIET_ART in action:
            actions_to_dispatch = [Action.QUIET_ART]
        elif Action.IDLE in action:
            # IDLE: no pipeline action, just state save + sleep
            logger.debug("Scheduler returned IDLE, skipping to sleep")
        else:
            # NONE: no action needed
            pass

        # Dispatch actions in WATCH_ACTION_DISPATCH_ORDER
        executed_actions: list[str] = []
        action_error: str | None = None
        for act in actions_to_dispatch:
            # Check for shutdown between actions
            if shutdown_requested:
                break

            try:
                if act == Action.REFRESH_NEWS:
                    logger.info("Dispatching REFRESH_NEWS")
                    await run_news_refresh(config, typed_state)
                    scheduler_state.last_news_refresh = now
                    executed_actions.append(str(act.name))

                elif act == Action.ROTATE:
                    logger.info("Dispatching ROTATE")
                    await run_once(config, typed_state, RunOptions())
                    scheduler_state.last_rotation = now
                    executed_actions.append(str(act.name))

                elif act == Action.BACKFILL:
                    logger.info("Dispatching BACKFILL")
                    added = await run_backfill(config, typed_state)
                    scheduler_state.last_backfill = now
                    logger.info("Backfill added %d paintings", added)
                    executed_actions.append(str(act.name))

                elif act == Action.QUIET_ART:
                    logger.info("Dispatching QUIET_ART")
                    await run_once(config, typed_state, RunOptions(no_news=True))
                    scheduler_state.last_rotation = now
                    executed_actions.append(str(act.name))

            except Exception as e:
                # Per-action recoverable errors: scope to current action
                # Daemon loop continues on next scheduled tick
                logger.error(
                    "Action %s failed (recoverable): %s. "
                    "Daemon will continue on next cycle.",
                    act.name,
                    e,
                )
                action_error = f"{act.name}: {e}"
                # Don't update timestamps for failed actions
                # Next cycle will retry

        # Stage 4: state_save
        try:
            state.save_all()
            write_health(
                "degraded" if action_error else "ok",
                executed_actions,
                error=action_error,
            )
        except Exception as e:
            # Persistence failure: fatal, propagate
            logger.critical("State persistence failed, terminating daemon: %s", e)
            raise

        # Stage 5: sleep_until_next_action
        if shutdown_requested:
            # Don't sleep if shutdown requested, exit immediately after state save
            break

        sleep_seconds = scheduler.seconds_until_next_action(now, scheduler_state)
        if sleep_seconds > 0:
            logger.debug("Sleeping for %.1f seconds until next action", sleep_seconds)
            await asyncio.sleep(sleep_seconds)

    logger.info("Watch daemon shut down gracefully")
    write_health("stopped", [], error=None)


async def init_project(config: AppConfig) -> None:
    """Initialize the sfumato project.

    This is a potentially long operation (~50 LLM calls for 50 paintings).
    Progress is printed to stdout.

    Steps:
    1. Create config file if not present
    2. Create state directory structure
    3. Fetch seed_size paintings from configured sources
    4. Analyze each painting (layout + description)
    5. Compute embeddings for each painting

    Args:
        config: Application configuration.

    Raises:
        OSError: If directories cannot be created.
        Exception: Propagated from painting fetch, layout analysis, or embedding compute.
    """
    from sfumato.config import generate_default_config

    # Step 1: Create config file if not present
    config_path = config.data_dir / "config.toml"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(generate_default_config())
        print(f"Created config file: {config_path}")
    else:
        print(f"Config file already exists: {config_path}")

    # Step 2: Create state directory structure
    state_dir = config.data_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    paintings_dir = config.paintings.cache_dir
    paintings_dir.mkdir(parents=True, exist_ok=True)
    output_dir = config.data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directories:")
    print(f"  - State: {state_dir}")
    print(f"  - Paintings: {paintings_dir}")
    print(f"  - Output: {output_dir}")

    # Steps 3-5: Pipeline — download paintings and analyze them concurrently
    # Producer: downloads paintings, puts them in queue
    # Consumer: analyzes layout + computes embeddings as paintings arrive
    print(f"\nFetching and analyzing {config.paintings.seed_size} paintings...")
    print(f"  Sources: {config.paintings.sources}")

    state = AppState.load(state_dir)
    queue: asyncio.Queue[PaintingsPaintingInfo | None] = asyncio.Queue(maxsize=5)
    download_count = 0
    success_count = 0

    async def producer() -> None:
        """Download paintings and feed them into the queue."""
        nonlocal download_count
        paintings = await fetch_paintings(
            sources=config.paintings.sources,
            count=config.paintings.seed_size,
            cache_dir=config.paintings.cache_dir,
            exclude_ids=None,
        )
        download_count = len(paintings)
        for painting in paintings:
            await queue.put(painting)
        await queue.put(None)  # Sentinel: signal consumer to stop

    async def consumer() -> None:
        """Analyze and embed paintings as they arrive from the queue."""
        nonlocal success_count
        idx = 0
        while True:
            painting = await queue.get()
            if painting is None:
                break
            idx += 1

            try:
                print(f"  [{idx}] {painting.title} — {painting.artist}")

                # Layout analysis (cached)
                if state.layout_cache.has(painting.content_hash):
                    print(f"      Layout cached, skipping")
                else:
                    layout = await analyze_painting(painting.image_path, config.ai)
                    state.layout_cache.put(painting.content_hash, layout)
                    print(f"      Layout: {layout.template_hint}")

                # Embedding (cached)
                if state.embedding_cache.has(painting.content_hash):
                    print(f"      Embedding cached, skipping")
                else:
                    layout = state.layout_cache.get(painting.content_hash)
                    if layout and layout.painting_description:
                        embedding_result = await compute_embedding(
                            layout.painting_description, config.ai
                        )
                        state.embedding_cache.put(
                            painting.content_hash, embedding_result.vector
                        )
                        print(f"      Embedded ({len(embedding_result.vector)}d)")

                success_count += 1

            except Exception as e:
                print(f"      Error: {e}")
                logger.warning(
                    "Failed to process painting %s: %s",
                    painting.content_hash,
                    e,
                )

    # Run producer and consumer concurrently
    await asyncio.gather(producer(), consumer())

    state.save_all()
    print(f"\nInitialization complete!")
    print(f"  - Paintings downloaded: {download_count}")
    print(f"  - Successfully processed: {success_count}")
    print(f"  - State saved to: {state_dir}")


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

    Replay wiring contract:
        - primary news queue precedes replay queue for content selection
        - consumed primary batches are transferred to replay before downstream use
        - replay is a fallback only after the primary dequeue/refresh path is empty

    Art-fact wiring contract:
    - ``layout_analysis`` is the sole source of ``layout.art_facts`` for the cycle,
      whether produced freshly or loaded from ``state.layout_cache``.
    - ``RenderContext.layout`` preserves the selected ``LayoutParams`` unchanged,
      including ordered ``art_facts``.
    - ``RenderContext.whisper_fact_index`` is caller-owned orchestration metadata;
      render consumes it but does not rotate or persist it.
    - Rotate-mode index advancement happens only after a successful rotate commit
      boundary, so failed attempts do not consume an art fact.

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
    # CONTRACT: If queue is empty, trigger on-demand refresh then dequeue
    batch: QueuedBatch | None = None
    if not options.no_news:
        batch = state.news_queue.dequeue()
        if batch is None:
            # On-demand refresh when queue is empty
            await run_news_refresh(config, state)
            batch = state.news_queue.dequeue()

    # Stage 2: painting_selection
    # CONTRACT: Use painting_path if specified, semantic matching when available
    selected_painting, match_score = await _select_painting(
        config=config,
        state=state,
        painting_path=options.painting_path,
        batch=batch,
    )
    # Convert from paintings.PaintingInfo to render.PaintingInfo
    painting = _convert_painting_info(selected_painting)

    # Stage 3: layout_analysis
    # CONTRACT: Check cache first, analyze if not cached, propagate errors
    layout = await _analyze_layout(
        painting=painting,
        state=state,
        config=config,
    )

    # Stage 3.5: resolve_whisper_fact_index
    # CONTRACT: ART_FACT_INDEX_STATE_OWNERSHIP - orchestrator resolves index
    # from rotation state before render, empty art_facts => index is None
    art_fact_count = len(layout.art_facts)
    whisper_fact_index: int | None = state.art_fact_rotation.get_next_index(
        painting.content_hash, art_fact_count
    )

    # Stage 4: palette_extraction
    # CONTRACT: Always extract (no caching in palette module), propagate errors
    palette = extract_palette(painting.image_path)

    # Stage 5: template_selection
    # CONTRACT: Use layout.template_hint with orientation fallback
    template_name = _select_template(layout, painting)

    # Stage 6: render_4k_png
    # CONTRACT: Render to 4K PNG, propagate errors
    # BUG #8 FIX: Apply recommended_stories from layout to limit stories rendered
    # ARCHITECTURE.md A.4: Batch sizing is determined by selected painting's layout
    all_stories = batch.stories if batch else []
    if all_stories and layout.recommended_stories:
        # Limit stories to what the layout recommends can fit
        stories = all_stories[: layout.recommended_stories]
    else:
        stories = all_stories
    render_result = await _render_4k(
        painting=painting,
        stories=stories,
        layout=layout,
        palette=palette,
        template_name=template_name,
        config=config,
        whisper_fact_index=whisper_fact_index,
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

    # Stage 8.5: commit_art_fact_rotation
    # CONTRACT: Only advance rotation after successful render (failure boundary)
    # ROTATE_ART_FACT_ACCEPTANCE_CRITERIA: successful rotate advances modulo count
    if art_fact_count > 0:
        state.art_fact_rotation.commit_rotation(painting.content_hash, art_fact_count)

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
        match_score=match_score,
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
) -> tuple[PaintingsPaintingInfo, float | None]:
    """Select a painting for this rotation.

    Selection order:
    1. If painting_path is specified, use it directly (returns match_score=None)
    2. If batch is None (no_news), use random unused painting from pool
    3. If match_strategy is "random", use random selection (returns match_score=0.0)
    4. Otherwise, use semantic matching with batch.tone_description

    Args:
        config: Application configuration.
        state: Application state.
        painting_path: Optional path to specific painting.
        batch: Current news batch (None if no_news).

    Returns:
        Tuple of (selected_painting, similarity_score).
        - match_score is None when painting_path is specified.
        - match_score is 0.0 for random strategy.
        - match_score is in [0.0, 1.0] for semantic strategy.
        - Painting is PaintingsPaintingInfo from paintings module.

    Raises:
        ValueError: If painting_path specified but file doesn't exist.
        RuntimeError: If no paintings available in pool.
        MatcherError: If semantic matching fails.
    """
    # If specific painting is requested, create PaintingInfo from file
    if painting_path is not None:
        if not painting_path.exists():
            raise ValueError(f"Painting file not found: {painting_path}")

        return (_create_painting_info_from_path(painting_path), None)

    # Load painting pool from cache
    paintings = list_cached_paintings(config.paintings.cache_dir)
    if not paintings:
        raise RuntimeError("No paintings available in pool")

    # Filter out used paintings
    available = [
        p for p in paintings if not state.used_paintings.is_used(p.content_hash)
    ]
    if not available:
        # All paintings used, reset and use any
        state.used_paintings.reset()
        available = paintings

    # If no_news (pure art mode), use random selection
    if batch is None:
        selected = random.choice(available)
        return (selected, None)

    # Get match strategy from config
    strategy = config.paintings.match_strategy

    # If random strategy, select randomly
    if strategy == "random":
        selected = random.choice(available)
        return (selected, 0.0)

    # Semantic matching strategy
    # Build painting_descriptions dict from layout cache
    # For paintings without cached layouts, we need descriptions for embedding
    painting_descriptions: dict[str, str] = {}
    for p in available:
        cached_layout = state.layout_cache.get(p.content_hash)
        if cached_layout is not None:
            painting_descriptions[p.content_hash] = cached_layout.painting_description

    # Build embedding cache for lookups (state.embedding_cache is a dict-like cache)
    embedding_cache: dict[str, np.ndarray] = {}
    # Copy existing embeddings from state cache
    for p in available:
        if state.embedding_cache.has(p.content_hash):
            vec = state.embedding_cache.get(p.content_hash)
            if vec is not None:
                embedding_cache[p.content_hash] = vec

    # Check tone cache
    from sfumato.matcher import _compute_tone_cache_key

    tone_key = _compute_tone_cache_key(batch.tone_description)
    if state.embedding_cache.has(tone_key):
        vec = state.embedding_cache.get(tone_key)
        if vec is not None:
            embedding_cache[tone_key] = vec

    try:
        selected, score = await select_painting(
            news_tone=batch.tone_description,
            paintings=available,
            painting_descriptions=painting_descriptions,
            embedding_cache=embedding_cache,
            ai_config=config.ai,
            strategy=strategy,
        )

        # Cache tone embedding if it was computed
        if not state.embedding_cache.has(tone_key):
            tone_embedding = embedding_cache.get(tone_key)
            if tone_embedding is not None:
                state.embedding_cache.put(tone_key, tone_embedding)

        return (selected, score)

    except MatcherError:
        # Fallback to random selection if semantic matching fails
        logger.warning(
            "Semantic matching failed, falling back to random selection",
        )
        selected = random.choice(available)
        return (selected, 0.0)


def _create_painting_info_from_path(painting_path: Path) -> PaintingsPaintingInfo:
    """Create PaintingInfo from an image file path.

    Args:
        painting_path: Path to the painting image file.

    Returns:
        PaintingInfo from paintings module with detected orientation and generated hash.

    Raises:
        OSError: If file cannot be read.
    """
    from PIL import Image

    # Load image to get dimensions
    img = Image.open(painting_path)
    width, height = img.size

    # Detect orientation (using paintings.Orientation enum)
    orientation = (
        PaintingsOrientation.LANDSCAPE
        if width >= height
        else PaintingsOrientation.PORTRAIT
    )

    # Compute content hash
    import hashlib

    content = painting_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()

    return PaintingsPaintingInfo(
        image_path=painting_path.resolve(),
        content_hash=content_hash,
        title="Unknown",
        artist="Unknown",
        year="Unknown",
        orientation=orientation,
        width=width,
        height=height,
        source=ArtSource.MET,  # Placeholder source for user-provided paintings
        source_id=content_hash[:12],
        source_url="",
    )


def _convert_painting_info(p: PaintingsPaintingInfo) -> PaintingInfo:
    """Convert from paintings.PaintingInfo to render.PaintingInfo.

    This is needed because the two modules have separate PaintingInfo types.
    The paintings module has full metadata while render has a minimal version.

    Args:
        p: PaintingInfo from paintings module.

    Returns:
        PaintingInfo for render module.
    """
    # Convert orientation enum
    render_orientation = (
        Orientation.LANDSCAPE
        if p.orientation == PaintingsOrientation.LANDSCAPE
        else Orientation.PORTRAIT
    )

    return PaintingInfo(
        image_path=p.image_path,
        content_hash=p.content_hash,
        title=p.title,
        artist=p.artist,
        year=p.year,
        orientation=render_orientation,
        width=p.width,
        height=p.height,
        source=p.source.value if hasattr(p.source, "value") else str(p.source),
        source_id=p.source_id,
        source_url=p.source_url,
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
    whisper_fact_index: int | None = None,
) -> "RenderResult":
    """Render the 4K PNG.

    Args:
        painting: Painting info.
        stories: News stories to render.
        layout: Layout parameters.
        palette: Extracted color palette.
        template_name: Template to use.
        config: Application configuration.
        whisper_fact_index: Caller-selected art-fact index for whisper rendering.

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
        whisper_fact_index=whisper_fact_index,
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
