"""Scheduler for the sfumato daemon's dual-timer architecture.

Implements the contract from ARCHITECTURE.md#2.11:
- Action enum for scheduling decisions
- SchedulerState for tracking last action timestamps
- Scheduler class enforcing quiet_hours/active_hours and interval logic

This module is a CONTRACT STUB. Implementation will follow in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Flag, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sfumato.config import ScheduleConfig


class Action(Flag):
    """Scheduling actions that can be combined using bitwise OR.

    The scheduler returns Action flags to indicate what should happen.
    Multiple actions can be combined: REFRESH_NEWS | ROTATE means both
    should run in sequence.

    Semantics (from ARCHITECTURE.md#2.11):
        NONE:        No action needed; daemon can sleep.
        REFRESH_NEWS: Fetch RSS feeds, curate via LLM, enqueue story batches.
        ROTATE:     Dequeue news, select painting, render 4K PNG, push to TV.
        BACKFILL:   Background painting pool expansion; fetch + analyze + embed.
        QUIET_ART:  Display pure painting (no news overlay); used in quiet_hours.
        IDLE:       Outside active hours; no rendering, no pushing, no network.
    """

    NONE = 0
    REFRESH_NEWS = auto()
    ROTATE = auto()
    BACKFILL = auto()
    QUIET_ART = auto()
    IDLE = auto()


@dataclass
class SchedulerState:
    """Timestamps of last actions for interval calculations.

    The scheduler uses these timestamps to determine if enough time has
    elapsed since the last news refresh, rotation, or backfill.

    Fields:
        last_news_refresh: When news was last fetched and curated, or None
            if never refreshed. Used to check news_interval_hours.
        last_rotation: When a painting was last rotated, or None if never
            rotated. Used to check rotate_interval_minutes.
        last_backfill: When the painting pool was last expanded, or None
            if never backfilled. Used for background expansion logic.
    """

    last_news_refresh: datetime | None
    last_rotation: datetime | None
    last_backfill: datetime | None


class Scheduler:
    """Manages the dual-timer architecture for the watch daemon.

    Responsibility:
        Determine which actions should run at a given time, respecting
        quiet_hours and active_hours configuration, and calculating sleep
        durations between actions.

    Non-Responsibility:
        Does not execute actions (orchestrator does that).
        Does not persist state (caller is responsible).
        Does not know about painting pool status (passed as context).

    Time Window Precedence (ARCHITECTURE.md#6.2):
        The scheduler processes time windows in this order:

        1. OUTSIDE ACTIVE HOURS (and NOT IN QUIET HOURS) → IDLE
           No rendering, no pushing, no network calls.

        2. WITHIN QUIET HOURS (must also be outside active_hours) → QUIET_ART
           Display pure painting without news overlay. TV may still be
           in Art Mode during these hours.

        3. WITHIN ACTIVE HOURS (and outside quiet_hours) → Normal operation
           Check intervals for ROTATE, REFRESH_NEWS, and BACKFILL.

    IMPORTANT: With default configuration where quiet_hours=(0,6) and
    active_hours=(7,23), these ranges DO NOT OVERLAP. This means:
        - Hours 0-5: outside active_hours → falls through to check quiet_hours
        - Hour 6: outside both → IDLE
        - Hours 7-23: within active_hours, outside quiet_hours → normal operation

    For QUIET_ART to activate, configure overlapping ranges such as:
        quiet_hours = (0, 6)
        active_hours = (0, 23)  # Covers entire day including quiet hours

    Interval Semantics:
        - news_interval_hours: Minimum time between news refreshes.
          If `now - last_news_refresh >= news_interval_hours`, trigger REFRESH_NEWS.
        - rotate_interval_minutes: Minimum time between rotations.
          If `now - last_rotation >= rotate_interval_minutes`, trigger ROTATE.
        - REFRESH_NEWS and ROTATE can combine: on first refresh of the day,
          both may be scheduled together.

    Backfill Trigger Semantics:
        BACKFILL should only be returned when:
        - Pool has room for more paintings (paintings_used < pool_size)
        - No heavy action (REFRESH_NEWS or ROTATE) is scheduled
        - System is within active_hours and outside quiet_hours

        BACKFILL is a background activity that shouldn't compete with
        foreground operations.

    Edge Cases:
        - Daemon starts during quiet_hours: Initialize state but return QUIET_ART,
          not IDLE (we're in a valid display window).
        - Daemon starts outside active_hours (but not in quiet_hours): Return IDLE.
        - last_rotation is None (first run): Treat as immediate ROTATE needed.
        - last_news_refresh is None (first run): Treat as REFRESH_NEWS needed.
        - Crossing boundary from quiet to active: May trigger REFRESH_NEWS | ROTATE.
    """

    def __init__(self, config: "ScheduleConfig") -> None:
        """Initialize scheduler with configuration.

        Args:
            config: ScheduleConfig containing:
                - news_interval_hours: Hours between news refreshes (default 6)
                - rotate_interval_minutes: Minutes between rotations (default 15)
                - quiet_hours: (start, end) tuple in 24h, [start, end) interval
                - active_hours: (start, end) tuple in 24h, [start, end] interval
        """
        # Contract: Store config for interval and time window calculations.
        # Implementation will compute thresholds and boundaries.
        ...

    def what_to_do(self, now: datetime, state: SchedulerState) -> Action:
        """Determine which actions should run at the given time.

        This is the primary scheduling decision function. It returns one
        or more Actions (combined via | for multiple) indicating what the
        daemon should do right now.

        Decision Logic (in order of precedence):

        1. CHECK ACTIVE HOURS
           If `now` is NOT within active_hours:
               a. If `now` IS within quiet_hours: return QUIET_ART
               b. Else: return IDLE

           The precedence ensures quiet_hours behavior activates even
           when outside active_hours, supporting the "night art mode"
           use case.

        2. CHECK NEWS INTERVAL
           If `state.last_news_refresh` is None OR
              `now - state.last_news_refresh >= news_interval_hours`:
               Return REFRESH_NEWS | ROTATE
               (Both actions combined; rotation follows news refresh)

        3. CHECK ROTATE INTERVAL
           If `state.last_rotation` is None OR
              `now - state.last_rotation >= rotate_interval_minutes`:
               Return ROTATE

        4. CHECK BACKFILL (placeholder for future pool status check)
           Contract: If pool has room and no heavy work scheduled,
           return BACKFILL.

           NOTE: This step requires pool status which is NOT in
           SchedulerState. Implementation will need additional context
           or a method signature extension.

           For this contract: Return NONE if no other action needed.
           BACKFILL logic will be deferred to a follow-up.

        5. DEFAULT
           Return NONE (nothing to do, sleep until next interval)

        Args:
            now: Current datetime (timezone-naive or aware, implementation
                 should handle both by normalizing).
            state: SchedulerState with timestamps of last actions.

        Returns:
            Action flag indicating what to do. Can be combined:
            - REFRESH_NEWS | ROTATE (first run after news interval)
            - ROTATE (regular rotation)
            - BACKFILL (background painting expansion)
            - QUIET_ART (during quiet hours)
            - IDLE (outside active hours, not in quiet hours)
            - NONE (within active hours, intervals not elapsed)
        """
        ...

    def seconds_until_next_action(self, now: datetime, state: SchedulerState) -> float:
        """Calculate seconds until the next action is due.

        Used by the daemon's main loop to sleep efficiently. Returns the
        minimum time until any of the following:
        - News interval boundary
        - Rotate interval boundary
        - Transition into active_hours (if currently outside)
        - Transition out of active_hours (if currently within)

        Calculation Strategy:

        1. COMPUTE ALL RELEVANT WAIT TIMES
           - Time until next rotate boundary
           - Time until next news boundary
           - Time until active_hours starts (if outside)
           - Time until active_hours ends (if inside and relevant)

        2. RETURN THE MINIMUM
           The daemon should wake up at the earliest opportunity.

        Edge Cases:
           - If any action is due NOW (e.g., what_to_do returns non-NONE),
             this should return 0.0.
           - If last_rotation is None, rotate is due immediately → return 0.0.
           - If last_news_refresh is None, news refresh is due → return 0.0
             (or time until next rotation boundary, whichever is sooner).
           - Crossing midnight: Correctly compute time to next active_hours
             start when currently before start or after end.

        Args:
            now: Current datetime.
            state: SchedulerState with last action timestamps.

        Returns:
            Floating-point seconds until the next action. Returns 0.0
            if an action is already due. Guarantees non-negative result.
        """
        ...

    def is_quiet_hour(self, now: datetime) -> bool:
        """Check if the given time falls within quiet_hours.

        Quiet hours are typically late night (e.g., 0:00-6:00) where the
        TV may still be on in Art Mode, but full operation is not desired.
        During quiet hours, the scheduler returns QUIET_ART, displaying
        paintings without news overlays.

        Interval Semantics:
           - quiet_hours[0] is the start hour (inclusive)
           - quiet_hours[1] is the end hour (exclusive)
           - Hour values are in 24-hour format (0-23)
           - Example: (0, 6) means hours 0, 1, 2, 3, 4, 5 are quiet

        Edge Cases:
           - Crossing midnight: quiet_hours=(22, 2) means 22:00-23:59 and 0:00-1:59.
             Implementation must handle end < start by checking both ranges.
           - Quiet hours of (0, 0) or (x, x) is an empty interval → always False.
           - Hour boundary: is_quiet_hour(5:59:59) should return True for (0, 6).
           - Hour boundary: is_quiet_hour(6:00:00) should return False for (0, 6).

        Args:
            now: Current datetime. Uses the hour component only.

        Returns:
            True if the hour falls within [start, end), False otherwise.
        """
        ...

    def is_active_hour(self, now: datetime) -> bool:
        """Check if the given time falls within active_hours.

        Active hours are when the daemon operates at full capacity:
        news refreshes, painting rotations, and backfill operations.
        Outside active hours, the daemon may be IDLE or in QUIET_ART mode.

        Interval Semantics:
           - active_hours[0] is the start hour (inclusive)
           - active_hours[1] is the end hour (inclusive, NOT exclusive)
           - Hour values are in 24-hour format (0-23)
           - Example: (7, 23) means hours 7, 8, ..., 22, 23 are active

        This differs from quiet_hours which uses exclusive end.
        (See ARCHITECTURE.md#2.1 for rationale.)

        Edge Cases:
           - Crossing midnight: active_hours=(22, 2) with inclusive end is
             NOT well-defined. Implementations should either:
             a) Reject such config at construction time, or
             b) Interpret as 22:00-23:59 AND 0:00-2:00 (wrapping).
           - Active hours of (0, 23) means active all day.
           - Active hours of (7, 6) where end < start: Invalid, but
             implementation should handle gracefully (return False or raise).
           - Hour boundary: is_active_hour(22:59:59) returns True for (7, 23).
           - Hour boundary: is_active_hour(23:00:00) returns True for (7, 23).
           - Hour boundary: is_active_hour(23:00:01) should return True for (7, 23).
             Implementation should use hour comparison, not timedelta.

        Args:
            now: Current datetime. Uses the hour component only.

        Returns:
            True if the hour falls within [start, end] (both inclusive),
            False otherwise.
        """
        ...

    # -----------------------------------------------------------------------
    # Internal Helper Methods (contract signatures, no implementation)
    # -----------------------------------------------------------------------

    def _is_time_in_range(
        self,
        now: datetime,
        start: int,
        end: int,
        end_inclusive: bool,
    ) -> bool:
        """Check if `now.hour` falls within the specified range.

        Handles the midnight-wrapping case where start > end.

        Args:
            now: Current datetime.
            start: Start hour (inclusive).
            end: End hour (exclusive or inclusive based on end_inclusive).
            end_inclusive: If True, end is inclusive; if False, exclusive.

        Returns:
            True if now.hour is in range, False otherwise.
        """
        ...

    def _seconds_until_boundary(
        self,
        now: datetime,
        last: datetime | None,
        interval: timedelta,
    ) -> float:
        """Calculate seconds until an interval boundary is reached.

        If `last` is None, returns 0.0 (immediately due).
        If interval has elapsed, returns 0.0.
        Otherwise, returns seconds remaining.

        Args:
            now: Current datetime.
            last: Datetime of last action, or None if never.
            interval: The minimum time between actions.

        Returns:
            Non-negative seconds until next boundary.
        """
        ...

    def _seconds_until_hour(
        self,
        now: datetime,
        target_hour: int,
    ) -> float:
        """Calculate seconds until the next occurrence of target_hour.

        If target_hour == now.hour, assumes the NEXT day's occurrence.
        Handles midnight wrapping correctly.

        Args:
            now: Current datetime.
            target_hour: Target hour in 24-hour format (0-23).

        Returns:
            Seconds until the next time the clock shows target_hour.
        """
        ...
