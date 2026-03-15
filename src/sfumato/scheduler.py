"""Scheduler for the sfumato daemon's dual-timer architecture.

Implements the contract from ARCHITECTURE.md#2.11:
- Action enum for scheduling decisions
- SchedulerState for tracking last action timestamps
- Scheduler class enforcing quiet_hours/active_hours and interval logic
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

        2. WITHIN QUIET HOURS (and outside active_hours) → QUIET_ART
           Display pure painting without news overlay. TV may still be
           in Art mode during these hours.

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
        self._config = config
        self._news_interval = timedelta(hours=config.news_interval_hours)
        self._rotate_interval = timedelta(minutes=config.rotate_interval_minutes)
        self._quiet_hours = config.quiet_hours
        self._active_hours = config.active_hours

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
        # Step 1: Check active hours
        if not self.is_active_hour(now):
            # Outside active hours: check if in quiet hours
            if self.is_quiet_hour(now):
                return Action.QUIET_ART
            else:
                return Action.IDLE

        # Step 2: In active hours - check news interval
        # Note: is_quiet_hour should be False here if ranges don't overlap
        # but if they do overlap (e.g., quiet=(0,23), active=(0,23)),
        # QUIET_ART takes precedence via the first check
        # However, since we passed is_active_hour, we're in normal operation

        # Actually, re-check: if in active AND quiet hours, prioritize quiet
        # This handles the edge case where ranges overlap
        if self.is_quiet_hour(now):
            return Action.QUIET_ART

        # Step 3: Check news interval
        news_due = self._is_interval_due(
            now, state.last_news_refresh, self._news_interval
        )
        if news_due:
            # News refresh triggers both REFRESH_NEWS and ROTATE
            return Action.REFRESH_NEWS | Action.ROTATE

        # Step 4: Check rotation interval
        rotation_due = self._is_interval_due(
            now, state.last_rotation, self._rotate_interval
        )
        if rotation_due:
            return Action.ROTATE

        # Step 5: Default - no action needed
        # Note: BACKFILL is deferred until pool status is available
        return Action.NONE

    def seconds_until_next_action(self, now: datetime, state: SchedulerState) -> float:
        """Calculate seconds until the next action is due.

        Used by the daemon's main loop to sleep efficiently. Returns the
        minimum time until any of the following:
        - News interval boundary
        - Rotate interval boundary
        - Transition into active_hours (if currently outside)

        Calculation Strategy:

        1. Compute time until all possible state changes:
           - Time until news is due
           - Time until rotation is due
           - Time until active_hours starts (if outside active)
           - Time until quiet_hours ends (if currently in quiet)

        2. Return the minimum of all applicable wait times.

        Edge Cases:
           - If any action is due NOW (e.g., what_to_do returns non-NONE),
             this should return 0.0.
           - If last_rotation is None, rotate is due immediately → return 0.0.
           - If last_news_refresh is None, news refresh is due → return 0.0.

        Args:
            now: Current datetime.
            state: SchedulerState with last action timestamps.

        Returns:
            Floating-point seconds until the next action. Returns 0.0
            if an action is already due. Guarantees non-negative result.
        """
        # Check if we need to act now (any action other than NONE, IDLE, or QUIET_ART)
        action = self.what_to_do(now, state)
        if action not in (Action.NONE, Action.IDLE, Action.QUIET_ART):
            return 0.0

        # Collect all relevant wait times
        wait_times: list[float] = []

        # Always compute time until active hours start (relevant when outside)
        if not self.is_active_hour(now):
            # Outside active hours - wait until active starts
            wait_times.append(self._seconds_until_hour(now, self._active_hours[0]))

            # If in quiet hours, also compute time until quiet ends
            if self.is_quiet_hour(now):
                # Time until end of quiet hours
                wait_times.append(self._seconds_until_hour(now, self._quiet_hours[1]))
        else:
            # In active hours (and not in quiet): compute wait times for intervals

            # News interval wait time
            if state.last_news_refresh is not None:
                elapsed_news = now - state.last_news_refresh
                remaining_news = self._news_interval - elapsed_news
                if remaining_news.total_seconds() > 0:
                    wait_times.append(remaining_news.total_seconds())
                else:
                    # Interval elapsed - should have been caught by what_to_do
                    # But handle it here for safety
                    return 0.0
            else:
                # None means due now
                return 0.0

            # Rotation interval wait time
            if state.last_rotation is not None:
                elapsed_rotation = now - state.last_rotation
                remaining_rotation = self._rotate_interval - elapsed_rotation
                if remaining_rotation.total_seconds() > 0:
                    wait_times.append(remaining_rotation.total_seconds())
                else:
                    return 0.0
            else:
                # None means due now
                return 0.0

        # Return minimum wait time, with a floor of 0
        if not wait_times:
            # This shouldn't happen in normal operation
            return 0.0

        return max(0.0, min(wait_times))

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
           - Quiet hours of (0, 0) or (x, x) is an empty interval → always False.
           - Hour boundary: is_quiet_hour(5:59:59) should return True for (0, 6).
           - Hour boundary: is_quiet_hour(6:00:00) should return False for (0, 6).

        Args:
            now: Current datetime. Uses the hour component only.

        Returns:
            True if the hour falls within [start, end), False otherwise.
        """
        return self._is_time_in_range(
            now, self._quiet_hours[0], self._quiet_hours[1], end_inclusive=False
        )

    def is_active_hour(self, now: datetime) -> bool:
        """Check if the given time falls within active_hours.

        Active hours are when the daemon operates at full capacity:
        news refreshes, painting rotations, and backfill operations.
        Outside active hours, the daemon may be IDLE or in QUIET_ART mode.

        Interval Semantics:
           - active_hours[0] is the start hour (inclusive)
           - active_hours[1] is the end hour (exclusive)
           - Hour values are in 24-hour format (0-23)
           - Same semantics as quiet_hours (both use exclusive end)
           - Example: (7, 23) means hours 7, 8, ..., 21, 22 are active
           - Example: (10, 2) means 10, 11, ..., 23, 0, 1 are active (wraps midnight)
           - Example: (0, 24) means all day active

        Edge Cases:
           - Crossing midnight: active_hours=(10, 2) means 10:00-1:59.
           - Active hours of (0, 0) or (x, x) is empty → always False.

        Args:
            now: Current datetime. Uses the hour component only.

        Returns:
            True if the hour falls within [start, end), False otherwise.
        """
        return self._is_time_in_range(
            now, self._active_hours[0], self._active_hours[1], end_inclusive=False
        )

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
        hour = now.hour

        # Handle same hour range (e.g., (0, 0) or (5, 5)) - empty interval
        if start == end:
            if end_inclusive:
                # (x, x) inclusive means just that one hour
                return hour == start
            else:
                # (x, x) exclusive means empty range
                return False

        # Handle midnight wrapping (e.g., (22, 6) means 22-23 and 0-5)
        if start > end:
            # Start is after end: wraps around midnight
            # Hour is in range if it's >= start (evening) OR < end (morning)
            # For end_inclusive: <= end instead of < end
            if end_inclusive:
                return hour >= start or hour <= end
            else:
                return hour >= start or hour < end
        else:
            # Normal range (start < end)
            # Hour is in range if it's >= start and (< or <= end)
            if end_inclusive:
                return start <= hour <= end
            else:
                return start <= hour < end

    def _is_interval_due(
        self,
        now: datetime,
        last: datetime | None,
        interval: timedelta,
    ) -> bool:
        """Check if an interval has elapsed since the last action.

        Args:
            now: Current datetime.
            last: Datetime of last action, or None if never.
            interval: Minimum time between actions.

        Returns:
            True if interval has elapsed (or last is None), False otherwise.
        """
        if last is None:
            return True
        elapsed = now - last
        return elapsed >= interval

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
        current_hour = now.hour
        current_minute = now.minute
        current_second = now.second
        current_microsecond = now.microsecond

        # Seconds until the start of the next hour
        seconds_into_current_hour = (
            current_minute * 60 + current_second + current_microsecond / 1_000_000
        )

        if target_hour > current_hour:
            # Target is today, later
            hours_until = target_hour - current_hour
            return hours_until * 3600 - seconds_into_current_hour
        elif target_hour < current_hour:
            # Target is tomorrow
            hours_until = 24 - current_hour + target_hour
            return hours_until * 3600 - seconds_into_current_hour
        else:
            # target_hour == current_hour, need to wait until tomorrow
            return 24 * 3600 - seconds_into_current_hour
