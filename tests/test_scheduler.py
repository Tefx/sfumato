"""Contract tests for scheduler module.

Implements test stubs that will be filled with verification logic
once the scheduler implementation is complete. Tests cover:

- Action enum and flag behavior
- Time window logic (active_hours)
- Interval calculations
- Edge cases: midnight wrapping, None timestamps, first run
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Flag

import pytest

from sfumato.config import ScheduleConfig
from sfumato.scheduler import Action, Scheduler, SchedulerState


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> ScheduleConfig:
    """Return ScheduleConfig with default values."""
    return ScheduleConfig()


@pytest.fixture
def custom_config() -> ScheduleConfig:
    """Return ScheduleConfig with custom values for edge case testing."""
    return ScheduleConfig(
        news_interval_hours=4,
        rotate_interval_minutes=10,
        active_hours=(7, 23),
    )


@pytest.fixture
def all_day_active_config() -> ScheduleConfig:
    """Return ScheduleConfig with 24-hour active hours."""
    return ScheduleConfig(
        news_interval_hours=6,
        rotate_interval_minutes=15,
        active_hours=(0, 23),  # All day
    )


@pytest.fixture
def fresh_state() -> SchedulerState:
    """Return a SchedulerState with all None timestamps (first run)."""
    return SchedulerState(
        last_news_refresh=None,
        last_rotation=None,
        last_backfill=None,
    )


@pytest.fixture
def recent_state() -> SchedulerState:
    """Return a SchedulerState with recent action timestamps."""
    now = datetime.now()
    return SchedulerState(
        last_news_refresh=now - timedelta(hours=3),
        last_rotation=now - timedelta(minutes=5),
        last_backfill=now - timedelta(hours=12),
    )


@pytest.fixture
def overdue_state() -> SchedulerState:
    """Return a SchedulerState with overdue action timestamps."""
    now = datetime.now()
    return SchedulerState(
        last_news_refresh=now - timedelta(hours=10),  # Overdue for 6h interval
        last_rotation=now - timedelta(minutes=30),  # Overdue for 15min interval
        last_backfill=now - timedelta(days=7),
    )


# ---------------------------------------------------------------------------
# Action Enum Tests
# ---------------------------------------------------------------------------


class TestActionEnum:
    """Tests for Action flag enum behavior."""

    def test_action_none_is_zero(self) -> None:
        """Action.NONE should have value 0."""
        assert Action.NONE.value == 0

    def test_action_none_is_falsy(self) -> None:
        """Action.NONE should be falsy in boolean context."""
        assert not Action.NONE
        assert not bool(Action.NONE)

    def test_action_single_flags_are_truthy(self) -> None:
        """Single action flags should be truthy."""
        assert Action.REFRESH_NEWS
        assert Action.ROTATE
        assert Action.BACKFILL
        assert Action.QUIET_ART

    def test_action_can_combine_with_or(self) -> None:
        """Actions should combine with bitwise OR."""
        combined = Action.REFRESH_NEWS | Action.ROTATE
        assert Action.REFRESH_NEWS in combined
        assert Action.ROTATE in combined

    def test_action_combined_is_truthy(self) -> None:
        """Combined actions should be truthy."""
        combined = Action.REFRESH_NEWS | Action.ROTATE
        assert combined

    def test_action_none_and_combined(self) -> None:
        """NONE combined with another action should equal the other action."""
        combined = Action.NONE | Action.ROTATE
        assert combined == Action.ROTATE

    def test_action_can_check_membership(self) -> None:
        """Should be able to check if an action is contained in a combined flag."""
        combined = Action.REFRESH_NEWS | Action.ROTATE | Action.BACKFILL
        assert Action.REFRESH_NEWS in combined
        assert Action.ROTATE in combined
        assert Action.BACKFILL in combined
        assert Action.QUIET_ART not in combined

    def test_action_unique_bit_values(self) -> None:
        """Each action should have a unique bit position."""
        values = [
            Action.NONE.value,
            Action.REFRESH_NEWS.value,
            Action.ROTATE.value,
            Action.BACKFILL.value,
            Action.QUIET_ART.value,
        ]
        # All non-NONE values should be distinct powers of 2
        non_zero = [v for v in values if v != 0]
        assert len(set(non_zero)) == len(non_zero), (
            "Actions have overlapping bit values"
        )


# ---------------------------------------------------------------------------
# SchedulerState Tests
# ---------------------------------------------------------------------------


class TestSchedulerState:
    """Tests for SchedulerState dataclass."""

    def test_state_all_none(self, fresh_state: SchedulerState) -> None:
        """Fresh state should have all None timestamps."""
        assert fresh_state.last_news_refresh is None
        assert fresh_state.last_rotation is None
        assert fresh_state.last_backfill is None

    def test_state_with_timestamps(self, recent_state: SchedulerState) -> None:
        """State with timestamps should store them correctly."""
        assert recent_state.last_news_refresh is not None
        assert recent_state.last_rotation is not None
        assert recent_state.last_backfill is not None


# ---------------------------------------------------------------------------
# Scheduler Initialization Tests
# ---------------------------------------------------------------------------


class TestSchedulerInit:
    """Tests for Scheduler initialization."""

    def test_scheduler_init_with_default_config(
        self, default_config: ScheduleConfig
    ) -> None:
        """Scheduler should initialize with default config."""
        # Contract: Should accept ScheduleConfig and store it.
        scheduler = Scheduler(default_config)
        assert scheduler is not None  # Placeholder; implementation needed

    def test_scheduler_init_with_custom_config(
        self, custom_config: ScheduleConfig
    ) -> None:
        """Scheduler should initialize with custom config."""
        # Contract: Should handle custom intervals and time windows.
        scheduler = Scheduler(custom_config)
        assert scheduler is not None


# ---------------------------------------------------------------------------
# is_active_hour Tests
# ---------------------------------------------------------------------------


class TestIsActiveHour:
    """Tests for is_active_hour time window logic."""

    def test_active_hour_default_range(self, default_config: ScheduleConfig) -> None:
        """Default active_hours=(10, 2) should cover hours 10-1 (wraps midnight)."""
        scheduler = Scheduler(default_config)

        # Hours within active period (10-1 wrapping midnight)
        for hour in [10, 11, 12, 15, 22, 23, 0, 1]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Hours outside active period (2-9)
        for hour in [2, 3, 4, 5, 6, 7, 8, 9]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_active_hour(now), (
                f"Hour {hour} should NOT be active"
            )

    def test_active_hour_boundary_start(self, default_config: ScheduleConfig) -> None:
        """At the start of active_hours boundary (inclusive)."""
        scheduler = Scheduler(default_config)

        # Just before active_hours (9:59:59)
        before_start = datetime(2024, 1, 15, 9, 59, 59)
        assert not scheduler.is_active_hour(before_start)

        # At start of active_hours (10:00)
        at_start = datetime(2024, 1, 15, 10, 0, 0)
        assert scheduler.is_active_hour(at_start)

    def test_active_hour_boundary_end(self, default_config: ScheduleConfig) -> None:
        """At the end of active_hours boundary (exclusive at hour 2)."""
        scheduler = Scheduler(default_config)

        # At hour 1 (still active, wraps midnight)
        at_1 = datetime(2024, 1, 16, 1, 30, 0)
        assert scheduler.is_active_hour(at_1)

        # At hour 2 (exclusive end, no longer active)
        at_end = datetime(2024, 1, 16, 2, 0, 0)
        assert not scheduler.is_active_hour(at_end)

    def test_active_hour_all_day(self, all_day_active_config: ScheduleConfig) -> None:
        """active_hours=(0, 23) should be active hours 0-22 (exclusive end)."""
        scheduler = Scheduler(all_day_active_config)

        # Hours 0-22 are active (end is exclusive)
        for hour in range(23):
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Hour 23 is NOT active (exclusive end)
        now = datetime(2024, 1, 15, 23, 30, 0)
        assert not scheduler.is_active_hour(now), "Hour 23 should NOT be active (exclusive end)"


# ---------------------------------------------------------------------------
# what_to_do Tests
# ---------------------------------------------------------------------------


class TestWhatToDo:
    """Tests for the what_to_do scheduling decision function."""

    def test_what_to_do_outside_active_returns_quiet_art(
        self, default_config: ScheduleConfig, recent_state: SchedulerState
    ) -> None:
        """Outside active_hours should return QUIET_ART (TV always shows something)."""
        scheduler = Scheduler(default_config)

        # Hour 5 is outside active_hours (10-2)
        now = datetime(2024, 1, 15, 5, 30, 0)
        action = scheduler.what_to_do(now, recent_state)
        assert action == Action.QUIET_ART

    def test_what_to_do_outside_active_morning(
        self, default_config: ScheduleConfig, recent_state: SchedulerState
    ) -> None:
        """Morning hours outside active_hours should return QUIET_ART."""
        scheduler = Scheduler(default_config)

        # Hour 8 is outside active_hours (10-2)
        now = datetime(2024, 1, 15, 8, 30, 0)
        action = scheduler.what_to_do(now, recent_state)
        assert action == Action.QUIET_ART

    def test_what_to_do_first_run(
        self, default_config: ScheduleConfig, fresh_state: SchedulerState
    ) -> None:
        """First run (all None timestamps) in active_hours should return REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)

        # Hour 12 is within active_hours
        now = datetime(2024, 1, 15, 12, 0, 0)
        action = scheduler.what_to_do(now, fresh_state)

        # First run: need news refresh (None timestamp)
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action

    def test_what_to_do_overdue_news(self, default_config: ScheduleConfig) -> None:
        """Overdue news refresh should return REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)

        # Hour 12 is within active_hours, state has 10h old news refresh
        now = datetime(2024, 1, 15, 12, 0, 0)
        # Create state with timestamps relative to test's 'now'
        overdue_state = SchedulerState(
            last_news_refresh=now
            - timedelta(hours=10),  # 10h ago = overdue for 6h interval
            last_rotation=now
            - timedelta(minutes=30),  # 30 min ago = overdue for 15min interval
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, overdue_state)

        # news_interval_hours=6 (default), last_news_refresh=10h ago
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action

    def test_what_to_do_rotation_due(
        self, default_config: ScheduleConfig, recent_state: SchedulerState
    ) -> None:
        """Rotation interval elapsed should return ROTATE."""
        scheduler = Scheduler(default_config)

        # Hour 12 is within active_hours, recent_state has rotation 5 min ago
        # But we need a state where rotation is overdue (15+ minutes)
        now = datetime(2024, 1, 15, 12, 0, 0)
        rotation_due_state = SchedulerState(
            last_news_refresh=now - timedelta(hours=3),
            last_rotation=now - timedelta(minutes=20),  # 20 min > 15 min interval
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, rotation_due_state)

        # Rotation is overdue
        assert Action.ROTATE in action
        # News is not overdue (3h < 6h)
        assert Action.REFRESH_NEWS not in action

    def test_what_to_do_nothing_due(self, default_config: ScheduleConfig) -> None:
        """Within intervals, should return NONE."""
        scheduler = Scheduler(default_config)

        # Hour 12 is within active_hours
        now = datetime(2024, 1, 15, 12, 0, 0)
        # State with very recent actions
        up_to_date_state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=2),
            last_backfill=now - timedelta(hours=1),
        )
        action = scheduler.what_to_do(now, up_to_date_state)

        assert action == Action.NONE


# ---------------------------------------------------------------------------
# seconds_until_next_action Tests
# ---------------------------------------------------------------------------


class TestSecondsUntilNextAction:
    """Tests for the sleep calculation function."""

    def test_seconds_until_rotation_due(self, default_config: ScheduleConfig) -> None:
        """If rotation is due now, should return 0.0."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation was 20 minutes ago (overdue for 15-minute interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=3),
            last_rotation=now - timedelta(minutes=20),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        assert seconds == 0.0

    def test_seconds_until_first_run(
        self, default_config: ScheduleConfig, fresh_state: SchedulerState
    ) -> None:
        """First run (None timestamps) should return 0.0."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        seconds = scheduler.seconds_until_next_action(now, fresh_state)
        assert seconds == 0.0

    def test_seconds_until_future_rotation(
        self, default_config: ScheduleConfig
    ) -> None:
        """Should compute seconds until next rotation boundary."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation was 2 minutes ago (3 minutes remaining for 5-minute interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=2),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return roughly 3 minutes (180 seconds)
        assert 170 <= seconds <= 190  # Allow small tolerance

    def test_seconds_until_active_start(self) -> None:
        """Outside active hours, should compute time until active start."""
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            active_hours=(10, 2),
        )
        scheduler = Scheduler(config)

        # Hour 5: outside active_hours (10-2)
        # Time until active_hours start (10:00) = 5 hours
        now = datetime(2024, 1, 15, 5, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return time until hour 10 (~5 hours = 18000 seconds)
        assert 17990 <= seconds <= 18010

    def test_seconds_never_negative(self, default_config: ScheduleConfig) -> None:
        """Should never return negative seconds."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=100),  # Way overdue
            last_rotation=now - timedelta(minutes=1000),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        assert seconds >= 0.0


# ---------------------------------------------------------------------------
# Edge Cases and Integration Tests
# ---------------------------------------------------------------------------


class TestSchedulerEdgeCases:
    """Tests for edge cases and complex scenarios."""

    def test_midnight_wrapping_in_active_hours(self) -> None:
        """active_hours crossing midnight should work correctly."""
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            active_hours=(22, 6),
        )
        scheduler = Scheduler(config)

        # Test evening hours (active)
        for hour in [22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Test early morning hours (active)
        for hour in [0, 1, 2, 3, 4, 5]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Test daytime hours (not active -> QUIET_ART)
        for hour in [6, 7, 12, 15, 20, 21]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_active_hour(now), f"Hour {hour} should NOT be active"

    def test_transition_inactive_to_active(self) -> None:
        """Transition from outside active_hours to inside should work."""
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            active_hours=(7, 23),
        )
        scheduler = Scheduler(config)

        # Hour 6: outside active (QUIET_ART)
        now_6 = datetime(2024, 1, 15, 6, 30, 0)
        assert scheduler.is_active_hour(now_6) is False

        # Hour 7: inside active
        now_7 = datetime(2024, 1, 15, 7, 30, 0)
        assert scheduler.is_active_hour(now_7) is True

    def test_outside_active_always_quiet_art(
        self, default_config: ScheduleConfig
    ) -> None:
        """Outside active hours, what_to_do always returns QUIET_ART, never IDLE."""
        scheduler = Scheduler(default_config)
        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )

        # Hours outside active_hours (10-2) should all be QUIET_ART
        for hour in [2, 3, 4, 5, 6, 7, 8, 9]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            action = scheduler.what_to_do(now, state)
            assert action == Action.QUIET_ART, f"Hour {hour} should be QUIET_ART"


class TestSchedulerContractCompliance:
    """Tests to verify contract requirements from ARCHITECTURE.md#2.11."""

    def test_action_flag_semantics(self) -> None:
        """Verify Action follows Flag semantics for bitwise operations."""
        # Contract: Action is a Flag, allowing combination with |
        combined = Action.REFRESH_NEWS | Action.ROTATE
        assert isinstance(combined, Action)
        assert isinstance(combined, Flag)

    def test_scheduler_has_required_methods(
        self, default_config: ScheduleConfig
    ) -> None:
        """Verify Scheduler implements all required methods."""
        scheduler = Scheduler(default_config)
        assert hasattr(scheduler, "what_to_do")
        assert hasattr(scheduler, "seconds_until_next_action")
        assert hasattr(scheduler, "is_active_hour")
        assert callable(scheduler.what_to_do)
        assert callable(scheduler.seconds_until_next_action)
        assert callable(scheduler.is_active_hour)

    def test_scheduler_state_dataclass(self) -> None:
        """Verify SchedulerState is a proper dataclass."""
        from dataclasses import fields

        field_names = {f.name for f in fields(SchedulerState)}
        assert "last_news_refresh" in field_names
        assert "last_rotation" in field_names
        assert "last_backfill" in field_names

    def test_default_config_values(self) -> None:
        """Verify default ScheduleConfig values."""
        config = ScheduleConfig()
        assert config.news_interval_hours == 6
        assert config.rotate_interval_minutes == 5
        assert config.active_hours == (10, 2)


# ---------------------------------------------------------------------------
# Dual-Timer Decision Tests (news + rotate interaction)
# ---------------------------------------------------------------------------


class TestDualTimerDecisions:
    """Tests for dual-timer architecture: news_interval vs rotate_interval."""

    def test_refresh_news_triggers_combined_action(
        self, default_config: ScheduleConfig
    ) -> None:
        """When news refresh is due, should return REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News refresh is overdue (7 hours > 6 hour interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=7),
            last_rotation=now - timedelta(hours=1),  # Recent rotation
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # Both should be triggered when news is due
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action

    def test_rotation_only_when_news_not_due(
        self, default_config: ScheduleConfig
    ) -> None:
        """When only rotation is due, should return ROTATE alone."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News was recent (2 hours < 6 hour interval)
        # Rotation is due (20 min > 15 min interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=2),
            last_rotation=now - timedelta(minutes=20),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.ROTATE in action
        assert Action.REFRESH_NEWS not in action

    def test_news_interval_boundary_exact(self, default_config: ScheduleConfig) -> None:
        """At exact news interval boundary, should trigger refresh."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Exactly 6 hours since last news refresh (boundary case)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=6),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # At boundary, should trigger (>= means due)
        assert Action.REFRESH_NEWS in action

    def test_rotation_interval_boundary_exact(
        self, default_config: ScheduleConfig
    ) -> None:
        """At exact rotation interval boundary, should trigger rotation."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Exactly 15 minutes since last rotation (boundary case)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=15),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # At boundary, should trigger (>= means due)
        assert Action.ROTATE in action

    def test_both_intervals_due_simultaneously(
        self, default_config: ScheduleConfig
    ) -> None:
        """When both intervals align, should return combined action."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Both news (6h) and rotation (15min) are overdue
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=6, minutes=30),
            last_rotation=now - timedelta(minutes=30),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action
        # Verify it's a combined action, not just one
        assert action != Action.REFRESH_NEWS
        assert action != Action.ROTATE

    def test_news_interval_elapsed_rotation_not(
        self, default_config: ScheduleConfig
    ) -> None:
        """News overdue but rotation recent: still combine REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News 7h ago (overdue), rotation 5min ago (not due)
        # Contract says REFRESH_NEWS triggers ROTATE alongside
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=7),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action  # Combined with news refresh

    def test_first_rotation_of_day_after_quiet(
        self, default_config: ScheduleConfig
    ) -> None:
        """First action of day after quiet art period triggers immediate refresh."""
        scheduler = Scheduler(default_config)
        # 10:00 AM - first active hour with default active_hours=(10, 2)
        now = datetime(2024, 1, 15, 10, 0, 0)

        # Last actions were during previous day's active hours
        state = SchedulerState(
            last_news_refresh=datetime(2024, 1, 14, 20, 0, 0),  # 14h overnight gap
            last_rotation=datetime(2024, 1, 14, 23, 0, 0),  # 11h overnight gap
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # News is overdue (14h > 6h), rotation is overdue (11h > any interval)
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action


# ---------------------------------------------------------------------------
# Backfill Flag Condition Tests
# ---------------------------------------------------------------------------


class TestBackfillConditions:
    """Tests for BACKFILL action triggering conditions.

    BACKFILL should only trigger when:
    - Pool has room for more paintings
    - No heavy action (REFRESH_NEWS or ROTATE) is scheduled
    - System is within active_hours

    Note: Current contract stub does NOT include pool status in SchedulerState.
    Implementation may need additional context or method signature extension.
    These tests define the expected behavior.
    """

    def test_backfill_not_triggered_when_news_due(
        self, default_config: ScheduleConfig
    ) -> None:
        """BACKFILL should not appear when news refresh is due."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News overdue, rotation fine
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=8),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=now - timedelta(hours=24),
        )
        action = scheduler.what_to_do(now, state)

        # No backfill when heavy work pending
        assert Action.BACKFILL not in action

    def test_backfill_not_triggered_when_rotation_due(
        self, default_config: ScheduleConfig
    ) -> None:
        """BACKFILL should not appear when rotation is due."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation overdue, news recent
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=2),
            last_rotation=now - timedelta(minutes=20),
            last_backfill=now - timedelta(hours=12),
        )
        action = scheduler.what_to_do(now, state)

        # Rotation takes priority over backfill
        assert Action.ROTATE in action
        assert Action.BACKFILL not in action

    def test_backfill_not_possible_outside_active_hours(
        self, default_config: ScheduleConfig
    ) -> None:
        """BACKFILL should not run outside active hours."""
        scheduler = Scheduler(default_config)
        # Hour 5 is outside active_hours (10-2)
        now = datetime(2024, 1, 15, 5, 30, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=12),
            last_rotation=now - timedelta(hours=8),
            last_backfill=now - timedelta(hours=48),
        )
        action = scheduler.what_to_do(now, state)

        # Outside active hours -> QUIET_ART
        assert action == Action.QUIET_ART
        assert Action.BACKFILL not in action

    def test_no_heavy_work_but_no_backfill_yet(
        self, default_config: ScheduleConfig
    ) -> None:
        """When no heavy work due, should check backfill conditions.

        Note: Contract says backfill requires pool status which is not
        in SchedulerState. For this test, we define expected behavior
        when backfill COULD run but pool status is unknown.
        """
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # All actions recent - no heavy work due
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=2),  # Well within 5-min interval
            last_backfill=now - timedelta(hours=2),
        )
        action = scheduler.what_to_do(now, state)

        # Without pool status, should return NONE
        # (Backfill decision deferred to implementation with pool context)
        assert action == Action.NONE


# ---------------------------------------------------------------------------
# Seconds Until Next Action Boundary Tests
# ---------------------------------------------------------------------------


class TestSecondsUntilNextActionBoundaries:
    """Tests for precise seconds_until_next_action calculations.

    These tests focus on boundary conditions and edge cases that
    are most likely to cause real-world bugs.
    """

    def test_seconds_at_59_minute_boundary(
        self, default_config: ScheduleConfig
    ) -> None:
        """Handle rotation near minute boundary correctly."""
        scheduler = Scheduler(default_config)
        # 11:59:30 - rotation due at 12:00 boundary
        now = datetime(2024, 1, 15, 11, 59, 30)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=4, seconds=30),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Rotation due in 30 seconds (interval is 5 min)
        assert 25 <= seconds <= 35  # Allow some tolerance

    def test_seconds_at_hour_boundary(self, default_config: ScheduleConfig) -> None:
        """Handle transition across hour boundary."""
        scheduler = Scheduler(default_config)
        # 7:59:45 - approaching active_hours start at 8:00
        # But default config active_hours starts at 7, so we're already active
        # Use custom config
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            active_hours=(8, 22),  # Active starts at 8
        )
        scheduler = Scheduler(config)
        now = datetime(2024, 1, 15, 7, 59, 45)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return ~15 seconds until active_hours start
        assert 10 <= seconds <= 20

    def test_seconds_midnight_crossing(self) -> None:
        """Handle midnight crossing in time calculations."""
        # 23:30 -> next active start at 7:00 next day
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            active_hours=(7, 21),
        )
        scheduler = Scheduler(config)
        now = datetime(2024, 1, 15, 23, 30, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should compute time until 7:00 next day = 7.5 hours = 27000 seconds
        assert seconds > 0

    def test_seconds_rotation_due_after_just_completed(
        self, default_config: ScheduleConfig
    ) -> None:
        """Just after a rotation, compute remaining time correctly."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation just completed at 11:59 (1 minute ago)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=2),
            last_rotation=now - timedelta(minutes=1),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # ~4 minutes remaining (5 min interval - 1 min elapsed)
        assert 230 <= seconds <= 250

    def test_seconds_news_rotation_race(self, default_config: ScheduleConfig) -> None:
        """When both intervals matter, return minimum time."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News due in 4 hours, rotation due in 2 minutes
        # Should return 2 minutes (120 seconds)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=2),  # Due in 4 hours
            last_rotation=now - timedelta(minutes=3),  # Due in 2 min
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Rotation wins - ~120 seconds
        assert 110 <= seconds <= 130

    def test_seconds_with_state_timestamps_in_future(
        self, default_config: ScheduleConfig
    ) -> None:
        """Handle state timestamps in the future (clock skew/correction)."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # State has future timestamps (clock was ahead, now corrected)
        state = SchedulerState(
            last_news_refresh=now + timedelta(hours=1),  # Future!
            last_rotation=now + timedelta(minutes=5),  # Future!
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Future timestamps mean rotation/news are "not due"
        # Should compute remaining time from now
        # Behavior: treat as very recent -> return full interval
        # Implementation choice: clamp or use max(0, elapsed)
        assert seconds >= 0

    def test_seconds_exact_interval_minus_one_second(
        self, default_config: ScheduleConfig
    ) -> None:
        """One second before interval boundary returns 1.0."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation was exactly 4:59 ago (5 min interval - 1 second)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=3),
            last_rotation=now - timedelta(minutes=4, seconds=59),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # ~1 second remaining
        assert 0.5 <= seconds <= 1.5

    def test_seconds_multiple_intervals_distant_future(self) -> None:
        """When next action is hours away, compute correctly."""
        config = ScheduleConfig(
            news_interval_hours=24,  # Very long interval
            rotate_interval_minutes=60,  # 1 hour interval
            active_hours=(7, 21),
        )
        scheduler = Scheduler(config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # All recent
        state = SchedulerState(
            last_news_refresh=now - timedelta(minutes=30),
            last_rotation=now - timedelta(minutes=30),
            last_backfill=now - timedelta(minutes=30),
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Rotation due in ~30 minutes
        assert 1700 <= seconds <= 1900


# ---------------------------------------------------------------------------
# Active Hour Boundary and Edge Cases
# ---------------------------------------------------------------------------


class TestActiveHourBoundaries:
    """Tests for boundary conditions in active_hours time window logic."""

    def test_exact_hour_start_is_active(self) -> None:
        """At exactly active_hours[0], should be active."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        # Exactly 7:00:00
        now = datetime(2024, 1, 15, 7, 0, 0)
        assert scheduler.is_active_hour(now)

    def test_exact_hour_end_is_not_active(self) -> None:
        """At exactly active_hours[1], should NOT be active (exclusive end)."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        # Exactly 23:00:00 - end is exclusive, so hour 23 is NOT active
        now = datetime(2024, 1, 15, 23, 0, 0)
        assert not scheduler.is_active_hour(now)

        # Hour 22 is the last active hour
        now_last = datetime(2024, 1, 15, 22, 59, 59)
        assert scheduler.is_active_hour(now_last)

    def test_midnight_is_not_active_with_non_wrapping_config(self) -> None:
        """Hour 0 (midnight) is not active with active_hours=(7, 23)."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        now = datetime(2024, 1, 15, 0, 0, 0)
        assert not scheduler.is_active_hour(now)

        now_with_minutes = datetime(2024, 1, 15, 0, 30, 0)
        assert not scheduler.is_active_hour(now_with_minutes)

    def test_active_hour_all_hours(self) -> None:
        """active_hours=(0, 0) with same start/end is empty. Use wrapping for all day."""
        # (0, 23) with exclusive end means hours 0-22 are active
        config = ScheduleConfig(active_hours=(0, 23))
        scheduler = Scheduler(config)

        for hour in range(23):
            now = datetime(2024, 1, 15, hour, 0, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Hour 23 is NOT active (exclusive end)
        now = datetime(2024, 1, 15, 23, 0, 0)
        assert not scheduler.is_active_hour(now)


# ---------------------------------------------------------------------------
# None Timestamp Edge Cases (First Run Behavior)
# ---------------------------------------------------------------------------


class TestNoneTimestampEdgeCases:
    """Tests for handling None timestamps (first run, no history)."""

    def test_first_run_all_none_returns_actions(
        self, default_config: ScheduleConfig
    ) -> None:
        """First run with all None timestamps should trigger actions."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)  # Active hour

        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # First run: need both news refresh and rotation
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action

    def test_first_run_outside_active_hours(
        self, default_config: ScheduleConfig
    ) -> None:
        """First run outside active hours returns QUIET_ART."""
        scheduler = Scheduler(default_config)
        # Hour 5 is outside active (10-2)
        now = datetime(2024, 1, 15, 5, 0, 0)

        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # Outside active hours -> QUIET_ART (TV always shows something)
        assert action == Action.QUIET_ART

    def test_partial_none_rotation_only(self, default_config: ScheduleConfig) -> None:
        """Only last_rotation is None: should trigger ROTATE."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),  # Recent
            last_rotation=None,  # Never rotated
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.ROTATE in action
        # News was recent, shouldn't trigger
        assert Action.REFRESH_NEWS not in action

    def test_partial_none_news_only(self, default_config: ScheduleConfig) -> None:
        """Only last_news_refresh is None: should trigger REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        state = SchedulerState(
            last_news_refresh=None,  # Never refreshed
            last_rotation=now - timedelta(minutes=5),  # Recent
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # None for news_refresh means immediate refresh needed
        # REFRESH_NEWS always pairs with ROTATE per contract
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action


# ---------------------------------------------------------------------------
# Precedence and Priority Tests
# ---------------------------------------------------------------------------


class TestActionPrecedence:
    """Tests for action precedence in what_to_do decisions."""

    def test_outside_active_always_quiet_art_precedence(self) -> None:
        """Outside active hours always returns QUIET_ART."""
        config = ScheduleConfig(
            active_hours=(8, 22),
        )
        scheduler = Scheduler(config)

        # Hour 7: outside active (8-22) -> QUIET_ART
        now = datetime(2024, 1, 15, 7, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=10),
            last_rotation=now - timedelta(hours=2),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert action == Action.QUIET_ART

    def test_heavy_work_takes_priority_over_backfill(self) -> None:
        """When both heavy work and backfill are due, heavy work wins."""
        config = ScheduleConfig()
        scheduler = Scheduler(config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Both news and rotation overdue
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=10),
            last_rotation=now - timedelta(hours=1),
            last_backfill=now - timedelta(days=7),
        )
        action = scheduler.what_to_do(now, state)

        # Heavy work takes priority
        assert Action.REFRESH_NEWS in action
        assert Action.ROTATE in action
        # Backfill is NOT in action (implementation note: pool status unknown)
        assert Action.BACKFILL not in action

    def test_none_result_when_nothing_due(self, default_config: ScheduleConfig) -> None:
        """When nothing is due, return NONE (not IDLE or empty)."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),  # 5h till due
            last_rotation=now - timedelta(minutes=2),  # 3min till due
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # Must be Action.NONE (which is 0 and falsy)
        assert action == Action.NONE
        assert action.value == 0
        assert not action


# ---------------------------------------------------------------------------
# Time Arithmetic Edge Cases
# ---------------------------------------------------------------------------


class TestTimeArithmeticEdgeCases:
    """Tests for time arithmetic edge cases that commonly cause bugs."""

    def test_rotation_interval_exactly_at_boundary(
        self, default_config: ScheduleConfig
    ) -> None:
        """Rotation at exact interval boundary should be due."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Last rotation exactly 15 minutes ago (at boundary)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=3),
            last_rotation=now - timedelta(minutes=15),  # Exact boundary
            last_backfill=None,
        )

        # At boundary, should be due (>= interval)
        action = scheduler.what_to_do(now, state)
        assert Action.ROTATE in action

    def test_news_interval_exactly_at_boundary(
        self, default_config: ScheduleConfig
    ) -> None:
        """News at exact interval boundary should be due."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Last news refresh exactly 6 hours ago (at boundary)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=6),  # Exact boundary
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        # At boundary, should be due (>= interval)
        action = scheduler.what_to_do(now, state)
        assert Action.REFRESH_NEWS in action

    def test_seconds_one_microsecond_after_rotation(
        self, default_config: ScheduleConfig
    ) -> None:
        """Microseconds after rotation: compute correct sleep."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # Rotation just completed ( microseconds ago, well within interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(microseconds=1),  # Just now
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return ~5 minutes (full interval, minus 1 microsecond)
        # But implementation may round or truncate
        assert 295 <= seconds <= 300  # ~5 min give-or-take

    def test_timezone_naive_datetime_handling(
        self, default_config: ScheduleConfig
    ) -> None:
        """Scheduler should handle timezone-naive datetimes correctly."""
        scheduler = Scheduler(default_config)

        # Naive datetime (no timezone)
        now_naive = datetime(2024, 1, 15, 12, 0, 0)
        state = SchedulerState(
            last_news_refresh=now_naive - timedelta(hours=3),
            last_rotation=now_naive - timedelta(minutes=5),
            last_backfill=None,
        )

        # Should not raise
        action = scheduler.what_to_do(now_naive, state)
        assert action is not None

        seconds = scheduler.seconds_until_next_action(now_naive, state)
        assert seconds >= 0


# ---------------------------------------------------------------------------
# Stress and Invariant Tests
# ---------------------------------------------------------------------------


class TestSchedulerInvariants:
    """Tests for invariants that should always hold."""

    def test_seconds_never_negative(self) -> None:
        """seconds_until_next_action should never return negative values."""
        config = ScheduleConfig()
        scheduler = Scheduler(config)

        # Test across all hours
        for hour in range(24):
            now = datetime(2024, 1, 15, hour, 30, 0)
            state = SchedulerState(
                last_news_refresh=now - timedelta(hours=hour + 1),
                last_rotation=now - timedelta(minutes=hour + 1),
                last_backfill=None,
            )
            seconds = scheduler.seconds_until_next_action(now, state)
            assert seconds >= 0.0, f"Negative seconds at hour {hour}: {seconds}"

    def test_action_always_valid_in_active_hours(self) -> None:
        """In active hours, action should never be QUIET_ART."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        for hour in range(7, 23):  # Active hours
            now = datetime(2024, 1, 15, hour, 0, 0)
            state = SchedulerState(
                last_news_refresh=now - timedelta(hours=3),
                last_rotation=now - timedelta(minutes=5),
                last_backfill=None,
            )
            action = scheduler.what_to_do(now, state)
            assert Action.QUIET_ART not in action, f"QUIET_ART in active hours {hour}"

    def test_active_hours_get_normal_actions(self) -> None:
        """In active hours, first run should get normal actions, not QUIET_ART."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        # In active hours (12), should get normal actions
        now = datetime(2024, 1, 15, 12, 0, 0)
        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.QUIET_ART not in action
        assert Action.REFRESH_NEWS in action
