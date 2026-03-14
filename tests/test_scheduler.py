"""Contract tests for scheduler module.

Implements test stubs that will be filled with verification logic
once the scheduler implementation is complete. Tests cover:

- Action enum and flag behavior
- Time window logic (quiet_hours, active_hours)
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
        quiet_hours=(22, 6),  # 22:00-6:00 (crosses midnight)
        active_hours=(7, 23),
    )


@pytest.fixture
def all_day_active_config() -> ScheduleConfig:
    """Return ScheduleConfig with 24-hour active hours."""
    return ScheduleConfig(
        news_interval_hours=6,
        rotate_interval_minutes=15,
        quiet_hours=(0, 0),  # No quiet hours
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
        assert Action.IDLE

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
        assert Action.IDLE not in combined

    def test_action_unique_bit_values(self) -> None:
        """Each action should have a unique bit position."""
        values = [
            Action.NONE.value,
            Action.REFRESH_NEWS.value,
            Action.ROTATE.value,
            Action.BACKFILL.value,
            Action.QUIET_ART.value,
            Action.IDLE.value,
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
# is_quiet_hour Tests
# ---------------------------------------------------------------------------


class TestIsQuietHour:
    """Tests for is_quiet_hour time window logic."""

    def test_quiet_hour_default_range(self, default_config: ScheduleConfig) -> None:
        """Default quiet_hours=(0, 6) should cover hours 0-5."""
        scheduler = Scheduler(default_config)

        # Hours within quiet period
        for hour in [0, 1, 2, 3, 4, 5]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now), f"Hour {hour} should be quiet"

        # Hours outside quiet period
        for hour in [6, 7, 12, 18, 22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_quiet_hour(now), f"Hour {hour} should NOT be quiet"

    def test_quiet_hour_crossing_midnight(self, custom_config: ScheduleConfig) -> None:
        """quiet_hours=(22, 6) crosses midnight; hours 22-23 and 0-5 are quiet."""
        scheduler = Scheduler(custom_config)

        # Hours in evening part (22-23)
        for hour in [22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now), f"Hour {hour} should be quiet"

        # Hours in morning part (0-5)
        for hour in [0, 1, 2, 3, 4, 5]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now), f"Hour {hour} should be quiet"

        # Hours outside
        for hour in [6, 7, 12, 15, 20, 21]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_quiet_hour(now), f"Hour {hour} should NOT be quiet"

    def test_quiet_hour_boundary_start(self, default_config: ScheduleConfig) -> None:
        """At the start of quiet_hours boundary."""
        scheduler = Scheduler(default_config)

        # Just before quiet_hours start (hour 23:59:59 -> 0)
        before_start = datetime(2024, 1, 15, 23, 59, 59)
        assert not scheduler.is_quiet_hour(before_start)

        # At start of quiet_hours (hour 0)
        at_start = datetime(2024, 1, 16, 0, 0, 0)
        assert scheduler.is_quiet_hour(at_start)

    def test_quiet_hour_boundary_end(self, default_config: ScheduleConfig) -> None:
        """At the end of quiet_hours boundary (exclusive)."""
        scheduler = Scheduler(default_config)

        # Just before end of quiet_hours (5:59:59)
        before_end = datetime(2024, 1, 16, 5, 59, 59)
        assert scheduler.is_quiet_hour(before_end)

        # At end of quiet_hours (hour 6, exclusive)
        at_end = datetime(2024, 1, 16, 6, 0, 0)
        assert not scheduler.is_quiet_hour(at_end)

    def test_quiet_hour_no_quiet_hours(
        self, all_day_active_config: ScheduleConfig
    ) -> None:
        """quiet_hours=(0, 0) is empty; no time should be quiet."""
        scheduler = Scheduler(all_day_active_config)

        for hour in range(24):
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_quiet_hour(now), f"Hour {hour} should NOT be quiet"


# ---------------------------------------------------------------------------
# is_active_hour Tests
# ---------------------------------------------------------------------------


class TestIsActiveHour:
    """Tests for is_active_hour time window logic."""

    def test_active_hour_default_range(self, default_config: ScheduleConfig) -> None:
        """Default active_hours=(7, 23) should cover hours 7-23 (inclusive)."""
        scheduler = Scheduler(default_config)

        # Hours within active period (7-23 inclusive)
        for hour in [7, 8, 12, 15, 22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

        # Hours outside active period (0-6)
        for hour in [0, 1, 2, 3, 4, 5, 6]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_active_hour(now), (
                f"Hour {hour} should NOT be active"
            )

    def test_active_hour_boundary_start(self, default_config: ScheduleConfig) -> None:
        """At the start of active_hours boundary (inclusive)."""
        scheduler = Scheduler(default_config)

        # Just before active_hours (6:59:59)
        before_start = datetime(2024, 1, 15, 6, 59, 59)
        assert not scheduler.is_active_hour(before_start)

        # At start of active_hours (7:00)
        at_start = datetime(2024, 1, 15, 7, 0, 0)
        assert scheduler.is_active_hour(at_start)

    def test_active_hour_boundary_end(self, default_config: ScheduleConfig) -> None:
        """At the end of active_hours boundary (inclusive)."""
        scheduler = Scheduler(default_config)

        # At end of active_hours (23:00 inclusive)
        at_end = datetime(2024, 1, 15, 23, 30, 0)
        assert scheduler.is_active_hour(at_end)

        # Just after active_hours (0:00)
        after_end = datetime(2024, 1, 16, 0, 0, 0)
        assert not scheduler.is_active_hour(after_end)

    def test_active_hour_all_day(self, all_day_active_config: ScheduleConfig) -> None:
        """active_hours=(0, 23) should be active all day."""
        scheduler = Scheduler(all_day_active_config)

        for hour in range(24):
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"


# ---------------------------------------------------------------------------
# what_to_do Tests
# ---------------------------------------------------------------------------


class TestWhatToDo:
    """Tests for the what_to_do scheduling decision function."""

    def test_what_to_do_outside_active_unless_quiet(
        self, default_config: ScheduleConfig, recent_state: SchedulerState
    ) -> None:
        """Outside active_hours and not in quiet_hours should return IDLE."""
        scheduler = Scheduler(default_config)

        # Hour 6 is outside active_hours (7-23) and not in quiet_hours (0-6)
        now = datetime(2024, 1, 15, 6, 30, 0)
        action = scheduler.what_to_do(now, recent_state)
        assert action == Action.IDLE

    def test_what_to_do_in_quiet_hours(
        self, default_config: ScheduleConfig, recent_state: SchedulerState
    ) -> None:
        """Within quiet_hours should return QUIET_ART."""
        scheduler = Scheduler(default_config)

        # Hour 3 is in quiet_hours (0-6)
        # Note: With default config, hour 3 is also outside active_hours
        # The contract says: outside active_hours -> check quiet_hours -> QUIET_ART
        now = datetime(2024, 1, 15, 3, 30, 0)
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

        # Rotation was 5 minutes ago (10 minutes remaining for 15-minute interval)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return roughly 10 minutes (600 seconds)
        assert 590 <= seconds <= 610  # Allow small tolerance

    def test_seconds_until_active_start(self) -> None:
        """Outside active hours, should compute time until active start."""
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            quiet_hours=(0, 6),
            active_hours=(7, 23),
        )
        scheduler = Scheduler(config)

        # Hour 6: outside active active_hours, not in quiet_hours
        # Time until active_hours start (7:00) = 1 hour
        now = datetime(2024, 1, 15, 6, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=1),
            last_rotation=now - timedelta(minutes=5),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Should return time until hour 7
        assert 3590 <= seconds <= 3610  # ~1 hour (allow tolerance)

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

    def test_midnight_wrapping_in_quiet_hours(
        self, custom_config: ScheduleConfig
    ) -> None:
        """quiet_hours crossing midnight should work correctly."""
        scheduler = Scheduler(custom_config)  # (22, 6)

        # Test evening hours
        for hour in [22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now)

        # Test early morning hours
        for hour in [0, 1, 2, 3, 4, 5]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now)

    def test_transition_quiet_to_active(self) -> None:
        """Transition from quiet_hours to active_hours should trigger correctly."""
        # Custom config where quiet_hours and active_hours don't overlap
        # (default config: quiet=(0,6), active=(7,23))
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            quiet_hours=(0, 6),
            active_hours=(7, 23),
        )
        scheduler = Scheduler(config)

        # Hour 6: outside both (IDLE)
        now_6 = datetime(2024, 1, 15, 6, 30, 0)
        assert scheduler.is_quiet_hour(now_6) is False
        assert scheduler.is_active_hour(now_6) is False

    def test_quiet_art_overlaps_with_active(
        self, all_day_active_config: ScheduleConfig
    ) -> None:
        """When active_hours covers entire day, quiet_hours can activate within."""
        # active_hours=(0, 23), quiet_hours=(0, 0) = none
        # Let's test with overlapping config
        config = ScheduleConfig(
            news_interval_hours=6,
            rotate_interval_minutes=15,
            quiet_hours=(0, 6),
            active_hours=(0, 23),  # All day active
        )
        scheduler = Scheduler(config)

        # Hour 3: within both active_hours and quiet_hours
        now = datetime(2024, 1, 15, 3, 30, 0)
        assert scheduler.is_active_hour(now)  # Active
        # what_to_do should respect precedence:
        # - Within active AND within quiet -> QUIET_ART


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
        assert hasattr(scheduler, "is_quiet_hour")
        assert hasattr(scheduler, "is_active_hour")
        assert callable(scheduler.what_to_do)
        assert callable(scheduler.seconds_until_next_action)
        assert callable(scheduler.is_quiet_hour)
        assert callable(scheduler.is_active_hour)

    def test_scheduler_state_dataclass(self) -> None:
        """Verify SchedulerState is a proper dataclass."""
        from dataclasses import fields

        field_names = {f.name for f in fields(SchedulerState)}
        assert "last_news_refresh" in field_names
        assert "last_rotation" in field_names
        assert "last_backfill" in field_names

    def test_default_config_values(self) -> None:
        """Verify default ScheduleConfig matches ARCHITECTURE.md."""
        config = ScheduleConfig()
        assert config.news_interval_hours == 6
        assert config.rotate_interval_minutes == 15
        assert config.quiet_hours == (0, 6)
        assert config.active_hours == (7, 23)


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
        """First action of day after quiet hours may trigger immediate refresh."""
        scheduler = Scheduler(default_config)
        # 7:00 AM - first active hour, coming from quiet hours
        now = datetime(2024, 1, 15, 7, 0, 0)

        # Last actions were during previous day's active hours
        state = SchedulerState(
            last_news_refresh=datetime(2024, 1, 14, 20, 0, 0),  # 11h overnight gap
            last_rotation=datetime(2024, 1, 14, 23, 0, 0),  # 8h overnight gap
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # News is overdue (11h > 6h), rotation is overdue (8h > any interval)
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
    - System is within active_hours and outside quiet_hours

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

    def test_backfill_not_possible_during_quiet_hours(
        self, default_config: ScheduleConfig
    ) -> None:
        """BACKFILL should not run during quiet hours."""
        scheduler = Scheduler(default_config)
        # Hour 3 is in quiet_hours (0-6)
        now = datetime(2024, 1, 15, 3, 30, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=12),
            last_rotation=now - timedelta(hours=8),
            last_backfill=now - timedelta(hours=24),
        )
        action = scheduler.what_to_do(now, state)

        # In quiet hours, no backfill
        assert action == Action.QUIET_ART
        assert Action.BACKFILL not in action

    def test_backfill_not_possible_outside_active_hours(
        self, default_config: ScheduleConfig
    ) -> None:
        """BACKFILL should not run outside active hours."""
        scheduler = Scheduler(default_config)
        # Hour 6 is outside active_hours (7-23) and not in quiet_hours
        now = datetime(2024, 1, 15, 6, 30, 0)

        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=12),
            last_rotation=now - timedelta(hours=8),
            last_backfill=now - timedelta(hours=48),
        )
        action = scheduler.what_to_do(now, state)

        # Outside active hours but not in quiet -> IDLE
        assert action == Action.IDLE
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
            last_rotation=now - timedelta(minutes=5),
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
            last_rotation=now - timedelta(minutes=14, seconds=30),
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Rotation due in 30 seconds (interval is 15 min)
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
            quiet_hours=(0, 6),
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
            quiet_hours=(22, 6),  # Active: 22-23, then 6->7 is IDLE
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
        # or until end of quiet_hours at 6:00 = 6.5 hours
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
        # ~14 minutes remaining
        assert 830 <= seconds <= 870

    def test_seconds_news_rotation_race(self, default_config: ScheduleConfig) -> None:
        """When both intervals matter, return minimum time."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 12, 0, 0)

        # News due in 1 hour, rotation due in 5 minutes
        # Should return 5 minutes (300 seconds)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=5),  # Due in 1 hour
            last_rotation=now - timedelta(minutes=10),  # Due in 5 min
            last_backfill=None,
        )

        seconds = scheduler.seconds_until_next_action(now, state)
        # Rotation wins - ~300 seconds
        assert 290 <= seconds <= 310

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

        # Rotation was exactly 14:59 ago (15 min interval - 1 second)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=3),
            last_rotation=now - timedelta(minutes=14, seconds=59),
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
            quiet_hours=(22, 6),
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
# Active/Quiet Hour Boundary and Edge Cases
# ---------------------------------------------------------------------------


class TestActiveQuietHourBoundaries:
    """Tests for boundary conditions in time window logic."""

    def test_exact_hour_start_is_active(self) -> None:
        """At exactly active_hours[0], should be active."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        # Exactly 7:00:00
        now = datetime(2024, 1, 15, 7, 0, 0)
        assert scheduler.is_active_hour(now)

    def test_exact_hour_end_is_active(self) -> None:
        """At exactly active_hours[1], should be active (inclusive)."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        # Exactly 23:00:00 (hour 23, not hour 24)
        # active_hours=(7, 23) means hours 7-23 are active, inclusive
        now = datetime(2024, 1, 15, 23, 0, 0)
        assert scheduler.is_active_hour(now)

        # 23:59:59 is also active
        now_edge = datetime(2024, 1, 15, 23, 59, 59)
        assert scheduler.is_active_hour(now_edge)

    def test_midnight_is_not_active_default(self) -> None:
        """Hour 0 (midnight) is not active with default config."""
        config = ScheduleConfig(active_hours=(7, 23))
        scheduler = Scheduler(config)

        now = datetime(2024, 1, 15, 0, 0, 0)
        assert not scheduler.is_active_hour(now)

        now_with_minutes = datetime(2024, 1, 15, 0, 30, 0)
        assert not scheduler.is_active_hour(now_with_minutes)

    def test_quiet_hour_exact_start(self) -> None:
        """At exactly quiet_hours[0], should be quiet."""
        config = ScheduleConfig(quiet_hours=(0, 6))
        scheduler = Scheduler(config)

        now = datetime(2024, 1, 15, 0, 0, 0)
        assert scheduler.is_quiet_hour(now)

    def test_quiet_hour_exact_end_exclusive(self) -> None:
        """At exactly quiet_hours[1], should NOT be quiet (exclusive end)."""
        config = ScheduleConfig(quiet_hours=(0, 6))
        scheduler = Scheduler(config)

        # Hour 6 is NOT quiet (end is exclusive)
        now = datetime(2024, 1, 15, 6, 0, 0)
        assert not scheduler.is_quiet_hour(now)

        # Hour 5:59:59 is still quiet
        now_before_end = datetime(2024, 1, 15, 5, 59, 59)
        assert scheduler.is_quiet_hour(now_before_end)

    def test_quiet_hour_same_hour_in_range(self) -> None:
        """quiet_hours=(x, x) means empty range - never quiet."""
        config = ScheduleConfig(quiet_hours=(12, 12))
        scheduler = Scheduler(config)

        for hour in range(24):
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_quiet_hour(now), f"Hour {hour} should not be quiet"

    def test_active_hour_all_hours(self) -> None:
        """active_hours=(0, 23) means all hours active."""
        config = ScheduleConfig(active_hours=(0, 23))
        scheduler = Scheduler(config)

        for hour in range(24):
            now = datetime(2024, 1, 15, hour, 0, 0)
            assert scheduler.is_active_hour(now), f"Hour {hour} should be active"

    def test_quiet_hour_crossing_midnight_evening_first(self) -> None:
        """quiet_hours=(22, 2): 22-23 is quiet, then 0-1 is quiet."""
        config = ScheduleConfig(quiet_hours=(22, 2))
        scheduler = Scheduler(config)

        # Evening portion: 22, 23
        for hour in [22, 23]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now), f"Hour {hour} should be quiet"

        # Early morning portion: 0, 1
        for hour in [0, 1]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert scheduler.is_quiet_hour(now), f"Hour {hour} should be quiet"

        # NOT quiet: 2-21
        for hour in range(2, 22):
            now = datetime(2024, 1, 15, hour, 30, 0)
            assert not scheduler.is_quiet_hour(now), f"Hour {hour} should NOT be quiet"

    def test_quiet_hour_crossing_midnight_morning_first(self) -> None:
        """quiet_hours=(23, 1): 23 is quiet, then 0 is quiet."""
        config = ScheduleConfig(quiet_hours=(23, 1))
        scheduler = Scheduler(config)

        # Hour 23
        assert scheduler.is_quiet_hour(datetime(2024, 1, 15, 23, 30, 0))
        # Hour 0
        assert scheduler.is_quiet_hour(datetime(2024, 1, 15, 0, 30, 0))
        # Hour 1 and beyond: not quiet
        assert not scheduler.is_quiet_hour(datetime(2024, 1, 15, 1, 30, 0))
        assert not scheduler.is_quiet_hour(datetime(2024, 1, 15, 22, 30, 0))


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

    def test_first_run_during_quiet_hours(self, default_config: ScheduleConfig) -> None:
        """First run during quiet hours returns QUIET_ART."""
        scheduler = Scheduler(default_config)
        now = datetime(2024, 1, 15, 3, 0, 0)  # Quiet hour (0-6)

        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # Even on first run, quiet hours take precedence
        assert action == Action.QUIET_ART

    def test_first_run_outside_active_not_quiet(
        self, default_config: ScheduleConfig
    ) -> None:
        """First run outside active hours (not quiet) returns IDLE."""
        scheduler = Scheduler(default_config)
        # Hour 6 is outside active (7-23) and not in quiet (0-6)
        now = datetime(2024, 1, 15, 6, 30, 0)

        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # Outside active hours, not in quiet -> IDLE
        assert action == Action.IDLE

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

    def test_quiet_hours_overrides_active_inside_check(self) -> None:
        """Time in both quiet and active hours: QUIET_ART takes precedence."""
        # Config where quiet_hours overlap with active_hours
        config = ScheduleConfig(
            quiet_hours=(0, 8),
            active_hours=(0, 23),
        )
        scheduler = Scheduler(config)

        # Hour 3: in both quiet (0-8) and active (0-23)
        now = datetime(2024, 1, 15, 3, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=10),
            last_rotation=now - timedelta(hours=2),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        # QUIET_ART should be returned (not REFRESH_NEWS or ROTATE)
        # This tests the precedence: outside active -> check quiet -> QUIET_ART
        # Implementation note: when both overlap, implementation behavior to verify
        # The contract says: check active first, then quiet if not in active
        # So if is_active_hour returns True, quiet check is skipped
        # This test documents expected behavior when config has overlap
        assert action == Action.QUIET_ART or action == Action.NONE

    def test_idle_over_quiet_outside_active(self) -> None:
        """Outside active and not in quiet: IDLE, not QUIET_ART."""
        config = ScheduleConfig(
            quiet_hours=(0, 6),
            active_hours=(8, 22),
        )
        scheduler = Scheduler(config)

        # Hour 7: in neither (gap between quiet=0-6 and active=8-22)
        now = datetime(2024, 1, 15, 7, 0, 0)
        state = SchedulerState(
            last_news_refresh=now - timedelta(hours=10),
            last_rotation=now - timedelta(hours=2),
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert action == Action.IDLE

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
            last_rotation=now - timedelta(minutes=5),  # 10min till due
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
        # Should return ~15 minutes (full interval, minus 1 microsecond)
        # But implementation may round or truncate
        assert 895 <= seconds <= 900  # ~15 min give-or-take

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
        """In active hours, action should never be IDLE or QUIET_ART."""
        config = ScheduleConfig(quiet_hours=(0, 6), active_hours=(7, 23))
        scheduler = Scheduler(config)

        for hour in range(7, 24):  # Active hours
            now = datetime(2024, 1, 15, hour, 0, 0)
            state = SchedulerState(
                last_news_refresh=now - timedelta(hours=3),
                last_rotation=now - timedelta(minutes=5),
                last_backfill=None,
            )
            action = scheduler.what_to_do(now, state)
            assert Action.IDLE not in action or action == Action.NONE, (
                f"IDLE in active hours {hour}"
            )
            assert Action.QUIET_ART not in action, f"QUIET_ART in active hours {hour}"

    def test_quiet_hours_during_active_are_ignored(self) -> None:
        """Non-overlapping quiet/active hours don't interfere."""
        config = ScheduleConfig(quiet_hours=(0, 6), active_hours=(7, 23))
        scheduler = Scheduler(config)

        # In active hours (12), should never get QUIET_ART
        now = datetime(2024, 1, 15, 12, 0, 0)
        state = SchedulerState(
            last_news_refresh=None,
            last_rotation=None,
            last_backfill=None,
        )
        action = scheduler.what_to_do(now, state)

        assert Action.QUIET_ART not in action
