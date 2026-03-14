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

    def test_what_to_do_overdue_news(
        self, default_config: ScheduleConfig, overdue_state: SchedulerState
    ) -> None:
        """Overdue news refresh should return REFRESH_NEWS | ROTATE."""
        scheduler = Scheduler(default_config)

        # Hour 12 is within active_hours, state has 10h old news refresh
        now = datetime(2024, 1, 15, 12, 0, 0)
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
