"""Configuration contracts for sfumato core.

This module intentionally pins API and behavior contracts from
`ARCHITECTURE.md#2.1` without implementing parsing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CONFIG_SOURCE_PRECEDENCE: tuple[str, ...] = (
    "path argument",
    "SFUMATO_CONFIG",
    "~/.config/sfumato/config.toml",
    "./sfumato.toml",
)
"""Source precedence for `load_config` from highest to lowest authority."""

SUPPORTED_ENV_OVERRIDES: tuple[str, ...] = ("SFUMATO_CONFIG",)
"""Supported env overrides in core config contract for this phase."""


class ConfigError(Exception):
    """Raised when a config source is authoritative but invalid or unusable."""


@dataclass(frozen=True)
class TvConfig:
    """Typed TV configuration contract."""

    ip: str
    port: int = 8002
    max_uploads: int = 5


@dataclass(frozen=True)
class ScheduleConfig:
    """Typed schedule configuration contract."""

    news_interval_hours: int = 6
    rotate_interval_minutes: int = 15
    quiet_hours: tuple[int, int] = (0, 6)
    active_hours: tuple[int, int] = (7, 23)


@dataclass(frozen=True)
class FeedConfig:
    """Typed RSS feed configuration contract."""

    name: str
    url: str
    category: str


@dataclass(frozen=True)
class NewsConfig:
    """Typed news configuration contract."""

    language: str = "zh"
    stories_per_refresh: int = 12
    max_age_days: int = 3
    expire_days: int = 7
    feeds: list[FeedConfig] = field(default_factory=list)


@dataclass(frozen=True)
class PaintingsConfig:
    """Typed paintings configuration contract."""

    cache_dir: Path = Path("~/.sfumato/paintings")
    seed_size: int = 50
    pool_size: int = 200
    sources: list[str] = field(
        default_factory=lambda: ["rijksmuseum", "met", "wikimedia"]
    )
    match_strategy: str = "semantic"


@dataclass(frozen=True)
class AiConfig:
    """Typed AI backend configuration contract."""

    cli: str = "gemini"
    model: str = "gemini-3.1-pro-preview"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration contract."""

    tv: TvConfig = field(default_factory=TvConfig)  # type: ignore[arg-type] # architecture pins this exact contract
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    paintings: PaintingsConfig = field(default_factory=PaintingsConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    data_dir: Path = Path("~/.sfumato")


def load_config(path: Path | None = None) -> AppConfig:
    """Load config from TOML using authoritative precedence.

    Args:
        path: Explicit config file path. If provided, it is authoritative.

    Returns:
        AppConfig with path-valued fields expanded and resolved to absolute paths.

    Raises:
        ConfigError: If the selected authoritative source is missing, malformed TOML,
            or fails schema/type validation. Error messages include file context.

    Contract:
        Source precedence (highest first):
        1. `path` argument
        2. `SFUMATO_CONFIG`
        3. `~/.config/sfumato/config.toml`
        4. `./sfumato.toml`

        Missing-file behavior:
        - If explicit `path` is provided but missing, raise `ConfigError`.
        - If `SFUMATO_CONFIG` is set to a missing target, raise `ConfigError`.
        - If no source exists in the search path at all, return default `AppConfig`.

        Path behavior:
        - `~` expansion is required.
        - Path-valued TOML fields are resolved relative to the config file they came from.
        - Returned path-valued fields are absolute.

        Env behavior:
        - Core supports only `SFUMATO_CONFIG` as an environment override.
        - Additional per-field environment overrides are intentionally out of scope for
          this contract phase.
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def generate_default_config() -> str:
    """Return a complete default config TOML document with comments.

    Returns:
        A string containing a full default config document.

    Raises:
        NotImplementedError: Always in this contract-only step.
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )
