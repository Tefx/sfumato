"""Configuration loading and validation for sfumato.

Implements the config contract from ARCHITECTURE.md#2.1:
- source precedence and fallback behavior
- TOML parsing with type validation
- path normalization (tilde expansion + absolute resolution)
- default config generation for ``sfumato init``
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CONFIG_SOURCE_PRECEDENCE: tuple[str, ...] = (
    "path argument",
    "SFUMATO_CONFIG",
    "SFUMATO_DATA_DIR",
    "~/.config/sfumato/config.toml",
    "./sfumato.toml",
)
"""Source precedence for `load_config` from highest to lowest authority."""

SUPPORTED_ENV_OVERRIDES: tuple[str, ...] = ("SFUMATO_CONFIG", "SFUMATO_DATA_DIR")
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
class ApiKeysConfig:
    """API keys for external services.

    Keys can be set in config.toml or via environment variables.
    Env vars take precedence over config values.
    """

    rijksmuseum: str = ""


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration contract."""

    tv: TvConfig = field(default_factory=lambda: TvConfig(ip=""))
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    paintings: PaintingsConfig = field(default_factory=PaintingsConfig)
    ai: AiConfig = field(default_factory=AiConfig)
    api_keys: ApiKeysConfig = field(default_factory=ApiKeysConfig)
    data_dir: Path = Path("~/.sfumato")

    def __post_init__(self) -> None:
        # Expand ~ in all Path fields so they resolve to real filesystem paths
        object.__setattr__(self, "data_dir", self.data_dir.expanduser())
        object.__setattr__(
            self,
            "paintings",
            PaintingsConfig(
                cache_dir=self.paintings.cache_dir.expanduser(),
                seed_size=self.paintings.seed_size,
                pool_size=self.paintings.pool_size,
                sources=list(self.paintings.sources),
                match_strategy=self.paintings.match_strategy,
            ),
        )


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
    selected = _select_config_source(path)

    if selected is None:
        return _apply_env_overrides(_normalize_paths(AppConfig(), base_dir=Path.cwd()))

    data = _read_toml(selected)
    return _apply_env_overrides(_build_app_config(data=data, source_path=selected))


def generate_default_config() -> str:
    """Return a complete default config TOML document with comments.

    Returns:
        A string containing a full default config document.

    Raises:
        NotImplementedError: Always in this contract-only step.
    """
    return """# sfumato configuration
# See ARCHITECTURE.md#2.1 for config contract details.

data_dir = "~/.sfumato"

[tv]
ip = "192.168.1.100"
port = 8002
max_uploads = 5

[schedule]
news_interval_hours = 6
rotate_interval_minutes = 15
quiet_hours = [0, 6]
active_hours = [7, 23]

[news]
language = "zh"
stories_per_refresh = 12
max_age_days = 3
expire_days = 7

# [[news.feeds]]
# name = "Example Feed"
# url = "https://example.com/rss.xml"
# category = "Tech"

[[news.feeds]]
name = "TLDR Tech"
url = "https://tldr.tech/api/rss/tech"
category = "Tech"

[[news.feeds]]
name = "TLDR AI"
url = "https://tldr.tech/api/rss/ai"
category = "AI"

[[news.feeds]]
name = "Hacker News 100+"
url = "https://hnrss.org/frontpage?points=100"
category = "Tech"

[[news.feeds]]
name = "Ars Technica"
url = "https://feeds.arstechnica.com/arstechnica/index"
category = "Tech"

[[news.feeds]]
name = "BBC News"
url = "https://feeds.bbci.co.uk/news/rss.xml"
category = "World"

[[news.feeds]]
name = "Nature"
url = "https://www.nature.com/nature.rss"
category = "Science"

[paintings]
cache_dir = "~/.sfumato/paintings"
seed_size = 50
pool_size = 200
sources = ["rijksmuseum", "met", "wikimedia"]
match_strategy = "semantic"

[ai]
cli = "gemini"
model = "gemini-3.1-pro-preview"

[api_keys]
# rijksmuseum = "your-api-key-here"
# Get a free key at https://www.rijksmuseum.nl/en/register
# (account settings → advanced → API key)
# Env var RIJKSMUSEUM_API_KEY also works and takes precedence.
"""


def _select_config_source(path: Path | None) -> Path | None:
    if path is not None:
        return _resolve_authoritative_path(path, source_label="path argument")

    env_value = os.environ.get("SFUMATO_CONFIG")
    if env_value:
        return _resolve_authoritative_path(
            Path(env_value),
            source_label="SFUMATO_CONFIG",
        )

    optional_locations = (
        Path.home() / ".config" / "sfumato" / "config.toml",
        Path.cwd() / "sfumato.toml",
    )

    for candidate in optional_locations:
        if candidate.exists():
            if not candidate.is_file():
                raise ConfigError(
                    f"Config source '{candidate}' exists but is not a file"
                )
            return candidate.resolve()

    return None


def _resolve_authoritative_path(path: Path, source_label: str) -> Path:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else (Path.cwd() / expanded)
    resolved = absolute.resolve()

    if not resolved.exists():
        raise ConfigError(
            f"Authoritative config from {source_label} does not exist: {resolved}"
        )
    if not resolved.is_file():
        raise ConfigError(
            f"Authoritative config from {source_label} is not a file: {resolved}"
        )
    return resolved


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        parsed = tomllib.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise ConfigError(f"Failed to read config file '{path}': {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Config file '{path}' is not valid UTF-8: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Malformed TOML in '{path}': {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(f"Top-level TOML in '{path}' must be a table")
    return parsed


def _build_app_config(data: dict[str, Any], source_path: Path) -> AppConfig:
    defaults = AppConfig()

    tv_data = _expect_table(data, "tv", source_path)
    schedule_data = _expect_table(data, "schedule", source_path)
    news_data = _expect_table(data, "news", source_path)
    paintings_data = _expect_table(data, "paintings", source_path)
    ai_data = _expect_table(data, "ai", source_path)

    tv = TvConfig(
        ip=_expect_str_or_default(tv_data, "ip", defaults.tv.ip, source_path, "tv"),
        port=_expect_int_or_default(
            tv_data, "port", defaults.tv.port, source_path, "tv"
        ),
        max_uploads=_expect_int_or_default(
            tv_data,
            "max_uploads",
            defaults.tv.max_uploads,
            source_path,
            "tv",
        ),
    )

    schedule = ScheduleConfig(
        news_interval_hours=_expect_int_or_default(
            schedule_data,
            "news_interval_hours",
            defaults.schedule.news_interval_hours,
            source_path,
            "schedule",
        ),
        rotate_interval_minutes=_expect_int_or_default(
            schedule_data,
            "rotate_interval_minutes",
            defaults.schedule.rotate_interval_minutes,
            source_path,
            "schedule",
        ),
        quiet_hours=_expect_hours_tuple_or_default(
            schedule_data,
            "quiet_hours",
            defaults.schedule.quiet_hours,
            source_path,
            "schedule",
        ),
        active_hours=_expect_hours_tuple_or_default(
            schedule_data,
            "active_hours",
            defaults.schedule.active_hours,
            source_path,
            "schedule",
        ),
    )

    feeds_default: list[FeedConfig] = defaults.news.feeds
    feeds_value = news_data.get("feeds", feeds_default)
    feeds = _parse_feeds(feeds_value, source_path)

    news = NewsConfig(
        language=_expect_str_or_default(
            news_data,
            "language",
            defaults.news.language,
            source_path,
            "news",
        ),
        stories_per_refresh=_expect_int_or_default(
            news_data,
            "stories_per_refresh",
            defaults.news.stories_per_refresh,
            source_path,
            "news",
        ),
        max_age_days=_expect_int_or_default(
            news_data,
            "max_age_days",
            defaults.news.max_age_days,
            source_path,
            "news",
        ),
        expire_days=_expect_int_or_default(
            news_data,
            "expire_days",
            defaults.news.expire_days,
            source_path,
            "news",
        ),
        feeds=feeds,
    )

    paintings = PaintingsConfig(
        cache_dir=Path(
            _expect_str_or_default(
                paintings_data,
                "cache_dir",
                str(defaults.paintings.cache_dir),
                source_path,
                "paintings",
            )
        ),
        seed_size=_expect_int_or_default(
            paintings_data,
            "seed_size",
            defaults.paintings.seed_size,
            source_path,
            "paintings",
        ),
        pool_size=_expect_int_or_default(
            paintings_data,
            "pool_size",
            defaults.paintings.pool_size,
            source_path,
            "paintings",
        ),
        sources=_expect_str_list_or_default(
            paintings_data,
            "sources",
            defaults.paintings.sources,
            source_path,
            "paintings",
        ),
        match_strategy=_expect_str_or_default(
            paintings_data,
            "match_strategy",
            defaults.paintings.match_strategy,
            source_path,
            "paintings",
        ),
    )

    ai = AiConfig(
        cli=_expect_str_or_default(ai_data, "cli", defaults.ai.cli, source_path, "ai"),
        model=_expect_str_or_default(
            ai_data,
            "model",
            defaults.ai.model,
            source_path,
            "ai",
        ),
    )

    data_dir = Path(
        _expect_str_or_default(
            data,
            "data_dir",
            str(defaults.data_dir),
            source_path,
            "root",
        )
    )

    api_keys_data = _expect_table(data, "api_keys", source_path)
    # Config value, then env var override
    rijks_key = _expect_str_or_default(
        api_keys_data, "rijksmuseum", "", source_path, "api_keys"
    )
    rijks_key = os.environ.get("RIJKSMUSEUM_API_KEY", rijks_key)
    api_keys = ApiKeysConfig(rijksmuseum=rijks_key)

    config = AppConfig(
        tv=tv,
        schedule=schedule,
        news=news,
        paintings=paintings,
        ai=ai,
        api_keys=api_keys,
        data_dir=data_dir,
    )
    return _normalize_paths(config, base_dir=source_path.parent)


def _expect_table(data: dict[str, Any], key: str, source_path: Path) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(
            f"Invalid type for '{key}' in '{source_path}': expected table, got {type(value).__name__}"
        )
    return value


def _expect_str_or_default(
    data: dict[str, Any],
    key: str,
    default: str,
    source_path: Path,
    section: str,
) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(
            f"Invalid type for '{section}.{key}' in '{source_path}': expected str, got {type(value).__name__}"
        )
    return value


def _expect_int_or_default(
    data: dict[str, Any],
    key: str,
    default: int,
    source_path: Path,
    section: str,
) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"Invalid type for '{section}.{key}' in '{source_path}': expected int, got {type(value).__name__}"
        )
    return value


def _expect_hours_tuple_or_default(
    data: dict[str, Any],
    key: str,
    default: tuple[int, int],
    source_path: Path,
    section: str,
) -> tuple[int, int]:
    value = data.get(key, list(default))
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ConfigError(
            f"Invalid value for '{section}.{key}' in '{source_path}': expected two-item array"
        )

    first, second = value
    if isinstance(first, bool) or not isinstance(first, int):
        raise ConfigError(
            f"Invalid type for '{section}.{key}[0]' in '{source_path}': expected int, got {type(first).__name__}"
        )
    if isinstance(second, bool) or not isinstance(second, int):
        raise ConfigError(
            f"Invalid type for '{section}.{key}[1]' in '{source_path}': expected int, got {type(second).__name__}"
        )

    return (first, second)


def _expect_str_list_or_default(
    data: dict[str, Any],
    key: str,
    default: list[str],
    source_path: Path,
    section: str,
) -> list[str]:
    value = data.get(key, default)
    if not isinstance(value, list):
        raise ConfigError(
            f"Invalid type for '{section}.{key}' in '{source_path}': expected list[str], got {type(value).__name__}"
        )

    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(
                f"Invalid type for '{section}.{key}[{index}]' in '{source_path}': expected str, got {type(item).__name__}"
            )
        parsed.append(item)
    return parsed


def _parse_feeds(value: Any, source_path: Path) -> list[FeedConfig]:
    if not isinstance(value, list):
        raise ConfigError(
            f"Invalid type for 'news.feeds' in '{source_path}': expected array of tables"
        )

    feeds: list[FeedConfig] = []
    for index, feed in enumerate(value):
        if not isinstance(feed, dict):
            raise ConfigError(
                f"Invalid type for 'news.feeds[{index}]' in '{source_path}': expected table"
            )

        name = feed.get("name")
        url = feed.get("url")
        category = feed.get("category")

        if not isinstance(name, str):
            raise ConfigError(
                f"Invalid type for 'news.feeds[{index}].name' in '{source_path}': expected str"
            )
        if not isinstance(url, str):
            raise ConfigError(
                f"Invalid type for 'news.feeds[{index}].url' in '{source_path}': expected str"
            )
        if not isinstance(category, str):
            raise ConfigError(
                f"Invalid type for 'news.feeds[{index}].category' in '{source_path}': expected str"
            )

        feeds.append(FeedConfig(name=name, url=url, category=category))

    return feeds


def _normalize_paths(config: AppConfig, base_dir: Path) -> AppConfig:
    resolved_paintings_cache = _normalize_path(config.paintings.cache_dir, base_dir)
    resolved_data_dir = _normalize_path(config.data_dir, base_dir)

    return AppConfig(
        tv=config.tv,
        schedule=config.schedule,
        news=config.news,
        paintings=PaintingsConfig(
            cache_dir=resolved_paintings_cache,
            seed_size=config.paintings.seed_size,
            pool_size=config.paintings.pool_size,
            sources=list(config.paintings.sources),
            match_strategy=config.paintings.match_strategy,
        ),
        api_keys=config.api_keys,
        ai=config.ai,
        data_dir=resolved_data_dir,
    )


def _apply_env_overrides(config: AppConfig) -> AppConfig:
    env_data_dir = os.environ.get("SFUMATO_DATA_DIR")
    if not env_data_dir:
        return config

    resolved_data_dir = _normalize_path(Path(env_data_dir), Path.cwd())
    default_cache_dir = PaintingsConfig().cache_dir.expanduser().resolve()
    resolved_cache_dir = config.paintings.cache_dir
    if resolved_cache_dir == default_cache_dir:
        resolved_cache_dir = resolved_data_dir / "paintings"

    return AppConfig(
        tv=config.tv,
        schedule=config.schedule,
        news=config.news,
        paintings=PaintingsConfig(
            cache_dir=resolved_cache_dir,
            seed_size=config.paintings.seed_size,
            pool_size=config.paintings.pool_size,
            sources=list(config.paintings.sources),
            match_strategy=config.paintings.match_strategy,
        ),
        api_keys=config.api_keys,
        ai=config.ai,
        data_dir=resolved_data_dir,
    )


def _normalize_path(path: Path, base_dir: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (base_dir / expanded).resolve()
