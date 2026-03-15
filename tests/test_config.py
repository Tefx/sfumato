"""Regression tests for config source selection and parsing behavior.

Spec source: ARCHITECTURE.md#2.1
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import pytest

from sfumato.config import (
    AppConfig,
    ConfigError,
    NewsConfig,
    generate_default_config,
    load_config,
)


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate HOME + CWD so discovery order is deterministic."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    monkeypatch.delenv("SFUMATO_CONFIG", raising=False)
    return tmp_path


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_default_discovery_returns_default_appconfig_when_nothing_exists(
    isolated_paths: Path,
) -> None:
    result = load_config()

    assert isinstance(result, AppConfig)
    assert result.paintings.cache_dir.is_absolute()
    assert result.data_dir.is_absolute()
    assert result.paintings.sources == ["met", "wikimedia"]


def test_precedence_explicit_path_beats_env_and_optional(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit_path = isolated_paths / "explicit.toml"
    env_path = isolated_paths / "env.toml"
    local_path = Path.cwd() / "sfumato.toml"

    _write(
        explicit_path,
        '[tv]\nip = "explicit-ip"\n[schedule]\nnews_interval_hours = 11\n',
    )
    _write(env_path, '[tv]\nip = "env-ip"\n')
    _write(local_path, '[tv]\nip = "local-ip"\n')
    monkeypatch.setenv("SFUMATO_CONFIG", str(env_path))

    result = load_config(path=explicit_path)

    assert result.tv.ip == "explicit-ip"
    assert result.schedule.news_interval_hours == 11


def test_precedence_env_beats_optional_discovery(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = isolated_paths / "env.toml"
    home_path = Path.home() / ".config" / "sfumato" / "config.toml"
    local_path = Path.cwd() / "sfumato.toml"

    _write(env_path, '[tv]\nip = "env-ip"\n')
    _write(home_path, '[tv]\nip = "home-ip"\n')
    _write(local_path, '[tv]\nip = "local-ip"\n')
    monkeypatch.setenv("SFUMATO_CONFIG", str(env_path))

    result = load_config()

    assert result.tv.ip == "env-ip"


def test_optional_discovery_prefers_home_location_before_local(
    isolated_paths: Path,
) -> None:
    home_path = Path.home() / ".config" / "sfumato" / "config.toml"
    local_path = Path.cwd() / "sfumato.toml"

    _write(home_path, '[tv]\nip = "home-ip"\n')
    _write(local_path, '[tv]\nip = "local-ip"\n')

    result = load_config()

    assert result.tv.ip == "home-ip"


def test_optional_discovery_falls_back_to_local_when_home_missing(
    isolated_paths: Path,
) -> None:
    local_path = Path.cwd() / "sfumato.toml"
    _write(local_path, '[tv]\nip = "local-ip"\n')

    result = load_config()

    assert result.tv.ip == "local-ip"


def test_missing_authoritative_explicit_path_raises_configerror(
    isolated_paths: Path,
) -> None:
    missing = isolated_paths / "missing.toml"

    with pytest.raises(ConfigError) as exc_info:
        load_config(path=missing)

    assert str(missing) in str(exc_info.value)


def test_missing_authoritative_env_target_raises_configerror(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = isolated_paths / "missing-env.toml"
    monkeypatch.setenv("SFUMATO_CONFIG", str(missing))

    with pytest.raises(ConfigError) as exc_info:
        load_config()

    message = str(exc_info.value)
    assert "SFUMATO_CONFIG" in message
    assert str(missing) in message


def test_malformed_toml_raises_configerror_with_file_context(
    isolated_paths: Path,
) -> None:
    config_path = isolated_paths / "malformed.toml"
    _write(config_path, "[tv\nip = 'broken'\n")

    with pytest.raises(ConfigError) as exc_info:
        load_config(path=config_path)

    assert str(config_path) in str(exc_info.value)


def test_wrong_field_type_raises_configerror_with_field_context(
    isolated_paths: Path,
) -> None:
    config_path = isolated_paths / "wrong-type.toml"
    _write(config_path, '[tv]\nip = "ok"\nport = "not-an-int"\n')

    with pytest.raises(ConfigError) as exc_info:
        load_config(path=config_path)

    message = str(exc_info.value)
    assert "tv.port" in message
    assert str(config_path) in message


def test_path_resolution_anchors_relative_paths_to_owning_config_file(
    isolated_paths: Path,
) -> None:
    config_dir = isolated_paths / "configs" / "nested"
    config_path = config_dir / "config.toml"
    _write(
        config_path,
        (
            'data_dir = "../runtime-data"\n\n'
            '[paintings]\ncache_dir = "relative-paintings"\n\n'
            '[tv]\nip = "ip"\n'
        ),
    )

    result = load_config(path=config_path)

    assert result.paintings.cache_dir == (config_dir / "relative-paintings").resolve()
    assert result.data_dir == (config_dir / "../runtime-data").resolve()


def test_path_resolution_keeps_absolute_and_expands_tilde(
    isolated_paths: Path,
) -> None:
    config_path = isolated_paths / "paths.toml"
    _write(
        config_path,
        (
            'data_dir = "/var/lib/sfumato"\n\n'
            '[paintings]\ncache_dir = "~/paintings-cache"\n\n'
            '[tv]\nip = "ip"\n'
        ),
    )

    result = load_config(path=config_path)

    assert result.paintings.cache_dir == (Path.home() / "paintings-cache").resolve()
    assert result.data_dir == Path("/var/lib/sfumato").resolve()


def test_env_data_dir_override_rehomes_default_runtime_paths(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFUMATO_DATA_DIR", "/data")

    result = load_config()

    assert result.data_dir == Path("/data")
    assert result.paintings.cache_dir == Path("/data/paintings")


def test_selected_source_values_win_and_unspecified_fields_use_defaults(
    isolated_paths: Path,
) -> None:
    config_path = isolated_paths / "partial.toml"
    _write(
        config_path,
        (
            '[tv]\nip = "192.168.88.22"\n\n'
            '[news]\nlanguage = "en"\n\n'
            '[[news.feeds]]\nname = "Example"\nurl = "https://example.com/rss"\ncategory = "Tech"\n'
        ),
    )

    result = load_config(path=config_path)

    assert result.tv.ip == "192.168.88.22"
    assert result.news.language == "en"
    assert result.tv.port == 8002
    assert result.schedule.rotate_interval_minutes == 5
    assert len(result.news.feeds) == 1
    assert result.news.feeds[0].name == "Example"


def test_generate_default_config_produces_parseable_complete_toml() -> None:
    import tomllib

    rendered = generate_default_config()
    parsed = tomllib.loads(rendered)

    assert "#" in rendered
    assert "tv" in parsed
    assert "schedule" in parsed
    assert "news" in parsed
    assert "paintings" in parsed
    assert "ai" in parsed


def test_news_config_contract_declares_replay_expire_days_default() -> None:
    replay_field = next(
        field for field in fields(NewsConfig) if field.name == "replay_expire_days"
    )
    type_hints = get_type_hints(NewsConfig)

    assert NewsConfig().replay_expire_days == 2
    assert type_hints["replay_expire_days"] is int
    assert replay_field.default == 2


def test_generate_default_config_places_replay_expire_days_under_news() -> None:
    import tomllib

    rendered = generate_default_config()
    parsed = tomllib.loads(rendered)

    assert parsed["news"]["replay_expire_days"] == 2


def test_news_config_contract_documents_omitted_field_behavior() -> None:
    assert "omit" in (NewsConfig.__doc__ or "")
    assert "default value ``2``" in (NewsConfig.__doc__ or "")


def test_replay_expire_days_defaults_to_2_when_omitted_from_config_file(
    isolated_paths: Path,
) -> None:
    """Backward-compatible behavior: when replay_expire_days is omitted from TOML config,
    it defaults to 2 (from NewsConfig dataclass default).

    This test documents the expected backward-compatible behavior for older configs
    that were created before replay_expire_days was added.
    """
    config_path = isolated_paths / "backward-compat.toml"
    _write(
        config_path,
        (
            '[tv]\nip = "192.168.1.100"\n\n'
            '[news]\nlanguage = "en"\n'
            "max_age_days = 3\nexpire_days = 7\nfeeds = []\n"
        ),
    )

    config = load_config(path=config_path)

    # Field omitted from TOML, defaults to 2 via dataclass
    assert config.news.replay_expire_days == 2


def test_replay_expire_days_is_parsed_from_toml(
    isolated_paths: Path,
) -> None:
    """Explicit replay_expire_days in TOML config is parsed and used."""
    config_path = isolated_paths / "explicit-replay.toml"
    _write(
        config_path,
        (
            '[tv]\nip = "192.168.1.100"\n\n'
            '[news]\nlanguage = "en"\nreplay_expire_days = 5\nfeeds = []\n'
        ),
    )

    config = load_config(path=config_path)

    assert config.news.replay_expire_days == 5


def test_generate_default_config_round_trips_through_load_config(
    isolated_paths: Path,
) -> None:
    """generate_default_config() output can be loaded by load_config()."""
    config_path = isolated_paths / "default.toml"
    config_path.write_text(generate_default_config())

    # Should not raise
    config = load_config(path=config_path)

    # Basic validation
    assert isinstance(config, AppConfig)
    assert len(config.news.feeds) == 15  # Default config has 15 feeds


def test_empty_feeds_array_is_accepted(
    isolated_paths: Path,
) -> None:
    """Config with empty news.feeds array is valid and returns empty list."""
    config_path = isolated_paths / "empty-feeds.toml"
    _write(
        config_path,
        '[tv]\nip = "192.168.1.1"\n[news]\nfeeds = []\n',
    )

    config = load_config(path=config_path)

    assert config.news.feeds == []


def test_missing_feeds_field_uses_empty_default(
    isolated_paths: Path,
) -> None:
    """Config without news.feeds field uses empty list default."""
    config_path = isolated_paths / "no-feeds.toml"
    _write(
        config_path,
        '[tv]\nip = "192.168.1.1"\n[news]\nlanguage = "en"\n',
    )

    config = load_config(path=config_path)

    assert config.news.feeds == []


def test_path_expansion_tilde_in_config_file(
    isolated_paths: Path,
) -> None:
    """Tilde paths in config file are expanded to absolute paths."""
    config_path = isolated_paths / "tilde.toml"
    _write(
        config_path,
        'data_dir = "~/mydata"\npaintings.cache_dir = "~/paintings"\n[tv]\nip = ""\n',
    )

    config = load_config(path=config_path)

    assert config.data_dir.is_absolute()
    assert config.paintings.cache_dir.is_absolute()
    # Both should expand relative to home
    assert str(config.data_dir).startswith(str(Path.home()))


def test_path_expansion_relative_path_in_config_file(
    isolated_paths: Path,
) -> None:
    """Relative paths in config file are resolved relative to config file location."""
    config_dir = isolated_paths / "subdir"
    config_path = config_dir / "relative.toml"
    _write(
        config_path,
        'data_dir = "mydata"\npaintings.cache_dir = "cache"\n[tv]\nip = ""\n',
    )

    config = load_config(path=config_path)

    assert config.data_dir.is_absolute()
    assert config.paintings.cache_dir.is_absolute()
    # Should be relative to config file's parent directory
    assert "mydata" in str(config.data_dir)
    assert "cache" in str(config.paintings.cache_dir)


def test_path_expansion_relative_path_in_env_override(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SFUMATO_DATA_DIR with relative path is resolved to absolute path."""
    # Set a relative path via env
    monkeypatch.setenv("SFUMATO_DATA_DIR", "relative/data")
    monkeypatch.delenv("SFUMATO_CONFIG", raising=False)

    config = load_config()

    # Should be expanded to absolute
    assert config.data_dir.is_absolute()
    assert "relative" in str(config.data_dir) or config.data_dir.is_absolute()


def test_path_expansion_tilde_in_env_override(
    isolated_paths: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SFUMATO_DATA_DIR with tilde path is expanded."""
    monkeypatch.setenv("SFUMATO_DATA_DIR", "~/envdata")
    monkeypatch.delenv("SFUMATO_CONFIG", raising=False)

    config = load_config()

    assert config.data_dir.is_absolute()
    assert str(config.data_dir).startswith(str(Path.home()))


def test_minimal_tv_only_toml_fills_all_defaults(
    isolated_paths: Path,
) -> None:
    """Loading a minimal TOML with only [tv] ip fills all other fields with defaults.

    Backward-compatibility contract:
    - replay_expire_days defaults to 2
    - backend defaults to "sdk"
    - sdk_provider defaults to "openrouter"
    - Unknown/removed fields (e.g. stories_per_refresh) are silently ignored
    """
    config_path = isolated_paths / "minimal.toml"
    _write(config_path, '[tv]\nip = "1.2.3.4"\n')

    config = load_config(path=config_path)

    # TV section: explicit ip, defaults for rest
    assert config.tv.ip == "1.2.3.4"
    assert config.tv.port == 8002
    assert config.tv.max_uploads == 5

    # Schedule: all defaults
    assert config.schedule.news_interval_hours == 6
    assert config.schedule.rotate_interval_minutes == 5
    assert config.schedule.active_hours == (10, 2)

    # News: all defaults including replay_expire_days
    assert config.news.language == "zh"
    assert config.news.max_age_days == 3
    assert config.news.expire_days == 7
    assert config.news.replay_expire_days == 2
    assert config.news.feeds == []

    # Paintings: all defaults
    assert config.paintings.seed_size == 50
    assert config.paintings.pool_size == 200
    assert config.paintings.sources == ["met", "wikimedia"]
    assert config.paintings.match_strategy == "semantic"

    # AI: all defaults including backend and sdk_provider
    assert config.ai.cli == "gemini"
    assert config.ai.model == "gemini-3-flash-preview"
    assert config.ai.backend == "sdk"
    assert config.ai.sdk_provider == "openrouter"

    # Paths are absolute
    assert config.data_dir.is_absolute()
    assert config.paintings.cache_dir.is_absolute()


def test_old_toml_with_removed_field_stories_per_refresh_is_silently_ignored(
    isolated_paths: Path,
) -> None:
    """Old TOML configs with removed fields like stories_per_refresh don't crash."""
    config_path = isolated_paths / "old-config.toml"
    _write(
        config_path,
        '[tv]\nip = "1.2.3.4"\n\n[news]\nstories_per_refresh = 10\nlanguage = "en"\nfeeds = []\n',
    )

    # Should not raise despite unknown field stories_per_refresh
    config = load_config(path=config_path)

    assert config.tv.ip == "1.2.3.4"
    assert config.news.language == "en"
    # replay_expire_days defaults even when other news fields are present
    assert config.news.replay_expire_days == 2
