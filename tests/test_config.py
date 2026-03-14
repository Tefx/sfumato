"""Regression tests for config source selection and parsing behavior.

Spec source: ARCHITECTURE.md#2.1
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sfumato.config import AppConfig, ConfigError, generate_default_config, load_config


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
    assert result.paintings.sources == ["rijksmuseum", "met", "wikimedia"]


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
    assert result.schedule.rotate_interval_minutes == 15
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
