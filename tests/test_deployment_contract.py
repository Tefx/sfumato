"""Contract tests for Docker deployment artifacts.

These tests validate the production deployment `Dockerfile` and
`docker-compose.yml` artifacts defined by `ARCHITECTURE.md#9` and
`DEPLOYMENT_CONTRACT.md`.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from validate_deployment_contract import (
    validate_compose_text,
    validate_docker_compose_config,
    validate_dockerfile_text,
)


ROOT = Path(__file__).resolve().parents[1]


class TestDockerfileContract:
    def test_dockerfile_declares_required_stages(self) -> None:
        dockerfile_text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        validate_dockerfile_text(dockerfile_text)

    def test_dockerfile_defaults_to_watch_mode(self) -> None:
        dockerfile_text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert 'ENTRYPOINT ["/opt/sfumato/docker/entrypoint.sh"]' in dockerfile_text
        assert (
            'CMD ["sfumato", "watch", "--config", "/data/config.toml"]'
            in dockerfile_text
        )

    def test_dockerfile_installs_playwright_chromium(self) -> None:
        dockerfile_text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "python -m playwright install --with-deps chromium" in dockerfile_text


class TestComposeContract:
    def test_compose_declares_runtime_service_contract(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        validate_compose_text(compose_text)

    def test_compose_healthcheck_targets_freshness_artifact(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert (
            'test: ["CMD", "python", "/opt/sfumato/docker/healthcheck.py"]'
            in compose_text
        )
        assert "SFUMATO_HEALTH_MAX_AGE_SECONDS: 2100" in compose_text

    def test_compose_stop_contract_is_bounded(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "stop_signal: SIGTERM" in compose_text
        assert "stop_grace_period: 90s" in compose_text

    def test_compose_restarts_watch_service(self) -> None:
        compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        assert "restart: unless-stopped" in compose_text


class TestValidationScript:
    def test_validation_script_succeeds(self) -> None:
        result = subprocess.run(
            [sys.executable, "validate_deployment_contract.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr or result.stdout

    def test_validation_script_optionally_accepts_compose_cli(self) -> None:
        validate_docker_compose_config()
