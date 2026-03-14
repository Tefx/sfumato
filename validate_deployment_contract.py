"""Validate Dockerfile and compose contract artifacts.

This script enforces the deployment contract defined in:
- ARCHITECTURE.md#9
- DEPLOYMENT_CONTRACT.md

It is intentionally static and contract-focused. It does not require a runnable
production container implementation.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DOCKERFILE_PATH = ROOT / "Dockerfile"
COMPOSE_PATH = ROOT / "docker-compose.yml"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(text: str, needle: str, message: str) -> None:
    if needle not in text:
        raise AssertionError(message)


def require_regex(text: str, pattern: str, message: str) -> None:
    if re.search(pattern, text, re.MULTILINE) is None:
        raise AssertionError(message)


def validate_dockerfile_text(text: str) -> None:
    require_regex(
        text,
        r"^FROM\s+\$\{PYTHON_IMAGE\}\s+AS\s+base-runtime$",
        "Dockerfile must define base-runtime stage",
    )
    require_regex(
        text,
        r"^FROM\s+base-runtime\s+AS\s+builder$",
        "Dockerfile must define builder stage",
    )
    require_regex(
        text,
        r"^FROM\s+base-runtime\s+AS\s+runtime$",
        "Dockerfile must define runtime stage",
    )
    require(
        text,
        "ENV SFUMATO_CONFIG=/data/config.toml",
        "Dockerfile must default SFUMATO_CONFIG to /data/config.toml",
    )
    require(
        text,
        "ENV SFUMATO_DATA_DIR=/data",
        "Dockerfile must default SFUMATO_DATA_DIR to /data",
    )
    require(
        text,
        "ENV RIJKSMUSEUM_API_KEY=",
        "Dockerfile must expose RIJKSMUSEUM_API_KEY for runtime injection",
    )
    require(text, 'VOLUME ["/data"]', "Dockerfile must declare /data volume")
    require(
        text,
        'ENTRYPOINT ["python3", "-c", "raise SystemExit(\'Dockerfile contract stub only; see DEPLOYMENT_CONTRACT.md\')"]',
        "Dockerfile must remain a contract stub in this phase",
    )


def validate_compose_text(text: str) -> None:
    require_regex(
        text,
        r"^services:\n\s+sfumato:\n",
        "Compose file must define an sfumato service",
    )
    require(text, "target: runtime", "Compose file must target the runtime stage")
    require(
        text,
        "SFUMATO_CONFIG: /data/config.toml",
        "Compose file must set SFUMATO_CONFIG",
    )
    require(text, "SFUMATO_DATA_DIR: /data", "Compose file must set SFUMATO_DATA_DIR")
    require(
        text,
        "RIJKSMUSEUM_API_KEY: ${RIJKSMUSEUM_API_KEY:-}",
        "Compose file must pass through RIJKSMUSEUM_API_KEY",
    )
    require(text, "network_mode: host", "Compose file must default to host networking")
    require(
        text, "stop_signal: SIGTERM", "Compose file must define SIGTERM stop signal"
    )
    require_regex(
        text,
        r"stop_grace_period:\s*[1-9]\d*s",
        "Compose file must define a non-zero stop_grace_period",
    )
    require(
        text,
        'test: ["CMD-SHELL", "test -s /data/state/last_action.json"]',
        "Compose healthcheck must read the freshness artifact path",
    )
    for source, target in (
        ("source: ./data/config.toml", "target: /data/config.toml"),
        ("source: ./data/paintings", "target: /data/paintings"),
        ("source: ./data/state", "target: /data/state"),
        ("source: ./data/output", "target: /data/output"),
    ):
        require(text, source, f"Compose file must mount {source}")
        require(text, target, f"Compose file must mount {target}")


def validate_docker_compose_config() -> None:
    docker = shutil.which("docker")
    if docker is None:
        return

    result = subprocess.run(
        [docker, "compose", "config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"docker compose config failed: {result.stderr.strip()}")


def main() -> int:
    validate_dockerfile_text(read_text(DOCKERFILE_PATH))
    validate_compose_text(read_text(COMPOSE_PATH))
    validate_docker_compose_config()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
