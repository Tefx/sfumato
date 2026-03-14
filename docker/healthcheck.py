from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import sys


STATE_DIR = Path(os.environ.get("SFUMATO_DATA_DIR", "/data")) / "state"
HEALTH_PATH = STATE_DIR / "last_action.json"
MAX_AGE_SECONDS = int(os.environ.get("SFUMATO_HEALTH_MAX_AGE_SECONDS", "2100"))


def load_payload() -> dict[str, object]:
    if not HEALTH_PATH.exists():
        raise RuntimeError(f"missing health file: {HEALTH_PATH}")
    if not os.access(STATE_DIR, os.W_OK):
        raise RuntimeError(f"state directory is not writable: {STATE_DIR}")
    return json.loads(HEALTH_PATH.read_text(encoding="utf-8"))


def main() -> int:
    payload = load_payload()
    timestamp_raw = payload.get("timestamp")
    if not isinstance(timestamp_raw, str):
        raise RuntimeError("health payload missing timestamp")

    timestamp = datetime.fromisoformat(timestamp_raw)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    age = datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)
    if age > timedelta(seconds=MAX_AGE_SECONDS):
        raise RuntimeError(f"health payload stale: {age.total_seconds():.0f}s old")

    proc_cmdline = Path("/proc/1/cmdline")
    if proc_cmdline.exists():
        cmdline = proc_cmdline.read_text(encoding="utf-8", errors="ignore").replace(
            "\x00", " "
        )
        if "sfumato" not in cmdline or "watch" not in cmdline:
            raise RuntimeError(f"pid 1 is not sfumato watch: {cmdline.strip()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
