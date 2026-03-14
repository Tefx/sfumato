#!/bin/sh
set -eu

STATE_DIR="${SFUMATO_DATA_DIR:-/data}/state"
OUTPUT_DIR="${SFUMATO_DATA_DIR:-/data}/output"
PAINTINGS_DIR="${SFUMATO_DATA_DIR:-/data}/paintings"
CONFIG_PATH="${SFUMATO_CONFIG:-/data/config.toml}"

mkdir -p "$STATE_DIR" "$OUTPUT_DIR" "$PAINTINGS_DIR"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "sfumato: missing config file at $CONFIG_PATH" >&2
  echo "sfumato: mount a config file before starting watch mode" >&2
  exit 1
fi

exec "$@"
