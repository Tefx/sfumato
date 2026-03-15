"""CLI entry point for sfumato.

Implements the CLI contract from ARCHITECTURE.md#2.13:
- Typer app with init, run, watch, preview commands
- TV management subcommands (status, list, clean)
- Config loading and error handling
- User-friendly error messages and exit codes

EXIT CODE SEMANTICS (CONTRACT):
    0: Success
    1: General/unexpected error
    2: Configuration error (invalid config file)
    3: State initialization error (cannot create/load state)
    4: Input validation error (invalid arguments)
    5: File/IO error (missing file, permission denied)

COMMAND BEHAVIOR CONTRACTS:
    - init: Idempotent, creates config/state dirs if missing
    - run: Single execution, outputs render path on success
    - watch: Daemon mode, runs until SIGINT/SIGTERM
    - preview: Single execution, opens result in system viewer
    - tv status: Outputs JSON if --json, else human-readable
    - tv list: Outputs JSON if --json, else table format
    - tv clean: Outputs count of deleted images

FLAG SEMANTICS (see CLI_FLAG_SEMANTICS constant for details):
    --config: Path to TOML config file
    --verbose: Enable verbose output
    --no-upload: Skip TV upload (local only)
    --no-news: Pure painting mode
    --painting: Specific painting file
    --cli: Override AI CLI backend
    --model: Override AI model
    --json: JSON output format
    --keep: Number of uploads to retain
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import typer

from sfumato.config import ConfigError, generate_default_config, load_config
from sfumato.orchestrator import RunOptions, init_project, run_once

if TYPE_CHECKING:
    from sfumato.layout_ai import LayoutParams
    from sfumato.news import CurationResult
    from sfumato.news import Story

app = typer.Typer(
    name="sfumato",
    help="Turn Samsung The Frame into a living art + news terminal.",
    no_args_is_help=True,
)

# Subcommand group: sfumato tv ...
tv_app = typer.Typer(help="TV management commands.")
app.add_typer(tv_app, name="tv")


# =============================================================================
# EXIT CODE CONSTANTS (CONTRACT)
# =============================================================================

EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_STATE_ERROR = 3
EXIT_INPUT_ERROR = 4
EXIT_FILE_ERROR = 5


# =============================================================================
# FLAG SEMANTICS (CONTRACT)
# =============================================================================

CLI_FLAG_SEMANTICS: dict[str, str] = {
    "--config": "Path to TOML config file. Searches default locations if not specified.",
    "--verbose": "Enable verbose output with debug information.",
    "--no-upload": "Render locally but skip TV upload. Implies local output only.",
    "--no-news": "Pure painting mode (no news overlay).",
    "--painting": "Use specific painting file instead of pool selection.",
    "--cli": "Override AI CLI backend (gemini|codex|claude-code).",
    "--model": "Override AI model for layout analysis and curation.",
    "--json": "Output in JSON format for programmatic consumption.",
    "--keep": "Number of recent uploads to retain on clean.",
}
"""Documented flag semantics for CLI commands."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _verbose_print(verbose: bool, message: str) -> None:
    """Print message only if verbose mode is enabled."""
    if verbose:
        typer.echo(message)


def _load_config_or_exit(config_path: Path | None, verbose: bool) -> Any:
    """Load config, printing user-friendly error on failure.

    Args:
        config_path: Optional explicit config file path.
        verbose: Whether to print verbose output.

    Returns:
        Loaded AppConfig.

    Raises:
        SystemExit: On config load failure with user-friendly message (exit code 2).
    """
    try:
        config = load_config(config_path)
        if verbose:
            typer.echo(f"Loaded config from: {config_path or 'default search path'}")
        return config
    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=EXIT_CONFIG_ERROR) from e


def _output_json(data: dict[str, Any] | list[Any]) -> None:
    """Output data as JSON to stdout.

    Args:
        data: Dictionary or list to output as JSON.

    Contract:
        - Outputs exactly one JSON value to stdout
        - No trailing newline required (json.dump handles it)
        - Never includes ANSI color codes
        - Keys are snake_case for dicts
    """
    import json

    json.dump(data, sys.stdout, indent=2, sort_keys=True)
    typer.echo("")  # Final newline


# =============================================================================
# MINIMAL STATE IMPLEMENTATION (inline for step scope)
# =============================================================================


@dataclass
class _QueuedBatch:
    """Minimal queued batch implementation."""

    stories: list[Story]
    tone_description: str
    enqueued_at: datetime.datetime


class _NewsQueue:
    """Minimal news queue implementation for CLI."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._queue: list[_QueuedBatch] = []

    def dequeue(self) -> _QueuedBatch | None:
        """Remove and return the next batch."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    def enqueue(self, result: "CurationResult", batch_size: int) -> int:
        """Split curation result into batches and append to queue."""
        if not hasattr(result, "stories") or not result.stories:
            return 0
        enqueued = 0
        for idx in range(0, len(result.stories), batch_size):
            self._queue.append(
                _QueuedBatch(
                    stories=result.stories[idx : idx + batch_size],
                    tone_description=getattr(result, "tone_description", ""),
                    enqueued_at=datetime.datetime.now().astimezone(),
                )
            )
            enqueued += 1
        return enqueued

    def expire(self, expire_days: int) -> int:
        """Drop batches older than expire_days. Returns removed count."""
        if not self._queue:
            return 0
        cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(
            days=expire_days
        )
        before = len(self._queue)
        self._queue = [b for b in self._queue if b.enqueued_at >= cutoff]
        return before - len(self._queue)

    def peek(self) -> _QueuedBatch | None:
        """Return the next batch without removal."""
        if not self._queue:
            return None
        return self._queue[0]

    @property
    def size(self) -> int:
        """Number of batches in queue."""
        return len(self._queue)

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        return None

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        return None


class _UsedPaintings:
    """Minimal used paintings tracking."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._used: set[str] = set()

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting as used."""
        self._used.add(content_hash)

    def is_used(self, content_hash: str) -> bool:
        """Check if painting has been used."""
        return content_hash in self._used

    def reset(self) -> None:
        """Reset used paintings tracking."""
        self._used.clear()

    @property
    def count(self) -> int:
        """Number of used paintings tracked."""
        return len(self._used)

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        return None

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        return None


class _LayoutCache:
    """Minimal layout cache implementation for CLI."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._cache: dict[str, LayoutParams] = {}

    def get(self, content_hash: str) -> "LayoutParams | None":
        """Get cached layout params."""
        return self._cache.get(content_hash)

    def put(self, content_hash: str, layout: "LayoutParams") -> None:
        """Cache layout params."""
        self._cache[content_hash] = layout

    def has(self, content_hash: str) -> bool:
        """Check if layout is cached."""
        return content_hash in self._cache

    @property
    def size(self) -> int:
        """Number of cached layouts."""
        return len(self._cache)

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        return None

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        return None


class _EmbeddingCache:
    """Minimal embedding cache implementation for CLI."""

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._cache: dict[str, np.ndarray] = {}

    def get(self, key: str) -> np.ndarray | None:
        """Get cached embedding vector."""
        return self._cache.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        """Cache embedding vector."""
        if vector.ndim != 1:
            raise ValueError("Embedding vectors must be 1-dimensional")
        self._cache[key] = vector

    def has(self, key: str) -> bool:
        """Check if embedding is cached."""
        return key in self._cache

    @property
    def size(self) -> int:
        """Number of cached embedding vectors."""
        return len(self._cache)

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        return None

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        return None


class _ReplayTransferResult:
    """Minimal replay transfer result."""

    def __init__(self, accepted: bool, reason: str, overlap_ratio: float) -> None:
        self.accepted = accepted
        self.reason = reason
        self.overlap_ratio = overlap_ratio
        self.matched_batch_index: int | None = None


class _ReplayBatch:
    """Minimal replay batch."""

    def __init__(
        self,
        stories: list["Story"],
        tone_description: str,
        source_enqueued_at: datetime.datetime,
    ) -> None:
        self.stories = stories
        self.tone_description = tone_description
        self.source_enqueued_at = source_enqueued_at
        self.transferred_at = datetime.datetime.now().astimezone()
        self.replay_count: int = 0
        self.last_replayed_at: datetime.datetime | None = None


class _ReplayQueue:
    """Minimal replay queue implementation for CLI.

    Satisfies orchestrator.ReplayQueueProtocol.
    """

    def __init__(self, state_dir: Path) -> None:
        self._state_dir = state_dir
        self._batches: list[_ReplayBatch] = []
        self._next_index: int = 0

    def next(self) -> _ReplayBatch | None:
        """Return the next replay batch cyclically without removal."""
        if not self._batches:
            return None
        batch = self._batches[self._next_index % len(self._batches)]
        batch.replay_count += 1
        batch.last_replayed_at = datetime.datetime.now().astimezone()
        self._next_index = (self._next_index + 1) % len(self._batches)
        return batch

    def expire(self, expire_days: int) -> int:
        """Drop replay batches older than expire_days."""
        if not self._batches:
            return 0
        cutoff = datetime.datetime.now().astimezone() - datetime.timedelta(
            days=expire_days
        )
        before = len(self._batches)
        self._batches = [
            b for b in self._batches if b.source_enqueued_at >= cutoff
        ]
        removed = before - len(self._batches)
        if self._batches:
            self._next_index = self._next_index % len(self._batches)
        else:
            self._next_index = 0
        return removed

    def transfer_from_news_queue(self, batch: Any) -> _ReplayTransferResult:
        """Transfer a consumed primary batch into replay storage."""
        stories = getattr(batch, "stories", [])
        if not stories:
            return _ReplayTransferResult(
                accepted=False, reason="rejected-empty-batch", overlap_ratio=0.0
            )
        tone = getattr(batch, "tone_description", "")
        enqueued_at = getattr(
            batch, "enqueued_at", datetime.datetime.now().astimezone()
        )
        self._batches.append(
            _ReplayBatch(
                stories=list(stories),
                tone_description=tone,
                source_enqueued_at=enqueued_at,
            )
        )
        return _ReplayTransferResult(
            accepted=True, reason="accepted", overlap_ratio=0.0
        )

    @property
    def size(self) -> int:
        """Number of replay batches."""
        return len(self._batches)

    def persist(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        return None

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        return None


class _ArtFactRotation:
    """Minimal art-fact rotation tracking."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}

    def get_next_index(self, content_hash: str, fact_count: int) -> int | None:
        """Return the next fact index for a painting, or None if no facts."""
        if fact_count <= 0:
            return None
        current = self._counters.get(content_hash, 0)
        return current % fact_count

    def commit_rotation(self, content_hash: str, fact_count: int) -> None:
        """Advance the rotation counter after successful render."""
        if fact_count <= 0:
            return
        current = self._counters.get(content_hash, 0)
        self._counters[content_hash] = (current + 1) % fact_count

    def save(self) -> None:
        return None

    def load(self) -> None:
        return None


@dataclass
class AppState:
    """Minimal application state implementation for CLI.

    Satisfies orchestrator.AppStateProtocol for run_once().
    """

    news_queue: _NewsQueue
    used_paintings: _UsedPaintings
    layout_cache: _LayoutCache
    embedding_cache: _EmbeddingCache
    art_fact_rotation: _ArtFactRotation
    replay_queue: _ReplayQueue

    @classmethod
    def load(cls, state_dir: Path) -> "AppState":
        """Load state from directory (minimal implementation)."""
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            news_queue=_NewsQueue(state_dir),
            used_paintings=_UsedPaintings(state_dir),
            layout_cache=_LayoutCache(state_dir),
            embedding_cache=_EmbeddingCache(state_dir),
            art_fact_rotation=_ArtFactRotation(),
            replay_queue=_ReplayQueue(state_dir),
        )

    def save_all(self) -> None:
        """Persist all state components."""
        self.news_queue.save()
        self.used_paintings.save()
        self.layout_cache.save()
        self.embedding_cache.save()
        self.replay_queue.persist()


# =============================================================================
# CLI COMMANDS
# =============================================================================


@app.command()
def init(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (creates default if not found)",
        exists=False,
        dir_okay=False,
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Skip interactive prompts, use defaults.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Initialize sfumato: interactive setup, fetch seed paintings, analyze them.

    On first run, walks you through configuration interactively.
    Safe to run multiple times (idempotent).
    """
    from sfumato.config import AppConfig, ConfigError, generate_default_config

    if config is None:
        config = Path.home() / ".config" / "sfumato" / "config.toml"

    config_path = config.expanduser()

    if config_path.exists():
        # Config exists, load it
        try:
            loaded_config = _load_config_or_exit(config, verbose)
        except (SystemExit, ConfigError):
            loaded_config = AppConfig()
            _verbose_print(verbose, "Config exists but failed to load, using defaults")
    else:
        # First time — create default config
        if non_interactive:
            # Non-interactive mode: create default config file
            loaded_config = AppConfig()
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_content = generate_default_config()
            config_path.write_text(config_content)
            typer.echo(f"Created default config at {config_path}")
        else:
            typer.echo("\nWelcome to sfumato! Let's set things up.\n")

            tv_ip = typer.prompt("  TV IP address", default="", show_default=False)
            if not tv_ip:
                typer.echo("    (skipped — set later in config.toml)")
                tv_ip = ""

            cli_choice = typer.prompt(
                "  AI CLI (gemini/codex/claude-code)", default="gemini"
            )
            model_choice = typer.prompt("  AI Model", default="gemini-3.1-pro-preview")
            language = typer.prompt("  Display language", default="zh")

            # Build config with user's choices
            from sfumato.config import (
                TvConfig,
                AiConfig,
                NewsConfig,
            )

            loaded_config = AppConfig(
                tv=TvConfig(ip=tv_ip) if tv_ip else TvConfig(ip=""),
                ai=AiConfig(cli=cli_choice, model=model_choice),
                news=NewsConfig(language=language),
            )

            # Write config file
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Generate TOML with user's values
            config_content = generate_default_config()
            # Patch in user values
            if tv_ip:
                config_content = config_content.replace(
                    'ip = "192.168.1.100"', f'ip = "{tv_ip}"'
                )
            config_content = config_content.replace(
                'cli = "gemini"', f'cli = "{cli_choice}"'
            )
            config_content = config_content.replace(
                'model = "gemini-3.1-pro-preview"', f'model = "{model_choice}"'
            )
            config_content = config_content.replace(
                'language = "zh"', f'language = "{language}"'
            )
            config_path.write_text(config_content)
            typer.echo(f"\n  ✓ Config saved to {config_path}")
            typer.echo(f"    (edit anytime: {config_path})\n")

            # Reload the config we just wrote
            loaded_config = _load_config_or_exit(config, verbose)

    # Initialize state directory
    state_dir = loaded_config.data_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        typer.echo(f"State directory: {state_dir}")

    # Run init_project asynchronously
    async def _init() -> None:
        try:
            await init_project(loaded_config)
        except FileNotFoundError as e:
            typer.echo(f"File not found: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except Exception as e:
            typer.echo(f"Initialization error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=EXIT_GENERAL_ERROR) from e

    asyncio.run(_init())


@app.command()
def run(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,  # Don't validate existence here; load_config handles it
        dir_okay=False,
    ),
    no_upload: bool = typer.Option(
        False,
        "--no-upload",
        help="Render only, skip TV upload. Implies local output.",
    ),
    no_news: bool = typer.Option(
        False,
        "--no-news",
        help="Pure painting mode (no news overlay).",
    ),
    painting: Path = typer.Option(
        None,
        "--painting",
        "-p",
        help="Use a specific painting image file instead of pool selection.",
        exists=True,
        dir_okay=False,
    ),
    cli_override: str = typer.Option(
        None,
        "--cli",
        help="Override AI CLI backend (gemini|codex|claude-code).",
    ),
    model_override: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Override AI model for layout analysis and curation.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Execute a single full pipeline cycle.

    BEHAVIOR CONTRACT:
        - Single execution: one rotation cycle then exit
        - Outputs render path on success (stdout)
        - Errors go to stderr with context

    PIPELINE ORDER (ARCHITECTURE.md#2.12):
        1. Load configuration
        2. Dequeue next news batch (or on-demand refresh if empty)
        3. Select painting (specific or from pool)
        4. Analyze layout with LLM
        5. Extract color palette
        6. Render 4K PNG
        7. Upload to TV (unless --no-upload)
        8. Mark painting as used

    EXIT CODES:
        0: Success
        1: Pipeline error (LLM failure, render failure, etc.)
        2: Configuration error
        3: State initialization error
        4: Input validation error
        5: File not found or IO error

    Use --no-upload for local testing without TV connection.
    Use --painting to test with a specific image file.
    """
    loaded_config = _load_config_or_exit(config, verbose)

    # Apply CLI overrides
    if cli_override:
        _verbose_print(verbose, f"Overriding AI CLI backend to: {cli_override}")
        typer.echo(
            f"Warning: --cli override not yet implemented. Using: {loaded_config.ai.cli}",
            err=True,
        )

    if model_override:
        _verbose_print(verbose, f"Overriding AI model to: {model_override}")
        typer.echo(
            f"Warning: --model override not yet implemented. Using: {loaded_config.ai.model}",
            err=True,
        )

    # Build run options
    options = RunOptions(
        no_upload=no_upload,
        no_news=no_news,
        painting_path=painting.resolve() if painting else None,
        preview=False,
    )

    _verbose_print(verbose, f"Run options: no_upload={no_upload}, no_news={no_news}")

    # Initialize state
    try:
        state = AppState.load(loaded_config.data_dir)
    except Exception as e:
        typer.echo(f"Failed to initialize state: {e}", err=True)
        raise typer.Exit(code=EXIT_STATE_ERROR) from e

    # Run the pipeline asynchronously
    async def _run_pipeline() -> None:
        try:
            result = await run_once(
                config=loaded_config,
                state=state,  # type: ignore[arg-type]
                options=options,
            )

            # Output result
            _verbose_print(verbose, f"Action performed: {result.action}")
            _verbose_print(verbose, f"Story count: {result.story_count}")
            _verbose_print(verbose, f"Uploaded: {result.uploaded}")

            if result.render_result:
                typer.echo(f"Rendered: {result.render_result.png_path}")
                typer.echo(f"HTML: {result.render_result.html_path}")
                typer.echo(f"Template: {result.render_result.template_used}")
                typer.echo(f"Stories: {result.render_result.story_count}")

                if result.painting:
                    _verbose_print(
                        verbose,
                        f"Painting: {result.painting.title} by {result.painting.artist}",
                    )

                if result.match_score is not None:
                    _verbose_print(verbose, f"Match score: {result.match_score:.3f}")
            else:
                typer.echo("No render produced (IDLE or skipped)")

        except ValueError as e:
            typer.echo(f"Input error: {e}", err=True)
            raise typer.Exit(code=EXIT_INPUT_ERROR) from e
        except FileNotFoundError as e:
            typer.echo(f"File not found: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except Exception as e:
            typer.echo(f"Pipeline error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=EXIT_GENERAL_ERROR) from e

    asyncio.run(_run_pipeline())


@app.command()
def watch(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,
        dir_okay=False,
    ),
    cli_override: str = typer.Option(
        None,
        "--cli",
        help="Override AI CLI backend (gemini|codex|claude-code).",
    ),
    model_override: str = typer.Option(
        None,
        "--model",
        "-m",
        help="Override AI model for layout analysis and curation.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output with action logging.",
    ),
) -> None:
    """Start the daemon (long-running watch mode).

    BEHAVIOR CONTRACT:
        - Runs indefinitely until SIGINT/SIGTERM
        - Handles graceful shutdown on interrupt
        - Respects quiet_hours and active_hours from config
        - Outputs status messages to stdout (one per action)
        - Errors go to stderr

    DAEMON LOOP (ARCHITECTURE.md#6):
        1. Load state from disk
        2. Expire old news queue entries
        3. Check scheduler for next action
        4. Execute action (refresh_news, rotate, backfill)
        5. Save state to disk
        6. Sleep until next action
        7. Repeat from step 3

    SIGNAL HANDLING:
        - SIGINT (Ctrl+C): Graceful shutdown after current action
        - SIGTERM: Graceful shutdown after current action

    EXIT CODES:
        0: Graceful shutdown via signal
        1: Unexpected error during daemon operation
        2: Configuration error
        3: State initialization error

    LOGGING:
        - Startup: "Starting sfumato daemon..."
        - Action: "Action: <action_type> at <timestamp>"
        - Shutdown: "Shutting down..."
        - Verbose mode: Additional debug information
    """
    loaded_config = _load_config_or_exit(config, verbose)

    # Apply CLI overrides
    if cli_override:
        _verbose_print(verbose, f"Overriding AI CLI backend to: {cli_override}")
        typer.echo(
            f"Warning: --cli override not yet implemented. Using: {loaded_config.ai.cli}",
            err=True,
        )

    if model_override:
        _verbose_print(verbose, f"Overriding AI model to: {model_override}")
        typer.echo(
            f"Warning: --model override not yet implemented. Using: {loaded_config.ai.model}",
            err=True,
        )

    # Import orchestrator.watch for actual implementation
    from sfumato.orchestrator import watch as orchestrator_watch

    typer.echo("Starting sfumato daemon...")
    typer.echo(f"TV: {loaded_config.tv.ip}:{loaded_config.tv.port}")
    typer.echo(
        f"Schedule: rotate every {loaded_config.schedule.rotate_interval_minutes}min"
    )
    typer.echo(f"News refresh: every {loaded_config.schedule.news_interval_hours}h")
    typer.echo("Press Ctrl+C to stop.")

    # Run the daemon loop
    async def _watch() -> None:
        try:
            await orchestrator_watch(loaded_config)
        except KeyboardInterrupt:
            typer.echo("\nShutdown requested...")
        except Exception as e:
            typer.echo(f"Daemon error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=EXIT_GENERAL_ERROR) from e

    try:
        asyncio.run(_watch())
    except KeyboardInterrupt:
        typer.echo("Shutting down...")
        typer.echo("Goodbye.")


@app.command()
def preview(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,
        dir_okay=False,
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Render and open the result in the system image viewer.

    BEHAVIOR CONTRACT:
        - Equivalent to: sfumato run --no-upload --preview
        - Single execution with preview enabled
        - Opens PNG in system default viewer
        - No TV upload

    EXIT CODES:
        0: Success
        1: Pipeline error
        2: Configuration error
        3: State initialization error
        4: Input validation error
        5: File not found or IO error

    Useful for testing layout, colors, and overall appearance
    before pushing to the TV.
    """
    loaded_config = _load_config_or_exit(config, verbose)

    # Build run options with preview enabled
    options = RunOptions(
        no_upload=True,
        no_news=False,
        painting_path=None,
        preview=True,
    )

    try:
        state = AppState.load(loaded_config.data_dir)
    except Exception as e:
        typer.echo(f"Failed to initialize state: {e}", err=True)
        raise typer.Exit(code=EXIT_STATE_ERROR) from e

    async def _run_preview() -> None:
        try:
            result = await run_once(
                config=loaded_config,
                state=state,  # type: ignore[arg-type]
                options=options,
            )

            if result.render_result:
                typer.echo(f"Rendered: {result.render_result.png_path}")
                _verbose_print(
                    verbose,
                    f"Template: {result.render_result.template_used}, "
                    f"Stories: {result.render_result.story_count}",
                )
                typer.echo("Opening preview...")
            else:
                typer.echo("No render produced (nothing to preview)")

        except ValueError as e:
            typer.echo(f"Input error: {e}", err=True)
            raise typer.Exit(code=EXIT_INPUT_ERROR) from e
        except FileNotFoundError as e:
            typer.echo(f"File not found: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=EXIT_FILE_ERROR) from e
        except Exception as e:
            typer.echo(f"Pipeline error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=EXIT_GENERAL_ERROR) from e

    asyncio.run(_run_preview())


# =============================================================================
# TV SUBCOMMANDS (CONTRACT)
# =============================================================================


@tv_app.command("status")
def tv_status(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,
        dir_okay=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format for programmatic consumption.",
    ),
) -> None:
    """Check TV connection and Art Mode status.

    BEHAVIOR CONTRACT:
        - Non-throwing: always outputs status, never raises
        - Human-readable output by default
        - JSON output with --json flag
        - Quick check: completes within 10 seconds (timeout boundary)

    OUTPUT FORMAT (human-readable):
        TV Status:
          Reachable: yes/no
          Art Mode: supported/unsupported/active/inactive
          Uploads: <count>

    OUTPUT FORMAT (JSON):
        {
          "reachable": bool,
          "art_mode_supported": bool,
          "art_mode_active": bool,
          "uploaded_count": int,
          "error": string or null
        }

    EXIT CODES:
        0: TV reachable (check output for Art Mode status)
        1: Unexpected error during status check
        2: Configuration error

    Note: Exit code 0 does not guarantee Art Mode is active.
    Check the "art_mode_active" field in output for that.
    """
    from sfumato.tv import TvConnectionError, check_status

    loaded_config = _load_config_or_exit(config, False)

    try:
        status = check_status(loaded_config.tv)

        if json_output:
            _output_json(
                {
                    "reachable": status.reachable,
                    "art_mode_supported": status.art_mode_supported,
                    "art_mode_active": status.art_mode_active,
                    "uploaded_count": status.uploaded_count,
                    "error": status.error,
                }
            )
        else:
            typer.echo("TV Status:")
            typer.echo(f"  Reachable: {'yes' if status.reachable else 'no'}")
            if status.reachable:
                typer.echo(
                    f"  Art Mode: {'active' if status.art_mode_active else 'inactive'}"
                )
                typer.echo(
                    f"  Art Mode Supported: {'yes' if status.art_mode_supported else 'no'}"
                )
                typer.echo(f"  Uploads: {status.uploaded_count}")
            else:
                typer.echo(f"  Error: {status.error}")

    except TvConnectionError as e:
        if json_output:
            _output_json(
                {
                    "reachable": False,
                    "art_mode_supported": False,
                    "art_mode_active": False,
                    "uploaded_count": 0,
                    "error": str(e),
                }
            )
        else:
            typer.echo(f"TV connection error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e


@tv_app.command("list")
def tv_list(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,
        dir_okay=False,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output in JSON format for programmatic consumption.",
    ),
) -> None:
    """List uploaded images on the TV.

    BEHAVIOR CONTRACT:
        - Lists all images currently stored in TV's Art Mode
        - Human-readable table format by default
        - JSON output with --json flag
        - Raises TvConnectionError on TV unreachable

    OUTPUT FORMAT (human-readable):
        Content ID      | File Name  | Uploaded
        ----------------|------------|----------
        MY_F0001        | art1.png   | (date if available)
        MY_F0002        | art2.png   | (date if available)

    OUTPUT FORMAT (JSON):
        [
          {"content_id": "MY_F0001", "file_name": "art1.png"},
          {"content_id": "MY_F0002", "file_name": null}
        ]

    EXIT CODES:
        0: Success (list retrieved)
        1: TV connection error
        2: Configuration error

    Requires TV to be reachable and paired.
    """
    from sfumato.tv import TvConnectionError, list_uploaded

    loaded_config = _load_config_or_exit(config, False)

    try:
        images = list_uploaded(loaded_config.tv)

        if json_output:
            _output_json(
                [
                    {"content_id": img.content_id, "file_name": img.file_name}
                    for img in images
                ]
            )
        else:
            if not images:
                typer.echo("No images uploaded to TV.")
                return

            # Table format
            typer.echo("Uploaded Images:")
            typer.echo("-" * 50)
            typer.echo(f"{'Content ID':<15} | {'File Name':<20}")
            typer.echo("-" * 50)
            for img in images:
                file_name = img.file_name or "(unknown)"
                typer.echo(f"{img.content_id:<15} | {file_name:<20}")
            typer.echo("-" * 50)
            typer.echo(f"Total: {len(images)} image(s)")

    except TvConnectionError as e:
        typer.echo(f"TV connection error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e


@tv_app.command("clean")
def tv_clean(
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (searches default locations if not specified)",
        exists=False,
        dir_okay=False,
    ),
    keep: int = typer.Option(
        5,
        "--keep",
        "-k",
        help="Number of most recent uploads to retain.",
    ),
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Show details of deleted images.",
    ),
) -> None:
    """Remove old uploads from the TV.

    BEHAVIOR CONTRACT:
        - Removes oldest uploads, keeping most recent `--keep` images
        - Ordering by TV metadata date (or content_id lexical if unavailable)
        - Non-throwing for individual delete failures: logs and continues
        - Outputs count of successfully deleted images

    KEEP-POLICY (ARCHITECTURE.md#2.8):
        - Retains `--keep` most recent uploads
        - Uses TV metadata date for ordering when available
        - Falls back to lexical ascending by content_id
        - Protected: retained images are NEVER deleted due to failures

    OUTPUT FORMAT (human-readable):
        Deleted 3 images, keeping 5 most recent.

    OUTPUT FORMAT (verbose):
        Deleted 3 images:
          - MY_F0001 (oldest)
          - MY_F0002
        Kept 5 images:
          - MY_F0003
          - MY_F0004
          ...

    EXIT CODES:
        0: Success (cleanup complete)
        1: TV connection error (on list failure)
        2: Configuration error

    Note: Individual delete failures are logged but do not cause exit 1.
    """
    from sfumato.tv import TvConnectionError, clean_old_uploads, list_uploaded

    loaded_config = _load_config_or_exit(config, verbose)

    if keep < 0:
        typer.echo("Error: --keep must be >= 0", err=True)
        raise typer.Exit(code=EXIT_INPUT_ERROR)

    try:
        # Get initial count for verbose output
        if verbose:
            images_before = list_uploaded(loaded_config.tv)
            count_before = len(images_before)
            _verbose_print(verbose, f"Images before cleanup: {count_before}")

        # Perform cleanup
        deleted_count = clean_old_uploads(loaded_config.tv, keep=keep)

        if verbose:
            images_after = list_uploaded(loaded_config.tv)
            count_after = len(images_after)
            _verbose_print(verbose, f"Images after cleanup: {count_after}")
            _verbose_print(verbose, f"Deleted: {deleted_count} images")

        # Output result
        if deleted_count == 0:
            typer.echo(f"No images to clean. Keeping {keep} most recent.")
        else:
            typer.echo(f"Deleted {deleted_count} images, keeping {keep} most recent.")

    except TvConnectionError as e:
        typer.echo(f"TV connection error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(code=EXIT_GENERAL_ERROR) from e


def main() -> None:
    """Entry point for `sfumato` command."""
    app()


if __name__ == "__main__":
    main()
