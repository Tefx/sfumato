"""CLI entry point for sfumato.

Implements the CLI contract from ARCHITECTURE.md#2.13:
- Typer app with run and preview commands
- Config loading and error handling
- User-friendly error messages and exit codes
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from sfumato.config import ConfigError, generate_default_config, load_config
from sfumato.orchestrator import RunOptions, init_project, run_once

if TYPE_CHECKING:
    from sfumato.layout_ai import LayoutParams
    from sfumato.news import Story

app = typer.Typer(
    name="sfumato",
    help="Turn Samsung The Frame into a living art + news terminal.",
    no_args_is_help=True,
)


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
        SystemExit: On config load failure with user-friendly message.
    """
    try:
        config = load_config(config_path)
        if verbose:
            typer.echo(f"Loaded config from: {config_path or 'default search path'}")
        return config
    except ConfigError as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=2) from e


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
        self._queue: list[_QueuedBatch] = []

    def dequeue(self) -> _QueuedBatch | None:
        """Remove and return the next batch."""
        if not self._queue:
            return None
        return self._queue.pop(0)

    @property
    def size(self) -> int:
        """Number of batches in queue."""
        return len(self._queue)

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        pass

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        pass


class _UsedPaintings:
    """Minimal used paintings tracking."""

    def __init__(self, state_dir: Path) -> None:
        self._used: set[str] = set()

    def mark_used(self, content_hash: str) -> None:
        """Mark a painting as used."""
        self._used.add(content_hash)

    def is_used(self, content_hash: str) -> bool:
        """Check if painting has been used."""
        return content_hash in self._used

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        pass

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        pass


class _LayoutCache:
    """Minimal layout cache implementation for CLI."""

    def __init__(self, state_dir: Path) -> None:
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

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        pass

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        pass


class _EmbeddingCache:
    """Minimal embedding cache implementation for CLI."""

    def __init__(self, state_dir: Path) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        """Get cached embedding vector."""
        return self._cache.get(key)

    def put(self, key: str, vector: Any) -> None:
        """Cache embedding vector."""
        self._cache[key] = vector

    def has(self, key: str) -> bool:
        """Check if embedding is cached."""
        return key in self._cache

    def save(self) -> None:
        """Persist state (no-op for minimal implementation)."""
        pass

    def load(self) -> None:
        """Load state (no-op for minimal implementation)."""
        pass


@dataclass
class AppState:
    """Minimal application state implementation for CLI.

    Satisfies orchestrator.AppStateProtocol for run_once().
    """

    news_queue: _NewsQueue
    used_paintings: _UsedPaintings
    layout_cache: _LayoutCache
    embedding_cache: _EmbeddingCache

    @classmethod
    def load(cls, state_dir: Path) -> "AppState":
        """Load state from directory (minimal implementation)."""
        state_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            news_queue=_NewsQueue(state_dir),
            used_paintings=_UsedPaintings(state_dir),
            layout_cache=_LayoutCache(state_dir),
            embedding_cache=_EmbeddingCache(state_dir),
        )

    def save_all(self) -> None:
        """Persist all state components."""
        self.news_queue.save()
        self.used_paintings.save()
        self.layout_cache.save()
        self.embedding_cache.save()


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
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Initialize sfumato project: create config, fetch seed paintings, analyze them.

    This command:
    1. Creates config file if not present (with sensible defaults)
    2. Creates state directory structure (~/.sfumato/state)
    3. Fetches seed_size paintings from configured sources
    4. Analyzes each painting (layout AI + description)
    5. Computes embeddings for semantic matching

    This is a potentially long operation (~50 LLM calls for 50 paintings).
    Progress is printed to stdout.

    Use this for first-time setup before running `sfumato run` or `sfumato watch`.
    """
    # For init, we use default config if no path specified
    # This ensures we have a valid config for creating directories
    if config is None:
        config = Path.home() / ".config" / "sfumato" / "config.toml"

    # Try to load existing config, or use defaults
    try:
        loaded_config = _load_config_or_exit(config, verbose)
    except SystemExit:
        # Config doesn't exist yet, create default
        from sfumato.config import AppConfig

        loaded_config = AppConfig()
        if verbose:
            typer.echo("Using default configuration")

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
            raise typer.Exit(code=5) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=5) from e
        except Exception as e:
            typer.echo(f"Initialization error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=1) from e

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

    Pipeline:
    1. Load configuration
    2. Dequeue next news batch (or on-demand refresh if empty)
    3. Select painting (specific or from pool)
    4. Analyze layout with LLM
    5. Extract color palette
    6. Render 4K PNG
    7. Upload to TV (unless --no-upload)
    8. Mark painting as used

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
        raise typer.Exit(code=3) from e

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
            raise typer.Exit(code=4) from e
        except FileNotFoundError as e:
            typer.echo(f"File not found: {e}", err=True)
            raise typer.Exit(code=5) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=5) from e
        except Exception as e:
            typer.echo(f"Pipeline error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=1) from e

    asyncio.run(_run_pipeline())


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

    Equivalent to: sfumato run --no-upload --preview

    This command:
    1. Runs the full pipeline (news + painting + render)
    2. Skips TV upload
    3. Opens the generated PNG in your system's default image viewer

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
        raise typer.Exit(code=3) from e

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
            raise typer.Exit(code=4) from e
        except FileNotFoundError as e:
            typer.echo(f"File not found: {e}", err=True)
            raise typer.Exit(code=5) from e
        except OSError as e:
            typer.echo(f"IO error: {e}", err=True)
            raise typer.Exit(code=5) from e
        except Exception as e:
            typer.echo(f"Pipeline error: {e}", err=True)
            if verbose:
                import traceback

                typer.echo(traceback.format_exc(), err=True)
            raise typer.Exit(code=1) from e

    asyncio.run(_run_preview())


def main() -> None:
    """Entry point for `sfumato` command."""
    app()


if __name__ == "__main__":
    main()
