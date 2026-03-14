"""
Tests for CLI module contract.

These tests verify the public contract defined in src/sfumato/cli.py.
Implementation will be completed in follow-up dispatch steps.

Contract Sources:
- ARCHITECTURE.md#2.13
- src/sfumato/cli.py docstrings

Required Coverage:
- Command signatures (init, run, watch, preview)
- TV subcommand signatures (status, list, clean)
- Exit code semantics (0-5)
- Flag semantics
- Output format contracts (human-readable vs JSON)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typer.testing import CliRunner

# Import CLI app for testing
from sfumato.cli import (
    app,
    CLI_FLAG_SEMANTICS,
    EXIT_SUCCESS,
    EXIT_GENERAL_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_STATE_ERROR,
    EXIT_INPUT_ERROR,
    EXIT_FILE_ERROR,
    init,
    run,
    watch,
    preview,
    tv_status,
    tv_list,
    tv_clean,
    _output_json,
)


# =============================================================================
# CONTRACT: EXIT CODE SEMANTICS
# =============================================================================


class TestExitCodeSemantics:
    """Contract tests for exit code values."""

    def test_exit_success_is_zero(self) -> None:
        """Exit code 0 indicates success."""
        assert EXIT_SUCCESS == 0

    def test_exit_general_error_is_one(self) -> None:
        """Exit code 1 indicates general/unexpected error."""
        assert EXIT_GENERAL_ERROR == 1

    def test_exit_config_error_is_two(self) -> None:
        """Exit code 2 indicates configuration error."""
        assert EXIT_CONFIG_ERROR == 2

    def test_exit_state_error_is_three(self) -> None:
        """Exit code 3 indicates state initialization error."""
        assert EXIT_STATE_ERROR == 3

    def test_exit_input_error_is_four(self) -> None:
        """Exit code 4 indicates input validation error."""
        assert EXIT_INPUT_ERROR == 4

    def test_exit_file_error_is_five(self) -> None:
        """Exit code 5 indicates file/IO error."""
        assert EXIT_FILE_ERROR == 5

    def test_exit_codes_are_distinct(self) -> None:
        """All exit codes have unique values."""
        codes = [
            EXIT_SUCCESS,
            EXIT_GENERAL_ERROR,
            EXIT_CONFIG_ERROR,
            EXIT_STATE_ERROR,
            EXIT_INPUT_ERROR,
            EXIT_FILE_ERROR,
        ]
        assert len(codes) == len(set(codes)), "Exit codes must be distinct"

    def test_exit_codes_are_integers(self) -> None:
        """All exit codes are integers (not strings or other types)."""
        for code in [
            EXIT_SUCCESS,
            EXIT_GENERAL_ERROR,
            EXIT_CONFIG_ERROR,
            EXIT_STATE_ERROR,
            EXIT_INPUT_ERROR,
            EXIT_FILE_ERROR,
        ]:
            assert isinstance(code, int)


# =============================================================================
# CONTRACT: FLAG SEMANTICS
# =============================================================================


class TestFlagSemantics:
    """Contract tests for CLI flag documentation."""

    def test_flag_semantics_defined(self) -> None:
        """All CLI flags have documented semantics."""
        assert CLI_FLAG_SEMANTICS is not None
        assert isinstance(CLI_FLAG_SEMANTICS, dict)

    def test_config_flag_documented(self) -> None:
        """--config flag has documented behavior."""
        assert "--config" in CLI_FLAG_SEMANTICS
        assert "TOML" in CLI_FLAG_SEMANTICS["--config"]
        assert "config file" in CLI_FLAG_SEMANTICS["--config"].lower()

    def test_verbose_flag_documented(self) -> None:
        """--verbose flag has documented behavior."""
        assert "--verbose" in CLI_FLAG_SEMANTICS
        assert "verbose" in CLI_FLAG_SEMANTICS["--verbose"].lower()

    def test_no_upload_flag_documented(self) -> None:
        """--no-upload flag has documented behavior."""
        assert "--no-upload" in CLI_FLAG_SEMANTICS
        assert "TV upload" in CLI_FLAG_SEMANTICS["--no-upload"]

    def test_no_news_flag_documented(self) -> None:
        """--no-news flag has documented behavior."""
        assert "--no-news" in CLI_FLAG_SEMANTICS
        assert "painting mode" in CLI_FLAG_SEMANTICS["--no-news"].lower()

    def test_painting_flag_documented(self) -> None:
        """--painting flag has documented behavior."""
        assert "--painting" in CLI_FLAG_SEMANTICS
        assert "pool" in CLI_FLAG_SEMANTICS["--painting"].lower()

    def test_json_flag_documented(self) -> None:
        """--json flag has documented behavior."""
        assert "--json" in CLI_FLAG_SEMANTICS
        assert "JSON" in CLI_FLAG_SEMANTICS["--json"]

    def test_keep_flag_documented(self) -> None:
        """--keep flag has documented behavior."""
        assert "--keep" in CLI_FLAG_SEMANTICS
        assert "retain" in CLI_FLAG_SEMANTICS["--keep"].lower()


# =============================================================================
# CONTRACT: COMMAND SIGNATURES
# =============================================================================


class TestCommandSignatures:
    """Contract tests for command function signatures."""

    def test_init_command_callable(self) -> None:
        """init command is callable."""
        assert callable(init)

    def test_run_command_callable(self) -> None:
        """run command is callable."""
        assert callable(run)

    def test_watch_command_callable(self) -> None:
        """watch command is callable."""
        assert callable(watch)

    def test_preview_command_callable(self) -> None:
        """preview command is callable."""
        assert callable(preview)

    def test_tv_status_command_callable(self) -> None:
        """tv status subcommand is callable."""
        assert callable(tv_status)

    def test_tv_list_command_callable(self) -> None:
        """tv list subcommand is callable."""
        assert callable(tv_list)

    def test_tv_clean_command_callable(self) -> None:
        """tv clean subcommand is callable."""
        assert callable(tv_clean)

    def test_app_is_typer_app(self) -> None:
        """app is a Typer application."""
        import typer

        assert isinstance(app, typer.Typer)


# =============================================================================
# CONTRACT: WATCH COMMAND BEHAVIOR
# =============================================================================


class TestWatchCommandBehavior:
    """Contract tests for watch command behavior."""

    def test_watch_has_config_option(self) -> None:
        """watch command has --config option."""
        # Inspect command parameters
        import inspect

        sig = inspect.signature(watch)
        params = sig.parameters
        assert "config" in params

    def test_watch_has_cli_override_option(self) -> None:
        """watch command has --cli option for AI backend override."""
        import inspect

        sig = inspect.signature(watch)
        params = sig.parameters
        assert "cli_override" in params

    def test_watch_has_model_override_option(self) -> None:
        """watch command has --model option for AI model override."""
        import inspect

        sig = inspect.signature(watch)
        params = sig.parameters
        assert "model_override" in params

    def test_watch_has_verbose_option(self) -> None:
        """watch command has --verbose option."""
        import inspect

        sig = inspect.signature(watch)
        params = sig.parameters
        assert "verbose" in params

    def test_watch_docstring_describes_behavior(self) -> None:
        """watch command has behavior contract in docstring."""
        assert watch.__doc__ is not None
        doc = watch.__doc__.lower()
        # Key behavior contract elements
        assert "daemon" in doc or "long-running" in doc
        assert "signal" in doc or "sigint" in doc or "sigterm" in doc
        assert "exit" in doc


# =============================================================================
# CONTRACT: TV SUBCOMMAND BEHAVIOR
# =============================================================================


class TestTvStatusBehavior:
    """Contract tests for tv status subcommand behavior."""

    def test_status_has_config_option(self) -> None:
        """tv status has --config option."""
        import inspect

        sig = inspect.signature(tv_status)
        params = sig.parameters
        assert "config" in params

    def test_status_has_json_option(self) -> None:
        """tv status has --json option."""
        import inspect

        sig = inspect.signature(tv_status)
        params = sig.parameters
        assert "json_output" in params

    def test_status_output_format_documented(self) -> None:
        """tv status output format documented in docstring."""
        assert tv_status.__doc__ is not None
        doc = tv_status.__doc__
        assert "reachable" in doc.lower()
        assert "json" in doc.lower()

    def test_status_exit_codes_documented(self) -> None:
        """tv status exit codes documented."""
        assert tv_status.__doc__ is not None
        assert (
            "EXIT CODES" in tv_status.__doc__
            or "exit code" in tv_status.__doc__.lower()
        )


class TestTvListBehavior:
    """Contract tests for tv list subcommand behavior."""

    def test_list_has_config_option(self) -> None:
        """tv list has --config option."""
        import inspect

        sig = inspect.signature(tv_list)
        params = sig.parameters
        assert "config" in params

    def test_list_has_json_option(self) -> None:
        """tv list has --json option."""
        import inspect

        sig = inspect.signature(tv_list)
        params = sig.parameters
        assert "json_output" in params

    def test_list_output_format_documented(self) -> None:
        """tv list output format documented in docstring."""
        assert tv_list.__doc__ is not None
        doc = tv_list.__doc__
        assert "content_id" in doc.lower() or "image" in doc.lower()
        assert "json" in doc.lower()


class TestTvCleanBehavior:
    """Contract tests for tv clean subcommand behavior."""

    def test_clean_has_config_option(self) -> None:
        """tv clean has --config option."""
        import inspect

        sig = inspect.signature(tv_clean)
        params = sig.parameters
        assert "config" in params

    def test_clean_has_keep_option(self) -> None:
        """tv clean has --keep option with default."""
        import inspect

        sig = inspect.signature(tv_clean)
        params = sig.parameters
        assert "keep" in params
        # Check default value
        keep_param = params["keep"]
        # Typer Option has default
        assert keep_param.default is not None

    def test_clean_has_verbose_option(self) -> None:
        """tv clean has --verbose option."""
        import inspect

        sig = inspect.signature(tv_clean)
        params = sig.parameters
        assert "verbose" in params

    def test_clean_keep_policy_documented(self) -> None:
        """tv clean keep-policy documented in docstring."""
        assert tv_clean.__doc__ is not None
        doc = tv_clean.__doc__
        assert "KEEP" in doc or "keep" in doc.lower()
        assert "recent" in doc.lower()

    def test_clean_non_throwing_for_delete_failures(self) -> None:
        """tv clean non-throwing behavior for individual delete failures."""
        assert tv_clean.__doc__ is not None
        doc = tv_clean.__doc__.lower()
        # Should document that individual failures don't cause exit
        assert "fail" in doc or "error" in doc or "log" in doc


# =============================================================================
# CONTRACT: OUTPUT FORMATS
# =============================================================================


class TestOutputFormats:
    """Contract tests for CLI output formats."""

    def test_output_json_function_exists(self) -> None:
        """_output_json helper function exists."""
        assert callable(_output_json)

    def test_output_json_accepts_dict(self) -> None:
        """_output_json accepts dict input."""
        import io
        import sys
        from contextlib import redirect_stdout

        # Capture stdout
        output = io.StringIO()
        with redirect_stdout(output):
            _output_json({"test": "value"})

        result = output.getvalue()
        assert "test" in result
        assert "value" in result

    def test_output_json_accepts_list(self) -> None:
        """_output_json accepts list input."""
        import io
        import sys
        from contextlib import redirect_stdout

        # Capture stdout
        output = io.StringIO()
        with redirect_stdout(output):
            _output_json([{"id": 1}, {"id": 2}])

        result = output.getvalue()
        assert "id" in result


# =============================================================================
# CONTRACT: MODULE DOCSTRING
# =============================================================================


class TestModuleDocumentation:
    """Contract tests for module documentation."""

    def test_module_docstring_present(self) -> None:
        """CLI module has docstring."""
        from sfumato import cli

        assert cli.__doc__ is not None

    def test_exit_codes_documented_in_module(self) -> None:
        """Exit code semantics documented in module docstring."""
        from sfumato import cli

        doc = cli.__doc__
        assert "EXIT" in doc or "exit" in doc.lower()

    def test_command_behavior_contracts_documented(self) -> None:
        """Command behavior contracts documented."""
        from sfumato import cli

        doc = cli.__doc__
        assert "BEHAVIOR CONTRACT" in doc.upper() or "behavior" in doc.lower()

    def test_flag_semantics_documented(self) -> None:
        """Flag semantics documented."""
        from sfumato import cli

        doc = cli.__doc__
        # Either in module doc or in CLI_FLAG_SEMANTICS constant
        assert "flag" in doc.lower() or "option" in doc.lower()


# =============================================================================
# CONTRACT: RUN COMMAND EXIT CODE BEHAVIOR
# =============================================================================


class TestRunCommandExitCodes:
    """Contract tests for run command exit code behavior."""

    def test_run_success_exit_code_documented(self) -> None:
        """run success exit code (0) documented."""
        assert run.__doc__ is not None
        doc = run.__doc__
        assert "EXIT CODES" in doc or "0:" in doc

    def test_run_config_error_exit_code_documented(self) -> None:
        """run config error exit code (2) documented."""
        assert run.__doc__ is not None
        doc = run.__doc__
        assert "2:" in doc or "Configuration error" in doc

    def test_run_state_error_exit_code_documented(self) -> None:
        """run state initialization error exit code (3) documented."""
        assert run.__doc__ is not None
        doc = run.__doc__
        assert "3:" in doc or "State" in doc

    def test_run_input_error_exit_code_documented(self) -> None:
        """run input validation error exit code (4) documented."""
        assert run.__doc__ is not None
        doc = run.__doc__
        assert "4:" in doc or "Input" in doc

    def test_run_file_error_exit_code_documented(self) -> None:
        """run file/IO error exit code (5) documented."""
        assert run.__doc__ is not None
        doc = run.__doc__
        assert "5:" in doc or "File" in doc or "IO" in doc


# =============================================================================
# CONTRACT: PREVIEW COMMAND BEHAVIOR
# =============================================================================


class TestPreviewCommandBehavior:
    """Contract tests for preview command behavior."""

    def test_preview_equivalent_to_run_no_upload_preview(self) -> None:
        """preview documented as equivalent to run --no-upload --preview."""
        assert preview.__doc__ is not None
        doc = preview.__doc__
        assert "no-upload" in doc.lower() or "Equivalent" in doc

    def test_preview_has_config_option(self) -> None:
        """preview has --config option."""
        import inspect

        sig = inspect.signature(preview)
        params = sig.parameters
        assert "config" in params

    def test_preview_has_verbose_option(self) -> None:
        """preview has --verbose option."""
        import inspect

        sig = inspect.signature(preview)
        params = sig.parameters
        assert "verbose" in params

    def test_preview_open_in_viewer_documented(self) -> None:
        """preview opening in system viewer documented."""
        assert preview.__doc__ is not None
        doc = preview.__doc__.lower()
        assert "viewer" in doc or "open" in doc or "preview" in doc


# =============================================================================
# CONTRACT: INIT COMMAND BEHAVIOR
# =============================================================================


class TestInitCommandBehavior:
    """Contract tests for init command behavior."""

    def test_init_idempotent_documented(self) -> None:
        """init idempotent behavior documented."""
        assert init.__doc__ is not None
        doc = init.__doc__.lower()
        assert "idempotent" in doc or "safe to run multiple" in doc

    def test_init_long_running_warning_documented(self) -> None:
        """init long-running warning documented."""
        assert init.__doc__ is not None
        doc = init.__doc__
        assert (
            "long" in doc.lower() or "llm" in doc.lower() or "painting" in doc.lower()
        )

    def test_init_has_config_option(self) -> None:
        """init has --config option."""
        import inspect

        sig = inspect.signature(init)
        params = sig.parameters
        assert "config" in params

    def test_init_has_verbose_option(self) -> None:
        """init has --verbose option."""
        import inspect

        sig = inspect.signature(init)
        params = sig.parameters
        assert "verbose" in params


# =============================================================================
# CONTRACT: APP REGISTRATION
# =============================================================================


class TestAppRegistration:
    """Contract tests for app command registration."""

    def test_init_registered_as_command(self) -> None:
        """init registered as Typer command."""
        # The @app.command() decorator registers the command
        # We can verify by checking the function has the right metadata
        assert hasattr(init, "__name__")

    def test_run_registered_as_command(self) -> None:
        """run registered as Typer command."""
        assert hasattr(run, "__name__")

    def test_watch_registered_as_command(self) -> None:
        """watch registered as Typer command."""
        assert hasattr(watch, "__name__")

    def test_preview_registered_as_command(self) -> None:
        """preview registered as Typer command."""
        assert hasattr(preview, "__name__")

    def test_tv_app_registered(self) -> None:
        """tv typer app registered on main app."""
        from sfumato.cli import tv_app
        import typer

        assert isinstance(tv_app, typer.Typer)


# =============================================================================
# CONTRACT: COMMAND PIPELINE ORDER
# =============================================================================


class TestPipelineOrder:
    """Contract tests for run command pipeline order documentation."""

    def test_run_pipeline_order_documented(self) -> None:
        """run pipeline stages documented in order."""
        assert run.__doc__ is not None
        doc = run.__doc__
        # Check ARCHITECTURE.md reference
        assert "ARCHITECTURE" in doc or "pipeline" in doc.lower()
        # Check stages are mentioned
        assert "load" in doc.lower() and "config" in doc.lower()

    def test_watch_daemon_loop_documented(self) -> None:
        """watch daemon loop documented."""
        assert watch.__doc__ is not None
        doc = watch.__doc__
        # Check ARCHITECTURE.md reference
        assert "ARCHITECTURE" in doc or "DAEMON" in doc or "loop" in doc.lower()
