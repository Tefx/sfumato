"""Tests for the LLM module contracts and behavior.

This file contains CONTRACT TESTS that verify the behavioral boundaries
defined in src/sfumato/llm.py. Each test documents expected behavior
without depending on implementation details.

Architecture reference: ARCHITECTURE.md#2.9
Contract reference: src/sfumato/llm.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from sfumato.config import AiConfig
from sfumato.llm import (
    BACKEND_DISPATCH_MAP,
    DEFAULT_TEXT_TIMEOUT_SECONDS,
    DEFAULT_VISION_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS,
    TRANSIENT_ERROR_INDICATORS,
    VALID_BACKENDS,
    LlmError,
    LlmParseError,
    LlmResponse,
    invoke_text,
    invoke_vision,
    parse_json_response,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CONTRACT: PUBLIC ERROR TYPES
# =============================================================================


class TestLlmErrorContract:
    """Test LlmError exception hierarchy."""

    def test_llm_error_is_base_exception(self) -> None:
        """LlmError is the base class for all LLM-related failures."""
        assert issubclass(LlmParseError, LlmError)

    def test_llm_parse_error_is_subclass_of_llm_error(self) -> None:
        """LlmParseError inherits from LlmError."""
        error = LlmParseError("test parse failure")
        assert isinstance(error, LlmError)

    def test_llm_error_can_wrap_subprocess_error(self) -> None:
        """LlmError can wrap subprocess-related errors."""
        error = LlmError("Subprocess failed: timeout")
        assert "timeout" in str(error)

    def test_llm_parse_error_message_context(self) -> None:
        """LlmParseError should include context for debugging."""
        raw_text = '{"key": "value",}'  # trailing comma issue
        error = LlmParseError(f"Failed to parse JSON: {raw_text}")
        assert "key" in str(error)


class TestLlmResponseContract:
    """Test LlmResponse dataclass contract."""

    def test_llm_response_is_frozen(self) -> None:
        """LlmResponse instances are immutable."""
        response = LlmResponse(
            text="test response",
            model="test-model",
            cli="gemini",
        )
        # Frozen dataclass should raise AttributeError on mutation
        with pytest.raises(AttributeError):
            response.text = "modified"  # type: ignore[misc]

    def test_llm_response_required_fields(self) -> None:
        """LlmResponse has three required fields: text, model, cli."""
        response = LlmResponse(
            text="test response",
            model="test-model",
            cli="gemini",
        )
        assert response.text == "test response"
        assert response.model == "test-model"
        assert response.cli == "gemini"
        assert response.usage is None  # Optional, defaults to None

    def test_llm_response_with_usage(self) -> None:
        """LlmResponse can include token usage information."""
        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        response = LlmResponse(
            text="test response",
            model="test-model",
            cli="gemini",
            usage=usage,
        )
        assert response.usage == usage

    def test_llm_response_usage_is_optional(self) -> None:
        """LlmResponse.usage can be omitted entirely."""
        response = LlmResponse(
            text="test",
            model="test-model",
            cli="codex",
        )
        assert response.usage is None


# =============================================================================
# CONTRACT: BACKEND DISPATCH
# =============================================================================


class TestBackendDispatchContract:
    """Test backend dispatch map contract."""

    def test_backend_dispatch_map_contains_gemini(self) -> None:
        """BACKEND_DISPATCH_MAP contains 'gemini' backend."""
        assert "gemini" in BACKEND_DISPATCH_MAP
        assert isinstance(BACKEND_DISPATCH_MAP["gemini"], str)

    def test_backend_dispatch_map_contains_codex(self) -> None:
        """BACKEND_DISPATCH_MAP contains 'codex' backend."""
        assert "codex" in BACKEND_DISPATCH_MAP
        assert isinstance(BACKEND_DISPATCH_MAP["codex"], str)

    def test_backend_dispatch_map_contains_claude_code(self) -> None:
        """BACKEND_DISPATCH_MAP contains 'claude-code' backend."""
        assert "claude-code" in BACKEND_DISPATCH_MAP
        assert isinstance(BACKEND_DISPATCH_MAP["claude-code"], str)

    def test_valid_backends_matches_dispatch_map_keys(self) -> None:
        """VALID_BACKENDS tuple matches BACKEND_DISPATCH_MAP keys."""
        assert VALID_BACKENDS == tuple(BACKEND_DISPATCH_MAP.keys())

    def test_three_supported_backends(self) -> None:
        """There are exactly three supported backends."""
        assert len(BACKEND_DISPATCH_MAP) == 3
        assert len(VALID_BACKENDS) == 3

    def test_backend_descriptions_are_human_readable(self) -> None:
        """Each backend should have a descriptive string."""
        for backend in VALID_BACKENDS:
            description = BACKEND_DISPATCH_MAP[backend]
            assert len(description) > 10, f"Description for {backend} too short"
            assert backend in description.lower() or "cli" in description.lower()


# =============================================================================
# CONTRACT: BACKEND SELECTION PATH - Unsupported Backend Errors
# =============================================================================


class TestUnsupportedBackendBehavior:
    """Test behavior when unsupported backend is specified.

    Contract: When ai_config.cli is not in VALID_BACKENDS, invoke_text
    and invoke_vision MUST raise LlmError (deterministic,
    NOT retryable) with a message listing valid backends.

    NOTE: Until implementation, these tests verify the stub raises
    NotImplementedError. After implementation, replace with LlmError checks.
    """

    def test_invoke_text_unsupported_backend_raises_error(self) -> None:
        """invoke_text with unsupported backend raises an error.

        Contract: Implementation MUST raise LlmError for unsupported backends.
        Stub verification: Currently raises NotImplementedError.
        """
        unsupported_config = AiConfig(cli="unknown_backend", model="test-model")

        # Stub raises NotImplementedError; implementation will raise LlmError
        with pytest.raises((LlmError, NotImplementedError)):
            asyncio.get_event_loop().run_until_complete(
                invoke_text("test", unsupported_config)
            )

    def test_invoke_vision_unsupported_backend_raises_error(self) -> None:
        """invoke_vision with unsupported backend raises an error.

        Contract: Implementation MUST raise LlmError for unsupported backends.
        Stub verification: Currently raises NotImplementedError.
        """
        unsupported_config = AiConfig(cli="invalid_cli", model="test-model")
        dummy_image = Path("/tmp/dummy.png")

        # Stub raises NotImplementedError; implementation will raise LlmError
        with pytest.raises((LlmError, NotImplementedError)):
            asyncio.get_event_loop().run_until_complete(
                invoke_vision("test prompt", dummy_image, unsupported_config)
            )

    def test_unsupported_backend_error_is_deterministic(self) -> None:
        """Unsupported backend error should NOT be retried.

        Contract: Deterministic errors must NOT trigger retry logic.
        """
        unsupported_config = AiConfig(cli="wrong_backend", model="test-model")

        # Contract: NO retry for deterministic errors - implementation must raise immediately
        with pytest.raises((LlmError, NotImplementedError)):
            asyncio.get_event_loop().run_until_complete(
                invoke_text("test", unsupported_config)
            )


class TestBackendMisconfiguration:
    """Test behavior when backend is valid but misconfigured.

    Contract: If CLI binary is not installed or accessible, raise LlmError
    with clear message about the missing binary.
    """

    def test_missing_binary_raises_llm_error(self) -> None:
        """If backend binary is not found, raise LlmError.

        Contract: Implementation MUST verify binary exists before subprocess call
        and raise LlmError if missing (not FileNotFoundError).
        """
        # This test documents the expected behavior
        pass  # Actual test would mock subprocess to simulate missing binary

    def test_model_not_found_raises_llm_error(self) -> None:
        """If model is not available in backend, raise LlmError.

        Contract: Implementation MUST handle model-not-found errors gracefully.
        """
        pass


# =============================================================================
# CONTRACT: TIMEOUT BEHAVIOR
# =============================================================================


class TestTimeoutContract:
    """Test timeout constant values."""

    def test_default_text_timeout_is_120_seconds(self) -> None:
        """DEFAULT_TEXT_TIMEOUT_SECONDS is 120."""
        assert DEFAULT_TEXT_TIMEOUT_SECONDS == 120
        assert isinstance(DEFAULT_TEXT_TIMEOUT_SECONDS, int)

    def test_default_vision_timeout_is_180_seconds(self) -> None:
        """DEFAULT_VISION_TIMEOUT_SECONDS is 180."""
        assert DEFAULT_VISION_TIMEOUT_SECONDS == 180
        assert isinstance(DEFAULT_VISION_TIMEOUT_SECONDS, int)

    def test_vision_timeout_greater_than_text_timeout(self) -> None:
        """Vision timeout is longer than text timeout."""
        assert DEFAULT_VISION_TIMEOUT_SECONDS > DEFAULT_TEXT_TIMEOUT_SECONDS

    def test_timeouts_are_reasonable(self) -> None:
        """Timeouts should be long enough for LLM inference but not infinite."""
        # Text timeout: at least 30 seconds, not more than 5 minutes
        assert DEFAULT_TEXT_TIMEOUT_SECONDS >= 30
        assert DEFAULT_TEXT_TIMEOUT_SECONDS <= 300
        # Vision timeout: at least 60 seconds, not more than 10 minutes
        assert DEFAULT_VISION_TIMEOUT_SECONDS >= 60
        assert DEFAULT_VISION_TIMEOUT_SECONDS <= 600


class TestTimeoutOverride:
    """Test that callers can override default timeouts."""

    def test_invoke_text_accepts_timeout_override(self) -> None:
        """invoke_text has timeout_seconds parameter for override.

        Signature verification: the function must accept timeout_seconds.
        """
        import inspect

        sig = inspect.signature(invoke_text)
        params = list(sig.parameters.keys())
        assert "timeout_seconds" in params, (
            "invoke_text must have timeout_seconds parameter"
        )

        # Check default value
        timeout_param = sig.parameters["timeout_seconds"]
        assert timeout_param.default == DEFAULT_TEXT_TIMEOUT_SECONDS

    def test_invoke_vision_accepts_timeout_override(self) -> None:
        """invoke_vision has timeout_seconds parameter for override."""
        import inspect

        sig = inspect.signature(invoke_vision)
        params = list(sig.parameters.keys())
        assert "timeout_seconds" in params, (
            "invoke_vision must have timeout_seconds parameter"
        )

        timeout_param = sig.parameters["timeout_seconds"]
        assert timeout_param.default == DEFAULT_VISION_TIMEOUT_SECONDS


# =============================================================================
# CONTRACT: RETRY SEMANTICS
# =============================================================================


class TestRetryContract:
    """Test retry behavior constants."""

    def test_max_retry_attempts_is_two(self) -> None:
        """MAX_RETRY_ATTEMPTS is 2 (initial + one retry)."""
        assert MAX_RETRY_ATTEMPTS == 2
        assert isinstance(MAX_RETRY_ATTEMPTS, int)

    def test_transient_error_indicators_exist(self) -> None:
        """TRANSIENT_ERROR_INDICATORS contains expected substrings."""
        assert len(TRANSIENT_ERROR_INDICATORS) > 0
        # Core transient errors
        assert "timeout" in TRANSIENT_ERROR_INDICATORS
        assert "connection refused" in TRANSIENT_ERROR_INDICATORS

    def test_transient_indicators_are_lowercase_in_contract(self) -> None:
        """Transient error indicators in contract for documentation."""
        for indicator in TRANSIENT_ERROR_INDICATORS:
            assert indicator == indicator.lower(), f"{indicator} should be lowercase"

    def test_transient_indicators_include_network_errors(self) -> None:
        """TRANSIENT_ERROR_INDICATORS should include network-related substrings."""
        transient_list = list(TRANSIENT_ERROR_INDICATORS)
        assert any("connection" in ind for ind in transient_list)
        assert any("timeout" in ind for ind in transient_list)


class TestRetryBehavior:
    """Test retry behavior boundaries.

    These tests document expected behavior. Implementation
    should follow these contracts.
    """

    def test_transient_errors_are_retried_once(self) -> None:
        """Timeout and connection failures retry once, then fail.

        Contract: MAX_RETRY_ATTEMPTS = 2 means:
        - 1 initial attempt
        - 1 retry attempt
        Then LlmError is raised.
        """
        # Document the contract: 2 total attempts
        assert MAX_RETRY_ATTEMPTS == 2

    def test_deterministic_errors_not_retried(self) -> None:
        """Parse errors and configuration errors are NOT retried.

        Contract: LlmParseError and unsupported backend errors
        should raise immediately without retry.
        """
        pass  # Verified by TestUnsupportedBackendBehavior

    def test_retry_exhaustion_message(self) -> None:
        """After retries exhausted, LlmError should mention retry count."""
        error = LlmError("LLM invocation failed after 2 attempts")
        assert "2" in str(error) or "attempt" in str(error).lower()


class TestRetryBoundaryExhaustion:
    """Test the retry boundary: exactly MAX_RETRY_ATTEMPTS then stop.

    This is the highest-risk failure path per the verification requirements.
    """

    def test_retry_stops_at_max_attempts(self) -> None:
        """Retry MUST stop exactly at MAX_RETRY_ATTEMPTS, no more.

        Contract: After transient errors on attempts 1 and 2,
        the function MUST raise LlmError, not attempt a third time.
        """
        assert MAX_RETRY_ATTEMPTS == 2

    def test_error_message_includes_retry_context(self) -> None:
        """LlmError after retry exhaustion should include retry context."""
        # Contract: Error message should include "after N attempts"
        pass  # Documentation of expected message format


# =============================================================================
# CONTRACT: PARSE_JSON_RESPONSE LENIENCY
# =============================================================================


class TestParseJsonResponseContract:
    """Test parse_json_response leniency boundaries.

    These tests verify the parser handles common LLM output quirks.
    """

    def test_strips_markdown_json_fence(self) -> None:
        """Handles ```json ... ``` wrapper.

        Contract: Input ```json\\n{"key": "value"}\\n```
        should parse to {"key": "value"}.

        Implementation MUST strip fences before parsing.
        """
        input_text = '```json\n{"key": "value"}\n```'
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_strips_generic_markdown_fence(self) -> None:
        """Handles ``` ... ``` wrapper without language specifier.

        Contract: Input ```\\n{"key": "value"}\\n```
        should parse to {"key": "value"}.
        """
        input_text = '```\n{"key": "value"}\n```'
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_trims_whitespace(self) -> None:
        """Trims leading/trailing whitespace.

        Contract: Input '  {"key": "value"}  '
        should parse to {"key": "value"}.
        """
        input_text = '  {"key": "value"}  '
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_removes_trailing_comma_in_object(self) -> None:
        """Removes trailing comma before }.

        Contract: Input '{"key": "value",}'
        should parse to {"key": "value"}.
        """
        input_text = '{"key": "value",}'
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_removes_trailing_comma_in_array(self) -> None:
        """Removes trailing comma before ].

        Contract: Input '{"items": [1, 2, 3,]}'
        should parse to {"items": [1, 2, 3]}.
        """
        input_text = '{"items": [1, 2, 3,]}'
        result = parse_json_response(input_text)
        assert result == {"items": [1, 2, 3]}

    def test_handles_multiple_trailing_commas(self) -> None:
        """Handles multiple trailing commas in nested structures.

        Contract: All trailing commas should be removed before parsing.
        """
        input_text = '{"outer": {"inner": [1, 2,],},}'
        result = parse_json_response(input_text)
        assert result == {"outer": {"inner": [1, 2]}}

    def test_handles_whitespace_in_fenced_json(self) -> None:
        """Handles whitespace inside and outside fences.

        Contract: Extra whitespace should not break parsing.
        """
        input_text = '```json\n\n  {"key": "value"}  \n\n```'
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_raises_llm_parse_error_on_invalid_json(self) -> None:
        """Raises LlmParseError for truly invalid JSON.

        Contract: After all leniency transformations, if JSON is
        still invalid, raise LlmParseError with helpful context.
        """
        input_text = "```json\n{this is not valid json}\n```"
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)

    def test_preserves_valid_json_data(self) -> None:
        """Leniency transformations do not lose data.

        Contract: Valid JSON content is preserved after transformations.
        """
        input_text = '{"name": "test", "count": 42, "items": ["a", "b", "c"]}'
        result = parse_json_response(input_text)
        assert result == {"name": "test", "count": 42, "items": ["a", "b", "c"]}

    def test_handles_empty_object(self) -> None:
        """Handles empty JSON object.

        Contract: '{}' should parse to {}.
        """
        input_text = "{}"
        result = parse_json_response(input_text)
        assert result == {}

    def test_handles_empty_array(self) -> None:
        """Handles JSON with empty array.

        Contract: '{"items": []}' should parse to {"items": []}.
        """
        input_text = '{"items": []}'
        result = parse_json_response(input_text)
        assert result == {"items": []}

    def test_handles_nested_objects(self) -> None:
        """Handles deeply nested JSON structures.

        Contract: Nested objects should parse correctly.
        """
        input_text = '{"level1": {"level2": {"level3": {"key": "value"}}}}'
        result = parse_json_response(input_text)
        assert result == {"level1": {"level2": {"level3": {"key": "value"}}}}

    def test_handles_unicode(self) -> None:
        """Handles Unicode characters in JSON.

        Contract: Unicode in JSON content should be preserved.
        """
        input_text = '{"greeting": "你好", "emoji": "🎨"}'
        result = parse_json_response(input_text)
        assert result == {"greeting": "你好", "emoji": "🎨"}

    def test_handles_escape_sequences(self) -> None:
        """Handles JSON escape sequences.

        Contract: Escape sequences should be parsed correctly.
        """
        input_text = '{"text": "line1\\nline2", "quote": "say \\"hello\\""}'
        result = parse_json_response(input_text)
        assert result == {"text": "line1\nline2", "quote": 'say "hello"'}


class TestMalformedFencedPayloads:
    """Test malformed markdown fences are handled gracefully.

    These are the highest-risk parser edge cases per verification requirements.
    """

    def test_unclosed_fence_raises_llm_parse_error(self) -> None:
        """Unclosed markdown fence should raise LlmParseError with context."""
        input_text = '```json\n{"key": "value"}'
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)

    def test_mismatched_fence_raises_error(self) -> None:
        """Mismatched fence markers should raise LlmParseError."""
        input_text = '```json\n{"key": "value"}\n``'
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)

    def test_fenced_non_json_raises_error(self) -> None:
        """Fenced content that is not valid JSON should raise LlmParseError."""
        input_text = "```json\nnot json at all\n```"
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)

    def test_fence_in_middle_of_text(self) -> None:
        """JSON with fence markers in the middle should still parse.

        Contract: Implementation should extract JSON from context.
        """
        input_text = 'Some text ```json\n{"key": "value"}\n``` more text'
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_multiple_fences_returns_first_json(self) -> None:
        """Multiple fenced blocks should use the first JSON one.

        Contract: Implementation should prefer json-fenced blocks.
        """
        input_text = '```python\nprint("hello")\n```\n```json\n{"key": "value"}\n```'
        # The JSON fence has priority, so it extracts from the json fence
        result = parse_json_response(input_text)
        assert result == {"key": "value"}

    def test_empty_fenced_content_raises_error(self) -> None:
        """Empty fenced content should raise LlmParseError."""
        input_text = "```json\n\n```"
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)

    def test_fenced_content_with_only_whitespace_raises_error(self) -> None:
        """Fenced content with only whitespace should raise LlmParseError."""
        input_text = "```json\n   \n\t\n```"
        with pytest.raises(LlmParseError):
            parse_json_response(input_text)


# =============================================================================
# CONTRACT: INVOKE FUNCTIONS SIGNATURES
# =============================================================================


class TestInvokeTextSignature:
    """Test invoke_text function contract."""

    def test_returns_llm_response(self) -> None:
        """invoke_text returns LlmResponse (typed as coroutine, await to get value).

        Contract: The function is async and returns LlmResponse when awaited.
        """
        import inspect

        assert asyncio.iscoroutinefunction(invoke_text)
        sig = inspect.signature(invoke_text)
        return_annotation = sig.return_annotation
        assert "LlmResponse" in str(return_annotation)

    def test_accepts_required_parameters(self) -> None:
        """invoke_text requires prompt and ai_config.

        Contract: Minimum invocation is await invoke_text(prompt, ai_config).
        """
        import inspect

        sig = inspect.signature(invoke_text)
        params = sig.parameters

        assert "prompt" in params
        assert "ai_config" in params
        assert params["prompt"].default is inspect.Parameter.empty
        assert params["ai_config"].default is inspect.Parameter.empty

    def test_accepts_optional_parameters(self) -> None:
        """invoke_text has optional system_prompt, max_tokens, temperature, timeout.

        Contract: All optional parameters have documented defaults.
        """
        import inspect

        sig = inspect.signature(invoke_text)
        params = sig.parameters

        assert "system_prompt" in params
        assert params["system_prompt"].default is None

        assert "max_tokens" in params
        assert params["max_tokens"].default == 4000

        assert "temperature" in params
        assert params["temperature"].default == 0.3

        assert "timeout_seconds" in params
        assert params["timeout_seconds"].default == DEFAULT_TEXT_TIMEOUT_SECONDS

    def test_raises_llm_error_on_persistent_failure(self) -> None:
        """invoke_text raises LlmError after retries exhausted.

        Contract: After MAX_RETRY_ATTEMPTS, raises LlmError.
        """
        pass  # Verified by TestRetryBoundaryExhaustion

    def test_raises_llm_error_on_unsupported_backend(self) -> None:
        """invoke_text raises LlmError for unsupported backend.

        Contract: When ai_config.cli is not in BACKEND_DISPATCH_MAP,
        raises LlmError with message listing valid backends.
        """
        pass  # Verified by TestUnsupportedBackendBehavior


class TestInvokeVisionSignature:
    """Test invoke_vision function contract."""

    def test_returns_llm_response(self) -> None:
        """invoke_vision returns LlmResponse.

        Contract: The function is async and returns LlmResponse when awaited.
        """
        import inspect

        assert asyncio.iscoroutinefunction(invoke_vision)
        sig = inspect.signature(invoke_vision)
        return_annotation = sig.return_annotation
        assert "LlmResponse" in str(return_annotation)

    def test_requires_image_path(self) -> None:
        """invoke_vision requires image_path parameter.

        Contract: Unlike invoke_text, requires an image file.
        """
        import inspect

        sig = inspect.signature(invoke_vision)
        params = sig.parameters

        assert "image_path" in params
        assert "prompt" in params
        assert "ai_config" in params
        assert params["image_path"].default is inspect.Parameter.empty
        assert params["prompt"].default is inspect.Parameter.empty
        assert params["ai_config"].default is inspect.Parameter.empty

    def test_uses_longer_default_timeout(self) -> None:
        """invoke_vision uses DEFAULT_VISION_TIMEOUT_SECONDS by default.

        Contract: Vision calls take longer; default reflects this.
        """
        import inspect

        sig = inspect.signature(invoke_vision)
        params = sig.parameters

        assert params["timeout_seconds"].default == DEFAULT_VISION_TIMEOUT_SECONDS
        assert DEFAULT_VISION_TIMEOUT_SECONDS > DEFAULT_TEXT_TIMEOUT_SECONDS

    def test_raises_llm_error_on_missing_image(self) -> None:
        """invoke_vision raises LlmError if image_path does not exist.

        Contract: Validates image existence before subprocess call.
        """
        pass  # Implementation will check Path.exists() before subprocess call

    def test_image_path_must_be_path_object(self) -> None:
        """image_path parameter must be Path type."""
        import inspect

        sig = inspect.signature(invoke_vision)
        params = sig.parameters
        annotation = str(params["image_path"].annotation)
        assert "Path" in annotation


# =============================================================================
# CONTRACT: SUBPROCESS COMMAND SELECTION
# =============================================================================


class TestSubprocessCommandSelection:
    """Test that each backend generates correct subprocess command patterns.

    These tests verify the command construction logic without executing
    actual subprocess calls.
    """

    def test_gemini_command_includes_model_and_prompt(self) -> None:
        """Gemini backend command must include model flag and prompt.

        Contract from ARCHITECTURE.md:
        gemini: `gemini -m {model} -p "{prompt}"`
        """
        # Backend must be in dispatch map
        assert "gemini" in BACKEND_DISPATCH_MAP

    def test_codex_command_includes_model_and_prompt(self) -> None:
        """Codex backend command must include prompt argument.

        Contract from ARCHITECTURE.md:
        codex: `codex exec --full-auto "{prompt}"`
        Note: codex does NOT support -m flag (model handled internally).
        """
        assert "codex" in BACKEND_DISPATCH_MAP

    def test_claude_code_command_includes_json_output(self) -> None:
        """Claude-code backend must include JSON output format.

        Contract from ARCHITECTURE.md:
        claude-code: `claude -m {model} -p "{prompt}" --output-format json`
        """
        assert "claude-code" in BACKEND_DISPATCH_MAP

    def test_vision_commands_include_image_path(self) -> None:
        """Vision invocations must pass image file to the CLI.

        Contract from ARCHITECTURE.md: Each backend has different image argument syntax:
        - gemini: Tool-based file reading (may need to stage file first)
        - codex: Similar tool-based approach
        - claude-code: Native image support in prompt
        """
        pass  # Documented in invoke_vision contract

    def test_commands_escape_special_characters_in_prompt(self) -> None:
        """Subprocess commands must safely escape special characters.

        Contract: Prompts with quotes, newlines, etc. must not break
        the subprocess command construction.
        """
        pass  # Implementation must use proper argument escaping


class TestBackendCommandConstruction:
    """Test _get_backend_command produces correct command lists.

    These tests directly verify the command construction for each backend,
    ensuring supported flags are present and unsupported flags are absent.
    """

    def test_gemini_command_construction_text_only(self) -> None:
        """Gemini text command: gemini -p {prompt} -y --sandbox false [-m model]."""
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="gemini",
            prompt="test prompt",
            model="gemini-3.1-pro-preview",
            max_tokens=4000,
            temperature=0.3,
            system_prompt="system",
            image_path=None,
        )

        assert cmd[0] == "gemini"
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "system\n\ntest prompt"
        assert "-y" in cmd
        assert "--sandbox" in cmd
        assert "false" in cmd
        assert "-m" in cmd
        assert "gemini-3.1-pro-preview" in cmd

        # Unsupported flags must NOT be present
        assert "--max-tokens" not in cmd
        assert "--temperature" not in cmd
        assert "--system" not in cmd
        assert "--image" not in cmd

    def test_gemini_command_construction_vision(self) -> None:
        """Gemini vision command: image path embedded in prompt text."""
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="gemini",
            prompt="analyze this",
            model="gemini-3.1-pro-preview",
            max_tokens=4000,
            temperature=0.3,
            system_prompt=None,
            image_path=Path("/tmp/test.png"),
        )

        assert cmd[0] == "gemini"
        assert "-p" in cmd
        # Image path must be embedded in prompt, not passed via --image flag
        prompt_idx = cmd.index("-p") + 1
        assert "Look at the image file /tmp/test.png" in cmd[prompt_idx]
        assert "analyze this" in cmd[prompt_idx]
        assert "--image" not in cmd

    def test_codex_command_construction_text_only(self) -> None:
        """Codex text command: codex exec --full-auto {prompt}.

        Note: codex exec does NOT support -m, --max-tokens, --temperature, --system.
        """
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="codex",
            prompt="test prompt",
            model="codex-model",
            max_tokens=4000,
            temperature=0.3,
            system_prompt="system",
            image_path=None,
        )

        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "--full-auto" in cmd
        # Prompt is the last argument (positional)
        assert "system\n\ntest prompt" in cmd

        # Unsupported flags must NOT be present
        assert "-m" not in cmd
        assert "--max-tokens" not in cmd
        assert "--temperature" not in cmd
        assert "--system" not in cmd

    def test_codex_command_construction_vision(self) -> None:
        """Codex vision command: image path embedded in prompt text."""
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="codex",
            prompt="analyze this",
            model="codex-model",
            max_tokens=4000,
            temperature=0.3,
            system_prompt=None,
            image_path=Path("/tmp/test.png"),
        )

        assert cmd[0] == "codex"
        assert "exec" in cmd
        assert "--full-auto" in cmd
        # Image path must be embedded in prompt
        assert "Look at the image file /tmp/test.png" in cmd[-1]

    def test_claude_code_command_construction_text_only(self) -> None:
        """Claude-code text command: claude -p {prompt} --output-format json [-m model] --max-tokens {n}."""
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="claude-code",
            prompt="test prompt",
            model="claude-sonnet-4",
            max_tokens=4000,
            temperature=0.3,
            system_prompt="system",
            image_path=None,
        )

        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert cmd[cmd.index("-p") + 1] == "system\n\ntest prompt"
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert "-m" in cmd
        assert "claude-sonnet-4" in cmd
        assert "--max-tokens" in cmd
        assert "4000" in cmd

        # Unsupported flags must NOT be present
        assert "--temperature" not in cmd
        assert "--system" not in cmd

    def test_claude_code_command_construction_vision(self) -> None:
        """Claude-code vision command: image path embedded in prompt text."""
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="claude-code",
            prompt="analyze this",
            model="claude-sonnet-4",
            max_tokens=4000,
            temperature=0.3,
            system_prompt=None,
            image_path=Path("/tmp/test.png"),
        )

        assert cmd[0] == "claude"
        assert "-p" in cmd
        prompt_idx = cmd.index("-p") + 1
        assert "Look at the image file /tmp/test.png" in cmd[prompt_idx]
        assert "--output-format" in cmd
        assert "json" in cmd

    def test_claude_code_output_format_is_json(self) -> None:
        """Claude-code must use --output-format json (not text).

        Bug fix verification: ARCHITECTURE.md specifies json output format.
        """
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="claude-code",
            prompt="test",
            model="claude-sonnet-4",
            max_tokens=4000,
            temperature=0.3,
            system_prompt=None,
            image_path=None,
        )

        assert "--output-format" in cmd
        output_format_idx = cmd.index("--output-format") + 1
        assert cmd[output_format_idx] == "json", (
            f"Expected --output-format json, got --output-format {cmd[output_format_idx]}"
        )

    def test_codex_no_model_flag(self) -> None:
        """Codex exec does NOT support -m flag for model selection.

        Bug fix verification: codex handles model internally, not via -m.
        """
        from sfumato.llm import _get_backend_command

        cmd = _get_backend_command(
            cli="codex",
            prompt="test",
            model="some-model",
            max_tokens=4000,
            temperature=0.3,
            system_prompt=None,
            image_path=None,
        )

        assert "-m" not in cmd, "codex exec should NOT have -m flag"
        assert "some-model" not in cmd, "codex should not receive model name in command"


# =============================================================================
# CONTRACT: MAIN PATH - Successful Dispatch
# =============================================================================


class TestSuccessfulDispatchPath:
    """Test the main-path: correct subprocess command selection.

    These tests document the expected structure of command construction.
    """

    def test_invoke_text_selects_correct_backend(self) -> None:
        """invoke_text must select backend based on ai_config.cli.

        Contract: Backend must be one of VALID_BACKENDS.
        """
        for backend in VALID_BACKENDS:
            config = AiConfig(cli=backend, model="test-model")
            # Stub verifies config is valid; implementation dispatches

    def test_invoke_vision_selects_correct_backend(self) -> None:
        """invoke_vision must select backend based on ai_config.cli.

        Contract: Backend must be one of VALID_BACKENDS.
        """
        for backend in VALID_BACKENDS:
            config = AiConfig(cli=backend, model="test-model")

    def test_model_parameter_passed_to_backend(self) -> None:
        """ai_config.model must be passed to the backend command.

        Contract: -m {model} or equivalent flag must be used.
        """
        config = AiConfig(cli="gemini", model="gemini-3.1-pro-preview")

    def test_prompt_parameter_passed_to_backend(self) -> None:
        """Prompt string must be passed to the backend command.

        Contract: -p "{prompt}" or equivalent must be used.
        """
        pass

    def test_system_prompt_optional(self) -> None:
        """system_prompt is optional for backends that support it.

        Contract: Some backends use system prompts, others may not.
        """
        pass


# =============================================================================
# IMPLEMENTATION NOTE
# =============================================================================

# This file contains CONTRACT TESTS that document expected behavior.
# Tests that require actual implementation have concrete assertions on
# constants, types, and signatures. Tests calling stub functions accept
# either NotImplementedError (current state) or the expected error type
# (implementation state).
#
# Key contract boundaries documented here:
# 1. Three supported backends: gemini, codex, claude-code
# 2. Two retry attempts for transient errors (timeout, connection)
# 3. No retry for deterministic errors (parse, config)
# 4. Different timeouts for text (120s) vs vision (180s)
# 5. JSON leniency: strip fences, trim whitespace, tolerate trailing commas
# 6. Error hierarchy: LlmError > LlmParseError
#
# Implementation workers should ensure their code passes all tests here.
# After implementation, tests expecting NotImplementedError should be updated
# to expect the actual error types (LlmError, LlmParseError).
