"""Tests for the LLM module contracts and behavior.

This file contains CONTRACT TESTS that verify the behavioral boundaries
defined in src/sfumato/llm.py. Each test documents expected behavior
without depending on implementation details.

Architecture reference: ARCHITECTURE.md#2.9
Contract reference: src/sfumato/llm.py
"""

from __future__ import annotations

import pytest

from sfumato.config import AiConfig
from sfumato.llm import (
    BACKEND_DISPATCH_MAP,
    DEFAULT_TEXT_TIMEOUT_SECONDS,
    DEFAULT_VISION_TIMEOUT_SECONDS,
    MAX_RETRY_ATTEMPTS,
    TRANSIENT_ERROR_INDICATORS,
    VALID_BACKENDS,
    EmbeddingError,
    LlmError,
    LlmParseError,
    LlmResponse,
    compute_embedding,  # noqa: F401 - contract export verification
    invoke_text,
    invoke_vision,  # noqa: F401 - contract export verification
    parse_json_response,  # noqa: F401 - contract export verification
)


# =============================================================================
# CONTRACT: PUBLIC ERROR TYPES
# =============================================================================


class TestLlmErrorContract:
    """Test LlmError exception hierarchy."""

    def test_llm_error_is_base_exception(self) -> None:
        """LlmError is the base class for all LLM-related failures."""
        assert issubclass(LlmParseError, LlmError)
        assert issubclass(EmbeddingError, LlmError)

    def test_llm_parse_error_is_subclass_of_llm_error(self) -> None:
        """LlmParseError inherits from LlmError."""
        error = LlmParseError("test parse failure")
        assert isinstance(error, LlmError)

    def test_embedding_error_is_subclass_of_llm_error(self) -> None:
        """EmbeddingError inherits from LlmError."""
        error = EmbeddingError("test embedding failure")
        assert isinstance(error, LlmError)


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


class TestUnsupportedBackendBehavior:
    """Test behavior when unsupported backend is specified."""

    def test_unsupported_backend_raises_llm_error(self) -> None:
        """invoke_text with unsupported backend raises LlmError.

        This test documents the expected behavior; the actual
        implementation stub raises NotImplementedError.
        """
        # Contract expectation: unsupported backend should raise LlmError
        # with a message listing valid backends.
        unsupported_config = AiConfig(cli="unknown_backend", model="test-model")

        # Calling the stub function should raise NotImplementedError
        # (contract-only stub behavior)
        # Implementation will raise LlmError for unsupported backends
        with pytest.raises(NotImplementedError):
            # The stub raises NotImplementedError; implementation
            # should raise LlmError with message listing valid backends
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                invoke_text("test prompt", unsupported_config)
            )


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


class TestTimeoutOverride:
    """Test that callers can override default timeouts."""

    def test_invoke_text_accepts_timeout_override(self) -> None:
        """invoke_text has timeout_seconds parameter for override.

        This test documents the interface; implementation is deferred.
        """
        # The stub accepts the parameter, documented in signature
        # Actual implementation would use it for subprocess timeout
        pass  # Interface documented in stub signature

    def test_invoke_vision_accepts_timeout_override(self) -> None:
        """invoke_vision has timeout_seconds parameter for override."""
        # The stub accepts the parameter, documented in signature
        pass


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
        """Transient error indicators in contract for documentation.

        Actual matching should be case-insensitive, but contract
        defines them in lowercase for consistency.
        """
        for indicator in TRANSIENT_ERROR_INDICATORS:
            assert indicator == indicator.lower(), f"{indicator} should be lowercase"


class TestRetryBehavior:
    """Test retry behavior boundaries.

    These tests document expected behavior. Implementation
    should follow these contracts.
    """

    def test_transient_errors_are_retried(self) -> None:
        """Timeout and connection failures retry up to MAX_RETRY_ATTEMPTS.

        Contract: Transient errors (timeout, connection refused) should
        trigger retry logic before raising LlmError.
        """
        # Documentation test - actual behavior verified by integration tests
        pass

    def test_deterministic_errors_not_retried(self) -> None:
        """Parse errors and configuration errors are NOT retried.

        Contract: LlmParseError and unsupported backend errors
        should raise immediately without retry.
        """
        pass


# =============================================================================
# CONTRACT: PARSE_JSON_RESPONSE LENIENCY
# =============================================================================


class TestParseJsonResponseContract:
    """Test parse_json_response leniency boundaries.

    These tests document the expected leniency transformations.
    Implementation should handle these cases.
    """

    def test_strips_markdown_json_fence(self) -> None:
        """Handles ```json ... ``` wrapper.

        Contract: Input ```json\\n{"key": "value"}\\n```
        should parse to {"key": "value"}.
        """
        pass  # Implementation deferred

    def test_strips_generic_markdown_fence(self) -> None:
        """Handles ``` ... ``` wrapper without language specifier.

        Contract: Input ```\\n{"key": "value"}\\n```
        should parse to {"key": "value"}.
        """
        pass

    def test_trims_whitespace(self) -> None:
        """Trims leading/trailing whitespace.

        Contract: Input '  {"key": "value"}  '
        should parse to {"key": "value"}.
        """
        pass

    def test_removes_trailing_comma_in_object(self) -> None:
        """Removes trailing comma before }.

        Contract: Input '{"key": "value",}'
        should parse to {"key": "value"}.
        """
        pass

    def test_removes_trailing_comma_in_array(self) -> None:
        """Removes trailing comma before ].

        Contract: Input '{"items": [1, 2, 3,]}'
        should parse to {"items": [1, 2, 3]}.
        """
        pass

    def test_raises_llm_parse_error_on_invalid_json(self) -> None:
        """Raises LlmParseError for truly invalid JSON.

        Contract: After all leniency transformations, if JSON is
        still invalid, raise LlmParseError with helpful context.
        """
        pass

    def test_preserves_valid_json_data(self) -> None:
        """Leniency transformations do not lose data.

        Contract: Valid JSON content is preserved after transformations.
        """
        pass


# =============================================================================
# CONTRACT: INVOKE FUNCTIONS SIGNATURES
# =============================================================================


class TestInvokeTextSignature:
    """Test invoke_text function contract."""

    def test_returns_llm_response(self) -> None:
        """invoke_text returns LlmResponse (typed as coroutine, await to get value).

        Contract: The function is async and returns LlmResponse when awaited.
        """
        # Signature documented in stub
        # Returns coroutine[LlmResponse] when called
        pass

    def test_accepts_required_parameters(self) -> None:
        """invoke_text requires prompt and ai_config.

        Contract: Minimum invocation is await invoke_text(prompt, ai_config).
        """
        pass

    def test_accepts_optional_parameters(self) -> None:
        """invoke_text has optional system_prompt, max_tokens, temperature, timeout.

        Contract: All optional parameters have documented defaults.
        """
        pass

    def test_raises_llm_error_on_persistent_failure(self) -> None:
        """invoke_text raises LlmError after retries exhausted.

        Contract: After MAX_RETRY_ATTEMPTS, raises LlmError.
        """
        pass

    def test_raises_llm_error_on_unsupported_backend(self) -> None:
        """invoke_text raises LlmError for unsupported backend.

        Contract: When ai_config.cli is not in BACKEND_DISPATCH_MAP,
        raises LlmError with message listing valid backends.
        """
        pass


class TestInvokeVisionSignature:
    """Test invoke_vision function contract."""

    def test_returns_llm_response(self) -> None:
        """invoke_vision returns LlmResponse.

        Contract: The function is async and returns LlmResponse when awaited.
        """
        pass

    def test_requires_image_path(self) -> None:
        """invoke_vision requires image_path parameter.

        Contract: Unlike invoke_text, requires an image file.
        """
        pass

    def test_uses_longer_default_timeout(self) -> None:
        """invoke_vision uses DEFAULT_VISION_TIMEOUT_SECONDS by default.

        Contract: Vision calls take longer; default reflects this.
        """
        pass

    def test_raises_llm_error_on_missing_image(self) -> None:
        """invoke_vision raises LlmError if image_path does not exist.

        Contract: Validates image existence before subprocess call.
        """
        pass


class TestComputeEmbeddingSignature:
    """Test compute_embedding function contract."""

    def test_returns_list_of_floats(self) -> None:
        """compute_embedding returns list[float], not numpy array.

        Contract: Keep dependencies minimal; return plain Python list.
        """
        pass

    def test_accepts_text_and_config(self) -> None:
        """compute_embedding requires text and ai_config.

        Contract: Minimum invocation is await compute_embedding(text, ai_config).
        """
        pass

    def test_raises_embedding_error_on_failure(self) -> None:
        """compute_embedding raises EmbeddingError on failure.

        Contract: Embedding-specific failures raise EmbeddingError.
        """
        pass

    def test_uses_backend_specific_embedding_approach(self) -> None:
        """Each backend may use different embedding strategy.

        Contract:
        - gemini: uses Gemini embedding API
        - codex: may use local sentence-transformers fallback
        - claude-code: may use Anthropic embedding or local fallback
        """
        pass


# =============================================================================
# IMPLEMENTATION NOTE
# =============================================================================

# This file contains CONTRACT TESTS that document expected behavior.
# Tests marked with `pass` are interface documentation; they will be
# activated once implementation is complete.
#
# Key contract boundaries documented here:
# 1. Three supported backends: gemini, codex, claude-code
# 2. Two retry attempts for transient errors (timeout, connection)
# 3. No retry for deterministic errors (parse, config)
# 4. Different timeouts for text (120s) vs vision (180s)
# 5. JSON leniency: strip fences, trim whitespace, tolerate trailing commas
# 6. Embedding returns list[float], not numpy array
#
# Implementation workers should ensure their code passes all tests here.
