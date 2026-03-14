"""LLM backend invocation and response parsing.

This module defines the unified interface for invoking LLM backends (gemini CLI,
codex CLI, claude-code CLI) via subprocess. It handles prompt construction, response
parsing, retries, and timeout behavior.

This file contains CONTRACT STUBS ONLY - implementation is deferred to a later step.
All public functions raise NotImplementedError. The signatures, behavior contracts,
and error types are pinned here for worker agents to implement.

Architecture reference: ARCHITECTURE.md#2.9
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from sfumato.config import AiConfig


# =============================================================================
# PUBLIC ERROR TYPES
# =============================================================================


class LlmError(Exception):
    """LLM invocation failed after retries.

    This is the base exception for all LLM-related failures. It represents
    persistent failures that could not be resolved through retry attempts.

    Contract:
        - Raised after all retry attempts are exhausted
        - Message includes the CLI backend name and the underlying error
        - Wraps subprocess errors, timeouts, and connection failures
    """

    pass


class LlmParseError(LlmError):
    """LLM response could not be parsed as expected format.

    Raised when the LLM returns a response that cannot be parsed into
    the required structure (e.g., invalid JSON, missing required fields).

    Contract:
        - Subclass of LlmError (parse errors are a category of invocation failure)
        - Message includes the raw response that failed parsing
        - NOT retried (deterministic error - same bad input yields same bad output)
    """

    pass


class EmbeddingError(LlmError):
    """Embedding computation failed.

    Raised when the configured backend cannot compute an embedding vector
    for the given text input.

    Contract:
        - Subclass of LlmError
        - Wraps embedding-specific failures (API errors, dimension mismatches)
        - Retry behavior follows parent LlmError semantics
    """

    pass


# =============================================================================
# PUBLIC DATA TYPES
# =============================================================================


@dataclass(frozen=True)
class LlmResponse:
    """Typed response from an LLM invocation.

    Attributes:
        text: Raw text response from the LLM backend.
        model: The model identifier that produced the response.
        cli: The CLI backend name that was invoked ("gemini" | "codex" | "claude-code").
        usage: Optional token usage dict with keys like 'prompt_tokens',
            'completion_tokens', 'total_tokens'. Present when the backend
            provides it.

    Contract:
        - Instances are immutable (frozen dataclass)
        - `text` is the raw output from the CLI, including any markdown fencing
        - `model` reflects the model that actually responded (may differ from config
          if the backend selected a fallback)
        - `cli` is the canonical backend name from BACKEND_DISPATCH_MAP
        - `usage` is None if the backend does not report token counts
    """

    text: str
    model: str
    cli: str
    usage: dict | None = None


# =============================================================================
# BACKEND DISPATCH CONTRACT
# =============================================================================

# Canonical mapping from AiConfig.cli value to subprocess invocation behavior.
# This dict defines which backends are supported and their invocation patterns.
#
# Contract: Workers must implement support for exactly these three backends.
# Any other value in AiConfig.cli must raise NotImplementedError with a clear
# message listing the supported backends.
#
# Implementation note: The actual command templates are implementation details,
# but the contract requires:
#   - gemini: Invokes `gemini` CLI with prompt and model args
#   - codex: Invokes `codex` CLI with prompt and model args
#   - claude-code: Invokes `claude` CLI with prompt, model, and JSON output format

BACKEND_DISPATCH_MAP: dict[str, str] = {
    "gemini": "gemini CLI - Google Gemini via gemini tool",
    "codex": "codex CLI - OpenAI Codex via codex tool",
    "claude-code": "claude CLI - Anthropic Claude via claude-code tool",
}
"""Canonical backend name to description mapping.

Keys must match valid values for AiConfig.cli.
Values are human-readable descriptions for error messages.

Contract:
    - If AiConfig.cli is not in this dict, raise UnsupportedBackendError
      (which is a subtype of LlmError informing the user of valid options)
    - The dict is the single source of truth for supported backend names
    - Adding a new backend requires updating this dict AND the invoke implementations
"""

# Valid backend names for type checking and error messages
VALID_BACKENDS: tuple[str, ...] = tuple(BACKEND_DISPATCH_MAP.keys())
"""Tuple of valid AiConfig.cli values. Derived from BACKEND_DISPATCH_MAP."""


# =============================================================================
# TIMEOUT CONTRACT
# =============================================================================

# Default timeouts per invocation type. These are the architectural defaults.
# Callers may override via timeout_seconds parameter, but the contract
# establishes these as the baseline behavior.

DEFAULT_TEXT_TIMEOUT_SECONDS: int = 120
"""Default timeout for text-only LLM invocations.

Contract:
    - Used when invoke_text() is called without explicit timeout_seconds
    - Applies to the entire subprocess call (connection + response)
    - Must be sufficient for typical text generation tasks
    - Vision invocations use a separate, longer default
"""

DEFAULT_VISION_TIMEOUT_SECONDS: int = 180
"""Default timeout for vision (image + text) LLM invocations.

Contract:
    - Used when invoke_vision() is called without explicit timeout_seconds
    - Longer than text-only because vision processing takes more time
    - Applies to the entire subprocess call (connection + response)
"""


# =============================================================================
# RETRY CONTRACT
# =============================================================================

# Architectural decision: Two total attempts for transient failures.
# This is the single source of truth for retry behavior across all backends.

MAX_RETRY_ATTEMPTS: int = 2
"""Maximum number of invocation attempts for transient failures.

Contract:
    - Value is 2 (one initial attempt + one retry)
    - Applies to timeout errors, connection refused, and other transient failures
    - Does NOT apply to deterministic errors:
        - Invalid JSON response (LlmParseError) - no retry
        - Configuration errors (unknown backend) - no retry
        - Malformed prompts - no retry
    - Each attempt uses the same timeout value
    - Backends must respect this limit; no additional retry loops within
"""

# Categories of errors that are transient and eligible for retry.
# Errors not in these categories are deterministic and should NOT be retried.
TRANSIENT_ERROR_INDICATORS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection refused",
    "connection reset",
    "network",
    "econnrefused",
    "econnreset",
    "etimedout",
)
"""Substrings that indicate a transient/failure that can be retried.

Contract:
    - If a subprocess error message contains any of these (case-insensitive),
      it is eligible for retry up to MAX_RETRY_ATTEMPTS
    - If none match, treat as deterministic and raise immediately without retry
"""


# =============================================================================
# PUBLIC API - INVOCATION FUNCTIONS (STUBS)
# =============================================================================


async def invoke_text(
    prompt: str,
    ai_config: AiConfig,
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    timeout_seconds: int = DEFAULT_TEXT_TIMEOUT_SECONDS,
) -> LlmResponse:
    """Invoke LLM with a text-only prompt.

    Dispatches to the configured CLI backend:
        - gemini: `gemini -m {model} -p "{prompt}"`
        - codex: `codex -m {model} -p "{prompt}"`
        - claude-code: `claude -m {model} -p "{prompt}" --output-format json`

    Args:
        prompt: The user prompt text to send to the LLM.
        ai_config: Configuration specifying the backend (ai_config.cli) and model.
        system_prompt: Optional system prompt to prepend. Backend-specific handling.
        max_tokens: Maximum tokens for the response. Passed to the backend.
        temperature: Sampling temperature. Passed to the backend.
        timeout_seconds: Timeout for the entire invocation. Defaults to
            DEFAULT_TEXT_TIMEOUT_SECONDS.

    Returns:
        LlmResponse with raw text, model name, and cli backend used.

    Raises:
        LlmError: If invocation fails after MAX_RETRY_ATTEMPTS for transient errors.
        LlmError: If ai_config.cli is not in BACKEND_DISPATCH_MAP (unsupported backend).
        LlmParseError: Never raised here (parse errors only in parse_json_response).

    Contract Behavior:
        1. Backend dispatch: Look up ai_config.cli in BACKEND_DISPATCH_MAP.
           - If not found, raise LlmError with message listing VALID_BACKENDS.
        2. Subprocess invocation: Execute the CLI command with proper argument escaping.
        3. Timeout: Enforce timeout_seconds. On timeout, retry if attempts < MAX_RETRY_ATTEMPTS.
        4. Connection failures: Retry transient errors up to MAX_RETRY_ATTEMPTS.
        5. Deterministic errors: Raise immediately without retry.
        6. Return LlmResponse on success.

    Non-goals for implementation:
        - No streaming support (collect full response before returning)
        - No multi-turn conversation management (each call is independent)
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


async def invoke_vision(
    prompt: str,
    image_path: Path,
    ai_config: AiConfig,
    max_tokens: int = 4000,
    timeout_seconds: int = DEFAULT_VISION_TIMEOUT_SECONDS,
) -> LlmResponse:
    """Invoke LLM with an image + text prompt.

    Passes the image as a file reference to the CLI backend. Used for
    painting layout analysis where the LLM must "see" the artwork.

    Args:
        prompt: The user prompt text to send to the LLM.
        image_path: Path to the image file (PNG or JPEG). Must exist.
        ai_config: Configuration specifying the backend and model.
        max_tokens: Maximum tokens for the response. Passed to the backend.
        timeout_seconds: Timeout for the entire invocation. Defaults to
            DEFAULT_VISION_TIMEOUT_SECONDS (longer than text-only).

    Returns:
        LlmResponse with raw text, model name, and cli backend used.

    Raises:
        LlmError: If invocation fails after MAX_RETRY_ATTEMPTS for transient errors.
        LlmError: If image_path does not exist or is not a valid image file.
        LlmError: If ai_config.cli is not in BACKEND_DISPATCH_MAP.

    Contract Behavior:
        1. Validate image_path exists and is readable. Raise LlmError if not.
        2. Dispatch to backend. Each backend has different image argument syntax:
           - gemini: Tool-based file reading (may need to stage file first)
           - codex: Similar tool-based approach
           - claude-code: Native image support in prompt
        3. Apply DEFAULT_VISION_TIMEOUT_SECONDS by default (longer than text).
        4. Same retry semantics as invoke_text for transient failures.
        5. Return LlmResponse on success.

    Non-goals for implementation:
        - No image format conversion (caller must provide PNG or JPEG)
        - No image resizing (caller handles preprocessing)
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


async def compute_embedding(
    text: str,
    ai_config: AiConfig,
) -> list[float]:
    """Compute text embedding using the configured backend.

    Returns an embedding vector for semantic similarity matching between
    painting descriptions and news tone descriptions.

    Args:
        text: The text string to embed.
        ai_config: Configuration specifying the backend and model.

    Returns:
        Embedding vector as a list of floats (dimension depends on model).

    Raises:
        EmbeddingError: If embedding computation fails.
        EmbeddingError: If the backend does not support embedding (unlikely
            for the supported backends).

    Contract Behavior:
        1. Backend dispatch: Each backend has different embedding support:
           - gemini: Uses Gemini embedding API endpoint
           - codex: May fall back to local sentence-transformers
           - claude-code: May use Anthropic embedding or local fallback
        2. Return the embedding as list[float] (not numpy array - keep deps minimal).
        3. Embedding dimension is model-dependent. Callers should not assume.
        4. Retry semantics: Same as invoke_text for transient failures.
        5. Do NOT cache here - caller is responsible for caching.

    Backend-specific embedding strategies:
        - If the configured backend has a native embedding API, use it.
        - If not, fall back to a local model (sentence-transformers).
        - The fallback model should be documented in the error if unavailable.

    Non-goals for implementation:
        - No batch embedding (callers invoke per-text)
        - No embedding storage (caller handles caching)
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def parse_json_response(text: str) -> dict:
    """Parse LLM response as JSON, applying leniency transformations.

    LLMs often wrap JSON in markdown code fences or include trailing commas.
    This function handles these common issues before parsing.

    Args:
        text: Raw text response from an LLM (LlmResponse.text).

    Returns:
        Parsed JSON as a Python dict.

    Raises:
        LlmParseError: If the text cannot be parsed as valid JSON after
            applying all leniency transformations.

    Contract Behavior (leniency boundaries in order):
        1. Strip markdown code fences:
           - If text contains ```json ... ```, extract the inner content
           - If text contains ```...```, extract the inner content
           - Strip leading/trailing whitespace after fence removal
        2. Trim leading/trailing whitespace from the entire text.
        3. Tolerate trailing commas:
           - Remove trailing commas before ] and }
           - This is a common LLM output quirk
        4. Attempt JSON parse.
        5. If parse fails, raise LlmParseError with:
           - The original text (for debugging)
           - The transformed text (after leniency)
           - The JSON parse error message

    Example transformations:
        Input: '```json\\n{"key": "value",}\\n```'
        After step 1: '{"key": "value",}'
        After step 3: '{"key": "value"}'
        Parse result: {"key": "value"}

    Important:
        - This function does NOT validate the JSON schema
        - Caller is responsible for checking required fields
        - The parse is lenient but NOT lossy - it preserves all valid JSON data

    Non-goals for implementation:
        - No schema validation (caller handles that)
        - No YAML or other format support (JSON only)
        - No streaming parse (complete text required)
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


# =============================================================================
# INTERNAL HELPERS (NOT PUBLIC API)
# =============================================================================

# The following are implementation helpers that workers MAY use or implement.
# They are NOT part of the public contract and may change during implementation.


def _get_backend_command(
    cli: str,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None,
    image_path: Path | None,
) -> list[str]:
    """Build the subprocess command for a specific backend.

    This is an implementation helper, NOT public API.
    Workers may implement differently based on backend specifics.

    Contract (for reference during implementation):
        - Returns a list of strings suitable for subprocess.run()
        - Must properly escape/quote the prompt for shell safety
        - Must handle image_path=None vs image_path=Path for vision calls
        - Must pass max_tokens and temperature to backends that support them
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    This is an implementation helper for parse_json_response, NOT public API.

    Contract:
        - Handles ```json ... ``` and ``` ... ``` patterns
        - Returns stripped content without the fence markers
        - If no fences found, returns the original text trimmed
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] and }.

    This is an implementation helper for parse_json_response, NOT public API.

    Contract:
        - Removes commas that precede closing brackets/braces
        - Handles both ] and } cases
        - Does not modify other content
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )


def _check_transient_error(error_message: str) -> bool:
    """Check if an error message indicates a transient failure eligible for retry.

    This is an implementation helper, NOT public API.

    Contract:
        - Returns True if error_message contains any TRANSIENT_ERROR_INDICATORS
        - Case-insensitive matching
        - Returns False for all other errors (deterministic)
    """
    raise NotImplementedError(
        "Contract-only stub: implementation deferred to a later step"
    )
