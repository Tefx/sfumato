"""LLM backend invocation and response parsing.

This module provides a unified interface for invoking LLM backends (gemini CLI, codex CLI,
claude-code CLI) via subprocess. It handles prompt construction, response parsing, retries,
and timeout behavior.

Architecture reference: ARCHITECTURE.md#2.9
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sfumato.config import AiConfig

if TYPE_CHECKING:
    pass


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
# Any other value in AiConfig.cli must raise LlmError with a clear
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
    - If AiConfig.cli is not in this dict, raise LlmError
      with a message listing valid options
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
# INTERNAL HELPERS
# =============================================================================


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences from LLM output.

    This is an implementation helper for parse_json_response, NOT public API.

    Contract:
        - Handles ```json ... ``` and ``` ... ``` patterns
        - Returns stripped content without the fence markers
        - If no fences found, returns the original text trimmed

    Args:
        text: Raw LLM response that may contain code fences.

    Returns:
        Text with fences stripped, trimmed of leading/trailing whitespace.
    """
    text = text.strip()

    # Pattern 1: ```json ... ```
    json_fence_match = re.search(
        r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL | re.IGNORECASE
    )
    if json_fence_match:
        return json_fence_match.group(1).strip()

    # Pattern 2: ``` ... ``` (generic fence)
    # Look for backtick fence at start
    if text.startswith("```"):
        # Find the end of the opening fence
        end_of_lang = text.find("\n", 3)
        if end_of_lang != -1:
            # Find the closing fence
            close_fence = text.rfind("```")
            if close_fence > end_of_lang:
                return text[end_of_lang + 1 : close_fence].strip()

    # Pattern 3: Fence in middle of text - extract JSON from inside
    fence_match = re.search(r"```(?:\w*)\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return text


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] and }.

    This is an implementation helper for parse_json_response, NOT public API.

    Contract:
        - Removes commas that precede closing brackets/braces
        - Handles both ] and } cases
        - Does not modify other content

    Args:
        text: JSON text that may have trailing commas.

    Returns:
        Text with trailing commas removed.
    """
    # Remove trailing comma before ]
    text = re.sub(r",\s*]", "]", text)
    # Remove trailing comma before }
    text = re.sub(r",\s*}", "}", text)
    return text


def _check_transient_error(error_message: str) -> bool:
    """Check if an error message indicates a transient failure eligible for retry.

    This is an implementation helper, NOT public API.

    Contract:
        - Returns True if error_message contains any TRANSIENT_ERROR_INDICATORS
        - Case-insensitive matching
        - Returns False for all other errors (deterministic)

    Args:
        error_message: The error message to check.

    Returns:
        True if the error appears to be transient, False otherwise.
    """
    lower_message = error_message.lower()
    return any(indicator in lower_message for indicator in TRANSIENT_ERROR_INDICATORS)


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

    Contract (for reference during implementation):
        - Returns a list of strings suitable for subprocess.run()
        - Must properly escape/quote the prompt for shell safety
        - Must handle image_path=None vs image_path=Path for vision calls
        - Must pass max_tokens and temperature to backends that support them

    Args:
        cli: Backend name from BACKEND_DISPATCH_MAP.
        prompt: User prompt text.
        model: Model identifier.
        max_tokens: Maximum tokens for response.
        temperature: Sampling temperature.
        system_prompt: Optional system prompt.
        image_path: Optional image path for vision calls.

    Returns:
        List of command arguments.

    Raises:
        LlmError: If cli is not a valid backend.
    """
    if cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise LlmError(f"Unsupported backend '{cli}'. Valid backends: {valid_options}")

    if cli == "gemini":
        # Validated flags from PROTOTYPING.md#4:
        #   gemini -p "prompt" -y --sandbox false
        # gemini CLI does NOT support: --max-tokens, --temperature, --system, --image
        # System prompt and image references must be embedded in the prompt text.
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        if image_path:
            # gemini reads files via its tool system when referenced in prompt
            full_prompt = f"Look at the image file {image_path} .\n\n{full_prompt}"
        cmd = ["gemini", "-p", full_prompt, "-y", "--sandbox", "false"]
        if model:
            cmd.extend(["-m", model])
        return cmd

    if cli == "codex":
        # Validated flags from PROTOTYPING.md#4:
        #   codex exec --full-auto "prompt"
        # codex exec does NOT support: --max-tokens, --temperature, --system, -m
        # Note: Model selection is handled by codex itself, not via -m flag
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        if image_path:
            full_prompt = f"Look at the image file {image_path} .\n\n{full_prompt}"
        cmd = ["codex", "exec", "--full-auto", full_prompt]
        return cmd

    if cli == "claude-code":
        # claude -p "prompt" --output-format json -m {model} --max-tokens {n}
        # claude supports: -p, -m, --max-tokens, --output-format
        # Note: --output-format should be "json" per ARCHITECTURE.md
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        if image_path:
            full_prompt = f"Look at the image file {image_path} .\n\n{full_prompt}"
        cmd = ["claude", "-p", full_prompt, "--output-format", "json"]
        if model:
            cmd.extend(["-m", model])
        cmd.extend(["--max-tokens", str(max_tokens)])
        return cmd

    # This should never be reached due to the check above
    valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
    raise LlmError(f"Unsupported backend '{cli}'. Valid backends: {valid_options}")


async def _run_subprocess_with_retry(
    cmd: list[str],
    timeout_seconds: int,
    cli: str,
) -> str:
    """Run subprocess with retry logic for transient errors.

    This is an internal helper that implements the retry contract.

    Args:
        cmd: Command to execute.
        timeout_seconds: Timeout for each attempt.
        cli: Backend name for error messages.

    Returns:
        stdout content on success.

    Raises:
        LlmError: After retries exhausted for transient errors.
        LlmError: Immediately for deterministic errors.
    """
    last_error: str = ""

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                if not error_msg:
                    error_msg = f"Process exited with code {process.returncode}"

                if _check_transient_error(error_msg):
                    last_error = error_msg
                    if attempt < MAX_RETRY_ATTEMPTS:
                        continue  # Retry
                    raise LlmError(
                        f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
                        f"(backend: {cli}): {error_msg}"
                    )
                else:
                    # Deterministic error, no retry
                    raise LlmError(
                        f"LLM invocation failed (backend: {cli}): {error_msg}"
                    )

            result = stdout.decode("utf-8", errors="replace")
            return result

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout_seconds}s"
            if attempt < MAX_RETRY_ATTEMPTS:
                continue  # Retry
            raise LlmError(
                f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
                f"(backend: {cli}): {last_error}"
            ) from None

        except FileNotFoundError:
            # Binary not found - deterministic error
            raise LlmError(
                f"Backend binary not found: {cli}. "
                f"Please ensure '{cli}' is installed and available on PATH."
            ) from None

        except PermissionError as e:
            # Permission denied - deterministic error
            raise LlmError(
                f"Permission denied executing backend '{cli}': {e}"
            ) from None

        except OSError as e:
            error_msg = str(e)
            if _check_transient_error(error_msg):
                last_error = error_msg
                if attempt < MAX_RETRY_ATTEMPTS:
                    continue  # Retry
                raise LlmError(
                    f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
                    f"(backend: {cli}): {error_msg}"
                ) from None
            else:
                # Deterministic OS error
                raise LlmError(
                    f"LLM invocation failed (backend: {cli}): {error_msg}"
                ) from None

    # This should never be reached, but satisfy type checker
    raise LlmError(
        f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
        f"(backend: {cli}): {last_error}"
    )


# =============================================================================
# PUBLIC API - INVOCATION FUNCTIONS
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
        - gemini: `gemini -p "{prompt}" -y --sandbox false [-m {model}]`
        - codex: `codex exec --full-auto "{prompt}"`
        - claude-code: `claude -p "{prompt}" --output-format json [-m {model}] --max-tokens {n}`

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
    # Validate backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise LlmError(
            f"Unsupported backend '{ai_config.cli}'. Valid backends: {valid_options}"
        )

    # Build command
    cmd = _get_backend_command(
        cli=ai_config.cli,
        prompt=prompt,
        model=ai_config.model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        image_path=None,
    )

    # Execute with retry
    result = await _run_subprocess_with_retry(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        cli=ai_config.cli,
    )

    return LlmResponse(
        text=result,
        model=ai_config.model,
        cli=ai_config.cli,
        usage=None,  # CLI outputs may not include usage info
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
    # Validate backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise LlmError(
            f"Unsupported backend '{ai_config.cli}'. Valid backends: {valid_options}"
        )

    # Validate image path
    if not image_path.exists():
        raise LlmError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise LlmError(f"Image path is not a file: {image_path}")

    # Build command
    cmd = _get_backend_command(
        cli=ai_config.cli,
        prompt=prompt,
        model=ai_config.model,
        max_tokens=max_tokens,
        temperature=0.3,  # Default temperature for vision calls
        system_prompt=None,  # Vision calls typically don't use system prompt
        image_path=image_path,
    )

    # Execute with retry
    result = await _run_subprocess_with_retry(
        cmd=cmd,
        timeout_seconds=timeout_seconds,
        cli=ai_config.cli,
    )

    return LlmResponse(
        text=result,
        model=ai_config.model,
        cli=ai_config.cli,
        usage=None,
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
    # Validate backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise EmbeddingError(
            f"Unsupported backend '{ai_config.cli}'. Valid backends: {valid_options}"
        )

    # Build embedding command based on backend
    # Note: Different backends have different embedding APIs
    if ai_config.cli == "gemini":
        # gemini has an embedding API
        cmd = ["gemini", "embed", "-m", ai_config.model, "-t", text]
    elif ai_config.cli == "codex":
        # codex may need external embedding support
        cmd = ["codex", "embed", "-m", ai_config.model, "-t", text]
    elif ai_config.cli == "claude-code":
        # claude-code may have embedding support
        cmd = ["claude", "embed", "-m", ai_config.model, "-t", text]
    else:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise EmbeddingError(
            f"Unsupported backend '{ai_config.cli}'. Valid backends: {valid_options}"
        )

    try:
        result = await _run_subprocess_with_retry(
            cmd=cmd,
            timeout_seconds=DEFAULT_TEXT_TIMEOUT_SECONDS,
            cli=ai_config.cli,
        )

        # Parse the embedding from the result
        # The CLI should output JSON with the embedding vector
        parsed = parse_json_response(result)

        if "embedding" in parsed:
            embedding = parsed["embedding"]
            if isinstance(embedding, list) and all(
                isinstance(x, (int, float)) for x in embedding
            ):
                return [float(x) for x in embedding]

        if "vector" in parsed:
            vector = parsed["vector"]
            if isinstance(vector, list) and all(
                isinstance(x, (int, float)) for x in vector
            ):
                return [float(x) for x in vector]

        # Try direct list output
        if isinstance(parsed, list):
            if all(isinstance(x, (int, float)) for x in parsed):
                return [float(x) for x in parsed]

        raise EmbeddingError(
            f"Invalid embedding response from {ai_config.cli}: "
            f"expected embedding list, got {type(parsed).__name__}"
        )

    except LlmParseError as e:
        raise EmbeddingError(
            f"Failed to parse embedding response from {ai_config.cli}: {e}"
        ) from e
    except LlmError as e:
        raise EmbeddingError(
            f"Embedding computation failed for {ai_config.cli}: {e}"
        ) from e


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
    # Store original for error messages
    original_text = text

    # Step 1 & 2: Strip code fences and trim whitespace
    stripped = _strip_code_fences(text)

    # Step 3: Remove trailing commas
    cleaned = _remove_trailing_commas(stripped)

    # Step 4: Attempt JSON parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
        else:
            raise LlmParseError(
                f"Expected JSON object (dict), got {type(result).__name__}. "
                f"Original: {original_text[:200]}..."
            )
    except json.JSONDecodeError as e:
        # Step 5: Raise with context
        raise LlmParseError(
            f"Failed to parse JSON response: {e.msg}. "
            f"Original (first 200 chars): {original_text[:200]}..., "
            f"Transformed: {cleaned[:200]}..."
        ) from e
