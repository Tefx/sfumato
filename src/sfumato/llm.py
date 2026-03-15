"""LLM backend invocation and response parsing.

This module provides a unified interface for invoking LLM backends (gemini CLI, codex CLI,
claude-code CLI) via subprocess, or via LiteLLM SDK (openrouter, google, openai).

Architecture reference: ARCHITECTURE.md#2.9
"""

from __future__ import annotations

import asyncio
import base64
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
        cli: The backend that was invoked:
            - "gemini" | "codex" | "claude-code" for CLI backends
            - "sdk" for SDK-based invocations (openrouter, google, openai)
        usage: Optional token usage dict with keys like 'prompt_tokens',
            'completion_tokens', 'total_tokens'. Present when the backend
            provides it (always present for SDK, may be None for CLI).

    Contract:
        - Instances are immutable (frozen dataclass)
        - `text` is the raw output from the LLM (CLI or SDK), including any markdown fencing
        - `model` reflects the model that actually responded
        - `cli` is "sdk" for SDK invocations, or the CLI backend name for CLI invocations
        - `usage` is available for SDK invocations, may be None for CLI invocations
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
# SDK PROVIDER MAPPING
# =============================================================================

# Valid values for ai_config.backend field
VALID_BACKEND_TYPES: tuple[str, ...] = ("cli", "sdk")
"""Valid values for ai_config.backend field."""

# Valid SDK providers for ai_config.sdk_provider field
VALID_SDK_PROVIDERS: tuple[str, ...] = ("openrouter", "google", "openai")
"""Valid SDK providers when backend='sdk'."""

# Provider prefix mapping for LiteLLM model names
SDK_PROVIDER_PREFIX_MAP: dict[str, str] = {
    "openrouter": "openrouter/",
    "google": "gemini/",
    "openai": "",  # OpenAI models use the model name directly
}
"""Mapping from ai_config.sdk_provider to LiteLLM model name prefix."""

# OpenRouter model ID mapping: short name → full OpenRouter ID
# Users write "gemini-3-flash-preview" in config, we map to "google/gemini-3-flash-preview"
_OPENROUTER_MODEL_PREFIX_GUESSES: dict[str, str] = {
    "gemini": "google/",
    "gpt": "openai/",
    "o3": "openai/",
    "o4": "openai/",
    "claude": "anthropic/",
}


def _map_sdk_model(sdk_provider: str, model: str) -> str:
    """Map SDK provider and model to LiteLLM model name.

    Contract:
        - If model already contains "/", use as-is with provider prefix
        - openrouter: infer vendor prefix from model name (gemini→google/, gpt→openai/)
        - google: gemini/{model}
        - openai: {model} (no prefix)

    Args:
        sdk_provider: The SDK provider ("openrouter" | "google" | "openai").
        model: The model identifier from ai_config.model.

    Returns:
        LiteLLM-formatted model string.

    Raises:
        LlmError: If sdk_provider is not in VALID_SDK_PROVIDERS.
    """
    if sdk_provider not in SDK_PROVIDER_PREFIX_MAP:
        valid_providers = ", ".join(repr(p) for p in VALID_SDK_PROVIDERS)
        raise LlmError(
            f"Unsupported SDK provider '{sdk_provider}'. Valid providers: {valid_providers}"
        )

    prefix = SDK_PROVIDER_PREFIX_MAP[sdk_provider]

    if sdk_provider == "openrouter" and "/" not in model:
        # Infer vendor prefix from model name
        vendor_prefix = ""
        for hint, vp in _OPENROUTER_MODEL_PREFIX_GUESSES.items():
            if model.startswith(hint):
                vendor_prefix = vp
                break
        return f"{prefix}{vendor_prefix}{model}"

    return f"{prefix}{model}" if prefix else model


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
# SDK INVOCATION HELPERS
# =============================================================================


async def _invoke_sdk_completion(
    prompt: str,
    ai_config: AiConfig,
    system_prompt: str | None = None,
    max_tokens: int = 4000,
    temperature: float = 0.3,
    timeout_seconds: int = DEFAULT_TEXT_TIMEOUT_SECONDS,
    image_path: Path | None = None,
) -> LlmResponse:
    """Invoke LLM via LiteLLM SDK.

    This is an internal helper for SDK-based invocations.

    Args:
        prompt: The user prompt text to send to the LLM.
        ai_config: Configuration specifying SDK provider and model.
        system_prompt: Optional system prompt.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.
        timeout_seconds: Timeout for the entire invocation.
        image_path: Optional image path for vision calls.

    Returns:
        LlmResponse with raw text, model name, and cli="sdk".

    Raises:
        LlmError: If SDK invocation fails or provider is invalid.
    """
    import os

    try:
        import litellm
    except ImportError as e:
        raise LlmError(
            "LiteLLM SDK is required for backend='sdk'. "
            "Install with: pip install litellm"
        ) from e

    # Ensure API key env var is set (support common aliases)
    if ai_config.sdk_provider == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        # Try common aliases
        for alias in ("OPENROUTER_KEY", "OR_API_KEY"):
            val = os.environ.get(alias)
            if val:
                os.environ["OPENROUTER_API_KEY"] = val
                break

    # Validate provider
    if ai_config.sdk_provider not in VALID_SDK_PROVIDERS:
        valid_providers = ", ".join(repr(p) for p in VALID_SDK_PROVIDERS)
        raise LlmError(
            f"Unsupported SDK provider '{ai_config.sdk_provider}'. "
            f"Valid providers: {valid_providers}"
        )

    # Map provider + model to LiteLLM format
    model = _map_sdk_model(ai_config.sdk_provider, ai_config.model)

    # Build messages
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Handle vision (image) payloads
    if image_path:
        if not image_path.exists():
            raise LlmError(f"Image file not found: {image_path}")
        if not image_path.is_file():
            raise LlmError(f"Image path is not a file: {image_path}")

        # Read and encode image
        try:
            image_bytes = image_path.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            # Detect image type from extension
            suffix = image_path.suffix.lower()
            mime_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}"
                            },
                        },
                    ],
                }
            )
        except OSError as e:
            raise LlmError(f"Failed to read image file {image_path}: {e}") from e
    else:
        messages.append({"role": "user", "content": prompt})

    # Invoke with retry
    last_error: str = ""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    litellm.completion,
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                timeout=timeout_seconds,
            )

            # Extract response text
            text = response.choices[0].message.content or ""

            # Extract usage if available
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LlmResponse(
                text=text,
                model=ai_config.model,
                cli="sdk",
                usage=usage,
            )

        except asyncio.TimeoutError:
            last_error = f"Timeout after {timeout_seconds}s"
            if attempt < MAX_RETRY_ATTEMPTS:
                continue
            raise LlmError(
                f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
                f"(backend: sdk, provider: {ai_config.sdk_provider}): {last_error}"
            ) from None

        except Exception as e:
            error_msg = str(e)
            if _check_transient_error(error_msg.lower()):
                last_error = error_msg
                if attempt < MAX_RETRY_ATTEMPTS:
                    continue
                raise LlmError(
                    f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
                    f"(backend: sdk, provider: {ai_config.sdk_provider}): {error_msg}"
                ) from None
            else:
                # Deterministic error
                raise LlmError(
                    f"LLM invocation failed (backend: sdk, provider: {ai_config.sdk_provider}): "
                    f"{error_msg}"
                ) from e

    # This should never be reached
    raise LlmError(
        f"LLM invocation failed after {MAX_RETRY_ATTEMPTS} attempts "
        f"(backend: sdk, provider: {ai_config.sdk_provider}): {last_error}"
    )


async def _invoke_sdk_embedding(
    text: str,
    ai_config: AiConfig,
) -> list[float]:
    """Invoke embedding via LiteLLM SDK.

    Args:
        text: The text string to embed.
        ai_config: Configuration specifying SDK provider and model.

    Returns:
        Embedding vector as list of floats.

    Raises:
        EmbeddingError: If embedding computation fails.
    """
    try:
        import litellm
    except ImportError as e:
        raise EmbeddingError(
            "LiteLLM SDK is required for backend='sdk'. "
            "Install with: pip install litellm"
        ) from e

    # Validate provider
    if ai_config.sdk_provider not in VALID_SDK_PROVIDERS:
        valid_providers = ", ".join(repr(p) for p in VALID_SDK_PROVIDERS)
        raise EmbeddingError(
            f"Unsupported SDK provider '{ai_config.sdk_provider}'. "
            f"Valid providers: {valid_providers}"
        )

    # Map provider + model to LiteLLM format
    model = _map_sdk_model(ai_config.sdk_provider, ai_config.model)

    last_error: str = ""
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    litellm.embedding,
                    model=model,
                    input=[text],
                ),
                timeout=DEFAULT_TEXT_TIMEOUT_SECONDS,
            )

            # Extract embedding vector
            embedding = response.data[0]["embedding"]
            if isinstance(embedding, list) and all(
                isinstance(x, (int, float)) for x in embedding
            ):
                return [float(x) for x in embedding]

            raise EmbeddingError(
                f"Invalid embedding response from {ai_config.sdk_provider}: "
                f"expected list of floats, got {type(embedding).__name__}"
            )

        except asyncio.TimeoutError:
            last_error = f"Timeout after {DEFAULT_TEXT_TIMEOUT_SECONDS}s"
            if attempt < MAX_RETRY_ATTEMPTS:
                continue
            raise EmbeddingError(
                f"Embedding failed after {MAX_RETRY_ATTEMPTS} attempts "
                f"(backend: sdk, provider: {ai_config.sdk_provider}): {last_error}"
            ) from None

        except EmbeddingError:
            raise  # Re-raise EmbeddingError directly

        except Exception as e:
            error_msg = str(e)
            if _check_transient_error(error_msg.lower()):
                last_error = error_msg
                if attempt < MAX_RETRY_ATTEMPTS:
                    continue
                raise EmbeddingError(
                    f"Embedding failed after {MAX_RETRY_ATTEMPTS} attempts "
                    f"(backend: sdk, provider: {ai_config.sdk_provider}): {error_msg}"
                ) from None
            else:
                raise EmbeddingError(
                    f"Embedding failed (backend: sdk, provider: {ai_config.sdk_provider}): "
                    f"{error_msg}"
                ) from e

    # This should never be reached
    raise EmbeddingError(
        f"Embedding failed after {MAX_RETRY_ATTEMPTS} attempts "
        f"(backend: sdk, provider: {ai_config.sdk_provider}): {last_error}"
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

    Dispatches based on ai_config.backend:
        - backend="cli": Use subprocess to invoke CLI backend (gemini, codex, claude-code)
        - backend="sdk": Use LiteLLM SDK to invoke API (openrouter, google, openai)

    For CLI backends:
        - gemini: `gemini -p "{prompt}" -y --sandbox false [-m {model}]`
        - codex: `codex exec --full-auto "{prompt}"`
        - claude-code: `claude -p "{prompt}" --output-format json [-m {model}] --max-tokens {n}`

    For SDK backends:
        - openrouter: Uses litellm.completion with model "openrouter/{model}"
        - google: Uses litellm.completion with model "gemini/{model}"
        - openai: Uses litellm.completion with model "{model}"

    Args:
        prompt: The user prompt text to send to the LLM.
        ai_config: Configuration specifying the backend (ai_config.backend and
            ai_config.cli for CLI, or ai_config.sdk_provider for SDK).
        system_prompt: Optional system prompt to prepend. Backend-specific handling.
        max_tokens: Maximum tokens for the response. Passed to the backend.
        temperature: Sampling temperature. Passed to the backend.
        timeout_seconds: Timeout for the entire invocation. Defaults to
            DEFAULT_TEXT_TIMEOUT_SECONDS.

    Returns:
        LlmResponse with raw text, model name, and cli backend used.
            For SDK invocations, cli="sdk".

    Raises:
        LlmError: If invocation fails after MAX_RETRY_ATTEMPTS for transient errors.
        LlmError: If unsupported backend or invalid configuration.

    Contract Behavior:
        CLI path (backend="cli"):
            1. Validate ai_config.cli is in BACKEND_DISPATCH_MAP.
            2. Subprocess invocation with proper argument escaping.
            3. Timeout enforcement with retry for transient errors.
            4. Return LlmResponse(cli=ai_config.cli).

        SDK path (backend="sdk"):
            1. Validate ai_config.sdk_provider is in VALID_SDK_PROVIDERS.
            2. Map provider + model to LiteLLM format.
            3. Invoke with retry for transient errors.
            4. Return LlmResponse(cli="sdk").

    Non-goals for implementation:
        - No streaming support (collect full response before returning)
        - No multi-turn conversation management (each call is independent)
    """
    # Route based on backend type
    if ai_config.backend == "sdk":
        return await _invoke_sdk_completion(
            prompt=prompt,
            ai_config=ai_config,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            image_path=None,
        )

    # CLI path
    if ai_config.backend not in VALID_BACKEND_TYPES:
        valid_backend_types = ", ".join(repr(b) for b in VALID_BACKEND_TYPES)
        raise LlmError(
            f"Unsupported backend type '{ai_config.backend}'. "
            f"Valid types: {valid_backend_types}"
        )

    # Validate CLI backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise LlmError(
            f"Unsupported CLI backend '{ai_config.cli}'. Valid backends: {valid_options}"
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

    Dispatches based on ai_config.backend:
        - backend="cli": Use subprocess to invoke CLI backend (gemini, codex, claude-code)
        - backend="sdk": Use LiteLLM SDK with base64-encoded image

    For CLI backends:
        - Passes image as file reference in prompt text
        - Each backend has different image handling approach

    For SDK backends:
        - Reads and base64-encodes the image
        - Passes as image_url content in messages

    Args:
        prompt: The user prompt text to send to the LLM.
        image_path: Path to the image file (PNG or JPEG). Must exist.
        ai_config: Configuration specifying the backend and model.
        max_tokens: Maximum tokens for the response. Passed to the backend.
        timeout_seconds: Timeout for the entire invocation. Defaults to
            DEFAULT_VISION_TIMEOUT_SECONDS (longer than text-only).

    Returns:
        LlmResponse with raw text, model name, and cli backend used.
            For SDK invocations, cli="sdk".

    Raises:
        LlmError: If invocation fails after MAX_RETRY_ATTEMPTS for transient errors.
        LlmError: If image_path does not exist or is not a valid image file.
        LlmError: If unsupported backend or invalid configuration.

    Contract Behavior:
        CLI path (backend="cli"):
            1. Validate image_path exists and is readable.
            2. Validate ai_config.cli is in BACKEND_DISPATCH_MAP.
            3. Dispatch to CLI with image file reference in prompt.
            4. Same retry semantics as invoke_text for transient failures.
            5. Return LlmResponse(cli=ai_config.cli).

        SDK path (backend="sdk"):
            1. Validate image_path exists and is readable.
            2. Validate ai_config.sdk_provider is in VALID_SDK_PROVIDERS.
            3. Base64-encode image and pass as multimodal content.
            4. Same retry semantics with LiteLLM.
            5. Return LlmResponse(cli="sdk").

    Non-goals for implementation:
        - No image format conversion (caller must provide PNG or JPEG)
        - No image resizing (caller handles preprocessing)
    """
    # Validate image path exists (required for both CLI and SDK)
    if not image_path.exists():
        raise LlmError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise LlmError(f"Image path is not a file: {image_path}")

    # Route based on backend type
    if ai_config.backend == "sdk":
        return await _invoke_sdk_completion(
            prompt=prompt,
            ai_config=ai_config,
            system_prompt=None,
            max_tokens=max_tokens,
            temperature=0.3,  # Default temperature for vision calls
            timeout_seconds=timeout_seconds,
            image_path=image_path,
        )

    # CLI path
    if ai_config.backend not in VALID_BACKEND_TYPES:
        valid_backend_types = ", ".join(repr(b) for b in VALID_BACKEND_TYPES)
        raise LlmError(
            f"Unsupported backend type '{ai_config.backend}'. "
            f"Valid types: {valid_backend_types}"
        )

    # Validate CLI backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise LlmError(
            f"Unsupported CLI backend '{ai_config.cli}'. Valid backends: {valid_options}"
        )

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

    Dispatches based on ai_config.backend:
        - backend="cli": Use subprocess CLI for CLI-based embedding
        - backend="sdk": Use LiteLLM SDK for embedding

    Args:
        text: The text string to embed.
        ai_config: Configuration specifying the backend and model.

    Returns:
        Embedding vector as a list of floats (dimension depends on model).

    Raises:
        EmbeddingError: If embedding computation fails.
        EmbeddingError: If unsupported backend or invalid configuration.

    Contract Behavior:
        CLI path (backend="cli"):
            1. Validate ai_config.cli is in BACKEND_DISPATCH_MAP.
            2. Invoke CLI embedding command.
            3. Parse JSON response for embedding vector.
            4. Return list[float].

        SDK path (backend="sdk"):
            1. Validate ai_config.sdk_provider is in VALID_SDK_PROVIDERS.
            2. Map provider + model to LiteLLM format.
            3. Call litellm.embedding().
            4. Return list[float].

    Non-goals for implementation:
        - No batch embedding (callers invoke per-text)
        - No embedding storage (caller handles caching)
    """
    # Route based on backend type
    if ai_config.backend == "sdk":
        return await _invoke_sdk_embedding(text=text, ai_config=ai_config)

    # CLI path
    if ai_config.backend not in VALID_BACKEND_TYPES:
        valid_backend_types = ", ".join(repr(b) for b in VALID_BACKEND_TYPES)
        raise EmbeddingError(
            f"Unsupported backend type '{ai_config.backend}'. "
            f"Valid types: {valid_backend_types}"
        )

    # Validate CLI backend
    if ai_config.cli not in BACKEND_DISPATCH_MAP:
        valid_options = ", ".join(repr(b) for b in VALID_BACKENDS)
        raise EmbeddingError(
            f"Unsupported CLI backend '{ai_config.cli}'. Valid backends: {valid_options}"
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
