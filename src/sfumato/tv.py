"""
Samsung The Frame TV connection and Art Mode control.

Contract module defining the public interface for TV operations.
All network behavior is stubbed pending implementation dispatch.

CONTRACT BOUNDARIES (ENFORCED):
- check_status(): Non-throwing on EVERY failure (connection, art-mode init, status listing).
  Returns TvStatus(reachable=False, error=...) rather than propagating exceptions.
- clean_old_uploads(): Non-throwing for individual delete failures. Continues cleanup,
  preserves retained set, returns count of confirmed deletions only.
- is_available_for_push(): Pure convenience wrapper over check_status() fields.
  No new network semantics beyond what check_status() provides.
- Timeout boundary: 10 seconds for status probing (hard limit).
- Error mapping: TvConnectionError for connection/setup failures,
  TvUploadError for upload-path failures after connection/art-mode setup.

KNOWN RISKS (from PROTOTYPING.md#1):
- set_artmode(True) may hang on validated TV model: This module MUST NOT introduce
  that path while implementing display switching. Use select_image only.
- First-pairing prompts and transient websocket setup failures MUST be surfaced as
  contracted reachable=False/error states, NOT uncategorized exceptions.
- get_thumbnail() hangs indefinitely on 2024 model: Do NOT use this API.

NON-GOALS (this contract step):
- No working SamsungTVWS upload/list/delete logic.
- No completed network behavior beyond stubs and contract notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sfumato.config import TvConfig


# =============================================================================
# Public Data Definitions
# =============================================================================


@dataclass(frozen=True)
class TvStatus:
    """
    TV connectivity and Art Mode status.

    Contract: check_status() ALWAYS returns this dataclass, never raises.
    On any failure (connection, art-mode init, status listing), returns
    TvStatus(reachable=False, error="<description>").

    Attributes:
        reachable: Can we establish a connection to the TV?
        art_mode_supported: Does this TV model support Art Mode?
            False if reachable is False.
        art_mode_active: Is the TV currently in Art Mode?
            False if reachable is False or art_mode_supported is False.
        uploaded_count: Number of images currently uploaded to Art Mode.
            0 if reachable is False.
        error: Human-readable error description if reachable is False.
            None if reachable is True.
    """

    reachable: bool
    art_mode_supported: bool
    art_mode_active: bool
    uploaded_count: int
    error: str | None = None


@dataclass(frozen=True)
class UploadedImage:
    """
    Metadata for an image uploaded to the TV's Art Mode.

    Attributes:
        content_id: Samsung's unique content ID for the image (e.g., "MY_F0001").
        file_name: Original file name if available from TV metadata, else None.
            TV may not provide original filenames for user uploads.
    """

    content_id: str
    file_name: str | None = None


# =============================================================================
# Public Exception Definitions
# =============================================================================


class TvError(Exception):
    """Base exception for TV operations."""

    pass


class TvConnectionError(TvError):
    """
    TV is unreachable or connection was refused.

    Raised by:
        - upload_image(): when TV cannot be reached before upload attempt
        - set_displayed(): when TV cannot be reached
        - list_uploaded(): when TV cannot be reached
        - delete_uploaded(): when TV cannot be reached

    Includes connection failures, websocket setup failures, and
    first-pairing prompt scenarios (user hasn't accepted pairing on TV).
    """

    pass


class TvUploadError(TvError):
    """
    Image upload to TV failed.

    Raised by:
        - upload_image(): when upload fails after successful connection/art-mode setup

    Does NOT include connection failures (those raise TvConnectionError).
    Use for: invalid image format, TV rejection, storage full, etc.
    """

    pass


# =============================================================================
# Public Function Signatures (Stubs)
# =============================================================================


def check_status(tv_config: "TvConfig") -> TvStatus:
    """
    Check TV connectivity and Art Mode status.

    CONTRACT (ENFORCED):
        - NON-THROWING: Always returns TvStatus, never raises exceptions.
        - On any failure (connection refused, art-mode not supported, status listing failure),
          returns TvStatus(reachable=False, error="<description>").
        - TIMEOUT: Maximum 10 seconds for entire status probing operation.
          If timeout exceeded, returns TvStatus(reachable=False, error="timeout after 10s").
        - First-pairing prompts and transient websocket setup failures surface as
          reachable=False with descriptive error field.

    Args:
        tv_config: TV configuration containing IP, port, and settings.

    Returns:
        TvStatus with reachability, Art Mode support/active status, upload count,
        and optional error description.

    Implementation Notes:
        - Uses samsungtvws SamsungTVWS(host=ip, port=port, name=...)
        - Calls art().supported() for art_mode_supported
        - Calls art().get_artmode() for art_mode_active
        - Calls art().available() for uploaded_count
        - MUST NOT use set_artmode() (known hang risk)
    """
    raise NotImplementedError("Contract stub - not implemented")


def upload_image(tv_config: "TvConfig", image_path: Path) -> str:
    """
    Upload a PNG image to the TV's Art Mode.

    Error Contract:
        - TvConnectionError: TV is unreachable, connection refused, or
          first-pairing not completed.
        - TvUploadError: Upload failed after connection was established
          (invalid format, TV rejection, storage issues).

    Args:
        tv_config: TV configuration containing IP, port, and settings.
        image_path: Path to PNG image file to upload.

    Returns:
        Content ID assigned by the TV (e.g., "MY_F0003").

    Raises:
        TvConnectionError: If TV cannot be reached.
        TvUploadError: If upload fails after connection.

    Implementation Notes:
        - Uses samsungtvws art().upload(data, file_type='PNG', matte='none')
        - Content IDs follow pattern MY_F followed by incrementing number
    """
    raise NotImplementedError("Contract stub - not implemented")


def set_displayed(tv_config: "TvConfig", content_id: str) -> None:
    """
    Switch the TV to display a specific uploaded image.

    Behavior Note:
        - TV screen must be on/Art Mode active to see the change.
        - If TV is in standby, the image is selected but screen stays dark.

    Args:
        tv_config: TV configuration containing IP, port, and settings.
        content_id: Content ID of the image to display.

    Raises:
        TvConnectionError: If TV cannot be reached.

    Implementation Notes:
        - Uses samsungtvws art().select_image(content_id, show=True)
        - MUST NOT use set_artmode(True) (known hang risk from PROTOTYPING.md#1)
    """
    raise NotImplementedError("Contract stub - not implemented")


def list_uploaded(tv_config: "TvConfig") -> list[UploadedImage]:
    """
    List all images currently uploaded to the TV's Art Mode.

    Args:
        tv_config: TV configuration containing IP, port, and settings.

    Returns:
        List of UploadedImage objects with content_id and optional file_name.

    Raises:
        TvConnectionError: If TV cannot be reached.

    Implementation Notes:
        - Uses samsungtvws art().available()
        - TV metadata may include image date for recency ordering
    """
    raise NotImplementedError("Contract stub - not implemented")


def delete_uploaded(tv_config: "TvConfig", content_id: str) -> None:
    """
    Delete a specific uploaded image from the TV.

    Args:
        tv_config: TV configuration containing IP, port, and settings.
        content_id: Content ID of the image to delete.

    Raises:
        TvConnectionError: If TV cannot be reached.

    Implementation Notes:
        - Uses samsungtvws art().delete(content_id)
    """
    raise NotImplementedError("Contract stub - not implemented")


def clean_old_uploads(tv_config: "TvConfig", keep: int) -> int:
    """
    Delete oldest uploads, keeping only the most recent `keep` images.

    KEEP-POLICY CONTRACT (ENFORCED):
        - Ordering uses TV-reported metadata (image date from art().available()).
        - If recency metadata is unavailable, falls back to deterministic ordering:
          lexical ascending by content_id (e.g., MY_F0001 before MY_F0002).
        - Retained set: the `keep` most recent images (or lexical-first if no dates).
        - Deleted set: all images NOT in the retained set.

    DELETE-FAILURE CONTRACT (ENFORCED):
        - NON-THROWING for individual deletion failures.
        - If a delete call fails, LOG WARNING and continue cleanup.
        - Only count confirmed successful deletions in return value.
        - NEVER delete from the retained set due to delete failures elsewhere.

    Args:
        tv_config: TV configuration containing IP, port, and settings.
        keep: Number of most recent images to retain.

    Returns:
        Number of images successfully deleted (confirmed deletions only).

    Raises:
        TvConnectionError: If list_uploaded() cannot reach TV initially.
            Individual delete failures do NOT raise; they are logged and skipped.

    Implementation Notes:
        - Uses list_uploaded() to get current images
        - Sorts by date if available, else by content_id
        - Calls delete_uploaded() for each image in delete set
        - Logs warnings for individual deletion failures (not exceptions)
    """
    raise NotImplementedError("Contract stub - not implemented")


def is_available_for_push(tv_config: "TvConfig") -> bool:
    """
    Check if the TV is reachable AND in Art Mode.

    PURE WRAPPER CONTRACT (ENFORCED):
        - Equivalent to: check_status(tv_config).reachable and check_status(tv_config).art_mode_active
        - NO new network semantics beyond check_status().
        - NON-THROWING: Returns False on any error (reachable=False or exception).

    Args:
        tv_config: TV configuration containing IP, port, and settings.

    Returns:
        True if TV is reachable AND Art Mode is active.
        False otherwise (unreachable, Art Mode off, Art Mode unsupported, any error).
    """
    # Pure convenience wrapper - no network calls beyond check_status
    status = check_status(tv_config)
    return status.reachable and status.art_mode_active
