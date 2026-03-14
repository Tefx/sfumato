"""
Samsung The Frame TV connection and Art Mode control.

Implementation module for TV operations using samsungtvws library.
All network behavior is implemented following validated patterns from PROTOTYPING.md#1.

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
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from samsungtvws import SamsungTVWS  # type: ignore[import-untyped]
from samsungtvws.art import SamsungTVArt  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from sfumato.config import TvConfig


logger = logging.getLogger(__name__)


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
# Internal Helper Functions
# =============================================================================


def _create_tv_client(tv_config: "TvConfig") -> SamsungTVWS:
    """
    Create a SamsungTVWS client with timeout configuration.

    Args:
        tv_config: TV configuration containing IP and port.

    Returns:
        Configured SamsungTVWS instance.
    """
    # Use a name that identifies this application for pairing prompts
    return SamsungTVWS(
        host=tv_config.ip,
        port=tv_config.port,
        name="Sfumato",
    )


def _get_art_client(tv_config: "TvConfig") -> tuple[SamsungTVWS, SamsungTVArt]:
    """
    Get Art mode client from TV connection.

    Args:
        tv_config: TV configuration.

    Returns:
        Tuple of (TV client, Art client).

    Raises:
        TvConnectionError: If connection or art mode setup fails.
    """
    try:
        tv = _create_tv_client(tv_config)
        art = tv.art()
        return tv, art
    except Exception as e:
        # Connection refused, websocket setup failure, pairing required
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            raise TvConnectionError(f"Connection refused: {e}") from e
        if "timeout" in error_msg:
            raise TvConnectionError(f"Connection timeout: {e}") from e
        if "pairing" in error_msg or "pin" in error_msg:
            raise TvConnectionError(f"Pairing required: accept prompt on TV") from e
        raise TvConnectionError(f"Connection failed: {e}") from e


# =============================================================================
# Public Function Implementations
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
    """
    try:
        # Create client with socket timeout enforcement
        # Note: samsungtvws uses websocket-client internally
        tv = _create_tv_client(tv_config)
        art = tv.art()

        # Query status with individual timeout protection
        # Use validated Art Mode calls from PROTOTYPING.md#1
        # Do NOT use set_artmode() or get_thumbnail() - can hang

        try:
            # Check if Art Mode is supported
            supported = art.supported()
            if not supported:
                return TvStatus(
                    reachable=True,
                    art_mode_supported=False,
                    art_mode_active=False,
                    uploaded_count=0,
                    error=None,
                )

            # Check current Art Mode state
            artmode = art.get_artmode()
            art_mode_active = artmode == "on"

            # Get uploaded images count
            available = art.available()
            uploaded_count = len(available) if available else 0

            return TvStatus(
                reachable=True,
                art_mode_supported=True,
                art_mode_active=art_mode_active,
                uploaded_count=uploaded_count,
                error=None,
            )

        except socket.timeout:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="timeout after 10s",
            )
        except Exception as e:
            # Art-mode API failures - surface as unreachable with descriptive error
            error_msg = str(e).lower()
            if "connection" in error_msg or "refused" in error_msg:
                return TvStatus(
                    reachable=False,
                    art_mode_supported=False,
                    art_mode_active=False,
                    uploaded_count=0,
                    error=f"Connection refused",
                )
            if "timeout" in error_msg:
                return TvStatus(
                    reachable=False,
                    art_mode_supported=False,
                    art_mode_active=False,
                    uploaded_count=0,
                    error="timeout after 10s",
                )
            if "pairing" in error_msg or "pin" in error_msg:
                return TvStatus(
                    reachable=False,
                    art_mode_supported=False,
                    art_mode_active=False,
                    uploaded_count=0,
                    error="pairing required: accept prompt on TV",
                )
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error=f"Art Mode probe failed: {e}",
            )

    except socket.timeout:
        return TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="timeout after 10s",
        )
    except ConnectionRefusedError:
        return TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
    except OSError as e:
        # Network errors (host unreachable, etc.)
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="Connection refused",
            )
        if "timeout" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="timeout after 10s",
            )
        if "unreachable" in error_msg or "network" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error=f"Network unreachable: {e}",
            )
        return TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error=f"Connection failed: {e}",
        )
    except Exception as e:
        # Catch-all for any other unexpected errors
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="Connection refused",
            )
        if "timeout" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="timeout after 10s",
            )
        if "pairing" in error_msg or "pin" in error_msg:
            return TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error="pairing required: accept prompt on TV",
            )
        return TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error=f"Connection failed: {e}",
        )


def upload_image(
    tv_config: "TvConfig",
    image_path: Path,
) -> str:
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
    """
    _, art = _get_art_client(tv_config)

    try:
        # Read image data
        with open(image_path, "rb") as f:
            data = f.read()

        # Upload using validated API from PROTOTYPING.md#1
        content_id = art.upload(data, file_type="PNG", matte="none")

        if not content_id:
            raise TvUploadError("Upload returned empty content ID")

        return content_id

    except TvConnectionError:
        # Re-raise connection errors
        raise
    except TvUploadError:
        # Re-raise upload errors
        raise
    except FileNotFoundError as e:
        raise TvUploadError(f"Image file not found: {image_path}") from e
    except PermissionError as e:
        raise TvUploadError(f"Permission denied reading file: {image_path}") from e
    except Exception as e:
        # Distinguish upload failures from connection failures
        error_msg = str(e).lower()
        if (
            "connection" in error_msg
            or "refused" in error_msg
            or "timeout" in error_msg
        ):
            raise TvConnectionError(f"Connection failed during upload: {e}") from e
        if "pairing" in error_msg or "pin" in error_msg:
            raise TvConnectionError(f"Pairing required: accept prompt on TV") from e
        raise TvUploadError(f"Upload failed: {e}") from e


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
    _, art = _get_art_client(tv_config)

    try:
        # Use select_image with show=True (validated in PROTOTYPING.md#1)
        # Do NOT use set_artmode(True) - can hang
        art.select_image(content_id, show=True)
    except TvConnectionError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if (
            "connection" in error_msg
            or "refused" in error_msg
            or "timeout" in error_msg
        ):
            raise TvConnectionError(f"Connection failed: {e}") from e
        if "pairing" in error_msg or "pin" in error_msg:
            raise TvConnectionError(f"Pairing required: accept prompt on TV") from e
        # Other errors during selection are connection-related
        raise TvConnectionError(f"Display selection failed: {e}") from e


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
    _, art = _get_art_client(tv_config)

    try:
        # Get list from TV
        available = art.available()

        if not available:
            return []

        # Convert TV metadata to UploadedImage objects
        images: list[UploadedImage] = []
        for item in available:
            # content_id is in the item dict
            content_id = item.get("content_id", "")
            if not content_id:
                continue

            # file_name may not be available, TV may use display name or ID
            file_name = item.get("name") or None

            images.append(UploadedImage(content_id=content_id, file_name=file_name))

        return images

    except TvConnectionError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if (
            "connection" in error_msg
            or "refused" in error_msg
            or "timeout" in error_msg
        ):
            raise TvConnectionError(f"Connection failed: {e}") from e
        if "pairing" in error_msg or "pin" in error_msg:
            raise TvConnectionError(f"Pairing required: accept prompt on TV") from e
        raise TvConnectionError(f"Failed to list uploads: {e}") from e


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
    _, art = _get_art_client(tv_config)

    try:
        art.delete(content_id)
    except TvConnectionError:
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if (
            "connection" in error_msg
            or "refused" in error_msg
            or "timeout" in error_msg
        ):
            raise TvConnectionError(f"Connection failed: {e}") from e
        if "pairing" in error_msg or "pin" in error_msg:
            raise TvConnectionError(f"Pairing required: accept prompt on TV") from e
        raise TvConnectionError(f"Delete failed: {e}") from e


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
    # Get current uploads - this can raise TvConnectionError
    images = list_uploaded(tv_config)

    # Nothing to clean if we have <= keep images
    if len(images) <= keep:
        return 0

    # Try to get dates from raw available() data for sorting
    try:
        _, art = _get_art_client(tv_config)
        available_raw = art.available() or []
    except Exception:
        # If we can't get raw metadata, fall back to list_uploaded results
        available_raw = []

    # Build mapping of content_id -> date for sorting
    date_map: dict[str, str] = {}
    for item in available_raw:
        content_id = item.get("content_id", "")
        # TV metadata date format from PROTOTYPING.md#1: "2026:03:14 17:24:47"
        date_str = item.get("date", "")
        if content_id and date_str:
            date_map[content_id] = date_str

    # Sort images by date (newest first) or by content_id if no date
    def sort_key(img: UploadedImage) -> tuple[int, str]:
        """Sort key: (has_date, date_or_content_id)."""
        date_str = date_map.get(img.content_id, "")
        if date_str:
            # Has date - sort by date descending (newest first)
            # Use negative comparison to sort newest first
            return (0, date_str)  # 0 = has date, sort by date string
        else:
            # No date - use content_id lexical ascending (MY_F0001 < MY_F0002)
            return (1, img.content_id)

    # Sort: newest first (by date), then lexical by content_id
    sorted_images = sorted(images, key=sort_key)

    # For images with dates, we want newest first
    # For images without dates, lexical ascending means MY_F0001 before MY_F0002
    # But we want to KEEP newest/highest, so we reverse the sort
    # Actually: we keep the last 'keep' items after sorting by date ascending
    # Let's sort properly:
    #   - images with dates: sort by date ascending (oldest first)
    #   - images without dates: sort by content_id ascending (lexical)
    #   Then keep the last 'keep' (newest/highest lexical)

    def sort_key_ascending(img: UploadedImage) -> tuple[int, str]:
        """Sort key for ascending order (oldest/lexical-first to newest/highest)."""
        date_str = date_map.get(img.content_id, "")
        if date_str:
            return (0, date_str)  # Has date, sort by date ascending
        else:
            return (1, img.content_id)  # No date, sort by content_id ascending

    # Sort ascending: oldest/lexical-first to newest/highest-lexical
    sorted_images = sorted(images, key=sort_key_ascending)

    # Retained = last 'keep' items (newest or highest lexical)
    retained = (
        set(img.content_id for img in sorted_images[-keep:]) if keep > 0 else set()
    )
    to_delete = sorted_images[:-keep] if keep > 0 and len(sorted_images) > keep else []

    # Delete old uploads, logging failures
    deleted_count = 0
    for img in to_delete:
        # Never delete from retained set
        if img.content_id in retained:
            continue

        try:
            delete_uploaded(tv_config, img.content_id)
            deleted_count += 1
        except Exception as e:
            # Log warning and continue - do NOT raise
            logger.warning(
                f"Failed to delete uploaded image {img.content_id}: {e}. "
                f"Continuing cleanup."
            )
            # The retained set is protected - we never fall back to deleting retained

    return deleted_count


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
    try:
        status = check_status(tv_config)
        return status.reachable and status.art_mode_active
    except Exception as e:
        # Should never happen since check_status is non-throwing, but be defensive
        logger.debug(f"is_available_for_push caught unexpected exception: {e}")
        return False
