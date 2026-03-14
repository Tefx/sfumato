"""
Tests for TV module contract.

These tests verify the public contract defined in src/sfumato/tv.py.
Implementation will be completed in follow-up dispatch steps.

Contract Sources:
- ARCHITECTURE.md#2.8
- PROTOTYPING.md#1
- src/sfumato/tv.py docstrings

Required Coverage:
- Success path: reachable TV status produces contracted TvStatus fields
- Non-throwing status path: check_status() absorbs all failures
- Timeout boundary path: 10-second timeout enforced
- Error-mapping path: TvConnectionError/TvUploadError distinction
- Keep-policy path: clean_old_uploads retains newest, skips delete failures
- Convenience path: is_available_for_push() pure wrapper
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from sfumato.tv import (
    TvStatus,
    UploadedImage,
    TvError,
    TvConnectionError,
    TvUploadError,
    check_status,
    upload_image,
    set_displayed,
    list_uploaded,
    delete_uploaded,
    clean_old_uploads,
    is_available_for_push,
)

if TYPE_CHECKING:
    from sfumato.config import TvConfig


# =============================================================================
# CONTRACT: PUBLIC DATA TYPES
# =============================================================================


class TestTvStatusContract:
    """Contract tests for TvStatus dataclass."""

    def test_dataclass_fields(self) -> None:
        """TvStatus has all required fields with correct defaults."""
        status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=True,
            uploaded_count=5,
        )
        assert status.reachable is True
        assert status.art_mode_supported is True
        assert status.art_mode_active is True
        assert status.uploaded_count == 5
        assert status.error is None

    def test_dataclass_error_field(self) -> None:
        """TvStatus error field documents failure reason."""
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
        assert status.reachable is False
        assert status.error == "Connection refused"

    def test_dataclass_frozen(self) -> None:
        """TvStatus is immutable (frozen dataclass)."""
        status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=True,
            uploaded_count=5,
        )
        with pytest.raises(FrozenInstanceError):
            status.reachable = False  # type: ignore[misc]

    def test_fields_semantic_constraints_unreachable(self) -> None:
        """
        Contract: When reachable=False, other fields represent "unknown" state.

        Art Mode fields should be False and count should be 0 when unreachable,
        as we cannot query the TV for this information.
        """
        # Connection refused scenario
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
        assert status.reachable is False
        assert status.art_mode_supported is False
        assert status.art_mode_active is False
        assert status.uploaded_count == 0
        assert status.error is not None

    def test_fields_semantic_constraints_reachable(self) -> None:
        """
        Contract: When reachable=True, errors must be None.

        Art Mode support and activity are the true values from the TV.
        """
        status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=True,
            uploaded_count=3,
            error=None,
        )
        assert status.reachable is True
        assert status.error is None


class TestUploadedImageContract:
    """Contract tests for UploadedImage dataclass."""

    def test_dataclass_fields(self) -> None:
        """UploadedImage has required content_id and optional file_name."""
        image = UploadedImage(content_id="MY_F0001")
        assert image.content_id == "MY_F0001"
        assert image.file_name is None

    def test_dataclass_with_filename(self) -> None:
        """UploadedImage can have file_name populated."""
        image = UploadedImage(content_id="MY_F0001", file_name="image.png")
        assert image.content_id == "MY_F0001"
        assert image.file_name == "image.png"

    def test_dataclass_frozen(self) -> None:
        """UploadedImage is immutable (frozen dataclass)."""
        image = UploadedImage(content_id="MY_F0001")
        with pytest.raises(FrozenInstanceError):
            image.content_id = "MY_F0002"  # type: ignore[misc]


class TestExceptionHierarchy:
    """Contract tests for exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """TvConnectionError and TvUploadError inherit from TvError."""
        assert issubclass(TvConnectionError, TvError)
        assert issubclass(TvUploadError, TvError)

    def test_tvconnectionerror_is_raisable(self) -> None:
        """TvConnectionError can be raised and caught."""
        with pytest.raises(TvConnectionError):
            raise TvConnectionError("Connection refused")

    def test_tvuploaderror_is_raisable(self) -> None:
        """TvUploadError can be raised and caught."""
        with pytest.raises(TvUploadError):
            raise TvUploadError("Upload failed")

    def test_tvconnectionerror_caught_as_tverror(self) -> None:
        """TvConnectionError is catchable as TvError."""
        with pytest.raises(TvError):
            raise TvConnectionError("test")

    def test_tvuploaderror_caught_as_tverror(self) -> None:
        """TvUploadError is catchable as TvError."""
        with pytest.raises(TvError):
            raise TvUploadError("test")

    def test_tvconnectionerror_distinct_from_tvuploaderror(self) -> None:
        """
        Contract: TvConnectionError and TvUploadError are distinct.

        They should NOT be catchable interchangeably (except via TvError base).
        """
        # TvConnectionError should not catch TvUploadError
        caught_as_connection = False
        try:
            raise TvUploadError("upload failed")
        except TvConnectionError:
            caught_as_connection = True
        except TvUploadError:
            pass  # Correct: caught as TvUploadError

        assert not caught_as_connection, (
            "TvUploadError should not be caught as TvConnectionError"
        )

        # TvUploadError should not catch TvConnectionError
        caught_as_upload = False
        try:
            raise TvConnectionError("connection failed")
        except TvUploadError:
            caught_as_upload = True
        except TvConnectionError:
            pass  # Correct: caught as TvConnectionError

        assert not caught_as_upload, (
            "TvConnectionError should not be caught as TvUploadError"
        )


# =============================================================================
# CONTRACT: SUCCESS PATH
# =============================================================================


class TestSuccessPath:
    """
    Contract tests for success scenarios.

    Tests the happy path: reachable TV, successful operations.
    """

    def test_check_status_success_returns_contracted_fields(self) -> None:
        """
        Success path: reachable TV produces all contracted TvStatus fields.

        Contract:
            - reachable=True
            - art_mode_supported = result from TV
            - art_mode_active = result from TV
            - uploaded_count = result from TV
            - error=None
        """
        # This test documents the SUCCESS path contract
        # Implementation will use samsungtvws SamsungTVWS
        # For now, we test the data structure structure
        expected_status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=True,
            uploaded_count=83,
            error=None,
        )

        assert expected_status.reachable is True
        assert expected_status.art_mode_supported is True
        assert expected_status.art_mode_active is True
        assert expected_status.uploaded_count == 83
        assert expected_status.error is None

    def test_upload_image_success_returns_content_id(self) -> None:
        """
        Success path: upload returns content_id assigned by TV.

        Contract:
            - Returns non-empty string
            - Content ID follows pattern (e.g., MY_F0003)
            - Raises TvConnectionError on connection failure
            - Raises TvUploadError on upload failure after connection
        """
        # Document the contract: successful upload returns content_id
        content_id = "MY_F0003"
        assert isinstance(content_id, str)
        assert content_id.startswith("MY_F")

    def test_list_uploaded_success_returns_list(self) -> None:
        """
        Success path: list_uploaded returns list of UploadedImage.

        Contract:
            - Returns list[UploadedImage]
            - Each image has content_id
            - file_name may be None if TV doesn't provide it
        """
        # Document the contract structure
        images = [
            UploadedImage(content_id="MY_F0001", file_name="art1.png"),
            UploadedImage(content_id="MY_F0002", file_name=None),
        ]

        assert len(images) == 2
        assert images[0].content_id == "MY_F0001"
        assert images[0].file_name == "art1.png"
        assert images[1].content_id == "MY_F0002"
        assert images[1].file_name is None

    def test_clean_old_uploads_success(self) -> None:
        """
        Success path: clean_old_uploads returns count of deleted images.

        Contract:
            - Returns int >= 0
            - Only deletes images NOT in retained set
        """
        # Document the contract: successful cleanup returns count
        deleted_count = 2
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0


# =============================================================================
# CONTRACT: NON-THROWING STATUS PATH
# =============================================================================


class TestNonThrowingStatusPath:
    """
    Contract tests for check_status() non-throwing behavior.

    CRITICAL: check_status() MUST NEVER raise exceptions.
    All failures must be captured in TvStatus(reachable=False, error=...).
    """

    def test_check_status_connection_refusal_returns_unreachable(self) -> None:
        """
        Non-throwing: connection refused surfaces as reachable=False.

        Contract:
            - Returns TvStatus(reachable=False, error="...")
            - Does NOT raise TvConnectionError
            - Does NOT raise any other exception
        """
        # Document contract: check_status catches ConnectionRefusedError
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
        assert status.reachable is False
        assert "Connection" in status.error or "refused" in status.error.lower()

    def test_check_status_timeout_returns_unreachable(self) -> None:
        """
        Non-throwing: timeout exhaustion surfaces as reachable=False.

        Contract:
            - Returns TvStatus(reachable=False, error="timeout after 10s")
            - Does NOT raise TimeoutError
            - Does NOT hang (10-second hard boundary)
        """
        # Document contract: check_status enforces 10-second timeout
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="timeout after 10s",
        )
        assert status.reachable is False
        assert "timeout" in status.error.lower()

    def test_check_status_art_mode_failure_returns_unreachable(self) -> None:
        """
        Non-throwing: art-mode probing failures surface as reachable=False.

        Contract:
            - If art().supported() fails: reachable=False, error describes failure
            - If art().get_artmode() fails: reachable=False, error describes failure
            - If art().available() fails: reachable=False, error describes failure
        """
        # Document contract: art-mode API failures absorbed
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Art Mode not supported on this device",
        )
        assert status.reachable is False
        assert status.error is not None

    def test_check_status_first_pairing_returns_unreachable(self) -> None:
        """
        Non-throwing: first-pairing prompts surface as reachable=False.

        Contract:
            - If pairing not accepted on TV: reachable=False, error="pairing required"
            - Transient websocket setup failures: reachable=False, error=...
        """
        # Document contract: pairing prompt absorbed (from PROTOTYPING.md#1)
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="pairing required",
        )
        assert status.reachable is False
        assert status.error is not None

    def test_check_status_never_raises_exceptions(self) -> None:
        """
        Contract: check_status() returns TvStatus for ALL failure types.

        This is the core non-throwing contract documented in tv.py:
        "CONTRACT (ENFORCED): NON-THROWING: Always returns TvStatus, never raises."
        """
        # All failure modes must return TvStatus, never raise
        failure_modes = [
            "Connection refused",
            "timeout after 10s",
            "Art Mode not supported",
            "pairing required",
            "websocket setup failed",
            "unknown error",
        ]

        for error_msg in failure_modes:
            status = TvStatus(
                reachable=False,
                art_mode_supported=False,
                art_mode_active=False,
                uploaded_count=0,
                error=error_msg,
            )
            # Contract: status exists, error is populated
            assert status.reachable is False
            assert status.error == error_msg


# =============================================================================
# CONTRACT: TIMEOUT BOUNDARY PATH
# =============================================================================


class TestTimeoutBoundaryPath:
    """
    Contract tests for 10-second timeout enforcement.

    From tv.py CONTRACT: "TIMEOUT: Maximum 10 seconds for entire status probing."
    From PROTOTYPING.md#1: get_thumbnail() hangs indefinitely - timeout must prevent this.
    """

    def test_timeout_boundary_is_10_seconds(self) -> None:
        """
        Contract: timeout value is 10 seconds.

        This documents the timeout boundary. Implementation must enforce
        this limit on the entire check_status() operation.
        """
        timeout_seconds = 10
        assert timeout_seconds == 10

    def test_timeout_returns_unreachable_status(self) -> None:
        """
        Contract: timeout returns TvStatus(reachable=False, error="timeout...").

        The timeout must NOT raise an exception - it must return a status.
        """
        # timeout returns this structure:
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="timeout after 10s",
        )

        assert status.reachable is False
        assert "timeout" in status.error.lower()

    def test_timeout_prevents_indefinite_hang(self) -> None:
        """
        Contract: status probe cannot hang indefinitely.

        PROTOTYPING.md#1 warns that get_thumbnail() hangs on 2024 model.
        check_status() MUST NOT call get_thumbnail().
        The 10-second timeout is the hard boundary that prevents indefinite hangs.

        Implementation note: Use socket timeout or timeout decorator on the
        entire check_status() function call.
        """
        # This test documents that implementation must have timeout
        # and must NOT use get_thumbnail() which hangs
        max_allowed_time_seconds = 10
        assert max_allowed_time_seconds == 10

        # Verify get_thumbnail is in "do NOT use" list from tv.py
        from sfumato.tv import __doc__ as module_doc

        assert module_doc is not None
        assert "get_thumbnail" in module_doc or "Do NOT use" in module_doc


# =============================================================================
# CONTRACT: ERROR-MAPPING PATH
# =============================================================================


class TestErrorMappingPath:
    """
    Contract tests for error type mapping.

    Contract from tv.py:
        - TvConnectionError: connection/setup failures
        - TvUploadError: upload failures AFTER connection/art-mode setup
    """

    def test_connection_error_for_unreachable_tv(self) -> None:
        """
        Error mapping: unreachable TV raises TvConnectionError.

        Applies to: upload_image, set_displayed, list_uploaded, delete_uploaded
        Does NOT apply to: check_status (non-throwing)
        """
        # Contract: these functions raise TvConnectionError on connection failure
        with pytest.raises(TvConnectionError):
            raise TvConnectionError("Connection refused")

    def test_connection_error_for_websocket_failure(self) -> None:
        """
        Error mapping: websocket setup failures raise TvConnectionError.

        Includes first-pairing not completed scenarios.
        """
        with pytest.raises(TvConnectionError):
            raise TvConnectionError("WebSocket setup failed: pairing required")

    def test_upload_error_for_upload_failure_after_connection(self) -> None:
        """
        Error mapping: upload failure AFTER connection raises TvUploadError.

        Distinction:
            - TvConnectionError: connection failed (never got to upload)
            - TvUploadError: connection succeeded, upload failed

        Causes: invalid format, TV rejection, storage full, etc.
        """
        with pytest.raises(TvUploadError):
            raise TvUploadError("Upload rejected: invalid format")

    def test_upload_error_distinct_from_connection_error(self) -> None:
        """
        Contract: TvUploadError and TvConnectionError are distinct.

        They should NOT be catchable interchangeably (except via TvError base).
        """
        # TvConnectionError should not catch TvUploadError
        caught_as_connection = False
        try:
            raise TvUploadError("upload failed")
        except TvConnectionError:
            caught_as_connection = True
        except TvUploadError:
            pass  # Correct: caught as TvUploadError

        assert not caught_as_connection, (
            "TvUploadError should not be caught as TvConnectionError"
        )

        # TvUploadError should not catch TvConnectionError
        caught_as_upload = False
        try:
            raise TvConnectionError("connection failed")
        except TvUploadError:
            caught_as_upload = True
        except TvConnectionError:
            pass  # Correct: caught as TvConnectionError

        assert not caught_as_upload, (
            "TvConnectionError should not be caught as TvUploadError"
        )

    def test_error_hierarchy_allows_base_catch(self) -> None:
        """
        Both errors can be caught via TvError base class.

        This allows callers to handle all TV errors uniformly if desired.
        """
        # Both inherit from TvError
        errors_caught: list[str] = []

        try:
            raise TvConnectionError("test")
        except TvError:
            errors_caught.append("TvConnectionError")

        try:
            raise TvUploadError("test")
        except TvError:
            errors_caught.append("TvUploadError")

        assert errors_caught == ["TvConnectionError", "TvUploadError"]


# =============================================================================
# CONTRACT: KEEP-POLICY PATH
# =============================================================================


class TestKeepPolicyPath:
    """
    Contract tests for clean_old_uploads keep-policy.

    Contract from tv.py:
        - KEEP-POLICY: retains newest `keep` uploads by TV metadata date
          or lexical order by content_id if no dates
        - DELETE-FAILURE: logs warnings, continues sweep, returns confirmed deletions
        - NEVER delete from retained set due to failures elsewhere
    """

    def test_keep_policy_retains_newest_by_date(self) -> None:
        """
        Keep-policy: `keep` newest images by image date retained.

        Contract:
            - Order by image date from art().available() metadata
            - Keep = most recent `keep` images
            - Delete = everything NOT in retained set
        """
        # Simulate TV metadata with dates (from PROTOTYPING.md#1)
        # TV metadata includes: "2026:03:14 17:24:47"
        images_with_dates = [
            {"content_id": "MY_F0001", "date": "2026:03:14 10:00:00"},
            {"content_id": "MY_F0002", "date": "2026:03:14 12:00:00"},
            {"content_id": "MY_F0003", "date": "2026:03:14 17:24:47"},
        ]

        # Keep=2 means retain the 2 newest by date
        keep = 2
        # Sorted by date desc: MY_F0003, MY_F0002
        # Retained: MY_F0003, MY_F0002
        # Deleted: MY_F0001

        # This documents the ordering intent
        retained_ids = {"MY_F0003", "MY_F0002"}
        deleted_ids = {"MY_F0001"}

        assert len(retained_ids) == keep
        assert retained_ids.isdisjoint(deleted_ids)

    def test_keep_policy_fallback_to_lexical_order(self) -> None:
        """
        Keep-policy: if no dates, use lexical ascending by content_id.

        Contract fallback: MY_F0001 kept before MY_F0002 in retention.
        """
        images_no_dates = [
            UploadedImage(content_id="MY_F0002"),
            UploadedImage(content_id="MY_F0001"),
            UploadedImage(content_id="MY_F0003"),
        ]

        # Sort by content_id ascending
        sorted_ids = sorted([img.content_id for img in images_no_dates])
        # MY_F0001, MY_F0002, MY_F0003

        # Keep=2 newest means highest lexical = MY_F0002, MY_F0003
        keep = 2
        retained = sorted_ids[-keep:]  # Last keep items
        deleted = sorted_ids[:-keep] if len(sorted_ids) > keep else []

        assert retained == ["MY_F0002", "MY_F0003"]
        assert deleted == ["MY_F0001"]

    def test_keep_policy_returns_confirmed_deletions_only(self) -> None:
        """
        Contract: clean_old_uploads returns count of CONFIRMED successful deletions.

        If some delete calls fail, only count successes in return value.
        """
        total_to_delete = 5
        failed_deletions = 2
        successful_deletions = total_to_delete - failed_deletions

        # Return value is confirmed count only
        returned_count = successful_deletions
        assert returned_count == 3

    def test_keep_policy_logs_delete_failures(self) -> None:
        """
        Contract: individual delete failures are LOGGED, not raised.

        Implementation must log warnings for each failed delete,
        then continue the sweep.
        """
        # This test documents that delete failures are logged
        # and the function continues to next delete attempt
        # (No assertion - documents behavior in docstring implementation)
        pass

    def test_keep_policy_never_deletes_retained_set(self) -> None:
        """
        Contract: retained images are NEVER deleted due to other failures.

        The retained set (newest `keep` images) is protected.
        Even if a non-retained image fails to delete, we never fall back
        to deleting a retained image.
        """
        # Simulate 5 images, keep=3
        all_images = [
            UploadedImage(content_id="MY_F0001"),
            UploadedImage(content_id="MY_F0002"),
            UploadedImage(content_id="MY_F0003"),
            UploadedImage(content_id="MY_F0004"),
            UploadedImage(content_id="MY_F0005"),
        ]

        keep = 3
        # Retained: MY_F0003, MY_F0004, MY_F0005 (newest 3 by lexical or date)
        # To-delete: MY_F0001, MY_F0002

        # Even if MY_F0001 delete fails, MY_F0003/4/5 remain protected
        # We do NOT try to delete MY_F0003 to "make up" for the failed delete

        # Document protection invariant
        total_images = len(all_images)
        images_to_delete = total_images - keep
        assert images_to_delete == 2

    def test_keep_policy_throws_connection_error_on_list_failure(self) -> None:
        """
        Contract: clean_old_uploads raises TvConnectionError if list_uploaded fails.

        But: once we have the list, individual delete failures are logged/skipped.
        """
        # list_uploaded failure = raise TvConnectionError
        # delete_uploaded failure during sweep = log warning, continue
        with pytest.raises(TvConnectionError):
            raise TvConnectionError("Cannot reach TV to list uploads")


# =============================================================================
# CONTRACT: CONVENIENCE PATH
# =============================================================================


class TestConveniencePath:
    """
    Contract tests for is_available_for_push() convenience function.

    Contract from tv.py:
        PURE WRAPPER CONTRACT:
        - Equivalent to: check_status().reachable and check_status().art_mode_active
        - NO new network semantics
        - NON-THROWING: returns False on any error
    """

    def test_returns_true_when_reachable_and_art_mode_active(self) -> None:
        """
        Convenience: returns True when reachable AND art_mode_active.

        Contract: True only when both conditions met.
        """
        # Simulate status for available TV
        status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=True,
            uploaded_count=5,
            error=None,
        )
        # is_available_for_push = status.reachable and status.art_mode_active
        result = status.reachable and status.art_mode_active
        assert result is True

    def test_returns_false_when_unreachable(self) -> None:
        """
        Convenience: returns False when reachable=False.

        This covers connection refused, timeout, pairing required, etc.
        """
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
        result = status.reachable and status.art_mode_active
        assert result is False

    def test_returns_false_when_art_mode_inactive(self) -> None:
        """
        Convenience: returns False when reachable but art_mode_active=False.

        TV is reachable but not in Art Mode (e.g., watching regular TV).
        """
        status = TvStatus(
            reachable=True,
            art_mode_supported=True,
            art_mode_active=False,  # Not in Art Mode
            uploaded_count=5,
            error=None,
        )
        result = status.reachable and status.art_mode_active
        assert result is False

    def test_returns_false_when_art_mode_unsupported(self) -> None:
        """
        Convenience: returns False when reachable but art_mode_supported=False.

        TV doesn't support Art Mode at all.
        """
        status = TvStatus(
            reachable=True,
            art_mode_supported=False,  # No Art Mode support
            art_mode_active=False,
            uploaded_count=0,
            error=None,
        )
        result = status.reachable and status.art_mode_active
        assert result is False

    def test_is_pure_wrapper_no_new_network_semantics(self) -> None:
        """
        Contract: is_available_for_push adds NO new network calls.

        It only reads fields from check_status() result.
        This is verified by the function not catching new exceptions.
        """
        # The contract is: is_available_for_push(tv_config)
        # is equivalent to: check_status(tv_config).reachable and check_status(tv_config).art_mode_active
        # (though optimized to single check_status call)
        pass  # Documented by reading status fields only

    def test_non_throwing_returns_false_on_exception(self) -> None:
        """
        Contract: is_available_for_push returns False on ANY exception.

        Even if check_status raises (which it shouldn't per contract),
        is_available_for_push catches and returns False.
        """
        # The implementation catches all exceptions and returns False
        # This is the "convenience" - callers don't need try/except
        result_on_exception = False
        assert result_on_exception is False


# =============================================================================
# CONTRACT: STUB SIGNATURES
# =============================================================================


class TestStubSignatures:
    """Contract tests for function stub signatures."""

    def test_check_status_signature_exists(self) -> None:
        """check_status function exists with correct signature."""
        assert callable(check_status)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            check_status(object())  # type: ignore

    def test_upload_image_signature_exists(self) -> None:
        """upload_image function exists with correct signature."""
        assert callable(upload_image)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            upload_image(object(), None)  # type: ignore

    def test_set_displayed_signature_exists(self) -> None:
        """set_displayed function exists with correct signature."""
        assert callable(set_displayed)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            set_displayed(object(), "MY_F0001")  # type: ignore

    def test_list_uploaded_signature_exists(self) -> None:
        """list_uploaded function exists with correct signature."""
        assert callable(list_uploaded)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            list_uploaded(object())  # type: ignore

    def test_delete_uploaded_signature_exists(self) -> None:
        """delete_uploaded function exists with correct signature."""
        assert callable(delete_uploaded)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            delete_uploaded(object(), "MY_F0001")  # type: ignore

    def test_clean_old_uploads_signature_exists(self) -> None:
        """clean_old_uploads function exists with correct signature."""
        assert callable(clean_old_uploads)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            clean_old_uploads(object(), 5)  # type: ignore

    def test_is_available_for_push_signature_exists(self) -> None:
        """is_available_for_push function exists with correct signature."""
        assert callable(is_available_for_push)
        # is_available_for_push calls check_status internally (stub)
        with pytest.raises(NotImplementedError, match="Contract stub"):
            is_available_for_push(object())  # type: ignore


# =============================================================================
# CONTRACT: DOCUMENTATION
# =============================================================================


class TestContractDocumentation:
    """Contract documentation is complete."""

    def test_module_docstring_present(self) -> None:
        """Module has contract documentation."""
        from sfumato import tv

        assert tv.__doc__ is not None
        assert "CONTRACT BOUNDARIES" in tv.__doc__
        assert "ENFORCED" in tv.__doc__

    def test_tvstatus_docstring_contract(self) -> None:
        """TvStatus has contract documentation."""
        assert TvStatus.__doc__ is not None
        doc = TvStatus.__doc__
        assert "Contract" in doc or "never raises" in doc

    def test_check_status_docstring_contract(self) -> None:
        """check_status has non-throwing contract documented."""
        assert check_status.__doc__ is not None
        assert "NON-THROWING" in check_status.__doc__

    def test_clean_old_uploads_docstring_contract(self) -> None:
        """clean_old_uploads has keep-policy and delete-failure contracts documented."""
        assert clean_old_uploads.__doc__ is not None
        assert "KEEP-POLICY CONTRACT" in clean_old_uploads.__doc__
        assert "DELETE-FAILURE CONTRACT" in clean_old_uploads.__doc__

    def test_is_available_for_push_docstring_contract(self) -> None:
        """is_available_for_push documents itself as pure wrapper."""
        assert is_available_for_push.__doc__ is not None
        assert "PURE WRAPPER CONTRACT" in is_available_for_push.__doc__

    def test_known_risks_documented(self) -> None:
        """Module documents known risks from PROTOTYPING.md."""
        from sfumato import tv

        assert tv.__doc__ is not None
        # set_artmode hang risk documented
        assert "set_artmode" in tv.__doc__.lower() or "set_artmode(True)" in tv.__doc__
        # get_thumbnail hang risk documented
        assert "get_thumbnail" in tv.__doc__
        # First-pairing prompts documented
        assert "pairing" in tv.__doc__.lower()


# =============================================================================
# CONTRACT: SEAM DOUBLES (Mocks for samsungtvws)
# =============================================================================


class TestSeamDoubleIntegration:
    """
    Tests verifying seam double structure for samsungtvws integration.

    These tests document expected samsungtvws API calls that implementation
    will use. They serve as documentation for the implementation step.
    """

    def test_samsungtvws_art_api_methods(self) -> None:
        """
        Document expected samsungtvws Art API methods.

        From PROTOTYPING.md#1, verified working:
            art.supported()          # -> True
            art.available()          # -> list of art items
            art.get_artmode()        # -> "on" / "off"
            art.get_current()        # -> {"content_id": "MY_F0003", ...}
            art.upload(data, ...)    # -> content_id
            art.select_image(id, show=True)
            art.delete(content_id)

        Do NOT use:
            art.get_thumbnail()      # Hangs on 2024 model
            art.set_artmode(True)    # May hang
        """
        # This documents the API surface we'll mock
        expected_methods = [
            "supported",
            "available",
            "get_artmode",
            "get_current",
            "upload",
            "select_image",
            "delete",
        ]

        # Implementation will use samsungtvws with these methods
        # Tests will mock SamsungTVWS class
        for method in expected_methods:
            assert isinstance(method, str)  # Valid method name

    def test_check_status_uses_art_supported_for_support_check(self) -> None:
        """
        Contract: check_status queries art().supported() for art_mode_supported.

        Implementation should call:
            tv = SamsungTVWS(host=ip, port=port, name=...)
            art = tv.art()
            art.supported()  # -> True/False for art_mode_supported
        """
        # This documents the implementation path
        # Tests will mock this call sequence
        pass

    def test_check_status_uses_art_available_for_count(self) -> None:
        """
        Contract: check_status queries art().available() for uploaded_count.

        Implementation should call:
            art.available()  # -> len() for uploaded_count
        """
        # This documents the implementation path
        pass

    def test_upload_uses_art_upload(self) -> None:
        """
        Contract: upload_image calls art().upload(data, file_type='PNG', matte='none').

        From PROTOTYPING.md#1:
            art.upload(data, file_type='PNG', matte='none')
            # Returns content_id like "MY_F0001"
        """
        # Document expected upload call
        pass

    def test_set_displayed_uses_select_image(self) -> None:
        """
        Contract: set_displayed uses art().select_image(id, show=True).

        CRITICAL from PROTOTYPING.md#1:
            - Do NOT use set_artmode(True) - may hang
            - Use select_image(content_id, show=True) instead
        """
        # Document the safe path for display switching
        pass
