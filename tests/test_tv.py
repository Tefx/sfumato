"""
Tests for TV module contract.

These tests verify the public contract defined in src/sfumato/tv.py.
Implementation will be completed in follow-up dispatch steps.
"""

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


class TestTvStatusContract:
    """Contract tests for TvStatus dataclass."""

    def test_dataclass_fields(self) -> None:
        """TvStatus has all required fields."""
        # Contract: TvStatus must have these fields
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
        """TvStatus error field is optional."""
        status = TvStatus(
            reachable=False,
            art_mode_supported=False,
            art_mode_active=False,
            uploaded_count=0,
            error="Connection refused",
        )
        assert status.reachable is False
        assert status.error == "Connection refused"


class TestUploadedImageContract:
    """Contract tests for UploadedImage dataclass."""

    def test_dataclass_fields(self) -> None:
        """UploadedImage has required content_id and optional file_name."""
        image = UploadedImage(content_id="MY_F0001")
        assert image.content_id == "MY_F0001"
        assert image.file_name is None

    def test_dataclass_with_filename(self) -> None:
        """UploadedImage can have file_name."""
        image = UploadedImage(content_id="MY_F0001", file_name="image.png")
        assert image.content_id == "MY_F0001"
        assert image.file_name == "image.png"


class TestExceptionHierarchy:
    """Contract tests for exception hierarchy."""

    def test_exception_hierarchy(self) -> None:
        """TvConnectionError and TvUploadError inherit from TvError."""
        assert issubclass(TvConnectionError, TvError)
        assert issubclass(TvUploadError, TvError)

    def test_exceptions_are_raisable(self) -> None:
        """Exceptions can be raised and caught."""
        with pytest.raises(TvError):
            raise TvConnectionError("test")

        with pytest.raises(TvError):
            raise TvUploadError("test")


class TestStubSignatures:
    """Contract tests for function stub signatures."""

    def test_check_status_signature_exists(self) -> None:
        """check_status function exists with correct signature."""
        # Function exists
        assert callable(check_status)
        # Stub raises NotImplementedError
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
        assert "CONTRACT" in TvStatus.__doc__ or "never raises" in TvStatus.__doc__

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
