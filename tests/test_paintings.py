"""Behavior tests for painting fetch/cache semantics.

Spec links:
- ARCHITECTURE.md#2.3
- PROTOTYPING.md#semantic-mood-matching-validated
"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

import sfumato.paintings as paintings
from sfumato.paintings import (
    ArtSource,
    Orientation,
    PaintingInfo,
    content_hash,
    detect_orientation,
    fetch_from_met,
    fetch_paintings,
    list_cached_paintings,
)


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Return a temporary cache directory path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def _jpeg_bytes(size: tuple[int, int] = (20, 10)) -> bytes:
    """Create valid in-memory JPEG bytes for tests."""
    image = Image.new("RGB", size, color="white")
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _write_cache_entry(
    cache_dir: Path,
    source: ArtSource,
    source_id: str,
    image_bytes: bytes,
) -> tuple[Path, Path]:
    """Create an image + sidecar pair in the cache."""
    source_dir = cache_dir / source.value
    source_dir.mkdir(parents=True, exist_ok=True)

    image_path = source_dir / f"{source_id}.jpg"
    image_path.write_bytes(image_bytes)

    metadata = {
        "content_hash": hashlib.sha256(image_bytes).hexdigest(),
        "title": f"Title {source_id}",
        "artist": "Artist",
        "year": "1900",
        "source": source.value,
        "source_id": source_id,
        "source_url": f"https://example.test/{source_id}",
        "orientation": "landscape",
        "width": 20,
        "height": 10,
    }
    sidecar_path = source_dir / f"{source_id}.json"
    sidecar_path.write_text(json.dumps(metadata), encoding="utf-8")
    return image_path, sidecar_path


def _painting_info(source: ArtSource, source_id: str, image_path: Path) -> PaintingInfo:
    """Build a deterministic PaintingInfo for dispatch tests."""
    return PaintingInfo(
        image_path=image_path,
        content_hash="a" * 64,
        title=f"Title {source_id}",
        artist="Artist",
        year="1900",
        source=source,
        source_id=source_id,
        source_url=f"https://example.test/{source_id}",
        orientation=Orientation.LANDSCAPE,
        width=20,
        height=10,
    )


def test_detect_orientation_landscape_portrait_and_square(temp_cache_dir: Path) -> None:
    """orientation: landscape/portrait/square semantics are correct."""
    landscape = temp_cache_dir / "landscape.jpg"
    portrait = temp_cache_dir / "portrait.jpg"
    square = temp_cache_dir / "square.jpg"

    Image.new("RGB", (1001, 1000)).save(landscape)
    Image.new("RGB", (1000, 1001)).save(portrait)
    Image.new("RGB", (1000, 1000)).save(square)

    assert detect_orientation(landscape) == Orientation.LANDSCAPE
    assert detect_orientation(portrait) == Orientation.PORTRAIT
    assert detect_orientation(square) == Orientation.PORTRAIT


def test_detect_orientation_off_by_one_boundary(temp_cache_dir: Path) -> None:
    """orientation: 1-pixel difference is handled without boundary bugs."""
    wider = temp_cache_dir / "wider.jpg"
    taller = temp_cache_dir / "taller.jpg"
    Image.new("RGB", (11, 10)).save(wider)
    Image.new("RGB", (10, 11)).save(taller)

    assert detect_orientation(wider) == Orientation.LANDSCAPE
    assert detect_orientation(taller) == Orientation.PORTRAIT


def test_content_hash_stable_for_unchanged_bytes_and_path_independent(
    temp_cache_dir: Path,
) -> None:
    """content_hash depends on bytes only, not path."""
    bytes_data = _jpeg_bytes()
    first = temp_cache_dir / "first.jpg"
    second = temp_cache_dir / "second.jpg"
    first.write_bytes(bytes_data)
    second.write_bytes(bytes_data)

    hash_a = content_hash(first)
    hash_b = content_hash(second)

    assert hash_a == hash_b
    assert len(hash_a) == 64


def test_content_hash_changes_when_bytes_change(temp_cache_dir: Path) -> None:
    """content_hash changes when image bytes change."""
    first = temp_cache_dir / "first.jpg"
    second = temp_cache_dir / "second.jpg"
    first.write_bytes(_jpeg_bytes((20, 10)))
    second.write_bytes(_jpeg_bytes((21, 10)))

    assert content_hash(first) != content_hash(second)


def test_list_cached_paintings_sidecar_round_trip_absolute_paths(
    temp_cache_dir: Path,
) -> None:
    """sidecar: metadata reconstructs stable absolute image paths."""
    image_path, _ = _write_cache_entry(
        cache_dir=temp_cache_dir,
        source=ArtSource.MET,
        source_id="123",
        image_bytes=_jpeg_bytes(),
    )

    cached = list_cached_paintings(temp_cache_dir)

    assert len(cached) == 1
    assert cached[0].source == ArtSource.MET
    assert cached[0].source_id == "123"
    assert cached[0].image_path.is_absolute()
    assert cached[0].image_path == image_path.resolve()


def test_list_cached_paintings_skips_missing_and_malformed_sidecar(
    temp_cache_dir: Path,
) -> None:
    """sidecar: missing/malformed files are skipped without crashing scan."""
    _write_cache_entry(temp_cache_dir, ArtSource.MET, "valid", _jpeg_bytes())

    missing_sidecar_image = temp_cache_dir / ArtSource.MET.value / "missing.jpg"
    missing_sidecar_image.write_bytes(_jpeg_bytes())

    broken_image, broken_sidecar = _write_cache_entry(
        temp_cache_dir,
        ArtSource.WIKIMEDIA,
        "broken",
        _jpeg_bytes(),
    )
    assert broken_image.exists()
    broken_sidecar.write_text("{not-json", encoding="utf-8")

    cached = list_cached_paintings(temp_cache_dir)
    ids = {painting.source_id for painting in cached}

    assert ids == {"valid"}


@pytest.mark.asyncio
async def test_dispatch_routes_to_requested_sources_only(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """dispatch: fetch_paintings calls only requested source fetchers."""
    calls: list[str] = []

    async def fake_met(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        calls.append("met")
        return []

    async def fake_wikimedia(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        calls.append("wikimedia")
        return []

    monkeypatch.setattr(paintings, "fetch_from_met", fake_met)
    monkeypatch.setattr(paintings, "fetch_from_wikimedia", fake_wikimedia)

    result = await fetch_paintings(
        sources=["met", "wikimedia"],
        count=2,
        cache_dir=temp_cache_dir,
    )

    assert result == []
    assert calls == ["met", "wikimedia"]


@pytest.mark.asyncio
async def test_dispatch_aggregates_successful_results(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """dispatch: successful results across sources are aggregated."""
    image = temp_cache_dir / "placeholder.jpg"
    image.write_bytes(_jpeg_bytes())

    async def fake_met(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        return [_painting_info(ArtSource.MET, "1", image)]

    async def fake_wikimedia(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        return [_painting_info(ArtSource.WIKIMEDIA, "2", image)]

    monkeypatch.setattr(paintings, "fetch_from_met", fake_met)
    monkeypatch.setattr(paintings, "fetch_from_wikimedia", fake_wikimedia)

    result = await fetch_paintings(
        sources=["met", "wikimedia"],
        count=5,
        cache_dir=temp_cache_dir,
    )

    assert {p.source for p in result} == {ArtSource.MET, ArtSource.WIKIMEDIA}


@pytest.mark.asyncio
async def test_dispatch_empty_sources_returns_empty_list(temp_cache_dir: Path) -> None:
    """dispatch: empty source list returns empty result set."""
    assert await fetch_paintings([], count=3, cache_dir=temp_cache_dir) == []


@pytest.mark.asyncio
async def test_exclude_ids_filters_before_download(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """exclude_ids: excluded source_ids are filtered before download."""
    candidates = [
        paintings._SourceCandidate(
            source_id="skip-me",
            title="Skip",
            artist="Artist",
            year="1900",
            source_url="https://example.test/skip",
            image_url="https://example.test/skip.jpg",
        ),
        paintings._SourceCandidate(
            source_id="keep-me",
            title="Keep",
            artist="Artist",
            year="1900",
            source_url="https://example.test/keep",
            image_url="https://example.test/keep.jpg",
        ),
    ]

    downloaded_ids: list[str] = []
    original_download_one = paintings._download_one

    async def fake_discover(count: int) -> list[paintings._SourceCandidate]:
        return candidates

    async def fake_download_image_bytes(image_url: str) -> bytes:
        return _jpeg_bytes()

    async def tracked_download_one(
        source: ArtSource,
        candidate: paintings._SourceCandidate,
        cache_dir: Path,
    ) -> PaintingInfo:
        downloaded_ids.append(candidate.source_id)
        return await original_download_one(source, candidate, cache_dir)

    monkeypatch.setattr(paintings, "_discover_met_candidates", fake_discover)
    monkeypatch.setattr(paintings, "_download_image_bytes", fake_download_image_bytes)
    monkeypatch.setattr(paintings, "_download_one", tracked_download_one)

    result = await fetch_from_met(
        count=5,
        cache_dir=temp_cache_dir,
        exclude_ids={"met:skip-me"},
    )

    assert downloaded_ids == ["keep-me"]
    assert [item.source_id for item in result] == ["keep-me"]


@pytest.mark.asyncio
async def test_exclude_ids_preserves_non_excluded_candidates(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """exclude_ids: non-excluded candidates are still downloaded and returned."""
    candidates = [
        paintings._SourceCandidate(
            source_id="A",
            title="A",
            artist="Artist",
            year="1900",
            source_url="https://example.test/a",
            image_url="https://example.test/a.jpg",
        ),
        paintings._SourceCandidate(
            source_id="B",
            title="B",
            artist="Artist",
            year="1900",
            source_url="https://example.test/b",
            image_url="https://example.test/b.jpg",
        ),
    ]

    async def fake_discover(count: int) -> list[paintings._SourceCandidate]:
        return candidates

    async def fake_download_image_bytes(image_url: str) -> bytes:
        return _jpeg_bytes()

    monkeypatch.setattr(paintings, "_discover_met_candidates", fake_discover)
    monkeypatch.setattr(paintings, "_download_image_bytes", fake_download_image_bytes)

    result = await fetch_from_met(
        count=5,
        cache_dir=temp_cache_dir,
        exclude_ids={"met:A"},
    )

    assert [item.source_id for item in result] == ["B"]


@pytest.mark.asyncio
async def test_sidecar_written_and_reconstructed_via_list_cache(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """sidecar: fetch writes image/json and list cache reconstructs metadata."""
    candidate = paintings._SourceCandidate(
        source_id="42",
        title="The Test",
        artist="Tester",
        year="2024",
        source_url="https://example.test/42",
        image_url="https://example.test/42.jpg",
    )

    async def fake_discover(count: int) -> list[paintings._SourceCandidate]:
        return [candidate]

    async def fake_download_image_bytes(image_url: str) -> bytes:
        return _jpeg_bytes((40, 20))

    monkeypatch.setattr(paintings, "_discover_met_candidates", fake_discover)
    monkeypatch.setattr(paintings, "_download_image_bytes", fake_download_image_bytes)

    fetched = await fetch_from_met(count=1, cache_dir=temp_cache_dir)
    assert len(fetched) == 1

    image_file = temp_cache_dir / "met" / "42.jpg"
    sidecar_file = temp_cache_dir / "met" / "42.json"
    assert image_file.exists()
    assert sidecar_file.exists()

    sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    assert "image_path" not in sidecar
    assert sidecar["source"] == "met"
    assert sidecar["source_id"] == "42"

    cached = list_cached_paintings(temp_cache_dir)
    assert len(cached) == 1
    assert cached[0].image_path == image_file.resolve()
    assert cached[0].orientation == Orientation.LANDSCAPE


@pytest.mark.asyncio
async def test_regression_one_source_fails_others_succeed(
    monkeypatch: pytest.MonkeyPatch,
    temp_cache_dir: Path,
) -> None:
    """dispatch: one failing source does not discard successful sibling results."""
    image = temp_cache_dir / "ok.jpg"
    image.write_bytes(_jpeg_bytes())

    async def failing_met(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        raise TimeoutError("simulated timeout")

    async def successful_wikimedia(
        count: int, cache_dir: Path, exclude_ids: set[str] | None = None
    ) -> list[PaintingInfo]:
        return [_painting_info(ArtSource.WIKIMEDIA, "wik-1", image)]

    monkeypatch.setattr(paintings, "fetch_from_met", failing_met)
    monkeypatch.setattr(paintings, "fetch_from_wikimedia", successful_wikimedia)

    result = await fetch_paintings(
        sources=["met", "wikimedia"],
        count=2,
        cache_dir=temp_cache_dir,
    )

    assert [item.source for item in result] == [ArtSource.WIKIMEDIA]


def test_content_hash_not_affected_by_sidecar_changes(temp_cache_dir: Path) -> None:
    """content_hash identity is derived from image bytes only, not sidecar."""
    image_path, sidecar_path = _write_cache_entry(
        cache_dir=temp_cache_dir,
        source=ArtSource.MET,
        source_id="SK-123",
        image_bytes=_jpeg_bytes(),
    )
    before = content_hash(image_path)

    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["title"] = "Changed Title"
    payload["artist"] = "Changed Artist"
    sidecar_path.write_text(json.dumps(payload), encoding="utf-8")

    after = content_hash(image_path)
    assert before == after
