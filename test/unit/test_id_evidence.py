"""Unit: identity-evidence frames (soak observability) — flag gate,
annotate + write, empty-scene no-op, ring prune."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from src.services.detection.id_evidence import _RING, dump_id_evidence


def _jpeg(w: int = 120, h: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (30, 30, 30)).save(buf, "JPEG")
    return buf.getvalue()


def _identity(*, tracks=(), patient_bbox=None, frames=None):
    return SimpleNamespace(
        extracted_frames=frames if frames is not None else [_jpeg()],
        best_frame_index=0,
        tracks=tuple(tracks),
        patient_bbox=patient_bbox,
    )


def _track(label="UNKNOWN", tid=1):
    return SimpleNamespace(
        track_id=tid, label=label, confidence=0.42,
        bbox_series=((0, (10.0, 10.0, 60.0, 80.0)),), face_locked=False,
    )


def _settings(tmp_path, on=True):
    return SimpleNamespace(
        soak_id_frames_enabled=on, soak_id_frames_dir=str(tmp_path),
    )


@pytest.mark.asyncio
async def test_writes_annotated_frame_for_tracked_person(tmp_path) -> None:
    await dump_id_evidence(
        _identity(tracks=[_track()]), camera_id="cafe0123-x",
        settings=_settings(tmp_path),
    )
    files = list((tmp_path / "cafe0123").glob("*.jpg"))
    assert len(files) == 1
    assert files[0].read_bytes()[:2] == b"\xff\xd8"      # real JPEG out


@pytest.mark.asyncio
async def test_flag_off_and_empty_scene_write_nothing(tmp_path) -> None:
    await dump_id_evidence(
        _identity(tracks=[_track()]), camera_id="cafe0123-x",
        settings=_settings(tmp_path, on=False),
    )
    await dump_id_evidence(              # nobody seen -> no file
        _identity(tracks=[]), camera_id="cafe0123-x",
        settings=_settings(tmp_path),
    )
    await dump_id_evidence(              # nothing extracted -> no file
        _identity(tracks=[_track()], frames=[]), camera_id="cafe0123-x",
        settings=_settings(tmp_path),
    )
    assert not list(tmp_path.rglob("*.jpg"))


@pytest.mark.asyncio
async def test_ring_prune_keeps_newest(tmp_path) -> None:
    cam = tmp_path / "cafe0123"
    cam.mkdir(parents=True)
    for i in range(_RING + 7):
        (cam / f"20260714T{i:06d}.jpg").write_bytes(b"x")
    await dump_id_evidence(
        _identity(tracks=[_track("PATIENT")]), camera_id="cafe0123-x",
        settings=_settings(tmp_path),
    )
    left = sorted(cam.glob("*.jpg"))
    assert len(left) == _RING
    assert left[-1].read_bytes()[:2] == b"\xff\xd8"      # newest = real frame
