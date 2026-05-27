"""Unit tests for the source_type + pi_stream_url additions to the
camera Pydantic schema and ORM model.

The DB CHECK constraint enforces the same invariant at the storage
layer. These tests verify the same rules at the API boundary so
schema drift between DB and code fails loudly in tests rather than
silently corrupting downstream consumers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.db.camera import Camera
from src.schemas.camera import CameraResponse


def _valid_iphone_kwargs() -> dict:
    """Minimum valid kwargs for an iPhone CameraResponse."""
    return {
        "id": uuid4(),
        "patient_id": uuid4(),
        "display_id": 1,
        "room": "Living Room",
        "icon_name": "tv",
        "online": True,
        "disabled": False,
        "desired_state": "connected",
        "source_type": "iphone",
        "pi_stream_url": None,
        "livekit_identity": "cam_abcd1234",
        "last_seen_at": datetime.now(UTC),
        "created_at": datetime.now(UTC),
    }


def _valid_pi_kwargs() -> dict:
    """Minimum valid kwargs for a Pi CameraResponse."""
    return {
        **_valid_iphone_kwargs(),
        "source_type": "pi",
        "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
    }


# ─── happy paths ─────────────────────────────────────────────────


def test_iphone_camera_with_no_pi_url_validates() -> None:
    cam = CameraResponse(**_valid_iphone_kwargs())
    assert cam.source_type == "iphone"
    assert cam.pi_stream_url is None


def test_pi_camera_with_url_validates() -> None:
    cam = CameraResponse(**_valid_pi_kwargs())
    assert cam.source_type == "pi"
    assert cam.pi_stream_url == "http://10.0.0.44:8000/stream.mjpg"


def test_pi_camera_accepts_https_url() -> None:
    """No protocol restriction in the schema; URL validity is left
    to the worker's connect attempt.
    """
    kw = {**_valid_pi_kwargs(), "pi_stream_url": "https://cam.local/stream"}
    cam = CameraResponse(**kw)
    assert cam.pi_stream_url == "https://cam.local/stream"


# ─── invariant violations ────────────────────────────────────────


def test_pi_source_without_url_is_accepted() -> None:
    """After the push refactor, pi_stream_url is vestigial and Pi
    rows do not need to carry it. Null is valid.
    """
    kw = {**_valid_pi_kwargs(), "pi_stream_url": None}
    cam = CameraResponse(**kw)
    assert cam.source_type == "pi"
    assert cam.pi_stream_url is None


def test_iphone_source_with_url_is_accepted() -> None:
    """The pi_stream_url / pi_audio_url columns are vestigial after
    the push refactor; the Pydantic boundary no longer enforces the
    source-type XOR that used to live in the validator. A spurious
    URL on an iPhone row is harmless because nothing reads it.
    """
    kw = {
        **_valid_iphone_kwargs(),
        "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    cam = CameraResponse(**kw)
    assert cam.source_type == "iphone"
    assert cam.pi_stream_url == "http://10.0.0.44:8000/stream.mjpg"


def test_iphone_source_with_audio_url_is_accepted() -> None:
    """Same rationale as the pi_stream_url test above: vestigial
    columns are no longer XOR-validated. Spurious audio URL on an
    iPhone row is ignored downstream.
    """
    kw = {
        **_valid_iphone_kwargs(),
        "pi_audio_url": "http://10.0.0.44:8001/audio.pcm",
    }
    cam = CameraResponse(**kw)
    assert cam.source_type == "iphone"
    assert cam.pi_audio_url == "http://10.0.0.44:8001/audio.pcm"


def test_pi_source_can_carry_audio_url() -> None:
    """A Pi camera with a USB mic gets a pi_audio_url. Optional
    field; pi_stream_url is still required."""
    kw = {**_valid_pi_kwargs(), "pi_audio_url": "http://10.0.0.44:8001/audio.pcm"}
    cam = CameraResponse(**kw)
    assert cam.pi_audio_url == "http://10.0.0.44:8001/audio.pcm"


def test_pi_source_without_audio_url_is_valid() -> None:
    """Video-only Pi camera. Most common case for v1."""
    kw = {**_valid_pi_kwargs()}  # audio omitted
    cam = CameraResponse(**kw)
    assert cam.pi_audio_url is None


def test_invalid_source_type_value_is_rejected() -> None:
    kw = {**_valid_iphone_kwargs(), "source_type": "android"}
    with pytest.raises(ValidationError):
        CameraResponse(**kw)


# ─── pi_stream_url length bounds ─────────────────────────────────


def test_pi_url_too_short_is_rejected() -> None:
    """Min length 8 covers `http://x` at minimum. Shorter values
    cannot be valid HTTP URLs.
    """
    kw = {**_valid_pi_kwargs(), "pi_stream_url": "http://"}  # 7 chars
    with pytest.raises(ValidationError):
        CameraResponse(**kw)


def test_pi_url_too_long_is_rejected() -> None:
    """Cap of 2048 prevents accidental large-blob storage."""
    kw = {
        **_valid_pi_kwargs(),
        "pi_stream_url": "http://" + "a" * 3000,
    }
    with pytest.raises(ValidationError):
        CameraResponse(**kw)


# ─── ORM model wiring ────────────────────────────────────────────


def test_camera_orm_has_source_type_field_with_iphone_default() -> None:
    """The ORM mapping must define source_type and pi_stream_url with
    the correct nullability, Python-side default, and column length
    cap. A regression that flipped any of these would corrupt the
    cameras table without being caught by the type checker.
    """
    assert hasattr(Camera, "source_type")
    assert hasattr(Camera, "pi_stream_url")

    src_col = Camera.__table__.columns["source_type"]
    assert src_col.nullable is False
    # Python-side default is load-bearing: SQLAlchemy fills this in
    # before INSERT, and a wrong default would create rows that
    # immediately violate the cameras_pi_stream_url_required CHECK
    # constraint (e.g. default='pi' with pi_stream_url=NULL).
    assert src_col.default is not None
    assert src_col.default.arg == "iphone"

    pi_url_col = Camera.__table__.columns["pi_stream_url"]
    assert pi_url_col.nullable is True
    # Column length cap aligns with the Pydantic schema's max_length
    # and the DB-level CHECK on length(pi_stream_url).
    assert pi_url_col.type.length == 2048
