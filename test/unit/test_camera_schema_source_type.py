"""`source_type` at the API boundary.

The DB CHECK constraint enforces the same set at the storage layer;
these check it at the boundary so drift between the two fails loudly in
tests rather than silently reaching downstream consumers.

Was mostly about `pi_stream_url` / `pi_audio_url` and their XOR with
`source_type`. Both columns and the whole 'pi' path went on 2026-08-22 —
the Pi ingest endpoint had already been deleted, so a 'pi' row could be
created and could never deliver a frame, and no Pi ever ran in
production. The cases that outlived the columns are kept here; the eight
that only described them are gone rather than adjusted, because there is
nothing left for them to be about.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.db.camera import Camera
from src.schemas.camera import CameraResponse


def _valid_kwargs(**overrides) -> dict:
    """Minimum valid kwargs for a CameraResponse."""
    return {
        "id": uuid4(),
        "patient_id": uuid4(),
        "display_id": 1,
        "room": "Living Room",
        "icon_name": "tv",
        "online": True,
        "disabled": False,
        "ai_paused": False,
        "paired": True,
        "desired_state": "connected",
        "source_type": "iphone",
        "livekit_identity": "cam_abcd1234",
        "created_at": datetime.now(UTC),
        **overrides,
    }


def test_a_wall_phone_validates() -> None:
    assert CameraResponse(**_valid_kwargs()).source_type == "iphone"


def test_a_gateway_sensor_validates() -> None:
    cam = CameraResponse(**_valid_kwargs(source_type="gateway"))
    assert cam.source_type == "gateway"


def test_pi_is_no_longer_a_source_type() -> None:
    """The path is gone, so the value must stop being accepted.

    Left in the Literal it would keep parsing — rows could still be
    written as 'pi' and would then be handled by nothing at all, which
    is the same silence the removal was meant to end.
    """
    with pytest.raises(ValidationError):
        CameraResponse(**_valid_kwargs(source_type="pi"))


def test_an_unknown_source_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CameraResponse(**_valid_kwargs(source_type="martian"))


def test_camera_orm_defaults_source_type_to_iphone() -> None:
    """A row written without one is a wall phone, matching the DB default."""
    assert Camera.__table__.c.source_type.default.arg == "iphone"
