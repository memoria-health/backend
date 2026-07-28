"""Per-camera room-derived privacy — API surface (2026-07-27).

The patient-level ``privacy_mode_enabled`` toggle is RETIRED: it is no
longer PATCH-able (422 via ``extra="forbid"``) and no longer enforced;
the GET response still echoes the deprecated column until both clients
ship without the toggle. Privacy now lives per-camera
(``cameras.privacy_enabled``, derived from ``room_class``), is never
settable via the public camera PATCH, and is written only by the room
classifier and the support-only ``/internal/cameras/{id}/reclassify``
endpoint (which these tests exercise end-to-end, including the
clear-and-rerun + forced-override modes and the room-rename
re-classification hook).
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from test.integration.test_patients import _FULL_BODY, _provision


async def _create_patient(client: AsyncClient, headers: dict[str, str]) -> dict:
    r = await client.post("/v1/patients", headers=headers, json=_FULL_BODY)
    assert r.status_code == 201
    return r.json()


async def _create_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str, room: str = "Den"
) -> dict:
    r = await client.post(
        f"/v1/patients/{patient_id}/cameras/pairing-code",
        headers=headers,
        json={"room": room, "icon_name": "tv"},
    )
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_get_still_echoes_deprecated_patient_flag(
    client: AsyncClient, auth_user
) -> None:
    """The deprecated column stays on the wire (read-only, false) until
    the clients ship without the toggle."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.get(f"/v1/patients/{created['id']}", headers=headers)
    assert r.status_code == 200
    assert r.json()["privacy_mode_enabled"] is False


@pytest.mark.asyncio
async def test_patch_patient_privacy_mode_is_rejected(
    client: AsyncClient, auth_user
) -> None:
    """Retired toggle: a client still sending privacy_mode_enabled is
    422'd like any unknown field (extra="forbid")."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    for value in (True, False):
        r = await client.patch(
            f"/v1/patients/{created['id']}",
            headers=headers,
            json={"privacy_mode_enabled": value},
        )
        assert r.status_code == 422


@pytest.mark.asyncio
async def test_camera_response_exposes_privacy_fields_defaulting_off(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)
    await _create_camera(client, headers, created["id"])

    r = await client.get(f"/v1/patients/{created['id']}/cameras", headers=headers)
    assert r.status_code == 200
    cam = r.json()[0]
    assert cam["privacy_enabled"] is False
    assert cam["room_class"] is None
    assert cam["room_class_confidence"] is None
    assert cam["room_classified_at"] is None


@pytest.mark.asyncio
async def test_camera_patch_cannot_set_privacy_fields(
    client: AsyncClient, auth_user
) -> None:
    """The privacy fields are locked — never settable via the public
    camera PATCH (422 via extra="forbid")."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)
    cam = await _create_camera(client, headers, created["id"])

    for body in (
        {"privacy_enabled": False},
        {"room_class": "other"},
        {"room_classified_at": None},
        {"room_class_confidence": 0.9},
    ):
        r = await client.patch(
            f"/v1/cameras/{cam['camera_id']}", headers=headers, json=body,
        )
        assert r.status_code == 422, body


@pytest.mark.asyncio
async def test_internal_reclassify_forced_override_locks_private(
    client: AsyncClient, auth_user, session
) -> None:
    """Support override: forcing bathroom finalizes the verdict and
    flips privacy_enabled on (no key needed in env=local)."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)
    cam = await _create_camera(client, headers, created["id"])

    r = await client.post(
        f"/internal/cameras/{cam['camera_id']}/reclassify",
        json={"room_class": "bathroom"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["room_class"] == "bathroom"
    assert body["privacy_enabled"] is True
    assert body["room_classified_at"] is not None

    # And the caregiver-facing camera object reflects it.
    r = await client.get(f"/v1/patients/{created['id']}/cameras", headers=headers)
    assert r.json()[0]["privacy_enabled"] is True


@pytest.mark.asyncio
async def test_internal_reclassify_clear_rearms_but_keeps_privacy(
    client: AsyncClient, auth_user, session
) -> None:
    """Clear-and-rerun: room_classified_at + confidence are cleared so
    the classifier re-runs, but the privacy verdict is KEPT until a
    fresh verdict lands (fail-closed)."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)
    cam = await _create_camera(client, headers, created["id"])

    # Lock it private first.
    r = await client.post(
        f"/internal/cameras/{cam['camera_id']}/reclassify",
        json={"room_class": "bedroom"},
    )
    assert r.status_code == 200 and r.json()["privacy_enabled"] is True

    # Now clear (empty body).
    r = await client.post(f"/internal/cameras/{cam['camera_id']}/reclassify")
    assert r.status_code == 200
    body = r.json()
    assert body["room_classified_at"] is None
    assert body["room_class_confidence"] is None
    assert body["privacy_enabled"] is True  # fail-closed until re-verdict
    assert body["room_class"] == "bedroom"  # stale class kept, informational


@pytest.mark.asyncio
async def test_internal_reclassify_404_for_unknown_camera(
    client: AsyncClient, auth_user
) -> None:
    r = await client.post(f"/internal/cameras/{uuid4()}/reclassify")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_room_rename_clears_classification_for_rerun(
    client: AsyncClient, auth_user, session
) -> None:
    """Misclassification escape hatch ⚑3: renaming the camera's room via
    the public PATCH clears room_classified_at (+ confidence) so the
    classifier re-runs; privacy stays as-is until the new verdict."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)
    cam = await _create_camera(client, headers, created["id"], room="Snug")

    # Simulate a finalized classification directly in the DB.
    await session.execute(
        text(
            "UPDATE cameras SET room_class = 'bathroom', "
            "room_class_confidence = 0.4, room_classified_at = now(), "
            "privacy_enabled = true WHERE id = :id"
        ),
        {"id": cam["camera_id"]},
    )
    await session.commit()

    r = await client.patch(
        f"/v1/cameras/{cam['camera_id']}",
        headers=headers,
        json={"room": "Reading Nook"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["room"] == "Reading Nook"
    assert body["room_classified_at"] is None  # re-armed
    assert body["room_class_confidence"] is None
    assert body["privacy_enabled"] is True  # fail-closed through the window
