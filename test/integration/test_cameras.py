"""Integration tests for the camera API.

Cameras are CREATED by wall phones in a later plan. v0 only lists +
deletes, so the test surface is small: list shape, delete happy path,
404s, cross-caregiver isolation, cascade-on-patient-delete.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt, seed_camera


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _insert_camera(
    session: AsyncSession,
    *,
    patient_id: str,
    display_id: int,
    room: str,
    icon_name: str = "tv",
    online: bool = True,
) -> UUID:
    """This suite's defaults, on the shared seeder."""
    return await seed_camera(
        session, patient_id=patient_id, display_id=display_id,
        room=room, icon_name=icon_name, online=online,
    )


@pytest.mark.asyncio
async def test_list_cameras_empty(client: AsyncClient, auth_user) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.get(f"/v1/patients/{pid}/cameras", headers=headers)
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_cameras_returns_sorted_by_display_id(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    # Insert out of order — endpoint should still return ascending display_id.
    await _insert_camera(session, patient_id=pid, display_id=3, room="Bedroom",  icon_name="bed.double")
    await _insert_camera(session, patient_id=pid, display_id=1, room="Living room", icon_name="tv")
    await _insert_camera(session, patient_id=pid, display_id=2, room="Kitchen", icon_name="fork.knife")

    r = await client.get(f"/v1/patients/{pid}/cameras", headers=headers)
    assert r.status_code == 200
    rooms = [c["room"] for c in r.json()]
    assert rooms == ["Living room", "Kitchen", "Bedroom"]


@pytest.mark.asyncio
async def test_list_cameras_full_field_assertion(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_camera(
        session, patient_id=pid, display_id=1, room="Living room",
        icon_name="tv", online=True,
    )

    cams = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert len(cams) == 1
    c = cams[0]
    assert c["patient_id"] == pid
    assert c["display_id"] == 1
    assert c["room"] == "Living room"
    assert c["icon_name"] == "tv"
    assert c["online"] is True
    assert "id" in c
    assert "created_at" in c


@pytest.mark.asyncio
async def test_list_cameras_unauthenticated_returns_401(
    client: AsyncClient
) -> None:
    r = await client.get(
        "/v1/patients/00000000-0000-0000-0000-000000000000/cameras"
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_delete_camera(client: AsyncClient, auth_user, session) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")

    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204

    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert listed == []


@pytest.mark.asyncio
async def test_delete_camera_404_when_missing(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.delete(
        "/v1/cameras/00000000-0000-0000-0000-000000000000", headers=headers,
    )
    assert r.status_code == 404


# ─────────────────────────────────────────────────────────────────────
# P2: DELETE handler → DetectorWorker.stop_worker_for_camera wiring.
#
# Pins the contract that camera DELETE actually tells the in-process
# detector worker to tear down. Without this, deleting a camera while
# it's actively streaming leaves a zombie worker spinning until P1's
# self-destruct catches it ~30s later. P2 closes the gap to ~1s.
#
# All three tests inject a fake detector_worker into app.state — the
# real one needs LiveKit creds + a live LiveKit room, which are out of
# scope for the API-layer test. We're proving the WIRING here; the
# detector-worker logic itself is unit-tested at
# test/unit/test_detector_worker_stop_for_camera.py.
# ─────────────────────────────────────────────────────────────────────


class _RecordingDetectorWorker:
    """Captures the (camera_id, timeout) arguments stop_worker_for_camera
    was called with, and can be configured to hang or raise for the
    failure-mode tests.
    """

    def __init__(
        self,
        *,
        hang_seconds: float = 0.0,
        raise_exc: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[UUID, float]] = []
        self._hang_seconds = hang_seconds
        self._raise_exc = raise_exc

    async def stop_worker_for_camera(
        self, camera_id: UUID, *, timeout: float = 5.0,
    ) -> bool:
        import asyncio

        self.calls.append((camera_id, timeout))
        if self._hang_seconds > 0:
            await asyncio.sleep(self._hang_seconds)
        if self._raise_exc is not None:
            raise self._raise_exc
        return True


@pytest.mark.asyncio
async def test_delete_camera_calls_stop_worker_for_camera(
    client: AsyncClient, auth_user, session, app,
) -> None:
    """DELETE /v1/cameras/{id} must call
    detector_worker.stop_worker_for_camera with the deleted camera's
    UUID. This is the wiring guarantee — without it, P2's whole
    purpose is defeated.
    """
    fake = _RecordingDetectorWorker()
    app.state.detector_worker = fake

    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Bedroom")

    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == cid


@pytest.mark.asyncio
async def test_delete_camera_succeeds_when_worker_missing(
    client: AsyncClient, auth_user, session, app,
) -> None:
    """Cold-boot edge case: if the detector_worker isn't on app.state
    (lifespan didn't run, misconfig, etc.), the DELETE must still
    succeed with 204. Without this guard, every test that doesn't
    install a fake would 500.
    """
    # Explicitly remove any worker that might have been set by a
    # prior test in the same session.
    if hasattr(app.state, "detector_worker"):
        delattr(app.state, "detector_worker")

    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Bedroom")

    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204


@pytest.mark.asyncio
async def test_delete_camera_succeeds_when_worker_stop_times_out(
    client: AsyncClient, auth_user, session, app,
) -> None:
    """If detector_worker.stop_worker_for_camera hangs (e.g.
    LiveKit network blip during disconnect), the DELETE must STILL
    return 204 within a bounded time — not park the API request
    forever waiting on the worker.

    The handler's outer wait_for is 5s; we install a worker that
    hangs for 30s and assert the request returns in well under
    that.
    """
    import time

    fake = _RecordingDetectorWorker(hang_seconds=30.0)
    app.state.detector_worker = fake

    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Bedroom")

    t0 = time.monotonic()
    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    elapsed = time.monotonic() - t0

    assert r.status_code == 204
    assert elapsed < 7.0, (
        f"DELETE took {elapsed:.2f}s; handler should bound at ~5s"
    )


@pytest.mark.asyncio
async def test_delete_camera_succeeds_when_worker_stop_raises(
    client: AsyncClient, auth_user, session, app,
) -> None:
    """If the worker raises, the DB row has ALREADY been committed
    (handler ordering is delete → commit → worker stop). DELETE
    must still return 204 — failure of worker stop is logged but
    not propagated.
    """
    fake = _RecordingDetectorWorker(
        raise_exc=RuntimeError("simulated worker stop blew up"),
    )
    app.state.detector_worker = fake

    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Bedroom")

    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204
    # And the row is actually gone.
    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert listed == []


@pytest.mark.asyncio
async def test_other_caregiver_cannot_see_or_delete_camera(
    client: AsyncClient, auth_user_factory, session
) -> None:
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")

    # B's list omits the camera.
    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=h_b)).json()
    assert listed == []
    # B's DELETE 404 (no existence leak).
    r = await client.delete(f"/v1/cameras/{cid}", headers=h_b)
    assert r.status_code == 404

    # And A's camera is still there.
    listed_a = (await client.get(f"/v1/patients/{pid}/cameras", headers=h_a)).json()
    assert len(listed_a) == 1


@pytest.mark.asyncio
async def test_delete_patient_cascades_to_cameras(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")
    await _insert_camera(session, patient_id=pid, display_id=2, room="Kitchen")

    assert (await client.delete(f"/v1/patients/{pid}", headers=headers)).status_code == 204

    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert listed == []
