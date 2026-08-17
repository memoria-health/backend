"""Integration tests for the cameras_detectors reconciler.

`camera_detector_repo.sync_links_for_patient` derives the links from
each detector's `where_rooms` (state-drift bug fix: an all-rooms preset
with ZERO links meant its is_on toggle did nothing while the app showed
it armed). These tests exercise the lifecycle hooks that call it —
camera create / room rename / delete, detector create — plus the two
invariants that make it safe: null-scoped customs are never touched,
and a drifted link set is repaired idempotently.

Room matching is EXACT string compare, consistent with
`set_detector_rooms` (the clients build
the rooms picker from the cameras' own `room` values). All SEEDED
presets default to all-rooms (2026-07-28 decision), so room-scoped
behavior is exercised by explicitly scoping a preset in each test.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.repos import camera_detector_repo
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}

_CUSTOM_DETECTOR_BODY = {
    "name": "Late kitchen visits",
    "icon_name": "moon.stars",
    "description": "Kitchen activity between 11 PM and 6 AM.",
    "category": "safety",
    "when_scope": "11 PM – 6 AM",
    "custom_prompt": "Trigger when motion is detected in the kitchen between 11pm and 6am.",
}


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _create_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str, room: str,
) -> str:
    r = await client.post(
        f"/v1/patients/{patient_id}/cameras/pairing-code",
        headers=headers,
        json={"room": room},
    )
    assert r.status_code == 200, r.text
    return r.json()["camera_id"]


async def _detectors_by_name(
    client: AsyncClient, headers: dict[str, str], patient_id: str,
) -> dict[str, dict]:
    r = await client.get(f"/v1/patients/{patient_id}/detectors", headers=headers)
    assert r.status_code == 200
    return {d["name"]: d for d in r.json()}


async def _camera_ids_linked_to(session: AsyncSession, detector_id: str) -> set[str]:
    rows = (
        await session.execute(
            text("SELECT camera_id FROM cameras_detectors WHERE detector_id = :d"),
            {"d": detector_id},
        )
    ).scalars().all()
    return {str(r) for r in rows}


async def _detector_ids_linked_to(session: AsyncSession, camera_id: str) -> set[str]:
    rows = (
        await session.execute(
            text("SELECT detector_id FROM cameras_detectors WHERE camera_id = :c"),
            {"c": camera_id},
        )
    ).scalars().all()
    return {str(r) for r in rows}


async def _scope_preset(
    session: AsyncSession, detector_id: str, rooms: list[str],
) -> None:
    """Give a seeded (all-rooms) preset an explicit room scope, the way
    a caregiver would from the detector editor."""
    import json

    await session.execute(
        text("UPDATE detectors SET where_rooms = CAST(:r AS jsonb) WHERE id = :d"),
        {"r": json.dumps(rooms), "d": detector_id},
    )
    await session.commit()


# ─── (1) camera create links all-rooms presets ───────────────────


@pytest.mark.asyncio
async def test_create_camera_links_all_rooms_presets(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    # All seeds are all-rooms; scope two of them for the room-match case.
    await _scope_preset(session, dets["Wandering"]["id"], ["HALLWAY"])
    await _scope_preset(session, dets["Medication adherence"]["id"], ["KITCHEN"])
    cam = await _create_camera(client, headers, pid, "Living room")

    linked = await _detector_ids_linked_to(session, cam)
    assert dets["Falls"]["id"] in linked
    assert dets["Agitation"]["id"] in linked
    # Room-scoped presets only attach to matching rooms.
    assert dets["Wandering"]["id"] not in linked
    assert dets["Medication adherence"]["id"] not in linked

    # A camera in a room a preset is scoped to picks that preset up too.
    kitchen_cam = await _create_camera(client, headers, pid, "KITCHEN")
    linked_kitchen = await _detector_ids_linked_to(session, kitchen_cam)
    assert dets["Falls"]["id"] in linked_kitchen
    assert dets["Medication adherence"]["id"] in linked_kitchen
    assert dets["Wandering"]["id"] not in linked_kitchen


# ─── (2) room rename re-syncs room-scoped links ──────────────────


@pytest.mark.asyncio
async def test_rename_camera_room_resyncs_room_scoped_links(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    med = dets["Medication adherence"]["id"]
    wander = dets["Wandering"]["id"]
    await _scope_preset(session, med, ["KITCHEN"])
    await _scope_preset(session, wander, ["HALLWAY"])

    cam = await _create_camera(client, headers, pid, "KITCHEN")
    assert cam in await _camera_ids_linked_to(session, med)
    assert cam not in await _camera_ids_linked_to(session, wander)

    r = await client.patch(
        f"/v1/cameras/{cam}", headers=headers, json={"room": "HALLWAY"},
    )
    assert r.status_code == 200, r.text

    # Loses the kitchen-scoped preset, gains the hallway-scoped one.
    assert cam not in await _camera_ids_linked_to(session, med)
    assert cam in await _camera_ids_linked_to(session, wander)
    # All-rooms presets are unaffected by the rename.
    assert cam in await _camera_ids_linked_to(session, dets["Falls"]["id"])


# ─── (3) camera delete: no dangling links, others untouched ──────


@pytest.mark.asyncio
async def test_delete_camera_leaves_no_dangling_links(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    cam_a = await _create_camera(client, headers, pid, "Living room")
    cam_b = await _create_camera(client, headers, pid, "Bedroom")

    r = await client.delete(f"/v1/cameras/{cam_a}", headers=headers)
    assert r.status_code == 204

    assert await _detector_ids_linked_to(session, cam_a) == set()
    # The surviving camera keeps its all-rooms preset links.
    linked_b = await _detector_ids_linked_to(session, cam_b)
    assert dets["Falls"]["id"] in linked_b
    assert dets["Agitation"]["id"] in linked_b


# ─── (4) null-scoped customs: opt-in links are authoritative ─────


@pytest.mark.asyncio
async def test_sync_leaves_null_scoped_custom_opt_in_links_alone(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cam_a = await _create_camera(client, headers, pid, "Living room")
    cam_b = await _create_camera(client, headers, pid, "Bedroom")

    # No where_rooms in the body → custom with where_rooms IS NULL.
    r = await client.post(
        f"/v1/patients/{pid}/detectors", headers=headers,
        json=_CUSTOM_DETECTOR_BODY,
    )
    assert r.status_code == 201, r.text
    det_id = r.json()["id"]
    # Creation links it everywhere (existing behavior)…
    assert await _camera_ids_linked_to(session, det_id) == {cam_a, cam_b}

    # …then the caregiver opts camera B out.
    await session.execute(
        text(
            "DELETE FROM cameras_detectors "
            "WHERE camera_id = :c AND detector_id = :d"
        ),
        {"c": cam_b, "d": det_id},
    )
    await session.commit()

    inserted, deleted = await camera_detector_repo.sync_links_for_patient(
        session, UUID(pid),
    )
    await session.commit()
    assert (inserted, deleted) == (0, 0)
    # The pruned opt-in set survives the reconcile untouched.
    assert await _camera_ids_linked_to(session, det_id) == {cam_a}


# ─── (5) drift repair: deleted preset links come back ────────────


@pytest.mark.asyncio
async def test_sync_repairs_drifted_preset_links(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    cam_a = await _create_camera(client, headers, pid, "Living room")
    cam_b = await _create_camera(client, headers, pid, "Bedroom")
    assert await _camera_ids_linked_to(session, falls) == {cam_a, cam_b}

    # Simulate the field drift: an all-rooms preset with ZERO links.
    await session.execute(
        text("DELETE FROM cameras_detectors WHERE detector_id = :d"),
        {"d": falls},
    )
    await session.commit()
    assert await _camera_ids_linked_to(session, falls) == set()

    inserted, deleted = await camera_detector_repo.sync_links_for_patient(
        session, UUID(pid),
    )
    await session.commit()
    assert inserted == 2
    assert deleted == 0
    assert await _camera_ids_linked_to(session, falls) == {cam_a, cam_b}

    # Idempotent: a second pass is a no-op.
    inserted, deleted = await camera_detector_repo.sync_links_for_patient(
        session, UUID(pid),
    )
    await session.commit()
    assert (inserted, deleted) == (0, 0)

# ─── (6) picker edits write back through where_rooms ─────────────


@pytest.mark.asyncio
async def test_picker_edit_survives_sync(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Removing a preset from one camera via the per-camera picker must
    stick: the PUT rewrites where_rooms from the resulting links, so a
    later lifecycle sync doesn't silently re-add it (the two-writers
    reconciliation, 2026-07-28)."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    cam_a = await _create_camera(client, headers, pid, "Living room")
    cam_b = await _create_camera(client, headers, pid, "Bedroom")
    assert cam_a in await _camera_ids_linked_to(session, falls)

    keep = [d for d in await _detector_ids_linked_to(session, cam_a) if d != falls]
    r = await client.put(
        f"/v1/cameras/{cam_a}/detectors", headers=headers,
        json={"detector_ids": keep},
    )
    assert r.status_code == 200, r.text

    # where_rooms now names only the still-linked room…
    row = (
        await session.execute(
            text("SELECT where_rooms FROM detectors WHERE id = :d"), {"d": falls},
        )
    ).scalar_one()
    assert row == ["Bedroom"]
    # …so a full sync no longer re-adds the removed camera.
    from src.repos import camera_detector_repo as repo
    await repo.sync_links_for_patient(session, UUID(pid))
    await session.commit()
    assert cam_a not in await _camera_ids_linked_to(session, falls)
    assert cam_b in await _camera_ids_linked_to(session, falls)



# ─── (7) ON must mean "running somewhere" ────────────────────────────
#
# `is_on` is only one of three conditions the runtime checks; the third
# is a `cameras_detectors` row (detection/registry.py). Nothing tied
# them together, so a detector could be switched on while scoped to a
# room no camera has — armed in the app, running on nothing. Live for a
# full night on 2026-08-06: `sleep` scoped ["Bedroom"] against a camera
# whose room was "Aryaman Bedroom", 3270 ticks, zero output, no signal.


async def test_turning_on_a_detector_scoped_to_no_camera_is_refused(
    client: AsyncClient, session: AsyncSession, auth_user,
) -> None:
    """The toggle fails loudly instead of lying.

    A caregiver can then pick a real room and turn it on again; what the
    app shows is what the runtime is doing. Self-healing to all-rooms
    was the alternative and is worse: it silently runs a bedroom
    detector in the kitchen."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    await _create_camera(client, headers, pid, "Aryaman Bedroom")

    # The exact shape that cost a night: a room name no camera has.
    await _scope_preset(session, falls, ["Bedroom"])
    await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": False},
    )

    r = await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": True},
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "detector_scope_matches_no_camera"

    # And it really is still off — the refusal is not cosmetic.
    row = (
        await session.execute(
            text("SELECT is_on FROM detectors WHERE id = :d"), {"d": falls},
        )
    ).scalar_one()
    assert row is False


async def test_turning_on_reconciles_links_so_it_actually_runs(
    client: AsyncClient, session: AsyncSession, auth_user,
) -> None:
    """Pressing ON compiles.

    The toggle used to skip link reconciliation to 'keep the hot path
    link-free', so a detector whose links had drifted stayed dead no
    matter how many times it was toggled — only editing the rooms
    rebuilt them. Here the links are deleted behind the API's back (the
    drift a rename or a hand-edited row produces) and the toggle alone
    has to bring them back."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    cam = await _create_camera(client, headers, pid, "Bedroom")
    assert cam in await _camera_ids_linked_to(session, falls)

    await session.execute(
        text("DELETE FROM cameras_detectors WHERE detector_id = :d"), {"d": falls},
    )
    await session.commit()
    assert await _camera_ids_linked_to(session, falls) == set()

    r = await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": True},
    )
    assert r.status_code == 200, r.text
    assert cam in await _camera_ids_linked_to(session, falls)


async def test_a_scoped_detector_still_turns_on_when_the_room_exists(
    client: AsyncClient, session: AsyncSession, auth_user,
) -> None:
    """The guard must not block the ordinary case — an explicit scope
    that names a room the patient actually has."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    cam = await _create_camera(client, headers, pid, "Bedroom")
    await _scope_preset(session, falls, ["Bedroom"])

    r = await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": True},
    )
    assert r.status_code == 200, r.text
    assert cam in await _camera_ids_linked_to(session, falls)


async def test_rescoping_a_live_detector_to_a_dead_room_is_refused(
    client: AsyncClient, session: AsyncSession, auth_user,
) -> None:
    """The other way into the same dead state: leave it on and move the
    scope somewhere no camera lives."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]
    await _create_camera(client, headers, pid, "Bedroom")

    r = await client.patch(
        f"/v1/detectors/{falls}", headers=headers,
        json={"where_rooms": ["Attic"]},
    )
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "detector_scope_matches_no_camera"


async def test_toggling_on_before_any_camera_is_paired_is_allowed(
    client: AsyncClient, session: AsyncSession, auth_user,
) -> None:
    """Onboarding is not breakage.

    Before the first camera exists every detector resolves to zero
    cameras, exactly like a broken scope does. The difference is that
    this one self-heals: an all-rooms scope picks up each camera as it
    is paired, because camera create runs the reconciler. Refusing here
    would block the ordinary first-run flow to prevent a problem that
    cannot exist yet — caught by test_patch_toggle_is_on when the guard
    was first written without this carve-out."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = await _detectors_by_name(client, headers, pid)
    falls = dets["Falls"]["id"]

    await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": False},
    )
    r = await client.patch(
        f"/v1/detectors/{falls}", headers=headers, json={"is_on": True},
    )
    assert r.status_code == 200, r.text

    # And it wires itself up the moment a camera arrives.
    cam = await _create_camera(client, headers, pid, "Bedroom")
    assert cam in await _camera_ids_linked_to(session, falls)
