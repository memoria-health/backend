"""Integration tests for the event API.

Events are written by wall phones in a later plan. For these tests
we INSERT events directly via SQL so we can exercise the read + review
path the iOS app exercises today.

Coverage:
- List filter chips (all / unreviewed / reviewed) match what the iOS
  Events tab shows.
- Cursor pagination is stable and complete.
- PATCH stamps `reviewed_at` exactly once (when reviewed flips false→true).
- RLS: another caregiver sees nothing.
- Cascade: deleting the patient removes their events.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt


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


async def _insert_event(
    session: AsyncSession,
    *,
    patient_id: str,
    title: str,
    category: str = "safety",
    source: str = "preset",
    icon_name: str = "figure.fall",
    location: str = "Living room",
    occurred_at: datetime | None = None,
    has_clip: bool = True,
    narrative: str = "",
    opened: bool = False,
    reviewed: bool = False,
) -> UUID:
    """Insert one event row via raw SQL so the test doesn't have to
    invent a wall-phone ingest endpoint. Returns the new event id.
    """
    now = occurred_at or datetime.now(timezone.utc)
    eid = uuid4()
    await session.execute(
        text("""
            INSERT INTO events (
                id, patient_id, category, source, icon_name, title, location,
                occurred_at, has_clip, narrative, opened, reviewed, created_at
            ) VALUES (
                :id, :pid, :cat, :src, :icon, :title, :loc, :occ, :clip,
                :narr, :opened, :reviewed, :created
            )
        """),
        {
            "id": str(eid),
            "pid": patient_id,
            "cat": category,
            "src": source,
            "icon": icon_name,
            "title": title,
            "loc": location,
            "occ": now,
            "clip": has_clip,
            "narr": narrative,
            "opened": opened,
            "reviewed": reviewed,
            "created": now,
        },
    )
    await session.commit()
    return eid


# ─── List + filter ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_events_empty_returns_empty_page(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.get(f"/v1/patients/{pid}/events", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body == {"items": [], "next_cursor": None}


@pytest.mark.asyncio
async def test_list_events_filter_all_returns_everything(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_event(session, patient_id=pid, title="A", reviewed=False)
    await _insert_event(session, patient_id=pid, title="B", reviewed=True)
    await _insert_event(session, patient_id=pid, title="C", reviewed=False)

    r = await client.get(
        f"/v1/patients/{pid}/events?filter=all", headers=headers
    )
    assert r.status_code == 200
    titles = sorted(e["title"] for e in r.json()["items"])
    assert titles == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_list_events_filter_unreviewed(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_event(session, patient_id=pid, title="Unreviewed1", reviewed=False)
    await _insert_event(session, patient_id=pid, title="Unreviewed2", reviewed=False)
    await _insert_event(session, patient_id=pid, title="DoneA", reviewed=True)

    r = await client.get(
        f"/v1/patients/{pid}/events?filter=unreviewed", headers=headers
    )
    assert r.status_code == 200
    titles = sorted(e["title"] for e in r.json()["items"])
    assert titles == ["Unreviewed1", "Unreviewed2"]


@pytest.mark.asyncio
async def test_list_events_filter_reviewed(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_event(session, patient_id=pid, title="Live1", reviewed=False)
    await _insert_event(session, patient_id=pid, title="DoneA", reviewed=True)
    await _insert_event(session, patient_id=pid, title="DoneB", reviewed=True)

    r = await client.get(
        f"/v1/patients/{pid}/events?filter=reviewed", headers=headers
    )
    assert r.status_code == 200
    titles = sorted(e["title"] for e in r.json()["items"])
    assert titles == ["DoneA", "DoneB"]


@pytest.mark.asyncio
async def test_list_events_sorted_newest_first(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    now = datetime.now(timezone.utc)
    await _insert_event(session, patient_id=pid, title="Oldest", occurred_at=now - timedelta(hours=2))
    await _insert_event(session, patient_id=pid, title="Middle", occurred_at=now - timedelta(hours=1))
    await _insert_event(session, patient_id=pid, title="Newest", occurred_at=now)

    items = (await client.get(f"/v1/patients/{pid}/events", headers=headers)).json()["items"]
    assert [e["title"] for e in items] == ["Newest", "Middle", "Oldest"]


# ─── Cursor pagination ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_cursor_pagination_complete_no_gaps(
    client: AsyncClient, auth_user, session
) -> None:
    """7 events, limit=3: page1=3, page2=3, page3=1, page4=empty.
    No duplicates, no gaps, newest-first throughout."""
    headers, pid = await _setup_patient(client, auth_user)
    now = datetime.now(timezone.utc)
    for i in range(7):
        await _insert_event(
            session, patient_id=pid, title=f"e{i}",
            occurred_at=now - timedelta(minutes=i),
        )

    all_titles: list[str] = []
    cursor: str | None = None
    for _ in range(5):  # 5 is enough to drain a 7-row dataset at limit=3
        url = f"/v1/patients/{pid}/events?limit=3"
        if cursor:
            url += f"&cursor={cursor}"
        body = (await client.get(url, headers=headers)).json()
        all_titles.extend(e["title"] for e in body["items"])
        cursor = body["next_cursor"]
        if cursor is None:
            break
    assert all_titles == ["e0", "e1", "e2", "e3", "e4", "e5", "e6"]


@pytest.mark.asyncio
async def test_cursor_pagination_last_page_has_null_cursor(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_event(session, patient_id=pid, title="only")

    body = (await client.get(f"/v1/patients/{pid}/events", headers=headers)).json()
    assert len(body["items"]) == 1
    assert body["next_cursor"] is None


# ─── GET one ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_one_event_returns_full_row(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(
        session, patient_id=pid,
        title="Possible fall", location="Living room",
        category="safety", source="preset",
        narrative="A sharp downward motion at 14:32...",
        has_clip=True,
    )

    r = await client.get(f"/v1/events/{eid}", headers=headers)
    assert r.status_code == 200
    e = r.json()
    assert e["id"] == str(eid)
    assert e["patient_id"] == pid
    assert e["category"] == "safety"
    assert e["source"] == "preset"
    assert e["title"] == "Possible fall"
    assert e["location"] == "Living room"
    assert e["narrative"] == "A sharp downward motion at 14:32..."
    assert e["has_clip"] is True
    assert e["opened"] is False
    assert e["reviewed"] is False
    assert e["review_choice"] is None
    assert e["review_notes"] is None
    assert e["reviewed_at"] is None


@pytest.mark.asyncio
async def test_get_one_404_when_missing(client: AsyncClient, auth_user) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.get(
        "/v1/events/00000000-0000-0000-0000-000000000000", headers=headers
    )
    assert r.status_code == 404


# ─── PATCH (review flow) ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_mark_opened_does_not_stamp_reviewed_at(
    client: AsyncClient, auth_user, session
) -> None:
    """Opening an event (tapping into Event Detail) is distinct from
    reviewing it. Only `reviewed=true` stamps `reviewed_at`."""
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="x")

    r = await client.patch(
        f"/v1/events/{eid}", headers=headers, json={"opened": True},
    )
    assert r.status_code == 200
    e = r.json()
    assert e["opened"] is True
    assert e["reviewed"] is False
    assert e["reviewed_at"] is None


@pytest.mark.asyncio
async def test_patch_full_review_stamps_reviewed_at(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="x")

    r = await client.patch(
        f"/v1/events/{eid}",
        headers=headers,
        json={
            "opened": True,
            "reviewed": True,
            "review_choice": "correct",
            "review_notes": "Yes, real fall. Tripped on the rug.",
        },
    )
    assert r.status_code == 200
    e = r.json()
    assert e["opened"] is True
    assert e["reviewed"] is True
    assert e["review_choice"] == "correct"
    assert e["review_notes"] == "Yes, real fall. Tripped on the rug."
    assert e["reviewed_at"] is not None


@pytest.mark.asyncio
async def test_patch_review_again_does_not_re_stamp(
    client: AsyncClient, auth_user, session
) -> None:
    """If a row is already reviewed, a PATCH that sets reviewed=true
    again should NOT bump reviewed_at — preserves the original audit
    timestamp."""
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="x")

    r1 = await client.patch(
        f"/v1/events/{eid}",
        headers=headers,
        json={"reviewed": True, "review_choice": "correct"},
    )
    first_stamp = r1.json()["reviewed_at"]

    r2 = await client.patch(
        f"/v1/events/{eid}",
        headers=headers,
        json={"reviewed": True, "review_notes": "Adding a note later"},
    )
    assert r2.json()["reviewed_at"] == first_stamp
    assert r2.json()["review_notes"] == "Adding a note later"


@pytest.mark.asyncio
async def test_patch_rejects_invalid_review_choice(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="x")
    r = await client.patch(
        f"/v1/events/{eid}", headers=headers,
        json={"review_choice": "maybe"},                # not in {correct, wrong, unsure}
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_rejects_unknown_field(
    client: AsyncClient, auth_user, session
) -> None:
    """`extra="forbid"` — can't sneak in a `title` change via PATCH."""
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="x")
    r = await client.patch(
        f"/v1/events/{eid}", headers=headers, json={"title": "renamed"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_404_when_event_missing(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.patch(
        "/v1/events/00000000-0000-0000-0000-000000000000",
        headers=headers,
        json={"opened": True},
    )
    assert r.status_code == 404


# ─── Cross-caregiver isolation ────────────────────────────────────


@pytest.mark.asyncio
async def test_other_caregiver_cannot_see_event(
    client: AsyncClient, auth_user_factory, session
) -> None:
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]
    eid = await _insert_event(session, patient_id=pid, title="A's fall")

    # B's list omits.
    listed = (await client.get(f"/v1/patients/{pid}/events", headers=h_b)).json()
    assert listed == {"items": [], "next_cursor": None}
    # B's GET 404.
    r = await client.get(f"/v1/events/{eid}", headers=h_b)
    assert r.status_code == 404
    # B's PATCH 404.
    r = await client.patch(f"/v1/events/{eid}", headers=h_b, json={"opened": True})
    assert r.status_code == 404


# ─── Cascade ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_patient_cascades_to_events(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    eid = await _insert_event(session, patient_id=pid, title="will be orphaned")
    # Confirm row exists.
    assert (await client.get(f"/v1/events/{eid}", headers=headers)).status_code == 200

    # Delete the patient.
    assert (await client.delete(f"/v1/patients/{pid}", headers=headers)).status_code == 204

    # Event is gone.
    assert (await client.get(f"/v1/events/{eid}", headers=headers)).status_code == 404
