"""Integration tests for POST /v1/clips/ingest.

This is the wall phone's path. Auth is the camera api-key (bcrypt-checked).
We monkeypatch the Gemini + OpenAI provider callables to return canned
`ProviderResult`s so the tests never touch real LLMs.

Coverage:
- happy path INSERT into `analyses`
- auth rejections (missing / bad / mismatched bearer)
- form-validation rejections (motion, timestamp, empty file)
- the closed-state short-circuit (LLM skipped, provider='skipped')
- fall promotion + confidence gate + 30-second dedup
- gemini → openai fallback
- both-providers-failed → 502
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.event import Event
from src.services.detection.providers.types import ProviderResult
from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


# ─── Provider mock harness ─────────────────────────────────────────


class _ProviderMock:
    """Stand-in for `gemini_provider` / `openai_provider`. Captures
    call count + last args; returns the configured ProviderResult or
    raises if `raises` is set.
    """

    def __init__(
        self,
        *,
        result: ProviderResult | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        *,
        mp4: bytes,
        detectors,
        log_tag: str,
        settings,
    ) -> ProviderResult:
        self.calls.append({
            "mp4_len": len(mp4),
            "detectors": list(detectors),
            "log_tag": log_tag,
            "settings": settings,
        })
        if self.raises is not None:
            raise self.raises
        assert self.result is not None
        return self.result


def _default_result(**overrides) -> ProviderResult:
    payload = {
        "description": "A person is walking in the living room.",
        "people_present": True,
        "people_count": 1,
        "detector_fields": {
            "fall_detected": False,
            "fall_confidence": 0.05,
            "fall_reasoning": "no abnormal motion",
            "wandering_detected": False,
            "wandering_confidence": 0.1,
            "wandering_reasoning": "no exit interaction",
            "agitation_detected": False,
            "agitation_confidence": 0.1,
            "agitation_reasoning": "calm",
            "medication_taken": False,
            "medication_confidence": 0.0,
            "medication_reasoning": "no pills visible",
        },
    }
    payload.update(overrides)
    return ProviderResult(**payload)


@pytest.fixture
def gemini_mock(monkeypatch):
    """Default: Gemini returns a "person walking, no fall" result.
    Tests that need a different shape override `mock.result`.
    """
    mock = _ProviderMock(result=_default_result())
    monkeypatch.setattr("src.api.clips.gemini_provider", mock)
    return mock


@pytest.fixture
def openai_mock(monkeypatch):
    """Default: OpenAI raises if called. Tests that exercise the
    fallback path override `mock.raises = None` and set `.result`.
    """
    mock = _ProviderMock(raises=RuntimeError("openai mock not configured"))
    monkeypatch.setattr("src.api.clips.openai_provider", mock)
    return mock


@pytest.fixture
def fast_bcrypt_app(app, monkeypatch):
    """Same trick as test_pairing — drop bcrypt rounds to 4 so the
    pair endpoint + camera_auth path don't dominate test runtime.
    """
    from src.api.deps import get_settings_dep

    real = get_settings()
    fast = Settings(**{**real.model_dump(), "bcrypt_rounds": 4})
    app.dependency_overrides[get_settings_dep] = lambda: fast
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


# ─── Helpers ───────────────────────────────────────────────────────


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _pair_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str,
    *, room: str = "Living room", icon_name: str = "tv",
) -> tuple[UUID, str]:
    """Mint a pairing code + exchange it. Returns (camera_id, api_key)."""
    mint = (await client.post(
        f"/v1/patients/{patient_id}/cameras/pairing-code",
        headers=headers,
        json={"room": room, "icon_name": icon_name},
    )).json()
    pair = (await client.post(
        "/v1/cameras/pair", json={"pairing_code": mint["pairing_code"]},
    )).json()
    return UUID(pair["camera_id"]), pair["api_key"]


def _form_data(
    camera_id: UUID,
    *,
    motion_in_clip: str = "false",
    timestamp: str | None = None,
) -> dict[str, str]:
    return {
        "camera_id": str(camera_id),
        "motion_in_clip": motion_in_clip,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


def _dummy_mp4(size: int = 1024) -> bytes:
    """1 KB of zeros. The mocked provider never reads the bytes."""
    return b"\x00" * size


# ─── Happy path ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_inserts_analysis_row(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """The happy path: a paired camera ingests a 1KB MP4 and an
    analyses row is INSERTed with all fields populated. We don't
    test SSE / realtime here (analyses isn't in the publication).
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)
    captured_at = datetime.now(timezone.utc).replace(microsecond=0)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, motion_in_clip="false", timestamp=captured_at.isoformat()),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "gemini"
    assert body["analysis"] == "A person is walking in the living room."
    assert body["people_present"] is True
    assert body["people_count"] == 1
    # detector_fields in the wire response equals the LLM's detector_fields
    # (not the merged-with-people_count store value).
    assert body["detector_fields"]["fall_detected"] is False
    assert body["state"] == "open"
    assert body["no_people_count"] == 0  # people=true reset
    assert body["close_threshold"] == 6
    assert UUID(body["id"])  # round-trips as UUID

    # provider was called exactly once with the mp4 bytes and the
    # patient's 4 preset detectors.
    assert len(gemini_mock.calls) == 1
    assert gemini_mock.calls[0]["mp4_len"] == 1024
    assert {d.identifier for d in gemini_mock.calls[0]["detectors"]} == {
        "fall", "agitation", "wandering", "medication",
    }
    assert len(openai_mock.calls) == 0

    # DB-side: full analysis row populated with merged detector_fields.
    rows = (await session.execute(select(Analysis).where(Analysis.camera_id == cid))).scalars().all()
    assert len(rows) == 1
    a = rows[0]
    assert a.id == UUID(body["id"])
    assert a.camera_id == cid
    assert a.patient_id == UUID(pid)
    # caregiver_id denormalized from patient → camera. Pull the patient
    # row to compute the expected value.
    cam = await session.get(Camera, cid)
    assert cam is not None
    pat_row = await session.execute(
        text("SELECT caregiver_id FROM patients WHERE id = :pid"), {"pid": pid},
    )
    expected_caregiver_id = pat_row.scalar_one()
    assert a.caregiver_id == expected_caregiver_id
    assert a.provider == "gemini"
    assert a.description == "A person is walking in the living room."
    # detector_fields stored = LLM detector_fields + people_count merged in.
    assert a.detector_fields == {
        **gemini_mock.result.detector_fields,
        "people_count": 1,
    }
    assert a.people_present is True
    assert a.motion_in_clip is False
    assert a.state_after == "open"
    assert int(a.captured_at.timestamp()) == int(captured_at.timestamp())
    assert a.created_at is not None


# ─── Auth ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_api_key(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, _api_key = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": "Bearer wrong-token-not-a-real-api-key"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 401
    # No analysis inserted.
    rows = (await session.execute(select(Analysis))).scalars().all()
    assert rows == []
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_missing_bearer(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, _ = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 401
    rows = (await session.execute(select(Analysis))).scalars().all()
    assert rows == []
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_unpaired_camera(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """A camera that hasn't completed pairing (api_key_hash IS NULL)
    has no api_key the wall phone can present. The auth path can't
    accept any bearer for it. We verify by minting a code but never
    pairing — any bearer is 401.
    """
    headers, pid = await _setup_patient(client, auth_user)
    mint = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers, json={"room": "Living room", "icon_name": "tv"},
    )).json()
    cid = UUID(mint["camera_id"])

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": "Bearer any-token"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 401

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.api_key_hash is None  # confirms truly unpaired
    rows = (await session.execute(select(Analysis))).scalars().all()
    assert rows == []
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_form_camera_id_mismatch(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """The api-key resolves to camera A but the form claims camera B.
    The route must 403 even though auth itself passed.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid_a, api_key_a = await _pair_camera(client, headers, pid, room="A")
    cid_b, _api_key_b = await _pair_camera(client, headers, pid, room="B")

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key_a}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid_b),
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "camera_id_mismatch"
    rows = (await session.execute(select(Analysis))).scalars().all()
    assert rows == []
    # Camera A was resolved (auth ran) but provider wasn't called.
    assert len(gemini_mock.calls) == 0


# ─── Form validation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_rejects_missing_motion_in_clip(
    fast_bcrypt_app, client: AsyncClient, auth_user, gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data={"camera_id": str(cid), "timestamp": datetime.now(timezone.utc).isoformat()},
    )
    # FastAPI's Form(...) returns 422 for missing required field.
    assert r.status_code in (400, 422)
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_motion_value(
    fast_bcrypt_app, client: AsyncClient, auth_user, gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, motion_in_clip="maybe"),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == 'motion_in_clip must be "true" or "false"'
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_invalid_timestamp(
    fast_bcrypt_app, client: AsyncClient, auth_user, gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, timestamp="not-an-iso-timestamp"),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "timestamp must be ISO-8601"
    assert len(gemini_mock.calls) == 0


@pytest.mark.asyncio
async def test_ingest_rejects_empty_file(
    fast_bcrypt_app, client: AsyncClient, auth_user, gemini_mock, openai_mock,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", b"", "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "file is empty"
    assert len(gemini_mock.calls) == 0


# ─── Event promotion ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fall_promotes_to_event(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """A fall_detected=true with confidence ≥ 0.70 fires an events row
    with category='safety', detector_id pointing at the Fall preset,
    location = camera.room, narrative = description.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid, room="Living room")

    fall_fields = {**gemini_mock.result.detector_fields}
    fall_fields["fall_detected"] = True
    fall_fields["fall_confidence"] = 0.9
    fall_fields["fall_reasoning"] = "person dropped to the floor"
    gemini_mock.result = _default_result(
        description="The elder fell from the couch.",
        detector_fields=fall_fields,
    )

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, motion_in_clip="true"),
    )
    assert r.status_code == 200, r.text

    # Pull the events row, assert every field.
    events = (await session.execute(select(Event))).scalars().all()
    assert len(events) == 1
    ev = events[0]

    # Find the fall detector for this patient so we can compare detector_id.
    fall_det = (await session.execute(
        text(
            "SELECT id, category, source, icon_name, name "
            "FROM detectors WHERE patient_id = :pid AND identifier = 'fall'"
        ),
        {"pid": pid},
    )).one()

    assert ev.patient_id == UUID(pid)
    assert ev.detector_id == fall_det.id
    assert ev.camera_id == cid
    assert ev.category == "safety"
    assert ev.source == "preset"
    assert ev.icon_name == "figure.fall"
    assert ev.title == "Falls"
    assert ev.location == "Living room"
    assert ev.has_clip is False
    assert ev.narrative == "The elder fell from the couch."
    assert ev.opened is False
    assert ev.reviewed is False
    assert ev.review_choice is None
    assert ev.review_notes is None
    assert ev.reviewed_at is None
    assert ev.occurred_at is not None
    assert ev.created_at is not None


@pytest.mark.asyncio
async def test_low_confidence_fall_does_not_promote(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """fall_confidence=0.3 is below the 0.70 gate — analyses row is
    inserted but NO event lands.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    fall_fields = {**gemini_mock.result.detector_fields}
    fall_fields["fall_detected"] = True
    fall_fields["fall_confidence"] = 0.3
    gemini_mock.result = _default_result(detector_fields=fall_fields)

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 200, r.text

    analyses = (await session.execute(select(Analysis))).scalars().all()
    assert len(analyses) == 1
    events = (await session.execute(select(Event))).scalars().all()
    assert events == []


@pytest.mark.asyncio
async def test_fall_dedup_within_30s(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """Two fall-positive clips within 30s → only one event lands."""
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    fall_fields = {**gemini_mock.result.detector_fields}
    fall_fields["fall_detected"] = True
    fall_fields["fall_confidence"] = 0.9
    gemini_mock.result = _default_result(detector_fields=fall_fields)

    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    t1 = t0 + timedelta(seconds=10)

    r1 = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("a.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, timestamp=t0.isoformat()),
    )
    r2 = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("b.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, timestamp=t1.isoformat()),
    )
    assert r1.status_code == 200
    assert r2.status_code == 200

    analyses = (await session.execute(select(Analysis))).scalars().all()
    assert len(analyses) == 2
    events = (await session.execute(select(Event))).scalars().all()
    assert len(events) == 1


@pytest.mark.asyncio
async def test_fall_after_30s_promotes_second_event(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """Two fall-positive clips > 30s apart → two events."""
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    fall_fields = {**gemini_mock.result.detector_fields}
    fall_fields["fall_detected"] = True
    fall_fields["fall_confidence"] = 0.9
    gemini_mock.result = _default_result(detector_fields=fall_fields)

    t0 = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    t1 = t0 + timedelta(seconds=60)

    r1 = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("a.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, timestamp=t0.isoformat()),
    )
    r2 = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("b.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, timestamp=t1.isoformat()),
    )
    assert r1.status_code == 200
    assert r2.status_code == 200

    analyses = (await session.execute(select(Analysis))).scalars().all()
    assert len(analyses) == 2
    events = (await session.execute(select(Event))).scalars().all()
    assert len(events) == 2


# ─── State machine via the route ──────────────────────────────────


@pytest.mark.asyncio
async def test_state_machine_open_to_closed(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """Six consecutive ticks with people=false, motion=false → camera
    reaches state='closed', no_people_count=6, state_changed_at
    updated. The first 5 LLM calls return people_present=false (open
    state, so LLM runs); the 6th flips the state to closed.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    # Tweak the mock to return people_present=false so the counter rises.
    gemini_mock.result = _default_result(
        description="Empty room.", people_present=False, people_count=0,
    )

    pre = await session.get(Camera, cid)
    assert pre is not None
    initial_state_changed_at = pre.state_changed_at
    assert pre.state == "open"
    assert pre.no_people_count == 0

    for i in range(1, 7):
        r = await client.post(
            "/v1/clips/ingest",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (f"{i}.mp4", _dummy_mp4(), "video/mp4")},
            data=_form_data(cid, motion_in_clip="false"),
        )
        assert r.status_code == 200, r.text

    # Only the first 6 ticks ran the LLM (the 6th still ran because
    # state was open at the start of that tick).
    assert len(gemini_mock.calls) == 6

    session.expire_all()
    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    assert cam.no_people_count == 6
    assert cam.close_threshold == 6
    assert cam.state_changed_at is not None
    assert initial_state_changed_at is None or cam.state_changed_at > initial_state_changed_at


@pytest.mark.asyncio
async def test_state_machine_closed_skips_llm(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """A closed camera ingesting a clip with motion=false must
    short-circuit: provider='skipped', LLM NOT called, synthetic
    analyses row inserted.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    # Force closed state.
    await session.execute(
        text(
            "UPDATE cameras SET state='closed', no_people_count=6 WHERE id=:cid"
        ),
        {"cid": str(cid)},
    )
    await session.commit()

    captured_at = datetime.now(timezone.utc).replace(microsecond=0)
    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, motion_in_clip="false", timestamp=captured_at.isoformat()),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "skipped"
    assert body["analysis"] == "(closed — LLM skipped)"
    assert body["people_present"] is False
    assert body["people_count"] == 0
    assert body["state"] == "closed"
    assert body["no_people_count"] == 6
    assert body["close_threshold"] == 6
    assert body["detector_fields"] == {
        "people_present": False,
        "skipped_reason": "camera_closed",
    }

    # Provider was NOT called.
    assert len(gemini_mock.calls) == 0
    assert len(openai_mock.calls) == 0

    # Analyses row exists with provider='skipped'.
    rows = (await session.execute(select(Analysis).where(Analysis.camera_id == cid))).scalars().all()
    assert len(rows) == 1
    a = rows[0]
    assert a.provider == "skipped"
    assert a.description == "(closed — LLM skipped)"
    assert a.people_present is False
    assert a.motion_in_clip is False
    assert a.state_after == "closed"
    # detector_fields = synthetic + people_count merged in.
    assert a.detector_fields == {
        "people_present": False,
        "skipped_reason": "camera_closed",
        "people_count": 0,
    }


@pytest.mark.asyncio
async def test_state_machine_closed_to_open_on_motion(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """A closed camera + motion=true must NOT short-circuit. The LLM
    IS called (someone might have walked back in), and on a
    people_present=true result the state flips to open.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    await session.execute(
        text("UPDATE cameras SET state='closed', no_people_count=6 WHERE id=:cid"),
        {"cid": str(cid)},
    )
    await session.commit()

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid, motion_in_clip="true"),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "gemini"
    assert body["state"] == "open"
    assert body["no_people_count"] == 0
    assert body["close_threshold"] == 6

    # LLM was called once.
    assert len(gemini_mock.calls) == 1
    assert len(openai_mock.calls) == 0


# ─── Provider fallback ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_provider_fallback(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """Gemini raises → OpenAI runs and the analysis row records
    provider='openai'.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    gemini_mock.raises = RuntimeError("gemini went down")
    openai_mock.raises = None
    openai_mock.result = _default_result(description="OpenAI saw a person.")

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert body["analysis"] == "OpenAI saw a person."

    assert len(gemini_mock.calls) == 1
    assert len(openai_mock.calls) == 1

    rows = (await session.execute(select(Analysis).where(Analysis.camera_id == cid))).scalars().all()
    assert len(rows) == 1
    assert rows[0].provider == "openai"
    assert rows[0].description == "OpenAI saw a person."


@pytest.mark.asyncio
async def test_both_providers_fail(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
    gemini_mock, openai_mock,
) -> None:
    """Both Gemini and OpenAI raise → 502, no analyses row inserted."""
    headers, pid = await _setup_patient(client, auth_user)
    cid, api_key = await _pair_camera(client, headers, pid)

    gemini_mock.raises = RuntimeError("gemini down")
    openai_mock.raises = RuntimeError("openai down")

    r = await client.post(
        "/v1/clips/ingest",
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("clip.mp4", _dummy_mp4(), "video/mp4")},
        data=_form_data(cid),
    )
    assert r.status_code == 502
    assert r.json()["detail"] == "both_providers_failed"

    assert len(gemini_mock.calls) == 1
    assert len(openai_mock.calls) == 1
    rows = (await session.execute(select(Analysis).where(Analysis.camera_id == cid))).scalars().all()
    assert rows == []
