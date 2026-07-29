"""Integration: P4 decision layer — windowed-count state entry and
per-detector cooldown, both flag-gated (`detection_v2_decision_enabled`)
and parity-exact when off or unconfigured.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.detector import Detector
from src.services.detection.analyses_store import add_analysis
from src.services.detection.event_promotion import maybe_promote_analysis
from test.integration.identity_helpers import _setup_patient_with_camera

_V2 = SimpleNamespace(detection_v2_decision_enabled=True)


async def _add_detector(
    session: AsyncSession, patient_id, ident: str, *,
    is_sustained: bool = True, eventing: dict | None = None,
    cooldown_seconds: int | None = None, gap: int = 300,
    category: str = "routine", subject_scope: str | None = None,
) -> Detector:
    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(), patient_id=patient_id, name=ident, icon_name="x",
        description=ident, category=category, source="custom",
        subject_scope=subject_scope,
        alert_trigger="leading", is_sustained=is_sustained, when_scope="ALWAYS",
        sensitivity="standard", is_on=True, created_at=now, updated_at=now,
        identifier=ident, prompt_fragment=None, schema_json=None,
        always_on=False, episode_gap_seconds=gap if is_sustained else None,
        eventing=eventing, cooldown_seconds=cooldown_seconds,
    )
    session.add(det)
    await session.flush()
    return det


def _fields(ident: str, conf: float = 0.9) -> dict:
    return {
        f"{ident}_detected": True,
        f"{ident}_confidence": conf,
        f"{ident}_reasoning": "seen",
    }


async def _seed_tick(session, camera_id, ident, at) -> None:
    await add_analysis(
        session, camera_id=camera_id, provider="gemini",
        description="seed", detector_fields=_fields(ident),
        people_present=True, people_count=1, motion_in_clip=True,
        state_after="open", captured_at=at,
    )
    await session.commit()


@pytest.mark.asyncio
async def test_windowed_entry_waits_for_m_ticks_flag_off_is_on_sight(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_tvwatch"
    await _add_detector(
        session, patient_id, ident,
        eventing={"entry": {"m": 3, "t_seconds": 600}},
    )
    await session.commit()
    now = datetime.now(timezone.utc)

    # One prior qualifying tick + the current one = 2 < M=3 ⇒ held.
    await _seed_tick(session, camera_id, ident, now - timedelta(seconds=60))
    held = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="tick 2",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert held.fired == []

    # A second prior tick ⇒ 3 ≥ M ⇒ the state enters and fires.
    await _seed_tick(session, camera_id, ident, now - timedelta(seconds=30))
    fired = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now + timedelta(seconds=1), description="tick 3",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(fired.fired) == 1

    # Parity control: flag OFF (settings=None) fires on sight even
    # with the same eventing config on a fresh detector.
    ident2 = "custom_tvwatch2"
    await _add_detector(
        session, patient_id, ident2,
        eventing={"entry": {"m": 3, "t_seconds": 600}},
    )
    await session.commit()
    parity = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="first sight",
        detector_fields=_fields(ident2),
    )
    await session.commit()
    assert len(parity.fired) == 1


@pytest.mark.asyncio
async def test_null_eventing_state_still_enters_on_sight_flag_on(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE parity pin: flag on + NULL eventing must equal today."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_nullcfg"
    await _add_detector(session, patient_id, ident, eventing=None)
    await session.commit()
    r = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc), description="x",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(r.fired) == 1


@pytest.mark.asyncio
async def test_per_detector_cooldown_replaces_global_dedup_flag_on(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_doorslam"
    await _add_detector(
        session, patient_id, ident, is_sustained=False,
        cooldown_seconds=5,
    )
    await session.commit()
    t0 = datetime.now(timezone.utc)

    first = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0, description="one",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(first.fired) == 1

    # 10 s later: inside the legacy 30 s window, outside the row's 5 s.
    again = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0 + timedelta(seconds=10), description="two",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(again.fired) == 1, "row cooldown (5s) must own re-fire spacing"

    # Same shape, flag OFF: 10 s later is deduped by the global 30 s.
    ident2 = "custom_doorslam2"
    await _add_detector(
        session, patient_id, ident2, is_sustained=False,
        cooldown_seconds=5,
    )
    await session.commit()
    a = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0, description="one",
        detector_fields=_fields(ident2),
    )
    await session.commit()
    assert len(a.fired) == 1
    b = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0 + timedelta(seconds=10), description="two",
        detector_fields=_fields(ident2),
    )
    await session.commit()
    assert b.fired == [] and ident2 in b.skipped_for_dedup


@pytest.mark.asyncio
async def test_saturated_window_is_ongoing_never_refires(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE anti-spam arm (P4 review finding 4): once the window holds
    ≥M qualifying ticks, further qualifying ticks are ONGOING — a long
    episode must not refire even after the cooldown/dedup window has
    lapsed."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_longwatch"
    await _add_detector(
        session, patient_id, ident,
        eventing={"entry": {"m": 3, "t_seconds": 600}},
    )
    await session.commit()
    now = datetime.now(timezone.utc)

    for age in (120, 90, 60):   # 3 qualifying ticks: window saturated
        await _seed_tick(session, camera_id, ident, now - timedelta(seconds=age))

    # Crossing already happened earlier in the episode; this tick (and
    # any later one while the window stays saturated) is ongoing —
    # even though NO event exists within any dedup window.
    later = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now + timedelta(seconds=45), description="still going",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert later.fired == []
    assert ident in later.skipped_ongoing


@pytest.mark.asyncio
async def test_repeat_inside_row_cooldown_is_still_deduped(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A v2 bug disabling dedup entirely must not pass: a repeat
    INSIDE the row's cooldown is squashed (P4 review test gap)."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_fastfire"
    await _add_detector(
        session, patient_id, ident, is_sustained=False,
        cooldown_seconds=20,
    )
    await session.commit()
    t0 = datetime.now(timezone.utc)
    first = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0, description="one",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(first.fired) == 1
    inside = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=t0 + timedelta(seconds=10), description="two",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert inside.fired == [] and ident in inside.skipped_for_dedup


# ── Criticality split (2026-07-28): routine patient detectors fail-quiet,
#    safety patient detectors fail-loud. Applies to CUSTOMS via the same
#    category × subject_scope columns the compiler infers — no preset
#    special-casing anywhere.

@pytest.mark.asyncio
async def test_routine_patient_custom_fails_quiet_without_confirmed_patient(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_tvhabit"
    await _add_detector(
        session, patient_id, ident, is_sustained=False,
        category="routine", subject_scope="patient",
    )
    await session.commit()
    now = datetime.now(timezone.utc)

    # Identity did NOT confirm the patient (a visitor may be in frame).
    res = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="visitor tick",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert res.fired == []

    # Patient positively confirmed → promotes normally.
    f = _fields(ident)
    f["identity_patient_in_clip"] = True
    res2 = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now + timedelta(seconds=30), description="patient tick",
        detector_fields=f, settings=_V2,
    )
    await session.commit()
    assert len(res2.fired) == 1


@pytest.mark.asyncio
async def test_safety_patient_custom_fails_loud_without_confirmation(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    ident = "custom_collapse"
    await _add_detector(
        session, patient_id, ident, is_sustained=False,
        category="safety", subject_scope="patient",
    )
    await session.commit()

    # No identity confirmation at all — a safety detector must still fire
    # (a fall is never suppressed for want of a visible face).
    res = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc), description="unknown person down",
        detector_fields=_fields(ident), settings=_V2,
    )
    await session.commit()
    assert len(res.fired) == 1
