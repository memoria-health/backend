"""The exact JSON the Home sleep card decodes.

The card is the only surface sleep has, and it is fed by one endpoint
whose shape nothing pinned. Two mismatches shipped before this file
existed, both silent:

  * the wake-up windows are keyed `from`/`to`; the journal renderer read
    `start`/`end` inside a try/except, so every gap raised, was
    swallowed, and a night with three wake-ups drew as unbroken sleep.
  * those timestamps went out with NO UTC offset, because the
    summarizer works in naive UTC and `.isoformat()` does not add one.
    A decoder that requires an offset rejects the night outright; one
    that does not shifts it by the phone's timezone.

Neither would fail a test that only checked HTTP 200.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def _seed_night(session: AsyncSession, patient_id, night: date) -> None:
    onset = datetime(2026, 8, 23, 21, 59, tzinfo=timezone.utc)
    wake = onset + timedelta(hours=6)
    windows = [{
        "from": (onset + timedelta(hours=1)).isoformat(),
        "to": (onset + timedelta(hours=1, minutes=30)).isoformat(),
        "duration_minutes": 30,
    }]
    await session.execute(text("""
        insert into sleep_nights (patient_id, night_date, bedtime,
            sleep_onset, wake_time, duration_minutes, disturbances,
            out_of_bed_events, flags)
        values (:p, :n, :bt, :so, :wk, 306, 1, cast(:ob as jsonb), '[]'::jsonb)
        on conflict (patient_id, night_date) do update set
            sleep_onset = excluded.sleep_onset
    """), {"p": patient_id, "n": night, "bt": onset - timedelta(hours=2),
           "so": onset, "wk": wake, "ob": json.dumps(windows)})
    await session.commit()


@pytest.mark.asyncio
async def test_the_card_gets_every_field_it_renders(
    client, auth_user, session: AsyncSession,
) -> None:
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    pid = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "A", "preferred_name": "A", "relation": "parent",
    })).json()["id"]
    await _seed_night(session, pid, date.today() - timedelta(days=1))

    r = await client.get(f"/v1/patients/{pid}/sleep-nights", headers=headers)
    assert r.status_code == 200, r.text
    nights = r.json()["nights"]
    assert nights, "the endpoint returned no nights for a seeded patient"
    n = nights[-1]

    # Every key SleepNight.swift names. A rename here is a card that
    # silently stops decoding, which looks identical to "no data yet".
    for key in ("night", "bedtime", "sleep_onset", "wake_time",
                "duration_minutes", "disturbances", "out_of_bed_events",
                "flags"):
        assert key in n, f"{key} missing — SleepCard cannot render without it"

    assert n["duration_minutes"] == 306
    assert n["disturbances"] == 1


@pytest.mark.asyncio
async def test_wake_up_windows_keep_their_keys_and_their_offset(
    client, auth_user, session: AsyncSession,
) -> None:
    """`from`/`to`, and an explicit UTC offset on both."""
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    pid = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "B", "preferred_name": "B", "relation": "parent",
    })).json()["id"]
    await _seed_night(session, pid, date.today() - timedelta(days=1))

    r = await client.get(f"/v1/patients/{pid}/sleep-nights", headers=headers)
    windows = r.json()["nights"][-1]["out_of_bed_events"]
    assert windows, "a night with disturbances=1 returned no windows"
    w = windows[0]
    assert "from" in w and "to" in w, (
        "the renderer and the card both read from/to; start/end is the "
        "pre-2026-08-03 spelling and reading it drew nights as unbroken")
    for key in ("from", "to"):
        parsed = datetime.fromisoformat(str(w[key]))
        assert parsed.tzinfo is not None, (
            f"{key} has no UTC offset — the phone would place this "
            f"wake-up in its own timezone")


@pytest.mark.asyncio
async def test_a_night_that_could_not_be_measured_says_why(
    client, auth_user, session: AsyncSession,
) -> None:
    """Nulls plus a flag, never zeroes.

    `duration_minutes = 0` reads as "they did not sleep". The truth is
    "we could not tell", and the card only knows the difference because
    the flag survives the round trip.
    """
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    pid = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "C", "preferred_name": "C", "relation": "parent",
    })).json()["id"]
    await session.execute(text("""
        insert into sleep_nights (patient_id, night_date, disturbances,
            out_of_bed_events, flags)
        values (:p, :n, 0, '[]'::jsonb, '["no_confirmed_bed_entry"]'::jsonb)
    """), {"p": pid, "n": date.today() - timedelta(days=1)})
    await session.commit()

    n = (await client.get(f"/v1/patients/{pid}/sleep-nights",
                          headers=headers)).json()["nights"][-1]
    assert n["duration_minutes"] is None
    assert n["sleep_onset"] is None
    assert n["flags"] == ["no_confirmed_bed_entry"]
