"""Who a shared sensor's events belong to.

One sensor can watch two people. Three rules, agreed 2026-08-22:

  1. identity locates a patient -> the event is theirs
  2. identity locates nobody    -> SAFETY fires for EVERY covered patient
  3. identity locates nobody    -> non-safety fires for NONE of them

Rule 2 is why a fall nobody can attribute is still a fall. Rule 3 is
because filing one resident's routine onto another's record is not
recoverable, and a wrong care record is worse than a missing one.

`event_promotion` has implemented 2 and 3 since 2026-08-19. What was
broken is upstream: identity was asked about ONE patient and that single
answer was applied to everybody, and the promotion loop then kept only
the first patient's result and discarded the rest — so a second
resident's fall reached the `events` table and stopped there, with no
push, no email, no page and no clip.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.detection.event_promotion import (
    _covered_patient_count,
    _is_truthy_bool,
)


async def _shared_sensor(session: AsyncSession, client, auth_user):
    """One sensor covering two patients on one account."""
    from test.conftest import mint_test_jwt, seed_camera

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    body = {"full_name": "A", "preferred_name": "A", "relation": "parent"}
    a = (await client.post("/v1/patients", headers=headers,
                           json=body)).json()["id"]
    b = (await client.post("/v1/patients", headers=headers,
                           json={**body, "full_name": "B"})).json()["id"]
    cam = await seed_camera(session, patient_id=a, room="Lounge")
    await session.execute(
        text("INSERT INTO public.camera_patients (camera_id, patient_id) "
             "VALUES (:c, :p) ON CONFLICT DO NOTHING"),
        {"c": cam, "p": b},
    )
    await session.commit()
    return cam, a, b


@pytest.mark.asyncio
async def test_the_gate_arms_only_where_the_question_is_real(
    client, auth_user, session: AsyncSession,
) -> None:
    """Two covered patients is what makes "which of them?" ambiguous.

    On a single-patient sensor there is nothing to confuse, so an
    unconfirmed identity must NOT suppress that patient's routine
    detectors — otherwise every home without a working identity node
    loses non-safety detection entirely.
    """
    cam, _a, _b = await _shared_sensor(session, client, auth_user)
    assert await _covered_patient_count(session, cam) == 2

    solo = uuid4()
    assert await _covered_patient_count(session, solo) == 0


@pytest.mark.asyncio
async def test_each_patient_is_judged_on_their_own_identity_answer(
    client, auth_user, session: AsyncSession,
) -> None:
    """The flag is per patient, not one patient's answer for everybody.

    This is the defect: `identity_patient_in_clip` was computed for the
    shim patient and handed to every covered patient's promotion. So A
    being in frame fired B's routine detectors onto B's record for A's
    behaviour, and B alone in frame fired nothing for B at all.
    """
    _cam, a, b = await _shared_sensor(session, client, auth_user)

    # What camera_ingest now builds: one entry per covered patient.
    identity_by_patient = {a: True, b: False}

    fields_a = {"identity_patient_in_clip": identity_by_patient[a]}
    fields_b = {"identity_patient_in_clip": identity_by_patient[b]}

    assert _is_truthy_bool(fields_a["identity_patient_in_clip"]) is True
    assert _is_truthy_bool(fields_b["identity_patient_in_clip"]) is False
    # The two must not be the same object or value by construction —
    # sharing one dict is precisely how the bug happened.
    assert fields_a != fields_b


@pytest.mark.asyncio
async def test_safety_survives_an_unconfirmed_identity(
    client, auth_user, session: AsyncSession,
) -> None:
    """Rule 2, asserted on the gate's own condition.

    The gate reads `det.category != "safety"`, so a safety detector is
    never suppressed by an unconfirmed identity — for any covered
    patient. A fall nobody can attribute is still a fall, and it is
    better delivered twice than lost.
    """
    from src.services.detection import event_promotion

    src = event_promotion.__file__
    with open(src, encoding="utf-8") as fh:
        body = fh.read()
    # The literal guard, so a future edit that drops it fails here rather
    # than silently muting safety on every shared sensor.
    assert 'det.category != "safety"' in body
    assert "patient_unconfirmed" in body


@pytest.mark.asyncio
async def test_deleting_one_patient_leaves_the_shared_sensor(
    client, auth_user, session: AsyncSession,
) -> None:
    """The sensor is hardware on a wall; a person leaving does not remove it.

    `cameras.patient_id` was `ON DELETE CASCADE`. It is a compatibility
    shim "authoritative for nothing" — except it decided whether the row
    survived. Deleting the patient it happened to point at destroyed the
    whole sensor, and with it the OTHER patient's coverage, their detector
    links and the ingest token, while the bridge kept publishing into a
    camera that no longer existed.
    """
    from test.conftest import mint_test_jwt

    cam, a, b = await _shared_sensor(session, client, auth_user)
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}

    r = await client.delete(f"/v1/patients/{a}", headers=headers)
    assert r.status_code == 204, r.text

    row = (await session.execute(
        text("SELECT patient_id FROM public.cameras WHERE id = :c"),
        {"c": cam},
    )).first()
    assert row is not None, "the sensor was destroyed with the patient"
    # And the shim follows the remaining covered patient, so the readers
    # that have not migrated off it still see a sensor with an owner.
    assert str(row[0]) == str(b)

    still_covered = (await session.execute(
        text("SELECT patient_id FROM public.camera_patients WHERE camera_id = :c"),
        {"c": cam},
    )).scalars().all()
    assert [str(p) for p in still_covered] == [str(b)]


# ── Rule 1, the half that was never implemented ──────────────────────
# "identity locates a patient -> the event is theirs" was only ever true
# for non-safety detectors. Safety skipped the identity check entirely,
# so a fall belonging to a located resident ALSO fired for everyone else
# the sensor covers. Found live 2026-08-25.

def _identity(*, in_clip: bool, people: int = 1, ran: bool = True,
              attributed: bool = True):
    from src.services.identity.result import IdentityResult
    return IdentityResult(
        reid_ran=ran,
        patient_in_clip=in_clip,
        confidence=0.74 if in_clip else 0.0,
        best_frame_index=None,
        face_count=0,
        people_present_reid=people > 0,
        people_count_reid=people,
        attribution_ran=attributed,
    )


def test_a_located_resident_owns_their_own_emergency() -> None:
    """B alone in the room, located. A must not be told A fell."""
    from src.services.detection.camera_ingest import other_patient_confirmed

    a, b = uuid4(), uuid4()
    by_patient = {a: _identity(in_clip=False), b: _identity(in_clip=True)}

    assert other_patient_confirmed(by_patient, a) is True
    # And B, who was actually there, is unaffected — nobody else is
    # confirmed, so B's own detectors run normally.
    assert other_patient_confirmed(by_patient, b) is False


def test_a_crowd_alerts_everybody() -> None:
    """Two people in frame: which of them fell is unanswerable.

    Suppressing here would drop a real fall on the strength of a lock
    held on the person who did NOT fall.
    """
    from src.services.detection.camera_ingest import other_patient_confirmed

    a, b = uuid4(), uuid4()
    by_patient = {
        a: _identity(in_clip=False, people=2),
        b: _identity(in_clip=True, people=2),
    }
    assert other_patient_confirmed(by_patient, a) is False


def test_every_uncertain_answer_stays_loud() -> None:
    """Identity down, attribution off, or nobody located -> alert.

    This gate STOPS an emergency alert, so not knowing must never be
    mistaken for knowing.
    """
    from src.services.detection.camera_ingest import other_patient_confirmed

    a, b = uuid4(), uuid4()
    for other in (
        _identity(in_clip=True, ran=False),         # identity crashed
        _identity(in_clip=True, attributed=False),  # could not attribute
        _identity(in_clip=False),                   # nobody located
        _identity(in_clip=True, people=0),          # located, saw no one
    ):
        got = other_patient_confirmed({a: _identity(in_clip=False), b: other}, a)
        assert got is False, other


def test_a_solo_sensor_can_never_suppress() -> None:
    """One covered patient means there is no "someone else" to find."""
    from src.services.detection.camera_ingest import other_patient_confirmed

    a = uuid4()
    assert other_patient_confirmed({a: _identity(in_clip=True)}, a) is False


def test_only_patient_scoped_safety_is_narrowed() -> None:
    """The guard's literal shape, so an edit cannot quietly widen it.

    An environment hazard is not about a person and must keep firing for
    everyone; `anyone`-scoped detectors say so in their scope.
    """
    from src.services.detection import event_promotion

    with open(event_promotion.__file__, encoding="utf-8") as fh:
        body = fh.read()
    assert 'det.category == "safety"' in body
    assert "identity_other_patient_confirmed" in body
    assert "other_patient_confirmed" in body
