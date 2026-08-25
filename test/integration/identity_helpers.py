"""Shared provisioning helpers for identity-adjacent integration tests.

Promoted 2026-07-27 from the deleted test_reid_safety_invariant.py
(which tested the legacy in-process brain and died with it) — these
helpers provision DB state, not brain behavior, and multiple live
tests depend on them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.identity.embeddings import embedding_to_bytes
from test.conftest import mint_test_jwt, seed_camera

_PATIENT_BODY = {
    "full_name": "Maggie",
    "preferred_name": "Maggie",
    "relation": "parent",
}


def _ready_embedding(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return embedding_to_bytes(v)


async def _setup_patient_with_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> tuple[UUID, UUID]:
    """Provision a caregiver + patient + a camera under that patient.
    Returns (patient_id, camera_id)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT_BODY,
    )).json()
    patient_id = UUID(p["id"])
    camera_id = await seed_camera(
        session, patient_id=patient_id, caregiver_id=uid,
        room="Test Room", icon_name="bed.double", online=True,
        api_key_hash="test_hash", state="open", no_people_count=0,
        close_threshold=6, state_changed_at=datetime.now(timezone.utc),
        disabled=False, desired_state="connected",
    )
    # seed_camera writes the camera->patient link too — coverage is what
    # the account view and the identity gate read, not cameras.patient_id.
    return patient_id, camera_id


async def _share_camera_with_another_patient(session, camera_id, patient_id) -> UUID:
    """Make a camera cover a SECOND patient.

    The fail-quiet identity gate arms only where a camera watches more
    than one person — that is the only case where an unattributed routine
    event could land on the wrong record.
    """
    other = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.patients "
            "  (id, caregiver_id, full_name, preferred_name, relation) "
            "SELECT :o, caregiver_id, 'Housemate', 'Housemate', 'other' "
            "  FROM public.patients WHERE id = :p"
        ),
        {"o": other, "p": patient_id},
    )
    await session.execute(
        text(
            "INSERT INTO public.camera_patients (camera_id, patient_id) "
            "VALUES (:c, :o) ON CONFLICT DO NOTHING"
        ),
        {"c": camera_id, "o": other},
    )
    await session.commit()
    return other
