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
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.identity.embeddings import embedding_to_bytes
from test.conftest import mint_test_jwt

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
    from sqlalchemy import insert

    from src.db.camera import Camera
    camera_id = uuid4()
    await session.execute(
        insert(Camera).values(
            id=camera_id,
            patient_id=patient_id,
            display_id=1,
            room="Test Room",
            icon_name="bed.double",
            online=True,
            api_key_hash="test_hash",
            livekit_identity=f"cam_{camera_id.hex[:8]}",
            state="open",
            no_people_count=0,
            close_threshold=6,
            state_changed_at=datetime.now(timezone.utc),
            disabled=False,
            desired_state="connected",
            created_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return patient_id, camera_id


async def _enroll_face(session, patient_id) -> None:
    """Give a home a usable face enrollment.

    The fail-quiet identity gate only ARMS for a home that can actually
    identify someone (2026-08-19). Tests asserting that an unconfirmed
    person is suppressed must therefore enroll first — otherwise they are
    asserting the pre-fix behaviour, where an unenrolled household had
    every routine detector silenced forever by a confirmation that could
    never arrive.
    """
    from sqlalchemy import text
    await session.execute(
        text(
            "INSERT INTO public.patient_face_embeddings "
            "  (patient_id, embedding, det_score, source, status) "
            "VALUES (:p, :e, 0.99, 'selfie', 'ready')"
        ),
        {"p": patient_id, "e": b"\x00" * 2048},
    )
    await session.commit()
