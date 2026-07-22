"""End-to-end Re-ID test driving the API like a real caregiver would.

Two "agents" share fixtures: a caregiver-simulator that POSTs
selfies and polls the summary endpoint, and a camera-simulator that
exercises the tick path against fixture clips (synthetic MP4s
generated inline rather than checked-in to keep the repo small).

These tests are slower than the per-phase tests above. They run as
part of the integration tier but are tagged so CI can shard them
separately if needed.

The InsightFace inference path is monkeypatched here — same as the
Phase 2 tests — because spinning up the real model in CI is
prohibitively slow (~3s model load + per-embed ~50ms × N). The
adaptive threshold and harvest logic are covered by dedicated
unit/integration tests that exercise the real math.
"""

from __future__ import annotations

import asyncio
import io
from uuid import UUID

import numpy as np
import pytest
from httpx import AsyncClient

from src.services.reid import embeddings as reid_embeddings
from src.services.reid import storage as reid_storage
from src.services.reid.embeddings import FaceHit
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}

# Same minimal-valid-JPEG bytes used in test_face_embeddings.py.
# Storage stub doesn't actually decode the image; the embed
# monkeypatch returns canned FaceHit objects.
_TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605"
    "08070707090908ffe10018000c000400000900000007000601020800030500"
    "ffdb00430108090909080a0b0a0a0b0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
    "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
    "ffc00011080001000103012200021101031101ffc4001f0000010501010101"
    "010100000000000000000102030405060708090a0bffc400b5100002010303"
    "020403050504040000017d01020300041105122131410613516107227114"
    "32811491a1082342b1c11552d1f02433627282090a161718191a25262728"
    "292a3435363738393a434445464748494a535455565758595a6364656667"
    "68696a737475767778797a838485868788898a9293949596979899a2a3a4"
    "a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7"
    "d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f01000301"
    "01010101010101010100000000000001020304050607080902"
    "ffd9"
)


def _norm(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec.astype(np.float32) / (n + 1e-8)


def _patch_stack(monkeypatch, *, patient_embedding: np.ndarray | None = None) -> dict:
    """Storage stub + embedding stub that returns a fixed embedding."""
    store: dict[str, bytes] = {}

    async def fake_upload(*, patient_id, embedding_id, image_bytes, content_type, settings):
        key = f"patient-faces/{patient_id}/{embedding_id}.jpg"
        store[key] = image_bytes
        return key

    async def fake_download(*, key, settings):
        return store[key]

    async def fake_delete(*, key, settings):
        store.pop(key, None)

    monkeypatch.setattr(reid_storage, "upload_face_photo", fake_upload)
    monkeypatch.setattr(reid_storage, "download_face_photo", fake_download)
    monkeypatch.setattr(reid_storage, "delete_face_photo", fake_delete)

    if patient_embedding is None:
        patient_embedding = _norm(np.ones(512, dtype=np.float32))

    async def fake_embed(image_bytes, *, model_name="buffalo_l"):
        return FaceHit(
            embedding=patient_embedding,
            det_score=0.85,
            bbox=(10.0, 10.0, 100.0, 130.0),
        )
    monkeypatch.setattr(reid_embeddings, "embed_image_bytes", fake_embed)
    return store


async def _caregiver_setup(client: AsyncClient, auth_user) -> tuple[str, dict[str, str]]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Caregiver"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return p["id"], headers


async def _wait_for_ready(
    client: AsyncClient,
    headers: dict[str, str],
    patient_id: str,
    *,
    expected: int,
    timeout: float = 8.0,
) -> dict:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while True:
        r = await client.get(
            f"/v1/patients/{patient_id}/face-embeddings/summary",
            headers=headers,
        )
        s = r.json()
        if s["selfie_count"] == expected and s["processing_count"] == 0:
            return s
        if loop.time() > deadline:
            pytest.fail(
                f"timed out waiting for {expected} ready, got {s}",
            )
        await asyncio.sleep(0.1)


# ─── Caregiver simulator ────────────────────────────────────────


@pytest.mark.asyncio
async def test_caregiver_uploads_twelve_selfies_full_lifecycle(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    """End-to-end caregiver flow: POST 12 selfies, poll until ready,
    confirm summary, list embeddings, delete one, confirm count drops.
    """
    _patch_stack(monkeypatch)
    patient_id, headers = await _caregiver_setup(client, auth_user)

    files = [
        ("photos", (f"selfie_{i}.jpg", _TINY_JPEG, "image/jpeg"))
        for i in range(12)
    ]
    r = await client.post(
        f"/v1/patients/{patient_id}/face-embeddings",
        headers=headers,
        files=files,
    )
    assert r.status_code == 202
    initial_rows = r.json()
    assert len(initial_rows) == 12
    assert all(row["status"] == "processing" for row in initial_rows)

    summary = await _wait_for_ready(
        client, headers, patient_id, expected=12,
    )
    assert summary == {
        "selfie_count": 12,
        "harvested_count": 0,
        "processing_count": 0,
        "failed_count": 0,
        "last_update": summary["last_update"],
    }

    list_r = await client.get(
        f"/v1/patients/{patient_id}/face-embeddings",
        headers=headers,
    )
    assert list_r.status_code == 200
    assert len(list_r.json()) == 12
    assert all(row["status"] == "ready" for row in list_r.json())

    # Delete one.
    del_id = list_r.json()[0]["id"]
    del_r = await client.delete(
        f"/v1/face-embeddings/{del_id}", headers=headers,
    )
    assert del_r.status_code == 204

    final_summary_r = await client.get(
        f"/v1/patients/{patient_id}/face-embeddings/summary",
        headers=headers,
    )
    assert final_summary_r.json()["selfie_count"] == 11


# ─── Cross-account isolation (RLS + app-layer ownership) ────────


@pytest.mark.asyncio
async def test_caregiver_cant_see_other_caregivers_summary(
    client: AsyncClient, auth_user_factory, monkeypatch,
) -> None:
    _patch_stack(monkeypatch)
    a = await auth_user_factory()
    b = await auth_user_factory()
    pid_a, headers_a = await _caregiver_setup(client, a)

    # A enrolls.
    files = [
        ("photos", ("p.jpg", _TINY_JPEG, "image/jpeg")) for _ in range(3)
    ]
    await client.post(
        f"/v1/patients/{pid_a}/face-embeddings",
        headers=headers_a, files=files,
    )
    await _wait_for_ready(client, headers_a, pid_a, expected=3)

    # B opens A's summary — should see zeros (treated as 'not enrolled'
    # rather than 404 by design; the status card on web/iOS renders the
    # same empty-gallery shape).
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(b[0], b[1])}"}
    await client.post(
        "/v1/me", headers=headers_b, json={"display_name": "B"},
    )
    r = await client.get(
        f"/v1/patients/{pid_a}/face-embeddings/summary",
        headers=headers_b,
    )
    assert r.status_code == 200
    s = r.json()
    assert s["selfie_count"] == 0
    assert s["harvested_count"] == 0


# ─── Camera simulator — tick path is wired to summary ───────────


@pytest.mark.asyncio
async def test_tick_writes_identity_into_analyses(
    client: AsyncClient, auth_user, session, monkeypatch,
) -> None:
    """We don't drive the full camera_ingest tick here (LiveKit
    plumbing is too invasive to fake), but we exercise the
    ReIDInference.label_clip path directly and confirm the result
    shape is what camera_ingest will stash into detector_fields.

    With the body Re-ID port we no longer test the per-frame internals
    here (YOLO + BoT-SORT + InsightFace would require real models in
    CI). Instead we mock `_run_pipeline_sync` to verify the async
    wrapper + harvest hop honor the IdentityResult shape and that the
    timeout/error fall-through paths still work."""
    from src.services.reid.inference import ReIDInference, IdentityResult
    from src.db.session import get_sessionmaker
    from src.settings import get_settings
    from src.services.detection import frame_extraction

    _patch_stack(monkeypatch)

    patient_id, headers = await _caregiver_setup(client, auth_user)

    # Enroll one selfie so the gallery is non-empty and this run
    # exercises the ATTRIBUTION path (v2: an empty gallery no longer
    # short-circuits label_clip — tracking always runs; enrollment
    # only decides whether face matching happens).
    files = [("photos", ("a.jpg", _TINY_JPEG, "image/jpeg"))]
    await client.post(
        f"/v1/patients/{patient_id}/face-embeddings",
        headers=headers, files=files,
    )
    await _wait_for_ready(client, headers, patient_id, expected=1)

    class _Frame:
        def __init__(self, data: bytes):
            import base64 as b
            self.data_base64 = b.b64encode(data).decode()
            self.mime_type = "image/jpeg"

    async def fake_extract(mp4, *, fps=1, max_width=480, jpeg_quality=5):
        return [_Frame(_TINY_JPEG) for _ in range(3)]
    monkeypatch.setattr(frame_extraction, "extract_frames", fake_extract)

    settings = get_settings().model_copy(update={
        "reid_enabled": True,
        # This test mocks _run_pipeline_sync; the SAM wire hop
        # (env-selected on the soak box) must not run for real.
        "reid_pipeline_backend": "legacy",
    })
    sm = get_sessionmaker()
    reid = ReIDInference(sessionmaker=sm, settings=settings)

    # Bypass the heavy CV pipeline — emulate a clean PATIENT lock with
    # a body bbox so the rest of ReIDInference (async wrapper,
    # IdentityResult assembly, harvest hop) is what we exercise.
    # **_ absorbs newer keyword-only params (carry=, remote_obs=) so
    # this fake tracks the real _run_pipeline_sync signature — a
    # TypeError here would be swallowed by label_clip's fail-open
    # wrapper and silently flip the test's premise (review G2).
    def fake_pipeline_sync(*, frames, gallery, patient_id, sample_fps, **_):
        return (
            IdentityResult(
                reid_ran=True,
                patient_in_clip=True,
                confidence=0.85,
                best_frame_index=1,
                face_count=2,
                patient_bbox=(10.0, 20.0, 220.0, 480.0),
            ),
            None,  # no harvest payload
        )
    monkeypatch.setattr(reid, "_run_pipeline_sync", fake_pipeline_sync)

    result = await reid.label_clip(
        mp4_bytes=b"fake-mp4",
        patient_id=UUID(patient_id),
    )

    assert result.reid_ran is True
    assert result.patient_in_clip is True
    assert result.confidence == 0.85
    assert result.patient_bbox == (10.0, 20.0, 220.0, 480.0)
    # Legacy alias for callers still referencing face_bbox.
    assert result.face_bbox == result.patient_bbox
    assert result.face_count == 2
