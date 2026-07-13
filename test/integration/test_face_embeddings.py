"""Integration tests for the face-embedding API.

Exercises the multipart POST → background-processing → ready row
lifecycle end to end, plus list, summary, and delete. The InsightFace
embedding step is monkeypatched (mocking the model is fine here; the
embedding-extraction correctness is covered by the threshold +
adaptive-match unit tests). Storage uploads do hit real Supabase
Storage running locally.

Auth + RLS + cross-account behavior follow the same patterns as the
detector and event suites in this directory.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest
from httpx import AsyncClient

from src.services.reid import embeddings as reid_embeddings
from src.services.reid import storage as reid_storage
from src.services.reid.embeddings import FaceHit
from test.conftest import mint_test_jwt


# Minimal valid 1×1 JPEG. Supabase Storage stores by content-type
# header; the bytes don't have to be a useful image because the
# embedding step is monkeypatched in every test that needs to "see"
# the photo.
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


def _tiny_jpeg(seed: int = 0) -> bytes:
    """The same minimal JPEG bytes for every photo. `seed` is unused
    in this prebuilt-bytes flavor; kept in the signature so callers
    using `_tiny_jpeg(i)` still work. The embedding step is
    monkeypatched, so the actual pixel content is irrelevant."""
    return _TINY_JPEG


def _patch_storage(monkeypatch) -> dict[str, bytes]:
    """In-memory Storage shim. Returns the dict so tests can assert
    which keys got uploaded. The real Supabase Storage HTTP path is
    covered by `test_storage.py`; here we want to test the
    embedding pipeline without depending on bucket configuration."""
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
    return store


def _patch_embed_ok(monkeypatch, det_score: float = 0.85) -> None:
    """Monkeypatch storage + embedding extractor to a happy path.
    Single helper because every test that needs a "success" embed
    flow also needs storage stubbed."""
    import numpy as np

    _patch_storage(monkeypatch)

    async def fake_embed(image_bytes, *, model_name="buffalo_l"):
        return FaceHit(
            embedding=np.zeros(512, dtype=np.float32),
            det_score=det_score,
            bbox=(10.0, 10.0, 60.0, 80.0),
        )
    monkeypatch.setattr(reid_embeddings, "embed_image_bytes", fake_embed)


def _patch_embed_no_face(monkeypatch) -> None:
    _patch_storage(monkeypatch)
    async def fake_embed(image_bytes, *, model_name="buffalo_l"):
        return None
    monkeypatch.setattr(reid_embeddings, "embed_image_bytes", fake_embed)


async def _wait_for_status(
    client: AsyncClient,
    headers: dict[str, str],
    patient_id: str,
    *,
    expected_ready: int,
    expected_failed: int = 0,
    timeout_seconds: float = 10.0,
) -> dict:
    """Poll the summary endpoint until counts match or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    while True:
        r = await client.get(
            f"/v1/patients/{patient_id}/face-embeddings/summary",
            headers=headers,
        )
        assert r.status_code == 200, r.text
        s = r.json()
        if (s["selfie_count"] == expected_ready
                and s["failed_count"] == expected_failed
                and s["processing_count"] == 0):
            return s
        if asyncio.get_event_loop().time() > deadline:
            pytest.fail(
                f"timed out waiting for ready={expected_ready} "
                f"failed={expected_failed}, got {s}"
            )
        await asyncio.sleep(0.1)


# ─── POST happy path ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_three_selfies_all_become_ready(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    """Three valid selfies → 3 rows, all ready after processing,
    summary reflects the count, last_update set."""
    _patch_embed_ok(monkeypatch)
    headers, pid = await _setup_patient(client, auth_user)

    files = [
        ("photos", (f"selfie_{i}.jpg", _tiny_jpeg(i), "image/jpeg"))
        for i in range(3)
    ]
    r = await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=files,
    )
    assert r.status_code == 202, r.text
    rows = r.json()
    assert len(rows) == 3
    for row in rows:
        assert row["status"] == "processing"
        assert row["source"] == "selfie"
        assert row["patient_id"] == pid

    summary = await _wait_for_status(
        client, headers, pid, expected_ready=3,
    )
    assert summary["selfie_count"] == 3
    assert summary["harvested_count"] == 0
    assert summary["processing_count"] == 0
    assert summary["failed_count"] == 0
    assert summary["last_update"] is not None


# ─── POST: photo with no face → row marked failed ────────────────


@pytest.mark.asyncio
async def test_post_photo_without_face_marked_failed(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    _patch_embed_no_face(monkeypatch)
    headers, pid = await _setup_patient(client, auth_user)

    files = [("photos", ("blank.jpg", _tiny_jpeg(), "image/jpeg"))]
    r = await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=files,
    )
    assert r.status_code == 202

    summary = await _wait_for_status(
        client, headers, pid, expected_ready=0, expected_failed=1,
    )
    assert summary["selfie_count"] == 0
    assert summary["failed_count"] == 1


# ─── POST: low-quality face below enrollment threshold ───────────


@pytest.mark.asyncio
async def test_post_low_quality_face_below_threshold_marked_failed(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    """The enrollment processor rejects faces below
    `reid_enrollment_min_det_score` (default 0.4). Adaptive matching
    handles low-quality at MATCH time, but we keep the enrollment bar
    a little higher so the seed selfie set has variety."""
    _patch_embed_ok(monkeypatch, det_score=0.2)
    headers, pid = await _setup_patient(client, auth_user)

    files = [("photos", ("low_q.jpg", _tiny_jpeg(), "image/jpeg"))]
    await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=files,
    )

    summary = await _wait_for_status(
        client, headers, pid, expected_ready=0, expected_failed=1,
    )
    assert "low_quality_face" in (
        # Pull the actual row to confirm the failure_reason is set.
        (await client.get(
            f"/v1/patients/{pid}/face-embeddings", headers=headers,
        )).json()[0]["failure_reason"]
    )


# ─── POST: validation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_zero_photos_returns_422(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=[],  # FastAPI's File(...) requires presence; this surfaces as 422
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_post_too_many_photos_rejected(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    """Submitting one photo over the per-enrollment cap should 400 before
    any processing kicks off. Derived from settings so a cap change can't
    strand this test (it previously hardcoded 21 against an old cap of 20)."""
    from src.settings import get_settings

    cap = get_settings().reid_max_photos_per_enrollment
    headers, pid = await _setup_patient(client, auth_user)
    files = [
        ("photos", (f"p{i}.jpg", _tiny_jpeg(i), "image/jpeg"))
        for i in range(cap + 1)
    ]
    r = await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=files,
    )
    assert r.status_code == 400
    assert "too many photos" in r.json()["detail"]


@pytest.mark.asyncio
async def test_post_unsupported_content_type_rejected(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    files = [("photos", ("weird.gif", b"GIF89a", "image/gif"))]
    r = await client.post(
        f"/v1/patients/{pid}/face-embeddings",
        headers=headers,
        files=files,
    )
    assert r.status_code == 400
    assert "unsupported content type" in r.json()["detail"]


# ─── POST: cross-account ────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_to_other_caregivers_patient_returns_404(
    client: AsyncClient, auth_user_factory, monkeypatch,
) -> None:
    user_a = await auth_user_factory()
    user_b = await auth_user_factory()
    _, pid_a = await _setup_patient(client, user_a)

    # Caregiver B tries to upload to patient owned by caregiver A.
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(user_b[0], user_b[1])}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "Other"})
    files = [("photos", ("evil.jpg", _tiny_jpeg(), "image/jpeg"))]
    r = await client.post(
        f"/v1/patients/{pid_a}/face-embeddings",
        headers=headers_b,
        files=files,
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "patient_not_found"


# ─── DELETE ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_embedding_removes_row(
    client: AsyncClient, auth_user, monkeypatch,
) -> None:
    _patch_embed_ok(monkeypatch)
    headers, pid = await _setup_patient(client, auth_user)
    files = [("photos", ("a.jpg", _tiny_jpeg(), "image/jpeg"))]
    rows = (await client.post(
        f"/v1/patients/{pid}/face-embeddings", headers=headers, files=files,
    )).json()
    embedding_id = rows[0]["id"]

    await _wait_for_status(client, headers, pid, expected_ready=1)

    r = await client.delete(
        f"/v1/face-embeddings/{embedding_id}", headers=headers,
    )
    assert r.status_code == 204

    after = (await client.get(
        f"/v1/patients/{pid}/face-embeddings", headers=headers,
    )).json()
    assert after == []


@pytest.mark.asyncio
async def test_delete_other_caregivers_embedding_returns_404(
    client: AsyncClient, auth_user_factory, monkeypatch,
) -> None:
    _patch_embed_ok(monkeypatch)
    user_a = await auth_user_factory()
    user_b = await auth_user_factory()
    headers_a, pid_a = await _setup_patient(client, user_a)
    files = [("photos", ("a.jpg", _tiny_jpeg(), "image/jpeg"))]
    rows = (await client.post(
        f"/v1/patients/{pid_a}/face-embeddings",
        headers=headers_a,
        files=files,
    )).json()
    embedding_id = rows[0]["id"]
    await _wait_for_status(client, headers_a, pid_a, expected_ready=1)

    headers_b = {"Authorization": f"Bearer {mint_test_jwt(user_b[0], user_b[1])}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "Other"})
    r = await client.delete(
        f"/v1/face-embeddings/{embedding_id}", headers=headers_b,
    )
    assert r.status_code == 404


# ─── Summary edge cases ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_summary_empty_gallery_returns_zeros(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.get(
        f"/v1/patients/{pid}/face-embeddings/summary",
        headers=headers,
    )
    assert r.status_code == 200
    s = r.json()
    assert s == {
        "selfie_count": 0,
        "harvested_count": 0,
        "processing_count": 0,
        "failed_count": 0,
        "last_update": None,
    }


@pytest.mark.asyncio
async def test_summary_for_other_caregivers_patient_returns_zeros(
    client: AsyncClient, auth_user_factory, monkeypatch,
) -> None:
    """Cross-account read returns empty-gallery shape rather than 404.
    The status card on both clients treats this as 'not enrolled'."""
    _patch_embed_ok(monkeypatch)
    user_a = await auth_user_factory()
    user_b = await auth_user_factory()
    headers_a, pid_a = await _setup_patient(client, user_a)
    files = [("photos", ("a.jpg", _tiny_jpeg(), "image/jpeg"))]
    await client.post(
        f"/v1/patients/{pid_a}/face-embeddings",
        headers=headers_a,
        files=files,
    )
    await _wait_for_status(client, headers_a, pid_a, expected_ready=1)

    headers_b = {"Authorization": f"Bearer {mint_test_jwt(user_b[0], user_b[1])}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "Other"})
    r = await client.get(
        f"/v1/patients/{pid_a}/face-embeddings/summary",
        headers=headers_b,
    )
    assert r.status_code == 200
    assert r.json()["selfie_count"] == 0
