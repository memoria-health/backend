"""Rolling-deploy safety for the pod's `/observe` contract.

Three processes have to agree here and they restart independently: the
backend (Docker), the identity service (`gpu/identity/service.py`, on the
Mac or GPU host) and the perception pod (RunPod, behind an SSH tunnel).
There is no coordinated deploy and no version negotiation, so a wire
change is only safe if BOTH restart orders work:

  new identity service → old pod   (the extra field must be ignored)
  old identity service → new pod   (the missing field must default)

`capture_ts` was added on 2026-08-06 so the pod can stamp retained masks
with the frame's real capture time instead of its own processing time.
It is an OPTIONAL form field precisely so neither order can break. This
test pins the FastAPI behaviour that claim rests on, against handler
signatures mirroring the two versions — the real module cannot be
imported here (it constructs a CUDA face worker at import time).

If this ever fails, do not "fix" it by requiring the field: make the new
side tolerate its absence again. A required field turns a routine pod
restart into a tracking outage.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, File, Form, UploadFile

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402

_FILES = [("frames", ("00000.jpg", b"\xff\xd8not-a-real-jpeg", "image/jpeg"))]
_OLD_CALLER = {"person_thr": "0.4", "camera_id": "identity_cafe1234"}
_NEW_CALLER = {**_OLD_CALLER, "capture_ts": '{"0": 1786000000.5}'}


def _old_pod() -> FastAPI:
    """`/observe` as it was before the capture_ts field."""
    app = FastAPI()

    @app.post("/observe")
    async def observe(  # noqa: ANN202
        frames: list[UploadFile] = File(...),
        person_thr: float = Form(0.4),
        camera_id: str = Form("default"),
    ):
        return {"frames": len(frames), "camera_id": camera_id}

    return app


def _new_pod() -> FastAPI:
    """`/observe` with the optional stamp field (perception/server.py)."""
    app = FastAPI()

    @app.post("/observe")
    async def observe(  # noqa: ANN202
        frames: list[UploadFile] = File(...),
        person_thr: float = Form(0.4),
        camera_id: str = Form("default"),
        capture_ts: str = Form(""),
    ):
        return {
            "frames": len(frames), "camera_id": camera_id,
            "capture_ts": capture_ts,
        }

    return app


def test_a_new_caller_does_not_break_an_old_pod() -> None:
    """Deploy the identity service first: the pod has not restarted yet
    and knows nothing about `capture_ts`. It must ignore the field, not
    422 — a rejected tick is a tracking gap, and the seat decays."""
    resp = TestClient(_old_pod()).post(
        "/observe", data=_NEW_CALLER, files=_FILES,
    )
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "identity_cafe1234"


def test_an_old_caller_does_not_break_a_new_pod() -> None:
    """Deploy the pod first: the identity service is still sending the
    old form. The field must default to empty, and the pod then falls
    back to stamping masks with its own clock — degraded matching for
    the censor, never a failed tick."""
    resp = TestClient(_new_pod()).post(
        "/observe", data=_OLD_CALLER, files=_FILES,
    )
    assert resp.status_code == 200
    assert resp.json()["capture_ts"] == ""


def test_both_new_carries_the_stamps() -> None:
    resp = TestClient(_new_pod()).post(
        "/observe", data=_NEW_CALLER, files=_FILES,
    )
    assert resp.status_code == 200
    assert resp.json()["capture_ts"] == '{"0": 1786000000.5}'
