"""A tick that dies ANYWHERE must leave a visible analyses row.

Aryaman 2026-07-17: during the depleted-Gemini unpause, ticks whose
whole VLM chain failed wrote nothing — the dashboard showed silence
while the worker error-looped every 10s. The fix: the outer tick
wrapper (`_run_tick_safely`) writes a `skipped_reason='tick_failed'`
row (provider must stay inside the analyses CHECK set) stamped with
the error class + first message line, from a FRESH session.

Runs through `_run_tick_safely` — the wrapper that owns the writer —
not `_run_tick_inner` like the other tick suites.
"""

from __future__ import annotations

import shutil

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.reid.result import IdentityResult

from test.integration.identity_helpers import _setup_patient_with_camera

from test.integration.test_signals_wiring import (
    _last_analysis,
    _make_ingest,
    _push_frames,
)

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


@pytest.mark.asyncio
async def test_failed_tick_writes_visible_row(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )

    async def _boom(**_kwargs):
        raise RuntimeError("VLM chain exhausted (test)")

    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=_boom, sad=True,
    )
    _push_frames(ingest, flicker=True)

    await ingest._run_tick_safely()

    row = await _last_analysis(session, patient_id)
    f = row.detector_fields
    assert row.provider == "skipped"
    assert f["skipped_reason"] == "tick_failed"
    assert f["error_class"] == "RuntimeError"
    assert "VLM chain exhausted" in f["error_msg"]
