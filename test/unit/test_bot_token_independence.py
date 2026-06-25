"""Regression guard: the detector bot's LiveKit token is independent of
privacy mode (Phase 4.3).

Privacy mode severs the CAREGIVER's media (token-mint gate + live
UpdateParticipant). The bot must keep subscribing so analysis — and the
text insights — never stop. The bot token is minted in
``RoomWorker._mint_viewer_token`` which has no access to (and never
reads) the patient's privacy flag, so it is unconditionally a
subscriber. This test locks that in: if someone ever wires privacy into
the bot's grants, "the stream still reaches the backend" breaks and this
fails.
"""

from __future__ import annotations

from types import SimpleNamespace

import jwt as pyjwt

from src.services.detection.room_worker import RoomWorker


def test_bot_token_is_always_a_subscriber() -> None:
    stub = SimpleNamespace(
        room_name="patient_abc",
        _settings=SimpleNamespace(
            livekit_api_key="test-key",
            livekit_api_secret="test-secret-long-enough-for-hs256-signing-1234567890",
        ),
        _identity=lambda: "detector_worker_abc_1",
    )

    token = RoomWorker._mint_viewer_token(stub)
    video = pyjwt.decode(token, options={"verify_signature": False})["video"]

    assert video["canSubscribe"] is True  # bot keeps analyzing under privacy
    assert video.get("canPublish", False) is False  # bot never publishes
