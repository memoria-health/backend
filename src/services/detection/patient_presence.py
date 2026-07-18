"""Cross-camera patient-confirmation registry (Aryaman 2026-07-18:
"if patient is confirmed in one camera, others do not fire — that
saves us money").

One instance per RoomWorker (= per patient); every CameraIngest in the
room shares it. A camera marks itself when identity POSITIVELY
confirms the patient in its clip (face lock — never on unresolved).
Other cameras' patient-scoped detectors then skip for the TTL window,
with the visible skip reason `patient_elsewhere`.

Fail-open is the law: staleness, ambiguity, or an empty registry mean
NO suppression — absence of certainty never mutes a safety detector.
The camera that confirms keeps running everything; a camera whose own
tick sees the patient also runs regardless of the registry (the
scheduler checks its tick-local identity first).

Single-writer-per-camera within one asyncio loop — no lock needed
(same contract as MotionCache / FrameBuffer).
"""

from __future__ import annotations

import time
from uuid import UUID

# How long one positive confirmation shields the OTHER cameras. Long
# enough to cover consecutive ticks between confirmations of a seated
# patient; short enough that walking to another room re-enables its
# detectors within seconds of the track/face being lost.
CONFIRM_TTL_S = 45.0


class PatientPresence:
    def __init__(self, *, ttl_s: float = CONFIRM_TTL_S) -> None:
        self._ttl = ttl_s
        self._confirmed_at: dict[UUID, float] = {}

    def mark(self, camera_id: UUID) -> None:
        """This camera's tick positively confirmed the patient."""
        self._confirmed_at[camera_id] = time.monotonic()

    def confirmed_elsewhere(self, camera_id: UUID) -> bool:
        """True when ANOTHER camera confirmed the patient within TTL."""
        now = time.monotonic()
        return any(
            cid != camera_id and now - ts <= self._ttl
            for cid, ts in self._confirmed_at.items()
        )
