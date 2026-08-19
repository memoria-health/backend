"""THE seam between the backend and the GPU-host identity service.

This is the ONLY module the two deployment targets share (plan rule,
2026-07-26): the backend's sidecar imports it to validate and consume
/state; the identity service imports it to stamp and guard /state.
Everything else is private to its side of the wire.

Versioning: bump STATE_CONTRACT whenever the /state shape changes in a
way an old consumer would misread. A version mismatch is treated by
the consumer exactly like an unreachable service — fail-open, loud in
logs, never a silent misparse (the observe-v2 lesson: a stale server
must read as "no verdict", not "zero people all day").

Auth: production sets NEMARA_IDENTITY_TOKEN on both sides; the service
then requires `Authorization: Bearer <token>` on /state. Empty token =
no auth (local dev, host-only traffic).
"""

from __future__ import annotations

import hmac

STATE_CONTRACT = "state-v2"
AUTH_ENV = "NEMARA_IDENTITY_TOKEN"

# /state response, state-v2 (reference for both sides):
# {
#   "contract": "state-v2",
#   "ready": bool,
#   "seats": { "<patient uuid>": {camera, camera_name, track, score,
#              anchor, face_confirmed, since, confirmed_at}, ... },
#   "cameras": { "<8-hex prefix>": {name, tracks: {tid: {bbox, age_s}},
#                 solo, lag_s, pod_ms, stage_ms, probe_ms}, ... },
# }
# Camera keys are the FIRST 8 HEX CHARS of the camera UUID (the LiveKit
# identity suffix); consumers match by str(uuid).startswith(key).
#
# v1 -> v2 (2026-08-20): the single `seat` became `seats`, keyed by
# patient, because a camera can watch more than one resident. A patient
# ABSENT from `seats` is not located; that is not the same as absent
# from the room, and it must never resolve to another patient's seat.
#
# The backend reads both for one release (identity_client), so the node
# image and the backend can ship in either order without an
# identity-blind window. Delete the v1 branch once every node is on v2.


def auth_ok(header_value: str | None, configured_token: str) -> bool:
    """True iff the request may read /state. Empty configured token
    disables auth; otherwise constant-time compare of the Bearer
    credential (hmac.compare_digest — no timing oracle)."""
    if not configured_token:
        return True
    if not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(
        header_value.removeprefix("Bearer "), configured_token,
    )
