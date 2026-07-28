"""The data-collection gate: a pure decision over existing signals.

``should_collect`` is the ONE place the recorder decides whether a
frame should be recorded. It reuses signals the detection pipeline
already computes — it never re-runs detection — so recording can never
diverge from "the patient opted in AND something is happening AND
someone is present."
"""

from __future__ import annotations


def should_collect(
    *,
    camera_privacy_enabled: bool,
    data_collection_enabled: bool,
    motion_in_clip: bool | None,
    gate_people_present: bool,
) -> bool:
    """Record only when the patient opted in, there's motion, and the
    detection layer says someone is present.

    ``camera_privacy_enabled`` is a hard override: data collection
    NEVER records a camera whose room-derived privacy is on
    (``cameras.privacy_enabled`` — bathroom/bedroom, or the fail-closed
    classifier fallback), no matter what the other signals say. Privacy
    EXCLUDES the camera without mutating the patient's consent flag, so
    it is evaluated here at decision time rather than by flipping
    ``data_collection_enabled``. (Replaces the retired patient-level
    ``privacy_mode_enabled`` check — 2026-07-27, per-camera model.)

    ``motion_in_clip`` is the tick's unified motion verdict (SAD hint
    OR backend frame-diff since v2 P2). The ``None`` in the signature
    is defensive back-compat — the tick always passes a real bool
    today; falsy still gates closed. Always returns a real ``bool``.
    """
    if camera_privacy_enabled:
        return False
    return bool(data_collection_enabled and motion_in_clip and gate_people_present)
