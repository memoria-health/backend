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
    privacy_mode_enabled: bool,
    data_collection_enabled: bool,
    motion_in_clip: bool | None,
    gate_people_present: bool,
) -> bool:
    """Record only when the patient opted in, there's motion, and the
    detection layer says someone is present.

    ``privacy_mode_enabled`` is a hard override: when True the gate is
    always closed, no matter what the other signals say. Privacy
    SUSPENDS data collection (it never mutates the consent flag), so it
    is evaluated here at decision time rather than by flipping
    ``data_collection_enabled``.

    ``motion_in_clip`` is the tick's unified motion verdict (SAD hint
    OR backend frame-diff since v2 P2). The ``None`` in the signature
    is defensive back-compat — the tick always passes a real bool
    today; falsy still gates closed. Always returns a real ``bool``.
    """
    if privacy_mode_enabled:
        return False
    return bool(data_collection_enabled and motion_in_clip and gate_people_present)
