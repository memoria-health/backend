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
    data_collection_enabled: bool,
    motion_in_clip: bool | None,
    gate_people_present: bool,
) -> bool:
    """Record only when the patient opted in, there's motion, and the
    detection layer says someone is present.

    ``motion_in_clip`` may be ``None`` (the motion cache returns None
    when a camera hasn't sent a recent heartbeat) — that gates closed.
    Always returns a real ``bool``.
    """
    return bool(data_collection_enabled and motion_in_clip and gate_people_present)
