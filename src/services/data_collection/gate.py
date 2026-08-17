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

    Room-derived privacy (``cameras.privacy_enabled``) deliberately does
    NOT gate collection (Aryaman 2026-08-12): the patient's consent flag
    is the recording boundary, and a sensitive VERDICT already switches
    the session to the masked encoder, so a privacy room records under
    the same per-moment protection as any other room. (Privacy still
    governs LIVE VIEW subscriptions — that severing is unchanged.) This
    reverses the 2026-07-27 hard override, which excluded bathroom /
    bedroom cameras from collection entirely.

    ``motion_in_clip`` is the tick's unified motion verdict (SAD hint
    OR backend frame-diff since v2 P2). The ``None`` in the signature
    is defensive back-compat — the tick always passes a real bool
    today; falsy still gates closed. Always returns a real ``bool``.
    """
    return bool(data_collection_enabled and motion_in_clip and gate_people_present)
