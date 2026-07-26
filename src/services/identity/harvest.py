"""Harvest eligibility — one condition (Aryaman, 2026-07-24).

Face-confirmed patient, alone in frame, on a settled track → harvest
liberally. The face confirmation lives on the TRACK (seat.anchor ==
"face"), so a lock earned at the door keeps harvesting on the couch
three hours later — across every pose and distance the pool was
starving for. Everything else never harvests: body-anchored seats
can't self-feed the pool that scored them, and multi-person harvesting
simply doesn't exist. Pool hygiene (cap/TTL/dedup/golden) is the
existing CrossClipBodyGallery machinery, unchanged.
"""

from __future__ import annotations

from typing import Optional

from src.services.identity.seat import Seat


def harvest_eligible(
    seat: Optional[Seat],
    *,
    camera_id: str,
    track_id: int,
    solo: bool,
    track_age_s: float,
    min_age_s: float,
) -> bool:
    if seat is None:
        return False
    if not seat.face_confirmed:
        # The face confirmation lives on the TRACK (Seat.face_confirmed
        # — sticky for the track's life), NOT on the current anchor: a
        # later body-score ratchet must not silently stop harvesting.
        return False
    if seat.face_confirms < 2:
        # Anti-poisoning depth (2026-07-24): one face-grade round can
        # be an impostor's outlier frame; a lock that can't RE-confirm
        # by face never writes the body gallery. It was the harvest,
        # not the lock itself, that entrenched the wrong patient live.
        return False
    if (seat.camera_id, seat.track_id) != (camera_id, track_id):
        return False
    if not solo:
        return False
    if track_age_s < min_age_s:
        # Births are the messy moments (crossings, occlusion re-entry) —
        # never learn from them.
        return False
    return True
