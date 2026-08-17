"""Read what the pipeline already recorded for one clip's window.

The database half of the sidecar. `sidecar.py` decides what a note says;
this decides what goes into it, and nothing more — no new analysis, no
extra model call, no third write in the detection tick. Everything here
already exists because detection ran.

The firing rules are BORROWED, never restated. Field names come from
`event_promotion._promotion_spec_for`, the confidence bar from
`_effective_threshold`, the run-ending gap from
`scheduling.decision.episode_exit_gap_for`. If any of those change, the
sidecar changes with them. Copying their logic here would give the corpus
a second, quietly diverging opinion about what fired — the one failure
that would make every label in it suspect.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, select

from src.db.analysis import Analysis
from src.db.detector import Detector
from src.db.event import Event
from src.services.annotations.sidecar import ClipFacts, DetectorSpec, build_sidecar

log = logging.getLogger(__name__)


def _specs_for(detectors: list[Detector]) -> list[DetectorSpec]:
    """Turn this patient's detectors into read instructions.

    Imported from `event_promotion` at call time rather than module level:
    that module pulls in the detection stack, and the recorder must not
    grow a startup dependency on it.
    """
    from src.services.detection.event_promotion import (
        _effective_threshold,
        _promotion_spec_for,
    )
    from src.services.detection.scheduling.decision import episode_exit_gap_for

    specs: list[DetectorSpec] = []
    for det in detectors:
        if not det.identifier:
            continue  # nothing to read its verdict out of
        detected, conf, base = _promotion_spec_for(det)
        try:
            threshold = _effective_threshold(base, det.sensitivity)
        except Exception:  # noqa: BLE001 — a bad row must not kill the note
            threshold = base
        # Only sustained detectors get their episode gap; a point-shape
        # detector uses the sidecar's own short merge window, so two falls
        # a minute apart stay two falls.
        gap = episode_exit_gap_for(det) if det.is_sustained else None
        specs.append(
            DetectorSpec(
                identifier=det.identifier,
                detected_field=detected,
                confidence_field=conf,
                name=det.name,
                category=str(getattr(det.category, "value", det.category)),
                alert_trigger=det.alert_trigger,
                is_sustained=bool(det.is_sustained),
                exit_gap_seconds=gap,
                duration_threshold_seconds=(
                    float(det.duration_seconds) if det.duration_seconds else None
                ),
                threshold=threshold,
                reasoning_field=f"{det.identifier}_reasoning",
            )
        )
    return specs


async def build_for_clip(
    session: Any, clip: ClipFacts, *, generated_at: datetime,
) -> dict[str, Any]:
    """Assemble the sidecar for ``clip`` from the database."""
    started = clip.started_at
    ended = clip.ended_at

    observations = (
        await session.execute(
            select(Analysis)
            .where(
                and_(
                    Analysis.camera_id == UUID(str(clip.camera_id)),
                    Analysis.captured_at >= started,
                    Analysis.captured_at <= ended,
                )
            )
            .order_by(Analysis.captured_at)
        )
    ).scalars().all()

    # Alerts are matched on the PATIENT, not the camera. A camera_id is
    # nullable on events and a scheduled/absence alert has none at all —
    # filtering by camera would silently drop exactly the alerts that most
    # need explaining.
    events = (
        await session.execute(
            select(Event)
            .where(
                and_(
                    Event.patient_id == UUID(str(clip.patient_id)),
                    Event.occurred_at >= started,
                    Event.occurred_at <= ended,
                )
            )
            .order_by(Event.occurred_at)
        )
    ).scalars().all()

    detectors = (
        await session.execute(
            select(Detector).where(
                Detector.patient_id == UUID(str(clip.patient_id))
            )
        )
    ).scalars().all()

    by_id = {d.id: d for d in detectors}
    alerts: list[Any] = []
    for ev in events:
        det = by_id.get(getattr(ev, "detector_id", None))
        # The sidecar needs the detector's identifier and trigger to match
        # an alert to its evidence and to know whether it has footage at
        # all. Carry them on a lightweight view rather than mutating ORM
        # rows attached to a live session.
        alerts.append(
            _AlertView(
                row=ev,
                identifier=det.identifier if det else None,
                alert_trigger=det.alert_trigger if det else None,
            )
        )

    return build_sidecar(
        clip,
        list(observations),
        alerts,
        generated_at=generated_at,
        detector_specs=_specs_for(list(detectors)),
    )


class _AlertView:
    """An event plus the two detector facts the sidecar needs.

    Attribute lookups fall through to the row, so a column added later
    (the operator gate's `gate_state`) is picked up with no change here.
    """

    __slots__ = ("_row", "identifier", "alert_trigger")

    def __init__(self, *, row: Any, identifier: str | None,
                 alert_trigger: str | None) -> None:
        self._row = row
        self.identifier = identifier
        self.alert_trigger = alert_trigger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._row, name, None)


def clip_facts_from_upload(
    *, key: str, started_at: datetime, duration_seconds: float,
    caregiver_id: Any, patient_id: Any, camera_id: Any, room: str,
    masked: bool, timeline_exact: bool,
) -> ClipFacts:
    """Adapt what the recorder knows at upload time into `ClipFacts`."""
    return ClipFacts(
        key=key,
        started_at=started_at,
        duration_seconds=duration_seconds,
        caregiver_id=str(caregiver_id),
        patient_id=str(patient_id),
        camera_id=str(camera_id),
        room=room,
        masked=masked,
        timeline_exact=timeline_exact,
    )


__all__ = [
    "build_for_clip",
    "clip_facts_from_upload",
    "timedelta",  # re-exported for callers building windows
]
