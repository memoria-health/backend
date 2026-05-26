"""Promote interesting analyses into caregiver-facing `events` rows.

The dashboard streams every analysis to its web UI via SSE; the iOS
caregiver app expects a higher signal-to-noise ratio — only fires that
warrant a notification land in the Events tab.

Per-detector thresholds are inline for v0. If we add many more
custom detectors, lift these to a per-detector column
(`event_threshold_confidence`).

De-dup: events for the same (patient, detector) within 30s of each
other are squashed. Prevents a noisy 3-tick run of fall_detected=true
from filing three separate events.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.db.patient_face_embedding import PatientFaceEmbedding

_log = logging.getLogger(__name__)

# (detection-field-name, confidence-field-name, min-confidence)
# Falling back to confidence-field-name = None means no confidence
# gate (any True fires).
_PROMOTION_RULES: dict[str, tuple[str, str | None, float]] = {
    "fall":       ("fall_detected",       "fall_confidence",       0.70),
    "wandering":  ("wandering_detected",  "wandering_confidence",  0.60),
    "agitation":  ("agitation_detected",  "agitation_confidence",  0.60),
    "medication": ("medication_taken",    "medication_confidence", 0.60),
}

_DEDUP_WINDOW = timedelta(seconds=30)


@dataclass(frozen=True)
class PromotionResult:
    fired: list[Event]
    skipped_for_dedup: list[str]   # detector identifiers


async def maybe_promote_analysis(
    session: AsyncSession,
    *,
    patient_id: UUID,
    camera_id: UUID,
    occurred_at: datetime,
    description: str,
    detector_fields: Mapping[str, Any],
) -> PromotionResult:
    """Inspect `detector_fields` for any detector that hit its
    promotion rule and insert events for them. Returns the events
    inserted + any rules that matched but were squashed by de-dup.
    """
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise ValueError(f"[event_promotion] camera {camera_id} not found")
    patient = await session.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"[event_promotion] patient {patient_id} not found")

    # Identity gate. Three states, and only one suppresses:
    #
    #   (a) Re-ID did not run (timeout / crash / feature disabled) →
    #       identity_reid_ran=False → bypass the gate, promote
    #       identity-agnostic (today's behavior). This is the
    #       load-bearing safety invariant: a Re-ID infrastructure
    #       failure must NEVER silence detector promotion. We'd
    #       rather over-alert than miss a fall.
    #
    #   (b) Re-ID ran, patient_in_clip=True → promote normally.
    #
    #   (c) Re-ID ran, patient_in_clip=False, AND a ready enrollment
    #       exists → SUPPRESS. The patient is confidently not in this
    #       clip; we don't want to alert the caregiver about
    #       household-member behavior.
    #
    #   (d) Re-ID ran, patient_in_clip=False, NO enrollment yet →
    #       bypass (day-zero before the caregiver has enrolled).
    identity_reid_ran = bool(detector_fields.get("identity_reid_ran", False))
    identity_patient_in_clip = bool(detector_fields.get("identity_patient_in_clip", False))
    if identity_reid_ran and not identity_patient_in_clip:
        gallery_exists = await _patient_has_ready_enrollment(session, patient_id)
        if gallery_exists:
            _log.info(
                "event_promotion.identity_gated patient=%s camera=%s "
                "suppressing all detector promotions for this tick",
                patient_id, camera_id,
            )
            return PromotionResult(fired=[], skipped_for_dedup=[])

    # Load this patient's detectors so we can map identifier → (id, category, source).
    stmt = select(Detector).where(Detector.patient_id == patient_id)
    detectors_by_id: dict[str, Detector] = {}
    for d in (await session.execute(stmt)).scalars().all():
        if d.identifier:
            detectors_by_id[d.identifier] = d

    fired: list[Event] = []
    skipped: list[str] = []

    # Built-in rules.
    for ident, (detected_field, conf_field, min_conf) in _PROMOTION_RULES.items():
        if not _is_truthy_bool(detector_fields.get(detected_field)):
            continue
        if conf_field is not None:
            conf = detector_fields.get(conf_field)
            if not isinstance(conf, (int, float)) or isinstance(conf, bool):
                continue
            if float(conf) < min_conf:
                continue
        det = detectors_by_id.get(ident)
        if det is None or not det.is_on:
            continue

        if await _within_dedup_window(
            session, patient_id=patient_id, detector_id=det.id, now=occurred_at,
        ):
            skipped.append(ident)
            continue

        fired.append(
            await _insert_event(
                session,
                detector=det,
                patient=patient,
                camera=cam,
                occurred_at=occurred_at,
                description=description,
            )
        )

    # Custom detectors (anything not in _PROMOTION_RULES): fire on bare
    # `<identifier>_detected: true` with no confidence gate.
    for ident, det in detectors_by_id.items():
        if ident in _PROMOTION_RULES:
            continue
        if not det.is_on:
            continue
        if not _is_truthy_bool(detector_fields.get(f"{ident}_detected")):
            continue
        if await _within_dedup_window(
            session, patient_id=patient_id, detector_id=det.id, now=occurred_at,
        ):
            skipped.append(ident)
            continue
        fired.append(
            await _insert_event(
                session,
                detector=det,
                patient=patient,
                camera=cam,
                occurred_at=occurred_at,
                description=description,
            )
        )

    return PromotionResult(fired=fired, skipped_for_dedup=skipped)


def _is_truthy_bool(v: Any) -> bool:
    return isinstance(v, bool) and v is True


async def _patient_has_ready_enrollment(
    session: AsyncSession,
    patient_id: UUID,
) -> bool:
    """True iff the patient has at least one ready face embedding.
    Used by the identity gate to decide whether to fall back to
    today's identity-agnostic behavior (no enrollment) or actively
    suppress non-patient promotions.
    """
    stmt = (
        select(PatientFaceEmbedding.id)
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            PatientFaceEmbedding.status == "ready",
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _within_dedup_window(
    session: AsyncSession,
    *,
    patient_id: UUID,
    detector_id: UUID,
    now: datetime,
) -> bool:
    cutoff = now - _DEDUP_WINDOW
    # Strict `>` — open interval. An event landing exactly at `cutoff`
    # no longer blocks a fresh promotion.
    stmt = (
        select(Event.id)
        .where(
            and_(
                Event.patient_id == patient_id,
                Event.detector_id == detector_id,
                Event.occurred_at > cutoff,
            )
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def _insert_event(
    session: AsyncSession,
    *,
    detector: Detector,
    patient: Patient,
    camera: Camera,
    occurred_at: datetime,
    description: str,
) -> Event:
    row = Event(
        id=uuid4(),
        patient_id=patient.id,
        detector_id=detector.id,
        camera_id=camera.id,
        category=detector.category,
        source=detector.source,
        icon_name=detector.icon_name,
        title=detector.name,
        location=camera.room,
        occurred_at=occurred_at,
        # Phase 0: no clips yet (HTTP clip ingest doesn't archive).
        # Phase 4 wires LiveKit Track Egress → has_clip=true.
        has_clip=False,
        narrative=description,
        opened=False,
        reviewed=False,
        review_choice=None,
        review_notes=None,
        reviewed_at=None,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.flush()
    _log.info(
        "[event_promotion] fired event %s for detector %s (%s) on patient %s",
        row.id, detector.identifier, detector.name, patient.id,
    )
    return row
