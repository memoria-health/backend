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
from src.services.detection import catalog

_log = logging.getLogger(__name__)

# Human-friendly event titles. Every preset is prefixed with "Possible"
# so the lock-screen alert, the in-app event header, and the history
# row never assert a definitive medical/safety judgment — the model
# can be wrong, and the caregiver is the human-in-the-loop. This
# wording is intentional for liability reasons; do not strip it.
_EVENT_TITLE_BY_IDENTIFIER: dict[str, str] = {
    "fall":       "Possible Fall",
    "wandering":  "Possible Wandering",
    "agitation":  "Possible Agitation",
    "medication": "Possible Medication Event",
    # Internal setup self-test — a success confirmation, NOT a "Possible …" alert.
    "setup_test": "Memoria Test Alert successful",
}

# Identifiers that fire exactly once then disable themselves (the hidden setup
# self-test). On promotion we flip the detector row's `is_on=false`; because the
# row is per-patient (shared across all their cameras), that one flip stops it
# everywhere — it can never fire a second time on any camera.
_FIRE_ONCE_KEYS = catalog.fire_once_keys()

# Fixed narrative for the setup self-test. The default event narrative is the
# VLM's free-text scene description; for the test we want a deterministic,
# reassuring confirmation that the whole alert path worked.
_SETUP_TEST_NARRATIVE = (
    "Your alerts are working. Memoria saw three fingers held up to the camera "
    "and ran this test end-to-end — detection, push, and email. This test runs "
    "only once."
)


def _event_title_for(detector: Detector) -> str:
    """Pick the human-readable event title for a fire from this detector.
    Preset detectors get their canonical "Possible …" wording. Custom
    detectors fall back to prefixing the user-defined detector name.
    """
    if title := _EVENT_TITLE_BY_IDENTIFIER.get(detector.identifier):
        return title
    name = detector.name.strip()
    if name.lower().startswith("possible"):
        return name
    return f"Possible {name}"


# Per-built-in (detected-field, confidence-field, standard-base-threshold), from
# the catalog (handles medication's `medication_taken`). Custom detectors aren't
# in the catalog — they use the default base below. Either way the FINAL bar is
# `base + sensitivity offset` (see `_effective_threshold`), so EVERY detector is
# confidence-gated and the caregiver's sensitivity picker actually moves the bar.
_PROMOTION_SPECS: dict[str, tuple[str, str, float]] = catalog.promotion_rules()

# Custom detectors (not in the catalog) gate at this standard base; their
# compiled schema emits `<id>_detected` / `<id>_confidence`.
_DEFAULT_BASE_THRESHOLD = 0.70

# Sensitivity moves the per-detector base. With every base at 0.70 this yields
# low=0.85, standard=0.70, high=0.40 — matching the iOS picker ("low = fewer /
# clearest", "high = more alerts"). Offsets (not fixed numbers) so it stays
# correct if a base is ever tuned per-detector later.
_SENSITIVITY_OFFSET: dict[str, float] = {
    "low":      0.15,    # stricter → fewer alerts
    "standard": 0.0,
    "high":     -0.30,   # looser → more alerts
}
_THRESHOLD_FLOOR = 0.05
_THRESHOLD_CEIL = 0.95


def _effective_threshold(base: float, sensitivity: str) -> float:
    """Final firing bar = base + sensitivity offset, clamped to a sane band.
    An unknown sensitivity (impossible via the DB enum + the iOS picker, which
    only offers low/standard/high) fails LOUD and falls back to standard rather
    than silently inventing a bar."""
    offset = _SENSITIVITY_OFFSET.get(sensitivity)
    if offset is None:
        _log.error(
            "[event_promotion] unknown sensitivity %r — using standard", sensitivity,
        )
        offset = _SENSITIVITY_OFFSET["standard"]
    return min(_THRESHOLD_CEIL, max(_THRESHOLD_FLOOR, base + offset))


def _promotion_spec_for(det: Detector) -> tuple[str, str, float]:
    """(detected-field, confidence-field, base-threshold) for a detector — from
    the catalog for built-ins (so medication's `medication_taken` is respected),
    else the standard `<id>_detected`/`<id>_confidence` naming + default base for
    custom detectors."""
    spec = _PROMOTION_SPECS.get(det.identifier or "")
    if spec is not None:
        return spec
    ident = det.identifier
    return f"{ident}_detected", f"{ident}_confidence", _DEFAULT_BASE_THRESHOLD


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

    # Identity gating happens UPSTREAM in `camera_ingest._run_tick`:
    # when Re-ID has positive evidence the patient is not in this clip
    # (face_count > 0 and patient_in_clip=False with an enrollment),
    # the LLM call is skipped entirely and we never reach this code
    # path. That's the only filter — no defense-in-depth duplicate
    # here, so the two paths can't fall out of sync. The identity_*
    # fields are still written into detector_fields for the dashboard
    # to render.
    #
    # If `maybe_promote_analysis` is called, it means: either the
    # patient is confirmed in the clip, or Re-ID couldn't confirm
    # they aren't (no faces detected, timeout, no enrollment). In
    # both cases we promote any detector findings normally.

    # Load this patient's detectors so we can map identifier → (id, category, source).
    stmt = select(Detector).where(Detector.patient_id == patient_id)
    detectors_by_id: dict[str, Detector] = {}
    for d in (await session.execute(stmt)).scalars().all():
        if d.identifier:
            detectors_by_id[d.identifier] = d

    fired: list[Event] = []
    skipped: list[str] = []

    # One unified firing path for EVERY detector (preset, library, custom):
    # `<id>_detected` is true AND `<id>_confidence` clears the sensitivity-
    # adjusted threshold. The caregiver's sensitivity picker sets the bar.
    for ident, det in detectors_by_id.items():
        if not det.is_on:
            continue
        # Scheduled ("absence") detectors land their `<id>_detected` in analyses
        # for the scheduler to read, but must NOT fire reactively here — they
        # fire only via the ScheduledDetectorRunner at their deadline, on ABSENCE.
        if det.shape == "scheduled":
            continue

        detected_field, conf_field, base = _promotion_spec_for(det)
        if not _is_truthy_bool(detector_fields.get(detected_field)):
            continue

        conf = detector_fields.get(conf_field)
        if not isinstance(conf, (int, float)) or isinstance(conf, bool):
            # Detected=true but no usable confidence. Don't silently fire AND
            # don't silently drop — log loudly so a miswired detector surfaces.
            _log.warning(
                "[event_promotion] detector %s (%s) detected=true but confidence "
                "field %r missing/invalid (%r) — NOT firing",
                det.id, ident, conf_field, conf,
            )
            continue

        threshold = _effective_threshold(base, det.sensitivity)
        if float(conf) < threshold:
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

        # Fire-once: the moment a self-test promotes, disable its (single,
        # per-patient) row so it can never fire again — on this camera or any
        # other. The caller commits this session, persisting is_on=false; the
        # next tick's registry query (is_on=true filter) drops it everywhere.
        if ident in _FIRE_ONCE_KEYS:
            det.is_on = False
            _log.info(
                "[event_promotion] fire-once detector %s (%s) disabled after "
                "first fire for patient %s", det.id, ident, patient_id,
            )

    return PromotionResult(fired=fired, skipped_for_dedup=skipped)


def _is_truthy_bool(v: Any) -> bool:
    return isinstance(v, bool) and v is True


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
    # The setup self-test gets a fixed, reassuring confirmation rather than the
    # VLM's raw scene description.
    narrative = (
        _SETUP_TEST_NARRATIVE
        if detector.identifier in _FIRE_ONCE_KEYS
        else description
    )
    row = Event(
        id=uuid4(),
        patient_id=patient.id,
        detector_id=detector.id,
        camera_id=camera.id,
        category=detector.category,
        source=detector.source,
        icon_name=detector.icon_name,
        title=_event_title_for(detector),
        location=camera.room,
        occurred_at=occurred_at,
        # Phase 0: no clips yet (HTTP clip ingest doesn't archive).
        # Phase 4 wires LiveKit Track Egress → has_clip=true.
        has_clip=False,
        # Privacy mode suppresses the clip entirely. Stamp it at creation
        # so the apps show "No clip — Privacy mode" rather than a forever
        # "Generating clip…". (The post-roll race — privacy flipping on
        # DURING archival — is stamped in clip_archival instead.)
        clip_suppressed=patient.privacy_mode_enabled,
        narrative=narrative,
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
