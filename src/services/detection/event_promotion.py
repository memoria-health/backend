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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import Float, and_, cast, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.services.detection.scheduling.decision import (
    cooldown_seconds_for,
    parse_eventing,
    windowed_entry_verdict,
)

if TYPE_CHECKING:
    from src.settings import Settings
from src.services.detection import catalog

_log = logging.getLogger(__name__)

# Duration trigger (Phase E): fire once the activity has PERSISTED for
# ~duration_seconds. Reuses the windowed-count crossing — require ~all ticks
# in the D-window to qualify, so the window saturates (and fires, once) only
# after the activity has genuinely been present that long. Assumes the ~60s
# scheduler cadence; the exact m is soak-tunable.
_DURATION_TICK_SECONDS = 60


def _apply_duration_entry(cfg, duration_seconds: int):
    """Override an EventingConfig's entry window with the duration threshold."""
    from dataclasses import replace
    m = max(2, duration_seconds // _DURATION_TICK_SECONDS)
    return replace(cfg, entry_m=m, entry_t_seconds=duration_seconds)

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
    # Detectors that crossed threshold on this clip but lost the
    # one-event-per-clip pick (the model's `primary_detection`, else the
    # highest-confidence detection won). Surfaced for logging/observability.
    skipped_for_primary: list[str] = field(default_factory=list)
    # Continuous ("state") detectors that crossed threshold but were
    # suppressed because the activity was already ongoing (not a rising edge).
    skipped_ongoing: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Candidate:
    """A detector that crossed its firing threshold on one clip and is in
    the running to be THE single event that clip yields.
    """

    identifier: str
    confidence: float
    category: str
    detector: Any   # the ORM Detector row (None in pure unit tests)


def _select_primary(
    candidates: Sequence["_Candidate"], *, primary_hint: str | None
) -> "_Candidate | None":
    """Pick the single detection to fire for one clip.

    Several detectors can legitimately be true for one action (a fall is
    also a general emergency; agitation overlaps calling-for-help), which
    used to file one event per matching detector. We now fire exactly one:

      1. catalog SEVERITY (higher wins) — if it's choking we don't also
         want the agitation alert on the same clip;
      2. the model's `primary_detection` pick, breaking ties WITHIN the
         same severity, IF it names a detector that actually crossed
         threshold this clip (the model judges which action dominates);
      3. otherwise the highest-confidence detection. Ties break toward
         `safety` over `routine`, then identifier, for determinism.

    Returns None when nothing crossed threshold.
    """
    if not candidates:
        return None
    hint = (primary_hint or "").strip()

    def _rank(c: "_Candidate") -> tuple[int, int, float, int, str]:
        # Precedence, in order:
        #   1. SEVERITY (higher wins) — so Choking beats Agitation regardless of
        #      what the model or confidence say (if it's choking, we don't also
        #      want the agitation alert on the same clip);
        #   2. the model's own `primary_detection` pick — breaks ties WITHIN the
        #      same severity (which action dominates);
        #   3. confidence; 4. safety over routine; 5. identifier for determinism.
        # v2: severity lives on the detector row (backfilled for built-ins,
        # compiler-set for future customs); the catalog map is the fallback
        # for NULL rows and detector-less unit fixtures. catalog_v2's import
        # validation keeps row values and the map in lockstep until the map
        # is retired.
        row_severity = getattr(c.detector, "severity", None)
        severity = (
            row_severity if row_severity is not None
            else catalog.severity_for(c.identifier)
        )
        is_hint = 0 if c.identifier == hint else 1
        safety_first = 0 if c.category == "safety" else 1
        return (-severity, is_hint, -c.confidence, safety_first, c.identifier)

    return min(candidates, key=_rank)


# Fallback episode gap (seconds) for a `state` detector that has no
# inferred `episode_gap_seconds` — e.g. the legacy preset Agitation. Safety
# states tighten the gap (re-alert sooner after a lull); routine states use
# a longer one. New custom state detectors carry their own LLM-inferred gap.
_DEFAULT_STATE_GAP_SAFETY = 60
_DEFAULT_STATE_GAP_ROUTINE = 300


def _episode_gap_for(det: Detector) -> int:
    """Hysteresis gap (seconds) for a `state` detector: its inferred value,
    else a category-aware default."""
    if det.episode_gap_seconds:
        return det.episode_gap_seconds
    category = str(getattr(det.category, "value", det.category))
    return _DEFAULT_STATE_GAP_SAFETY if category == "safety" else _DEFAULT_STATE_GAP_ROUTINE


async def _state_episode_ongoing(
    session: AsyncSession,
    *,
    patient_id: UUID,
    detected_field: str,
    conf_field: str,
    threshold: float,
    since: datetime,
    before: datetime,
) -> bool:
    """True if this detector was already PROMOTABLE — detected=true AND
    confidence >= threshold — on a PRIOR tick within [since, before), i.e. the
    continuous activity was already firing-eligible, so this tick is NOT a
    fresh rising edge.

    The confidence gate matters: a prior SUB-threshold blip (detected=true but
    below the bar, which never fires) must NOT count as "ongoing", or it would
    swallow the genuine first above-threshold event of an episode. Reads the
    per-tick `analyses` history; the current tick's row is excluded via the
    half-open `before` (it has captured_at == occurred_at).
    """
    stmt = (
        select(Analysis.id)
        .where(
            and_(
                Analysis.patient_id == patient_id,
                Analysis.captured_at >= since,
                Analysis.captured_at < before,
                Analysis.detector_fields[detected_field].astext == "true",
                cast(Analysis.detector_fields[conf_field].astext, Float) >= threshold,
            )
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


async def maybe_promote_analysis(
    session: AsyncSession,
    *,
    patient_id: UUID,
    camera_id: UUID,
    occurred_at: datetime,
    description: str,
    detector_fields: Mapping[str, Any],
    settings: "Settings | None" = None,
) -> PromotionResult:
    """Inspect `detector_fields` for any detector that hit its
    promotion rule and insert events for them. Returns the events
    inserted + any rules that matched but were squashed by de-dup.

    `settings` is optional for back-compat: None (or the
    detection_v2_decision_enabled flag off) keeps the legacy decision
    logic byte-for-byte. Flag on hands two decisions to the v2
    primitives (P4, parallel-systems rule): state entry becomes the
    windowed count (M qualifying ticks within T; NULL eventing = M=1
    = today's rising edge exactly) and point re-fire spacing reads the
    row's cooldown_seconds instead of the global 30 s.
    """
    v2_decisions = bool(
        settings is not None
        and getattr(settings, "detection_v2_decision_enabled", False)
    )
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
    ongoing: list[str] = []

    # Pass 1 — collect every detector that crosses its bar this clip
    # (preset, library, custom share one path): `<id>_detected` is true AND
    # `<id>_confidence` clears the sensitivity-adjusted threshold. The
    # caregiver's sensitivity picker sets the bar.
    candidates: list[_Candidate] = []
    for ident, det in detectors_by_id.items():
        if not det.is_on:
            continue
        # Clock-driven triggers land their `<id>_detected` in analyses for the
        # runner to read, but must NOT fire reactively here: schedule fires on
        # ABSENCE at its deadline; trailing fires on the FALLING edge when the
        # activity ends. Both are handled by the ScheduledDetectorRunner.
        if det.alert_trigger in ("absence", "trailing"):
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

        # Continuous ("state") detectors fire only on the RISING EDGE of an
        # episode. If the activity was already detected within this detector's
        # hysteresis gap on a prior tick, it's ongoing — suppress, so a whole
        # session ("watching TV") collapses to one event instead of one per
        # tick. Instant (non-sustained) detectors are unaffected.
        if det.is_sustained:
            cfg = parse_eventing(det.eventing) if v2_decisions else None
            # Duration trigger: the entry window IS the duration threshold, so
            # it fires on the crossing (present for ~D) and re-arms after a lull.
            if cfg is not None and det.alert_trigger == "duration" and det.duration_seconds:
                cfg = _apply_duration_entry(cfg, det.duration_seconds)
            if cfg is not None and cfg.entry_m > 1:
                # v2 (P4) M>1: the count-crossing edge REPLACES the
                # activity rising edge — prior qualifying ticks inside
                # the window are the evidence being accumulated, so the
                # legacy check would misread them as "ongoing" and an
                # M=3 state could never fire. Crossing semantics carry
                # their own anti-spam (a saturated window = ongoing);
                # a lull that drains the window below M re-arms entry.
                verdict, count = await windowed_entry_verdict(
                    session,
                    patient_id=patient_id,
                    detected_field=detected_field,
                    conf_field=conf_field,
                    threshold=threshold,
                    now=occurred_at,
                    cfg=cfg,
                )
                if verdict != "fire":
                    _log.info(
                        "[event_promotion] %s windowed entry %s (%d/%d)",
                        ident, verdict, count, cfg.entry_m,
                    )
                    ongoing.append(ident)
                    continue
            elif await _state_episode_ongoing(
                session,
                patient_id=patient_id,
                detected_field=detected_field,
                conf_field=conf_field,
                threshold=threshold,
                since=occurred_at - timedelta(
                    seconds=_episode_gap_for(det),
                ),
                before=occurred_at,
            ):
                # M=1 (or flag off): today's on-sight rising edge,
                # byte-for-byte.
                ongoing.append(ident)
                continue

        candidates.append(
            _Candidate(
                identifier=ident,
                confidence=float(conf),
                category=str(getattr(det.category, "value", det.category)),
                detector=det,
            )
        )

    # Pass 2 — ONE event per clip. Overlapping detectors (fall + general
    # emergency, agitation + calling-for-help) used to file one event each
    # for the same clip; now we fire only the single best-fitting detection.
    # If the best pick is within its 30s cross-tick dedup window (a repeat of a
    # recent event), fall back to the next-best DISTINCT candidate rather than
    # yielding no event at all — otherwise a deduped winner would suppress a
    # genuinely different detection that also fired this clip.
    primary_hint = detector_fields.get("primary_detection")
    primary_hint = primary_hint if isinstance(primary_hint, str) else None
    remaining = list(candidates)
    winner: _Candidate | None = None
    while remaining:
        pick = _select_primary(remaining, primary_hint=primary_hint)
        if pick is None:
            break
        if await _within_dedup_window(
            session,
            patient_id=patient_id,
            detector_id=pick.detector.id,
            now=occurred_at,
            window_seconds=(
                cooldown_seconds_for(
                    pick.detector, int(_DEDUP_WINDOW.total_seconds()),
                )
                if v2_decisions
                else int(_DEDUP_WINDOW.total_seconds())
            ),
        ):
            skipped.append(pick.identifier)
            remaining.remove(pick)
            continue
        winner = pick
        break

    # Candidates that crossed threshold but neither fired nor deduped — they
    # lost the one-event-per-clip pick.
    suppressed = [
        c.identifier for c in candidates
        if c is not winner and c.identifier not in skipped
    ]

    if winner is not None:
        det = winner.detector
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
        # per-patient) row so it can never fire again — on this camera or
        # any other. The caller commits this session, persisting
        # is_on=false; the next tick's registry query (is_on=true filter)
        # drops it everywhere.
        if winner.identifier in _FIRE_ONCE_KEYS:
            det.is_on = False
            _log.info(
                "[event_promotion] fire-once detector %s (%s) disabled after "
                "first fire for patient %s", det.id, winner.identifier,
                patient_id,
            )

    if suppressed:
        _log.info(
            "[event_promotion] one-event-per-clip: winner=%s (hint=%r) "
            "suppressed=%r for patient %s",
            winner.identifier if winner else None,
            primary_hint, suppressed, patient_id,
        )
    if ongoing:
        _log.info(
            "[event_promotion] state-episode ongoing (no new event): %r "
            "for patient %s", ongoing, patient_id,
        )

    return PromotionResult(
        fired=fired, skipped_for_dedup=skipped, skipped_for_primary=suppressed,
        skipped_ongoing=ongoing,
    )


def _is_truthy_bool(v: Any) -> bool:
    return isinstance(v, bool) and v is True


async def _within_dedup_window(
    session: AsyncSession,
    *,
    patient_id: UUID,
    detector_id: UUID,
    now: datetime,
    window_seconds: int | None = None,
) -> bool:
    cutoff = now - (
        timedelta(seconds=window_seconds)
        if window_seconds is not None
        else _DEDUP_WINDOW
    )
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
        # Private cameras get clips too (2026-07-27 censor pipeline):
        # every clip is sensitivity-screened before upload and censored
        # when needed, so nothing is suppressed at creation time.
        # clip_suppressed is stamped ONLY by clip_archival, for
        # genuinely dropped clips (censoring failed → fail-closed drop).
        clip_suppressed=False,
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
