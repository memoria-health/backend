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

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.services.detection.scheduling.decision import (
    cooldown_seconds_for,
    episode_exit_gap_for,
    parse_eventing,
    windowed_entry_verdict,
)

if TYPE_CHECKING:
    from src.settings import Settings
from src.services.detection import catalog
from src.services import pager

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


# ── state episodes: entry on evidence, exit on ABSENCE of evidence ────
#
# An episode ENDS when the detector's positive evidence stops for
# `exit gap` seconds. It does NOT end after a fixed duration, so an
# episode lasts exactly as long as the behaviour does — a 3-minute nap
# and a 3-hour one both yield one event, with no per-detector guess
# about how long the activity "should" take.
#
# This replaced `episode_gap_seconds` as a rising-edge hysteresis
# (2026-07-29). That scheme had two defects, both seen live: it needed
# an LLM-guessed duration per detector, and it was applied as a sliding
# COUNT window that drains mid-episode — a Watching-TV session produced
# 5 events in 25 minutes because the count of ABOVE-THRESHOLD ticks fell
# below M during ordinary confidence wobble, re-arming entry while the
# man never moved. See the soak log: "held (1/3)" → fire → "ongoing
# (4/3)".
#
# The gap is derived from CRITICALITY, not from the behaviour, so it
# stays scale-free: safety states re-alert sooner after a real lull (a
# caregiver needs nudging again), routine states stay quiet. Its two
# constants keep the values the old category defaults used, so a detector
# that never had an inferred gap behaves as it did before.
#
# The constants and accessor live in `scheduling.decision`
# (`episode_exit_gap_for`) so the rising edge here and the falling
# ("finished") edge in services/scheduling/scheduler cannot drift apart.

# How far back the episode-boundary scan looks. Bounds query cost; an
# episode running longer than this re-fires once, which is logged.
_EPISODE_LOOKBACK_SECONDS = 12 * 3600


def _continuation_field(det: "Detector", detected_field: str) -> str:
    """The analyses field that keeps a sustained episode ALIVE.

    `<identifier>_state_active` when the detector's schema declares it
    (the detection/state split: entry is voted on the strict
    `_detected`, continuation latches on the loose state question),
    else the detected field — presets and pre-split customs keep their
    exact prior behaviour. Duck-typed on `.identifier`/`.schema_json`
    so unit tests can pass a SimpleNamespace.
    """
    ident = getattr(det, "identifier", None)
    if not ident:
        return detected_field
    state_field = f"{ident}_state_active"
    schema = getattr(det, "schema_json", None) or {}
    return state_field if state_field in schema else detected_field


async def _episode_boundary(
    session: AsyncSession,
    *,
    patient_id: UUID,
    detected_field: str,
    now: datetime,
    exit_gap_seconds: int,
) -> datetime | None:
    """Start of the episode in progress at `now`, or None if the detector
    has no positive evidence in the lookback.

    The boundary is the oldest positive tick reachable from `now` without
    crossing a gap of >= `exit_gap_seconds`. Computed with one window
    function: mark every positive tick whose predecessor is missing or
    too far back as a run-start, then take the most recent such mark.

    Continuation counts `detected=true` at ANY confidence, deliberately
    looser than the promotion threshold. The whole failure being fixed
    was an episode "ending" because confidence wobbled under the bar
    while the behaviour continued — so a sub-threshold tick is still
    evidence the activity is ongoing, even though it could never fire.
    """
    lookback_start = now - timedelta(seconds=_EPISODE_LOOKBACK_SECONDS)
    positives = (
        select(
            Analysis.captured_at.label("at"),
            func.lag(Analysis.captured_at)
            .over(order_by=Analysis.captured_at)
            .label("prev"),
        )
        .where(
            and_(
                Analysis.patient_id == patient_id,
                Analysis.captured_at > lookback_start,
                Analysis.captured_at <= now,
                Analysis.detector_fields[detected_field].astext == "true",
            )
        )
        .subquery()
    )
    gap = timedelta(seconds=exit_gap_seconds)
    stmt = select(func.max(positives.c.at)).where(
        or_(
            positives.c.prev.is_(None),
            positives.c.at - positives.c.prev >= gap,
        )
    )
    boundary = (await session.execute(stmt)).scalar_one_or_none()
    # An episode older than the lookback gets truncated here, which lets it
    # re-fire once. Say so rather than letting it look like a fresh episode.
    if boundary is not None and boundary - lookback_start < timedelta(seconds=30):
        _log.warning(
            "[event_promotion] %s episode boundary hit the %dh lookback edge "
            "(%s) — an episode longer than the lookback will re-fire once",
            detected_field, _EPISODE_LOOKBACK_SECONDS // 3600,
            boundary.isoformat(),
        )
    return boundary


async def _episode_already_fired(
    session: AsyncSession,
    *,
    detector_id: UUID,
    patient_id: UUID,
    since: datetime,
) -> bool:
    """True if this detector already filed an event at or after `since`
    (the current episode's start).

    Keyed on the EVENTS table rather than on tick history: "have we
    already told the caregiver about this episode" is the question the
    anti-spam rule actually asks, and it stays correct no matter how the
    tick evidence wobbles in between.
    """
    stmt = (
        select(Event.id)
        .where(
            and_(
                Event.detector_id == detector_id,
                Event.patient_id == patient_id,
                Event.occurred_at >= since,
            )
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none() is not None


# Record-only occurrence paging, behind settings.record_only_paging.
# Sleep/activity excluded: continuous state labels, paging them is spam.
_RECORD_ONLY_PAGE_EXCLUDE = frozenset({"sleep", "activity"})
_RECORD_ONLY_PAGE_MIN_GAP_S = 300.0
_record_only_last_page: dict[str, float] = {}


async def _maybe_page_record_only(
    session: AsyncSession,
    *,
    det: "Detector",
    ident: str,
    camera_id: UUID,
    occurred_at: datetime,
    detector_fields: Mapping[str, Any],
    settings: "Settings | None",
) -> None:
    if settings is None or not getattr(settings, "record_only_paging", False):
        return
    if ident in _RECORD_ONLY_PAGE_EXCLUDE:
        return
    detected_field, _conf_field, _base = _promotion_spec_for(det)
    if not _is_truthy_bool(detector_fields.get(detected_field)):
        return
    now_mono = time.monotonic()
    last = _record_only_last_page.get(ident)
    if last is not None and now_mono - last < _RECORD_ONLY_PAGE_MIN_GAP_S:
        return
    # Rising edge only: consecutive positive ticks are one occurrence.
    prev = (await session.execute(
        select(Analysis.detector_fields[detected_field].astext)
        .where(
            Analysis.camera_id == camera_id,
            Analysis.captured_at < occurred_at,
            Analysis.detector_fields.has_key(detected_field),
        )
        .order_by(Analysis.captured_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if prev == "true":
        return
    _record_only_last_page[ident] = now_mono
    reasoning = str(detector_fields.get(f"{ident}_reasoning") or "")
    confidence = detector_fields.get(f"{ident}_confidence")
    det_name, det_id = det.name, det.id
    from src.db.session import get_sessionmaker
    from src.services.event_gate import tick_verdict_url

    # Deferred: the caller's transaction may still roll back (a tick
    # failure after promotion did, 2026-08-18, and the page pointed at
    # a row that never committed). Re-verify the row, then page.
    async def _page_after_commit() -> None:
        await asyncio.sleep(3)
        try:
            async with get_sessionmaker()() as s:
                exists = (await s.execute(
                    select(Analysis.id).where(
                        Analysis.camera_id == camera_id,
                        Analysis.captured_at == occurred_at,
                    ).limit(1)
                )).scalar_one_or_none()
            if exists is None:
                log.info("[record-only-page] %s tick rolled back — no page",
                         ident)
                return
            verdict_url = tick_verdict_url(
                camera_id=camera_id, captured_at=occurred_at, ident=ident,
                settings=settings,
            )
            ok = await pager.trigger_keyed(
                settings=settings,
                dedup_key=f"record-only:{det_id}:{occurred_at.isoformat()}",
                summary=f"[record-only] {det_name}: {reasoning[:120]}",
                severity="info",
                source="memoria-record-only-soak",
                link_url=verdict_url,
                details={
                    "detector": det_name,
                    "identifier": ident,
                    "reasoning": reasoning[:400],
                    "confidence": confidence,
                    "verdict_url": verdict_url,
                    "note": (
                        "no event exists — this rules on the COUNT: accept "
                        "keeps the occurrence, deny removes it and files a "
                        "labeled false positive"
                    ),
                },
            )
            if ok:
                log.info("[record-only-page] %s (%s) paged — %r",
                         det_name, ident, reasoning[:80])
        except Exception:  # noqa: BLE001 — paging must never hurt a tick
            log.exception("[record-only-page] %s failed", ident)

    asyncio.create_task(
        _page_after_commit(), name=f"record-only-page-{ident}")


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
        if det.alert_trigger in ("absence", "trailing", "record_only"):
            # record_only (sleep/activity): fields feed the summarizers,
            # never an event — and the scheduled runner ignores it too.
            if det.alert_trigger == "record_only":
                await _maybe_page_record_only(
                    session, det=det, ident=ident, camera_id=camera_id,
                    occurred_at=occurred_at, detector_fields=detector_fields,
                    settings=settings,
                )
            continue

        # Criticality split (Aryaman 2026-07-28, from the soak's first
        # false fire — Watching TV attributed to a visiting aunt):
        # NON-CRITICAL patient detectors FAIL-QUIET — they fire only when
        # identity POSITIVELY confirms the patient in the clip, so a
        # visitor's routine never lands on the patient's record. SAFETY
        # detectors keep FAIL-LOUD semantics (fire unless the patient is
        # confirmed elsewhere — the upstream skip): a fall must never be
        # missed because a face wasn't visible.
        #
        # Keyed on NOT-SAFETY, never on an enumeration of the quiet
        # categories (2026-08-08). This read `category == "routine"`, and
        # when 2026-08-05 added `behaviour` and `care` the gate silently
        # stopped covering them — the compiler's identical
        # stakes-vs-category conflation was fixed the SAME DAY, but this
        # copy was missed. Found by soak: four "Laptop work" (behaviour)
        # events fired with identity_patient_in_clip=false, one of them
        # on a clip whose scene call read "the room is empty throughout"
        # (people_count=0) while the crop call — cropped to a stale
        # track — narrated a seated person into an empty room. A new
        # category added tomorrow must default to fail-quiet, not to
        # bypassing identity.
        if (det.subject_scope == "patient"
                and det.category != "safety"
                and not _is_truthy_bool(
                    detector_fields.get("identity_patient_in_clip"))):
            skipped.append(f"{ident}(patient_unconfirmed)")
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

        # Continuous ("state") detectors fire ONCE PER EPISODE. An episode
        # runs from its first positive tick until positive evidence stops
        # for the exit gap; one event is filed at entry and everything
        # after it is "ongoing", however long the behaviour lasts.
        # Instant (non-sustained) detectors are unaffected.
        if det.is_sustained:
            exit_gap = episode_exit_gap_for(det)
            boundary = await _episode_boundary(
                session,
                patient_id=patient_id,
                # CONTINUATION is judged on the state field when the
                # detector has one (detection/state split, 2026-08-07):
                # `_detected` is strict ENTRY evidence and flaps with
                # momentary attention, so an episode latched on it
                # "ends" every time the person glances at a laptop —
                # measured as 3 Watching-TV alerts in 18 minutes for
                # one continuous session. `_state_active` answers the
                # looser question the latch actually needs (is the
                # session plausibly still in progress), so the episode
                # survives the flapping and one entry event covers the
                # whole sitting. Detectors without the field (all
                # presets, pre-split customs) keep the old behaviour.
                detected_field=_continuation_field(det, detected_field),
                now=occurred_at,
                exit_gap_seconds=exit_gap,
            )
            # Already filed an event inside THIS episode ⇒ ongoing. This is
            # the whole anti-spam rule, and it is bounded by the behaviour
            # rather than by any timer, which is why it no longer needs a
            # per-detector duration guess.
            if boundary is not None and await _episode_already_fired(
                session, detector_id=det.id, patient_id=patient_id,
                since=boundary,
            ):
                # Keep the episode's end fresh: ended_at converges on
                # the last positive tick with no exit-gap bookkeeping.
                await session.execute(
                    Event.__table__.update()
                    .where(Event.detector_id == det.id,
                           Event.patient_id == patient_id,
                           Event.occurred_at >= boundary)
                    .values(ended_at=occurred_at))
                _log.info(
                    "[event_promotion] %s episode ongoing since %s "
                    "(exit gap %ds) — no new event",
                    ident, boundary.isoformat(), exit_gap,
                )
                ongoing.append(ident)
                continue

            cfg = parse_eventing(det.eventing) if v2_decisions else None
            # Duration trigger: the entry window IS the duration threshold, so
            # it fires on the crossing (present for ~D) and re-arms after a lull.
            if cfg is not None and det.alert_trigger == "duration" and det.duration_seconds:
                cfg = _apply_duration_entry(cfg, det.duration_seconds)
            if cfg is not None and cfg.entry_m > 1:
                # M>1: accumulate entry evidence, clamped to this episode.
                # The clamp matters — an unclamped T wider than the exit gap
                # would still see the previous episode's ticks and report
                # "ongoing" for a genuinely new one.
                verdict, count = await windowed_entry_verdict(
                    session,
                    patient_id=patient_id,
                    detected_field=detected_field,
                    conf_field=conf_field,
                    threshold=threshold,
                    now=occurred_at,
                    cfg=cfg,
                    floor=boundary,
                )
                if verdict != "fire":
                    _log.info(
                        "[event_promotion] %s windowed entry %s (%d/%d)",
                        ident, verdict, count, cfg.entry_m,
                    )
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
        # Event gate: stamp the hold IN THE INSERT, not afterwards.
        # `events` is in the supabase_realtime publication, so the row is
        # broadcast to the caregiver app the moment it commits. Stamping
        # 'pending' in a later UPDATE left a window — small, but the whole
        # point of the gate is that the family never sees a false alarm —
        # where the row was published with gate_state NULL and the RLS
        # policy therefore treated it as visible. Setting it here means a
        # gated event is never visible to a caregiver at any instant.
        # Null for every patient with the gate off, which is all of them.
        gate_state="pending" if patient.event_gate_enabled else None,
        gate_held_at=(
            datetime.now(timezone.utc) if patient.event_gate_enabled else None
        ),
    )
    session.add(row)
    await session.flush()
    _log.info(
        "[event_promotion] fired event %s for detector %s (%s) on patient %s",
        row.id, detector.identifier, detector.name, patient.id,
    )
    return row
