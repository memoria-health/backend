"""L1 scheduler — which detectors are due this tick, against which
targets (detection v2, P3 T0).

ONE pure function evaluated once per detector row. All behavior comes
from the row's effective dimensions (`dims.resolve_dims`) plus the
tick context — never from per-detector code, which is exactly why
customs scale: once the P5 compiler fills a custom's columns, the
scheduler cannot tell it from a built-in.

Parity contract (D2, Aryaman 2026-07-13): at default settings (floors
0.0, sample k=1, presence not required) every enabled detector is due
on every tick — scheduled shapes included (the absence runner feeds on
their per-tick fields); only the scheduled_only duty cycle (unused
before P4) and verified-absence suppression can skip. Identical
coverage to today's monolithic call. The cost inversion is a settings
change at soak.

Subject rule (FIXED, not configurable — spec §1.2): run unless there
is POSITIVE verified evidence the patient is absent. Suppression of a
patient-scoped detector requires attribution to have actually run AND
every track to have ended verified-OTHER. Per-track flavor of the same
rule: a verified-OTHER track is never a crop target for a
patient-scoped detector, but UNKNOWN tracks always are (fail-open).

Crowd hardening (D3): more than `scheduler_crowd_fallback_tracks`
concurrent tracks ⇒ no crop targets anywhere this tick — everything
runs full-scene.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from src.services.detection.scheduling.dims import CROP_BUCKETS, EffectiveDims
from src.services.detection.scheduling.when_scope import is_active
from src.services.reid.body_pipeline import LABEL_OTHER, LABEL_PATIENT

if TYPE_CHECKING:
    from datetime import datetime

    from src.services.detection.prompt import DetectorSpec
    from src.services.detection.signals.tick_signals import TickSignals
    from src.services.reid.inference import IdentityResult
    from src.settings import Settings


@dataclass(frozen=True)
class ScheduledDetector:
    """One loadout entry: the compiled spec + its effective dims.

    `when_scope` is the row's operator-facing time window; None/"ALL
    DAY"-flavored means always-on (see scheduling.when_scope)."""

    spec: "DetectorSpec"
    dims: EffectiveDims
    when_scope: str | None = None


@dataclass(frozen=True)
class DueDetector:
    """A detector that runs this tick.

    `target_track_ids` is None for scene-level artifacts (full frame /
    zone); a tuple of track ids means the packer builds ONE CROP CALL
    PER TRACK (D3 — per-person, never a union box).

    `windowed` marks a detector inside a real when_scope window this
    tick — the packer groups windowed detectors into their own call,
    never shared with always-on ones.
    """

    spec: "DetectorSpec"
    dims: EffectiveDims
    target_track_ids: tuple[int, ...] | None
    windowed: bool = False


@dataclass(frozen=True)
class DueSet:
    due: tuple[DueDetector, ...]
    crowd_fallback: bool
    # (identifier, reason) for every skipped detector — stashed on the
    # analyses row so "why didn't X run" is always answerable.
    skipped: tuple[tuple[str, str], ...]


def resolve_due(
    entries: Sequence[ScheduledDetector],
    *,
    signals: "TickSignals",
    identity: "IdentityResult",
    settings: "Settings",
    tick_index: int,
    local_now: "datetime | None" = None,
    patient_confirmed_elsewhere: bool = False,
) -> DueSet:
    """Resolve the due set for one tick. Pure: no I/O, no clock — the
    caller supplies `local_now` (caregiver-local wall time) for
    when_scope windows; None skips window gating entirely (fail open).
    """
    tracks = identity.tracks
    n_tracks = len(tracks) if tracks else 0
    crowd = tracks is not None and n_tracks > settings.scheduler_crowd_fallback_tracks

    # A window too short to measure (frames_used < 4 ⇒ tick_signals
    # returned zeros, not readings) must FAIL OPEN: motion floors are
    # treated as satisfied. Otherwise a truncated/renegotiating window
    # reads as "still" and — once the post-soak dials are raised — a
    # measurement failure would gate seizure/fall off (T0 review F1).
    measurement_ok = signals.frames_used >= 4

    amount_by_track = {tm.track_id: tm.amount for tm in signals.track_motion}
    patient_ids = {
        t.track_id for t in (tracks or ()) if t.label == LABEL_PATIENT
    }
    other_ids = {
        t.track_id for t in (tracks or ()) if t.label == LABEL_OTHER
    }

    due: list[DueDetector] = []
    skipped: list[tuple[str, str]] = []

    for entry in entries:
        d = entry.dims
        ident = entry.spec.identifier

        # Scheduled-shape ("absence") detectors STAY in the per-tick
        # due set: the 60 s absence runner never runs them — it only
        # QUERIES analyses for the `<id>_detected` fields the per-tick
        # calls produce, so skipping them here would starve its
        # evidence stream and fire false absence events (T3/T4 review
        # F3). They are promotion-inert per tick, so no double-fire.
        # Only the explicit scheduled_only duty cycle opts out of
        # per-tick collection (nothing uses it before P4).
        if d.duty_cycle == "scheduled_only":
            skipped.append((ident, "scheduled_only_duty"))
            continue

        # when_scope gate (2026-07-14). Scheduled-shape rows are exempt
        # — their per-tick fields feed the absence runner all day and
        # their schedule_config governs timing. Everything else skips
        # outside its window; inside it, the entry is marked `windowed`
        # so the packer gives it a distinct call.
        windowed = False
        if local_now is not None and d.shape != "scheduled":
            eligible, windowed = is_active(entry.when_scope, local_now)
            if not eligible:
                skipped.append((ident, "outside_when_scope"))
                continue

        if d.duty_cycle == "thinned" and tick_index % settings.scheduler_sample_every_k:
            skipped.append((ident, "thinned_off_tick"))
            continue

        # Cross-camera exclusivity (Aryaman 2026-07-18): the patient
        # was POSITIVELY face-confirmed on a DIFFERENT camera within
        # the TTL, and THIS tick's own identity does not see them —
        # patient-scoped detectors skip here (they cannot be about a
        # person who is provably elsewhere). Anyone/environment scopes
        # are untouched; any ambiguity fails open by construction
        # (the registry only holds positive confirmations).
        if (
            d.subject_scope == "patient"
            and patient_confirmed_elsewhere
            and not identity.patient_in_clip
        ):
            skipped.append((ident, "patient_elsewhere"))
            continue

        # Verified-absence suppression is a SUBJECT rule, so it is
        # evaluated before the branch dispatch — an (unlikely)
        # patient-scoped environment-branch detector must suppress
        # like any other patient-scoped one (T0 review F4; spec §1.2
        # keys suppression on subject alone).
        if (
            d.subject_scope == "patient"
            and identity.attribution_ran
            and identity.has_only_verified_other_tracks
            and not identity.patient_in_clip
        ):
            skipped.append((ident, "verified_not_patient"))
            continue

        # ── ENVIRONMENT scope: no person needed, gated only by env
        # change (parity floor 0.0 never skips; unmeasurable windows
        # fail open). With no tracks, env_motion is whole-frame —
        # still the right reading here. (evidence_branch collapse
        # 2026-07-18: 'environment' was always ≡ subject_scope.)
        if d.subject_scope == "environment":
            if measurement_ok and signals.env_motion < settings.scheduler_env_motion_floor:
                skipped.append((ident, "env_floor"))
                continue
            due.append(DueDetector(
                spec=entry.spec, dims=d, target_track_ids=None,
                windowed=windowed,
            ))
            continue

        # ── PERSON branches (person_motion / person_still) ────────
        # Presence gate. Only when the operator opted in
        # (requires_track=True) AND the tracker positively saw nobody.
        # tracks is None (tracking didn't run) always fails open.
        if settings.scheduler_person_requires_track and tracks is not None and n_tracks == 0:
            skipped.append((ident, "no_person"))
            continue

        # Qualifying tracks. requires_motion applies the per-track
        # motion floor; still-condition detectors are presence-only (a
        # still person is the whole point of those).
        qualifying: list[int] | None = None
        if tracks:
            if d.requires_motion and measurement_ok:
                floor = settings.scheduler_person_motion_floor
                qualifying = [
                    t.track_id for t in tracks
                    if amount_by_track.get(t.track_id, 0.0) >= floor
                ]
            else:
                # person_still is presence-only; person_motion on an
                # unmeasurable window fails open to every track.
                qualifying = [t.track_id for t in tracks]
            if not qualifying and settings.scheduler_person_requires_track:
                skipped.append((ident, "below_motion_floor"))
                continue

        # Target filtering — the per-track flavor of the fixed rule:
        # verified-OTHER tracks are never patient targets, UNKNOWN
        # tracks always are (suppression itself already ran above).
        if d.subject_scope == "patient" and identity.attribution_ran:
            if qualifying:
                if patient_ids:
                    qualifying = [t for t in qualifying if t in patient_ids]
                else:
                    qualifying = [t for t in qualifying if t not in other_ids]
                if not qualifying and settings.scheduler_person_requires_track:
                    skipped.append((ident, "no_qualifying_patient_track"))
                    continue

        # Target resolution: crop buckets get per-track targets (D3);
        # crowd ticks and box-less ticks fall back to full-scene.
        targets: tuple[int, ...] | None = None
        if d.bucket in CROP_BUCKETS and not crowd and qualifying:
            targets = tuple(qualifying)
        due.append(DueDetector(
            spec=entry.spec, dims=d, target_track_ids=targets,
            windowed=windowed,
        ))

    return DueSet(due=tuple(due), crowd_fallback=crowd, skipped=tuple(skipped))
