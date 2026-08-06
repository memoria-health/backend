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
from src.services.identity.observations import LABEL_PATIENT

# Emergency line (2026-07-18, the enacted-choking miss): detectors at
# or above this severity are EXEMPT from every cost gate — cross-camera
# exclusivity and the per-person motion floor. Real loadout values:
# choking 100 / seizure 96 / fall 95 / calling_for_help 88 stay
# ungateable; agitation 60 / wandering 55 / medication 20 / customs
# remain cost-managed. A missed emergency is never worth a saved call.
_EMERGENCY_SEVERITY = 80

if TYPE_CHECKING:
    from datetime import datetime

    from src.services.detection.prompt import DetectorSpec
    from src.services.detection.signals.tick_signals import TickSignals
    from src.services.identity.result import IdentityResult
    from src.settings import Settings


@dataclass(frozen=True)
class ScheduledDetector:
    """One loadout entry: the compiled spec + its effective dims.

    `when_scope` is the row's operator-facing time window; None/"ALL
    DAY"-flavored means always-on (see scheduling.when_scope)."""

    spec: "DetectorSpec"
    dims: EffectiveDims
    when_scope: str | None = None
    # Detector row category — OBSERVABILITY ONLY. The cost-gate
    # exemptions (cross-camera exclusivity, per-person motion floor)
    # key on `severity >= _EMERGENCY_SEVERITY`, NOT on category: the
    # category axis proved unusable (2026-07-18 — everything is
    # 'safety' except medication), so severity is the exemption
    # contract. Nothing gates on this field.
    category: str = "safety"


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
    # GLOBAL crowd rule (Aryaman 2026-07-27): the threshold applies to
    # the WHOLE home's concurrent tracks, not this camera's — a packed
    # room disables crop targeting everywhere. Local count still
    # counts (a home_count of 0 from an older provider degrades to the
    # per-camera behavior).
    n_home = max(n_tracks, getattr(identity, "people_count_home", 0))
    crowd = tracks is not None and n_home > settings.scheduler_crowd_fallback_tracks

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
    # NOTE (2026-08-06): there is no verified-OTHER any more. The ledger
    # rewrite deleted LABEL_OTHER along with the verified-OTHER audit (see
    # track_ledger's "Deleted vs the track-anchored ledger"), so a track is
    # only ever PATIENT or UNKNOWN and this set is ALWAYS empty. It is kept
    # named here purely so the filter below reads honestly; the branch that
    # consumed it was removed rather than left looking like a live guard.

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
        if local_now is not None and d.alert_trigger != "absence":
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
            # Emergency exemption (2026-07-18, the enacted-choking
            # miss): one misidentified face at lunch poisoned the
            # registry and suppressed choking on the REAL patient.
            # Cost gates never apply above the emergency line.
            and d.severity < _EMERGENCY_SEVERITY
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
        # Every body on camera, floor-independent. This is the crop pool:
        # whether a detector RUNS is a cost question, but once it runs it
        # must still be looking at a person (see the crop note below).
        admissible: list[int] | None = None
        if tracks:
            admissible = [t.track_id for t in tracks]
            # Emergency detectors ignore the motion floor: choking/
            # fidget-scale distress is LOW gross motion by nature
            # (hands at throat while seated reads ~0.01 — under any
            # useful floor).
            if (
                d.requires_motion and measurement_ok
                and d.severity < _EMERGENCY_SEVERITY
            ):
                floor = settings.scheduler_person_motion_floor
                qualifying = [
                    t.track_id for t in tracks
                    if amount_by_track.get(t.track_id, 0.0) >= floor
                ]
            else:
                # person_still is presence-only; person_motion on an
                # unmeasurable window fails open to every track.
                qualifying = list(admissible)
            if not qualifying and settings.scheduler_person_requires_track:
                skipped.append((ident, "below_motion_floor"))
                continue

        # Target filtering — the per-track flavor of the fixed rule:
        # verified-OTHER tracks are never patient targets, UNKNOWN
        # tracks always are (suppression itself already ran above).
        # Applied to BOTH lists: a non-patient body must never become a
        # crop target either, or the fallback would hand a patient-scoped
        # detector a picture of the wrong person.
        if d.subject_scope == "patient" and identity.attribution_ran:
            def _patient_only(ids: list[int]) -> list[int]:
                # Patient located on THIS camera ⇒ target only them.
                # Patient NOT verified anywhere ⇒ keep every body in view
                # and crop each (up to the crowd cap). Absence of a lock is
                # not evidence that the person on screen isn't the patient,
                # and a patient-scoped detector must still run for them.
                if patient_ids:
                    return [t for t in ids if t in patient_ids]
                return list(ids)

            if admissible:
                admissible = _patient_only(admissible)
            if qualifying:
                qualifying = _patient_only(qualifying)
                if not qualifying and settings.scheduler_person_requires_track:
                    skipped.append((ident, "no_qualifying_patient_track"))
                    continue

        # Target resolution: crop buckets get per-track targets (D3);
        # crowd ticks and box-less ticks fall back to full-scene.
        #
        # `crop_pool`, NOT `qualifying` (2026-08-06). Those answer two
        # different questions and were sharing one list:
        #
        #   qualifying  — should this detector RUN for this person? The
        #                 motion floor lives here, as a COST gate.
        #   crop_pool   — if it runs, WHICH BODY do we crop to?
        #
        # The floor emptying `qualifying` used to empty the targets too, and
        # because `scheduler_person_requires_track` defaults to False the
        # detector then ran anyway — with no crop, on the full frame, and its
        # answer trusted identically. That is the worst of both: the gate
        # meant to save a call instead degraded the evidence and still paid.
        #
        # Observed 08-05 21:33: track 10 was present and labelled PATIENT
        # with amount=0.0 (under the 0.03 floor), so `medication` — a
        # hand-to-mouth detector — ran on the whole room at 4 fps and
        # reported a pill being swallowed while the man held a camera.
        #
        # So: if we have decided to run a crop-bucket detector, it crops.
        # Prefer the tracks that qualified; fall back to whatever bodies are
        # admissible for this detector's subject scope.
        # The rule (Aryaman 2026-08-06): the BUCKET decides. A person-bucket
        # detector that runs is cropped — always — unless this is a crowd
        # tick. If it cannot be cropped it does not run, because answering
        # from the whole room is not a degraded answer, it is a wrong one.
        crop_pool = qualifying if qualifying else admissible
        targets: tuple[int, ...] | None = None
        if d.bucket in CROP_BUCKETS and not crowd:
            if crop_pool:
                targets = tuple(crop_pool)
            elif d.severity >= _EMERGENCY_SEVERITY:
                # No body to crop to, but a missed emergency is never worth a
                # saved call — the same line that exempts these detectors from
                # every other cost gate. Fall open to the full frame.
                #
                # This is not hypothetical: only 62.5% of ticks with a person
                # visible carried a track box on the 08-05 night, so a rule of
                # "no box ⇒ no run" would have blinded emergency detectors on
                # a third of ticks, including while someone lay on the floor —
                # precisely when a tracker is most likely to lose them.
                pass
            else:
                # Cost-managed detector with nothing to crop. It does NOT get
                # to answer from the whole room: that is how `medication`
                # reported a pill being swallowed from a full-frame 4 fps view
                # while the man held a camera (08-05 21:33).
                #
                # Covers both causes, deliberately — tracker ran and saw
                # nobody, and tracker unavailable (dead sidecar, timeout).
                # Boxes arrive FROM the identity sidecar, so its death takes
                # cropping with it; `identity_client`'s "a dead sidecar must
                # never silence detection" is honoured by the emergency branch
                # above, which is where that doctrine actually matters.
                skipped.append((ident, "no_croppable_target"))
                continue
        due.append(DueDetector(
            spec=entry.spec, dims=d, target_track_ids=targets,
            windowed=windowed,
        ))

    return DueSet(due=tuple(due), crowd_fallback=crowd, skipped=tuple(skipped))
