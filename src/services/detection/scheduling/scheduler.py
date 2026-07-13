"""L1 scheduler — which detectors are due this tick, against which
targets (detection v2, P3 T0).

ONE pure function evaluated once per detector row. All behavior comes
from the row's effective dimensions (`dims.resolve_dims`) plus the
tick context — never from per-detector code, which is exactly why
customs scale: once the P5 compiler fills a custom's columns, the
scheduler cannot tell it from a built-in.

Parity contract (D2, Aryaman 2026-07-13): at default settings (floors
0.0, sample k=1, presence not required) every non-scheduled enabled
detector is due on every tick — identical coverage to today's
monolithic call. The cost inversion is a settings change at soak.

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
from src.services.reid.body_pipeline import LABEL_OTHER, LABEL_PATIENT

if TYPE_CHECKING:
    from src.services.detection.prompt import DetectorSpec
    from src.services.detection.signals.tick_signals import TickSignals
    from src.services.reid.inference import IdentityResult
    from src.settings import Settings


@dataclass(frozen=True)
class ScheduledDetector:
    """One loadout entry: the compiled spec + its effective dims."""

    spec: "DetectorSpec"
    dims: EffectiveDims


@dataclass(frozen=True)
class DueDetector:
    """A detector that runs this tick.

    `target_track_ids` is None for scene-level artifacts (full frame /
    zone); a tuple of track ids means the packer builds ONE CROP CALL
    PER TRACK (D3 — per-person, never a union box).
    """

    spec: "DetectorSpec"
    dims: EffectiveDims
    target_track_ids: tuple[int, ...] | None


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
) -> DueSet:
    """Resolve the due set for one tick. Pure: no I/O, no clock."""
    tracks = identity.tracks
    n_tracks = len(tracks) if tracks else 0
    crowd = tracks is not None and n_tracks > settings.scheduler_crowd_fallback_tracks

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

        # Scheduled-shape detectors keep their existing separate
        # runtime (the 60 s scheduler loop) until P4 dissolves the
        # shape; running them per-tick too would double-fire.
        if d.shape == "scheduled" or d.duty_cycle == "scheduled_only":
            skipped.append((ident, "scheduled_runtime"))
            continue

        if d.duty_cycle == "sampled" and tick_index % settings.scheduler_sample_every_k:
            skipped.append((ident, "sampled_off_tick"))
            continue

        # ── ENVIRONMENT branch: no person needed, attribution
        # irrelevant, gated only by env change (parity floor 0.0
        # never skips). With no tracks, env_motion is whole-frame —
        # still the right reading for this branch.
        if d.evidence_branch == "environment":
            if signals.env_motion < settings.scheduler_env_motion_floor:
                skipped.append((ident, "env_floor"))
                continue
            due.append(DueDetector(spec=entry.spec, dims=d, target_track_ids=None))
            continue

        # ── PERSON branches (person_motion / person_still) ────────
        # Presence gate. Only when the operator opted in
        # (requires_track=True) AND the tracker positively saw nobody.
        # tracks is None (tracking didn't run) always fails open.
        if settings.scheduler_person_requires_track and tracks is not None and n_tracks == 0:
            skipped.append((ident, "no_person"))
            continue

        # Qualifying tracks. person_motion applies the per-track
        # motion floor; person_still is presence-only (a still person
        # is the whole point of that branch).
        qualifying: list[int] | None = None
        if tracks:
            if d.evidence_branch == "person_motion":
                floor = settings.scheduler_person_motion_floor
                qualifying = [
                    t.track_id for t in tracks
                    if amount_by_track.get(t.track_id, 0.0) >= floor
                ]
            else:
                qualifying = [t.track_id for t in tracks]
            if not qualifying and settings.scheduler_person_requires_track:
                skipped.append((ident, "below_motion_floor"))
                continue

        # Subject filter — the fixed rule. Suppression only on
        # verified-absence; target filtering only removes
        # verified-OTHER tracks, never UNKNOWN ones.
        if d.subject_scope == "patient" and identity.attribution_ran:
            if (
                identity.has_only_verified_other_tracks
                and not identity.patient_in_clip
            ):
                skipped.append((ident, "verified_not_patient"))
                continue
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
        due.append(DueDetector(spec=entry.spec, dims=d, target_track_ids=targets))

    return DueSet(due=tuple(due), crowd_fallback=crowd, skipped=tuple(skipped))
