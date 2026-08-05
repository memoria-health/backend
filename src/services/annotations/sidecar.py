"""The note that travels with a clip.

A recorded clip is 15 minutes of pixels. On its own it says nothing about
what happened in it. The sidecar is a JSON object written beside the MP4
in B2, and its job is to answer "what happened, and when" without opening
the video.

It is deliberately NOT a dump of model output. A blob of `detector_fields`
would mean anyone using this corpus has to know our field-naming
conventions and re-implement our firing rules to get anything out of it.
Instead the sidecar makes the determination itself:

    this detector was active from 00:12 to 06:42, peaking at 0.91

**Detectors do not all start and stop the same way**, and getting those
boundaries right is the whole job:

  * **leading / point-in-time** — fires the instant it sees the thing (a
    fall). Consecutive positive ticks are ONE occurrence, not three, so
    they merge across a short gap.
  * **state / sustained** — runs until its evidence has been absent for
    that detector's exit gap. A brief lull inside a Watching-TV session
    is not the end of it; the same gap the alerting path uses decides.
  * **duration** — the same run, plus the moment it crossed the "gone on
    too long" bar (stove left on).
  * **schedule / absence** — fires because something did NOT happen.
    There is no footage of that, and the surrounding video is not
    evidence for it, so it is marked `no_footage` and never given a span.

**Life does not fit in 15-minute boxes.** Someone starts pacing at 2:58
and the clip ends at 3:00 while they are still pacing. A stove alert
fires at 3:10 for a stove on since 2:30 — three clips back.

The answer is not to make each sidecar complete; that would mean holding
clips back until every episode closed, which for sleep means all night.
Instead each span says whether its own edges are CLOSED (we watched it
start / stop) or OPEN (it was already running when the clip began, or
still running when it ended). Pages of a book: a sentence may run across a
page break, so you number the pages and make the break visible rather
than reformatting the book. Stitching offline is then "join open ends to
the neighbouring clip".

Nothing new is computed during detection. This reads what the pipeline
already wrote to `analyses` and `events`, so there is no second source of
truth and no third write in the tick.

**A sidecar is a SNAPSHOT.** Written when the clip lands, so it can never
carry a reviewer's true/false-positive verdict — that is marked hours
later. The travelling copy is this; the labelled copy is this same builder
run again at export against the database as it stands then.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

SCHEMA_VERSION = "1"

# How close to a clip boundary a tick must sit before we will claim
# anything about what was happening there. Ticks are ~10s, so a wider gap
# means we have no evidence at the edge and say so with null rather than
# guessing.
_EDGE_MARGIN_SECONDS = 15.0

# Gap that ends a run for a detector with no exit gap of its own — the
# point-in-time shapes. Roughly two tick intervals, so a fall seen on
# three consecutive ticks is one occurrence rather than three, while two
# genuinely separate falls a minute apart stay separate.
_DEFAULT_MERGE_GAP_SECONDS = 25.0

# Alert triggers whose evidence is an ABSENCE. No footage exists of a
# thing not happening.
_ABSENCE_TRIGGERS = frozenset({"schedule"})
# Triggers whose alert moment points BACKWARD, often into an earlier clip.
_BACKWARD_TRIGGERS = frozenset({"trailing", "duration"})


@dataclass(frozen=True)
class DetectorSpec:
    """How to read one detector's verdict, and how its runs begin and end.

    The field NAMES come from `event_promotion._promotion_spec_for` and
    the exit gap from `scheduling.decision.episode_exit_gap_for`, never
    re-guessed here. Built-ins can name their verdict whatever the catalog
    says (medication's is `medication_taken`, not `medication_detected`),
    so a local convention would quietly disagree with the real pipeline
    for exactly those detectors.
    """

    identifier: str
    detected_field: str
    confidence_field: str
    name: str | None = None
    category: str | None = None
    # leading | trailing | duration | schedule
    alert_trigger: str | None = None
    is_sustained: bool = False
    # Silence that ends a run. None → `_DEFAULT_MERGE_GAP_SECONDS`.
    exit_gap_seconds: float | None = None
    # The "gone on too long" bar for duration detectors, if any.
    duration_threshold_seconds: float | None = None

    @property
    def merge_gap(self) -> float:
        if self.exit_gap_seconds is not None and self.exit_gap_seconds > 0:
            return float(self.exit_gap_seconds)
        return _DEFAULT_MERGE_GAP_SECONDS


@dataclass(frozen=True)
class ClipFacts:
    """What the recorder knows about the clip this sidecar describes."""

    key: str
    started_at: datetime
    duration_seconds: float
    caregiver_id: str
    patient_id: str
    camera_id: str
    room: str
    masked: bool = False
    # False when the recorder could not keep up, so the clip is shorter
    # than the session and offsets after the stall have shifted.
    timeline_exact: bool = True

    @property
    def ended_at(self) -> datetime:
        return self.started_at + timedelta(seconds=self.duration_seconds)


def _offset(clip: ClipFacts, at: datetime) -> float:
    """Seconds from the clip's start, clamped into the clip."""
    raw = (at - clip.started_at).total_seconds()
    return max(0.0, min(raw, clip.duration_seconds))


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fired(fields: dict[str, Any], spec: DetectorSpec) -> bool:
    """Did this detector say yes on this tick?

    ``is True`` exactly, matching `event_promotion._is_truthy_bool`. A
    string "true" or a 1 is a miswired detector; counting it here while
    the alerting path ignores it would make the sidecar contradict the
    events sitting beside it.
    """
    return fields.get(spec.detected_field) is True


def _spans_for(
    clip: ClipFacts, observations: list[Any], spec: DetectorSpec,
) -> list[dict[str, Any]]:
    """Turn this detector's positive ticks into start/stop spans.

    Positive ticks separated by less than the detector's gap belong to the
    same span; a wider silence closes it. A span touching the first or
    last tick of the clip is marked OPEN at that edge — we cannot see
    beyond the clip, and "started exactly at the boundary" and "was
    already running" must not be confused.
    """
    if spec.alert_trigger in _ABSENCE_TRIGGERS:
        return []  # absence has no footage to span

    spans: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    first_at = observations[0].captured_at if observations else None
    last_at = observations[-1].captured_at if observations else None

    for obs in observations:
        fields = getattr(obs, "detector_fields", None) or {}
        if not _fired(fields, spec):
            continue
        conf = _confidence(fields.get(spec.confidence_field))
        at = obs.captured_at
        if current is not None and (
            (at - current["_last_at"]).total_seconds() <= spec.merge_gap
        ):
            current["_last_at"] = at
            current["tick_count"] += 1
            if conf is not None:
                current["_confs"].append(conf)
            continue
        if current is not None:
            spans.append(current)
        current = {
            "_first_at": at,
            "_last_at": at,
            "tick_count": 1,
            "_confs": [conf] if conf is not None else [],
        }
    if current is not None:
        spans.append(current)

    out: list[dict[str, Any]] = []
    for span in spans:
        start, end = span["_first_at"], span["_last_at"]
        confs: list[float] = span["_confs"]
        # OPEN means the span reaches the outermost tick we have, so
        # whatever it was doing outside the clip is invisible to us.
        open_start = first_at is not None and start <= first_at
        open_end = last_at is not None and end >= last_at
        observed = (end - start).total_seconds()
        entry = {
            "identifier": spec.identifier,
            "name": spec.name,
            "category": spec.category,
            "alert_trigger": spec.alert_trigger,
            "is_sustained": spec.is_sustained,
            "started_at": _iso(start),
            "ended_at": _iso(end),
            "start_offset_seconds": round(_offset(clip, start), 3),
            "end_offset_seconds": round(_offset(clip, end), 3),
            # Time between the first and last positive tick. With an open
            # edge the real activity is at least this long, not exactly.
            "observed_seconds": round(observed, 3),
            "open_start": open_start,
            "open_end": open_end,
            "tick_count": span["tick_count"],
            "max_confidence": max(confs) if confs else None,
            "mean_confidence": (
                round(sum(confs) / len(confs), 4) if confs else None
            ),
        }
        if spec.duration_threshold_seconds:
            # Only meaningful for a closed span: an open one may already
            # have been running long before the clip started.
            entry["exceeded_duration_threshold"] = (
                observed >= spec.duration_threshold_seconds
            )
            entry["duration_threshold_seconds"] = (
                spec.duration_threshold_seconds
            )
        out.append(entry)
    return out


def _edge_state(
    clip: ClipFacts, observations: list[Any], *, at_start: bool,
) -> bool | None:
    """Was somebody there right at this edge of the clip?

    None when the nearest tick is too far from the boundary to say —
    "no evidence" and "evidence of nobody" are different claims.
    """
    if not observations:
        return None
    obs = observations[0] if at_start else observations[-1]
    boundary = clip.started_at if at_start else clip.ended_at
    if abs((obs.captured_at - boundary).total_seconds()) > _EDGE_MARGIN_SECONDS:
        return None
    return bool(obs.people_present)


def build_sidecar(
    clip: ClipFacts,
    observations: list[Any],
    alerts: list[Any],
    *,
    generated_at: datetime,
    detector_specs: list[DetectorSpec] | None = None,
) -> dict[str, Any]:
    """Assemble the sidecar for one clip.

    ``observations`` are `analyses`-shaped rows and ``alerts`` are
    `events`-shaped rows, both already filtered to this clip's window.
    Pure — no database, no storage — so the whole shape is testable with
    plain values.
    """
    observations = sorted(observations, key=lambda o: o.captured_at)
    alerts = sorted(alerts, key=lambda a: a.occurred_at)
    specs = detector_specs or []

    detections: list[dict[str, Any]] = []
    for spec in specs:
        detections.extend(_spans_for(clip, observations, spec))
    detections.sort(key=lambda d: (d["started_at"], d["identifier"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "clip": {
            "key": clip.key,
            "started_at": _iso(clip.started_at),
            "ended_at": _iso(clip.ended_at),
            "duration_seconds": round(clip.duration_seconds, 3),
            # Offsets below are only exact when this is true.
            "timeline_exact": clip.timeline_exact,
            "masked": clip.masked,
            "room": clip.room,
            "caregiver_id": clip.caregiver_id,
            "patient_id": clip.patient_id,
            "camera_id": clip.camera_id,
        },
        # THE ANSWER: what was active in this clip and between which
        # moments. Open edges point at the neighbouring clip.
        "detections": detections,
        # Alerts that fired inside this window. A detection span can exist
        # with no alert (below threshold, deduped, or not a rising edge),
        # and an alert can exist with no span (absence detectors).
        "alerts": [_alert_entry(clip, a) for a in alerts],
        "edges": {
            "people_present_at_start": _edge_state(
                clip, observations, at_start=True
            ),
            "people_present_at_end": _edge_state(
                clip, observations, at_start=False
            ),
            "observation_count": len(observations),
            "people_present_ticks": sum(
                1 for o in observations if o.people_present
            ),
            "motion_ticks": sum(1 for o in observations if o.motion_in_clip),
        },
        # The raw per-tick record, kept whole: confidence, reasoning,
        # k-of-N counters, latch and quality flags. Filtering belongs at
        # export where the audience is known; dropping it here loses it.
        "observations": [
            {
                "at": _iso(o.captured_at),
                "offset_seconds": round(_offset(clip, o.captured_at), 3),
                "people_present": bool(o.people_present),
                "motion": bool(o.motion_in_clip),
                "state_after": getattr(o, "state_after", None),
                "description": getattr(o, "description", None),
                "detectors": getattr(o, "detector_fields", None) or {},
            }
            for o in observations
        ],
    }


def _alert_entry(clip: ClipFacts, alert: Any) -> dict[str, Any]:
    trigger = getattr(alert, "alert_trigger", None)
    return {
        "at": _iso(alert.occurred_at),
        "offset_seconds": round(_offset(clip, alert.occurred_at), 3),
        "event_id": str(alert.id),
        "detector_id": (
            str(alert.detector_id) if getattr(alert, "detector_id", None) else None
        ),
        "identifier": getattr(alert, "identifier", None),
        "title": getattr(alert, "title", None),
        "category": getattr(alert, "category", None),
        "narrative": getattr(alert, "narrative", None),
        # Which moment fired: leading (the instant it was seen), trailing
        # (when it finished), duration (when it had gone on too long) or
        # schedule (a deadline passed).
        "alert_trigger": trigger,
        # No footage exists of a thing not happening; the surrounding
        # video is not evidence for it.
        "no_footage": trigger in _ABSENCE_TRIGGERS,
        # The moment this fired points backward — the footage it is about
        # may be in an earlier clip.
        "may_precede_clip": trigger in _BACKWARD_TRIGGERS,
    }


def sidecar_key(clip_key: str) -> str:
    """The sidecar's object key: the clip's, with a .json extension.

    Same prefix and filename so the pair sorts together in a bucket
    listing and either can be found from the other without an index.
    """
    if clip_key.endswith(".mp4"):
        return clip_key[: -len(".mp4")] + ".json"
    return clip_key + ".json"
