"""L2 packer — due set → provider call plans (detection v2, P3 T1).

Pure function: takes the scheduler's due set and emits CallPlans, each
one exactly one provider request: one artifact recipe + the detectors
riding it. No I/O — artifact building and provider routing are the
executor's job (T2/T3).

Packing rules (D3/D4, Aryaman 2026-07-13):
  - The wide/SCENE call is UNCAPPED — load it up (today's parity; the
    caregiver's full detector set already rides one call in prod).
  - Fine-grained crop buckets (PERSON / FINE-MOTION / RAPID / FACE)
    and SNAPSHOT hold the measured ≤3 cap, chunked severity-first.
  - Crop buckets pack per (bucket, track): ONE CALL PER PERSON, never
    a union box. A detector targeting two tracks appears in two calls.
  - A crop-bucket detector with NO target (crowd fallback, no boxes)
    degrades to full-frame. For PERSON that makes its recipe identical
    to SCENE (4 fps / clean / lite / full frame), so it merges into
    the scene call rather than buying its own; FINE-MOTION and FACE
    keep their own full-frame call (8 fps / flash — not mergeable).
  - bucket=rapid is a first-class crop bucket (Ledger row 5, Aryaman
    2026-07-14 — the P0 standalone lane is retired): per-person
    overlay/flash calls at rapid fps. With no target (no boxes, crowd
    fallback) it keeps its own full-frame OVERLAY call — the P0 lane's
    no-box fallback — never a clean-clip ride.
  - needs_audio: if ANY member needs audio the CALL needs audio — the
    executor must land it on a native-video provider or mark the tick
    degraded.

Recipe vocabulary (locked, spec §1.4): frames ("video" | "frame"),
fps, channel ("clean" | "overlay"), tier ("lite" | "flash"), crop
("full" | "person" | "face" | "zone").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.detection.scheduling.scheduler import DueDetector, DueSet
    from src.settings import Settings

# Per-call member cap by FINAL recipe group. None = uncapped (D4).
_SCENE_KEY_CAP: int | None = None
_CROP_CAP = 3


@dataclass(frozen=True)
class CallPlan:
    """One provider request the executor will run."""

    frames: str                 # "video" | "frame"
    fps: int                    # 4 | 8 | 1 (frame)
    channel: str                # "clean" | "overlay"
    tier: str                   # "lite" | "flash"
    crop: str                   # "full" | "person" | "face" | "zone"
    target_track_id: int | None
    detectors: tuple["DueDetector", ...]
    needs_audio: bool
    # when_scope subgroup (2026-07-14): windowed detectors never share
    # a call with always-on ones, even when the recipes match.
    windowed: bool = False

    @property
    def plan_key(self) -> str:
        """Stable log/debug label."""
        tgt = "scene" if self.target_track_id is None else f"track{self.target_track_id}"
        base = f"{self.crop}:{self.frames}{self.fps}:{self.channel}:{self.tier}:{tgt}"
        return f"{base}:win" if self.windowed else base


# (frames, fps, channel, tier, crop-when-targeted). The scene tuple is
# also the degraded shape for person-bucket members with no target.
_RECIPES: dict[str, tuple[str, int, str, str, str]] = {
    "scene":       ("video", 4, "clean",   "lite",  "full"),
    "person":      ("video", 4, "clean",   "lite",  "person"),
    "fine_motion": ("video", 8, "clean",   "flash", "person"),
    "rapid":       ("video", 8, "overlay", "flash", "person"),
    "face":        ("video", 8, "clean",   "flash", "face"),
}

_SCENE_KEY = ("video", 4, "clean", "lite", "full", None, False)


def _group_key(
    d: "DueDetector", target: int | None,
) -> tuple[str, int, str, str, str, int | None, bool]:
    """Final recipe group for one (detector, target) pairing.

    2026-07-19 (Aryaman): windowed detectors FOLD into the same call
    as always-on ones — the contract is 3 media classes and no extra
    call shapes. The old when_scope subgroup split sent the identical
    clip twice per tick (`track1` + `track1:win`, ~1/3 of occupied
    cost) for zero semantic gain: when_scope eligibility is enforced
    upstream (a detector outside its window never reaches the packer),
    and the executor's only windowed dependence (`_is_scene`) is
    unaffected. `win` stays in the tuple shape as a constant so key
    comparisons and tests keep their arity."""
    bucket = d.dims.bucket
    win = False
    # (snapshot bucket retired 2026-07-18: it always rode the wide call
    # anyway — full-frame stills measurably failed object checks
    # (phone-vs-remote 100% FP, plate 0/10, spec §1.4) and stills stay
    # dead per the experiment record. resolve_dims normalizes legacy
    # 'snapshot' rows to 'scene'; the unknown-bucket fallback below
    # covers any straggler.)
    if bucket not in _RECIPES:
        # unknown bucket: never brick a tick — go wide
        return _SCENE_KEY[:6] + (win,)
    frames, fps, channel, tier, crop = _RECIPES[bucket]
    if target is None:
        # Degraded (no box): full-frame. PERSON's degraded recipe is
        # byte-identical to SCENE ⇒ same group. FINE-MOTION / RAPID /
        # FACE keep their fps/channel/tier in their own full-frame call
        # (RAPID full-frame = the P0 lane's exact no-box fallback).
        if bucket == "person":
            return _SCENE_KEY[:6] + (win,)
        crop = "full"
    return (frames, fps, channel, tier, crop, target, win)


def pack_calls(due_set: "DueSet", *, settings: "Settings") -> tuple[CallPlan, ...]:
    """Pack the due set into call plans. Deterministic: groups ordered
    scene-first then by max severity; members severity-desc; overflow
    chunks split severity-first so the highest-stakes detectors ride
    the first call of a group.

    INVARIANT: a scene call is ALWAYS emitted, even with zero scene
    members — the always-on people_present block makes an empty-
    detector wide call meaningful (it drives the presence state
    machine), which is exactly what flag-off does when every detector
    is toggled off. The executor anchors the tick's shared block on it
    (T3/T4 review F1)."""
    groups: dict[tuple, list["DueDetector"]] = {_SCENE_KEY: []}
    seen: dict[tuple, set[str]] = {_SCENE_KEY: set()}
    for d in due_set.due:
        targets: tuple[int | None, ...] = (
            d.target_track_ids if d.target_track_ids else (None,)
        )
        for target in targets:
            key = _group_key(d, target)
            members = groups.setdefault(key, [])
            idents = seen.setdefault(key, set())
            # A multi-target detector whose targets all collapse to
            # the same group (rapid/snapshot → scene) must appear in
            # that call exactly once (T3/T4 review F4 — the ledger
            # must not list phantom members).
            if d.spec.identifier in idents:
                continue
            idents.add(d.spec.identifier)
            members.append(d)

    plans: list[CallPlan] = []
    for key, members in groups.items():
        frames, fps, channel, tier, crop, target, win = key
        members = sorted(members, key=lambda m: -m.dims.severity)
        # Only the always-on scene group is uncapped; a WINDOWED wide
        # group holds the measured crop cap — windowed members are
        # typically micro-evidence (medication), exactly the dilution-
        # sensitive class.
        cap = _SCENE_KEY_CAP if key == _SCENE_KEY else _CROP_CAP
        chunks = (
            [members] if cap is None
            else [members[i:i + cap] for i in range(0, len(members), cap)]
        )
        for chunk in chunks:
            plans.append(CallPlan(
                frames=frames, fps=fps, channel=channel, tier=tier,
                crop=crop, target_track_id=target,
                detectors=tuple(chunk),
                needs_audio=any(m.dims.needs_audio for m in chunk),
                windowed=win,
            ))

    # Scene call(s) first, then by descending top severity, then by
    # plan_key for full determinism. A windowed wide call is NOT the
    # scene anchor — the executor pins the tick's shared block on the
    # always-on scene call only.
    def _order(p: CallPlan) -> tuple:
        is_scene = (
            p.crop == "full" and p.fps == 4 and p.tier == "lite"
            and p.target_track_id is None and not p.windowed
        )
        top = max((m.dims.severity for m in p.detectors), default=0)
        return (0 if is_scene else 1, -top, p.plan_key)

    return tuple(sorted(plans, key=_order))
