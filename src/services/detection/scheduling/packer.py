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
  - bucket=rapid members pack as SCENE: when the P0 rapid flag is ON,
    split_rapid removes them before the scheduler ever sees them (the
    validated solo lane owns them); when it's OFF, riding the wide
    lite call is exactly today's behavior — parity, not regression.
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

    @property
    def plan_key(self) -> str:
        """Stable log/debug label."""
        tgt = "scene" if self.target_track_id is None else f"track{self.target_track_id}"
        return f"{self.crop}:{self.frames}{self.fps}:{self.channel}:{self.tier}:{tgt}"


# (frames, fps, channel, tier, crop-when-targeted). The scene tuple is
# also the degraded shape for person-bucket members with no target.
_RECIPES: dict[str, tuple[str, int, str, str, str]] = {
    "scene":       ("video", 4, "clean",   "lite",  "full"),
    "person":      ("video", 4, "clean",   "lite",  "person"),
    "fine_motion": ("video", 8, "clean",   "flash", "person"),
    "rapid":       ("video", 8, "overlay", "flash", "person"),
    "face":        ("video", 8, "clean",   "flash", "face"),
    "snapshot":    ("frame", 1, "clean",   "lite",  "zone"),
}

_SCENE_KEY = ("video", 4, "clean", "lite", "full", None)


def _group_key(
    d: "DueDetector", target: int | None,
) -> tuple[str, int, str, str, str, int | None]:
    """Final recipe group for one (detector, target) pairing."""
    bucket = d.dims.bucket
    # Rapid members reaching the packer mean the P0 lane isn't running
    # them (flag off) — ride the wide call, exactly like today.
    if bucket == "rapid":
        return _SCENE_KEY
    # SNAPSHOT rides the wide call until per-camera zone configs exist
    # (P6): the still recipe was validated with ZONE CROPS; a full-
    # frame still measurably FAILS object checks (phone-vs-remote 100%
    # FP, plate 0/10 — spec §1.4), so packing one would be strictly
    # worse than today. When zones land, this becomes a zone-crop
    # frame call.
    if bucket == "snapshot":
        return _SCENE_KEY
    if bucket not in _RECIPES:
        return _SCENE_KEY  # unknown bucket: never brick a tick — go wide
    frames, fps, channel, tier, crop = _RECIPES[bucket]
    if target is None:
        # Degraded (no box): full-frame. PERSON's degraded recipe is
        # byte-identical to SCENE ⇒ same group. FINE-MOTION / FACE
        # keep their fps/tier in their own full-frame call.
        if bucket == "person":
            return _SCENE_KEY
        crop = "full"
    return (frames, fps, channel, tier, crop, target)


def pack_calls(due_set: "DueSet", *, settings: "Settings") -> tuple[CallPlan, ...]:
    """Pack the due set into call plans. Deterministic: groups ordered
    scene-first then by max severity; members severity-desc; overflow
    chunks split severity-first so the highest-stakes detectors ride
    the first call of a group."""
    groups: dict[tuple, list["DueDetector"]] = {}
    for d in due_set.due:
        targets: tuple[int | None, ...] = (
            d.target_track_ids if d.target_track_ids else (None,)
        )
        for target in targets:
            groups.setdefault(_group_key(d, target), []).append(d)

    plans: list[CallPlan] = []
    for key, members in groups.items():
        frames, fps, channel, tier, crop, target = key
        members = sorted(members, key=lambda m: -m.dims.severity)
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
            ))

    # Scene call(s) first, then by descending top severity, then by
    # plan_key for full determinism.
    def _order(p: CallPlan) -> tuple:
        is_scene = (
            p.crop == "full" and p.fps == 4 and p.tier == "lite"
            and p.target_track_id is None
        )
        top = max((m.dims.severity for m in p.detectors), default=0)
        return (0 if is_scene else 1, -top, p.plan_key)

    return tuple(sorted(plans, key=_order))
