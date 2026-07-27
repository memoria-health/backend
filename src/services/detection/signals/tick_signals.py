"""Layer-1 tick signals (detection v2, P2).

One cheap, VLM-free measurement pass per tick producing the gate-lattice
signals from the v2 spec (§1.3):

  person_present / people_count   from body tracks (peak concurrent)
  patient_present                 from attribution labels (None when
                                  attribution didn't run this tick)
  motion — per track              ONE full-frame diff pass over the
                                  buffered window, reduced inside each
                                  track's swept box; two readouts per
                                  track: `amount` (coverage) + `churn`
                                  (persistence concentration — logged
                                  as a tuning dial, gates nothing)
  env_motion                      the SAME diff pass reduced OUTSIDE
                                  every person box — a door opening or
                                  a burner lighting is never masked by
                                  a still person
  face_valid                      tallest usable face across tracks,
                                  scaled to native px, vs the ~300 px
                                  FACE-bucket legibility floor

Pure computation: no I/O, no DB, and — deliberately — NO decision
thresholds besides the face floor. In P2 these numbers are stashed on
analyses rows for observability only; P3 (scheduler) is where floors
and gating live.

Deliberate split from `rapid_energy`: the P0 RAPID gate keeps its own
crop-first, per-person measurement — those floors were tuned against
the labeled convulsion eval set and must not silently change. This
module diffs the WHOLE frame once at map resolution and reduces per
region, which is cheaper for N tracks + environment but yields amounts
that are NOT 1:1 comparable with the gate's. The 24 h soak decides
whether the gate can collapse onto this map (then `rapid_energy`
becomes a reduction here). Both use the same active-pixel semantics
(diff threshold, persistence ratio) imported from rapid_energy so the
two stay in lockstep.

Coordinate spaces: track boxes arrive in the Re-ID EXTRACTED-frame
space (width = min(sample_width, native)); the diff map lives in
MAP space (native downscaled to ≤ `_MAP_WIDTH`). Boxes scale
extracted→map with one linear factor since both preserve aspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.signals.rapid_energy import (
    _DIFF_THRESHOLD,
    _MIN_FRAMES,
    _PERSIST_RATIO,
)
from src.services.reid.observations import LABEL_PATIENT

if TYPE_CHECKING:
    from src.services.reid.observations import TrackObservation

# Full-frame diff map width. Wider than rapid_energy's 160 crop width
# because a person box occupies only part of the frame here — at 320
# a typical box keeps enough pixels for a meaningful coverage readout.
_MAP_WIDTH = 320
# FACE-bucket legibility floor (Law 7): faces below ~300 native px
# lose the micro-expressions FACE detectors read (lip smacking,
# grimacing). Also drives the distance_class degradation warning.
FACE_MIN_PX_DEFAULT = 300


@dataclass(frozen=True)
class TrackMotion:
    """Motion readouts for one track, reduced inside its swept box."""

    track_id: int
    label: str
    amount: float   # active pixels / box pixels (0..1)
    churn: float    # persistent-active / active (0..1) — logged dial


@dataclass(frozen=True)
class TickSignals:
    """All Layer-1 signals for one tick. `None` = unknowable this tick
    (tracking didn't run / attribution didn't run), never "false"."""

    person_present: bool | None
    people_count: int | None
    patient_present: bool | None
    track_motion: tuple[TrackMotion, ...]
    env_motion: float
    face_valid: bool | None
    frames_used: int

    def stash_fields(self) -> dict:
        """Flat `sig_*` dict for the analyses detector_fields stash.
        Unknowable signals are omitted (absent key = didn't run),
        floats rounded so the JSONB stays lean."""
        d: dict = {
            "sig_env_motion": round(self.env_motion, 4),
            "sig_frames": self.frames_used,
        }
        if self.person_present is not None:
            d["sig_person_present"] = self.person_present
            d["sig_people_count"] = self.people_count
        if self.patient_present is not None:
            d["sig_patient_present"] = self.patient_present
        if self.face_valid is not None:
            d["sig_face_valid"] = self.face_valid
        if self.track_motion:
            d["sig_tracks"] = [
                {
                    "id": t.track_id,
                    "label": t.label,
                    "amount": round(t.amount, 4),
                    "churn": round(t.churn, 4),
                }
                for t in self.track_motion[:8]
            ]
            if len(self.track_motion) > 8:
                d["sig_tracks_truncated"] = len(self.track_motion)
        return d


def _map_plane(frame: FrameSnapshot) -> np.ndarray | None:
    """Grayscale plane → downscale to map width → denoise blur.

    Live buffer frames are JPEG (2026-07-19 JPEG-in-buffer) carrying a
    320×180 luma thumbnail — exactly map-sized, so it is used as-is.
    Raw I420 frames (tests, legacy) read the Y plane. A JPEG without a
    thumbnail is unreadable to this per-tick path (never decode full
    JPEGs here): skip. Missing this consumer was the 2026-07-19 soak
    bug where env_motion pinned at 0.0 for every tick.
    """
    if frame_codec.is_jpeg(frame.data):
        luma = frame.luma
        if len(luma) != frame_codec.LUMA_W * frame_codec.LUMA_H:
            return None
        y = np.frombuffer(luma, dtype=np.uint8).reshape(
            frame_codec.LUMA_H, frame_codec.LUMA_W
        )
    elif len(frame.data) < frame.width * frame.height:
        return None
    else:
        y = np.frombuffer(
            frame.data, dtype=np.uint8, count=frame.width * frame.height
        ).reshape(frame.height, frame.width)
    if y.shape[1] > _MAP_WIDTH:
        scale = _MAP_WIDTH / y.shape[1]
        y = cv2.resize(
            y,
            (_MAP_WIDTH, max(1, int(round(y.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.GaussianBlur(y, (3, 3), 0)


def _scaled_box(
    bbox: tuple[float, float, float, float],
    factor: float,
    map_w: int,
    map_h: int,
) -> tuple[int, int, int, int]:
    """Extracted-space box → map-space int box, clamped, ≥1 px."""
    x0 = max(0, min(int(bbox[0] * factor), map_w - 1))
    y0 = max(0, min(int(bbox[1] * factor), map_h - 1))
    x1 = max(0, min(int(round(bbox[2] * factor)), map_w))
    y1 = max(0, min(int(round(bbox[3] * factor)), map_h))
    if x1 <= x0:
        x1 = x0 + 1
    if y1 <= y0:
        y1 = y0 + 1
    return (x0, y0, x1, y1)


def compute_tick_signals(
    frames: Sequence[FrameSnapshot],
    *,
    tracks: Sequence["TrackObservation"] | None,
    attribution_ran: bool,
    sample_width: int,
    face_min_px: int = FACE_MIN_PX_DEFAULT,
) -> TickSignals:
    """Compute all Layer-1 signals for one tick.

    `frames`: the buffered window (native I420) — same snapshot the
    main VLM call sees. `tracks`: IdentityResult.tracks (None when the
    tracking pipeline didn't run). `sample_width` is
    settings.reid_inference_sample_width — used to recover the
    extracted-frame width the track boxes live in.
    """
    # ── Presence, from tracks alone ──────────────────────────────
    if tracks is None:
        person_present: bool | None = None
        people_count: int | None = None
    else:
        person_present = len(tracks) > 0
        # Peak concurrent tracks in any single extracted frame — same
        # semantic as people_count_reid, recomputed from the series so
        # this module has one input, not two.
        concurrent: dict[int, int] = {}
        for t in tracks:
            for frame_idx, _ in t.bbox_series:
                concurrent[frame_idx] = concurrent.get(frame_idx, 0) + 1
        people_count = max(concurrent.values()) if concurrent else 0

    patient_present: bool | None = None
    if tracks is not None and attribution_ran:
        patient_present = any(t.label == LABEL_PATIENT for t in tracks)

    # ── One diff pass over the window, dominant shape only ───────
    native_w = 0
    planes: list[np.ndarray] = []
    if len(frames) >= _MIN_FRAMES:
        shapes: dict[tuple[int, int], int] = {}
        for f in frames:
            shapes[(f.width, f.height)] = shapes.get((f.width, f.height), 0) + 1
        keep = max(shapes, key=shapes.get)  # type: ignore[arg-type]
        native_w = keep[0]
        planes = [
            p
            for f in frames
            if (f.width, f.height) == keep and (p := _map_plane(f)) is not None
        ]

    env_motion = 0.0
    track_motion: list[TrackMotion] = []
    # Extracted-frame width the track boxes live in. 0 = no native
    # frames this window, so no scaling (and no face_valid verdict) is
    # possible.
    extracted_w = min(sample_width, native_w) if native_w else 0
    face_px_native = 0.0
    if tracks and extracted_w > 0:
        # Native face px: extracted-space px × (native / extracted).
        face_px_native = (
            max(t.face_px_max for t in tracks) * (native_w / extracted_w)
        )

    if len(planes) >= _MIN_FRAMES:
        counts = np.zeros(planes[0].shape, dtype=np.int32)
        n_pairs = 0
        for a, b in zip(planes, planes[1:]):
            counts += cv2.absdiff(a, b) > _DIFF_THRESHOLD
            n_pairs += 1
        active = counts > 0
        # Same persistence floor rationale as rapid_energy: a floor of
        # 2 would count a translating edge as persistent.
        need = max(3, int(round(n_pairs * _PERSIST_RATIO)))
        persistent = counts >= need
        map_h, map_w = counts.shape

        person_mask = np.zeros_like(active)
        if tracks:
            factor = map_w / max(1, extracted_w)
            for t in tracks:
                union: tuple[int, int, int, int] | None = None
                for _, bbox in t.bbox_series:
                    b = _scaled_box(bbox, factor, map_w, map_h)
                    person_mask[b[1]:b[3], b[0]:b[2]] = True
                    union = b if union is None else (
                        min(union[0], b[0]), min(union[1], b[1]),
                        max(union[2], b[2]), max(union[3], b[3]),
                    )
                if union is None:
                    continue
                x0, y0, x1, y1 = union
                box_active = active[y0:y1, x0:x1]
                n_active = int(box_active.sum())
                amount = n_active / box_active.size
                churn = (
                    int(persistent[y0:y1, x0:x1].sum()) / n_active
                    if n_active else 0.0
                )
                track_motion.append(TrackMotion(
                    track_id=t.track_id, label=t.label,
                    amount=float(amount), churn=float(churn),
                ))

        outside = ~person_mask
        n_outside = int(outside.sum())
        if n_outside:
            env_motion = float(int((active & outside).sum()) / n_outside)

    face_valid: bool | None = None
    if tracks and attribution_ran and extracted_w > 0:
        # Only meaningful when face checks could run, someone was
        # actually in frame, AND we can scale to native px. "Saw
        # people, no usable face" reads as False (the FACE bucket
        # genuinely can't work this tick); an empty room stays None —
        # there is no face question to answer.
        face_valid = face_px_native >= face_min_px

    return TickSignals(
        person_present=person_present,
        people_count=people_count,
        patient_present=patient_present,
        track_motion=tuple(track_motion),
        env_motion=env_motion,
        face_valid=face_valid,
        frames_used=len(planes),
    )
