"""Cheap per-tick person counter — the reground TRIGGER (2026-07-24).

Problem it solves: a person can only (re)appear at a SAM3 reground, and
regrounds land every ~40-60s of wall time at current tick sizes — a
50s blind spot for new entries (measured live: the uncle encounter).

Role: YOLO26 counts persons on each carry tick's newest frame. A count
DISAGREEMENT with the carry is the only signal — it forces a SAM3
reground next tick. The trigger never creates, kills, or moves tracks;
SAM3 remains the sole authority on existence and masks.

Model choice (researched 2026-07-24): YOLO26x — Ultralytics' production
flagship. NMS-free end-to-end head → stable counts (no duplicate-box
flicker, and counts are our whole signal); STAL small-object assignment
→ recall on distant persons; 57.5 COCO mAP. Cost is irrelevant at one
frame per ~7s tick on the A40, so we buy maximum FP robustness.

Fail-open: any import/download/inference error permanently disables the
trigger for the process and the engine behaves exactly as before.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("sam_service")

_WEIGHTS = os.environ.get("NEMARA_TRIGGER_WEIGHTS", "yolo26x.pt")
_IMGSZ = int(os.environ.get("NEMARA_TRIGGER_IMGSZ", "1280"))
_CONF = float(os.environ.get("NEMARA_TRIGGER_CONF", "0.45"))

_model = None
_dead = False


def person_boxes(bgr) -> "Optional[list]":
    """Person boxes [(x1,y1,x2,y2,conf), ...] on one BGR frame, or
    None if the detector is unavailable (callers treat None as 'no
    signal'). Used by the trigger (counts) and, since 2026-07-24, by
    YOLO-grounding (NEMARA_GROUNDER=yolo) as the box source that seeds
    EdgeTAM — SAM3's fresh-look monopoly was the 30-50s new-person
    latency."""
    global _model, _dead
    if _dead:
        return None
    try:
        if _model is None:
            from ultralytics import YOLO

            _model = YOLO(_WEIGHTS)
            log.info(
                "trigger.loaded weights=%s imgsz=%d conf=%.2f",
                _WEIGHTS, _IMGSZ, _CONF,
            )
        res = _model.predict(
            bgr, imgsz=_IMGSZ, conf=_CONF, classes=[0],
            verbose=False, device=0,
        )
        out = []
        for b in res[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out
    except Exception:  # noqa: BLE001 — fail-open by design
        _dead = True
        log.exception("trigger.dead — person trigger disabled (fail-open)")
        return None


def count_persons(bgr) -> Optional[int]:
    """Person count on one BGR frame, or None if unavailable."""
    boxes = person_boxes(bgr)
    return None if boxes is None else len(boxes)


def should_force_reground(
    dets: Optional[int], live: int, under_streak: int,
) -> tuple[bool, int]:
    """Pure decision: (force_reground, new_under_streak).

    dets > live  → someone new is visible: force IMMEDIATELY (this is
                   the 50s-blind-spot fix; a false fire costs one ground
                   tick, ~12s GPU).
    dets < live  → possible exit/phantom: require 2 CONSECUTIVE ticks
                   (a detector missing a seated person for one tick must
                   not burn grounds every cycle).
    dets == live or dets is None → no signal, streak resets/holds.
    """
    if dets is None:
        return False, under_streak
    if dets > live:
        return True, 0
    if dets < live:
        streak = under_streak + 1
        if streak >= 2:
            return True, 0
        return False, streak
    return False, 0
