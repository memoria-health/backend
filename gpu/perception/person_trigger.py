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

import contextlib
import logging
import os
import threading
from typing import Optional

log = logging.getLogger("perception")

_WEIGHTS = os.environ.get("NEMARA_TRIGGER_WEIGHTS", "yolo26x.pt")
_IMGSZ = int(os.environ.get("NEMARA_TRIGGER_IMGSZ", "1280"))
_CONF = float(os.environ.get("NEMARA_TRIGGER_CONF", "0.45"))

_model = None
_dead = False

# Arm C (2026-08-16 process-global falsification, see server._startup):
# this YOLO instance is the one model the predictor pool did NOT
# duplicate — every concurrent pass still runs through it. Serializing
# it isolates whether ultralytics' predictor carries the unsafe state.
_YOLO_SERIALIZE = os.environ.get(
    "NEMARA_EXP_YOLO_LOCK", "",
).strip() in ("1", "true")
_YOLO_LOCK = threading.Lock()


def _yolo_gate():
    return _YOLO_LOCK if _YOLO_SERIALIZE else contextlib.nullcontext()


def _load():
    """Lazy one-time model load, shared by boxes and masks."""
    global _model
    if _model is None:
        from ultralytics import YOLO

        _model = YOLO(_WEIGHTS)
        log.info(
            "trigger.loaded weights=%s task=%s imgsz=%d conf=%.2f",
            _WEIGHTS, getattr(_model, "task", "?"), _IMGSZ, _CONF,
        )
    return _model


def person_boxes(bgr) -> "Optional[list]":
    """Person boxes [(x1,y1,x2,y2,conf), ...] on one BGR frame, or
    None if the detector is unavailable (callers treat None as 'no
    signal'). Feeds the reground trigger's counts."""
    global _dead
    if _dead:
        return None
    try:
        with _yolo_gate():
            res = _load().predict(
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


def person_masks(bgr) -> "Optional[list]":
    """Person masks (FULL-res bool arrays via retina head) from seg
    weights, or None when there is no signal — detector dead, or the
    loaded weights are det-only (no seg head). None means 'fall back
    to SAM3'; an EMPTY LIST means 'zero people' and is authoritative.
    Mask source for NEMARA_GROUNDER=yoloseg (2026-07-25): masks are
    the matching currency that survives adjacency; the 07-24 box
    trial failed on currency, not detection."""
    global _dead
    if _dead:
        return None
    try:
        model = _load()
        if getattr(model, "task", "") != "segment":
            return None
        with _yolo_gate():
            res = model.predict(
                bgr, imgsz=_IMGSZ, conf=_CONF, classes=[0],
                verbose=False, device=0, retina_masks=True,
            )
        m = res[0].masks
        if m is None:  # zero detections, NOT a missing seg head
            return []
        return [(mm > 0.5) for mm in m.data.cpu().numpy()]
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
