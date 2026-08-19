"""Structured tracker-eval trace — one JSONL file, one source of truth.

The eval rig (SAM3 vs SAM3+EdgeTAM) needs every TP/FP decision point
captured so a failure can be attributed to the exact stage that caused
it. This module is that tap: it appends newline-delimited JSON records
that the soak dashboard renders (nested camera -> tick -> track) and
`tools/trace_tracks.py` replays as a per-track timeline.

Record types (all carry ts=capture-time, cam, tick):
  trk      one per track per frame — box, label, origin, body/face scores
  idevent  only on a change — BOUND/REACQUIRE/SWAP/UNBOUND + cause + score
  harvest  a pool write — the pool-poison tap (a swap that seeds a guest)
  latency  per-tick per-stage wall-clock — capture->decode->track->face->body->e2e

OFF by default (`reid_trace_enabled=False`) -> every emit early-returns,
zero overhead in prod. Enabled only for the eval rig.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_lock = threading.Lock()
_enabled = False
_path = "out/track_trace.jsonl"

# Per-tick context (cam, tick, arm, capture_ts) set once by label_clip and
# read by every emit, so call sites only pass record-specific fields. A
# contextvar is async-safe AND propagates into the pipeline thread that
# asyncio.to_thread spawns (where the ledger runs), so IDEVENTs there
# inherit the same tick key without threading it through every signature.
_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "reid_trace_ctx", default=None,
)


def configure(*, enabled: bool, path: str | None = None) -> None:
    """Called once at startup from settings. Idempotent."""
    global _enabled, _path
    _enabled = bool(enabled)
    if path:
        _path = path
    if _enabled:
        try:
            os.makedirs(os.path.dirname(_path) or ".", exist_ok=True)
        except OSError:
            log.exception("reid.trace: could not create %s", _path)


def enabled() -> bool:
    return _enabled


def reset(*, cam: str, kind: str) -> None:
    """Worker teardown / track re-subscribe — this WIPES the in-memory
    ledger, so identity continuity breaks for reasons unrelated to the
    tracker. Emitted outside any tick ctx (the worker is being recreated).
    The dashboard flags any run containing one as contaminated."""
    if not _enabled:
        return
    _emit({"type": "reset", "ts": time.time(), "cam": cam, "kind": kind})


def begin_tick(*, cam: str, tick: int, arm: str, capture_ts: float) -> None:
    """Stamp the current tick. Every emit until the next begin_tick
    inherits (cam, tick, arm, capture_ts). No-op cost when disabled."""
    if not _enabled:
        return
    _ctx.set({"cam": cam, "tick": tick, "arm": arm, "capture_ts": capture_ts})


def _base() -> dict | None:
    c = _ctx.get()
    if c is None:
        return None
    return {"ts": c["capture_ts"], "cam": c["cam"], "tick": c["tick"]}


def _emit(rec: dict[str, Any]) -> None:
    if not _enabled:
        return
    try:
        line = json.dumps(rec, separators=(",", ":"))
        with _lock, open(_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 — a trace write must NEVER break the pipeline
        pass


def trk(
    *, tid: int, frame: int, box: list[float] | tuple, color: str,
    label: str, origin: str = "?", stitch_iou: float | None = None,
    body_cos: float | None = None, face_px: float | None = None,
    face_score: float | None = None, face_lock: bool | None = None,
) -> None:
    b = _base()
    if b is None:
        return
    b.update({
        "type": "trk", "tid": tid, "frame": frame,
        "box": [round(float(v), 1) for v in box] if box else None,
        "color": color, "label": label, "origin": origin,
        "stitch_iou": None if stitch_iou is None else round(float(stitch_iou), 3),
        "body_cos": None if body_cos is None else round(float(body_cos), 3),
        "face_px": None if face_px is None else round(float(face_px), 1),
        "face_score": None if face_score is None else round(float(face_score), 3),
        "face_lock": face_lock,
    })
    _emit(b)


def body_score(
    *, scores: dict[str, float], verdict: str, tid: int | None = None,
) -> None:
    """Every body match attempt — the real top-3 cosines against each
    covered patient's pool, incl. no-matches. This is the 'stream the
    scores and see what wins out' tap: the full per-tick score
    distribution.

    `scores` is {patient_id: top3_mean}, one entry per patient the
    camera covers. It replaced a fixed patient/other pair when the
    gallery stopped keying pools by PATIENT/OTHER (2026-08-20) — a
    camera can watch more than one resident, so the number of scores
    per attempt is a property of the camera, not a constant.
    """
    b = _base()
    if b is None:
        return
    b.update({
        "type": "body",
        "scores": {k: round(float(v), 3) for k, v in scores.items()},
        "verdict": verdict, "tid": tid,
    })
    _emit(b)


def idevent(
    *, tid: int, event: str, cause: str, score: float | None = None,
    gap_s: float | None = None, prev_tid: int | None = None,
    challenger: str | None = None,
) -> None:
    b = _base()
    if b is None:
        return
    b.update({
        "type": "idevent", "tid": tid, "event": event, "cause": cause,
        "score": None if score is None else round(float(score), 3),
        "gap_s": None if gap_s is None else round(float(gap_s), 1),
        "prev_tid": prev_tid, "challenger": challenger,
    })
    _emit(b)


def harvest(
    *, tid: int, pool: str, score: float,
    source_box: list[float] | tuple | None = None,
) -> None:
    b = _base()
    if b is None:
        return
    b.update({
        "type": "harvest", "tid": tid, "pool": pool,
        "score": round(float(score), 3),
        "source_box": [round(float(v), 1) for v in source_box] if source_box else None,
    })
    _emit(b)


def latency(*, frames: int, e2e_ms: float, stages: dict[str, float] | None = None) -> None:
    b = _base()
    if b is None:
        return
    c = _ctx.get() or {}
    b.update({
        "type": "latency", "arm": c.get("arm", "?"), "frames": frames,
        "e2e_ms": round(float(e2e_ms), 1),
    })
    for k, v in (stages or {}).items():
        b[k] = round(float(v), 1)
    _emit(b)
