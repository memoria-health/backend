"""Unsupervised-soak event journal (Aryaman, 2026-07-24).

The dashboard's live windows are useless for a night nobody watched.
This journal records every identity-relevant EVENT — track birth,
track death, seat lock/death/move/heal — as one JSONL line plus a
snapshot JPEG of the moment, so the morning review is a scrollable
story: what happened, when, to whom, with the picture to judge it by.

Also the metrology home: seat_locked events carry confirm_latency_s
(track birth → seat lock), the number that answers "how long until we
know it's the patient" — per lock, per anchor, measured not guessed.

Files: <root>/events.jsonl + <root>/snapshots/NNNNNN_<kind>.jpg.
Append-only; survives service restarts; rate is human-scale (events,
not frames), so plain open-append is fine.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

_SNAP_W = 960          # snapshot width — evidence, not archival footage
_SNAP_QUALITY = 78

# Event → box color (BGR), matching the dashboard's event palette.
COLORS = {
    "track_birth": (248, 180, 138),   # blue
    "track_death": (166, 160, 154),   # gray
    "seat_locked": (153, 211, 52),    # green
    "seat_moved": (36, 191, 251),     # amber
    "seat_died": (113, 113, 248),     # red
    "seat_healed": (252, 132, 192),   # purple
}


def annotate(
    frame: np.ndarray, bbox, label: str, kind: str,
    context_boxes: "Optional[list]" = None,
) -> np.ndarray:
    """Copy of `frame` with the EVENT's track box + label drawn — the
    journal is image EVIDENCE (whose track was born/died/locked), not
    just a scene photo. `context_boxes` = [(tid, bbox), ...] for ALL
    other live tracks, drawn thin gray — without them a reviewer can't
    tell "not the event's track" from "not tracked at all" (live
    2026-07-25: the patient, tracked and seated, appeared boxless in
    every event image and reasonably concluded tracking had died)."""
    img = frame.copy()
    color = COLORS.get(kind, (220, 220, 220))
    for tid, cb in (context_boxes or []):
        if cb is None:
            continue
        cx1, cy1, cx2, cy2 = (int(v) for v in cb)
        cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (190, 190, 190), 1)
        cv2.putText(
            img, f"T{tid}", (cx1, max(18, cy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1,
        )
    if bbox is not None:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            img, label, (x1, max(24, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )
    else:
        cv2.putText(
            img, label, (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )
    return img


# Snapshot budget. Every journalled seat change can write a JPEG, and
# nothing ever deleted them: 7.2G of frame dumps by 2026-08-19, and the
# multi-patient seats write one entry PER RESIDENT rather than one per
# home, so the rate went up. Oldest-first eviction keeps the directory
# bounded without a cron job or an operator remembering.
_SNAP_MAX = int(os.environ.get("NEMARA_JOURNAL_SNAP_MAX", "3000"))
_LOG_MAX_BYTES = int(os.environ.get("NEMARA_JOURNAL_MAX_BYTES", str(64 << 20)))


class EventJournal:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.snaps = self.root / "snapshots"
        self.snaps.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "events.jsonl"
        self._lock = threading.Lock()
        self._seq = int(time.time()) % 10**6  # unique-enough across restarts
        self._since_gc = 0

    def _gc_locked(self) -> None:
        """Evict oldest snapshots past the cap, and roll the jsonl once.

        Runs every _SNAP_MAX//10 writes rather than every write — a
        directory listing per seat change would cost more than the JPEG.
        """
        try:
            snaps = sorted(
                self.snaps.glob("*.jpg"), key=lambda p: p.stat().st_mtime,
            )
            for p in snaps[:-_SNAP_MAX] if len(snaps) > _SNAP_MAX else []:
                p.unlink(missing_ok=True)
            if self.path.exists() and self.path.stat().st_size > _LOG_MAX_BYTES:
                # One generation only. The journal is a live debugging
                # tail, not an archive; keeping N rotations is how the
                # disk filled in the first place.
                self.path.replace(self.path.with_suffix(".jsonl.1"))
        except Exception:  # noqa: BLE001 — housekeeping must never kill a loop
            log.exception("identity.journal_gc_failed")

    def log(
        self,
        kind: str,
        *,
        cam: str,
        frame_bgr: Optional[np.ndarray] = None,
        **fields,
    ) -> None:
        """Append one event; never raises (journal must not hurt the
        pipeline — a lost snapshot beats a dead camera loop)."""
        try:
            now = time.time()
            evt = {
                "ts": round(now, 3),
                "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "kind": kind,
                "cam": cam,
                **{k: v for k, v in fields.items() if v is not None},
            }
            if frame_bgr is not None:
                with self._lock:
                    self._seq += 1
                    name = f"{self._seq:06d}_{kind}.jpg"
                img = frame_bgr
                if img.shape[1] > _SNAP_W:
                    s = _SNAP_W / img.shape[1]
                    img = cv2.resize(img, (_SNAP_W, int(img.shape[0] * s)))
                ok, jpg = cv2.imencode(
                    ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _SNAP_QUALITY],
                )
                if ok:
                    (self.snaps / name).write_bytes(jpg.tobytes())
                    evt["img"] = name
            with self._lock:
                with open(self.path, "a") as f:
                    f.write(json.dumps(evt) + "\n")
                self._since_gc += 1
                if self._since_gc >= max(50, _SNAP_MAX // 10):
                    self._since_gc = 0
                    self._gc_locked()
        except Exception:  # noqa: BLE001
            log.exception("identity.journal_failed kind=%s", kind)

    def tail(self, n: int = 200) -> list[dict]:
        try:
            lines = self.path.read_text().splitlines()[-n:]
            return [json.loads(ln) for ln in lines]
        except FileNotFoundError:
            return []
        except Exception:  # noqa: BLE001
            log.exception("identity.journal_tail_failed")
            return []
