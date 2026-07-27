"""ONE-SHOT PROBE (2026-07-21) — not part of the serving path.

Answers two questions for the track-anchored identity design review:

  B. Seeded-identity mechanics: does the sam3 multiplex predictor
     accept SAM2-style point prompts with CALLER-CHOSEN obj_ids, do
     those ids survive propagation, and can they coexist with the
     "person" text concept in one session?
  C. Display-veto: do screen concepts ("television, laptop, or phone
     screen", ...) segment the laptop in the 8:44 screen-face incident
     frames, and do the phantom person/face boxes fall INSIDE the
     screen region while the real patient falls OUTSIDE?

Run on the pod:
  cd /workspace/nemara && python perception/probe_continuity.py \
      out/ab_clips/probe_frames /workspace/probe_out

Writes results.json + viz jpgs to the output dir. Delete after the
design lands (or keep as a reference probe).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, "/workspace/sam3")
sys.path.insert(0, "/workspace/nemara")

from perception import sam_track  # noqa: E402

CKPT = os.environ.get(
    "NEMARA_SAM_CKPT", "/workspace/nemara/models/sam3.1_multiplex.pt",
)
SCREEN_PROMPTS = [
    "television, laptop, or phone screen",
    "laptop computer",
    "glowing electronic display screen",
]


def _mask_np(m):
    m = sam_track._np(m)
    if m is not None and m.ndim == 3:
        m = m[0]
    return m


def parse(outputs, thr=0.3):
    """-> [(obj_id, prob, bbox_xyxy_px)] using the serving parser's
    mask-derived bbox convention."""
    dets = []
    for tid, bbox, _m, p in sam_track.parse_frame(outputs, thr):
        dets.append((int(tid), round(float(p), 3), [round(v, 1) for v in bbox]))
    return dets


class Session:
    """Minimal session ctx over a list of BGR frames (temp mp4)."""

    def __init__(self, pred, frames):
        self.pred = pred
        h, w = frames[0].shape[:2]
        fd, self.path = tempfile.mkstemp(suffix=".mp4", dir="/dev/shm")
        os.close(fd)
        wr = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (w, h),
        )
        for f in frames:
            wr.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        wr.release()
        self.sid = pred.handle_request(
            request=dict(type="start_session", resource_path=self.path),
        )["session_id"]

    def prompt(self, **kw):
        return self.pred.handle_request(
            request=dict(type="add_prompt", session_id=self.sid, **kw),
        )

    def propagate(self, start=0):
        per_frame = {}
        for res in self.pred.handle_stream_request(request=dict(
            type="propagate_in_video", session_id=self.sid,
            propagation_direction="forward", start_frame_index=start,
        )):
            idx = int(res.get("frame_index", res.get("frame_idx")))
            per_frame[idx] = parse(res.get("outputs", res))
        return per_frame

    def close(self):
        try:
            self.pred.handle_request(request=dict(
                type="close_session", session_id=self.sid,
            ))
        finally:
            try:
                os.unlink(self.path)
            except OSError:
                pass


def centers_of(dets):
    return {
        tid: [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
        for tid, _p, b in dets
    }


def contain_frac(inner, outer):
    """area(inner ∩ outer) / area(inner)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    a = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return round(iw * ih / a, 3)


def main() -> None:
    frames_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    names = [os.path.basename(p) for p in paths]
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    R: dict = {"frames": names, "experiments": {}}

    import torch

    pred = sam_track.get_sam(CKPT)
    autocast = torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
    )
    with autocast, torch.inference_mode():
        # ── E1: concept-only baseline over the sequence ─────────────
        t0 = time.time()
        try:
            s = Session(pred, frames)
            s.prompt(frame_index=0, text="person")
            e1 = s.propagate(0)
            s.close()
            R["experiments"]["E1_concept_only"] = {
                "ok": True, "secs": round(time.time() - t0, 1),
                "per_frame": {str(k): v for k, v in sorted(e1.items())},
            }
        except Exception:
            e1 = {}
            R["experiments"]["E1_concept_only"] = {
                "ok": False, "err": traceback.format_exc(limit=3),
            }

        # Seed geometry: person dets on the 2-track frame (index 2 =
        # the 15:44:10 park frame) from E1, largest two by area.
        seed_frame = 2 if len(frames) > 2 else 0
        seeds = sorted(
            e1.get(seed_frame, []),
            key=lambda d: (d[2][2] - d[2][0]) * (d[2][3] - d[2][1]),
            reverse=True,
        )[:2]
        h, w = frames[0].shape[:2]

        # ── E2: SAM2 point seeds with CHOSEN obj_ids, no text ───────
        t0 = time.time()
        try:
            s = Session(pred, frames)
            claimed = []
            for want_id, (_tid, _p, b) in zip((101, 102), seeds):
                cx, cy = (b[0] + b[2]) / 2.0 / w, (b[1] + b[3]) / 2.0 / h
                s.prompt(
                    frame_index=seed_frame, points=[[cx, cy]],
                    point_labels=[1], obj_id=want_id,
                )
                claimed.append(want_id)
            e2 = s.propagate(seed_frame)
            s.close()
            ids_seen = sorted({t for v in e2.values() for t, _, _ in v})
            R["experiments"]["E2_point_seeds_chosen_ids"] = {
                "ok": True, "secs": round(time.time() - t0, 1),
                "claimed_ids": claimed, "ids_seen": ids_seen,
                "ids_preserved": all(i in ids_seen for i in claimed),
                "frames_with_all_claimed": sum(
                    1 for v in e2.values()
                    if all(i in {t for t, _, _ in v} for i in claimed)
                ),
                "n_frames": len(e2),
                "per_frame": {str(k): v for k, v in sorted(e2.items())},
            }
        except Exception:
            R["experiments"]["E2_point_seeds_chosen_ids"] = {
                "ok": False, "err": traceback.format_exc(limit=3),
            }

        # ── E3: coexistence — text concept THEN a point seed ────────
        t0 = time.time()
        try:
            s = Session(pred, frames)
            s.prompt(frame_index=0, text="person")
            if seeds:
                b = seeds[0][2]
                cx, cy = (b[0] + b[2]) / 2.0 / w, (b[1] + b[3]) / 2.0 / h
                s.prompt(
                    frame_index=seed_frame, points=[[cx, cy]],
                    point_labels=[1], obj_id=777,
                )
            e3 = s.propagate(0)
            s.close()
            ids_seen = sorted({t for v in e3.values() for t, _, _ in v})
            R["experiments"]["E3_text_plus_point_seed"] = {
                "ok": True, "secs": round(time.time() - t0, 1),
                "ids_seen": ids_seen,
                "seed_id_present": 777 in ids_seen,
                "per_frame_counts": {
                    str(k): len(v) for k, v in sorted(e3.items())
                },
            }
        except Exception:
            R["experiments"]["E3_text_plus_point_seed"] = {
                "ok": False, "err": traceback.format_exc(limit=3),
            }

        # ── E4: display-veto — screen concepts, single frames ───────
        veto = {}
        for fi in range(len(frames)):
            persons = e1.get(fi, [])
            frame_res = {"persons": persons, "screens": {}}
            try:
                s = Session(pred, [frames[fi]])
                for prompt in SCREEN_PROMPTS:
                    res = s.prompt(frame_index=0, text=prompt)
                    outputs = (
                        res.get("outputs") if isinstance(res, dict) else res[1]
                    )
                    screens = parse(outputs, thr=0.3)
                    frame_res["screens"][prompt] = {
                        "dets": screens,
                        "person_containment": [
                            {
                                "person_id": pid,
                                "max_contain": max(
                                    [
                                        contain_frac(pb, sb)
                                        for _sid, _sp, sb in screens
                                    ] or [0.0],
                                ),
                            }
                            for pid, _pp, pb in persons
                        ],
                    }
                s.close()
            except Exception:
                frame_res["err"] = traceback.format_exc(limit=3)
            veto[names[fi]] = frame_res
            # viz: screens (cyan) + persons (yellow) for the first prompt
            try:
                im = frames[fi].copy()
                first = veto[names[fi]]["screens"].get(SCREEN_PROMPTS[0], {})
                for _sid, sp, b in first.get("dets", []):
                    cv2.rectangle(
                        im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                        (255, 255, 0), 3,
                    )
                    cv2.putText(
                        im, f"screen {sp}", (int(b[0]), int(b[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                    )
                for pid, _pp, b in persons:
                    cv2.rectangle(
                        im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                        (0, 255, 255), 2,
                    )
                cv2.imwrite(
                    os.path.join(out_dir, f"veto_{names[fi]}"), im,
                )
            except Exception:
                pass
        R["experiments"]["E4_display_veto"] = veto

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("PROBE_DONE", os.path.join(out_dir, "results.json"))


if __name__ == "__main__":
    main()
