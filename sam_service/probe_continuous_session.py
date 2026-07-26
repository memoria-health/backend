"""Validate continuous.CameraSession: STABLE track ids across simulated
ticks (overlapping windows), through both carry AND re-ground (reground_every=2
so tick0=ground, tick1=carry, tick2=ground-remap, tick3=carry)."""
import glob  # noqa: F401
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace/sam3")
sys.path.insert(0, "/workspace/nemara")
sys.path.insert(0, "/workspace/edgetam")
from sam_service import continuous, sam_track  # noqa: E402

CLIP = sys.argv[1] if len(sys.argv) > 1 else \
    "/workspace/nemara/out/ab_clips/concat30s_clean.mp4"
NF = 36
cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
idxs = np.linspace(0, max(0, total - 1), NF).astype(int)
frames = []
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
print(f"clip={CLIP.split('/')[-1]} frames={len(frames)}")

sam_pred = sam_track.get_sam("/workspace/nemara/models/sam3.1_multiplex.pt")
sess = continuous.get_session("test-cam", reground_every=2)

WIN, OV = 12, 2  # overlapping windows -> clean carry seed
step = WIN - OV
ticks, s = [], 0
while s < len(frames) and len(ticks) < 4:
    ticks.append(frames[s:s + WIN])
    s += step

report = []
for k, tk in enumerate(ticks):
    out = sess.observe(tk, person_thr=0.4, sam_pred=sam_pred)
    ids = sorted(out[max(out)].keys()) if out else []
    boxes = {
        tid: [round(v) for v in out[max(out)][tid][0]] for tid in ids
    } if out else {}
    report.append({"tick": k, "ids": ids, "boxes": boxes})

print("CONTINUOUS_TEST " + json.dumps(report))
