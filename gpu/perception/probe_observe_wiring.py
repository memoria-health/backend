"""End-to-end wiring test: call the WIRED _observe_sync twice (two
overlapping ticks, same camera) and verify the observe-v2 payload carries
STABLE track ids across ticks. face_cap=0 skips the face pass (orthogonal).
Imports server from /workspace/nemara_edge (hardlink copy) so the live
server on :8000 is never touched."""
import os
import sys

sys.path.insert(0, "/workspace/sam3")
sys.path.insert(0, "/workspace/nemara_edge")
sys.path.insert(0, "/workspace/edgetam")
os.environ.setdefault(
    "NEMARA_SAM_CKPT", "/workspace/nemara/models/sam3.1_multiplex.pt",
)

import cv2
import numpy as np

from perception import server  # noqa: E402

CLIP = "/workspace/nemara/out/ab_clips/concat30s_clean.mp4"
cap = cv2.VideoCapture(CLIP)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
idxs = np.linspace(0, total - 1, 24).astype(int)
jpegs = []
for k, fi in enumerate(idxs):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
    ok, fr = cap.read()
    if not ok:
        break
    _ok, buf = cv2.imencode(".jpg", fr)
    jpegs.append((k, buf.tobytes()))
cap.release()

t1 = jpegs[0:14]
t2 = jpegs[10:24]  # overlaps t1 by ~4 frames
p1 = server._observe_sync(t1, 0.4, 5, 0, False, "camA")
p2 = server._observe_sync(t2, 0.4, 5, 0, False, "camA")


def tids(p):
    return sorted(t["track_id"] for t in p.get("tracks", []))


print(
    "WIRING_TEST tick1_ids=%s tick2_ids=%s tracks1=%d tracks2=%d "
    "sam1_ms=%s sam2_ms=%s" % (
        tids(p1), tids(p2), len(p1.get("tracks", [])),
        len(p2.get("tracks", [])),
        p1.get("timings_ms", {}).get("sam"),
        p2.get("timings_ms", {}).get("sam"),
    )
)
