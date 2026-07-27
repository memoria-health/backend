"""E2E smoke for the v2 perception/brain split — runs ON THE POD.

Full path under test: JPEG frames -> POST /observe (SAM track + mask
crops + GPU face worker) -> RemoteObservations -> RemoteReplayPipeline
(the UNCHANGED BodyReIDPipeline brain) -> track labels/locks — i.e.
everything the live SAM arm does except Docker + the SSH tunnel
(already proven) and carry/pool (exercised live; needs cross-window
state).

Windows of 44 and 88 frames reproduce the live worst-case frame
counts (11s @ 4fps buffer / 11s @ 8fps) from the 4fps debug clips —
frame COUNT is what drives latency, so the numbers are comparable
even though wall-span differs.

Run: PYTHONPATH=/workspace/sam3:/workspace/nemara \
     python perception/e2e_smoke.py
"""
import asyncio
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/workspace/nemara")

from src.services.reid.body_pipeline import PipelineConfig  # noqa: E402
from src.services.reid.remote_replay import RemoteReplayPipeline  # noqa: E402
from src.services.reid.sam_client import SamServiceClient  # noqa: E402


def frames_of(clip: str, *, width=1280, n=None):
    """Native-fps frames as (idx, jpeg_bytes, bgr) — mirrors the live
    path: buffer JPEGs at <=1280px wide, indices in extracted space."""
    cap = cv2.VideoCapture(clip)
    out = []
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if width and im.shape[1] > width:
            h = int(im.shape[0] * width / im.shape[1])
            im = cv2.resize(im, (width, h))
        ok2, enc = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            out.append((i, enc.tobytes(), im))
            i += 1
        if n and len(out) >= n:
            break
    cap.release()
    return out


async def run_window(client, gallery, clip, n):
    frames = frames_of(clip, n=n)
    if not frames:
        print(f"{os.path.basename(clip)}: no frames")
        return
    t0 = time.time()
    obs = await client.observe(
        [(i, b) for i, b, _ in frames],
        person_thr=0.4, face_interval=5, face_cap=8,
    )
    wire_ms = (time.time() - t0) * 1000.0
    pipe = RemoteReplayPipeline(
        observations=obs, gallery=gallery,
        config=PipelineConfig(),          # production defaults
        pre_match_callback=None,          # carry/pool exercised live
    )
    t1 = time.time()
    for i, _, im in frames:
        pipe.process_frame(im, i)
    replay_ms = (time.time() - t1) * 1000.0
    tracks = [
        (o.track_id, o.label, round(o.confidence, 2), o.face_locked)
        for o in pipe.track_observations()
    ]
    t = obs.timings_ms
    total = wire_ms + replay_ms
    print(
        f"{os.path.basename(clip):<42} n={len(frames):>3} "
        f"tracks={pipe.num_tracks} concurrent={pipe.max_concurrent_tracks} "
        f"patient={'YES' if pipe.has_any_patient_track else 'no'} "
        f"best={pipe.best_score:.2f} faces={pipe.face_count} | "
        f"TOTAL={total:.0f}ms wire={wire_ms:.0f} "
        f"(pod: dec={t.get('decode', 0):.0f} sam={t.get('sam', 0):.0f} "
        f"face={t.get('face', 0):.0f}) replay={replay_ms:.0f} | {tracks}"
    )


async def main():
    client = SamServiceClient(base_url="http://localhost:8000", timeout_s=300)
    print("health:", await client.health())
    gallery = np.load("/workspace/nemara/out/gallery_75a30940.npy")
    sims = gallery @ gallery.T
    iu = np.triu_indices(len(gallery), k=1)
    print(f"gallery(ArcFace enrollment): {gallery.shape} "
          f"intra_cos={sims[iu].mean():.2f}")
    clips = (
        sorted(glob.glob("/workspace/nemara/out/ab_clips/*_clean.mp4"))
        or sorted(glob.glob("/workspace/nemara/out/ab_clips/*.mp4"))
    )
    for clip in clips:
        for n in (44, 88):
            await run_window(client, gallery, clip, n)
    await client.aclose()


asyncio.run(main())
print("[done]")
