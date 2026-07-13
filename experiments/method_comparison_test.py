"""3-way METHOD comparison on ONE medication clip (recall + precision).

The question: for the medication detector, which preprocessing method wins?
Three arms, N runs each, holding model/detector-set constant (prod's 7):

  A  current   — native full scene + motion overlay, VIDEO       (prod today)
  B  trackcrop — motion overlay FIRST, THEN YOLO track-crop, VIDEO
                 (motion is a frame-diff, so overlaying the *stationary* full
                  frame before cropping keeps the tint on REAL subject motion
                  instead of the crop window's pan — Aryaman's ordering fix)
  C  stills    — 3 sharpest, temporally-spread frames as HIGH-RES STILLS,
                 describe-then-decide (no overlay, no video motion — pure
                 perception at ~1120 tok/still vs ~280 tok/video-frame)

Metric = medication_taken rate over N runs.
  - Run on a TRUE clip (you actually taking a pill)  -> higher = better (RECALL)
  - Run on the benign FP clip we already have        -> lower  = better (PRECISION)

Usage:
  PYTHONPATH=. uv run python experiments/method_comparison_test.py <clip.mp4> [N]

Drop your real medication clip at e.g.
  ~/Downloads/memoria-crop-experiment/med_true.mp4
and point this at it. Intermediates are written next to the input clip so you
can eyeball exactly what each arm fed Gemini.
"""

import asyncio
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types as genai_types
from ultralytics import YOLO

from src.services.detection.catalog import by_key
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.prompt import DetectorSpec, build_prompt_and_schema
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

_BACKEND = Path(__file__).resolve().parents[1]
CLIP = str(Path(sys.argv[1]).resolve())
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WEIGHTS = str(_BACKEND / "yolo11s.pt")
_OUT = Path(CLIP).parent
# The 7 detectors that were live on this patient in prod. Held CONSTANT across
# all three arms so the ONLY variable is the preprocessing method.
DETECTOR_KEYS = ["calling_for_help", "emergency", "fall", "agitation",
                 "medication", "choking", "seizure"]
CROP_PAD = 1.5     # crop size = 90th-pct person box * this
EMA = 0.30         # center smoothing (lower = smoother/laggier)
N_STILLS = 3       # sharpest frames for arm C


def _even(v: int) -> int:
    return v - (v % 2)


# ── arm B geometry: track on CLEAN frames, apply to ANY source clip ──────────

def _track_geometry(clip: str, sample_every: int = 2):
    """Per-frame largest-person centroid (from CLEAN pixels) + crop size."""
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    centers: list[tuple[float, float] | None] = []
    sizes: list[tuple[float, float]] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = None
        if i % sample_every == 0:
            best, area = None, 0.0
            for r in model.predict(frame, classes=[0], verbose=False):
                if r.boxes is not None:
                    for b in r.boxes.xyxy.cpu().numpy():
                        a = (b[2] - b[0]) * (b[3] - b[1])
                        if a > area:
                            area, best = a, b
            if best is not None:
                c = ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2)
                sizes.append((best[2] - best[0], best[3] - best[1]))
        centers.append(c)
        i += 1
    cap.release()
    if not sizes:
        raise RuntimeError("YOLO found no person — cannot crop")
    sw = _even(min(W, int(np.percentile([s[0] for s in sizes], 90) * CROP_PAD)))
    sh = _even(min(H, int(np.percentile([s[1] for s in sizes], 90) * CROP_PAD)))
    # forward-fill gaps, backfill the lead, EMA-smooth the center path.
    filled, last = [], None
    for c in centers:
        if c is not None:
            last = c
        filled.append(last)
    seed = next((c for c in filled if c is not None), (W / 2, H / 2))
    filled = [c if c is not None else seed for c in filled]
    smooth, sx, sy = [], filled[0][0], filled[0][1]
    for cx, cy in filled:
        sx, sy = (1 - EMA) * sx + EMA * cx, (1 - EMA) * sy + EMA * cy
        smooth.append((sx, sy))
    return smooth, sw, sh, W, H


def _apply_crop(src: str, out: str, geom) -> None:
    """Crop `src` with precomputed geometry (so the SAME window that was
    tracked on the clean clip is applied to the OVERLAID clip)."""
    smooth, sw, sh, W, H = geom
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 4
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sw, sh))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = smooth[min(i, len(smooth) - 1)]
        x = max(0, min(W - sw, int(cx - sw / 2)))
        y = max(0, min(H - sh, int(cy - sh / 2)))
        vw.write(frame[y:y + sh, x:x + sw])
        i += 1
    cap.release()
    vw.release()
    _mux_audio(tmp, src, out)


def _apply_overlay(clip: str, out: str) -> None:
    """Prod red-tint overlay via the real `overlay_motion`, on FULL frames."""
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 4
    snaps: list[FrameSnapshot] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = _even(bgr.shape[0]), _even(bgr.shape[1])
        i420 = cv2.cvtColor(bgr[:h, :w], cv2.COLOR_BGR2YUV_I420)
        snaps.append(FrameSnapshot(w, h, i420.tobytes(), float(len(snaps))))
    cap.release()
    over = overlay_motion(snaps)
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (over[0].width, over[0].height))
    for f in over:
        arr = np.frombuffer(f.data, np.uint8).reshape(f.height * 3 // 2, f.width)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420))
    vw.release()
    _mux_audio(tmp, clip, out)


def _mux_audio(video_only: str, audio_src: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_only, "-i", audio_src, "-map", "0:v",
         "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a",
         "aac", "-shortest", out],
        check=True, capture_output=True)
    Path(video_only).unlink(missing_ok=True)


# ── arm C: sharpest, temporally-spread stills ────────────────────────────────

def _sharpest_stills(clip: str, k: int) -> list[bytes]:
    """Split the clip into k temporal segments, take the Laplacian-sharpest
    frame from each (spread coverage, no motion blur), return HIGH-quality JPEG."""
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded for stills")
    picks, seg = [], max(1, len(frames) // k)
    for s in range(0, len(frames), seg):
        window = frames[s:s + seg]
        best = max(window, key=lambda f: cv2.Laplacian(
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        ok, buf = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            picks.append(buf.tobytes())
        if len(picks) >= k:
            break
    return picks


def _map_type(t: str):
    return {"boolean": genai_types.Type.BOOLEAN,
            "number": genai_types.Type.NUMBER,
            "string": genai_types.Type.STRING}[t]


async def gemini_stills(images: list[bytes], specs, settings):
    """Arm C: same prompt+schema as prod (description-first == describe-then-
    decide via field order), but fed HIGH-RES STILLS instead of video."""
    built = build_prompt_and_schema(specs, motion_overlay=False)
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT, required=built.schema_required,
        properties={n: genai_types.Schema(type=_map_type(f["type"]),
                                           description=f.get("description"))
                    for n, f in built.schema_properties.items()})
    parts = [genai_types.Part(inline_data=genai_types.Blob(
        data=img, mime_type="image/jpeg")) for img in images]
    parts.append(genai_types.Part(text=built.prompt))
    client = genai.Client(api_key=settings.gemini_api_key)
    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,
            media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH))
    raw = (resp.text or "").strip()
    return parse_and_validate(raw, provider="gemini",
                              required=built.schema_required,
                              properties=built.schema_properties)


# ── shared ───────────────────────────────────────────────────────────────────

def _specs() -> list[DetectorSpec]:
    return [DetectorSpec(identifier=e.key, name=e.name,
                         prompt_fragment=e.prompt_fragment,
                         schema_fields=e.schema_json, always_on=False)
            for e in (by_key(k) for k in DETECTOR_KEYS)]


async def _run_arm(label: str, caller) -> None:
    hits = 0
    per: Counter = Counter()
    for i in range(N):
        try:
            df = (await caller()).detector_fields
            taken = bool(df.get("medication_taken"))
            hits += taken
            fired = [k for k in DETECTOR_KEYS
                     if df.get(by_key(k).detected_key) is True]
            per.update(fired)
            print(f"  {label} run {i + 1:2d}: medication_taken={taken}  "
                  f"conf={df.get('medication_confidence')}  fired={fired or '—'}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {label} run {i + 1:2d}: ERROR {ex}")
    print(f">>> {label}: medication_taken in {hits}/{N} runs; all-fires={dict(per)}\n")


async def main() -> None:
    s_on = get_settings().model_copy(update={"motion_overlay_enabled": True})
    s_off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    specs = _specs()
    print(f"clip={CLIP}  model={s_on.gemini_model}  "
          f"media_res={s_on.gemini_media_resolution!r}  fps={s_on.gemini_video_fps}  "
          f"detectors={len(DETECTOR_KEYS)}  N={N}\n")

    a = str(_OUT / "mc_A_native_overlay.mp4")
    b = str(_OUT / "mc_B_trackcrop_overlay.mp4")
    d = str(_OUT / "mc_D_trackcrop_clean.mp4")

    print("arm A: overlay(native)…")
    _apply_overlay(CLIP, a)
    print("arm B/D: track-crop (SAME window: B on overlaid clip, D on clean clip)…")
    geom = _track_geometry(CLIP)          # track on CLEAN pixels
    _apply_crop(a, b, geom)               # B: crop the OVERLAID clip
    _apply_crop(CLIP, d, geom)            # D: crop the CLEAN clip (overlay is the ONLY diff vs B)
    print(f"  crop size = {geom[1]}x{geom[2]}")
    print("arm C: sharpest high-res stills…")
    stills = _sharpest_stills(CLIP, N_STILLS)
    print(f"  {len(stills)} stills, {[len(x) // 1024 for x in stills]} KB each\n")

    await _run_arm("A current    ", lambda: gemini_provider(
        mp4=Path(a).read_bytes(), detectors=specs, log_tag="A",
        settings=s_on, patient_in_clip=False))
    await _run_arm("B crop+ovl   ", lambda: gemini_provider(
        mp4=Path(b).read_bytes(), detectors=specs, log_tag="B",
        settings=s_on, patient_in_clip=False))
    await _run_arm("D crop-noovl ", lambda: gemini_provider(
        mp4=Path(d).read_bytes(), detectors=specs, log_tag="D",
        settings=s_off, patient_in_clip=False))
    await _run_arm("C stills     ", lambda: gemini_stills(stills, specs, s_on))


if __name__ == "__main__":
    asyncio.run(main())
