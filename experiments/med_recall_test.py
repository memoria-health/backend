"""RECALL test on 3 real medication events (6 clips), 5 arms.

These are TRUE positives (Aryaman actually took medication), so higher
medication_taken rate = better. Subject is far from the camera (small in a
1080p frame) — the micro-object case crop is meant to rescue.

Events (grouped by the user; a "sequence" spans consecutive 10s ticks):
  E1 living room : 233526, 233536, 233546   (3 clips)
  E2 bedroom     : 233606, 233616           (2 clips)
  E3 living room : 233636                    (1 clip)

Arms (all measure medication_taken, N runs per clip):
  A  native + overlay          medication-only          (prod-like)
  B  crop(FAN-OUT) + overlay   medication-only          overlay-first-then-crop
  D  crop(FAN-OUT) clean       medication-only          no overlay
  C  sharpest stills           medication-only          no motion, max detail
  E  native + overlay          medication + 4 detectors (fall/seizure/choking/agitation)
      -> identical input to A; isolates the CROWDED-PROMPT dilution effect

Crop arms FAN OUT: one tracking crop per person (YOLO multi-object track,
capped at 2), run the detector on EACH crop, the arm fires if ANY person's
crop fires. This is the production multi-person routing (no faceID -> fan out).

Usage:  PYTHONPATH=. uv run python experiments/med_recall_test.py [N]
"""

import asyncio
import subprocess
import sys
from collections import defaultdict
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
WEIGHTS = str(_BACKEND / "yolo11s.pt")
_D = Path("/Users/aryamandarda/Downloads/memoria-crop-experiment/med_tp_raw")
_WORK = Path("/private/tmp/claude-501/-Users-aryamandarda-Desktop-elephant-labs-code-elephantLabs-elephant/"
             "c93eaa1b-e1da-4fcf-a4a3-0c6364d94e0c/scratchpad/med_work")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MAX_PERSONS = 2
CROP_PAD = 1.5
EMA = 0.30
CONCURRENCY = 3

MED = ["medication"]
MULTI = ["medication", "fall", "seizure", "choking", "agitation"]
# A native+ovl(med) · D crop-clean(med) · B crop+ovl(med) · C stills(med)
# E native+ovl(med+4) · F crop-clean(med+4)
ARMS = ("A", "D", "B", "C", "E", "F")

EVENTS = [
    ("E1_living", ["233526_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4",
                   "233536_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4",
                   "233546_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4"]),
    ("E2_bed",    ["233606_9271358c-647f-44d8-a3ab-1b88a2044566.mp4",
                   "233616_9271358c-647f-44d8-a3ab-1b88a2044566.mp4"]),
    ("E3_living", ["233636_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4"]),
]

_sem = asyncio.Semaphore(CONCURRENCY)


def _even(v: int) -> int:
    return v - (v % 2)


# ── overlay (full frame) ────────────────────────────────────────────────────

def _apply_overlay(clip: str, out: str) -> None:
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
    _mux(tmp, clip, out)


# ── multi-person tracking crop (fan-out) ────────────────────────────────────

def _geom_from_track(track: dict, total: int, W: int, H: int):
    ws = [v[2] for v in track.values()]
    hs = [v[3] for v in track.values()]
    sw = _even(min(W, int(np.percentile(ws, 90) * CROP_PAD)))
    sh = _even(min(H, int(np.percentile(hs, 90) * CROP_PAD)))
    centers, last = [], None
    for fi in range(total):
        if fi in track:
            last = (track[fi][0], track[fi][1])
        centers.append(last)
    seed = next((c for c in centers if c is not None), (W / 2, H / 2))
    centers = [c if c is not None else seed for c in centers]
    smooth, sx, sy = [], centers[0][0], centers[0][1]
    for cx, cy in centers:
        sx, sy = (1 - EMA) * sx + EMA * cx, (1 - EMA) * sy + EMA * cy
        smooth.append((sx, sy))
    return (smooth, sw, sh, W, H)


def _track_persons(clip: str):
    """YOLO multi-object track -> one geometry per person (capped MAX_PERSONS),
    ranked by frame coverage. Falls back to a single whole-frame 'person'."""
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    tracks: dict[int, dict] = defaultdict(dict)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for r in model.track(frame, classes=[0], persist=True, verbose=False):
            if r.boxes is None or r.boxes.id is None:
                continue
            for b, tid in zip(r.boxes.xyxy.cpu().numpy(),
                              r.boxes.id.cpu().numpy()):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                tracks[int(tid)][i] = (cx, cy, b[2] - b[0], b[3] - b[1])
        i += 1
    cap.release()
    total = max(i, 1)
    ranked = sorted(tracks.items(), key=lambda kv: -len(kv[1]))
    geoms = [_geom_from_track(d, total, W, H)
             for _, d in ranked if len(d) >= total * 0.15][:MAX_PERSONS]
    if not geoms:  # nobody tracked — degrade to whole frame
        geoms = [([(W / 2, H / 2)] * total, _even(W), _even(H), W, H)]
    return geoms


def _apply_crop(src: str, out: str, geom) -> None:
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
    _mux(tmp, src, out)


def _mux(video_only: str, audio_src: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_only, "-i", audio_src, "-map", "0:v",
         "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a",
         "aac", "-shortest", out],
        check=True, capture_output=True)
    Path(video_only).unlink(missing_ok=True)


# ── stills ──────────────────────────────────────────────────────────────────

def _sharpest_stills(clip: str, k: int = 3) -> list[bytes]:
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    if not frames:
        return []
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
    return {"boolean": genai_types.Type.BOOLEAN, "number": genai_types.Type.NUMBER,
            "string": genai_types.Type.STRING}[t]


async def _gemini_stills(images, specs, settings):
    built = build_prompt_and_schema(specs, motion_overlay=False)
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT, required=built.schema_required,
        properties={n: genai_types.Schema(type=_map_type(f["type"]),
                                           description=f.get("description"))
                    for n, f in built.schema_properties.items()})
    parts = [genai_types.Part(inline_data=genai_types.Blob(data=img, mime_type="image/jpeg"))
             for img in images]
    parts.append(genai_types.Part(text=built.prompt))
    client = genai.Client(api_key=settings.gemini_api_key)
    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,
            media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH))
    return parse_and_validate((resp.text or "").strip(), provider="gemini",
                              required=built.schema_required,
                              properties=built.schema_properties)


# ── detector specs + calls ──────────────────────────────────────────────────

def _specs(keys) -> list[DetectorSpec]:
    return [DetectorSpec(identifier=e.key, name=e.name,
                         prompt_fragment=e.prompt_fragment,
                         schema_fields=e.schema_json, always_on=False)
            for e in (by_key(k) for k in keys)]


async def _retry(make_call, tries: int = 6):
    """Ride out transient Gemini 503/429 spikes with growing backoff."""
    for i in range(tries):
        try:
            async with _sem:
                return await make_call()
        except Exception as ex:  # noqa: BLE001
            if i == tries - 1:
                print(f"  !! give-up after {tries} tries: {type(ex).__name__}",
                      file=sys.stderr)
                return None
            await asyncio.sleep(3 * (i + 1))


async def _vid(mp4: bytes, specs, settings):
    r = await _retry(lambda: gemini_provider(
        mp4=mp4, detectors=specs, log_tag="rec", settings=settings,
        patient_in_clip=False))
    return None if r is None else bool(r.detector_fields.get("medication_taken"))


async def _still(images, specs, settings):
    r = await _retry(lambda: _gemini_stills(images, specs, settings))
    return None if r is None else bool(r.detector_fields.get("medication_taken"))


# ── per-clip build + run ────────────────────────────────────────────────────

async def _build(clip: Path):
    stem = clip.stem
    natovl = str(_WORK / f"{stem}_natovl.mp4")
    _apply_overlay(str(clip), natovl)
    geoms = _track_persons(str(clip))
    crop_clean, crop_ovl = [], []
    for pi, g in enumerate(geoms):
        cc = str(_WORK / f"{stem}_cc{pi}.mp4"); _apply_crop(str(clip), cc, g); crop_clean.append(cc)
        co = str(_WORK / f"{stem}_co{pi}.mp4"); _apply_crop(natovl, co, g); crop_ovl.append(co)
    return {
        "natovl": Path(natovl).read_bytes(),
        "crop_clean": [Path(p).read_bytes() for p in crop_clean],
        "crop_ovl": [Path(p).read_bytes() for p in crop_ovl],
        "stills": _sharpest_stills(str(clip)),
        "npersons": len(geoms),
    }


async def _run_clip(built, s_on, s_off):
    """N runs. Returns {arm: [Optional[bool] x N]} — the PER-RUN outcome for
    THIS clip (so main() can OR across an event's clips per sample). Fan-out
    arms fire if ANY person's crop fires that run."""
    med, multi = _specs(MED), _specs(MULTI)

    async def any_person(clips, specs, settings):
        res = await asyncio.gather(*(_vid(c, specs, settings) for c in clips))
        vals = [v for v in res if v is not None]
        return any(vals) if vals else None

    async def one_run():
        a, d, b, c, e, f = await asyncio.gather(
            _vid(built["natovl"], med, s_on),                 # A native+ovl med
            any_person(built["crop_clean"], med, s_off),      # D crop-clean med
            any_person(built["crop_ovl"], med, s_on),         # B crop+ovl med
            _still(built["stills"], med, s_on),               # C stills med
            _vid(built["natovl"], multi, s_on),               # E native+ovl med+4
            any_person(built["crop_clean"], multi, s_off),    # F crop-clean med+4
        )
        return {"A": a, "D": d, "B": b, "C": c, "E": e, "F": f}

    runs = await asyncio.gather(*(one_run() for _ in range(N)))
    return {arm: [r[arm] for r in runs] for arm in ARMS}


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s_on = get_settings().model_copy(update={"motion_overlay_enabled": True})
    s_off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"model={s_on.gemini_model}  N={N}  events={len(EVENTS)}  "
          f"arms=A(nat+ovl) B(crop+ovl,fanout) D(crop,fanout) C(stills) E(nat+ovl,MULTI)\n")

    ev_recall = {a: [] for a in ARMS}     # per-event success counts (out of N)
    clip_fires = {a: 0 for a in ARMS}     # total clip-run fires (transparency)
    clip_total = 0

    for ev_name, files in EVENTS:
        print(f"── {ev_name} ({len(files)} clip(s)) ──")
        per_clip = {}
        for fn in files:
            built = await _build(_D / fn)
            per_clip[fn] = await _run_clip(built, s_on, s_off)
            clip_total += 1
            rate = {a: sum(1 for x in per_clip[fn][a] if x) for a in ARMS}
            for a in ARMS:
                clip_fires[a] += rate[a]
            row = "  ".join(f"{a}={rate[a]}/{N}" for a in ARMS)
            print(f"  clip {fn[:6]} (persons={built['npersons']}):  {row}")
        # event-level: per-SAMPLE OR across the event's clips.
        ev_row = []
        for a in ARMS:
            succ = sum(1 for i in range(N)
                       if any(bool(per_clip[fn][a][i]) for fn in files))
            ev_recall[a].append(succ)
            ev_row.append(f"{a}={succ}/{N}")
        print(f"  » EVENT (any-clip):  {'  '.join(ev_row)}\n")

    labels = {"A": "A native+ovl (med)",   "D": "D crop-clean (med)",
              "B": "B crop+ovl   (med)",   "C": "C stills     (med)",
              "E": "E native+ovl (med+4)", "F": "F crop-clean (med+4)"}
    print("═══ SUMMARY — EVENT RECALL (event = ANY clip fires that sample) ═══")
    for a in ARMS:
        per = " ".join(f"E{i + 1}:{x}/{N}" for i, x in enumerate(ev_recall[a]))
        tot, den = sum(ev_recall[a]), len(EVENTS) * N
        print(f"  {labels[a]:<22} {per}   mean={tot}/{den} ({round(100 * tot / den)}%)")
    print("\n── per-clip fire rate (transparency, not event-OR'd) ──")
    for a in ARMS:
        print(f"  {labels[a]:<22} {clip_fires[a]}/{clip_total * N}")


if __name__ == "__main__":
    asyncio.run(main())
