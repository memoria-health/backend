"""RECALL test — collar_tug (custom micro-slow detector), 4 real events, 6 arms.

Generalization test for the crop-clean rule beyond medication: same arms as
med_recall_test.py, target = COLLAR_TUG from experiments/custom_detectors.py.
TRUE positives (Aryaman tugging his shirt collar), so higher = better.

  A  native+ovl   (collar only)      D  crop-clean fanout (collar only)
  B  crop+ovl     (collar only)      C  stills            (collar only)
  E  native+ovl   (collar + 4 catalog detectors — crowded)
  F  crop-clean   (collar + 4 — does crop rescue the crowded prompt again?)

Usage:  PYTHONPATH=. uv run python experiments/collar_recall_test.py [N]
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

from experiments.custom_detectors import COLLAR_TUG
from src.services.detection.catalog import by_key
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.prompt import DetectorSpec, build_prompt_and_schema
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

_BACKEND = Path(__file__).resolve().parents[1]
WEIGHTS = str(_BACKEND / "yolo11s.pt")
_D = Path("/Users/aryamandarda/Desktop/eval-set/collar_tug")
_WORK = Path("/private/tmp/claude-501/-Users-aryamandarda-Desktop-elephant-labs-code-elephantLabs-elephant/"
             "c93eaa1b-e1da-4fcf-a4a3-0c6364d94e0c/scratchpad/collar_work")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_PERSONS = 2
CROP_PAD = 1.5
EMA = 0.30
CONCURRENCY = 3
TARGET_KEY = "collar_tug_detected"

ARMS = ("A", "D", "B", "C", "E", "F")

# 1 representative clip per event (all are confirmed TPs — per-clip rate is
# true recall). Same picks as the previous run for comparability.
EVENTS = [
    ("E1_living",  ["pos_1_c_living.mp4"]),
    ("E2_living",  ["pos_2_a_living.mp4"]),
    ("E3_living",  ["pos_3_a_living.mp4"]),
    ("E4_bedroom", ["pos_4_a_bedroom.mp4"]),
]

_sem = asyncio.Semaphore(CONCURRENCY)


def _even(v: int) -> int:
    return v - (v % 2)


def _specs_solo() -> list[DetectorSpec]:
    return [COLLAR_TUG]


def _specs_multi() -> list[DetectorSpec]:
    cat = [DetectorSpec(identifier=e.key, name=e.name,
                        prompt_fragment=e.prompt_fragment,
                        schema_fields=e.schema_json, always_on=False)
           for e in (by_key(k) for k in
                     ("medication", "fall", "agitation", "choking"))]
    return [COLLAR_TUG] + cat


# ── preprocessing (same as med harness) ─────────────────────────────────────

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


def _box_iou(a, b) -> float:
    """(cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _cluster_tracks(tracks: dict[int, dict]) -> list[dict]:
    """Track IDs are NOT people: YOLO emits duplicate boxes on one person
    (simultaneous, heavily overlapping) and fragments on occlusion. Merge
    tracks into person-clusters by spatial overlap:
      - shared frames + high mean IoU        -> duplicate boxes, same person
      - disjoint frames + near at boundary   -> fragments, same person
    (E4 bedroom bug: top-2-by-coverage picked two duplicate boxes of the
    same person and dropped the real second person ranked 3rd by 3%.)"""
    order = sorted(tracks, key=lambda t: -len(tracks[t]))
    clusters: list[dict] = []
    for tid in order:
        tr = tracks[tid]
        placed = False
        for cl in clusters:
            shared = set(tr) & set(cl)
            if shared:
                mean_iou = sum(_box_iou(tr[f], cl[f]) for f in shared) / len(shared)
                if mean_iou > 0.35:
                    for f, b in tr.items():
                        cl.setdefault(f, b)      # keep cluster's box on conflicts
                    placed = True
                    break
            else:
                # temporally disjoint: same person if spatially near at the gap
                tf, cf = max(tr), max(cl)
                a = tr[min(tr)] if min(tr) > cf else tr[tf]
                b = cl[cf] if min(tr) > cf else cl[min(cl)]
                dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                if dist < max(a[2], b[2]) * 0.75:
                    cl.update(tr)
                    placed = True
                    break
        if not placed:
            clusters.append(dict(tr))
    return sorted(clusters, key=len, reverse=True)


def _track_persons(clip: str):
    """PROD Layer-1 tracking stack (ported 2026-07-12): PersonDetector
    (yolo11s @480px, conf .4) -> boxmot BoT-SORT + OSNet appearance re-ID —
    the same detector+tracker `src/services/reid/inference.py` runs per tick.
    Replaces the hand-rolled ultralytics `model.track`, whose duplicate/
    fragmented IDs produced the E4 wrong-person crops. `_cluster_tracks`
    stays as a safety net for any residual duplicates/fragments."""
    from src.services.reid.body_detector import PersonDetector
    from src.services.reid.tracker import PersonTracker

    det = PersonDetector(model_name="yolo11s.pt", confidence=0.4)
    trk = PersonTracker(frame_rate=4)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    scale = W / 480.0
    sh_small = int(H / scale)
    tracks: dict[int, dict] = defaultdict(dict)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (480, sh_small))
        for x1, y1, x2, y2, tid, _conf, _cls, _di in trk.update(
                det.detect(small), small):
            cx, cy = (x1 + x2) / 2 * scale, (y1 + y2) / 2 * scale
            w, h = (x2 - x1) * scale, (y2 - y1) * scale
            tracks[int(tid)][i] = (cx, cy, w, h)
        i += 1
    cap.release()
    total = max(i, 1)
    clusters = _cluster_tracks(tracks)
    geoms = [_geom_from_track(cl, total, W, H)
             for cl in clusters if len(cl) >= total * 0.15][:MAX_PERSONS]
    if not geoms:
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


async def _retry(make_call, tries: int = 6):
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
        mp4=mp4, detectors=specs, log_tag="col", settings=settings,
        patient_in_clip=False))
    return None if r is None else bool(r.detector_fields.get(TARGET_KEY))


async def _still(images, specs, settings):
    r = await _retry(lambda: _gemini_stills(images, specs, settings))
    return None if r is None else bool(r.detector_fields.get(TARGET_KEY))


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
    solo, multi = _specs_solo(), _specs_multi()

    async def any_person(clips, specs, settings):
        res = await asyncio.gather(*(_vid(c, specs, settings) for c in clips))
        vals = [v for v in res if v is not None]
        return any(vals) if vals else None

    async def one_run():
        a, d, b, c, e, f = await asyncio.gather(
            _vid(built["natovl"], solo, s_on),
            any_person(built["crop_clean"], solo, s_off),
            any_person(built["crop_ovl"], solo, s_on),
            _still(built["stills"], solo, s_on),
            _vid(built["natovl"], multi, s_on),
            any_person(built["crop_clean"], multi, s_off),
        )
        return {"A": a, "D": d, "B": b, "C": c, "E": e, "F": f}

    runs = await asyncio.gather(*(one_run() for _ in range(N)))
    return {arm: [r[arm] for r in runs] for arm in ARMS}


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s_on = get_settings().model_copy(update={"motion_overlay_enabled": True})
    s_off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"target=collar_tug  model={s_on.gemini_model}  N={N}  events={len(EVENTS)}\n")

    ev_recall = {a: [] for a in ARMS}
    clip_fires = {a: 0 for a in ARMS}
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
            print(f"  {fn[:-4]} (persons={built['npersons']}):  {row}")
        ev_row = []
        for a in ARMS:
            succ = sum(1 for i in range(N)
                       if any(bool(per_clip[fn][a][i]) for fn in files))
            ev_recall[a].append(succ)
            ev_row.append(f"{a}={succ}/{N}")
        print(f"  » EVENT (any-clip):  {'  '.join(ev_row)}\n")

    labels = {"A": "A native+ovl (solo)",   "D": "D crop-clean (solo)",
              "B": "B crop+ovl   (solo)",   "C": "C stills     (solo)",
              "E": "E native+ovl (multi5)", "F": "F crop-clean (multi5)"}
    print("═══ SUMMARY — EVENT RECALL (event = ANY clip fires that sample) ═══")
    for a in ARMS:
        per = " ".join(f"E{i + 1}:{x}/{N}" for i, x in enumerate(ev_recall[a]))
        tot, den = sum(ev_recall[a]), len(EVENTS) * N
        print(f"  {labels[a]:<24} {per}   mean={tot}/{den} ({round(100 * tot / den)}%)")
    print("\n── per-clip fire rate ──")
    for a in ARMS:
        print(f"  {labels[a]:<24} {clip_fires[a]}/{clip_total * N}")


if __name__ == "__main__":
    asyncio.run(main())
