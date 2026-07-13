"""RAPID-MOTION test — convulsion vs body_shaking (shivering), 2×2 arms.

Both rapid detectors ride ONE prompt as mutual confounders on the tracked
person crop, describe-then-decide schema (motion_{region,spread,rhythm,
amplitude,voluntary} before any boolean).

Arms — {4fps, 8fps} × {clean-only, dual split-screen}:
  C4  crop 4fps clean (single video)
  C8  crop 8fps clean (single video)
  D4  crop 4fps dual  (split-screen: left clean | right overlay)
  D8  crop 8fps dual  (split-screen)

Packaging findings baked in (measured 2026-07-12 on gemini-3.1-flash-lite):
  - TWO video parts in one request COLLAPSE — the model reports "one video"
    and attends to only one channel. Dual therefore ships as ONE hstacked
    split-screen video.
  - Overlay-only calls UNDER-read motion (a whole-body-red acted convulsion
    scored "sporadic, localized"); clean 8fps crop read the same clip as
    "whole_body, continuous, involuntary". The 2×2 measures whether the
    overlay helps at all next to clean.

Clips in ~/Desktop/eval-set/rapid_motion/, captured at 8fps (4fps arms are
decimated offline). Truth from filename prefix:
  conv_*  convulsion TP    shiv_*  shivering TP    neg_*  confounder FP take

Scoring is discrimination: recall on own TPs minus fire rate everywhere else.

Usage:  PYTHONPATH=. uv run python experiments/rapid_motion_test.py [N]
"""

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types as genai_types

from experiments.collar_recall_test import (
    CROP_PAD, EMA, MAX_PERSONS, _apply_crop, _apply_overlay, _cluster_tracks,
    _even, _geom_from_track, _map_type, _mux, _retry,
)
from experiments.custom_detectors import (
    BODY_SHAKING, CONVULSION, MOTION_DESCRIPTION, MOTION_DESCRIPTION_CLEAN,
)
from src.services.detection.prompt import build_prompt_and_schema
from src.services.detection.providers._parse import parse_and_validate
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/rapid_motion")
_WORK = Path("/private/tmp/claude-501/-Users-aryamandarda-Desktop-elephant-labs-code-elephantLabs-elephant/"
             "c93eaa1b-e1da-4fcf-a4a3-0c6364d94e0c/scratchpad/rapid_work")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = ("C4", "C8", "D4", "D8")
# arm -> (fps tag, input kind)
ARM_DEF = {"C4": ("4", "clean"), "C8": ("8", "clean"),
           "D4": ("4", "sbs"),   "D8": ("8", "sbs")}
KEYS = ("convulsion_detected", "body_shaking_detected")

SPECS_DUAL = [MOTION_DESCRIPTION, CONVULSION, BODY_SHAKING]
SPECS_CLEAN = [MOTION_DESCRIPTION_CLEAN, CONVULSION, BODY_SHAKING]

DUAL_CHANNEL_NOTE = (
    "The video is a SPLIT-SCREEN: the LEFT half and the RIGHT half show the "
    "SAME moment from the SAME camera, side by side. The LEFT half is the RAW "
    "clip. The RIGHT half is the IDENTICAL clip with a semi-transparent RED "
    "MOTION TINT overlaid on the areas that moved since the previous frame. "
    "The tint marks pixel CHANGE only (more red = more motion there); it is "
    "NOT blood, a light, or any real object — never report the red itself as "
    "an object or injury. Use the RIGHT half to LOCATE and CHARACTERIZE "
    "motion (where, spread, rhythm, amplitude); use the LEFT half to judge "
    "WHAT the person is actually doing (posture, purpose, context). There is "
    "ONE person shown twice — never report two people because of the split. "
    "Red alone is never sufficient evidence for any detector."
)


# ── preprocessing ────────────────────────────────────────────────────────────

def _decimate(src: str, out: str) -> None:
    """8fps -> 4fps by dropping alternate frames (audio muxed back from src).
    The 4fps arm must be RE-derived pixels: 4fps frame diffs differ from 8fps
    diffs, so overlay is computed AFTER decimation by the caller."""
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    tmp = out + ".v.mp4"
    vw = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 2 == 0:
            if vw is None:
                vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps / 2, (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        i += 1
    cap.release()
    if vw is not None:
        vw.release()
    _mux(tmp, src, out)


def _hstack(clean: str, ovl: str, out: str) -> None:
    import subprocess
    subprocess.run(
        ["ffmpeg", "-y", "-i", clean, "-i", ovl, "-filter_complex",
         "[0:v][1:v]hstack=inputs=2", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", "-an", out],
        check=True, capture_output=True)


def _track_persons_fps(clip: str, fps: int):
    """collar_recall_test._track_persons with the tracker frame rate matched
    to the clip (BoT-SORT's motion model assumes it)."""
    from src.services.reid.body_detector import PersonDetector
    from src.services.reid.tracker import PersonTracker

    det = PersonDetector(model_name="yolo11s.pt", confidence=0.4)
    trk = PersonTracker(frame_rate=fps)
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


def _build(clip: Path):
    """Per fps variant: decimate (4 only) -> overlay at THAT rate -> per-person
    clean crop + overlay crop -> hstack split-screen. Stores clean and sbs
    bytes per person under keys '<fps>_clean' / '<fps>_sbs'."""
    stem = clip.stem
    clip4 = str(_WORK / f"{stem}_4fps.mp4")
    _decimate(str(clip), clip4)
    out = {}
    for tag, src, fps in (("8", str(clip), 8), ("4", clip4, 4)):
        natovl = str(_WORK / f"{stem}_R{tag}_natovl.mp4")
        _apply_overlay(src, natovl)
        geoms = _track_persons_fps(src, fps)
        cleans, sbss = [], []
        for pi, g in enumerate(geoms):
            cc = str(_WORK / f"{stem}_R{tag}_cc{pi}.mp4")
            co = str(_WORK / f"{stem}_R{tag}_co{pi}.mp4")
            sbs = str(_WORK / f"{stem}_R{tag}_sbs{pi}.mp4")
            _apply_crop(src, cc, g)
            _apply_crop(natovl, co, g)
            _hstack(cc, co, sbs)
            cleans.append(Path(cc).read_bytes())
            sbss.append(Path(sbs).read_bytes())
        out[f"{tag}_clean"] = cleans
        out[f"{tag}_sbs"] = sbss
        out[f"{tag}_npersons"] = len(geoms)
    return out


# ── calls (one video + one schema; specs/note vary by input kind) ───────────

async def _call(vid: bytes, specs, note: str, settings):
    built = build_prompt_and_schema(specs, motion_overlay=False)
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT, required=built.schema_required,
        properties={n: genai_types.Schema(type=_map_type(f["type"]),
                                          description=f.get("description"))
                    for n, f in built.schema_properties.items()})
    text = (note + "\n\n" + built.prompt) if note else built.prompt
    parts = [
        genai_types.Part(inline_data=genai_types.Blob(data=vid, mime_type="video/mp4")),
        genai_types.Part(text=text),
    ]
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


async def _dual(sbs: bytes, settings):
    return await _call(sbs, SPECS_DUAL, DUAL_CHANNEL_NOTE, settings)


async def _clean(cc: bytes, settings):
    return await _call(cc, SPECS_CLEAN, "", settings)


async def _run_arm(vids, kind: str, settings):
    """One sample = every person's call; per detector key, any-person OR."""
    fn = _dual if kind == "sbs" else _clean
    res = await asyncio.gather(*(_retry(lambda v=v: fn(v, settings))
                                 for v in vids))
    fields = [r.detector_fields for r in res if r is not None]
    if not fields:
        return {k: None for k in KEYS}, {}
    verdict = {k: any(bool(f.get(k)) for f in fields) for k in KEYS}
    forensic = next((f for f in fields if any(bool(f.get(k)) for k in KEYS)),
                    fields[0])
    return verdict, forensic


# ── scoring ──────────────────────────────────────────────────────────────────

def _truth(stem: str) -> str:
    for p, t in (("conv", "conv"), ("shiv", "shiv"), ("neg", "neg")):
        if stem.startswith(p):
            return t
    raise ValueError(f"clip {stem}: name must start with conv_/shiv_/neg_")


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    settings = get_settings().model_copy(update={"motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    if not clips:
        sys.exit(f"no clips in {_D} (expected conv_/shiv_/neg_ prefixed .mp4)")
    print(f"RAPID 2x2  model={settings.gemini_model}  N={N}  "
          f"clips={[c.stem for c in clips]}\n")

    fires = {a: {t: {k: 0 for k in KEYS} for t in ("conv", "shiv", "neg")}
             for a in ARMS}
    denom = {t: 0 for t in ("conv", "shiv", "neg")}
    misfires = []

    for clip in clips:
        truth = _truth(clip.stem)
        denom[truth] += N
        built = _build(clip)
        for arm in ARMS:
            fps_tag, kind = ARM_DEF[arm]
            samples = await asyncio.gather(
                *(_run_arm(built[f"{fps_tag}_{kind}"], kind, settings)
                  for _ in range(N)))
            rate = {k: sum(1 for v, _ in samples if v[k]) for k in KEYS}
            for k in KEYS:
                fires[arm][truth][k] += rate[k]
            expected = {"conv": "convulsion_detected",
                        "shiv": "body_shaking_detected", "neg": None}[truth]
            for v, f in samples:
                for k in KEYS:
                    if v[k] and k != expected and f:
                        misfires.append(
                            (clip.stem, arm, k,
                             str(f.get(k.replace('_detected', '_reasoning'), ''))[:160],
                             f.get('motion_amplitude'), f.get('motion_voluntary')))
                        break
            print(f"  {clip.stem} [{truth}] {arm} "
                  f"(persons={built[f'{fps_tag}_npersons']}):  "
                  f"conv={rate['convulsion_detected']}/{N}  "
                  f"shake={rate['body_shaking_detected']}/{N}")
        print()

    print("═══ SUMMARY — fire rate per (arm × clip-truth × detector) ═══")
    print(f"{'':16}{'conv clips':<24}{'shiv clips':<24}{'neg clips'}")
    labels = {"C4": "C4 clean 4fps", "C8": "C8 clean 8fps",
              "D4": "D4 dual  4fps", "D8": "D8 dual  8fps"}
    for arm in ARMS:
        row = [f"{labels[arm]:<16}"]
        for t in ("conv", "shiv", "neg"):
            c, s = fires[arm][t]["convulsion_detected"], fires[arm][t]["body_shaking_detected"]
            row.append(f"conv={c}/{denom[t]} shake={s}/{denom[t]}".ljust(24))
        print("".join(row))

    print("\n═══ DISCRIMINATION (recall on own TPs − fire rate elsewhere) ═══")
    for arm in ARMS:
        for key, own in (("convulsion_detected", "conv"),
                         ("body_shaking_detected", "shiv")):
            others = [t for t in ("conv", "shiv", "neg") if t != own]
            rec = fires[arm][own][key] / denom[own] if denom[own] else 0.0
            fp_n = sum(fires[arm][t][key] for t in others)
            fp_d = sum(denom[t] for t in others)
            fp = fp_n / fp_d if fp_d else 0.0
            print(f"  {labels[arm]} {key.replace('_detected',''):<13} "
                  f"recall={rec:.0%}  fp={fp:.0%}  discrimination={rec - fp:+.0%}")

    print("\n═══ INVOLUNTARY-UNION (either detector fires on a TP — safety net) ═══")
    for arm in ARMS:
        for own in ("conv", "shiv"):
            hit = max(fires[arm][own][k] for k in KEYS)  # lower bound proxy
            print(f"  {labels[arm]} {own}: strongest-key={hit}/{denom[own]}")

    if misfires:
        print("\n── misfire forensics (amplitude/voluntary from schema) ──")
        for stem, arm, k, why, amp, vol in misfires[:15]:
            print(f"  {stem} {arm} {k}: [{amp}/{vol}] {why}")


if __name__ == "__main__":
    asyncio.run(main())
