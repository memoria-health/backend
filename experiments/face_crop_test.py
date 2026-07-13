"""FACE-CROP test — lip-smacking automatisms, gate #1 before implementation.

Face-scale micro-rapid signals (lip smacking ≈ 1–3 Hz — under the 8fps Nyquist
limit; the open question is RESOLUTION, answered here). Face located per frame
with the prod InsightFace stack (buffalo_l, same as attribution), EMA-smoothed
crop at 2× the face box so mouth/jaw context survives.

Arms (all 8fps, gemini-3-flash-preview per the RAPID tier finding):
  FC  face-crop clean
  FO  face-crop + motion overlay   (face-scale overlay is unmeasured — RAPID's
                                    overlay-only win was body-scale; measure it)

Clips in ~/Desktop/eval-set/face_crop/, 8fps captures. Truth by prefix:
  lip_*   lip-smacking TP        neg_*  confounder (talking, chewing, yawning,
                                        drinking, coughing)

Both detectors of the session ride one prompt? No — this session is ONE
detector (lip_smacking); confounders are FP takes, per the certification
pattern. Discrimination = TP fire rate − FP fire rate.

Usage:  PYTHONPATH=. uv run python experiments/face_crop_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

from experiments.collar_recall_test import _apply_overlay, _even, _mux, _retry
from experiments.rapid_motion_test import _call
from src.services.detection.prompt import DetectorSpec
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/face_crop")
_WORK = Path("/private/tmp/claude-501/-Users-aryamandarda-Desktop-elephant-labs-code-elephantLabs-elephant/"
             "c93eaa1b-e1da-4fcf-a4a3-0c6364d94e0c/scratchpad/face_work")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = "gemini-3-flash-preview"
ARMS = ("FC", "FO")
PAD = 2.0
EMA = 0.30
KEY = "lip_smacking_detected"

LIP_SMACKING = DetectorSpec(
    identifier="lip_smacking", name="Lip-smacking automatism", always_on=False,
    prompt_fragment=(
        "lip_smacking_detected: boolean. TRUE when the person shows a "
        "LIP-SMACKING / CHEWING AUTOMATISM: repetitive, involuntary mouth "
        "movements — lips repeatedly pressing/parting/puckering, or chewing "
        "motions of the jaw — WITHOUT food, drink, or speech, sustained across "
        "the clip, while the person is otherwise vacant or unresponsive. This "
        "is a focal-seizure sign: the movement is stereotyped (same small "
        "motion repeating), serves no purpose, and the person does not appear "
        "engaged in any activity with the mouth.\n"
        "\n"
        "The REQUIRED distinctions — name which you see in your reasoning:\n"
        "  - TALKING (FALSE): mouth movements are varied and irregular, "
        "synchronized with facial expression and engagement; often the head "
        "and eyes are directed at something/someone.\n"
        "  - EATING/CHEWING FOOD or GUM (FALSE): chewing motion but the "
        "person brought food/gum to the mouth, or holds food/a wrapper, or "
        "there are visible swallowing pauses of purposeful eating.\n"
        "  - DRINKING (FALSE): vessel raised to the mouth.\n"
        "  - YAWNING / COUGHING (FALSE): single large mouth opening, not "
        "repetitive small movements.\n"
        "If you cannot tell automatism from talking/eating, set FALSE with "
        "confidence <= 0.4.\n"
        "\n"
        "lip_smacking_confidence: number 0-1. 0 = none. 0.3-0.6 = repetitive "
        "mouth movement but purpose uncertain (must stay FALSE). 0.7+ = clear "
        "sustained purposeless stereotyped mouth movements.\n"
        "lip_smacking_reasoning: 1 short sentence NAMING the alternative you "
        "ruled out (e.g. 'repetitive purposeless lip pressing, no food or "
        "speech — automatism', 'mouth movement synchronized with animated "
        "expression — talking')."),
    schema_fields={"lip_smacking_detected": {"type": "boolean"},
                   "lip_smacking_confidence": {"type": "number"},
                   "lip_smacking_reasoning": {"type": "string"}})

EYE_TWITCH = DetectorSpec(
    identifier="eye_twitch", name="Eye twitching", always_on=False,
    prompt_fragment=(
        "eye_twitch_detected: boolean. TRUE when one or both EYELIDS show "
        "rapid, repetitive, involuntary fluttering or twitching — many small "
        "fast contractions in succession, distinct from normal blinking.\n"
        "Set FALSE for: normal blinking (regular, full closures, ~10-20/min); "
        "squinting at light or a screen; closing eyes to rest/sleep; winking; "
        "rubbing the eyes. If you cannot distinguish twitching from normal "
        "blinking at this resolution, set FALSE with confidence <= 0.4 and "
        "SAY SO in your reasoning.\n"
        "eye_twitch_confidence: number 0-1. 0.7+ = clear repeated fluttering "
        "beyond normal blinking.\n"
        "eye_twitch_reasoning: 1 short sentence (state whether eyelid detail "
        "was resolvable)."),
    schema_fields={"eye_twitch_detected": {"type": "boolean"},
                   "eye_twitch_confidence": {"type": "number"},
                   "eye_twitch_reasoning": {"type": "string"}})

FACIAL_GRIMACING = DetectorSpec(
    identifier="facial_grimacing", name="Facial grimacing (pain)", always_on=False,
    prompt_fragment=(
        "facial_grimacing_detected: boolean. TRUE when the person's face "
        "shows a SUSTAINED PAIN expression: brows drawn together/lowered, "
        "eyes tightened or squeezed, mouth tightened, corners pulled down, "
        "possibly wincing — held or recurring across the clip (this is the "
        "facial-expression item of the PAINAD dementia pain scale).\n"
        "Set FALSE for: laughing or smiling (raised cheeks, open relaxed "
        "mouth); animated TALKING expressions that change fluidly; squinting "
        "at light/screen only (no mouth/brow pain pattern); a single brief "
        "frown; neutral or sleepy face. If pain vs expression is uncertain, "
        "set FALSE with confidence <= 0.4.\n"
        "facial_grimacing_confidence: number 0-1. 0.7+ = clear sustained "
        "pain-pattern expression.\n"
        "facial_grimacing_reasoning: 1 short sentence NAMING the expression "
        "you saw (pain pattern / laughing / talking / squint / neutral)."),
    schema_fields={"facial_grimacing_detected": {"type": "boolean"},
                   "facial_grimacing_confidence": {"type": "number"},
                   "facial_grimacing_reasoning": {"type": "string"}})

SPECS = [LIP_SMACKING, EYE_TWITCH, FACIAL_GRIMACING]   # exactly the ≤3/call limit

OVERLAY_NOTE_FACE = (
    "The video has a semi-transparent RED MOTION TINT overlaid on areas that "
    "moved since the previous frame (more red = more motion). It is NOT blood "
    "or a real object. Red concentrated around the MOUTH/JAW that pulses in "
    "place across many frames indicates repetitive mouth movement. Judge WHAT "
    "the movement is from the face beneath the tint. Red alone is never "
    "sufficient evidence."
)


def _track_face(clip: str):
    """Per-frame face box via prod InsightFace (largest face), EMA-smoothed
    center, fixed crop size = 90th-pct box × PAD. Falls back to the previous
    box on missed frames."""
    from src.services.reid.embeddings import get_face_app
    app = get_face_app(get_settings().reid_insightface_model)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    boxes, last = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        faces = app.get(frame)
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            x1, y1, x2, y2 = f.bbox
            last = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        boxes.append(last)
    cap.release()
    seed = next((b for b in boxes if b is not None), None)
    if seed is None:
        raise RuntimeError(f"no face found in {clip}")
    boxes = [b if b is not None else seed for b in boxes]
    ws = [b[2] for b in boxes]
    side = _even(min(W, H, int(np.percentile(ws, 90) * PAD * 2)))  # square-ish
    smooth, sx, sy = [], boxes[0][0], boxes[0][1]
    for cx, cy, _w, _h in boxes:
        sx, sy = (1 - EMA) * sx + EMA * cx, (1 - EMA) * sy + EMA * cy
        smooth.append((sx, sy))
    return smooth, side, W, H


def _crop_face(src: str, out: str, geom) -> None:
    smooth, side, W, H = geom
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (side, side))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = smooth[min(i, len(smooth) - 1)]
        x = max(0, min(W - side, int(cx - side / 2)))
        y = max(0, min(H - side, int(cy - side / 2)))
        vw.write(frame[y:y + side, x:x + side])
        i += 1
    cap.release()
    vw.release()
    _mux(tmp, src, out)


def _build(clip: Path):
    stem = clip.stem
    geom = _track_face(str(clip))
    natovl = str(_WORK / f"{stem}_natovl.mp4")
    _apply_overlay(str(clip), natovl)
    fc = str(_WORK / f"{stem}_fc.mp4")
    fo = str(_WORK / f"{stem}_fo.mp4")
    _crop_face(str(clip), fc, geom)
    _crop_face(natovl, fo, geom)
    return {"FC": (Path(fc).read_bytes(), ""),
            "FO": (Path(fo).read_bytes(), OVERLAY_NOTE_FACE),
            "side": geom[1]}


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s = get_settings().model_copy(update={"gemini_model": MODEL,
                                          "motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    if not clips:
        sys.exit(f"no clips in {_D} (expected lip_*/neg_* .mp4 at 8fps)")
    print(f"FACE-CROP  model={MODEL}  N={N}  clips={[c.stem for c in clips]}\n")

    KEYS = ("lip_smacking_detected", "eye_twitch_detected",
            "facial_grimacing_detected")
    TRUTHS = ("lip", "eye", "grim", "neg")
    OWN = {"lip_smacking_detected": "lip", "eye_twitch_detected": "eye",
           "facial_grimacing_detected": "grim"}

    def truth_of(stem: str) -> str:
        for p in TRUTHS:
            if stem.startswith(p):
                return p
        raise ValueError(f"clip {stem}: name must start with lip_/eye_/grim_/neg_")

    fires = {a: {t: {k: 0 for k in KEYS} for t in TRUTHS} for a in ARMS}
    denom = {t: 0 for t in TRUTHS}
    for clip in clips:
        truth = truth_of(clip.stem)
        denom[truth] += N
        built = _build(clip)
        for arm in ARMS:
            vid, note = built[arm]
            res = await asyncio.gather(*(
                _retry(lambda: _call(vid, SPECS, note, s))
                for _ in range(N)))
            fields = [r.detector_fields for r in res if r is not None]
            rate = {k: sum(1 for f in fields if bool(f.get(k))) for k in KEYS}
            for k in KEYS:
                fires[arm][truth][k] += rate[k]
            print(f"  {clip.stem} [{truth}] {arm} (crop={built['side']}px): "
                  f"lip={rate[KEYS[0]]}/{len(fields)} "
                  f"eye={rate[KEYS[1]]}/{len(fields)} "
                  f"grim={rate[KEYS[2]]}/{len(fields)}")
        print()

    print("═══ DISCRIMINATION (per arm × detector) ═══")
    for arm in ARMS:
        for k in KEYS:
            own = OWN[k]
            others = [t for t in TRUTHS if t != own]
            rec = fires[arm][own][k] / denom[own] if denom[own] else 0.0
            fp_n = sum(fires[arm][t][k] for t in others)
            fp_d = sum(denom[t] for t in others)
            fp = fp_n / fp_d if fp_d else 0.0
            print(f"  {arm} {k.replace('_detected',''):<17} "
                  f"recall={rec:.0%}  fp={fp:.0%}  discrimination={rec - fp:+.0%}")


if __name__ == "__main__":
    asyncio.run(main())
