"""Run labelled clips through the REAL prompt builder + provider.

The point of this tool is that it is not a reimplementation. It loads
detector rows from the database, builds specs with
`registry.detector_spec_from_row`, and calls the same provider a live
tick calls, so a number it produces is a number production would
produce. Anything it cannot borrow from production, it says so.

    uv run python tools/eval_detectors.py --detector custom_b761ccdf \
        --dir ~/Desktop/eval-set/drinking_water --mode full --repeats 3

Labels come from the filename, matching the eval-set convention:
`pos_*` is a true occurrence, `neg_*` is not. Anything else is scored
as unlabelled and reported separately rather than silently counted.

`--mode crop` approximates the scheduler's person bucket by cropping
each frame to the largest detected person before the call. It is an
APPROXIMATION: the live path crops per TRACK using the pod's boxes,
this crops per frame using local YOLO. It answers "does cropping change
the verdict", not "is production byte-identical to this".

`--repeats` exists because single runs mislead. Prompt work on
2026-08-06 was judged on single runs repeatedly and the conclusions did
not survive a second pass; the harness reports per-clip agreement so an
unstable clip is visible rather than averaged away.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection.registry import detector_spec_from_row
from src.settings import get_settings


def _label(path: Path) -> str | None:
    n = path.name.lower()
    if n.startswith("pos"):
        return "pos"
    if n.startswith("neg"):
        return "neg"
    return None


def _crop_to_person(src: Path, dst: Path) -> bool:
    """Re-encode `src` cropped to the largest person, padded.

    ONE box for the whole clip (the union of per-frame boxes) rather
    than a moving crop: a per-frame crop makes the subject jitter
    against a shifting background, which is its own confusion for a
    video model. Returns False when no person is found in any frame, so
    the caller can report the clip as un-croppable instead of silently
    evaluating the full frame and calling it a crop.
    """
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(2)
    det = YOLO("yolo11n.pt")
    cap = cv2.VideoCapture(str(src))
    boxes: list[tuple[float, float, float, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % 4 == 0:                      # every 4th frame is plenty
            r = det.predict(frame, classes=[0], conf=0.25,
                            verbose=False, imgsz=960)[0]
            for b in r.boxes:
                boxes.append(tuple(float(v) for v in b.xyxy[0].cpu().numpy()))
        idx += 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not boxes:
        return False
    arr = np.array(boxes)
    x0, y0 = arr[:, 0].min(), arr[:, 1].min()
    x1, y1 = arr[:, 2].max(), arr[:, 3].max()
    pad = 0.15 * max(x1 - x0, y1 - y0)
    x0, y0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
    x1, y1 = min(W, int(x1 + pad)), min(H, int(y1 + pad))
    w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if w < 32 or h < 32:
        return False
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"crop={w}:{h}:{x0}:{y0}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        check=True,
    )
    return True


async def _load_spec(identifier: str, bundle: bool):
    """The target spec, plus (with --bundle) every other enabled
    detector for the same patient — the loadout a live tick carries."""
    sm = get_sessionmaker()
    async with sm() as s:
        row = (await s.execute(
            select(Detector).where(Detector.identifier == identifier)
        )).scalars().first()
        if row is None:
            raise SystemExit(f"no detector row with identifier {identifier!r}")
        specs = [detector_spec_from_row(row)]
        if bundle:
            others = (await s.execute(
                select(Detector).where(
                    Detector.patient_id == row.patient_id,
                    Detector.is_on.is_(True),
                    Detector.identifier.is_not(None),
                    Detector.identifier != identifier,
                )
            )).scalars().all()
            specs += [detector_spec_from_row(o) for o in others]
        return specs, row.name, row.bucket


async def _run_clip(mp4: bytes, specs, settings, model: str | None):
    from src.services.detection.providers.gemini import gemini_provider

    res = await gemini_provider(
        mp4=mp4, detectors=specs, log_tag="eval",
        settings=settings, model=model,
    )
    return res


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", required=True, help="detector identifier")
    ap.add_argument("--dir", required=True, help="directory of .mp4 clips")
    ap.add_argument("--mode", choices=["full", "crop"], default="full")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default=None, help="override the VLM model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="write raw results as JSON")
    ap.add_argument(
        "--state-field", default="",
        help="report a CATEGORICAL field (e.g. activity_state) instead of "
             "scoring a boolean. Activity is not a yes/no detector — the "
             "product signal is which state it assigns and how stable that "
             "is across repeats.",
    )
    ap.add_argument(
        "--flag-state", default="",
        help="with --state-field, count clips whose modal state is this "
             "value (e.g. 'active' for the seated-fidgeting negatives).",
    )
    ap.add_argument(
        "--window-seconds", type=float, default=0.0,
        help="trim each clip to the FIRST N seconds before the call. The "
             "eval set stores ARCHIVED event clips (~38s pre+post); a live "
             "tick reasons over livekit_worker_window_seconds (11s). Without "
             "this the harness evaluates an easier task than production.",
    )
    ap.add_argument(
        "--bundle", action="store_true",
        help="send EVERY enabled detector for the patient in ONE call, the "
             "way a live tick does, instead of the target detector alone. "
             "Isolates whether bundling is what degrades the verdict.",
    )
    ap.add_argument(
        "--no-ordering", action="store_true",
        help="send the schema WITHOUT property_ordering — the pre-2026-08-07 "
             "behaviour, for A/B-ing whether generation order is what binds "
             "evidence-before-verdict",
    )
    args = ap.parse_args()

    if args.no_ordering:
        from src.services.detection.providers import gemini as _g
        _g._evidence_first_ordering = lambda props: None

    specs, name, bucket = await _load_spec(args.detector, args.bundle)
    settings = get_settings()
    model = args.model or settings.gemini_model
    clips = sorted(Path(os.path.expanduser(args.dir)).glob("*.mp4"))
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no clips in {args.dir}")

    field = f"{args.detector}_detected"
    print(f"detector : {name} ({args.detector})  db_bucket={bucket}")
    print(f"model    : {model}   mode={args.mode}   repeats={args.repeats}")
    print(f"bundled  : {len(specs)} detector(s) in one call")
    print(f"clips    : {len(clips)} from {args.dir}\n")

    rows, uncroppable = [], []
    tmp = Path(tempfile.mkdtemp())
    for i, clip in enumerate(clips, 1):
        src = clip
        if args.mode == "crop":
            dst = tmp / f"crop_{clip.name}"
            if not _crop_to_person(clip, dst):
                uncroppable.append(clip.name)
                continue
            src = dst
        if args.window_seconds:
            trimmed = tmp / f"win_{clip.name}"
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(src), "-t", str(args.window_seconds),
                 "-c", "copy", str(trimmed)], check=True)
            src = trimmed
        mp4 = src.read_bytes()
        verdicts, confs, seens, descs, states = [], [], [], [], []
        for _ in range(args.repeats):
            try:
                res = await _run_clip(mp4, specs, settings, model)
                f = res.detector_fields or {}
                verdicts.append(bool(f.get(field)))
                confs.append(f.get(f"{args.detector}_confidence"))
                seens.append(f.get("seen"))
                if args.state_field:
                    states.append(f.get(args.state_field))
                descs.append(res.description)
            except Exception as exc:  # noqa: BLE001
                verdicts.append(None)
                confs.append(None)
                seens.append(f"ERROR {exc}")
                descs.append(None)
        gold = _label(clip)
        fired = sum(1 for v in verdicts if v)
        rows.append({
            "clip": clip.name, "label": gold, "verdicts": verdicts,
            "fired": fired, "of": len(verdicts),
            "confidences": confs, "seen": seens[0],
            "description": descs[0] if descs else None,
            "states": states,
        })
        if args.state_field:
            counts = Counter(x for x in states if x)
            modal = counts.most_common(1)[0][0] if counts else "?"
            stable = len(counts) <= 1
            hit = (modal == args.flag_state) if args.flag_state else False
            print(f"  [{i:2}/{len(clips)}] {'FLAG' if hit else 'ok  '} "
                  f"{clip.name[:40]:40} state={modal:9} "
                  f"{'' if stable else 'UNSTABLE ' + str(dict(counts))}")
            continue
        mark = "?" if gold is None else (
            "ok " if (gold == "pos") == (fired > len(verdicts) / 2) else "MISS"
        )
        print(f"  [{i:2}/{len(clips)}] {mark} {clip.name[:46]:46} "
              f"fired {fired}/{len(verdicts)}  label={gold}")

    print()
    if args.state_field:
        allc = Counter(s for r in rows for s in r["states"] if s)
        print("state distribution across all runs:", dict(allc))
        if args.flag_state:
            flagged = [
                r["clip"] for r in rows
                if r["states"] and Counter(
                    x for x in r["states"] if x
                ).most_common(1)[0][0] == args.flag_state
            ]
            print(f"clips whose modal state is {args.flag_state!r}: "
                  f"{len(flagged)}/{len(rows)}")
            for c in flagged:
                print(f"    {c}")
        unstable = [r["clip"] for r in rows
                    if len({x for x in r["states"] if x}) > 1]
        if unstable:
            print(f"UNSTABLE across repeats: {len(unstable)}/{len(rows)}")
            for c in unstable:
                print(f"    {c}")
        if args.out:
            Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
            print(f"\nraw results -> {args.out}")
        return
    lab = [r for r in rows if r["label"]]
    tp = sum(1 for r in lab if r["label"] == "pos" and r["fired"] > r["of"] / 2)
    fn = sum(1 for r in lab if r["label"] == "pos" and r["fired"] <= r["of"] / 2)
    fp = sum(1 for r in lab if r["label"] == "neg" and r["fired"] > r["of"] / 2)
    tn = sum(1 for r in lab if r["label"] == "neg" and r["fired"] <= r["of"] / 2)
    unstable = sum(1 for r in rows if 0 < r["fired"] < r["of"])
    print(f"labelled {len(lab)}   TP {tp}  FN {fn}  FP {fp}  TN {tn}")
    if tp + fp:
        print(f"precision {tp/(tp+fp):.2f}", end="   ")
    if tp + fn:
        print(f"recall {tp/(tp+fn):.2f}", end="")
    print()
    if unstable:
        print(f"UNSTABLE (verdict changed across repeats): {unstable} clip(s)")
    if uncroppable:
        print(f"NO PERSON FOUND, skipped in crop mode: {len(uncroppable)}")
        for u in uncroppable:
            print(f"    {u}")
    if len(rows) - len(lab):
        print(f"unlabelled (not counted): {len(rows) - len(lab)}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
        print(f"\nraw results -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
