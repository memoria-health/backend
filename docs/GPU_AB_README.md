# SAM identity A/B — GPU box runbook (2026-07-20)

Tests whether **SAM 3.1 + AdaFace + OSNet-x1_0** (mask-clean crops)
recognizes the patient on clips where the legacy **YOLO + BoT-SORT +
ArcFace + OSNet-x0_25** stack fails. Both arms share the patient
gallery and the match thresholds — only the models change.

## What runs where

- **Legacy arm**: runs anywhere (CPU), already validated on the Mac.
- **SAM arm**: needs an NVIDIA GPU (SAM 3.1 is CUDA; MPS unproven).

## One-time box setup

```bash
# on a RunPod/Vast box, "PyTorch 2.4 / CUDA 12.4" template
export HUGGINGFACE_TOKEN=hf_...          # license-accept SAM 3.1 + AdaFace first
git clone <this branch> && cd nemara-v2-build
bash scripts/gpu_setup.sh               # deps + AdaFace source + model download
```

## Stage the clips (from a box that can reach the local Supabase, or
run on the Mac and rsync `out/ab_clips` up)

```bash
python tools/stage_ab_clips.py          # incident MP4s + id-frame JPEGs -> out/ab_clips/
# then fill ground truth on the clips that matter:
#   out/ab_labels.csv   (clip_name,patient_present 1/0)
```

Evidence-frame filenames already encode truth: `..._1p-pat-1face` =
patient present with a face; `..._6p-unk...` = crowd, no patient.

## Run

```bash
python tools/pipeline_ab.py \
  --clips out/ab_clips \
  --patient 75a30940-b5e2-4608-afa3-b31a1e1fcda6 \
  --backends legacy,sam \
  --labels out/ab_labels.csv
```

## Read the result

The money line: **"clips where SAM caught a patient legacy MISSED"**.
On the Mac smoke run, the legacy arm already missed a
patient-present-with-face frame at face score 0.438 (< the 0.45 bar) —
the exact case AdaFace should rescue. If SAM catches a meaningful
share of these, the model upgrade is validated and the live flag
(`reid_pipeline_backend=sam`) is worth wiring into the backend.

## Ablate (isolate the variable)

The SAM arm's models are individually swappable so you learn WHICH
upgrade mattered:
- `reid_sam_face_model`  — AdaFace vs (set to the ArcFace path) to isolate the face model
- `reid_sam_body_model`  — OSNet-x1_0 vs x0_25
- SAM mask on/off        — isolate the clean-crop effect

## Live integration (phase 2, only if the A/B validates)

`reid_pipeline_backend` already exists as the flag; the seam is
`ReIDInference.label_clip` returning an `IdentityResult`. Phase 2 wires
`_run_sam_pipeline_sync` behind the flag so the live soak + dashboard
run on the SAM stack unchanged. Not built until the A/B earns it.
```
