# Handoff — replace the censor's segmenter with the pod's masks

Branch: `feat/data-collection-timing-truth`. Backend running locally, healthy.
Everything below is committed and the suite is green (1388 passed, 1 skipped).

---

## The problem being fixed

The event gate pages an operator with a masked clip so they can judge an alert
in 60s. At normal seated distance that clip comes back a **solid green
rectangle for its whole duration** — unjudgeable, so every alert auto-releases
unreviewed. The gate is therefore not doing its job.

Cause, already investigated in `censor.py:88-100`: the backend censors with
`yolo11n-seg`, which at that distance scores ~0.05, and both failure paths
treat "no mask" as "cover everything". A bigger YOLO is NOT the fix — the
comment records `yolo11x-seg` managing 7/21 frames for far more compute.

**The fix**: EdgeTAM on the pod already computes a full person mask for every
frame of every tick and tracks the person reliably at that distance. Those
masks were being thrown away. They are now retained. The censor must consume
them and the YOLO path must be **deleted, not demoted to a fallback** — a
weaker second segmenter would preserve the exact failure being removed.

---

## Done (commit `23494b9`) — pod side, inert

`gpu/perception/continuous.py`
- `rle_encode(mask) -> list[int]` — row-major run lengths, first run is ZEROs;
  a leading `0` is emitted when the mask starts with 1 so parity is unambiguous
- `record_masks(camera_id, captured_at, w, h, {tid: mask})` — retains one
  frame, evicts anything older than `_MASK_RETENTION_SECONDS = 45.0`
- `masks_between(camera_id, t0, t1) -> [{captured_at, width, height, masks}]`
- `forget_masks(camera_id)` — wired into `drop_session`

`gpu/perception/server.py`
- records each tick's per-frame masks after `session.observe(...)`, wrapped so
  it can never break a tick
- `GET /masks/{camera_id}?t0=&t1=` → `{camera_id, count, frames: [...]}`

Verified by round-trip: encode/decode parity, eviction of stale frames,
out-of-order tolerance, window filtering, empty-tick no-op, no leak after
teardown.

**Nothing consumes this yet.** That is the remaining work.

---

## To do — the censor consumer

### 1. `src/services/detection/pod_masks.py` (new)
```
fetch(camera_id, t0, t1, *, base_url, token) -> {frame_time: {tid: rle}}
decode(rle, width, height) -> np.ndarray  # bool, at RECORDED dims
```
Round-trip test it against `continuous.rle_encode` directly — the two must
never drift.

### 2. Subprocess boundary — `censor.py`
The tier-1 censor runs in an **isolated child process** (see the 2026-07-28
OpenMP-deadlock note at `_censor_person_mask_subprocess`). The child today
needs no network and no config; it will now need both.

- `_person_mask_subprocess_cmd` (~line 605): add `--camera-id`, `--clip-start`,
  `--pod-url`, `--pod-token`
- `_cli_main` (~line 816): parse them
- Decide deliberately how the child gets the token. Inheriting the parent's
  env is simplest; passing it in argv puts a credential in the process table.

### 3. `_person_mask` (~line 192)
`(model, frame)` → `(window, frame_time, shape)`. Decode at the **recorded**
width/height from the pod payload (NOT the clip's), union across tracks,
upscale to frame resolution — the same track-res→frame-res step
`server.mask_crop` already does. Keep the existing float-union-then-threshold
order; it is why the outline is smooth (see the comment there).

### 4. Entry points
- `_censor_person_mask_sync` (~417): fetch the window ONCE before the decode
  loop. `frame_time = clip_start + frame_index / fps`.
- `mask_stream_sync` (~538): same prefetch. It is inline in the recorder at
  15fps and **cannot** fetch per frame.
- `mask_people_in_still_sync` (~497): single JPEG, needs one frame time.

### 5. Delete
`_get_seg_model` (~173), `_SEG_MODEL_NAME`, `_SEG_CONFIDENCE`,
`_PERSON_CLASS_ID`, the ultralytics import, and the warmup at `main.py:122`.
Also drop `yolo11n-seg.pt` from the repo root if nothing else uses it.

### 6. Callers must thread camera_id + clip start
`services/clip_archival.py:193`, `api/cameras.py:1170`.

### 7. Fail-closed still applies
If the pod has no masks for the window (sidecar down, window evicted), the
censor must still cover the frame — never serve raw pixels. That is the
existing contract; keep it.

---

## Verify

1. Unit: RLE round-trip; decode at recorded dims; window indexing picks the
   nearest frame.
2. E2E: trigger a gate event at seated distance and confirm the operator clip
   shows a person-shaped mask, not a green rectangle. The clip that failed is
   the 60s-hold page from `event_gate.held`.
3. Confirm `yolo11n-seg` is gone from the process — no ultralytics import at
   startup.

---

## Also open (unrelated to masks)

- **`pos_6_real_help_2026-08-06_1406.mp4`** — a real call for help that misses
  0/3 in eval. Aryaman wants it TP: he is asking for help while talking to his
  mother, and a patient alone would sound identical. Every run transcribes
  garble. Likely a transcription-tier problem, not criteria.
- **`neg_fp_2026-08-05_223120.mp4`** — scores 0.0 / 0.2 / 1.0 on byte-identical
  input. It is a conversation ABOUT the calling-for-help detector, so the words
  really are in the audio. May be unfixable by prompt.
- **Prompt changes must be judged on 3x repeats.** Single-run eval numbers
  misled us repeatedly on 2026-08-06.
- **`confirm_k/n`** silently spends extra model calls per firing tick. NULL
  everywhere today, so inert, but unexamined.
- Eval harness: `scratchpad/eval_new_config.py` runs the labelled set through
  the real prompt builder + provider. Clips in `~/Desktop/eval-set`, ground
  truth in `GROUND-TRUTH-2026-08-05.md`.

---

## Today's commits

```
f64d7dd  calling-for-help: audio_source corroborates, never vetoes
23494b9  pod: retain per-frame masks for the event-gate censor
5b188d4  calling-for-help: literal help requests only
ee8a6f6  prompt: objects/actions only claimable if `seen` records them
7769dfc  remove episode_gap_seconds (dead since 2026-07-29)
8f085f2  compiler: co-occurring != competing; add record_only
af0b1ec  unbundle calls, evidence before verdict, always crop
9938d7e  sleep: evidenced awakenings, tz-correct window
```

---

## Security — state as of 2026-08-06 (NOT part of the mask task)

The 08-05 security pass is **code-complete and committed**; what remains is
credential rotation, which is deliberately deferred and is the riskiest item
outstanding.

**Done and committed** — rate limiting (`src/api/rate_limit.py`, a reusable
FastAPI dependency, applied to `/cameras/pair` and detector creation), security
headers (`src/api/security_headers.py`), owner-scoped deletes
(`repos/access.py:owned_patient_ids`), the demo-camera seeder scoped to its own
patient with a `--remote` guard. Findings and corrections are written up in
`security/SECURITY-AUDIT-2026-08-05-REMEDIATION.md`.

**NOT rotated — the open risk.** Deferred by Aryaman until he and Sunny can do
it together on a call:

- Backend keys swept into `origin/main` in commit `f1c0564` (Sunny's local
  `.env.bak`): Gemini, OpenAI x2, OpenRouter, Resend, Dashscope, and the
  Supabase service-role key + JWT secret. Supabase/DB values in that file were
  local-only and are safe.
- Gateway credentials from the committed `nemara-gateway/GATEWAY_ACCESS.md`:
  the Tapo camera password (still `qwerty123`), the gateway SSH/sudo password,
  the home Wi-Fi password, and the production LiveKit key + secret.
- History purge is blocked by GitHub `refs/pull/*`, which cannot be
  force-deleted — so **rotation is the only real remedy**, not history rewriting.

**Also open**

- Insights in `nemara-web-app` query production Postgres directly as
  `nemara_readonly` with BYPASSRLS; tenant isolation is hand-written WHERE
  clauses only. Fix is to route Insights through the backend API.
- `apply_rls_sleep_activity_review_delta.sql` must ship in the SAME deploy
  batch as the sleep/review deltas, and the demo camera must be seeded after it
  (App Review pairing 404s in prod today).
- iOS Universal Links (M4) — custom URL scheme is PKCE-mitigated but not fixed.
- `NEMARA_LOG_LEVEL=DEBUG` makes the OpenAI SDK log whole request payloads,
  including base64 video frames of the patient's bedroom. Prod is `INFO`, so
  not live — but it is one env var from a PHI leak into Railway logs.
  Suggested fix: add `openai._base_client` to the noisy-logger suppression
  already in `main.py` alongside boxmot/ultralytics.
- Local `.env` now carries real PagerDuty / ops / event-gate values (added
  2026-08-06 so the gate can actually page). Local only; do not commit.
