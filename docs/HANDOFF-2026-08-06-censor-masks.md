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

---

## Sleep — what changed, and what tonight's soak must answer

### Ground truth from the 08-05/06 night
Resident: asleep ~02:00, woke ~07:28-07:30, used a phone in bed ~8 min, dozed
again, got out of bed 07:55. No fall, no genuine call for help except a
deliberate 20:10 test.

### Shipped (commit `9938d7e`)

**1. Awakenings can be established by EVIDENCE, not just elapsed time.**
`WAKE_MIN_EVIDENCED_TICKS = 3` consecutive positively-sighted awake ticks
(`awake_in_bed` / `sitting_up` / `leaving_bed`, and only when the model already
said awake) now end a sleep run even under `WAKE_MIN_SECONDS`. The 10-minute
actigraphy rule buys certainty when the only signal is movement; it cost us a
real 8-minute wake backed by 30 consecutive ticks of "looking at a glowing
smartphone" plus a lamp being switched on. Unevidenced ticks
(`asleep=False, activity=none`) keep the full 10 minutes — that is an absence
of a sleep call, not a sighting.

**2. A single false `leaving_bed` can no longer fabricate an awakening.**
A crossing CLOSES the occupancy span, and `_sleep_runs` only ever saw one span,
so the flicker rule that would have absorbed a lone tick never saw the seam.
Runs are now merged across span boundaries with the same test, and the onset
filter runs after the merge so a split night is not discarded as two short
ones. On 08-06 this was one tick claiming "got out of bed and abruptly
collapsed to the floor" between two ticks of lying asleep under the covers.

**3. `run_daily_summaries.py` bound NAIVE window bounds against a timestamptz
column**, so they were read in the client's local zone. On an IST machine every
night silently truncated at 06:30 local, with no flag. Same tool, same rows:
IST host reported 274 min, UTC container 359 min. Fixed with `_bind()`.

Also corrected `MIN_CONFIDENCE`'s docstring: `sleep_confidence` is a VISIBILITY
score, not a sleep probability (conf=1.0 ticks were 44.8% asleep; conf=0.95
were 97.2%), because 1.0 is used equally for a confident ASLEEP and a confident
AWAKE. Behaviour unchanged; only the claim it makes.

### Result on that night, after the fixes
```
onset      01:56:53   (stable sleep; he announced "going to sleep" 01:54:12)
wake       07:54:48   (matches getting out of bed)
duration   350 min
awakening  07:30:18-07:38:28, 8 min  — the real phone episode
```
Previously: onset 01:55:33, wake 06:29:58, 274 min, and a 0-minute phantom
awakening at 05:49 invented by the hallucinated crossing.

### What tonight's soak must answer

1. **Does the evidenced-awakening rule hold on a second night?** It was tuned
   to one. Watch for the opposite failure: 3 consecutive `awake_in_bed` ticks
   fragmenting a night that was really continuous.
2. **Does the new evidence-first prompt change sleep labelling at all?**
   `heard` / `audio_source` / `seen` were added AFTER last night's data, so
   every sleep number above predates them. Sleep labels could shift.
3. **Does the IR blackout carry (`f9a07a7`) ever fire?** It has had NOTHING to
   act on so far. It handles a track SWAP; the 01:57 loss was
   `patient_left_view` — the body vanishing entirely — which is a different
   failure. Grep for `track_ledger.blackout_carry`.
4. **How long is the under-duvet blackout?** Last night: 03:03:33-04:02:14,
   ~59 min, with only 2 visible ticks. Nothing in the perception stack sees a
   person under covers in IR — `id_tracks` was `{}` and `people_count` 0 while
   frames arrived fine. The night survived only because sparse glimpses stitch
   the runs. A night with ZERO glimpses after onset would still truncate.
5. **Sanity: is `Sleep` still assigned to the camera?** It was `is_on=true` but
   assigned to NO camera and had never run once — found and fixed 08-06 01:59.
   `cameras_detectors` is the join that matters, not `is_on`.

### Sleep-adjacent product gap, unfixed
A detector can be `is_on` and assigned to nothing, and every surface still
shows it as ON. For a pilot family that is silent failure of a feature they
believe they bought. Worth an explicit "assigned to N cameras" signal.

---

## Activity — what it is for, and the edge cases to test

**What we actually want from it (Aryaman, 2026-08-06): movement vs lethargy.**
NOT granular posture classification. Whether the label is `sitting` or
`standing` does not matter; both mean "not moving". The product signal is the
ACTIVE FRACTION and how it trends, so the aggregation matters more than any
individual label.

### Measured over 2026-08-06 (3,630 person-present ticks)
```
sitting    59.7%   avg confidence 0.51
lying      32.1%   avg confidence 0.96
active      6.0%   avg confidence 0.98
standing     2.2%   avg confidence 0.77
```
`lying` and `active` are confident and trustworthy — which is the half the
product depends on. `sitting` is the low-confidence majority class and looks
like the model's default bucket when unsure.

**Under the movement-vs-lethargy framing that is mostly fine**, with ONE
residual risk worth testing: low-confidence `sitting` may be SWALLOWING
genuine movement, which would UNDERSTATE activity. That direction matters —
overstating lethargy in an elderly patient is the error that triggers a false
concern.

### The edge case: seated exercise

The prompt already makes a specific, falsifiable claim
(`catalog.py:96-103`, `activity_summary.py:5`):

> "actual physical activity and movement — engagement, not raw pixel motion.
>  Seated EXERCISE is active; seated fidgeting is sitting."

So this is not a new requirement, it is an existing assertion nobody has
tested. If it fails, every seated-exercise minute is counted as lethargy.

### Clips to capture (file under `~/Desktop/eval-set/activity/`)

POSITIVES — must label `active` despite the person being seated:
- `pos_seated_arm_raises` — seated, lifting arms overhead repeatedly
- `pos_seated_leg_extensions` — seated, straightening/lowering legs
- `pos_seated_weights` — seated, light dumbbells or a bottle as weight
- `pos_seated_resistance_band`
- `pos_seated_marching` — seated, alternating knee lifts
- `pos_physio_routine` — a slow, low-amplitude physio set (the hardest case:
  genuinely therapeutic movement that looks almost still)

NEGATIVES — must NOT label `active`, and are the ones most likely to trip:
- `neg_seated_fidgeting` — shifting, crossing legs, adjusting position
- `neg_seated_gesturing` — talking with the hands, animated conversation
- `neg_seated_eating` — repetitive hand-to-mouth, high pixel motion
- `neg_seated_typing` — sustained hand movement, zero whole-body engagement
- `neg_seated_phone_scrolling`

The two adversarial pairs are `pos_seated_arm_raises` vs `neg_seated_gesturing`
(similar arm motion, different intent) and `pos_seated_marching` vs
`neg_seated_fidgeting` (similar leg motion, different intent). If the model can
separate those, the "engagement, not raw pixel motion" claim holds.

### How to test
Run through `scratchpad/eval_new_config.py` with `activity` as the detector.
**3x per clip** — single runs misled us repeatedly on 2026-08-06.

Also worth checking on the same pass: does an explicit `unclear` state absorb
much of the 59.7% `sitting`? If most of it moves, the timeline has been
overstating sitting; if little moves, it is real and merely poorly calibrated.
That is an hour against existing clips, not a soak — and it only matters if
the seated-exercise cases above fail.

---

## Found while implementing (2026-08-06 evening) — the tracker, not the censor

The consumer is built, deployed and verified: at seated/standing distance
the operator clip is a person-shaped redaction with the room readable
(0 fully-covered frames of 24, live camera and real pod). Commit
`29eb471`.

Three things the plan above did not anticipate, all found live:

1. **Clock.** The pod stamped masks with ITS OWN clock at PROCESSING
   time. Measured: pod runs +1.54s ahead of the backend host, processing
   lags capture ~2.6s — every mask was ~4.1s from the frame it described.
   The identity service now sends capture times through `/observe` and
   the pod echoes them. Both wire fields are optional, so either service
   can restart first (`test_pod_wire_compat`).

2. **A bug in `23494b9`.** `record_masks` took width/height as a
   parameter and the caller passed the FRAME's size, but EdgeTAM returns
   masks at TRACK resolution. Every mask was labelled 1920x1080 while
   actually being 640x360; decoding put the silhouette in a thin band
   across the top of the frame — person-SIZED coverage, nowhere near the
   person, invisible to anything downstream. Dimensions now come off the
   array. Found by rendering a mask, not by reading numbers.

3. **Motion.** Masks arrive at ~3fps, clips encode at 8, so a frame gets
   a silhouette up to 170ms old — ~63px of walking at 1080p against a
   5px research margin. The margin now scales with the measured gap and
   the subject's own height.

### OPEN, and the reason the gate still fails at night

**SAM3 does not ground a recumbent person.** Watched live at 22:45 local
as the subject got into bed:

```
tick=5360 mode=carry  seeded=1 live=1     <- tracking
tick=5361 mode=ground seeded=0 live=0     <- lost, re-ground finds nobody
tick=5362 mode=ground seeded=0 live=0
```

Mask retention then decays 96 -> 35 -> 0 as the pre-bed window ages out.
The pod IS being polled (this is not the empty-room nap); grounding
simply returns nothing. Reproduced offline against
`neg_fp_2026-08-05_013723.mp4`, a bedroom clip with someone reclining
partly behind the monitor: 0 tracks.

Consequence, measured with the real live mask window at that moment: a
gate clip would come back **24/24 frames fully covered** — the same green
rectangle the censor work exists to remove, now caused upstream. Also no
seat/patient attribution overnight, and any recorder session judged
sensitive while the subject is in bed produces an all-green research clip
worth nothing to the corpus.

What is NOT affected: detection itself. Every suppression gate that reads
tracks is behind `scheduler_person_requires_track` (unset -> False), and
`patient_elsewhere` needs a SECOND camera holding a fresh claim. Sleep
detectors run normally with `tracks=[]`.

Not a quick fix, and not a mid-soak change. Both grounders struggle:
`yoloseg` fragmented a recumbent person (torso as its own mask — see the
note in `scripts/deploy_continuous_remote.sh`), which is why
`NEMARA_GROUNDER=sam3`; SAM3 grounds nothing at all. Worth trying a
compound prompt ("person lying in bed, or under a blanket") the way the
display-veto concept was tuned, and measuring against the eval-set
bedroom clips before touching the live pod.
