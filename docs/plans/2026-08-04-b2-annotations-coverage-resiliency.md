# B2 data collection — annotations, coverage, censoring, resiliency

> **Status (2026-08-04):** planned, not started. Phase 0 (clip timing) is
> DONE and committed on `feat/data-collection-timing-truth` (`db2ec3a`).
> Everything below is downstream of it.

**Goal:** turn the B2 recorder from "a pile of MP4s" into a labelled research
corpus — one that says what happened and when, covers the moments that matter,
keeps sensitive footage usable without exposing it, and doesn't lose clips.

**Standing constraint (unchanged, 2026-07-29):** pilot families are
OBSERVATION-FIRST. `data_collection_enabled` ships **OFF**; research consent is
opt-in per family, later. Nothing here is pilot-blocking, and nothing here may
assume a consented corpus exists.

---

## Phase 0 — clip timing (DONE)

Recorded clips were shorter than the sessions they recorded and the audio drifted
inside them: a 20s session produced 18.7s of video on an idle loop, 6.6s under
load. The pacer counted its own wakeups instead of asking the clock. Fixed by
applying the principle `mp4_remux.frames_to_vfr_mp4` already uses for event
clips — keep true timestamps, materialise onto a uniform fps grid — in streaming
form. Audio now anchors on a measured capture time and stream dropouts are filled
with silence.

**Why it comes first:** an annotation points into a clip as
`captured_at - started_at`. That arithmetic is only valid because clip duration
is now truthful. Every phase below depends on it.

**Consequence to expect on deploy:** clips get slightly LARGER. We were writing
fewer frames than wall clock demanded (as few as 1 in 6 under load); now we write
all of them. The added frames are duplicates of a held frame and x264 encodes
those as skip frames, so the increase should be small — but it is an increase,
and it should be watched on the first soak.

---

## Decisions locked in discussion (2026-08-03/04)

| # | Decision | Notes |
|---|---|---|
| 1 | Annotations live in **Postgres AND sidecar JSON** | DB is the source of truth; the sidecar is a *rendering* of it |
| 2 | Three layers: **observations → episodes → events** | Episodes are intervals, revisable, and are the new work |
| 3 | Episodes store **everything** — reasoning text, k-of-N counters, latch state | Filtering happens at EXPORT, never at write. Losing it at write loses it forever |
| 4 | Expert review IS the labelling tool | `review_gate` approve/reject is the TP/FP label |
| 5 | Un-reviewed events stay **NULL**, queried and revisited later | No second queue. Replaces the earlier `auto` stamp |
| 6 | Sensitive clips: **keep audio in B2**, behind an internal toggle | `censor_clip` already keeps audio by default; the toggle only overrides the intimate-situation silence |
| 7 | Sensitive + segmentation can't find the subject → **drop the clip** | Research corpora may have holes; caregiver alerts may not |
| 8 | **Masking is driven ONLY by the VLM sensitivity verdict** | Room class does NOT mask. A bedroom records raw unless a verdict says otherwise (2026-08-04) |
| 9 | Censoring happens **inline, as frames arrive** — never as a post-pass | See "correct implementation" below |
| 10 | On a first sensitive verdict, **discard the raw prefix** and restart masked | ~10s lost at the transition; nothing raw-and-sensitive ever persists |
| 11 | Research mask = **flat fill, tight dilation**, research-only | DONE, `502717d`. Deliberately leaks body outline; see the module comment |
| 12 | **Partner export is out of scope** | Dropped 2026-08-04 |
| 13 | **Linger stays at 45s** | The increase was a workaround for the motion gate; redundant if motion is dropped |

---

## The correct implementation — inline censoring

Doing this as a post-pass over the finished MP4 was wrong: it has to decode
H.264 back into frames before it can segment anything, and a 15-minute clip is
13,500 frames against an event clip's ~100. Inline, detection has **already
decoded the frames** — so the decode and the re-encode both disappear, and raw
uncensored video never reaches disk at all.

**The seam already exists.** `RecorderManager` takes an injectable
`encoder_factory`, and `SessionRecorder` only ever calls `start()`, `feed()` and
`aclose()` on what it gets back. So a masking encoder is a drop-in peer of
`StreamingMp4Writer`, and the session machine does not change at all.

```
room_worker._push_video → recorder.feed_video → session.feed_video
                                                      │
                             encoder_factory picks one of:
                                                      │
        StreamingMp4Writer ────────────────┬───────────┴── MaskingMp4Writer
        (raw; today's path)                │                (new)
                                           │
                        raw I420 → child process → segment → flat fill
                                     → _FrameEncoder → H.264 file
```

**Why a child process, not a thread.** Tier-1 censoring is isolated today
because a native OpenMP deadlock in torch froze the whole backend on 2026-07-28,
event loop included — a hang below Python that `asyncio.to_thread` cannot
contain but `kill()` always can. Inline masking inherits that constraint. The
child is long-lived for the session instead of per-clip, which also avoids
paying model load (~2-4s) per clip.

**Switching mid-session.** A verdict lands ~10s after the footage it judges, so
frames before it are already written. Rather than leave them, the tick aborts the
raw session (`SessionRecorder.abort()` — already built and tested for the privacy
path, discards temps without muxing or uploading) and opens a fresh masked one.

**Segmentation rate.** 15fps leaves 66ms a frame and segmentation costs tens of
ms. Segment at a lower rate and carry the mask across intervening frames with
wider dilation — `censor.py` already does exactly this for detection dropouts
(`_MASK_GAP_BRIDGE_FRAMES`, `_dilate_mask(extra=True)`).

**Open cost question.** The verdict needs a per-tick sensitivity screen.
`sensitivity.screen_clip` is a separate VLM call, so running it every tick
roughly doubles inference cost for a collecting camera. Adding the verdict to the
existing detection call would be free but means touching the detection prompt,
which the recorder has never done. **Implement with a settable cadence (screen
every Nth tick) and let the number be the lever.**

---

## Phase A — coverage (the gate)

**The gap.** We do not store raw video as it comes in. The gate is
`data_collection_enabled AND motion AND presence AND NOT privacy`, with a 45s
linger. A person sitting still for more than 45s ends the session; if they then
fall, the new session opens at the tick that noticed, and the start of the fall
is in the window we didn't record. That is the highest-value slice of footage we
have, and it is exactly what we miss.

**Grounding.** From the live `memoria-dev` bucket (2026-08-04): 171 objects,
9.9 GB, largest clip 338 MB. Against a 15-minute roll that is ~3 Mbps of real
time, i.e. **~1.35 GB per camera-hour**.

| Option | What it records | Per camera-day | Verdict |
|---|---|---|---|
| A1 Full continuous | everything, always | ~32 GB | ~1 TB/camera/month, accumulating. Not viable alone |
| A2 **Presence, drop motion** | anyone in the room, still or not | ~8 GB @ 6h occupancy | **Recommended.** Closes the actual gap |
| A3 **Longer linger** (45s → 5 min) | bridges stillness | small delta | **Recommended.** One constant, big return |
| A4 Two-tier | full rate on gate, 1fps/480p otherwise | ~1.6 GB + bursts | True 24/7 at ~5% of A1. Real work; probably premature |

**Recommendation: A2 + A3 now, A4 only if 24/7 turns out to be a hard
requirement.** Together they cover the still-person case at roughly 4x current
storage rather than 12x, and both reuse signals that already exist.

**The risk that decides A2, and it must be settled before coding.** Presence
comes from the detection tick. If a camera's state machine closes it when nothing
is happening, ticks stop, and there is no presence signal to record on — so A2
would need the camera held open, which costs a VLM call per tick (~$2 per
occupied-room-hour at `media_res=high`). That would make a storage decision
silently into a much larger inference bill.

The way out is the already-agreed decoupling: body tracking is meant to be
**always-on and independent of VLM attribution** (see
`nemara-reid-decouple-tracking-attribution`). Recording coverage should gate on
the cheap always-on tracker, not on the expensive VLM verdict.

- **A.1** Establish empirically whether a still-but-occupied room keeps ticking
  today, and what it costs. No code until this is answered.
- **A.2** Route `should_collect`'s presence input to the always-on tracker rather
  than the VLM tick verdict; drop the motion term. Truth-table unit tests.
- **A.3** Raise `linger_seconds` to 300 and make it a setting.
- **A.4** Re-measure GB/camera-day on soak before this goes anywhere near a
  consented home.

### Lead-in (small, separate)

Now cheap: the encoder knows what time a frame belongs to, so the analysis window
held in `FrameBuffer` can be replayed at its true rate instead of collapsing into
the one-frame slot. Frames are JPEG in the buffer since 2026-07-19, so a replay
needs a decode per frame (~40 frames, bounded). Worth doing after A2, which
shrinks how often it matters.

---

## Phase B — annotations

### B.1 Episodes table

```
episodes(
  id, patient_id, camera_id, kind,
  started_at, ended_at NULL,        -- open while it is happening
  state,                            -- armed | fired | resolved | retracted | ...
  payload JSONB,                    -- reasoning, k-of-N, latch, quality flags
  superseded_by NULL,               -- revision, never destructive update
  created_at, updated_at
)
```

Revisability is the point. Sleep opens at 22:14 and does not close until 06:03,
crosses ~30 clip rolls and gaps where nothing recorded, and can be split or
retracted after the fact. Nothing per-clip can hold that.

`superseded_by` rather than in-place mutation, so a label a human applied to an
episode is never silently re-pointed at a different interval.

### B.2 Writers

Detectors emit episode open/extend/close alongside the `analyses` row they
already write. Stateful detectors (sleep, and the k-of-N cadence ones) own their
own episode lifecycle; stateless ones open and close in one tick.

**Must not alter detection verdicts.** Same read-only discipline as the recorder
hooks.

### B.3 Clip projection

Given an episode, return every overlapping clip and the offset within each:
`offset = episode.started_at - clip.started_at`, valid because of Phase 0. An
episode crossing four clips returns four segments; one crossing a gap returns
what exists and reports the hole rather than papering over it.

`started_at` and `duration_seconds` already ride on the B2 object (Phase 0), so
this works without opening a file.

### B.4 Review as labelling

`review_gate` becomes `null | pending | approved | rejected`. **`auto` is
retired**: an event nobody reviewed stays NULL, which makes "everything still
unlabelled" a plain query and the labelling backlog free. The 60s auto-release
behaviour is untouched — it just stops stamping a value.

An approve/reject writes a verdict onto the overlapping episode. That is the
highest-value annotation in the corpus: human-adjudicated ground truth on a real
detector firing in a real home.

> **Open:** NULL will mean both "gate was never on" and "was held, nobody
> looked". Fine for labelling, but it loses any measure of reviewer coverage. Add
> a nullable `held_at` if that measure is wanted.

### B.5 Sidecar + export

- **At finalize:** minimal sidecar next to the MP4 — clip identity, true start,
  true duration, room, gate reason, and the layer-1 observations inside the clip.
  Makes the object self-describing.
- **At export:** regenerate in full from the DB, including episodes that had not
  closed when the clip was written. Carries `schema_version` and `generated_at` —
  the same clip exported twice legitimately differs, and a partner must be able
  to tell which they hold.

The export renderer is the single place that decides what a given recipient sees.
Reasoning text and detector internals are stored always and filtered here.

---

## Phase C — censoring + audio

- **C.1** Route recordings through `censor.py` when the sensitivity screen flags
  them, using a **tight mask** rather than pixelation. Note this is *less* safe
  than the current generous dilation — that is the tradeoff explicit consent buys,
  and the plan should say so out loud.
- **C.2** Screen says not-sensitive, or the screen itself fails → **store raw**.
  This inverts the event-clip posture on purpose: a research clip goes to a
  consented corpus, not to a caregiver's phone.
- **C.3** Scene IS sensitive but segmentation cannot find the subject → **drop**.
  Never full-frame blur (worthless for research), never raw.
- **C.4** Internal toggle: retain audio on sensitive research clips even though
  caregiver-facing censored clips stay stripped.
  **Composition rule:** `patients.audio_capture_enabled` is a facilities-level
  legal kill switch and wins unconditionally. The internal toggle decides whether
  we KEEP sensitive audio; it can never cause capture of audio that is legally
  off.
- **C.5** Consent text must state audio retention in sensitive moments
  explicitly. Not implied by a general research opt-in.
- **C.6** Worklist item 9 — censor soak on Railway. Never run. Local numbers are
  Docker-VM numbers.

---

## Phase D — resiliency (all found in the 2026-08-04 audit)

Ordered by what actually loses data.

- **D.1 Upload failure is permanent loss.** 3 retries with backoff, then the
  clip is logged and deleted in the `finally`. A short B2 outage silently
  destroys every clip finalised during it. Needs a disk spool with a retry
  sweeper, bounded by size and age.
- **D.2 No temp sweep — disk fill AND a privacy hole.** `clip_archival` sweeps
  `<tmp>/nemara-clip-scratch` at startup; the recorder's
  `<tmp>/memoria-data-collection` has **no equivalent**. After an OOM kill,
  orphans up to 1.5 GB each accumulate forever on ephemeral disk — and they are
  **raw uncensored footage**, which is the same failure `mp4_remux` was fixed for
  on 2026-07-29. Mirror `sweep_scratch_dir()`.
- **D.3 Region fail-open is a residency violation.** A region lookup failure
  defaults an India patient's clip to the US bucket, silently. Against the DPDP
  guardrails this should spool and retry, not fail open. Deliberate decision at
  `manager.py:306`, but taken before India was real.
- **D.4 `data_collection_max_clip_bytes` is dead.** Defined in settings, used
  nowhere. The 15-min roll is the only bound; a high-bitrate scene can exceed it.
  Enforce it or delete it.
- **D.5 No observability.** Nothing counts clips recorded, uploaded, dropped,
  privacy-discarded, cap-skipped, or `deficit_frames`. Collection could stop
  entirely and nothing would say so. Cheapest high-value item here.
- **D.6 No post-upload integrity check.** No size or checksum verification.
- **D.7 Temp disk is unbounded in practice.** cap(8) × 1.5 GB = 12 GB of
  headroom assumed, never measured against the Railway disk.

**Recommended order: D.5 → D.2 → D.1 → D.3 → D.4 → D.6/D.7.** Observability
first, because everything else is currently invisible.

---

## Not doing: repairing the existing corpus

Old clips are CFR files claiming 15fps. Real capture times were never written
anywhere, so the timeline is absent rather than degraded.

It IS partially recoverable. `analyses` rows carry `captured_at`,
`people_present` and `motion_in_clip` per tick, so replaying the gate over them
reconstructs each session's true boundaries, and a uniform stretch (stream-copy
with a new timebase — lossless) would restore average playback speed *and*
re-align audio, since audio was always real-time-correct while video was
compressed.

Not worth building. The drift tracked event-loop load and was not uniform within
a clip, so a uniform stretch fixes the average and not the local jitter; the
audio tail `-shortest` discarded is gone regardless; and the corpus is 171 soak
clips in `memoria-dev`, not consented family data. Revisit only if prod turns out
to hold something worth keeping.

---

## Verification

- Baseline `.venv/bin/pytest test/unit -q` → **2 failed, 941 passed, 21 errors**
  (the 2 `test_face_embedding_repo` failures and the 21 errors are pre-existing).
  Phase 0 moved 932 → 941 by adding 9 tests.
- Never buffer frames in RAM (`streaming_encoder` stream-writes; buffering a
  1080p clip got a child OOM-killed 2026-07-28).
- Do not weaken `test_recorder_privacy_drive.py` — a private-room camera must
  abort an open session rather than drift toward upload.
- Storage re-measured on soak after Phase 0 and again after Phase A.
