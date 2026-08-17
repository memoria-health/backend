# Handoff — testing the B2 recorder, masking and sidecar work

> **For Claude:** this is a TESTING handoff, not a build one. The code is
> written, committed and unit-tested. What it has never had is contact
> with a real camera. Read the whole file before running anything —
> **test 1 gates the rest**, and if it fails the design needs rethinking
> rather than debugging.

**Who is testing:** Aryaman's own home is the test household. There is no
customer exposure and no consented family to protect, so everything below
can be run locally against real cameras.

---

## Start here

| | |
|---|---|
| Repo | `nemara-backend` |
| Branch | `feat/data-collection-timing-truth` (shared — see *Shared branch* below) |
| Feature flag | `patients.data_collection_enabled`, default **FALSE**. Nothing records until it is switched on for one patient |
| Unit baseline | `.venv/bin/pytest test/unit -q` → **1158 passed** as of 2026-08-05 |
| Local DB deltas applied | `apply_data_collection_policy_version_delta.sql` (done) |
| Prod deltas NOT applied | the same one, plus whatever the event-gate work needs |

---

## What this work is, in one paragraph

When a patient opts in, the backend records what its cameras are already
streaming to disk, muxes audio in, and uploads one MP4 per presence
session to Backblaze. Three things were added: the clips are now the
right *length* (they were not), a sensitive moment now switches the
recording to a masked version where people are painted flat green, and
every clip gets a JSON note beside it saying what the detectors saw and
when. None of it changes detection — the recorder reads signals the tick
already computed and never writes back.

---

## Why the timing fix exists, and why it is the strongest thing here

The recorder's pacer counted **its own wakeups** and assumed each meant
1/15th of a second. `asyncio.sleep` sleeps *at least* that long and the
event loop it shares with detection stretches it further, so the frame
count always ran behind real time while ffmpeg was told the input was
exactly 15fps. Clips came out short and time-compressed, and the audio
drifted away from the picture inside them.

**Measured on 15 real clips from the dev bucket** (each one hit the
15-minute roll, so its true length was known to be ~900s):

| | Loss |
|---|---|
| Median | 3.8% — 34s missing from every 15-minute clip |
| Worst | **91.9%** — 907 seconds of a living room became a 74-second file |
| Mean | 12.9% |

Every clip lost time. The fix makes the pacer ask the clock how many
frames should exist and fill the gap with duplicates. Re-measured
locally: 20.00s in, 20.00s out at every load level tested.

This is the only part of the work with before-and-after evidence from
real footage. Treat it as high confidence; treat everything else as
unproven.

---

## Confidence, honestly split

Read this before deciding how much time to spend where.

| Area | State |
|---|---|
| Clip timing | **High.** Measured on real clips both sides |
| Startup sweep of raw tempfiles | **High.** Mirrors `clip_archival.sweep_scratch_dir`, which has been live for a week |
| Audio anchoring, gap filling | **Medium.** Correct by construction, never watched |
| Masking switch, abort-and-restart | **Medium.** Every test uses fakes |
| Counters, audio toggle | **Medium.** Simple, untested in reality |
| **Inline masking at recording speed** | **LOW. Never run with the real segmenter on real frames.** This is test 1 |
| Sidecar | **Unproven.** 30 tests, all with hand-made rows. Never seen a real `analyses` row |

---

## The test sequence

Ordered so a failure kills the cheapest work first.

### 1. Masking speed — THE GATE. Do this before anything else.

**Why first:** the masker runs in a child process fed raw frames at
15fps. Segmentation costs tens of milliseconds and there are only ~66ms
per frame. `censor_stream_segment_every` (default 5) segments 3x a second
and carries the mask between, but nobody has measured whether that is
enough on real hardware. If it cannot keep up, inline masking is the
wrong design and tests 2-7 are wasted effort.

**Run:** opt one patient in, force a sensitive verdict, let one masked
session complete.

**Pass:**
- the clip's B2 metadata has `timeline_exact = "true"` and
  `lost_seconds = "0"`
- no `cannot keep up: wrote off …` warnings in the log
- CPU for the child process stays under roughly half a core

**If it fails:** raise `censor_stream_segment_every` and re-run. If it
still fails at 15 (once a second), stop and report — the approach needs
rethinking, not tuning.

### 2. The safety claim — masked clip, covered person, no raw leak

Three checks, and the third is the one to be strict about.

A verdict arrives ~10s *after* the footage it judges, so the recorder
**discards the session so far and restarts masked**. That means the
seconds before the verdict are lost on purpose, and nothing raw and
sensitive should reach B2.

**Pass:**
- a masked clip landed
- watching it, the person is actually covered in flat green
- **no raw clip from before the verdict exists in the bucket** — check by
  listing keys around that timestamp

### 3. Duration truth in the field

Once two 15-minute rolls exist, re-run the roll-pair measurement: find
clips whose successor starts ~900s later, and compare each file's
duration against 900. Should read ~0% where it read 3.8% median before.

This is the direct proof the headline fix works in production rather than
on a laptop.

### 4. Audio alignment, by watching

Play a real clip and check sound matches picture **at the end**, not just
the start. Drift is cumulative, so the first ten seconds always look
fine. No synthetic test can catch this.

### 5. The sidecar against real data

Download a `.json` beside a clip and read it. Expect field-shape
surprises here more than anywhere else — every test used hand-made rows.

**Pass:**
- `detections` spans line up with what the video shows
- `justification` reads sensibly
- a span that produced an alert has `promoted: true` and a matching
  `alert_event_ids`
- offsets actually land on the right moment when you seek to them

### 6. Crash recovery

Kill the backend mid-session (`kill -9`), restart, confirm
`<tmp>/memoria-data-collection` is empty. Those files are **raw
uncensored footage** and nothing cleaned them up before this work.

### 7. Encoder failure

Kill the masking child mid-session. The clip must be **dropped**, not
uploaded — a half-masked file reaching B2 is the failure that matters.

### 8. Counters

After a session, read `RecorderManager.stats` and check the tally matches
what you watched happen.

---

## Sequencing with other testing

- **Nothing here is customer-facing.** With `data_collection_enabled`
  false — the default — none of this code runs at all. It cannot
  interfere with event-gate, detector or identity testing, so it can be
  slotted wherever it is convenient.
- **It shares the detection tick.** Masking and the sensitivity screen
  consume CPU and one extra model call per screened tick on a *collecting*
  camera. If you are measuring detection latency or VLM cost, either turn
  collection off for that run or expect the numbers to move.
- **Only one backend can run detection.** `worker.py` takes a global
  Postgres advisory lock, so a second instance refuses to start the
  worker. Two backends against one database will not double-record — the
  second simply will not detect. Worth knowing if a test looks inert.
- **Test 1 is short and decisive.** Run it early even if the rest waits,
  because a failure changes the plan rather than adding a bug.

---

## Gotchas

**Shared branch.** Several agent sessions commit here. Stage files by
name — never `git add -A`, and never `git add -A -- src/` either, which
still sweeps a neighbour's in-flight work. Check `git status` before
every commit.

**B2 deletes do not free space.** The bucket keeps hidden versions. On
2026-08-05 the dev bucket listed as empty while **348 versions were still
billing the full 10 GB** and had filled the free tier. Deleting needs
`list_object_versions` + delete every version *and* delete-marker. The dev
bucket is now genuinely empty.

**`sensitive=None` is not "clean".** The screen runs on a cadence, and a
failed screen also returns None. None means *no verdict*, so nothing
changes: an unmasked session stays unmasked, a masked one stays masked.
Do not read a null as a negative.

**Masking is sticky per session and only per session.** Once a moment is
judged sensitive the rest of that session is masked; the next session
starts unmasked. A verdict describes the moment it judged, not the camera
forever.

**Room class does not mask anything.** Only a sensitive verdict does. A
bedroom that is not judged sensitive records raw, deliberately.

**The research mask leaks body outline on purpose.** It hugs the body far
more closely than the caregiver-facing pixelation, which snaps to a block
grid precisely to hide the silhouette. That is a different exposure trade,
defensible only because the corpus is opt-in and never reaches a
caregiver. Anything that puts this mode in front of a caregiver is a bug.

**A missing sidecar means "no note", never "nothing happened".** It is
best-effort and always uploaded *after* the clip, so a database failure
costs the note and never the video.

**`timeline_exact = false` means offsets are approximate.** If the encoder
fell far enough behind, the clip is genuinely shorter than the session and
everything after the stall has shifted. The sidecar carries the flag; a
consumer must honour it.

**Never buffer frames in RAM.** Buffering a 1080p clip got a child
OOM-killed on 2026-07-28. The encoder stream-writes for that reason.

**Do not weaken `test_recorder_privacy_drive.py`.** A private-room camera
must abort an open session rather than drift toward upload.

---

## Where the code lives

| Path | What |
|---|---|
| `src/services/data_collection/streaming_encoder.py` | The pacer (timing fix) and `MaskingMp4Writer` |
| `src/services/data_collection/session.py` | Per-camera state machine, audio anchoring |
| `src/services/data_collection/manager.py` | Mux → privacy re-check → region → upload → sidecar; `stats`; `sweep_temp_dir` |
| `src/services/detection/censor.py` | `CENSOR_MODE_RESEARCH`, `mask_stream_sync` |
| `src/services/annotations/sidecar.py` | What a note says (pure) |
| `src/services/annotations/fetch.py` | What goes in it (DB) |
| `src/services/detection/camera_ingest.py` | `_screen_for_masking`, the tick's verdict |

**Settings added:** `censor_stream_segment_every` (5),
`data_collection_sensitivity_every_n_ticks` (3),
`data_collection_keep_sensitive_audio` (true).

---

## Known-unfinished

- **Export-time regeneration.** A sidecar is a snapshot written when the
  clip lands, so it can never carry a reviewer's true/false-positive
  verdict — that is marked hours later. Re-running the same builder at
  export against the current database is the labelled copy. Not built.
- **Coverage.** The gate is still `enabled AND motion AND presence AND
  NOT privacy`, so a person sitting still for more than 45s ends the
  session. Aryaman wants closer to continuous; the lever is the gate, and
  the blocking question is whether presence can come from the always-on
  tracker instead of the VLM tick (a storage decision otherwise becomes a
  much larger inference bill). Not started.
- **Lead-in.** Clips start at the moment of detection, missing the ~10s
  that triggered them. Cheap to restore now the encoder knows what time a
  frame belongs to. Not started.
- **Prod deltas.** Nothing here has run against the production database.
