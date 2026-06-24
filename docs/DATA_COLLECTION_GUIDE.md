# Video Data Collection — Feature Guide & Handoff

*Last updated: 2026-06-24. Audience: anyone on the team (engineering or not) who needs to
understand what this feature does, how to use it, and why it's built the way it is.*

This document has three parts:

1. **What it is** — the product, in plain language.
2. **How to operate it** — turning it on, the env vars, the deploy steps.
3. **How it works & why** — the architecture and the decisions, kept human-readable.

---

## 1. What it is

We can now (optionally, per patient) **store short video + audio clips of a patient to
help improve our AI's understanding of elderly health and disease.** This is a research
data-collection feature. It is **off by default** and only ever runs after a caregiver
**explicitly opts in** for that specific patient.

When it's on for a patient, the system records the camera **only while someone is actually
present in the room** (we never record an empty room), bundles each "someone's here"
stretch into a single video file with sound, and uploads it to our cloud storage bucket.
Caregivers can turn it off again at any time.

### What the caregiver sees

There are two touch-points, on **both the web app and the iOS app**:

- **A consent banner the first time they open a patient.** Plain-language explanation of
  what we store and why, with two buttons: **"Share recordings"** and **"Not now."**
  Either choice is remembered, so the banner never nags them again.
- **A toggle in that patient's Settings** — *"Contribute recordings to research"* — that
  they can flip on or off whenever they want.

The exact words shown:

> **Help improve care for everyone.** With your permission, Memoria will securely store
> short video and audio clips of *[patient]* — only when someone is present — to help
> improve our AI's understanding of elderly health and disease. This is optional, never
> required, and you can turn it off anytime in Settings.

### What gets stored

- One MP4 (video **and** audio) per "presence session" — i.e. per continuous stretch of
  someone being in the room. A long stretch is split into ~15-minute pieces.
- Files land in the **Backblaze B2** bucket under a tidy, sortable path:
  `nemara-data/v1/<caregiver>/<patient>/<year>/<month>/<day>/<time>Z__<room>.mp4`
- Default video quality is 720p H.264 at ~2.5 Mbps (the same the camera already streams).

### What is **not** affected

- **Detection / alerts are completely untouched.** This feature only *reads* the signals
  the detection pipeline already produces; it never changes what the AI detects or when an
  alert fires. If data collection is off (the default), the system behaves exactly as
  before.
- **No new cameras, no new streams, no extra LiveKit cost.** The recorder reuses the video
  frames the detection pipeline already receives.

---

## 2. How to operate it

### Turning it on for a patient
Just use the app: open the patient → **Share recordings** on the banner, or flip the
Settings toggle. That's it. (Under the hood this sets a `data_collection_enabled` flag on
the patient.)

### Environment variables (backend)
The recorder uploads to Backblaze B2. Set these on the backend (Railway in prod, `.env`
locally). All default to empty/safe — **if the B2 credentials are missing, the recorder
simply doesn't upload** (it won't crash), so a box with no creds is harmless.

| Variable | What it is | Example |
|---|---|---|
| `NEMARA_B2_KEY_ID` | B2 application key ID | `0057...` |
| `NEMARA_B2_APP_KEY` | B2 application key (secret) | `K005...` |
| `NEMARA_B2_BUCKET` | bucket name | `nemara-data` |
| `NEMARA_B2_ENDPOINT_URL` | S3-compatible endpoint **for your bucket's region** | `https://s3.us-west-004.backblazeb2.com` |
| `NEMARA_B2_REGION` | the bucket's region | `us-west-004` |
| `NEMARA_DATA_COLLECTION_MAX_CONCURRENT` | max simultaneous recordings per worker (safety cap) | `8` |
| `NEMARA_DATA_COLLECTION_MAX_CLIP_BYTES` | per-file size ceiling | `1500000000` |

> ⚠️ **Endpoint + region must match the region you created the B2 bucket in** (B2
> application keys are region-scoped). If the bucket is in `eu-central-003`, set both
> accordingly.

### Deploy checklist (prod)
Order matters — **the database change goes first** (see `PROD_MIGRATION_RUNBOOK.md`):

1. **Apply the DB delta** to prod Supabase (adds two patient columns; idempotent, additive,
   safe to re-run):
   ```bash
   psql "<prod NEMARA_DB_URL without +asyncpg>" -f scripts/apply_data_collection_delta.sql
   ```
2. **Set the `NEMARA_B2_*` env vars** on Railway.
3. **Merge + deploy** the backend (`railway up`) and the two apps.
4. **Smoke test:** opt one test patient in, have someone walk in front of the camera, and
   confirm a clip appears in the B2 bucket under that patient's path.

### Where the code lives
- Backend recorder: `src/services/data_collection/` (self-contained) + two read-only hooks
  in `src/services/detection/room_worker.py` and `src/services/detection/camera_ingest.py`.
- The patient flag: `src/db/patient.py`, `src/schemas/patient.py`, `src/api/patients.py`,
  `scripts/apply_data_collection_delta.sql`.
- Web consent UI: `nemara-web-app` (`DataCollectionConsentModal`, `DataCollectionSettings`).
- iOS consent UI: `nemara-caregiver-app` (`DataCollectionConsentSheet`, `DataCollectionCard`).

---

## 3. How it works & why

### The flow, end to end

```
 Camera ──video+audio──▶ Detection backend (already running, one worker per room)
                              │
                              │  every frame is ALSO forked (a cheap copy) to:
                              ▼
                        RecorderManager ── one recording "session" per camera
                              │
   each ~10s detection tick asks: "should we be recording right now?"
        = patient opted in  AND  motion  AND  someone present
                              │
              yes ──▶ write frames to ffmpeg (H.264, 720p, 15fps)
                       + write audio to a side file
                              │
        person leaves ──▶ wait ~45s (in case they come back) ──▶ finalize:
                              │
                       mux video + audio into one MP4 ──▶ upload to Backblaze B2 ──▶ delete local temp
```

The key idea: **the recorder is a second reader of frames the detection pipeline already
has.** Detection decodes every camera frame anyway; we just hand a copy to the recorder.
That's why there's no extra streaming cost and no second camera connection.

### The gate: when do we record?
Exactly one rule, reusing signals detection already computes:

> **record = patient opted in  AND  there's motion  AND  someone is present**

"Someone is present" is detection's own presence verdict (the same one that drives alerts);
"motion" is the camera's motion signal. We never re-run any AI for the recorder — we just
read these. So recording can never disagree with "is anyone actually here."

### One file per "presence session"
When the gate turns on, we start one recording. We keep recording while people are around.
If they briefly leave and come back within ~45 seconds, it stays one file (people flicker
in and out of view). If they're gone longer, we finalize the file. A very long session is
cut into ~15-minute pieces so no single file is huge.

Each finished file gets audio mixed in and is uploaded to B2 under a date-organized path,
then the local copy is deleted.

### The decisions, and why

These are the choices that shaped the build. Most were deliberate trade-offs for **v1:
keep it cheap and simple, leave the door open for later.**

- **We built our own recorder instead of using LiveKit's "Egress" recording service.**
  Egress *with audio* is billed per-minute (~$0.02/min) and at our scale would cost on the
  order of **$14k/month at 50 homes**. Our in-process recorder reuses frames we already
  have, so it costs essentially nothing extra. *(Egress stays documented as a fallback if
  we ever want to trade money for operational simplicity.)*

- **We record in H.264, not the more space-efficient HEVC/AV1.** H.264 is cheap on CPU
  (~a tenth of a core per active camera); HEVC would cost a lot more CPU on every recording
  box. Storage is the cheap axis here, CPU is the expensive one. *If storage ever becomes
  the pain point, the plan is to re-compress to HEVC **offline**, on separate compute, not
  on the live recording path.*

- **We store on Backblaze B2.** It's the cheapest simple, S3-compatible object store
  (~$6/TB-month), and its egress allowance comfortably covers how we plan to read the data
  back (occasional, sequential, for offline analysis — not a constant firehose). See
  `india-video-storage-cost-model` notes for the full math.

- **We reuse the frames detection already buffered as the clip's "lead-in."** When a
  session starts, we want the few seconds *before* the AI confirmed a person (the AI takes
  a moment to decide). Rather than keep a second, memory-hungry buffer of full-rate frames
  for every camera, we reuse the small 4-frames-per-second window detection already keeps.
  Zero extra memory.

- **The recorder runs inside the detection process, and we scale by adding more detection
  workers — not by splitting recording into its own service.** A separate recording service
  would have to subscribe to the camera streams a *second* time, which doubles the LiveKit
  bandwidth bill. Keeping it in-process avoids that.

- **A slow upload or encoder can never slow down detection.** Handing a frame to the
  recorder is a non-blocking, instant operation; if the encoder ever falls behind it quietly
  drops frames rather than make detection wait. Uploads happen in the background with
  automatic retries.

- **Consent is explicit, per-patient, default-off, and revocable — and we record *when*
  the decision was made.** This is sensitive audio/video of a vulnerable person, so the bar
  for "yes" is an explicit tap, never a pre-checked box, and we keep an audit timestamp of
  the decision.

### Things intentionally left for later (not built in v1)
These were conscious "not yet" calls, recorded here so they're not forgotten:

1. **Offline re-compression to HEVC/AV1** to shrink stored files — only if/when storage cost
   hurts, and always on separate compute, never the live path.
2. **A cold-storage tier** (e.g. Glacier-style, ~$1/TB) once the archive gets very large.
3. **Data residency** — B2's default regions are US/EU. India's DPDP rules may eventually
   require clips be stored in-country (e.g. AWS Mumbai); that's a config change when needed.
4. **Precise audio/video sync.** v1 sync is "good enough for offline analysis" (both streams
   anchored to the session start). If frame-accurate sync ever matters, there's a known
   upgrade path.
5. **Higher resolution (1080p).** We stay at 720p to match the camera's current stream.

### ⚠️ Legal / ethical — needs counsel before scaling
This feature stores **identifiable audio and video of elderly people, some of whom may have
diminished capacity to consent.** Before we scale data collection broadly we should get
legal review of (a) **who can validly consent** on a patient's behalf, and (b) **biometric/
health-data obligations** (e.g. India's DPDP, and equivalents elsewhere). This is **not a
code blocker** — the machinery is built and off-by-default — but it **is** a go-to-market
blocker for turning it on at scale.

---

## Appendix — what's tested, and a couple of pre-existing notes

**Tested (all green):**
- The patient flag + API (defaults off, toggling stamps the consent timestamp) — integration
  tests against a real local DB.
- Every recorder piece — key building, B2 upload wiring, the H.264 streaming encoder (real
  ffmpeg), the session state machine (start / linger / roll / finalize), the manager
  (mux + upload + cleanup + retry + concurrency cap + graceful shutdown), and an
  **end-to-end test** that drives a synthetic presence session through real ffmpeg and
  asserts one valid audio+video clip is produced and "uploaded" with the right path.
- Web consent UI: component unit tests + typecheck + production build.
- iOS consent UI: full simulator build.

**Pre-existing notes (unrelated to this feature, surfaced while testing):**
- The local backend test suite has **9 failing tests** in `test_prompt_builder.py` and
  `test_state_machine*.py` that predate this work — they come from the recent `main` merge
  (a prompt-builder change and a `cameras` schema change where the tests weren't updated).
  They are **not** touched by this feature.
- Onboarding now seeds a 5th preset detector named `setup_test` for every new patient, which
  breaks two patient-creation tests that expect exactly 4. Also from the merge — worth a
  look to confirm `setup_test` is intended for production.
- Two columns the merge added to `schema.sql` (`caregivers.vlm_provider`,
  `cameras.last_seen_at`) were missing from the **local** dev database; they were applied
  additively (no data lost). Prod should already have them via the normal delta process.
