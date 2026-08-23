# Video Data-Collection Recorder — Implementation Plan (v1, ACTIVE)

> Status (2026-07-27): implemented — `src/services/data_collection/` (b2, gate, keys, manager, mux, session, streaming_encoder) all live; hooks wired in `src/services/detection/room_worker.py` + `camera_ingest.py`; consent columns on `src/db/patient.py`; companion consent UI also shipped (see 2026-06-24-data-collection-consent-ui.md).

> **STATUS — ACTIVE (chosen path).** We evaluated **LiveKit Egress** and rejected it on cost: audio requires *Track Composite* egress at **$0.02/min ≈ $14k/mo at 50 homes**, vs **~$0** for this in-process recorder (it reuses frames detection already decodes — no extra LiveKit subscription, no double-charge). This is the v1 plan. The egress option stays documented as the at-large-scale fallback.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a per-patient `data_collection_enabled` flag is on, record the camera's stream to **one MP4 per presence session** (video **+ audio**), gated on the *existing* motion + person/face signals, and upload to **Backblaze B2** — all **in the detection backend, reusing the frames it already has**.

**Architecture:** A second consumer of the frames already flowing through `RoomWorker._consume_video` (full 15fps, *before* the 4fps detector downsample). A `RecorderManager` owns one `SessionRecorder` per actively-collecting camera. Each session: (a) **seeds a 4fps lead-in** from the `FrameBuffer` window detection already holds — the exact frames analyzed — up to the tick where `people_present` (or `patient_in_clip`, when face embeddings are enrolled) first fires; then (b) records **15fps live forward** until the gate goes false (+ ~45s linger), with a **15-min max roll**. Video is encoded **H.264** (cheap CPU) by a **CFR pacer** (writes the latest frame at 15fps wall-clock, duplicating when stale — handles variable LiveKit fps and the 4fps lead-in, and isolates back-pressure). Audio is captured as a **side PCM file** and **muxed in once at finalize** (`-c:v copy -c:a aac`). The finished MP4 streams to B2 at `data/v1/<caregiver_id>/<patient_id>/<YYYY>/<MM>/<DD>/<HHMMSS>Z__<room-slug>.mp4` + S3 object metadata. The tick (`camera_ingest._run_tick`) only computes a boolean `should_collect` from existing signals and pokes the manager — **detection logic is never touched.**

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy async, pydantic-settings, pytest + pytest-asyncio (`asyncio_mode=auto`), `ffmpeg` subprocess (libx264 + AAC; **no PyAV in v1**), `aioboto3` for B2.

---

## Locked decisions (authoritative — the cross-check)

| # | Decision | Why |
|---|---|---|
| 1 | **DIY in-process recorder**, not LiveKit Egress | Egress w/ audio = $0.02/min ≈ $14k/mo @ 50 homes; DIY ≈ $0 (reuses detection's frames) |
| 2 | Tap frames in `_consume_video` (full 15fps, pre-downsample) | **Verified**: `_push_video` runs per-frame before `frame_buffer.push` |
| 3 | Gate = `data_collection_enabled AND motion_in_clip AND gate_people_present` | Reuses existing signals; `people_present` is **always-on, detector-independent** (verified) |
| 4 | **Codec H.264** (libx264), not HEVC | Low CPU per active camera (~0.05–0.15 core); storage is the cheap axis. HEVC only via *offline* transcode later if storage ever hurts |
| 5 | **One MP4 / presence session**, 15-min max roll, ~45s gate-off linger | Bounds file size; linger bridges flicker |
| 6 | **4fps lead-in** from existing `FrameBuffer` window → then 15fps live | Captures the analyzed window (LLM latency) with **zero extra RAM** (no full-rate look-back buffer) |
| 7 | **Audio**: side PCM file + single finalize mux (`-c:v copy -c:a aac`) | Avoids live A/V-sync; **approximate sync** (both anchored to session start) is fine for offline analysis |
| 8 | **CFR pacer**, drop-under-pressure, non-blocking `feed()` | Variable LiveKit fps; a slow encoder can never stall `_consume_video`/detection |
| 9 | **Resolution: scale-to-fixed 720p**, never roll | Normalize like real recorders; jitter → brief soft frames, not file storms |
| 10 | **B2** (S3-compatible), multipart upload | Cheapest simple object store; clips ~280 MB need multipart |
| 11 | **In-process + shard the worker fleet** to scale; do NOT separate recording | A separate service must re-subscribe → LiveKit double-charges bandwidth |
| 12 | **Consent**: `data_collection_enabled` + `data_collection_decided_at` | Audit trail; `decided_at IS NULL` drives the first-open banner |

**Verified in code (2026-06-24):** full-rate tap exists (`_push_video` pre-downsample); I420 byte layout correct (same `bytes(frame.convert(I420).data)` `mp4_remux` already feeds ffmpeg); `people_present` is an `always_on` block independent of detector config; both video + audio carry shared `time.monotonic()` timestamps.

**⚠️ UI is a SEPARATE plan (not this one).** This plan is **backend only**. The consent banner + settings toggle on **web + iOS** is a companion plan that must ship with the feature — see "Companion plan needed" at the end. This backend plan provides the flag + `decided_at` the UI drives.

---

## Conventions (apply to every task)
- **Branch, never main.** Create `feat/data-collection-recorder` first. (Cloud-sync caveat: use plumbing if a checkout aborts — `docs/git-incident-2026-04-15.md`.)
- **TDD rhythm:** failing test → run (fail) → minimal impl → run (pass) → commit. Bite-sized steps.
- **Run one test:** `uv run pytest <path>::<test> -v`.
- **Never modify detection logic.** `camera_ingest.py` `gate_people_present` + the motion gate are read-only inputs.
- **Secrets** (`NEMARA_B2_*`) live in `.env` / Railway, never committed.
- **DB-before-deploy:** the patient column delta runs on prod Supabase *before* the backend deploys (Task 13).

---

## Phase 1 — Per-patient flag + consent state (ships independently)

### Task 1: `data_collection_enabled` + `data_collection_decided_at` columns
**Files:** `src/db/patient.py`, `supabase/schema.sql`, `scripts/deltas/apply_data_collection_delta.sql`, `test/integration/test_patient_data_collection_flag.py`.

TDD: test that a new patient defaults `data_collection_enabled is False` and `data_collection_decided_at is None`. Implement:
```python
# src/db/patient.py
data_collection_enabled: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), default=False)
data_collection_decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)
```
```sql
-- supabase/schema.sql (patients table)
data_collection_enabled  boolean not null default false,
data_collection_decided_at timestamptz,
-- scripts/deltas/apply_data_collection_delta.sql  (idempotent prod delta)
alter table public.patients add column if not exists data_collection_enabled boolean not null default false;
alter table public.patients add column if not exists data_collection_decided_at timestamptz;
```
Commit: `feat(db): per-patient data_collection flag + consent timestamp + delta`.

### Task 2: expose on the patient API (read + toggle; stamps `decided_at`)
**Files:** `src/schemas/patient.py` (`UpdatePatientRequest` + `PatientResponse` gain `data_collection_enabled`; response also exposes `data_collection_decided_at`), `src/api/patients.py`.
TDD: PATCH `{data_collection_enabled: true}` → 200, response reflects it, and `data_collection_decided_at` is now set (the handler stamps it whenever `data_collection_enabled` is in the payload). Commit: `feat(api): toggle data_collection + stamp consent timestamp`.

---

## Phase 2 — Keys + B2 uploader (pure / I-O, no detection coupling)

### Task 3: `slugify` + `build_clip_key` (pure)
`src/services/data_collection/keys.py` + unit tests. Key = `data/v1/<caregiver_id>/<patient_id>/<YYYY>/<MM>/<DD>/<HHMMSS>Z__<room-slug>.mp4` (UTC, no colons). Commit.

### Task 4: B2 uploader (multipart-capable) + settings
**Files:** `pyproject.toml` (add `aioboto3`), `src/settings.py`, `src/services/data_collection/b2.py`, unit test (stub the client).
Settings (env-prefixed `NEMARA_`):
```python
b2_key_id: str = Field(default="")
b2_app_key: str = Field(default="")
b2_bucket: str = Field(default="memoria-data")
b2_endpoint_url: str = Field(default="https://s3.us-west-004.backblazeb2.com")
b2_region: str = Field(default="us-west-004")
data_collection_max_clip_bytes: int = Field(default=1_500_000_000)  # ~15min H.264 720p headroom
data_collection_max_concurrent: int = Field(default=8)              # session cap (Task 11)
```
`upload_file_to_b2(path, key, metadata, settings)` uses `aioboto3` `upload_file` (multipart automatic for big files) with `ExtraArgs={ContentType: video/mp4, Metadata: {...}}`. Commit.

---

## Phase 3 — Streaming encoder (CFR pacer, H.264, no PyAV)

### Task 5: `StreamingMp4Writer`
**Files:** `src/services/data_collection/streaming_encoder.py`, ffmpeg-gated unit tests.

**Design:** one `ffmpeg` reads raw `yuv420p` from `pipe:0` at fixed `-r 15`, **H.264** (`libx264 -preset ultrafast -tune zerolatency`), fragmented MP4 (`+frag_keyframe+empty_moov+default_base_moof`) to a temp **video** file (crash-safe, no RAM buffering). A **CFR pacer** task ticks at 15fps wall-clock and writes the **latest frame**, duplicating when none is newer — this gives correct-duration 15fps output across (a) the 4fps lead-in and (b) variable live fps. `feed(i420)` is **synchronous, non-blocking**: it scales the frame to 1280×720 if off-size (resolution normalization) and overwrites the latest-frame slot; if the encoder is wedged the pacer drops (counted), never blocking `_consume_video`. `aclose()` flushes, closes stdin, awaits exit, returns the path.

TDD: (1) feeding 30 frames yields a valid `ftyp` MP4; (2) `feed()` never blocks/raises when the pacer is stalled (drop counter increments); (3) an off-size frame is scaled, not corrupting the file. Commit.

---

## Phase 4 — Session machine + manager (lead-in, linger, roll, audio, upload)

### Task 6: `should_collect` (pure) — `data_collection_enabled AND motion_in_clip AND gate_people_present`. Unit-test the truth table. Commit.

### Task 7: `SessionRecorder` (injected clock + encoder factory)
States IDLE → RECORDING → LINGERING → IDLE driven by `on_tick(collect, lead_in_frames)`. On first start it **seeds the lead-in**: writes the provided 4fps `FrameBuffer` snapshot frames (held to their timestamps) into the encoder, then routes live frames. 15-min roll finalizes + restarts; 45s linger bridges flicker; finalize calls `on_finalized(video_path, audio_path, ctx)`. Unit-test with fakes (no ffmpeg/real time). Commit.

### Task 8: `RecorderManager` (+ audio side-file + finalize mux + upload)
`dict[camera_id → SessionRecorder]`. `on_tick(camera_id, collect, ctx, lead_in)` finds-or-creates (respecting `data_collection_max_concurrent`); `feed_video(camera_id, i420)` and `feed_audio(camera_id, pcm)` route to the active session (both non-blocking; audio appends to `<session>.audio.pcm`, seeded from the existing `AudioBuffer` window to match the video lead-in). `on_finalized` runs **one ffmpeg mux** (`-i video.mp4 -f s16le -ar 16000 -ac 1 -i audio.pcm -c:v copy -c:a aac -movflags +faststart final.mp4`), uploads `final.mp4` to B2 (`asyncio.create_task`, exception-logged, **retry w/ backoff**), then deletes temps. Unit-test with fakes. Commit.

---

## Phase 5 — Two surgical hooks (detection untouched)

### Task 9: fork video + audio frames
In `RoomWorker._consume_video._push_video`, after the existing `frame_buffer.push`, add `self._recorder_manager.feed_video(camera_id, i420)`. Mirror in the audio consumer with `feed_audio`. Both O(1) no-ops unless a session is active. Resolve `identity → camera_id/patient_id/caregiver_id/room` via the cameras the worker already holds. Unit-test the fork. Commit.

### Task 10: compute `should_collect` + seed lead-in in the tick
In `_run_tick`, **after** `gate_people_present` is computed: `collect = should_collect(patient.data_collection_enabled, motion_in_clip, gate_people_present)`; when a session is *starting*, pass the existing `frame_buffer.snapshot_last_seconds(window)` as the lead-in; `await self._recorder_manager.on_tick(camera_id, collect, ctx=RecorderCtx(caregiver_id, patient_id, room, gate=("poi" if identity.patient_in_clip else "person"), confidence), lead_in=...)`. Read-only on all detection state. Unit-test the decision + lead-in handoff. Commit.

---

## Phase 6 — Ops hardening, E2E, deploy

### Task 11: ops defaults (the "7–12" list)
- **Concurrent-session cap** (`data_collection_max_concurrent`): over cap → don't start, log (semaphore).
- **Temp disk**: clips under a temp dir, deleted post-upload; bounded by cap × max-clip-bytes.
- **Upload retry**: backoff ×3, then log + drop; orphaned temp cleanup on failure.
- **Graceful shutdown**: FastAPI lifespan stop → finalize + upload in-flight sessions (best-effort, timeout). Hard crash loses in-flight (acceptable).
Unit-test cap + retry + shutdown-finalize. Commit.

### Task 12: end-to-end test (mocked B2, real ffmpeg)
Synthetic tick sequence: flag on + motion + presence for N ticks → absence. Assert exactly one MP4 finalized, lead-in present, audio muxed, one B2 upload with the expected key. Commit.

### Task 13: deploy (DB delta BEFORE deploy — per the prod runbook)
① `scripts/deltas/apply_data_collection_delta.sql` on prod Supabase → ② Railway env (`NEMARA_B2_*`) → ③ PR + merge → ④ `railway up` → ⑤ smoke-test one opted-in patient lands a clip in B2.

---

## Scaling (decided)
Recording stays **in-process** with detection (reuses the single LiveKit subscription → no double-charge). To scale, **shard the worker fleet by room** — each instance subscribes to its rooms once and records its own cameras in-process. Do **not** split recording into its own service (it would re-subscribe and double LiveKit bandwidth). At extreme scale, LiveKit Egress is the only "separate service that reuses the stream," at its per-minute price.

## Companion plan needed (UI — NOT in this plan)
**Consent banner + settings toggle, web + iOS.** Drives `data_collection_enabled` / `data_collection_decided_at`:
- **First-open banner** when `decided_at IS NULL`: plain-language modal (web) / sheet (iOS) — *what's stored* (video + audio of the patient, only when present) and *why* (improve AI understanding of elderly health/disease), **Share / Not now**, records the decision.
- **Settings toggle** ("Contribute recordings to research"), flippable anytime, same copy; default **OFF** until explicit opt-in; revocable.
- Shared backend flag; consent is sensitive A/V (DPDP) — explicit, not pre-checked, audited via `decided_at`.
Write this as its own plan after a quick read of the web + caregiver-app patterns.

## Verification checklist
- [ ] Nothing under `src/services/detection/` changed except the two read-only hooks (video+audio fork; tick `should_collect`/lead-in). Detection verdicts untouched.
- [ ] Codec is **H.264**; per-active-camera CPU ~0.1 core; idle cameras pay ~zero (O(1) no-op feeds).
- [ ] **No full-rate look-back buffer** — lead-in reuses the existing 4fps `FrameBuffer`/`AudioBuffer` windows.
- [ ] Back-pressure never reaches `_consume_video` (sync non-blocking feed; drop-under-pressure counted).
- [ ] One MP4/session, audio muxed at finalize, 15-min roll, 45s linger, resolution scaled-to-fixed.
- [ ] Concurrent-session cap, temp cleanup, multipart upload + retry, finalize-on-shutdown all present.
- [ ] DB delta is idempotent and ran on prod before deploy; flag defaults **false**; nothing records until opt-in.
- [ ] B2 creds only in `.env`/Railway.
- [ ] Companion UI plan (web + iOS consent) tracked separately before feature ships.
