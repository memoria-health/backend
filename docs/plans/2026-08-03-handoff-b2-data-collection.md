# Handoff — B2 video data-collection recorder

> **For Claude:** this is a FINISH-AND-VERIFY task, not a greenfield build. Most
> of it exists and is tested. Read this whole file before writing code.

**Goal:** get collected video actually landing in Backblaze B2 for real, and add
the cold-storage transcode job that was designed but never built.

---

## Start here

| | |
|---|---|
| Repo | `nemara-backend` |
| Branch | cut a new one from `main` — `feat/b2-transcode-and-verify` |
| `main` state | clean, but **4 commits unpushed** (security work — leave them alone, they are unrelated) |
| Design docs | `docs/plans/2026-06-24-video-data-collection-recorder.md`, `docs/plans/2026-06-24-data-collection-consent-ui.md` |

Do **not** start from any other branch. The other repos are irrelevant to this
task except for the consent toggle, which is already shipped.

---

## What already exists (do not rebuild)

`src/services/data_collection/` is a complete package:

| File | Role |
|---|---|
| `gate.py` | `should_collect()` — the ONE decision point |
| `session.py` | `RecorderCtx`, per-session lifecycle |
| `manager.py` | `RecorderManager`, owned by the RoomWorker |
| `streaming_encoder.py` | ffmpeg stream-write — never buffers frames in RAM |
| `mux.py` | audio/video muxing |
| `b2.py` | S3-compatible upload, `b2_configured()` |
| `keys.py` | object key scheme |

**Wired into the tick path already** — `src/services/detection/camera_ingest.py`
imports `should_collect`, `RecorderCtx` and `RecorderManager`. When the recorder
is absent every hook is a deliberate no-op (see the comment at
`camera_ingest.py:240`).

**The gate** (`gate.py:39`) is exactly the designed contract — no new signals:

```python
return bool(data_collection_enabled and motion_in_clip and gate_people_present)
```

**Tests: 9 files, all passing.** `test_data_collection_{b2,e2e,gate,keys,mux}.py`,
`test_recorder_{manager,privacy_drive}.py`, `test_session_recorder.py`,
`test_camera_ingest_recorder.py`. Run them before you touch anything so you know
your baseline.

**Settings** (`src/settings.py:425`): `b2_key_id`, `b2_app_key` (both default
`""`), `b2_bucket` (`memoria-data`), `b2_endpoint_url`, `b2_region`. Region
routing for India isolation lives in `services/region.b2_bucket_for`.

**Consent** is shipped: `patients.data_collection_enabled` +
`data_collection_decided_at`, with UI on iOS and web.

---

## What is missing — the actual work

### 1. The batch transcode job (the main build)

The design called for an **off-hot-path** job transcoding cold H.264 clips to
HEVC, replacing the originals and updating object metadata. It does not exist —
there is no such module in `data_collection/`.

Requirements from the design, restated so you don't have to re-derive them:

- **Decoupled from the recorder.** The real-time path must never wait on it.
- Operates on **cold** objects only — do not transcode something still being
  written. Use object age, not a flag you have to keep in sync.
- Replaces the original and updates metadata.
- AV1 was explicitly optional. Do not build it unless HEVC lands first.

Open question for Aryaman, **ask before building**: where does this run? A
Railway cron, a separate worker, or on-demand? That changes the design
materially and is not recorded anywhere.

### 2. It has never run against a real bucket

`b2_key_id` and `b2_app_key` default to `""`, and `b2_configured()` exists
precisely so everything no-ops without them. **Every passing test is against a
fake.** Nobody has confirmed a byte reaching Backblaze.

Before anything else: get credentials into a local `.env`, record one real
session, and confirm the object in the bucket — right key scheme, right
metadata, playable.

### 3. Deferred decisions still owe a partner handoff doc

The v1 design deliberately deferred a set of items that must be written up for
the data partner. They are listed in the design doc; they have never been
collated. Ask Aryaman for the partner context before writing it.

---

## Verification

1. `.venv/bin/pytest test/unit -q` — baseline is **2 failed, 932 passed, 21
   errors**. The 2 failures (`test_face_embedding_repo`) and the 21 errors are
   **pre-existing and unrelated**; do not chase them. Anything beyond that is
   yours.
2. Real B2 round trip: record → upload → fetch the object back → play it.
3. Transcode job: cold clip in, HEVC out, original replaced, metadata correct,
   and confirm the recorder's latency is untouched while it runs.

---

## Gotchas

- **Never buffer frames in RAM.** `streaming_encoder.py` stream-writes for a
  reason: buffering a 1080p clip got a child process OOM-killed on 2026-07-28.
- **Privacy interacts with this.** `test_recorder_privacy_drive.py` exists
  because a private-room camera must abort an open session rather than let it
  drift toward upload. Do not weaken that path.
- **The unit suite is the only one that runs.** The integration suite cannot
  reach its fixtures in this environment — that is pre-existing, not something
  you broke, and not worth fixing as part of this.
- Do not change `should_collect()`. The gate is settled and uses only existing
  signals by design.
