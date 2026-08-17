# Handoff — B2 data collection: annotations + recording censoring

> **For Claude:** this is a DISCUSSION-FIRST task. Two open design questions
> need Aryaman's decision before any code. Do not start building from the June
> design docs — they are superseded in the places noted below.

---

## Current state — what is actually true

The recorder is **built, deployed and exercised against a real Backblaze
bucket.** This is not a greenfield or a verify-it-works task.

`src/services/data_collection/` is complete: `gate.py`, `session.py`,
`manager.py`, `streaming_encoder.py` (stream-writes, never buffers frames),
`mux.py`, `b2.py`, `keys.py`. Wired into the tick path via
`src/services/detection/camera_ingest.py`. 9 test files, all passing.

The gate (`gate.py:39`) is settled and uses only pre-existing signals:

```python
return bool(data_collection_enabled and motion_in_clip and gate_people_present)
```

**Encoding is settled. No transcode or codec work is wanted.** An earlier
version of this handoff proposed an HEVC batch job from the June design doc —
that is not on the table. Do not raise it.

Today's object layout:

- **Key:** `<prefix>/<caregiver_id>/<patient_id>/<YYYY-MM-DD>/<HHMMSS>Z__<room-slug>.mp4`
  (`keys.py:54`)
- **Metadata:** flat S3 object metadata, string values only (`b2.py:73`)

---

## Discussion 1 — video annotations (Aryaman's primary ask)

**The need:** an efficient way to key into the raw video. Today a clip is an
opaque MP4 plus a handful of session-level strings. There is no way to ask "show
me the three seconds where the fall was detected" or "find every clip where
identity was confirmed" without watching the footage.

**Why the current mechanism cannot carry it:** S3 object metadata is capped at
~2KB and is string-valued key/value only. It can hold "which patient, which
room, which gate fired" and nothing time-indexed.

**The detection pipeline already produces exactly the data wanted** — every tick
writes an `analyses` row with `detector_fields` (per-detector confidence and
reasoning), `people_present`, `motion_in_clip`, identity confidence, track ids,
all stamped `captured_at`. The question is not how to generate annotations; it
is where they should live relative to the video.

**Options to put to Aryaman:**

1. **Sidecar JSON object** beside the MP4 (same key, `.json`). Simple, portable,
   travels to a data partner with the file. Costs a second object per clip and
   is not queryable without fetching.
2. **Embedded timed-metadata track inside the MP4.** Travels as one artifact and
   seeks natively in players/tools. More encoding work — and note the constraint
   above that encoding changes are unwanted, so this may be out on those grounds
   alone.
3. **Postgres table keyed by clip key.** Queryable across the whole corpus
   without touching B2 — best for internal analysis, worst for handing a
   partner a self-describing dataset.
4. **Hybrid:** DB for querying, sidecar written at finalize for portability.

**Questions that decide it:**
- Is the consumer *us* (eval, FP analysis) or an *external data partner*? That
  single answer probably picks the option.
- Should an annotation be per-tick, or one summary per clip session?
- Must a clip be self-describing once it leaves our infrastructure?

---

## Discussion 2 — recordings are RAW (pilot master worklist, item 6)

From `docs/plans/2026-07-28-pilot-master-worklist.md`:

> **6. [ ] Research/B2 recording censoring (recordings currently RAW) — gates
> consent PDF → gates attestation → gates research-consent pilots.**

Collected recordings are **uncensored**, unlike event clips, which go through
`src/services/detection/censor.py`. This is a dependency chain, not a standalone
item: censoring gates the consent PDF, which gates attestation, which gates any
research-consent pilot.

Related worklist items in the same chain: **7** consent attestation + BIPA
biometrics clause, **8** sensitivity eval corpus with a measured false-negative
rate, **9** censor soak on the Railway substrate (local numbers are Docker-VM
numbers and do not transfer).

**Never exercised** (worklist line 200): the data-collection exclusion field
test — research consent on, motion on both cameras, and B2 receives **only**
non-private clips. The privacy-exclusion path has never been tested in the
field.

---

## Standing constraint — read before proposing anything

Decided 2026-07-29, recorded in the worklist:

> Pilot families are **OBSERVATION-FIRST**. Research consent is NOT assumed from
> day one. The data-collection flag ships **OFF**; any research-consent home is
> opt-in per family, later. **Do NOT build anything that presumes a consented
> corpus exists.**

Items 6–9 stay where they are and are explicitly *not* pulled forward for the
pilot. So neither discussion above is pilot-blocking — they are the next
substantive build once the pilot is running.

---

## Start here

| | |
|---|---|
| Repo | `nemara-backend` |
| Branch | cut from `main` once a direction is agreed |
| `main` | clean; some unpushed commits are unrelated security work — leave them |
| Read first | `docs/plans/2026-07-28-pilot-master-worklist.md` items 6–9 and line 200 |
| Superseded | the June design docs, on encoding and on anything assuming consent |

**Baseline:** `.venv/bin/pytest test/unit -q` → **996 passed, 0 failed, 0 errors**. Anything beyond that is yours. The integration suite
cannot reach its fixtures in this environment; that is also pre-existing.

**Gotchas:** never buffer frames in RAM (`streaming_encoder.py` stream-writes
because buffering a 1080p clip got a child OOM-killed on 2026-07-28), and do not
weaken `test_recorder_privacy_drive.py` — a private-room camera must abort an
open session rather than let it drift toward upload.
