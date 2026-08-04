# Handoff — wire up the review gate (human evaluation of alerts)

> **For Claude:** the backend is built and locally tested by its author. It is
> **not runnable end-to-end** — one function it calls does not exist. Read this
> whole file before writing code.

**Goal:** make the human-in-the-loop alert review loop actually work — reviewer
gets notified, opens a queue, approves or rejects, and that decision reaches the
caregiver.

**Why it matters:** this is the answer to the false-positive problem. Aryaman's
stated top product risk is over-firing ("FPs are what kill product interest and
customers will delete it right away"). This gate lets a human remove false
alarms before a caregiver ever sees them.

---

## Start here

| | |
|---|---|
| Repo | `nemara-backend` (plus one client, see below) |
| Branch | cut a new one from `main` — `feat/review-gate-wiring` |
| `main` state | clean, **4 commits unpushed** (unrelated security work — leave alone) |
| Author's notes | `docs/plans/2026-08-02-sunny-handover.md` — read the review-gate section |

---

## The design (already decided, don't relitigate)

`src/services/review_gate.py::route_events()` replaced the direct push+email
fan-out. When `patients.require_review` is on (**default FALSE**), events are
held at `events.review_gate='pending'` instead of dispatching.

**The safety property that governs everything:** human review can only *remove*
false alarms, never block a real one. There is a **60-second auto-release** — if
no reviewer acts, the alert goes out anyway. Do not add anything that can hold
an alert indefinitely.

Clip handling is already clever and worth preserving: `clip_archival.py` has a
fast path that uploads the **pre-event MP4** immediately for held events
(`sub_key="review-pre"`, stamped on `events.review_clip_key`), so the reviewer
sees video in ~5s rather than waiting out the post-roll. **The reviewer sees
exactly the frames the model judged** — that is the point, keep it.

---

## What exists

- `src/services/review_gate.py` — the gate itself
- `src/api/review.py` — `GET /v1/review/queue`, `POST /v1/review/{id}`,
  `POST /v1/review/toggle`, `POST /v1/review/device`
- `scripts/apply_review_gate_delta.sql` — `patients.require_review`,
  `events.review_gate`, `events.review_clip_key`, `review_devices` table
- Fast-path clip upload in `src/services/clip_archival.py`
- RLS on `review_devices` — **added 2026-08-03**, see
  `scripts/apply_rls_sleep_activity_review_delta.sql`. That table holds APNs
  tokens and shipped without RLS; do not undo it.

---

## The three gaps — this is the work

### 1. `push.dispatch_review_pushes` does not exist (hard blocker)

`review_gate.py:45` does:

```python
asyncio.create_task(push.dispatch_review_pushes(...))
```

**There is no such function anywhere in `src/`.** Grep it yourself. Every time an
event is held, that task raises and the reviewer is never notified — so the only
outcome is the 60s auto-release, i.e. the gate silently does nothing.

Implement it in `src/services/push.py` alongside the existing APNs dispatch. It
must fan out to rows in `review_devices` (not `caregiver_devices` — different
audience, different table). Note `review_devices.environment` defaults to
`'sandbox'`; honour it, or prod pushes silently vanish.

### 2. `REVIEWER_EMAILS = {"admin@gmail.com"}` is hardcoded

`src/api/review.py:18`. A literal placeholder gates who can see **every
patient's** held events across all accounts. The author's own note says "move to
settings/DB before real use."

Ask Aryaman which he wants — a settings-based allow-list or a DB role — before
building. Do not invent a permissions model.

### 3. No reviewer UI exists

The API has no consumer. Grep for `review/queue` across `nemara-caregiver-app`
and `nemara-web-app`: **zero hits**.

Ask Aryaman where this belongs before starting. His stated intent was *"I will
make expert review a toggle where our experts will review some masked version of
the video"* — note **masked**. The censoring pipeline in
`src/services/detection/censor.py` already produces masked renditions; a
reviewer UI should almost certainly serve those rather than raw footage,
especially for private rooms. That is a product decision, not yours to assume.

---

## Verification

1. `.venv/bin/pytest test/unit -q` — baseline **2 failed, 932 passed, 21
   errors**. Those 2 failures and 21 errors are pre-existing and unrelated.
2. Apply `scripts/apply_review_gate_delta.sql` locally (it has **not** run on
   prod — it is one of 23 pending deltas).
3. Full loop by hand: turn `require_review` on for one patient → fire a detector
   → confirm the event holds → confirm a push actually arrives (this is the bit
   that has never worked) → approve and confirm dispatch → repeat with reject
   and confirm silence.
4. **Auto-release:** hold an event, do nothing for 60s, confirm it dispatches
   anyway. This is the safety property; test it explicitly.

---

## Gotchas

- **`require_review` defaults FALSE.** Nothing changes for existing patients
  until it's switched on. Behaviour with the toggle off must stay byte-identical
  to the old direct fan-out.
- **The delta must go to prod with the RLS delta**, never alone —
  `review_devices` without RLS is publicly writable via the anon key.
- The author verified the loop locally *with the push path stubbed*. "Verified"
  in his handover does not mean the notification works.
- Integration tests cannot run in this environment (pre-existing). The unit
  suite is your gate.
