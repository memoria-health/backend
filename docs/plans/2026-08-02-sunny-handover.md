# Sunny's handover — 2026-08-02

For Aryaman (and his agent). Everything below is on SEPARATE BRANCHES, all
based on `origin/main` or a snapshot of your branch — nothing merged, nothing
deployed to prod, every backend change is ADDITIVE and DEFAULT-OFF. Merge
order/timing is entirely yours.

| Repo | Branch | What it is |
|---|---|---|
| nemara-backend | `feat/review-gate-and-home-endpoints` | Review gate + 3 Home-AI endpoints (this doc lives here) |
| nemara-caregiver-app | `feat/care-polish` | iOS Home redesign + UI declutter (based on YOUR branch @ e6e3970) |
| nemara-caregiver-app | `fix/login-resilience` | Your worklist item #1, done (off main, one commit) |
| NemaraCam | `feat/review-mode` | Reviewer mode (human-in-the-loop alert gate client) |
| nemara-caregiver-web | `feat/insight-merge` | Unified web app (caregiver-web + insight dashboard merged) |
| memoria-insight (separate repo) | `main` | Standalone insight dashboard, already deployed |

## nemara-backend — `feat/review-gate-and-home-endpoints`

New files (all additive):
- `src/api/briefing.py` — POST `/v1/home-briefing`: LLM proxy (gpt-5.4-mini via
  `settings.openai_api_key`) writing the Home check-in sentence from
  client-computed facts. No DB.
- `src/api/presence.py` — GET `/v1/patients/{id}/last-presence`: last
  people_present tick + last analysis tick (observed-presence facts).
- `src/api/home_report.py` — POST `/v1/home-report`: gathers per-patient
  evidence (cameras, 48h events, presence, 7d event pattern, deterministic
  activity labels) + LIVE frames from the detection worker's frame buffers,
  LLM returns strict JSON {severity, highlight, description}. Guardrail
  prompt: names-only bold, no health advice, no repetition, caregiver-local
  times, shortest-true-sentence.
- `src/services/review_gate.py` — human-in-the-loop alert gate:
  `route_events()` replaces the direct push+email fan-out; when
  `patients.require_review` (DEFAULT FALSE) is on, events are HELD
  (`events.review_gate='pending'`), reviewer approves (dispatch) / rejects
  (drop), 60s auto-release safety valve (human review can only REMOVE false
  alarms, never block a real one). Fires APNs to `review_devices` on hold.
- `src/api/review.py` — GET `/v1/review/queue` (held events + signed clip
  URL), POST `/v1/review/{id}` {approve}, POST `/v1/review/toggle`,
  POST `/v1/review/device` (APNs registration). Admin allow-list
  `REVIEWER_EMAILS = {"admin@gmail.com"}` sees ALL patients (move to
  settings/DB before real use).
- `scripts/apply_review_gate_delta.sql` — RUN BEFORE DEPLOY (additive):
  patients.require_review, events.review_gate, events.review_clip_key,
  review_devices table.

Modified files (⚠ = will conflict with your branch, both are small):
- ⚠ `src/services/detection/camera_ingest.py` — the `promotion.fired`
  push+email `create_task` block is replaced by ONE
  `review_gate.route_events(...)` task (behavior identical when toggle off).
- `src/services/clip_archival.py` — review fast-path: for HELD events the
  pre-event MP4 (the exact clip the VLM judged) is uploaded IMMEDIATELY
  (`sub_key="review-pre"`, stamped on `events.review_clip_key`) so the
  reviewer has video in ~5s instead of waiting out the post-roll. The queue
  serves review_clip_key by design — reviewer sees exactly what the model saw.
- `src/schemas/patient.py` + `src/db/patient.py` — `require_review`
  (optional in schema; ORM column placed LAST — dataclass ordering).
- `src/main.py` — 4 `include_router` lines.

Verified: full loop tested locally (hold → queue → approve/reject →
dispatch/silence → auto-release), `import src.main` clean on this branch.

## nemara-caregiver-app — `feat/care-polish`
Based on YOUR `feat/gateway-camera-ownership` @ e6e3970 (snapshot; rebase
once after your wave lands — cherry-picked login fix will dedupe).
- `Features/Home/HomeStatusView.swift` (new): per-person verdict fold
  (urgent > blind > attention > quiet; zero cameras = blind, absence of
  data is never good news), AI report rendering (bolded names, markdown),
  streak chip, template fallback when the endpoint is absent.
- Home briefing/report calls: `homeBriefing`, `homeReport`, `lastPresence`
  in NemaraAPIClient (all optional-degrade against older backends).
- Declutter: InfoDot house pattern (`DesignSystem/Components/InfoDot.swift`),
  captions removed/converted across settings/sensors/events; SAFETY/ROUTINE
  pills removed; events show "Xd ago + h:mm a".
- Wandering: adding it prompts which cameras (sets whereRooms); exit-door
  section renders ONLY on those cameras (case-insensitive vs whereScope —
  backend uppercases). Duplicate camera room names blocked at creation.
- `fix(models)`: tolerant Detector decode (alert_trigger/is_sustained
  optional) — NOTE: your branch's strict decode breaks sign-in against prod
  (no migrations) — this fix is upstream-worthy for your wave.

## NemaraCam — `feat/review-mode`
- `Review/ReviewView.swift` (new): reviewer sign-in (Supabase password →
  JWT; admin@gmail.com), 1s queue poll, auto-playing MUTED=OFF 2x clip
  (exactly the VLM's pre-event MP4), YES/NO, per-patient require-review
  toggles, APNs registration (needs push entitlement + real device).
- Entry points: "Review" button on ContentView + "I'm a reviewer" on the
  pairing screen. Camera pipeline untouched.
- NOTE: a git stash on this repo holds your in-flight AppSettings.swift
  edit from `feat/audio-capture-toggle` (`git stash list`).

## nemara-caregiver-web — `feat/insight-merge`
The unified app (live preview: memoria-care-unified.vercel.app, separate
Vercel project "memoria-app" — old prod project/domains untouched). Person
chip (single patient = static label, multi = picker + Everyone), tabs
Home/Live/Ask/Explore/Settings, insight routes tenant-scoped via backend
/v1/patients (401 anon, 403 foreign), Cache-Control PRIVATE on scoped data.
Overlaps your `feat/web-customs-visibility` in 6 files — all additive on my
side; merge after your wave lands on main.

## Prod ops that already happened (no action needed, context only)
- 2026-07-29 mobile sign-in hang: prod Postgres hit 60-conn ceiling →
  Realtime starved. Fixed: idle conns purged, insight moved to :6543
  transaction pooling. Supavisor default_pool_size MUST STAY 45 (10 broke
  the backend: EMAXCONNSESSION).
- 2026-07-29 OOM on Railway: likely 1080p footprint; auto-recovered;
  your 1280w cap is the fix.
- Dev LiveKit webhook (your fix) → local backends get online/RoomWorkers.
- Demo learning: identity gate (`identity_not_patient`) correctly blocks
  non-enrolled actors — demos must use the enrolled person or a
  zero-embedding patient.
