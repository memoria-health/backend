# Master test checklist — 2026-08-12

Goal: everything on `feat/data-collection-timing-truth` (~100 commits) plus
`nemara-web-app fix/dashboard-local-dev` verified well enough to merge and
compose the prod deploy batch. Sources: the 2026-08-05 B2 recorder handoff,
the 2026-08-11 GPU fleet handoff, and the Aug 7-11 detector/insights work.

Legend: ✅ verified · 🟡 partially / passively verified · ⬜ not run · 🔴 blocked

## 0. Preflight — gates every live test

### 0a. Rig restore after the Aug-11 wipe (found 08-12 morning)

The wipe took patients, cameras, detectors, events, analyses — all zero.
Survivors: 3 caregivers (incl. jane@patel.co) and the GATEWAY BOX, which is
alive and polling the roster. Also found and fixed this morning:

- An orphaned `identity_supervisor.sh` (script deleted by the fleet cleanup,
  process alive 19h) was crash-looping the identity service against the dead
  patient room. Killed. That supervisor is retired — do not resurrect.
- Perception NEVER ran on the Mac: `localhost:8000` was an ssh tunnel to the
  old RunPod pod, and both tunnel + pod scripts are deleted. The ONLY current
  way to run identity+perception is the GPU node image (= F1). Until a node
  serves this home, the fail-quiet gate suppresses ALL non-safety patient
  detectors; safety events, per-tick activity labels, aggregation, the whole
  B-series recorder, and insights all work identity-less.
- `reid_sam_service_url` is settings-only dead config; the backend consumes
  ONLY the identity service. No separate SAM wiring needed.
- Model pinned per Aryaman: `.env` `NEMARA_GEMINI_MODEL=gemini-3.5-flash-lite`
  (was 3.1-flash-lite); fallback stays 3.1-flash-lite. Container recreated.

Restore sequence:

| step | who | action |
|---|---|---|
| R1 | Aryaman | app login (jane@patel.co) → create patient → enroll face photos |
| R2 | Claude | mint local dev JWT → `POST /v1/patients/<id>/cameras/gateway-provision` (real code path, detector links sync) |
| R3 | box | auto-discovers the new camera on its next roster poll (~30s), publishes `cam_*` into `patient_<new-id>` — NO box SSH |
| R4 | Aryaman | recreate customs in app: Drinking water + Laptop work (+ TV if wanted) — doubles as D7 compile-path proof on the new OpenAI key |
| R5 | Claude | re-apply Sleep preset when-scope 7 PM–12 PM; verify presets seeded |
| R6 | both | identity/perception via the F1 node (decision below) or run morning identity-less |

HARD RULE (Aryaman 08-12): no destructive DB commands. Never `make db-reset`
/ `fresh` / `fresh-seed`; DB writes limited to narrow additive updates.
`make backend-up` verified safe (starts Supabase if stopped + rebuilds the
backend container only).

| id | test | pass criteria | status |
|---|---|---|---|
| P1 | Local DB state after the Aug-11 fleet-test wipe | patients/cameras/detectors rows exist; if not, restore per 0a | 🔴 wiped — restore in progress |
| P2 | Rig alive: backend + camera streaming | ticks flowing in `analyses` < 1 min old; model recorded = gemini-3.5-flash-lite | ⬜ |
| P3 | Both suites green at HEAD | backend unit+integration; web `tsc` + build | ✅ 08-12 — backend 1455 passed / 1 skipped (114s); web build clean |
| P4 | Fleet Stage 0: CI image build from tag (free, no hardware) | workflow green → `ghcr.io/aryamandarda/memoria-gpu:gpu-v1` | ✅ 08-12 run 31563950055 (attempt 7). Six first-execution bugs fixed on the way: pip resolver → uv.lock; Settings env; ffmpeg + 5 DB-backed tests excluded from gate; onnxruntime-gpu 1.20.1 never existed → 1.20.2; gdown missing → pinned 5.x (6.0 broke fuzzy=). Registration gotcha: workflow file had to exist on `main` before tag pushes trigger (0c28f89); runs execute the file at the TAG's commit. |

## 1. Detector stack (Aug 7-8 work) — mostly verified live already

| id | test | pass criteria | status |
|---|---|---|---|
| D1 | Crop coordinate fix | production `person_crop_box` wraps the person; empty-room claims ≈ baseline | ✅ Aug 7 (rendered live, 28%→6%) |
| D2 | Concurrent executor | tick shed ≈ 0%, capture→row < ~15s | ✅ Aug 7 (25.2%→0%, 31.7→12.4s) |
| D3 | Reasoning-first + base-rate prompt | drinking-water FP rate collapsed, evidence coherent per-call | ✅ Aug 7-8 (2%→0.27%/tick) · week-watch 🟡 |
| D4 | Sustained state split — entry / latch / concurrent / exit / re-latch | one event per episode; re-latch after >5 min gap | ✅ Aug 8 (TV + Laptop, all edges) · TV-exit edge 🟡 (same code path as laptop, never directly observed) |
| D5 | Identity fail-quiet gate covers behaviour/care | next behaviour event has `identity_patient_in_clip=true`; unit test pins the category rule | ✅ unit · 🟡 live (absence-of-FP only) |
| D6 | **Activity verification — the 3-layer plan** | see §1a | ⬜ **today's centrepiece** |
| D7 | VLM chain after OpenRouter removal | a compile (OpenAI-first) succeeds with the new key; routing tests green | 🟡 (unit green; new key probed 200; no end-to-end compile since key swap) |
| D8 | Event gate round-trip | held → paged → verdict/auto-release → dispatch | ✅ Aug 5-7 live |

### 1a. D6 — Activity verification, all levels

1. **Per-tick labels** (guided soak, ~15 min): sitting_still → arm raises
   (`sitting_moving`, the case that failed pre-split) → gesturing trap →
   standing → active/walking → lying. ≥80% correct per phase, no
   `sitting` at conf 1.0 during arm raises.
2. **Timeline layer**: tick shed stayed ≈0 during the soak (persistence
   assumption holds); label spans match the announced phase boundaries.
3. **Aggregation invariants** (unit tests, no hardware): state minutes sum
   to labeled minutes; breaks alternate sedentary→upright; longest bout ≤
   sedentary total; legacy `sitting` still counted; `sitting_moving` in
   neither sedentary nor upright band.
4. **Diary cross-check** (~30-45 min, passive): user notes timestamps while
   living normally; diff against derived timeline; headline metrics
   (sedentary/upright/breaks) within tolerance.

## 2. B2 recorder / masking / sidecar (Aug-5 handoff — none ever run)

Order matters: **B1 gates B2-B8.** Enable `data_collection_enabled` for the
test patient early so 15-minute rolls accumulate during other work.

GATE CHANGE (Aryaman, 08-12, commit 4d2b9be): room privacy no longer
excludes collection — consent + motion + presence is the whole gate, and
sensitive verdicts still switch to the masked encoder. The B-series runs
against the privacy-on "Aryaman Bedroom" camera directly. Accepted
residuals, on the record: (a) a sensitive moment the sensitivity screen
MISSES in a privacy room now uploads raw (test 2's strictness is the
evidence the screen+mask deliver); (b) a mid-session room-reclassification
no longer retro-drops the in-flight clip.

REDESIGNED MID-DAY (Aryaman, 08-12): inline masking → finalize-time
redaction (raw local recording; spans = tick priors ∪ chunked re-screen;
person-pixelation via the event censor, span-gated; fail-closed drop).
B1/B2/B7 retired with the machinery; R-series replaced them. All run
against live camera + real B2 the same afternoon:

| id | test | result |
|---|---|---|
| R0 | Gate & session lifecycle | ✅ session opened on presence, closed on 45s stillness, nothing pre-toggle |
| R1 | Span detection | ✅ staged 19:01:00 IST → span [87,153] over a 13:29:34 start = 19:01:01 hit; 2 chunk hits + 2 tick priors (both nets) |
| R2 | Safety claim | ✅ redaction visually confirmed (green first clip, pixelation after the mode fix); **bucket sweep: zero raw objects**; spans metadata exact |
| R3 | Duration truth | ✅ 903.5s / 904.0s clips, timeline_exact=true, lost=0.0 — pacer fix field-proven (was 3.8% median loss) |
| R4 | Audio alignment | ✅ user-verified in sync end-to-end, continuous through redacted spans |
| R5 | Sidecar vs reality | ✅ staged fall → event 30fea6ae promoted in 12s; sidecar span carries promoted:true + that event id; offsets land |
| R6 | Crash recovery | ✅ kill -9 mid-session → boot sweep "removed 2 stranded recorder tempfile(s)" |
| R7 | Redaction failure → drop | ✖ SKIPPED live (Aryaman) — drop path pinned by unit test_redaction_failure_drops_the_clip |
| R8 | Counters | ✅ RESOLVED 08-12 night: no undercount — sessions_started counts manager-level sessions and a ROLL restarts inside the same recorder (3 clips = 1 session + 2 rolls, counter correctly read 1). The suspicious inter-clip gaps were the inline-redaction recording hole, independently found and fixed the same evening. |

Post-series fixes landed same day: pixelation replaces green in research
censor (commit after user review of clip 1); redaction moved OFF the
recorder-lock path into the upload task (inline version opened a 5-min
recording hole after every roll — observed, fixed, suite green). The
recording hole fix is committed but NOT live-re-verified (collection was
shut off) — first post-merge occupied roll should confirm contiguity.

## 3. GPU node fleet (Aug-11 handoff — never executed on hardware)

| id | test | pass criteria | status |
|---|---|---|---|
| F0 | CI build from tag (= P4) | image builds; Dockerfile assertions pass | ✅ 08-12 (see P4) |
| F1 | One node, one home (~2h, ~$0.80/hr) | 6 ordered assertions ending in **a real PATIENT verdict via tailnet** | ✅ 08-12 pod k562y9osxpa0yd (RTX 4090, EU-RO-1, $0.74/hr): checkpoints ✓, tailnet ✓, roster claim 60s post-boot ✓, backend→node 200 from inside Docker ✓, /ready identity:ok ✓, **patient_bound via=face score=0.73** ✓. CUDA-provider check 🟡 (read `perception.ready face_provider` in RunPod console). 10 first-execution bugs fixed to get here — see git log a1c5ab5..0c28f89. Local-rig quirks: userspace tailscale needs proxy/socat-bridge (all conditional on 100.64/10 targets); node registers tailnet IP not MagicDNS name; India DCs can't host pods+volumes together. |
| F2 | Measure K (RSS per identity process, not GPU) | homes/node number replaces the arithmetic guess; concurrency=2 latency A/B | ✅ 08-12 night (idle floor): 3 synthetic homes → node claimed 4 total, ports 8012-8015 sequential. Container RAM UNCHANGED at 12.2GB/61GB (marginal ≲0.6GB/idle home — RAM-bound fear unfounded); VRAM +~0.65GB/home (2%→10% of 24GB) → ~30 idle homes/card. ACTIVE-home cost unmeasured (rooms empty, AI paused) — rider on a future occupied soak before raising capacity past 4. Synthetic patients F2-Syn-1..3 left in local DB (feed F3; delete after). Latency A/B not run. |
| F3 | Two nodes: scale + failover | second node claims homes; kill A → B claims in ~90s; FleetWatch pages; **measure cold-start seat re-acquire** | ✅ 08-12 night: node B (capacity 8) live+claiming **61s** from release; A terminated → **B serving all 6 homes 82s later**, ports 8012-8017 sequential, no collisions. Duplication policy foundation: survivors need spare capacity ≥ biggest node's load (B@8 absorbed 2+4); failover is autonomous, pager-observed. Seat re-acquire on live video unmeasured (AI paused — rider on next occupied soak). B is now THE rig node (POD_ID + dashboard repointed; /ready identity:ok). |
| F4 | Prod: identity_nodes delta before backend deploy | delta idempotent/additive | ⬜ deploy-day |
| FS | Security invariants hold while testing | perception loopback-only; state token set; ACL port RANGE; secrets out of argv | ⬜ each stage |

## 4. Insights / BYPASSRLS flip (Aug 11)

| id | test | pass criteria | status |
|---|---|---|---|
| I1 | Backend endpoints: scope + shapes | two-caregiver isolation, foreign id 404s, term sanitiser | ✅ integration tests |
| I2 | Web live smoke: dashboard + widgets + chat + home view against local backend (`pnpm build && pnpm start`) | pages render real data; no pg anywhere | ⬜ |
| I3 | Prod ordering staged | backend deploy → web deploy → NOBYPASSRLS delta | ✅ staged (`apply_insights_readonly_revoke_delta.sql`) — deploy-day |

## 5. Merge & deploy prep

| id | item | status |
|---|---|---|
| M1 | Push both branches to origin (backend + web) | ⬜ **overdue** |
| M2 | Untangle the two cross-labeled commits (`5a7865c`/`acb33b0`) — needs a quiet branch | ⬜ optional, PR-note otherwise |
| M3 | Compose the deploy batch: identity_nodes delta + data_collection policy delta + RLS-batch leftovers (runbook memory) + insights revoke (AFTER web) | ⬜ |
| M4 | Revoke the Tailscale auth key pasted into the fleet session's transcript; reissue | ✖ WAIVED by Aryaman 08-12 — leaked into local transcript only, judged non-issue |
| M5 | Full suites, both repos, final run before PR | ⬜ |

iOS app polish todos (Aryaman, 08-12, found while re-onboarding — for a
future iOS session, not today):
- Basic profile "add a patient" flow is slightly glitchy
- First-time detector add is wordy, and detector colors don't follow the
  safety / routine / behaviour / care category palette
- The "Contact Memoria" floating button should be flexible/movable
- Edit the welcome tour content, and inspect the welcome-tour logic
- "Contribute recordings to research" belongs under a DATA sub-header in
  patient settings

PARKED (Aryaman, 08-12, apply AFTER the R-series): drop the tick-time
sensitivity screen entirely — the finalize re-screen is the sole net.
Delete `_screen_for_masking` + the `sensitive` tick plumbing +
`data_collection_sensitivity_every_n_ticks`; `note_sensitive`/priors
machinery goes with it. Saves ~30% tick cost on recording cameras.

SCALE NOTE (Aryaman, 08-12): shard the recording/backend workload once
we pass ~5 homes. Ceiling is the single backend instance's CPU + the
data_collection_max_concurrent=8 session cap: per actively-recording
camera it carries an ffmpeg encoder child, ~1.5Mbps upload, and per
finalized clip a ~clip-length redaction pass. Long-term home for
media-heavy work is the GPU node fleet, not the API backend.

Post-merge debt from the F0 build session (08-12): (a) move the five
DB-backed test/unit files to test/integration or mock them — they are
excluded from the CI gate only; (b) pin ultralytics/boxmot in
gpu/Dockerfile — unpinned latest-at-build-time is how the gdown-6.0
breakage class recurs.

## Suggested order today

1. P1-P3 preflight (now) → 2. P4/F0 CI tag (parallel, free) → 3. enable
data collection so rolls accumulate → 4. B1 gate → 5. D6 activity soak
(user present) → 6. B2, B4-B8 → 7. I2 web smoke → 8. B3 (rolls ready by
now) → 9. F1-F3 if daylight remains → 10. M1-M5.

## 6. Built-in FP soak (designed 2026-08-17 — env stance settled, unblocked)

Aryaman's framing: "not as worried about TP as FP — we might need a massive
run." Env taxonomy is now pinned (action detectors for glass/water, state
for fire, duration for stove — commit bcc1651), so every built-in has final
semantics and the soak measures the product as it will ship.

**Design: one enable-everything passive soak, triaged daily.**

| id | step | detail | status |
|---|---|---|---|
| S0 | Prereqs | AI unpaused on all cams; identity node serving the home (green box on soak dash); audio flowing (calling_for_help is audio-dependent) | ⬜ |
| S1 | Enable ALL adult built-ins via the app library | fall, agitation, wandering, medication, calling_for_help, choking, meal_eaten, stove_left_on, emergency, possible_intruder, glass, water, fire — plus existing customs stay on. App-side add doubles as a library/UI pass | ⬜ |
| S2 | Passive soak, 48h of normal living | no staging; every fired event is a presumed FP unless Aryaman logged a real occurrence. Cost ≈ occupied-rate (~$0.50/occupied-hr — the extra detectors are schema fields on calls already happening, not extra calls) | ⬜ |
| S3 | Daily triage | per-detector FP tally + root-cause bucket (reuse the Jul-22 six-category FP taxonomy); verdict per detector: SHIP / TUNE (prompt or threshold) / PULL from library | ⬜ |
| S4 | Staged TP spot-checks (after S2, ~30 min) | pour water on the floor on camera (action semantics: must fire on the pour, then NOT re-fire on the standing puddle within cooldown — both halves are the test); enacted choking + fall (established protocol); stove on ≥10 min unattended; possible_intruder with a mask | ⬜ |
| S5 | Fire self-wake check | closed empty room, small real flame (candle) in view: does flicker wake the camera and fire the detector? This is the one place "detect as it happens" rests on an unverified assumption. Also try a fire VIDEO on a TV screen — that's the FP half | ⬜ |
| S6 | Sleep verification rider | overnight during S2: night summary correctness + the home-screen data contract (widget reads without glitches) — closes the standing sleep item. KNOWN LIMITATION to measure, not fix (Aryaman 2026-08-17): sleep is scene-bucket, NOT track-bound — a partner in the same bed who wakes first will likely be read as the patient's wake time, and partner movements as the patient's disturbances. Tonight is a natural test of exactly this; log what the summary claims vs what actually happened. Fix path exists (bind sleep ticks to the seated patient track, the mechanism activity already uses) — future work. | ⬜ |
| S7 | Data-collection annotation audit (Aryaman 2026-08-17) | collection ON for the whole soak. Every staged S4 event is annotation ground truth: the recorder sidecar must carry a span at the right offset with promoted:true + the event id (R5, generalized to every detector class); for env actions the span must sit on the POUR, not the puddle. Also closes the recording-hole re-verify on the first occupied roll (clip contiguity, no 5-min gaps). One timestamped diary diffs against all three layers: activity timeline, journal events, sidecar spans. | ⬜ |

Go-state (configured 2026-08-17 23:30, AI still PAUSED — the Go is the unpause):
all 16 detectors on (13 library adds via the real from-library API — first live
proof of the env cooldown seeding: glass/water 1800s, fire 600s), sleep
when-scope 7 PM–12 PM, data_collection_enabled=true (decided_at stamped via
PATCH), node serving, dash roster-following. Overview must be checked for
"0 of 24" BEFORE unpausing — the clean read disappears at Go.

Acceptance bars (proposal — confirm with Aryaman at S3): emergency-severity
detectors (fall, choking, emergency, fire, calling_for_help) ≈ 0 FP/day
each — a false page is the product killing itself; sub-emergency ≤ 1 FP/day
each; anything worse gets TUNE or PULL, not a shrug.

## 2026-08-13 — GPU load test + optimization sprint (F-series continuation)

DONE. Full record: docs/architecture/identity-gpu-architecture.md. Headline: at 3
concurrently-occupied cameras, lag p50 10.9s -> 0.33s (SLA <7-10s now
holds with ~30x headroom); root causes were CPU-wheel onnxruntime
shadowing (faces on CPU since gpu-v1) + the JPEG loopback hop. Fleet is
on node G (bfb433h74ng3ou, image gpu-v2.2); nodes C/D/E/F terminated.
Six fleet post-mortems from the day are guarded in-image (see doc §post-
mortems). F1's open assertion 2 (face_provider CUDA) is now CLOSED —
verified live on node G.

Still open from this sprint: L4/5 capacity run (pin the occupied-camera
ceiling); soak dashboard node lookup must follow the roster (was aimed
at dead node C); backend liveness alarm; identity_nodes dead-row reaper
(M3 batch); FleetWatch lag>tick alarm.

SOAK CONFIG BOUNDARY (2026-08-18 ~14:15 IST): media_res high->medium,
thinking high->medium (Aryaman, cost). FP/FN rates before vs after this
point are not directly comparable.
CONFIG: 15:25 IST media_res back to HIGH (fall FPs: couch read as floor at medium). thinking stays medium, scene stays consolidated.
ROOT CAUSE CLOSED (15:5x IST): PD alert grouping silently folded fall alerts into open incidents (proven by ordered probes). Fix: grouping OFF in PD service settings (Aryaman) + record-only incidents self-resolve after 15 min. Re-verify: staged fall with grouping off must page.
