# Pilot master worklist — consolidated across all sessions (2026-07-28)

> Canonical order-of-operations for getting every repo to main and the system
> to pilot families. Consolidates three workstreams: identity productionization
> + resilience (this session), room-privacy + clip censoring (Jul 27–28 session,
> canonical plan: `2026-07-27-room-privacy-and-clip-censoring.md`), and the
> prod login-hang incident (Jul 28 session). Update THIS file as items land;
> per-session trackers do not belong in the repo.

## Ground truth (verified 2026-07-28, do not re-derive)

- All repos share the feature wave: backend + caregiver on
  `feat/gateway-camera-ownership`, web on `feat/web-customs-visibility`,
  camera-app on `fix/camera-local-signing`, gateway already merged to main.
- Backend tip `7f62727` (identity resilience fixes) on top of `7963d9b`
  (privacy/censoring). 906 unit tests green at `7963d9b`.
- Gateway box runs origin/main `0c1878b` (identity-service allow-list) —
  verified on-box. The Jul-28 handoff's "box runs 17a9db7" was stale.
  `0c1878b` is a PRODUCTION fix, not a backdoor: identity-service is AI
  processing, same SFU trust class as detector workers; caregivers stay
  blocked in private rooms. Caveat: LiveKit API secret can mint any identity
  → key rotation is the real lock on the private-room door.
- Detection is LIVE locally (`NEMARA_REID_SOAK_ONLY=false`).
- Bedroom is room-classified private (0.99); identity tracks it via the
  allow-list. Counsel language must state: "AI processes private rooms;
  humans never see raw video."
- Prod incident root cause CLOSED: partner's prototype dashboard polled prod
  DB directly and exhausted pooler slots; fixed on his side (his repo, his
  responsibility). Remaining backend-side items are in Phase 3.
- SAM3 phantom churn root cause: SAM3 grounding runs on NATIVE input frames;
  gateway's Jul-27 1080p deploy moved SAM3 from a proven ≤1280-wide input to
  1920, where it intermittently splits a recumbent/seated person into two
  instances (YOLO trigger dets=1 the whole time). Births/deaths 8-13/hr in
  occupied rooms + a forced-reground loop (~2.4s SAM3 every ~15s) burning GPU.

## Phase 0 — pre-soak fixes (blocks soak data quality)

- [ ] SAM3 grounding-input resolution pin (pod): downscale frames fed to SAM3
      grounding to 1280-wide (the proven operating point); existing
      mask→track-res resize path unchanged. NOT the consensus-merge idea
      (withdrawn — wrong layer; YOLO is a trigger, not an arbiter).
      Fold in: `_iou` shape-mismatch rescale one-liner (latent landmine).
      Deploy = rsync + pod restart. Success = phantom births ~0 in occupied
      still rooms; forced-reground loop gone; discrimination test still
      tracks 2 real people as 2.
- [ ] Prod hygiene: delete Supabase auth user
      `claude-diag-probe-20260728@example.com`; confirm no local env/profile
      references the prod pooler (policy now has an incident behind it).
- [x] Backend branch closed out (`7f62727` pushed).

## Phase 1 — full-system soak (24–48h; THE ship-gate before any merge)

- MODEL PLAN (Aryaman 2026-07-28, supersedes "day 1 all-3.6"): the soak
  runs 3.5-FLASH-LITE EVERYWHERE (local NEMARA_GEMINI_MODEL override at
  soak open); escalate to 3.6-flash only if accuracy problems appear.
  Per-call model+token attribution makes the lite→flash bill translation
  pure arithmetic. Per-DETECTOR tier segregation is REJECTED (Aryaman):
  model is per-call, so intra-bucket tiering would mint NEW calls
  (2→4/tick) — model choice stays whole-deployment, lite-first.
- SLEEP joins the ON-set (10 detectors incl. Sleep); tonight = sleep
  validation night 1 vs Aryaman's self-report/wearable. IR-motion
  evidence already in hand (overnight 07-27/28 bedroom latches at
  00:15/02:32/03:50 — transitions detectable; fine stirring under
  covers may under-trip → failure direction = OVERCOUNT undisturbed
  sleep, validated tonight).
- [ ] Detector matrix — the soak ON-set (9), chosen to cover every active
      bucket, every subject scope, audio, schedule, door, preset+custom+
      fresh-create, and a deliberate negative case:

      | Detector | Bucket (recipe) | Scope | On-demand test |
      |---|---|---|---|
      | Falls | scene (4fps clean) | patient | lie on the floor |
      | Calling for help | scene + audio | patient | shout "help" |
      | Watching TV | scene — RECLASSIFIED 2026-07-28 (was person/zoom; TV-watching needs scene context — the crop excludes the evidence). Duration semantics via its N-in-M entry rule, set to ~5 MIN for the soak (Aryaman: don't waste time) | patient | watch TV ~5 min; a glance must NOT fire |
      | Medication adherence | person (4fps crop) | patient | pill-bottle mime |
      | Setup test | scene | anyone | uncle/aunt walks through (doubles as discrimination test) |
      | Wandering | scene + door_config | patient | exit through front door |
      | Missed evening medication | scheduled/absence (NO calls — evaluates from coverage at deadline) | patient | set window tonight, do NOTHING → must fire absence |
      | (new) "Drinking water" | person | patient | drink from a BOTTLE; exercises create→compile→NL-refine e2e |

      Convulsion REMOVED OUTRIGHT 2026-07-28 (not just soak-excluded):
      catalog entry + dims deleted, preset rows deleted via
      apply_convulsion_removal_delta.sql (run on prod at deploy), and
      the compiler now stamps degradation_warning="rapid_frequency" on
      any CUSTOM ask routed to rapid/fine_motion (high FP/FN at VLM
      sampling rates — benchmark-measured temporal ceiling). Client
      warning copy for rapid_frequency: TODO web + iOS. Result: exactly
      TWO active built-in buckets — scene and person. Rapid lane
      infrastructure retained for warned customs; FFT successor
      post-pilot. Left OFF: Meals (not on-demand),
      Agitation/Choking (optional mimes), General Emergency (redundant
      w/ Falls), Stove (no kitchen camera).
      COMPILER LESSON (from the Watching-TV miss): context-dependent
      activities (watching TV, cooking, eating) must route to scene even
      when the subject is a person; person crops are for body-evidence
      tasks only. Fold into compiler prompt/goldens at branch wrap.
- [ ] Explicit tests beyond daily living (each with expected outcome):
      1. Falls positive → event + clip in caregiver app ≤90s.
      2. Calling-for-help positive → audio ridden, transcript non-empty.
      3. Watching TV duration: 15+ min viewing fires; a 1-min glance
         does NOT (entry-rule negative).
      4. Missed-medication NEGATIVE → absence fires at deadline with zero
         VLM calls attributed to it.
      5. Discrimination: visitor in frame → tracked, NOT seated, Setup-test
         (anyone) fires, patient-scoped detectors do NOT.
      6. Patient-elsewhere: Aryaman seated in bedroom → LR runs anyone-only
         (skip reason identity_not_patient on patient-scoped LR calls).
      7. Towel test (censor chain) → clip_censored + dignity overlay.
      8. Wandering door exit → fires with door_config semantics.
      9. Cost audit at hour 6 and hour 24: dashboard $ vs recorded
         tokens×list-price by model; flag >10% drift.
- [ ] Privacy chain live: staged sensitive event → screen → censor → censored
      clip w/ audio in caregiver app; quiet fall → clean clip. Outputs seed
      the sensitivity eval corpus.
- [ ] Data-collection exclusion field test (never exercised): research consent
      on, motion both cameras, B2 receives ONLY non-private clips.
- [ ] Identity discrimination test (needs 2nd person): tracked but NOT seated;
      also validates the resolution pin doesn't merge two real people.
- [ ] Wifi drill mid-soak: router off 3 min, hands off, all green ≤60s.
      Known weak link: ssh tunnel keeper (unsupervised; colocation deletes it
      in prod). Full-router drill = standing pre-deploy ritual per home.
- [ ] Soak-end counters: identity restarts (~0 expected), gateway watchdog
      fires, backend swallowed-error volume, phantom merge count.

## Phase 2 — review → merge wave (after soak passes; merge order = deploy order)

1. [ ] Backend full-diff adjudicated review (3 feature streams interact —
       review them together), then telemetry-gate discussion (below), then PR.
       DB deltas collected in apply order: camera-privacy, dashscope-provider,
       detector-links normalize. Deltas run BEFORE backend deploy.
2. [ ] Telemetry gate (pre-PR, Aryaman-mandated). Agenda from real incidents:
       silent failures (stale client, dead tunnel, zombie gateway, wedged
       dashboard, login hang with no log line); process-alive ≠ service-healthy
       (identity truth = /state within 2s); /ready measures the wrong thing for
       connection exhaustion. Minimum set for a 2-person team (proposal):
       Logfire/Sentry capture on swallowed exceptions + one synthetic login
       canary incl. real Realtime join w/ timeout + one home-offline operator
       alert + identity restart counter. Decide, then build ONLY that.
3. [ ] camera-app PR (publisher; revert temp signing/local-IP fallbacks first).
4. [ ] caregiver PR (revert temp fallbacks; fold iOS 1080p change into same
       TestFlight build; `xcodegen generate` after merges).
5. [ ] web PR (validate with `pnpm build && pnpm start`, dev-mode hydration
       known-broken).
6. [ ] Deploy checklist execution (privacy plan doc's list is load-bearing):
       explicit NEMARA_ENV + INTERNAL_SERVICE_KEY, OMP/MKL single-thread env,
       bake yolo11n-seg.pt into image, Railway env ADD
       (IDENTITY_STATE_URL/TOKEN) + SCRUB deleted keys (extra=forbid boot
       crash), legacy `privacy_mode_enabled` family check, deploy order
       DB → backend → publishers → clients.

## Phase 3 — pilot-blocking builds (post-merge)

1. [ ] iOS login resilience: bootstrap() must not block sign-in on Realtime
       subscribe (timeout + fire-and-forget behind existing 1s REST poll);
       same for cold-launch restore. The incident's actual product bug.
2. [ ] New cameras PRIVATE-BY-DEFAULT until classified (closes live-observed
       15-min bedroom-stream gap; pairing copy already written).
3. [ ] Backend pool right-size (10+10 → ~5+5 single replica) + role-level
       idle timeouts (config-only).
4. [ ] Key rotation (LiveKit/APNs/Gemini/OpenAI/OpenRouter/Resend) — BLOCKS
       all prod env work; .env.bak still in git history on main.
5. [ ] /state bearer token BOTH sides + unify env name
       (backend `NEMARA_IDENTITY_STATE_TOKEN` vs service
       `NEMARA_IDENTITY_TOKEN` — currently mismatched, silent no-auth trap).
6. [ ] Research/B2 recording censoring (recordings currently RAW) — gates
       consent PDF → gates attestation → gates research-consent pilots.
7. [ ] Consent attestation on DataCollection sheet + BIPA biometrics clause.
8. [ ] Sensitivity eval corpus + measured FALSE-NEGATIVE rate (ship number).
9. [ ] Censor soak on Railway substrate (local numbers are Docker-VM numbers).
10.[ ] Identity colocation on GPU host (kills tunnel failure class; verify
       RunPod HTTP port exposure as part of the task) + reserved GPU + autopay
       + deep-health smoke; sr_server (24GB squatter) kill w/ Sunny.
11.[ ] One-way "Make private" per camera (PATCH accepts true only).
12.[ ] FACILITIES AUDIO KILL (Aryaman 2026-07-28): patient-level
       audio_capture_enabled=false for facility deployments — enforced at
       the SOURCE (gateway + iOS publishers never publish the audio
       track, roster-driven like privacy_enabled), needs_audio detectors
       auto-degrade via existing sched_degraded_audio machinery, clips/
       transcripts/speech fields silent for free. Counsel: jurisdictions
       + facility rules + two-party-consent states (add to Zinia/US list).
13.[ ] Empty-room ENVIRONMENT sampling decision: env detectors (stove)
       currently never run in empty rooms (presence gate) — "stove left
       on after leaving the house" is unmonitored. Option: low-cadence
       snapshot class (~1 frame/10min) only where an env detector is
       configured. Decide + price before pilots.
14.[ ] Creation-time cost note for STILL-eligible customs (still-person /
       environment classes poll during stillness — the expensive class),
       mirroring the rapid_frequency warning pattern.

## Phase 4 — activity + sleep — V1 BUILT 2026-07-28 (commit 9ea6a05)

SHIPPED (Aryaman's simplified price-aware design superseding the epoch/HMM
sketch below): "Sleep" catalog detector (scene bucket, patient, severity 15,
motion-gated ON PURPOSE — burst ticks label moments, tick silence = asleep)
+ pure state machine `sleep_summary.py` (onset = in-bed + 15min quiet;
disturbances; out-of-bed = leave+return<=60min; wake = leave + 30min
confirm; incomplete nights FLAG, never fabricate) + `activity_summary.py`
(zero-VLM: active fraction, transitions, coverage) + `sleep_nights` /
`activity_days` tables (delta applied locally, RUNS ON PROD AT DEPLOY) +
`tools/run_daily_summaries.py` runner. Marginal cost ~$0.10-0.15/night
worst case on lite; ~$0 when other bedroom detectors already tick.
REMAINING: enable Sleep in the soak ON-set (bedroom); validate vs Aryaman's
wearable/self-report across soak nights (±15min bedtime/wake, ~100%
out-of-bed recall = ship gate); pin sleep to lite tier when difficulty
routing lands; family-report rendering post-pilot.

Original rigor notes (kept for the validation phase):

- Sleep — behavioral sleep measurement (honest boundary: in-bed quiescence,
  not physiological staging; no HR/HRV sensor exists):
  * 30s EPOCHS classified over fused signals: seat presence (identity),
    motion-energy series (frame-diff, already computed for the latch),
    track micro-motion (rustle cycles — clean post phantom-fix), scene
    luminance (lights-out is a strong bedtime prior; IR-mode step handled).
  * State model with SMOOTHING (HMM-style transition costs, hysteresis) —
    states: out-of-room / in-bed-active / quiescent / disturbed. No single
    thresholds: per-home percentile-calibrated bands learned from the
    patient's own nights (Whoop's baseline principle).
  * Nightly record: bedtime, sleep onset, disturbance intervals (+intensity),
    out-of-bed events (night wandering — the clinical headline), wake time,
    CONFIDENCE + data-quality flags ("camera offline 02:10-02:50 — night
    partially observed"). Coverage gaps degrade confidence, never fabricate.
  * Edge cases in scope: partner in bed (disturbances attributed only via
    track association / solo), bedroom TV glow (luminance flicker filter),
    pets (small-track filter), bathroom trips (seat-exit + return = one
    out-of-bed event with duration).
  * VALIDATION, not vibes: run vs Aryaman's own wearable + self-report
    nightly during soak/dev weeks; ship gate = bedtime/wake within ±15 min
    and out-of-bed recall ~100% on the validation nights.
- Activity — same rigor: epoch classification (stationary / moving / active)
  from track displacement distributions (angle-robust), room transitions,
  occupied-time normalization; daily score vs the patient's own 14-day
  baseline (trend, not absolute); confidence + coverage flags; posture
  split deferred (ceiling-angle bbox geometry unreliable — revisit with
  sampled VLM labeling in non-private rooms if pilots want it).
- Pilot mode: RECORD-ONLY (no notifications) → week-2 family report artifact.

## Event-generation taxonomy (Aryaman 2026-07-28: "evaluate now — so many
## flavours"). Current truth + the agreed direction; sleep-event build
## pends Aryaman's sign-off on the OPEN DECISIONS below.

| Detector flavour | Trigger | Event today | Clip | Notifies |
|---|---|---|---|---|
| leading (Falls, Wandering, Help) | tick crosses bar | standard event | yes (censor-screened) | yes |
| duration (Stove) | sustained > duration_seconds | standard event | yes | yes |
| trailing (activity-end asks) | falling edge via scheduled runner | standard event | yes | yes |
| absence/schedule (Missed medication) | deadline, from coverage | standard event | NO (nothing to show) | yes |
| fire_once (Setup test) | first fire, then self-disables | standard event | yes | yes |
| record_only (Sleep, Activity) | never promotes | NO event — summary tables only | no | no |

AGREED DIRECTION (this session): record_only detectors additionally emit
a DAILY SUMMARY EVENT — a distinct event species: generated by the
summarizer RUNNER (not tick promotion), no clip, no severity page, body =
the summary payload (sleep_nights row / activity_days row), rendered as
its own card in the events feed ("Last night: 7h10m, 1 disturbance, one
10-min wake-up"). Keeps the recording/alerting separation (a runner
emits once daily; promotion never touches record_only) while making the
insight VISIBLE where caregivers already look.
OPEN DECISIONS (Aryaman): (a) event kind naming + one event per night
(sleep) / per day (activity) vs one combined daily digest event; (b)
push notification: none, or a quiet morning digest push; (c) does the
summary event ride the existing events table (needs a `kind` +
jsonb-body column check) or a parallel feed query over the summary
tables (zero schema change — feeds read two sources).
NIGHT WRAP-UP RULE (v1.5, build during soak): the night closes at
whichever comes FIRST — the noon window end, or DAY-BEGUN: sustained
patient activity in another room (Activity labels ≥15 min outside the
bedroom after a bed-exit). Fixes the morning-nap conflation: a post-
day-begun return to bed becomes a NAP (separate, or logged in
activity), not an extension of the night.

## EXIT REVIEW (mandatory, before pilot go-live)

- [ ] Walk the "Parked" list AND "Known residual risks" line by line with
      Aryaman — each gets an explicit "accepted for pilots" or "do now".
      Nothing ships as silently-parked.

## Open decisions (Aryaman)

- [ ] intimate_situation: the one silent carve-out to "censored clips keep
      audio"? (flagged by privacy session)
- [ ] Pilot families: research consent from day 1 (pulls items 6-9 forward)
      or observation-first?

## Parked (explicitly NOT for pilots)

Counsel questions (Zinia + US: bathroom statutes, BIPA, household notice,
"encrypted in transit/at rest" wording — runs in parallel, not code);
patients.privacy_mode column drop + client legacy-wiring delta (after clients
ship); detector hygiene (zero-link cleanup, room chips, duplicate prevention);
web room-rename UI; Docker Desktop launchd keepalive (dev-only); legacy
env-named settings scrub (move with container env); stale docstring sweep;
yoloseg containment-dedup retrial; seizure-FFT + gait ST-GCN (post-pilot).

## Known residual risks (accepted, from privacy plan doc)

Raw clip bytes survivable in scratch only on SIGKILL mid-censor (boot sweep +
ephemeral FS); Tapo classification lacks motion-settled frame gate (fails
closed); two cameras sharing a room name can't diverge for scoped detectors;
legacy silent-clip mystery never root-caused (probes guard the live chain).
