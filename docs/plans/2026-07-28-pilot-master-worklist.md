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

- [x] IDENTITY LATENCY — CLOSED 2026-08-01, and NOT the way this item
      predicted. Per-camera pod processes were the wrong fix; the pod
      was never GPU-bound. Three harness defects, each measured before
      and after (commits fd24075, 86b88dd):
        1. every tick JPEG-encoded frames it already held decoded, wrote
           them to a temp dir, and init_state PIL-decoded them back —
           68% of every call, 98.7 ms/frame. Same work as one batched
           GPU op: 4.2 ms.
        2. ground ticks ran SAM3 propagation over 6 frames and used only
           frame 0 — 2101 -> 694 ms, same detections, mask IoU 0.982.
        3. the image encoder ran batch-of-1 per frame into a 1-entry
           cache. Priming it: 1.166x, masks unchanged.
      Result, both rooms occupied: sam 4965 -> ~800 ms, pod round-trip
      7541 -> ~1400 ms, GPU 5-8% -> ~24%, capacity ~2.3 -> ~7 occupied
      cameras. Open question #1 in identity-latency.md is answered:
      EdgeTAM runs at 42 ms/frame against its published ~36. The model
      was never slow.
      MEASURED-NEGATIVE, do not re-try without new evidence: cross-camera
      batching (only touches the ~31% encoder slice) and concurrency
      (nothing left to overlap once the work got small — 2 workers and
      in-process concurrency both landed within noise of 1 worker).
      In-process concurrency is committed and default-OFF
      (NEMARA_PERCEPTION_CONCURRENCY=1); it matters at 4-5 cameras, not 2.
- [x] SOAK KEEPS DYING ON LAPTOP SLEEP — FIXED 2026-08-01 (4855ebc).
      `restart: on-failure` only applies while the Docker daemon runs, so
      a Docker Desktop restart left the backend dead. It was down
      EIGHTEEN HOURS on 08-01 and nobody noticed until someone queried
      for events. Now `unless-stopped`. THIS is why sleep/event data was
      thin — not quiet detectors, a stopped system.
- [ ] NIGHTLY SUMMARISER IS NOT SCHEDULED. `run_daily_summaries.py` is a
      manual "every morning run this" ritual documented only in the soak
      dashboard — no cron, no launchd. That is why sleep_nights had ONE
      row. Sleep validation cannot depend on remembering a command.
      Schedule it before the next soak night.
- [ ] **SLEEP IS BROKEN — the blocker for the sleep ship gate.** All four
      backfilled nights return `incomplete_night` DESPITE abundant input
      (182-278 in-bed ticks/hour, 01:00-07:00). Ruled out: missing data,
      and the night window (`_utc_naive` converts correctly). The defect
      is in the state machine: onset lands in the MORNING (08:00 local on
      07-30 with in-bed evidence from 22:00), wake is never found, and
      one night produced TWENTY out-of-bed events clustered in 11 minutes
      with duration_minutes=0 each — the confirmation guard that exists
      to stop exactly that is not firing. Next step is tick-level: dump
      sleep_activity classifications for one night and find where the
      classifier and the state machine disagree.
      ACTIVITY, by contrast, produces plausible output every day and
      needs only validation against recollection.
- [ ] Choking (5 events) and Agitation (1) fired in the last 7 days
      though both are in this plan's "Left OFF" set. Either false
      positives or the ON-set is not what we think — check before merge.
- [ ] ~~PER-CAMERA POD PROCESSES~~ (superseded above). Original: Proven
      that the pod's `_perception_lock` (`perception/server.py:80`) is
      the real serialiser and that the A40 runs two independent pod
      processes in genuine parallel (1.6-1.8x, pod compute UNCHANGED at
      ~2s). Full measurements + the three-stacked-bottlenecks model:
      `docs/architecture/identity-latency.md` ROOT CAUSE section.
      Build: one pod process per camera on its own port, cameras routed
      to their own process (`_SESSIONS` is process-global, so EdgeTAM
      session affinity comes free instead of round-robining). ~11GB of
      46GB for two model copies. Touches the client (per-camera pod URL)
      + `identity_supervisor.sh` (N processes) + the tunnel (N ports) —
      wants its own design pass, do not bolt on. Only AFTER this does
      the uplink become the binding constraint and frame-size work
      become worth revisiting. Symptom this closes: 12-16s lag whenever
      both rooms are occupied.
- [ ] SAM3 grounding-input resolution pin (pod): downscale frames fed to SAM3
      grounding to 1280-wide (the proven operating point); existing
      mask→track-res resize path unchanged. NOT the consensus-merge idea
      (withdrawn — wrong layer; YOLO is a trigger, not an arbiter).
      Fold in: `_iou` shape-mismatch rescale one-liner (latent landmine).
      Deploy = rsync + pod restart. Success = phantom births ~0 in occupied
      still rooms; forced-reground loop gone; discrimination test still
      tracks 2 real people as 2.
- [x] IDENTITY AUTH — CLOSED 2026-07-29 ~18:30. Token generated, in
      backend `.env`, exported to the identity supervisor. Verified:
      every endpoint 401s without it, 200s with it; backend `/ready`
      reads identity ok (it now sends the bearer, and a 401 reports
      `unauthorized` instead of the old `< 500` → "ok", which would
      have made the check meaningless). ALL SIX endpoints are gated
      now, not just three — `/timeline`, `/events` and `/snap` were
      open, and `/snap` serves camera-cropped JPEGs, same exposure
      class as `/stream`.
      CONSUMERS THAT BROKE AND WERE FIXED (this is the part that made
      it not a 30-second job): `tools/sentinel.py` (urllib RAISES on
      401, so identity read as DOWN — it logged `identity: 0` for two
      cycles before the restart), `tools/soak_dashboard.py`'s four
      server-side proxies, and the dashboard's `<img>` tags, which
      cannot carry an Authorization header — `/identity_stream/{cam}`
      and `/identity_snap/{name}` proxies now front them server-side.
      RELAUNCH PATTERN (anything talking to identity needs the env
      var; a process started before the token was set fails closed):
        export NEMARA_IDENTITY_STATE_TOKEN=$(grep '^NEMARA_IDENTITY_STATE_TOKEN=' .env | cut -d= -f2)
      Applies to identity_supervisor.sh, soak_dashboard.py, sentinel.py.
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

0. [ ] BEHAVIOR category (Aryaman 2026-07-28, MUST-DO post-soak, ~half
       day): three-way split — safety (health/harm, red), behavior
       (clinical observations: agitation, confusion, pacing,
       sundowning — GREEN/informational, data for the care team not
       alarms), routine (rest). Touchpoints: detector_category enum +
       delta, catalog reclass (Agitation), compiler rubric, criticality
       gate (behavior = fail-quiet like routine — a visitor's pacing
       must never enter the patient's behavioral record), web + iOS
       category color/label maps. Trace the FULL enum fanout (DDL,
       schema, both client enums — the record_only lesson).
0b.[x] MULTI-WINDOW when_scope — FIXED same night (2026-07-28).
       CORRECTED facts: per-tick ELIGIBILITY (is_active) always
       honored both windows — leading detectors like Medication
       adherence were never truncated (earlier claim wrong). The real
       truncation was parse_schedule (the ABSENCE runner's lens, first
       window only) + the web editor displaying "All day" for scopes it
       couldn't parse. Fixed: parse_schedules() returns every window;
       the absence runner evaluates each at its own deadline (stateless
       per-window single-fire via event occurred_at == deadline;
       last_scheduled_eval_date = whole-day completion only); web
       editor fail-honest 'other' state shows the scope verbatim.
       Medication row restored to "8 AM, 8 PM +-1H" (the truth).
0c.[x] ALERT-MEANING NAMING — compiler outputs suggested_name (absence
       = miss-phrased, else the activity); web create flow applies it.
       [ ] iOS create flow applies suggested_name (rides next build).
       ALERT-TRIGGER SEMANTICS (documented from the med incident): a
       detector always watches for a POSITIVE activity; alert_trigger
       is the alerting polarity on top — leading fires on sight,
       absence fires on deadline-without-sight, record_only never
       fires. Absence events: title "Possible <name>", watched-for
       narrative, clip_suppressed=True always (no camera, no clip).

1. [x] iOS login resilience — DONE 2026-07-29. `bootstrap()` no longer
       awaits `openRealtimeChannels()` (detached Task), and each
       `subscribeWithError()` now has a 10s budget: the 07-28 hang was
       a call that never RETURNED, so its own catch block never ran.
       The 1s camera poll + foreground resync already carry state
       without Realtime. Cold-launch restore rides the same path.
       Build verified (`xcodebuild` succeeds under Swift 6 strict
       concurrency).
2. [~] New cameras PRIVATE-BY-DEFAULT until classified — DROPPED for
       pilots (Aryaman 2026-07-29). The 15-min bedroom-stream window
       stays open until the classifier lands a verdict.
3. [ ] Backend pool right-size (10+10 → ~5+5 single replica) + role-level
       idle timeouts (config-only).
4. [ ] Key rotation (LiveKit/APNs/Gemini/OpenAI/OpenRouter/Resend) — BLOCKS
       all prod env work; .env.bak still in git history on main.
5. [x] /state token name UNIFIED 2026-07-29 — both sides read
       `NEMARA_IDENTITY_STATE_TOKEN`; the service still accepts the old
       `NEMARA_IDENTITY_TOKEN` with a deprecation warning, and now logs
       loudly at boot when NO token is set (it binds 0.0.0.0, so an
       empty token means /state, /probes and /stream are open).
       REMAINING (ops, not code): actually SET the token on both sides
       — it is unset today, so identity is currently open on the LAN.
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

DECIDED (Aryaman 2026-07-28, final): record_only stays TABLE-ONLY — no
event of any kind for sleep/activity. The events feed remains purely
alert-worthy moments; sleep/activity insight surfaces later through a
dedicated reader of the summary tables (sleep card / weekly family
report), never through the events pipeline.
NIGHT WRAP-UP (Aryaman final 2026-07-28: keep it simple — SHELVED:
episode scorer, main-sleep selection, day-begun heuristic). What ships
is v1+reopen exactly as built: wake = a seen not-asleep moment with no
asleep sighting for 30 min (WAKE_CONFIRM) — so a re-sleep within 30 min
counts as part of the night (Aryaman's rule, already the code). Known
quirk accepted pending real-night data: a LATE return (hours later,
still before noon) also merges — night SPAN stretches but
duration_minutes correctly excludes awake time. Revisit only from soak
data.

0d.[x] CLIP A/V SYNC — FIXED 2026-07-28 night (frames_to_vfr_mp4 in
       mp4_remux.py): true per-frame durations via concat demuxer,
       materialized on a clip_fps grid (`-vf fps=N` + exact `-t` cap —
       raw `-vsync vfr` writes lying container durations), leading-PCM
       trim for A/V t=0 alignment, no `-shortest`. Symptoms this
       killed: 10-12s wandering clips for ~26s windows, the 21:27 fall
       couch→floor "teleport", audio drift. Soak test B10 verifies on
       every event clip from ~23:30 on. VLM-path frames_to_mp4 kept
       (model doesn't watch in real time).

0e.[x] WANDERING DOOR-GATE — 2026-07-28 night: registry.py legacy
       grandfather branch DELETED; NULL door_config == unset == the
       sensor does not run wandering (matches the app's "inactive
       until exit doors set" promise; the generic prompt produced two
       FP events in 2.5 min on family room-exits). Prompt targeting
       also hardened: ONLY the designated exit door counts — any other
       exit from frame (other doorway / off-edge / behind furniture)
       is explicit FALSE. NOTE: test_rapid_seizure_path.py deleted
       with Convulsion (Aryaman call) — the rapid-overlay pipeline has
       NO end-to-end test now; re-add via a synthetic rapid-bucket
       custom detector if rapid customs ship to users.

## Resilience + telemetry architecture (as-built 2026-07-29 — the
## seed of the post-soak hardening discussion)

Born from three DISTINCT same-day infrastructure failures, all with
the same disease — process-alive ≠ data-flowing:
  1. 05:37–07:25 backend Docker-container egress wedge (new outbound
     connections to LiveKit timed out; gateway + internet fine;
     /health said 200 throughout; 108 min blind).
  2. 11:02 gateway box wifi blip → LiveKit publisher ZOMBIE
     (participant + publications registered, zero media; systemd saw
     a running process; backend logged "walking for missed
     track_subscribed" forever).
  3. Identity service stranded subscriptions after the gateway
     restart (its /state answered with frozen data; separately it had
     silently frozen 04:19→08:07 holding the pod lock after a tunnel
     death — watchdog blind).

Layer 1 — in-process recovery (each service saves itself first):
  - backend room_worker: reconnect loop + publication walk (existed).
  - identity service: NEW missed-subscription walk (10s cadence:
    unsubscribed publication → subscribe; subscribed-but-intake-quiet
    15s → toggle). gpu/identity/service.py main loop.
  - tunnel: gpu/deploy/tunnel_supervisor.sh active /health probe
    (3 fails → kill+relaunch; ssh stderr captured to
    /tmp/pod_tunnel.log). NOTE 2026-07-29: the legacy
    scripts/pod_tunnel.sh loop from Jul 27 was found still running
    and RACING it over port 8000 — killed; delete the script file.
  - identity supervisor: gpu/deploy/identity_supervisor.sh (existed).

Layer 2 — data-freshness probes (the truth signals):
  - GET /health/freshness (backend): per-ONLINE-camera newest-analysis
    age, 503 when any > 300s. Excludes disabled/ai_paused/offline so
    intentional disconnects can never read as outages. THE portable
    prod artifact: point a Railway healthcheck / external pinger at it.
  - GET /ready (backend): dependency diagnosis layer — db (hard 503),
    livekit reachability, identity /state, gateway_poll age (stamped
    by /v1/gateway/wait). Reported-not-gated: status ok|degraded;
    only db kills readiness. Caveat encoded in the docstring: an HTTP
    200 from a dependency proves reachability, NOT data flow.
  - identity /state lag_s per camera; /probes (per-probe face/body/
    cand scores, 2026-07-29) for the dashboard's tick-correlated view.

Layer 3 — the sentinel (tools/sentinel.py, host-run, stdlib-only,
dev-rig; journal /tmp/sentinel.jsonl):
  - Probes every 30s: backend/identity/tunnel/dashboard HTTP, frames
    age via /health/freshness, identity lag via /state, gateway ping,
    Docker gvisor in-progress-connection trend.
  - Actions (shared 600s cooldown + 3/hr cap; forensics captured
    BEFORE every restart into /tmp/sentinel_forensics/):
      frames stale + backend 200 + gateway pings
        + walk-signature in backend logs → GATEWAY restart (ssh)
        + no walk-signature → BACKEND docker restart
      identity lag > 300s WHILE frames fresh → IDENTITY restart
        (high bar: costs the in-memory body pool).
  - First live fire 2026-07-29 11:08: detected in <5 min, forensics
    correctly ruled out the egress-wedge theory in one file read.
  - Deliberate non-goals: never restarts on dead gateway/backend
    probes alone (wrong-side restarts), never acts on an
    intentionally-quiet home.

Prod mapping (the post-soak discussion): freshness endpoint +
external pinger replaces the sentinel's probe layer; caregiver-facing
home-offline alert MUST key on data freshness (pilot-blocker);
Railway's own healthcheck must NOT gate on /ready degraded (only
db-fail); colocation deletes the tunnel class outright.

DECIDED (Aryaman 2026-07-29): identity is stream-scoped, not
toggle-scoped. "Pause AI" / Hide stop DETECTION only (streams keep
flowing by design for instant resume) and identity keeps processing —
it has no user-facing output and is what makes detection correct at
resume. Parity is required ONLY for actual camera disconnects, and
there it is structural: disconnect unpublishes → identity reader
dies + participant_disconnected teardown + stale-intake track clear.
No code change. Pilot consent language should describe identity
processing as tied to the camera being CONNECTED, not to AI toggles.

CLOSED 2026-07-29 — gateway publish-freshness watchdog. The box
cannot see its own zombie (room, publications and connection_state
all read healthy while zero media leaves), so the SUBSCRIBER tells
it: `/gateway/roster` now returns `online` per camera (the backend's
frame-arrival view, independent of AI pause), and
`RosterManager._check_publish_freshness` exits the process after
WD_PUBLISH_DARK_S=120s of publishing-into-the-void. Discovered while
wiring it: the 2026-07-27 `_watchdog` was DEAD (never scheduled,
and it read a RosterManager attribute off CameraSource) — which is
exactly why the second zombie survived. Its LiveKit-unhealthy
judgment now runs for real in the same method. BOTH judgments are
gated on this box actually decoding frames for that camera, because
an unplugged camera tears its publishers down and would otherwise
trip the unhealthy branch every 45s forever, restarting the whole
box (and every other camera on it) on a fault it cannot fix.
Tradeoff accepted: a wedged BACKEND that reports every camera
offline will make the gateway restart every ~2 min. Cheap, and the
alternative is a blind home.

CLOSED 2026-07-29 — sentinel is KEY-ONLY. `~/.ssh/memoria_ops.pub`
installed in the box's authorized_keys; the expect/password path is
DELETED, not made optional (a shared password usable from any machine
on the LAN is not a control for hardware in someone's home). ssh and
rsync to the box now work plainly. Host + key path override via
`NEMARA_GATEWAY_SSH_HOST` / `NEMARA_GATEWAY_SSH_KEY`.

- FLEET SSH — root gap found + closed in code 2026-07-30. A box cut
  from the golden image accepted only the account PASSWORD: prep and
  firstboot regenerate HOST keys but never seeded authorized_keys, so
  the reverse tunnel reached port 22 on any box and nothing got you
  through it except the shared string in the repo. That is why the
  password was load-bearing, and why "retire the password" was
  premature advice. nemara-gateway `1c8421a`:
  `deploy/ops_authorized_keys` (public keys, committed on purpose) is
  seeded by prep-golden-image.sh and re-asserted by firstboot, both
  additive + idempotent, verified on the real box (fresh home, re-run,
  foreign key preserved, ssh-keygen -l parses, 0600/0700).
  - [ ] Cut a golden image WITH the seeding step, boot ONE clone, and
        confirm key-only ssh works on it.
  - [ ] Ops key on the RELAY (pending Sunny) — the other half of
        reaching a box nobody has touched.
  - [ ] THEN PasswordAuthentication no, and append the key by hand to
        every box already in the field. Not before: until then the
        password is the only way into deployed hardware.
- [ ] Sudoers rule — OPTIONAL, deprioritised (Aryaman 2026-07-30). The
      gateway restarts ITSELF (os._exit + systemd, no ssh, no sudo, no
      credentials), so the sentinel's ssh restart is only a backup for a
      process alive-but-wedged past its own watchdog, on a dev-rig tool
      prod replaces with an external pinger. If ever wanted, ON THE BOX
      (not the Mac — that was why it failed):
      `/etc/sudoers.d/memoria-gateway-watchdog` granting NOPASSWD on
      `systemctl restart|is-active|status memoria-gateway` ONLY. Exact
      lines in nemara-gateway/GATEWAY_ACCESS.md §3. Until then the
      sentinel's gateway restart reports
      `GATEWAY_RESTART_FAILED: sudo wants a password` and does not
      hang — and the gateway's OWN watchdog (which needs no ssh at
      all) is unaffected, so the home is still covered.

### 2026-07-29 adjudicated review — OPEN findings (fix before merge)

Three review agents audited the day's diff. 9 defects fixed same-day
(see changelog below); these remain OPEN and are merge-blockers unless
noted:

- **door_config wiped on detector-list edit** (`camera_detector_repo.
  replace_assignments_for_camera`): the PUT deletes+reinserts
  cameras_detectors rows without carrying `door_config`, so a
  caregiver toggling any detector on a camera silently DISABLES
  wandering there. Fix: preserve door_config (select-then-restore).
  Also `child_wandering` can never obtain a door_config — the picker
  hard-codes "wandering" — so it is structurally dark; parameterize
  or drop it from _WANDERING_IDENTIFIERS.
- **Sleep summariser uses the RUNNER HOST's timezone**, not
  `caregivers.timezone`, and captures a fixed offset at import (also
  wrong across DST). Blocks the India pilot. `camera_ingest.
  _resolve_caregiver_tz` is the pattern to copy.
- **Sleep ticks are not camera-scoped** — `_TICKS_SQL` pulls every
  camera carrying the sleep detector; a second camera's ticks
  manufacture out-of-bed events. `SleepTick` needs a camera field.
- **Onset walk-back crosses blind gaps and the identity gate**, so
  `sleep_onset` can be attributed to a visitor hours early; and
  `bedtime` is no longer identity-gated and is None whenever onset
  isn't found (v1 set it first).
- **PCM alignment is back-computed from byte length** — any audio
  underrun silently displaces the whole track (measured: 16s offset).
  Buffers must return the first chunk's captured_at instead.
- **Sleep-watch session self-extinguishes ~1h after the last in-bed
  sighting**, so a 3am wake under a blanket isn't covered; and
  `DIMS["sleep"].requires_motion=True` means raising the post-soak
  scheduler dials would strip sleep fields from forced looks entirely.
- [x] FIXED 2026-07-29 (below): `/ready` auth + identity URL,
  `/state` token name, gateway publish-freshness watchdog, iOS
  sign-in blocking on Realtime.
- ACCEPTED, no fix (Aryaman 2026-07-29): censor cost doubling with
  honest ~26s clips — we take the hit.
- DROPPED (Aryaman 2026-07-29): new cameras private-by-default until
  classified.
- **Sentinel**: `gateway_zombie_signature` keys on a join-time log
  line, so it does NOT fire in the zombie case it was written for
  (restarts the backend instead); `pkill -f` can hit the uv wrapper /
  unrelated processes and does nothing if identity is off-host;
  restart budget is shared, so identity restarts can starve a real
  backend wedge. Per-target budgets + a backend-exported signal.
- **`/ready` holds a DB session across ~6s of outbound HTTP**,
  is unauthenticated, and `get_settings()` has no lru_cache (re-reads
  .env per call) — close to the 2026-07-28 pool-exhaustion incident.
  Also `identity_state_url` defaults to host.docker.internal, so
  /ready will read "degraded" on Railway forever.
- **Censor cost doubles** with the now-honest ~26s clips (208 decoded
  frames vs ~83, single-threaded YOLO, 180s timeout → a timeout
  downgrades to full-frame blur); `censor.py`'s fps fallback is 4.0
  but clips are clip_fps=8. Sensitivity screen spend ~2.4x/clip.
- **Identity `_remove_camera` never cancels `_probe_task`** — a
  removed camera's probe loop runs forever and pins its native-res
  ring (GBs). Pre-existing, amplified by camera churn.
- VFR encode does per-frame JPEG conversion + file writes ON the event
  loop; move to a thread. `end_at` earlier than the last frame is
  silently ignored (drift, never truncation — verified).

## EXIT REVIEW (mandatory, before pilot go-live)

- [ ] Walk the "Parked" list AND "Known residual risks" line by line with
      Aryaman — each gets an explicit "accepted for pilots" or "do now".
      Nothing ships as silently-parked.

## Open decisions (Aryaman)

- [x] intimate_situation IS the one silent carve-out — DECIDED + BUILT
      2026-07-29. `sensitivity.SILENT_CATEGORIES`; `censor_clip` /
      `_censor_full_frame` take `keep_audio`. Everywhere else audio
      stays (the cry-for-help rule). A MISSING verdict does NOT
      silence — that path is the screen-failure fallback and fires on
      ordinary clips, so silencing it would strip the shout from a fall
      whose screen timed out. Silence requires a positive finding.
- [x] Pilot families: OBSERVATION-FIRST — research consent is NOT
      assumed from day 1 (Aryaman 2026-07-29). Consequences: items 6-9
      (B2 recording censoring, consent attestation + BIPA, sensitivity
      eval corpus, censor soak) stay in their current position rather
      than being pulled forward; the data-collection flag ships OFF and
      any research-consent home is opt-in per family, later. Do NOT
      build anything that presumes a consented corpus exists.

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
