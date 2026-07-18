# Soak #2 — 24h cost + stability soak (2026-07-18)

Successor to the 2026-07-14 protocol (aborted at ~50min on substrate).
This run's PRIMARY question is **cost** ("the best way to measure cost
is just through soak" — Aryaman); secondary is 24h stability on the
Docker rig. Detector/crop **accuracy is indicative-only at 640×360**
— per-bucket resolution floors come from the separate replay study,
not this soak.

Rig: `~/nemara-v2-soak` @ e0b41ab in Docker (livekit 1.1.13, layer
pinned MEDIUM 640×360, in-flight cap 2 with visible shed rows,
`restart: on-failure`, reid ON — weights lazy-download on the FIRST
reid tick, expect that one tick to be slow). Gemini funded (probed
200). Dashboard: http://localhost:8010 — fresh since events wipe.

## Objectives

| # | Question | Data | Pass bar |
|---|---|---|---|
| O1 | **$/day for 2 cameras, real living** | hourly chart + `sched_calls` ledger | a number Aryaman can price against; no call storms |
| O2 | **24h unattended stability** | container uptime, red rows, `tick_backlog` count | zero silent death; any crash self-heals visibly |
| O3 | **Delivery health at 360p** | `fps=recv→kept` on every row | recv ≥ 8 sustained; no multi-second gap regime |
| O4 | **FP pace under Gemini** (vs Nagpur's Qwen day) | events list vs reality | FPs adjudicated; compare against the 5-FP Qwen baseline |
| O5 | **Identity + tracking health** | id-consistency % on cam headers, id-evidence frames | tracked% high while seated (known birth-bar risk — measure it) |

## Pre-flight (once, before unpausing)

- [ ] Laptop on power, lid open, `caffeinate -is` running in a terminal
- [ ] Docker Desktop: start-at-login enabled
- [ ] Dashboard :8010 loads, "Calls per hour" card present, tick log empty (fresh)
- [ ] Both cameras online in app (Bedroom + Living Room), AI paused
- [ ] Aryaman enrolled on the LOCAL stack (face enrollment — needed for O5 / identity tests)
- [ ] Detector loadout as intended (presets + convulsion; note what's on, it defines the cost being measured)

## Smoke tests (first ~15 min after unpausing, you in frame)

**Detectors:**
- [ ] 1. Walk into Living Room → analyzed tick, `provider=gemini`, one scene call in `calls=[…]`
- [ ] 2. Sit still 2 min → ticks continue, no detector flags, kept≈3.4fps steady
- [ ] 3. Rapid-shake near camera 10s → `rapid_gateAF` rises, convulsion call appears (gate open), then ⛔ again after
- [ ] 4. Leave room → ~6 quiet ticks → camera **closes** → `skipped · camera_closed` at $0
- [ ] 5. Re-enter → block-motion wake within ~1 tick, camera reopens

**Face recognition:**
- [ ] 6. Face the camera squarely → row shows `identity_patient_in_clip=true`, confidence > 0
- [ ] 7. Back turned / profile → unresolved (face_count=0) → ticks STILL analyze (fail-open is correct)
- [ ] 8. Second person alone in frame, face visible → `SKIP identity-gated` on that tick

**Tracking (first reid tick downloads weights — slow once, then normal):**
- [ ] 9. Walk across the room → `trk=1` on the tick line, track follows on id-evidence frames
- [ ] 10. Sit still 5 min → id-track % on the camera header stays high (seated birth-bar measurement)
- [ ] 11. Two people cross paths → count correct, no label swap on evidence frames

## During the 24h — what watches what

- **You**: nothing. Live your day normally — that IS the experiment.
- **Dashboard**: fps on every line (O3), red `⛔ TICK FAILED` rows (O2), `tick_backlog` sheds (O2), hourly call/token bars (O1), events list (O4), id-track % (O5).
- **Claude watchdog**: alert if analyses rows stop flowing >5 min while cameras online.
- **Self-healing**: container `restart: on-failure`; ingress + failure rows make any incident post-hoc diagnosable.

## End-of-soak deliverables

1. $/day (ledger-priced) + calls/hour curve + per-provider split
2. Uptime story: crashes (if any), shed count, worst delivery window
3. FP list with clip frames, adjudicated, vs the Qwen-day baseline
4. Id-consistency + seated-tracking numbers → feeds birth-bar decision
5. Dial candidates for the leanness pass (duty, cadence, frame diet)

## Known issues going in (logged, not blockers)

- **Clip generation is not clean** (Aryaman, 2026-07-18): inconsistent
  lengths, seam skips where pre/post stitch, frozen last seconds.
  Mechanism: clips are cut from the decoded analytics buffer — length
  = whatever the buffer held (short after wake/reconnect), seam gap =
  delivery gap at the pre/post boundary, frozen tail = CFR padding
  duplicating the last frame when delivery stalls mid post-roll.
  Fix belongs to the encoded clip plane (designed, not built). Do NOT
  judge clip quality this soak; do log egregious examples.
- Fallback (dashscope/openrouter) ticks report 0 tokens in the ledger
  — usage capture pending; the chart's provider split shows how often
  it happened so the undercount is bounded.
- 640×360 person crops are soft — convulsion/micro accuracy is
  indicative only (replay study owns accuracy).
- Seated people may not birth tracks (BoT-SORT birth bar 0.6 vs seated
  conf ~0.3-0.6) — O5 measures it; the 0.4-bar decision is parked.
