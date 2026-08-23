# Detection v2 — 4-hour soak protocol (2026-07-14)

> Status (2026-07-27): moved from docs/plans/ — this is the reusable soak methodology (objectives, truth-CSV scoring, instrumentation map, rollback), not a one-shot plan. The original 2026-07-14 run itself aborted at ~50 min on substrate issues (Docker VM ≠ native experiment substrate); the 2026-07-18 24h cost soak succeeded it for the cost question. Date-specific setup details (branch commit, convulsion-rename delta, that day's detector loadout) are historical — adapt per run. The 2026-07-26 detector-axis reconciliation section at the bottom is still an OPEN gate: duration / trailing / absence / M-of-N firing paths have not yet been soak-validated and must be before detection is re-enabled.

A formal experiment, not a vibe check. Four objectives, each with its
own data, scoring, and pass bar. Locked context: no baseline period
(fresh system), NO k-of-N configs anywhere (Aryaman: confirm is a
per-detector opt-in for miss-prone/false-fire-prone detectors, never a
blanket 3×; nothing carries it in this soak), `scheduler_env_motion_floor`
kept as a dial at 0.0, enrollment done fresh by Aryaman at rebuild.

## Objectives

| # | Question | Data | Pass bar |
|---|---|---|---|
| O1 | **TP** — does every acted event fire exactly once? | truth-CSV takes vs events | every take TP; no doubles |
| O2 | **FP** — does ambient living stay quiet? | ambient segments (FP denominator hours) | candidate-FP list adjudicates to ~zero real FPs; convulsion 0 fires outside its takes |
| O3 | **Gate tuning** — what should the dials be? | `sig_*` percentiles per labeled segment + rapid gate readouts on takes vs confounders | clean separation: idle percentiles ≪ acted readings; dial candidates written down |
| O4 | **Cost** — calls/tick and $/day pace | `sched_calls` ledger | $/day pace acceptable to Aryaman; no call-error storms |

## Setup (T−20 min)

1. Local stack up on branch `feat/gateway-camera-ownership` (≥406b61a).
2. Apply `scripts/deltas/apply_convulsion_rename_delta.sql` to the local DB
   (all other deltas already applied during P1).
3. `.env`: `NEMARA_DETECTION_V2_SCHEDULER_ENABLED=true`,
   `NEMARA_DETECTION_V2_DECISION_ENABLED=true` → **recreate** the
   container (hot-reload lies about env).
4. Aryaman: create the fresh patient, enroll himself, connect cameras,
   assign rooms.
5. Detectors: seeded presets ON (fall, agitation, wandering incl. exit
   doors, medication) + Convulsion + Stove left on; create through the
   app: custom **"watching TV for more than 15 minutes"** (exercises
   compiler v2 → person-still + sampled + entry preset (5, 900)); two
   scheduled checks with deadlines inside the window — one Aryaman
   satisfies, one he deliberately misses.
6. Start the watcher + note the start time:
   `uv run python tools/soak_watch.py --since <ISO-now>`
7. First-tick sanity: analyses row carries `sched_calls`; exactly one
   scene call; no doubled events; Convulsion visible in the app under
   its new name.

## Segment schedule (parallel across cameras; ~4h wall clock)

| Segment | Where | Duration | Purpose (objective) |
|---|---|---|---|
| A — idle empty room | cam 1 | ≥30 min | floors data: idle `sig_env_motion` / track-amount percentiles (O3) + FP hours (O2) |
| B — TV on, room empty | cam 1, after A | ≥40 min | TV-as-person: tracker-only presence share on a known-empty room (O2/O3) |
| C — fan/curtain, closed room | cam 2, whole session | ≥60 min | perpetual-wake count on a sleeping camera (O3/O4) |
| D — acted takes | cam 3 (living area) | ~60–75 min | the TP matrix below (O1) with confounders (O2) |
| E — normal living | cam 3, gaps between takes | whatever's left | ambient FP hours (O2) |
| F — privacy mode | any | 15 min | detection keeps flowing, zero clips (integration sanity) |
| — scheduled pair | patient-level | deadlines in-window | absence runner both directions (O1) |

Segments A/B/C run unattended in parallel while D happens — that is
how 4 hours covers everything.

## Acted-take matrix (segment D)

Every take gets one truth-CSV line. Nearest confounders are takes too
(their truth is "must NOT fire" — they get NO row; if they fire they
surface on the candidate-FP list).

| Detector | Takes | Confounder (must stay quiet) |
|---|---|---|
| fall | 2 (fast collapse; slow slump) | 1 sitting-down-hard |
| convulsion | 2 (~20 s each, lying + sitting) | 1 exercise burst (jumping jacks) |
| medication | 2 (pill-take gesture) | 1 snack/phone-to-face |
| wandering | 1 exit through the tracked door | 1 walk-past-the-door |
| stove_left_on | 1 burner on, walk away ≥10 min | — |
| watching TV >15 min | 1 continuous 20-min watch | 1 two-minute glance (entry rule must hold it) |
| visitor/suppression | if a second person is available; else skipped | — |

Truth CSV (`soak_truth.csv`, repo root; Claude maintains it live from
Aryaman's play-by-play in chat):

```csv
detector,start,end,note
fall,2026-07-14T14:32:10-07:00,2026-07-14T14:32:25-07:00,fast collapse living room
convulsion,2026-07-14T14:40:00-07:00,2026-07-14T14:40:25-07:00,lying on couch
```

## Instrumentation map (what is recorded, per layer, automatically)

| Layer | Fields on every analyses row | Read via |
|---|---|---|
| signals | `sig_env_motion`, `sig_tracks[]` (amount/churn per track), `face_valid`, `motion_src`, `people_present_src`, `woke_by_backend` | watcher / report percentiles |
| scheduler | `sched_skipped[]` (ident:reason), `sched_crowd_fallback` | watcher / report skip histogram |
| packer/executor | `sched_calls[]` (key, detectors, provider, tokens, gate state), `sched_degraded_*`, `sched_call_errors` | watcher / report cost + degradation |
| rapid | `rapid_gate_active_fraction`, `rapid_gate_score`, `rapid_motion_description` | O3 margins: takes vs confounders vs idle |
| decision | `sched_confirm` (none this soak), event rows, `clip_suppressed` under privacy | events feed / report |

Nothing extra needs saving — the analyses table IS the experiment log,
immutable and queryable mid-soak. Live view: `tools/soak_watch.py`
(per-camera last-tick readout of every layer + running $ pace +
events). Ad-hoc mid-soak questions ("what did the gate read at 14:32?")
are one SQL query — ask Claude in chat, answers come from the rows.

## Scoring (end of session)

```bash
uv run python tools/soak_report.py --hours 4 --truth soak_truth.csv
```

- O1: every truth row TP (FN rows print how many ticks inside the
  window answered detected=true — separating "model never saw it"
  from "decision layer held it").
- O2: walk the candidate-FP list; each is real-FP or explainable.
- O3: dial candidates from labeled segments — person floor ≈ p90 idle
  track amount (segment A/E sitting-still periods), env floor stays a
  dial with its candidate noted from segment A p90/p99, rapid-gate
  margin = takes' readouts vs exercise-confounder readout vs idle.
  Dials MOVE only after this soak, via settings, one at a time.
- O4: calls/tick histogram + $/day pace vs Aryaman's comfort line.

## Rollback

Remove the two env lines, recreate the container: legacy owns the tick
again in ~30 s. Nothing to migrate; analyses rows keep the soak data.

## Out of scope (4h)

Multi-day FP base rates, night/IR, >4-person crowds, k-of-N efficacy
(configs deliberately zero), face/zone artifact buckets (P6), floor
raising itself (post-soak, settings-only).

## Detector-axis reconciliation — validation requirements (added 2026-07-26)

The reconciliation replaced `shape` with `alert_trigger`
(`leading`/`trailing`/`duration`/`absence`) + `is_sustained`, made M-of-N an
internal confidence gate, and added two new firing paths. These MUST be
validated before re-enabling detection — the soak is their correctness gate:

- **Duration** (`alert_trigger=duration`, e.g. stove-left-on): fires **once**
  after the activity has been continuously present ~`duration_seconds`
  (windowed-count crossing in `event_promotion._apply_duration_entry`); does
  **not** fire before the threshold; re-arms after a lull drains the window.
  **Requires `detection_v2_decision_enabled=ON`** — with it off, duration falls
  back to the rising edge (fires early). Tunable: `_DURATION_TICK_SECONDS` (60),
  the `max(2, D//tick)` m.
- **Trailing** (`alert_trigger=trailing`): fires **once** on the FALLING edge —
  a sustained activity detected, then absent for `episode_gap_seconds`
  (`ScheduledDetectorRunner._evaluate_trailing`). Verify: it does NOT
  rising-edge fire (event_promotion skips it), fires exactly once per session,
  and a fresh session re-arms. Tunable: `_DEFAULT_TRAILING_GAP_SECONDS` (300),
  `_TRAILING_LOOKBACK_SECONDS` (6h).
- **Absence** (renamed from `schedule`; positive-`schedule` folded into
  `leading`): unchanged semantics — fire if NOT seen within its `when_scope`
  window; 0%-coverage suppression still applies.
- **M-of-N internal confidence** (`compiler_v2._entry_confidence_for`):
  hard-to-detect sustained states (distance_class / rapid / fine_motion)
  require 2–3 corroborating ticks; easy ones fire on-sight; safety caps
  corroboration lower than routine. The `(m, t)` tiers are **placeholders to
  calibrate on golden data**.
- **Duration/absence with no time given**: a detector inferred as duration
  (or absence) with a null threshold/deadline must be blocked in the UI until
  the caregiver supplies one — never fire on a guessed time.
