# Tracker eval — morning runbook (SAM3 vs SAM3+EdgeTAM)

> **2026-07-24 — the IDENTITY SERVICE supersedes this rig for live testing.** EdgeTAM won
> the arm comparison (ids held through crossing/swap/partial-occlusion, live).
> The continuous service is `gpu/identity/service.py` — formerly `tools/run_identity.py`,
> consolidated under `gpu/identity/` in the 2026-07-27 refactor (`ecbefea`); see
> "Identity-service quickstart" at the bottom. This rig stays for A/B reruns and
> trace-based scoring (`src/services/identity/trace.py` + the soak-dashboard card survive).

Built + deployed overnight 2026-07-23. The lean rig is **live now on the SAM3 arm** and verified. This is what to do in the morning.

## What's running right now

- Backend recreated with `reid_lean_eval_mode=true` + `reid_trace_enabled=true`. Confounders stripped, trace emitting, verified on live ticks.
- Pod on `NEMARA_TRACKER=sam3`, stride=1.
- Dashboard live at **http://localhost:8010** → the **"Tracker eval — live trace"** card (metrics + per-tick panel, polls every 5s).
- Trace file: `out/track_trace.jsonl` (bind-mounted, the dashboard reads it).

## To run a scenario

1. **Truncate the trace for a clean slate:** `: > out/track_trace.jsonl`
2. Do the scenario (walk in, sit, cross, leave — the S0–S8 ladder in the protocol artifact).
3. Read the dashboard card. Everything auto-updates.
4. On-screen box labels already exist in the id-evidence thumbnails: `P7 0.53 *face` = label-initial + track-id + confidence (+ `*face` if face-locked). Use them to tell me "T7 was me, T4 was the guest."

## To switch to the SAM3+EdgeTAM arm

Two spots (pod + backend; the backend env is the arm *label* the trace records):

> **Rewritten 2026-08-11.** Everything this section used to describe is gone:
> `scripts/pod/pod.sh deploy`, `scripts/deploy_continuous_remote.sh` and the
> supervisor heredoc it wrote were all deleted when the node became an image.
> The tracker arm is still selected by `NEMARA_TRACKER`
> (`gpu/perception/continuous.py`, default `edgetam`) — only the way it is
> set has changed.
>
> `NEMARA_TRACKER` is now node ENV, set at create time. To flip the arm,
> add it to `.pod.env`, forward it in `pod.sh release`'s optional-env list,
> and release a node. For a one-off experiment it is faster to ssh in and
> restart the perception loop with the var set — but that change dies with
> the node, which is the point.

```bash
# 1) Node (functional — the actual tracker):
#    NEMARA_TRACKER=sam3 in .pod.env, then scripts/pod/pod.sh release <tag>
#    (omit it to revert to edgetam)
# 2) Backend (label only):
sed -i '' 's/NEMARA_TRACKER=sam3/NEMARA_TRACKER=edgetam/' .env
docker compose -f docker/docker-compose.local.yml up -d --force-recreate --no-deps backend
```
Run the exact same scenario ladder. Same rig, same panel, only the tracker changed.

## What the panel shows (and what it deliberately doesn't)

**Computable without ground truth — shown:**
- **latency** per stage (`track_ms` is the arm's cost) + e2e — the black box, opened.
- **body score distribution** (min/median/max of real cosines) — "what wins out."
- **event counts** (BOUND / UNBOUND / SEAT_STOLEN / SEAT_KEPT / HARVEST).
- **reset flag** — ⚠ if a worker teardown wiped the ledger mid-run → **that run is contaminated, redo it.**

**Needs your ground-truth labels — you overlay:**
- acquire / release latency and false-positives. I refused to fake these from events; the raw timeline is there so it's a read, not a guess.

## Confounders stripped (verify with `reid mode=full` in the log + `resets=0`)

governor → full · recent-lane → off · scene-extension (150s ghost) → off · face throttle → full cadence · grace 15s → 2s. The **ledger stays on** — it's the binding brain (single-seat/bind/conflict), not hysteresis; grace=2 + scene-ext-off make its hold a 2s blink-bridge.

## The 8fps capability config (set for the morning — NOT prod)

Max-frame capability test. Current `.env` + pod values:
- `NEMARA_REID_LEAN_EVAL_SAMPLE_FPS=8` → **~65 frames/observe** (delivery-capped; 10fps is best-effort/variable).
- `NEMARA_LIVEKIT_WORKER_TICK_SECONDS=40` — **the critical one.** Each dense observe takes ~22-25s; the tick is spaced past that so they don't overlap. At tick=10 the pod queues and **e2e spirals** (24→28→30→35s) toward the timeout → drops. Measured this directly.
- `NEMARA_REID_TIMEOUT_SECONDS=60` (schema max), `NEMARA_REID_SAM_REQUEST_TIMEOUT_S=55`, pod `NEMARA_PERCEPTION_ACQUIRE_S=90` — backstops.

**Behavior:** ~65-frame observations, **e2e ~20-25s, one dense observation every ~40s.** Dense-within-window (the tracking capability you're testing), sparse-between (the wire cost). **Time a crossing to land inside an observed window** (~11s of every 40s).

**Single-camera only.** 8fps saturates the single-flight pod. Two cameras (S7 room-to-room) can't both do a dense observe per cycle → drop `SAMPLE_FPS` to 2-3 and `TICK_SECONDS` back to ~15 for multi-camera scenarios.

**Prod-unrealistic** — 20-25s latency, sparse coverage. This finds the capability ceiling with max frames; prod needs the latency work (JPEG compression at same res / co-located pod — measure whether the wire is bytes or pod-serial-wait first).

## Sweeping the accuracy levers (the actual point)

All env, one recreate each (`docker compose -f docker/docker-compose.local.yml up -d --force-recreate --no-deps backend`):
- **frames** `NEMARA_REID_LEAN_EVAL_SAMPLE_FPS` (move `TICK_SECONDS` with it) — the frames↔coverage trade.
- **tracker arm** `NEMARA_TRACKER` (pod + `.env`).
- **resolution** `NEMARA_REID_INFERENCE_SAMPLE_WIDTH` (1280 now; lower = more frames but smaller faces → worse acquire).
- **face floors** `NEMARA_REID_FACE_LOCK_MIN_PX` (36 — the thing blocking your far-camera locks), `..._SCORE_SINGLE/MULTI` (0.45/0.55).
- **body bar** `NEMARA_REID_BODY_MATCH_PATIENT_THRESHOLD` (0.72).
- **grace** `NEMARA_REID_BINDING_RETIRE_AFTER_S` (2).

## Findings from last night (context, not solutions)

- **Harvest is face-gated** — every pool write requires a face lock (11 harvests = 11 face locks, exact-ms match). Face rarely locks on far cameras (20–33px, under the 36px floor), so the body pool starves.
- **A clothes change wipes body identity** — OSNet keys on appearance; new clothes score ~0 against the old-clothes pool until a fresh face lock re-harvests.
- **A LiveKit re-subscribe wipes the ledger** — the in-memory binding doesn't survive a worker teardown (that's what the reset markers catch).
- The bedroom couldn't identify you for ~20 min until you got close enough for one 0.601 face lock.

## Teardown when done

Remove the `# Tracker eval rig` block at the bottom of `.env`, then `make backend-up`.

## Identity-service quickstart (2026-07-24 architecture; paths updated 2026-07-27)

One process: continuous tracking + three-rule identity + live overlay.
(`tools/run_identity.py` became `gpu/identity/service.py` in the `ecbefea`
consolidation — same service, same port.)

```bash
pkill -f live_tracker_view.py   # spine replaces the viewer (tools/live_tracker_view.py)
# LiveKit creds: exported in env, or already present via .env
uv run python gpu/identity/service.py --room patient_75a30940-b5e2-4608-afa3-b31a1e1fcda6
# production form (restart-on-exit supervisor, same service):
#   NEMARA_IDENTITY_ROOM=patient_<uuid> gpu/deploy/identity_supervisor.sh
# open http://localhost:8012/    (all cameras, PATIENT box, lag HUD)
# state JSON: http://localhost:8012/state
```

Prereqs: pod up in EdgeTAM mode (`scripts/pod/pod.sh health`, or check
http://localhost:8000/health through the tunnel), DB up (face enrollment
gallery). Historical note (2026-07-24 rig): the backend container stayed
stopped so the spine owned the A40 — since then the backend consumes this
service's `/state` (`src/services/identity/identity_client.py`), so both run.

What to verify against the test matrix
(`docs/research/2026-07-23-tracker-eval-findings.md` §5):
acquire at the door (face lock → green PATIENT box), hold through movement
/ crossing / partial occlusion (seat rides the track), solo harvest filling
the body pool (watch body scores rise on far/seated re-locks), release on
exit (seat dies with the track), cross-room hand-off (seat moves cameras),
guest-only room stays amber, lag HUD ~1-2s.

## Not built (deliberately deferred, low value / high risk)

- **Per-frame box TRK with pod origin/stitch** — would localize the swap to the exact frame, but needs a pod redeploy (risk on the decommissioning host). The panel shows per-tick scores/events; per-frame boxes are a fast-follow.
- Auto acquire/release/FP metrics — need your labels (above).
