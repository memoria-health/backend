# Deploy handoff — 2026-08-26, ~03:00 IST

Written so the next session starts from facts rather than from memory.
Everything below was measured tonight unless it says otherwise.

## Where the code is

| repo | branch | state |
|---|---|---|
| `nemara-backend` | `chore/deploy-prep` | 1 unpushed commit (docs deletion) |
| `nemara-bridge` | `main` | clean, PR #1 merged |
| `nemara-caregiver-app` | `main` | clean, PR #32 merged |
| `nemara-gateway` | `docs/simulcast-ladder-restored` | **not main** |
| `nemara-web-app` | `fix/dashboard-local-dev` | **2 dirty — the privacy notice** |
| `nemara-camera-app` | `main` | 2 dirty — local build fallbacks, must not ship |
| `nemara-caregiver-app-android` | `phase-2-networking-auth` | 3 dirty (tests) |

PRs #44 (backend), #1 (bridge), #32 (iOS) are merged with merge commits,
so every commit message survives in `git log` and `git blame`.

## What is running right now

- **Identity node** `7yettlzalhkkl0`, tailnet `100.79.162.85`. Both shards
  ticking, `tick_health.failed=0`.
- **Bridge** on the Mac, publishing the TV Room at 1920x1080.
- **Gateway** on box-01, publishing the bedroom Tapo at 1920x1080.

**The node runs `NEMARA_DEV_CODE=1`** — its code is an overlay in
`/persist/dev-code`, on container disk (`volumeInGb: 0`). A pod recreate
wipes every fix from tonight. `gpu-v3.5` was tagged to end that; check
`gh run list --workflow=build-gpu-image.yml`.

## Next actions, in order

1. **`pod.sh release gpu-v3.5`**, then `NEMARA_DEV_CODE=0`, clear the
   overlay, confirm the node reproduces tonight's numbers. This is what
   makes the work durable and it is the highest-value item.
2. **Probe prod**: run `scripts/probe_prod_schema.sql` (Supabase MCP, or
   paste into the SQL editor). 68 checks over 42 of 62 deltas.
   **TRAP:** `episode_gap`, `where_scope`, `is_builtin`, `pi_source`,
   `shape` read `present=false` because a later delta DROPPED them.
   Re-applying undoes the drop. 20 deltas cannot be probed at all and
   are listed at the bottom of the file for a human.
3. **Backend to Railway**: deltas → env → `railway up --service
   nemara-backend --detach` → `/ready` → seed the App Review demo camera.
4. **Bridge to Railway**: see below.

## Bridge to production — read before starting

`CLOUD_DEPLOYMENT.md` is accurate and names the risk itself:

> Not yet proven: a real camera streaming from a cloud host. The vendor
> transport is WebRTC over Warner cloud/P2P/TURN, and a container sits
> behind NAT with no inbound UDP.

So this is an experiment, not a config exercise. Provision ONE camera and
watch `last_frame_age_s` before moving anything real.

Three constraints that are correctness, not preference:

- **A volume at `/data` is mandatory.** `inventory.json` holds each unit's
  device key and admin password; without it every redeploy un-provisions
  the fleet.
- **`numReplicas: 1`.** Two replicas share `BRIDGE_TOKEN`, publish as the
  same LiveKit identity, and disconnect each other forever.
- **`/api/health` is unauthenticated** and exposes the camera list. On a
  public domain that is readable by anyone with the URL.

A green health check does NOT mean it works: with no Chromium the bridge
still answers `/api/health` and publishes nothing. Use
`scripts/health_check.py`.

**The gateway cannot follow.** It pulls RTSP off cameras on the LAN, so
the bedroom Tapo stays on box-01. Only Warner cameras can move to cloud.

## Owed, and not done

- **Rotate four credentials** leaked into the 08-25 transcript: dev
  LiveKit API key + secret, node state token, internal service key. All
  dev, none gate patient data. Deferred by Aryaman, not resolved.
- **web-app privacy notice** uncommitted — a 700-line legal page swap.
- **camera-app** `AppSettings.swift` points Debug at a LAN IP and says
  `REVERT before PR`; `project.pbxproj` carries a local signing team.
- **Warner** drops resolution once per ~20 min (was 3-4/min) and runs
  8-12 fps against the Tapo's 15. Aryaman accepted both.
- **Safety gate has never fired on a real event.** The test that matters
  is the inverse: BOTH residents in the room, one falls, the alert must
  still fire.
- **Face harvest** has one data point, ever.
- **Sleep** unusable; summariser gated off.

## What tonight actually fixed

Measured, not asserted:

- `carry_seat` had never executed its body since it was written. Identity
  was re-earned by face match every ~40s. Now: 0.8/min → 0.1/min.
- EdgeTAM dropped a seeded track on 2% of carry ticks, the only churn
  source common to both cameras. Renumbers 6→2 and 5→1 per 5 min.
- Painted faces: vetoed, excluded from the pod's people count (re-looks
  34% → 20% of ticks), and no longer blocking the idle step-down.
- The dashboard showed green through a total perception outage;
  `tick_health` now separates an empty room from a dead loop.
