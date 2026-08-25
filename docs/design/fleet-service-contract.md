# The fleet service contract

How anything that is not the backend talks TO the backend: cameras,
gateways, camera-vendor bridges, GPU identity nodes, and whatever comes
next.

This is not a new invention. The same shape has been arrived at
independently three times in this codebase, each time with a different
piece missing. Writing it down is how the next integration stops
rediscovering it badly.

## What already exists, and what each one got right

| service | endpoint | auth | poll = heartbeat | reports its own health |
|---|---|---|---|---|
| iPhone camera | `GET /v1/cameras/me/desired-state` (1s) | bearer → `cameras.api_key_hash` | yes, per camera | yes (`app_present`) |
| gateway (Pi/Tapo) | `GET /v1/gateway/roster` (3s) | bearer → `gateways.token_hash` | yes, gateway + each camera | partial |
| Warner bridge | same gateway roster | same | **per process, not per camera** | **no** |
| GPU identity node | `POST /v1/identity/roster` (30s) | `x-internal-key` shared secret | yes, per node | partial (`/state` pulled by backend) |

Three things they agree on, and those three are the contract:

1. **The poll IS the heartbeat.** No separate liveness endpoint. Both
   the gateway roster and the identity roster say so in their own
   docstrings. It works because a service that stops working stops
   polling, and there is no second mechanism to disagree with.
2. **The response IS the instruction.** The service holds no config of
   its own; it reconciles to whatever came back. Nothing to drift.
3. **Liveness is stamped as a side effect** of the poll, in the same
   statement, so it cannot be forgotten or fail separately.

## The four rules

### 1. Per-instance credentials, hashed at rest

The camera and gateway paths mint a per-instance token and store only
`sha256`. The identity node uses `NEMARA_INTERNAL_SERVICE_KEY`, one
static secret shared by every node — which is why a dev node can claim
production homes (checklist B.15), and why rotating it requires moving
both sides at once.

**Rule:** every instance gets its own credential, the backend stores a
hash, comparison is constant-time. No shared static secrets. Revoking
one instance never touches another.

### 2. Telemetry is a PRECONDITION of instruction, not a side channel

This is the rule that would have prevented the bridge's failures.

The bridge polls, receives `desired_state`, `privacy_enabled`,
`audio_capture_enabled` and `online` — and reports nothing back, and
until 2026-08-20 acted on none of them. Nothing detected this, because
reporting was optional and acting was unverified.

**Rule:** the poll is a `POST` whose body carries the service's health
and the state of every entity it owns. A poll without a report is
rejected. You cannot receive instructions without saying what you did
with the last ones.

That single change makes telemetry loss impossible rather than
merely discouraged: a service that stops reporting stops being told
what to do, and goes visibly stale within one poll interval.

### 3. Health is PER ENTITY, never per process

The structural bug in the bridge. One process polls for N cameras, so
`last_heartbeat_at` is stamped for all of them while the bridge lives —
even for a camera whose worker died ten minutes ago. The iPhone path
does not have this problem only because one camera is one device is one
poller.

**Rule:** the report body carries a row per entity. A supervisor that
owns many entities must say something about each, every poll. "I am
alive" is not a statement about the things I am responsible for.

### 4. Capabilities are declared, never assumed

`app_present` defaults to `true` for anything that is not an iPhone, so
a Pi camera "can never show app closed" — the column lies rather than
abstains. The app then cannot distinguish *not supported* from *fine*.

**Rule:** each service declares what it can do. The backend and the app
branch on the declared set, and a capability that is absent renders as
absent, not as healthy. A vendor without a battery reports no battery;
it does not report a full one.

## The shape

```
POST /v1/fleet/poll
Authorization: Bearer <per-instance token>

{
  "service": {
    "kind": "camera-bridge",           # camera | gateway | camera-bridge | identity-node
    "instance_id": "bridge-warner-01",
    "version": "1.4.2",
    "contract": "fleet-v1",            # bump when a consumer would misread
    "capabilities": ["talk", "privacy_sever", "audio_kill"]
  },
  "entities": [                        # RULE 3 — one row per owned thing
    {
      "id": "<camera uuid>",
      "state": "publishing",           # vendor-agnostic vocabulary
      "healthy": true,
      "detail": "",
      "last_frame_age_s": 0.4,
      "restarts": 0,
      "applied": {                     # RULE 2 — what we did with the
        "desired_state": "connected",  # last instruction, echoed back
        "audio_capture_enabled": false,
        "privacy_enabled": true
      },
      "battery_percent": null          # RULE 4 — null means "cannot know"
    }
  ]
}
```

Response — the instruction, and nothing the service must remember:

```
{
  "contract": "fleet-v1",
  "entities": [
    { "id": "...", "desired_state": "connected",
      "audio_capture_enabled": false, "privacy_enabled": true,
      "online": true }                 # backend's own view, fed back
  ]
}
```

`online` fed back is the 2026-07-29 wifi-blip lesson: the gateway's own
state all read healthy while zero media left the box. A service cannot
see itself; the backend can. Feeding the subscriber's view back is what
lets a supervisor notice it is a zombie.

## Why `applied` matters

It is the difference between "we sent the instruction" and "the
instruction took effect". A caregiver pressing disconnect wrote a
database row for months while the camera kept publishing, and every
dashboard showed the row. Echoing what was applied makes divergence a
value the backend can compare and alarm on, rather than something a
human notices eventually.

## Migration

Nothing is rewritten at once. The existing endpoints keep working; each
service moves when it is next touched.

1. Define `fleet-v1` and the shared client. Contract-version guard
   copied from `gpu/identity/contract.py`, which already treats a
   version mismatch as "no verdict" rather than a silent misparse.
2. Bridge first — it has the most gaps and is not yet deployed, so it
   is the cheapest to change and the most in need.
3. Identity node next, which also retires the shared
   `NEMARA_INTERNAL_SERVICE_KEY` (B.15).
4. Gateway and iPhone last. Both work today; move them for uniformity,
   not urgency.

## What this does not cover

Backend → service calls (the backend pulling `/state` from an identity
node, or minting a LiveKit token). That direction has its own contract
in `gpu/identity/contract.py` and is deliberately separate: it is a
read, it fails open, and conflating the two would make a fetch failure
look like a heartbeat failure.
