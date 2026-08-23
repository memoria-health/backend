# Soak plan — sensor/account-room migration

Two sensors, two patients, two wifi networks, one account. Validates
`feat/camera-ownership` (backend), `feat/bridge-core-vendor-split`
(bridge) and `feat/detector-polish` (iOS) against real hardware.

**Assert, do not watch.** `scripts/soak_check.py` checks every invariant
this migration introduced and exits non-zero on failure:

```bash
watch -n 60 '.venv/bin/python scripts/soak_check.py'
```

`SKIP` is not a pass — the precondition does not exist yet. Person
tracking is on the soak dashboard at **:8010**.

---

## Where things stand

| | |
|---|---|
| Tapo (`Aryaman Bedroom`) | online, publishing into `account_a30c505e…`, privacy ON |
| Mini-PC gateway | polling the public URL, following the account room |
| GPU node `7yettlzalhkkl0` | running the reshard, serving 1/1 ports |
| Patients | Aryaman Darda, Margaret Ellis (+ your father, to add) |
| Warner bridge sensor | not provisioned yet |

The gateway stays as it is. It re-reads its room from the roster and
never constructs one, so it follows the migration untouched — it only
has to not interfere.

---

## Phase 1 — single Tapo, single patient (running now)

Baseline. Leave it for an hour before adding anything, so a later failure
has something to be compared against.

| Watch | Passes when |
|---|---|
| `soak_check` | 0 failed |
| :8010 | tracks and a seat appear when you are in frame |
| App | live tile plays; audio kill persists when toggled |

**Do once:** flip `audio_capture_enabled` off in the app, confirm the
gateway log says `audio=False` within ~5s, flip it back. That switch
returned 200 and changed nothing until yesterday.

**Do once:** restart the backend (`make backend-up`). Both sensors must
come back online. Until yesterday a restart spawned no worker for an
account room and then swept every sensor to `online=false`.

---

## Phase 2 — add the Warner sensor

Order matters. Provision it **before** assigning anyone, because
unassigned-and-streaming is the state this migration created and the one
nothing had ever run in.

1. App → **Sensors → Add a sensor** → read the 6-digit code.
2. Bridge page → pick the Warner camera → enter the code → provision.
3. **Before assigning**: `soak_check` shows it online and the app lists it
   as unassigned. *This is the installer's moment — the whole point.*
4. Assign it to your father.

| Check | Why it matters |
|---|---|
| code expires in 5 min | binding used to take an email, which is on the invoice |
| second sensor needs no second code | a bound service provisions freely |
| a used code is refused | single-use, or reading it aloud once binds forever |
| `fleet: a bridge is enrolled multi-tenant` | the bridge declared `service_kind`; without it, provisioning for a second customer silently drops the first |
| two sensors, one account, one room | distinct identity ports, both answering `/state` |

---

## Phase 3 — the shared sensor, which is where attribution is decided

Assign the **Tapo** to both your father and Aryaman Darda. One sensor,
two people: this is what turns the attribution rules on.

`coverage: a shared sensor exists` flips SKIP → PASS.

| Scenario | Expected |
|---|---|
| you alone in frame, identity confirms | event on you only |
| father alone, identity confirms | event on him only |
| both in frame | each detector fires for whoever identity located |
| nobody identifiable (back turned, dark) | **no** care/behaviour event for either |
| same, but a **safety** detector fires | event for **both** |

The last two are the agreed rule: a fall nobody can attribute is still a
fall; a care record filed against the wrong resident is not recoverable.

Then: **delete a patient.** The shared sensor must survive with the other
patient still covered. Until yesterday this destroyed the sensor row, its
coverage, its detector links and its ingest token.

---

## Phase 4 — sleep and care/behaviour

The two categories never fully tested, and not by accident — both are
`subject_scope="patient"` and non-safety, which is exactly the class the
gate suppresses when identity cannot say who is in frame. They prove the
rules rather than merely running.

**Sleep** needs a bedroom sensor and a night. It is *sustained*, so it
exercises episode dedup, which reads `analyses.patient_id`. On a shared
sensor the second patient previously had no analyses rows at all, so
boundaries were never found and the event re-fired every tick. **Watch
for one sleep event per episode, not one per tick** — that single number
is the tell.

**Care / behaviour** — run the matrix above. Medication and drinking are
the easiest to stage deliberately.

**Identity off** (stop the node): on a *single-patient* sensor, safety
AND non-safety must keep firing. The gate only arms where coverage is
greater than one. If everything goes quiet, the gate is wrong in the
direction that takes detection down with the node.

---

## What each change is validated by

| Change | Observable |
|---|---|
| Account rooms | Tapo followed with no gateway restart ✅ already seen |
| Unassigned sensors stream | Warner publishes before any patient exists |
| No primary patient | assigning two does not move the stream |
| Audio kill on the sensor | PATCH persists, reaches the publisher |
| Sensor survives patient delete | delete one, sensor stays |
| Identity per sensor | 2 sensors → 2 ports, both answering |
| `PatientPresence` per patient | you confirmed on the Tapo does not suppress your father's detectors on the Warner |
| Per-patient attribution | the Phase 3 matrix |
| Bridge multi-tenancy | `multi_tenant` true on the gateways row |
| One-step enrolment | the Phase 2 code checks |
| `worker.py` room shapes | backend restart, both sensors return |

---

## Still manual — flag for production

| Step | Why | Fix |
|---|---|---|
| **Warner device selection** | the vendor's camera list only exists inside their JS SDK, in a browser | the real question is whether an installer signs into a *vendor* account per install, or we hold one across customers. Product decision, not scripting |
| **Node code deploy** | `NEMARA_DEV_CODE=1` + dev-code overlay is a soak tool; production is an immutable tagged image | before prod: `git tag gpu-vN` → CI → `pod.sh release` |
| **`.pod.env` goes stale** | RunPod reassigns the endpoint on every restart — it did again today, the node moved to a new tailnet IP | `pod.sh up` rediscovers; `push_node_code.sh` should read the reader endpoint from `.pod.env` instead of taking it as an argument |
| **Mini-PC restart** | `sudo` needs an interactive password | the unit is `Restart=always` and runs as `potaton00b`, so `kill <pid>` is a clean restart |

---

## Known hazards during the run

- **DHCP churn.** Your Mac's LAN IP moved three times today and broke the
  gateway, two test runs and the Supabase config. The gateway now points
  at the public URL and `.env` at loopback, so both are immune — but iOS
  `LocalAppConfig.swift` still pins a LAN IP and will need the current one
  before you run on device.
- **The dev DB is shared with the live fleet.** Node rows appear and go
  stale mid-run. The fleet-watch baselines are measured after the fact now,
  so this no longer causes false failures.
- **Test pollution.** The suite creates `node-*` rows in the same
  `identity_nodes` table the soak reads. `soak_check` ignores them; a
  human reading the table directly should too.

---

## Aborting

`desired_state='disconnected'` on both sensors halts publishing without
deleting anything — coverage and identity slots survive for the next run.
Nothing here touches production.
