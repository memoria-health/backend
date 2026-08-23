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
| GPU node `7yettlzalhkkl0` | 8012 + 8013 both answering (2026-08-24) |
| Patients | Aryaman Darda, Margaret Ellis (+ your father, to add) |
| Warner bridge sensor | provisioned, publishing 2304x1296 H265 as `cam_c51733e6` |

### Before Phase 3 — two things must land first

**1. Merge `fix/analyses-patient-keying` and apply its delta.** Phase 3
is the shared-sensor configuration, and until that branch is in, the
second covered patient has NO observations at all — `analyses.patient_id`
is the `cameras.patient_id` shim, so every row belongs to whichever
resident sorted first. Running Phase 3 without it does not test
attribution; it tests a known-broken join, and every "no event for the
second patient" result will be indistinguishable from the gate working.

```bash
psql "$DB" -f scripts/deltas/apply_analyses_coverage_delta.sql   # DB first
```

**2. Revert the two soak overrides in `.env`.**
`NEMARA_SENSITIVITY_SCREENING_ENABLED=false` and
`NEMARA_SLEEP_CLIPS_DIR` are both dev instruments that write unscreened
frames. They are local-only and ignored outside env=local/test, but they
must be off before this stack points at anyone but its operator.

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
exercises episode dedup, which used to read `analyses.patient_id` — on a
shared sensor the second patient had no analyses rows at all, so
boundaries were never found and the event re-fired every tick.
`fix/analyses-patient-keying` moves that lookback onto coverage. **Watch
for one sleep event per episode, not one per tick** — that single number
is the tell, and it is now a real test rather than a known failure.

---

## Phase 5 — the masked capture test

**Why this is its own phase: the first attempt was inconclusive.** The
night of 2026-08-23 recorded 496 objects to B2, of which three sessions
carried `masked=True` and most did not, and the one window that would
have settled the question was not captured at all — a 24-minute gap sits
exactly across it. Two sessions were pulled to `out/verify-masking/` for
a side-by-side and neither answers whether the mask fires when it
matters. So the pipeline is currently **unverified in both directions**:
we have not seen it censor something sensitive, and we have not
established what decides `masked`.

Run it deliberately instead of hoping the night produces the case.

| Step | Watch |
|---|---|
| 1. `NEMARA_SENSITIVITY_SCREENING_ENABLED` back to its default | it must be ON for this to mean anything |
| 2. Bedroom camera, `privacy_enabled` on, data collection ON | the configuration a real household would run |
| 3. Stage a sensitive moment on purpose, in frame, ~1 min | this is the case that has never been observed working |
| 4. Stage a plainly non-sensitive minute right after | the control — a mask that fires on everything is not a mask |
| 5. Pull BOTH sessions from B2 and watch them | pixelation present in (3), absent in (4) |

Then answer the question the last run could not:

- **What decides `masked`?** Three of ~10 sessions had it true. Until
  that is a rule and not a coincidence, no household can be told what
  their footage does.
- **Does the screen fail closed in the dark?** `.env` records that it
  censors full-frame on any doubt, and that in a dark bedroom this blurs
  essentially everything. If that is what happens, collection at night is
  paying for storage of a grey rectangle.
- **What is the gap?** 02:40:24 to 03:05:38 with the recorder armed and a
  person present. A recorder that stops for 24 minutes without saying so
  is a bigger finding than the mask.

**Do not delete the 496 existing objects until this runs** — they are the
only evidence of what the pipeline did unsupervised, including the ~10
unmasked bedroom sessions. Deciding their fate is Aryaman's call and it
should be made after the mechanism is understood, not before.

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
| **The overlay only works if the agent is newer than it** | `roster_agent.sh` applies the dev-code overlay, but the RUNNING agent started 07:56 and the overlay block was added to the file at 11:38 — bash reads a script lazily, so the live agent never had that code. Every `push_node_code.sh` since has been a silent no-op, and 2026-08-24's identity fix had to be `cp`'d into `/workspace` by hand. There is no log line for a push that did nothing | after any change to `gpu/deploy/`, restart the container (`pod.sh up`) before trusting a push. Better: have the agent log its own mtime at boot so "the agent predates the overlay" is visible |
| **Orphaned supervisors** | the same stale agent predates the `_stop_room` fix, so re-provisioning a camera left the OLD supervisor subshell running and un-stoppable (its pid was no longer in the run state). Two supervisors for `c51733e6` each started an identity process for the same shard, which then evicted each other on `DuplicateIdentity` | fixed in the repo; lands on the next container restart. Until then check `ps -eo pid,ppid,args \| grep roster_agent` — one child per camera, no more |
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
