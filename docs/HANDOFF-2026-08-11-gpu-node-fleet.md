# Handoff — GPU nodes as a fleet: what was built, and what testing it costs

Branch: `feat/data-collection-timing-truth`, ~95 commits ahead of `main`.
15 of those are this work. Unit suite green (**1158 passed**). Integration
green as of the last full run (**292 passed, 1 skipped**) with one caveat in
§7.

**You are being handed the testing, not the building.** The code is written
and reviewed; none of it has ever been executed on real hardware. This
document exists so you can size that, not so you can re-read 15 diffs.

---

## 1. What problem this solves

Identity — the thing that decides *which person in the frame is the patient* —
had **no production deployment**. It ran on Aryaman's Mac. The backend reached
it via a single hardcoded `NEMARA_IDENTITY_STATE_URL` that only resolved
because both were the same laptop.

Perception ran on a hand-built RunPod pod that has been lost **twice**:
terminated at $0 balance with its disk, and on 2026-08-10 unable to resume
because RunPod had reallocated its GPU. Savings plans are a rate discount, not
a capacity reservation, so money does not prevent this.

The ask was 5-10+ homes. Two things had to change: the node had to become
reproducible, and the backend had to be able to address more than one of them.

## 2. What it is now, in one page

**A node is an image.** One container = perception + N identity processes,
serving up to K homes. Built from a git tag by CI, released with
`scripts/pod.sh release <tag>`. A lost node is replaced, never repaired.

Perception and identity are **colocated deliberately**: identity ships ~13MB
per cycle to perception, RunPod's inter-pod link is 100 Mbps (~1s/batch), and
perception has no authentication so it must never bind a network interface.

**Nodes ask for work; nobody configures them.** Every 30s a node posts
`POST /v1/identity/roster` with `{node_id, base_url, capacity}` and is told
which rooms to run **and on which port each**. It reconciles: start what is
new, stop what is gone, leave the rest alone. The poll is also the heartbeat.

That single mechanism gives discovery, scale-up, failover and rebalance. There
is no orchestrator and no config file listing homes.

**The node derives its own identity at boot**, because neither value exists at
create time — the pod id comes back from the API call that creates it, and the
tailnet name is assigned at login:

- `NEMARA_NODE_ID` ← `RUNPOD_POD_ID` (stable across restarts of the same pod,
  so a restart re-claims its own homes instead of waiting out the 90s window)
- `NEMARA_NODE_BASE_URL` ← the name **tailscaled reports**, not `TS_HOSTNAME`
  (Tailscale suffixes a name already taken, which happens whenever a replaced
  node's record lingers — assuming it would register the *dead* node's address)

A node that cannot derive a reachable address **exits** rather than
registering. Registering unreachable is worse than absent: it claims homes up
to capacity, serves none, and the backend fails open over it in silence.

**The backend routes per patient.** `IdentityRouter` resolves
`patient_id → identity_nodes.base_url : patients.identity_port`, one httpx pool
per node, 15s route cache. It replaced a single shared client.

**Both ends are on a tailnet.** Identity ports are not public. Railway reaches
them through tailscaled's outbound HTTP proxy, scoped to the identity client
only.

**A watchdog pages a human**, because every failure here is silent by design.

### Files that matter

| | |
|---|---|
| `gpu/Dockerfile` | the node image |
| `gpu/deploy/entrypoint.sh` | boot: checkpoints → tailscale → perception → roster |
| `gpu/deploy/roster_agent.sh` | poll + reconcile rooms |
| `src/api/identity_roster.py` | `POST /v1/identity/roster` |
| `src/services/identity/identity_router.py` | per-patient routing |
| `src/services/identity/fleet_watch.py` | dead-man alert |
| `scripts/pod.sh` | `release` / `up` / `down` |
| `scripts/apply_identity_nodes_delta.sql` | schema |
| `infra/tailscale-acl.hujson` | network boundary |
| `docs/runbooks/pod.md` | operator runbook (rewritten) |

## 3. Why the failures here are invisible — read this before testing

**Identity fails OPEN.** When a node is unreachable the backend returns
`IdentityResult.empty()` and carries on. Safety detectors (fall, choking) are
identity-agnostic and keep firing; `/health` stays green. What stops is
patient-scoped **routine** work — medication, drinking, TV, sleep and activity
minutes — which fails quiet and does not retry.

So **"it looks fine" is not a test result here.** A completely dead fleet and
a healthy one differ only in the absence of routine detections. This is why the
2026-07-26 identity OOM ran for hours unnoticed, and it is the single most
important thing to hold onto when designing the test plan: you must assert on
*positive* identity signal, never on the absence of errors.

The specific ways this bit during the build, all now fixed, are the shape of
what to watch for:

- the roster addressed **nodes** while rooms live on **ports** → every home
  after the first on each node was silently starved
- `/ready` probed one URL → would report `ok` with every other node dead
- `pod.sh release` forwarded no roster env → every node it created booted the
  static fallback, which cannot be scaled or failed over
- the GPU image never installed tailscale at all, while its entrypoint called
  `tailscaled`

## 4. Testing lift — honest sizing

Roughly **one focused day** to first green, plus a soak. Sequenced, because
each stage's failures are cheaper to find than the next stage's.

### Stage 0 — CI build (~30 min, no hardware, no risk)

```bash
git tag gpu-v1 && git push origin gpu-v1
```

Tag-triggered workflows run the workflow file **at that tag's commit**, so this
does **not** require merging to `main`. That matters: `main` is developmental
and Railway deliberately does not auto-deploy from it.

This is the first time any of it is compiled. Expect failures here; the
Dockerfile has build-time assertions for both upstream patches, the tailscale
binaries, and smoke imports, so a break should be loud and specific.

**Cost if it fails: zero.** Do this first.

### Stage 1 — one node, one home (~2h, ~$0.80/hr)

```bash
scripts/pod.sh release gpu-v1   # → new pod id, put it in .pod.env
scripts/pod.sh up               # endpoint + host key re-pin
```

Assert, in order — each depends on the last:

1. `pod.sh ssh 'tail /persist/state/logs/checkpoints.log'` — first boot pulls
   ~3.8GB onto the network volume and is slow
2. perception answers on **loopback only**; `/health` reports `face_provider`
   as CUDA, not CPU (the onnxruntime-pin regression)
3. `tailscale.log` shows an `Online` self, and `roster.log` shows the derived
   node id and base_url
4. the roster row exists and a home is claimed **with a port**
5. `GET /ready` on the backend reports `identity: ok`, through the tailnet
6. **a real verdict**: a `PATIENT` label bound to a track. This is the only
   assertion that proves the whole chain; everything above can pass with
   identity producing nothing.

### Stage 2 — the number that matters (~4h, one node)

**Measure K.** This sets homes-per-node and therefore $/home, and it is the
single largest unknown in the build (§6).

Watch **RSS per identity process**, not GPU utilisation. See §6 for why.

Then the `NEMARA_PERCEPTION_CONCURRENCY=2` A/B that commit `86b88dd` left
explicitly "NOT VALIDATED FOR LATENCY".

### Stage 3 — scale + failover (~2h, two nodes, ~$1.60/hr)

- release a second node → it claims unassigned homes on its first poll
- kill node A → within ~90s node B claims its homes; `FleetWatch` pages
  **warning** for the node and **critical** only if the homes stay unplaced
- **measure the cold-start seat re-acquire time.** The 30-60s figure in the
  docs is for a *patient changing rooms*, not a node starting with no seat at
  all. That number sets the right staleness threshold and the true cost of a
  failover.

### Stage 4 — production

DB delta **before** backend deploy, per `docs/runbooks/` deploy order. The
delta is idempotent and additive (`nemara-backend/scripts/apply_identity_nodes_delta.sql`):
one new table, two nullable columns on `patients`, one partial unique index.
It does not touch existing rows.

## 5. Sequencing with your other work

You are on the same branch, working on OpenRouter removal and the Insights
read API. Relevant overlaps:

- **`src/settings.py`** — both of us added fields. Already merged cleanly, but
  it is the likely conflict point.
- **`test/integration/`** — see §7. Our test isolation strategies must agree
  or we will corrupt each other's fixtures.
- **Nothing else touches the same runtime paths.** Routing is VLM provider
  selection; this is identity. They meet only in `settings`.

**Sequencing recommendation:** Stage 0 costs nothing and needs no hardware —
run it in parallel with whatever you are doing. Stages 1-3 want a quiet branch,
because a failed assertion that turns out to be someone else's concurrent
commit is the most expensive kind of debugging.

## 6. Where we took real liberties

| Claim | Status |
|---|---|
| a pass costs 0.6-1.3s | **measured** (`86b88dd`), at low load |
| A40 at 17% SM | **measured**, one node, light |
| concurrency=2 → 1.6-1.8× | **measured for throughput**; same commit says NOT validated for latency |
| **K ≈ 4 homes/node** | **never run.** 1s ÷ 5s cadence, arithmetic only |
| **K ≈ 8 at concurrency=2** | extrapolation on an extrapolation |
| **65% of homes occupied at peak** | **invented.** Every $/home figure inherits it |
| 30-60s re-ground after a move | documented for a *patient changing rooms* — wrong scenario |
| 90s staleness threshold | 3× the poll interval, arbitrary, never tested |

### The one that should shape your test plan

**More than one identity process has never run on one machine.** Identity has
only ever run on a Mac, one room, one process. The capacity model is built on
*perception's GPU duty cycle* — but the thing that has actually killed this
system was *identity*, an OOM.

Each identity process holds its own OSNet model, torch runtime, EdgeTAM
sessions and ledger. Four of those alongside perception is a memory profile
nobody has observed. **We modelled the resource that has never failed and
guessed at the one that has.**

Practical consequence: K may be bound by RAM, not GPU, and may be lower than 4.
That changes the money, not the architecture — the roster does not care what K
is; it is a number the node reports in `NEMARA_NODE_CAPACITY`.

## 7. Known-bad and unverified — do not rediscover these

1. **`test_a_brand_new_home_is_not_paged_immediately` has not been re-run.**
   It was rewritten from absolute counts to a before/after delta and never
   executed, because the database was frozen (see 2 below). Correct by
   construction; wants one run.

2. **I wiped the local dev database.** An earlier version of
   `test/integration/test_identity_fleet_watch.py` used
   `DELETE FROM public.patients` as setup. It committed and cascaded:
   patients, cameras, analyses, events and detectors are all 0. Caregivers and
   gateways survived. **Production was never touched.** Both that file and
   `test_identity_roster.py` now scope cleanup to rows the tests create, using
   a sentinel node to park foreign rows.
   **Rule for both of us: never delete rows a test did not create.**
   The local rig needs its camera re-paired before any local end-to-end test.

3. **Two commits have mismatched messages.** My `git add -A` crossed with your
   staging:
   - `5a7865c` (*"scope test cleanup"*) contains my 2 test files **plus your**
     OpenRouter removal
   - `acb33b0` (*"delete the OpenRouter provider entirely"*) contains **my**
     four superseded script deletions

   Nothing is lost and the tree is clean. Not rewritten because the branch has
   concurrent writers. Untangle only when nobody else is committing.

4. **`pod.sh release` is unrun.** It POSTs to `rest.runpod.io/v1/pods` with a
   `volumeMountPath` of `/persist` and a private-GHCR `containerRegistryAuthId`.
   The GHCR-credentials-on-RunPod path has never been verified.

5. **Nothing has been compiled.** The image has never been built.

## 8. Two traps that have each cost a machine

- **A stopped pod may not resume.** RunPod reallocates the GPU. The recovery
  is not "wait", it is "release a new node".
- **A pod without a network volume is TERMINATED, not stopped, at $0
  balance,** and the disk goes with it. `pod.sh release` refuses without
  `RUNPOD_NETWORK_VOLUME_ID` for exactly this reason.

## 9. Security invariants — do not regress while testing

- **Perception has no auth by default and must stay on loopback.**
  `PERCEPTION_HOST` exists so exposing it is a deliberate, reviewable act.
- **`NEMARA_IDENTITY_STATE_TOKEN` must be set.** Empty disables the check on
  `/state`, `/probes` and `/stream` — live annotated patient video. The
  entrypoint refuses to start without it.
- **The tailnet ACL grants a port RANGE** (`tag:gpu:8012-8111`). Granting only
  `:8012` lets one home through and denies the rest *at the tailnet*, which the
  backend cannot distinguish from "service down" — so it fails open and those
  homes silently have no identity.
- **Secrets stay out of argv.** `ps` is readable by any local process.
  `.pod.env*` is gitignored **including** the `.pod.env.bak-*` files `pod.sh up`
  writes, which hold the RunPod key, DB URL, LiveKit secret and HF token.
- A Tailscale auth key was pasted into a chat transcript during the build.
  **It should be revoked and reissued** before any real deployment.
