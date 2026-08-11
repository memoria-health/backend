# GPU node runbook — release, scale, recover

**Rewritten 2026-08-11.** The previous version described a hand-built pet:
`pod_bootstrap.sh` mutated a running container over ssh for ~40 minutes, and
`pod.sh deploy` rsynced perception onto it. That is gone — those scripts were
deleted, and `pod.sh deploy` now refuses and prints the release steps. If you
need what they did, they are in git history.

**A node is now an image.** The artifact is a tag, a lost node is replaced
rather than repaired, and "what is running?" has an answer that is not
"whatever we last scp'd".

> **NOTHING IN THIS FILE HAS BEEN RUN END TO END YET.** The image has never
> been built and no node has been created from it. Every command below is
> written from the source it drives. Treat the first run as the test, and
> fix this file where it is wrong.

---

## What a node is

One pod = **one perception + N identity processes**, serving up to K homes.

Both services live in one container on purpose. Identity sends ~13MB batches
to perception every cycle; over a network that is ~2.2 TB/month/camera, and
RunPod's inter-pod link is 100 Mbps (~1s per batch). Colocated it is a
localhost copy, and **perception never binds a network interface**.

Nodes are interchangeable. All per-room state — EdgeTAM sessions, the seat,
the track ledger — is rebuildable, so a home can move between nodes at the
cost of a re-ground.

```
                    ┌─ perception :8000 (loopback ONLY, no auth by default)
  RunPod pod ───────┤
   (one cell)       ├─ identity :8012  room A ─┐
                    ├─ identity :8013  room B ─┼─ published on the tailnet
                    └─ roster agent ───────────┘
                            │ POST /v1/identity/roster every 30s
                            ▼
                    Railway backend ── reads /state over the tailnet
```

## How a node finds its work

Nobody configures which homes a node serves. It polls
`POST /v1/identity/roster` with its id, address and capacity, and is told
which rooms to run and **on which ports**. The same call is its heartbeat.

That one mechanism gives four things:

| | |
|---|---|
| discovery | backend joins `patients` → `identity_nodes.base_url` |
| scale up | start another node; it drains the unassigned queue on first poll |
| failover | a node that stops polling has its homes released to whoever polls next |
| rebalance | null out `patients.identity_node_id` and it is re-placed |

The node **derives its own identity** at boot, because neither value exists
at create time:

- `NEMARA_NODE_ID` ← `RUNPOD_POD_ID`, which RunPod injects. Stable across
  restarts of the same pod, so a restarting node re-claims its own homes
  instead of waiting out the staleness window while they sit blind.
- `NEMARA_NODE_BASE_URL` ← the name **tailscaled reports**, not `TS_HOSTNAME`.
  Tailscale suffixes a name already in use, which happens whenever a replaced
  node's record lingers. Assuming the name would register an address pointing
  at the dead node.

A node that cannot derive a reachable address **exits** rather than
registering. Registering an unreachable node is worse than being absent: it
claims homes up to its capacity and serves none, and the backend fails open
over it.

---

## 1. Release a new node

```bash
git tag gpu-v3 && git push origin gpu-v3      # CI builds + pushes to GHCR
scripts/pod.sh release gpu-v3                 # create a node from that tag
# → prints the new pod id
# set POD_ID=<new id> in .pod.env, then:
scripts/pod.sh up                             # discover endpoint, re-pin host key
scripts/pod.sh health                         # confirm ticks
curl -H "X-Internal-Key: $KEY" $BACKEND/ready  # confirm the fleet
```

The tag push is what triggers the build. Tag-triggered workflows run the
workflow file **at that tag's commit**, so this does not require merging to
`main` — which matters, because `main` is developmental and Railway
deliberately does not auto-deploy from it.

`release` refuses unless `NEMARA_BACKEND_URL`, `NEMARA_INTERNAL_SERVICE_KEY`
and `TS_AUTHKEY` are set. Without them the entrypoint falls through to a
static room list, which is for a local `docker run`, not for real homes: a
statically-pinned node cannot be scaled or failed over and keeps serving
rooms the backend has since reassigned — two nodes writing seats for one home.

**The old pod is not touched.** Terminate it deliberately once the new one is
verified.

## 2. Add capacity

Release another node. It claims unassigned homes on its first poll. There is
nothing else to do — no config, no restart of anything existing.

**When?** The constraint is **concurrently occupied rooms, not home count** —
an empty room makes zero perception calls. At ~0.6-1.3s per pass on a ~5s
cadence, each *occupied* camera is 12-26% duty, so K ≈ 4 at
`NEMARA_PERCEPTION_CONCURRENCY=1` and K ≈ 8 at 2 (the 1.6-1.8x is measured,
the latency impact is **not** — see `docs/architecture/identity-latency.md`).

The signal already exists: perception returns **429** the moment its
admission gate saturates, and `identity.batch` logs lag per batch. A rising
429 rate on a node is the "add a node" trigger.

## 3. Lose a node

Nothing to do. A node that stops polling has its homes released after
`identity_roster_stale_after_s` (90s), and the next node to poll claims them.

You will be told: `FleetWatch` pages **warning** for the dead node, and
**critical** only if the homes stay unplaced past a 300s grace period — that
second page means no node had spare capacity, and the remedy is §2.

Cost of a move: the receiving node starts with no seat, no EdgeTAM sessions
and an empty ledger, and needs to re-ground. Safety detectors are unaffected
(fail-loud and identity-agnostic), but patient-scoped **routine** detectors —
medication, drinking, TV — fail quiet, and sleep/activity show a gap.

> The ~30-60s re-ground figure is documented for a patient changing rooms,
> **not** for a cold-start node with no prior seat. Measure it during the
> first real failover; that number sets the right staleness threshold.

## 4. Daily cost control

`scripts/pod.sh stop` halts billing, `up` resumes. Two traps, both of which
have cost us a machine:

- **A stopped pod may not resume.** On 2026-08-10 the host had reallocated
  its GPU. Savings plans are a **rate discount, not a capacity reservation**,
  so no amount of money prevents this. The recovery is §1: release a new node.
- **A pod without a network volume is TERMINATED, not stopped, at $0
  balance,** and the disk goes with it. That is how the first pod died.
  `release` refuses without `RUNPOD_NETWORK_VOLUME_ID` for this reason.

The volume also carries the ~3.8GB of checkpoints across node recreation, so
only the first boot pays the download.

## 5. Health signals

| check | what it tells you |
|---|---|
| `GET /ready` on the backend | one probe per registered node, through the tailnet. `degraded_1_of_3_nodes` names the count; failing nodes get their own line. |
| `scripts/pod.sh health` | recent `continuous.tick` lines on that node. |
| `identity.roster` in backend logs | logged only when a claim or a reclaim happens — the only two events that change which machine watches a home. |
| perception 429 rate | admission gate saturated: this node is at capacity. |
| `identity.node_down` / `identity.homes_blind` | FleetWatch. The second means families are receiving no identity right now. |

`/ready` returning `no_nodes_registered` in production means the fleet is
absent, not misconfigured.

## 6. Debugging a node

```bash
scripts/pod.sh ssh 'tail -50 /persist/state/logs/roster.log'      # what it was told
scripts/pod.sh ssh 'tail -50 /persist/state/logs/identity-patient_<uuid>.log'
scripts/pod.sh ssh 'tail -50 /persist/state/logs/perception.log'
scripts/pod.sh ssh 'tail -50 /persist/state/logs/tailscale.log'
scripts/pod.sh ssh 'ls /persist/state/run/'                       # pid + port per room
```

`roster.log` saying **"roster REJECTED this node (HTTP 4xx)"** is
configuration, not a blip, and will not recover on its own — a rotated
internal key (403) or a `base_url` the roster refuses (422). The body names
the field.

`roster.log` saying **"roster unreachable — keeping current rooms"** is the
node deliberately NOT stopping anything on a failed poll. A backend blip must
not blind every home on the node.

To iterate without a release, ssh in and edit under `/workspace` — but the
next node will not have your change. That is the whole point of §1.

## 7. Security invariants (do not regress)

- **Perception has no authentication by default and must stay on loopback.**
  `PERCEPTION_HOST` exists only so exposing it is a deliberate, reviewable
  act. `NEMARA_PERCEPTION_TOKEN` gates `/observe` and `/faces` if it ever
  does leave.
- **`NEMARA_IDENTITY_STATE_TOKEN` must be set.** An empty value disables the
  check on `/state`, `/probes` and `/stream` — live annotated patient video.
  The entrypoint refuses to start without it.
- **The tailnet ACL grants a port RANGE** (`tag:gpu:8012-8111`), because a
  node runs one process per room. Granting only `:8012` would let one home
  through and deny the rest at the tailnet — which the backend cannot
  distinguish from "the service is down", so it fails open and those homes
  silently have no identity.
- **Host key is pinned** (`StrictHostKeyChecking=yes` + `UserKnownHostsFile`),
  because RunPod recycles direct-TCP endpoints across tenants. A new node
  means a re-scan; `pod.sh up` does it.
- **Secrets stay in `.pod.env` / `.pod/`** (gitignored, including the
  `.pod.env.bak-*` files `up` writes). Nothing sensitive goes on a command
  line — argv is visible in `ps`.
- **`NEMARA_EDGE_DEBUG_DIR` writes raw patient frames to disk.** Keep it off.
