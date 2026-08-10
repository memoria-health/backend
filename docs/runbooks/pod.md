# Pod runbook — spin up / down / deploy the perception GPU

The EdgeTAM-continuous `sam_service` runs on a single RunPod GPU (an A40
is plenty; ~$0.44/hr). The backend reaches it over an SSH tunnel. This is
how to operate it. All commands go through `scripts/pod.sh`, which reads
secrets from a gitignored `.pod.env`.

## 0. One-time: credentials

```bash
cp .pod.env.example .pod.env      # then fill in (see the file's comments)
mkdir -p .pod                     # holds the SSH key + pinned known_hosts
# put your RunPod SSH private key at .pod/runpod_key  (chmod 600)
```

`.pod.env` and `.pod/` are gitignored — never commit them. `.pod.env`
holds the RunPod API key and the pod's direct-TCP endpoint.

## 1. Daily cost control — stop when idle, and the catch

```bash
scripts/pod.sh up        # start + rediscover endpoint + re-pin + wait healthy
scripts/pod.sh down      # billing off (desiredStatus -> EXITED)
scripts/pod.sh status    # check desiredStatus + $/hr
```

Always use `up`, never bare `start`. Two traps it handles that manual
bring-up does not:

**⚠️ Stopping is NOT reliably reversible.** A stopped Secure Cloud pod
holds **no GPU reservation**. The GPU goes back to the host machine, and
if the host reallocates it, `podResume` fails with *"There are not enough
free GPUs on the host machine to start this pod"* and the pod stays
EXITED until one frees up **on that same physical host**. Hit for real on
2026-08-10: the pod would not come back, with no ETA. `up` retries for
~10 min (`RESUME_TRIES=90` for longer) and prints the real reason.
If it stays stuck, rebuild on a different host with §3 (~40 min).
So: stopping overnight to save $0.44/hr is a genuine gamble against
needing the pod back on a schedule. Weigh ~$320/mo against that.

**The endpoint moves.** RunPod assigns the direct-TCP `ip:port` only when
the pod reaches RUNNING (`runtime` is `null` while stopped) and reassigns
it across restarts. So `POD_HOST`/`POD_PORT` in `.pod.env` are stale the
moment you stop, and the pinned host key no longer matches — ssh then
refuses, correctly, because RunPod recycles those endpoints across
tenants. `up` rediscovers the endpoint from the API, rewrites `.pod.env`
(backing it up) and re-scans the host key, so nobody has to remember any
of this.

Storage: the pod now has a **20 GB network volume** (`volumeInGb=20`)
alongside the 70 GB container disk; the volume survives independently.
Give it ~60–90 s after `up` before the server answers.

## 2. Deploy code changes

The pod code lives in `gpu/perception/` (`continuous.py` / `server.py` /
`sam_track.py` / `face_worker.py` / `person_trigger.py`; renamed from
`sam_service/` on 2026-07-26). After editing:

```bash
scripts/pod.sh deploy    # ships gpu/perception/ + relaunches the supervisor
scripts/pod.sh health    # confirm ticks resume (mode=ground/carry, drift=N)
scripts/pod.sh logs 80   # tail the server log
```

`deploy` runs `scripts/deploy_continuous_remote.sh` on the pod: it kills
any old supervisor (including the disabled legacy `start_continuous.sh`),
rewrites the file-based supervisor `/workspace/start_sam_supervisor.sh`,
and launches `uvicorn perception.server:app` on the pod's **system
python** (`/usr/bin/python3 -m uvicorn` — the old edgetam `.venv` no
longer exists; one system env hosts sam2 + sam3 + insightface).

To watch the mask during a verification run, uncomment the
`NEMARA_EDGE_DEBUG_DIR` line in `deploy_continuous_remote.sh`, deploy, do
the walk-out, pull `/workspace/edge_debug/`, then **redeploy with it off**.

## 3. Rebuild from scratch (new pod, or disk reclaimed)

> Battle-tested for real on 2026-07-25, after RunPod TERMINATED the
> original pod (credits hit zero → pod AND disk deleted, not stopped).
> Total rebuild time ~40 min. Everything below is reproducible from
> this repo — nothing may ever live only on the pod disk again.

1. Create an A40 (or better) **On-Demand** pod (Spot/Community can be
   reclaimed mid-soak). Template "PyTorch 2.4 / CUDA 12.4", ~70 GB
   container disk, port 22/tcp + public IP. Scriptable via the REST
   API (`RUNPOD_API_KEY` in `.pod.env`; pass our SSH public key as the
   `PUBLIC_KEY` env so access works from birth):
   `POST https://rest.runpod.io/v1/pods` — see git history 2026-07-25
   for the exact JSON body.
2. Capture the endpoint into `.pod.env` (`POD_HOST`, `POD_PORT`,
   `POD_ID`) and **pin the host key**:
   ```bash
   ssh-keyscan -p "$POD_PORT" "$POD_HOST" > .pod/pod_known_hosts
   ```
3. One-shot bootstrap — repos, pip layer, ALL source patches, ALL
   checkpoints, smoke imports (idempotent, safe to re-run):
   ```bash
   scripts/pod.sh ssh "export HUGGINGFACE_TOKEN=<token>; bash -s" \
     < scripts/pod_bootstrap.sh
   ```
   The token is needed for the **gated** `facebook/sam3.1` repo
   (source of `sam3.1_multiplex.pt`); one lives as
   `NEMARA_HF_ACCESS_TOKEN` in the backend `.env`. The script also
   applies, with rationale inline:
   - the **EdgeTAM Perceiver reshape patch** (multi-object crash —
     `.expand().view()` on non-contiguous tensors 500s the instant a
     2nd person is tracked; invisible to solo testing),
   - the **SAM3 bool-argsort patch** (torch 2.4 cannot sort bool on
     CUDA; the old pod ran torch 2.8-dev where it works),
   - the **onnxruntime-gpu pin (1.20.1)** — 1.28+ wants CUDA 13; and
     the supervisor must export `LD_LIBRARY_PATH` to torch's bundled
     nvidia cudnn/cublas wheels or faces silently fall back to
     `CPUExecutionProvider` (~10x slower probes; check `/health`).
4. Ship + launch: `scripts/pod.sh deploy`. Mac-side services
   (identity, dashboard) reconnect on their own — do not restart them.
5. Retarget the Mac tunnel at the new endpoint and confirm
   `scripts/pod.sh health` + new `identity.batch` lines.

**Operational traps learned 2026-07-24/25 (read before touching the pod):**
- **Exactly ONE supervisor.** A legacy `start_continuous.sh` and the
  deploy-script supervisor once raced for port 8000 for hours — every
  deploy "worked" (new code) while env changes silently never applied
  (old winner's env). The deploy script now disables the legacy file.
- **`pkill -f` self-match.** Ad-hoc `pkill -f 'uvicorn sam_service'`
  from an ssh one-liner matches the ssh session's OWN command line and
  kills it (looks like a hang / empty output / phantom short-lived
  processes). Always use the bracket trick (`sam_servic[e]`) and never
  combine a pkill with a command containing the literal pattern.
- **Launch supervisors as FILES, not inline heredocs** — two layers of
  ssh quoting mangled inline scripts twice. Ship
  `start_sam_supervisor.sh` via `ssh "cat > file" < localfile`
  (md5-verify), then `setsid --fork` it in a session with NO kill
  patterns.
- **A backgrounded remote process holding the session's pipes makes
  ssh "hang"** — that's the launch *succeeding* while sshd waits on
  fds; redirect all three (`</dev/null >/dev/null 2>&1`).
- `/proc/PID/environ` shows the INITIAL env only — exports made inside
  a running script are invisible there; verify env via the *server's*
  behavior (`/health` face_provider), not the supervisor's environ.

## 4. Backend tunnel

The backend connects to the pod's `127.0.0.1:8000` over an autossh-style
tunnel bound to host-loopback (so the rest of the LAN cannot reach patient
frames). See the backend's tunnel helper; the security rationale is the
same host-key pinning as above.

## 5. Health signals

- `scripts/pod.sh health` → recent `continuous.tick` lines. Healthy:
  `mode` alternates ground/carry, `seeded` and `live` track occupancy,
  `drift=0` most ticks (a nonzero `drift` is the guard doing its job).
- `continuous.drift_dropped` lines → the drift guard fired (forced a
  re-ground). A burst on every tick means the thresholds are too tight for
  that install — raise `NEMARA_EDGE_DRIFT_BALLOON`.
- Server not answering after `start` → give it 90 s; then
  `scripts/pod.sh logs 100` for a stack trace.

## 6. Security invariants (do not regress)

- Host key is **pinned** (`StrictHostKeyChecking=yes` +
  `UserKnownHostsFile`), because RunPod recycles direct-TCP endpoints
  across tenants. A new pod = re-scan (`ssh-keyscan`).
- Secrets stay in `.pod.env` / `.pod/` (gitignored). Rotate the RunPod API
  key and SSH key if they were ever pasted into chat/logs.
- `NEMARA_EDGE_DEBUG_DIR` writes raw patient frames to disk — keep it OFF
  in production.
