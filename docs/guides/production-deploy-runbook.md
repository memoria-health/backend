# Production deploy runbook

How to ship the backend + identity service to production and prove they
work. Written from the first full v2 deploy (2026-08-19); every trap at
the bottom was hit that night.

## The three services and how they connect

```
caregiver apps ──HTTPS──▶ Railway backend ──tailnet (userspace proxy)──▶ RunPod GPU node
                              │                                              │
                              └────────── Supabase prod Postgres ◀──────────┘
```

- **Backend**: Railway, built from `docker/Dockerfile` (see `railway.toml`).
  Joins the tailnet at boot via the entrypoint (userspace tailscaled +
  HTTP proxy on :1055) when `TS_AUTHKEY` is set.
- **Identity node**: RunPod GPU pod running `ghcr.io/memoria-health/memoria-gpu:<tag>`.
  Joins the tailnet at boot, derives its own address, self-registers in
  `identity_nodes` via the backend's roster API, and heartbeats every poll.
- **DB**: Supabase project `zgifmvjnggmgzvtnosnu`. Apply-deltas-only —
  never reset, never migrate destructively (`docs/guides/setup-hosted-stack.md`).

## The secret handshake pairs (all self-minted random strings)

| Secret | Direction | Lives in |
|---|---|---|
| `NEMARA_IDENTITY_STATE_TOKEN` | backend → node (`/state`, `/stream`) | Railway env + node env |
| `NEMARA_INTERNAL_SERVICE_KEY` | node → backend (roster register/heartbeat) | Railway env + node env |
| `TS_AUTHKEY` (tag:backend) | backend joins tailnet | Railway env |
| `TS_AUTHKEY` (fleet/gpu) | node joins tailnet | node env (`.pod.env`) |

**Both directions must be configured or identity fails OPEN and silent**:
the backend looks healthy, ticks run, but every home reads as
unidentified and patient-scoped routine detectors never promote.

## Deploy order (backend)

1. **DB deltas first.** `psql "<prod NEMARA_DB_URL>" -1 -v ON_ERROR_STOP=1 -f scripts/apply_<x>_delta.sql`
   — one transaction per file, chronological order. Never `-f` a delta
   containing `drop column`/`delete from` without reading it first.
2. **Prompt reseeds**: `NEMARA_DB_URL=<prod> PYTHONPATH=. .venv/bin/python scripts/reseed_prompt.py <identifiers…>`
3. **Env vars** on Railway (`railway variables --set … --skip-deploys`
   from the repo dir — the link is per-directory).
4. **Merge to main via PR.** The `main` ruleset requires a review;
   solo-maintainer override is `gh pr merge N --merge --admin`
   (Repository admin is on the ruleset bypass list for exactly this).
5. **`railway up --detach`** from the repo root, then watch
   `railway deployment list` until SUCCESS.
6. Verify (checklist below) before touching the node.

## Deploy order (identity node)

**Always use `scripts/pod.sh release <image-tag>`. Never hand-roll the
pod create.** The script encodes: `cloudType: SECURE` (community hosts
get reclaimed within minutes), the CUDA ≥12.4 host filter, non-interruptible,
the org registry path, the network volume + pinned datacenter, and the
full env contract (it refuses to release without the roster-mode vars).

- Image comes from CI: push a `gpu-v*` tag → `.github/workflows/build-gpu-image.yml`
  builds and pushes `ghcr.io/memoria-health/memoria-gpu:<tag>` (~30 min).
- For a PROD node, run with a prod env file:
  copy `.pod.env`, append overrides for `NEMARA_BACKEND_URL` (Railway
  public URL), `NEMARA_DB_URL`, `NEMARA_IDENTITY_STATE_TOKEN`,
  `NEMARA_LIVEKIT_*` (prod project), `TS_HOSTNAME`, then
  `POD_ENV_FILE=<file> scripts/pod.sh release <tag>`.
- Do NOT set `NEMARA_DEV_CODE=1` on prod — that flag makes the
  entrypoint overlay `/persist/dev-code` over the image at boot.

## Verification checklist (in order, each proves the previous)

1. `curl https://<railway-domain>/health` → `{"status":"ok"}`
2. Tailscale admin → Machines: `nemara-backend` online, tagged
   `tag:backend`; the node's hostname online under the gpu tag.
3. Prod DB: `select * from identity_nodes` → node row with fresh
   `last_seen_at` (heartbeats every roster poll, ~10s).
4. `select count(*) from patients where identity_node_id is not null`
   → online homes assigned.
5. Backend logs: no `tailscale up failed`, no roster 401s.
6. On a live camera: `analyses.detector_fields` rows carry
   `identity_patient_in_clip` / `id_binding` — identity is truly in the
   tick path.

## Traps (each cost real time on 2026-08-19)

- **`ModuleNotFoundError: identity`** at boot → the backend imports the
  node package (`identity.matching`); `docker/Dockerfile` must `COPY
  gpu/identity` + `ENV PYTHONPATH=/app:/app/gpu` (mirrors local compose).
- **`requested tags [tag:backend] are invalid or not permitted`** →
  the Railway `TS_AUTHKEY` must be minted with tag `tag:backend`; a
  gpu-tagged fleet key cannot advertise it. ACL source of truth:
  `infra/tailscale-acl.hujson` (must actually be applied in the admin console).
- **Node up but never registers** → `NEMARA_INTERNAL_SERVICE_KEY`
  missing on Railway. The node 401s silently on every poll.
- **Pod "Exited by Runpod" within minutes** → hand-rolled create landed
  on community cloud and/or the wrong registry path
  (`ghcr.io/aryamandarda/...` is the DEAD pre-transfer path). Use `pod.sh release`.
- **`railway` CLI says "No linked project"** → the link is
  per-directory; `cd` into `nemara-backend` first.
- **Event gate delta**: prod applies the DEFAULT-FALSE variant (no
  patient backfill) until PagerDuty env + client "Evaluation mode" UI
  exist. Enabling is per-patient:
  `update patients set event_gate_enabled=true where id='…'`.

## Standing state (2026-08-19)

- Fleet: exactly 2 GPU pods (dev `nemara-cap4090-*`, prod
  `nemara-prod-gpu-*`, $0.74/hr each) + the $0.06/hr volume-reader CPU
  helper. Dev pod can be STOPPED when not developing.
- Model routing (prod env): `NEMARA_GEMINI_MODEL=gemini-3.1-flash-lite`
  (primary screen), `NEMARA_CONFIRM_MODEL=gemini-3.7-flash` (all k-of-N
  confirm votes), `NEMARA_GEMINI_FALLBACK_MODEL=gemini-3.5-flash-lite`
  (outage hop only).
- Debt, deliberate: dev and prod share `NEMARA_INTERNAL_SERVICE_KEY`
  and the gpu `TS_AUTHKEY` — rotate prod to its own values;
  `scripts/apply_insights_readonly_revoke_delta.sql` is HELD until the
  web app stops querying Postgres directly.
