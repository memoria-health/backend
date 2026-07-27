# Identity Productionization & GPU Split Implementation Plan

> **For Claude:** execute task-by-task; every task ends with the full test
> suite green and a commit; adjudicated checkpoint review after each phase;
> full-diff review before PR. Telemetry/monitoring discussion with Aryaman
> happens BEFORE the PR is opened (his explicit gate, 2026-07-26).

**Goal:** Turn the proven identity system into a deployable production unit:
a clean `gpu/` deployable (perception + identity co-located on the GPU host),
a single versioned+authenticated `/state` seam to the backend, zero
experiment flags, and zero legacy reid code.

**Architecture:** Monorepo, two deployment targets. `gpu/` holds everything
that runs on the GPU host (perception = ex-sam_service; identity = ex-
src/services/identity + run_identity, now flagless with baked defaults;
deploy = supervisor/bootstrap/tunnel). `src/` is backend-only and shares
exactly ONE module with the GPU side: `gpu/identity/contract.py` (the /state
schema, version, auth). The legacy in-process reid brain is deleted; the
sidecar is the only identity provider.

**Locked decisions (Aryaman, 2026-07-26):**
- Separate identity service deployed directly on the GPU host. Monorepo.
- RunPod reserved secure-cloud pod (NOT serverless — stateful streaming
  workload, no scale-to-zero, cold starts unacceptable).
- Baked production config: SAM3 grounder, native ship resolution,
  3fps in-batch rate, queue-cap 2, reground every 5 ticks, faces on.
- All experiment flags deleted. All legacy reid code deleted this branch.
- Telemetry deferred, but discussed before PR (pilot must not be blind).
- Capacity truth: 3–4s pod time per batch at ~5s cadence (60–80% duty per
  occupied room) → size at 1–2 homes per A40 at peak; measure upside.

---

## Phase 0 — Land the current pile as reviewable commits
Working tree holds ~3 days of proven-live work. Commit as logical units
(no behavior change, suite green as-is):
- 0.1 pod engine: sam_service (yoloseg grounder kept-but-inactive, epoch,
      trigger weights), scripts/deploy+bootstrap, POD_RUNBOOK
- 0.2 identity service experiment flags + journal/timeline changes +
      run_identity overrides (they die in Phase 2, but land them first so
      the deletion diff is honest)
- 0.3 backend sidecar: settings, SidecarReID, worker switch, ingest
      people_present/arm/latency stashes, tests
- 0.4 dashboard: real timings, native 10s timeline, names, labels,
      tick-grouping fix

## Phase 1 — The gpu/ split (pure moves, no behavior change)
- 1.1 `git mv sam_service gpu/perception`; python package rename
      sam_service→perception (imports, uvicorn target, supervisor,
      deploy/bootstrap scripts, tests). Suite green.
- 1.2 `git mv src/services/identity gpu/identity`; move
      tools/run_identity.py → gpu/identity/service.py; move the GPU-side
      wire client (src/services/reid/sam_client.py, remote_replay wire
      dataclasses) into gpu/identity/wire.py — after Phase 3 the backend
      has zero pod knowledge. Tests move imports only.
- 1.3 Extract `gpu/identity/contract.py`: /state response schema
      (versioned, `contract: state-v1`), camera-key convention, bearer
      auth check. `src/services/reid/sidecar.py` imports ONLY this.
      Sidecar sends Authorization header; service requires it when
      NEMARA_IDENTITY_TOKEN is set.

## Phase 2 — Bake production config, delete flags, supervise
- 2.1 Delete --preset/--track-width/--batch-fps/--queue-cap; constants in
      gpu/identity/config.py = the proven values. service.py runs the
      production config with zero arguments.
- 2.2 gpu/deploy: identity supervisor (file-based, setsid, restart-on-exit
      — same discipline as perception; the 11:57 OOM kill proved the need)
      + docker-compose for the GPU host running perception + N identity
      instances (one per home/room env).
- 2.3 Update pod deploy scripts + POD_RUNBOOK for the new layout; redeploy
      pod; live validation (ticks, seat re-lock, dashboard).

## Phase 3 — Legacy deletion (the brain comes out)
- 3.1 Extract survivors first: IdentityResult → src/services/reid/result.py;
      TrackObservation + LABEL_* → src/services/reid/observations.py.
      Update all importers (worker, ingest, ledger, scheduler, executor,
      id_evidence, sidecar, tests). Suite green BEFORE any deletion.
- 3.2 Delete: inference.py, remote_replay.py (pipeline half),
      cross_clip_body_gallery.py, harvester.py, tracker.py,
      body_pipeline.py internals (keep only what observations.py absorbed),
      embeddings.py → gpu/identity (it serves the identity OSNet fold).
      worker.py: sidecar is the only provider; identity_provider flag
      deleted; NEMARA_IDENTITY_STATE_URL stays (it's config, not a flag).
- 3.3 Delete legacy tests wholesale where they test deleted code
      (test_reid_e2e, test_pipeline_sync_attribution, ...); adapt the ones
      covering kept semantics (ledger, suppression gate) to sidecar-shaped
      inputs. Suite green.
- 3.4 Pod probe scripts, retired yolo BOX grounding path (yoloseg MASK path
      stays switchable — it has a dedup fix pending), tracker-eval scratch
      docs → docs/, dashboard dead cards (legacy lanes stub, id_frames
      card, idticks panel).

## Phase 4 — Validation & wrap
- 4.1 Full pod redeploy from clean branch; e2e: camera → perception →
      identity → /state → backend tick → dashboard. Gemini re-enabled
      (reid_soak_only off) for one supervised hour — the last unproven
      interaction.
- 4.2 Adjacency test with relatives on the final artifact (Aryaman).
- 4.3 Full-diff adjudicated review (correctness / efficiency / memory /
      security / goal-adherence), fix findings.
- 4.4 **Telemetry & monitoring discussion with Aryaman** → then PR.

## Deferred (explicitly out of this branch)
- Telemetry implementation (post-discussion), sleep-occupancy layer,
  server-side GPU fairness queue (single-home pilot doesn't hit it),
  vLLM/self-hosted VLM evaluation, eval-suite benchmark harness,
  yoloseg containment-dedup retrial, edge-box path.
