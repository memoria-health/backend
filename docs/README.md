# docs

> Last updated: 2026-08-16

Documentation for nemara-backend. The repo root keeps only [README.md](../README.md) (local dev quickstart); everything else lives here.

**Identity / perception GPU stack — start here:**
- [architecture/identity-gpu-architecture.md](architecture/identity-gpu-architecture.md)
  — the master doc: architecture, the in-process concurrency law,
  worker pool + stateless dispatch, measured capacity, and the
  production deployment runbook (§0 is current; later sections are
  history).
- [plans/2026-08-16-cross-camera-batching.md](plans/2026-08-16-cross-camera-batching.md)
  — deferred capacity enhancement, with the full falsification
  evidence behind the concurrency law.

## Layout

```
docs/
├── architecture/    how the system is built
│   ├── backend-architecture.md          repo layout, conventions, pipeline wiring
│   ├── detection-architecture-v2.md     v2 detection contract (alert_trigger spine, G0–G7)
│   ├── detection-inference-architecture.md  SUPERSEDED — historical evidence base kept
│   ├── continuous-tracking.md           SAM3+EdgeTAM perception core (gpu/perception/)
│   └── identity-gpu-architecture.md     GPU fleet master doc: concurrency law, stateless
│                                        workers, capacity, deployment runbook (§0 current)
│
├── guides/          how to use / operate features
│   ├── setup-hosted-stack.md            the deployed prod stack (Railway/Supabase/LiveKit/Vercel)
│   ├── vlm-routing.md                   switching a family between Qwen/Gemini/OpenAI/DashScope
│   ├── data-collection.md               video data-collection feature guide + handoff
│   ├── detection-accuracy-knobs.md      accuracy tuning knobs
│   └── detector-config-knobs.md         per-detector config knobs
│
├── runbooks/        step-by-step operational procedures
│   ├── pod.md                           GPU pod spin up/down/deploy (scripts/pod/pod.sh)
│   ├── soak-protocol.md                 reusable soak methodology + open detector-axis validation
│   ├── tracker-eval.md                  SAM3 vs EdgeTAM A/B rig (superseded live by identity service)
│   ├── ghost-deadlock-known-fix.md      legacy track-ledger bug + unapplied fix (dies with P3 cleanup)
│   └── detector-testing-checklist.md    detector validation master checklist (2026-08-12)
│
├── plans/           dated design/implementation plans, each with a "> Status:" header
│   └── 2026-06-24 → 2026-08-16          data collection, detection v2, patient sharing,
│                                        identity succession, censor-masks + gpu-fleet handoffs,
│                                        cross-camera batching (deferred)
│
└── research/        experiment findings and evidence records
    ├── 2026-07-19-person-locking-identity-persistence.md
    └── 2026-07-23-tracker-eval-findings.md
```

## Conventions

- **Plans are historical records.** Bodies stay as written; a `> Status (date):` header at the top says what actually shipped and what superseded it. Don't rewrite history — annotate it.
- **New plans**: `docs/plans/YYYY-MM-DD-short-name.md`. New runbooks/guides: kebab-case, no date unless one-shot.
- **One-shot migration runbooks are deleted once verified shipped** (the migrations workflow itself is documented in `guides/setup-hosted-stack.md`).
- Colocated exceptions: `gpu/identity/README.md` (module orientation, lives next to the code).
