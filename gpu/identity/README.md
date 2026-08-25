# Identity service — continuous tracking + sparse identity

The 2026-07-24 architecture (docs/plans/2026-07-24-continuous-identity-
architecture.md): per-camera continuous batches drive the pod's
persistent EdgeTAM session (`gpu/perception/`, see
docs/architecture/continuous-tracking.md); a three-rule identity layer
sits on top.

1. **Score** everyone on one calibrated scale — face weighted by pixel
   size, body (OSNet pool) as the workhorse (`scoring.py`).
2. **Seat** — highest combined score takes the single home-wide patient
   seat; challengers must beat it by δ; the seat rides the track and
   dies with it; silence holds (`seat.py`).
3. **Harvest** — face-confirmed track + alone in frame → harvest the
   body pool liberally; nothing else ever harvests (`harvest.py`).

Entry point: `gpu/identity/service.py`, ex `tools/run_identity.py`
(MJPEG overlay :8012 + the versioned, token-authenticated `/state` seam
defined in `contract.py`).

On a **node**, processes are started and supervised by
`gpu/deploy/roster_agent.sh`, one per room, on ports the backend assigns
via `POST /v1/identity/roster`. Nothing is launched by hand there — the
node is told which homes to run and reconciles to that list.
(`gpu/deploy/identity_supervisor.sh`, which ran a single room from an env
var, was deleted 2026-08-11.)

**Locally**, run one directly:

    uv run python gpu/identity/service.py --room patient_<uuid>

There is no `--patient` (removed 2026-08-20). The process loads a face
gallery for every resident its room's cameras cover, resolved from
`cameras.livekit_room` -> `camera_patients`, because a shared camera
watches more than one person and a single flag could only name one of
them. The 60s gallery refresh re-reads coverage too, so sharing a
camera with another resident takes effect without a restart.

Pure modules are unit-tested (`test/unit/test_identity_*.py`).

## Relationship to `src/services/identity/` (the backend half)

The consolidation this section used to track is DONE (2026-07-27,
commits ecbefea / dfb9b4e / f61d9d8): everything identity lives HERE —
pod wire client (`pod_client.py`), body gallery (`gallery.py`),
embedding math + codec (`matching.py`), screen/rendered-person veto
(`display_veto.py`), evidence loop (`loop.py`), journal, service. The
legacy per-tick brain (`inference.py` / `body_pipeline.py` / the old
per-clip ledger) was deleted at P3 — **the sidecar is the only identity
provider**; detection reads Patient State through the `/state` seam.

`src/services/identity/` (renamed from `services/reid`) keeps only what
the BACKEND needs to consume and feed this service: the `/state` client
(`identity_client.py`), verdict shapes (`result.py`, `observations.py`),
the sticky bind-once-hold lock (`track_ledger.py` — keyed by the pod's
stable track ids, PATIENT/UNKNOWN only), instrumentation (`trace.py`),
and the caregiver-facing face-enrollment flow (`embeddings.py`,
`enrollment_processor.py`, `storage.py`) whose vectors this pipeline
reads back through the DB.

Only `service.py` may import from `src/*` (DB session + repos +
settings for gallery loads) — the known packaging exception until the
GPU host gets its own config surface.
