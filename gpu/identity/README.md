# Identity service — continuous tracking + sparse identity

The 2026-07-24 architecture (docs/plans/2026-07-24-continuous-identity-
architecture.md): per-camera continuous batches drive the pod's
persistent EdgeTAM session; a three-rule identity layer sits on top.

1. **Score** everyone on one calibrated scale — face weighted by pixel
   size, body (OSNet pool) as the workhorse (`scoring.py`).
2. **Seat** — highest combined score takes the single home-wide patient
   seat; challengers must beat it by δ; the seat rides the track and
   dies with it; silence holds (`seat.py`).
3. **Harvest** — face-confirmed track + alone in frame → harvest the
   body pool liberally; nothing else ever harvests (`harvest.py`).

Entry point: `tools/run_identity.py` (MJPEG overlay :8012 + /state).
Pure modules are unit-tested (`test/unit/test_identity_*.py`).

## Relationship to `src/services/identity/` (the legacy folder)

`reid/` holds TWO different things — do not delete it wholesale:

| reid/ file | Role today | Fate |
|---|---|---|
| `sam_client.py` | pod wire client — **used by this service** | migrates here after P3 |
| `embeddings.py` | ArcFace cosine/gallery — **used by this service** | migrates here after P3 |
| `body_embeddings.py` | OSNet extraction — **used by this service** | migrates here after P3 |
| `cross_clip_body_gallery.py` | body pool + hygiene — **used by this service** | migrates here after P3 |
| `display_veto.py` | screen/rendered-person veto — **used by this service** | migrates here after P3 |
| `remote_replay.py` | `RemoteObservations` types — **used by this service** | types migrate; pipeline dies |
| `enrollment_processor.py` | face enrollment ingestion (API) | stays (serves the DB gallery) |
| `inference.py` | LEGACY per-tick brain | **deleted at P3** (detection reads Patient State instead) |
| `track_ledger.py` | LEGACY bind-once ledger | **deleted at P3** (replaced by `seat.py`) |
| `body_pipeline.py` | LEGACY in-clip brain | **deleted at P3** |
| `harvester.py` | LEGACY DB face harvest | reviewed at P3 |
| `trace.py` | eval-rig instrumentation | stays with the eval rig |

P3 = detection-pipeline integration at `camera_ingest.py:1122` (the
`IdentityResult` seam) after this service passes the eval matrix
(TRACKER_EVAL_FINDINGS.md §5). Until then the legacy brain must stay
runnable for A/B and fallback.
