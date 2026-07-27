"""Backend half of the identity seam.

NOT the identity pipeline — that lives in gpu/identity (the
consolidated GPU-host deployable). This package is what the BACKEND
needs to consume and feed it: the /state client (identity_client),
the verdict shapes (result, observations), the sticky lock
(track_ledger), instrumentation (trace), and the caregiver-facing
face-enrollment flow (embeddings, enrollment_processor, storage)
whose vectors the pipeline reads back through the DB.

Renamed from services/reid 2026-07-27 (Aryaman): the brain this
folder was named for is deleted; what remains is identity plumbing.
"""
