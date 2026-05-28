"""Standalone worker process entrypoints.

Each module under this package is a long-running asyncio process
designed to be deployed alongside the FastAPI API but in its own
container or systemd unit. They share the DB, ORM models, and
settings with the API but do not share the API's event loop, so a
runaway worker can crash without taking the caregiver-app-facing
HTTP surface down with it.

Modules:

    pi_ingest_worker  -- supervises one `services.ingest.pi.manager.
                         PiIngestManager`, which spawns a LiveKit
                         publisher per Pi camera row in the
                         `cameras` table.
"""
