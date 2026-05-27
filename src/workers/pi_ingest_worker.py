"""Standalone process: Pi camera ingest worker.

Run as:
    python -m src.workers.pi_ingest_worker

The process owns one `PiIngestManager` which spawns a LiveKit
publisher coroutine per Pi camera row in the `cameras` table. The
caregiver-app-facing FastAPI process is unaffected: this worker
shares the database, the ORM models, and the `Settings`, but does
NOT share the API's event loop. A WebRTC publisher hang here cannot
starve API workers.

Lifecycle:

  - SIGINT / SIGTERM: cancel the manager, wait for all relays to
    flush their LiveKit disconnects, exit cleanly.
  - Uncaught exception inside the manager loop: log, exit non-zero
    so systemd restarts us.

Deployment: this entrypoint is meant to live behind a process
supervisor (docker-compose service, systemd unit, k8s deployment).
The supervisor is responsible for restart-on-crash, log rotation,
and resource limits. The worker itself has no built-in restart.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from src.db.session import get_engine, get_sessionmaker
from src.services.ingest.pi.manager import PiIngestManager
from src.settings import get_settings

log = logging.getLogger(__name__)


async def _run() -> int:
    """Main coroutine. Returns the process exit code (0 on clean
    shutdown, non-zero on fatal error).
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info(
        "pi_ingest_worker.start env=%s livekit_url=%s",
        settings.env, settings.livekit_url,
    )

    # Validate the LiveKit creds early so a misconfigured worker
    # fails on startup, not the first time a Pi camera shows up.
    if not settings.livekit_url:
        log.error("NEMARA_LIVEKIT_URL is empty; refusing to start")
        return 2
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error(
            "NEMARA_LIVEKIT_API_KEY / SECRET are required for the publisher"
        )
        return 2

    session_factory = get_sessionmaker()
    manager = PiIngestManager(settings, session_factory)

    stop_event = asyncio.Event()

    def _on_signal(signum: int) -> None:
        log.info("pi_ingest_worker.signal signum=%d, draining", signum)
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _on_signal(int(s)))

    try:
        await manager.start()
        await stop_event.wait()
        return 0
    except Exception:
        log.exception("pi_ingest_worker.fatal")
        return 1
    finally:
        log.info("pi_ingest_worker.stopping")
        await manager.stop()
        engine = get_engine()
        await engine.dispose()
        log.info("pi_ingest_worker.stopped")


def main() -> None:
    """Sync entrypoint suitable for `python -m`."""
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
