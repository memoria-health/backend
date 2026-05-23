"""FastAPI app factory.

Wires the lifespan (engine init/dispose + detector worker
start/stop) and registers routers. Stays small on purpose — every
new aggregate appends one `include_router` line.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Register every ORM model in SQLAlchemy metadata at startup so foreign
# keys resolve when the first INSERT/SELECT runs. Imported for side
# effects only.
from src.api.cameras import router as cameras_router
from src.api.caregivers import router as caregivers_router
from src.api.detectors import router as detectors_router
from src.api.events import router as events_router
from src.api.health import router as health_router
from src.api.livekit import router as livekit_router
from src.api.livekit_webhooks import router as livekit_webhooks_router
from src.api.pairing import router as pairing_router
from src.api.patients import router as patients_router
from src.api.talk import router as talk_router
from src.db import analysis as _analysis  # noqa: F401
from src.db import camera as _camera  # noqa: F401
from src.db import caregiver as _caregiver  # noqa: F401
from src.db import detector as _detector  # noqa: F401
from src.db import event as _event  # noqa: F401
from src.db import patient as _patient  # noqa: F401
from src.db.session import get_engine, get_sessionmaker
from src.services.detection.worker import DetectorWorker
from src.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Touch the engine so the lazy `lru_cache` factory binds at startup
    # instead of on the first request — surfaces bad NEMARA_DB_URL early.
    engine = get_engine()
    # Phase 2: in-process detector subscriber-bot. Default off; flip
    # NEMARA_LIVEKIT_WORKER_ENABLED=true to enable. Webhook handler
    # in `/v1/livekit/webhooks` reads `app.state.detector_worker` to
    # dispatch participant_joined/left events.
    worker = DetectorWorker(
        settings=get_settings(), sessionmaker=get_sessionmaker(),
    )
    app.state.detector_worker = worker
    await worker.start()
    try:
        yield
    finally:
        await worker.stop()
        await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="nemara-backend",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(caregivers_router)
    app.include_router(patients_router)
    app.include_router(detectors_router)
    app.include_router(events_router)
    app.include_router(cameras_router)
    # Pairing flow + LiveKit token minting.
    app.include_router(pairing_router)
    app.include_router(livekit_router)
    # Phase 2: webhook receiver feeding the in-process detector worker.
    app.include_router(livekit_webhooks_router)
    # Phase 3 caregiver Talk mode.
    app.include_router(talk_router)
    return app


app = create_app()
