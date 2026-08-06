"""FastAPI app factory.

Wires the lifespan (engine init/dispose + detector worker
start/stop) and registers routers. Stays small on purpose — every
new aggregate appends one `include_router` line.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Connect Python's root logger to stdout so application loggers
# (`logging.getLogger(__name__)` calls throughout src/) actually emit.
# Uvicorn configures its own access/error loggers separately; this
# covers everything else (detector worker tick logs, webhook handlers,
# clip archival, push dispatch, etc.). Honors NEMARA_LOG_LEVEL.
logging.basicConfig(
    level=os.getenv("NEMARA_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Silence verbose third-party loggers that flood every tick.
#
# - boxmot's `boxmot.utils.__init__` calls `configure_logging()` at
#   import time, which installs a RichHandler on the "boxmot" logger
#   with propagate=False and a hardcoded INFO level. We need to
#   import that submodule explicitly — `import boxmot` alone doesn't
#   pull in boxmot.utils, so configure_logging would only fire later
#   (when the tracker first instantiates) and clobber any override
#   we set here.
# - ultralytics also chatters at INFO about model loading.
#
# After both have done their import-time logger setup, we strip
# RichHandler and set propagate=True so anything WARNING+ flows
# through our standard root config.
import boxmot.utils  # noqa: F401, E402
import ultralytics  # noqa: F401, E402

for _noisy in ("boxmot", "ultralytics"):
    _lg = logging.getLogger(_noisy)
    _lg.setLevel(logging.WARNING)
    for _h in list(_lg.handlers):
        _lg.removeHandler(_h)
    _lg.propagate = True


# Quiet uvicorn's access log for the docs/schema endpoints. They're hit
# constantly by browsers + scanners and drown out real request traffic.
# `/health` is intentionally KEPT so uptime/healthcheck pings stay
# visible; every other route logs as normal. uvicorn.access emits args as
# (client_addr, method, full_path, http_version, status_code), so we match
# on the path (index 2) and drop only the chatty docs routes.
class _AccessLogQuietFilter(logging.Filter):
    _QUIET_PATHS = frozenset({"/openapi.json", "/docs", "/redoc"})

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3:
            path = str(args[2]).split("?", 1)[0]
            if path in self._QUIET_PATHS:
                return False
        return True


logging.getLogger("uvicorn.access").addFilter(_AccessLogQuietFilter())

# Register every ORM model in SQLAlchemy metadata at startup so foreign
# keys resolve when the first INSERT/SELECT runs. Imported for side
# effects only.
from src.api.briefing import router as briefing_router
from src.api.cameras import router as cameras_router
from src.api.cameras_ingest import router as cameras_ingest_router
from src.api.caregivers import router as caregivers_router
from src.api.detectors import router as detectors_router
from src.api.devices import router as devices_router
from src.api.event_gate import router as event_gate_router
from src.api.events import router as events_router
from src.api.face_embeddings import router as face_embeddings_router
from src.api.feedback import router as feedback_router
from src.api.health import router as health_router
from src.api.home_report import router as home_report_router
from src.api.internal.cameras import router as internal_cameras_router
from src.api.livekit import router as livekit_router
from src.api.livekit_webhooks import router as livekit_webhooks_router
from src.api.pairing import router as pairing_router
from src.api.patient_members import router as patient_members_router
from src.api.patients import router as patients_router
from src.api.presence import router as presence_router
from src.api.security_headers import SecurityHeadersMiddleware
from src.api.support import router as support_router
from src.api.talk import router as talk_router
from src.db import analysis as _analysis  # noqa: F401
from src.db import camera as _camera  # noqa: F401
from src.db import camera_detector as _camera_detector  # noqa: F401
from src.db import caregiver as _caregiver  # noqa: F401
from src.db import caregiver_device as _caregiver_device  # noqa: F401
from src.db import detector as _detector  # noqa: F401
from src.db import event as _event  # noqa: F401
from src.db import feedback as _feedback  # noqa: F401
from src.db import patient as _patient  # noqa: F401
from src.db import patient_face_embedding as _patient_face_embedding  # noqa: F401
from src.db import pi_discovery as _pi_discovery  # noqa: F401
from src.db.session import get_engine, get_sessionmaker
from src.services import clip_archival, email, event_gate, pager, push
from src.services.data_collection import manager as recorder_manager
from src.services.detection.presence_cache import PresenceCache
from src.services.detection.presence_sweeper import PresenceSweeper
from src.services.detection.worker import DetectorWorker
from src.services.scheduling.scheduler import ScheduledDetectorRunner
from src.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Touch the engine so the lazy `lru_cache` factory binds at startup
    # instead of on the first request — surfaces bad NEMARA_DB_URL early.
    engine = get_engine()
    # Sweep clip tempfiles stranded by a hard-killed (SIGKILL/OOM)
    # previous process — raw clip bytes must never outlive a crash.
    # See clip_archival.sweep_scratch_dir; runs before any clip work.
    clip_archival.sweep_scratch_dir()
    # Same for the data-collection recorder's tempfiles, which are raw
    # uncensored video/audio and had no sweep of their own until now.
    recorder_manager.sweep_temp_dir()
    # (No censor model warmup since 2026-08-06: the censor's person
    # masks come from the perception pod's EdgeTAM segmentation over
    # HTTP, so there are no local weights to download and no torch to
    # initialize. The warmup existed to get `yolo11n-seg.pt` cached in
    # the process cwd before the first censored clip; nothing in the
    # censor path loads a model any more.)
    # Phase 2: in-process detector subscriber-bot. Webhook-driven; the
    # `/v1/livekit/webhooks` route reads `app.state.detector_worker`
    # to dispatch participant_joined/left events. Worker idles at
    # zero cost when no rooms are active.
    worker = DetectorWorker(
        settings=get_settings(), sessionmaker=get_sessionmaker(),
    )
    app.state.detector_worker = worker
    await worker.start()
    # Scheduled ("absence") detector runtime — standalone background
    # loop, independent of the webhook-driven DetectorWorker.
    scheduler = ScheduledDetectorRunner(
        settings=get_settings(), sessionmaker=get_sessionmaker(),
    )
    app.state.scheduler = scheduler
    await scheduler.start()
    # `cameras.online` is now owned by each `RoomWorker`'s
    # `LiveKitRoomReconciler` (see room_reconciler.py). The standalone
    # `StaleOnlineSweeper` was removed because REST polling produces
    # the same staleness signal without a separate DB sweep.
    #
    # `cameras.app_present` (is the Memoria Sensor iPhone app open?) is a
    # SEPARATE, poll-based, room-independent signal: the desired-state
    # poll handler touches an in-memory PresenceCache (no DB write on
    # that 1s hot path) and PresenceSweeper flips the column on
    # transitions. Unlike `online`, this has no per-room owner — a
    # camera whose app is alive but not publishing has no RoomWorker —
    # so it MUST be swept globally here. The cache lives on app.state so
    # the poll handler (api/cameras.py) and the sweeper share one
    # instance. SINGLE REPLICA ONLY (see presence_cache.py).
    presence_cache = PresenceCache(
        stale_seconds=get_settings().sensor_presence_stale_seconds,
    )
    app.state.presence_cache = presence_cache
    presence_sweeper = PresenceSweeper(
        settings=get_settings(),
        sessionmaker=get_sessionmaker(),
        cache=presence_cache,
    )
    app.state.presence_sweeper = presence_sweeper
    await presence_sweeper.start()
    # Durable half of the event gate's auto-release. The per-event
    # asyncio timer dies with this process, so a deploy inside a hold
    # window would strand an alert at 'pending' forever and no
    # caregiver would ever be told. This sweeps those out. Runs
    # unconditionally — it is a safety net, not a feature.
    gate_sweeper = event_gate.GateSweeper(
        settings=get_settings(), sessionmaker=get_sessionmaker(),
    )
    app.state.gate_sweeper = gate_sweeper
    await gate_sweeper.start()
    try:
        yield
    finally:
        await gate_sweeper.stop()
        await presence_sweeper.stop()
        await scheduler.stop()
        await worker.stop()
        await push.close()
        await pager.close()
        await email.close()
        await engine.dispose()


def create_app() -> FastAPI:
    # Interactive docs and the raw schema are developer conveniences, and
    # outside local/test they are pure reconnaissance: production served
    # /openapi.json unauthenticated, handing over all 53 paths and 68
    # operations with request/response schemas. Same fail-closed shape as
    # _require_internal_key — local and test keep the convenience, every
    # other env gets nothing. Passing None also unmounts the routes, so
    # this cannot be defeated by requesting the path directly.
    _docs_enabled = get_settings().env in ("local", "test")
    app = FastAPI(
        title="nemara-backend",
        version="0.0.1",
        lifespan=lifespan,
        docs_url="/docs" if _docs_enabled else None,
        redoc_url="/redoc" if _docs_enabled else None,
        openapi_url="/openapi.json" if _docs_enabled else None,
    )
    # Baseline browser-facing headers on every response (HSTS, CSP,
    # nosniff, frame-deny). See src/api/security_headers.py.
    app.add_middleware(SecurityHeadersMiddleware)
    app.include_router(health_router)
    app.include_router(caregivers_router)
    app.include_router(briefing_router)
    app.include_router(presence_router)
    app.include_router(home_report_router)
    # Operator-only event gate: pre-dispatch false-positive check.
    # Bearer token separate from caregiver auth (deps.operator_auth).
    app.include_router(event_gate_router)
    app.include_router(patients_router)
    app.include_router(patient_members_router)
    app.include_router(detectors_router)
    app.include_router(events_router)
    app.include_router(cameras_router)
    app.include_router(cameras_ingest_router)
    # APNs device-token registration.
    app.include_router(devices_router)
    # Pairing flow + LiveKit token minting.
    app.include_router(pairing_router)
    app.include_router(livekit_router)
    # Phase 2: webhook receiver feeding the in-process detector worker.
    app.include_router(livekit_webhooks_router)
    # Phase 3 caregiver Talk mode.
    app.include_router(talk_router)
    # Re-ID face enrollment + status.
    app.include_router(face_embeddings_router)
    # Contact Support (Resend forwarding) from the iOS app's Settings.
    app.include_router(support_router)
    # Product Feedback (persisted + best-effort Resend) from Settings.
    app.include_router(feedback_router)
    # Support-only /internal routes (room-privacy reclassify escape
    # hatch). Guarded by the X-Internal-Key shared secret; fails closed
    # outside local/test when NEMARA_INTERNAL_SERVICE_KEY is unset, so
    # mounting unconditionally is safe.
    app.include_router(internal_cameras_router)

    # Experimental, non-production routes under src/internals/. Gated by
    # NEMARA_ENABLE_INTERNALS (default off). The import lives inside the
    # guard so a deploy without src/internals/ can never break startup.
    if get_settings().enable_internals:
        from src.internals.record_clip import router as internals_record_clip_router

        app.include_router(internals_record_clip_router)
        logging.getLogger(__name__).info(
            "internals enabled: mounted record-clip endpoint"
        )
    return app


app = create_app()
