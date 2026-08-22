"""FastAPI app factory.

Wires the lifespan (engine init/dispose + detector worker
start/stop) and registers routers. Stays small on purpose — every
new aggregate appends one `include_router` line.
"""

import logging
import subprocess
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Process-wide logging: single stdout handler (JSON on Railway so log
# levels classify correctly, plain locally), uvicorn loggers routed
# through it, livekit burst-dedup. Honors NEMARA_LOG_LEVEL and
# NEMARA_LOG_JSON.
from src.logging_setup import configure_logging

configure_logging()

# NOTE: torch-family imports (ultralytics, boxmot, torch) are banned at
# module level in this process. This file used to import boxmot.utils +
# ultralytics here purely to silence their loggers, which paid the full
# torch runtime (hundreds of MB resident) in a process that runs YOLO
# only in child processes. boxmot is no longer imported anywhere in the
# backend (the re-id brain lives in the GPU sidecar); ultralytics is
# lazy-imported by `censor._get_seg_model`, which silences its logger
# at that point.


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
from src.api.insights import router as insights_router
from src.api.insights_widgets import router as insights_widgets_router
from src.api.journal import router as journal_router
from src.api.summaries import router as summaries_router
from src.api.internal.cameras import router as internal_cameras_router
from src.api.livekit import router as livekit_router
from src.api.livekit_webhooks import router as livekit_webhooks_router
from src.api.pairing import router as pairing_router
from src.api.patient_members import router as patient_members_router
from src.api.patients import router as patients_router
from src.api.presence import router as presence_router
from src.api.security_headers import SecurityHeadersMiddleware
from src.api.identity_roster import router as identity_roster_router
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
from src.db.session import get_engine, get_sessionmaker
from src.services import clip_archival, email, event_gate, pager, push
from src.services.data_collection import manager as recorder_manager
from src.services.detection.presence_cache import PresenceCache
from src.services.detection.presence_sweeper import PresenceSweeper
from src.services.identity.fleet_watch import FleetWatch
from src.services.detection.worker import DetectorWorker
from src.services.scheduling.scheduler import ScheduledDetectorRunner
from src.settings import get_settings


def _warm_censor_weights() -> None:
    """Fetch the censor's YOLO seg weights into cwd via a THROWAWAY
    child process (daemon thread wrapper).

    Replaces `_warm_censor_model`, which loaded the model in THIS
    process and pinned the full torch runtime (hundreds of MB) for the
    process lifetime — even though tier-1 censoring runs in child
    processes and the only in-parent consumer (privacy-masked door
    stills) can lazy-load on first use. The child pays the download +
    torch init, dies, and leaves the weights file cached in cwd where
    every later censor subprocess (and a lazy in-parent load) finds it.
    """
    from src.services.detection.censor import _SEG_MODEL_NAME

    try:
        proc = subprocess.run(
            [
                sys.executable, "-c",
                "from ultralytics import YOLO; "
                f"YOLO({_SEG_MODEL_NAME!r})",
            ],
            capture_output=True, timeout=600,
        )
        if proc.returncode == 0:
            logging.getLogger(__name__).info("censor weights warmed (subprocess)")
        else:
            logging.getLogger(__name__).warning(
                "censor weights warmup exited %d: %s",
                proc.returncode, proc.stderr.decode(errors="replace")[-500:],
            )
    except Exception:
        logging.getLogger(__name__).warning(
            "censor weights warmup failed; first censored clip or masked "
            "still pays the download", exc_info=True,
        )


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
    # Warm the censor's YOLO seg WEIGHTS off the startup path — in a
    # throwaway subprocess, so this process never pays torch residency
    # for a model it runs in children. ultralytics resolves the bare
    # "yolo11n-seg.pt" name against cwd (repo root locally, WORKDIR
    # /app in the container); the child downloads it there and every
    # censor subprocess finds it cached instead of re-downloading.
    threading.Thread(
        target=_warm_censor_weights, name="censor-weights-warmup", daemon=True
    ).start()
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
    # Dead-man watch over the GPU nodes. Identity fails OPEN, so a node
    # that dies costs its homes their routine detections while everything
    # keeps looking healthy — that is how the 2026-07-26 OOM went unnoticed
    # for hours. Nothing else notices, so this does.
    fleet_watch = FleetWatch(
        settings=get_settings(), sessionmaker=get_sessionmaker(),
    )
    app.state.fleet_watch = fleet_watch
    await fleet_watch.start()
    try:
        yield
    finally:
        await fleet_watch.stop()
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
    # Insights read API: the caregiver dashboard's data layer — replaces
    # the web app's direct nemara_readonly/BYPASSRLS Postgres connection.
    app.include_router(insights_router)
    app.include_router(insights_widgets_router)
    app.include_router(summaries_router)
    app.include_router(journal_router)
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
    # GPU node roster — nodes poll to be told which homes to run. This is
    # also their heartbeat, so it is what makes failover and scale-out work.
    app.include_router(identity_roster_router)
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
