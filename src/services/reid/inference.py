"""Per-tick Re-ID labeller: YOLO person detect → BoT-SORT track →
periodic face check per track → state machine → locked PATIENT bbox.

Verbatim port of `nemara-reid-experiment/src/reid/pipeline/identifier.py`
adapted for the production tick budget (one 15-second clip per tick,
state is intentionally per-tick).

Pipeline per tick:

  1. extract frames at `reid_inference_sample_fps` (default 5 fps).
  2. for each frame in temporal order:
       YOLO -> person bboxes
       BoT-SORT.update(dets, frame) -> tracked bboxes with stable IDs
       for each track:
         if UNKNOWN and (frame_idx - last_face_check) >= face_check_interval:
           face check inside the person bbox
           if face matches gallery above adaptive threshold:
             face_hits += 1; lock as PATIENT after `hits_to_lock` hits
           else after `face_checks_for_other` checks: lock as OTHER
       record the last frame's bbox for any locked PATIENT track
  3. emit IdentityResult:
       patient_in_clip = True iff at least one track locked PATIENT
       patient_bbox    = body bbox of the most-recent locked PATIENT frame
       confidence      = best face-match cosine score across the tick
       face_count      = total face checks that returned at least one face

Why this matters vs the v0 face-only path: the body Re-ID embeddings
inside BoT-SORT carry the patient identification through frames where
the face is turned, occluded, or motion-blurred — exactly the failure
mode that hid a patient fall earlier (face pointed toward the floor
during the fall). Body Re-ID locks on the patient's clothing /
silhouette and the label sticks for the rest of the tick.

Threading: the whole per-tick pipeline runs in `asyncio.to_thread`.
YOLO + BoT-SORT + InsightFace are all sync torch / numpy / onnxruntime
work, so we keep them off the event loop with one thread hop instead
of per-frame fan-out.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.repos import face_embedding_repo
from src.services.detection import frame_extraction
from src.services.reid import embeddings as reid_embeddings
from src.services.reid import harvester
from src.services.reid.body_pipeline import BodyReIDPipeline, PipelineConfig
from src.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityResult:
    """Per-tick identity decision for a single patient.

    `reid_ran`: True iff the pipeline executed and produced a verdict.
    False on timeout, crash, feature disabled, or no enrollment yet.
    The upstream gate in `camera_ingest` keys off this: when False,
    it bypasses identity-based suppression. Load-bearing safety
    invariant: a Re-ID infrastructure failure must NEVER silence the
    detector pipeline.

    `patient_in_clip`: True iff at least one BoT-SORT track was
    locked as PATIENT during this tick. Only meaningful when
    `reid_ran=True`.

    `confidence`: best cosine face-match score across all face checks
    in the tick. 0.0 when no match.

    `best_frame_index`: which sampled frame produced the patient
    lock's most recent body bbox. Useful diagnostics; primarily
    informational.

    `face_count`: total face checks that returned at least one face.
    Used by the upstream gate to distinguish "Re-ID saw no faces"
    (face_count=0, ambiguous) from "saw faces, none matched the
    patient" (face_count>0, confident not-patient).

    `patient_bbox`: pixel-space BODY bbox `(x0, y0, x1, y1)` of the
    most-recently-seen locked PATIENT track. None when no track
    locked. This is the BODY box from YOLO, not the face box — that
    matches what the experiment draws and what the clip annotation
    paths burn into the LLM frames + archived clips.
    """
    reid_ran: bool
    patient_in_clip: bool
    confidence: float
    best_frame_index: int | None
    face_count: int
    patient_bbox: tuple[float, float, float, float] | None = None

    # Legacy alias so existing call sites that reference `face_bbox`
    # keep compiling. The semantic shifted from "face bbox" to "body
    # bbox" when we ported the experiment's YOLO + BoT-SORT pipeline.
    @property
    def face_bbox(self) -> tuple[float, float, float, float] | None:
        return self.patient_bbox

    @classmethod
    def empty(cls) -> "IdentityResult":
        return cls(
            reid_ran=False,
            patient_in_clip=False,
            confidence=0.0,
            best_frame_index=None,
            face_count=0,
            patient_bbox=None,
        )


@dataclass
class _HarvestRequest:
    """Surface for the post-thread harvest hop. The thread can't
    cleanly call into the async sessionmaker (it's bound to the parent
    loop), so we hand the payload back to the async caller."""
    embedding: np.ndarray
    det_score: float
    match_score: float


class ReIDInference:
    """In-process Re-ID labeler. One instance per FastAPI process —
    constructed by `main.create_app` at startup so the first tick
    doesn't pay the model-load cost.

    Owns the per-patient gallery cache. The InsightFace, YOLO, and
    OSNet models are owned by their respective modules (lazy
    singletons).
    """

    def __init__(self, sessionmaker: async_sessionmaker, settings: Settings) -> None:
        self._sessionmaker = sessionmaker
        self._settings = settings
        self._gallery_cache: dict[UUID, tuple[float, np.ndarray]] = {}
        self._cache_lock = asyncio.Lock()

    async def label_clip(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
    ) -> IdentityResult:
        """Run Re-ID on a tick clip. Returns the empty result on
        timeout, infrastructure error, or no-enrollment.

        Hard timeout is `settings.reid_timeout_seconds`. The full body
        pipeline is heavier than the face-only path; calibrate that
        setting accordingly.
        """
        if not self._settings.reid_enabled:
            return IdentityResult.empty()
        try:
            return await asyncio.wait_for(
                self._label_clip_inner(mp4_bytes=mp4_bytes, patient_id=patient_id),
                timeout=self._settings.reid_timeout_seconds,
            )
        except asyncio.TimeoutError:
            log.warning(
                "reid.label_clip.timeout patient=%s timeout=%.1fs",
                patient_id, self._settings.reid_timeout_seconds,
            )
            return IdentityResult.empty()
        except Exception as exc:
            log.exception(
                "reid.label_clip.error patient=%s err=%s", patient_id, exc,
            )
            return IdentityResult.empty()

    async def _label_clip_inner(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
    ) -> IdentityResult:
        gallery = await self._get_gallery(patient_id)
        if gallery.size == 0:
            return IdentityResult.empty()

        sample_fps = self._settings.reid_inference_sample_fps
        sample_width = self._settings.reid_inference_sample_width
        try:
            frames = await frame_extraction.extract_frames(
                mp4_bytes,
                fps=sample_fps,
                max_width=sample_width,
            )
        except RuntimeError as exc:
            log.warning("reid.frame_extract_failed err=%s", exc)
            return IdentityResult.empty()

        if not frames:
            return IdentityResult.empty()

        # The whole pipeline (decode + YOLO + BoT-SORT + face checks +
        # state machine) is sync work — hop into a thread once so the
        # event loop stays responsive. The async harvest hop happens
        # AFTER the thread returns; the sessionmaker is bound to this
        # event loop and can't be safely reused from inside a thread.
        result, harvest = await asyncio.to_thread(
            self._run_pipeline_sync,
            frames=frames,
            gallery=gallery,
            patient_id=patient_id,
            sample_fps=sample_fps,
        )
        if harvest is not None:
            try:
                harvested = await self._harvest(
                    patient_id=patient_id,
                    embedding=harvest.embedding,
                    det_score=harvest.det_score,
                    match_score=harvest.match_score,
                )
                if harvested:
                    await self.invalidate_gallery(patient_id)
            except Exception:
                log.exception("reid.harvest_failed patient=%s", patient_id)
        return result

    def _run_pipeline_sync(
        self,
        *,
        frames,
        gallery: np.ndarray,
        patient_id: UUID,
        sample_fps: int,
    ) -> tuple[IdentityResult, Optional[_HarvestRequest]]:
        import cv2  # local: heavy, only needed on the inference path
        started = time.monotonic()

        s = self._settings
        pipeline = BodyReIDPipeline(
            gallery=gallery,
            config=PipelineConfig(
                sample_fps=sample_fps,
                yolo_confidence=s.reid_yolo_confidence,
                face_check_interval=s.reid_face_check_interval,
                hits_to_lock=s.reid_hits_to_lock,
                face_min_det_score=s.reid_face_min_det_score,
                face_checks_for_other=s.reid_face_checks_for_other,
                match_threshold_base=s.reid_match_threshold_base,
                insightface_model=s.reid_insightface_model,
            ),
        )

        for frame_idx, frame in enumerate(frames):
            try:
                jpeg = base64.b64decode(frame.data_base64)
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                log.exception("reid.frame_decode_failed frame=%d", frame_idx)
                continue
            if img is None:
                continue
            pipeline.process_frame(img, frame_idx)

        # Adaptive learning: hand the best face embedding we saw back
        # to the async caller, which will run the harvester + cache
        # invalidation outside this thread (SQLAlchemy session bound
        # to the parent event loop).
        harvest: Optional[_HarvestRequest] = None
        if (
            pipeline.has_any_patient_track
            and pipeline.best_face_embedding is not None
            and pipeline.best_score > 0
        ):
            harvest = _HarvestRequest(
                embedding=pipeline.best_face_embedding,
                det_score=pipeline.best_face_det_score,
                match_score=pipeline.best_score,
            )

        elapsed_ms = (time.monotonic() - started) * 1000.0
        result = IdentityResult(
            reid_ran=True,
            patient_in_clip=pipeline.has_any_patient_track,
            confidence=pipeline.best_score,
            best_frame_index=pipeline.best_frame_idx,
            face_count=pipeline.face_count,
            patient_bbox=pipeline.latest_patient_bbox,
        )
        log.info(
            "reid.label_clip patient=%s frames=%d face_count=%d "
            "tracks=%d patient_in_clip=%s confidence=%.3f elapsed_ms=%.0f",
            patient_id, len(frames), pipeline.face_count, pipeline.num_tracks,
            result.patient_in_clip, result.confidence, elapsed_ms,
        )
        return result, harvest

    async def _harvest(
        self,
        *,
        patient_id: UUID,
        embedding: np.ndarray,
        det_score: float,
        match_score: float,
    ) -> bool:
        return await harvester.maybe_harvest_embedding(
            patient_id=patient_id,
            embedding=embedding,
            det_score=det_score,
            match_score=match_score,
            sessionmaker=self._sessionmaker,
            settings=self._settings,
        )

    # ── Gallery cache ───────────────────────────────────────────

    async def _get_gallery(self, patient_id: UUID) -> np.ndarray:
        """Return the (N, 512) stacked gallery matrix for `patient_id`.

        Cached per-patient with a TTL from settings. Cache miss reads
        from `patient_face_embeddings` (status=ready). Empty array on
        no-enrollment. See module docstring of the previous version
        for the race-safety reasoning around `load_started_at`.
        """
        now = time.monotonic()
        ttl = self._settings.reid_gallery_cache_ttl_seconds
        async with self._cache_lock:
            cached = self._gallery_cache.get(patient_id)
            if cached is not None and (now - cached[0]) < ttl:
                return cached[1]
            load_started_at = time.monotonic()

        async with self._sessionmaker() as session:
            rows = await face_embedding_repo.list_ready_for_patient_unscoped(
                session, patient_id,
            )

        if not rows:
            stacked = np.empty((0, reid_embeddings.EMBEDDING_DIM), dtype=np.float32)
        else:
            vecs = []
            for row in rows:
                try:
                    vecs.append(reid_embeddings.embedding_from_bytes(row.embedding))
                except ValueError as exc:
                    log.warning(
                        "reid.gallery_skip row=%s err=%s", row.id, exc,
                    )
            stacked = (
                np.stack(vecs) if vecs
                else np.empty((0, reid_embeddings.EMBEDDING_DIM), dtype=np.float32)
            )

        async with self._cache_lock:
            existing = self._gallery_cache.get(patient_id)
            if existing is None or existing[0] <= load_started_at:
                self._gallery_cache[patient_id] = (time.monotonic(), stacked)
            return stacked

    async def invalidate_gallery(self, patient_id: UUID) -> None:
        """Drop the cached gallery for a patient. Called from the
        harvester when it appends a new row so the next tick picks it
        up immediately rather than waiting for TTL."""
        async with self._cache_lock:
            self._gallery_cache.pop(patient_id, None)
