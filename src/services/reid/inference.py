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
from src.services.detection.frame_extraction import ExtractedFrame
from src.services.reid import body_embeddings, embeddings as reid_embeddings
from src.services.reid import harvester
from src.services.reid.body_pipeline import (
    LABEL_OTHER,
    LABEL_PATIENT,
    BodyReIDPipeline,
    PipelineConfig,
)
from src.services.reid.cross_clip_body_gallery import CrossClipBodyGallery
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

    `people_present_reid` / `people_count_reid`: body-track-derived
    presence and concurrent count. The state machine reads these
    instead of the LLM's `people_present` field — body tracking
    catches presence even when no face is visible (turned away,
    occluded, motion-blurred). `people_count_reid` is peak concurrent
    tracks in any one frame, not cumulative across the clip.
    """
    reid_ran: bool
    patient_in_clip: bool
    confidence: float
    best_frame_index: int | None
    face_count: int
    patient_bbox: tuple[float, float, float, float] | None = None
    people_present_reid: bool = False
    people_count_reid: int = 0
    # True iff num_tracks > 0 AND every track in the clip ended labeled
    # OTHER (face seen + match failed enough times to be confidently
    # non-patient). Drives the camera_ingest gate's "skip LLM when
    # everyone in frame is verified not-patient" branch — replaces the
    # old `face_count > 0` gate for the face-hidden-second-half-of-clip
    # case. False when any track is UNKNOWN, when patient is in clip,
    # or when no tracks at all.
    has_only_verified_other_tracks: bool = False

    # True iff this patient has at least one enrolled face embedding
    # (non-empty gallery). Drives the camera_ingest presence gate:
    #   faces_enrolled=True  -> trust YOLO/re-ID for open/close
    #   faces_enrolled=False -> fall back to the LLM's people_present
    #                           verdict (and the YOLO pipeline is skipped
    #                           entirely for these patients to save CPU).
    # Independent of reid_ran: a patient with faces enrolled whose tick
    # timed out has faces_enrolled=True but reid_ran=False.
    faces_enrolled: bool = False

    # Frames the Re-ID pipeline already decoded + sampled. The LLM
    # provider reuses these instead of running its own ffmpeg
    # extract_frames pass (the OpenAI fps/width settings match Re-ID's
    # sampling: 5 fps, 480px). None when Re-ID didn't run (disabled,
    # no enrollment, timeout, error).
    extracted_frames: list[ExtractedFrame] | None = None

    # One body bbox per entry in `extracted_frames`, in the same pixel
    # space. `None` for frames where no PATIENT track was locked at
    # that instant (e.g., the lock hasn't happened yet, or the track
    # was lost on this frame). The provider uses these to draw a
    # frame-accurate green rectangle that follows the patient — the
    # caregiver-facing archive uses the same per-frame approach via
    # `clip_annotate`, so the LLM sees what the caregiver sees.
    per_frame_bboxes: list[tuple[float, float, float, float] | None] | None = None

    # Sampling rate of `extracted_frames`. The provider uses this for
    # the "sampled at N fps" line in its prompt so the LLM's temporal
    # reasoning matches the actual frame cadence.
    sample_fps: int | None = None

    # Legacy alias so existing call sites that reference `face_bbox`
    # keep compiling. The semantic shifted from "face bbox" to "body
    # bbox" when we ported the experiment's YOLO + BoT-SORT pipeline.
    @property
    def face_bbox(self) -> tuple[float, float, float, float] | None:
        return self.patient_bbox

    @classmethod
    def empty(cls, *, faces_enrolled: bool = False) -> "IdentityResult":
        return cls(
            reid_ran=False,
            patient_in_clip=False,
            confidence=0.0,
            best_frame_index=None,
            face_count=0,
            patient_bbox=None,
            extracted_frames=None,
            per_frame_bboxes=None,
            sample_fps=None,
            people_present_reid=False,
            people_count_reid=0,
            has_only_verified_other_tracks=False,
            faces_enrolled=faces_enrolled,
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

    def __init__(
        self,
        sessionmaker: async_sessionmaker,
        settings: Settings,
        body_gallery: "CrossClipBodyGallery | None" = None,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._settings = settings
        self._gallery_cache: dict[UUID, tuple[float, np.ndarray]] = {}
        self._cache_lock = asyncio.Lock()
        # Shared cross-clip body gallery. None = disable cross-clip
        # persistence (production default constructs one; tests can
        # opt out by passing None).
        self._body_gallery = body_gallery

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
            return IdentityResult.empty(faces_enrolled=False)
        # Determine enrollment up front (the gallery is cached) so the
        # verdict is known even if the pipeline below times out, and so we
        # can SKIP the whole YOLO/re-ID pipeline for patients with no
        # enrolled faces — for them, presence falls back to the LLM's
        # people_present verdict (see camera_ingest gate). This is the
        # CPU saver: re-ID only runs for patients who actually enrolled.
        gallery = await self._get_gallery(patient_id)
        faces_enrolled = gallery is not None and len(gallery) > 0
        if not faces_enrolled:
            return IdentityResult.empty(faces_enrolled=False)
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
            return IdentityResult.empty(faces_enrolled=True)
        except Exception as exc:
            log.exception(
                "reid.label_clip.error patient=%s err=%s", patient_id, exc,
            )
            return IdentityResult.empty(faces_enrolled=True)

    async def _label_clip_inner(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
    ) -> IdentityResult:
        # Empty gallery (no enrollment yet) is no longer a hard stop:
        # body tracking still runs and populates
        # `people_present_reid`/`people_count_reid` so the state
        # machine can close the camera when nobody is in frame, even
        # before the caregiver enrolls a patient face. `patient_in_clip`
        # stays False on this path because identity matching needs a
        # gallery.
        gallery = await self._get_gallery(patient_id)

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
            return IdentityResult.empty(faces_enrolled=True)

        if not frames:
            return IdentityResult.empty(faces_enrolled=True)

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

        # Cross-clip body gallery pre-match callback. Body pipeline
        # calls this once per new track. Extract OSNet body embedding
        # and match against the day's gallery for this patient. Sync
        # because body_pipeline runs in to_thread and the gallery uses
        # threading.Lock.
        pre_match_cb = None
        if (
            self._body_gallery is not None
            and self._settings.reid_body_gallery_enabled
        ):
            gallery_ref = self._body_gallery
            pid = patient_id

            def _pre_match(
                frame_bgr: np.ndarray,
                bbox: tuple[float, float, float, float],
            ) -> Optional[str]:
                emb = body_embeddings.extract_body_embedding(
                    frame_bgr=frame_bgr, bbox=bbox,
                )
                if emb is None:
                    return None
                match = gallery_ref.match(
                    patient_id=pid, probe_embedding=emb,
                )
                if match is None:
                    return None
                log.info(
                    "reid.pre_match_hit patient=%s label=%s score=%.3f age=%.0fs",
                    pid, match.label, match.score, match.source_age_s,
                )
                return match.label

            pre_match_cb = _pre_match

        pipeline = BodyReIDPipeline(
            gallery=gallery,
            config=PipelineConfig(
                sample_fps=sample_fps,
                yolo_model=s.reid_yolo_model,
                yolo_confidence=s.reid_yolo_confidence,
                face_check_interval=s.reid_face_check_interval,
                hits_to_lock=s.reid_hits_to_lock,
                face_min_det_score=s.reid_face_min_det_score,
                face_checks_for_other=s.reid_face_checks_for_other,
                match_threshold_base=s.reid_match_threshold_base,
                insightface_model=s.reid_insightface_model,
            ),
            pre_match_callback=pre_match_cb,
        )

        # One bbox slot per input frame, aligned by index. Slots stay
        # None when the frame decode fails or no PATIENT track is
        # currently locked. The provider consumes this list directly
        # so the annotation it draws matches the frames it sends.
        per_frame_bboxes: list[tuple[float, float, float, float] | None] = []
        # Cache decoded BGR frames during the process loop so the
        # cross-clip gallery harvest step at the end can grab the
        # specific best-frame for each verified track without
        # re-decoding. Held only for the duration of this thread.
        decoded_frames: list[np.ndarray | None] = []
        for frame_idx, frame in enumerate(frames):
            try:
                jpeg = base64.b64decode(frame.data_base64)
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                log.exception("reid.frame_decode_failed frame=%d", frame_idx)
                per_frame_bboxes.append(None)
                decoded_frames.append(None)
                continue
            if img is None:
                per_frame_bboxes.append(None)
                decoded_frames.append(None)
                continue
            result = pipeline.process_frame(img, frame_idx)
            per_frame_bboxes.append(result.patient_bbox)
            decoded_frames.append(img)

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

        # Cross-clip body gallery harvest. For every track that ended
        # this clip with a verified label (PATIENT or OTHER), extract an
        # OSNet body embedding from its best frame (the one where YOLO
        # was most confident about the body box). Skip tracks that were
        # pre-labeled WITHOUT additional face evidence — those
        # embeddings would just re-confirm what's already in the gallery
        # and inflate dedup work. We harvest:
        #   - PATIENT tracks that were face-locked this clip (Path D)
        #   - OTHER tracks that were face-locked this clip (Path C)
        # The `face_locked` flag distinguishes "earned this clip's
        # verdict via face match" from "trusted a gallery pre-label."
        if self._body_gallery is not None and self._settings.reid_body_gallery_enabled:
            for state in pipeline._tracks.values():
                if not state.face_locked:
                    continue
                if state.label not in (LABEL_PATIENT, LABEL_OTHER):
                    continue
                if (
                    state.best_frame_idx is None
                    or state.best_bbox is None
                    or state.best_frame_idx >= len(decoded_frames)
                ):
                    continue
                best_img = decoded_frames[state.best_frame_idx]
                if best_img is None:
                    continue
                try:
                    body_emb = body_embeddings.extract_body_embedding(
                        frame_bgr=best_img, bbox=state.best_bbox,
                    )
                except Exception:
                    log.exception("reid.harvest_body_embedding_failed track=%d", state.track_id)
                    continue
                if body_emb is None:
                    continue
                # Confidence: use the face match score for PATIENT
                # (pipeline.best_score) or the YOLO det_score as a proxy
                # for OTHER (no face match score is meaningful for
                # OTHER tracks).
                conf = (
                    float(state.confidence) if state.label == LABEL_PATIENT
                    else float(state.best_det_score)
                )
                try:
                    self._body_gallery.harvest(
                        patient_id=patient_id,
                        embedding=body_emb,
                        label=state.label,
                        confidence=conf,
                    )
                except Exception:
                    log.exception("reid.body_gallery_harvest_failed track=%d", state.track_id)

        elapsed_ms = (time.monotonic() - started) * 1000.0
        result = IdentityResult(
            reid_ran=True,
            patient_in_clip=pipeline.has_any_patient_track,
            confidence=pipeline.best_score,
            best_frame_index=pipeline.best_frame_idx,
            face_count=pipeline.face_count,
            patient_bbox=pipeline.latest_patient_bbox,
            extracted_frames=list(frames),
            per_frame_bboxes=per_frame_bboxes,
            sample_fps=sample_fps,
            people_present_reid=pipeline.num_tracks > 0,
            people_count_reid=pipeline.max_concurrent_tracks,
            has_only_verified_other_tracks=pipeline.has_only_verified_other_tracks,
            faces_enrolled=True,
        )
        yolo_stats = pipeline.yolo_stats
        log.info(
            "reid.label_clip patient=%s frames=%d face_count=%d "
            "tracks=%d concurrent=%d patient_in_clip=%s "
            "people_present_reid=%s confidence=%.3f elapsed_ms=%.0f",
            patient_id, len(frames), pipeline.face_count, pipeline.num_tracks,
            pipeline.max_concurrent_tracks,
            result.patient_in_clip, result.people_present_reid,
            result.confidence, elapsed_ms,
        )
        # Separate line for YOLO telemetry so the parser can group
        # detection-quality metrics independently of identity outcome.
        # Critical for "Gemini saw a person, YOLO didn't" diagnosis.
        log.info(
            "reid.yolo_stats patient=%s frames=%d frames_with_dets=%d "
            "zero_det_pct=%.3f total_dets=%d max_conf=%.3f mean_conf=%.3f "
            "conf_threshold=%.2f",
            patient_id,
            yolo_stats["frames_processed"],
            yolo_stats["frames_with_detections"],
            yolo_stats["zero_detection_frame_pct"],
            yolo_stats["total_detections"],
            yolo_stats["max_confidence"],
            yolo_stats["mean_confidence"],
            yolo_stats["confidence_threshold"],
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
