"""Per-tick Re-ID labeller. Decides whether the patient appears in
the clip's video, with what confidence, and at what frame.

Design choice for v0: face-only identification. We do NOT run YOLO
person detection or BoT-SORT body re-ID at production tick time.
Reasons:

  1. The detector tick already has frames extracted (or extractable
     via `frame_extraction.extract_frames`). A face-detection pass on
     a handful of those frames is much cheaper than the full
     YOLO+BoT-SORT+OSNet pipeline.
  2. The output we need is a single boolean ("patient present in this
     clip") plus a confidence score and best-frame index. We don't
     need per-track stability across the clip — the LLM will reason
     over the whole clip whether or not we label individual tracks.
  3. Removes ~80% of the model footprint (no YOLO weights, no OSNet
     weights, no BoT-SORT tracker state).

If we later need per-track labeling for SoM-style multi-person scenes,
we can layer the full pipeline back in. v0 ships the simpler thing.

Threading: each call runs `embed_image_bytes` per sampled frame
concurrently via `asyncio.gather`. The InsightFace model is shared
across calls (lazy-init singleton). Total per-tick latency bound by
the slowest face check, not the sum.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.repos import face_embedding_repo
from src.services.detection import frame_extraction
from src.services.reid import embeddings as reid_embeddings
from src.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityResult:
    """Per-tick identity decision for a single patient.

    `patient_in_clip`: True iff at least one sampled frame had a face
    that matched the gallery above the adaptive threshold.

    `confidence`: best cosine score over all sampled frames. 0.0 when
    no face matched. Surfaced in the analyses row's
    `detector_fields.identity_confidence` for diagnostics.

    `best_frame_index`: which sampled frame produced the best match.
    Useful for future SoM-style overlays; currently informational.

    `face_count`: total faces detected across all sampled frames.
    Useful for "how busy is this clip" reasoning later.
    """
    patient_in_clip: bool
    confidence: float
    best_frame_index: int | None
    face_count: int

    @classmethod
    def empty(cls) -> "IdentityResult":
        return cls(
            patient_in_clip=False,
            confidence=0.0,
            best_frame_index=None,
            face_count=0,
        )


class ReIDInference:
    """In-process Re-ID labeler. One instance per FastAPI process —
    loaded by `main.create_app` at startup so the first tick doesn't
    pay the model-load cost.

    The class owns the per-patient gallery cache. The InsightFace
    model itself is owned by the `embeddings` module (lazy module-
    level singleton).
    """

    def __init__(self, sessionmaker: async_sessionmaker, settings: Settings) -> None:
        self._sessionmaker = sessionmaker
        self._settings = settings
        # gallery cache: patient_id -> (cached_at_monotonic, gallery_array)
        # gallery_array is the stacked (N, 512) float32 matrix ready for
        # cosine match. Empty array signals "no enrollment yet" and the
        # caller falls back to identity-agnostic.
        self._gallery_cache: dict[UUID, tuple[float, np.ndarray]] = {}
        self._cache_lock = asyncio.Lock()

    async def label_clip(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
    ) -> IdentityResult:
        """Run Re-ID on a 15-second tick clip. Returns the empty
        result on timeout, infrastructure error, or no-enrollment.

        The hard timeout is `settings.reid_timeout_seconds`. If
        exceeded, we cancel and return empty so the caller (the tick
        body) proceeds with identity-agnostic behavior.
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
            # Crashes degrade to identity-agnostic, never to silence.
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
            # No enrollment yet: identity-agnostic (today's behavior).
            return IdentityResult.empty()

        # Sample frames at a low fps — face detection doesn't need 15
        # fps; 1 fps over a 15s clip = 15 frames, more than enough to
        # catch a glimpse of the patient if they're in frame.
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

        started = time.monotonic()
        # Embed each frame concurrently. embed_image_bytes already wraps
        # the sync work in asyncio.to_thread.
        face_hits = await asyncio.gather(
            *(
                reid_embeddings.embed_image_bytes(
                    base64.b64decode(f.data_base64),
                    model_name=self._settings.reid_insightface_model,
                )
                for f in frames
            ),
            return_exceptions=True,
        )

        best_score: float = -1.0
        best_frame_idx: int | None = None
        face_count = 0
        any_match = False

        for idx, hit in enumerate(face_hits):
            if isinstance(hit, BaseException):
                log.warning("reid.embed_failed frame=%d err=%s", idx, hit)
                continue
            if hit is None:
                continue
            face_count += 1
            match = reid_embeddings.topk_mean_cosine_match(hit.embedding, gallery)
            threshold = reid_embeddings.effective_match_threshold(
                self._settings.reid_match_threshold_base,
                hit.det_score,
                hit.face_pixel_area,
            )
            log.debug(
                "reid.frame patient=%s frame=%d det_score=%.3f score=%.3f thr=%.3f",
                patient_id, idx, hit.det_score, match.score, threshold,
            )
            if match.score >= threshold and match.score > best_score:
                best_score = match.score
                best_frame_idx = idx
                any_match = True

        elapsed_ms = (time.monotonic() - started) * 1000.0
        result = IdentityResult(
            patient_in_clip=any_match,
            confidence=max(0.0, best_score),
            best_frame_index=best_frame_idx,
            face_count=face_count,
        )
        log.info(
            "reid.label_clip patient=%s frames=%d face_count=%d "
            "patient_in_clip=%s confidence=%.3f elapsed_ms=%.0f",
            patient_id, len(frames), face_count,
            result.patient_in_clip, result.confidence, elapsed_ms,
        )
        return result

    # ── Gallery cache ───────────────────────────────────────────

    async def _get_gallery(self, patient_id: UUID) -> np.ndarray:
        """Return the (N, 512) stacked gallery matrix for `patient_id`.

        Cached per-patient with a TTL from settings. Cache miss reads
        from `patient_face_embeddings` (status=ready). Empty array on
        no-enrollment.
        """
        now = time.monotonic()
        ttl = self._settings.reid_gallery_cache_ttl_seconds
        async with self._cache_lock:
            cached = self._gallery_cache.get(patient_id)
            if cached is not None and (now - cached[0]) < ttl:
                return cached[1]

        # Load from DB outside the cache lock — the DB call could take
        # tens of ms and we don't want to serialize ticks for different
        # patients behind one DB read.
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
            self._gallery_cache[patient_id] = (time.monotonic(), stacked)
        return stacked

    def invalidate_gallery(self, patient_id: UUID) -> None:
        """Drop the cached gallery for a patient. Called from the
        harvester when it appends a new row so the next tick picks it
        up immediately rather than waiting for TTL."""
        self._gallery_cache.pop(patient_id, None)
