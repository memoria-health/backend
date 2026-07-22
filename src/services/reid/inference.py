"""Per-tick Re-ID labeller: YOLO person detect → BoT-SORT track →
periodic face check per track → state machine → locked PATIENT bbox.

Verbatim port of `nemara-reid-experiment/src/reid/pipeline/identifier.py`
adapted for the production tick budget (one ~11s clip per tick,
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
    LABEL_UNKNOWN,
    BodyReIDPipeline,
    PipelineConfig,
    TrackObservation,
)
from src.services.reid.cross_clip_body_gallery import CrossClipBodyGallery
from src.services.reid.remote_replay import (
    RemoteObservations,
    RemoteReplayPipeline,
)
from src.services.reid.sam_client import SamServiceClient
from src.services.reid.display_veto import ScreenMap, apply_veto
from src.settings import Settings

log = logging.getLogger(__name__)

# Stand-in gallery for attribution-less ticks (reid_enabled off, or no
# ready enrollment). Shape matches the real (N, EMBEDDING_DIM) stacked
# matrix so the pipeline's `gallery.size > 0` check is the single
# source of truth for "can face-match this tick."
_EMPTY_GALLERY = np.empty((0, reid_embeddings.EMBEDDING_DIM), dtype=np.float32)


@dataclass(frozen=True)
class IdentityResult:
    """Per-tick identity decision for a single patient.

    `reid_ran`: True iff the tracking pipeline executed and produced a
    verdict. False on timeout or crash ONLY — since v2 (P2) body
    tracking always runs; `reid_enabled` and enrollment gate face
    ATTRIBUTION, not tracking. The upstream gate in `camera_ingest`
    keys off this: when False, it bypasses identity-based suppression.
    Load-bearing safety invariant: a Re-ID infrastructure failure must
    NEVER silence the detector pipeline.

    `attribution_ran`: True iff the pipeline ran WITH a usable face
    gallery (reid_enabled AND ≥1 ready enrollment) — i.e. identity
    labels (PATIENT / OTHER / pre-labels) were even possible this
    tick. The suppression gate additionally requires this: a tick
    where attribution could not run must never suppress the LLM, no
    matter what the track labels claim.

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
    attribution_ran: bool = False
    # Per-track observations for the clip (v2 P2): one TrackObservation
    # per BoT-SORT track — label, confidence, and the full (frame_idx,
    # bbox) series in extracted-frame pixel space. The signals layer
    # scopes motion per track and masks env-motion with these; the
    # aggregate fields above stay for existing consumers. None iff the
    # pipeline didn't run (timeout/crash); empty tuple = ran, no people.
    tracks: tuple["TrackObservation", ...] | None = None
    # True iff num_tracks > 0 AND every track in the clip ended labeled
    # OTHER (face seen + match failed enough times to be confidently
    # non-patient). Drives the camera_ingest gate's "skip LLM when
    # everyone in frame is verified not-patient" branch — replaces the
    # old `face_count > 0` gate for the face-hidden-second-half-of-clip
    # case. False when any track is UNKNOWN, when patient is in clip,
    # or when no tracks at all.
    has_only_verified_other_tracks: bool = False

    # True iff attribution was ELIGIBLE this tick: reid_enabled AND at
    # least one ready face embedding (non-empty gallery). Drives the
    # camera_ingest presence-source branch. NOTE the v2 semantic shift:
    # with reid_enabled=False this reads False even if embeddings exist
    # in the DB — "enrolled" here means "the gallery was actually in
    # play." Independent of reid_ran: an eligible patient whose tick
    # timed out has faces_enrolled=True but reid_ran=False.
    faces_enrolled: bool = False

    # Frames the Re-ID pipeline already decoded + sampled (5 fps,
    # ≤480px). The frame-sampling LLM providers reuse these instead of
    # running their own ffmpeg extract — but ONLY on attribution ticks
    # (vlm_routing gates on attribution_ran: Re-ID's 5 fps exceeds the
    # providers' configured 4, and the reuse would silently inflate
    # image tokens for the always-on-tracking population whose bboxes
    # are all None anyway). None when the pipeline didn't run.
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

    # ── Input-quality admissibility (identity-persistence W2) ────
    # Measured on the decoded extraction frames: delivered width
    # (<400 ⇒ the SFU sent the 320×180 rung), mean Laplacian variance
    # over ≤3 sampled frames (blur), and how many frames the window
    # had at all. `quality_degraded` is the verdict: this tick's
    # pixels are inadmissible as evidence of ABSENCE — camera_ingest
    # freezes the state machine's close countdown on it, and the
    # identity lock (Batch B) holds through it. It never blocks
    # analysis: the scene call still runs.
    quality_width: int | None = None
    quality_sharpness: float | None = None
    quality_frames: int = 0
    quality_degraded: bool = False

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
            attribution_ran=False,
            tracks=None,
            has_only_verified_other_tracks=False,
            faces_enrolled=faces_enrolled,
        )


def assess_quality_degraded(
    *,
    width: int | None,
    sharpness: float | None,
    frames_n: int,
    min_width: int,
    sharpness_floor: float,
    min_frames: int,
) -> bool:
    """Identity-persistence W2: is this tick's input inadmissible as
    evidence of absence? Any floor violated ⇒ degraded. `width`/
    `sharpness` None (nothing decoded) with a sub-floor frame count is
    the degenerate thin-window case and counts as degraded."""
    if frames_n < min_frames:
        return True
    if width is not None and width < min_width:
        return True
    if sharpness is not None and sharpness < sharpness_floor:
        return True
    return False


def _bbox_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1.0, ua)


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
        # Lazily-constructed client for the SAM identity microservice
        # (reid_pipeline_backend="sam"). None until the first SAM tick;
        # holds one long-lived httpx connection pool once created.
        self._sam_client: SamServiceClient | None = None
        # Display-veto: per-camera screen-region maps + last screen
        # pass timestamps (2026-07-21 screen-face incident).
        self._screen_maps: dict[UUID, ScreenMap] = {}
        self._screen_last_pass: dict[UUID, float] = {}
        # Consecutive SAM-arm tick failures. Fail-open keeps detection
        # alive, but a dead pod/tunnel must be LOUD — reid coasting
        # dark at 2am looks identical to an empty house otherwise
        # (review S9).
        self._sam_fail_streak = 0

    async def label_clip(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
        camera_id: UUID | None = None,
        face_evidence: bool = True,
        harvest_hint: tuple | None = None,
        mode: str = "full",
    ) -> IdentityResult:
        """Run body tracking (+ optional face attribution) on a tick
        clip. Returns the empty result on timeout or infra error.

        `mode` is the patient-focus governor's verdict (2026-07-22):
          full    — identify at full fidelity (default)
          lean    — patient locked here, scene fully identified:
                    fewer frames, throttled faces
          minimal — patient held on ANOTHER camera: presence-only,
                    zero face work (no compute identifying guests)
          split   — nobody holds the patient and several rooms are
                    occupied: frame budget shared so total GPU demand
                    always fits (kills the saturation spiral)

        v2 (P2): body tracking ALWAYS runs — presence, motion scoping,
        and the subject filter all hang off tracks, so neither the
        `reid_enabled` flag nor a missing enrollment kills the pipeline
        anymore. The two gate ATTRIBUTION only: face matching happens
        iff `reid_enabled` AND the patient has a non-empty ready
        gallery; otherwise the pipeline runs gallery-less (tracks stay
        UNKNOWN, presence/count still populate).

        Hard timeout is `settings.reid_timeout_seconds`. The full body
        pipeline is heavier than the face-only path; calibrate that
        setting accordingly.
        """
        # The gallery is fetched (cached, TTL) only when the flag is
        # on — with the flag off we don't touch the DB and simply run
        # attribution-less. Eligibility is decided up front so the
        # verdict is right even if the pipeline below times out.
        if self._settings.reid_enabled:
            gallery = await self._get_gallery(patient_id)
        else:
            gallery = _EMPTY_GALLERY
        faces_enrolled = gallery is not None and gallery.size > 0
        try:
            result = await asyncio.wait_for(
                self._label_clip_inner(
                    mp4_bytes=mp4_bytes,
                    patient_id=patient_id,
                    camera_id=camera_id,
                    gallery=gallery if faces_enrolled else _EMPTY_GALLERY,
                    face_evidence=face_evidence,
                    harvest_hint=harvest_hint,
                    mode=mode,
                ),
                timeout=self._settings.reid_timeout_seconds,
            )
            self._sam_fail_streak = 0
            return result
        except asyncio.TimeoutError:
            log.warning(
                "reid.label_clip.timeout patient=%s timeout=%.1fs",
                patient_id, self._settings.reid_timeout_seconds,
            )
            self._note_sam_failure()
            return IdentityResult.empty(faces_enrolled=faces_enrolled)
        except Exception as exc:
            log.exception(
                "reid.label_clip.error patient=%s err=%s", patient_id, exc,
            )
            self._note_sam_failure()
            return IdentityResult.empty(faces_enrolled=faces_enrolled)

    def _note_sam_failure(self) -> None:
        """Escalate consecutive SAM-arm failures to ERROR: fail-open
        means reid quietly reads 'nobody present' while the pod or
        tunnel is down — someone has to notice. Detection itself is
        never affected (the fail-open invariant)."""
        if self._settings.reid_pipeline_backend != "sam":
            return
        self._sam_fail_streak += 1
        if self._sam_fail_streak >= 5 and self._sam_fail_streak % 5 == 0:
            log.error(
                "reid.sam_dark consecutive_failures=%d — pod/tunnel "
                "likely down; identity coasting empty (detection "
                "unaffected, fail-open)",
                self._sam_fail_streak,
            )

    async def _label_clip_inner(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: UUID,
        gallery: np.ndarray,
        camera_id: UUID | None = None,
        face_evidence: bool = True,
        harvest_hint: tuple | None = None,
        mode: str = "full",
    ) -> IdentityResult:
        # `gallery` is the caller-resolved attribution gallery — empty
        # array when attribution is ineligible this tick (flag off or
        # no enrollment). Body tracking runs either way and populates
        # `people_present_reid`/`people_count_reid` so the state
        # machine can close the camera when nobody is in frame, even
        # before the caregiver enrolls a patient face. `patient_in_clip`
        # stays False on the gallery-less path because identity
        # matching needs a gallery.
        faces_enrolled = gallery.size > 0

        # Patient-focus governor sampling (2026-07-22): frames are the
        # GPU cost, so the mode decides the frame budget. `minimal`
        # additionally kills ALL face work — a room the patient
        # verifiably isn't in never spends compute identifying guests
        # (Aryaman: "find the patient, waste nothing anywhere else").
        base = self._settings.reid_inference_sample_fps
        if mode == "minimal":
            face_evidence = False
            sample_fps = 1
        elif mode == "lean":
            sample_fps = self._settings.reid_inference_sample_fps_lean
        elif mode == "split":
            sample_fps = max(1, base // 2)
        else:
            sample_fps = base
        sample_width = self._settings.reid_inference_sample_width
        try:
            frames = await frame_extraction.extract_frames(
                mp4_bytes,
                fps=sample_fps,
                max_width=sample_width,
            )
        except RuntimeError as exc:
            log.warning("reid.frame_extract_failed err=%s", exc)
            return IdentityResult.empty(faces_enrolled=faces_enrolled)

        if not frames:
            return IdentityResult.empty(faces_enrolled=faces_enrolled)

        # SAM backend (v2, 2026-07-20): the pod is PERCEPTION ONLY —
        # SAM tracks + mask-clean SCRFD/ArcFace face observations come
        # back raw (no labels, no gallery on the pod), and the SAME
        # _run_pipeline_sync below replays them through the unchanged
        # brain: locks, quality floors, carry, cross-clip pools, and
        # harvests all run verbatim on both arms. A remote/network
        # failure raises here and label_clip's wrapper returns empty —
        # the Re-ID-never-silences-detection invariant holds across
        # the wire too.
        remote_obs: RemoteObservations | None = None
        if self._settings.reid_pipeline_backend == "sam":
            remote_obs = await self._fetch_remote_observations(
                frames, faces_enrolled=faces_enrolled,
                camera_id=camera_id, face_evidence=face_evidence,
            )

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
            remote_obs=remote_obs,
            harvest_hint=harvest_hint,
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

    def _get_sam_client(self) -> SamServiceClient:
        if self._sam_client is None:
            s = self._settings
            if s.reid_sam_request_timeout_s >= s.reid_timeout_seconds:
                # Timeout inversion (review C4): the outer label_clip
                # budget must also cover local replay — an inner wire
                # timeout at/above it means healthy slow ticks get
                # cancelled and the wire timeout is dead config.
                log.warning(
                    "reid.sam_timeout_inversion wire=%.0fs >= outer=%.0fs "
                    "— raise reid_timeout_seconds for the SAM arm",
                    s.reid_sam_request_timeout_s, s.reid_timeout_seconds,
                )
            self._sam_client = SamServiceClient(
                base_url=s.reid_sam_service_url,
                timeout_s=s.reid_sam_request_timeout_s,
                expected_face_model=s.reid_insightface_model,
            )
        return self._sam_client

    async def _fetch_remote_observations(
        self,
        frames: "list[ExtractedFrame]",
        *,
        faces_enrolled: bool,
        camera_id: UUID | None = None,
        face_evidence: bool = True,
    ) -> RemoteObservations:
        """v2 SAM wire hop: ship the tick's ALREADY-JPEG frames as-is
        (no decode/re-encode — the v1 overhead lesson), named by
        original frame index; the pod returns raw per-track
        observations in that same index space. Wire round-trip time is
        stashed into observations.timings_ms["wire"] for the per-step
        `reid.sam_steps` log in _run_pipeline_sync. Raises on any
        transport/remote failure (label_clip's wrapper fails open).

        Face budget: 0 when attribution is ineligible (no enrolled
        gallery — the brain would never consume the embeddings, review
        G1); otherwise the legacy arm's effective uncapped budget
        (n_frames // interval + 1) under the settings ceiling, so both
        arms see identical face-evidence economics (review G3)."""
        s = self._settings
        t0 = time.monotonic()
        jpegs: list[tuple[int, bytes]] = []
        for idx, frame in enumerate(frames):
            try:
                jpegs.append((idx, base64.b64decode(frame.data_base64)))
            except Exception:
                log.exception("reid.sam_frame_b64_failed frame=%d", idx)
        if not jpegs:
            raise RuntimeError("sam wire: no sendable frames")
        if faces_enrolled and face_evidence:
            face_cap = min(
                s.reid_sam_face_cap,
                len(jpegs) // s.reid_face_check_interval + 1,
            )
        else:
            # face_evidence=False = the ledger's binding-aware throttle
            # (optimization #1): everyone in view is bound and
            # streak-free, so this tick skips SCRFD + ArcFace + the
            # countersign entirely (~1/3 of an occupied tick's GPU).
            face_cap = 0
        # Display-veto map refresh: screens are furniture, so one
        # concept pass every couple of minutes per camera keeps the
        # region map fresh at ~0 marginal cost.
        want_screens = False
        if camera_id is not None and getattr(
            s, "reid_screen_veto_enabled", True,
        ):
            last = self._screen_last_pass.get(camera_id, 0.0)
            if time.monotonic() - last >= 120.0:
                want_screens = True
        obs = await self._get_sam_client().observe(
            jpegs,
            person_thr=s.reid_sam_person_thr,
            face_interval=s.reid_face_check_interval,
            face_cap=face_cap,
            detect_screens=want_screens,
        )
        obs.timings_ms["wire"] = (time.monotonic() - t0) * 1000.0
        if camera_id is not None:
            if want_screens:
                self._screen_last_pass[camera_id] = time.monotonic()
            smap = self._screen_maps.get(camera_id)
            if smap is None:
                smap = self._screen_maps[camera_id] = ScreenMap()
            if obs.screen_regions:
                smap.add([r[:4] for r in obs.screen_regions])
            apply_veto(obs, smap)
        return obs

    def _run_pipeline_sync(
        self,
        *,
        frames,
        gallery: np.ndarray,
        patient_id: UUID,
        sample_fps: int,
        remote_obs: "RemoteObservations | None" = None,
        harvest_hint: tuple | None = None,
    ) -> tuple[IdentityResult, Optional[_HarvestRequest]]:
        import cv2  # local: heavy, only needed on the inference path
        started = time.monotonic()

        s = self._settings

        # Attribution eligibility for this run. Gallery-less ⇒ face
        # checks are skipped inside the pipeline (`_has_gallery`), and
        # everything downstream that exists only to produce or consume
        # identity labels (body-gallery pre-match, harvests) is skipped
        # here — no OSNet inference for lookups that can't match.
        attribution = gallery is not None and gallery.size > 0

        # Cross-clip body gallery pre-match callback. Body pipeline
        # calls this once per new track. Extract OSNet body embedding
        # and match against the day's gallery for this patient. Sync
        # because body_pipeline runs in to_thread and the gallery uses
        # threading.Lock.
        pre_match_cb = None
        if (
            attribution
            and self._body_gallery is not None
            and self._settings.reid_body_gallery_enabled
        ):
            gallery_ref = self._body_gallery
            pid = patient_id

            def _pre_match(
                frame_bgr: np.ndarray,
                bbox: tuple[float, float, float, float],
            ) -> Optional[tuple]:
                emb = body_embeddings.extract_body_embedding(
                    frame_bgr=frame_bgr, bbox=bbox,
                )
                if emb is None:
                    return None
                match = gallery_ref.match(
                    patient_id=pid, probe_embedding=emb,
                )
                if match is None:
                    return (None, "gallery")
                log.info(
                    "reid.pre_match_hit patient=%s label=%s score=%.3f age=%.0fs",
                    pid, match.label, match.score, match.source_age_s,
                )
                return (match.label, "gallery")

            pre_match_cb = _pre_match

        pipeline_config = PipelineConfig(
            sample_fps=sample_fps,
            yolo_model=s.reid_yolo_model,
            yolo_confidence=s.reid_yolo_confidence,
            face_check_interval=s.reid_face_check_interval,
            hits_to_lock=s.reid_hits_to_lock,
            face_min_det_score=s.reid_face_min_det_score,
            face_checks_for_other=s.reid_face_checks_for_other,
            match_threshold_base=s.reid_match_threshold_base,
            insightface_model=s.reid_insightface_model,
            yolo_detect_floor=s.reid_yolo_detect_floor,
            track_birth_confidence=s.reid_track_birth_confidence,
            face_lock_min_px=s.reid_face_lock_min_px,
            face_lock_min_det=s.reid_face_lock_min_det,
            face_lock_score_single=s.reid_face_lock_score_single,
            face_lock_score_multi=s.reid_face_lock_score_multi,
            face_lock_margin=s.reid_face_lock_margin,
        )
        if remote_obs is not None:
            # SAM arm: same brain, remote eyes — every decision below
            # this line is the identical code path the legacy arm runs.
            pipeline = RemoteReplayPipeline(
                observations=remote_obs,
                gallery=gallery,
                config=pipeline_config,
                pre_match_callback=pre_match_cb,
            )
        else:
            pipeline = BodyReIDPipeline(
                gallery=gallery,
                config=pipeline_config,
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
        # re-decoding. Held only for the duration of this thread — and
        # ONLY when that harvest can actually run: with always-on
        # tracking (v2 P2) this loop executes for every camera every
        # tick, and ~55 retained BGR frames ≈ 15-20 MB per concurrent
        # attribution-less tick would be provably dead weight (T1
        # review finding). Index alignment is preserved via None slots.
        keep_decoded = (
            attribution
            and self._body_gallery is not None
            and self._settings.reid_body_gallery_enabled
        )
        decoded_frames: list[np.ndarray | None] = []
        # Per-frame (w, h), aligned with `frames` — the SFU can flap
        # resolution MID-clip, so any bbox normalization must use the
        # dims of the frame the bbox lives in (review F17).
        decoded_dims: list[tuple[int, int] | None] = []
        # Input-quality probes (W2): delivered width from the first
        # decodable frame; blur from Laplacian variance on ≤3 frames
        # spread across the window (~1ms each on ≤480px frames).
        quality_width: int | None = None
        decoded_ok = 0
        sharp_samples: list[float] = []
        quality_probe_idxs = {0, len(frames) // 2, len(frames) - 1}
        for frame_idx, frame in enumerate(frames):
            try:
                jpeg = base64.b64decode(frame.data_base64)
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                log.exception("reid.frame_decode_failed frame=%d", frame_idx)
                per_frame_bboxes.append(None)
                decoded_frames.append(None)
                decoded_dims.append(None)
                continue
            if img is None:
                per_frame_bboxes.append(None)
                decoded_frames.append(None)
                decoded_dims.append(None)
                continue
            decoded_ok += 1
            if quality_width is None:
                quality_width = int(img.shape[1])
            if frame_idx in quality_probe_idxs:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                sharp_samples.append(
                    float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                )
            result = pipeline.process_frame(img, frame_idx)
            per_frame_bboxes.append(result.patient_bbox)
            decoded_frames.append(img if keep_decoded else None)
            decoded_dims.append((int(img.shape[1]), int(img.shape[0])))

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

        # Cross-clip body gallery harvest — MULTI-POSE (W4): for every
        # track that ended this clip with a face-locked verified label,
        # harvest up to 4 body embeddings SPREAD across the track
        # (first / one-third / two-thirds / best-YOLO frame) so one
        # face lock teaches the pool that outfit from several angles —
        # the 00:27Z soak measurement showed a single fresh outfit
        # embedding lifting the body match from ~0.65 to 0.94.
        # Quality bars (W5): the lock must carry a face at/above the
        # lock pixel floor, and PATIENT harvests additionally need the
        # match score ≥ reid_harvest_min_confidence — marginal locks
        # can no longer poison the pool (the lunch green-box lesson).
        s_cfg = self._settings
        if (
            attribution
            and self._body_gallery is not None
            and self._settings.reid_body_gallery_enabled
        ):
            # Binding-aware harvest (2026-07-21): the ledger vouches
            # for ONE track by geometry while its PATIENT binding is
            # in face-afterglow (face-grade evidence ≤60s ago; pool-
            # provenance bindings never emit a hint — no pool→pool
            # self-feeding). That track may donate body crops without
            # a face THIS clip — how the pool learns far/seated
            # appearances (the couch finding). Guard: exactly one
            # candidate near the hint, else nobody harvests.
            hint_state = None
            if harvest_hint is not None:
                hb = harvest_hint[0]
                cands = [
                    st for st in pipeline._tracks.values()
                    if len(st.bbox_series) >= 3
                    and not st.face_locked
                    and st.label == LABEL_UNKNOWN
                    and _bbox_iou(st.bbox_series[0][1], hb) > 0.3
                ]
                if len(cands) == 1:
                    hint_state = cands[0]
            for state in pipeline._tracks.values():
                is_hint = state is hint_state
                if not state.face_locked and not is_hint:
                    continue
                if not is_hint and state.label not in (
                    LABEL_PATIENT, LABEL_OTHER,
                ):
                    continue
                if not is_hint and state.face_px_max < s_cfg.reid_face_lock_min_px:
                    continue
                if (
                    not is_hint
                    and state.label == LABEL_PATIENT
                    and float(state.confidence) < s_cfg.reid_harvest_min_confidence
                ):
                    continue
                if is_hint:
                    # Trust anchor is the afterglow envelope, not this
                    # clip's (absent) face.
                    conf = float(harvest_hint[1])
                else:
                    conf = (
                        float(state.confidence)
                        if state.label == LABEL_PATIENT
                        else float(state.best_det_score)
                    )
                series = state.bbox_series
                if not series:
                    continue
                pick_idxs: list[int] = []
                for frac in (0.0, 1 / 3, 2 / 3):
                    pick_idxs.append(
                        series[int(frac * (len(series) - 1))][0],
                    )
                if state.best_frame_idx is not None:
                    pick_idxs.append(state.best_frame_idx)
                bbox_at = dict(series)
                harvested_n = 0
                for fi in dict.fromkeys(pick_idxs):     # dedupe, keep order
                    if harvested_n >= 4:
                        break
                    if fi >= len(decoded_frames):
                        continue
                    img = decoded_frames[fi]
                    bbox = bbox_at.get(fi) or (
                        state.best_bbox if fi == state.best_frame_idx
                        else None
                    )
                    if img is None or bbox is None:
                        continue
                    try:
                        body_emb = body_embeddings.extract_body_embedding(
                            frame_bgr=img, bbox=bbox,
                        )
                    except Exception:
                        log.exception(
                            "reid.harvest_body_embedding_failed track=%d",
                            state.track_id,
                        )
                        continue
                    if body_emb is None:
                        continue
                    try:
                        if self._body_gallery.harvest(
                            patient_id=patient_id,
                            embedding=body_emb,
                            label=(
                                LABEL_PATIENT if is_hint else state.label
                            ),
                            confidence=conf,
                        ):
                            harvested_n += 1
                            if is_hint:
                                log.info(
                                    "reid.afterglow_harvest tid=%d "
                                    "conf=%.2f",
                                    state.track_id, conf,
                                )
                    except Exception:
                        log.exception(
                            "reid.body_gallery_harvest_failed track=%d",
                            state.track_id,
                        )

        elapsed_ms = (time.monotonic() - started) * 1000.0
        quality_sharpness = (
            float(np.mean(sharp_samples)) if sharp_samples else None
        )
        quality_degraded = assess_quality_degraded(
            width=quality_width,
            # DECODED count, not input count (review F9): a window of
            # 12 undecodable frames carries zero visual evidence and
            # must read degraded.
            sharpness=quality_sharpness,
            frames_n=decoded_ok,
            min_width=s.reid_quality_min_width,
            sharpness_floor=s.reid_quality_sharpness_floor,
            min_frames=s.reid_quality_min_frames,
        )
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
            attribution_ran=attribution,
            tracks=tuple(pipeline.track_observations()),
            has_only_verified_other_tracks=pipeline.has_only_verified_other_tracks,
            faces_enrolled=attribution,
            quality_width=quality_width,
            quality_sharpness=quality_sharpness,
            quality_frames=decoded_ok,
            quality_degraded=quality_degraded,
        )
        yolo_stats = pipeline.yolo_stats
        log.info(
            "reid.label_clip patient=%s frames=%d face_count=%d "
            "tracks=%d concurrent=%d patient_in_clip=%s "
            "people_present_reid=%s confidence=%.3f elapsed_ms=%.0f "
            "q_width=%s q_sharp=%s q_degraded=%s",
            patient_id, len(frames), pipeline.face_count, pipeline.num_tracks,
            pipeline.max_concurrent_tracks,
            result.patient_in_clip, result.people_present_reid,
            result.confidence, elapsed_ms,
            quality_width,
            f"{quality_sharpness:.0f}" if quality_sharpness is not None else "-",
            quality_degraded,
        )
        if remote_obs is not None:
            # Per-step SAM-arm breakdown (the "measure every step"
            # requirement): wire = backend b64 + POST + parse + network;
            # pod_* = the pod's own clock; replay_local = this sync run
            # (decode + brain replay + harvests). Compare directly
            # against legacy's elapsed_ms above.
            t = remote_obs.timings_ms
            log.info(
                "reid.sam_steps patient=%s wire_ms=%.0f pod_decode_ms=%.0f "
                "pod_sam_ms=%.0f pod_face_ms=%.0f pod_total_ms=%.0f "
                "replay_local_ms=%.0f frames_sent=%d pod_frames=%d "
                "dets=%d max_prob=%.2f",
                patient_id, t.get("wire", 0.0), t.get("decode", 0.0),
                t.get("sam", 0.0), t.get("face", 0.0), t.get("total", 0.0),
                elapsed_ms, len(frames), remote_obs.frames_seen,
                remote_obs.det_count, remote_obs.max_prob,
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
