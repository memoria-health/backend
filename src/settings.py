from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Nemara backend runtime settings. Loaded from environment + `.env`.

    Same shape as wolf's Settings, minus everything that's out of scope
    for the caregiver-app-facing skeleton (LiveKit, APNs, pairing, hawk,
    elephant, agent, intervention page, trigger shortening).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEMARA_",
        frozen=True,
        extra="forbid",
    )

    env: Literal["local", "test", "staging", "prod"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Internals (experimental, non-production) ─────────────────
    # Master switch for everything under `src/internals/`. Off by
    # default so no experimental routes mount in prod. Set
    # `NEMARA_ENABLE_INTERNALS=true` in a local `.env` to enable the
    # data-collection record-clip endpoint (see src/internals/).
    enable_internals: bool = Field(default=False)

    # ── Public base URL ─────────────────────────────────────────
    # Public URL the backend is reachable at from Pi cameras out in
    # customers' homes. The announce response embeds this into the
    # ingest_url it returns to a freshly-paired Pi so the Pi knows
    # where to POST its frame stream. In local dev set this to the
    # LAN IP+port (e.g. http://10.0.0.245:8002); in production set it
    # to the public HTTPS URL of the deployed backend.
    public_base_url: str = Field(min_length=1, default="http://localhost:8002")

    # ── Database ────────────────────────────────────────────────
    db_url: str = Field(min_length=1)
    db_pool_size: int = Field(ge=1, le=100, default=10)
    db_max_overflow: int = Field(ge=0, le=100, default=10)
    db_pool_recycle_seconds: int = Field(ge=60, le=86400, default=1800)

    # ── Supabase auth (JWT verification) ────────────────────────
    supabase_url: str = Field(min_length=1)
    supabase_publishable_key: str = Field(min_length=1)
    supabase_jwt_secret: str = Field(min_length=1)
    supabase_service_role_key: str = Field(min_length=1)

    # ── Detector pipeline (Phase 0+) ─────────────────────────────
    # Provider order is gemini → openai. Both keys must be present in
    # production; in tests we monkeypatch the provider callables.
    gemini_api_key: str = Field(default="")
    # Default model and fps shape match the dashboard's `.env.example`.
    # Video sampling rate sent to Gemini. "" = Gemini auto (~1fps); we pin 4 to
    # match the 4fps frame source + openai_frames_fps (lower than the old 5 to
    # cut frames-per-call cost).
    gemini_model: str = Field(default="gemini-3.5-flash")
    gemini_video_fps: str = Field(default="4")
    # 'low' | 'medium' | 'high' | '' (Gemini default). Lowercased at call.
    gemini_media_resolution: str = Field(default="")

    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-5.4")
    # Model used to COMPILE a detector's seeded prompt (one-off on Create).
    # Decoupled from the per-tick runtime model so compile can use a stronger
    # frontier/thinking model without slowing or inflating the cost of the hot
    # detection loop. Override via NEMARA_OPENAI_COMPILE_MODEL.
    openai_compile_model: str = Field(default="gpt-5.4")
    openai_frames_fps: int = Field(ge=1, le=30, default=4)

    # ── LiveKit (Phase 0+) ──────────────────────────────────────
    # Server-side token minting. The publish URL the wall phone connects
    # to is also returned to the camera via /v1/cameras/pair.
    livekit_url: str = Field(default="")
    livekit_api_key: str = Field(default="")
    livekit_api_secret: str = Field(default="")
    # LiveKit Cloud signs webhook deliveries with the same API
    # key/secret; the receiver verifies via livekit-api's TokenVerifier.
    livekit_webhook_api_key: str = Field(default="")  # falls back to livekit_api_key

    # ── Detector subscriber-bot (Phase 2) ───────────────────────
    # The bot lives inside the FastAPI process as an asyncio task tree
    # that subscribes to LiveKit rooms and runs the detector pipeline.
    # It's webhook-driven (joins rooms only when LiveKit Cloud tells us
    # a camera has connected), so it idles at zero cost until a wall
    # phone is paired and publishing.
    livekit_worker_tick_seconds: int = Field(ge=5, le=60, default=10)
    # Sliding ANALYSIS window the tick re-muxes to MP4 for the LLM. 11s (down
    # from 15s) trims frames per call to save cost: 11s × 4fps ≈ 44 frames vs
    # the old 75. (Distinct from the incident-clip post-roll, which is unchanged.)
    livekit_worker_window_seconds: int = Field(ge=5, le=60, default=11)
    # Target fps after publish-side downsampling in the frame buffer.
    # 4 matches `openai_frames_fps` + `gemini_video_fps`.
    livekit_worker_target_fps: int = Field(ge=1, le=15, default=4)
    # TTL for the per-camera motion_state cache. Wall phones publish
    # motion heartbeats every ~5s; missing/stale → motion_in_clip=false.
    livekit_motion_state_ttl_seconds: int = Field(ge=5, le=60, default=15)

    # ── Memoria Sensor app-present heartbeat (iPhone sensors) ────
    # A live iPhone sensor polls /v1/cameras/me/desired-state every 1s
    # while its app is open. PresenceSweeper records that heartbeat
    # in-memory and flips `cameras.app_present` on transitions, so the
    # caregiver can tell "disconnected (reconnectable)" from "app closed
    # (must reopen)". Pi cameras never poll this endpoint and are never
    # touched. `enabled=false` is the kill switch: the poll no-ops and
    # the sweeper doesn't run, so the column stays at its true default.
    sensor_presence_enabled: bool = Field(default=True)
    # A sensor unseen for this many seconds is considered "app closed"
    # (covers force-quit AND backgrounded — the sensor stops polling on
    # both). 8s ≈ 16 missed 0.5s polls — snappy enough that backgrounding
    # the sensor surfaces quickly, while still absorbing brief network
    # jitter. Stays well under the ~30s iOS background-suspend window.
    sensor_presence_stale_seconds: int = Field(ge=5, le=300, default=8)
    # How often the sweep compares the heartbeat cache to the DB and
    # flips app_present on transitions. Cheap (one scoped SELECT + only
    # transition UPDATEs), so a few seconds keeps the UI responsive.
    sensor_presence_sweep_seconds: int = Field(ge=2, le=60, default=5)

    # ── Pairing / camera api-key auth (Phase 0+) ────────────────
    # bcrypt rounds for the per-camera api_key_hash. 10 matches the
    # dashboard's bcryptjs default; bumping costs CPU on every clip
    # ingest (api-key verify path).
    bcrypt_rounds: int = Field(ge=4, le=15, default=10)
    # Pairing-code lifetime in seconds. Dashboard uses 10 minutes; we
    # keep the same for parity.
    pairing_code_ttl_seconds: int = Field(ge=60, le=3600, default=600)
    # App Review demo pairing code (NEMARA_DEMO_PAIRING_CODE). When set,
    # `POST /v1/cameras/pair` special-cases this exact code: it pairs the
    # camera row that holds it WITHOUT expiry or single-use clearing, so
    # an Apple reviewer can pair a fresh NemaraCam install repeatedly
    # against the seeded demo camera. None (unset) disables the path
    # entirely — production is the only env that sets it.
    demo_pairing_code: str | None = Field(default=None)

    # ── Supabase Storage (Phase 4) ──────────────────────────────
    # Two private buckets, signed URLs at read time. Bucket names are
    # overridable so test envs can point at scratch buckets. Buckets
    # are created manually in Supabase Studio per environment — there
    # is no SDK bucket-bootstrap step.
    storage_bucket_patient_photos: str = Field(default="patient-photos", min_length=1)
    storage_bucket_incident_clips: str = Field(default="incident-clips", min_length=1)
    # Per-payload size ceilings enforced at the storage layer. Photo
    # cap matches wolf (1 MiB). Clip cap fits a ~30s combined pre+post
    # MP4 at the encoder's bitrate with comfortable headroom.
    storage_max_photo_bytes: int = Field(ge=1024, le=10 * 1024 * 1024, default=1 * 1024 * 1024)
    storage_max_clip_bytes: int = Field(ge=1024, le=100 * 1024 * 1024, default=25 * 1024 * 1024)
    # Upload timeouts: photos finish in <1s on dev WiFi; clips on
    # cellular need more headroom. Both come from settings so a slow
    # CI VM can bump them via env without code changes.
    storage_upload_timeout_seconds: float = Field(gt=0, le=120, default=10.0)
    storage_clip_upload_timeout_seconds: float = Field(gt=0, le=300, default=60.0)
    # Default TTL for signed URLs handed to the caregiver app. 10 min
    # is comfortable for "tap event → watch clip" and short enough
    # that a shared URL goes stale fast.
    storage_signed_url_default_ttl_seconds: int = Field(ge=30, le=3600, default=600)

    # ── Re-ID (Phase 1+) ────────────────────────────────────────
    # Feature flag: when False, the enrollment API still accepts
    # photos but the inference pipeline does NOT consult the gallery
    # at tick time. Lets us roll the schema + API out without flipping
    # the detector behavior until we're ready.
    reid_enabled: bool = Field(default=False)
    # Hard timeout for the per-tick Re-ID call. If exceeded, the tick
    # proceeds with no identity_map and detectors fire identity-agnostic
    # (same as today). Larger than the current ~36s offline-CPU pace so
    # GPU-on-Railway can land well under, with CPU as a fallback for
    # local dev.
    # Hard timeout for the per-tick Re-ID pipeline. Body Re-ID
    # (YOLO + BoT-SORT + face checks on a ~75-frame tick) lands
    # around 6-10s on CPU; 15s gives reasonable headroom without
    # letting a stuck tick block the next one indefinitely (tick
    # interval is 10s but the loop is fire-and-forget).
    reid_timeout_seconds: float = Field(gt=0, le=60, default=15.0)
    # Base cosine threshold for face matches. The adaptive curve in
    # `services/reid/embeddings.py:effective_match_threshold` scales
    # this up for clear faces (~0.50) and down for unclear ones
    # (~0.24 floor); the value here is the "wall-cam normal" point.
    reid_match_threshold_base: float = Field(gt=0, lt=1, default=0.40)
    # Minimum SCRFD det_score for an enrollment selfie to be accepted.
    # Lower than the experiment's 0.6 because we now use adaptive
    # thresholds at match time — we want to accept a wider band of
    # selfie quality so the gallery has variety.
    reid_enrollment_min_det_score: float = Field(ge=0, le=1, default=0.4)
    # TTL on the per-patient gallery cache. The gallery is read once
    # at tick time and cached so repeat ticks for the same patient
    # don't hammer Postgres. 60s is long enough that a fresh
    # enrollment becomes visible within a minute, short enough that
    # harvested rows show up promptly too.
    reid_gallery_cache_ttl_seconds: int = Field(ge=1, le=600, default=60)
    # Cosine score required to harvest a face crop back into the
    # gallery as `source='harvested'`. Much higher than the lock
    # threshold so only highly-confident matches enrich the gallery.
    reid_harvest_enabled: bool = Field(default=True)
    reid_harvest_threshold: float = Field(gt=0, le=1, default=0.80)
    # Per-patient gallery cap. When inserting a harvested row would
    # exceed this, the oldest harvested row is evicted (selfies are
    # never evicted).
    reid_gallery_cap: int = Field(ge=20, le=1000, default=200)
    # Per-enrollment upload bounds. iOS captures 10-15 selfies; we
    # cap at 20 so a buggy client can't flood the gallery in one shot.
    reid_max_photos_per_enrollment: int = Field(ge=1, le=50, default=20)
    # InsightFace model bundle. `buffalo_l` is the production accuracy
    # tier (~280 MB); `buffalo_sc` is the lighter option (~40 MB) for
    # memory-constrained deploys. Override via env if Railway OOMs.
    reid_insightface_model: str = Field(default="buffalo_l")
    # Per-tick frame sampling for the inference path. 1 fps × 15s = 15
    # frames is plenty to catch a face glimpse without spending the
    # 75-frames-at-5fps budget the LLM uses. 480 px wide matches the
    # OpenAI provider's frame size, so the ffmpeg pass is reusable.
    # 5 fps gives BoT-SORT enough temporal continuity to keep track
    # IDs stable as the patient moves. 15s tick × 5 fps = 75 frames
    # per tick, processed sequentially through YOLO + tracker.
    reid_inference_sample_fps: int = Field(ge=1, le=15, default=5)
    reid_inference_sample_width: int = Field(ge=240, le=1920, default=480)
    # ── Body Re-ID pipeline (mirrors the experiment) ───────────────
    # YOLO weights. `yolo11n.pt` (nano, 2.5M params) is fastest but
    # misses partial / distant / occluded bodies — observed
    # 2026-05-30 with kitchen-counter helper clips where YOLO returned
    # 0 detections and Gemini still saw the person. `yolo11s.pt`
    # (small, ~9M params) has much better recall on those cases at
    # ~2x inference cost; default raised to `s` after that incident.
    # `yolo11m.pt` is a further step up if `s` still misses.
    reid_yolo_model: str = Field(min_length=1, default="yolo11s.pt")
    # YOLO confidence threshold for the person detector. Default
    # matches the experiment's PersonDetector(confidence=0.4). Once
    # the per-clip detection logs ship we can tune this from observed
    # confidence distributions on missed bodies. Lowering increases
    # recall at the cost of more false positives (re-ID filters most
    # of those, so the trade-off is cheap).
    reid_yolo_confidence: float = Field(gt=0, le=1, default=0.4)
    # Per-track face check cadence: every N frames per UNKNOWN track,
    # run SCRFD inside the person bbox. 5 frames @ 5 fps = 1s
    # between checks, same as the experiment.
    reid_face_check_interval: int = Field(ge=1, le=60, default=5)
    # Consecutive above-threshold face matches required to lock a
    # track as PATIENT. 2 means a single false-positive embedding
    # cannot lock the wrong identity.
    reid_hits_to_lock: int = Field(ge=1, le=10, default=2)
    # SCRFD detection confidence floor for a face check to count.
    # Faces below this are ignored (motion blur, partial occlusion).
    reid_face_min_det_score: float = Field(ge=0, le=1, default=0.5)
    # Face checks (with no patient match) before a track locks as
    # OTHER. Higher = slower OTHER locks but fewer wrong-OTHER calls
    # if the patient was briefly face-occluded early in the tick.
    reid_face_checks_for_other: int = Field(ge=1, le=20, default=5)
    # Per-frame Re-ID annotation of archived event clips. Display-only
    # post-processing; the detector pipeline never reads these. When
    # enabled, every event clip uploaded to Supabase Storage has a
    # thin green box tracking the patient body bbox on every native
    # frame so caregivers always see who triggered the alert. Adds
    # ~25-35s to the archive path (already async/fire-and-forget) —
    # the trade-off for matching nemara-reid-experiment's visual
    # quality.
    reid_clip_annotation_enabled: bool = Field(default=True)

    # ── Cross-clip body Re-ID gallery (Phase 2) ─────────────────
    # In-memory, per-patient. See cross_clip_body_gallery.py for the
    # full design. Defaults aim for "4-hour outfit cycle, conservative
    # OTHER threshold to avoid false-skipping the patient."
    reid_body_gallery_ttl_seconds: int = Field(ge=60, default=14400)  # 4h
    # Asymmetric thresholds: PATIENT is more liberal (false positive
    # = LLM runs with patient context anyway, mild), OTHER is more
    # conservative (false positive = LLM skipped on a patient clip,
    # severe).
    reid_body_match_patient_threshold: float = Field(ge=0, le=1, default=0.72)
    reid_body_match_other_threshold: float = Field(ge=0, le=1, default=0.80)
    # Intra-cluster diversity floor. When harvesting a new embedding
    # into an existing cluster, skip if it cosine-matches any existing
    # member above this threshold — keeps each cluster's buffer
    # diverse (we want N representative views of the same person, not
    # N near-identical embeddings from consecutive frames).
    reid_body_dedup_threshold: float = Field(ge=0, le=1, default=0.95)
    # Cluster-merge threshold. A new harvest joins an existing
    # same-label cluster if its cosine to that cluster (mean over the
    # cluster's embeddings) is >= this. Otherwise a new cluster is
    # seeded — i.e. we believe this is a different person. Must be
    # HIGHER than match_other_threshold so we only merge when we're
    # SURE it's the same identity (a wrong merge contaminates the
    # cluster mean and degrades future matches).
    reid_body_cluster_merge_threshold: float = Field(ge=0, le=1, default=0.85)
    # Max embeddings retained per cluster. FIFO eviction past the
    # cap — keeps the most recent N views. Higher = richer match
    # signal but more memory and slower per-match cosine sum.
    reid_body_max_embeddings_per_cluster: int = Field(ge=1, default=8)
    # Max clusters retained per label per patient. Evict the
    # least-recently-updated on overflow. PATIENT cluster count
    # should stay ~1 in practice; OTHER may grow to handle multiple
    # helpers/family members.
    reid_body_max_clusters_per_label: int = Field(ge=1, default=20)
    # Master switch — set False to disable the gallery entirely
    # (harvest + match become no-ops). Useful for A/B comparison.
    reid_body_gallery_enabled: bool = Field(default=True)

    # ── APNs (Phase 4) ──────────────────────────────────────────
    # Apple Developer credentials for HTTP/2 push. team_id +
    # apns_key_id identify the auth key file; apns_auth_key is the
    # PEM contents of the .p8 (ES256 ECDSA P-256 private key) as a
    # single string in the env (newlines preserved). bundle_id matches
    # the caregiver app's CFBundleIdentifier. environment is "sandbox"
    # for Xcode dev builds and "production" for TestFlight + App Store.
    # All default empty so tests / local dev don't need real APNs creds;
    # the push dispatcher noops when team_id is empty.
    apns_team_id: str = Field(default="")
    apns_key_id: str = Field(default="")
    apns_auth_key: str = Field(default="")
    apns_bundle_id: str = Field(default="com.elephantlabs.nemara.caregiver")
    apns_environment: Literal["sandbox", "production"] = "sandbox"
    # Token TTL refresh policy. Apple rejects JWTs older than 60 min;
    # we refresh slightly under that to leave headroom.
    apns_jwt_lifetime_seconds: int = Field(ge=60, le=3600, default=3300)

    # ── Clip archival (Phase 4) ─────────────────────────────────
    # Wait this many seconds after an event fires before snapshotting
    # the post-event buffer window. Matches livekit_worker_window_seconds
    # by default so pre.mp4 (T-15s..T) and post.mp4 (T..T+15s) are
    # contiguous. Test overrides bump it down to ~0.1 to avoid waiting.
    clip_post_event_wait_seconds: float = Field(gt=0, le=120, default=15.0)

    # ── Scheduled ("absence") detector runtime ──────────────────
    # How often the scheduler loop wakes to check for scheduled
    # detectors past their deadline. 60s is plenty — deadlines are
    # minute-granular and the single-fire guard makes re-checks cheap.
    scheduler_tick_seconds: int = Field(ge=15, le=300, default=60)

    # ── Email alerts (Resend) ───────────────────────────────────
    # When resend_api_key is empty the email sender no-ops (logs a
    # warning) so dev + tests run without the integration. email_from
    # must be a Resend-verified domain for real sends.
    resend_api_key: str = Field(default="")
    email_from: str = Field(default="onboarding@resend.dev")
    # Recipients for caregiver-submitted Contact Support messages from
    # the iOS app. Comma-separated in env (pydantic-settings parses).
    # replyTo is set to the caregiver's account email so hitting reply
    # in Gmail lands directly in their inbox.
    support_email_recipients: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["sunny.work.2002@gmail.com"]
    )
    # Recipients for caregiver-submitted product feedback (Settings →
    # Send feedback in the iOS app). Comma-separated in env. replyTo is
    # set to the caregiver's account email so the founders can reply
    # straight back. Distinct from support: storing the feedback row is
    # the source of truth, so an email failure must not fail the request.
    feedback_email_recipients: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["aryaman@memoria-health.com", "sunny@memoria-health.com"]
    )
    # Sender for feedback emails. Like email_from, must be a
    # Resend-verified domain for real sends; otherwise Resend rejects
    # the send (the feedback row is still persisted regardless).
    feedback_email_from: str = Field(default="feedback@memoria-health.com")

    @field_validator(
        "support_email_recipients",
        "feedback_email_recipients",
        mode="before",
    )
    @classmethod
    def _split_csv_recipients(cls, value: object) -> object:
        """Accept a comma-separated env string (e.g.
        "a@x.com,b@x.com") for the recipient lists. pydantic-settings
        would otherwise try to JSON-decode the raw env value for a
        `list[str]` field and choke on a plain CSV string. A value
        that's already a list (the default / programmatic construction)
        passes through untouched.
        """
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


def get_settings() -> Settings:
    return Settings()
