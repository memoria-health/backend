from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Empty fps = let Gemini choose its own video sampling rate.
    gemini_model: str = Field(default="gemini-3.5-flash")
    gemini_video_fps: str = Field(default="")
    # 'low' | 'medium' | 'high' | '' (Gemini default). Lowercased at call.
    gemini_media_resolution: str = Field(default="")

    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-5.4")
    openai_frames_fps: int = Field(ge=1, le=30, default=5)

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
    # Sliding window the tick re-muxes to MP4 for Gemini. 15s matches
    # the dashboard's clip length.
    livekit_worker_window_seconds: int = Field(ge=5, le=60, default=15)
    # Target fps after publish-side downsampling in the frame buffer.
    # 5 matches `openai_frames_fps`; Gemini accepts the same rate.
    livekit_worker_target_fps: int = Field(ge=1, le=15, default=5)
    # TTL for the per-camera motion_state cache. Wall phones publish
    # motion heartbeats every ~5s; missing/stale → motion_in_clip=false.
    livekit_motion_state_ttl_seconds: int = Field(ge=5, le=60, default=15)

    # ── Pairing / camera api-key auth (Phase 0+) ────────────────
    # bcrypt rounds for the per-camera api_key_hash. 10 matches the
    # dashboard's bcryptjs default; bumping costs CPU on every clip
    # ingest (api-key verify path).
    bcrypt_rounds: int = Field(ge=4, le=15, default=10)
    # Pairing-code lifetime in seconds. Dashboard uses 10 minutes; we
    # keep the same for parity.
    pairing_code_ttl_seconds: int = Field(ge=60, le=3600, default=600)

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
    reid_timeout_seconds: float = Field(gt=0, le=30, default=5.0)
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
    reid_inference_sample_fps: int = Field(ge=1, le=15, default=1)
    reid_inference_sample_width: int = Field(ge=240, le=1920, default=480)

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


def get_settings() -> Settings:
    return Settings()
