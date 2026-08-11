from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Extra seconds the per-camera frame/audio buffers retain BEYOND the
# longest window any consumer asks for. Absorbs scheduling slack and the
# frames that keep arriving while the post-roll task sleeps, so the post
# snapshot's oldest frame is never trimmed before it's read.
_CLIP_BUFFER_MARGIN_SECONDS = 2.0


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
    gemini_model: str = Field(default="gemini-3.6-flash")
    # Second Gemini tier: tried by the ROUTING chain on the same
    # native-video path when the primary gemini_model fails, before
    # dropping to another provider. Aryaman 2026-07-27: primary is the
    # benchmark winner (3.6-flash), fallback the newest lite tier —
    # outages degrade quality, never availability. Env:
    # NEMARA_GEMINI_FALLBACK_MODEL.
    #
    # 2026-08-03 (Aryaman): moved off 3.5-flash-lite. `.env` pins the
    # PRIMARY to 3.5-flash-lite, so an identical fallback made the chain
    # a no-op — a 3.5-lite outage had nowhere to go but the next
    # provider. A different tier is the whole point of the hop.
    gemini_fallback_model: str = Field(default="gemini-3.1-flash-lite")
    gemini_video_fps: str = Field(default="4")
    # Per-frame token budget Gemini spends on each image/video frame — the real
    # "how much detail does the VLM see" dial (independent of the pixels we
    # upload; Gemini downsamples to this budget either way). 'low' | 'medium' |
    # 'high' | '' (Gemini default). Defaulted to 'high' for max detail on small/
    # subtle detections (distant doors, hand positions, fine motion); costs more
    # tokens/latency per frame — dial down per-account if needed. Lowercased at call.
    gemini_media_resolution: str = Field(default="high")
    # How hard the model reasons before answering. 'minimal' | 'low' |
    # 'medium' | 'high' | '' (send nothing, i.e. the model's own default).
    #
    # MEASURED 2026-08-01 on gemini-3.5-flash-lite over real event clips:
    # the model does NOT think by default, and 'minimal'/'low' are no-ops
    # on it — all three returned thoughts_token_count = 0. 'medium' and
    # 'high' both reason (~110-310 and ~160-770 thought tokens), costing
    # roughly +0.5-1.5s. So on lite the only live rungs are medium/high.
    # Set 'high' because the un-thought verdicts were not good enough
    # (Aryaman 2026-08-01). Caveat worth knowing before trusting this
    # dial to fix accuracy: on the four known choking false-positive
    # clips, thinking did NOT reduce false fires (off 1/4, medium 1/4,
    # high 1/4) — one rep each, so noise-dominated, and it measures
    # suppression only, never recall.
    #
    # Thought tokens bill at the OUTPUT rate. That is a small share of the
    # bill here — a scene call is ~13k input vs ~100 output — but it is not
    # free, and it is already accounted for: `usage["output"]` below adds
    # candidates + thoughts, with `usage["thoughts"]` carried separately.
    gemini_thinking_level: str = Field(default="high")

    # EXPERIMENT: tint moved regions red on the VLM clip (frame-diff overlay) so
    # the model can see small/sustained motion — a vibrating hand / tremor — that
    # is invisible in raw sampled frames. Off by default.
    # (`motion_overlay_enabled` removed 2026-08-01. It was a global toggle
    # for what is a per-call artifact property — the RAPID/overlay channel
    # tints its own crop, nothing else does. It is now an explicit
    # `motion_overlay=` argument to gemini_provider, passed by the executor
    # as `plan.channel == "overlay"`. As a global it was actively harmful:
    # left true in .env, it added the red-tint note to every clean scene
    # and person prompt.)

    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-5.4")
    # Model used to COMPILE a detector's seeded prompt (one-off on Create).
    # Decoupled from the per-tick runtime model so compile can use a stronger
    # frontier/thinking model without slowing or inflating the cost of the hot
    # detection loop. Override via NEMARA_OPENAI_COMPILE_MODEL.
    openai_compile_model: str = Field(default="gpt-5.4")
    openai_frames_fps: int = Field(ge=1, le=30, default=4)

    # ── OpenRouter (brokers Qwen3-VL etc.) ───────────────────────────
    # OpenAI-compatible endpoint, so the provider reuses the frame-based
    # path (OpenRouter does NOT accept native video). Used when a caregiver
    # is routed to a non-Gemini/OpenAI model. Model ids are namespaced
    # slugs, e.g. "qwen/qwen3-vl-32b-instruct".
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="qwen/qwen3-vl-32b-instruct")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")

    # ── DashScope (Alibaba Model Studio — Qwen3-VL, NATIVE video+fps) ──
    # SANDBOX: go direct to Alibaba instead of OpenRouter. The OpenAI-
    # compatible endpoint accepts a real `video` content part with an `fps`
    # field, so Qwen3-VL applies native temporal video understanding instead
    # of a flat list of independent images. US (Virginia) region base URL;
    # model ids are plain, e.g. "qwen3-vl-32b-instruct".
    dashscope_api_key: str = Field(default="")
    dashscope_model: str = Field(default="qwen3-vl-32b-instruct")
    dashscope_base_url: str = Field(
        default="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    )
    # Frames are sent as a video at this fps (sibling of the `video` array),
    # extracted at this width (1920 = 1080p source) and JPEG quality (ffmpeg
    # -q:v; 2 best .. 31 worst). Higher res/quality = more detail/specificity
    # but more tokens + latency + cost — tune for the analysis you need.
    dashscope_frames_fps: int = Field(ge=1, le=10, default=5)  # DashScope caps fps at 10
    dashscope_frame_max_width: int = Field(ge=240, le=3840, default=854)  # 480p (16:9)
    dashscope_jpeg_quality: int = Field(ge=2, le=31, default=3)
    # Cap on how many frames we send (we sample at fps, then trim to this,
    # keeping the most recent). 5fps × 11s = 55 (full 11s window, no blind spot).
    dashscope_max_frames: int = Field(ge=1, le=768, default=55)

    # Per-account VLM routing: a caregiver may carry a `vlm_provider`
    # override ("gemini" | "openai" | "openrouter" | "dashscope"); when
    # null/unset we fall back to this global default. So you can move
    # individual families onto a different model without a deploy (just flip
    # the caregivers column). See services/detection/vlm_routing.py.
    default_vlm_provider: str = Field(default="gemini")

    # ── LiveKit (Phase 0+) ──────────────────────────────────────
    # Server-side token minting. The publish URL the wall phone connects
    # to is also returned to the camera via /v1/cameras/pair.
    livekit_url: str = Field(default="")
    livekit_api_key: str = Field(default="")
    livekit_api_secret: str = Field(default="")
    # LiveKit Cloud signs webhook deliveries with the same API
    # key/secret; the receiver verifies via livekit-api's TokenVerifier.
    livekit_webhook_api_key: str = Field(default="")  # falls back to livekit_api_key

    # ── Room-derived privacy (2026-07-27) ────────────────────────
    # Master switch for the automatic room classifier: when a camera
    # with `room_classified_at IS NULL` starts delivering frames, one
    # clean-frame VLM call classifies its room (bathroom/bedroom →
    # privacy_enabled=true, locked). Default ON everywhere; turn OFF
    # (NEMARA_AUTO_PRIVACY_MODE=false) only in internal/dev
    # environments so we can test with live feeds. Never user-visible —
    # this is an env setting, not a product toggle. See
    # docs/plans/2026-07-27-room-privacy-and-clip-censoring.md and
    # services/room_classifier.py.
    auto_privacy_mode: bool = Field(default=True)
    # Shared secret guarding the support-only /internal/* routes (e.g.
    # POST /internal/cameras/{id}/reclassify). Callers present it in the
    # X-Internal-Key header. Fails CLOSED outside env=local/test when
    # empty — an unset key in prod means every /internal call is
    # rejected, never open. Env: NEMARA_INTERNAL_SERVICE_KEY.
    internal_service_key: str = Field(default="")

    # ── Sensitivity screening + clip censoring (2026-07-27) ──────
    # Master switch for the per-clip sensitivity screen: EVERY event
    # clip from EVERY camera is screened before upload, and sensitive
    # clips are censored (person-mask blur + audio stripped) — company
    # baseline, no per-account setting. Default ON everywhere; turn OFF
    # (NEMARA_SENSITIVITY_SCREENING_ENABLED=false) only in internal/dev
    # environments, mirroring auto_privacy_mode. Never user-visible.
    # See docs/plans/2026-07-27-room-privacy-and-clip-censoring.md and
    # services/detection/sensitivity.py.
    # NOTE: honored ONLY when env is local/test — in any other env a
    # false value is IGNORED (screening stays on, loud warning logged),
    # mirroring how internal_service_key fails closed by env. See
    # clip_archival._screening_active.
    sensitivity_screening_enabled: bool = Field(default=True)
    # Confidence floor for a "not sensitive" clearance. A well-formed
    # sensitive=false verdict BELOW this self-reported confidence is
    # treated as sensitive (censored): doubt → censor, in code, not
    # just in the prompt. sensitive=true verdicts censor at ANY
    # confidence. Env: NEMARA_SENSITIVITY_CONFIDENCE_FLOOR.
    sensitivity_confidence_floor: float = Field(ge=0.0, le=1.0, default=0.6)
    # PINNED model for the sensitivity classifier. Deliberately NOT the
    # per-account vlm_provider routing — a safety-critical classifier
    # must behave identically for every account, and QA/eval is run
    # against exactly one model. Same Gemini flash family as the
    # detection default. Env: NEMARA_SENSITIVITY_MODEL.
    sensitivity_model: str = Field(default="gemini-3.6-flash")
    # Hard wall-clock cap on the tier-1 (person-mask) censor, which runs
    # in an ISOLATED child process (censor.py `__main__`). 2026-07-28: a
    # tier-1 YOLO inference hit a native torch/OpenMP deadlock and froze
    # the ENTIRE backend process — event loop included — in the local
    # Docker VM; `asyncio.to_thread` cannot contain a native hang, so
    # tier 1 now runs out-of-process and is `kill()`ed at this deadline,
    # falling back to the pure-ffmpeg full-frame tier in-process.
    #
    # 180s. This is a HANG DETECTOR, not a budget.
    #
    # MEASURED 2026-08-01 in the local Docker VM (single-threaded CPU,
    # 1080p, 8 fps source), because the module docstring's "a few
    # ms-to-tens-of-ms per frame" is far too optimistic here:
    #     11.1s clip /  88 frames -> 19s
    #     26.3s clip / 210 frames -> 41s
    # i.e. ~195 ms per frame, scaling linearly. A 26s clip is the
    # TYPICAL case, so a healthy pass is ~40s and the longest clips seen
    # (~34s) land near 55s.
    #
    # A 45s deadline was tried and reverted the same day: it sits below
    # the normal-case cost, so ordinary clips would have timed out and
    # degraded to the coarse full-frame blur — manufacturing the exact
    # failure the deadline exists to catch. 180s is ~3x the realistic
    # worst case, which is the right shape for a hang detector.
    # If faster failover is wanted, make tier 1 cheaper (downscale
    # before segmentation, or sample fewer frames) rather than cutting
    # the deadline into the working range.
    # Env: NEMARA_CENSOR_TIMEOUT_SECONDS.
    censor_timeout_seconds: float = Field(gt=0, le=900, default=180.0)

    # Inline recorder masking: run segmentation every Nth frame and carry
    # the (extra-dilated) mask across the rest. At 15fps there is only
    # ~66ms per frame and segmentation costs tens of ms, so segmenting
    # every frame does not fit the budget. 5 → segment 3x a second; a
    # person moves very little in 200ms and the wider dilation covers it.
    # Lower is safer and more expensive. Env:
    # NEMARA_CENSOR_STREAM_SEGMENT_EVERY.
    censor_stream_segment_every: int = Field(ge=1, le=60, default=5)

    # How often a recording camera asks whether the moment is sensitive.
    # This is the ONLY thing that turns masking on — room class does not.
    # It reuses the MP4 the tick already built for the detector, so the
    # cost is one extra VLM call, not another encode. 3 → roughly every
    # 30s of a recording session. Lower reacts sooner and costs more.
    # Env: NEMARA_DATA_COLLECTION_SENSITIVITY_EVERY_N_TICKS.
    data_collection_sensitivity_every_n_ticks: int = Field(
        ge=1, le=60, default=3
    )

    # Whether a research clip whose video was masked keeps its audio.
    # ON by default: a masked clip is exactly where sound carries the most
    # of what happened, and the corpus is opt-in. This is OURS to flip,
    # not a per-family setting — hence a setting and not a column.
    #
    # It only decides whether we KEEP the audio. It can never cause any to
    # be captured: `patients.audio_capture_enabled` severs the track at
    # the publisher for facilities where recording audio is illegal, so
    # when that is false no audio reaches us to keep.
    # Env: NEMARA_DATA_COLLECTION_KEEP_SENSITIVE_AUDIO.
    data_collection_keep_sensitive_audio: bool = True

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

    # ── RAPID recipe (detection v2) ──────────────────────────────
    # Recipe params for RAPID-bucket detectors (seizure): person-cropped,
    # motion-overlay-only clip at `rapid_fps` on `rapid_model`, behind the
    # cheap rapid-energy pre-gate — run by the v2 executor as a regular
    # bucket call, no per-detector flag (Aryaman 2026-07-14, Ledger row 5;
    # the P0 standalone lane and its `rapid_seizure_enabled` flag are gone).
    # While a camera's loadout carries a RAPID detector, the frame buffer
    # retains `rapid_fps` natively; every other consumer decimates back to
    # `livekit_worker_target_fps` via snapshot at_fps, so main-call cost
    # and clip cadence are unchanged. Validated recipe (conv 8/10 R,
    # 1/30 FP): docs/architecture/detection-architecture-v2.md §1.4.
    rapid_fps: int = Field(ge=2, le=15, default=8)
    rapid_model: str = Field(default="gemini-3-flash-preview")

    # ── Layer-1 signals (detection v2, P2) ───────────────────────
    # Backend-owned motion floor: minimum active-pixel fraction on the
    # tick_signals full-frame diff map for a window to count as "in
    # motion". Unifies `motion_in_clip` (SAD packet is a hint OR'd with
    # this reading) and the closed-state wake, so gateway cameras that
    # never publish SAD packets still get motion gating. Whole-frame
    # scale — deliberately below the person-scoped RAPID floor (0.03);
    # the map's blur + diff threshold already reject sensor noise.
    # Soak dial: sig_env_motion is stashed on every analyses row.
    # 2026-07-19 recalibrated 0.02 → 0.003 against live data: Aryaman's
    # collar-tug (the agreed minimum detectable motion) measured
    # 0.003-0.007 track-active fraction; map noise baseline sits at
    # ~0.0007, so 0.003 keeps ~4x margin. Watch empty-room ticks for
    # false "backend" motion; noise held 0.0006-0.0007 on both cameras
    # when set.
    signals_motion_min_active_fraction: float = Field(ge=0.0, le=1.0, default=0.003)

    # ── Detection v2 scheduler + packer (P3) ─────────────────────
    # Master flag: OFF = today's single-call tick, byte-for-byte. ON =
    # L1 scheduler → L2 packer path (per-bucket calls). Landed parity-
    # first (D2, Aryaman 2026-07-13): with the dials below at their
    # defaults, flag-on runs the SAME detectors as flag-off — only the
    # packing changes. The cost inversion (action buckets only under
    # motion) is a post-soak settings change, never a code change.
    detection_v2_scheduler_enabled: bool = Field(default=False)
    # soak 0.5: run reid (identity) every tick but skip ALL VLM/Gemini
    # calls — including the always-on scene call — so identity continuity
    # can be validated for ~$0. Default off; set NEMARA_REID_SOAK_ONLY=true.
    reid_soak_only: bool = Field(default=False)
    # Base URL of the identity service (the GPU-host sidecar that
    # synthesizes per-tick IdentityResults from /state — the ONLY
    # identity provider since the legacy brain's deletion 2026-07-26).
    # How long a GPU node may go without polling /v1/identity/roster before
    # its homes are released to whichever node polls next. MUST exceed the
    # node's poll interval with margin, or nodes will steal homes from each
    # other on a slow poll and every home pays a ~30-60s re-ground for
    # nothing. 90s against a 30s poll gives two misses of slack.
    identity_roster_stale_after_s: int = Field(ge=30, le=900, default=90)
    identity_state_url: str = Field(default="http://host.docker.internal:8012")
    # HTTP proxy used ONLY for calls to identity nodes. On Railway that is
    # tailscaled's --outbound-http-proxy-listen, which is how a container
    # with no TUN device reaches a tailnet host.
    #
    # An HTTP proxy rather than tailscaled's SOCKS one: httpx needs the
    # `socksio` package for socks5:// and we do not ship it, so an
    # ALL_PROXY=socks5://... would raise ImportError on the first identity
    # call — at runtime, in production, on a path that otherwise fails open.
    #
    # SCOPED to this client rather than set as a process-wide ALL_PROXY,
    # because that would also route Gemini, OpenAI, LiveKit, Supabase and
    # Resend through tailscaled. Only identity lives on the tailnet.
    #
    # It also fixes DNS: MagicDNS names do not resolve in the container's
    # resolver, but a CONNECT carries the hostname and the proxy resolves it.
    identity_proxy_url: str = Field(default="")
    # Bearer token for /state (gpu/identity/contract.py). Empty = no
    # auth (local dev, host-only traffic). ONE env name on BOTH sides:
    # NEMARA_IDENTITY_STATE_TOKEN. The service used to read
    # NEMARA_IDENTITY_TOKEN, so setting one name left the other side
    # open with no error anywhere; it now reads this name and only
    # falls back to the old one with a warning.
    identity_state_token: str = Field(default="")
    # Per-track motion floor for the person_motion branch gate.
    # 0.0 = parity (any present track qualifies).
    scheduler_person_motion_floor: float = Field(ge=0.0, le=1.0, default=0.0)
    # env_motion floor for the environment branch gate. 0.0 = parity.
    scheduler_env_motion_floor: float = Field(ge=0.0, le=1.0, default=0.0)
    # When True, person-branch detectors are skipped on ticks where the
    # tracker positively saw nobody (and, with a raised motion floor,
    # when no track clears it). False = parity: person detectors run
    # regardless, exactly like today's monolithic call.
    scheduler_person_requires_track: bool = Field(default=False)
    # duty_cycle='sampled' detectors run every k-th tick. 1 = parity.
    scheduler_sample_every_k: int = Field(ge=1, le=60, default=1)
    # Crowd hardening (D3): more than this many concurrent tracks ⇒ no
    # per-person crops this tick — every detector runs full-scene.
    scheduler_crowd_fallback_tracks: int = Field(ge=1, le=16, default=4)

    # ── Soak observability ───────────────────────────────────────
    # One annotated identity-evidence JPEG per analyzed tick with
    # people in it (track boxes + PATIENT/OTHER/UNKNOWN labels +
    # face-lock marker) — the visual audit for tracking/face-ID.
    # Ring-pruned per camera; written under out/ (host-mounted).
    soak_id_frames_enabled: bool = Field(default=False)
    soak_id_frames_dir: str = Field(default="out/id_frames")

    # ── Detection v2 decision layer (P4) ─────────────────────────
    # OFF = legacy decision logic byte-for-byte (30s global dedup,
    # on-sight state entry, absence runner's own counting). ON = the
    # v2 primitives own the decisions (parallel-systems rule, one
    # owner per flag state): windowed-count state entry (M-of-T from
    # eventing JSONB; NULL = M=1 = today), per-detector
    # cooldown_seconds, k-of-N confirmation re-checks in the executor,
    # absence counting via the shared qualifying_tick_count query.
    detection_v2_decision_enabled: bool = Field(default=False)

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

    # ── Data-collection recorder + Backblaze B2 (research opt-in) ──
    # The in-process recorder uploads one MP4 per presence session to a
    # Backblaze B2 bucket (S3-compatible). All default empty/safe so
    # local dev + tests run without B2 creds — the uploader no-ops when
    # creds are absent (see services/data_collection/b2.b2_configured),
    # so the recorder degrades to "record but don't upload" rather than
    # crashing. endpoint_url + region MUST match the region the bucket
    # was created in (e.g. us-west-004, eu-central-003) — B2 application
    # keys are region-scoped.
    b2_key_id: str = Field(default="")
    b2_app_key: str = Field(default="")
    b2_bucket: str = Field(default="memoria-data")
    b2_endpoint_url: str = Field(default="https://s3.us-west-004.backblazeb2.com")
    b2_region: str = Field(default="us-west-004")
    # Data-residency: India patients' research clips route to their OWN bucket
    # (services/region.b2_bucket_for) so India collected video is isolated and
    # liftable as a unit. Empty until an India bucket exists — while empty, IN
    # clips fall back to `b2_bucket` with a warning (see region.b2_bucket_for).
    # NOTE: if the India bucket is created in a different B2 region, add
    # matching b2_endpoint_url_in / b2_region_in and thread them through
    # region.b2_bucket_for; today B2 has no India POP so a single endpoint is
    # assumed.
    b2_bucket_in: str = Field(default="")
    # Hard ceiling on a single clip file. ~15min of H.264 720p @2.5Mbps
    # is ~280 MB; 1.5 GB leaves generous headroom for the roll + audio.
    data_collection_max_clip_bytes: int = Field(ge=1024, default=1_500_000_000)
    # Max simultaneous recording sessions per worker process. Over the
    # cap a new session is not started (logged + skipped), so a surge of
    # active cameras can't exhaust CPU/disk. Shard the fleet to scale
    # past it (see the recorder plan's Scaling note).
    data_collection_max_concurrent: int = Field(ge=1, le=128, default=8)

    # ── Re-ID (Phase 1+) ────────────────────────────────────────
    # Gates face ATTRIBUTION only (detection v2, P2): when False, the
    # enrollment API still accepts photos but tick-time inference never
    # consults the gallery — tracks stay unlabeled and the identity
    # gate can't suppress anything. Body tracking (YOLO + BoT-SORT)
    # runs UNCONDITIONALLY either way — presence, motion scoping, and
    # the subject filter all hang off tracks, so no flag may kill it.
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
    # `services/identity/embeddings.py:effective_match_threshold` scales
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
    # Per-enrollment upload bounds. The guided-capture iOS flow sweeps a
    # full pose cone and auto-selects the best frame per direction, so it
    # sends ~15-30 (frontal + up to 2 per dial arc). Cap at 40 for headroom
    # while still stopping a buggy client from flooding the gallery.
    reid_max_photos_per_enrollment: int = Field(ge=1, le=50, default=40)
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
    # LEAN sampling (2026-07-22 headroom fix): when a camera has the
    # patient locked AND everyone in view is identified, it samples
    # fewer frames — the binding is already stable and just needs light
    # continuity, so this frees GPU for a co-occupied camera still
    # working to identify someone (Aryaman: "enough headroom for both
    # cameras when the patient isn't confirmed in one of them"). SAM
    # per-frame cost is the bottleneck, so fewer frames = proportional
    # headroom. Full sampling resumes the instant anyone is unbound.
    reid_inference_sample_fps_lean: int = Field(ge=1, le=15, default=3)
    reid_inference_sample_width: int = Field(ge=240, le=1920, default=480)
    # ── Tracker-eval rig (SAM3 vs SAM3+EdgeTAM) ─────────────────────
    # LEAN EVAL MODE: strips every optimization/hidden hold so we test
    # the RAW tracker + identity. When True: governor forced to FULL
    # (no adaptive sampling throttle), recent-lane OFF, scene-extension
    # OFF, and sampling runs at reid_inference_sample_fps regardless of
    # occupancy. Face + body ID stay on — that's the whole point. The
    # only per-run flag is NEMARA_TRACKER (sam3 | edgetam). Off in prod.
    reid_lean_eval_mode: bool = Field(default=False)
    # Frames/sec fed to the tracker under lean eval. The pod costs ~275ms
    # per frame, and the reid window is ~4s, so fps=5 → ~20 frames → ~5.5s
    # observe, which fits the 10s tick even with a second camera. Going to
    # 8-10 (~32-40 frames) buys smoothness but risks observe > tick →
    # pileups/429s that DROP ticks and contaminate the run. Raise it only
    # for solo scenarios where one camera has the whole budget.
    reid_lean_eval_sample_fps: int = Field(ge=1, le=15, default=5)
    # Structured JSONL trace for the eval: per-frame TRK, IDEVENT (with
    # de-floored real scores), pool-write HARVEST, per-stage LATENCY.
    # The soak dashboard + tools/trace_tracks.py read this one file.
    reid_trace_enabled: bool = Field(default=False)
    reid_trace_path: str = Field(default="out/track_trace.jsonl")
    # Tracker-eval arm label (sam3 | edgetam). The pipeline reads this
    # straight from os.environ so the pod and backend always agree on the
    # label; it is modeled here ONLY so a host process that loads the .env
    # FILE (the soak dashboard) doesn't trip extra="forbid" on the
    # NEMARA_TRACKER key. (pydantic filters os.environ extras but validates
    # .env-file extras — which is why the container was fine and the
    # dashboard 500'd.)
    tracker: str = Field(default="edgetam")
    # SAM identity microservice (GPU pod) — the ONLY perception arm since
    # the legacy in-Docker YOLO+BoT-SORT arm was removed (2026-07-22).
    reid_sam_service_url: str = "http://localhost:8000"
    # NOTE: on the SAM arm, reid_timeout_seconds must exceed the wire
    # round trip (pod SAM ≈ 210ms/frame) + local replay — inference
    # logs a loud warning when this timeout ordering is inverted.
    reid_sam_request_timeout_s: float = Field(gt=0, default=30.0)
    # SAM concept-detection threshold for the "person" prompt (pod).
    reid_sam_person_thr: float = Field(ge=0.0, le=1.0, default=0.4)
    # CEILING on per-track pod face detect+embed attempts per clip.
    # The actual request cap is computed per tick as
    # min(this, n_frames // face_check_interval + 1) — the legacy
    # arm's effective (uncapped) budget — so both arms see the same
    # face-evidence budget and OTHER-lock economics (review G3); 0 is
    # sent when the patient has no enrolled gallery, skipping all pod
    # face work (review G1). GPU faces are ~25ms/attempt, so the
    # ceiling is safety, not tuning.
    reid_sam_face_cap: int = Field(ge=1, le=64, default=32)
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

    # ── Identity persistence, Batch A (2026-07-19) ──────────────
    # See docs/plans/2026-07-19-identity-persistence.md.
    #
    # W1 — bounded tick queue: when the 2-generation in-flight cap
    # sheds a tick, its WINDOW is snapshotted at shed time and the
    # tick runs as soon as a slot frees — unless it has aged past
    # this. Rationale (Aryaman): the action may live only in the shed
    # window; a verdict 15s late beats a dropped window. 0 disables
    # (old drop-on-shed behavior).
    # 15 keeps the age ≤ the frame buffer's ~17 s retention so a
    # queued tick's moment is never mostly evicted (review F5), and
    # bounds the frozen payload's lifetime.
    scheduler_tick_queue_max_age_s: float = Field(ge=0, default=15.0)
    # W2 — input-quality admissibility floors. A tick below any floor
    # is "degraded": analysis still runs (the LLM reads blur far
    # better than YOLO) but the tick cannot testify toward EMPTINESS —
    # the state machine's close countdown is frozen so a blurry/180p
    # window can never close an occupied room or (Batch B) release an
    # identity lock. Floors calibrated on 2026-07-19 soak data:
    # sharpness (Laplacian variance on ≤480px extraction frames) ran
    # median 4745 on tracked frames vs 2436 on tracking-lost frames;
    # width <400 means the SFU delivered the 320×180 rung.
    # Sharpness floor re-derived 2026-07-19 evening from the outcome
    # curve (patient-found rate by sharpness band at 1280w occupied):
    # ≥3000 → 100%, 2500-2975 → 94%, 2160-2500 → 37%. The cliff is at
    # ~2500; 3200 branded dim-but-working evening ticks degraded.
    # Laplacian variance is luminance-sensitive — expect this absolute
    # floor to be WRONG under IR (no IR data exists yet; first IR rows
    # land tonight). Durable fix = per-camera rolling baseline.
    reid_quality_min_width: int = Field(ge=0, default=400)
    reid_quality_sharpness_floor: float = Field(ge=0, default=2500.0)
    reid_quality_min_frames: int = Field(ge=0, default=6)
    # W6 — still-person track continuation (ByteTrack two-stage
    # association). YOLO now returns candidates down to the detect
    # floor so BoT-SORT's low-confidence second pass can keep an
    # EXISTING track alive on a seated/lying person (observed conf
    # 0.10-0.35 at 360p). Track BIRTH stays at the birth confidence —
    # weak boxes can continue a proven track, never create one.
    reid_yolo_detect_floor: float = Field(gt=0, le=1, default=0.10)
    reid_track_birth_confidence: float = Field(gt=0, le=1, default=0.40)

    # ── Identity persistence (2026-07-19 → 2026-07-21) ──────────
    # Track-anchored identity: the TrackLedger (persistent geometric
    # track stitching + identity bindings — confirm once, follow the
    # track, audit passively) IS the identity layer; the IdentityLock
    # ENTER/HOLD/PARK/RELEASE machine and tracklet carry were deleted
    # 2026-07-21. See docs/plans/2026-07-21-track-anchored-identity.md.
    # How long a patient binding survives its track being unseen on
    # GOOD ticks before retiring (degraded ticks never count). 45s —
    # the 30s experiment regressed live (2026-07-22): under shedding
    # and normal in-and-out movement it retired bindings the grace
    # exists to bridge. Display staleness is the truth fields' job.
    reid_binding_retire_after_s: float = Field(ge=0, default=15.0)
    # Fix B (2026-07-22, re-applied): cap on how long the scene-presence
    # layer may extend a PATIENT binding with NO real body re-sighting.
    # Bounds the phantom lock — an EdgeTAM mask that drifted onto the couch
    # keeps num_tracks>0 -> people_present -> scene_present, which unbounded
    # held a departed patient present for 890s over an empty room. A still
    # patient in bed is re-grounded by SAM3 within this window. 0 = old
    # unbounded behaviour.
    reid_scene_extension_max_s: float = Field(ge=0, default=150.0)
    # W5 — face-lock quality floors: dim faces may VOTE, only quality
    # faces may LOCK (the lunch green-box fix: 15-30px guest faces
    # cleared the quality-ADAPTIVE match bar and locked PATIENT).
    # Height in extracted-frame pixels (~480px wide frames).
    reid_face_lock_min_px: float = Field(ge=0, default=36.0)
    reid_face_lock_min_det: float = Field(ge=0, le=1, default=0.6)
    # Wrong-green margin split (2026-07-19 late soak: the aunt case).
    # Dim light puts the true patient AND a blood relative in the same
    # 0.38-0.51 face-score band — a flat bar cannot separate them and
    # the green box alternated. Locks now need: single-person clip
    # score ≥ _single; multi-person clip score ≥ _multi AND a margin
    # gap over every rival track's best face score. The 0.40 base
    # remains the VOTE bar (hits, OTHER-block) only.
    reid_face_lock_score_single: float = Field(ge=0, le=1, default=0.45)
    reid_face_lock_score_multi: float = Field(ge=0, le=1, default=0.55)
    reid_face_lock_margin: float = Field(ge=0, le=1, default=0.15)
    # W4 — flat per-label embedding POOL (replaces merge-threshold
    # clusters, which fragmented into singletons: same-person
    # cross-pose cosines 0.6-0.75 can never clear a 0.85 merge bar).
    # Match = mean of top-3 cosines against the pool.
    reid_body_pool_max: int = Field(ge=1, default=64)
    # Harvest bar: only face locks at/above this match score feed the
    # pool (with the W5 pixel floor) — ends silent gallery poisoning
    # by marginal locks. 0.40 calibrated on soak #2: Aryaman's six
    # genuine bedroom locks scored 0.402-0.495 (all pass) while the
    # lunch green-box locks sat ~0.34 (all blocked). 0.45 would have
    # starved half the legitimate harvests (review finding). Stays
    # 0.40 after the 2026-07-19 margin split: anti-poison moved
    # upstream into the LOCK bars (a near-tie face can no longer lock,
    # so it can no longer harvest) — raising this too would only
    # starve legitimate dim solo harvests.
    reid_harvest_min_confidence: float = Field(ge=0, le=1, default=0.40)
    # Harvest hygiene (2026-07-19 late soak): the pool churned at up to
    # 3 harvests/tick, FIFO-evicting good entries within minutes.
    # Rate-cap per (patient,label); protect the top-scored "golden"
    # entries from eviction so the pool keeps its best evidence.
    reid_gallery_harvest_min_interval_s: float = Field(ge=0, default=20.0)
    # One lock event may deposit this many poses inside a window (the
    # multi-pose design teaches an outfit from several angles at once);
    # the interval then gates the NEXT event, not the burst itself.
    reid_gallery_harvest_burst: int = Field(ge=1, default=4)
    reid_gallery_golden_count: int = Field(ge=0, default=8)
    # Negative (OTHER) pool is deliberately SHORT-LIVED (Aryaman
    # 2026-07-19): body embeddings are outfit embeddings — a guest's
    # entry is meaningful for the visit that produced it and becomes
    # pure wrong-mute risk after (a stale negative can verify the
    # PATIENT as not-patient, releasing locks + muting his room).
    # Positive pool: 4h refreshed continuously; negatives: 2h, small.
    reid_other_pool_max: int = Field(ge=1, default=32)
    reid_other_pool_ttl_seconds: float = Field(ge=0, default=7200.0)

    # ── JPEG-in-buffer + native quality (2026-07-19) ────────────
    # Storage quality for buffered frames (see frame_codec.py). The
    # per-round H.264 the VLM sees is lossier than this, so 88 adds no
    # accuracy-relevant loss while shrinking 2K frames ~6-10×.
    frame_jpeg_quality: int = Field(ge=50, le=100, default=88)
    # Which simulcast layer the detector bot subscribes to. "high" =
    # the top layer (the camera's full resolution). "medium" was the
    # soak-1 crash-night triage — kept selectable for rollback.
    livekit_subscribe_quality: str = Field(default="high")

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

    # ── Event gate: operator paging + verdict links ─────────────
    # The gate holds an alert and pages an operator (not a caregiver —
    # see src/services/event_gate.py). Paging goes through PagerDuty
    # rather than our own APNs because their app already holds Apple's
    # critical-alerts entitlement and does repeat-until-acknowledged.
    #
    # `pagerduty_routing_key` is the Integration Key of a PagerDuty
    # service (Service → Integrations → Events API v2). Empty disables
    # paging: the gate still holds and still auto-releases, but nobody
    # is told, so the gate silently does nothing. Never leave it empty
    # in an environment where any patient has event_gate_enabled on.
    pagerduty_routing_key: str = Field(default="")
    pagerduty_events_url: str = Field(
        default="https://events.pagerduty.com/v2/enqueue"
    )

    # Public origin of THIS backend, used to build the verdict link that
    # travels inside the page. Must be reachable from a phone on cellular
    # — a LAN address here produces a link that works only at your desk.
    event_gate_public_base_url: str = Field(default="")

    # HMAC key for verdict links. One link grants sight of a single
    # household's clip and the power to suppress a single real alert, so
    # treat it as a live credential: rotating it kills every outstanding
    # link. Empty disables link minting (fail closed — the page then
    # carries no action link rather than an unsigned one).
    event_gate_link_secret: str = Field(default="")

    # How long a verdict link stays valid. Deliberately longer than the
    # auto-release so a link opened late renders "already dispatched"
    # instead of a signature error, which would look like a bug.
    event_gate_link_ttl_seconds: int = Field(ge=30, le=900, default=180)

    # Bearer token for the operator-only queue endpoint. Cross-tenant by
    # design — it reads every household. Never ship it in a client.
    ops_api_token: str = Field(default="")

    # Cadence of the durable auto-release sweeper. The in-process timer
    # gives an exact 60s release but dies with the process; this sweeper
    # is what stops a deploy mid-window from stranding an alert forever.
    event_gate_sweep_seconds: int = Field(ge=5, le=300, default=15)

    # ── Clip archival (Phase 4) ─────────────────────────────────
    # Wait this many seconds after an event fires before snapshotting
    # the post-event buffer window. pre.mp4 (T-window..T) and post.mp4
    # (T..T+wait) are concatenated into the incident clip; the buffer
    # retention (frame_buffer_retention_seconds, below) is sized to cover
    # THIS so post.mp4 is contiguous with pre.mp4 regardless of how short
    # the analysis window gets. Test overrides bump it to ~0.1 to skip waiting.
    clip_post_event_wait_seconds: float = Field(gt=0, le=120, default=15.0)
    # Frame rate of the incident clip (2026-07-27). Decoupled from the
    # 4fps LLM cadence so caregivers get watchable playback: the clip
    # is a SINGLE encode (pre+post frames + one contiguous PCM audio
    # track in one ffmpeg pass — no concat seam; audio restored
    # 2026-07-28, censored clips still strip it). Cost of raising
    # this: the frame buffer must RETAIN this rate (see
    # frame_buffer_target_fps), so steady-state memory scales
    # linearly, and the censor's per-frame YOLO pass scales with it.
    clip_fps: int = Field(ge=1, le=15, default=8)

    @property
    def frame_buffer_target_fps(self) -> int:
        """Native retention rate for the per-camera FrameBuffer.

        DERIVED (like frame_buffer_retention_seconds) so the buffer can
        never retain less than its fastest consumer needs: the LLM tick
        decimates to `livekit_worker_target_fps` via at_fps either way,
        but the incident clip snapshots at `clip_fps`.
        """
        return max(self.livekit_worker_target_fps, self.clip_fps)

    @property
    def frame_buffer_retention_seconds(self) -> float:
        """Seconds of history the per-camera frame + audio buffers retain.

        DERIVED, not a separate knob, so it can never drift below what a
        consumer needs. Two consumers read these buffers:
          * the LLM tick — snapshots `livekit_worker_window_seconds`
          * the incident-clip post-roll — snapshots
            `clip_post_event_wait_seconds` AFTER the event fires

        The buffer must out-live the LARGER of the two. Sizing it to only
        the analysis window (the bug this fixes) starved the post-roll:
        cutting the window 15s→11s while the post-roll stayed 15s made the
        post snapshot silently drop its oldest ~4s, leaving a visible gap
        at the pre/post concat seam. Tying retention to clip length here
        makes that drift impossible.
        """
        return (
            max(
                float(self.livekit_worker_window_seconds),
                float(self.clip_post_event_wait_seconds),
            )
            + _CLIP_BUFFER_MARGIN_SECONDS
        )

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide Settings, built ONCE.

    Every call used to construct a fresh Settings, and Settings reads
    `.env` from disk — so every request that injected settings did a
    synchronous file read and parse inside an async handler. Env is
    fixed for a process's lifetime in every environment we run
    (container, Railway dyno, local uvicorn), so caching is safe and
    the read disappears.

    Tests that need to change env must call
    `get_settings.cache_clear()` after patching (or construct
    `Settings()` directly, which is unaffected).
    """
    return Settings()
