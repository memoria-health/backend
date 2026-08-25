# Room-Derived Privacy + Sensitive-Clip Censoring — Implementation Plan

> Status: IMPLEMENTED 2026-07-27 (all 5 repos, uncommitted working trees; 3 ⚑ decisions resolved as recommended — all-persons blur, pinned classifier model, support-only reclassify). Reviewed (3-agent adjudicated pass) + all findings fixed. See "Deploy checklist" below before shipping.
> Date: 2026-07-27
> Scope: nemara-backend (primary), nemara-caregiver-app, nemara-web-app, nemara-camera-app, nemara-gateway
> Branches: build on current checked-out branches — backend + iOS `feat/gateway-camera-ownership`, web `feat/web-customs-visibility`, camera-app `fix/camera-local-signing`, gateway `main`.

## Product model (locked)

- Privacy is **per-camera, derived from the room the camera watches**. No user privacy toggles.
- On first connect, one clean frame → VLM classifies `bathroom | bedroom | other`.
  - `bathroom`, `bedroom` → **Private**, locked. No live video or audio, ever.
  - `other` → **Full** (live view + clips), locked.
- **Clips are retained for Private cameras** (video + audio) — *unless* the clip contains a
  sensitive moment, in which case: person-mask blur + audio stripped, raw never persisted.
- **Sensitivity screening runs on every clip from every camera** (Full rooms included).
  Company baseline, invisible, no setting. Extra-cautious prompt; any doubt or any
  pipeline failure → censor.
- Patient-level `privacy_mode_enabled` is **removed** (UI + enforcement move per-camera).
  The only remaining patient-wide action is "Disconnect all" (emergency stop).
- `auto_privacy_mode` is a backend **env setting**, default ON. OFF only in internal/dev
  environments so we can test with live feeds. Never user-visible.
- Viewer-side "Blur People" stays, repositioned under a Privacy sub-header in app-level
  settings, copy clarifying "on this device only".
- Pause AI moves to camera detail → Advanced.

## Flagged decisions (need sign-off before build) ⚑

1. **⚑ Blur target = all persons in frame, not patient-only (V1).** Re-identifying the
   patient inside the archival path is unreliable (and the archival path has no re-id
   today); a caregiver mid-bathing-assist is also a dignity subject. Macro positions and
   "what happened" stay readable either way. Patient-only masking becomes a V2 if ever
   needed.
2. **⚑ Sensitivity classifier is pinned to one model** (Gemini flash family, dedicated
   prompt + structured output), bypassing per-account `vlm_provider` routing. A
   safety-critical classifier must behave identically for every account; QA is against
   one model.
3. **⚑ Misclassification escape hatch.** Locked-private + no user toggle means a
   misclassified living room loses live view with no user remedy. Proposal: (a) auto
   re-classify whenever a camera is re-paired or its room name is edited, (b) an
   internal/support-only reclassify endpoint. No user-facing unlock.

## Architecture

### State (backend)

New delta `scripts/deltas/apply_camera_privacy_delta.sql`:

```sql
alter table cameras add column room_class text
  check (room_class in ('bathroom','bedroom','other'));
alter table cameras add column room_class_confidence real;
alter table cameras add column room_classified_at timestamptz;
alter table cameras add column privacy_enabled boolean not null default false;
alter table events add column clip_censored boolean not null default false;
alter table events add column sensitivity_categories text[];
-- patients.privacy_mode_enabled: mark deprecated now; DROP in follow-up delta
-- after both clients ship without the toggle.
```

`privacy_enabled` is denormalized from `room_class` for hot-path reads
(token mint, desired-state poll, clip archival). Written only by the classifier
service and the support reclassify endpoint — no general PATCH.

Existing cameras: `room_classified_at IS NULL` → classified lazily on next connect.
Until classified, treat as Full (today's behavior; avoids surprise loss of live view
before the messaging ships).

### Room classification (backend, new `src/services/room_classifier.py`)

- Trigger: camera ingest sees frames for a camera with `room_classified_at IS NULL`
  and `settings.auto_privacy_mode` is true.
- Clean-frame selection: wait for motion-settled window (existing motion state) —
  avoids classifying a dark/mid-mount/installer-in-frame first frame.
- One VLM call, structured output `{room_class, confidence}`.
- Decision rule (fail-closed): any bathroom/bedroom answer → Private regardless of
  confidence. `other` accepted only at confidence ≥ threshold (start 0.7). Uncertain →
  re-sample (up to 3 tries across the first hour) → still uncertain → Private.
- On result: write fields, set `privacy_enabled`, surface via camera GET so clients can
  show the one-time "sensitive room → privacy is on by default" card.

### Live severing, per-camera (backend + edge)

Phase 0 spike (0.5–1d): confirm SFU-enforced per-track subscription control.
Primary mechanism: **publisher-side track subscription permissions**
(`setTrackSubscriptionPermissions` — allowed-participants list), which the SFU enforces
regardless of what a subscriber client requests. Server-side `UpdateSubscriptions` alone
is best-effort (client can re-subscribe) — acceptable only as defense-in-depth.

- Backend: include `privacy_enabled` in `/v1/cameras/me/desired-state` response
  (wall phone) and in the gateway roster payload. On classification flip, nudge
  publishers (they already poll).
- camera-app (`LiveKitPublisher.swift`): when private, restrict both tracks' subscription
  permissions to `detector_worker_*` identities only; ignore inbound `set_talk_active`
  (talk/call dead for private cameras).
- gateway (`publisher.py`): same restriction per room via livekit-rtc.
- Backend defense-in-depth: keep the `livekit_privacy.py` walk, now per-camera —
  on privacy flip, force-unsubscribe caregivers from that camera's tracks.
- Analysis is unaffected: detector worker keeps subscribing everywhere.

### Sensitivity screening + censoring (backend)

Hook point: `clip_archival.py` (already re-checks privacy immediately before upload).
New flow for **every** clip:

1. `src/services/detection/sensitivity.py` — one pinned-model VLM call on the assembled
   clip. Structured output `{sensitive: bool, categories: [...], confidence}`.
   Category enum: `nudity_or_undress`, `toileting`, `bathing_showering`,
   `personal_hygiene`, `medical_care` (wounds/catheters/devices), `incontinence`,
   `intimate_situation`, `emotional_distress`, `other_dignity`.
   Prompt is EXTRA CAUTIOUS by instruction: partial states, ambiguous poses, occlusion,
   low light → sensitive. Any doubt → sensitive.
2. Not sensitive → archive as today.
3. Sensitive → `src/services/detection/censor.py`:
   - Person-mask blur: YOLO-seg person masks per frame (CPU is fine at clip scale,
     ~44 frames for an 11s @ 4fps window), dilate masks, heavy gaussian/pixelate via
     OpenCV, re-encode with ffmpeg, strip audio (`-an`).
   - **Fail-safe:** VLM says sensitive but segmentation finds no person → full-frame
     blur. Segmentation must never be the weak link.
   - Only the censored MP4 is uploaded; raw buffer discarded. `clip_censored = true`,
     categories stored.
4. **Any failure anywhere** (VLM error/timeout, seg crash, encode error) → censor
   (full-frame blur, no audio) or drop the clip; never archive raw on error.

### Client UI (iOS + web)

iOS (`nemara-caregiver-app`):
- Delete `PrivacyModeCard` from patient settings; remove patient-wide privacy banner
  logic from `PatientLiveView`.
- Per-tile Private placard ("Private room — live view off"), hide per-tile audio + Call
  for private cameras (per-tile placard pattern already exists — rework, not greenfield).
- Camera detail: privacy badge + explainer; Pause AI → Advanced section.
- Event detail: censored-clip treatment ("Sensitive moment — censored for dignity")
  from `clip_censored` + categories.
- One-time "This looks like a bathroom/bedroom — privacy is on by default" card after
  classification (client-side ack, keyed on camera id).
- Blur People → app settings under "Privacy", copy "on this device".

Web (`nemara-web-app`):
- Delete `PrivacyModeSettings`; `VideoGrid` per-camera private placard + audio disabled.
- Events feed: censored badge (web is text-only for clips; badge + copy only).
- Same one-time classification card; settings IA mirror of iOS; `BlurPeopleSettings`
  under Privacy sub-header.
- `AddCameraFlow`: room name stays cosmetic; add a line that room type is detected
  automatically.

### Removal of patient-level privacy mode

Backend: retire the patient PATCH path for `privacy_mode_enabled`, the patient-level
token-mint gate, and patient-wide clip suppression (superseded by per-camera). Keep
`events.clip_suppressed` (historical rows + still used if a camera flips private
mid-clip). Column drop = follow-up delta after clients ship. Update/replace
`test_patient_privacy_mode_api.py` with per-camera equivalents.

## Eval + QA (critical path)

- Staged eval set for the sensitivity classifier: acted/clothed-proxy clips across all
  categories + negatives (normal kitchen/living scenes). Target: measured
  **false-negative rate ~0** on the staged set; false-positive rate is the tunable.
- Unit: classifier decision rule, fail-closed paths, censor fallback (no-person →
  full-frame), audio stripping.
- Integration: per-camera severing (caregiver cannot subscribe to private track even
  when requesting), classification lifecycle, clip pipeline end-to-end.
- Manual: soak with `auto_privacy_mode` off vs on; verify detector worker unaffected.

## Sequencing

| Phase | Work | Repos |
|---|---|---|
| 0 | LiveKit per-track permission spike | backend (+SDK probes) |
| 1 | Schema delta + settings flag + room classifier | backend |
| 2 | Sensitivity call + censor pipeline + eval set | backend |
| 3 | Per-camera live severing | backend, camera-app, gateway |
| 4 | Client UI + messaging + settings IA | iOS, web |
| 5 | Patient privacy-mode removal + column-drop delta + copy/docs pass | all |

Phases 1–2 parallel with the Phase 0 spike. Checkpoint review after each phase per the
standing review convention. Deploy order per prod runbook: DB delta → backend → clients.

## Deploy checklist (adjudicated-review additions, 2026-07-27)

- **`NEMARA_ENV` must be explicitly set in prod.** `Settings.env` defaults to
  `local`, and env=local waives the internal-key check — an unset prod deploy
  would leave the internal reclassify endpoint UNAUTHENTICATED (reviewer
  finding). Set `NEMARA_INTERNAL_SERVICE_KEY` as well.
- **`NEMARA_SENSITIVITY_SCREENING_ENABLED` must never be false in prod.** Now
  enforced in code (Fix 2): the kill-switch is honored only in env=local/test;
  elsewhere a false value is ignored, screening stays on, and a loud warning is
  logged.
- **Pre-warm `yolo11n-seg.pt`** (~6 MB): the censor's person-mask model
  auto-downloads on first censor. Bake it into the image or allow egress to the
  Ultralytics asset host, or the first sensitive clip after a cold deploy drops.
- **Publisher apps (camera-app / gateway) must ship before relying on the
  SFU-enforced cut.** Until their track-permission polls are live, the backend
  webhook re-sever on every caregiver join (Fix 1) is the interim guarantee —
  best-effort, self-healing, but not SFU-enforced.
- **Accepted residual risks:** (1) raw clip bytes can survive in `/tmp` only if
  the process is SIGKILLed mid-censor (tempfiles are otherwise always unlinked);
  (2) gateway cameras classify without a clean-frame motion gate — a mid-install
  first look may burn attempts, but the fail-closed direction is preserved
  (uncertainty finalizes private, never open).
- **Legacy whole-home privacy silently stops being enforced.** Patients with the
  old `privacy_mode_enabled=true` flag lose whole-home privacy on this deploy —
  the flag is no longer read anywhere (privacy is per-camera via
  `cameras.privacy_enabled` now). At pilot scale, check for such rows BEFORE
  deploying (`SELECT id FROM patients WHERE privacy_mode_enabled`) and
  coordinate with those families / set their non-private cameras'
  `desired_state` off as appropriate.
