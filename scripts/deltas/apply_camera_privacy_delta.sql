-- Adds the per-CAMERA room-derived privacy columns to an EXISTING database.
-- Idempotent. Apply BEFORE deploying the room-privacy backend
-- (DB-before-deploy — see docs/PROD_MIGRATION_RUNBOOK.md):
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_camera_privacy_delta.sql
--
-- Privacy moves from the patient level (patients.privacy_mode_enabled,
-- now DEPRECATED below) to per-camera, derived from the ROOM the camera
-- watches (see docs/plans/2026-07-27-room-privacy-and-clip-censoring.md):
--   room_class            — VLM verdict for what room this camera sees.
--   room_class_confidence — the model's self-reported confidence [0,1].
--   room_classified_at    — when the verdict was finalized. NULL = not
--                           yet classified → the ingest worker classifies
--                           lazily on the next connect. Cleared on room
--                           rename / re-pair so the classifier re-runs.
--   privacy_enabled       — DENORMALIZED from room_class for hot-path
--                           reads (desired-state poll, gateway roster,
--                           clip archival). bathroom/bedroom → true,
--                           locked. Written ONLY by the classifier and
--                           the internal reclassify endpoint — never via
--                           the public camera PATCH.

alter table public.cameras
  add column if not exists room_class text
    check (room_class in ('bathroom', 'bedroom', 'other'));
alter table public.cameras
  add column if not exists room_class_confidence real;
alter table public.cameras
  add column if not exists room_classified_at timestamptz;
alter table public.cameras
  add column if not exists privacy_enabled boolean not null default false;

-- events.clip_censored — true when the archived clip was CENSORED
-- (person-mask blur + audio stripped) because the sensitivity screen
-- flagged a dignity-sensitive moment. Columns land now; the censor
-- pipeline that writes them ships separately (Phase 2 of the plan).
alter table public.events
  add column if not exists clip_censored boolean not null default false;
-- events.sensitivity_categories — which sensitivity categories the
-- screen flagged (e.g. '{bathing_showering,nudity_or_undress}'). NULL
-- until the sensitivity pipeline ships / for non-sensitive clips.
alter table public.events
  add column if not exists sensitivity_categories text[];

-- patients.privacy_mode_enabled is superseded by per-camera
-- cameras.privacy_enabled. Kept (NOT dropped) so historical rows and
-- not-yet-updated clients keep working; DROP lands in a follow-up delta
-- after both clients ship without the toggle.
comment on column public.patients.privacy_mode_enabled is
  'DEPRECATED 2026-07-27: superseded by per-camera cameras.privacy_enabled '
  '(room-derived privacy). No longer PATCH-able or enforced; column kept for '
  'historical rows until a follow-up drop delta after clients ship.';

-- 2026-07-28: all built-in preset detectors default to ALL ROOMS
-- (users rescope in-app). Normalizes the two seeded presets that
-- carried hardcoded room scopes (Wandering="HALLWAY", Medication
-- adherence="KITCHEN") — under derive-from-where_rooms link syncing,
-- an exact-match room scope that matches no real room name would
-- silently disarm them. Idempotent.
UPDATE detectors SET where_rooms = NULL
WHERE source = 'preset' AND where_rooms IS NOT NULL;
