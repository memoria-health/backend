-- Wandering exit-door selection.
--
-- Adds detectors.door_config to an EXISTING database (local dev or prod).
-- schema.sql already includes it for fresh creates; this delta is for a DB
-- that predates the column. Idempotent — safe to re-run.
--
-- Only meaningful on the per-patient 'wandering' detector. Holds the caregiver's
-- exit-door choice from the setup picker (see POST /v1/patients/{id}/wandering-door):
--   { "box": [x, y, w, h],      -- selected door, normalized 0-1 (null if no_exit_door)
--     "label": "left wall",      -- human label for the door
--     "source": "vlm"|"manual",  -- highlighted box vs tap-override
--     "no_exit_door": false }    -- true => no door leads outside; detector is turned off
-- The wandering prompt_fragment is regenerated from this so the per-tick VLM
-- only counts leaving through THAT doorway as wandering.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_wandering_door_delta.sql
-- or paste into the Supabase SQL editor / run via the Management API.

alter table public.detectors
  add column if not exists door_config jsonb;
