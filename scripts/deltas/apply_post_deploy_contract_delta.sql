-- The CONTRACT halves of two deltas, split out to deploy at zero downtime.
--
-- WHY THIS FILE EXISTS. apply_account_room_delta.sql and
-- apply_identity_shard_per_sensor_delta.sql each do two things: ADD what the
-- new code needs, and DROP what the old code used. Run whole, they are
-- unrunnable in either order --
--
--   before `railway up`: they drop `patients.audio_capture_enabled` and
--     `patients.identity_node_id`, which the deployed image reads (the latter
--     in 4 files), and they rename `livekit_room` to `account_<uuid>` -- which
--     the old image REWRITES back to `patient_<uuid>` (src/api/cameras.py:666)
--     while `identity_roster.py:269` builds `patient_<pid>` and ignores the
--     column, so the rename would also point the roster at dead rooms;
--
--   after `railway up`: the new image needs `cameras.audio_capture_enabled`
--     and `cameras.identity_node_id` in its fleet roster on the first request.
--
-- So the ADD halves were applied BEFORE the deploy (both schemas coexist --
-- every added column is nullable or defaulted, and nothing reads the new ones
-- until the new image runs), and this file is everything that had to wait.
--
-- RUN THIS IMMEDIATELY AFTER `/ready` GOES GREEN on the new image.
-- Between the deploy and this file, the new code is running against columns
-- that still exist, so the gap is safe but should be short.

BEGIN;

-- ── 1. Guard: the new image must already be live ─────────────────────
-- These columns are what the new image reads. If they are missing, the ADD
-- halves never ran and this file would drop the old ones out from under a
-- backend that has nothing to replace them with.
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema='public' AND table_name='cameras'
                    AND column_name='audio_capture_enabled') THEN
    RAISE EXCEPTION 'cameras.audio_capture_enabled missing -- run the ADD half first.';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                  WHERE table_schema='public' AND table_name='cameras'
                    AND column_name='identity_node_id') THEN
    RAISE EXCEPTION 'cameras.identity_node_id missing -- run the ADD half first.';
  END IF;
END $$;

-- ── 2. account_room: the account is not optional ─────────────────────
ALTER TABLE public.cameras ALTER COLUMN caregiver_id SET NOT NULL;

-- ── 3. account_room: the room follows the account ────────────────────
-- Only sensors whose publisher RE-READS its room from the fleet poll. An
-- 'iphone' wall phone is handed its room once at pairing and would be taken
-- off the air with no way back, so it keeps `patient_<uuid>` until it re-pairs.
UPDATE public.cameras
   SET livekit_room = 'account_' || caregiver_id::text
 WHERE source_type IN ('gateway', 'pi')
   AND (livekit_room IS DISTINCT FROM 'account_' || caregiver_id::text);

-- ── 4. account_room: there is no primary ─────────────────────────────
DROP INDEX IF EXISTS public.camera_patients_one_primary;
ALTER TABLE public.camera_patients DROP COLUMN IF EXISTS is_primary;

-- ── 5. account_room: the audio kill leaves the patient ───────────────
-- The backfill onto cameras already ran in the ADD half. Dropping this is
-- what makes the move real: left in place it is a privacy control on the
-- patient API that no longer reaches anything, which is worse than no control.
ALTER TABLE public.patients DROP COLUMN IF EXISTS audio_capture_enabled;

-- ── 6. account_room: sensor names are unique per ACCOUNT ─────────────
-- Blocked until 2026-08-26 by three rows for one physical camera, all named
-- "Room 1" -- see scripts/cleanup_duplicate_camera_rows.sql, which must have
-- run first.
CREATE UNIQUE INDEX IF NOT EXISTS cameras_account_room_unique
  ON public.cameras (caregiver_id, lower(room))
  WHERE room IS NOT NULL AND room <> '';

-- ── 7. identity_shard: the slot leaves the patient ───────────────────
-- No backfill by design: every node re-claims its sensors on its next poll
-- (~30s) and re-grounds. Expect one window of identity churn per node.
DROP INDEX IF EXISTS public.patients_identity_node_port_uidx;

ALTER TABLE public.patients
  DROP COLUMN IF EXISTS identity_node_id,
  DROP COLUMN IF EXISTS identity_port;

COMMIT;

-- ── Verification ─────────────────────────────────────────────────────
--   -- gateway/pi sensors on account rooms, wall phones untouched:
--   select source_type, count(*) filter (where livekit_room like 'account_%') as account_rooms,
--          count(*) filter (where livekit_room like 'patient_%') as patient_rooms
--     from public.cameras group by source_type;
--
--   -- must be 0 -- the old columns are gone:
--   select count(*) from information_schema.columns
--    where table_schema='public'
--      and (table_name='patients' and column_name in
--             ('audio_capture_enabled','identity_node_id','identity_port'))
--       or (table_name='camera_patients' and column_name='is_primary');
--
-- ── Rollback note ────────────────────────────────────────────────────
-- Re-add the three patient columns and roll the image back; nodes re-claim on
-- their next poll either way, and rooms can be restored from camera_patients.
-- `is_primary` VALUES are not recoverable -- restore as earliest added_at per
-- camera, which is what the application uses for patient_id anyway.
