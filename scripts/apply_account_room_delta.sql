-- A sensor's room belongs to the ACCOUNT, and assignment is only attribution.
--
-- This is the CONTRACT half of the migration whose expand half landed in
-- apply_camera_ownership_delta.sql on 2026-08-19. That delta said, in as
-- many words, that `cameras.livekit_room` stays authoritative "until the
-- application no longer reads them". This is that moment.
--
-- WHY. `livekit_room` was `patient_<uuid>`, inherited from the original
-- design where one iPhone watched one patient. A camera publishes into
-- exactly ONE LiveKit room, so once a physical camera became one row
-- covering 0..N patients, something had to choose whose room it landed
-- in. That choice is `camera_patients.is_primary`. It is not a product
-- concept and nobody ever asked for it; it is the scaffolding that let
-- one-row-per-camera ship without renaming the room.
--
-- Naming the room after the ACCOUNT removes the question. The account is
-- the one thing provisioning always knows -- an installer types the
-- caregiver's email and nothing else -- so a sensor has a room the
-- instant it exists, before any patient does. Assignment stops being the
-- act that creates the stream and becomes what it should always have
-- been: a statement about who the AI is watching for, revocable without
-- the video pausing.
--
-- WHAT THIS DELIBERATELY DOES NOT DO. `cameras.patient_id` survives.
-- 77 call sites still read it, and dropping a column while readers exist
-- is the unrecoverable direction. It is kept in sync with the FIRST
-- covered patient purely so unmigrated readers see what they saw
-- yesterday, and it is authoritative for nothing. Its own delta follows
-- once those readers are gone.

BEGIN;

-- ── 1. The account is not optional ──────────────────────────────────
-- The expand delta backfilled this and left it nullable so older inserts
-- kept working. Every insert sets it now. Guarded rather than blind: a
-- surviving NULL means a row whose account cannot be determined, and it
-- must stop the delta rather than be silently given a wrong room below.
DO $$
DECLARE orphans int;
BEGIN
  SELECT count(*) INTO orphans FROM public.cameras WHERE caregiver_id IS NULL;
  IF orphans > 0 THEN
    RAISE EXCEPTION
      'aborting: % camera row(s) have no caregiver_id. Backfill them '
      '(apply_camera_ownership_delta.sql) before applying this delta.',
      orphans;
  END IF;
END $$;

ALTER TABLE public.cameras ALTER COLUMN caregiver_id SET NOT NULL;

-- ── 2. The room follows the account ─────────────────────────────────
-- Only sensors whose publisher RE-READS its room are moved.
--
-- A gateway/bridge camera re-reads `livekit_room` from the fleet poll
-- every few seconds and reconciles, so a rename here reaches it within
-- one poll with nothing to redeploy.
--
-- An 'iphone' wall phone does NOT: it is handed `livekit_room` once, in
-- the pairing response, and holds it until it is re-paired. Renaming one
-- here would take a live wall phone off the air with no way for it to
-- find its way back. They keep their `patient_<uuid>` room and migrate
-- when they next pair. Every reader is being changed to RESOLVE a room
-- rather than parse one, so the two shapes coexist safely for as long as
-- they need to.
UPDATE public.cameras
   SET livekit_room = 'account_' || caregiver_id::text
 WHERE source_type IN ('gateway', 'pi')
   AND (livekit_room IS DISTINCT FROM 'account_' || caregiver_id::text);

-- ── 3. There is no primary ──────────────────────────────────────────
-- Both of these exist only to answer "whose room does this publish
-- into". Coverage is now an unordered set of patients this sensor
-- watches, ordered for display by when they were added.
DROP INDEX IF EXISTS public.camera_patients_one_primary;
ALTER TABLE public.camera_patients DROP COLUMN IF EXISTS is_primary;

-- ── 4. The audio kill belongs to the sensor ─────────────────────────
-- It is the facilities audio kill -- a statement about a physical room,
-- "no recording in here" -- but it lived on `patients`, so the roster
-- had to compute the strictest covered patient with a bool_and across
-- coverage. Once a sensor can have no patients at all that expression
-- returns NULL, and NULL fails closed, which would have meant every
-- unassigned sensor silently deaf.
--
-- Backfilled from the strictest patient covering each sensor, so no
-- site's existing kill is loosened by this move. Sensors covering
-- nobody take the column default, which matches a fresh patient's.
ALTER TABLE public.cameras
  ADD COLUMN IF NOT EXISTS audio_capture_enabled boolean NOT NULL DEFAULT true;

UPDATE public.cameras c
   SET audio_capture_enabled = s.strictest
  FROM (SELECT cp.camera_id, bool_and(p.audio_capture_enabled) AS strictest
          FROM public.camera_patients cp
          JOIN public.patients p ON p.id = cp.patient_id
         GROUP BY cp.camera_id) s
 WHERE s.camera_id = c.id
   AND s.strictest IS NOT NULL
   AND c.audio_capture_enabled IS DISTINCT FROM s.strictest;

-- And the patient column goes, in the same transaction as the backfill
-- that reads it. Leaving it would leave a control on the patient API
-- that no longer reaches anything: a caregiver could turn audio "off"
-- for a person and every sensor would keep publishing it. A dead switch
-- on a privacy control is worse than no switch.
ALTER TABLE public.patients DROP COLUMN IF EXISTS audio_capture_enabled;

-- ── 5. Sensor names are unique per ACCOUNT ──────────────────────────
-- One account is one home, and a home does not have two Living Rooms.
--
-- The handler scoped its check on `patient_id`, which is NULL for an
-- unassigned sensor, and `NULL = NULL` is NULL in SQL -- so the check
-- matched no rows and every duplicate passed. Unassigned is the normal
-- state between the installer leaving and the caregiver opening the app,
-- so that was the common case, and the damage is not cosmetic:
-- `cameras.room` is the key detector `where_rooms` matches on.
--
-- Enforced here and not only in the handler, because the handler is one
-- of several writers. Case-insensitive to match the handler's own
-- comparison; partial so a blank name stays legal.
CREATE UNIQUE INDEX IF NOT EXISTS cameras_account_room_unique
  ON public.cameras (caregiver_id, lower(room))
  WHERE room IS NOT NULL AND room <> '';

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Reversible except for is_primary, whose VALUES are gone (the column
-- can be re-added, but which patient was primary cannot be recovered).
-- Rooms can be restored from camera_patients; the flag cannot. If this
-- has to come out, restore primaries as "earliest added_at per camera",
-- which is what the application now uses for patient_id anyway.
