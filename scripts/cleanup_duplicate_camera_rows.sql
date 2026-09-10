-- Collapse the one physical camera that exists as three rows (2026-08-26).
--
-- This is the 2026-08-19 incident named in apply_camera_ownership_delta.sql:
-- gateway provisioning wrote one camera row PER PATIENT of the caregiver, so
-- a single Trueview in "Room 1" became three rows, three ingest tokens, three
-- LiveKit identities, and three independent analysis streams over identical
-- footage. The schema fix shipped; nobody ever collapsed the rows that the
-- bug had already created.
--
-- Proof they are one device, not three: all three rows carry the SAME
-- `last_heartbeat_at` to the microsecond, and it advances together. One
-- heartbeat write touching three rows is one camera.
--
-- WHY NOW. apply_account_room_delta.sql ends with
--   CREATE UNIQUE INDEX cameras_account_room_unique ON cameras (caregiver_id, lower(room))
-- and these three rows violate it, so that delta cannot apply until this runs.
--
-- WHAT IS LOST. `analyses.camera_id` is ON DELETE CASCADE, so the ~36k
-- analyses rows on the discarded row go with it. That is deliberate: they are
-- a second model pass over the SAME video the survivor already analysed, and
-- keeping them by repointing would fabricate a camera with double-density
-- history that never existed. `events` is ON DELETE SET NULL and all three
-- rows have ZERO events, so no caregiver-visible record is touched.
--
-- One-off. Not a delta: it repairs data this database happens to hold, and it
-- is a no-op on any database that does not hold it.

BEGIN;

-- ── 0. Refuse to run against anything but the shape described above ──
-- A literal-UUID script that silently matches nothing is worse than one that
-- fails, because the account_room delta then aborts later with no explanation.
DO $$
DECLARE n int; hb int;
BEGIN
  SELECT count(*) INTO n FROM public.cameras
   WHERE id IN ('118dd183-dc62-4739-b5c4-be70c929c9de',
                '5e9fbc44-ef86-4108-95dd-ec37a8310ab4',
                'aa76ac0a-4cdd-49da-a9dc-8bc45026a119');
  IF n <> 3 THEN
    RAISE EXCEPTION 'expected 3 camera rows, found %. Already merged, or wrong database.', n;
  END IF;

  SELECT count(DISTINCT last_heartbeat_at) INTO hb FROM public.cameras
   WHERE id IN ('118dd183-dc62-4739-b5c4-be70c929c9de',
                '5e9fbc44-ef86-4108-95dd-ec37a8310ab4',
                'aa76ac0a-4cdd-49da-a9dc-8bc45026a119');
  IF hb <> 1 THEN
    RAISE EXCEPTION
      'the 3 rows have % distinct last_heartbeat_at values, not 1. They may be '
      'genuinely separate cameras -- do NOT merge without checking by hand.', hb;
  END IF;
END $$;

-- ── 1. Coverage: the survivor watches all three residents ────────────
-- This is what the duplicate rows were standing in for. `is_primary` stays
-- with the survivor's own backfilled row -- the partial unique index permits
-- exactly one, and apply_account_room_delta.sql drops the column entirely
-- shortly after, falling back to earliest `added_at`, which is that same row.
INSERT INTO public.camera_patients (camera_id, patient_id, is_primary)
SELECT '118dd183-dc62-4739-b5c4-be70c929c9de', c.patient_id, false
  FROM public.cameras c
 WHERE c.id IN ('5e9fbc44-ef86-4108-95dd-ec37a8310ab4',
                'aa76ac0a-4cdd-49da-a9dc-8bc45026a119')
   AND c.patient_id IS NOT NULL
ON CONFLICT (camera_id, patient_id) DO NOTHING;

-- ── 2. Detector links: union, not intersection ───────────────────────
-- Each duplicate accumulated its own assignments (6, 5 and 4). They describe
-- what should run on this physical camera, so the survivor takes every
-- detector any of them carried. Its own rows win on conflict, since those are
-- the ones whose door_config an operator most recently touched.
INSERT INTO public.cameras_detectors (camera_id, detector_id, created_at, door_config)
SELECT DISTINCT ON (cd.detector_id)
       '118dd183-dc62-4739-b5c4-be70c929c9de', cd.detector_id, cd.created_at, cd.door_config
  FROM public.cameras_detectors cd
 WHERE cd.camera_id IN ('5e9fbc44-ef86-4108-95dd-ec37a8310ab4',
                        'aa76ac0a-4cdd-49da-a9dc-8bc45026a119')
 ORDER BY cd.detector_id, cd.created_at DESC
ON CONFLICT (camera_id, detector_id) DO NOTHING;

-- ── 3. Drop the duplicates ───────────────────────────────────────────
-- analyses CASCADE (intended, see header); camera_patients CASCADE (the rows
-- were re-pointed at the survivor in step 1); events SET NULL (there are none).
DELETE FROM public.cameras
 WHERE id IN ('5e9fbc44-ef86-4108-95dd-ec37a8310ab4',
              'aa76ac0a-4cdd-49da-a9dc-8bc45026a119');

-- ── 4. The precondition this exists to satisfy ───────────────────────
DO $$
DECLARE dupes int;
BEGIN
  SELECT count(*) INTO dupes FROM (
    SELECT caregiver_id, lower(room) FROM public.cameras
     WHERE room IS NOT NULL AND room <> ''
     GROUP BY 1,2 HAVING count(*) > 1) d;
  IF dupes > 0 THEN
    RAISE EXCEPTION
      'still % duplicate (caregiver_id, room) group(s); '
      'apply_account_room_delta.sql would abort.', dupes;
  END IF;
END $$;

COMMIT;

-- ── Rollback note ────────────────────────────────────────────────────
-- There isn't one. The discarded rows and their analyses are gone. What is
-- recoverable is the SHAPE: re-provisioning would recreate per-patient rows,
-- which is the bug. Do not.
