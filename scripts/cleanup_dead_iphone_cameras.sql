-- Delete 14 long-dead iPhone camera rows (2026-08-26).
--
-- WHY NOW. Identity sharding moved from the PATIENT to the SENSOR
-- (apply_identity_shard_per_sensor_delta.sql). Under the old model 8 slots
-- covered 8 patients; under the new one every `cameras` row competes for a
-- slot, so 19 rows chase a capacity of 8 and the long tail of abandoned test
-- phones from May-July sits in the queue forever.
--
-- The claim order is NOT the problem and was not changed. `_LIVE_SENSOR`
-- ranks streaming sensors first and evicts a dormant holder the moment a live
-- sensor needs its slot; test/integration/test_identity_roster.py covers this
-- in five cases and all 22 tests pass. These rows are simply garbage that
-- makes the roster unreadable and holds warm-standby slots nothing will ever
-- use.
--
-- WHAT COUNTS AS DEAD. No iPhone camera in this database has EVER sent a
-- heartbeat (`last_heartbeat_at IS NULL` on all 16), so liveness cannot come
-- from there. The discriminator used is activity: no analysis in 30 days.
--
-- DELIBERATELY KEPT:
--   13295095 "Test Iphone"  -- 598 analyses, last 2026-08-22, still in use
--   60482a37 "Demo"         -- last 2026-07-30, and a camera named Demo is
--                              very likely the App Review demo row, which
--                              the prod runbook says must exist
--   all 3 `gateway` rows    -- real hardware; two are analysing today
--
-- WHAT IS LOST, and it is not recoverable:
--   ~44,165 `analyses` rows CASCADE with their camera. 42,651 of those belong
--   to 92e5804c "Test", idle since 2026-06-27. Aryaman approved the cascade.
--   34 `events` do NOT cascade -- `events.camera_id` is ON DELETE SET NULL --
--   so they survive and lose only their camera attribution.
--
-- One-off. Not a delta: it repairs data this database happens to hold.

BEGIN;

-- ── Guard: refuse if the set is not the one this script was written for ──
-- A predicate-driven delete would silently widen as rows age past 30 days.
-- The ids are pinned, and the count must match exactly.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n FROM public.cameras
   WHERE id IN ('84b50508-5cdc-4acb-b4ae-ea939280514d',
                '92e5804c-9465-413e-8926-ba2e88fd2ef4',
                '5ce257a7-cb8d-467f-aba4-a9bb80cdcf81',
                '78817a86-3182-440e-90cd-84adb07e73a6',
                'd52b8b69-f724-44a8-8645-7d46b601c83c',
                '2996a1c4-1b4f-4198-a6ba-2c6c8a1ad723',
                '6a79255d-3b75-4929-8d8c-f9f16efe27b5',
                '9268433c-6bc3-4ca1-b534-2ffcdd039972',
                '40d72d81-7ad1-4b60-9e86-c3c46c565149',
                'f0de6870-89b8-4d05-b963-9bd2d2ff8286',
                '5bc52a62-854d-458a-bd68-b4a846c5fce0',
                '3c1f3c09-8a91-4ec2-88b4-95c90309039b',
                'f0e6f6a7-4ebc-4a73-9a9a-f952f601ee36',
                '23be67c1-2dd2-4138-80c0-0578d70f06b0');
  IF n <> 14 THEN
    RAISE EXCEPTION 'expected 14 camera rows, found %. Already run, or wrong database.', n;
  END IF;
END $$;

-- ── Refuse to delete anything that is actually streaming ─────────────
-- Cheap insurance against a mistyped uuid landing on a live sensor.
DO $$
DECLARE live int;
BEGIN
  SELECT count(*) INTO live FROM public.cameras
   WHERE id IN ('84b50508-5cdc-4acb-b4ae-ea939280514d',
                '92e5804c-9465-413e-8926-ba2e88fd2ef4',
                '5ce257a7-cb8d-467f-aba4-a9bb80cdcf81',
                '78817a86-3182-440e-90cd-84adb07e73a6',
                'd52b8b69-f724-44a8-8645-7d46b601c83c',
                '2996a1c4-1b4f-4198-a6ba-2c6c8a1ad723',
                '6a79255d-3b75-4929-8d8c-f9f16efe27b5',
                '9268433c-6bc3-4ca1-b534-2ffcdd039972',
                '40d72d81-7ad1-4b60-9e86-c3c46c565149',
                'f0de6870-89b8-4d05-b963-9bd2d2ff8286',
                '5bc52a62-854d-458a-bd68-b4a846c5fce0',
                '3c1f3c09-8a91-4ec2-88b4-95c90309039b',
                'f0e6f6a7-4ebc-4a73-9a9a-f952f601ee36',
                '23be67c1-2dd2-4138-80c0-0578d70f06b0')
     AND (online OR source_type <> 'iphone');
  IF live > 0 THEN
    RAISE EXCEPTION 'aborting: % of the targets is online or is not an iphone row.', live;
  END IF;
END $$;

DELETE FROM public.cameras
 WHERE id IN ('84b50508-5cdc-4acb-b4ae-ea939280514d',
              '92e5804c-9465-413e-8926-ba2e88fd2ef4',
              '5ce257a7-cb8d-467f-aba4-a9bb80cdcf81',
              '78817a86-3182-440e-90cd-84adb07e73a6',
              'd52b8b69-f724-44a8-8645-7d46b601c83c',
              '2996a1c4-1b4f-4198-a6ba-2c6c8a1ad723',
              '6a79255d-3b75-4929-8d8c-f9f16efe27b5',
              '9268433c-6bc3-4ca1-b534-2ffcdd039972',
              '40d72d81-7ad1-4b60-9e86-c3c46c565149',
              'f0de6870-89b8-4d05-b963-9bd2d2ff8286',
              '5bc52a62-854d-458a-bd68-b4a846c5fce0',
              '3c1f3c09-8a91-4ec2-88b4-95c90309039b',
              'f0e6f6a7-4ebc-4a73-9a9a-f952f601ee36',
              '23be67c1-2dd2-4138-80c0-0578d70f06b0');

COMMIT;

-- The freed identity slots are NOT reassigned here. The node re-claims on its
-- next roster poll (~30s) using the ordering that was already correct.
--
-- ── Rollback note ────────────────────────────────────────────────────
-- None. The rows and their analyses are gone.
