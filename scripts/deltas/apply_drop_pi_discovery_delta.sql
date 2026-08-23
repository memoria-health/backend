-- Delete `pi_discoveries`. There is no Raspberry Pi, and there never was.
--
-- The table held announcements from a Pi daemon
-- (`scripts/pi/announce_daemon.py`) so a caregiver could pick a
-- discovered unit out of a list and claim it. That whole path is dead in
-- two independent ways:
--
--   * the ingest endpoint the Pi pushed frames to
--     (POST /v1/cameras/me/frames) was deleted on 2026-08-21, so a
--     claimed discovery led to a camera that could never deliver a frame;
--   * no Pi has ever run in production. The shipped hardware is the
--     mini-PC gateway, and the bridge replaces that.
--
-- It survived the 2026-08-21 sweep only because it owns a table with an
-- FK to `cameras` and was threaded into camera create and both delete
-- paths -- deleting the directory wholesale would have taken those with
-- it. Its own change, as the handoff said. This is that change.
--
-- The application side is already gone: src/db/pi_discovery.py,
-- src/repos/pi_discovery_repo.py, src/schemas/pi_discovery.py, the
-- `discovery_id` mode on pi-provision, and the unclaim calls in
-- delete_camera and the wall-phone unpair.

BEGIN;

-- Loud rather than quiet: this drops a table, and doing that while rows
-- exist should be a decision somebody makes on purpose. In dev and in
-- production this is empty -- the daemon was never deployed.
DO $$
DECLARE n int;
BEGIN
  IF to_regclass('public.pi_discoveries') IS NULL THEN
    RAISE NOTICE 'pi_discoveries already absent, nothing to do';
    RETURN;
  END IF;
  EXECUTE 'SELECT count(*) FROM public.pi_discoveries' INTO n;
  IF n > 0 THEN
    RAISE EXCEPTION
      'aborting: pi_discoveries holds % row(s). Nothing reads this table '
      'any more, but dropping data is not something a migration should do '
      'behind your back -- inspect them, then drop the table by hand.', n;
  END IF;
END $$;

-- The FK to cameras is ON DELETE SET NULL and lives on THIS table, so
-- dropping it takes the constraint with it and touches nothing in
-- `cameras`.
DROP TABLE IF EXISTS public.pi_discoveries;

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Recreate from scripts/apply_pi_schema_delta.sql, which still holds the
-- original definition. There is no data to restore: the guard above
-- refuses to run if there ever was any.
