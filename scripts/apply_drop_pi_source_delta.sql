-- Remove the last of the Raspberry Pi path from the schema.
--
-- Follows apply_drop_pi_discovery_delta.sql. That took the announce
-- table; this takes what a discovered Pi would have become:
--
--   * `source_type = 'pi'` -- a row that could be created and could
--     never deliver a frame, because POST /v1/cameras/me/frames was
--     deleted on 2026-08-21;
--   * `pi_stream_url` / `pi_audio_url` -- the pull-mode URLs, already
--     described in the ORM as "vestigial" since the push refactor,
--     written as NULL by every remaining creation path.
--
-- The application side is already gone: the pi-provision endpoint,
-- PiProvisionRequest/Response, the ORM columns, and `scripts/pi/` with
-- its installer and systemd units.
--
-- No Pi ever ran in production. `source_type` keeps 'iphone' (the wall
-- phone) and 'gateway' (anything publishing straight to LiveKit with an
-- ingest token -- the mini-PC and the hosted bridge).

BEGIN;

-- Loud if reality disagrees. Dropping the value out from under live rows
-- would leave them failing the new constraint and unreadable by code
-- whose Literal no longer contains it.
DO $$
DECLARE n int;
BEGIN
  SELECT count(*) INTO n FROM public.cameras WHERE source_type = 'pi';
  IF n > 0 THEN
    RAISE EXCEPTION
      'aborting: % camera row(s) still have source_type = ''pi''. They '
      'cannot ingest -- the endpoint is gone -- so decide whether to '
      'delete them or move them to ''gateway'' before applying this.', n;
  END IF;
END $$;

ALTER TABLE public.cameras DROP CONSTRAINT IF EXISTS cameras_source_type_check;
ALTER TABLE public.cameras
  ADD CONSTRAINT cameras_source_type_check
  CHECK (source_type = ANY (ARRAY['iphone'::text, 'gateway'::text]));

-- The length CHECKs go with their columns; naming them explicitly
-- because DROP COLUMN takes dependent constraints silently and a later
-- reader should be able to see that these were considered, not missed.
ALTER TABLE public.cameras
  DROP CONSTRAINT IF EXISTS cameras_pi_stream_url_length,
  DROP CONSTRAINT IF EXISTS cameras_pi_audio_url_length;

ALTER TABLE public.cameras
  DROP COLUMN IF EXISTS pi_stream_url,
  DROP COLUMN IF EXISTS pi_audio_url;

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Re-add both columns as nullable text and widen the CHECK back to
-- include 'pi'. No data is lost going forward: the guard above refuses
-- to run while any 'pi' row exists, and both columns were NULL on every
-- row every remaining creation path writes.
