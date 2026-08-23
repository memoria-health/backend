-- A service's token identifies exactly one gateways row.
--
-- `public.gateways.token_hash` had no unique constraint, and enrolment
-- is a SELECT-then-INSERT: two concurrent first-enrols with the same
-- token both miss the SELECT and both INSERT, leaving two rows for one
-- credential. Every lookup then does `.first()`, so which row a service
-- gets -- and therefore which caregiver it is bound to and which sensors
-- it is told to run -- depends on physical row order. A bridge would
-- serve half its cameras, consistently, with nothing in the logs.
--
-- A bridge re-enrols on every boot (see nemara-bridge core/server.py
-- `enroll_stored`), so the concurrent path is no longer hypothetical:
-- a restart loop is exactly two enrolments close together.
--
-- The index alone would turn a silent double-row into a 500 for the
-- losing racer, which is better but still a regression. It ships with
-- the ON CONFLICT handling in `_enroll_gateway_token` -- the two are one
-- change and must not be applied separately.

BEGIN;

DO $$
DECLARE dupes int;
BEGIN
  SELECT count(*) INTO dupes FROM (
    SELECT token_hash FROM public.gateways
     GROUP BY token_hash HAVING count(*) > 1) d;
  IF dupes > 0 THEN
    RAISE EXCEPTION
      'aborting: % duplicate token_hash value(s) in public.gateways. '
      'Merge them by hand first -- picking a survivor decides which '
      'account each affected service is bound to, which is not a choice '
      'a migration should make silently.', dupes;
  END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS gateways_token_hash_uidx
  ON public.gateways (token_hash);

COMMIT;
