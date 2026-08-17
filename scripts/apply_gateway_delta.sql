-- apply_gateway_delta.sql
--
-- Gateway camera support (promoted from the sandbox). Brings `public.cameras`
-- up to what the gateway roster/heartbeat/provision code needs:
--   1. allow source_type = 'gateway'
--   2. add last_heartbeat_at            (roster poll doubles as a heartbeat)
--   3. allow 'gateway' in the ingest-token-required check (gateway carries an
--      ingest token like Pi — reuses the existing Pi columns, no new tables)
--
-- ADDITIVE + idempotent (drop-if-exists by name, then add). No column drops,
-- no data changes. Dev-first; prod-safe (verify constraint names before prod).

BEGIN;

-- 1. allowed source types
ALTER TABLE public.cameras DROP CONSTRAINT IF EXISTS cameras_source_type_check;
ALTER TABLE public.cameras
  ADD CONSTRAINT cameras_source_type_check
  CHECK (source_type = ANY (ARRAY['iphone'::text, 'pi'::text, 'gateway'::text]));

-- 2. gateway heartbeat timestamp
ALTER TABLE public.cameras
  ADD COLUMN IF NOT EXISTS last_heartbeat_at timestamptz;

-- 3. ingest-token-required: gateway carries a token like Pi (dev allowed only 'pi')
ALTER TABLE public.cameras DROP CONSTRAINT IF EXISTS cameras_ingest_token_required;
ALTER TABLE public.cameras
  ADD CONSTRAINT cameras_ingest_token_required
  CHECK (
    ((source_type = ANY (ARRAY['pi'::text, 'gateway'::text])) AND (ingest_token_hash IS NOT NULL))
    OR ((source_type = 'iphone'::text) AND (ingest_token_hash IS NULL))
  );

-- 4. gateways table — the roster's site-token auth. Each mini-PC has a UNIQUE
--    token (generated on first boot) tied to a CAREGIVER at setup (by email, in
--    the field — nothing baked in). The roster returns gateway cameras across
--    ALL that caregiver's patients. patient_id kept nullable for back-compat.
CREATE TABLE IF NOT EXISTS public.gateways (
    id           uuid DEFAULT gen_random_uuid() NOT NULL,
    caregiver_id uuid,
    patient_id   uuid,
    token_hash   text NOT NULL,
    created_at   timestamptz DEFAULT now(),
    last_seen_at timestamptz,
    CONSTRAINT gateways_pkey PRIMARY KEY (id)
);
-- bring an EXISTING gateways table up to the caregiver model (idempotent):
ALTER TABLE public.gateways ADD COLUMN IF NOT EXISTS caregiver_id uuid;
ALTER TABLE public.gateways ALTER COLUMN patient_id DROP NOT NULL;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint
                 WHERE conrelid = 'public.gateways'::regclass
                   AND conname = 'gateways_caregiver_id_fkey') THEN
    ALTER TABLE public.gateways
      ADD CONSTRAINT gateways_caregiver_id_fkey
      FOREIGN KEY (caregiver_id) REFERENCES public.caregivers(id) ON DELETE CASCADE;
  END IF;
END $$;

COMMIT;

-- RLS (2026-08-03). Supabase grants anon/authenticated full DML on every
-- table in public and PostgREST exposes it, so RLS is the only control —
-- and anon is the PUBLISHABLE key that ships in the app binary. Enabling
-- it with no policy denies all non-owner access (fail closed); the backend
-- reaches these through the service role, which bypasses RLS.
-- Must live in THIS file: deltas are applied one at a time, so fixing it in
-- a later delta leaves a live exposure window in between.
alter table public.gateways enable row level security;
