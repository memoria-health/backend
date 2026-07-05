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

-- 4. gateways table — the roster's site-token auth: one stable token per
--    mini-PC, hashed, mapped to a patient. The gateway polls /gateway/roster
--    with this token; the roster returns that patient's source_type='gateway'
--    cameras. Matches the sandbox schema exactly.
CREATE TABLE IF NOT EXISTS public.gateways (
    id          uuid DEFAULT gen_random_uuid() NOT NULL,
    patient_id  uuid NOT NULL,
    token_hash  text NOT NULL,
    created_at  timestamptz DEFAULT now(),
    last_seen_at timestamptz,
    CONSTRAINT gateways_pkey PRIMARY KEY (id),
    CONSTRAINT gateways_patient_id_fkey FOREIGN KEY (patient_id)
        REFERENCES public.patients(id) ON DELETE CASCADE
);

COMMIT;
