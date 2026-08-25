-- apply_internal_tables_rls_delta.sql
--
-- SECURITY: lock down the backend-internal tables that shipped WITHOUT
-- row-level security — public.gateways and public.pi_discoveries.
--
-- Supabase grants the API roles (anon, authenticated) full CRUD on public
-- tables by default, and the anon key is PUBLIC (it ships in the web + iOS
-- clients). With RLS off, that key can reach these tables through PostgREST:
--   * gateways        -> read (caregiver_id, token_hash) rows, or INSERT a
--                        forged gateway to poll a caregiver's roster, or
--                        UPDATE/DELETE to tamper.
--   * pi_discoveries  -> read `pending_ingest_token` (a PLAINTEXT one-shot
--                        ingest bearer token) and the Pi stream URLs, or
--                        tamper with discovery / claim state.
--
-- Both are backend-internal: only the backend touches them, and it connects
-- as a bypassrls role — so enabling RLS with NO policy (deny-all to
-- anon/authenticated) + revoking the API grants closes the hole with zero
-- backend impact. Neither table has a caregiver_id to scope by, and no
-- client reads either directly, so deny-all is the correct fix (not a
-- caregiver-scoped policy).
--
-- ADDITIVE + idempotent. Safe to run repeatedly and anytime (no client
-- depends on these). Run on prod to close the exposure there too.

BEGIN;

alter table public.gateways enable row level security;
revoke all on public.gateways from anon, authenticated;

alter table public.pi_discoveries enable row level security;
revoke all on public.pi_discoveries from anon, authenticated;

COMMIT;
