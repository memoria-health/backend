-- RLS for sleep_nights / activity_days / review_devices (2026-08-03).
--
-- WHY THIS EXISTS
-- ---------------
-- apply_sleep_activity_tables_delta.sql and apply_review_gate_delta.sql each
-- create their tables with NO `enable row level security`. Supabase grants
-- `anon` and `authenticated` full DELETE/INSERT/SELECT/UPDATE/TRUNCATE on
-- every table in `public`, and PostgREST exposes them. With RLS off, those
-- grants are the ONLY control.
--
-- `anon` is the publishable key. It ships inside the iOS app binary and the
-- web bundle — it is public by design. So on any database where those two
-- deltas have run, anyone who extracts that key can read, overwrite, or
-- TRUNCATE:
--
--   sleep_nights    — per-patient sleep records (health data)
--   activity_days   — per-patient activity records (health data)
--   review_devices  — APNs push tokens + user_ids
--
-- Verified 2026-08-03 on the local DB:
--   pg_class.relrowsecurity = false, 0 policies, anon holds all seven privs.
--
-- Both deltas are still PENDING for production, so prod is not yet affected.
-- APPLY THIS ONE IN THE SAME BATCH — never those two alone.
--
-- Idempotent: safe to re-run.

-- ── sleep_nights ─────────────────────────────────────────────────────
alter table public.sleep_nights enable row level security;
drop policy if exists sleep_nights_member on public.sleep_nights;
create policy sleep_nights_member on public.sleep_nights
  using (has_patient_access(patient_id));

-- ── activity_days ────────────────────────────────────────────────────
alter table public.activity_days enable row level security;
drop policy if exists activity_days_member on public.activity_days;
create policy activity_days_member on public.activity_days
  using (has_patient_access(patient_id));

-- ── review_devices ───────────────────────────────────────────────────
-- No patient_id — this is a per-USER device registration, so it scopes to
-- the authenticated user rather than to patient membership. A caregiver may
-- only ever see their own push tokens.
alter table public.review_devices enable row level security;
drop policy if exists review_devices_self on public.review_devices;
create policy review_devices_self on public.review_devices
  using (user_id = auth.uid());

-- ── body_pool_embeddings / gateways / pi_discoveries ────────────────
-- These already have RLS enabled with ZERO policies, which denies every
-- non-owner role. That is deliberate: they are backend-only tables reached
-- through the service role, which bypasses RLS. Left as-is — adding a
-- permissive policy here would OPEN them, not secure them.
