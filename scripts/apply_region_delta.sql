-- apply_region_delta.sql
--
-- Data-residency partitioning, step 1 — the "India key".
--
-- Adds `caregivers.region` ('US' | 'IN'): the single authoritative axis for
-- splitting all tenant data by geography. Every sensitive table (patients,
-- cameras, analyses, patient_face_embeddings, events, gateways) resolves its
-- own region through this column via its existing caregiver_id / patient_id
-- FK, so no other table needs the column. Makes three future operations one
-- query (`where region = ?`): lifting a region's data to in-region infra,
-- scoping a breach notification, and answering a data-subject access request.
--
-- Also installs:
--   * a CHECK restricting region to {US, IN}
--   * an immutability trigger that blocks region changes from the client
--     (RLS/PostgREST) path — residency is set by the backend/admin only
--   * region_of(caregiver_id) — the canonical resolver
--
-- ADDITIVE + idempotent (IF NOT EXISTS / OR REPLACE / drop-then-add). No
-- column drops. The only data touched is backfilling existing rows to the
-- 'US' default via the NOT NULL DEFAULT. Prod-safe: adding a NOT NULL column
-- with a constant default is metadata-only in Postgres 11+ (no table rewrite,
-- no long lock). Run BEFORE deploying any backend that reads the column
-- (deploy runbook: DB delta first).

begin;

-- 1. region column — defaults every existing caregiver to 'US'.
alter table public.caregivers
  add column if not exists region text not null default 'US';

-- 2. allowed values (drop-then-add so re-runs converge).
alter table public.caregivers drop constraint if exists caregivers_region_check;
alter table public.caregivers
  add constraint caregivers_region_check check (region in ('US', 'IN'));

-- 3. client-immutability guard. Blocks a region change only when it arrives
--    on the RLS/PostgREST path (JWT claims present, or role authenticated|
--    anon). The backend/admin (direct connection, no JWT, bypasses RLS) may
--    still set/correct it — the authoritative path for tagging India accounts
--    until onboarding sets region at insert time.
create or replace function public.enforce_caregiver_region_immutable()
returns trigger
language plpgsql
as $$
begin
  if new.region is distinct from old.region then
    if current_setting('request.jwt.claims', true) is not null
       or current_user in ('authenticated', 'anon') then
      raise exception
        'caregivers.region is immutable from the client path (data-residency attribute); change it only via an admin/service connection'
        using errcode = 'check_violation';
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists caregivers_region_immutable on public.caregivers;
create trigger caregivers_region_immutable
  before update on public.caregivers
  for each row execute function public.enforce_caregiver_region_immutable();

-- 4. canonical resolver.
create or replace function public.region_of(p_caregiver_id uuid)
returns text
language sql
stable
as $$
  select region from public.caregivers where id = p_caregiver_id;
$$;

commit;

-- ─── Tagging India accounts (run manually, authoritatively) ──────────────
-- India caregivers default to 'US' like everyone else. Until the onboarding
-- flow sets region at insert time (partition step 2), tag them explicitly
-- from an admin/service connection (NOT through PostgREST/RLS, or the
-- immutability trigger will reject it), e.g.:
--
--   update public.caregivers set region = 'IN' where email = 'partner@example.in';
--
-- Verify afterwards:
--   select email, region from public.caregivers order by region;
