-- Onboarding country pick, separate from data-residency region.
--
-- Adds caregivers.country + patients.country to an EXISTING database (local dev
-- or prod). schema.sql includes them for fresh creates; this delta is for a DB
-- that predates the columns. Idempotent — safe to re-run.
--
-- Model: `country` is the full ISO 3166-1 alpha-2 pick the caregiver made at
-- signup (caregivers) or patient onboarding (patients), recorded verbatim.
-- `region` ('US' | 'IN') is DERIVED from it server-side (services/region.
-- region_for_country: India → 'IN', else 'US') and remains the only value that
-- drives storage routing + legal scoping. So the full country is captured for
-- records/analytics without changing what we physically support.
--
-- Existing rows come out 'US' (the default), matching their existing region
-- default — no backfill needed. Both columns are immutable from the client path
-- (the region-immutability triggers below now also freeze country).
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_country_delta.sql
-- or paste into the Supabase SQL editor. Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.caregivers
  add column if not exists country text not null default 'US'
    check (country ~ '^[A-Z]{2}$');

alter table public.patients
  add column if not exists country text not null default 'US'
    check (country ~ '^[A-Z]{2}$');

-- Extend the residency-immutability guards to also freeze `country` (region is
-- derived from it, so the two must not drift). `create or replace` is idempotent.
create or replace function public.enforce_caregiver_region_immutable()
returns trigger
language plpgsql
as $$
begin
  if new.region is distinct from old.region
     or new.country is distinct from old.country then
    if current_setting('request.jwt.claims', true) is not null
       or current_user in ('authenticated', 'anon') then
      raise exception
        'caregivers.region/country are immutable from the client path (data-residency attributes); change them only via an admin/service connection'
        using errcode = 'check_violation';
    end if;
  end if;
  return new;
end;
$$;

create or replace function public.enforce_patient_region_immutable()
returns trigger
language plpgsql
as $$
begin
  if new.region is distinct from old.region
     or new.country is distinct from old.country then
    if current_setting('request.jwt.claims', true) is not null
       or current_user in ('authenticated', 'anon') then
      raise exception
        'patients.region/country are immutable from the client path (data-residency attributes); change them only via an admin/service connection'
        using errcode = 'check_violation';
    end if;
  end if;
  return new;
end;
$$;
