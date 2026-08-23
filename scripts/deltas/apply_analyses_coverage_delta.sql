-- Observations resolve through coverage, not through the shim.
--
-- `apply_camera_ownership_delta.sql` made a camera cover 0..N patients and
-- said `cameras.patient_id` would stay "authoritative until the application
-- no longer reads them". This is the reader migration for the observation
-- tables, and it has to land before a shared sensor is assigned to a second
-- person, because that is the configuration where the shim is not merely
-- redundant but WRONG:
--
--   * `analyses.patient_id` is stamped from `cameras.patient_id` at tick
--     time, so on a shared sensor every row belongs to the earliest-added
--     resident. The second resident has no analyses at all -- no episode
--     boundary can resolve for them, `home_report` shows a covered person
--     as having no camera and no activity, and every insights widget is
--     empty. It looks like a quiet day rather than a missing join.
--
--   * `cameras.patient_id = any(:ids)` picks which sensors belong to a
--     patient, so a shared sensor is invisible to its second resident in
--     the camera lists too.
--
-- This delta itself is a BACKFILL AND AN INDEX. It changes no column and
-- drops nothing: `cameras.patient_id` and `analyses.patient_id` both stay,
-- so a rollback to the previous application build is clean.
--
-- Idempotent. Safe to re-run.

-- ── 1. Make the coverage invariant true for pre-migration rows ───────
--
-- `set_coverage` is the only writer of `cameras.patient_id` and sets it to
-- the earliest covered patient, or NULL when coverage is empty. So a
-- non-null shim implies a `camera_patients` row -- for every camera whose
-- coverage was written by that function. A camera created before the
-- ownership delta can predate it, and for that camera switching the
-- readers to coverage would silently drop its whole history.
--
-- Insert the missing links rather than trusting the invariant.
insert into public.camera_patients (camera_id, patient_id, added_at)
select c.id, c.patient_id, coalesce(c.created_at, now())
  from public.cameras c
 where c.patient_id is not null
   and not exists (
     select 1 from public.camera_patients cp where cp.camera_id = c.id
   )
on conflict do nothing;

-- ── 2. The index the new predicate reads ─────────────────────────────
--
-- The readers become `camera_id in (select camera_id from camera_patients
-- where patient_id = ...)`, so the hot path is a lookup by patient into
-- the join table and then `analyses` by camera + time.
--
-- `analyses (camera_id, captured_at)` already exists as
-- `analyses_camera_captured_at_uniq` and `camera_patients (patient_id)`
-- as `camera_patients_patient_idx`, both from schema.sql. These are
-- restated for a database that somehow skipped either, so the new
-- readers can never fall back to a sequential scan over a table
-- measured at 162k rows and growing without bound.
--
-- DISTINCT NAMES. `analyses_camera_created_idx` is already taken by an
-- index on (camera_id, created_at DESC) — a different column. Reusing
-- a name under `if not exists` is a silent no-op that leaves the wrong
-- index in place and reads as if it worked.
create index if not exists analyses_camera_captured_at_idx
  on public.analyses (camera_id, captured_at desc);

create index if not exists camera_patients_patient_id_idx
  on public.camera_patients (patient_id);

-- ── 3. What is deliberately NOT here ─────────────────────────────────
--
-- `analyses.patient_id` is not dropped and not made nullable. It stays as
-- a denormalization for RLS scoping alongside `caregiver_id`, and nothing
-- filters a person's observations on it any more. Dropping it is a THIRD
-- delta, after this one has soaked, on the same reasoning the ownership
-- delta gave for keeping the columns it superseded.

-- ── 4. Verification ──────────────────────────────────────────────────
--
--   -- must be 0: every camera with a shim has a coverage row
--   select count(*) from public.cameras c
--    where c.patient_id is not null
--      and not exists (select 1 from public.camera_patients cp
--                       where cp.camera_id = c.id);
--
--   -- the shared sensor, after assigning it to a second patient: both
--   -- rows present, and each patient now resolves to the same camera
--   select patient_id, camera_id from public.camera_patients
--    where camera_id = '<shared camera uuid>';
