-- Detection v2 (P2): drops detectors.is_builtin. Redundant with
-- `source`: every write path set preset⇔true / custom⇔false (one early
-- window let customs default to true — harmless, nothing ever READ the
-- column), and no code needed the two to diverge. All code now keys
-- catalog membership on source = 'preset' (the P1 backfill WHERE in
-- apply_detector_v2_delta.sql included; that WHERE also gates on
-- built-in identifiers, so the swap selects the identical row set).
--
-- ⚠ ORDERING — OPPOSITE of the usual DB-before-deploy rule: this delta
-- is DESTRUCTIVE, so it must run AFTER deploying a backend that no
-- longer references is_builtin (the pre-P2 backend sets the column on
-- every detector INSERT and would break the moment it's gone). Safe on
-- the old DB in the meantime: the column has a default, so the new
-- backend's INSERTs work fine before the drop.
--
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_drop_is_builtin_delta.sql
--
-- Idempotent: `if exists` makes re-runs a no-op.

alter table public.detectors
  drop column if exists is_builtin;
