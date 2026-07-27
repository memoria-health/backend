-- Phase C of the detector-axis reconciliation (2026-07-26): the runtime is
-- fully migrated onto alert_trigger + is_sustained, so the deprecated 3-way
-- `shape` column (and its Postgres enum) are dropped. Idempotent; LOCAL/soak
-- only — prod runs this via the deploy runbook AFTER the code that stops
-- reading `shape` is live. Column first (it references the type), then type.
alter table public.detectors drop column if exists shape;
drop type if exists detector_shape;
