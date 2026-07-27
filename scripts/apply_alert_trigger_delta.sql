-- Alerting spine (2026-07-26): the detector-axis reconciliation replaces the
-- 3-way `shape` (point_in_time/state/scheduled) with `alert_trigger`
-- (leading/trailing/duration/schedule) + `is_sustained` (temporal nature) +
-- `duration_seconds` (the "too long" threshold). Idempotent; LOCAL/soak only
-- — prod runs this via the deploy runbook (DB delta before backend deploy).
-- `shape` is kept as a bridge and dropped in the Phase C delta once the
-- runtime no longer reads it.
alter table public.detectors
  add column if not exists alert_trigger text not null default 'leading'
    check (alert_trigger in ('leading','trailing','duration','schedule'));
alter table public.detectors
  add column if not exists is_sustained boolean not null default false;
alter table public.detectors
  add column if not exists duration_seconds integer;

-- Backfill from the existing shape so no row diverges during the bridge.
update public.detectors
   set alert_trigger = case shape when 'scheduled' then 'schedule' else 'leading' end,
       is_sustained  = (shape = 'state')
 where true;
