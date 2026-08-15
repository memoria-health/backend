-- Home-tab dashboard widget spec (2026-08-15).
--
-- One compiled widget spec per detector, written only on detector
-- create / from-library / refine, and on update when a shape field
-- changes (category, alert_trigger, is_sustained, duration_seconds,
-- schedule_config, when_scope, where_rooms, subject_scope, severity).
-- Built-ins come from the pinned mapping in
-- src/services/detection/catalog.py; customs from
-- src/services/dashboard/widget_compiler.py. NULL = no widget
-- (hidden/internal detectors, or not yet compiled — run
-- tools/backfill_dashboard_widgets.py after applying).
--
-- Shape: {widget, size, title, params, caption_fields, ranges,
--         min_days, compiler, compiled_at}

alter table public.detectors
  add column if not exists dashboard_widget jsonb;
