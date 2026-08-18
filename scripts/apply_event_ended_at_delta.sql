-- Sustained episodes persist their falling edge (2026-08-18): ended_at
-- extends on every ongoing tick, so its final value is the last positive
-- tick. Null for instant events. Idempotent.
alter table public.events add column if not exists ended_at timestamptz;
