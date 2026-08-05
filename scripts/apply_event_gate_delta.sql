-- Event gate (additive, backward compatible; default OFF everywhere).
--
-- Replaces the never-applied apply_review_gate_delta.sql. Renamed because
-- `events.review_gate` sat one column away from `events.reviewed` — the
-- CAREGIVER retrospective-review feature — and the two are unrelated:
-- `reviewed` is a family member looking back at an alert they already got,
-- `gate_state` is us deciding whether they get it at all.
--
-- Safe to rename rather than migrate ONLY because the review-gate delta
-- never ran on prod (it was one of the pending deltas). If a local DB has
-- it applied, the old columns are dropped at the bottom of this file.

-- Per-patient opt in. FALSE everywhere means this whole feature is inert
-- until switched on for one patient at a time.
alter table public.patients
  add column if not exists event_gate_enabled boolean not null default false;

-- null | pending | approved | rejected | auto
alter table public.events add column if not exists gate_state text;

-- When the hold started. The auto-release deadline is measured from THIS,
-- not from created_at: the sweeper is the durable half of the safety
-- property and it needs an exact deadline that survives a restart.
alter table public.events add column if not exists gate_held_at timestamptz;

-- Key of the fast-path pre-event clip shown to the operator. Separate from
-- clip_storage_key because it is uploaded seconds earlier, covers only the
-- pre-roll, and is the footage the model actually judged.
alter table public.events add column if not exists gate_clip_key text;

-- Drives the sweeper's claim query. Partial, so it costs nothing while the
-- feature is off (no rows match) and stays tiny when it is on.
create index if not exists events_gate_pending
  on public.events (gate_held_at) where gate_state = 'pending';

-- `review_devices` is intentionally NOT recreated. Operator paging now goes
-- through PagerDuty (src/services/pager.py), so we store no APNs tokens for
-- reviewers, which removes both the table and the RLS surface it needed.
-- The drop is safe because nothing ever wrote to it: the delta that created
-- it never reached prod, and the endpoint that populated it had no client.
drop table if exists public.review_devices;

-- Old column names, dropped for local DBs that applied the superseded delta.
-- No-ops on prod, which never saw them.
alter table public.patients drop column if exists require_review;
drop index if exists public.events_review_pending;
alter table public.events drop column if exists review_gate;
alter table public.events drop column if exists review_clip_key;
