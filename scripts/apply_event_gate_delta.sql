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

-- Per-patient. ON BY DEFAULT (product decision, 2026-08-05).
--
-- READ THIS BEFORE APPLYING TO PRODUCTION. Default-true plus the backfill
-- below means every EXISTING patient starts having their alerts held for
-- up to 60s, and masked footage of their home shown to a Memoria reviewer,
-- the moment this runs. That is the intended behaviour, but it is a change
-- made TO households rather than BY them, so it should not land before the
-- caregiver-facing "Evaluation mode" setting and its explainer are live in
-- all three clients. Applying this alone leaves families opted in with no
-- way to see it or turn it off.
alter table public.patients
  add column if not exists event_gate_enabled boolean not null default true;
-- Separate statement so re-runs and databases that already have the column
-- (created with the old default of false) converge on the same state.
alter table public.patients
  alter column event_gate_enabled set default true;
update public.patients set event_gate_enabled = true
  where event_gate_enabled = false;

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

-- Hide gated events from the caregiver. WITHOUT THIS THE GATE DOES NOTHING
-- USEFUL: `events` is in the supabase_realtime publication, so the row is
-- broadcast to the iOS app the instant it is inserted, which happens before
-- the gate runs. Observed live on 2026-08-05 — the alert appeared in the app
-- a second before the operator was even paged, and a suppressed false alarm
-- would have stayed in the feed forever.
--
-- Realtime honours RLS, so narrowing the policy hides held events from the
-- broadcast and from direct reads at once, with no client release.
--
--   pending  -> hidden while the operator decides
--   rejected -> hidden forever (that is what suppression means)
--   approved -> visible
--   auto     -> visible (the 60s timer released it)
--   null     -> visible; every patient with the gate off is unaffected
drop policy if exists events_member on public.events;
create policy events_member on public.events
  using (
    has_patient_access(patient_id)
    and coalesce(gate_state, 'none') not in ('pending', 'rejected')
  );

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
