-- One-shot, idempotent schema delta to add the product-feedback table
-- to a running Supabase WITHOUT a full db reset. Companion to
-- scripts/deltas/apply_caregiver_devices_delta.sql + apply_face_embeddings_delta.sql.
--
-- Backs POST /v1/feedback (Settings → Send feedback in the iOS app).
-- Every statement is guarded (IF NOT EXISTS / pg_policies check) so
-- re-running is a no-op. Apply via:
--   psql "<connection string>" -f scripts/deltas/apply_feedback_delta.sql
-- or paste into the Supabase SQL editor.

begin;

create table if not exists public.feedback (
  id                uuid primary key default gen_random_uuid(),
  caregiver_id      uuid not null references public.caregivers(id) on delete cascade,
  rating            smallint,
  whats_working     text,
  whats_frustrating text,
  category          text,
  created_at        timestamptz not null default now()
);

create index if not exists feedback_caregiver_idx
  on public.feedback(caregiver_id);

alter table public.feedback enable row level security;

do $$ begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'feedback'
      and policyname = 'feedback_owner'
  ) then
    create policy feedback_owner on public.feedback
      for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());
  end if;
end $$;

commit;
