-- analyses.provider CHECK omitted 'dashscope' (found 2026-07-27):
-- any tick routed to DashScope analyzed but failed to PERSIST — the
-- verdict was computed and then thrown away at the insert. schema.sql
-- includes it for fresh creates; this delta is for existing DBs.
-- Idempotent — safe to re-run.
alter table public.analyses drop constraint if exists analyses_provider_check;
alter table public.analyses add constraint analyses_provider_check
  check (provider in ('gemini', 'openai', 'openrouter', 'dashscope', 'skipped'));
