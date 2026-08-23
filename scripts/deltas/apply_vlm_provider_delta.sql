-- Adds the per-account VLM routing column to an EXISTING database (local
-- dev, or prod when we eventually deploy). schema.sql already includes it
-- for fresh creates; this delta is for a DB that predates the column.
--
-- Idempotent. Apply locally with:
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_vlm_provider_delta.sql
-- or paste into the Supabase SQL editor.
--
-- vlm_provider: 'gemini' | 'openai' | 'openrouter'. NULL = use the
-- server's default_vlm_provider. Flip it per caregiver to route a single
-- family onto a different model — no deploy.
alter table public.caregivers
  add column if not exists vlm_provider text;

-- The analyses.provider CHECK constraint must allow 'openrouter' or every
-- OpenRouter (Qwen) tick succeeds at the model but fails on INSERT.
alter table public.analyses
  drop constraint if exists analyses_provider_check;
alter table public.analyses
  add constraint analyses_provider_check
  check (provider in ('gemini', 'openai', 'openrouter', 'skipped'));
