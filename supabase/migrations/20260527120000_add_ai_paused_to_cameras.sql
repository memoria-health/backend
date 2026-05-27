-- Adds the `ai_paused` flag to public.cameras for an LLM-only pause
-- toggle independent of `disabled` (which also hides the tile from
-- the live grid). When `ai_paused=true`, the detection tick writes a
-- synthetic "skipped" row with skipped_reason='ai_paused' instead of
-- calling Gemini; video keeps streaming and the tile stays visible.
--
-- Additive only: NOT NULL with a safe default; existing rows
-- automatically pick up `false`, preserving today's behavior.

alter table public.cameras
  add column ai_paused boolean not null default false;
