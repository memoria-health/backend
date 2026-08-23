-- Identity-fleet operations delta. Idempotent. PREPARED ONLY — do NOT
-- apply to production until the deploy batch (Aryaman, 2026-08-14);
-- apply locally for development:
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_identity_fleet_ops_delta.sql
--
-- 1. identity_occupancy_samples — one row per node per FleetWatch cycle:
--    how many cameras existed and how many were OCCUPIED (had live
--    tracks) at that instant. Turns "how many homes fit on a node" from
--    an assumption into a measured duty cycle (capacity work,
--    docs/IDENTITY-GPU-ARCHITECTURE.md). Backend-only table: RLS enabled
--    with no policies — service role reads/writes, tenants never see it.
--
-- 2. Dead identity_nodes reaping — nodes C/D/E/F/H from the 2026-08-13/14
--    fleet churn hold stale rows forever (the roster ignores them via
--    last_seen_at, but they clutter every join). Reap anything silent
--    for 7+ days; live nodes re-register on their next poll anyway.

CREATE TABLE IF NOT EXISTS public.identity_occupancy_samples (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sampled_at   timestamptz NOT NULL DEFAULT now(),
    node_id      text        NOT NULL,
    total_cameras    integer NOT NULL,
    occupied_cameras integer NOT NULL
);

CREATE INDEX IF NOT EXISTS identity_occupancy_samples_at_idx
    ON public.identity_occupancy_samples (sampled_at);

ALTER TABLE public.identity_occupancy_samples ENABLE ROW LEVEL SECURITY;

DELETE FROM public.identity_nodes
 WHERE last_seen_at < now() - interval '7 days';
