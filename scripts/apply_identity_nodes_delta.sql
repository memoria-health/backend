-- Identity node roster — which GPU node serves which home (2026-08-11).
--
-- WHY
-- ---
-- One identity process serves exactly ONE room: gpu/identity/loop.py's
-- IdentityHome holds a single patient_id and a single seat. A GPU node runs N
-- of them next to one perception process, so "which homes does this node
-- serve?" needs an answer somewhere.
--
-- Until now that answer was a single global setting, `identity_state_url`, and
-- a single shared IdentityClient. With more than one node that fails in the
-- worst possible way: the first room's cameras prefix-match the payload and
-- get real verdicts, and every OTHER patient silently gets
-- IdentityResult.empty() forever — no error, no log, just no identity. This
-- table is what makes a second node possible at all.
--
-- SHAPE
-- -----
-- Deliberately the same shape as gateways.tunnel_port: the fleet member polls
-- a roster endpoint, the backend lazily assigns, and the assignment lives in
-- Postgres rather than in config files on N machines. That pattern is already
-- in production for the gateway boxes, concurrency-safe, and costs nothing in
-- steady state.
--
--   identity_nodes            one row per GPU node, heartbeated
--   patients.identity_node_id which node currently serves this home
--
-- A node claims unassigned patients up to its capacity. If a node stops
-- heartbeating, its patients are released and any node with spare capacity
-- picks them up on its next poll. Adding capacity is "start another node".
--
-- Idempotent — safe to re-run.

-- ── The nodes ──────────────────────────────────────────────────────────
create table if not exists public.identity_nodes (
  -- The RunPod pod id. Stable for the life of the node, and the thing you
  -- can paste into the RunPod console when something looks wrong.
  node_id      text primary key,

  -- Where the backend reaches this node's identity processes. A tailnet
  -- MagicDNS name in production (e.g. http://nemara-gpu-abc:8012) precisely
  -- so it survives RunPod reassigning the pod's direct-TCP endpoint, which
  -- happens on every restart.
  base_url     text not null,

  -- How many rooms this node will accept. Set per-node because it depends on
  -- NEMARA_PERCEPTION_CONCURRENCY and the GPU: ~4 concurrently-occupied rooms
  -- at concurrency=1, ~8 at 2 (measured 1.6-1.8x, not yet latency-validated).
  capacity     int  not null default 4 check (capacity > 0),

  -- Heartbeat. The roster endpoint refreshes this on every poll; a node that
  -- stops polling has its patients released to whoever polls next.
  last_seen_at timestamptz not null default now(),
  created_at   timestamptz not null default now()
);

-- ── The assignment ─────────────────────────────────────────────────────
-- Nullable on purpose. NULL means "unassigned, claimable" and is both the
-- initial state and what a node failure returns a home to. `on delete set
-- null` so removing a node row releases its homes rather than orphaning them
-- behind a foreign key.
alter table public.patients
  add column if not exists identity_node_id text
    references public.identity_nodes(node_id) on delete set null;

-- The roster's hot query is "unassigned patients, oldest first". Partial
-- index: the assigned rows are the majority and are never scanned by it.
create index if not exists patients_identity_unassigned_idx
  on public.patients (created_at)
  where identity_node_id is null;

-- Counting a node's current load, on every poll from every node.
create index if not exists patients_identity_node_idx
  on public.patients (identity_node_id)
  where identity_node_id is not null;

-- ── RLS ────────────────────────────────────────────────────────────────
-- Supabase grants anon/authenticated full access to every table in `public`,
-- and anon is the PUBLISHABLE key shipped inside the iOS binary and the web
-- bundle — so RLS is the only access control. See
-- scripts/apply_auto_enable_rls_delta.sql; the trigger there would catch this
-- anyway, but a table should not rely on a safety net it could outlive.
--
-- NO POLICY IS CREATED, deliberately. RLS with no policy denies all non-owner
-- access, and nothing outside the backend has any business reading which GPU
-- serves which home. The backend uses the service role, which bypasses RLS.
alter table public.identity_nodes enable row level security;

-- ── Verify ─────────────────────────────────────────────────────────────
--   select node_id, base_url, capacity,
--          now() - last_seen_at as heartbeat_age,
--          (select count(*) from public.patients p
--             where p.identity_node_id = n.node_id) as homes
--     from public.identity_nodes n order by created_at;
--
--   select count(*) as unassigned from public.patients
--    where identity_node_id is null;
