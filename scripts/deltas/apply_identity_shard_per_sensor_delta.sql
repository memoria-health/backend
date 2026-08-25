-- The identity process is per SENSOR, not per patient.
--
-- WHY. A slot was allocated on `public.patients`, one process per
-- patient, and the process joined `patient_<uuid>` -- so "a home" and "a
-- LiveKit room" and "a patient" were the same thing and the shard key
-- could be any of them. `apply_account_room_delta.sql` made every sensor
-- on an account share ONE room, and that identity collapsed:
--
--   * two residents in one home each claim a slot, and BOTH processes
--     subscribe to the same account room -- twice the models, twice the
--     GPU, on identical video;
--   * a sensor covering nobody has no patient row to be allocated
--     through, so it can never be given a process at all -- and an
--     unassigned sensor is exactly what an installer leaves behind;
--   * the router had to pick one patient out of several covering a
--     sensor to find its port, which is `is_primary` growing back.
--
-- A sensor is the thing that actually produces frames, it is unique
-- within an account, and one process can resolve every patient the
-- sensor covers by looking at coverage. So the sensor is the shard key,
-- and "which people might this be" becomes a loop inside the process
-- rather than a fact baked into which process exists.
--
-- NOTE ON DEPLOY: this does NOT backfill ports. Every node re-claims its
-- sensors on its next poll (~30s) and re-grounds, so expect one window
-- of identity churn per node on rollout. Backfilling was possible for
-- single-patient homes only, and a partial backfill would have left the
-- fleet in two states at once, which is worse to reason about than one
-- known re-ground.

BEGIN;

-- ── 1. The slot moves to the sensor ─────────────────────────────────
-- The FK is load-bearing, not decoration. `apply_identity_fleet_ops_delta`
-- reaps identity_nodes rows unconditionally after 7 days; without
-- ON DELETE SET NULL a sensor keeps pointing at a node that no longer
-- exists, and the roster's release step only frees rows whose node is IN
-- identity_nodes -- so a dangling id is never released, never re-claimed,
-- and the sensor is permanently identity-blind while paging forever.
ALTER TABLE public.cameras
  ADD COLUMN IF NOT EXISTS identity_node_id text
    REFERENCES public.identity_nodes(node_id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS identity_port    integer;

-- Carried over from the patients columns. A port outside the range the
-- roster allocates from is unreachable, and silently so.
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint
                 WHERE conname = 'cameras_identity_port_range') THEN
    ALTER TABLE public.cameras
      ADD CONSTRAINT cameras_identity_port_range
      CHECK (identity_port IS NULL OR identity_port BETWEEN 8012 AND 8111);
  END IF;
END $$;

-- Same backstop the patients table had: two nodes polling at the same
-- instant must not be handed the same port. The claim statement pairs
-- free ports by row_number and locks SKIP LOCKED; this is what catches
-- it if they ever race anyway.
CREATE UNIQUE INDEX IF NOT EXISTS cameras_identity_node_port_uidx
  ON public.cameras (identity_node_id, identity_port)
  WHERE identity_node_id IS NOT NULL AND identity_port IS NOT NULL;

-- Claim and shed both filter on this constantly.
CREATE INDEX IF NOT EXISTS cameras_identity_node_idx
  ON public.cameras (identity_node_id);

-- The claim's hot query is "unassigned, oldest first". Partial, because
-- the assigned rows are the majority and it never scans them. This
-- matters MORE than it did on patients: the unassigned pool used to be a
-- handful of homes and is now every sensor row.
CREATE INDEX IF NOT EXISTS cameras_identity_unassigned_idx
  ON public.cameras (created_at)
  WHERE identity_node_id IS NULL;

-- ── 2. And leaves the patient ───────────────────────────────────────
-- Dropped rather than left in place: a stale allocation column that
-- nothing writes is exactly the sort of thing a future query joins
-- against and silently gets nothing from.
DROP INDEX IF EXISTS public.patients_identity_node_port_uidx;

ALTER TABLE public.patients
  DROP COLUMN IF EXISTS identity_node_id,
  DROP COLUMN IF EXISTS identity_port;

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Reversible: re-add the two patient columns and the index, and every
-- node re-claims on its next poll exactly as it does rolling forward.
-- No allocation is worth preserving in either direction -- a port is a
-- lease, not a fact about a home.
