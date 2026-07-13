-- apply_gateway_ownership_delta.sql
--
-- Camera -> gateway ownership (multi-gateway robustness).
--
-- Before this, a gateway camera was owned only by a CAREGIVER
-- (cameras.patient_id -> patient -> caregiver). With >1 gateway box per
-- caregiver that caused three problems:
--   1. the roster fanned EVERY gateway camera to EVERY box (each box then
--      skipped the ones it had no local RTSP mapping for),
--   2. liveness was fleet-collective — any box's roster poll stamped
--      last_heartbeat_at on ALL the caregiver's gateway cameras, so a dead
--      box's cameras still looked "online" while a sibling box polled,
--   3. nothing stopped two boxes claiming the same camera.
--
-- This ties each gateway camera to the specific box that runs it, so the
-- roster + heartbeat can scope per-box.
--
-- ADDITIVE + idempotent. Safe to re-run. Requires apply_gateway_delta.sql
-- (the public.gateways table) to have run first. Run BEFORE deploying the
-- backend that scopes the roster by gateway_id (DB-before-deploy).

BEGIN;

-- 1. Ownership column: which gateway box runs this camera. Nullable so
--    iphone/pi rows and not-yet-assigned gateway rows stay NULL. ON DELETE
--    SET NULL: deleting a box orphans its cameras (they fall back to the
--    roster's IS NULL path) rather than cascading the camera rows away.
ALTER TABLE public.cameras
  ADD COLUMN IF NOT EXISTS gateway_id uuid
  REFERENCES public.gateways(id) ON DELETE SET NULL;

-- 2. The roster + heartbeat now filter by gateway_id.
CREATE INDEX IF NOT EXISTS idx_cameras_gateway_id
  ON public.cameras (gateway_id);

-- 3. Backfill existing gateway cameras to their caregiver's gateway, but
--    ONLY when that caregiver has exactly ONE gateway (unambiguous). Any
--    multi-gateway caregiver (none today) stays NULL and keeps working via
--    the roster's OR-IS-NULL fallback until assigned. Idempotent: only
--    touches rows still NULL.
UPDATE public.cameras c
SET gateway_id = g.id
FROM public.patients p, public.gateways g
WHERE c.source_type = 'gateway'
  AND c.gateway_id IS NULL
  AND c.patient_id = p.id
  AND g.caregiver_id = p.caregiver_id
  AND (SELECT count(*) FROM public.gateways g2
       WHERE g2.caregiver_id = p.caregiver_id) = 1;

COMMIT;
