-- `camera_patients` writes must name a camera you own.
--
-- The policy from apply_camera_ownership_delta.sql constrains only the
-- PATIENT side:
--
--   for all using (public.has_patient_access(patient_id))
--   with check (public.has_patient_access(patient_id))
--
-- Nothing constrains `camera_id`, and `authenticated` holds INSERT on the
-- table through PostgREST with the anon key that ships inside the app
-- binary. So any signed-in caregiver could insert
-- (someone_elses_camera_id, their_own_patient_id) -- a row that then
-- satisfies the SECOND arm of the `cameras_member` policy and hands them
-- SELECT/UPDATE/DELETE on that camera row: its livekit_room, its
-- livekit_identity, its ingest_token_hash, its api_key_hash. `cameras` is
-- in the realtime publication with replica identity full, so they would
-- get live row broadcasts too.
--
-- Exploiting it needs the victim's camera UUID, which is a v4 and not
-- enumerable (the `cam_<8 hex>` identity leaks only 32 bits of it), so
-- this is a hole rather than an open door. It is still one the coverage
-- migration opened: the previous policy read `cameras.patient_id`
-- directly and had no row for an attacker to insert.
--
-- The API path was never affected -- `set_camera_coverage` validates the
-- camera against the caller's account before writing. This closes the
-- direct-PostgREST path to match.

BEGIN;

DROP POLICY IF EXISTS camera_patients_member ON public.camera_patients;

CREATE POLICY camera_patients_member ON public.camera_patients
  FOR ALL
  USING (
    public.has_patient_access(patient_id)
    AND EXISTS (
      SELECT 1 FROM public.cameras c
       WHERE c.id = camera_id AND c.caregiver_id = auth.uid()
    )
  )
  WITH CHECK (
    public.has_patient_access(patient_id)
    AND EXISTS (
      SELECT 1 FROM public.cameras c
       WHERE c.id = camera_id AND c.caregiver_id = auth.uid()
    )
  );

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Re-create the policy with the patient-only clauses. Note that doing so
-- reopens the write path above, so it is only correct as part of
-- reverting the coverage model itself. No data is touched either way --
-- the backend reaches this table through the service role, which
-- bypasses RLS, so nothing the API does is affected by this delta.
