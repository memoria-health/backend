-- Setup test becomes per-camera (2026-08-22, Aryaman): each camera gets to
-- pass the three-finger self-test once, and a camera restart re-arms it.
-- Previously the first pass on ANY camera flipped the single per-patient
-- detector row to is_on=false, so every other camera in the home lost its
-- chance to prove its own alert path.
--
-- Stamped when the test promotes on that camera; cleared on the camera's
-- next LiveKit participant_joined (a reboot/reconnect). NULL = armed.
-- Reads ONLY as the setup-test arming check — it gates nothing else.
-- Idempotent.
ALTER TABLE public.cameras
    ADD COLUMN IF NOT EXISTS setup_test_passed_at timestamptz;

-- Carry the old per-patient state over, in this order.
--
-- 1. Stamp every camera that ALREADY passed, so the re-enable below cannot
--    make a settled home re-run a test it has done. The camera is read off
--    the event the pass wrote; homes that never passed stamp nothing.
UPDATE public.cameras c
   SET setup_test_passed_at = e.occurred_at
  FROM (
    SELECT ev.camera_id, max(ev.occurred_at) AS occurred_at
      FROM public.events ev
      JOIN public.detectors d ON d.id = ev.detector_id
     WHERE d.identifier = 'setup_test'
     GROUP BY ev.camera_id
  ) e
 WHERE c.id = e.camera_id
   AND c.setup_test_passed_at IS NULL;

-- 2. Re-enable the rows the old fire-once switched off. Arming is the
--    camera's job now, so a false here is dead state that would keep the
--    test from ever running again — including on cameras added later, which
--    never got a chance in the first place.
UPDATE public.detectors
   SET is_on = true
 WHERE identifier = 'setup_test'
   AND is_on = false;
