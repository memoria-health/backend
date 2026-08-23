-- Facilities audio kill (2026-07-28, Aryaman): audio recording is illegal
-- in care facilities (and two-party-consent contexts) — patient-level
-- switch that disables audio CAPTURE at the SOURCE. Publishers (gateway;
-- iOS camera app follow-up) simply never publish the audio track, so
-- clips, transcripts, and speech fields go silent for free downstream;
-- needs_audio detectors degrade via the existing sched_degraded_audio
-- machinery. Default TRUE (family homes unchanged). Idempotent.
ALTER TABLE public.patients
    ADD COLUMN IF NOT EXISTS audio_capture_enabled boolean NOT NULL DEFAULT true;
