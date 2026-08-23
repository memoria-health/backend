-- Phase H1 (2026-07-26): collapse the legacy evidence_branch enum
-- (environment/person_motion/person_still) into requires_motion — only the
-- "is it person_motion" bit was ever read downstream (the motion floor).
-- LOCAL/soak only.
alter table public.detectors add column if not exists requires_motion boolean;
update public.detectors
   set requires_motion = (evidence_branch = 'person_motion')
 where evidence_branch is not null;
alter table public.detectors drop column if exists evidence_branch;
