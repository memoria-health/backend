-- snapshot bucket retired (2026-07-18): stills are dead per the
-- experiment record; environment checks ride the scene call (which
-- the packer already did unconditionally). Idempotent.
update public.detectors set bucket = 'scene' where bucket = 'snapshot';
