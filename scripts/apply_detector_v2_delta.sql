-- Detection v2 (P1): adds the six-dimension columns to public.detectors on
-- an EXISTING database, and backfills the 24 built-ins. Idempotent. Apply
-- BEFORE deploying the P1 backend (DB-before-deploy — see
-- docs/PROD_MIGRATION_RUNBOOK.md):
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_detector_v2_delta.sql
--
-- NOTHING reads these columns until P3 (scheduler/packer) except `severity`
-- (event promotion prefers the row value, falling back to the in-code
-- catalog map when NULL) — so this delta is additive and safe alongside a
-- pre-P1 backend. Text + named CHECK constraints instead of new enum types:
-- same guarantees, none of the enum-alter lifecycle pain.
--
-- Column meanings (spec: docs/architecture/detection-architecture-v2.md):
--   evidence_branch     which scene branch feeds this detector evidence
--   subject_scope       who it is about (while attribution is unresolved the
--                       RULE is fixed, not configurable: run — suppress only
--                       on VERIFIED not-patient evidence)
--   bucket              perception instrument (locked recipe)
--   needs_audio         must land in a call that carries sound
--   evidence_unit       verdict | instance | sample
--   eventing            jsonb: {shape, k, n, M, T, absence_deadline}
--   duty_cycle          every_eligible_tick | sampled | scheduled_only
--   severity            within-tick precedence (moves out of catalog._SEVERITY)
--   cooldown_seconds    per-detector re-fire spacing for point shapes
--   degradation_warning none | distance_class (perpetual app notice)

alter table public.detectors
  add column if not exists evidence_branch text,
  add column if not exists subject_scope text,
  add column if not exists bucket text,
  add column if not exists needs_audio boolean not null default false,
  add column if not exists evidence_unit text,
  add column if not exists eventing jsonb,
  add column if not exists duty_cycle text,
  add column if not exists severity integer,
  add column if not exists cooldown_seconds integer,
  add column if not exists degradation_warning text;

do $$ begin
  alter table public.detectors add constraint detectors_evidence_branch_check
    check (evidence_branch in ('environment', 'person_motion', 'person_still'));
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.detectors add constraint detectors_subject_scope_check
    check (subject_scope in ('patient', 'anyone', 'environment'));
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.detectors add constraint detectors_bucket_check
    check (bucket in ('scene', 'person', 'fine_motion', 'rapid', 'face', 'snapshot'));
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.detectors add constraint detectors_evidence_unit_check
    check (evidence_unit in ('verdict', 'instance', 'sample'));
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.detectors add constraint detectors_duty_cycle_check
    check (duty_cycle in ('every_eligible_tick', 'sampled', 'scheduled_only'));
exception when duplicate_object then null; end $$;

do $$ begin
  alter table public.detectors add constraint detectors_degradation_warning_check
    check (degradation_warning in ('none', 'distance_class'));
exception when duplicate_object then null; end $$;

-- ── Backfill the built-ins (rows keep NULLs for anything not listed; the
-- catalog map in src/services/detection/catalog_v2.py is the source of
-- truth this block was generated from — keep the two in sync). ──────────
update public.detectors as d
set evidence_branch     = v.branch,
    subject_scope       = v.subject,
    bucket              = v.bucket,
    needs_audio         = v.needs_audio,
    evidence_unit       = v.unit,
    duty_cycle          = v.duty,
    severity            = v.severity,
    degradation_warning = v.warn
from (values
  -- identifier            branch            subject        bucket        audio  unit       duty                    sev  warn
  ('fall',                 'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  95,  'none'),
  ('agitation',            'person_motion',  'patient',     'scene',      true,  'sample',  'sampled',              60,  'none'),
  ('seizure',              'person_motion',  'patient',     'rapid',      false, 'verdict', 'every_eligible_tick',  96,  'none'),
  ('wandering',            'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  55,  'none'),
  ('medication',           'person_motion',  'patient',     'person',     false, 'verdict', 'every_eligible_tick',  20,  'none'),
  ('climbing',             'person_motion',  'patient',     'scene',      false, 'sample',  'every_eligible_tick',  66,  'none'),
  ('distress',             'person_motion',  'patient',     'scene',      true,  'sample',  'every_eligible_tick',  82,  'none'),
  ('child_crib_exit',      'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  65,  'none'),
  ('child_cord_entanglement','person_motion','patient',     'person',     false, 'verdict', 'every_eligible_tick',  97,  'distance_class'),
  ('child_stairs',         'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  73,  'none'),
  ('child_hazard_reach',   'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  70,  'none'),
  ('child_head_impact',    'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  93,  'none'),
  ('child_wandering',      'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  55,  'none'),
  ('child_device_usage',   'person_still',   'patient',     'person',     false, 'sample',  'sampled',              25,  'none'),
  ('calling_for_help',     'person_motion',  'patient',     'scene',      true,  'verdict', 'every_eligible_tick',  88,  'none'),
  ('choking',              'person_motion',  'patient',     'person',     false, 'verdict', 'every_eligible_tick', 100,  'none'),
  ('meal_eaten',           'person_motion',  'patient',     'person',     false, 'verdict', 'every_eligible_tick',  22,  'none'),
  ('stove_left_on',        'environment',    'environment', 'snapshot',   false, 'sample',  'sampled',              74,  'distance_class'),
  ('emergency',            'person_motion',  'patient',     'scene',      false, 'verdict', 'every_eligible_tick',  80,  'none'),
  ('possible_intruder',    'person_motion',  'anyone',      'scene',      false, 'verdict', 'every_eligible_tick',  78,  'none'),
  ('env_hazard_glass',     'environment',    'environment', 'snapshot',   false, 'verdict', 'sampled',              62,  'distance_class'),
  ('env_hazard_water',     'environment',    'environment', 'snapshot',   false, 'verdict', 'sampled',              56,  'distance_class'),
  ('fire_smoke',           'environment',    'environment', 'scene',      false, 'verdict', 'every_eligible_tick',  98,  'none'),
  ('setup_test',           'person_motion',  'anyone',      'scene',      false, 'verdict', 'every_eligible_tick', 200,  'none')
) as v(identifier, branch, subject, bucket, needs_audio, unit, duty, severity, warn)
where d.identifier = v.identifier
  and d.source = 'preset'
  -- Re-run guard covers EVERY column this UPDATE sets (review finding: a
  -- severity-only guard would silently skip future non-severity edits on
  -- an already-migrated DB). No-op when all values are current.
  and (
       d.evidence_branch     is distinct from v.branch
    or d.subject_scope       is distinct from v.subject
    or d.bucket              is distinct from v.bucket
    or d.needs_audio         is distinct from v.needs_audio
    or d.evidence_unit       is distinct from v.unit
    or d.duty_cycle          is distinct from v.duty
    or d.severity            is distinct from v.severity
    or d.degradation_warning is distinct from v.warn
  );
-- NOTE on the constraint blocks above: `duplicate_object` no-ops mean a
-- future change to a CHECK's legal-value list needs a NEW constraint name
-- (drop old + add new) — re-running the same name keeps the old body.
