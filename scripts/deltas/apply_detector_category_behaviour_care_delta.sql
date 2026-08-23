-- Detector categories: add 'behaviour' and 'care' (2026-08-05)
--
-- The product pitches four things Memoria watches for — Safety, Routine,
-- Behaviour, Care. The schema modelled two.
--
-- Category is SEMANTIC ONLY. It drives labelling, colour and filtering.
-- Treatment (how urgently something is handled) rides on `detectors.severity`,
-- which already gates the scheduler's emergency exemptions and pack priority.
-- scheduling/scheduler.py:68 records why that split exists: the category axis
-- collapsed to "everything is safety except medication" (2026-07-18), so
-- severity became the exemption contract. Adding two values does not change
-- any behaviour, because no behaviour keys on this column.
--
-- ADDITIVE and non-destructive, per the apply-deltas-only rule:
-- ALTER TYPE ... ADD VALUE appends to the enum. Nothing is dropped, no column
-- is rewritten, and every existing row stays valid.
--
-- Ordering note: this must run BEFORE the backend deploy that ships the new
-- Python enum, or any INSERT/UPDATE carrying 'behaviour'/'care' fails with
-- InvalidTextRepresentation. Same rule as every other delta here.
--
-- IF NOT EXISTS makes it idempotent, so a re-run is a no-op.
--
-- Pre-PG12 could not run ADD VALUE inside a transaction block. Supabase is
-- well past that, but each statement is standalone so this file is safe to
-- apply either way.

alter type detector_category add value if not exists 'behaviour';
alter type detector_category add value if not exists 'care';

-- Verify (should list: safety, routine, behaviour, care)
--   select unnest(enum_range(null::detector_category));
