-- Phase H2 (2026-07-26): collapse the 3 "where" reps. where_rooms (intent) +
-- cameras_detectors (runtime) are the real reps; the where_scope scalar was a
-- synced-derived duplicate. Backfill where_rooms from where_scope for any row
-- that lacks it, then drop where_scope. It's re-derived on serialize.
-- LOCAL/soak only.
update public.detectors
   set where_rooms = case when where_scope = 'ALL ROOMS' then null
                          else jsonb_build_array(where_scope) end
 where where_rooms is null;
alter table public.detectors drop column if exists where_scope;
