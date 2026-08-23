-- Auto-enable RLS on every new table in `public` (2026-08-03).
--
-- WHY
-- ---
-- Supabase grants `anon` and `authenticated` full
-- SELECT/INSERT/UPDATE/DELETE/TRUNCATE on every table in `public`, and
-- PostgREST exposes them. RLS is therefore the ONLY access control, and `anon`
-- is the PUBLISHABLE key — it ships inside the iOS binary and the web bundle.
--
-- A `create table` that forgets `enable row level security` is not a small
-- omission; it publishes the table to anyone who can read the app bundle. That
-- happened on 2026-08-03 with sleep_nights, activity_days and review_devices.
--
-- test/unit/test_rls_invariant.py catches this at authoring time. This trigger
-- is the second layer: it catches a table created by a migration nobody
-- reviewed, by hand in the SQL editor, or by tooling.
--
-- Supabase ships this as "Enable automatic RLS" on new projects. `nemara`
-- predates that option, so we install it ourselves.
--
-- FAIL CLOSED
-- -----------
-- RLS with no policy denies all non-owner access. A new table is therefore
-- unreadable via PostgREST until someone writes a policy deliberately. That is
-- the intended behaviour: a forgotten policy shows up as "my query returns
-- nothing", which gets fixed, rather than as a silent public exposure, which
-- does not. The backend uses the service role, which bypasses RLS, so its
-- writes are unaffected.
--
-- Idempotent: safe to re-run.

create or replace function public.auto_enable_rls()
  returns event_trigger
  language plpgsql
  security definer
  set search_path = public, pg_catalog
as $$
declare
  obj record;
begin
  for obj in
    select * from pg_event_trigger_ddl_commands()
  loop
    -- Ordinary tables in `public` only. Views, matviews, partitions of an
    -- already-protected parent, and anything an extension creates in its own
    -- schema are left alone.
    if obj.schema_name = 'public'
       and obj.object_type = 'table'
       and obj.command_tag in ('CREATE TABLE', 'CREATE TABLE AS')
    then
      execute format(
        'alter table %s enable row level security', obj.object_identity
      );
      raise notice
        'auto_enable_rls: enabled RLS on % (add a policy, or it denies all)',
        obj.object_identity;
    end if;
  end loop;
end;
$$;

drop event trigger if exists auto_enable_rls_trg;
create event trigger auto_enable_rls_trg
  on ddl_command_end
  when tag in ('CREATE TABLE', 'CREATE TABLE AS')
  execute function public.auto_enable_rls();
