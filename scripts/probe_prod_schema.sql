-- READ-ONLY pre-flight probe. Run against PROD before applying any delta.
-- Each row reports whether an object main's code needs already exists.
-- present = true  -> that delta is a no-op (safe to skip or run anyway)
-- present = false -> run the matching delta below
--
-- Paste into the Supabase SQL editor of the PROD project, or:
--   psql "<prod connection string>" -f scripts/probe_prod_schema.sql

select 'cameras.source_type column     (apply_pi_schema_delta.sql)' as object,
       exists(select 1 from information_schema.columns
              where table_schema='public' and table_name='cameras'
                and column_name='source_type') as present
union all
select 'cameras.ingest_token_hash col  (apply_pi_schema_delta.sql)',
       exists(select 1 from information_schema.columns
              where table_schema='public' and table_name='cameras'
                and column_name='ingest_token_hash')
union all
select 'pi_discoveries table           (apply_pi_schema_delta.sql)',
       exists(select 1 from information_schema.tables
              where table_schema='public' and table_name='pi_discoveries')
union all
select 'patient_face_embeddings table  (apply_face_embeddings_delta.sql)',
       exists(select 1 from information_schema.tables
              where table_schema='public' and table_name='patient_face_embeddings')
union all
select 'caregiver_devices table        (apply_caregiver_devices_delta.sql)',
       exists(select 1 from information_schema.tables
              where table_schema='public' and table_name='caregiver_devices')
order by object;
