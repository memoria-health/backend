-- READ-ONLY pre-flight probe. Run against PROD before any delta.
-- present = true  -> already applied (safe to skip)
-- present = false -> run the matching delta, BUT SEE BELOW FIRST
--
-- SUPERSEDED DELTAS READ AS ABSENT. Some columns were added by one
-- delta and dropped by a later one (episode_gap, convulsion,
-- where_scope, is_builtin, pi_source ...). Those show present=false
-- because the column really is gone — re-applying them would undo
-- the drop. Before running anything, check whether a
-- scripts/deltas/apply_drop_*.sql exists for the same object.
--
-- GENERATED. Do not hand-edit:
--   python3 scripts/gen_prod_probe.py > scripts/probe_prod_schema.sql
--
-- Paste into the PROD Supabase SQL editor, or:
--   psql "<prod connection string>" -f scripts/probe_prod_schema.sql

select 'column cameras.audio_capture_enabled  (apply_account_room_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='audio_capture_enabled') as present
union all
select 'column detectors.alert_trigger  (apply_alert_trigger_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='alert_trigger') as present
union all
select 'column detectors.is_sustained  (apply_alert_trigger_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='is_sustained') as present
union all
select 'column detectors.duration_seconds  (apply_alert_trigger_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='duration_seconds') as present
union all
select 'index analyses_camera_captured_at_idx  (apply_analyses_coverage_delta.sql)' as object, exists(select 1 from pg_indexes where schemaname='public' and indexname='analyses_camera_captured_at_idx') as present
union all
select 'column patients.audio_capture_enabled  (apply_audio_capture_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='audio_capture_enabled') as present
union all
select 'table AS  (apply_auto_enable_rls_delta.sql)'              as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='AS') as present
union all
select 'column gateways.multi_tenant  (apply_bridge_multitenant_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='gateways' and column_name='multi_tenant') as present
union all
select 'column cameras.caregiver_id  (apply_camera_ownership_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='caregiver_id') as present
union all
select 'table camera_patients  (apply_camera_ownership_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='camera_patients') as present
union all
select 'column cameras.room_class  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='room_class') as present
union all
select 'column cameras.room_class_confidence  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='room_class_confidence') as present
union all
select 'column cameras.room_classified_at  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='room_classified_at') as present
union all
select 'column cameras.privacy_enabled  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='privacy_enabled') as present
union all
select 'column events.clip_censored  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='clip_censored') as present
union all
select 'column events.sensitivity_categories  (apply_camera_privacy_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='sensitivity_categories') as present
union all
select 'table caregiver_devices  (apply_caregiver_devices_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='caregiver_devices') as present
union all
select 'column caregivers.country  (apply_country_delta.sql)'     as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='country') as present
union all
select 'column patients.country  (apply_country_delta.sql)'       as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='country') as present
union all
select 'column patients.data_collection_enabled  (apply_data_collection_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='data_collection_enabled') as present
union all
select 'column patients.data_collection_decided_at  (apply_data_collection_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='data_collection_decided_at') as present
union all
select 'column patients.data_collection_policy_version  (apply_data_collection_policy_version_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='data_collection_policy_version') as present
union all
select 'column detectors.user_criteria  (apply_detector_criteria_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='user_criteria') as present
union all
select 'column detectors.refinements  (apply_detector_criteria_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='refinements') as present
union all
select 'column detectors.evidence_branch  (apply_detector_v2_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='evidence_branch') as present
union all
select 'column detectors.episode_gap_seconds  (apply_episode_gap_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='episode_gap_seconds') as present
union all
select 'column events.ended_at  (apply_event_ended_at_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='ended_at') as present
union all
select 'column patients.event_gate_enabled  (apply_event_gate_default_on_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='event_gate_enabled') as present
union all
select 'column patients.event_gate_enabled  (apply_event_gate_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='event_gate_enabled') as present
union all
select 'column events.gate_state  (apply_event_gate_delta.sql)'   as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='gate_state') as present
union all
select 'column events.gate_held_at  (apply_event_gate_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='gate_held_at') as present
union all
select 'column events.gate_clip_key  (apply_event_gate_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='gate_clip_key') as present
union all
select 'table patient_face_embeddings  (apply_face_embeddings_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='patient_face_embeddings') as present
union all
select 'table feedback  (apply_feedback_delta.sql)'               as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='feedback') as present
union all
select 'column cameras.last_heartbeat_at  (apply_gateway_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='last_heartbeat_at') as present
union all
select 'column gateways.caregiver_id  (apply_gateway_delta.sql)'  as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='gateways' and column_name='caregiver_id') as present
union all
select 'table gateways  (apply_gateway_delta.sql)'                as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='gateways') as present
union all
select 'column cameras.gateway_id  (apply_gateway_ownership_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='gateway_id') as present
union all
select 'table gateway_pairing_codes  (apply_gateway_pairing_code_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='gateway_pairing_codes') as present
union all
select 'index gateways_token_hash_uidx  (apply_gateway_token_unique_delta.sql)' as object, exists(select 1 from pg_indexes where schemaname='public' and indexname='gateways_token_hash_uidx') as present
union all
select 'column gateways.tunnel_port  (apply_gateway_tunnel_port_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='gateways' and column_name='tunnel_port') as present
union all
select 'table identity_occupancy_samples  (apply_identity_fleet_ops_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='identity_occupancy_samples') as present
union all
select 'column patients.identity_node_id  (apply_identity_nodes_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='identity_node_id') as present
union all
select 'column patients.identity_port  (apply_identity_nodes_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='identity_port') as present
union all
select 'table identity_nodes  (apply_identity_nodes_delta.sql)'   as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='identity_nodes') as present
union all
select 'column cameras.identity_node_id  (apply_identity_shard_per_sensor_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='identity_node_id') as present
union all
select 'column caregivers.show_live_analyses  (apply_live_analyses_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='show_live_analyses') as present
union all
select 'table patient_members  (apply_patient_sharing_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='patient_members') as present
union all
select 'column patients.timezone  (apply_patient_timezone_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='timezone') as present
union all
select 'column cameras.source_type  (apply_pi_schema_delta.sql)'  as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='source_type') as present
union all
select 'column pi_discoveries.pending_ingest_token  (apply_pi_schema_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='pi_discoveries' and column_name='pending_ingest_token') as present
union all
select 'table pi_discoveries  (apply_pi_schema_delta.sql)'        as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='pi_discoveries') as present
union all
select 'column patients.privacy_mode_enabled  (apply_privacy_mode_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='privacy_mode_enabled') as present
union all
select 'column events.clip_suppressed  (apply_privacy_mode_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='events' and column_name='clip_suppressed') as present
union all
select 'column caregivers.region  (apply_region_delta.sql)'       as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='region') as present
union all
select 'column patients.region  (apply_region_delta.sql)'         as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='patients' and column_name='region') as present
union all
select 'column detectors.requires_motion  (apply_requires_motion_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='requires_motion') as present
union all
select 'column cameras.setup_test_passed_at  (apply_setup_test_per_camera_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras' and column_name='setup_test_passed_at') as present
union all
select 'column activity_days.state_minutes  (apply_sleep_activity_tables_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='activity_days' and column_name='state_minutes') as present
union all
select 'table sleep_nights  (apply_sleep_activity_tables_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='sleep_nights') as present
union all
select 'table activity_days  (apply_sleep_activity_tables_delta.sql)' as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='activity_days') as present
union all
select 'column caregivers.tos_accepted_at  (apply_tos_acceptance_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='tos_accepted_at') as present
union all
select 'column caregivers.tos_version  (apply_tos_acceptance_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='tos_version') as present
union all
select 'table ui_events  (apply_ui_events_delta.sql)'             as object, exists(select 1 from information_schema.tables where table_schema='public' and table_name='ui_events') as present
union all
select 'column caregivers.vlm_provider  (apply_vlm_provider_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='caregivers' and column_name='vlm_provider') as present
union all
select 'column detectors.door_config  (apply_wandering_door_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='door_config') as present
union all
select 'column cameras_detectors.door_config  (apply_wandering_door_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='cameras_detectors' and column_name='door_config') as present
union all
select 'column detectors.where_rooms  (apply_where_rooms_delta.sql)' as object, exists(select 1 from information_schema.columns where table_schema='public' and table_name='detectors' and column_name='where_rooms') as present
order by 1;

-- NOT PROBED — these change data, policies or types rather than
-- adding an object, so presence cannot be inferred. Check by hand:
--   apply_absence_rename_delta.sql
--   apply_camera_patients_rls_delta.sql
--   apply_camera_survives_patient_delete_delta.sql
--   apply_convulsion_removal_delta.sql
--   apply_convulsion_rename_delta.sql
--   apply_detector_category_behaviour_care_delta.sql
--   apply_detector_category_reclassify_delta.sql
--   apply_drop_episode_gap_delta.sql
--   apply_drop_is_builtin_delta.sql
--   apply_drop_pi_discovery_delta.sql
--   apply_drop_pi_source_delta.sql
--   apply_drop_shape_delta.sql
--   apply_drop_where_scope_delta.sql
--   apply_duty_thinned_delta.sql
--   apply_insights_readonly_revoke_delta.sql
--   apply_internal_tables_rls_delta.sql
--   apply_medication_whenscope_delta.sql
--   apply_provider_dashscope_delta.sql
--   apply_rls_sleep_activity_review_delta.sql
--   apply_snapshot_bucket_delta.sql
