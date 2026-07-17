-- Gateway reverse-SSH tunnel ports (fleet ops access).
--
-- Adds gateways.tunnel_port to an EXISTING database. The gateways table is
-- delta-managed (apply_gateway_delta.sql), so this column is too.
-- Idempotent — safe to re-run.
--
-- Each gateway box holds a reverse-SSH tunnel open to the relay VPS
-- (memoria-relay, 144.126.211.110) on a UNIQUE port. The roster endpoint
-- (GET /v1/gateway/roster) assigns the port lazily from the sequence on the
-- box's first poll and returns it as "tunnel_port"; publisher.py writes it to
-- /opt/memoria-gateway/.tunnel-port and deploy/tunnel.sh (golden-image boxes)
-- (re)connects autossh on it. Ops then reach the box via
--   ssh -J root@relay -p <tunnel_port> potaton00b@localhost
--
-- PORT RANGES — all environments share ONE relay, so each database must
-- issue from a range no other database uses, or two environments can hand
-- the same port to different physical boxes (autossh bind fight on the
-- relay, ops ssh lands on the wrong box):
--   22001          reserved: box-01 (OptiPlex) legacy HARDCODED tunnel unit
--   22002 - 41999  PROD  (this delta's default)
--   42002 -        DEV / local stacks (change START below to 42002 before
--                  applying to a non-prod database)
--
-- ⚠️ PROD APPLY — RUN AS ONE TRANSACTION, PIN INCLUDED. Boxes poll the
-- roster every ~1s, so if the column commits before box-01's pin, its row is
-- lazily assigned 22002+ within a second while its legacy unit keeps dialing
-- 22001 — the DB then lies to ops about the port. Committing the pin
-- atomically with the column closes that window: no poll can see the column
-- without the pin already in place. Template (fill in box-01's row id from
-- `select id, last_seen_at from gateways`):
--
--   begin;
--     \i scripts/apply_gateway_tunnel_port_delta.sql
--     update public.gateways set tunnel_port = 22001
--       where id = '<box-01 row id>';
--   commit;
--
-- (Local/dev DBs with no legacy box skip the pin.)

create sequence if not exists public.gateway_tunnel_port_seq start 22002;

alter table public.gateways
  add column if not exists tunnel_port int unique;
