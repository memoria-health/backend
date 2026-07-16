-- Gateway reverse-SSH tunnel ports (fleet ops access).
--
-- Adds gateways.tunnel_port to an EXISTING database (local dev or prod).
-- schema.sql already includes it for fresh creates; this delta is for a DB
-- that predates the column. Idempotent — safe to re-run.
--
-- Each gateway box holds a reverse-SSH tunnel open to the relay VPS
-- (memoria-relay, 144.126.211.110) on a UNIQUE port. The roster endpoint
-- (GET /v1/gateway/roster) assigns the port lazily from the sequence on the
-- box's first poll and returns it as "tunnel_port"; publisher.py writes it to
-- /opt/memoria-gateway/.tunnel-port and deploy/tunnel.sh (golden-image boxes)
-- (re)connects autossh on it. Ops then reach the box via
--   ssh -J root@relay -p <tunnel_port> potaton00b@localhost
--
-- The sequence starts at 22002: port 22001 is deliberately reserved for the
-- original OptiPlex (box-01), which runs the legacy HARDCODED-22001 tunnel
-- unit. After applying, pin its row so the numbers agree:
--   update public.gateways set tunnel_port = 22001 where id = '<box-01 row id>';
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_gateway_tunnel_port_delta.sql
-- or paste into the Supabase SQL editor / run via the Management API.

create sequence if not exists public.gateway_tunnel_port_seq start 22002;

alter table public.gateways
  add column if not exists tunnel_port int unique;
