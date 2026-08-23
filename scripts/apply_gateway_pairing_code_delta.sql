-- A service binds to an account with a short-lived code, not an email.
--
-- Enrolment took an arbitrary bearer token the caller invented plus the
-- caregiver's EMAIL. An email is on the invoice and in the signature, so
-- anyone who knew one could bind a token and poll that account's roster:
-- camera ids, room names, LiveKit rooms, privacy and audio flags.
-- Metadata rather than video -- minting a publish token checks
-- `gateway_id` ownership separately -- but it is the customer's home.
--
-- Same shape as the wall phone's `cameras.pairing_code`, and for the same
-- reason: possession of something the caregiver just generated is the
-- trust anchor, because the installer and the caregiver are together (or
-- on the phone) at exactly that moment and never again.
--
-- Only the SHA-256 is stored. A leaked database does not leak usable
-- codes, and the window is five minutes regardless.

BEGIN;

CREATE TABLE IF NOT EXISTS public.gateway_pairing_codes (
  -- sha256 of the 6-digit code, hex. Primary key, so two live codes can
  -- never collide -- the minting loop retries on conflict.
  code_hash    text PRIMARY KEY,
  caregiver_id uuid NOT NULL REFERENCES public.caregivers(id) ON DELETE CASCADE,
  created_at   timestamptz NOT NULL DEFAULT now(),
  expires_at   timestamptz NOT NULL,
  -- Set when redeemed. Single-use: a code read aloud once must not go on
  -- binding services for the rest of the day.
  consumed_at  timestamptz,
  -- Which service took it, for the audit trail. Not a credential.
  consumed_by  uuid REFERENCES public.gateways(id) ON DELETE SET NULL
);

-- The redeem path looks up live codes only.
CREATE INDEX IF NOT EXISTS gateway_pairing_codes_live_idx
  ON public.gateway_pairing_codes (expires_at)
  WHERE consumed_at IS NULL;

-- Internal table: the backend reaches it through the service role, which
-- bypasses RLS. Enabling RLS with NO policy denies every other reader,
-- and the revoke keeps it out of PostgREST -- which matters more here
-- than usual, because the anon key ships inside the app binary and this
-- table is the thing that grants access to a home.
ALTER TABLE public.gateway_pairing_codes ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.gateway_pairing_codes FROM anon, authenticated;

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- DROP TABLE public.gateway_pairing_codes. Live codes are lost, which
-- costs an installer one re-generate. Rolling back also means restoring
-- email-based enrolment, so it is only correct alongside a revert of the
-- application change.
