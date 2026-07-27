"""The consolidated identity pipeline — GPU-host deployable.

Everything identity lives HERE (Aryaman's no-split-brain rule,
2026-07-27): the per-camera loops, evidence/seat/harvest stack, body
gallery, matching math + embedding codec, display veto, perception
wire client, journal, and the service entrypoint. The backend keeps
only the consumer shim (src/services/identity/identity_client.py) and
imports exactly two identity modules: `identity.contract` (the /state
seam) and `identity.matching` (the embedding codec enrollment shares
through the DB).

Runtime deps on src/* remain in service.py only (DB session + repos +
settings for gallery loads) — the known packaging exception until the
GPU host gets its own config surface.
"""
