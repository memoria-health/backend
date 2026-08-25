-- A hosted service serves MANY accounts; a box in a home serves one.
--
-- `public.gateways` binds a token to a single caregiver_id, which is
-- right for a mini-PC — one box, one home, one account — and wrong for
-- the Warner bridge, which fronts the vendor cloud for every customer at
-- once. On the box model, provisioning a camera for a second customer
-- REBOUND the whole bridge to them and the first customer's cameras
-- vanished from the roster and stopped publishing, silently.
--
-- ADDITIVE and default false, so every existing box keeps its current
-- behaviour byte for byte — including the `gateway_id IS NULL` fallback
-- that 4 of the 5 gateway cameras in production still depend on.
--
-- A multi-tenant service is scoped STRICTLY by the cameras carrying its
-- own gateway_id, across every account, with no caregiver filter and no
-- unowned fallback. That is NARROWER than a box's scope, not wider,
-- which is why a service may declare it at enrolment: claiming it can
-- only ever show you fewer cameras, never someone else's.

alter table public.gateways
  add column if not exists multi_tenant boolean not null default false;

comment on column public.gateways.multi_tenant is
  'true for a hosted service (camera bridge) that serves many caregiver '
  'accounts. Scopes its roster to cameras.gateway_id alone and stops '
  'provisioning from rewriting caregiver_id.';
