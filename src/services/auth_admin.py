"""Supabase Auth Admin (GoTrue) operations via the service-role key.

Currently one operation: hard-delete an auth user. Deleting the
`auth.users` row cascades to `public.caregivers` (FK `ON DELETE
CASCADE`, see supabase/schema.sql) and from there to the entire
account graph — so this is the load-bearing step of account deletion.

Mirrors the header/auth shape `scripts/seed/seed.py` uses for the admin
`users` endpoint (apikey + service-role bearer). Pure async; the caller
injects the frozen Settings instance.
"""

from __future__ import annotations

import logging
from uuid import UUID

import httpx

from src.errors import AuthAdminError
from src.settings import Settings

log = logging.getLogger(__name__)


async def delete_auth_user(user_id: UUID, *, settings: Settings) -> None:
    """Hard-delete the Supabase auth user with id `user_id`.

    GoTrue's `DELETE /auth/v1/admin/users/{id}` hard-deletes by default
    (no `should_soft_delete`), which removes the `auth.users` row and
    fires the `caregivers` cascade.

    A 404 is treated as success — the user is already gone, which
    satisfies the caller's intent ("this account must not exist") and
    makes the call idempotent on retry.

    Raises:
        AuthAdminError: transport failure or any non-2xx/non-404 status.
    """
    base = settings.supabase_url.rstrip("/")
    url = f"{base}/auth/v1/admin/users/{user_id}"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.delete(url, headers=headers)
    except httpx.HTTPError as exc:
        raise AuthAdminError(f"auth user delete transport error: {exc}") from exc

    if resp.status_code == 404:
        log.info("auth_admin.delete.absent user=%s", user_id)
        return
    if resp.status_code < 200 or resp.status_code >= 300:
        raise AuthAdminError(
            f"auth user delete failed: {resp.status_code} {resp.text}"
        )
    log.info("auth_admin.delete.success user=%s", user_id)
