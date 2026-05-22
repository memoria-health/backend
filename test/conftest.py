"""Shared pytest fixtures.

The integration suite assumes a local Supabase is running (`supabase
start`) and the schema has been applied (`make db-reset`). The
fixtures below mint a short-lived HS256 JWT signed with the local
`NEMARA_SUPABASE_JWT_SECRET` — same auth path the production app uses,
just with the symmetric local-dev key.

`auth_user` / `auth_user_factory` create real rows in `auth.users` via
the Supabase admin REST API. They're necessary because `caregivers.id`
has a FK to `auth.users(id)` — we can't provision a caregiver from a
fabricated UUID without the auth user existing first.

`cleanup_auth_users` deletes everything those factories created at the
end of each test so isolation holds.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_engine, get_sessionmaker
from src.main import create_app
from src.settings import get_settings


@pytest_asyncio.fixture(autouse=True)
async def _per_test_engine() -> AsyncIterator[None]:
    """asyncpg connections are bound to the event loop they were
    created on. pytest-asyncio creates a fresh loop per test, so the
    `lru_cache`-held engine from the previous test fails on the next
    one with `Event loop is closed` during pool teardown.

    Fix: dispose + cache-clear at every test boundary so each test
    gets an engine bound to its own loop. autouse=True so every
    integration test gets this for free.
    """
    yield
    engine = get_engine()
    await engine.dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """One async session per test. Caller controls commit / rollback —
    integration tests typically commit because the FastAPI endpoints
    they exercise commit through the route handler.
    """
    sm = get_sessionmaker()
    async with sm() as s:
        yield s


def mint_test_jwt(caregiver_id: UUID, email: str) -> str:
    """Mint an HS256 JWT identical in shape to a Supabase-issued token.

    Caller passes `caregiver_id` and `email` explicitly — these MUST
    match a real `auth.users` row (use the `auth_user` /
    `auth_user_factory` fixtures to create one).
    """
    settings = get_settings()
    now = int(time.time())
    claims = {
        "sub": str(caregiver_id),
        "email": email,
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
        "role": "authenticated",
    }
    return pyjwt.encode(claims, settings.supabase_jwt_secret, algorithm="HS256")


# ─── Supabase auth user lifecycle ────────────────────────────────


async def _create_supabase_auth_user(email: str) -> UUID:
    """Create a user in `auth.users` via the admin REST API.

    Uses `email_confirm=True` so we skip Supabase's confirmation flow
    in tests. Returns the new user's id.
    """
    s = get_settings()
    headers = {
        "apikey": s.supabase_service_role_key,
        "Authorization": f"Bearer {s.supabase_service_role_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=10.0) as c:
        r = await c.post(
            f"{s.supabase_url}/auth/v1/admin/users",
            headers=headers,
            json={
                "email": email,
                "password": "test-password-1234!",
                "email_confirm": True,
            },
        )
        r.raise_for_status()
        return UUID(r.json()["id"])


async def _delete_supabase_auth_user(user_id: UUID) -> None:
    """Best-effort delete of an `auth.users` row. Tolerates 404s so a
    failed-mid-test cleanup doesn't mask the real assertion failure.
    """
    s = get_settings()
    headers = {
        "apikey": s.supabase_service_role_key,
        "Authorization": f"Bearer {s.supabase_service_role_key}",
    }
    async with httpx.AsyncClient(timeout=10.0) as c:
        await c.delete(
            f"{s.supabase_url}/auth/v1/admin/users/{user_id}",
            headers=headers,
        )


@pytest_asyncio.fixture
async def auth_user_factory() -> AsyncIterator:
    """Factory fixture: call inside a test to mint multiple distinct
    auth.users rows. All are deleted at teardown via the
    auth.users → caregivers cascade (set on schema.sql).
    """
    created: list[UUID] = []

    async def _make(email: str | None = None) -> tuple[UUID, str]:
        addr = email or f"test-{uuid4().hex[:12]}@example.com"
        uid = await _create_supabase_auth_user(addr)
        created.append(uid)
        return uid, addr

    yield _make

    for uid in created:
        await _delete_supabase_auth_user(uid)


@pytest_asyncio.fixture
async def auth_user(auth_user_factory) -> tuple[UUID, str]:
    """Shorthand for `(uid, email) = await auth_user_factory()`."""
    return await auth_user_factory()


@pytest.fixture
def auth_headers(auth_user) -> dict[str, str]:
    """A fresh JWT for a real auth.users row. Use this in any test that
    actually hits `/v1/me` or further-protected routes.
    """
    uid, email = auth_user
    return {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}


# ─── HTTP client ─────────────────────────────────────────────────


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    """ASGI-transport HTTP client — no network, no port allocation."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
