"""Shared pytest fixtures.

The integration suite assumes a local Supabase is running (`supabase
start`) and the schema has been applied (`make db-reset`). The
fixtures below mint a short-lived HS256 JWT signed with the local
`NEMARA_SUPABASE_JWT_SECRET` — same auth path the production app uses,
just with the symmetric local-dev key.
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

from src.db.session import get_sessionmaker
from src.main import create_app
from src.settings import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """One async session per test. Wrapped in a transaction that's
    rolled back at the end so tests don't pollute each other.
    """
    sm = get_sessionmaker()
    async with sm() as s:
        async with s.begin():
            yield s
            await s.rollback()


def mint_test_jwt(caregiver_id: UUID | None = None, email: str | None = None) -> str:
    """Mint an HS256 JWT identical in shape to a Supabase-issued token.

    Local Supabase signs with the secret in `NEMARA_SUPABASE_JWT_SECRET`;
    our `verify_supabase_jwt` accepts HS256 + that secret as the
    legacy/dev path. Same JWT shape, same verification path the real
    app uses — just symmetric instead of ES256.
    """
    settings = get_settings()
    now = int(time.time())
    cid = str(caregiver_id or uuid4())
    claims = {
        "sub": cid,
        "email": email or f"test-{cid[:8]}@example.com",
        "aud": "authenticated",
        "iat": now,
        "exp": now + 3600,
        "role": "authenticated",
    }
    return pyjwt.encode(claims, settings.supabase_jwt_secret, algorithm="HS256")


@pytest_asyncio.fixture
async def client(app) -> AsyncIterator[httpx.AsyncClient]:
    """ASGI-transport HTTP client — no network, no port allocation."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """A fresh caregiver JWT for the default test caregiver. Tests that
    need a specific caregiver_id should mint their own via
    `mint_test_jwt(caregiver_id=...)`.
    """
    return {"Authorization": f"Bearer {mint_test_jwt()}"}
