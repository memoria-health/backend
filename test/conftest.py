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

Outbound email is disabled for the entire suite (see the env override
below the imports): the developer `.env` carries the production Resend
key, and without the override every feedback/support happy-path test
sends a real email to the founders.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import httpx
import jwt as pyjwt
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Env overrides MUST precede the src imports: get_settings() is
# @lru_cache(maxsize=1) and the src.main import tree touches settings
# at import time, freezing whatever the developer .env holds. (An
# older comment here claimed Settings was rebuilt per call — that
# stopped being true when the cache landed.)
#
# NEMARA_RESEND_API_KEY: blanking it disarms every email path (they
# all no-op on an empty key); email tests monkeypatch the sender.
os.environ["NEMARA_RESEND_API_KEY"] = ""
# NEMARA_INTERNAL_SERVICE_KEY (2026-08-17): the GPU-fleet work put a
# real key into the developer .env (nodes need it), which flipped
# every /internal-style route in the test app from "unset + env=test
# → allow" to enforcing a key the tests don't send. Blank it so the
# suite exercises the same local/test path it always did.
os.environ["NEMARA_INTERNAL_SERVICE_KEY"] = ""
# NEMARA_SUPABASE_URL (2026-08-23): the developer .env points this at
# `host.docker.internal`, because the backend runs in Docker and needs
# to reach Supabase on the Mac to fetch its JWKS. That name resolves
# ONLY inside a container — the suite runs natively, so every auth test
# died in DNS. Loopback is correct here for the same reason
# host.docker.internal is correct there: both mean "this Mac", from
# where the process happens to be standing.
os.environ["NEMARA_SUPABASE_URL"] = "http://127.0.0.1:54341"
# NEMARA_SENSITIVITY_SCREENING_ENABLED (2026-08-25): a developer who
# turns screening off locally — a normal thing to do while working on
# something else — silently broke three clip-archival tests, because
# they assert the censoring behaviour the flag governs. The suite must
# test the shipped default (True), not whatever a laptop happens to
# hold; a test that wants it off can override it explicitly.
os.environ["NEMARA_SENSITIVITY_SCREENING_ENABLED"] = "true"

from src.db.session import get_engine, get_sessionmaker
from src.main import create_app
from src.settings import get_settings


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Clear rate-limit counters between tests.

    The limiters are per-process and per-minute, so without this the
    pairing tests exhaust the budget and every later test in the run
    gets a 429 unrelated to what it asserts. Resetting (rather than
    disabling the limiter under NEMARA_ENV=test) keeps the production
    code path exercised by the integration suite.
    """
    from src.api.rate_limit import reset_all_rate_limits

    reset_all_rate_limits()
    yield
    reset_all_rate_limits()


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


# ─── camera seeding ──────────────────────────────────────────────


async def seed_camera(
    session: AsyncSession,
    *,
    patient_id: UUID | str | None = None,
    caregiver_id: UUID | str | None = None,
    room: str | None = None,
    online: bool = False,
    source_type: str = "iphone",
    livekit_room: str | None = None,
    livekit_identity: str | None = None,
    cover_patient: bool = True,
    camera_id: UUID | None = None,
    **columns,
) -> UUID:
    """Insert one camera row and return its id.

    Eleven test modules each grew their own near-identical version of
    this, and every schema change since had to be applied to all of them
    by hand — the account-room migration hit ten files before a single
    behaviour was under test, and each one failed differently depending
    on which columns its copy happened to set. One definition means the
    next schema change is one edit.

    The invariants it will not let a caller get wrong, because each has
    already produced a confusing failure somewhere:

      * `caregiver_id` is NOT NULL and is derived from the patient when
        it is not given. It is not an independent fact about a camera.
      * names are unique per account, so an unnamed camera gets a unique
        one rather than colliding with the last fixture's.
      * a 'gateway' row needs an `ingest_token_hash`
        (`cameras_ingest_token_required`).
      * coverage lives in `camera_patients`; `cameras.patient_id` is a
        compatibility shim. Both are written so a fixture is visible to
        old and migrated readers alike.

    Extra columns pass straight through as `**columns`, so a test that
    cares about `disabled`, `state`, `privacy_enabled` and so on still
    says so locally instead of growing another copy of this.
    """
    camera_id = camera_id or uuid4()

    if caregiver_id is None:
        if patient_id is None:
            raise AssertionError(
                "seed_camera needs a patient_id or a caregiver_id — a camera "
                "always belongs to an account"
            )
        caregiver_id = (await session.execute(
            text("SELECT caregiver_id FROM public.patients WHERE id = :p"),
            {"p": patient_id},
        )).scalar_one()

    values: dict = {
        "id": camera_id,
        "patient_id": patient_id,
        "caregiver_id": caregiver_id,
        "display_id": columns.pop("display_id", 1),
        "room": room or f"Room {camera_id.hex[:8]}",
        "icon_name": columns.pop("icon_name", "tv"),
        "online": online,
        "created_at": columns.pop("created_at", datetime.now(timezone.utc)),
        "source_type": source_type,
        "livekit_identity": livekit_identity or f"cam_{camera_id.hex[:8]}",
        "livekit_room": livekit_room if livekit_room is not None
        else f"account_{str(caregiver_id).lower()}",
        "ingest_token_hash": columns.pop(
            "ingest_token_hash",
            camera_id.hex * 2 if source_type == "gateway" else None),
        **columns,
    }
    cols = ", ".join(values)
    binds = ", ".join(f":{k}" for k in values)
    await session.execute(
        text(f"INSERT INTO public.cameras ({cols}) VALUES ({binds})"), values,
    )
    if cover_patient and patient_id is not None:
        await session.execute(
            text("INSERT INTO public.camera_patients (camera_id, patient_id) "
                 "VALUES (:c, :p) ON CONFLICT DO NOTHING"),
            {"c": camera_id, "p": patient_id},
        )
    await session.commit()
    return camera_id


async def mint_gateway_code(client, headers) -> str:
    """A fresh pairing code for this caregiver.

    Binding a service takes a code the caregiver just generated, so every
    test that provisions needs one. Centralised for the same reason
    `seed_camera` is: the email-based handshake lived inline in a dozen
    suites, and changing it meant editing all of them.
    """
    r = await client.post("/v1/gateway/pairing-code", headers=headers, json={})
    assert r.status_code == 200, r.text
    return r.json()["pairing_code"]


async def provision_sensor(
    client, headers, *, token: str, room: str = "Living Room",
    service_kind: str | None = None, expect: int = 201,
) -> dict:
    """Bind `token` to this account (first time) and create a sensor.

    Mints a code per call. A token already bound ignores it, which is the
    real behaviour — an installer mounting a second sensor does not go
    back for a second code.
    """
    body: dict = {"room": room, "pairing_code": await mint_gateway_code(client, headers)}
    if service_kind is not None:
        body["service_kind"] = service_kind
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"}, json=body)
    assert r.status_code == expect, r.text
    return r.json() if r.status_code == 201 else {}


# Presets a patient used to be born with. `1cd354e` stopped auto-seeding
# them — the caregiver picks from the library instead, and the seeded rows
# were a second, competing source of truth that showed up in the picker as
# "already added". Tests that need a preset must now ask for it, exactly
# as the app does.
#
# Named by CATALOG key; the tests read them back by display name.
PRESET_KEYS = ("fall", "agitation", "wandering", "medication")


async def add_library_detectors(
    client, headers: dict, patient_id: str, keys=PRESET_KEYS,
) -> dict:
    """Instantiate catalog detectors for a patient, returned by NAME.

    Mirrors what the app does on add: POST the catalog keys, then read the
    list back. Keys the patient already has are skipped server-side, so
    this is safe to call twice.
    """
    r = await client.post(
        f"/v1/patients/{patient_id}/detectors/from-library",
        headers=headers, json={"keys": list(keys)},
    )
    assert r.status_code == 201, r.text
    listed = await client.get(
        f"/v1/patients/{patient_id}/detectors", headers=headers)
    assert listed.status_code == 200, listed.text
    return {d["name"]: d for d in listed.json()}


async def seed_catalog_detectors(session, patient_id, keys):
    """Create catalog presets directly, for tests that work at DB level.

    The API route (`/detectors/from-library`) is the caregiver's path and
    `add_library_detectors` above covers it. Some suites operate on the
    session with no client — they only need the ROWS to exist so they can
    assert scheduling or linking behaviour, and used to get them free
    from onboarding's auto-seed (removed in 1cd354e).

    Built from the same CATALOG entry the endpoint uses, so a change to a
    preset's shape cannot make these tests disagree with production.
    """
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from src.db.detector import Detector
    from src.services.detection import catalog

    base = datetime.now(timezone.utc)
    made = []
    for i, key in enumerate(keys):
        e = next(c for c in catalog.CATALOG if c.key == key)
        now = base + timedelta(microseconds=i)
        det = Detector(
            id=uuid4(), patient_id=patient_id, name=e.name,
            icon_name=e.icon_name, description=e.description,
            category=e.category, source="preset",
            alert_trigger=e.alert_trigger, is_sustained=e.is_sustained,
            duration_seconds=e.duration_seconds,
            where_rooms=(None if e.where_scope == "ALL ROOMS" else [e.where_scope]),
            when_scope=e.when_scope, sensitivity="standard", is_on=True,
            created_at=now, updated_at=now, custom_prompt=None,
            identifier=e.key, prompt_fragment=e.prompt_fragment,
            schema_json=e.schema_json, always_on=False,
        )
        session.add(det)
        made.append(det)
    await session.flush()
    return made
