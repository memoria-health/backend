"""Tests for the operator event gate.

The gate's whole reason to exist is that it must NEVER be able to lose
an alert. So the tests that matter most here are the negative ones:
a tampered link cannot vote, a restart cannot strand a held event, and
model-generated text cannot execute in the verdict page.

Pure-logic tests (tokens, pager payloads, HTML) need no database. The
release-path tests use the same HTTP-driven fixtures as the rest of
test/unit so patient/event rows come from the real schema.
"""

from __future__ import annotations

import time
from uuid import UUID, uuid4

import pytest

from src.api import event_gate as api_gate
from src.services import event_gate, pager
from src.settings import Settings


def _settings(**over) -> Settings:
    base = {
        "event_gate_link_secret": "test-secret-not-a-real-key",
        "event_gate_public_base_url": "https://api.example.test",
        "event_gate_link_ttl_seconds": 180,
        "pagerduty_routing_key": "R0UT1NGK3Y",
    }
    base.update(over)
    return Settings(**base)


# ── verdict tokens ───────────────────────────────────────────────────


def test_token_round_trips_to_its_event() -> None:
    s = _settings()
    event_id = uuid4()
    token = event_gate.mint_verdict_token(event_id=event_id, settings=s)
    assert token is not None
    assert event_gate.verify_verdict_token(token, settings=s) == event_id


def test_token_for_one_event_cannot_be_edited_into_another() -> None:
    """The event id is inside the signed payload. Swapping it — the
    obvious attack on a URL you can see in your own notification — must
    fail rather than let one page suppress a different household's
    alert."""
    s = _settings()
    token = event_gate.mint_verdict_token(event_id=uuid4(), settings=s)
    _, exp, sig = token.split(".")
    forged = f"{uuid4()}.{exp}.{sig}"
    assert event_gate.verify_verdict_token(forged, settings=s) is None


def test_token_signed_with_another_secret_is_rejected() -> None:
    token = event_gate.mint_verdict_token(event_id=uuid4(), settings=_settings())
    other = _settings(event_gate_link_secret="a-different-secret-entirely")
    assert event_gate.verify_verdict_token(token, settings=other) is None


def test_expired_token_is_rejected() -> None:
    """Links outlive the 60s window on purpose, but not indefinitely —
    a page sitting in notification history must not stay live."""
    s = _settings()
    event_id = uuid4()
    body = f"{event_id}.{int(time.time()) - 1}"
    expired = f"{body}.{event_gate._sign(body, s.event_gate_link_secret)}"
    assert event_gate.verify_verdict_token(expired, settings=s) is None


@pytest.mark.parametrize(
    "bad", ["", "garbage", "a.b", "a.b.c.d", "....", "not-a-uuid.9999999999.beef"],
)
def test_malformed_tokens_are_rejected_without_raising(bad: str) -> None:
    assert event_gate.verify_verdict_token(bad, settings=_settings()) is None


def test_no_secret_means_no_link_rather_than_an_unsigned_one() -> None:
    """Fail closed. An unsigned link would let anyone who guessed the
    URL suppress a real alert."""
    s = _settings(event_gate_link_secret="")
    assert event_gate.mint_verdict_token(event_id=uuid4(), settings=s) is None
    assert event_gate.verify_verdict_token("anything", settings=s) is None


def test_verdict_url_is_absent_when_base_url_unset() -> None:
    s = _settings(event_gate_public_base_url="")
    assert event_gate._verdict_url(event_id=uuid4(), settings=s) is None


# ── paging ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trigger_sends_one_incident_keyed_to_the_event(monkeypatch) -> None:
    """dedup_key must be the event id: two households firing together
    have to produce two incidents, because settling one must never
    silence the other."""
    sent: list[dict] = []

    class _Resp:
        status_code = 202
        text = ""

    class _Client:
        async def post(self, url, json=None):  # noqa: A002
            sent.append(json)
            return _Resp()

    monkeypatch.setattr(pager, "_get_client", lambda: _fake(_Client()))
    event_id = uuid4()
    ok = await pager.trigger(
        settings=_settings(), event_id=event_id,
        summary="Maggie: Possible fall", link_url="https://x.test/act/tok",
    )
    assert ok is True
    assert sent[0]["dedup_key"] == str(event_id)
    assert sent[0]["event_action"] == "trigger"
    assert sent[0]["payload"]["severity"] == "critical"
    assert sent[0]["links"][0]["href"] == "https://x.test/act/tok"


@pytest.mark.asyncio
async def test_pager_reports_failure_instead_of_raising(monkeypatch) -> None:
    """A paging outage must degrade to the auto-release, never to an
    exception inside the detector tick."""
    class _Client:
        async def post(self, url, json=None):  # noqa: A002
            raise RuntimeError("pagerduty is down")

    monkeypatch.setattr(pager, "_get_client", lambda: _fake(_Client()))
    assert await pager.trigger(
        settings=_settings(), event_id=uuid4(), summary="x") is False


@pytest.mark.asyncio
async def test_pager_noops_when_unconfigured() -> None:
    s = _settings(pagerduty_routing_key="")
    assert pager.is_configured(s) is False
    assert await pager.trigger(settings=s, event_id=uuid4(), summary="x") is False
    assert await pager.resolve(settings=s, event_id=uuid4()) is False


def _fake(obj):
    async def _get():
        return obj
    return _get()


# ── the verdict page ─────────────────────────────────────────────────


def _row(**over) -> dict:
    base = {
        "title": "Possible fall", "narrative": "Maggie did not get up.",
        "patient_name": "Maggie", "room": "Kitchen", "gate_state": "pending",
        "seconds_left": 42, "has_clip": True,
    }
    base.update(over)
    return base


def test_narrative_cannot_inject_html() -> None:
    """Narratives are model-generated and land straight in the DOM. A
    page that executes them would execute them on the one device that
    can suppress alerts for every household."""
    hostile = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    page = api_gate._decision_page(_row(narrative=hostile), token="tok")
    # The payload survives as TEXT — that is fine and correct. What must
    # not survive is a live tag, so assert on the brackets, not on the
    # attribute string (which is inert once the brackets are entities).
    assert "<script>alert(1)" not in page
    assert "<img" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "&lt;img src=x onerror=alert(2)&gt;" in page


def test_patient_name_and_title_are_escaped() -> None:
    page = api_gate._decision_page(
        _row(patient_name="<b>Mags</b>", title="Fall & injury"), token="tok")
    assert "<b>Mags</b>" not in page
    assert "&amp;" in page


def test_decision_page_offers_both_verdicts_as_posts() -> None:
    page = api_gate._decision_page(_row(), token="tok")
    assert "method='post'" in page
    assert "action=suppress" in page
    assert "action=release" in page
    # The first tap must not carry confirm — that is the double-guard.
    assert "confirm=1" not in page


def test_confirm_page_is_the_only_thing_carrying_confirm() -> None:
    page = api_gate._confirm_page(_row(), token="tok", action="suppress")
    assert "action=suppress&confirm=1" in page
    assert "cannot be undone" in page


def test_settled_page_explains_the_outcome_and_offers_no_buttons() -> None:
    for state in ("approved", "rejected", "auto"):
        page = api_gate._settled_page(_row(gate_state=state))
        assert "method='post'" not in page
        assert "Already decided." in page


def test_missing_clip_still_renders_a_decidable_page() -> None:
    """A censor drop or a slow upload must not leave the operator with
    a blank page and a running clock."""
    page = api_gate._decision_page(_row(has_clip=False), token="tok")
    assert "<video" not in page
    assert "Maggie did not get up." in page


# ── the safety property, against a real database ─────────────────────


@pytest.mark.asyncio
async def test_sweeper_releases_an_event_stranded_by_a_restart(
    client, auth_user, session, settings,
) -> None:
    """The scenario that motivates GateSweeper: an event is held, the
    process that owned its 60s timer dies, and nothing in memory knows
    the event exists. It must still reach the caregiver.
    """
    from sqlalchemy import text

    from src.db.session import get_sessionmaker
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    patient = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Margaret O'Brien", "preferred_name": "Maggie",
        "relation": "parent",
    })).json()

    event_id = uuid4()
    await session.execute(text(
        """insert into events (id, patient_id, category, source, icon_name,
                               title, location, narrative, occurred_at,
                               opened, reviewed, gate_state, gate_held_at)
           values (:id, :pid, 'safety', 'preset', 'figure.fall',
                   'Possible fall', 'Kitchen', 'n', now(), false, false,
                   'pending', now() - interval '5 minutes')"""),
        {"id": str(event_id), "pid": patient["id"]})
    await session.commit()

    dispatched: list[list[UUID]] = []

    async def _capture(*, sessionmaker, event_ids, settings):  # noqa: ARG001
        dispatched.append(list(event_ids))

    original = event_gate._dispatch
    event_gate._dispatch = _capture
    try:
        sweeper = event_gate.GateSweeper(
            settings=settings, sessionmaker=get_sessionmaker())
        await sweeper._sweep()
    finally:
        event_gate._dispatch = original

    state = (await session.execute(text(
        "select gate_state from events where id = :id"),
        {"id": str(event_id)})).scalar_one()
    assert state == "auto"
    assert [event_id] in [list(d) for d in dispatched]


@pytest.mark.asyncio
async def test_sweeper_leaves_a_fresh_hold_alone(
    client, auth_user, session, settings,
) -> None:
    """The sweeper must not pre-empt the operator: an event still inside
    its window is the human's to decide."""
    from sqlalchemy import text

    from src.db.session import get_sessionmaker
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    patient = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Margaret O'Brien", "preferred_name": "Maggie",
        "relation": "parent",
    })).json()

    event_id = uuid4()
    await session.execute(text(
        """insert into events (id, patient_id, category, source, icon_name,
                               title, location, narrative, occurred_at,
                               opened, reviewed, gate_state, gate_held_at)
           values (:id, :pid, 'safety', 'preset', 'figure.fall',
                   'Possible fall', 'Kitchen', 'n', now(), false, false,
                   'pending', now())"""),
        {"id": str(event_id), "pid": patient["id"]})
    await session.commit()

    sweeper = event_gate.GateSweeper(
        settings=settings, sessionmaker=get_sessionmaker())
    await sweeper._sweep()

    state = (await session.execute(text(
        "select gate_state from events where id = :id"),
        {"id": str(event_id)})).scalar_one()
    assert state == "pending"


@pytest.mark.asyncio
async def test_a_settled_event_cannot_be_settled_twice(
    client, auth_user, session, settings,
) -> None:
    """Operator taps 'suppress' at the same moment the timer fires. One
    of them wins; the loser must be told it already happened rather than
    dispatching a suppressed alert."""
    from sqlalchemy import text

    from src.db.session import get_sessionmaker
    from test.conftest import mint_test_jwt

    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    patient = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Margaret O'Brien", "preferred_name": "Maggie",
        "relation": "parent",
    })).json()

    event_id = uuid4()
    await session.execute(text(
        """insert into events (id, patient_id, category, source, icon_name,
                               title, location, narrative, occurred_at,
                               opened, reviewed, gate_state, gate_held_at)
           values (:id, :pid, 'safety', 'preset', 'figure.fall',
                   'Possible fall', 'Kitchen', 'n', now(), false, false,
                   'pending', now())"""),
        {"id": str(event_id), "pid": patient["id"]})
    await session.commit()

    calls: list[list[UUID]] = []

    async def _capture(*, sessionmaker, event_ids, settings):  # noqa: ARG001
        calls.append(list(event_ids))

    original = event_gate._dispatch
    event_gate._dispatch = _capture
    try:
        first = await event_gate.act(
            sessionmaker=get_sessionmaker(), event_id=event_id,
            approve=False, settings=settings)
        second = await event_gate.act(
            sessionmaker=get_sessionmaker(), event_id=event_id,
            approve=True, settings=settings)
    finally:
        event_gate._dispatch = original

    assert first == "rejected"
    assert second == "already_settled"
    # The second call asked to approve. It must not have dispatched.
    assert calls == []


# ── the operator clip is always masked ───────────────────────────────


def test_gate_fast_path_never_uploads_unmasked_bytes() -> None:
    """Guard on the property that matters: the operator's clip is masked
    unconditionally, not because a model judged it sensitive.

    Read as source rather than executed because the fast path needs a
    real camera tick, YOLO and ffmpeg. That makes this a weak test of
    behaviour but a strong one of intent: it fails the moment someone
    reintroduces the screen-dependent path, which is exactly how raw
    footage of every household would quietly come back.
    """
    import inspect

    from src.services import clip_archival

    src = inspect.getsource(clip_archival.archive_event)
    fast_path = src.split("EVENT-GATE FAST PATH")[1].split("try:")[1]
    fast_path = fast_path.split("await _archive_one")[0]
    # Comments in this block discuss the screen path at length; assert on
    # the CODE, or the prose defeats the check.
    fast_path = "\n".join(
        ln for ln in fast_path.splitlines() if not ln.strip().startswith("#")
    )

    # It censors, in green-silhouette mode, keeping audio.
    assert "_censor_bytes(" in fast_path
    assert "CENSOR_MODE_RESEARCH" in fast_path
    assert "keep_audio=True" in fast_path
    # It does NOT consult the sensitivity screen for this clip.
    assert "resolve_clip_for_upload" not in fast_path
    # The only bytes uploaded are the censored ones.
    upload = fast_path.split("upload_clip(")[1].split(")")[0]
    assert "mp4_bytes=masked" in upload
    assert "region=patient_region" in upload


def test_gate_clip_is_spawned_not_awaited_before_the_caregiver_clip() -> None:
    """The operator's masked clip must never delay the caregiver's.

    Masking is ~14s of single-threaded YOLO. `_archive_one` has only
    `frame_buffer_retention_seconds - clip_post_event_wait_seconds` of
    slack (2s at current settings) before the post-roll it needs is
    evicted from the ring buffer. Awaiting the mask inline therefore
    silently truncated the FAMILY's clip from 26s to 11s — observed
    2026-08-05 on event 2bdf15b5.

    So: the fast path must be spawned, and the spawn must come before
    the `_archive_one` call.
    """
    import inspect

    from src.services import clip_archival

    src = inspect.getsource(clip_archival.archive_event)
    # Whitespace-insensitive: the spawn is a multi-line call.
    flat = " ".join(src.split())
    spawn = flat.find("asyncio.create_task( _gate_fast_path()")
    if spawn == -1:
        spawn = flat.find("asyncio.create_task(_gate_fast_path()")
    archive = flat.find("await _archive_one(")
    assert spawn != -1, "the gate clip is no longer spawned as a task"
    assert archive != -1
    assert spawn < archive, (
        "the gate's masked clip must be spawned BEFORE _archive_one, and "
        "never awaited ahead of it"
    )
    # And it must not be awaited inline anywhere.
    assert "await _gate_fast_path()" not in src
