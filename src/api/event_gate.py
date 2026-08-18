"""Operator endpoints for the pre-dispatch event gate.

Not the caregiver review feature — see `src/services/event_gate.py` for
why those two things share a word and nothing else.

Two surfaces, two different credentials, on purpose:

  `GET /v1/event-gate/queue`
      Operator bearer token (`OperatorAuth`). Cross-tenant listing for
      the ops dashboard and for debugging. No customer token satisfies
      it.

  `GET|POST /v1/event-gate/act/{token}`
      No session at all — the signed link IS the credential. This is
      the page PagerDuty puts on the operator's phone: it renders the
      clip inline and takes the verdict, because a login prompt inside
      a 60-second window is a login prompt nobody completes.

      GET only ever renders. POST performs the verdict, and only with
      `confirm=1`, so the first tap on a phone in the dark shows a
      confirmation rather than silently suppressing a real alert.

      The page is single-use by construction: acting moves `gate_state`
      off 'pending', and every path here refuses to act on anything not
      'pending'. A reopened link renders what already happened.
"""

from __future__ import annotations

import html
import logging
import secrets
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import text

from src.api.deps import OperatorAuth, SessionDep, SettingsDep
from src.db.session import get_sessionmaker
from src.services import event_gate, storage

router = APIRouter(prefix="/v1/event-gate", tags=["event-gate"])

log = logging.getLogger(__name__)

# Replaced with the real per-response nonce inside `_html`.
_NONCE_SLOT = "__CSP_NONCE__"


class QueueItem(BaseModel):
    event_id: str
    title: str
    narrative: str
    patient_name: str
    room: str
    occurred_at: str
    seconds_left: int
    clip_url: str | None = None  # null while the fast-path upload is in flight


_QUEUE_SQL = """
    select e.id, e.title, coalesce(e.narrative, ''),
           coalesce(nullif(p.preferred_name, ''), p.full_name),
           coalesce(e.location, ''), e.occurred_at,
           greatest(0, :auto - extract(epoch from now()
                     - coalesce(e.gate_held_at, e.created_at)))::int,
           e.gate_clip_key
    from events e join patients p on p.id = e.patient_id
    where e.gate_state = 'pending'
    order by e.gate_held_at desc nulls last, e.created_at desc
    limit 50
"""


@router.get("/queue", response_model=list[QueueItem])
async def queue(
    _: OperatorAuth, session: SessionDep, settings: SettingsDep
) -> list[QueueItem]:
    """Every currently-held event, across every household."""
    rows = (await session.execute(
        text(_QUEUE_SQL), {"auto": event_gate.AUTO_RELEASE_S})).all()
    out: list[QueueItem] = []
    for r in rows:
        out.append(QueueItem(
            event_id=str(r[0]), title=r[1], narrative=r[2], patient_name=r[3],
            room=r[4], occurred_at=r[5].isoformat(), seconds_left=r[6],
            clip_url=await _sign_clip(r[7], settings=settings),
        ))
    return out


class ToggleBody(BaseModel):
    patient_id: str
    enabled: bool


@router.post("/toggle", status_code=status.HTTP_204_NO_CONTENT)
async def toggle(_: OperatorAuth, body: ToggleBody, session: SessionDep) -> None:
    """Turn the gate on or off for one patient.

    Writes the SAME `patients.event_gate_enabled` column the caregiver's
    own "Evaluation mode" setting writes (PATCH /v1/patients/{id}). Last
    writer wins, by decision on 2026-08-05.

    That means this route can enable human review of a household that
    never opted in, so every write here is logged at WARNING with the
    before and after value. If we ever need to show who turned review on
    for a given family, that log line is the only record — there is no
    separate consent column to consult.

    Deliberately one patient at a time: there is no "enable everywhere",
    because the blast radius of a mistake there is every household.
    """
    # Read the prior value first, in the same transaction. A subquery
    # inside RETURNING would read the post-update snapshot, so it cannot
    # give us the "was" value the audit line needs.
    was = (await session.execute(text(
        "select event_gate_enabled from patients where id = :pid"),
        {"pid": body.patient_id})).scalar_one_or_none()
    if was is None:
        raise HTTPException(status_code=404, detail="patient_not_found")
    await session.execute(text(
        """update patients set event_gate_enabled = :enabled
           where id = :pid"""),
        {"enabled": body.enabled, "pid": body.patient_id})
    await session.commit()
    # WARNING, not INFO: this is us changing a household's privacy posture
    # from the outside. It should be greppable and it should stand out.
    log.warning(
        "event_gate.toggled_by_operator patient=%s was=%s now=%s",
        body.patient_id, was, body.enabled,
    )


async def _sign_clip(key: str | None, *, settings) -> str | None:
    """Signed URL for a clip key, or None. Never raises: a missing clip
    must still let the operator read the narrative and decide."""
    if not key:
        return None
    try:
        return await storage.signed_url(
            bucket=settings.storage_bucket_incident_clips,
            key=key, settings=settings,
            ttl_seconds=event_gate.AUTO_RELEASE_S * 4,
        )
    except Exception:  # noqa: BLE001
        log.warning("event_gate.clip_sign_failed key=%s", key)
        return None


# ── the verdict page ─────────────────────────────────────────────────


@router.get("/act/{token}/clip")
async def act_clip(token: str, session: SessionDep, settings: SettingsDep) -> Response:
    """Stream the operator's clip from OUR origin, under the same token.

    The page used to embed a storage-signed URL directly. That fails in
    exactly the situation the page is for: the signed URL is built from
    `settings.supabase_url`, which locally is a LAN address the phone
    cannot reach, and is plain http inside an https page, so Safari
    blocks it as mixed content before it even tries. Both failures are
    silent — you get a video element that will not play.

    Proxying fixes both and is better anyway: the bytes leave from the
    same origin as the page, under the same short-lived token, so no
    separately-shareable storage URL for a household's footage ever
    exists. Costs one hop through the backend for a file we only serve
    for 60 seconds.
    """
    event_id = event_gate.verify_verdict_token(token, settings=settings)
    if event_id is None:
        raise HTTPException(status_code=403, detail="invalid_token")
    # ONLY `gate_clip_key`. This used to fall back to `clip_storage_key`
    # when the gate's own clip was missing, which quietly handed the
    # operator the CAREGIVER's clip — screen-dependent, and unmasked
    # whenever the screen cleared the room, which is most of the time.
    # An operator seeing every household must never be served that. If
    # the masked clip is absent, they decide from the narrative.
    # Settlement closes the clip too, not just the page. The token TTL is
    # deliberately longer than the auto-release, so without this check a
    # link kept open served a household's footage for a further ~2 minutes
    # after the decision was already made. The page itself already refuses
    # to act once settled; this makes the bytes follow the same rule.
    row = (await session.execute(text(
        """select e.gate_clip_key
           from events e where e.id = :id and e.gate_state = 'pending'"""),
        {"id": str(event_id)})).first()
    if row is None or not row[0]:
        raise HTTPException(status_code=404, detail="clip_not_ready")
    try:
        data = await storage.download_object(
            bucket=settings.storage_bucket_incident_clips,
            key=row[0], settings=settings)
    except Exception as exc:  # noqa: BLE001
        log.warning("event_gate.clip_proxy_failed event=%s", event_id)
        raise HTTPException(status_code=502, detail="clip_unavailable") from exc
    return Response(
        content=data,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store", "Accept-Ranges": "none"},
    )


@router.get("/act/{token}", response_class=HTMLResponse)
async def act_page(
    token: str, session: SessionDep, settings: SettingsDep
) -> Response:
    """Render the decision page. Never mutates."""
    event_id = event_gate.verify_verdict_token(token, settings=settings)
    if event_id is None:
        return _html(_expired_page(), status_code=status.HTTP_403_FORBIDDEN)
    row = await _load(session, event_id, settings=settings)
    if row is None:
        return _html(_gone_page("This alert no longer exists."),
                     status_code=status.HTTP_404_NOT_FOUND)
    if row["gate_state"] != "pending":
        return _html(_settled_page(row))
    return _html(_decision_page(row, token=token))


@router.post("/act/{token}", response_class=HTMLResponse)
async def act_submit(
    token: str,
    action: str,
    session: SessionDep,
    settings: SettingsDep,
    confirm: int = 0,
) -> Response:
    """Take the verdict.

    `action` and `confirm` arrive as query parameters even though this
    is a POST — the buttons are plain HTML forms whose `action`
    attribute carries the query string and whose body is empty. That
    keeps the page dependency-free (no multipart parsing) while still
    keeping every mutation off GET, so a link preview or a prefetch in
    the PagerDuty app cannot cast a vote.
    """
    event_id = event_gate.verify_verdict_token(token, settings=settings)
    if event_id is None:
        return _html(_expired_page(), status_code=status.HTTP_403_FORBIDDEN)
    if action not in {"release", "suppress"}:
        raise HTTPException(status_code=400, detail="bad_action")

    row = await _load(session, event_id, settings=settings)
    if row is None:
        return _html(_gone_page("This alert no longer exists."),
                     status_code=status.HTTP_404_NOT_FOUND)
    if row["gate_state"] != "pending":
        return _html(_settled_page(row))

    # First tap shows the consequence in words and asks again. Suppress
    # is unrecoverable — the caregiver is never told, and there is no
    # second chance once the 60s is gone.
    if not confirm:
        return _html(_confirm_page(row, token=token, action=action))

    result = await event_gate.act(
        sessionmaker=get_sessionmaker(),
        event_id=event_id,
        approve=(action == "release"),
        settings=settings,
    )
    log.info(
        "event_gate.operator_verdict event=%s action=%s result=%s",
        event_id, action, result,
    )
    return _html(_result_page(row, result=result))


# Record-only tick verdicts: rule on the COUNT. GET renders, POST
# mutates; no confirm step (a count correction is recoverable).


@router.get("/tick/{token}", response_class=HTMLResponse)
async def tick_page(
    token: str, session: SessionDep, settings: SettingsDep
) -> Response:
    got = event_gate.verify_tick_token(token, settings=settings)
    if got is None:
        return _html(_expired_page(), status_code=status.HTTP_403_FORBIDDEN)
    camera_id, captured_at, ident = got
    row = await _load_tick(session, camera_id, captured_at, ident)
    if row is None:
        return _html(_gone_page("This tick no longer exists."),
                     status_code=status.HTTP_404_NOT_FOUND)
    if row["verdict"]:
        return _html(f"<h2>Already settled: {html.escape(row['verdict'])}</h2>")
    base = f"/v1/event-gate/tick/{html.escape(token, quote=True)}"
    name = html.escape(ident.replace("_", " ").title())
    return _html(
        f"<h2>{name}</h2>"
        f"<div class='narr'>{html.escape(row['reasoning'] or 'no reasoning recorded')}</div>"
        "<p class='sub'>This is a COUNT, not an alert — nothing is sent "
        "either way.</p>"
        + f"<form method='post' action='{base}?action=reject'>"
          "<button class='suppress'>Don't count — false positive</button></form>"
        + f"<form method='post' action='{base}?action=accept'>"
          "<button class='release'>Count it — this happened</button></form>"
    )


@router.post("/tick/{token}", response_class=HTMLResponse)
async def tick_submit(
    token: str, action: str, session: SessionDep, settings: SettingsDep,
) -> Response:
    got = event_gate.verify_tick_token(token, settings=settings)
    if got is None:
        return _html(_expired_page(), status_code=status.HTTP_403_FORBIDDEN)
    if action not in {"accept", "reject"}:
        raise HTTPException(status_code=400, detail="bad_action")
    camera_id, captured_at, ident = got
    stamped = "accepted" if action == "accept" else "rejected"
    # First writer wins — the merge applies only while no verdict key
    # exists, so a second tap cannot flip the first ruling.
    updated = (await session.execute(text(
        """update analyses
              set detector_fields = detector_fields ||
                  jsonb_build_object(
                      (:vf)::text, (:v)::text,
                      (:vat)::text, to_char(now() at time zone 'utc',
                                            'YYYY-MM-DD"T"HH24:MI:SS"Z"'))
            where camera_id = :c
              and captured_at between (:t)::timestamptz - interval '1 second'
                                  and (:t)::timestamptz + interval '1 second'
              and detector_fields->>:vf is null
            returning id"""),
        {"c": camera_id, "t": captured_at, "v": stamped,
         "vf": f"{ident}_verdict", "vat": f"{ident}_verdict_at"},
    )).first()
    await session.commit()
    if updated is None:
        row = await _load_tick(session, camera_id, captured_at, ident)
        prior = (row or {}).get("verdict")
        return _html(f"<h2>Already settled: {html.escape(prior or 'row missing')}</h2>")
    log.info(
        "event_gate.tick_verdict ident=%s verdict=%s tick=%s cam=%s",
        ident, stamped, captured_at.isoformat(), camera_id,
    )
    return _html(
        f"<h2>{'Counted' if stamped == 'accepted' else 'Removed from the count'}</h2>"
        f"<p class='sub'>{html.escape(ident)} — {stamped}.</p>"
    )


async def _load_tick(session, camera_id, captured_at, ident) -> dict | None:
    row = (await session.execute(text(
        """select detector_fields->>:rf, detector_fields->>:vf
             from analyses
            where camera_id = :c
              and captured_at between (:t)::timestamptz - interval '1 second'
                                  and (:t)::timestamptz + interval '1 second'
            limit 1"""),
        {"c": camera_id, "t": captured_at,
         "rf": f"{ident}_reasoning", "vf": f"{ident}_verdict"},
    )).first()
    if row is None:
        return None
    return {"reasoning": row[0], "verdict": row[1]}


async def _load(session, event_id: UUID, *, settings) -> dict | None:
    row = (await session.execute(text(
        """select e.title, coalesce(e.narrative, ''),
                  coalesce(nullif(p.preferred_name, ''), p.full_name),
                  coalesce(e.location, ''), e.gate_state,
                  greatest(0, :auto - extract(epoch from now()
                            - coalesce(e.gate_held_at, e.created_at)))::int,
                  e.gate_clip_key, e.category::text
           from events e join patients p on p.id = e.patient_id
           where e.id = :id"""),
        {"id": str(event_id), "auto": event_gate.AUTO_RELEASE_S})).first()
    if row is None:
        return None
    return {
        "title": row[0], "narrative": row[1], "patient_name": row[2],
        "room": row[3], "gate_state": row[4], "seconds_left": row[5],
        # Just whether there IS one. The page points at our own proxy
        # route rather than a storage URL — see `act_clip`.
        "has_clip": bool(row[6]),
        "urgent": row[7] == "safety",
    }


# ── HTML ─────────────────────────────────────────────────────────────
# Server-rendered, self-contained, no external assets. It loads on a
# phone on cellular in a dark room in under a second, which is the only
# requirement that matters. Every interpolated value is escaped: the
# narrative is model-generated text and goes straight into the DOM.

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:20px; background:#0b0d10; color:#e9edf2;
  font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
.wrap { max-width:520px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; }
.sub { color:#9aa4b2; font-size:15px; margin:0 0 16px; }
video { width:100%; border-radius:12px; background:#000; margin-bottom:16px; }
.narr { background:#151a21; border-radius:12px; padding:14px; margin-bottom:20px;
  font-size:15px; }
.count { font-variant-numeric:tabular-nums; color:#ffb020; font-weight:600; }
form { display:block; margin:0 0 10px; }
button { width:100%; padding:16px; font-size:17px; font-weight:600;
  border:0; border-radius:12px; color:#fff; }
.suppress { background:#c0392b; }
.release { background:#1f7a3f; }
.cancel { background:#2a313a; }
.note { color:#9aa4b2; font-size:13px; margin-top:18px; }
.big { font-size:19px; font-weight:600; margin:24px 0 8px; }
"""


def _html(body: str, *, status_code: int = 200) -> Response:
    """Render one page, with a CSP narrow enough to survive review.

    THIS PAGE MUST SET ITS OWN CSP. `SecurityHeadersMiddleware` applies
    `default-src 'none'; form-action 'none'` to everything, which is
    correct for a JSON API and fatal here: it blocks the inline style
    (unstyled page), the inline countdown script, the <video> (media-src
    falls back to default-src, so the clip renders as a dead black box),
    and — worst — form submission, so neither verdict button can fire.
    Observed live on 2026-08-05 after the middleware landed. The
    middleware yields to a handler that sets the header itself, which is
    why this works and why it must not be removed.
    """
    # Per-response nonce rather than 'unsafe-inline'. The narrative on
    # this page is model-generated text; it is escaped, but allowing
    # arbitrary inline script would mean one escaping slip is executable
    # on the one device that can suppress alerts for every household.
    nonce = secrets.token_urlsafe(16)
    csp = (
        "default-src 'none'; "
        f"style-src 'nonce-{nonce}'; "
        f"script-src 'nonce-{nonce}'; "
        "media-src 'self'; "      # the clip, proxied from our own origin
        "form-action 'self'; "    # the two verdict buttons
        "base-uri 'none'; frame-ancestors 'none'"
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Memoria alert review</title>"
        f"<style nonce='{nonce}'>{_CSS}</style></head>"
        f"<body><div class='wrap'>{body.replace(_NONCE_SLOT, nonce)}</div></body></html>"
    )
    # These pages carry one household's clip and narrative. Keep them out
    # of every cache and out of any referrer sent to a third party.
    return HTMLResponse(
        content=page,
        status_code=status_code,
        headers={
            "Content-Security-Policy": csp,
            "Cache-Control": "no-store, max-age=0",
            "Referrer-Policy": "no-referrer",
            "X-Robots-Tag": "noindex, nofollow",
        },
    )


def _header(row: dict) -> str:
    who = html.escape(row["patient_name"])
    what = html.escape(row["title"])
    where = html.escape(row["room"])
    sub = f"{who}{f' · {where}' if where else ''}"
    return f"<h1>{what}</h1><p class='sub'>{sub}</p>"


def _clip(row: dict, *, token: str) -> str:
    if not row["has_clip"]:
        return ("<div class='narr'>Clip still uploading. Decide on the "
                "description below, or wait a few seconds and reload.</div>")
    src = f"/v1/event-gate/act/{html.escape(token, quote=True)}/clip"
    # Deliberately NOT muted. The first version carried `muted` so the
    # clip would autoplay, since browsers block autoplay with sound —
    # but for "possible calling for help" the audio IS the evidence, and
    # a silent autoplay quietly throws away half of what the operator
    # needs to judge. Sound wins; the cost is one tap to start playback
    # on iOS, inside a 60-second window that can afford it.
    return (f"<video controls playsinline preload='auto' src='{src}'></video>")


def _decision_page(row: dict, *, token: str) -> str:
    left = int(row["seconds_left"])
    base = f"/v1/event-gate/act/{html.escape(token, quote=True)}"
    return (
        _header(row)
        + _clip(row, token=token)
        + f"<div class='narr'>{html.escape(row['narrative'])}</div>"
        + (f"<p class='sub'>Auto-sends to the caregiver in "
           f"<span class='count' id='c'>{left}</span>s if you do nothing.</p>"
           if row.get("urgent")
           else "<p class='sub'>Waits for your verdict — nothing is sent "
                "until you decide.</p>")
        + f"<form method='post' action='{base}?action=suppress'>"
          "<button class='suppress'>Suppress — false alarm</button></form>"
        + f"<form method='post' action='{base}?action=release'>"
          "<button class='release'>Send now — this is real</button></form>"
        + "<p class='note'>Suppressing means the caregiver is never told "
          "about this alert.</p>"
        + (f"<script nonce='{_NONCE_SLOT}'>var n=" + str(left) + ",e=document.getElementById('c');"
           "setInterval(function(){if(n>0){n--;e.textContent=n;}},1000);</script>"
           if row.get("urgent") else "")
    )


def _confirm_page(row: dict, *, token: str, action: str) -> str:
    base = f"/v1/event-gate/act/{html.escape(token, quote=True)}"
    if action == "suppress":
        question = "Suppress this alert?"
        detail = (f"{html.escape(row['patient_name'])}'s caregiver will never "
                  "be told about it. This cannot be undone.")
        cls, label = "suppress", "Yes, suppress it"
    else:
        question = "Send this alert now?"
        detail = (f"{html.escape(row['patient_name'])}'s caregiver is notified "
                  "immediately, without waiting out the timer.")
        cls, label = "release", "Yes, send it"
    return (
        _header(row)
        + f"<p class='big'>{question}</p>"
        + f"<p class='sub'>{detail}</p>"
        + f"<form method='post' action='{base}?action={action}&confirm=1'>"
          f"<button class='{cls}'>{label}</button></form>"
        + f"<form method='get' action='{base}'>"
          "<button class='cancel'>Cancel</button></form>"
    )


def _result_page(row: dict, *, result: str) -> str:
    if result == "rejected":
        line, note = "Suppressed.", "The caregiver will not be notified."
    elif result == "approved":
        line, note = "Sent.", "The caregiver has been notified."
    else:
        line, note = ("Already sent.",
                      "The 60-second timer released this alert before your "
                      "decision landed. The caregiver has been notified.")
    return (_header(row) + f"<p class='big'>{line}</p><p class='sub'>{note}</p>"
            + "<p class='note'>This link is now closed.</p>")


def _settled_page(row: dict) -> str:
    state = row["gate_state"]
    note = {
        "rejected": "It was suppressed. The caregiver was not notified.",
        "approved": "It was sent to the caregiver.",
        "auto": "The 60-second timer released it. The caregiver was notified.",
    }.get(state, "It is no longer awaiting a decision.")
    return (_header(row) + "<p class='big'>Already decided.</p>"
            + f"<p class='sub'>{note}</p>"
            + "<p class='note'>This link is now closed.</p>")


def _expired_page() -> str:
    return ("<p class='big'>This link has expired.</p>"
            "<p class='sub'>Verdict links are valid for a few minutes. The "
            "alert it referred to has already been sent or suppressed.</p>")


def _gone_page(msg: str) -> str:
    return f"<p class='big'>{html.escape(msg)}</p>"
