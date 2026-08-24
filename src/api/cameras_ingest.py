"""Fleet endpoints: how a camera service talks to this backend.

Gateway boxes and camera bridges enrol a bearer token, poll for the
cameras they own, and report per-camera health. See
docs/design/fleet-service-contract.md.

Was the Raspberry Pi MJPEG push ingest. That hardware never shipped and
the push path is deleted (2026-08-21); what remained and grew here is the
fleet contract.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import health as health_mod
from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.camera import Camera
from src.repos import camera_detector_repo
from src.services import livekit_rooms
from src.services.fleet_auth import hash_ingest_token, mint_ingest_token

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["fleet"])

# Which sensors a gateway box owns. Ownership is the ACCOUNT
# (cameras.caregiver_id), not the patient — a sensor is on the account
# before anyone assigns it, and it streams from that moment.
# `gateway_id IS NULL` is the transitional fallback until the ownership
# backfill fills it in.
#
# Module level because both /gateway/roster and /fleet/poll answer the
# same question. Two copies of this predicate would eventually disagree
# about which cameras a box owns, which is how the audio-kill bug got to
# live in two endpoints at once.
#
# `patient_id IS NOT NULL` was here until 2026-08-22 and was the reason an
# unassigned sensor never published: it had no patient, so it was not on
# the roster, so nothing told it to stream. The installer could not see
# their own work. A sensor is on the roster because it EXISTS on the
# account, not because someone has decided yet who it watches.
_GATEWAY_CAM_WHERE = (
    "WHERE (caregiver_id = :c OR patient_id IN "
    "      (SELECT id FROM public.patients WHERE caregiver_id = :c)) "
    "AND source_type = 'gateway' "
    "AND (gateway_id = :gw OR gateway_id IS NULL)")

# A hosted service (the Warner bridge) serves MANY accounts, so the
# caregiver filter above is exactly wrong for it — provisioning for a
# second customer rebound the whole service and the first customer's
# cameras vanished from the roster. Ownership is the gateway row and
# nothing else here: no caregiver scope, and no `gateway_id IS NULL`
# fallback, because an unowned camera belongs to whoever provisioned it
# and a shared service must never guess. This is NARROWER than a box's
# scope, which is what makes it safe to let a service ask for it.
_BRIDGE_CAM_WHERE = (
    "WHERE gateway_id = :gw "
    "AND source_type = 'gateway'")

# The row every fleet member gets back. `privacy_enabled` is room-derived;
# `audio_capture_enabled` is the SENSOR's own column; `online` is the
# SUBSCRIBER's view of whether frames are actually landing, which a
# publisher cannot see about itself.
def _cam_select(multi_tenant: bool) -> str:
    """The camera list, scoped by what this service IS."""
    where = _BRIDGE_CAM_WHERE if multi_tenant else _GATEWAY_CAM_WHERE
    return _GATEWAY_CAM_SELECT_HEAD + where + " ORDER BY created_at"


# The audio kill is the SENSOR's own column since 2026-08-22. It was a
# bool_and across covered patients, which is a statement about people
# used to answer a question about a room — and once a sensor can cover
# nobody, that expression returns NULL, and NULL fails closed, so every
# unassigned sensor would have been silently deaf.
_GATEWAY_CAM_SELECT_HEAD = (
    "SELECT id, livekit_identity, livekit_room, desired_state, "
    "privacy_enabled, online, audio_capture_enabled "
    "FROM public.cameras ")

FLEET_CONTRACT = "fleet-v1"


@router.post(
    "/cameras/me/heartbeat",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def post_heartbeat(
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict[str, bool]:
    """Liveness ping from a gateway (mini-PC pulling RTSP).

    The gateway posts this every few seconds per camera it manages. We stamp
    `cameras.last_heartbeat_at`. A stale stamp means the GATEWAY itself is down
    (crashed / power loss / lost internet) — which is DISTINCT from
    `desired_state='disconnected'` (a caregiver intentionally turned it off) and
    from `online=false` (not currently publishing). That three-way distinction
    is what lets the dashboard say "gateway offline" instead of mislabeling a
    dead mini-PC as an intentional disconnect.

    Same ingest-token bearer auth as the push + ingest-status endpoints, and a
    single raw UPDATE keyed by the token hash (no ORM model change needed).
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    result = await session.execute(
        text(
            "UPDATE public.cameras SET last_heartbeat_at = now() "
            "WHERE ingest_token_hash = :digest RETURNING id"
        ),
        {"digest": digest},
    )
    if result.first() is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_ingest_token",
        )
    await session.commit()
    return {"ok": True}


@router.get(
    "/gateway/roster",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def gateway_roster(
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """The gateway's source-of-truth roster.

    A mini-PC gateway holds ONE stable site token (not per-camera tokens) and
    polls this every few seconds. We return exactly the cameras it should be
    running, with their live desired_state — so a dashboard add/delete/connect/
    disconnect propagates on the next poll, and a deleted camera simply vanishes
    from the list (no orphaned per-camera token, no 401 flood). The act of
    polling also stamps liveness (gateway + each camera's last_heartbeat_at), so
    this single call replaces the separate per-camera desired-state + heartbeat
    polls. Same sha256 bearer-token shape as the ingest endpoints.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    gw = (await session.execute(
        text("SELECT id, caregiver_id, multi_tenant FROM public.gateways "
             " WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    gateway_id, caregiver_id, multi_tenant = gw[0], gw[1], bool(gw[2])
    # Liveness (polling the roster IS the heartbeat) + reverse-SSH tunnel port
    # in one statement. The port is assigned lazily on the box's FIRST poll
    # (covers rows that predate the column too — no backfill needed); the box
    # writes it to .tunnel-port and its tunnel unit dials the relay on it,
    # which is what makes every deployed box reachable by ops. COALESCE
    # short-circuits (Postgres §9.18), so nextval() fires only when the port
    # is NULL — steady-state polls burn nothing — and the row lock serializes
    # concurrent first polls, so RETURNING always yields the final port.
    # nextval() never issues 22001 (reserved for the legacy box-01 unit); see
    # scripts/deltas/apply_gateway_tunnel_port_delta.sql for the per-env port ranges.
    tunnel_port = (await session.execute(
        text("UPDATE public.gateways SET last_seen_at = now(), "
             "tunnel_port = COALESCE(tunnel_port, "
             "  nextval('public.gateway_tunnel_port_seq')) "
             "WHERE id = :id RETURNING tunnel_port"),
        {"id": gateway_id},
    )).scalar()
    # A gateway box serves a CAREGIVER, but each gateway camera is OWNED by
    # a specific box (cameras.gateway_id). Return only THIS box's cameras,
    # plus any not-yet-assigned (gateway_id IS NULL) rows as a transitional
    # fallback until the ownership backfill/provision fills them in. This is
    # what makes liveness per-box: the heartbeat below only stamps the
    # cameras this box actually runs.
    # Ownership is the ACCOUNT (cameras.caregiver_id), not the patient — a
    # camera can be on the account before anyone assigns it. `patient_id IS
    # NOT NULL` is what keeps those unassigned ones OUT of the roster, so
    # they never publish until someone says who they watch.
    _cam_where = _BRIDGE_CAM_WHERE if multi_tenant else _GATEWAY_CAM_WHERE
    await session.execute(
        text(f"UPDATE public.cameras SET last_heartbeat_at = now() {_cam_where}"),
        {"c": caregiver_id, "gw": gateway_id},
    )
    rows = (await session.execute(
        text(_cam_select(multi_tenant)), {"c": caregiver_id, "gw": gateway_id},
    )).all()
    await session.commit()
    # `privacy_enabled` (room-derived, 2026-07-27): when true the gateway
    # restricts this camera's track subscription permissions to
    # detector_worker_* identities (publisher-side sever) — same contract
    # as the wall phone's desired-state poll.
    # `audio_capture_enabled` (patient-level, 2026-07-28): when false the
    # gateway never publishes the audio track — the facilities audio kill,
    # severed at the source so nothing downstream can leak it.
    # `online` (2026-07-29): the SUBSCRIBER's view of whether this
    # camera's frames are actually landing here — stamped by the room
    # worker on real frame arrival and independent of detector activity
    # (an ai_paused camera still reads online). The gateway cannot see
    # this about itself: in the 2026-07-29 wifi-blip zombie its room,
    # publications and connection_state all said healthy while zero
    # media left the box. Handing it back closes that blind spot.
    # Deliberately NOT in /gateway/wait's fingerprint — `online` flaps
    # on ordinary blips, and fingerprinting it would wake the long poll
    # constantly for something the 3s roster poll delivers anyway.
    return {"tunnel_port": tunnel_port, "cameras": _camera_rows(rows)}


@router.get(
    "/gateway/wait",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def gateway_wait(
    session: SessionDep,
    since: str = "",
    authorization: str = Header(default=""),
) -> dict:
    """Long-poll: HOLD this request open until the gateway's roster (which
    cameras + their desired_state, fingerprinted) differs from `since`, then
    return the new roster. The gateway keeps ONE of these open and re-issues it,
    so a dashboard connect/disconnect reaches it in ~0.4s instead of waiting for
    the 3s poll. The plain `/gateway/roster` poll stays as the fallback if this
    drops. (Internal 0.4s DB check; a Postgres LISTEN/NOTIFY upgrade would make
    it truly instant + zero internal polling.)
    """
    # Dependency-health stamp: /ready reports how long ago the gateway
    # last long-polled — a box whose service died or whose network
    # dropped stops polling, and that age is the cheapest backend-side
    # signal of it. Must stay BELOW the docstring: put above it, the
    # string stops being the docstring and the contract above vanishes
    # from /docs and from help().
    health_mod.last_gateway_poll_mono = time.monotonic()
    # Same helper the roster uses, so this cannot drift from it — and it
    # is the ONLY source of `multi_tenant`. The inlined lookup that used
    # to be here selected `id, caregiver_id` alone, so a bridge long-
    # polled on the single-tenant BOX scope while its roster answered on
    # the bridge scope: the two disagreed about which sensors exist, and
    # a change to one the fingerprint could not see would never wake it.
    gateway_id, caregiver_id, multi_tenant = await _gateway_from_bearer(
        session, authorization,
    )

    async def snapshot():
        rows = (await session.execute(
            text("SELECT id, livekit_identity, livekit_room, desired_state, "
                 "privacy_enabled, "
                 # Same audio rule as the roster, and it has to match
                 # exactly: this feeds the fingerprint the long poll
                 # wakes on.
                 "audio_capture_enabled "
                 "FROM public.cameras "
                 + (_BRIDGE_CAM_WHERE if multi_tenant else _GATEWAY_CAM_WHERE)
                 + " ORDER BY livekit_identity"),
            {"c": caregiver_id, "gw": gateway_id},
        )).all()
        await session.commit()   # release the connection between checks
        cams = [{"camera_id": str(r[0]), "livekit_identity": r[1],
                 "livekit_room": r[2], "desired_state": r[3],
                 "privacy_enabled": bool(r[4]),
                 # Fails CLOSED, matching _camera_rows above.
                 "audio_capture_enabled":
                     bool(r[5]) if r[5] is not None else False}
                for r in rows]
        # privacy_enabled + audio_capture_enabled ride in the fingerprint
        # so a classifier / reclassify / audio-kill flip wakes the
        # long-poll immediately — the gateway must react within ~0.4s,
        # not wait for the 3s roster fallback.
        fp = "|".join(
            f"{c['livekit_identity']}:{c['desired_state']}"
            f":{int(c['privacy_enabled'])}"
            f":{int(c['audio_capture_enabled'])}"
            for c in cams
        )
        return cams, fp

    elapsed = 0.0
    while elapsed < 25.0:
        cams, fp = await snapshot()
        if fp != since:
            return {"changed": True, "cameras": cams, "fingerprint": fp}
        await asyncio.sleep(0.4)
        elapsed += 0.4
    cams, fp = await snapshot()
    return {"changed": False, "fingerprint": fp}


class GatewayProvisionRequest(BaseModel):
    """Bind a service and create a sensor, in one call.

    `pairing_code` is what proves the account. Required the first time a
    token is seen and ignored afterwards, so mounting a second sensor in
    the same home does not send the installer back for another code.

    `email` is an optional TYPO CHECK. The code already names the
    account; if the email disagrees, the install is wrong in a way that
    picking one of them cannot fix.

    `service_kind` is how a HOSTED service says it serves many accounts.
    Declaring it NARROWS what the service can see — sensors carrying its
    own gateway_id and nothing else — so it grants nothing.
    """

    pairing_code: str | None = None
    email: str | None = None
    service_kind: str | None = None
    room: str = "Camera"
    icon_name: str = "tv"


def _camera_rows(rows) -> list[dict]:
    """One shape for every fleet member, so the roster and the fleet poll
    cannot answer the same question differently."""
    return [
        {"camera_id": str(r[0]), "livekit_identity": r[1],
         "livekit_room": r[2], "desired_state": r[3],
         "privacy_enabled": bool(r[4]),
         "online": bool(r[5]),
         # NULL means we could not determine it, and the flag means
         # "this site does not permit recording audio". Defaulting to
         # permitted is the unsafe direction, and the bridge already
         # defaults the same field to False for the same reason — the
         # two ends of the contract must agree on which way is safe.
         "audio_capture_enabled": bool(r[6]) if r[6] is not None else False}
        for r in rows
    ]


async def _gateway_from_bearer(
    session: AsyncSession, authorization: str,
) -> tuple[UUID, UUID, bool]:
    """Identify the polling service by its own bearer token.

    Only the sha256 is stored, so this is a hash lookup rather than a
    comparison. Returns `(gateway_id, caregiver_id, multi_tenant)`.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    gw = (await session.execute(
        text("SELECT id, caregiver_id, multi_tenant FROM public.gateways "
             " WHERE token_hash = :d"),
        {"d": hash_ingest_token(presented)},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    return gw[0], gw[1], bool(gw[2])


# Kinds that serve MANY caregiver accounts. A box in one home does not.
_MULTI_TENANT_KINDS = frozenset({"camera-bridge"})


def _is_multi_tenant(service_kind: str | None) -> bool:
    return (service_kind or "").strip().lower() in _MULTI_TENANT_KINDS


async def _redeem_pairing_code(
    session: AsyncSession, pairing_code: str, now: datetime,
    *, email: str | None, gateway_id: UUID | None,
) -> UUID:
    """Consume a code and return the account that issued it.

    Atomic: the UPDATE is the claim, so two services racing on one code
    cannot both win. Unknown, expired and already-used collapse to one
    answer — a caller probing codes learns nothing from the difference.
    """
    code_hash = hash_ingest_token(pairing_code.strip())
    claimed = (await session.execute(
        text("UPDATE public.gateway_pairing_codes "
             "   SET consumed_at = :now, consumed_by = COALESCE(:g, consumed_by) "
             " WHERE code_hash = :h AND consumed_at IS NULL "
             "   AND expires_at > :now "
             "RETURNING caregiver_id"),
        {"h": code_hash, "now": now, "g": gateway_id},
    )).first()
    if claimed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_pairing_code",
        )
    caregiver_id = claimed[0]

    if email:
        owner = (await session.execute(
            text("SELECT lower(email) FROM public.caregivers WHERE id = :c"),
            {"c": caregiver_id},
        )).scalar_one_or_none()
        if owner != email.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="email_does_not_match_pairing_code",
            )
    return caregiver_id


async def _bind_service_token(
    session: AsyncSession, authorization: str, now: datetime,
    *, pairing_code: str | None, email: str | None,
    service_kind: str | None,
) -> tuple[UUID, UUID]:
    """Resolve this service's token to an account, binding it if new.

    ONE door. A token the backend has never seen binds only by redeeming
    a pairing code the caregiver just generated; a token it already knows
    needs nothing, so an installer mounting a second sensor ten minutes
    later does not have to walk back for another code.

    This took the caregiver's EMAIL until 2026-08-23. An email is on the
    invoice and in the signature, so it proved nothing — anyone who knew
    one could bind a token and poll that account's roster. It is now
    accepted only as a typo check: if it disagrees with the code, the
    install is wrong in a way that guessing cannot fix.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    token = authorization[len("bearer "):].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    token_hash = hash_ingest_token(token)

    existing = (await session.execute(
        text("SELECT id, caregiver_id, multi_tenant FROM public.gateways "
             " WHERE token_hash = :h"),
        {"h": token_hash},
    )).first()

    if existing is not None:
        gateway_id, bound_caregiver, was_mt = (
            existing[0], existing[1], bool(existing[2]))
        # Re-declared on every call, so a row created before the service
        # knew its own kind repairs itself without a human.
        if _is_multi_tenant(service_kind) and not was_mt:
            await session.execute(
                text("UPDATE public.gateways SET multi_tenant = true "
                     " WHERE token_hash = :h"), {"h": token_hash})
        if not pairing_code:
            # No code: this sensor belongs to the account the service is
            # bound to. That is the second-sensor-in-the-same-home case.
            return gateway_id, bound_caregiver
        # A code IS presented, so this sensor belongs to whoever issued
        # it — which is how ONE hosted bridge serves many customers. The
        # gateways row is NOT rebound: it is the service's identity, and
        # rewriting it is exactly what used to drop the previous
        # customer's sensors off the roster. Ownership of a SENSOR is
        # `cameras.caregiver_id`; ownership of the SERVICE is this row.
        code_caregiver = await _redeem_pairing_code(
            session, pairing_code, now, email=email, gateway_id=gateway_id)
        return gateway_id, code_caregiver

    if not pairing_code:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="pairing_code_required",
        )

    caregiver_id = await _redeem_pairing_code(
        session, pairing_code, now, email=email, gateway_id=None)

    gateway_id = (await session.execute(
        text("INSERT INTO public.gateways "
             " (caregiver_id, token_hash, created_at, multi_tenant) "
             "VALUES (:c, :h, :n, :mt) "
             "ON CONFLICT (token_hash) DO NOTHING RETURNING id"),
        {"c": caregiver_id, "h": token_hash, "n": now,
         "mt": _is_multi_tenant(service_kind)},
    )).scalar_one_or_none()
    if gateway_id is None:
        gateway_id = (await session.execute(
            text("SELECT id FROM public.gateways WHERE token_hash = :h"),
            {"h": token_hash},
        )).scalar_one()
    await session.execute(
        text("UPDATE public.gateway_pairing_codes SET consumed_by = :g "
             " WHERE code_hash = :h"),
        {"g": gateway_id, "h": hash_ingest_token(pairing_code.strip())})
    return gateway_id, caregiver_id


class GatewayPairingCodeResponse(BaseModel):
    pairing_code: str
    expires_in_seconds: int


# Five minutes. Long enough to read six digits down a phone line and type
# them; short enough that a code left on a screen is not a standing key
# to somebody's home.
_PAIRING_TTL_S = 300


@router.post(
    "/gateway/pairing-code",
    status_code=status.HTTP_200_OK,
    response_model=GatewayPairingCodeResponse,
)
async def create_gateway_pairing_code(
    auth: CaregiverAuth,
    session: SessionDep,
) -> GatewayPairingCodeResponse:
    """Mint the code an installer types to bind a service to THIS account.

    Caregiver-authenticated, so the person granting access is the person
    who owns the home — which is the whole point, and is what an email
    could never establish.

    Only the sha256 is stored; the plaintext is returned once and never
    again. Six digits over a five-minute window is ~3.3 codes per million
    guesses per attempt, and a code dies the moment it is used.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=_PAIRING_TTL_S)
    for _ in range(5):
        code = f"{secrets.randbelow(900_000) + 100_000}"
        inserted = (await session.execute(
            text("INSERT INTO public.gateway_pairing_codes "
                 " (code_hash, caregiver_id, created_at, expires_at) "
                 "VALUES (:h, :c, :n, :e) "
                 "ON CONFLICT (code_hash) DO NOTHING RETURNING code_hash"),
            {"h": hash_ingest_token(code), "c": auth.id, "n": now, "e": expires},
        )).first()
        if inserted is not None:
            await session.commit()
            return GatewayPairingCodeResponse(
                pairing_code=code, expires_in_seconds=_PAIRING_TTL_S)
    # Five collisions against a live-code set this small means something
    # is wrong with the entropy, not with luck.
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="could_not_allocate_pairing_code")


@router.post(
    "/gateway/provision-camera",
    status_code=status.HTTP_201_CREATED,
)
async def gateway_provision_camera(
    body: GatewayProvisionRequest,
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """Provision a gateway camera in the field with EMAIL ONLY — no password.

    Tenets: a generic box, set up in a new home, phone-only, no password.
      - Authorization: Bearer <box's UNIQUE token> (generated on first boot).
      - body.email identifies the CAREGIVER (account) this box belongs to.
    We tie this box's token -> that caregiver (public.gateways), then create the
    camera under EVERY one of the caregiver's patients. The roster returns them
    by caregiver. Nothing account-specific is baked into the image — the box
    learns its account HERE, at setup.
    """
    now = datetime.now(timezone.utc)
    # Bind and create in ONE call, which is also the only moment the
    # installer and the caregiver are together. A new token redeems the
    # code; a token already bound just provisions.
    gateway_id, caregiver_id = await _bind_service_token(
        session, authorization, now,
        pairing_code=body.pairing_code,
        email=body.email,
        service_kind=body.service_kind,
    )

    # No patient lookup any more: a camera belongs to the account, so it can
    # be provisioned before the caregiver has added anyone. The old
    # caregiver_has_no_patients rejection is gone with it.

    # ONE row per physical camera (2026-08-19). This used to loop over every
    # patient of the caregiver, so a single Trueview camera in a shared room
    # became N rows publishing N streams — N× ingest, N× model spend on
    # identical footage, duplicate events, and no way to add a patient later.
    # The camera now belongs to the ACCOUNT and starts unassigned: it is
    # absent from the roster, so it does not publish until someone says who
    # it watches (see POST /v1/cameras/{id}/assign).
    cameras = []
    for camera_id in (uuid4(),):
        livekit_identity = f"cam_{camera_id.hex[:8]}"
        # The room is the ACCOUNT's, so it is known right here — an
        # installer types a caregiver's email and nothing else, and that
        # is enough. It used to be None until a caregiver picked a
        # patient, which meant the person standing in the house holding
        # the sensor could not see it work before they left.
        livekit_room = livekit_rooms.account_room(caregiver_id)
        next_display_id = (await session.execute(
            select(func.coalesce(func.max(Camera.display_id), 0)).where(
                Camera.caregiver_id == caregiver_id)
        )).scalar_one() + 1
        ingest_token_plaintext, ingest_token_hash = mint_ingest_token()
        cam = Camera(
            id=camera_id,
            caregiver_id=caregiver_id,
            patient_id=None,
            display_id=next_display_id,
            room=body.room,
            icon_name=body.icon_name,
            online=False,
            created_at=now,
            pairing_code=None,
            pairing_expires_at=None,
            api_key_hash=None,
            livekit_identity=livekit_identity,
            livekit_room=livekit_room,
            state="open",
            no_people_count=0,
            state_changed_at=now,
            close_threshold=6,
            battery_percent=None,
            # A sensor an installer just mounted is meant to be running.
            # This defaulted to 'disconnected', copied from the wall
            # phone, where it is right — a phone is battery-powered and
            # someone must choose to spend it. A bridge sensor is mains
            # powered and was mounted for the purpose of streaming, so
            # defaulting it off meant the installer's own sensor was dark
            # until a caregiver, who is not in the house, turned it on.
            # The now-deleted pi-provision already defaulted to
            # connected; the two paths disagreeing was the tell.
            desired_state="connected",
            source_type="gateway",
            # Own this camera to THIS box, so the roster + heartbeat scope
            # per-gateway (not fleet-wide across the caregiver's boxes).
            gateway_id=gateway_id,
            ingest_token_hash=ingest_token_hash,
            ingest_token_created_at=now,
            app_present=True,
        )
        session.add(cam)
        await session.flush()
        cameras.append({
            # Always unassigned: the caregiver names it and says who it
            # watches in the app. One path, whether the account has one
            # patient or twenty.
            "patient_id": None,
            "camera_id": str(camera_id),
            "livekit_identity": livekit_identity,
            "livekit_room": livekit_room,
            "ingest_token": ingest_token_plaintext,
        })
    await session.commit()
    return {"caregiver_email": body.email, "provisioned": len(cameras),
            "cameras": cameras}


class FleetPollRequest(BaseModel):
    """A fleet member's report. See docs/design/fleet-service-contract.md.

    Loosely typed on purpose: the backend must accept a report from a
    service one version ahead without rejecting it, and `contract` is what
    guards against actually misreading one.
    """

    contract: str = ""
    service: dict = {}
    entities: list[dict] = []


@router.post(
    "/fleet/poll",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def fleet_poll(
    body: FleetPollRequest,
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """Report first, then be told what to run.

    The camera bridge polled `/gateway/roster` for `desired_state`,
    `privacy_enabled` and `audio_capture_enabled`, acted on none of them,
    and reported nothing back — and nothing detected that for weeks,
    because reporting was optional and acting was unverified. Making the
    report a PRECONDITION of the instruction fixes that structurally: a
    service that stops reporting stops being told what to do, and goes
    visibly stale within one poll interval.

    The second rule is that liveness is per CAMERA. `/gateway/roster`
    stamps `last_heartbeat_at` for every camera on the box simply because
    the box is polling, so a camera whose worker died ten minutes ago
    still reads as healthy — and a blind camera looks exactly like a quiet
    house. Here only the cameras the service reports HEALTHY are stamped,
    and only those it actually owns.
    """
    gateway_id, caregiver_id, multi_tenant = await _gateway_from_bearer(
        session, authorization)

    if not body.contract and not body.service and not body.entities:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="a poll must carry a report; see the fleet-service contract",
        )
    if body.contract != FLEET_CONTRACT:
        # A version we cannot read is refused rather than half-parsed,
        # the same rule gpu/identity/contract.py already applies where a
        # mismatch means "no verdict" instead of a silent misparse.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported contract {body.contract!r}, expected "
                   f"{FLEET_CONTRACT!r}",
        )
    if not isinstance(body.service, dict) or not body.service.get("instance_id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report must name the reporting instance",
        )

    # THE SERVICE'S OWN HEARTBEAT. Camera liveness below is per camera
    # and deliberately so, but it can only ever speak for cameras that
    # exist: a service with none — freshly installed, or one whose last
    # camera was deleted — produced no signal at all, and its
    # `gateways.last_seen_at` stayed NULL forever while it polled every
    # few seconds. `/gateway/roster` has always stamped this; the fleet
    # path never did, so the entire bridge fleet was invisible at the
    # service level. On 2026-08-24 the live bridge's row read
    # `last_seen_at = NULL` next to the mini-PC's `12:09`, and
    # soak_check's "a bridge is enrolled multi-tenant" passed on it —
    # a dead bridge and a healthy one are the same row.
    #
    # Placed AFTER the contract and instance checks, so a malformed poll
    # is not liveness: this endpoint's whole design is that reporting is
    # a precondition of being told what to run.
    #
    # Nothing here says the cameras are healthy. It says the SERVICE is
    # answering — which is the question "is the bridge up?", and the one
    # nothing could answer.
    await session.execute(
        text("UPDATE public.gateways SET last_seen_at = now() WHERE id = :id"),
        {"id": gateway_id},
    )

    # Parsed, not passed through. These ids are client data and go into a
    # `uuid[]` bind: one non-uuid among them ("warner_8d2506f65170d9f3",
    # a bridge sending its own vendor key) failed the whole statement, so
    # the poll 500'd and NOT ONE camera on that service got its heartbeat
    # stamped. The camera was publishing fine and read as never-seen —
    # the exact state the staleness sweep exists to catch, produced by
    # the reporting path itself. A malformed id is now that one entity's
    # problem; every well-formed sibling is still stamped.
    healthy: list[str] = []
    for e in body.entities:
        if not (isinstance(e, dict) and e.get("healthy") and e.get("id")):
            continue
        try:
            healthy.append(str(UUID(str(e["id"]))))
        except (ValueError, AttributeError, TypeError):
            log.warning(
                "fleet.poll_entity_id_not_a_uuid gateway=%s id=%r",
                gateway_id, e.get("id"),
            )
    if healthy:
        # Scoped by the ownership predicate as well as by the reported
        # ids: the report is data from a client, so trusting the ids
        # alone would let any enrolled service mark any camera in the
        # fleet as alive.
        await session.execute(
            text("UPDATE public.cameras SET last_heartbeat_at = now() "
                 + (_BRIDGE_CAM_WHERE if multi_tenant else _GATEWAY_CAM_WHERE)
                 + " AND id = ANY(CAST(:ids AS uuid[]))"),
            {"c": caregiver_id, "gw": gateway_id, "ids": healthy},
        )

    rows = (await session.execute(
        text(_cam_select(multi_tenant)),
        {"c": caregiver_id, "gw": gateway_id},
    )).all()
    await session.commit()
    return {"contract": FLEET_CONTRACT, "cameras": _camera_rows(rows)}
