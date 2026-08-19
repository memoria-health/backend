"""Scenario CLI — drives the live local nemara-backend stack so the
iOS app on a simulator can be verified end-to-end.

Each subcommand performs one action (insert event, flip a detector,
add a patient, etc.) against the local Postgres + Supabase stack the
backend already uses. Supabase Realtime broadcasts the change; the iOS
app (talking to the same stack via the bundled scheme env vars)
reacts immediately. Names like `jane` and `maggie` are resolved to
UUIDs via a case-insensitive substring match on `caregivers.display_name`
/ `patients.preferred_name`.

Run:
    make db-reset
    make dev                     # backend on :8002
    open NemaraCaregiver.xcodeproj from nemara-caregiver-app, ⌘R
    uv run e2e <command>          # fire scenarios

Help:
    uv run e2e --help
    uv run e2e fire-event --help
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID, uuid4

import typer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scripts import seed
from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker

app = typer.Typer(
    add_completion=False,
    help="Scenario CLI for end-to-end verification of nemara-backend + nemara-caregiver-app.",
    no_args_is_help=True,
)


# ─── helpers ─────────────────────────────────────────────────────


def _run(coro):
    """Run an async function from the sync Typer entrypoint."""
    return asyncio.run(coro)


async def _find_caregiver(session: AsyncSession, name: str) -> Caregiver:
    needle = name.lower()
    result = await session.execute(select(Caregiver))
    rows = result.scalars().all()
    matches = [c for c in rows if needle in c.display_name.lower()]
    if not matches:
        raise typer.BadParameter(f"No caregiver matches '{name}'.")
    if len(matches) > 1:
        names = ", ".join(c.display_name for c in matches)
        raise typer.BadParameter(f"'{name}' is ambiguous: {names}.")
    return matches[0]


async def _find_patient(session: AsyncSession, name: str) -> Patient:
    needle = name.lower()
    result = await session.execute(select(Patient))
    rows = result.scalars().all()
    matches = [
        p for p in rows
        if needle in p.preferred_name.lower() or needle in p.full_name.lower()
    ]
    if not matches:
        raise typer.BadParameter(f"No patient matches '{name}'.")
    if len(matches) > 1:
        names = ", ".join(p.preferred_name for p in matches)
        raise typer.BadParameter(f"'{name}' is ambiguous: {names}.")
    return matches[0]


async def _find_detector(
    session: AsyncSession, patient_id: UUID, name: str,
) -> Detector:
    needle = name.lower()
    result = await session.execute(
        select(Detector).where(Detector.patient_id == patient_id)
    )
    rows = result.scalars().all()
    matches = [d for d in rows if needle in d.name.lower()]
    if not matches:
        raise typer.BadParameter(f"No detector matches '{name}' for this patient.")
    if len(matches) > 1:
        names = ", ".join(d.name for d in matches)
        raise typer.BadParameter(f"'{name}' is ambiguous: {names}.")
    return matches[0]


# ─── commands ────────────────────────────────────────────────────


@app.command()
def reset() -> None:
    """Wipe the demo caregiver's patients and re-seed Jane + Maggie."""
    typer.echo("[e2e] reseeding via scripts.seed.main()...")
    _run(seed.main())
    typer.secho("[e2e] reset complete.", fg=typer.colors.GREEN)


@app.command("fire-event")
def fire_event(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
    kind: Annotated[
        str, typer.Option(help="Event kind: fall, agitation, routine, meds, door."),
    ] = "fall",
    location: Annotated[str, typer.Option(help="Room.")] = "Living room",
    title: Annotated[str | None, typer.Option(help="Override the title.")] = None,
) -> None:
    """Insert one event row. The iOS Events tab shows a new card."""
    presets = {
        "fall": ("safety", "preset", "figure.fall", "Possible fall", True,
                 "A sharp downward motion followed by stillness. No spoken response."),
        "agitation": ("safety", "preset", "figure.walk.motion", "Agitation episode", True,
                      "Pacing detected for several minutes; voice raised."),
        "routine": ("routine", "preset", "leaf", "Routine activity", False,
                    "Patient appears to be going about a normal routine."),
        "meds": ("routine", "preset", "pills", "Medication", False,
                 "Pills taken with water at the kitchen counter."),
        "door": ("safety", "preset", "door.left.hand.open", "Front door opened", True,
                 "Front door opened. Brief visit, door closed soon after."),
    }
    if kind not in presets:
        raise typer.BadParameter(f"Unknown kind '{kind}'. One of: {', '.join(presets)}.")
    category, source, icon, default_title, has_clip, narrative = presets[kind]
    final_title = title or default_title

    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, patient)
            now = datetime.now(timezone.utc)
            event = Event(
                id=uuid4(),
                patient_id=p.id,
                detector_id=None,
                camera_id=None,
                category=category,
                source=source,
                icon_name=icon,
                title=final_title,
                location=location,
                occurred_at=now,
                has_clip=has_clip,
                narrative=narrative,
                opened=False,
                reviewed=False,
                review_choice=None,
                review_notes=None,
                reviewed_at=None,
                created_at=now,
            )
            session.add(event)
            await session.commit()
            typer.echo(f"[e2e] fired {kind} event for {p.preferred_name}: id={event.id}")

    _run(go())


@app.command("mark-event-reviewed")
def mark_event_reviewed(
    event_id: Annotated[UUID, typer.Option("--event", help="Event UUID.")],
    review_as: Annotated[
        str, typer.Option("--as", help="correct | wrong | unsure."),
    ] = "correct",
    notes: Annotated[str | None, typer.Option(help="Optional reviewer note.")] = None,
) -> None:
    """Server-side review of an event. The iOS card flips to REVIEWED."""
    if review_as not in {"correct", "wrong", "unsure"}:
        raise typer.BadParameter("--as must be correct, wrong, or unsure.")

    async def go() -> None:
        async with get_sessionmaker()() as session:
            event = await session.get(Event, event_id)
            if event is None:
                raise typer.BadParameter(f"Event {event_id} not found.")
            now = datetime.now(timezone.utc)
            event.opened = True
            event.reviewed = True
            event.review_choice = review_as
            event.review_notes = notes
            event.reviewed_at = now
            await session.commit()
            typer.echo(f"[e2e] event {event_id} marked reviewed={review_as}")

    _run(go())


@app.command("update-patient")
def update_patient(
    name: Annotated[str, typer.Option(help="Patient name (substring).")],
    state: Annotated[
        str | None, typer.Option(help="calm | critical | offline."),
    ] = None,
    status_line: Annotated[str | None, typer.Option(help="Roster status line.")] = None,
    battery_percent: Annotated[int | None, typer.Option(help="0–100.")] = None,
) -> None:
    """Mutate the server-managed fields wall phones own. Drives the
    roster row + status line on the iOS Home tab."""
    if state is not None and state not in {"calm", "critical", "offline"}:
        raise typer.BadParameter("--state must be calm, critical, or offline.")
    if battery_percent is not None and not 0 <= battery_percent <= 100:
        raise typer.BadParameter("--battery-percent must be 0–100.")

    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, name)
            if state is not None: p.state = state
            if status_line is not None: p.status_line = status_line
            if battery_percent is not None: p.battery_percent = battery_percent
            p.updated_at = datetime.now(timezone.utc)
            await session.commit()
            typer.echo(f"[e2e] patient {p.preferred_name} updated")

    _run(go())


@app.command("toggle-detector")
def toggle_detector(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
    detector_name: Annotated[
        str, typer.Option("--name", help="Detector name (substring)."),
    ],
    on: Annotated[bool, typer.Option("--on/--off", help="Set is_on.")] = True,
) -> None:
    """Flip is_on on a detector. iOS Detector row reflects the toggle."""
    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, patient)
            d = await _find_detector(session, p.id, detector_name)
            d.is_on = on
            d.updated_at = datetime.now(timezone.utc)
            await session.commit()
            typer.echo(f"[e2e] detector '{d.name}' is_on={on}")

    _run(go())


@app.command("set-sensitivity")
def set_sensitivity(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
    detector_name: Annotated[
        str, typer.Option("--name", help="Detector name (substring)."),
    ],
    level: Annotated[
        str, typer.Option(help="low | standard | high | off."),
    ] = "standard",
) -> None:
    """Tune a detector's sensitivity. iOS Detector row updates."""
    if level not in {"low", "standard", "high", "off"}:
        raise typer.BadParameter("--level must be low, standard, high, or off.")

    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, patient)
            d = await _find_detector(session, p.id, detector_name)
            d.sensitivity = level
            d.updated_at = datetime.now(timezone.utc)
            await session.commit()
            typer.echo(f"[e2e] detector '{d.name}' sensitivity={level}")

    _run(go())


@app.command("pause-cameras")
def pause_cameras(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
) -> None:
    """Set cameras_paused=true on a patient. iOS shows the PAUSED chip."""
    _run(_set_paused(patient, True))


@app.command("unpause-cameras")
def unpause_cameras(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
) -> None:
    """Set cameras_paused=false. iOS clears the PAUSED chip."""
    _run(_set_paused(patient, False))


async def _set_paused(patient: str, paused: bool) -> None:
    async with get_sessionmaker()() as session:
        p = await _find_patient(session, patient)
        p.cameras_paused = paused
        p.updated_at = datetime.now(timezone.utc)
        await session.commit()
        typer.echo(f"[e2e] {p.preferred_name} cameras_paused={paused}")


@app.command("add-patient")
def add_patient(
    caregiver: Annotated[str, typer.Option(help="Caregiver name (substring).")],
    name: Annotated[str, typer.Option(help="Full name.")],
    preferred_name: Annotated[
        str | None, typer.Option(help="Preferred name (defaults to first word)."),
    ] = None,
    relation: Annotated[
        str, typer.Option(help="parent | spouse | grandparent | sibling | client | other."),
    ] = "other",
) -> None:
    """Insert a new patient for the named caregiver. iOS Roster appends
    the new card via Realtime."""
    if relation not in {"parent", "spouse", "grandparent", "sibling", "client", "other"}:
        raise typer.BadParameter("--relation must be one of the supported enum values.")

    async def go() -> None:
        async with get_sessionmaker()() as session:
            cg = await _find_caregiver(session, caregiver)
            pref = preferred_name or name.split()[0]
            now = datetime.now(timezone.utc)
            patient = Patient(
                id=uuid4(),
                caregiver_id=cg.id,
                full_name=name,
                preferred_name=pref,
                date_of_birth=None,
                gender=None,
                relation=relation,
                home_address=None,
                state="calm",
                status_line="All quiet",
                last_event_at=None,
                battery_percent=None,
                cameras_paused=False,
                created_at=now,
                updated_at=now,
            )
            session.add(patient)
            await session.flush()
            # Seeds only the hidden setup self-test — visible detectors are
            # picked from the library, so this patient starts with none.
            from src.services import onboarding
            await onboarding.seed_detectors_for_patient(session, patient.id)
            await session.commit()
            typer.echo(f"[e2e] added patient {pref} ({patient.id}) for {cg.display_name}")

    _run(go())


@app.command("remove-patient")
def remove_patient(
    name: Annotated[str, typer.Option(help="Patient name (substring).")],
) -> None:
    """Delete a patient. Cascades to cameras / detectors / events. iOS
    Roster drops the card via Realtime."""
    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, name)
            pid = p.id
            pref = p.preferred_name
            await session.delete(p)
            await session.commit()
            typer.echo(f"[e2e] removed patient {pref} ({pid})")

    _run(go())


@app.command("list")
def list_state(
    caregiver: Annotated[
        str | None,
        typer.Option(help="Caregiver name (substring). Defaults to all."),
    ] = None,
) -> None:
    """Dump current DB state for one (or every) caregiver as JSON."""
    async def go() -> None:
        async with get_sessionmaker()() as session:
            cg_rows = (await session.execute(select(Caregiver))).scalars().all()
            if caregiver:
                needle = caregiver.lower()
                cg_rows = [c for c in cg_rows if needle in c.display_name.lower()]
                if not cg_rows:
                    raise typer.BadParameter(f"No caregiver matches '{caregiver}'.")

            out: list[dict] = []
            for cg in cg_rows:
                patients = (
                    await session.execute(
                        select(Patient).where(Patient.caregiver_id == cg.id)
                    )
                ).scalars().all()
                p_blob: list[dict] = []
                for p in patients:
                    cams = (
                        await session.execute(
                            select(Camera).where(Camera.patient_id == p.id)
                        )
                    ).scalars().all()
                    dets = (
                        await session.execute(
                            select(Detector).where(Detector.patient_id == p.id)
                        )
                    ).scalars().all()
                    events = (
                        await session.execute(
                            select(Event).where(Event.patient_id == p.id)
                        )
                    ).scalars().all()
                    p_blob.append({
                        "id": str(p.id),
                        "preferred_name": p.preferred_name,
                        "state": p.state,
                        "status_line": p.status_line,
                        "cameras_paused": p.cameras_paused,
                        "cameras": [
                            {"id": str(c.id), "room": c.room, "online": c.online}
                            for c in cams
                        ],
                        "detectors": [
                            {
                                "id": str(d.id), "name": d.name,
                                "is_on": d.is_on, "sensitivity": d.sensitivity,
                            }
                            for d in dets
                        ],
                        "events": [
                            {
                                "id": str(e.id), "title": e.title,
                                "reviewed": e.reviewed,
                                "review_choice": e.review_choice,
                            }
                            for e in events
                        ],
                    })
                out.append({
                    "id": str(cg.id),
                    "display_name": cg.display_name,
                    "email": cg.email,
                    "patients": p_blob,
                })

            typer.echo(json.dumps(out, indent=2, default=str))

    _run(go())


# ─── Phase 0: pairing + clip ingest scenarios ─────────────────────


@app.command("mint-pairing-code")
def mint_pairing_code(
    patient: Annotated[str, typer.Option(help="Patient name (substring).")],
    room: Annotated[str, typer.Option(help="Room label, e.g. 'Kitchen'.")] = "Kitchen",
    icon: Annotated[str, typer.Option(help="SF Symbol or icon name.")] = "tv",
) -> None:
    """Insert a `cameras` row with a 6-digit pairing code, valid for 10
    minutes. Prints the code so the wall phone can pair against it.

    Skips the HTTP layer — writes directly through the same ORM the
    `/v1/patients/{id}/cameras/pairing-code` route uses. Useful when
    you want a code without standing up the caregiver app first.
    """
    import secrets
    from datetime import datetime, timedelta, timezone
    from src.db.camera import Camera
    from sqlalchemy import func, select

    async def go() -> None:
        async with get_sessionmaker()() as session:
            p = await _find_patient(session, patient)
            now = datetime.now(timezone.utc)
            max_display = (await session.execute(
                select(func.coalesce(func.max(Camera.display_id), 0)).where(
                    Camera.patient_id == p.id
                )
            )).scalar_one()
            code = str(secrets.randbelow(900_000) + 100_000)
            cam_id = uuid4()
            short = cam_id.hex[:8]
            cam = Camera(
                id=cam_id,
                patient_id=p.id,
                display_id=max_display + 1,
                room=room,
                icon_name=icon,
                online=False,
                created_at=now,
                pairing_code=code,
                pairing_expires_at=now + timedelta(minutes=10),
                api_key_hash=None,
                livekit_identity=f"cam_{short}",
                livekit_room=f"patient_{p.id}",
                state="open",
                no_people_count=0,
                state_changed_at=now,
                close_threshold=6,
                battery_percent=None,
            )
            session.add(cam)
            await session.commit()
            typer.echo(json.dumps({
                "pairing_code": code,
                "expires_at": (now + timedelta(minutes=10)).isoformat(),
                "camera_id": str(cam_id),
                "patient_id": str(p.id),
                "room": room,
            }, indent=2))

    _run(go())


@app.command("simulate-pair")
def simulate_pair(
    pairing_code: Annotated[str, typer.Option("--pairing-code", help="6-digit code.")],
    backend: Annotated[
        str, typer.Option("--backend", help="Backend base URL."),
    ] = "http://127.0.0.1:8002",
) -> None:
    """Call POST /v1/cameras/pair as a wall phone would. Useful for
    smoke-testing the pairing exchange without standing up the iOS
    camera app.

    Requires `make dev` running. Prints the api_key so subsequent
    commands (like `simulate-clip`) can authenticate.
    """
    import httpx

    async def go() -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{backend.rstrip('/')}/v1/cameras/pair",
                json={"pairing_code": pairing_code},
            )
            if r.status_code != 200:
                typer.secho(
                    f"pair failed: {r.status_code} {r.text}",
                    fg=typer.colors.RED, err=True,
                )
                raise typer.Exit(code=1)
            typer.echo(json.dumps(r.json(), indent=2))

    _run(go())


if __name__ == "__main__":
    app()
