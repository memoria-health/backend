#!/usr/bin/env python3
"""LiveKit room state inspector + zombie/idle flagger.

Run any time to snapshot all rooms on the configured LiveKit
deployment, print participants per room, and flag two production
hygiene problems:

  - ZOMBIE: multiple `detector_worker_<prefix>_*` participants in
    the same `patient_*` room. Means a previous backend process
    didn't clean up (SIGKILL, crash) and the new process spawned its
    own worker alongside the ghost. The new worker is fine; the
    ghost wastes a LiveKit identity slot until LiveKit's server-side
    timeout reaps it (~5 min).

  - IDLE: a `patient_*` room with at least one `detector_worker_*`
    participant but zero `cam_*` publishers. Our reconciliation
    joined a room with no actual camera publishing — usually OK if
    short-lived (the wall phone is briefly offline) but a persistent
    idle room across multiple audits means the idle-teardown path
    isn't firing.

Reads LiveKit creds from the same .env the backend uses
(NEMARA_LIVEKIT_URL / KEY / SECRET).

Usage:
    cd /path/to/nemara-backend-pi-ingest
    uv run python tools/lk_room_audit.py

Exit code 0 always; this is a diagnostic, not a CI gate.
"""

from __future__ import annotations

import asyncio
import collections
import os
import sys
from pathlib import Path

# Make `src.*` imports work when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from livekit import api as lkapi

from src.settings import get_settings


# ANSI color codes for terminal output. Defined as constants so the
# pretty-printer below stays readable.
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_GREEN = "\x1b[32m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


def _classify(identity: str) -> str:
    """Bucket a participant identity into one of the four shapes the
    nemara backend emits / accepts. Returns the bucket name as a
    short string for the printer.
    """
    if identity.startswith("cam_"):
        return "publisher"
    if identity.startswith("detector_worker_"):
        return "worker"
    if identity.startswith("caregiver_"):
        return "caregiver"
    return "other"


async def main() -> None:
    settings = get_settings()
    if not (settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret):
        print(
            f"{_RED}ERROR{_RESET}: LIVEKIT_URL/KEY/SECRET not set — "
            "audit needs them.",
            file=sys.stderr,
        )
        sys.exit(2)

    lk = lkapi.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )

    zombie_rooms: list[str] = []
    idle_rooms: list[str] = []
    healthy_rooms: list[str] = []

    try:
        rooms_resp = await lk.room.list_rooms(lkapi.ListRoomsRequest())
        print(f"{_BOLD}LiveKit deployment:{_RESET} {settings.livekit_url}")
        print(f"{_BOLD}Rooms found:{_RESET} {len(rooms_resp.rooms)}")
        print()

        for room in rooms_resp.rooms:
            participants_resp = await lk.room.list_participants(
                lkapi.ListParticipantsRequest(room=room.name),
            )
            buckets: dict[str, list[str]] = collections.defaultdict(list)
            for p in participants_resp.participants:
                buckets[_classify(p.identity)].append(p.identity)

            n_workers = len(buckets["worker"])
            n_publishers = len(buckets["publisher"])
            is_patient_room = room.name.startswith("patient_")
            flags: list[str] = []
            if is_patient_room and n_workers > 1:
                flags.append(f"{_RED}ZOMBIE x{n_workers}{_RESET}")
                zombie_rooms.append(room.name)
            if is_patient_room and n_workers >= 1 and n_publishers == 0:
                flags.append(f"{_YELLOW}IDLE{_RESET}")
                idle_rooms.append(room.name)
            if is_patient_room and not flags:
                healthy_rooms.append(room.name)
            flag_str = "  ".join(flags) if flags else f"{_GREEN}healthy{_RESET}"

            print(f"{_BOLD}{room.name}{_RESET}  {flag_str}")
            print(
                f"  {_DIM}participants={room.num_participants} "
                f"publishers={room.num_publishers}{_RESET}"
            )
            for bucket in ("publisher", "worker", "caregiver", "other"):
                for ident in buckets[bucket]:
                    label = bucket.upper() if bucket == "publisher" else bucket
                    print(f"  - [{label}] {ident}")
            print()

        print(f"{_BOLD}Summary:{_RESET}")
        print(f"  {_GREEN}healthy:{_RESET} {len(healthy_rooms)}")
        print(f"  {_YELLOW}idle:{_RESET}    {len(idle_rooms)}")
        print(f"  {_RED}zombies:{_RESET} {len(zombie_rooms)}")
        if zombie_rooms:
            print(
                f"\n{_RED}Action:{_RESET} the zombie rooms above have more "
                "than one detector_worker participant. Newer code with the "
                "startup zombie-reaper should kick the prior-process bots "
                "automatically on next backend restart."
            )
        if idle_rooms:
            print(
                f"\n{_YELLOW}Action:{_RESET} the idle rooms above have a "
                "detector_worker but no publisher. If they persist across "
                "two audits, the idle-teardown path isn't firing."
            )
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
