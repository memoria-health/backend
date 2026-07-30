"""One-shot: update every existing sleep detector's prompt_fragment (and
its schema_json) to the current catalog version.

Preset prompts are copied into `detectors.prompt_fragment` when the
detector row is created and read from that row at runtime — editing the
Python constant does NOT retroactively patch live rows. Found the hard
way 2026-07-29: the sleep prompt was rewritten to add the
'awake_in_bed' state (so lying in bed awake, on a phone or talking,
stops counting as sleep), and none of it reached the model because
every live row still carried the old text.

Run against local: `uv run python -m scripts.reseed_sleep_prompt`
Run against prod:  `railway run python -m scripts.reseed_sleep_prompt`
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection import catalog


async def main() -> None:
    entry = catalog.by_key("sleep")
    if entry is None:
        raise SystemExit("no 'sleep' entry in the catalog")
    fragment = entry.prompt_fragment
    schema = entry.schema_json

    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        rows = (
            await session.execute(
                select(
                    Detector.id, Detector.patient_id, Detector.prompt_fragment,
                ).where(Detector.identifier == "sleep")
            )
        ).all()
        stale = [r for r in rows if r.prompt_fragment != fragment]
        print(
            f"Found {len(rows)} sleep detector(s) — "
            f"{len(rows) - len(stale)} already current, {len(stale)} stale."
        )
        if not stale:
            return
        await session.execute(
            update(Detector)
            .where(Detector.identifier == "sleep")
            .values(prompt_fragment=fragment, schema_json=schema)
        )
        await session.commit()
        for r in stale:
            print(f"  updated detector {r.id} (patient {r.patient_id})")
        print(f"Reseeded {len(stale)} sleep detector(s).")


if __name__ == "__main__":
    asyncio.run(main())
