"""One-shot: update every existing wandering detector's prompt_fragment
to the latest version from `services.onboarding._WANDERING_PROMPT_FRAGMENT`.

The wandering preset prompt is stored per-detector in the DB at creation
time. Updating the Python constant doesn't retroactively patch live
rows — this script does.

Run against prod: `railway run python -m scripts.reseed_wandering_prompt`
Run against local: `uv run python -m scripts.reseed_wandering_prompt`
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.onboarding import _WANDERING_PROMPT_FRAGMENT


async def main() -> None:
    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        rows = (
            await session.execute(
                select(Detector.id, Detector.patient_id, Detector.prompt_fragment).where(
                    Detector.identifier == "wandering"
                )
            )
        ).all()
        already_current = sum(
            1 for r in rows if r.prompt_fragment == _WANDERING_PROMPT_FRAGMENT
        )
        stale = len(rows) - already_current
        print(
            f"Found {len(rows)} wandering detector(s) — "
            f"{already_current} already current, {stale} stale."
        )
        if stale == 0:
            return

        result = await session.execute(
            update(Detector)
            .where(Detector.identifier == "wandering")
            .values(prompt_fragment=_WANDERING_PROMPT_FRAGMENT)
        )
        await session.commit()
        print(f"Updated {result.rowcount} row(s).")


if __name__ == "__main__":
    asyncio.run(main())
