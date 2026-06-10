"""One-shot: update every existing fall detector's prompt_fragment to
the latest version from the detection catalog (`catalog.by_key`).

The fall preset prompt is stored per-detector in the DB at creation
time. Updating the Python constant doesn't retroactively patch live
rows — this script does.

Run against prod:  `railway run python -m scripts.reseed_fall_prompt`
Run against local: `uv run python -m scripts.reseed_fall_prompt`
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection.catalog import by_key

_entry = by_key("fall")
assert _entry is not None, "fall missing from catalog"
_FALL_PROMPT_FRAGMENT = _entry.prompt_fragment


async def main() -> None:
    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        rows = (
            await session.execute(
                select(Detector.id, Detector.patient_id, Detector.prompt_fragment).where(
                    Detector.identifier == "fall"
                )
            )
        ).all()
        already_current = sum(
            1 for r in rows if r.prompt_fragment == _FALL_PROMPT_FRAGMENT
        )
        stale = len(rows) - already_current
        print(
            f"Found {len(rows)} fall detector(s) — "
            f"{already_current} already current, {stale} stale."
        )
        if stale == 0:
            return

        result = await session.execute(
            update(Detector)
            .where(Detector.identifier == "fall")
            .values(prompt_fragment=_FALL_PROMPT_FRAGMENT)
        )
        await session.commit()
        print(f"Updated {result.rowcount} row(s).")


if __name__ == "__main__":
    asyncio.run(main())
