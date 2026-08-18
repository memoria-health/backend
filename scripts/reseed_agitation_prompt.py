"""One-shot: update every existing agitation detector's prompt_fragment
(and schema_json + description) to the current catalog version.

Rewritten 2026-08-18 (was importing the fragment from onboarding, its
pre-productionization home): agitation was recalibrated to EXPLICIT
AGGRESSION only — physical or verbal fighting, angry outbursts — with
pacing/restlessness removed entirely (Aryaman, soak day 1). Preset
prompts are copied into `detectors.prompt_fragment` at row creation and
read from the row at runtime, so catalog edits need this reseed to
reach live rows.

Run against local: `uv run python -m scripts.reseed_agitation_prompt`
Run against prod:  `railway run python -m scripts.reseed_agitation_prompt`
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection import catalog


async def main() -> None:
    entry = catalog.by_key("agitation")
    if entry is None:
        raise SystemExit("no 'agitation' entry in the catalog")

    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        rows = (
            await session.execute(
                select(
                    Detector.id, Detector.patient_id, Detector.prompt_fragment,
                ).where(Detector.identifier == "agitation")
            )
        ).all()
        stale = [r for r in rows if r.prompt_fragment != entry.prompt_fragment]
        print(
            f"Found {len(rows)} agitation detector(s) — "
            f"{len(rows) - len(stale)} already current, {len(stale)} stale."
        )
        if not stale:
            return
        await session.execute(
            update(Detector)
            .where(Detector.identifier == "agitation")
            .values(
                prompt_fragment=entry.prompt_fragment,
                schema_json=entry.schema_json,
                description=entry.description,
            )
        )
        await session.commit()
        for r in stale:
            print(f"  updated detector {r.id} (patient {r.patient_id})")
        print(f"Reseeded {len(stale)} agitation detector(s).")


if __name__ == "__main__":
    asyncio.run(main())
