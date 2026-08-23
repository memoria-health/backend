"""Reseed live detector rows from the current catalog, per identifier.

    uv run python -m scripts.reseed_prompt meal_eaten [agitation ...]
    railway run python -m scripts.reseed_prompt meal_eaten   # prod

Preset prompts are copied into `detectors.prompt_fragment` at row
creation and read from the ROW at runtime — editing the catalog does
NOT retroactively patch live rows (lesson learned on sleep, 2026-07-29).
For each identifier given, refreshes prompt_fragment / schema_json /
description on every matching row, and backfills cooldown_seconds from
catalog_v2 dims where the row is NULL (rows a caregiver tuned keep
their value).

Replaces the per-detector one-shots (reseed_sleep_prompt,
reseed_fall_prompt, reseed_wandering_prompt, reseed_agitation_prompt,
reseed_env_prompts) — one mechanism, parametrized (2026-08-18).
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection import catalog, catalog_v2


async def main(idents: list[str]) -> None:
    if not idents:
        raise SystemExit("usage: reseed_prompt <identifier> [identifier ...]")
    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        for key in idents:
            entry = catalog.by_key(key)
            if entry is None:
                raise SystemExit(f"no {key!r} entry in the catalog")
            dims = catalog_v2.dims_for(key)

            rows = (
                await session.execute(
                    select(
                        Detector.id, Detector.patient_id,
                        Detector.prompt_fragment,
                    ).where(Detector.identifier == key)
                )
            ).all()
            stale = [
                r for r in rows if r.prompt_fragment != entry.prompt_fragment
            ]
            print(
                f"{key}: {len(rows)} row(s) — "
                f"{len(rows) - len(stale)} current, {len(stale)} stale."
            )
            if not rows:
                continue
            await session.execute(
                update(Detector)
                .where(Detector.identifier == key)
                .values(
                    prompt_fragment=entry.prompt_fragment,
                    schema_json=entry.schema_json,
                    description=entry.description,
                )
            )
            if dims is not None and dims.cooldown_seconds is not None:
                await session.execute(
                    update(Detector)
                    .where(
                        Detector.identifier == key,
                        Detector.cooldown_seconds.is_(None),
                    )
                    .values(cooldown_seconds=dims.cooldown_seconds)
                )
            for r in stale:
                print(f"  updated {r.id} (patient {r.patient_id})")
        await session.commit()
        print("Reseed complete.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
