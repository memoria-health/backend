"""One-shot: bring every existing environment detector row up to the
2026-08-17 env taxonomy (Aryaman): glass/water are ACTION detectors
(the prompt fires on the event happening in-clip, never on the
lingering puddle/shards) and fire stays state-phrased on purpose
(re-alerts each cooldown while still burning).

Preset prompts are copied into `detectors.prompt_fragment` at row
creation and read from the row at runtime — editing the catalog does
NOT retroactively patch live rows (same lesson as reseed_sleep_prompt).
This refreshes prompt_fragment / schema_json / description from the
catalog and backfills cooldown_seconds from catalog_v2 (1800s glass and
water as the puddle-mistaken-for-fresh-spill backstop, 600s fire) — but
only where cooldown is NULL, so rows a caregiver already tuned keep
their value.

Run against local: `uv run python -m scripts.reseed_env_prompts`
Run against prod:  `railway run python -m scripts.reseed_env_prompts`
"""

from __future__ import annotations

import asyncio

from sqlalchemy import select, update

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection import catalog, catalog_v2

ENV_KEYS = ("env_hazard_glass", "env_hazard_water", "fire_smoke")


async def main() -> None:
    SessionMaker = get_sessionmaker()
    async with SessionMaker() as session:
        for key in ENV_KEYS:
            entry = catalog.by_key(key)
            dims = catalog_v2.dims_for(key)
            if entry is None or dims is None:
                raise SystemExit(f"no {key!r} entry in the catalog")

            rows = (
                await session.execute(
                    select(
                        Detector.id, Detector.patient_id,
                        Detector.prompt_fragment, Detector.cooldown_seconds,
                    ).where(Detector.identifier == key)
                )
            ).all()
            stale = [r for r in rows if r.prompt_fragment != entry.prompt_fragment]
            print(
                f"{key}: {len(rows)} row(s) — "
                f"{len(rows) - len(stale)} prompt-current, {len(stale)} stale."
            )
            if rows:
                await session.execute(
                    update(Detector)
                    .where(Detector.identifier == key)
                    .values(
                        prompt_fragment=entry.prompt_fragment,
                        schema_json=entry.schema_json,
                        description=entry.description,
                    )
                )
                await session.execute(
                    update(Detector)
                    .where(
                        Detector.identifier == key,
                        Detector.cooldown_seconds.is_(None),
                    )
                    .values(cooldown_seconds=dims.cooldown_seconds)
                )
            for r in stale:
                print(f"  updated detector {r.id} (patient {r.patient_id})")
        await session.commit()
        print("Env detector reseed complete.")


if __name__ == "__main__":
    asyncio.run(main())
