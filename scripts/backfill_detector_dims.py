"""Backfill v2 dim columns on detector rows that predate the P1 dims
backfill (found on soak day: stove_left_on had NULL columns).

Runtime behavior is ALREADY correct for these rows — `resolve_dims`
falls back to the in-code catalog (`dims_source="catalog"`); this
script just copies the catalog answer into the row columns so the DB
is truthful and introspection (SQL, dashboards) sees real dims.
Customs with NULL columns are left alone — the P5 compiler owns them.

    uv run python scripts/backfill_detector_dims.py --dry-run
    uv run python scripts/backfill_detector_dims.py
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.services.detection import catalog_v2


async def run(dry: bool) -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (await session.execute(
            select(Detector).where(Detector.bucket.is_(None))
        )).scalars().all()
        touched = 0
        for row in rows:
            dims = catalog_v2.dims_for(row.identifier or "")
            if dims is None:
                print(f"  skip {row.identifier or row.name!r} — not in catalog")
                continue
            print(
                f"  {row.identifier}: branch={dims.evidence_branch} "
                f"subject={dims.subject_scope} bucket={dims.bucket} "
                f"duty={dims.duty_cycle} audio={dims.needs_audio}"
            )
            if not dry:
                row.evidence_branch = dims.evidence_branch
                row.subject_scope = dims.subject_scope
                row.bucket = dims.bucket
                row.duty_cycle = dims.duty_cycle
                row.needs_audio = dims.needs_audio
                touched += 1
        if not dry:
            await session.commit()
        print(f"{'DRY RUN: would touch' if dry else 'backfilled'} "
              f"{len(rows) if dry else touched} row(s) with NULL bucket")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args.dry_run))
