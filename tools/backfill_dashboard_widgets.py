"""One-off backfill: compile dashboard_widget for existing detectors.

The write-path hooks (detectors API + onboarding seeding) cover every
row created after the dashboard ships; this fills the rows that already
exist. Idempotent — recompiling an unchanged detector yields the same
spec (only compiled_at moves), and rows that compile to None (hidden
internals) are left NULL.

    uv run python tools/backfill_dashboard_widgets.py            # dry run
    uv run python tools/backfill_dashboard_widgets.py --apply
    uv run python tools/backfill_dashboard_widgets.py --apply --recompile-all
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
from src.services.dashboard.widget_compiler import compile_widget


async def run(apply: bool, recompile_all: bool) -> None:
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        stmt = select(Detector)
        if not recompile_all:
            stmt = stmt.where(Detector.dashboard_widget.is_(None))
        rows = (await session.execute(stmt)).scalars().all()

        compiled = skipped = 0
        for det in rows:
            spec = compile_widget(det)
            if spec is None:
                skipped += 1
                continue
            compiled += 1
            print(f"  {det.identifier or 'custom':<20} {det.id}  "
                  f"-> {spec['widget']}")
            if apply:
                det.dashboard_widget = spec
        if apply:
            await session.commit()
        print(
            f"{'applied' if apply else 'DRY RUN'}: {compiled} compiled, "
            f"{skipped} no-widget, of {len(rows)} candidate rows"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="write specs (default: dry run)")
    p.add_argument("--recompile-all", action="store_true",
                   help="recompile rows that already have a spec too")
    a = p.parse_args()
    asyncio.run(run(apply=a.apply, recompile_all=a.recompile_all))
