"""Fold a night/day into `sleep_nights` / `activity_days`, by hand.

The backend now does this on its own every half hour
(`src/services/summaries/nightly.py`, started in `main.py`), so this is
for backfilling a specific date or checking a fold without waiting.

    uv run python tools/run_daily_summaries.py               # last night
    uv run python tools/run_daily_summaries.py --date 2026-07-28

This file used to CONTAIN the fold — roughly 200 lines of SQL, timezone
conversions and summarizer glue that existed nowhere else. That is why
nothing summarised a night unless somebody remembered to type this, and
why seven nights between 2026-08-18 and 2026-08-25 were collected and
never folded. The logic moved into the service; what is left here is the
argument parsing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.session import get_sessionmaker
from src.services.summaries.nightly import summarize_date


async def _run(night_date: date) -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        wrote = await summarize_date(session, night_date)
        await session.commit()
    print(f"{night_date}: {wrote} sleep row(s) written "
          f"(per-patient detail is on the logger)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="night/day date YYYY-MM-DD (default: yesterday)")
    a = ap.parse_args()
    asyncio.run(_run(
        date.fromisoformat(a.date) if a.date
        else date.today() - timedelta(days=1)
    ))


if __name__ == "__main__":
    main()
