"""Which timezone a home's day is measured in.

Consolidates a fallback that had drifted into three near-copies (the
journal endpoint, the scheduled runner, the per-tick window gate), each
reading `caregivers.timezone` directly and each with its own idea of the
fallback zone — one used UTC, two used America/New_York, so the same
home could roll its day over at two different instants depending on
which code path asked.

The order is patient, then caregiver, then a fixed default:

  - `patients.timezone` — where the HOME is. This is the answer whenever
    it is set, because the day being described is the patient's day.
  - `caregivers.timezone` — the historical behaviour, kept as the
    fallback so homes that predate the patient column keep the exact
    boundaries they have now.
  - `_DEFAULT` — a last resort for a row with neither.

An unparseable zone name degrades to the default rather than raising: a
bad string in one row must not take down a tick or a journal request.
"""

from __future__ import annotations

import logging
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

log = logging.getLogger(__name__)

# Matches the scheduled runner's long-standing fallback. Changing this
# silently moves the day boundary for every home with no timezone set.
_DEFAULT = "America/New_York"


def zone_for(patient_tz: str | None, caregiver_tz: str | None) -> ZoneInfo:
    """Resolve the home's zone from the two stored names."""
    name = patient_tz or caregiver_tz or _DEFAULT
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        log.warning("patient_tz.bad_zone %r — falling back to %s", name, _DEFAULT)
        return ZoneInfo(_DEFAULT)

