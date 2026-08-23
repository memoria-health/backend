"""Which zone a home's day is measured in.

The day boundary used to be the CAREGIVER's zone everywhere, which is
wrong the moment they don't live with the patient: a daughter in New
York watching a father in California saw his day roll over at 9pm his
time, and "medication by 5pm" meant 2pm to him.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.schemas.patient import CreatePatientRequest
from src.services import patient_tz


def test_patient_zone_wins_over_the_caregivers() -> None:
    tz = patient_tz.zone_for("America/Los_Angeles", "America/New_York")
    assert str(tz) == "America/Los_Angeles"


def test_falls_back_to_the_caregiver_when_home_has_none() -> None:
    """Every home predating the column has NULL here, so this fallback IS
    the old behaviour — it must not change for them."""
    assert str(patient_tz.zone_for(None, "Asia/Kolkata")) == "Asia/Kolkata"


def test_falls_back_to_the_default_when_neither_is_set() -> None:
    assert str(patient_tz.zone_for(None, None)) == "America/New_York"


def test_an_unresolvable_zone_degrades_instead_of_raising() -> None:
    """A bad string in one row must not take down a tick or a journal
    request — window gating degrades, it never crashes."""
    assert str(patient_tz.zone_for("Mars/Olympus", None)) == "America/New_York"
    assert str(patient_tz.zone_for("", "Europe/London")) == "Europe/London"


def test_the_api_rejects_a_zone_the_server_cannot_resolve() -> None:
    """Storing an unknown name wouldn't error anywhere — it would just
    silently fall back forever, reading as "the setting does nothing"."""
    ok = CreatePatientRequest(
        full_name="A", preferred_name="A", relation="parent",
        timezone="America/Los_Angeles")
    assert ok.timezone == "America/Los_Angeles"
    with pytest.raises(ValueError, match="unknown IANA timezone"):
        CreatePatientRequest(
            full_name="A", preferred_name="A", relation="parent",
            timezone="Mars/Olympus")


def test_omitting_the_zone_is_allowed_and_means_inherit() -> None:
    req = CreatePatientRequest(
        full_name="A", preferred_name="A", relation="parent")
    assert req.timezone is None
    assert str(patient_tz.zone_for(
        SimpleNamespace(timezone=None).timezone, "America/Denver",
    )) == "America/Denver"
