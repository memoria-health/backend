"""Unit tests for services/region.py — the data-residency storage router.

Pure-function tests (no DB, no network): US must be unchanged (backward
compatible), IN must be isolated, and an unconfigured India bucket must fall
back to the default rather than fail.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.services import region


@dataclass
class _FakeSettings:
    b2_bucket: str = "memoria-data"
    b2_bucket_in: str = ""


def test_normalize_defaults_unknown_to_us() -> None:
    assert region.normalize("US") == "US"
    assert region.normalize("IN") == "IN"
    assert region.normalize(None) == "US"
    assert region.normalize("ZZ") == "US"
    assert region.normalize("") == "US"


def test_storage_key_us_is_unchanged() -> None:
    # US keeps its exact existing key (no prefix, no migration, no collision).
    assert region.storage_key("US", "abc.jpg") == "abc.jpg"
    assert region.storage_key(None, "events/e1.mp4") == "events/e1.mp4"


def test_storage_key_in_is_prefixed() -> None:
    assert region.storage_key("IN", "abc.jpg") == "IN/abc.jpg"
    assert region.storage_key("IN", "patient-faces/p/e.jpg") == "IN/patient-faces/p/e.jpg"
    # leading slash on the base key is normalized away
    assert region.storage_key("IN", "/abc.jpg") == "IN/abc.jpg"


def test_storage_key_unknown_region_treated_as_us() -> None:
    assert region.storage_key("ZZ", "abc.jpg") == "abc.jpg"


def test_b2_bucket_us_uses_default() -> None:
    s = _FakeSettings(b2_bucket="memoria-data", b2_bucket_in="memoria-data-in")
    assert region.b2_bucket_for("US", s) == "memoria-data"
    assert region.b2_bucket_for(None, s) == "memoria-data"


def test_b2_bucket_in_uses_india_bucket_when_configured() -> None:
    s = _FakeSettings(b2_bucket="memoria-data", b2_bucket_in="memoria-data-in")
    assert region.b2_bucket_for("IN", s) == "memoria-data-in"


def test_b2_bucket_in_falls_back_to_default_when_unconfigured() -> None:
    # India infra not stood up yet: don't fail collection, fall back (warned).
    s = _FakeSettings(b2_bucket="memoria-data", b2_bucket_in="")
    assert region.b2_bucket_for("IN", s) == "memoria-data"
