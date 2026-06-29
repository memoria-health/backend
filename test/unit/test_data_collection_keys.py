"""Task 3 — B2 object-key construction for data-collection clips (pure)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from src.services.data_collection.keys import build_clip_key, slugify

_CG = UUID("11111111-1111-1111-1111-111111111111")
_PT = UUID("22222222-2222-2222-2222-222222222222")


def test_slugify_basic() -> None:
    assert slugify("Living Room") == "living-room"


def test_slugify_collapses_and_trims_punctuation() -> None:
    assert slugify("  Mom's Bedroom!! ") == "mom-s-bedroom"


def test_slugify_empty_or_all_symbols_uses_fallback() -> None:
    assert slugify("") == "room"
    assert slugify("***") == "room"


def test_build_clip_key_layout_and_has_no_colons() -> None:
    start = datetime(2026, 6, 24, 9, 5, 3, tzinfo=timezone.utc)
    key = build_clip_key(
        caregiver_id=_CG, patient_id=_PT, room="Living Room", session_start=start
    )
    assert key == f"data/v1/{_CG}/{_PT}/2026-06-24/090503Z__living-room.mp4"
    # Colons are legal in S3 keys but break downloaded-filename handling.
    assert ":" not in key


def test_build_clip_key_converts_to_utc() -> None:
    # 09:05:03 at +05:30 (India) == 03:35:03 UTC.
    tz = timezone(timedelta(hours=5, minutes=30))
    start = datetime(2026, 6, 24, 9, 5, 3, tzinfo=tz)
    key = build_clip_key(
        caregiver_id=_CG, patient_id=_PT, room="Bedroom", session_start=start
    )
    assert "/2026-06-24/033503Z__bedroom.mp4" in key


def test_build_clip_key_treats_naive_datetime_as_utc() -> None:
    start = datetime(2026, 6, 24, 9, 5, 3)  # naive
    key = build_clip_key(
        caregiver_id=_CG, patient_id=_PT, room="Den", session_start=start
    )
    assert "/2026-06-24/090503Z__den.mp4" in key
