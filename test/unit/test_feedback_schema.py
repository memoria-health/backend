"""Unit tests for the FeedbackRequest / FeedbackResponse boundary models.

Pure Pydantic — no DB, no network. Covers:
- a valid submission (rating + text) parses
- rating-only and text-only submissions are both accepted
- a wholly-empty submission is rejected
- a blank-whitespace-only submission is rejected
- rating out of the 1..5 band is rejected
- unknown fields are forbidden (extra="forbid")
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.schemas.feedback import FeedbackRequest, FeedbackResponse


def test_valid_full_submission_parses() -> None:
    req = FeedbackRequest(
        rating=5,
        whats_working="Alerts are timely.",
        whats_frustrating="Wish I could snooze them.",
        category="alerts",
    )
    assert req.rating == 5
    assert req.whats_working == "Alerts are timely."
    assert req.category == "alerts"


def test_rating_only_is_accepted() -> None:
    req = FeedbackRequest(rating=3)
    assert req.rating == 3
    assert req.whats_working is None


def test_text_only_no_rating_is_accepted() -> None:
    req = FeedbackRequest(whats_frustrating="The app crashed twice today.")
    assert req.rating is None
    assert req.whats_frustrating == "The app crashed twice today."


def test_category_only_is_accepted() -> None:
    req = FeedbackRequest(category="ui")
    assert req.category == "ui"


def test_completely_empty_submission_rejected() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest()


def test_all_fields_none_rejected() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(
            rating=None, whats_working=None, whats_frustrating=None, category=None
        )


def test_blank_whitespace_only_text_rejected() -> None:
    """Whitespace-only text counts as empty: nothing meaningful to store."""
    with pytest.raises(ValidationError):
        FeedbackRequest(whats_working="   ", whats_frustrating="\n\t ", category="  ")


@pytest.mark.parametrize("bad", [0, 6, -1, 10])
def test_rating_out_of_range_rejected(bad: int) -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(rating=bad, whats_working="something")


@pytest.mark.parametrize("ok", [1, 2, 3, 4, 5])
def test_rating_in_range_accepted(ok: int) -> None:
    req = FeedbackRequest(rating=ok)
    assert req.rating == ok


def test_unknown_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        FeedbackRequest(rating=4, sentiment="positive")  # type: ignore[call-arg]


def test_response_model_round_trips() -> None:
    rid = uuid4()
    resp = FeedbackResponse(status="received", id=rid)
    dumped = resp.model_dump()
    assert dumped["status"] == "received"
    assert dumped["id"] == rid
