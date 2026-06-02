"""Pydantic boundary models for the Send Feedback endpoint.

Mirrors `nemara-caregiver-app` Settings → Send feedback sheet. The
request rejects a completely empty submission: at minimum a star
rating OR one non-blank free-text field must be present.
"""

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeedbackRequest(BaseModel):
    """Body of POST /v1/feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rating: Annotated[int | None, Field(ge=1, le=5)] = None
    whats_working: Annotated[str | None, Field(max_length=5000)] = None
    whats_frustrating: Annotated[str | None, Field(max_length=5000)] = None
    category: Annotated[str | None, Field(max_length=120)] = None

    @model_validator(mode="after")
    def _require_some_content(self) -> "FeedbackRequest":
        """Reject a wholly-empty submission. A rating present OR at least
        one non-blank text field is required so we never store / email a
        blank report."""
        has_text = any(
            (val or "").strip()
            for val in (self.whats_working, self.whats_frustrating, self.category)
        )
        if self.rating is None and not has_text:
            raise ValueError(
                "feedback must include a rating or at least one non-blank field"
            )
        return self


class FeedbackResponse(BaseModel):
    """Echoed status + the persisted row id so the iOS UI can confirm."""

    model_config = ConfigDict(frozen=True)

    status: str
    id: UUID
