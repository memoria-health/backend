"""Pydantic boundary models for the Contact Support endpoint.

Mirrors `nemara-caregiver-app` Settings → Contact support sheet.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class SupportRequest(BaseModel):
    """Body of POST /v1/support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: Annotated[str, Field(min_length=0, max_length=200)] = ""
    message: Annotated[str, Field(min_length=1, max_length=5000)]


class SupportResponse(BaseModel):
    """Echoed status string so the iOS UI can show a friendly toast."""

    model_config = ConfigDict(frozen=True)

    status: str
