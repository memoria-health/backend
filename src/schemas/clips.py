"""Pydantic boundary models for the clip ingest endpoint.

`POST /v1/clips/ingest` takes `multipart/form-data` (the MP4 plus
form fields), so the request body is a FastAPI `UploadFile` + form
parameters rather than a JSON body — there's no `Request` model
here, just the `IngestResponse` shape.
"""

from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class IngestResponse(BaseModel):
    """Wire shape returned to the wall phone after a clip is processed.

    Mirrors the dashboard's response so the wall phone code in
    `nemara-camera-app` can switch backends with only a URL change.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID                          # analyses.id
    # `skipped` indicates the closed-state short-circuit fired (no LLM
    # was called this tick).
    provider: Annotated[str, Field(pattern=r"^(gemini|openai|skipped)$")]
    analysis: str                     # the LLM description
    people_present: bool
    people_count: int = Field(ge=0)
    detector_fields: dict[str, Any]
    state: Annotated[str, Field(pattern=r"^(open|closed)$")]
    no_people_count: int = Field(ge=0)
    close_threshold: int = Field(ge=1)
