"""Provider abstraction — same shape Gemini + OpenAI both satisfy.

Ports `nemara-temp-dashboard/lib/providers/types.ts`. Returns the
fields the state machine + analyses store need:
  - `description`           — LLM 1–2 sentence summary
  - `people_present`/_count — from the always-on `people_present` block
  - `detector_fields`       — flat dict of `<id>_detected`, `<id>_confidence`,
                              `<id>_reasoning`, etc. Anything beyond the three
                              top-level fields ends up here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.services.detection.prompt import DetectorSpec


@dataclass(frozen=True)
class ProviderResult:
    description: str
    people_present: bool
    people_count: int
    detector_fields: dict[str, Any]


class Provider(Protocol):
    """Both `gemini_provider` and `openai_provider` satisfy this."""

    async def __call__(
        self, *, mp4: bytes, detectors: Sequence[DetectorSpec], log_tag: str,
    ) -> ProviderResult: ...
