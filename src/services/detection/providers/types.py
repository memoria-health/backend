"""Provider abstraction — same shape Gemini + OpenAI both satisfy.

Ports `nemara-temp-dashboard/lib/providers/types.ts`. Returns the
fields the analyses store needs:
  - `description`           — LLM 1–2 sentence summary
  - `people_present`/_count — legacy fields, default False/0 since
                              Re-ID body tracking is now the source of
                              truth. Kept on the dataclass so any cached
                              provider response that still carries them
                              flows through without breakage; the state
                              machine + analyses store read the Re-ID
                              values instead.
  - `detector_fields`       — flat dict of `<id>_detected`, `<id>_confidence`,
                              `<id>_reasoning`, etc. Anything beyond the
                              three top-level fields ends up here.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.services.detection.prompt import DetectorSpec


@dataclass(frozen=True)
class ProviderResult:
    description: str
    detector_fields: dict[str, Any]
    people_present: bool = False
    people_count: int = 0
    # Per-call token spend {"input": n, "output": n} when the provider
    # reports it — the cost ledger's raw data (detection v2). None = unknown.
    usage: dict[str, int] | None = None


class Provider(Protocol):
    """Both `gemini_provider` and `openai_provider` satisfy this."""

    async def __call__(
        self, *, mp4: bytes, detectors: Sequence[DetectorSpec], log_tag: str,
    ) -> ProviderResult: ...
