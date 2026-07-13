"""Scripted VLM seam — the fake that lets tests drive a REAL CameraIngest
tick end-to-end (frames → gates → calls → promotion → event row) without
any network or model. Patch points:

    monkeypatch.setattr(camera_ingest, "run_vlm", FakeVLM(...))
    monkeypatch.setattr(camera_ingest, "gemini_provider", FakeRapidProvider(...))

Every call's kwargs are recorded on `.calls` for assertions.
"""

from __future__ import annotations

from typing import Any

from src.services.detection.providers.types import ProviderResult


class FakeVLM:
    """Stands in for `vlm_routing.run_vlm` (the main call)."""

    def __init__(
        self,
        detector_fields: dict[str, Any] | None = None,
        *,
        description: str = "fake scene",
        people_present: bool = True,
        people_count: int = 1,
        provider: str = "gemini",   # analyses.provider is check-constrained
        usage: dict[str, int] | None = None,
    ) -> None:
        self.detector_fields = dict(detector_fields or {})
        self.description = description
        self.people_present = people_present
        self.people_count = people_count
        self.provider = provider
        self.usage = usage
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> tuple[str, ProviderResult]:
        self.calls.append(kwargs)
        return self.provider, ProviderResult(
            description=self.description,
            detector_fields=dict(self.detector_fields),
            people_present=self.people_present,
            people_count=self.people_count,
            usage=self.usage,
        )


class FakeRapidProvider:
    """Stands in for `gemini_provider` on the RAPID path."""

    def __init__(
        self,
        detector_fields: dict[str, Any] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        self.detector_fields = dict(detector_fields or {})
        self.usage = usage
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> ProviderResult:
        self.calls.append(kwargs)
        return ProviderResult(
            description="rapid", detector_fields=dict(self.detector_fields),
            usage=self.usage,
        )
