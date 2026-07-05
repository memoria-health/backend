"""Per-account VLM routing + provider fallback dispatch.

Two jobs:
  1. `resolve_vlm_provider` — which VLM a caregiver's patients run on,
     cached (~60s TTL) so flipping `caregivers.vlm_provider` takes effect
     within a minute WITHOUT adding a DB query to every detection tick.
  2. `run_vlm` — run the chosen provider with a fallback chain so a single
     provider's outage / rate-limit / unsupported-feature can't blind a
     family. Gemini gets the native-video path; openai / openrouter share
     the frame-sampling path.
"""

import logging
import time
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.services.detection.prompt import DetectorSpec
from src.services.detection.providers.dashscope import dashscope_provider
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.openai import openai_provider
from src.services.detection.providers.openrouter import openrouter_provider
from src.services.detection.providers.types import ProviderResult
from src.settings import Settings

log = logging.getLogger(__name__)

_VALID = ("gemini", "openai", "openrouter", "dashscope")

# Chosen-first, then fallbacks — detection never goes dark if the primary
# errors out.
_CHAINS: dict[str, tuple[str, ...]] = {
    # Sandbox cost guard: gemini falls back to Qwen (cheap), never to the
    # expensive OpenAI, so a gemini 503/outage can't quietly run up an OpenAI bill.
    "gemini": ("gemini", "openrouter"),
    "openai": ("openai", "openrouter"),
    "openrouter": ("openrouter", "gemini"),
    # SANDBOX: Alibaba-direct Qwen3-VL (native video + fps) primary, OpenRouter
    # fallback so detection never goes dark if DashScope errors or the model
    # isn't activated on the account yet.
    "dashscope": ("dashscope", "openrouter"),
}

# caregiver_id -> (provider, expiry_monotonic). Module-level so it survives
# across ticks for the worker process. Single event loop → no lock needed;
# a concurrent double-miss just does two harmless identical reads.
_cache: dict[UUID, tuple[str, float]] = {}
_TTL_SECONDS = 60.0


def _invalidate_cache() -> None:
    """Test/admin hook — drop the cache so the next tick re-reads the DB."""
    _cache.clear()


async def resolve_vlm_provider(
    session: AsyncSession, caregiver_id: UUID, settings: Settings,
) -> str:
    """Cached lookup of a caregiver's VLM choice. NULL / invalid → the global
    `settings.default_vlm_provider`. ~60s TTL keeps the detection hot path
    from doing a per-tick caregivers query."""
    now = time.monotonic()
    hit = _cache.get(caregiver_id)
    if hit is not None and hit[1] > now:
        return hit[0]
    raw = (
        await session.execute(
            select(Caregiver.vlm_provider).where(Caregiver.id == caregiver_id)
        )
    ).scalar_one_or_none()
    choice = raw if raw in _VALID else settings.default_vlm_provider
    if choice not in _VALID:
        choice = "gemini"
    _cache[caregiver_id] = (choice, now + _TTL_SECONDS)
    return choice


async def run_vlm(
    *,
    choice: str,
    mp4: bytes,
    detectors: Sequence[DetectorSpec],
    identity,  # IdentityResult — duck-typed to avoid a reid import cycle
    log_tag: str,
    settings: Settings,
) -> tuple[str, ProviderResult]:
    """Run `choice`'s provider, falling back through `_CHAINS[choice]` on any
    error. Returns (used_provider, result). Raises only if EVERY provider in
    the chain fails (caller treats that like the old both-providers-failed)."""
    chain = _CHAINS.get(choice, _CHAINS["gemini"])
    errors: dict[str, str] = {}
    for prov in chain:
        try:
            if prov == "gemini":
                result = await gemini_provider(
                    mp4=mp4, detectors=detectors, log_tag=log_tag, settings=settings,
                    patient_in_clip=identity.patient_in_clip,
                    patient_bbox=identity.patient_bbox,
                )
            else:
                fn = {
                    "openai": openai_provider,
                    "openrouter": openrouter_provider,
                    "dashscope": dashscope_provider,
                }[prov]
                result = await fn(
                    mp4=mp4, detectors=detectors, log_tag=log_tag, settings=settings,
                    patient_in_clip=identity.patient_in_clip,
                    patient_bbox=identity.patient_bbox,
                    pre_extracted_frames=identity.extracted_frames,
                    per_frame_bboxes=identity.per_frame_bboxes,
                    sample_fps=identity.sample_fps,
                )
            return prov, result
        except Exception as e:  # noqa: BLE001
            errors[prov] = str(e)
            log.warning("%s vlm provider %s failed: %s", log_tag, prov, e)
    raise RuntimeError(f"all VLM providers failed (choice={choice}): {errors}")
