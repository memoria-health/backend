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
from dataclasses import replace as _dc_replace
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
    # Gemini path: primary model (gemini_model, e.g. flash-lite) → fallback model
    # (gemini_fallback_model, e.g. flash) on the SAME native-video path → DashScope
    # (Alibaba-direct Qwen3-VL, ALSO native video) → OpenAI (frame-based)
    # only as a last-resort net so detection never goes fully dark. Immediate
    # failover on first error, no retry. `gemini_fallback` is a chain-internal
    # token (NOT a user choice); run_vlm reports it as "gemini". A keyless
    # provider (e.g. DashScope) raises immediately, is caught, and the chain
    # moves on — so keyless environments still fail through to openai.
    #
    # OpenRouter was retired as the net (Aryaman 2026-08-11): its key was
    # part of the f1c0564 leak, the routed Qwen was measured poor out of
    # the box, and OpenAI already earns its keep here (compile path +
    # opt-in accounts). It survives only as an explicit account CHOICE
    # below. Two honest limits of the remaining net: gemini and
    # gemini_fallback share ONE credential/account, so those two tiers
    # are not independent against key/quota/billing failures — and the
    # OpenAI tier is frame-sampling, so a failover tick loses native
    # video pacing and audio (the executor already marks
    # `sched_degraded_audio` when that happens).
    "gemini": ("gemini", "gemini_fallback", "dashscope", "openai"),
    "openai": ("openai", "gemini"),
    "openrouter": ("openrouter", "dashscope", "gemini"),
    # SANDBOX: Alibaba-direct Qwen3-VL (native video + fps) primary, OpenAI
    # fallback so detection never goes dark if DashScope errors or the model
    # isn't activated on the account yet.
    "dashscope": ("dashscope", "openai"),
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
            if prov in ("gemini", "gemini_fallback"):
                # Two gemini tiers on the SAME native-video path: primary model,
                # then the fallback model. Reported as "gemini" either way — the
                # analyses.provider CHECK constraint has no separate value and
                # both ARE the gemini provider.
                gmodel = (
                    settings.gemini_fallback_model
                    if prov == "gemini_fallback"
                    else settings.gemini_model
                )
                result = await gemini_provider(
                    mp4=mp4, detectors=detectors, log_tag=log_tag, settings=settings,
                    patient_in_clip=identity.patient_in_clip,
                    model=gmodel,
                )
                return "gemini", _dc_replace(result, model=gmodel)
            fn = {
                "openai": openai_provider,
                "openrouter": openrouter_provider,
                "dashscope": dashscope_provider,
            }[prov]
            # Frame reuse only when attribution ran: with always-on
            # tracking (v2 P2) extracted_frames populate on every
            # successful tick, but Re-ID samples at 5 fps vs these
            # providers' configured 4 — reusing for the attribution-less
            # population would silently send ~25% more image tokens.
            # Attribution-less ticks keep their own extraction at the
            # provider's own fps — the pre-P2 cost profile.
            reuse = identity.attribution_ran
            result = await fn(
                mp4=mp4, detectors=detectors, log_tag=log_tag, settings=settings,
                patient_in_clip=identity.patient_in_clip,
                pre_extracted_frames=(
                    identity.extracted_frames if reuse else None
                ),
                sample_fps=identity.sample_fps if reuse else None,
            )
            served_model = {
                "openai": settings.openai_model,
                "openrouter": settings.openrouter_model,
                "dashscope": settings.dashscope_model,
            }[prov]
            return prov, _dc_replace(result, model=served_model)
        except Exception as e:  # noqa: BLE001
            errors[prov] = str(e)
            log.warning("%s vlm provider %s failed: %s", log_tag, prov, e)
    raise RuntimeError(f"all VLM providers failed (choice={choice}): {errors}")
