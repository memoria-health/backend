"""Call-plan executor (detection v2, P3 T3).

Runs the packer's CallPlans and merges their answers into ONE
tick-level result shaped like today's single call, so everything
downstream of the main call in `camera_ingest` (presence gate, state
machine, analyses row, promotion, archival) is completely unchanged.

Provider routing:
  lite  → the account's own chain via `run_main_vlm` (today's
          run_vlm: per-caregiver routing + fallback), so flag-on
          respects vlm_provider exactly like flag-off.
  flash → `gemini_provider` with the rapid model override (the P0
          pattern) — flash-tier recipes exist because lite is blind
          to what they detect.

Audio constraint: a needs_audio call that lands on a frame-sampling
provider (openai/openrouter/dashscope receive frames, not video) loses
its sound. The fallback is allowed (locked decision) but the tick is
MARKED: `sched_degraded_audio=true`.

Merge semantics:
  - description / people_present / people_count come from the SCENE
    call only (crop calls contain a person by construction; their
    people_present answer is meaningless for the room).
  - Each detector's schema-field group is taken from the call where it
    scored best (detected=True beats not-detected; then confidence) —
    a multi-track detector rides several calls and the winning
    person's answer carries.
  - SCENE-call failure fails the tick exactly like today (raise).
    A non-scene call failing is recorded (`sched_call_errors`) and the
    tick continues — a fall answer must never die because a crop
    call 503'd.

Ledger: vlm_input/output_tokens = SUM across all calls this tick (the
main-path spend, same key as flag-off); `sched_calls` breaks it down
per call for the soak's cost accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Awaitable, Callable, Sequence

from src.services.detection.providers.types import ProviderResult
from src.services.detection.scheduling.artifacts import build_person_crop_mp4
from src.services.detection.scheduling.packer import CallPlan

if TYPE_CHECKING:
    from src.services.detection.frame_buffer import FrameSnapshot
    from src.services.reid.inference import IdentityResult
    from src.settings import Settings

log = logging.getLogger(__name__)

# Providers that receive extracted frames, not native video — no audio.
_FRAME_SAMPLING_PROVIDERS = frozenset({"openai", "openrouter", "dashscope"})


@dataclass
class SchedulerTickResult:
    """Shaped like the single-call result the tick body expects."""

    description: str
    detector_fields: dict
    people_present: bool
    people_count: int
    provider: str                       # scene call's provider (row value)
    sched_fields: dict = field(default_factory=dict)


def _is_scene(plan: CallPlan) -> bool:
    return (
        plan.crop == "full" and plan.fps == 4
        and plan.tier == "lite" and plan.target_track_id is None
    )


def _score(fields: dict, ident: str) -> tuple[int, float]:
    detected = bool(fields.get(f"{ident}_detected"))
    conf = float(fields.get(f"{ident}_confidence") or 0.0)
    return (1 if detected else 0, conf)


async def run_scheduled_calls(
    plans: Sequence[CallPlan],
    *,
    scene_mp4: bytes,
    window: Sequence["FrameSnapshot"],
    pcm: bytes | None,
    pcm_sample_rate: int | None,
    pcm_num_channels: int | None,
    identity: "IdentityResult",
    settings: "Settings",
    vlm_choice: str,
    run_main_vlm: Callable[..., Awaitable],
    flash_provider: Callable[..., Awaitable[ProviderResult]],
    log_tag: str,
) -> SchedulerTickResult:
    """Execute every plan; merge into one tick-shaped result."""
    tracks_by_id = {t.track_id: t for t in (identity.tracks or ())}

    merged_fields: dict = {}
    # ident → (score, fields-group) so multi-call detectors keep their
    # best answer.
    best_by_ident: dict[str, tuple[tuple[int, float], dict]] = {}
    call_ledger: list[dict] = []
    call_errors: list[str] = []
    degraded_audio = False
    degraded_fps = False
    degraded_calls = 0
    total_in = 0
    total_out = 0

    scene_result: ProviderResult | None = None
    scene_provider = "gemini"

    for plan in plans:
        specs = [d.spec for d in plan.detectors]
        idents = [s.identifier for s in specs]

        # ── Artifact ─────────────────────────────────────────────
        # The window is snapshotted at the tick's target fps; a recipe
        # asking for more (flash 8 fps) would remux time-compressed
        # video, so the fps is clamped to what the frames really are
        # and the tick marked (T3/T4 review F5). Unreachable for the
        # default catalog; matters the day a P5 custom lands in
        # fine_motion before native-rate plumbing exists.
        artifact_fps = min(plan.fps, settings.livekit_worker_target_fps)
        if artifact_fps < plan.fps:
            degraded_fps = True
        mp4 = scene_mp4
        degraded_this_call = False
        if plan.crop == "person" and plan.target_track_id is not None:
            track = tracks_by_id.get(plan.target_track_id)
            crop_mp4 = None
            if track is not None:
                crop_mp4 = await build_person_crop_mp4(
                    window,
                    track=track,
                    sample_width=settings.reid_inference_sample_width,
                    fps=artifact_fps,
                    pcm=pcm if plan.needs_audio else None,
                    pcm_sample_rate=pcm_sample_rate,
                    pcm_num_channels=pcm_num_channels,
                )
            if crop_mp4 is None:
                # Never silently drop detectors: run this call's
                # members on the full scene artifact instead.
                degraded_this_call = True
                degraded_calls += 1
            else:
                mp4 = crop_mp4
        elif plan.crop != "full":
            # face / zone recipes have no builder yet (P6) and the
            # packer shouldn't emit them — full-scene fallback if one
            # ever appears.
            degraded_this_call = True
            degraded_calls += 1

        # ── Provider ─────────────────────────────────────────────
        try:
            if plan.tier == "flash":
                flash_settings = settings.model_copy(update={
                    "gemini_model": settings.rapid_model,
                    "gemini_video_fps": str(artifact_fps),
                })
                result = await flash_provider(
                    mp4=mp4, detectors=specs, log_tag=log_tag,
                    settings=flash_settings,
                    patient_in_clip=identity.patient_in_clip,
                    patient_bbox=None,
                )
                used_provider = "gemini"
            else:
                # Crop artifacts must actually be LOOKED AT: run_vlm's
                # frame-reuse hands frame-sampling providers Re-ID's
                # extraction of the ORIGINAL full clip, which would
                # silently replace the crop we just built (T3/T4
                # review F2). A reuse-disabled identity surrogate
                # forces those providers to decode the passed mp4;
                # patient_bbox is dropped too — full-frame coords are
                # meaningless on a crop. Scene / degraded-to-scene
                # calls keep the real identity (parity).
                call_identity = identity
                if mp4 is not scene_mp4:
                    call_identity = replace(
                        identity,
                        extracted_frames=None,
                        per_frame_bboxes=None,
                        sample_fps=None,
                        patient_bbox=None,
                    )
                used_provider, result = await run_main_vlm(
                    choice=vlm_choice,
                    mp4=mp4, detectors=specs, identity=call_identity,
                    log_tag=log_tag, settings=settings,
                )
        except Exception as exc:  # noqa: BLE001 — per-call isolation
            if _is_scene(plan):
                # No scene answer = no tick, exactly like today.
                raise
            call_errors.append(plan.plan_key)
            log.warning(
                "%s sched call failed key=%s detectors=%s err=%s",
                log_tag, plan.plan_key, idents, exc,
            )
            continue

        if plan.needs_audio and used_provider in _FRAME_SAMPLING_PROVIDERS:
            degraded_audio = True

        usage = result.usage or {}
        total_in += int(usage.get("input", 0))
        total_out += int(usage.get("output", 0))
        call_ledger.append({
            "key": plan.plan_key,
            "detectors": idents,
            "provider": used_provider,
            "input_tokens": int(usage.get("input", 0)),
            "output_tokens": int(usage.get("output", 0)),
            **({"degraded_artifact": True} if degraded_this_call else {}),
        })

        # ── Merge ────────────────────────────────────────────────
        if _is_scene(plan):
            scene_result = result
            scene_provider = used_provider
        for spec in specs:
            ident = spec.identifier
            keys = set(spec.schema_fields) | {
                f"{ident}_detected", f"{ident}_confidence", f"{ident}_reasoning",
            }
            group = {
                k: v for k, v in result.detector_fields.items() if k in keys
            }
            if not group:
                continue
            score = _score(result.detector_fields, ident)
            prev = best_by_ident.get(ident)
            if prev is None or score > prev[0]:
                best_by_ident[ident] = (score, group)

    if scene_result is None:
        # The packer GUARANTEES a scene plan on every tick (empty-
        # detector wide call when nothing else needs the wide look —
        # the always-on people_present block still drives the presence
        # machine, exactly like flag-off with all detectors toggled
        # off). Reaching here therefore means the scene call itself
        # failed — which the loop already re-raises — or a packer
        # regression. Fail the tick like today.
        raise RuntimeError("scheduler tick produced no scene call result")

    for _, group in best_by_ident.values():
        merged_fields.update(group)
    # Scene-call shared block wins for the non-detector keys.
    for k, v in scene_result.detector_fields.items():
        if k not in merged_fields:
            merged_fields[k] = v
    merged_fields["people_present"] = bool(scene_result.people_present)

    sched_fields: dict = {
        "sched_calls": call_ledger,
        "vlm_input_tokens": total_in,
        "vlm_output_tokens": total_out,
    }
    if call_errors:
        sched_fields["sched_call_errors"] = call_errors
    if degraded_audio:
        sched_fields["sched_degraded_audio"] = True
    if degraded_fps:
        sched_fields["sched_degraded_fps"] = True
    if degraded_calls:
        sched_fields["sched_degraded_artifacts"] = degraded_calls

    return SchedulerTickResult(
        description=scene_result.description,
        detector_fields=merged_fields,
        people_present=bool(scene_result.people_present),
        people_count=int(scene_result.people_count or 0),
        provider=scene_provider,
        sched_fields=sched_fields,
    )
