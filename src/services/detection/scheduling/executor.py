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
provider (openai/dashscope receive frames, not video) loses
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

Concurrency (2026-08-07): the provider calls run CONCURRENTLY, in
three phases — build artifacts, gather the calls, merge in plan order.

They used to run in one sequential loop, which made a tick cost the
SUM of its calls. Measured on the live soak: ~7s per call, three calls
due, so a 21.7s tick against a 10s cadence. `camera_ingest` allows 2
generations in flight plus one 16s queue slot, so a permanently
oversubscribed pipeline shed 25.2% of ticks (33 of 131 in 25 min) —
and shedding is blind to what a tick carries. On 2026-08-07 05:46 a
real fall landed in a 19s hole left by two shed ticks: the clip before
showed the patient standing, the clip after showed them already on the
floor, and `fall` correctly answered "no fall observed" both times.

The calls were ALREADY independent — `af0b1ec` unbundled them so no
detector's verdict could contaminate another's, and nothing in the
loop read another call's result. So this is latency-only: same calls,
same tokens, same answers. A tick becomes as slow as its SLOWEST call
rather than the sum.

Why three phases rather than gathering the whole loop body:
  - Artifact building (person-crop remux, `tint_window`) is ffmpeg and
    numpy on big native frames. It has OOM-killed the worker before
    (soak 2026-07-14), so it stays SEQUENTIAL — only the network waits
    overlap.
  - Merging stays sequential and in PLAN ORDER, so `sched_calls`,
    `best_by_ident` tie-breaks and the token totals are byte-identical
    to the sequential version regardless of which call returns first.
    A nondeterministic ledger would have made this morning's debugging
    impossible.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from src.services.detection.providers.types import ProviderResult
from src.services.detection.rapid_call import RAPID_MOTION_SPEC, rapid_seizure_spec
from src.services.detection.scheduling.artifacts import (
    build_overlay_crop_mp4,
    build_person_crop_mp4,
    person_crop_box,
    tint_window,
)
from src.services.detection.scheduling.decision import detection_keys_for
from src.services.detection.scheduling.packer import CallPlan
from src.services.detection.signals.rapid_energy import (
    MIN_ACTIVE_FRACTION_DEFAULT,
    MIN_SCORE_DEFAULT,
    rapid_energy,
)

if TYPE_CHECKING:
    from src.services.detection.frame_buffer import FrameSnapshot
    from src.services.identity.result import IdentityResult
    from src.settings import Settings

log = logging.getLogger(__name__)

# Providers that receive extracted frames, not native video — no audio.
_FRAME_SAMPLING_PROVIDERS = frozenset({"openai", "dashscope"})

# Overlay-channel prompt swaps: a detector whose row prompt describes the
# clean clip gets its validated overlay variant on RAPID calls (the P0
# recipe's prompt semantics — a clean-clip prompt on tinted footage, or
# vice versa, is exactly the mismatch that caused prod seizure
# overfiring). Detectors without an entry keep their own fragment; the
# provider's tint note (motion_overlay=True) explains the red channel.
_OVERLAY_SPECS = {"seizure": rapid_seizure_spec()}


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
        and not plan.windowed
    )


def _score(fields: dict, ident: str) -> tuple[int, float]:
    detected = bool(fields.get(f"{ident}_detected"))
    conf = float(fields.get(f"{ident}_confidence") or 0.0)
    return (1 if detected else 0, conf)


@dataclass
class _PreparedCall:
    """One plan whose artifact is built and which is ready to be sent.

    The seam between phase 1 (sequential, memory-bound) and phase 2
    (concurrent, network-bound). A plan that gated out or whose
    artifact failed never becomes one of these — that is the old
    `continue`.
    """

    plan: CallPlan
    specs: list
    idents: list[str]
    mp4: bytes
    artifact_fps: int
    overlay_specs: list | None
    degraded_artifact: bool


async def run_scheduled_calls(
    plans: Sequence[CallPlan],
    *,
    scene_mp4: bytes,
    window: Sequence["FrameSnapshot"],
    pcm: bytes | None,
    pcm_sample_rate: int | None,
    pcm_num_channels: int | None,
    native_window: Sequence["FrameSnapshot"] | None = None,
    native_fps: int | None = None,
    identity: "IdentityResult",
    settings: "Settings",
    vlm_choice: str,
    run_main_vlm: Callable[..., Awaitable],
    flash_provider: Callable[..., Awaitable[ProviderResult]],
    log_tag: str,
) -> SchedulerTickResult:
    """Execute every plan; merge into one tick-shaped result.

    `native_window` / `native_fps`: the frame buffer's native-rate
    snapshot for overlay (RAPID) plans — the recipe diffs and encodes at
    `rapid_fps`, which the decimated `window` cannot supply. When absent
    the overlay plan degrades to the decimated window (marked
    `sched_degraded_fps`) rather than silently skipping seizure coverage.
    """
    tracks_by_id = {t.track_id: t for t in (identity.tracks or ())}
    # Overlay (RAPID) state: the tint pass runs at most once per tick
    # (tint-then-crop is per-plan), and the gate readouts ride along for
    # soak tuning exactly like the P0 lane stamped them.
    overlay_tinted: Sequence["FrameSnapshot"] | None = None
    rapid_gate_af: float | None = None
    rapid_gate_score: float | None = None

    merged_fields: dict = {}
    # ident → (score, fields-group) so multi-call detectors keep their
    # best answer.
    best_by_ident: dict[str, tuple[tuple[int, float], dict]] = {}
    # ident → (DueDetector, artifact, plan) of the winning answer —
    # the k-of-N confirmation target.
    winning_call: dict[str, tuple] = {}
    call_ledger: list[dict] = []
    call_errors: list[str] = []
    degraded_audio = False
    degraded_fps = False
    degraded_calls = 0
    total_in = 0
    total_out = 0

    scene_result: ProviderResult | None = None
    scene_provider = "gemini"

    # ── Phase 1: artifacts (SEQUENTIAL — ffmpeg/numpy, memory-bound) ──
    # A plan that gates out or fails to build simply never reaches the
    # prepared list, which is the old `continue`.
    prepared: list[_PreparedCall] = []

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
        if plan.channel == "overlay":
            artifact_fps = min(
                plan.fps, native_fps or settings.livekit_worker_target_fps,
            )
        else:
            artifact_fps = min(plan.fps, settings.livekit_worker_target_fps)
        if artifact_fps < plan.fps:
            degraded_fps = True
        mp4 = scene_mp4
        degraded_this_call = False
        overlay_specs = None
        if plan.channel == "overlay":
            # RAPID recipe as a regular bucket call (Ledger row 5, was
            # the P0 standalone lane): native-rate frames, rapid-energy
            # pre-gate, tint-then-crop artifact. Gate closed = no spend
            # (the cost lever); a window too short to diff never runs;
            # an artifact failure is recorded — never judged as clean
            # footage under an overlay prompt.
            nw = native_window if native_window else window
            if native_window is None:
                degraded_fps = True  # decimated frames under a rapid recipe
            if len(nw) < 4:
                call_ledger.append({
                    "key": plan.plan_key, "detectors": idents,
                    "gate": "short_window",
                })
                continue
            track = (
                tracks_by_id.get(plan.target_track_id)
                if plan.target_track_id is not None else None
            )
            region = (
                person_crop_box(
                    nw, track=track,
                    sample_width=settings.reid_inference_sample_width,
                )
                if track is not None else None
            )
            energy = (
                rapid_energy(nw, region=region)
                if region is not None else rapid_energy(nw)
            )
            rapid_gate_af = max(rapid_gate_af or 0.0, energy.active_fraction)
            rapid_gate_score = max(rapid_gate_score or 0.0, energy.score)
            if not (
                energy.active_fraction >= MIN_ACTIVE_FRACTION_DEFAULT
                and energy.score >= MIN_SCORE_DEFAULT
            ):
                call_ledger.append({
                    "key": plan.plan_key, "detectors": idents,
                    "gate": "closed",
                })
                continue
            if overlay_tinted is None:
                overlay_tinted = await tint_window(nw)
            crop_mp4 = await build_overlay_crop_mp4(
                overlay_tinted, track=track,
                sample_width=settings.reid_inference_sample_width,
                fps=artifact_fps,
            )
            if crop_mp4 is None:
                degraded_calls += 1
                call_ledger.append({
                    "key": plan.plan_key, "detectors": idents,
                    "gate": "artifact_failed",
                })
                continue
            mp4 = crop_mp4
            if plan.needs_audio:
                degraded_audio = True  # overlay artifacts are video-only
            overlay_specs = [RAPID_MOTION_SPEC] + [
                _OVERLAY_SPECS.get(s.identifier, s) for s in specs
            ]
        elif plan.crop == "person" and plan.target_track_id is not None:
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

        prepared.append(_PreparedCall(
            plan=plan, specs=specs, idents=idents, mp4=mp4,
            artifact_fps=artifact_fps, overlay_specs=overlay_specs,
            degraded_artifact=degraded_this_call,
        ))
        if settings.artifact_dump_dir and mp4:
            try:
                import pathlib as _pl
                import time as _time
                _d = _pl.Path(settings.artifact_dump_dir)
                _d.mkdir(parents=True, exist_ok=True)
                _name = (f"{_time.strftime('%H%M%S')}_"
                         f"{plan.crop}_{'-'.join(idents)[:80]}.mp4")
                (_d / _name).write_bytes(mp4)
            except Exception:  # noqa: BLE001 — a dump must never hurt a tick
                log.exception("executor.artifact_dump_failed")

    # ── Phase 2: provider calls (CONCURRENT) ─────────────────────────
    # `return_exceptions=True` so one crop call 503'ing cannot cancel
    # its siblings — the per-call isolation contract is unchanged, it
    # just moves to the merge below. Order of `outcomes` matches
    # `prepared`, which is plan order.
    async def _call(pc: "_PreparedCall") -> tuple[str, ProviderResult]:
        plan = pc.plan
        if plan.tier == "flash":
            flash_update: dict = {
                "gemini_model": settings.rapid_model,
                "gemini_video_fps": str(pc.artifact_fps),
            }
            flash_settings = settings.model_copy(update=flash_update)
            result = await flash_provider(
                mp4=pc.mp4, detectors=(pc.overlay_specs or pc.specs),
                log_tag=log_tag,
                settings=flash_settings,
                patient_in_clip=identity.patient_in_clip,
                # Only the overlay channel tints its crop, so only it
                # gets the provider's red-tint note.
                motion_overlay=(plan.channel == "overlay"),
            )
            return "gemini", result
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
        if pc.mp4 is not scene_mp4:
            call_identity = replace(
                identity,
                extracted_frames=None,
                per_frame_bboxes=None,
                sample_fps=None,
                patient_bbox=None,
            )
        return await run_main_vlm(
            choice=vlm_choice,
            mp4=pc.mp4, detectors=pc.specs, identity=call_identity,
            log_tag=log_tag, settings=settings,
        )

    outcomes: list[Any] = list(await asyncio.gather(
        *(_call(pc) for pc in prepared), return_exceptions=True,
    ))

    # ── Phase 3: merge (SEQUENTIAL, PLAN ORDER) ──────────────────────
    # Iterating `prepared` rather than completion order is what keeps
    # the ledger, the token totals and every best-answer tie-break
    # identical to the pre-2026-08-07 sequential executor.
    for pc, outcome in zip(prepared, outcomes):
        plan, specs, idents, mp4 = pc.plan, pc.specs, pc.idents, pc.mp4
        artifact_fps = pc.artifact_fps
        if isinstance(outcome, BaseException):
            if _is_scene(plan):
                # No scene answer = no tick, exactly like today.
                raise outcome
            call_errors.append(plan.plan_key)
            log.warning(
                "%s sched call failed key=%s detectors=%s err=%s",
                log_tag, plan.plan_key, idents, outcome,
            )
            continue
        used_provider, result = outcome

        if plan.needs_audio and used_provider in _FRAME_SAMPLING_PROVIDERS:
            degraded_audio = True

        usage = result.usage or {}
        total_in += int(usage.get("input", 0))
        total_out += int(usage.get("output", 0))
        # Per-call evidence (2026-08-07). Every call generates its own
        # `seen`/`heard` describing ITS OWN artifact, but the row-level
        # copy is the scene call's (the shared-block rule below), so
        # a crop call's verdict was being stored beside a DIFFERENT
        # call's description of DIFFERENT pixels. A whole morning of
        # "the verdict contradicts `seen`" analysis was invalid because
        # of that — the contradiction was two calls looking at two
        # artifacts. Riding the evidence on the ledger entry makes a
        # verdict checkable against the evidence from its own call.
        evidence = {
            k: str(result.detector_fields[k])[:500]
            for k in ("seen", "heard")
            if result.detector_fields.get(k)
        }
        call_ledger.append({
            "key": plan.plan_key,
            "detectors": idents,
            "provider": used_provider,
            "model": result.model,
            "input_tokens": int(usage.get("input", 0)),
            "output_tokens": int(usage.get("output", 0)),
            **evidence,
            **({"degraded_artifact": True} if pc.degraded_artifact else {}),
        })

        # ── Merge ────────────────────────────────────────────────
        if _is_scene(plan):
            scene_result = result
            scene_provider = used_provider
        for d in plan.detectors:
            spec = d.spec
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
                # Remember the artifact + plan (+ the fps it was really
                # encoded at) behind the winning answer — the k-of-N
                # confirmation re-checks run against the SAME clip the
                # fire came from, at the SAME fps.
                winning_call[ident] = (d, mp4, plan, artifact_fps)

        if plan.channel == "overlay":
            # The describe-then-decide preamble's output is soak/debug
            # material — carry the first one seen onto the row.
            desc = result.detector_fields.get("rapid_motion_description")
            if desc and "rapid_motion_description" not in merged_fields:
                merged_fields["rapid_motion_description"] = desc

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

    if rapid_gate_af is not None:
        # Gate readouts ride along for soak tuning (P0 pattern) — was
        # the churn score ever worth a floor? what does the activity
        # distribution look like on real ticks?
        merged_fields["rapid_gate_active_fraction"] = round(rapid_gate_af, 4)
        merged_fields["rapid_gate_score"] = round(rapid_gate_score or 0.0, 3)

    # ── k-of-N confirmation (P4; Aryaman: N extra calls on the SAME
    # clip, ≥k must agree). Runs only with the decision flag on AND a
    # per-detector confirm config — no built-in carries one, so parity
    # holds. Spend-on-fires: re-checks happen only for detectors whose
    # merged verdict is True. A failed vote suppresses the fire (fields
    # flipped false + marked) BEFORE promotion ever sees it; re-check
    # infra errors count as non-confirming votes but can never crash
    # the tick.
    confirm_ledger: dict[str, dict] = {}
    if getattr(settings, "detection_v2_decision_enabled", False):
        for ident, (due, artifact, plan, art_fps) in winning_call.items():
            k, n = due.dims.confirm_k, due.dims.confirm_n
            detected_key, _conf_key = detection_keys_for(ident)
            if n <= 0 or not merged_fields.get(detected_key):
                continue
            votes = 0
            spend_in = spend_out = 0
            for i in range(n):
                try:
                    if plan.tier == "flash":
                        # Votes must see what the original call saw —
                        # same model, same fps (P4 review finding 3: a
                        # 4fps vote on an 8fps rapid artifact is a
                        # systematically blind re-check), and on overlay
                        # artifacts the same prompt variant + tint note.
                        vote_update: dict = {
                            "gemini_model": settings.rapid_model,
                            "gemini_video_fps": str(art_fps),
                        }
                        vote_spec = due.spec
                        if plan.channel == "overlay":
                            vote_spec = _OVERLAY_SPECS.get(ident, due.spec)
                        fs = settings.model_copy(update=vote_update)
                        res = await flash_provider(
                            mp4=artifact, detectors=[vote_spec],
                            log_tag=f"{log_tag} confirm:{ident}:{i}",
                            settings=fs, patient_in_clip=identity.patient_in_clip,
                            motion_overlay=(plan.channel == "overlay"),
                        )
                    else:
                        _, res = await run_main_vlm(
                            choice=vlm_choice, mp4=artifact,
                            detectors=[due.spec],
                            identity=replace(
                                identity, extracted_frames=None,
                                per_frame_bboxes=None, sample_fps=None,
                                patient_bbox=None,
                            ),
                            log_tag=f"{log_tag} confirm:{ident}:{i}",
                            settings=settings,
                        )
                except Exception:  # noqa: BLE001 — a vote, not a tick
                    log.warning(
                        "%s confirm call failed ident=%s vote=%d",
                        log_tag, ident, i,
                    )
                    continue
                u = res.usage or {}
                spend_in += int(u.get("input", 0))
                spend_out += int(u.get("output", 0))
                if bool(res.detector_fields.get(detected_key)):
                    votes += 1
            total_in += spend_in
            total_out += spend_out
            passed = votes >= k
            confirm_ledger[ident] = {
                "votes": votes, "k": k, "n": n, "passed": passed,
                "input_tokens": spend_in, "output_tokens": spend_out,
            }
            if not passed:
                merged_fields[detected_key] = False
                merged_fields[f"{ident}_confirm_suppressed"] = True
                log.info(
                    "%s confirm SUPPRESSED %s (%d/%d needed %d)",
                    log_tag, ident, votes, n, k,
                )
    # Scene-call shared block wins for the non-detector keys.
    for k, v in scene_result.detector_fields.items():
        if k not in merged_fields:
            merged_fields[k] = v
    merged_fields["people_present"] = bool(scene_result.people_present)

    sched_fields: dict = {
        "sched_calls": call_ledger,
        "vlm_input_tokens": total_in,
        "vlm_output_tokens": total_out,
        # Row-level served-model attribution for the SCHEDULER path
        # (2026-07-28): per-call models live in the ledger above; this
        # mirrors the scene call's model so hourly rollups and
        # confirm-vote pricing (which run the same primary) group
        # correctly without exploding the ledger.
        "vlm_model": getattr(scene_result, "model", None),
    }
    if confirm_ledger:
        sched_fields["sched_confirm"] = confirm_ledger
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
