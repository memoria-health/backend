"""Room classification → per-camera privacy (2026-07-27).

Privacy is per-camera, derived from the ROOM the camera watches
(docs/plans/2026-07-27-room-privacy-and-clip-censoring.md). When the
ingest worker sees frames for a camera whose ``room_classified_at`` is
NULL (and ``settings.auto_privacy_mode`` is on), this module runs ONE
clean-frame VLM call with structured output ``{room_class, confidence}``
and finalizes the camera's privacy state:

  - ``bathroom`` / ``bedroom`` → that class + ``privacy_enabled=true``,
    REGARDLESS of confidence (fail-closed: a hesitant "bathroom" is
    still a bathroom).
  - ``other`` → accepted only at confidence ≥
    ``OTHER_CONFIDENCE_THRESHOLD``; below it the answer is "uncertain"
    and we re-sample (a dark / mid-mount / installer-in-frame first
    look must not permanently unlock a room).
  - After ``MAX_ATTEMPTS`` uncertain attempts (spaced
    ``RETRY_SPACING_SECONDS`` apart, ~1h overall) → finalize with the
    top guess but ``privacy_enabled=true`` — the fail-closed fallback.
    A VLM error/timeout counts as an uncertain attempt.

``room_classified_at`` is ALWAYS written at finalization; the ingest
trigger keys on it being NULL, so a finalized camera never re-enters
the classifier until a room rename / re-pair / support reclassify
clears the timestamp again.

Hot-path discipline: the per-tick hook (``kick``) is synchronous
bookkeeping only — the frame grab, JPEG encode, VLM call, and DB write
all run inside a fire-and-forget asyncio task, exactly like the other
tick side-tasks (push fan-out, clip archival). Attempt state lives
in-process (the detector worker runs inside the FastAPI process), keyed
by camera id; no extra DB column needed.

On finalizing ``privacy_enabled=true`` the per-camera live sever runs
(``livekit_privacy.enforce_privacy_for_camera``) so caregivers already
watching the room are cut immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from google import genai

from src.services.genai_client import get_client
from google.genai import types as genai_types

from src.db.camera import Camera
from src.services.detection.frame_codec import is_jpeg
from src.services.livekit_privacy import enforce_privacy_for_camera

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.frame_buffer import FrameBuffer
    from src.settings import Settings

log = logging.getLogger(__name__)

ROOM_CLASSES = ("bathroom", "bedroom", "other")
PRIVATE_ROOM_CLASSES = ("bathroom", "bedroom")

# "other" is accepted only at/above this confidence — below it the
# answer is uncertain and we re-sample. Plan starting point: 0.7.
OTHER_CONFIDENCE_THRESHOLD = 0.7
# Total classification attempts before the fail-closed fallback fires.
MAX_ATTEMPTS = 3
# Spacing between attempts: 3 attempts x 20 min ~= the plan's "up to 3
# tries across the first hour".
RETRY_SPACING_SECONDS = 20.0 * 60.0

_PROMPT = (
    "You are looking at a single frame from a fixed in-home monitoring "
    "camera. Classify the ROOM the camera is watching.\n\n"
    "room_class must be exactly one of:\n"
    "  - 'bathroom' — any bathroom, toilet, or shower room.\n"
    "  - 'bedroom'  — any room whose primary purpose is sleeping (a bed "
    "is the central furniture).\n"
    "  - 'other'    — every other room (living room, kitchen, hallway, "
    "dining room, office, ...).\n\n"
    "Be conservative toward privacy: if the frame plausibly shows a "
    "bathroom or bedroom (even partially, at an angle, or in low "
    "light), answer that class. Answer 'other' only when the room is "
    "clearly NOT a bathroom or bedroom.\n\n"
    "confidence is your self-assessed probability (0.0-1.0) that "
    "room_class is correct. If the frame is dark, blocked, mid-mount, "
    "or otherwise unreadable, answer 'other' with LOW confidence."
)


@dataclass(frozen=True)
class Answer:
    """One VLM answer. ``confidence`` is the model's self-report [0,1]."""

    room_class: str
    confidence: float


@dataclass(frozen=True)
class Verdict:
    """A finalized classification. ``confidence`` is None only on the
    all-attempts-errored fallback (no answer to report)."""

    room_class: str
    confidence: float | None
    privacy_enabled: bool


def decide(answers: list[Answer | None]) -> Verdict | None:
    """The decision rule, pure for unit tests. ``answers`` is every
    attempt so far, oldest first; ``None`` entries are failed VLM calls
    (error/timeout — counted as uncertain, fail-closed).

    Returns a Verdict to finalize, or None to keep re-sampling.
    """
    real = [a for a in answers if a is not None]
    # Any bathroom/bedroom answer ANYWHERE in the ledger is final and
    # Private, regardless of confidence — doubt about a sensitive room
    # resolves private. Whole-ledger, not latest-only (review fix): if a
    # private verdict's finalize failed and a later attempt answered a
    # confident 'other', the room must NOT unlock — any-sensitive-answer
    # wins. Report the highest-confidence private answer.
    private = [a for a in real if a.room_class in PRIVATE_ROOM_CLASSES]
    if private:
        top = max(private, key=lambda a: a.confidence)
        return Verdict(top.room_class, top.confidence, True)
    latest = answers[-1] if answers else None
    if (
        latest is not None
        and latest.room_class == "other"
        and latest.confidence >= OTHER_CONFIDENCE_THRESHOLD
    ):
        return Verdict("other", latest.confidence, False)
    if len(answers) >= MAX_ATTEMPTS:
        # Fail-closed fallback: still uncertain after every attempt →
        # record the top guess (highest-confidence real answer; only
        # sub-threshold 'other's can be here — a private answer would
        # have finalized above) but keep the camera PRIVATE.
        if real:
            top = max(real, key=lambda a: a.confidence)
            return Verdict(top.room_class, top.confidence, True)
        # Every attempt errored — nothing to guess from. 'other' as the
        # class (least-wrong label), privacy ON.
        return Verdict("other", None, True)
    return None


@dataclass
class _AttemptState:
    """In-process per-camera attempt ledger."""

    answers: list[Answer | None] = field(default_factory=list)
    last_attempt_mono: float = float("-inf")
    in_flight: bool = False
    # Decided verdict whose `_finalize` DB write hasn't landed yet
    # (set before finalize, cleared with the whole state on success).
    # A failed finalize leaves it set so `kick` retries the WRITE on a
    # later tick instead of wedging (ledger-full early-return) or
    # burning more VLM attempts. See the retry path in `kick`.
    verdict: Verdict | None = None


_states: dict[UUID, _AttemptState] = {}
# Fire-and-forget classification tasks, held so the loop can't GC them.
_tasks: set[asyncio.Task] = set()


def reset_attempts(camera_id: UUID) -> None:
    """Forget the in-process attempt ledger for a camera. Called when
    its classification is cleared (room rename, re-pair, support
    reclassify) so the re-run starts from attempt 1."""
    _states.pop(camera_id, None)


def kick(
    *,
    camera_id: UUID,
    livekit_room: str | None,
    livekit_identity: str | None,
    in_motion: bool,
    frame_buffer: FrameBuffer,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> None:
    """Per-tick hook, called by the ingest worker ONLY for cameras with
    ``room_classified_at IS NULL`` while ``settings.auto_privacy_mode``
    is on. Cheap synchronous bookkeeping; if an attempt is due, spawns
    the actual classification as a fire-and-forget task so the tick's
    hot path never waits on a VLM call.
    """
    state = _states.setdefault(camera_id, _AttemptState())
    if state.in_flight:
        return
    verdict = state.verdict
    if verdict is None and len(state.answers) >= MAX_ATTEMPTS:
        # Full ledger but no recorded verdict (a crash landed between
        # the append and the decide) — re-derive it; `decide` always
        # returns a verdict at MAX_ATTEMPTS. Never wedge here.
        verdict = state.verdict = decide(state.answers)
    if verdict is not None:
        # A decided verdict is awaiting its DB write (a previous
        # `_finalize` raised). Retry the WRITE — no new VLM attempt, no
        # motion gate (nothing frame-dependent left to do). Idempotency
        # holds: `_finalize` only writes while room_classified_at IS
        # NULL, so a concurrent override still wins.
        state.in_flight = True
        task = asyncio.create_task(
            _retry_finalize(
                camera_id=camera_id,
                verdict=verdict,
                livekit_room=livekit_room,
                livekit_identity=livekit_identity,
                state=state,
                sessionmaker=sessionmaker,
                settings=settings,
            ),
            name=f"room-classify-finalize-{camera_id}",
        )
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        return
    if in_motion:
        # Clean-frame selection: wait for a motion-settled window so we
        # never classify off an installer-in-frame / mid-mount view.
        # Next tick re-checks — costs nothing.
        return
    now = time.monotonic()
    if state.answers and now - state.last_attempt_mono < RETRY_SPACING_SECONDS:
        return
    frames = frame_buffer.snapshot_last_seconds(2.0)
    if not frames:
        return
    state.in_flight = True
    state.last_attempt_mono = now
    task = asyncio.create_task(
        _attempt(
            camera_id=camera_id,
            livekit_room=livekit_room,
            livekit_identity=livekit_identity,
            frame=frames[-1],
            state=state,
            sessionmaker=sessionmaker,
            settings=settings,
        ),
        name=f"room-classify-{camera_id}",
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _attempt(
    *,
    camera_id: UUID,
    livekit_room: str | None,
    livekit_identity: str | None,
    frame,
    state: _AttemptState,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> None:
    """One classification attempt: VLM call → decision rule → (maybe)
    finalize. Never raises into the loop."""
    try:
        answer: Answer | None
        try:
            jpeg = await _frame_to_jpeg(frame)
            answer = await classify_frame(jpeg, settings)
        except Exception:
            log.exception(
                "room_classifier attempt failed camera=%s "
                "(counts as uncertain, fail-closed)",
                camera_id,
            )
            answer = None
        state.answers.append(answer)
        log.info(
            "room_classifier attempt=%d/%d camera=%s answer=%s",
            len(state.answers), MAX_ATTEMPTS, camera_id, answer,
        )
        verdict = decide(state.answers)
        if verdict is None:
            return  # re-sample on a later tick (RETRY_SPACING_SECONDS)
        # Record the verdict BEFORE the DB write: if `_finalize` raises,
        # `kick` retries the write from this pending verdict instead of
        # wedging with a full ledger and room_classified_at still NULL.
        state.verdict = verdict
        await _finalize(
            camera_id=camera_id,
            verdict=verdict,
            livekit_room=livekit_room,
            livekit_identity=livekit_identity,
            sessionmaker=sessionmaker,
            settings=settings,
        )
    except Exception:
        log.exception("room_classifier task crashed camera=%s", camera_id)
    finally:
        state.in_flight = False


async def _retry_finalize(
    *,
    camera_id: UUID,
    verdict: Verdict,
    livekit_room: str | None,
    livekit_identity: str | None,
    state: _AttemptState,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> None:
    """Retry a finalize whose DB write previously failed. On success
    `_finalize` clears the whole ledger (reset_attempts); on another
    failure the pending verdict stays and the next `kick` retries again.
    Never raises into the loop."""
    try:
        await _finalize(
            camera_id=camera_id,
            verdict=verdict,
            livekit_room=livekit_room,
            livekit_identity=livekit_identity,
            sessionmaker=sessionmaker,
            settings=settings,
        )
    except Exception:
        log.exception(
            "room_classifier finalize retry failed camera=%s "
            "(will retry on a later tick)",
            camera_id,
        )
    finally:
        state.in_flight = False


async def _finalize(
    *,
    camera_id: UUID,
    verdict: Verdict,
    livekit_room: str | None,
    livekit_identity: str | None,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> None:
    """Write the verdict. Idempotent: only a camera still awaiting
    classification (``room_classified_at IS NULL``) is written, so a
    concurrent support override / duplicate task can't stomp a fresher
    verdict. On a privacy flip to true, sever live caregivers."""
    async with sessionmaker() as session:
        cam = await session.get(Camera, camera_id)
        if cam is None:
            log.info("room_classifier camera vanished camera=%s", camera_id)
            reset_attempts(camera_id)
            return
        if cam.room_classified_at is not None:
            # Someone else finalized (support override) — keep theirs.
            reset_attempts(camera_id)
            return
        cam.room_class = verdict.room_class
        cam.room_class_confidence = verdict.confidence
        cam.room_classified_at = datetime.now(UTC)
        cam.privacy_enabled = verdict.privacy_enabled
        await session.commit()
    reset_attempts(camera_id)
    log.info(
        "room_classifier FINALIZED camera=%s room_class=%s confidence=%s "
        "privacy_enabled=%s",
        camera_id, verdict.room_class, verdict.confidence,
        verdict.privacy_enabled,
    )
    if verdict.privacy_enabled:
        # Instant per-camera sever of any caregiver already watching.
        # Best-effort (the helper swallows LiveKit errors); the durable
        # cut is publisher-side track permissions, which the publishers
        # pick up from their desired-state / roster polls.
        await enforce_privacy_for_camera(
            settings, room=livekit_room, camera_identity=livekit_identity,
        )


async def _frame_to_jpeg(frame) -> bytes:
    """A FrameSnapshot's `data` is JPEG on the live path since
    2026-07-19; legacy/synthetic raw-I420 frames are encoded here (in a
    thread — cv2 encode must not block the loop)."""
    if is_jpeg(frame.data):
        return frame.data
    from src.services.detection.frame_codec import (
        DEFAULT_JPEG_QUALITY,
        i420_to_jpeg,
    )

    return await asyncio.to_thread(
        lambda: i420_to_jpeg(
            frame.width, frame.height, frame.data,
            quality=DEFAULT_JPEG_QUALITY,
        ),
    )


async def classify_frame(jpeg: bytes, settings: Settings) -> Answer:
    """One structured-output Gemini call on a single frame. Reuses the
    same google-genai client stack as the detection providers, but this
    is deliberately a separate, minimal call: one image, fixed schema,
    no per-account vlm_provider routing (privacy classification must
    behave identically for every account)."""
    if not settings.gemini_api_key:
        raise RuntimeError("NEMARA_GEMINI_API_KEY not configured")
    response_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["room_class", "confidence"],
        properties={
            "room_class": genai_types.Schema(
                type=genai_types.Type.STRING,
                enum=list(ROOM_CLASSES),
            ),
            "confidence": genai_types.Schema(type=genai_types.Type.NUMBER),
        },
    )
    client = get_client(settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=jpeg, mime_type="image/jpeg",
                        ),
                    ),
                    genai_types.Part(text=_PROMPT),
                ],
            )
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
        ),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("room_classifier: gemini returned empty response")
    parsed = json.loads(raw)
    room_class = str(parsed.get("room_class", "")).strip().lower()
    if room_class not in ROOM_CLASSES:
        raise RuntimeError(
            f"room_classifier: model returned unknown room_class {room_class!r}"
        )
    confidence = float(parsed.get("confidence", 0.0))
    # Clamp a drifting self-report into [0,1] rather than erroring —
    # the decision rule only compares against the threshold.
    confidence = min(1.0, max(0.0, confidence))
    return Answer(room_class=room_class, confidence=confidence)
