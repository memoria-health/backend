"""Room classifier decision rule + ingest-hook bookkeeping (2026-07-27).

The decision rule is deliberately pure (``decide`` over the list of
answers so far) so every fail-closed path is testable without a DB or
a VLM:

  - bathroom/bedroom finalize Private IMMEDIATELY, at any confidence.
  - 'other' finalizes Full only at/above the confidence threshold.
  - uncertain answers (low-confidence 'other', errored calls) re-sample
    up to MAX_ATTEMPTS, then finalize the top guess with privacy ON —
    the fail-closed fallback. All-errors still finalizes Private.

``kick`` is the per-tick hook: motion-settled + spacing + single-flight
bookkeeping that spawns the actual attempt fire-and-forget.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

import src.services.room_classifier as rc
from src.services.room_classifier import Answer, Verdict, decide

# ── decide(): the pure decision rule ─────────────────────────────────


@pytest.mark.parametrize("room", ["bathroom", "bedroom"])
@pytest.mark.parametrize("confidence", [0.05, 0.4, 0.99])
def test_private_room_finalizes_immediately_at_any_confidence(
    room: str, confidence: float
) -> None:
    """Fail-closed: a hesitant 'bathroom' is still a bathroom."""
    verdict = decide([Answer(room, confidence)])
    assert verdict == Verdict(room, confidence, True)


def test_confident_other_finalizes_full() -> None:
    verdict = decide([Answer("other", 0.9)])
    assert verdict == Verdict("other", 0.9, False)


def test_other_at_exact_threshold_is_accepted() -> None:
    verdict = decide([Answer("other", rc.OTHER_CONFIDENCE_THRESHOLD)])
    assert verdict is not None
    assert verdict.privacy_enabled is False


def test_uncertain_other_retries() -> None:
    """Below-threshold 'other' must NOT unlock the room — re-sample."""
    assert decide([Answer("other", 0.5)]) is None
    assert decide([Answer("other", 0.5), Answer("other", 0.69)]) is None


def test_errored_call_counts_as_uncertain_and_retries() -> None:
    assert decide([None]) is None
    assert decide([None, None]) is None


def test_three_uncertain_attempts_finalize_top_guess_but_private() -> None:
    """The fail-closed fallback: record the best guess, keep privacy ON."""
    answers = [Answer("other", 0.3), Answer("other", 0.6), Answer("other", 0.4)]
    verdict = decide(answers)
    assert verdict == Verdict("other", 0.6, True)  # top guess, still private


def test_three_errored_attempts_finalize_private() -> None:
    """Even with zero usable answers the camera finalizes PRIVATE —
    a VLM outage must never leave a bathroom unlocked forever."""
    verdict = decide([None, None, None])
    assert verdict == Verdict("other", None, True)


def test_mixed_errors_and_uncertain_fall_back_to_real_top_guess() -> None:
    verdict = decide([None, Answer("other", 0.5), None])
    assert verdict == Verdict("other", 0.5, True)


def test_private_answer_on_final_attempt_wins_over_fallback() -> None:
    """The latest answer is evaluated before the attempt-cap fallback."""
    answers = [Answer("other", 0.2), None, Answer("bedroom", 0.35)]
    verdict = decide(answers)
    assert verdict == Verdict("bedroom", 0.35, True)


def test_confident_other_on_final_attempt_still_unlocks() -> None:
    answers = [Answer("other", 0.2), None, Answer("other", 0.95)]
    verdict = decide(answers)
    assert verdict == Verdict("other", 0.95, False)


def test_prior_private_answer_beats_later_confident_other() -> None:
    """Whole-ledger fail-closed: if ANY answer said bathroom/bedroom
    (e.g. its finalize failed and the classifier re-sampled), a later
    confident 'other' must NOT unlock the room — private wins."""
    verdict = decide([Answer("bathroom", 0.4), Answer("other", 0.95)])
    assert verdict == Verdict("bathroom", 0.4, True)


def test_prior_private_answer_wins_across_errors_too() -> None:
    verdict = decide([Answer("bedroom", 0.2), None, Answer("other", 0.99)])
    assert verdict == Verdict("bedroom", 0.2, True)


def test_multiple_private_answers_report_highest_confidence() -> None:
    verdict = decide([Answer("bedroom", 0.3), Answer("bathroom", 0.7)])
    assert verdict == Verdict("bathroom", 0.7, True)


# ── kick(): per-tick bookkeeping ─────────────────────────────────────


class _FakeFrameBuffer:
    def __init__(self, frames: list) -> None:
        self._frames = frames

    def snapshot_last_seconds(self, seconds: float, **_kw) -> list:
        return self._frames


def _jpeg_frame():
    return SimpleNamespace(width=2, height=2, data=b"\xff\xd8fakejpegbytes")


_SETTINGS = SimpleNamespace(
    auto_privacy_mode=True, gemini_api_key="k", gemini_model="m",
)


def _kick(camera_id, *, in_motion=False, frames=None):
    rc.kick(
        camera_id=camera_id,
        livekit_room="patient_x",
        livekit_identity="cam_1234",
        in_motion=in_motion,
        frame_buffer=_FakeFrameBuffer(
            frames if frames is not None else [_jpeg_frame()]
        ),
        sessionmaker=None,  # unused — _finalize is patched in these tests
        settings=_SETTINGS,
    )


@pytest.mark.asyncio
async def test_kick_classifies_and_finalizes_private(monkeypatch) -> None:
    cam = uuid4()
    finalized: list = []

    async def _fake_classify(jpeg, settings):
        assert jpeg.startswith(b"\xff\xd8")
        return Answer("bathroom", 0.3)

    async def _fake_finalize(**kw):
        finalized.append(kw["verdict"])
        rc.reset_attempts(kw["camera_id"])

    monkeypatch.setattr(rc, "classify_frame", _fake_classify)
    monkeypatch.setattr(rc, "_finalize", _fake_finalize)

    _kick(cam)
    await asyncio.sleep(0.01)  # let the fire-and-forget task run

    assert finalized == [Verdict("bathroom", 0.3, True)]


@pytest.mark.asyncio
async def test_kick_waits_for_motion_settled_window(monkeypatch) -> None:
    """In-motion ticks never attempt — clean-frame selection."""
    cam = uuid4()

    async def _must_not_run(jpeg, settings):
        raise AssertionError("classify must not run while in motion")

    monkeypatch.setattr(rc, "classify_frame", _must_not_run)
    _kick(cam, in_motion=True)
    await asyncio.sleep(0.01)
    rc.reset_attempts(cam)


@pytest.mark.asyncio
async def test_kick_skips_without_frames(monkeypatch) -> None:
    cam = uuid4()

    async def _must_not_run(jpeg, settings):
        raise AssertionError("classify must not run without a frame")

    monkeypatch.setattr(rc, "classify_frame", _must_not_run)
    _kick(cam, frames=[])
    await asyncio.sleep(0.01)
    rc.reset_attempts(cam)


@pytest.mark.asyncio
async def test_uncertain_attempt_respects_retry_spacing(monkeypatch) -> None:
    """After an uncertain answer, re-kicks inside RETRY_SPACING_SECONDS
    do nothing; once the spacing elapses the next attempt runs."""
    cam = uuid4()
    calls = {"n": 0}
    finalized: list = []

    async def _fake_classify(jpeg, settings):
        calls["n"] += 1
        return Answer("other", 0.2)  # uncertain every time

    async def _fake_finalize(**kw):
        finalized.append(kw["verdict"])
        rc.reset_attempts(kw["camera_id"])

    monkeypatch.setattr(rc, "classify_frame", _fake_classify)
    monkeypatch.setattr(rc, "_finalize", _fake_finalize)

    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 1 and finalized == []

    # Immediately re-kicking is a no-op (spacing not elapsed).
    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 1

    # Fast-forward the ledger past the spacing → attempt 2 runs.
    rc._states[cam].last_attempt_mono -= rc.RETRY_SPACING_SECONDS + 1
    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 2 and finalized == []

    # Attempt 3 (final) → fail-closed fallback finalizes PRIVATE.
    rc._states[cam].last_attempt_mono -= rc.RETRY_SPACING_SECONDS + 1
    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 3
    assert finalized == [Verdict("other", 0.2, True)]


@pytest.mark.asyncio
async def test_classify_error_counts_as_uncertain_attempt(monkeypatch) -> None:
    """A VLM error is an uncertain attempt (fail-closed), not a crash
    and not a free pass."""
    cam = uuid4()
    finalized: list = []

    async def _boom(jpeg, settings):
        raise RuntimeError("gemini down")

    async def _fake_finalize(**kw):
        finalized.append(kw["verdict"])
        rc.reset_attempts(kw["camera_id"])

    monkeypatch.setattr(rc, "classify_frame", _boom)
    monkeypatch.setattr(rc, "_finalize", _fake_finalize)

    for _ in range(rc.MAX_ATTEMPTS):
        _kick(cam)
        await asyncio.sleep(0.01)
        if cam in rc._states:
            rc._states[cam].last_attempt_mono -= rc.RETRY_SPACING_SECONDS + 1

    assert finalized == [Verdict("other", None, True)]


@pytest.mark.asyncio
async def test_finalize_failure_recovers_on_later_kick(monkeypatch) -> None:
    """A failed `_finalize` (DB blip) must NOT wedge the camera: the
    decided verdict is kept pending and the next kick retries the DB
    WRITE — no extra VLM attempt — even during motion (nothing
    frame-dependent left to do)."""
    cam = uuid4()
    calls = {"classify": 0, "finalize": 0}
    finalized: list = []

    async def _fake_classify(jpeg, settings):
        calls["classify"] += 1
        return Answer("bathroom", 0.8)

    async def _flaky_finalize(**kw):
        calls["finalize"] += 1
        if calls["finalize"] == 1:
            raise RuntimeError("db down")
        finalized.append(kw["verdict"])
        rc.reset_attempts(kw["camera_id"])

    monkeypatch.setattr(rc, "classify_frame", _fake_classify)
    monkeypatch.setattr(rc, "_finalize", _flaky_finalize)

    _kick(cam)
    await asyncio.sleep(0.01)
    # Verdict decided but the write failed — pending, camera not wedged.
    assert calls == {"classify": 1, "finalize": 1}
    assert finalized == []
    assert rc._states[cam].verdict == Verdict("bathroom", 0.8, True)

    _kick(cam, in_motion=True)  # retry ignores the motion gate
    await asyncio.sleep(0.01)
    assert calls == {"classify": 1, "finalize": 2}  # write retried, no new VLM call
    assert finalized == [Verdict("bathroom", 0.8, True)]
    assert cam not in rc._states  # ledger cleared on success


@pytest.mark.asyncio
async def test_full_ledger_without_verdict_still_finalizes(monkeypatch) -> None:
    """The wedge case: MAX_ATTEMPTS answers in the ledger but no
    finalize landed (crash between append and decide). kick() must
    re-derive the fail-closed verdict and finalize instead of
    early-returning forever."""
    cam = uuid4()
    finalized: list = []

    async def _must_not_run(jpeg, settings):
        raise AssertionError("no new VLM attempt — the ledger is full")

    async def _fake_finalize(**kw):
        finalized.append(kw["verdict"])
        rc.reset_attempts(kw["camera_id"])

    monkeypatch.setattr(rc, "classify_frame", _must_not_run)
    monkeypatch.setattr(rc, "_finalize", _fake_finalize)

    rc._states[cam] = rc._AttemptState(answers=[None, None, None])
    _kick(cam)
    await asyncio.sleep(0.01)
    assert finalized == [Verdict("other", None, True)]
    assert cam not in rc._states


@pytest.mark.asyncio
async def test_reset_attempts_restarts_the_ledger(monkeypatch) -> None:
    cam = uuid4()
    calls = {"n": 0}

    async def _fake_classify(jpeg, settings):
        calls["n"] += 1
        return Answer("other", 0.2)

    monkeypatch.setattr(rc, "classify_frame", _fake_classify)
    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 1

    # Rename/re-pair/support-clear resets the ledger → next kick runs
    # immediately (no spacing carry-over from the dropped state).
    rc.reset_attempts(cam)
    _kick(cam)
    await asyncio.sleep(0.01)
    assert calls["n"] == 2
    rc.reset_attempts(cam)
