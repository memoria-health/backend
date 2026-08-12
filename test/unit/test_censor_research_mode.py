"""Research mask mode — the B2-corpus-only censor treatment.

This mode deliberately differs from the caregiver-facing one, so the
tests here are mostly about keeping those differences true:

  - it fills the person flat instead of pixelating them,
  - it hugs the body far more closely (pose stays honest),
  - and it has NO fallback: no subject found means the clip is dropped,
    never blurred and never stored raw.

The last one is the safety contract. The event path answers "I can't
find anyone" with a full-frame pixelation; a research clip has no such
answer, because a full-frame smear is worthless to a corpus and raw is
the exposure we are avoiding.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.detection import censor


def _frame(h: int = 120, w: int = 160) -> np.ndarray:
    """A frame with structure, so "was it changed?" is answerable."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[:, :, 0] = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    f[:, :, 2] = 200
    return f


def _mask(h: int = 120, w: int = 160) -> np.ndarray:
    """A person-shaped blob in the middle."""
    m = np.zeros((h, w), dtype=np.uint8)
    m[40:90, 60:100] = 1
    return m


def test_person_is_filled_flat_not_pixelated() -> None:
    frame, mask = _frame(), _mask()
    out = censor._research_frame(frame, mask)
    assert out is not None
    # Centre of the blob is exactly the fill colour, not a blurred
    # average of the original pixels.
    assert tuple(int(c) for c in out[65, 80]) == censor._RESEARCH_FILL_BGR


def test_the_room_outside_the_person_is_untouched() -> None:
    """Research value lives in the scene; only the body is removed."""
    frame, mask = _frame(), _mask()
    out = censor._research_frame(frame, mask)
    assert np.array_equal(out[0:20, 0:20], frame[0:20, 0:20])
    assert np.array_equal(out[110:120, 140:160], frame[110:120, 140:160])


def test_mask_is_tighter_than_the_caregiver_treatment() -> None:
    """The whole point of the mode: pose fidelity. If this inverts, the
    mode has silently become the event-clip treatment."""
    mask = _mask()
    research = censor._research_dilate(mask).sum()
    caregiver = censor._dilate_mask(mask).sum()
    assert research < caregiver


def test_dilation_still_leaves_a_real_margin() -> None:
    """Tight is not the same as none — segmentation under-covers at
    body edges, so some margin has to survive."""
    mask = _mask()
    assert censor._research_dilate(mask).sum() > mask.sum()


def test_no_subject_covers_the_whole_frame() -> None:
    """Segmentation missing someone is a segmentation failure, not
    evidence the room is empty — the gate upstream already said a person
    is there. So the frame gets covered, never left raw."""
    frame = _frame()
    for mask in (None, np.zeros((120, 160), np.uint8)):
        out = censor._research_frame(frame, mask)
        assert out is not None
        assert (out == np.array(censor._RESEARCH_FILL_BGR, np.uint8)).all()


async def test_research_mode_never_falls_back_to_full_frame(monkeypatch) -> None:
    """A tier-1 failure must propagate, not quietly produce a blurred
    clip that then gets uploaded as though it were fine."""
    async def boom(in_path, out_path, *, mode=censor.CENSOR_MODE_PIXELATE, **kw):
        raise censor.CensorError("no subject")

    called = False

    async def full_frame(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(censor, "_censor_person_mask_subprocess", boom)
    monkeypatch.setattr(censor, "_censor_full_frame", full_frame)

    with pytest.raises(censor.CensorError):
        await censor.censor_clip(
            "in.mp4", "out.mp4", mode=censor.CENSOR_MODE_RESEARCH,
        )
    assert called is False


async def test_full_frame_request_in_research_mode_is_refused() -> None:
    with pytest.raises(censor.CensorError):
        await censor.censor_clip(
            "in.mp4", "out.mp4", full_frame=True,
            mode=censor.CENSOR_MODE_RESEARCH,
        )


async def test_default_mode_still_falls_back(monkeypatch) -> None:
    """The caregiver-facing path must be completely unchanged."""
    async def boom(in_path, out_path, *, mode=censor.CENSOR_MODE_PIXELATE, **kw):
        raise censor.CensorError("no subject")

    called = False

    async def full_frame(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(censor, "_censor_person_mask_subprocess", boom)
    monkeypatch.setattr(censor, "_censor_full_frame", full_frame)

    await censor.censor_clip("in.mp4", "out.mp4")
    assert called is True


def test_mode_reaches_the_child_process() -> None:
    argv = censor._person_mask_subprocess_cmd(
        "a.mp4", "b.mp4", mode=censor.CENSOR_MODE_RESEARCH,
    )
    assert "--mode" in argv
    assert argv[argv.index("--mode") + 1] == censor.CENSOR_MODE_RESEARCH


def test_default_mode_adds_no_flag() -> None:
    """Keeps the existing child invocation byte-identical."""
    assert "--mode" not in censor._person_mask_subprocess_cmd("a.mp4", "b.mp4")
