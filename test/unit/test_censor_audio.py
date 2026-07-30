"""Censored clips keep audio — except intimate situations.

2026-07-28 product call: censoring is VISUAL, the sound stays, because
a fall with a cry for help is exactly what a caregiver must hear.
2026-07-29 carve-out (Aryaman): intimate_situation ships silent, since
there the audio IS the exposure and keeping it defeats the blur.
"""

import asyncio

from src.services.clip_archival import _censor_bytes  # noqa: F401  (import guard)
from src.services.detection import censor, sensitivity


def _keep_audio_for(categories):
    """Mirror of the decision in resolve_clip_for_upload."""
    return not (set(categories or ()) & sensitivity.SILENT_CATEGORIES)


def test_intimate_situation_ships_silent():
    assert _keep_audio_for(["intimate_situation"]) is False


def test_intimate_situation_silences_even_alongside_others():
    assert _keep_audio_for(["nudity_or_undress", "intimate_situation"]) is False


def test_every_other_category_keeps_audio():
    for cat in sensitivity.SENSITIVITY_CATEGORIES:
        if cat in sensitivity.SILENT_CATEGORIES:
            continue
        assert _keep_audio_for([cat]) is True, cat


def test_missing_verdict_keeps_audio():
    """The screen-failure fallback must NOT silence.

    No verdict means the screen timed out / errored / was low
    confidence. That path blurs everything, and it fires on ordinary
    clips — silencing it would strip the shout from a fall whose screen
    happened to fail, which is the case the audio rule exists for.
    """
    assert _keep_audio_for(None) is True


async def _make_clip_with_audio(path: str) -> None:
    """1s of colour bars + a tone, so the audio track is real."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", path,
    )
    await proc.communicate()


async def _has_audio(path: str) -> bool:
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    return b"audio" in out


async def test_full_frame_censor_keeps_audio_by_default(tmp_path):
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    await _make_clip_with_audio(src)
    assert await _has_audio(src)
    await censor._censor_full_frame(src, dst)
    assert await _has_audio(dst)


async def test_full_frame_censor_drops_audio_when_asked(tmp_path):
    """The carve-out, through the real ffmpeg path — not a mirror."""
    src, dst = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4")
    await _make_clip_with_audio(src)
    assert await _has_audio(src)
    await censor._censor_full_frame(src, dst, keep_audio=False)
    assert not await _has_audio(dst)
