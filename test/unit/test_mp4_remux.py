"""Unit tests for `frames_to_mp4`.

These tests shell out to real ffmpeg (the bot does the same at
runtime). Skipped if ffmpeg isn't on PATH so CI doesn't fail
spuriously on an image that didn't install it. We assert two
contracts:

  1. The function returns bytes that ffprobe agrees is a valid MP4
     with the expected duration / framerate.
  2. The function raises Mp4RemuxError on the kinds of failures the
     worker should surface (bad subprocess exit, mismatched frame
     dimensions, ffmpeg missing).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

import pytest

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4

_pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _yuv420_blob(width: int, height: int, *, fill: int = 0x80) -> bytes:
    """An all-grey I420 frame at the given size. Grey because the
    Y plane fills to 128 and U/V planes fill to 128 (neutral chroma).
    The actual pixel values don't matter for these tests — only the
    byte count + the ability to re-decode.
    """
    return bytes([fill]) * (width * height * 3 // 2)


def _ffprobe(mp4: bytes) -> dict:
    """Pipe MP4 bytes into ffprobe and return the parsed JSON."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            "-i", "pipe:0",
        ],
        input=mp4,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_returns_decodable_mp4_with_expected_dimensions() -> None:
    width, height, fps = 320, 240, 5
    # 15 frames @ 5fps = a 3-second clip.
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + i / fps,
        )
        for i in range(15)
    ]
    mp4 = await frames_to_mp4(frames, fps=fps)
    assert mp4.startswith(b"\x00\x00\x00") or mp4[4:8] == b"ftyp"
    info = _ffprobe(mp4)
    streams = info["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert video["codec_name"] in {"h264", "avc1"}
    assert video["width"] == width
    assert video["height"] == height


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_empty_frame_list() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        await frames_to_mp4([], fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_mixed_dimensions_filters_to_most_recent_size() -> None:
    """WebRTC adaptive bitrate can change publish resolution mid-stream,
    so the frame buffer may contain mixed sizes. mp4_remux now filters
    the window to only frames matching the MOST RECENT (last) frame's
    size — that's the current publish state and what the LLM should
    see. Older off-size frames are dropped silently with an info log.
    """
    # 5 small startup frames + 10 steady-state frames at the larger
    # publish resolution. Expectation: ffmpeg encodes the 10 large
    # frames; the 5 startup frames are dropped.
    small = [
        FrameSnapshot(
            width=320, height=240,
            data=_yuv420_blob(320, 240),
            captured_at=100.0 + i / 5,
        )
        for i in range(5)
    ]
    big = [
        FrameSnapshot(
            width=640, height=480,
            data=_yuv420_blob(640, 480),
            captured_at=101.0 + i / 5,
        )
        for i in range(10)
    ]
    mp4 = await frames_to_mp4(small + big, fps=5)
    info = _ffprobe(mp4)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["width"] == 640
    assert video["height"] == 480


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_wrong_byte_length_for_size() -> None:
    f = FrameSnapshot(width=320, height=240, data=b"\x00" * 100, captured_at=100.0)
    with pytest.raises(ValueError, match="expected 115200"):
        await frames_to_mp4([f], fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_out_of_range_fps() -> None:
    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(ValueError, match="fps must be in"):
        await frames_to_mp4([f], fps=0)
    with pytest.raises(ValueError, match="fps must be in"):
        await frames_to_mp4([f], fps=120)


@pytest.mark.asyncio
async def test_raises_clear_error_when_ffmpeg_missing(monkeypatch) -> None:
    """When ffmpeg can't be launched we want a `Mp4RemuxError` with a
    legible message, not a bare `FileNotFoundError`. The worker logs
    these — readability matters.
    """
    async def _fake_exec(*_args, **_kw):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(Mp4RemuxError, match="ffmpeg binary not found"):
        await frames_to_mp4([f], fps=5)


# ── audio path ────────────────────────────────────────────────────────


def _silence_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    """`seconds` of silent int16 mono PCM at the given rate."""
    return b"\x00\x00" * int(seconds * sample_rate)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_pcm_produces_mp4_with_both_streams() -> None:
    """When PCM is provided, the resulting MP4 must contain both a
    video stream and an AAC audio stream. The Gemini prompt's
    `speech_heard` / `speech_transcript` fields depend on audio
    actually reaching the LLM through the MP4."""
    width, height, fps = 320, 240, 5
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + i / fps,
        )
        for i in range(15)  # 3s of video
    ]
    pcm = _silence_pcm(3.0)
    mp4 = await frames_to_mp4(frames, fps=fps, pcm=pcm)

    info = _ffprobe(mp4)
    streams = info["streams"]
    assert len(streams) == 2
    kinds = {s["codec_type"] for s in streams}
    assert kinds == {"video", "audio"}

    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert video["codec_name"] in {"h264", "avc1"}
    assert video["width"] == width
    assert video["height"] == height
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 16000
    assert audio["channels"] == 1


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_no_pcm_keeps_video_only() -> None:
    """Without PCM the output stays video-only — no audio stream gets
    added accidentally. Important because the dashboard era's `-an`
    flag was the only thing keeping audio out of the LLM input; the
    new code path must preserve that when audio isn't buffered yet."""
    width, height, fps = 320, 240, 5
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + i / fps,
        )
        for i in range(15)
    ]
    mp4 = await frames_to_mp4(frames, fps=fps, pcm=None)
    info = _ffprobe(mp4)
    streams = info["streams"]
    assert len(streams) == 1
    assert streams[0]["codec_type"] == "video"


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_empty_pcm_keeps_video_only() -> None:
    """Empty PCM (audio buffer warming up) takes the no-audio branch
    rather than handing ffmpeg an empty `-i` input that would fail.
    """
    width, height, fps = 320, 240, 5
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + i / fps,
        )
        for i in range(15)
    ]
    mp4 = await frames_to_mp4(frames, fps=fps, pcm=b"")
    info = _ffprobe(mp4)
    assert len(info["streams"]) == 1
    assert info["streams"][0]["codec_type"] == "video"


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_pcm_rejects_invalid_sample_rate() -> None:
    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(ValueError, match="pcm_sample_rate"):
        await frames_to_mp4([f], fps=5, pcm=_silence_pcm(1.0), pcm_sample_rate=0)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_pcm_rejects_invalid_channel_count() -> None:
    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(ValueError, match="pcm_num_channels"):
        await frames_to_mp4([f], fps=5, pcm=_silence_pcm(1.0), pcm_num_channels=0)


# ── frames_to_vfr_mp4: wall-clock-faithful clips ──────────────────────


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_vfr_clip_duration_matches_wall_clock_not_frame_count() -> None:
    """Sparse, unevenly spaced frames (like a still scene) must play
    for the real wall-clock span, not frame_count/fps. 8 frames over
    20s at fixed 8fps would play 1s; VFR must play ~20s."""
    from src.services.detection.mp4_remux import frames_to_vfr_mp4

    width, height = 320, 240
    # Burst at t=0-2s, then a long still gap, then a final frame at 18s.
    times = [100.0, 100.5, 101.0, 101.5, 102.0, 108.0, 114.0, 118.0]
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=t,
        )
        for t in times
    ]
    mp4 = await frames_to_vfr_mp4(frames, end_at=120.0)
    info = _ffprobe(mp4)
    duration = float(info["format"]["duration"])
    # True span: first frame 100.0 -> end_at 120.0 = 20s.
    assert 19.0 <= duration <= 21.0


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_vfr_audio_muxes_full_length_without_truncation() -> None:
    """PCM spanning the full window must not be -shortest-truncated by
    a sparse video track, and leading audio from before the first
    frame is trimmed so A/V t=0 align."""
    from src.services.detection.mp4_remux import frames_to_vfr_mp4

    width, height = 320, 240
    rate = 16000
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + 2.0 * i,
        )
        for i in range(5)  # 100..108, end_at 110 -> 10s video
    ]
    # 12s of audio ending at 110.0 -> starts at 98.0; 2s must be
    # trimmed so audio t=0 == first frame (100.0), leaving ~10s.
    pcm = b"\x00\x00" * (rate * 12)
    mp4 = await frames_to_vfr_mp4(
        frames, end_at=110.0,
        pcm=pcm, pcm_sample_rate=rate, pcm_num_channels=1,
        pcm_starts_at=98.0,          # measured, not inferred from length
    )
    info = _ffprobe(mp4)
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert 9.0 <= float(audio["duration"]) <= 11.0
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert 9.0 <= float(video["duration"]) <= 11.0


@pytest.mark.asyncio
async def test_vfr_rejects_empty_window() -> None:
    from src.services.detection.mp4_remux import frames_to_vfr_mp4

    with pytest.raises(ValueError):
        await frames_to_vfr_mp4([])


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_vfr_single_frame_produces_playable_clip() -> None:
    """A window that decimates to ONE frame (resolution switch at the
    tail, or an empty post-roll) must still yield a playable clip with
    a real duration — not a 0-length file that uploads as has_clip."""
    from src.services.detection.mp4_remux import frames_to_vfr_mp4

    frames = [FrameSnapshot(
        width=320, height=240, data=_yuv420_blob(320, 240),
        captured_at=100.0,
    )]
    mp4 = await frames_to_vfr_mp4(frames, end_at=105.0)
    info = _ffprobe(mp4)
    assert 4.0 <= float(info["format"]["duration"]) <= 6.0
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert 4.0 <= float(video["duration"]) <= 6.0


# NOTE (2026-07-29): the audio-underrun displacement fix — using the
# buffer's MEASURED first-chunk timestamp instead of inferring the
# start from len(pcm)/bytes-per-second — is deliberately NOT covered
# here. The MP4 muxer reports the audio track's duration as the
# container duration once the audio ends early, so both the correct
# and the displaced output probe identically. Proving it needs sample-
# level decoding of a marked (non-silent) tone; covered instead by the
# AudioBuffer.snapshot_window unit test, which pins the timestamp the
# encoder now relies on.
