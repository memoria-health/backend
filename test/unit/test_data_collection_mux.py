"""Task 8 (mux half) — finalize a session into one uploadable MP4.

Shells out to real ffmpeg: the streaming encoder produces a fragmented
H.264 MP4 and detection produces raw s16le PCM; mux_av copies the video
(no re-encode) and AAC-encodes the audio into one faststart MP4. Skipped
without ffmpeg/ffprobe.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services.data_collection.mux import MuxError, mux_av
from src.services.data_collection.streaming_encoder import StreamingMp4Writer

_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-i", str(path)],
        capture_output=True, check=True,
    )
    return json.loads(proc.stdout)


async def _make_video(path: Path) -> None:
    w = StreamingMp4Writer(path, width=320, height=240, fps=15)
    await w.start()
    blob = b"\x80" * (320 * 240 * 3 // 2)
    for _ in range(10):
        w.feed(320, 240, blob)
        await asyncio.sleep(0.03)
    await w.aclose()


@_ffmpeg
@pytest.mark.asyncio
async def test_mux_combines_video_and_audio(tmp_path) -> None:
    vid = tmp_path / "v.mp4"
    await _make_video(vid)
    aud = tmp_path / "a.pcm"
    aud.write_bytes(b"\x00\x00" * (16000 // 2))  # 0.5s of silent 16k mono
    out = tmp_path / "final.mp4"

    res = await mux_av(vid, aud, out)

    assert res == out
    streams = _ffprobe(out)["streams"]
    kinds = {s["codec_type"] for s in streams}
    assert kinds == {"video", "audio"}
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert video["codec_name"] in {"h264", "avc1"}
    assert audio["codec_name"] == "aac"


@_ffmpeg
@pytest.mark.asyncio
async def test_mux_without_audio_is_video_only(tmp_path) -> None:
    vid = tmp_path / "v.mp4"
    await _make_video(vid)

    out = tmp_path / "noaudio.mp4"
    await mux_av(vid, None, out)
    assert [s["codec_type"] for s in _ffprobe(out)["streams"]] == ["video"]

    # Empty PCM (audio buffer was warming) also takes the video-only path.
    empty = tmp_path / "empty.pcm"
    empty.write_bytes(b"")
    out2 = tmp_path / "emptyaudio.mp4"
    await mux_av(vid, empty, out2)
    assert [s["codec_type"] for s in _ffprobe(out2)["streams"]] == ["video"]


@pytest.mark.asyncio
async def test_mux_raises_clear_error_when_ffmpeg_missing(monkeypatch) -> None:
    async def _boom(*_a, **_k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _boom)
    with pytest.raises(MuxError, match="ffmpeg binary not found"):
        await mux_av("/tmp/x.mp4", None, "/tmp/out.mp4")
