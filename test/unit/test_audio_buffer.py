"""Unit tests for `AudioBuffer`.

Pure data structure — these tests do not touch LiveKit at all. Every
chunk is a synthetic ``b"\\x00" * N`` blob with an explicit monotonic
timestamp. The buffer's correctness lives in three behaviors:

  1. Empty pushes are silently dropped (LiveKit can emit a zero-length
     frame at stream-end; we shouldn't store a phantom chunk).
  2. The time-based trim removes chunks older than `window_seconds`
     relative to the most-recently-pushed chunk.
  3. `snapshot_last_seconds` returns the concatenated PCM for chunks
     within the trailing window, oldest first, relative to a pinned
     `now`.
"""

from __future__ import annotations

import pytest

from src.services.detection.audio_buffer import AudioBuffer, AudioSnapshot


def _pcm(n: int, fill: int = 0) -> bytes:
    """A throwaway int16-PCM byte string. Even-length so it would
    parse as int16 samples if anything bothered (nothing does — the
    buffer is byte-opaque).
    """
    return bytes([fill]) * (n * 2)


def test_init_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        AudioBuffer(window_seconds=0)


def test_init_rejects_nonpositive_sample_rate() -> None:
    with pytest.raises(ValueError):
        AudioBuffer(window_seconds=15, sample_rate=0)


def test_init_rejects_nonpositive_channel_count() -> None:
    with pytest.raises(ValueError):
        AudioBuffer(window_seconds=15, num_channels=0)


def test_default_shape_is_16khz_mono() -> None:
    """The room-worker asks `rtc.AudioStream` for these defaults; if
    they ever drift, the audio consumer's request shape and the
    buffer's stored shape will diverge silently. Pin them here."""
    buf = AudioBuffer(window_seconds=15)
    assert buf.sample_rate == 16000
    assert buf.num_channels == 1


def test_first_chunk_is_accepted() -> None:
    buf = AudioBuffer(window_seconds=15)
    buf.push(data=_pcm(100), captured_at=100.0)
    assert len(buf) == 1


def test_empty_chunk_is_silently_dropped() -> None:
    buf = AudioBuffer(window_seconds=15)
    buf.push(data=b"", captured_at=100.0)
    assert len(buf) == 0


def test_chunks_older_than_window_get_trimmed_on_push() -> None:
    buf = AudioBuffer(window_seconds=2.0)
    # Five chunks evenly spaced 0.2s apart from t=100.0.
    for i in range(5):
        buf.push(data=_pcm(10, fill=i + 1), captured_at=100.0 + 0.2 * i)
    assert len(buf) == 5

    # A chunk at 103.0 makes the cutoff 101.0 → all earlier chunks drop.
    buf.push(data=_pcm(10, fill=99), captured_at=103.0)
    assert len(buf) == 1
    pcm = buf.snapshot_last_seconds(10, now=103.0)
    assert pcm == _pcm(10, fill=99)


def test_snapshot_concatenates_chunks_in_temporal_order() -> None:
    buf = AudioBuffer(window_seconds=15)
    # Three distinguishable chunks at t=100.0, 100.5, 101.0.
    buf.push(data=_pcm(2, fill=0xAA), captured_at=100.0)
    buf.push(data=_pcm(2, fill=0xBB), captured_at=100.5)
    buf.push(data=_pcm(2, fill=0xCC), captured_at=101.0)
    pcm = buf.snapshot_last_seconds(5, now=101.0)
    assert pcm == _pcm(2, fill=0xAA) + _pcm(2, fill=0xBB) + _pcm(2, fill=0xCC)


def test_snapshot_only_returns_chunks_within_window() -> None:
    buf = AudioBuffer(window_seconds=15)
    buf.push(data=_pcm(2, fill=0x11), captured_at=100.0)
    buf.push(data=_pcm(2, fill=0x22), captured_at=100.5)
    buf.push(data=_pcm(2, fill=0x33), captured_at=101.0)
    # At now=101.0, a 0.7s window has cutoff 100.3 → chunks at 100.5
    # and 101.0 qualify; 100.0 does not.
    pcm = buf.snapshot_last_seconds(0.7, now=101.0)
    assert pcm == _pcm(2, fill=0x22) + _pcm(2, fill=0x33)


def test_snapshot_with_nonpositive_seconds_returns_empty_bytes() -> None:
    buf = AudioBuffer(window_seconds=15)
    buf.push(data=_pcm(10), captured_at=100.0)
    assert buf.snapshot_last_seconds(0) == b""
    assert buf.snapshot_last_seconds(-1) == b""


def test_snapshot_returns_empty_when_no_chunks_in_window() -> None:
    buf = AudioBuffer(window_seconds=15)
    buf.push(data=_pcm(10), captured_at=100.0)
    # 1s window at now=105.0 → cutoff 104.0; the only chunk is older.
    # The chunk itself isn't trimmed (window_seconds=15 governs trim);
    # snapshot just filters it out.
    assert buf.snapshot_last_seconds(1, now=105.0) == b""


def test_snapshots_preserve_caller_bytes_exactly() -> None:
    """The room-worker copies the SDK's memoryview into `bytes` before
    push; the buffer must hand that exact bytes back at snapshot time.
    """
    buf = AudioBuffer(window_seconds=15)
    sentinel = b"\x42\x00\xff\xff\x12\x34"  # 3 int16 samples
    buf.push(data=sentinel, captured_at=100.0)
    pcm = buf.snapshot_last_seconds(15, now=100.0)
    assert pcm == sentinel


def test_audio_snapshot_dataclass_is_frozen() -> None:
    """AudioSnapshot is shared across the snapshot list; tests
    accidentally mutating it would mask bugs in the buffer."""
    snap = AudioSnapshot(data=b"\x00\x00", captured_at=100.0)
    with pytest.raises(Exception):
        snap.data = b"\xff\xff"  # type: ignore[misc]


def test_continuous_pushing_stays_within_window() -> None:
    """Push at a steady rate for much longer than the window and
    verify chunk count stabilizes. Audio buffer is the real memory
    bound for long-running cameras.
    """
    buf = AudioBuffer(window_seconds=5.0)
    # 20ms chunks at 50 Hz (typical LiveKit audio frame cadence).
    for i in range(500):
        buf.push(data=_pcm(160), captured_at=100.0 + 0.02 * i)
    # At steady state, exactly 5s / 0.02s = 250 chunks fit in the window.
    assert 245 <= len(buf) <= 255
