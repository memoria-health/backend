"""Deterministic speech guard — VAD gate + deloop post-filter."""

from __future__ import annotations

from src.services.detection.speech_filter import (
    is_degenerate_transcript,
    sanitize_speech_fields,
    voiced_speech_present,
)


# ── deloop (pure) ─────────────────────────────────────────────────────


def test_deloop_flags_repetition_loop() -> None:
    # The exact failure mode observed in the wild.
    assert is_degenerate_transcript(
        "You stand by your ship? / Yes / You stand by your ship? / Yes "
        "/ You stand by your ship? / Yes"
    )


def test_deloop_passes_normal_speech() -> None:
    assert not is_degenerate_transcript(
        "Can you grab the remote / It's on the kitchen counter I think"
    )


def test_deloop_ignores_short_strings() -> None:
    assert not is_degenerate_transcript("Yes")
    assert not is_degenerate_transcript("")


# ── VAD on real silence (webrtcvad) ───────────────────────────────────


def test_vad_silence_has_no_voiced_speech() -> None:
    # 1s of 16kHz mono int16 silence → no voiced frames.
    assert voiced_speech_present(b"\x00\x00" * 16000, 16000) is False


def test_vad_failopen_on_unsupported_rate() -> None:
    # Odd rate can't run webrtcvad → fail-open (don't drop real speech).
    assert voiced_speech_present(b"\x01\x02" * 16000, 44100) is True


# ── sanitize (injected VAD verdict) ───────────────────────────────────


def test_sanitize_blanks_when_no_voiced_speech() -> None:
    fields = {"speech_heard": True, "speech_transcript": "what's your take on influencers"}
    sanitize_speech_fields(
        fields, pcm=b"x", sample_rate=16000, vad_fn=lambda *a, **k: False
    )
    assert fields["speech_heard"] is False
    assert fields["speech_transcript"] == ""


def test_sanitize_blanks_degenerate_even_with_voice() -> None:
    fields = {"speech_heard": True, "speech_transcript": "Yes / Yes / Yes / Yes"}
    sanitize_speech_fields(
        fields, pcm=b"x", sample_rate=16000, vad_fn=lambda *a, **k: True
    )
    assert fields["speech_heard"] is False
    assert fields["speech_transcript"] == ""


def test_sanitize_keeps_real_speech() -> None:
    fields = {"speech_heard": True, "speech_transcript": "Have you taken your pills yet"}
    sanitize_speech_fields(
        fields, pcm=b"x", sample_rate=16000, vad_fn=lambda *a, **k: True
    )
    assert fields["speech_heard"] is True
    assert fields["speech_transcript"] == "Have you taken your pills yet"


def test_sanitize_noop_without_speech_fields() -> None:
    # OpenAI path has no audio / no speech fields → untouched.
    fields = {"fall_detected": False}
    sanitize_speech_fields(
        fields, pcm=b"x", sample_rate=16000, vad_fn=lambda *a, **k: False
    )
    assert fields == {"fall_detected": False}
