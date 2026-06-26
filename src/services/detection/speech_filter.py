"""Deterministic guard on the VLM's speech_heard / speech_transcript fields.

Audio LLMs (Gemini here) hallucinate speech transcripts from non-speech
audio — keyboard typing, room noise — and degenerate into repetition
loops ("/ Yes / Yes / Yes"). This is a documented, systematic failure of
large audio-language models (audio is their weakest modality; they lean
on language priors over the actual audio), NOT a prompt bug — so we don't
try to prompt our way out of it. We guard the model's speech OUTPUT with
two deterministic, fully-local checks (no new data egress, ~ms):

  1. VAD gate — if a voice-activity detector finds no sustained voiced
     speech in the ACTUAL PCM, any transcript the model produced is
     fabricated, so blank it.
  2. Deloop — if the transcript is degenerate (absurd compression ratio
     or heavy segment repetition), it's a decoder loop, so blank it.

Detection reasoning over native audio (e.g. the agitation tone/content
cue) is deliberately untouched — only the displayed transcript fields are
sanitized. The guard is fail-OPEN: if it can't run cleanly (missing dep,
unexpected audio shape) it leaves the model's output alone rather than
risk dropping real speech; the deloop still backstops the worst loops.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Callable

log = logging.getLogger(__name__)

_SPEECH_HEARD = "speech_heard"
_SPEECH_TRANSCRIPT = "speech_transcript"

# webrtcvad accepts only these rates, mono, 16-bit, 10/20/30 ms frames.
_VAD_RATES = (8000, 16000, 32000, 48000)
_FRAME_MS = 30
_DEFAULT_MIN_VOICED_MS = 300  # need ~0.3s of voiced frames to count as speech


def voiced_speech_present(
    pcm: bytes,
    sample_rate: int,
    *,
    num_channels: int = 1,
    aggressiveness: int = 2,
    min_voiced_ms: int = _DEFAULT_MIN_VOICED_MS,
) -> bool:
    """True if WebRTC-VAD finds at least ``min_voiced_ms`` of voiced speech
    in ``pcm`` (mono 16-bit). Fail-OPEN (returns True) if VAD can't run, so
    a config/format hiccup never silently drops real speech."""
    if not pcm:
        return False
    if sample_rate not in _VAD_RATES or num_channels != 1:
        return True  # can't run cleanly → leave the transcript alone
    try:
        import webrtcvad
    except Exception:  # dep missing → don't gate (deloop still applies)
        return True

    vad = webrtcvad.Vad(aggressiveness)
    frame_bytes = int(sample_rate * (_FRAME_MS / 1000.0)) * 2  # 2 bytes/sample
    if frame_bytes <= 0:
        return True
    voiced_ms = 0
    for start in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        frame = pcm[start : start + frame_bytes]
        try:
            if vad.is_speech(frame, sample_rate):
                voiced_ms += _FRAME_MS
                if voiced_ms >= min_voiced_ms:
                    return True
        except Exception:
            return True  # malformed frame → fail-open
    return False


def is_degenerate_transcript(text: str) -> bool:
    """True if the transcript looks like a decoder hallucination loop.

    Uses the same two signals the Whisper community uses: an absurd gzip
    compression ratio (repetitive text compresses far too well — Whisper
    flags > 2.4), and heavy repetition of the ``/``-joined speaker segments.
    """
    t = (text or "").strip()
    if len(t) < 12:
        return False
    raw = t.encode("utf-8")
    if len(raw) / max(len(zlib.compress(raw, 9)), 1) > 2.4:
        return True
    segs = [s.strip().lower() for s in t.split("/") if s.strip()]
    # 3+ segments and at most half are unique → looping (e.g. "Yes / Yes / Yes").
    return len(segs) >= 3 and len(set(segs)) <= len(segs) // 2


def sanitize_speech_fields(
    detector_fields: dict,
    *,
    pcm: bytes | None,
    sample_rate: int,
    num_channels: int = 1,
    log_tag: str = "",
    vad_fn: Callable[..., bool] = voiced_speech_present,
) -> None:
    """In-place: blank ``speech_heard`` / ``speech_transcript`` when the
    audio carries no voiced speech, or the transcript is a degenerate loop.

    No-op when the provider emitted no speech fields (e.g. the OpenAI path
    has no audio) or they're already empty. ``vad_fn`` is injectable for
    tests.
    """
    if _SPEECH_HEARD not in detector_fields and _SPEECH_TRANSCRIPT not in detector_fields:
        return
    transcript = str(detector_fields.get(_SPEECH_TRANSCRIPT) or "")
    if not detector_fields.get(_SPEECH_HEARD) and not transcript:
        return  # nothing to scrub

    reason: str | None = None
    if not vad_fn(pcm or b"", sample_rate, num_channels=num_channels):
        reason = "no_voiced_speech"
    elif is_degenerate_transcript(transcript):
        reason = "degenerate_transcript"

    if reason is not None:
        detector_fields[_SPEECH_HEARD] = False
        detector_fields[_SPEECH_TRANSCRIPT] = ""
        log.info("%s speech filtered (%s): %r", log_tag, reason, transcript[:80])
