"""Finalize-time redaction: find the sensitive seconds, pixelate them.

Replaces inline masking (2026-08-12, Aryaman): the recorder now always
encodes RAW to local disk, and this pass runs between mux and upload —
so a sensitive verdict no longer discards the session, masking no
longer outlives the moment it judged, and nothing sensitivity-related
runs in the frame hot path at all. The safety invariant moves with it:
from "never write a raw sensitive frame" to "never UPLOAD an
unredacted clip" — on ANY failure here the caller drops the clip
(fail closed, never upload raw on error).

Two nets find the spans, and both must miss for a leak:

  1. The tick verdicts the session already collected (free priors —
     the same production screen that used to steer the encoder).
  2. A chunked re-screen of the finished clip: the file is cut into
     ~30s pieces (stream copy, cheap) and each piece goes through the
     PRODUCTION sensitivity screen (`sensitivity.screen_clip`) — same
     prompt, same parser, same fail-sensitive ambiguity rules. Chunked
     video screening costs the same tokens as 1fps frame-by-frame
     analysis (video calls sample at that rate) but keeps temporal
     context and reuses proven code end to end.

Span hits are padded (default ±3s) and merged; the union goes to the
existing event-clip censor (person-mask pixelation, audio untouched —
the 2026-07-28 product call) in span-gated mode.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

log = logging.getLogger(__name__)

Span = tuple[float, float]

# ffmpeg is invoked with -ss/-t stream copy: keyframe-aligned cuts are
# fine for SCREENING (the model does not need frame-exact boundaries;
# padding absorbs the slack).
_FFMPEG_TIMEOUT_S = 60.0


def merge_spans(
    spans: Sequence[Span], *, pad_seconds: float, duration: float,
) -> list[Span]:
    """Pad each span, clamp to [0, duration], merge overlaps. Pure.

    Adjacent-after-padding spans merge into one so the censor sees the
    fewest, widest windows — flicker at span edges is worse than a few
    extra pixelated seconds.
    """
    if not spans or duration <= 0:
        return []
    padded = sorted(
        (max(0.0, a - pad_seconds), min(duration, b + pad_seconds))
        for a, b in spans
        if b > 0 and a < duration and b > a - 1e-9
    )
    if not padded:
        return []
    out: list[list[float]] = [list(padded[0])]
    for a, b in padded[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b > a]


@dataclass
class RedactionPlan:
    """What `detect_spans` decided, kept for metadata/sidecar."""

    spans: list[Span]
    chunk_hits: int
    chunks_screened: int


async def _cut_chunk(
    src: Path, dst: Path, start: float, length: float,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-i", str(src), "-c", "copy", "-movflags", "+faststart",
        str(dst),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(
            proc.communicate(), timeout=_FFMPEG_TIMEOUT_S,
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"redaction: chunk cut timed out at {start:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError(
            "redaction: chunk cut failed at "
            f"{start:.1f}s: {(err or b'')[-300:]!r}"
        )


async def detect_spans(
    final_path: Path,
    *,
    duration: float,
    settings: Any,
    screen: Callable[..., Awaitable[Any]] | None = None,
) -> RedactionPlan:
    """Chunked re-screen of the whole clip — the sole redaction net
    (tick-verdict priors retired 2026-08-14 with the tick-time screen).

    Raises on screening/cut failure — the caller treats ANY exception
    as "drop the clip" (fail closed). A screen call that itself judges
    ambiguously already resolves sensitive inside `parse_verdict`.
    """
    if screen is None:
        from src.services.detection.sensitivity import screen_clip as screen

    pad = float(settings.data_collection_redaction_pad_seconds)
    chunk_s = float(settings.data_collection_redaction_chunk_seconds)
    concurrency = int(settings.data_collection_redaction_concurrency)

    starts = []
    t = 0.0
    while t < duration:
        starts.append(t)
        t += chunk_s

    sem = asyncio.Semaphore(concurrency)
    hits: list[Span] = []

    async def _screen_one(start: float, tmpdir: str) -> None:
        dst = Path(tmpdir) / f"chunk_{start:09.2f}.mp4"
        length = min(chunk_s, duration - start)
        if length <= 0.5:  # sub-half-second tail: covered by padding
            return
        async with sem:
            await _cut_chunk(final_path, dst, start, length)
            data = dst.read_bytes()
            dst.unlink(missing_ok=True)
            result = await screen(data, settings=settings)
        if result.sensitive:
            hits.append((start, start + length))

    with tempfile.TemporaryDirectory(prefix="memoria-redact-") as tmpdir:
        await asyncio.gather(*(_screen_one(s, tmpdir) for s in starts))

    spans = merge_spans(hits, pad_seconds=pad, duration=duration)
    plan = RedactionPlan(
        spans=spans,
        chunk_hits=len(hits),
        chunks_screened=len(starts),
    )
    log.info(
        "redaction: %s -> %d span(s) (%d chunk hits / %d chunks)",
        final_path.name, len(spans), plan.chunk_hits,
        plan.chunks_screened,
    )
    return plan
