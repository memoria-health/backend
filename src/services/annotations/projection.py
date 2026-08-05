"""Map an episode onto the recorded clips that contain it.

This is the join between the annotation layer and the video, and it is
COMPUTED rather than stored. An episode lives on a wall-clock timeline;
a clip is a window on that same timeline. Asking "where in the footage
is this?" is then arithmetic:

    offset_in_clip = episode.started_at - clip.started_at

Two things make that arithmetic valid, and both were prerequisites:

  * Clip duration is truthful. Before 2026-08-04 the recorder wrote
    fewer frames than wall clock demanded and declared them 15fps apart,
    so a clip claiming 871s covered 906s of real life. Every offset
    computed against it would have been wrong, by a load-dependent
    amount nobody could reconstruct.
  * The clip says so. `started_at`, `duration_seconds` and
    `timeline_exact` ride on the B2 object, so a consumer can do this
    without opening the file — and can tell when NOT to trust it.

Nothing here touches storage or the database. It takes an episode's span
and a list of clip spans and returns the overlaps, so it is exercised
entirely with plain values.

Gaps are reported, never papered over. Recording is gated on presence,
so a long episode routinely spans stretches where nothing was recorded
at all. A caller that got back only the covered segments would have no
way to tell "we have all of this" from "we have four minutes of a
seven-hour night".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class ClipSpan:
    """A recorded clip on the timeline. Built from what the B2 object
    already carries, so no download is needed to place it."""

    key: str
    started_at: datetime
    duration_seconds: float
    # False when the recorder could not keep up and the clip is shorter
    # than the session it recorded. Offsets into it are approximate;
    # carried through so the caller can decide whether that matters.
    timeline_exact: bool = True

    @property
    def ended_at(self) -> datetime:
        return self.started_at + timedelta(seconds=self.duration_seconds)


@dataclass(frozen=True)
class Segment:
    """The part of one clip that an episode covers."""

    key: str
    # Seconds from the start of the CLIP — where to seek to.
    start_offset: float
    end_offset: float
    # The same span as wall-clock, so a caller can line segments from
    # different clips up against each other.
    started_at: datetime
    ended_at: datetime
    timeline_exact: bool = True

    @property
    def duration(self) -> float:
        return self.end_offset - self.start_offset


@dataclass(frozen=True)
class Projection:
    """Where an episode is in the footage, and where it is not."""

    segments: tuple[Segment, ...]
    # Spans of the episode with no recording at all, as (start, end).
    # Recording is gated on presence, so these are normal.
    gaps: tuple[tuple[datetime, datetime], ...]

    @property
    def covered_seconds(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def is_complete(self) -> bool:
        return not self.gaps


def project(
    started_at: datetime,
    ended_at: datetime | None,
    clips: list[ClipSpan],
    *,
    now: datetime | None = None,
) -> Projection:
    """Find the parts of ``clips`` covering ``started_at``..``ended_at``.

    ``ended_at`` may be None — an episode that is still running. It is
    then treated as ending at ``now`` (or at the last clip we hold, when
    ``now`` isn't given), so an open episode still projects onto whatever
    has been recorded so far.
    """
    if ended_at is None:
        if now is not None:
            ended_at = now
        elif clips:
            ended_at = max(c.ended_at for c in clips)
        else:
            return Projection(segments=(), gaps=())
    if ended_at <= started_at:
        return Projection(segments=(), gaps=())

    overlapping = sorted(
        (
            c for c in clips
            if c.duration_seconds > 0
            and c.started_at < ended_at
            and c.ended_at > started_at
        ),
        key=lambda c: c.started_at,
    )

    segments: list[Segment] = []
    gaps: list[tuple[datetime, datetime]] = []
    cursor = started_at

    for clip in overlapping:
        # Clips can overlap each other (a roll boundary, or two cameras
        # in one room). Never emit the same wall-clock twice.
        if clip.ended_at <= cursor:
            continue
        if clip.started_at > cursor:
            gaps.append((cursor, min(clip.started_at, ended_at)))
            cursor = min(clip.started_at, ended_at)
            if cursor >= ended_at:
                break

        seg_start = max(cursor, clip.started_at)
        seg_end = min(ended_at, clip.ended_at)
        if seg_end <= seg_start:
            continue
        segments.append(
            Segment(
                key=clip.key,
                start_offset=(seg_start - clip.started_at).total_seconds(),
                end_offset=(seg_end - clip.started_at).total_seconds(),
                started_at=seg_start,
                ended_at=seg_end,
                timeline_exact=clip.timeline_exact,
            )
        )
        cursor = seg_end
        if cursor >= ended_at:
            break

    if cursor < ended_at:
        gaps.append((cursor, ended_at))

    return Projection(segments=tuple(segments), gaps=tuple(gaps))
