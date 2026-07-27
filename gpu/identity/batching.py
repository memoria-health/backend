"""Contiguous batch planning — never front-drop (the 2026-07-23 lesson).

Keeping only the newest frames while the pod catches up (`new[-N:]`)
opens blind gaps between windows and the tracker renumbers on any
movement. Instead: select frames spread EVENLY across the whole
unshipped interval, decimated to `target_fps`, capped at `max_batch`,
and prepend the previous batch's final frame so consecutive windows
overlap by one frame (EdgeTAM's carry seed matches its first frame).
"""

from __future__ import annotations

from typing import Optional, TypeVar

F = TypeVar("F", bound=tuple)  # (idx, ts, *payload)


def plan_batch(
    frames: list[F],
    *,
    last_shipped: Optional[F],
    target_fps: float,
    max_batch: int,
) -> list[F]:
    """Pick the next contiguous batch from `frames` (ascending by idx).

    `last_shipped` is the final frame of the previous batch; it is
    re-included as the overlap seed and only frames AFTER it are new.
    Returns [] when there is nothing new.
    """
    if last_shipped is not None:
        last_idx = last_shipped[0]
        new = [f for f in frames if f[0] > last_idx]
    else:
        new = list(frames)
    if not new:
        return []

    # Even decimation to target_fps across the WHOLE span — front
    # preserved, newest preserved, no tail-only selection.
    kept: list[F] = []
    next_t = -1.0
    for f in new:
        if f[1] >= next_t:
            kept.append(f)
            next_t = f[1] + 1.0 / target_fps
    if kept[-1][0] != new[-1][0]:
        kept.append(new[-1])                 # newest always ships
    # NOTE: compare by frame idx, never tuple equality — payloads may be
    # numpy arrays (elementwise == would raise).

    # Cap by even thinning (never a contiguous front- or tail-drop).
    if len(kept) > max_batch:
        if max_batch <= 1:
            kept = [kept[-1]]
        else:
            step = (len(kept) - 1) / (max_batch - 1)
            kept = [kept[round(i * step)] for i in range(max_batch)]

    if last_shipped is not None:
        kept.insert(0, last_shipped)
    return kept
