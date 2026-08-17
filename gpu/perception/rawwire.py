"""Raw frame wire between identity and perception (2026-08-13).

Identity and perception share the GPU box, and the original multipart-
JPEG contract meant every batch paid a JPEG encode (identity), a
multipart parse with >1MB parts spooling through disk (starlette), and
a JPEG decode (perception) — all CPU, all for a loopback hop. Under
3-camera load those stages helped peg the CPU at ~96% while the GPU
starved (stage profile, node D). This wire ships the pixels as-is:

    8 bytes   big-endian header length
    header    JSON — request params + "frames": [[orig_idx, w, h], ...]
    frames    concatenated BGR uint8, w*h*3 bytes each, header order

ONE module owns both directions so the two services cannot drift apart
on the format. Stdlib + numpy only — importable by pod_client, by
server.py, and by the unit suite without the SAM/torch stack.
"""

from __future__ import annotations

import json

import numpy as np

MAX_HEADER_BYTES = 1_000_000


def pack_frames(
    header: dict, frames: list[tuple[int, np.ndarray]],
) -> bytes:
    """Serialize `header` + (orig_idx, BGR uint8 array) frames. The
    header's "frames" table is written HERE from the arrays' real
    shapes — callers never hand-build it, so table and pixels cannot
    disagree. Any caller-supplied "frames" key is overwritten."""
    header = dict(header)
    header["frames"] = [
        [int(idx), int(f.shape[1]), int(f.shape[0])] for idx, f in frames
    ]
    hdr = json.dumps(header).encode()
    parts = [len(hdr).to_bytes(8, "big"), hdr]
    parts.extend(
        np.ascontiguousarray(f).tobytes() for _, f in frames
    )
    return b"".join(parts)


def parse_frames(
    data: bytes,
    *,
    max_frames: int = 128,
    max_side_px: int = 2560,
) -> tuple[dict, list[tuple[int, np.ndarray]]]:
    """Inverse of pack_frames. Raises ValueError on any malformed
    input — the server maps that to a 400. Arrays are zero-copy views
    into `data` (read-only), same sanity caps as the multipart path."""
    if len(data) < 8:
        raise ValueError("raw body too short")
    hlen = int.from_bytes(data[:8], "big")
    if not 0 < hlen <= MAX_HEADER_BYTES or 8 + hlen > len(data):
        raise ValueError("raw header length out of bounds")
    try:
        header = json.loads(data[8:8 + hlen])
    except ValueError:
        raise ValueError("raw header not JSON")
    metas = header.get("frames") or []
    if len(metas) > max_frames:
        raise ValueError(f"too many frames (>{max_frames})")
    off = 8 + hlen
    frames: list[tuple[int, np.ndarray]] = []
    for meta in metas:
        try:
            idx, w, h = (int(v) for v in meta[:3])
        except (TypeError, ValueError):
            raise ValueError("raw frame meta not [idx, w, h]")
        if not (0 < w <= max_side_px and 0 < h <= max_side_px):
            raise ValueError(f"raw frame dims {w}x{h} out of bounds")
        n = w * h * 3
        if off + n > len(data):
            raise ValueError("raw body shorter than frame table")
        frames.append((
            idx,
            np.frombuffer(data[off:off + n], np.uint8).reshape(h, w, 3),
        ))
        off += n
    return header, frames
