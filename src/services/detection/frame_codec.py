"""Frame codec helpers — JPEG-in-buffer storage (2026-07-19).

The rolling frame store used to hold raw I420 frames (5.5 MB each at
the Tapo's native 2K), which is what made native resolution un-storable
(the soak-1 OOM ledger: ~24× the designed footprint once frame size ×
retention × rate × overlapping rounds stacked up). Storing each kept
frame as a JPEG shrinks it ~6-10× at quality 88 with no accuracy cost
that survives the pipeline: every frame the VLM sees passes through a
lossier H.264 hop anyway, and the face/identity path has consumed
JPEG-decoded frames all along.

Two representations ride in each stored FrameSnapshot:

  - `data`  — the JPEG bytes (was: raw I420). Consumers that need
    pixels decode on use; the per-round MP4 build feeds them to ffmpeg
    as an MJPEG stream (no Python-side decode at all).
  - `luma`  — a tiny raw grayscale thumbnail (LUMA_W×LUMA_H) made from
    the I420 Y plane at push time, so the motion/wake math never pays
    a JPEG decode. Block means are scale-invariant, so the block-motion
    calibration (trip floor 12) carries over unchanged.

Every consumer FORMAT-SNIFFS (`is_jpeg`) and keeps its legacy raw-I420
path, so raw-frame tests and any transitional mixed windows keep
working — the system flips to compressed the moment the receive path
starts pushing JPEGs, with no flag day.
"""

from __future__ import annotations

import numpy as np

# Thumbnail dims for the motion-math luma copy. 320×180 keeps the
# 32×18 block grid at 10×10 px per block — plenty for block MEANS
# (the grid averages regions; averaging a downscaled region ≈
# averaging the native region).
LUMA_W = 320
LUMA_H = 180

_JPEG_MAGIC = b"\xff\xd8"

# Default storage quality. 88 keeps facial detail comfortably above
# what the face matcher needs while shrinking 2K frames ~6-10×; the
# later H.264 hop (what the VLM sees) is lossier than this anyway.
DEFAULT_JPEG_QUALITY = 88


def is_jpeg(data: bytes) -> bool:
    return data[:2] == _JPEG_MAGIC


def i420_to_bgr(width: int, height: int, data: bytes) -> "np.ndarray":
    """Decode a raw I420 byte plane into a BGR array (cv2 layout)."""
    import cv2

    yuv = np.frombuffer(data, dtype=np.uint8).reshape(height * 3 // 2, width)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)


def bgr_to_i420(bgr: "np.ndarray") -> tuple[int, int, bytes]:
    import cv2

    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return w, h, yuv.tobytes()


def i420_to_jpeg(
    width: int, height: int, data: bytes, *, quality: int,
) -> bytes:
    """Compress one raw I420 frame to JPEG bytes."""
    import cv2

    bgr = i420_to_bgr(width, height, data)
    ok, buf = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise ValueError("cv2.imencode failed for I420 frame")
    return bytes(buf)


def jpeg_to_i420(data: bytes) -> tuple[int, int, bytes]:
    """Decode JPEG bytes back to (width, height, raw I420 bytes)."""
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed — not a decodable JPEG")
    return bgr_to_i420(bgr)


def jpeg_to_bgr(data: bytes) -> "np.ndarray":
    import cv2

    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed — not a decodable JPEG")
    return bgr


def luma_thumbnail(width: int, height: int, i420_data: bytes) -> bytes:
    """Raw grayscale LUMA_W×LUMA_H thumbnail straight from the I420 Y
    plane (no color conversion — the Y plane IS grayscale)."""
    import cv2

    y = np.frombuffer(
        i420_data, dtype=np.uint8, count=width * height,
    ).reshape(height, width)
    small = cv2.resize(y, (LUMA_W, LUMA_H), interpolation=cv2.INTER_AREA)
    return small.tobytes()


def compress_for_buffer(
    width: int, height: int, i420_data: bytes, *, quality: int,
) -> tuple[bytes, bytes]:
    """The push-time package: (jpeg, luma_thumbnail). Runs in a worker
    thread (~15-25 ms at 2K) — never on the event loop."""
    return (
        i420_to_jpeg(width, height, i420_data, quality=quality),
        luma_thumbnail(width, height, i420_data),
    )
