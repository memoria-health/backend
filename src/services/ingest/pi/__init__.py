"""Pi camera ingest service (push-mode).

A Raspberry Pi running a small picamera2 client opens a long-lived
HTTPS POST to `/v1/cameras/me/frames` and streams `multipart/x-mixed-
replace` JPEG frames over the request body. The push endpoint
(src/api/cameras_ingest.py) authenticates the connection, parses
the multipart body, runs motion detection, and republishes to
LiveKit as a regular `cam_*` participant. From the existing
detection pipeline's perspective, a Pi camera is indistinguishable
from an iPhone camera joining the LiveKit room.

Module layout:

    auth.py             -- ingest bearer-token mint + verify helpers
    mjpeg_reader.py     -- multipart MJPEG parser (used by both the
                           push endpoint parsing the request body
                           and the legacy pull path)
    motion_detector.py  -- Python port of the iOS SAD-tile algorithm
                           that runs on each frame before LiveKit
                           re-encode
    jpeg_decoder.py     -- JPEG -> Y plane + I420 buffer for LiveKit
    audio_reader.py     -- PCM-over-HTTP audio reader (pull mode,
                           vestigial; push-mode audio is a follow-up
                           endpoint)
    relay_publisher.py  -- LiveKit publisher: connects, publishes a
                           video track frame-by-frame, fires motion
                           heartbeat data packets. Exposes
                           `run_from_iterator(frames)` so the push
                           endpoint can drive it directly from the
                           request body.

There is no longer a separate worker process: the push endpoint
runs inside the FastAPI app, one connection per Pi at a time.
"""
