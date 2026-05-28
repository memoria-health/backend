"""Camera ingest services.

Each subpackage encapsulates one camera source type. From the
detection pipeline's perspective (`services/detection/`), every
source ends up looking the same: I420 video frames in a
`FrameBuffer`, PCM audio in an `AudioBuffer`, and a TTL motion-state
entry in `MotionCache`. The work of getting bytes from the physical
device into those buffers is what differs per source.

Subpackages:

    pi/         Raspberry Pi + MJPEG-over-HTTP. The Pi runs a tiny
                streaming daemon; the backend pulls frames, runs
                motion detection, and republishes to LiveKit so the
                rest of the system (caregiver viewer, detection
                tick) sees a normal LiveKit participant.

    iphone/     (Phase 8 refactor target.) Currently the iPhone path
                lives at `services/detection/room_worker.py` and the
                webhook handler at `api/livekit_webhooks.py`. To be
                relocated here once the Pi path is in production.

    common/     (Phase 8 refactor target.) Shared
                FrameBuffer/AudioBuffer/MotionCache will move here so
                both ingest paths consume identical abstractions.
"""
