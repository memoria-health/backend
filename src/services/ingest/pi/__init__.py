"""Pi camera ingest service.

A Raspberry Pi running a tiny MJPEG-over-HTTP daemon streams JPEG
frames continuously. This package consumes that stream, performs
motion detection, and republishes the video to LiveKit as a regular
camera participant. From the existing detection pipeline's
perspective, a Pi camera is indistinguishable from an iPhone camera
joining the LiveKit room.

Module layout (built incrementally):

    mjpeg_reader.py     -- async MJPEG stream parser (Phase 1)
    motion_detector.py  -- Python port of the iOS SAD-tile algorithm
                           that runs on the same frames before they
                           are re-encoded for LiveKit (Phase 2)
    relay_publisher.py  -- LiveKit publisher: connects, publishes a
                           video track frame-by-frame, fires motion
                           heartbeat data packets (Phase 5)
    manager.py          -- PiIngestManager: reads `cameras` rows with
                           `source_type='pi'` at startup, spawns one
                           relay coroutine per row, manages reconnects
                           (Phase 5)

The orchestration (manager.py) is run by a standalone worker
entrypoint in `src/workers/pi_ingest_worker.py` so a publisher crash
or WebRTC hang does not affect the FastAPI process serving the
caregiver app.
"""
