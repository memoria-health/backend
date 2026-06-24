"""Per-patient video data-collection.

An in-process recorder that lives alongside the detection pipeline:
when a patient has opted into data collection, the camera's stream is
recorded to one MP4 per presence session (video + audio) and uploaded
to Backblaze B2. It reuses the frames detection already decodes, so it
adds no extra LiveKit subscription and no per-minute egress cost.

See ``docs/plans/2026-06-24-video-data-collection-recorder.md``.
"""
