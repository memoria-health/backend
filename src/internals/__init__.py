"""Experimental, non-production backend code.

Everything in this package is gated behind the `enable_internals`
setting (`NEMARA_ENABLE_INTERNALS=true`) and is NOT mounted in
production. Nothing under `src/` outside this folder should import
from here. Safe to delete wholesale to remove all experimental
features.

Current contents:
  - record_clip.py    — caregiver-triggered 15s clip capture endpoint
  - clip_receiver.py  — standalone laptop receiver for the uploaded MP4s
"""
