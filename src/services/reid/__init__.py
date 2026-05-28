"""Person Re-Identification service.

This package owns everything to do with identifying the patient in a
camera frame:

  - `storage.py`: Supabase Storage uploads + signed URLs for face photos
  - `embeddings.py`: InsightFace SCRFD + ArcFace wrapper, gallery match,
                     adaptive threshold
  - `enrollment_processor.py`: background pipeline that processes
                                uploaded selfies into ready embeddings
  - `inference.py`: per-tick Re-ID labeller (Phase 3)
  - `harvester.py`: adaptive learning hook (Phase 4)
"""
