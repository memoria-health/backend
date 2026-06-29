"""Tiny smoke test that doesn't need Supabase running — just verifies
that Settings loads from `.env`.
"""

from src.settings import get_settings


def test_settings_loads_with_required_fields() -> None:
    s = get_settings()
    assert s.db_url.startswith("postgresql+asyncpg://")
    assert s.supabase_url.startswith("http")
    assert s.env in {"local", "test", "staging", "prod"}


def test_frame_buffer_retention_covers_post_roll() -> None:
    """Regression: the per-camera frame/audio buffers must retain at
    least the incident-clip post-roll window, not just the (smaller) LLM
    analysis window.

    Sizing the buffer to only `livekit_worker_window_seconds` starved the
    post-roll: when that window was cut 15s→11s while the post-roll stayed
    15s, the post snapshot silently lost its oldest ~4s, leaving a visible
    gap at the pre/post concat seam. Retention must follow clip length.
    """
    base = get_settings()
    # The exact prod config that shipped the seam: 11s analysis window,
    # 15s post-roll (analysis window is the SHORTER of the two).
    s = base.model_copy(
        update={
            "livekit_worker_window_seconds": 11,
            "clip_post_event_wait_seconds": 15.0,
        }
    )
    assert s.frame_buffer_retention_seconds >= s.clip_post_event_wait_seconds
    # ...and strictly more than the analysis window, so the post-roll's
    # oldest frames survive until the post snapshot reads them.
    assert s.frame_buffer_retention_seconds > s.livekit_worker_window_seconds


def test_frame_buffer_retention_covers_analysis_window() -> None:
    """Inverse case: when the analysis window is the larger of the two,
    retention must still cover IT — the LLM tick needs its full window.
    """
    base = get_settings()
    s = base.model_copy(
        update={
            "livekit_worker_window_seconds": 30,
            "clip_post_event_wait_seconds": 15.0,
        }
    )
    assert s.frame_buffer_retention_seconds >= s.livekit_worker_window_seconds
