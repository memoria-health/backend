"""`--only-cam` decides which participant a node process serves.

One process per SENSOR (2026-08-22), so several processes can sit in one
account room and each must subscribe to exactly one publisher. That
filter is the only thing keeping them apart, and getting it wrong is
silent in the worst way: the process boots, answers /state, reports
`ready`, and serves an empty camera map forever. The backend reads that
as "service up, doesn't know this camera", falls open, and every tick
comes back `reid_unavailable_llm` — detection keeps running on the LLM
alone and nothing anywhere logs an error.

Which is what happened. The roster passes `camera_identity` verbatim
(`cam_650ca94d`); the filter compared it against the identity with
`cam_` already stripped (`650ca94d`). No roster-started process ever
matched its own camera.
"""

from identity.service import cam_selected


def test_the_roster_passes_the_full_livekit_identity():
    """The regression. This is the exact shape `roster_agent.sh` sends."""
    assert cam_selected("cam_650ca94d", "cam_650ca94d") is True


def test_a_human_passes_the_bare_camera_id():
    """The documented manual shape, from --help and every runbook."""
    assert cam_selected("650ca94d", "cam_650ca94d") is True


def test_an_empty_filter_serves_everything():
    """No --only-cam is how a single-sensor home has always run."""
    assert cam_selected("", "cam_650ca94d") is True


def test_another_camera_in_the_same_room_is_skipped():
    """The whole reason the flag exists: two sensors, one account room."""
    assert cam_selected("cam_650ca94d", "cam_9271358c") is False
    assert cam_selected("650ca94d", "cam_9271358c") is False
