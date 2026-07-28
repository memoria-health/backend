"""Unit: video ingress meter (soak #1 finding 3 instrumentation).

Pins the readings the SDK-delivery bisect depends on:
  - steady delivery reports recv vs kept fps and delivered res
  - a stall shows up as a growing max gap, never a frozen healthy line
  - the rolling window forgets old traffic (recent truth only)
  - a cold meter stamps NOTHING (no lying zeros on bypass paths)
plus the CameraIngest._stamp_ingress wiring (fields + no-op-when-cold).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

from src.services.detection.camera_ingest import CameraIngest
from src.services.detection.ingress_meter import IngressMeter
from src.services.detection.motion_cache import MotionCache


def _feed(
    m: IngressMeter,
    t0: float,
    *,
    n: int,
    fps: float,
    keep_every: int = 4,
    convert_ms: float = 2.0,
    width: int = 1920,
    height: int = 1080,
) -> float:
    for i in range(n):
        m.record(
            t0 + i / fps,
            accepted=(i % keep_every == 0),
            convert_ms=convert_ms,
            width=width,
            height=height,
        )
    return t0 + n / fps


def test_steady_delivery_reports_recv_kept_and_res() -> None:
    m = IngressMeter()
    now = _feed(m, 1000.0, n=45, fps=15.0, keep_every=4, convert_ms=3.5)
    snap = m.snapshot(now)
    assert snap is not None
    # 45 frames over ~3s → ~15 recv; every 4th accepted → ~3.75 kept.
    assert 13.0 <= snap["ingress_recv_fps"] <= 16.0
    assert 3.0 <= snap["ingress_kept_fps"] <= 4.5
    assert snap["ingress_res"] == "1920x1080"
    assert snap["ingress_convert_ms_max"] == 3.5
    # Contiguous 15fps: worst gap is ~one frame interval.
    assert snap["ingress_max_gap_ms"] < 200


def test_stall_reads_as_growing_gap_and_decaying_fps() -> None:
    m = IngressMeter()
    last = _feed(m, 1000.0, n=30, fps=15.0)
    healthy = m.snapshot(last)
    assert healthy is not None and healthy["ingress_max_gap_ms"] < 200
    # 5s of silence: the tail gap counts, fps decays instead of freezing.
    stalled = m.snapshot(last + 5.0)
    assert stalled is not None
    assert stalled["ingress_max_gap_ms"] >= 5000
    assert stalled["ingress_recv_fps"] < healthy["ingress_recv_fps"]


def test_window_forgets_old_traffic() -> None:
    m = IngressMeter(window_seconds=15.0)
    # 10s of slow 2fps traffic, then 5s of 12fps: the snapshot must be
    # dominated by what the window still holds, and once the slow era
    # ages out entirely the reading converges to the fast rate.
    mid = _feed(m, 1000.0, n=20, fps=2.0)
    now = _feed(m, mid, n=60, fps=12.0)
    later = m.snapshot(now + 9.0)  # slow era fully outside the window
    assert later is not None
    assert later["ingress_recv_fps"] >= 4.0
    # And a fully-idle window goes back to None, not stale numbers.
    assert m.snapshot(now + 40.0) is None


def test_cold_or_single_frame_returns_none() -> None:
    m = IngressMeter()
    assert m.snapshot(5.0) is None
    m.record(5.0, accepted=True, convert_ms=1.0, width=640, height=360)
    assert m.snapshot(5.1) is None


def _ingest() -> CameraIngest:
    # Same minimal-settings pattern as test_wake_verify: _stamp_ingress
    # touches only the meter, the buffer's target and the log tag.
    return CameraIngest(
        camera_id=uuid4(),
        patient_id=uuid4(),
        sessionmaker=None,
        motion_cache=MotionCache(ttl_seconds=60),
        settings=SimpleNamespace(
            frame_buffer_retention_seconds=20.0,
            livekit_worker_target_fps=4,
            frame_buffer_target_fps=4,
            clip_fps=4,
        ),
        reid=None,
    )


def test_stamp_ingress_cold_meter_stamps_nothing() -> None:
    fields: dict = {"skipped_reason": "camera_closed"}
    _ingest()._stamp_ingress(fields)
    assert fields == {"skipped_reason": "camera_closed"}


def test_stamp_ingress_adds_row_fields_when_fed() -> None:
    ing = _ingest()
    now = time.monotonic()
    for i in range(30):
        ing.ingress_meter.record(
            now - 2.0 + i / 15.0,
            accepted=(i % 4 == 0),
            convert_ms=1.5,
            width=640,
            height=360,
        )
    fields: dict = {}
    ing._stamp_ingress(fields)
    assert set(fields) == {
        "ingress_recv_fps",
        "ingress_kept_fps",
        "ingress_max_gap_ms",
        "ingress_convert_ms_max",
        "ingress_res",
    }
    assert fields["ingress_res"] == "640x360"
    assert fields["ingress_recv_fps"] > fields["ingress_kept_fps"]
