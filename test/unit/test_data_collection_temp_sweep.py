"""Recorder tempfiles must not outlive a hard-killed process.

The recorder writes raw video and a raw PCM side-file to a temp folder
and unlinks them once the clip is uploaded. A SIGKILL or OOM kill skips
that cleanup, leaving RAW UNCENSORED footage on disk with nothing to
remove it. Event clips have swept their scratch folder since 2026-07-29;
the recorder never did, so orphans accumulated forever, up to a
session's worth (hundreds of MB) per crash.
"""

from __future__ import annotations

from pathlib import Path

from src.services.data_collection import manager as recorder_manager


def test_sweep_removes_stranded_tempfiles(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recorder_manager.tempfile, "gettempdir", lambda: str(tmp_path))
    root = recorder_manager.temp_root()

    (root / "abc.video.mp4").write_bytes(b"raw video from a killed process")
    (root / "abc.audio.pcm").write_bytes(b"raw audio")
    (root / "abc.video.final.mp4").write_bytes(b"muxed but never uploaded")

    assert recorder_manager.sweep_temp_dir() == 3
    assert list(root.iterdir()) == []


def test_sweep_on_a_clean_dir_is_a_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(recorder_manager.tempfile, "gettempdir", lambda: str(tmp_path))
    recorder_manager.temp_root()
    assert recorder_manager.sweep_temp_dir() == 0


def test_sweep_survives_a_missing_dir(tmp_path, monkeypatch) -> None:
    """Must never raise into startup."""
    missing = tmp_path / "nope"
    monkeypatch.setattr(
        recorder_manager.tempfile, "gettempdir", lambda: str(missing)
    )
    # temp_root() creates it, so the sweep finds an empty dir, not an error.
    assert recorder_manager.sweep_temp_dir() == 0


def test_room_worker_and_sweep_agree_on_the_directory(tmp_path, monkeypatch) -> None:
    """The sweep is worthless if the recorder writes somewhere else.

    This is the actual bug: room_worker built its own path string, so a
    sweep pointed anywhere but there would have quietly cleaned nothing.
    """
    monkeypatch.setattr(recorder_manager.tempfile, "gettempdir", lambda: str(tmp_path))
    from src.services.detection.room_worker import recorder_temp_root

    assert Path(recorder_temp_root()) == recorder_manager.temp_root()
