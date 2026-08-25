"""Detector registry — load enabled detectors for a camera.

The dashboard caches detectors in-memory and hydrates from the DB on
boot + via Supabase Realtime; for nemara-backend we keep it simple
and read the detectors table per ingest call (small N per patient,
~5 rows). If profiling later shows this is hot, add an in-process
cache here.

## Filtering rules

A detector runs on a tick iff ALL of:
  1. `detectors.is_on = true`  (master kill switch)
  2. `detectors.identifier` is non-NULL  (runtime contract present)
  3. The camera has a row in `cameras_detectors` for it
     — i.e., the caregiver has assigned the detector to this camera.

`always_on=true` detectors (today only `people_present`, which is
synthesized in `prompt.build_prompt_and_schema` and not stored as a
row) bypass all three. That's enforced in the prompt builder, not
here — this function only returns rows that live in the detectors
table.
"""

from collections.abc import Sequence
from dataclasses import replace
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.services.detection import catalog
from src.services.detection.prompt import DetectorSpec, detector_spec_from_row

# Detectors whose prompt is specialized per-camera by the caregiver's exit-door
# selection (stored on cameras_detectors.door_config).
_WANDERING_IDENTIFIERS = {"wandering", "child_wandering"}

# Detectors armed once per camera rather than once per patient (the setup
# self-test). Their arming lives on `cameras.setup_test_passed_at`.
_FIRE_ONCE_IDENTIFIERS = frozenset(catalog.fire_once_keys())


class CameraVanishedError(ValueError):
    """The detector tick referenced a `cameras` row that no longer
    exists. Subclass of ValueError so any existing broad `except
    ValueError` keeps working. CameraIngest._run_tick_safely uses the
    isinstance check to count consecutive vanished-row failures and
    fire its self-destruct callback after N in a row.
    """


async def get_enabled_detectors_for_camera(
    session: AsyncSession, camera_id: UUID,
) -> Sequence[DetectorSpec]:
    """Return the runtime detector specs for `camera_id`.

    Joins through `cameras_detectors` so a detector only runs on a
    camera it has been explicitly assigned to. The assignment is
    seeded for every existing (camera, detector) pair so behavior is
    backward-compatible.

    `people_present` is implicit (merged in by `prompt.build_prompt_and_schema`)
    and `always_on=true` detectors bypass the camera assignment check —
    but at the moment only `people_present` is `always_on`, and it's
    not a DB row, so in practice the join-only filter is sufficient.
    """
    return [spec for spec, _row in await get_enabled_loadout_for_camera(
        session, camera_id,
    )]


async def get_enabled_loadout_for_camera(
    session: AsyncSession, camera_id: UUID,
) -> list[tuple[DetectorSpec, Detector]]:
    """Same filtering + per-camera specialization as
    `get_enabled_detectors_for_camera`, but returns each spec PAIRED
    with its `Detector` row. The detection-v2 scheduler
    (`scheduling/`) needs the row's dimension columns
    (requires_motion / subject_scope / bucket / duty_cycle / shape …)
    alongside the compiled spec; keeping one query + one wandering
    specialization here means the two paths can never drift.
    """
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise CameraVanishedError(
            f"[registry] camera {camera_id} does not exist; cannot resolve detectors"
        )

    # Detectors assigned to THIS camera that are also enabled (is_on=true)
    # and have a runtime contract (identifier non-NULL). `door_config` is the
    # PER-CAMERA wandering exit-door config from the assignment.
    stmt = (
        select(Detector, CameraDetector.door_config)
        .join(CameraDetector, CameraDetector.detector_id == Detector.id)
        .where(
            CameraDetector.camera_id == camera_id,
            Detector.is_on.is_(True),
            Detector.identifier.is_not(None),
        )
    )
    rows = (await session.execute(stmt)).all()

    loadout: list[tuple[DetectorSpec, Detector]] = []
    for det, door_config in rows:
        # Wandering is specialized PER SENSOR by the caregiver's exit-door
        # choice, and runs ONLY once >=1 exit door is set on the assignment.
        # A NULL door_config (room-scope seeding, never touched by the
        # exit-door picker) is the same as unset: not watching this sensor.
        # This matches the app's "inactive until exit doors are set" promise;
        # the old grandfathered generic any-doorway prompt fired on every
        # room exit (2026-07-28 soak: two false events in 2.5 minutes).
        # Setup self-test is armed PER CAMERA: it stops running on the one
        # camera that already passed, and its restart clears the stamp.
        # Other cameras in the home are untouched — each proves its own
        # alert path.
        if det.identifier in _FIRE_ONCE_IDENTIFIERS:
            if cam.setup_test_passed_at is not None:
                continue
            loadout.append((detector_spec_from_row(det), det))
            continue

        if det.identifier in _WANDERING_IDENTIFIERS:
            cfg = door_config or {}
            doors = cfg.get("doors") or []
            if not doors or cfg.get("no_exit_door"):
                continue
            loadout.append((
                replace(
                    detector_spec_from_row(det),
                    prompt_fragment=catalog.build_wandering_fragment(doors),
                ),
                det,
            ))
            continue
        loadout.append((detector_spec_from_row(det), det))
    return loadout
