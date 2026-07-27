"""Sidecar identity provider — implementation A (Aryaman, 2026-07-26).

Drop-in replacement for `ReIDInference` behind
`settings.identity_provider == "sidecar"`: instead of running the
in-process BodyReID brain over the tick mp4, `label_clip` synthesizes
the per-tick `IdentityResult` from the standalone identity service's
`/state` (tools/run_identity.py :8012 — the continuous seat/fusion/
harvest stack that already owns the GPU pod). The backend does ZERO
video identity processing in this mode; `mp4_bytes` is ignored.

Contract obligations (mapped from the live consumer surface):
- The TrackLedger binds/holds the patient ONLY from a non-empty
  `tracks` tuple whose PATIENT entry has `face_locked=True` or a
  gallery pre-label — so the seat is translated into exactly that.
  Seat anchor "face"/face_confirmed → face bind; body-gallery anchor →
  pool bind (`pre_labeled=True, pre_label_source="gallery"`), the same
  trust level the pool gallery had in the legacy brain.
- Fail-open is sacred: ANY failure (service down, timeout, not-ready,
  unknown camera) returns `IdentityResult.empty()` (`reid_ran=False`),
  which bypasses identity-based suppression exactly like a legacy
  pipeline crash. A dead sidecar must never silence detection.
- Camera matching: the identity service keys cameras by the LiveKit
  identity suffix (e.g. "9271358c" — the UUID's first 8 hex chars);
  the backend passes full UUIDs. Prefix match bridges the two.

The /state read is cached for STATE_CACHE_S so N cameras ticking in
the same second share one HTTP round-trip.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

import httpx

from src.services.reid.body_pipeline import (
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    TrackObservation,
)
from src.services.reid.inference import IdentityResult

log = logging.getLogger(__name__)

STATE_CACHE_S = 1.0
# A seat/track snapshot older than this is treated as gone — a wedged
# identity service must decay to fail-open, not freeze a stale lock.
STATE_TIMEOUT_S = 2.0

# Payload sanity caps (Phase 0 checkpoint review finding #2, mirroring
# sam_client's S5 discipline): a corrupt/hostile identity service must
# not be able to allocate unbounded backend memory before fail-open
# kicks in. Real homes have a handful of cameras and single-digit
# simultaneous tracks; these bounds are generous.
_MAX_CAMERAS = 32
_MAX_TRACKS = 64


class SidecarReID:
    """`label_clip`-compatible provider backed by the identity service.

    Constructor kwargs mirror `ReIDInference` (worker.py and tests
    construct with `sessionmaker=`, `settings=`, `body_gallery=`);
    only `settings` is used."""

    def __init__(
        self, *, sessionmaker: Any = None, settings: Any,
        body_gallery: Any = None,
    ) -> None:
        # Tracker-eval trace tap. Legacy armed this inside
        # ReIDInference.__init__, so sidecar mode silently left the
        # trace disabled and even the ledger's live idevent calls
        # no-op'd (live 2026-07-26: tracker-eval card starved). The
        # provider that replaces the brain inherits the arming duty.
        from src.services.reid import trace

        trace.configure(
            enabled=getattr(settings, "reid_trace_enabled", False),
            path=getattr(settings, "reid_trace_path", "out/track_trace.jsonl"),
        )
        self._url = settings.identity_state_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(STATE_TIMEOUT_S),
        )
        self._cache: tuple[float, dict] | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _state(self) -> dict | None:
        now = time.monotonic()
        if self._cache is not None and now - self._cache[0] < STATE_CACHE_S:
            return self._cache[1]
        try:
            resp = await self._client.get(f"{self._url}/state")
            resp.raise_for_status()
            j = resp.json()
        except Exception:  # noqa: BLE001 — fail-open by design
            log.warning("sidecar.state_unreachable url=%s", self._url)
            return None
        if not j.get("ready"):
            return None
        self._cache = (now, j)
        return j

    def timing(self, camera_id: "UUID | str") -> "dict[str, float] | None":
        """REAL pipeline timings for this camera, as measured by the
        identity service on its last batch (not synthesized): e2e_ms =
        capture→processed lag, track_ms = pod time (EdgeTAM carry +
        ground), face_ms = face-probe wire time. Feeds the tracker-eval
        trace so the dashboard reports measured perception latency, not
        the backend's ~5ms /state read (live 2026-07-26: 'sam3 e2e
        5.4ms' was the backend overhead wearing a stale arm label)."""
        if self._cache is None:
            return None
        entry = self._cam_entry(self._cache[1], camera_id)
        if not entry:
            return None
        out: "dict[str, float]" = {}
        if entry.get("lag_s") is not None:
            out["e2e_ms"] = float(entry["lag_s"]) * 1000.0
        pod = (entry.get("stage_ms") or {}).get("pod", entry.get("pod_ms"))
        if pod is not None:
            out["track_ms"] = float(pod)
        faces = (entry.get("probe_ms") or {}).get("faces_wire")
        if faces is not None:
            out["face_ms"] = float(faces)
        return out or None

    @staticmethod
    def _cam_entry(state: dict, camera_id: "UUID | str") -> dict | None:
        """Match the backend's full camera UUID to the service's
        short key ("9271358c" == str(uuid)[:8]) or an exact key."""
        cams = state.get("cameras") or {}
        if len(cams) > _MAX_CAMERAS:
            log.warning("sidecar.cameras_truncated %d", len(cams))
            cams = dict(list(cams.items())[:_MAX_CAMERAS])
        cid = str(camera_id)
        if cid in cams:
            return cams[cid]
        for key, entry in cams.items():
            if key and cid.startswith(key):
                return entry
        return None

    async def label_clip(
        self,
        *,
        mp4_bytes: bytes,
        patient_id: "UUID",
        camera_id: "UUID",
        face_evidence: bool = True,
        harvest_hint: Any = None,
        mode: str = "full",
        tick_seq: int = 0,
        capture_ts: float = 0.0,
    ) -> IdentityResult:
        """Synthesize this camera's IdentityResult from /state.
        `mp4_bytes` and the legacy tuning kwargs are accepted and
        ignored — the sidecar owns its own frame intake."""
        state = await self._state()
        if state is None:
            return IdentityResult.empty(faces_enrolled=True)
        entry = self._cam_entry(state, camera_id)
        if entry is None:
            # Service is up but doesn't know this camera (yet) — no
            # verdict is honest; fail open like a crash would.
            return IdentityResult.empty(faces_enrolled=True)

        seat = state.get("seat")
        seat_here = bool(
            seat and str(camera_id).startswith(str(seat.get("camera", "\x00")))
        )
        try:
            seat_tid = int(seat["track"]) if seat_here and seat else None
        except (KeyError, TypeError, ValueError):
            seat_tid = None
        seat_score = float(seat.get("score", 0.0)) if seat else 0.0
        seat_face = bool(seat.get("face_confirmed")) if seat else False
        seat_anchor = str(seat.get("anchor", "")) if seat else ""

        tracks: list[TrackObservation] = []
        patient_bbox: "tuple[float, float, float, float] | None" = None
        raw_tracks = entry.get("tracks") or {}
        if len(raw_tracks) > _MAX_TRACKS:
            log.warning("sidecar.tracks_truncated %d", len(raw_tracks))
            raw_tracks = dict(list(raw_tracks.items())[:_MAX_TRACKS])
        for tid_s, tv in raw_tracks.items():
            try:
                tid = int(tid_s)
            except (TypeError, ValueError):
                continue
            bbox = tv.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bb = tuple(float(v) for v in bbox)
            is_patient = seat_tid is not None and tid == seat_tid
            # Anchor → ledger bind lane: a face-confirmed seat rides
            # the face lane; a body-anchored seat is exactly a pool-
            # gallery pre-label (that's what built the body gallery).
            face_locked = is_patient and (seat_face or seat_anchor == "face")
            tracks.append(TrackObservation(
                track_id=tid,
                label=LABEL_PATIENT if is_patient else LABEL_UNKNOWN,
                confidence=seat_score if is_patient else 0.0,
                bbox_series=((0, bb),),
                face_locked=face_locked,
                pre_labeled=is_patient and not face_locked,
                pre_label_source=(
                    "gallery" if (is_patient and not face_locked) else None
                ),
            ))
            if is_patient:
                patient_bbox = bb  # type: ignore[assignment]

        return IdentityResult(
            reid_ran=True,
            patient_in_clip=seat_tid is not None,
            confidence=seat_score if seat_here else 0.0,
            best_frame_index=None,
            face_count=1 if (seat_here and seat_face) else 0,
            patient_bbox=patient_bbox,
            people_present_reid=bool(tracks),
            people_count_reid=len(tracks),
            attribution_ran=True,
            tracks=tuple(tracks),
            # Conservative: the sidecar's seat model has one PATIENT
            # seat and UNKNOWN others — never "verified others", so
            # identity-based suppression can never fire from here.
            has_only_verified_other_tracks=False,
            faces_enrolled=True,
        )
