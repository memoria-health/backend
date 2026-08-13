"""Persistent per-camera continuous-tracking session — perception core of
the EdgeTAM-continuous architecture (2026-07-22).

Cascade per camera:
  - SAM3 grounds "person" PERIODICALLY (every `reground_every` ticks, and on
    the first tick). SAM3 multiplex needs ~6 frames of temporal context — a
    single frame grounds nothing — so we ground over the tick's first frames
    and take frame 0's masks.
  - EdgeTAM PROPAGATES the tick's frames between grounds at ~36ms/frame.

Track ids stay STABLE across ticks two ways:
  1. mask-carry — a non-ground tick seeds EdgeTAM from the PREVIOUS tick's
     final masks under the SAME ids (validated: 0 lost, ~0px drift when the
     tick windows overlap by a frame).
  2. IoU re-map — a ground tick maps SAM3's fresh masks back onto existing
     persistent ids by IoU with the carried masks; unmatched masks get new
     ids. So a re-ground refreshes geometry without renumbering anyone.

One session per camera in a process-global store. Self-contained (sam3 +
sam2/edgetam + torch + cv2 + numpy) so one pod process hosts both models.
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("perception.continuous")


class SessionCapacityError(RuntimeError):
    """Raised when a NEW camera session would exceed the node's cap and
    no idle session can be evicted. The server maps this to a 429 so
    identity fails open on that camera — bounded degradation, never the
    unbounded VRAM growth that killed perception on 2026-08-13."""

_EDGE = None
_edge_lock = threading.Lock()
_SESSIONS: "dict[str, CameraSession]" = {}
_sessions_lock = threading.Lock()

_EDGE_CKPT = os.environ.get(
    "NEMARA_EDGE_CKPT", "/workspace/edgetam/checkpoints/edgetam.pt",
)
_EDGE_CFG = os.environ.get("NEMARA_EDGE_CFG", "configs/edgetam.yaml")
_EDGE_ROOT = os.environ.get("NEMARA_EDGE_ROOT", "/workspace/edgetam")
# IoU floor to consider a re-grounded SAM3 mask the SAME person as a carried
# id. Below it → a genuinely new person → new id.
_REMAP_IOU = float(os.environ.get("NEMARA_EDGE_REMAP_IOU", "0.30"))
# Optimization: EdgeTAM is built for small frames (16fps on an iPhone), so
# we track at a reduced width and scale bboxes back to full res on output.
# The face pass still uses the full-res crop (server mask_crop resizes the
# track-res mask up), so identity quality is untouched. 0 disables.
# (This constant's original note blamed "init_state RE-ENCODING every
# frame" — measured 2026-08-01 and confirmed: that WAS 68% of every call.
# It is now fixed at the source, see frame_tensor.py, so the width cap is
# no longer load-bearing for latency — only for what SAM3/EdgeTAM see.)
_TRACK_W = int(os.environ.get("NEMARA_EDGE_TRACK_W", "640"))
# Ticks between SAM3 re-grounds. Higher = fewer expensive ground ticks, but
# slower to catch a NEW person → keep modest until occupancy-triggered
# re-ground lands.
_REGROUND_EVERY = int(os.environ.get("NEMARA_EDGE_REGROUND", "5"))
# Debug: if set, dump per-frame mask overlays for each tick to this dir so we
# can SEE what EdgeTAM's mask does (follow the person, or drift to the sofa).
_DEBUG_DIR = os.environ.get("NEMARA_EDGE_DEBUG_DIR", "")
# Tracker backend: "edgetam" (default, the revert) = SAM3-ground + EdgeTAM-
# carry; "sam3" = SAM3's own tracker over the WHOLE clip (swap-robust
# within-clip ids from its memory) + IoU stitch across clips. Toggle with
# env NEMARA_TRACKER; edgetam is the one-var revert.
_TRACKER = os.environ.get("NEMARA_TRACKER", "edgetam").strip().lower()
# sam3 mode: track every Nth clip frame. SAM3 propagation cost is ~linear in
# frames tracked, and indoor motion is slow, so stride 2 on an 8-frame clip
# (~4fps) roughly HALVES the ~2.2s/clip cost with negligible tracking loss.
# Set NEMARA_SAM3_STRIDE=1 for max robustness. Output positions are remapped
# to ORIGINAL frame indices so the server's per-frame face pass stays aligned.
_SAM3_STRIDE = max(1, int(os.environ.get("NEMARA_SAM3_STRIDE", "1")))

# NOTE: the 2026-07-22 drift guard (mask balloon/shape heuristics vs the
# last ground's area) was PURGED 2026-07-24 — it mistook a person walking
# toward the camera for a smear and killed live tracks. Regrounds are
# the sole existence authority; a sticky-mask phantom dies at the next
# ground (≤reground_every ticks) when the grounder finds nobody there.

# Grounder selection: "sam3" (production — Aryaman's 2026-07-26
# verdict after yoloseg fragmented a recumbent person into two
# instances); "yoloseg" = YOLO26-seg person MASKS seed EdgeTAM via the
# exact same mask-IoU remap + add_new_mask path — same matching
# currency, ~140ms vs SAM3's ~2.4s fresh look — switchable pending a
# containment-dedup pass. (A third mode that seeded BOXES was deleted
# 2026-07-26: rectangles are the wrong matching currency for adjacent
# people — 40-60% box overlap where silhouettes share ~5%; it swapped
# ids live.) Grounder failures fall back to SAM3 per-tick.
_GROUNDER = os.environ.get("NEMARA_GROUNDER", "sam3").strip().lower()

# Width cap on the frames FED TO SAM3 grounding (2026-07-28). SAM3's
# instance decisions are input-scale-sensitive: the proven soak weeks ran
# ≤1280-wide input; the Jul-27 1080p gateway deploy silently moved it to
# 1920, where a recumbent/seated person intermittently splits into two
# instances (phantom births 8-13/hr in occupied rooms, YOLO trigger
# dets=1 throughout, plus a forced-reground loop arguing about it).
# Masks are resized to track-res downstream either way — only SAM3's view
# changes. 0 = native (the misbehaving regime); env knob is for live A/B.
_SAM3_GROUND_W = int(os.environ.get("NEMARA_SAM3_GROUND_W", "1280"))


def _track_dims(w: int, h: int) -> tuple[int, int, float]:
    """(track_w, track_h, scale) for a full-res frame. scale = track/orig."""
    if _TRACK_W <= 0 or w <= _TRACK_W:
        return w, h, 1.0
    s = _TRACK_W / float(w)
    return _TRACK_W, max(1, round(h * s)), s


def _get_edge():
    """Process-singleton EdgeTAM (sam2) video predictor."""
    global _EDGE
    if _EDGE is None:
        with _edge_lock:
            if _EDGE is None:
                import sys
                if _EDGE_ROOT not in sys.path:
                    sys.path.insert(0, _EDGE_ROOT)
                from sam2.build_sam import build_sam2_video_predictor
                _EDGE = build_sam2_video_predictor(
                    _EDGE_CFG, _EDGE_CKPT, device="cuda",
                )
                log.info("continuous.edgetam_loaded ckpt=%s", _EDGE_CKPT)
    return _EDGE


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        # Cross-resolution comparison (input res changed mid-session):
        # rescale instead of declaring zero overlap — a silent 0.0 makes
        # every carried track unmatchable and full-rebirths the room.
        import cv2
        b = cv2.resize(
            b.astype(np.uint8), (a.shape[1], a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _mask_bbox(m: np.ndarray):
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not rows.any() or not cols.any():
        return None
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return (float(xs[0]), float(ys[0]), float(xs[-1]) + 1.0, float(ys[-1]) + 1.0)


def _dump_overlay(frames, out, camera_id, tick, outdir) -> None:
    """Debug: save per-frame mask overlays for one tick, so we can SEE what
    EdgeTAM's mask does (follow the person vs drift onto the sofa)."""
    import cv2

    d = os.path.join(outdir, f"{str(camera_id)[:8]}_t{tick:04d}")
    os.makedirs(d, exist_ok=True)
    palette = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]
    for fidx in sorted(out):
        if fidx >= len(frames) or frames[fidx] is None:
            continue
        img = frames[fidx].copy()
        h, w = img.shape[:2]
        for oid, (bb, mm) in out[fidx].items():
            m = mm
            if m.shape[:2] != (h, w):
                m = cv2.resize(
                    m.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            col = palette[(oid - 1) % len(palette)]
            ov = img.copy()
            ov[m] = col
            img = cv2.addWeighted(img, 0.6, ov, 0.4, 0)
            x0, y0, x1, y1 = (int(v) for v in bb)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            cv2.putText(img, f"id{oid}", (x0, max(20, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"t{tick} f{fidx}", (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(d, f"{fidx:03d}.jpg"), img)


class CameraSession:
    """Per-camera continuous session: stable ids across ticks."""

    def __init__(self, camera_id: str, *, reground_every: int = 5) -> None:
        self.camera_id = camera_id
        self.reground_every = reground_every
        self.tick = 0
        self.last_ground = -(10 ** 9)
        self.next_id = 1
        # Session epoch: track ids are only meaningful WITHIN one
        # session lifetime — a server restart resets next_id and
        # recycles numbers (live 2026-07-25: "how was I born again as
        # T13?"). The service resets its state when this changes, so
        # a recycled number can never inherit the patient seat.
        import uuid as _uuid
        self.epoch = _uuid.uuid4().hex[:12]
        # Serialises calls for THIS camera only. Until 2026-08-01 the
        # pod held one global lock, so two requests for one camera were
        # structurally impossible; now that the gate admits several
        # cameras at once, the only thing left preventing it is the
        # client sending one batch per camera at a time. That is a
        # promise made in ANOTHER PROCESS, and everything below is
        # read-modify-write: tick, next_id, last_masks, and EdgeTAM's
        # session itself. A retry, a second identity instance, or a
        # future client that pipelines would interleave two ticks and
        # scramble track ids. Different cameras still run concurrently
        # — this lock is per session, which is exactly the granularity
        # the work has.
        self._lock = threading.Lock()
        # persistent track_id -> last final mask (frame-0 resolution, bool)
        self.last_masks: "dict[int, np.ndarray]" = {}
        # consecutive carry ticks where the person trigger saw FEWER
        # people than the carry (see person_trigger.should_force_reground)
        self._under_streak = 0

    # ── SAM3 grounding (periodic) ──────────────────────────────────
    def _ground(self, frames, person_thr, sam_pred) -> "dict[int, np.ndarray]":
        import cv2
        import torch
        from perception import sam_track

        ctx = list(frames[: min(6, len(frames))])
        if _SAM3_GROUND_W > 0 and ctx and ctx[0].shape[1] > _SAM3_GROUND_W:
            s = _SAM3_GROUND_W / float(ctx[0].shape[1])
            ctx = [
                cv2.resize(
                    f, (_SAM3_GROUND_W, max(1, round(f.shape[0] * s))),
                    interpolation=cv2.INTER_AREA,
                )
                for f in ctx
            ]
        # Frame 0's masks are the ONLY thing this function uses, so ask
        # for exactly that. The old shape opened a propagation stream
        # and broke at the first yield — by which point SAM3 had already
        # tracked all six context frames and we discarded five of them,
        # for 1.4s of every ground tick (see sam_track.ground_frame0).
        with torch.inference_mode():
            got: list = sam_track.ground_frame0(
                sam_pred, ctx, concept="person", thr=person_thr,
            )

        h, w = frames[0].shape[:2]
        tw, th, _scale = _track_dims(w, h)
        ms: "list[np.ndarray]" = []
        for (_tid, _bbox, mask, _prob) in got:
            m = np.asarray(mask) > 0.5
            # SAM3 mask -> TRACK res so it matches last_masks + what EdgeTAM
            # sees (the whole session works in track-res space).
            if m.shape != (th, tw):
                m = cv2.resize(
                    m.astype(np.uint8), (tw, th),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            ms.append(m)
        return self._remap_masks(ms)

    def _remap_masks(
        self, masks: "list[np.ndarray]",
    ) -> "dict[int, np.ndarray]":
        """Map fresh track-res grounding masks onto persistent ids by
        mask-IoU vs the carried masks — best overlap ≥ _REMAP_IOU keeps
        the id, otherwise a new id. Shared by every mask-currency
        grounder (SAM3 concept, YOLO-seg): the matching contract lives
        HERE, not per-grounder, so grounder choice never changes
        matching behavior."""
        seeded: "dict[int, np.ndarray]" = {}
        used_ids: "set[int]" = set()
        for m in masks:
            best_id, best_iou = None, _REMAP_IOU
            for pid, pm in self.last_masks.items():
                if pid in used_ids:
                    continue
                i = _iou(m, pm)
                if i > best_iou:
                    best_id, best_iou = pid, i
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used_ids.add(best_id)
            seeded[best_id] = m
        return seeded

    # ── YOLO-seg grounding (NEMARA_GROUNDER=yoloseg, 2026-07-25) ──
    def _ground_yoloseg(self, frames):
        """Person MASKS from the YOLO26 seg head seed EdgeTAM (~140ms
        vs SAM3's ~10s fresh look). Same currency as SAM3 grounding —
        masks — so the IoU remap and the add_new_mask seeding path are
        identical; only the mask source changes. (The 07-24 box trial
        failed on currency, not detection: adjacent people's rectangles
        overlap 40-60% where silhouettes share ~5%.) Returns
        {id: mask_trackres} or None (detector dead / det-only weights
        → caller falls back to SAM3). An empty dict is authoritative:
        YOLO saw nobody."""
        import cv2

        from perception import person_trigger

        masks = person_trigger.person_masks(frames[0])
        if masks is None:
            return None
        h, w = frames[0].shape[:2]
        tw, th, _scale = _track_dims(w, h)
        ms = []
        for m in masks:
            if m.shape != (th, tw):
                m = cv2.resize(
                    m.astype(np.uint8), (tw, th),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if m.any():
                ms.append(m)
        return self._remap_masks(ms)

    # ── EdgeTAM propagation (every tick) ───────────────────────────
    def _propagate(self, frames, seed) -> "dict[int, dict]":
        import cv2
        import torch

        if not seed:
            return {}
        from perception import frame_tensor

        edge = _get_edge()
        h, w = frames[0].shape[:2]
        tw, th, scale = _track_dims(w, h)
        inv = 1.0 / scale if scale else 1.0
        # Track-res frames, in memory. These used to be JPEG-encoded into
        # a temp dir so init_state could PIL-open and decode them back one
        # at a time — 68% of every call, 98.7 ms/frame (see
        # frame_tensor.py). Now they go to the GPU as one batched op.
        tracked = [
            fr if (tw, th) == (w, h) else cv2.resize(fr, (tw, th))
            for fr in frames
        ]
        out: "dict[int, dict]" = {}
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
        ), frame_tensor.staged_frames(
            tracked, edge, "cuda" if torch.cuda.is_available() else "cpu",
        ):
            # Path is ignored while frames are staged; kept only
            # because init_state's signature requires it.
            state = edge.init_state("<staged>")
            # Encode all frames in one pass rather than N passes of 1
            # (measured 1.166x end-to-end, masks unchanged). Fails safe:
            # on any problem the cache is empty and upstream encodes per
            # frame exactly as before.
            frame_tensor.prime_feature_cache(edge, state)
            # Seeds are ALWAYS masks — the one matching currency
            # (SAM3 concept masks or YOLO-seg retina masks; the
            # 2026-07-24 box-prompt path was deleted with its
            # grounder: rectangles were the wrong currency for
            # adjacent people).
            for tid, m in seed.items():
                edge.add_new_mask(
                    state, frame_idx=0, obj_id=int(tid),
                    mask=torch.from_numpy(
                        np.ascontiguousarray(m)
                    ).contiguous(),
                )
            for fidx, obj_ids, logits in edge.propagate_in_video(state):
                fr_out: dict = {}
                for j, oid in enumerate(obj_ids):
                    mm = (logits[j, 0] > 0).cpu().numpy()  # track res
                    bb = _mask_bbox(mm)
                    if bb is None:
                        continue
                    # bbox -> FULL res for rows + face crops; the mask
                    # stays track-res (server mask_crop resizes it up).
                    bb_full = (
                        bb[0] * inv, bb[1] * inv, bb[2] * inv, bb[3] * inv,
                    )
                    fr_out[int(oid)] = (bb_full, mm)
                out[fidx] = fr_out
        return out

    # ── SAM3-native tracking (NEMARA_TRACKER=sam3) ─────────────────
    def _track_native(self, frames, person_thr, sam_pred) -> "dict[int, dict]":
        """One tick, SAM3's own tracker over the WHOLE clip.

        SAM3 grounds "person" on frame 0 and propagates across every frame
        with swap-robust, clip-local ids from its memory (no EdgeTAM). We
        STITCH those clip ids onto persistent session ids by IoU of each
        id's first-appearance mask vs the previous clip's final masks — the
        cross-clip binding brought back on purpose. No drift guard: SAM3
        reports absence via its occlusion score, so the couch phantom never
        arises. Same return contract as observe()/_propagate():
        {frame_pos: {track_id: (bbox_xyxy_fullres, mask)}}."""
        import torch
        from perception import sam_track

        clip = list(frames)[::_SAM3_STRIDE]  # subsample; remap pos below
        per_pos: "dict[int, list]" = {}
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
        ):
            for pos, dets in sam_track.track_stream(
                sam_pred, clip, concept="person", thr=person_thr,
            ):
                # Remap subsampled position -> original clip frame index so
                # the server's pos->orig frame mapping stays correct.
                per_pos[int(pos) * _SAM3_STRIDE] = [
                    (int(stid), np.asarray(mask) > 0.5)
                    for (stid, _bbox, mask, _prob) in dets
                ]

        # Stitch clip-local ids onto persistent ids by first-appearance IoU.
        first_mask: "dict[int, np.ndarray]" = {}
        for pos in sorted(per_pos):
            for stid, m in per_pos[pos]:
                first_mask.setdefault(stid, m)
        id_map: "dict[int, int]" = {}
        used: "set[int]" = set()
        for stid, m0 in first_mask.items():
            best_id, best_iou = None, _REMAP_IOU
            for pid, pm in self.last_masks.items():
                if pid in used:
                    continue
                i = _iou(m0, pm)
                if i > best_iou:
                    best_id, best_iou = pid, i
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            id_map[stid] = best_id

        out: "dict[int, dict]" = {}
        for pos, dets in per_pos.items():
            fr: dict = {}
            for stid, m in dets:
                bb = _mask_bbox(m)
                if bb is not None:
                    fr[id_map[stid]] = (bb, m)
            out[pos] = fr

        # Shared tail (mirrors observe()): last_masks carries to the next
        # clip; cleared empty when nobody is tracked so no mask lingers.
        if out:
            last_pos = max(out)
            self.last_masks = {
                tid: mm for tid, (_bb, mm) in out[last_pos].items() if mm.any()
            }
        else:
            self.last_masks = {}
        self.last_ground = self.tick  # every clip is authoritative on existence
        if _DEBUG_DIR:
            try:
                _dump_overlay(frames, out, self.camera_id, self.tick, _DEBUG_DIR)
            except Exception:  # noqa: BLE001
                log.exception("continuous.debug_dump_failed")
        log.info(
            "continuous.tick cam=%s tick=%d mode=sam3 seeded=%d live=%d drift=0",
            self.camera_id, self.tick, len(first_mask), len(self.last_masks),
        )
        self.tick += 1
        return out

    def observe(self, frames, *, person_thr: float, sam_pred) -> "dict[int, dict]":
        with self._lock:
            return self._observe_locked(
                frames, person_thr=person_thr, sam_pred=sam_pred,
            )

    def _observe_locked(self, frames, *, person_thr: float, sam_pred) -> "dict[int, dict]":
        """One tick. Returns {frame_pos: {track_id: (bbox_xyxy, mask)}} with
        STABLE track ids across ticks. NEMARA_TRACKER=sam3 routes to SAM3's
        own whole-clip tracker; default 'edgetam' = SAM3-ground+EdgeTAM-carry."""
        if _TRACKER == "sam3":
            return self._track_native(frames, person_thr, sam_pred)
        reground = (
            self.tick == 0
            or (self.tick - self.last_ground) >= self.reground_every
            or not self.last_masks
        )
        if reground:
            seed = None
            mode = "ground"
            if _GROUNDER == "yoloseg":
                seed = self._ground_yoloseg(frames)
                if seed is None:
                    log.warning(
                        "continuous.yoloseg_ground_failed cam=%s tick=%d "
                        "— SAM3 fallback", self.camera_id, self.tick,
                    )
                else:
                    mode = "ground_ys"
            if seed is None:
                seed = self._ground(frames, person_thr, sam_pred)
            self.last_ground = self.tick
        else:
            seed = dict(self.last_masks)
            mode = "carry"

        out = self._propagate(frames, seed)

        # NO drift guard (purged 2026-07-24, Aryaman's call): its area
        # heuristic could not tell a person approaching the camera from a
        # mask smearing onto the couch, and it killed real walking tracks
        # (live: drift_dropped ids=[8],[9] mid-walk) — a renumber + seat
        # re-bind, worse than the phantom it shortened. SAM3 regrounds
        # remain the sole existence authority: a genuine phantom gets no
        # detection at the next ground (≤reground_every ticks) and dies
        # there.

        # last_masks reflects THIS tick's result. Crucially it is CLEARED to
        # empty when the tick produced no tracks — a ground that found nobody
        # (SAM3 seeded=0) or a carry that lost the person. Without this,
        # EdgeTAM (sticky by design) carries a departed person's mask forever
        # — the "green box on the empty sofa" phantom. A ground is
        # authoritative on existence: if SAM3 says nobody's here, nobody is.
        if out:
            last_pos = max(out)
            self.last_masks = {
                tid: mm for tid, (_bb, mm) in out[last_pos].items() if mm.any()
            }
        else:
            self.last_masks = {}

        # Reground trigger (2026-07-24): a cheap YOLO26 person count on
        # the newest frame of every CARRY tick. Count disagreement with
        # the carry forces a SAM3 reground next tick — new entries appear
        # in ~1 tick instead of waiting out the reground interval (~50s
        # blind spot, measured live). The trigger never touches tracks.
        dets = None
        if frames:
            from perception import person_trigger

            # Run on EVERY tick (not just carry): ground ticks pair the
            # YOLO count with SAM3's authoritative verdict on the same
            # scene — the passive agreement dataset that decides whether
            # YOLO earns grounding duty (and the GPU-rental endgame).
            dets = person_trigger.count_persons(frames[-1])
        if mode == "carry":
            force, self._under_streak = person_trigger.should_force_reground(
                dets, len(self.last_masks), self._under_streak,
            )
            if force:
                self.last_ground = -(10 ** 9)
                log.info(
                    "continuous.trigger cam=%s tick=%d dets=%s live=%d "
                    "— forcing reground", self.camera_id, self.tick,
                    dets, len(self.last_masks),
                )
        if _DEBUG_DIR:
            try:
                _dump_overlay(frames, out, self.camera_id, self.tick, _DEBUG_DIR)
            except Exception:  # noqa: BLE001
                log.exception("continuous.debug_dump_failed")
        log.info(
            "continuous.tick cam=%s tick=%d mode=%s seeded=%d live=%d dets=%s",
            self.camera_id, self.tick, mode, len(seed), len(self.last_masks),
            dets,
        )
        self.tick += 1
        return out


# Concurrent-session ceiling. Each live session holds an EdgeTAM memory
# bank; 10 measured survivable (thrashing but alive, 2026-08-13 ladder),
# and the SLA is already gone by 10 — the cap is a backstop against
# unbounded growth, not a performance rating (that is 4, surge 6-8; see
# docs/IDENTITY-GPU-ARCHITECTURE.md).
_MAX_SESSIONS = int(os.environ.get("NEMARA_MAX_SESSIONS", "10"))
# A session idle this long belongs to a camera that stopped (nothing
# calls drop_session in the normal flow) — evict it before refusing a
# newcomer, so churn in camera fleets self-heals.
_SESSION_IDLE_EVICT_S = float(os.environ.get("NEMARA_SESSION_IDLE_S", "180"))


def get_session(camera_id: str, *, reground_every: int = _REGROUND_EVERY) -> CameraSession:
    with _sessions_lock:
        s = _SESSIONS.get(camera_id)
        if s is None:
            if len(_SESSIONS) >= _MAX_SESSIONS:
                now = time.time()
                idle = [
                    cid for cid, cs in _SESSIONS.items()
                    if now - getattr(cs, "last_used", 0) > _SESSION_IDLE_EVICT_S
                ]
                for cid in idle:
                    _SESSIONS.pop(cid, None)
                    log.info("continuous.session_evicted_idle cam=%s", cid)
            if len(_SESSIONS) >= _MAX_SESSIONS:
                raise SessionCapacityError(
                    f"{len(_SESSIONS)} live sessions at cap "
                    f"{_MAX_SESSIONS} — refusing new camera {camera_id}",
                )
            s = CameraSession(camera_id, reground_every=reground_every)
            _SESSIONS[camera_id] = s
        s.last_used = time.time()
        return s


def drop_session(camera_id: str) -> None:
    with _sessions_lock:
        _SESSIONS.pop(camera_id, None)
