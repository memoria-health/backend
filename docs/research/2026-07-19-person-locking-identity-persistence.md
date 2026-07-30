# Persistent person identity in home video — literature + industry survey

> Status (2026-07-27): proposals shipped in two waves. The §(d) recommendations (sticky hysteresis lock, multi-prototype pool gallery with max/top-k matching, multi-pose harvest per face lock, carried-tracklet TTL) landed in the 2026-07-19 identity-persistence layer. The underlying problem this survey addresses — identity re-decided per 10s clip — was then eliminated at the root by the EdgeTAM-continuous pivot (2026-07-22): the pod holds one persistent tracking session per camera with stable track ids (`gpu/perception/continuous.py`), and the backend binds once and holds (`src/services/identity/track_ledger.py`), so cross-clip tracklet re-association no longer exists. Kept as the evidence record behind both designs.

> Researched 2026-07-19 (night of soak #2), prompted by Bedroom PATIENT↔UNKNOWN
> label flicker: identity is re-decided from scratch every 10 s tick; the only
> cross-tick memory (OSNet body gallery) fragmented into 11 singleton clusters
> and scores oscillate around the hard 0.72 threshold.

## (a) The four canonical mechanisms

**1. Tracklet linking / re-association (not longer-lived trackers).** SOTA
systems treat a track break as normal and repair it afterward: short tracklets
are matched across gaps using position, motion continuity, and appearance, then
merged, often by hierarchical clustering with a distance threshold
([Hierarchical Deep Tracklet Re-ID](https://arxiv.org/abs/1811.04091),
[GTA](https://arxiv.org/html/2411.08216v1)). NVIDIA DeepStream productizes this
as "shadow tracking": a lost target is kept alive invisibly for
`maxShadowTrackingAge` frames while new tracklets are tested against it
([nvtracker docs](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)).
A 2026 thermal-surveillance study found identity continuity is "primarily
governed by conservative spatial-temporal relinking rather than online tracker
complexity" (+2.7 IDF1 from relinking alone)
([arXiv 2606.01694](https://arxiv.org/html/2606.01694v1)).

**2. Appearance galleries with max-similarity matching, plus EMA smoothing
inside a track.** Within a live track, embeddings are EMA-smoothed (BoT-SORT
alpha=0.9 ([Ultralytics](https://docs.ultralytics.com/reference/trackers/bot_sort));
Deep OC-SORT alpha=0.95, adaptively raised toward 1.0 on low-confidence
detections so occluded/blurry crops are ignored
([arXiv 2302.11813](https://arxiv.org/pdf/2302.11813))). Across breaks, systems
keep a *gallery of many embeddings per identity* and match on the **max**
cosine over the gallery, not against one averaged prototype: DeepStream stores
up to `reidHistorySize=100` vectors per target. Unsupervised ReID likewise
maintains per-cluster memory updated with momentum, and clusters on
k-reciprocal Jaccard distance rather than raw cosine because raw cosine is
unreliable cross-view ([Cluster Contrast](https://arxiv.org/pdf/2103.11568)).

**3. Spatio-temporal priors ("nobody teleports").** st-ReID adds a
transition-time/location probability to appearance score and prunes most
impossible matches ([arXiv 1812.03282](https://arxiv.org/abs/1812.03282)); the
thermal paper found spatial proximity was the *dominant* cue, above appearance
and motion. In a closed-world home with 1-3 known occupants this prior is far
stronger than in public surveillance.

**4. Face anchors, body carries.** Sparse high-confidence face hits are
propagated along tracklets
([label propagation for video face recognition](https://ieeexplore.ieee.org/document/6976772)).
Industry does exactly this: Google Nest Familiar Faces "uses additional signals
(body size, clothing color, etc.) when a subject's face is not visible"
([Google support](https://support.google.com/googlehome/answer/9268625?hl=en));
healthcare/digital-home research fuses face + body color/shape with
**track-based majority voting**
([PMC4179041](https://pmc.ncbi.nlm.nih.gov/articles/PMC4179041/)).
Clothes-changing ReID confirms body appearance is outfit-dominated, so the body
model must be refreshed per outfit/day; durable cues are shape and gait
([LTCC](https://arxiv.org/pdf/2005.12633)).

## (b) Per-clip re-decision vs sticky state

Per-frame/per-clip re-decision is **not** standard; everything strong holds
sticky state with asymmetric enter/exit. Frigate assigns a name only after
consistent, face-area-weighted recognition (`unknown_score 0.8` to even count,
`recognition_threshold 0.9` to assign — an explicit two-threshold scheme), then
the sub_label persists for the tracked object's lifetime
([Frigate docs](https://docs.frigate.video/configuration/face_recognition/)).
An edge-perception paper formalizes hysteresis label locking: promote to
"locked" after ≥3 votes in a 5-frame window with mean confidence >60%; escape
only on sustained contradictory evidence with *higher* mean confidence; locking
alone recovered **+10.2 pp accuracy** over per-frame decisions
([arXiv 2606.17241](https://arxiv.org/html/2606.17241)).

## (c) Concrete numbers found

- BoT-SORT EMA alpha 0.9; Deep OC-SORT 0.95 confidence-adaptive. Track buffer
  for lost tracks: 30 frames (ByteTrack/BoT-SORT defaults; ≈6-7.5 s at 4-5 fps)
  ([Ultralytics config](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/trackers/botsort.yaml)).
- DeepStream: gallery 100 embeddings/target, cosine match threshold
  `minMatchingScore4ReidSimilarity=0.8` (max over gallery),
  `maxShadowTrackingAge` 38-42 frames, re-association search window 20 frames
  ([nvtracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html),
  [RidgeRun](https://developer.ridgerun.com/wiki/index.php/ReID_and_Target_Reassociation_using_NVIDIA_Deepstream)).
- OSNet cosine geometry: adjacent-frame same-person sims run ≥0.8;
  cross-viewpoint same-person can drop below different-person-same-pose
  similarity (torchreid issues
  [#568](https://github.com/KaiyangZhou/deep-person-reid/issues/568),
  [#483](https://github.com/KaiyangZhou/deep-person-reid/issues/483) report
  poorly separated distributions). Sports tracking uses staged cosine-distance
  gates 0.1/0.2/0.4 (sim 0.9/0.8/0.6)
  ([arXiv 2209.13154](https://arxiv.org/pdf/2209.13154)).
- Thermal relinking: gap ≤7 frames auto-repair; ≤30-frame stitching with
  velocity checks; ≤60-frame relinks accepted only at p≥0.95 mutual-best.
- Enrollment practice: Astro requires 5 face angles
  ([Amazon](https://www.amazon.com/gp/help/customer/display.html?nodeId=G5CVLVD5574Z5GT5));
  Frigate recommends 20-100 images across poses/lighting with blur/size quality
  filtering; Google auto-purges blurry/non-frontal library images
  ([9to5google](https://9to5google.com/2026/05/06/google-home-nest-cam-familiar-faces-improvements/)).

## (d) What maps onto our design

- **Sticky per-camera identity with single-person continuity: yes, clearly
  supported.** Mechanism 2(b)+3: hold PATIENT in a locked state, demote only on
  sustained contradictory evidence (second person appears, or repeated face
  mismatch), never on absence of evidence. Expect the largest flicker reduction
  of any change.
- **Cluster-merge 0.85→0.70: directionally right, but change the mechanism
  too.** 0.85 raw cosine will essentially never merge cross-viewpoint views of
  one person; 0.70 matches observed same-person cross-view ranges but risks
  same-pose inter-person collisions. The literature's answer is multi-prototype
  galleries matched on max similarity (DeepStream keeps 100), merging clusters
  only under a co-anchor (both face-locked) or spatio-temporal continuity, not
  appearance alone.
- **Multi-pose harvesting per face lock: yes, best-practice everywhere.**
  Harvest body embeddings across the whole face-locked track (front/back/side,
  sitting/standing), quality-gate by detection confidence and blur, cap the
  gallery (~50-100), and expire body/clothing prototypes daily (CC-ReID:
  appearance is outfit-bound).
- **Persistent tracker across 10 s gaps: persist tracklet memory, not the
  tracker.** Kalman state is worthless after 2+ s; no system carries motion
  state across such gaps. What production systems persist is exactly what we
  lose at clip boundaries: ended tracklets (last position, label, confidence,
  embedding gallery), then tracklet re-association at next clip start with a
  "nobody teleports" position gate plus max-gallery cosine. Standard buffers
  (30-60 frames, DeepStream 38-42) equal roughly one clip at our fps, so a
  one-to-two-clip TTL on carried tracklets is squarely in line with deployed
  configs.
