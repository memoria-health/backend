# Shared cameras and identity (2026-08-20)

What changed, why, and what is deliberately still open. Written because
the interactions here are not obvious from any one file.

## The physical layout, which forces everything else

The GPU node runs **one process per LiveKit room**. Each process joins
its room and watches whatever publishes into it.

A camera publishes into **exactly one** room. It is a video stream; it
has one destination. Which room is decided when the camera is assigned
and stored on the camera row as `livekit_room`.

So a lounge camera covering Alice and Bob arrives at **one** process.
It cannot arrive at two without decoding the same stream twice, which
is what would eat the ~15-camera-per-node ceiling.

## The two problems that follow

**1. The process only knew one person.** It loaded Alice's enrolled
face and compared every body to her. Bob walked through, scored low,
and the answer was "not Alice" — it never asked "is this Bob?", because
it had never heard of Bob.

**2. The backend asked the wrong process.** It resolved "which process
serves Bob" and asked that one, but the lounge camera's video went to
**Alice's** process. Bob's process returns "I don't know that camera",
which is indistinguishable from the service being down, so it correctly
fails open — silently, forever. Even with (1) fixed, the right answer
would be computed and thrown away.

## What was built

**The node asks N binary questions instead of one.** For each resident
the camera covers, score this track against that resident's gallery,
exactly the pre-existing math. Take the highest above the floor. Below
it for everyone, the answer is nobody.

Two arbitrations, kept deliberately separate:

| Question | Where | Changed? |
|---|---|---|
| which track is Alice | `seat.update_seat` | no |
| which resident is track 7 | `seat.resolve_track_conflicts` | new |

With one resident the second has nothing to arbitrate and returns its
input, which is why the single-patient path is unchanged. That property
is pinned by tests, not assumed —
`test_identity_seat_multipatient.py::test_one_resident_is_a_no_op` and
`test_cross_clip_body_gallery.py::test_single_patient_path_is_unchanged`.

**Routing follows the camera.** `IdentityRouter` resolves the process
from `cameras.livekit_room`. That is an address lookup and nothing in
it decides who anyone is. The room name embeds a patient uuid for
historical reasons; nothing is allowed to read meaning into that.

**Nothing falls back to a default person.** The places that could have,
and what they do instead:

- `gallery.match` — below threshold for everyone returns None.
- `IdentityClient._seat_for` — strict `seats[patient_id]`; a patient
  absent from `seats` is "not located", never another patient's seat.
- `CameraIngest` promotion — the `covered or [self.patient_id]`
  fallback was deleted; empty coverage promotes to nobody.
- Harvest — writes to the pool of the resident who holds the seat.
  A harvest filed under the wrong person poisons that pool for its
  whole 4h TTL.

**Roster liveness reads `livekit_room`.** A room is live when a camera
streams *into* it, not when a camera is *assigned* to its patient.
Those differ exactly when a camera is shared: the second resident is
covered by a streaming camera while no camera streams into their room.
Under the old rule they were handed a slot, a port and a full set of
models for a process with zero cameras in it.

**The caregiver app joins the camera's room.** It used to build
`patient_<viewed patient>`, so a shared camera in the second resident's
tab connected to a room that camera never appears in and showed a
permanently empty grid. `CameraResponse` now exposes `livekit_room` and
`NemaraAPIClient.liveRoom(forPatient:)` resolves it.

## What died

`PATIENT` / `OTHER` labels in the body gallery, with them the
asymmetric 0.80 OTHER threshold, the separate OTHER cap, the 2h OTHER
TTL, and the "prefer PATIENT within 0.05" tie-break. The identity node
harvested `LABEL_PATIENT` only and ignored any non-PATIENT verdict, so
none of it was reachable. Removing it is also what makes the
multi-patient question well-posed: "not this patient" is now just
"below the threshold".

`--patient` on `identity.service`, and the hardcoded pilot UUID that
was its argparse default.

## Deploy

`state-v2` is a breaking wire change and there is **no v1 compatibility
branch**, deliberately. A v1 node reports a single `seat` from whichever
patient that process served; now that the router picks the process by
camera, trusting that field would attribute one resident's lock to
another. A brief fail-open window is the safer failure — it is loud in
logs and it only costs identity gating, whereas a silent wrong
attribution writes someone else's record.

So: DB delta, then backend, then `git tag gpu-vN` → CI → `pod.sh
release gpu-vN` → `up` → `health`, then terminate the old pod. Identity
is fail-open between the backend and node deploys.

## Open, and deliberately not fixed here

**A second caregiver would leak cameras across residents.** LiveKit
tokens grant `can_subscribe` on the whole room with no per-track
restriction ([livekit.py:135](../../src/api/livekit.py#L135)), and the
join check reads the patient named in the room, not the camera's
coverage ([livekit.py:265](../../src/api/livekit.py#L265)). So a
caregiver admitted to a room to watch a shared lounge can also
subscribe to every private camera in that room — including bedrooms.

Harmless while one caregiver owns every patient, which is the current
product decision. But `patient_members`, `add_member` and
`accessible_patient_ids` are built and live, so per-patient sharing
works today if anyone uses it, and the leak arms the moment a second
caregiver is added to one patient in a multi-patient account.

The structural fix is to make a room mean one privacy scope — key it on
the coverage set, so a shared camera gets its own room — and to
decouple rooms (permissions) from processes (capacity) so extra rooms
cost a subscription rather than a whole process. Not done here because
it re-keys the roster that shipped 2026-08-19.

**A patient whose cameras span two rooms cannot be fully shown in the
app.** Needs them to own both a private camera and a shared one. The
client holds one room at a time; `liveRoom` picks the room covering
most of their cameras and returns the rest as `unreachable` so the
caller can say so rather than render a blank tile. Nothing consumes
`unreachable` yet.

**Accuracy is a measure-then-tune item.** An unenrolled stranger now
gets N chances to clear the bar instead of one, so false accepts scale
with the number of enrolled residents on a camera. Inherent to asking a
multi-way question. The floor, top-K history and provisional gate are
unchanged.

**Probe cadence prefers cost over speed on shared cameras.** Dense
while nobody is located, light once *anyone* is. The stricter reading
(everyone the camera covers) would hold a shared camera at search
cadence whenever one resident is out of the house, multiplying
face-probe cost for someone who is not there.
