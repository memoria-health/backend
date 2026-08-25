"""Two identity processes in one room must not share a participant name.

LiveKit permits one participant per identity per room: a second join
under the same name DISCONNECTS the first. Under `patient_<id>` rooms
there was exactly one identity service per room, so a constant
"identity-service" was safe. Account rooms put every camera a caregiver
owns in ONE room while the fleet still shards identity PER CAMERA — so
two processes joined as the same participant and evicted each other on
a ~60s cycle. Each survived 35-66s, was kicked (`DuplicateIdentity`),
exited rc=1, and the supervisor restarted it into the same collision.

The visible symptom was elsewhere and looked like a camera fault: the
backend read `sidecar.state_unreachable` in the gaps and every tick came
back `reid_unavailable_llm`, so the second sensor appeared to have no
identity at all. The room worker had already learned this exact lesson
(`RoomWorker._identity`) and made its own name unique; the identity
service never did.
"""

from identity.service import _mint_token

KEY, SECRET = "devkey", "secret"
ROOM = "account_a30c505e-23b8-4efd-8d41-acc17c6b8879"


def _identity_of(jwt: str) -> str:
    """The `sub` claim is the LiveKit participant identity."""
    import base64
    import json

    payload = jwt.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]


def test_two_cameras_in_one_room_get_distinct_names():
    """THE regression: this is the shape the roster starts, one process
    per sensor, both pointed at the same account room."""
    a = _identity_of(_mint_token(KEY, SECRET, ROOM, shard="cam_650ca94d"))
    b = _identity_of(_mint_token(KEY, SECRET, ROOM, shard="cam_c51733e6"))
    assert a != b, "same participant name = they evict each other forever"


def test_the_name_still_starts_with_identity_service():
    """The gateway and the bridge grant track permission by PREFIX
    (`identity-service`). A suffix is free; a rename is not — it would
    silently cost the service its subscription on every private camera."""
    name = _identity_of(_mint_token(KEY, SECRET, ROOM, shard="cam_650ca94d"))
    assert name.startswith("identity-service")


def test_the_name_is_stable_for_one_camera():
    """Reconnects must land on the SAME name, or a restarting process
    leaves its own stale participant behind instead of replacing it."""
    args = (KEY, SECRET, ROOM)
    first = _identity_of(_mint_token(*args, shard="cam_650ca94d"))
    second = _identity_of(_mint_token(*args, shard="cam_650ca94d"))
    assert first == second


def test_one_service_owning_the_whole_room_keeps_the_bare_name():
    """No shard means no sharding — the single-service deployment, where
    the bare name is still correct and nothing collides."""
    assert _identity_of(_mint_token(KEY, SECRET, ROOM)) == "identity-service"
