"""Ring overflow — bounded memory, and the batch chain must SURVIVE it.

The 2026-07-24 churn lesson (live): overflow must never reset
`_last_shipped`. The old "re-seed after the gap" reset made every
in-flight push count as unshipped (shipped_idx=0), re-latching the
overflow flag every cycle → permanent whole-ring batches → the pod
session rewound ~30s per tick and renumbered the subject on every
reground under motion. Dropping old unshipped CANDIDATES is harmless:
plan_batch picks from the remaining newer frames, all still after
last_shipped — the viewer's normal condition.
"""

from types import SimpleNamespace

import numpy as np

from identity.loop import CameraLoop


def _spine() -> CameraLoop:
    return CameraLoop(camera_id="a", home=SimpleNamespace())


def test_ring_bounded_and_chain_preserved_on_overflow():
    sp = _spine()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for i in range(5):
        sp.push_frame(float(i), frame)
    shipped = sp.ring[-1]
    sp._last_shipped = shipped
    for i in range(5, sp.RING_MAX + 40):
        sp.push_frame(float(i), frame)
    assert len(sp.ring) == sp.RING_MAX
    # The chain anchor survives overflow — overflow must never force a
    # whole-ring re-ship (the 2026-07-24 rewind bug).
    assert sp._last_shipped is shipped


def test_overflow_has_no_reseed_latch():
    # The self-perpetuating `_overflowed` latch is gone for good.
    sp = _spine()
    assert not hasattr(sp, "_overflowed")
