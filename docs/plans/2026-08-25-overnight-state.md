# State at 04:45 IST, 2026-08-25 — what is running and what to check

Everything below is committed and pushed, and every service is running
the pushed code. Branches: backend `feat/camera-ownership` (PR #44),
bridge `feat/bridge-core-vendor-split` (PR #1), iOS
`feat/detector-polish` (PR #32). All three: 0 unpushed, 0 dirty.

## Running now

| Thing | State |
|---|---|
| Backend container | restarted 21:54 UTC on HEAD; ticking, ~32 analyses / 3 min |
| Nightly summariser | in-process, every 30 min, folding tonight as it happens |
| Bridge | restarted 02:57 IST, has the privacy fix; TV Room publishing |
| Identity shards 8012/8013 | restarted on HEAD |
| Perception | restarted; `thr 0.05`, concepts `[screen, painting]`, both levers on, 16 passes 0 failed |
| Both cameras | online |
| Confirmer | DISARMED on every detector (`eventing = null`) — no confirm spend tonight |

## Check these in the morning

**1. The sleep card.** Open Home on today. It should show last night:
duration, asleep/woke times, wake-up count, and a band broken at each
wake-up. If the card is absent, the night could not be measured — that
is deliberate, and the reason appears in words instead of zeroes.

The night of 08-23 folds to bed 19:53, asleep 03:29, woke 09:36, 306
min, 3 disturbances. That is the first real night this table has ever
held; every prior row was demo seed data.

**2. Cost.** Nothing was armed that spends. The confirmer is off, the
tick cadence is unchanged, and the summariser is SQL plus arithmetic. If
the bill moved tonight it is not from anything shipped in this batch.

**3. The safety gate, still unverified live.** See
`2026-08-25-verification-owed.md`. The test that matters is the inverse
one: both residents in the TV Room, one falls, the alert must still
fire.

## Not done, and why

**The statue veto.** Asked for at ~03:50. Not a straight path: a
"motionless box is not a person" rule also describes a sleeping person,
and identity dropping that track makes the sleep record blank — which
would have cancelled the night's other work. Design and a cheaper thing
to try first are in `2026-08-25-motionless-object-veto.md`.

**Debouncing `sleep_detected`.** Scoped, then found unnecessary for the
night to fold: the 739 verdict changes were never the blocker. The real
cause was the identity gate, now fixed. Worth revisiting only if a
folded night looks wrong, not on principle.

## One number that was chosen, not measured

`SOLO_TICKS_BEFORE_SUPPRESSING = 3` in `camera_ingest.py`. Its invisible
failure is being a no-op — if track churn breaks every solo run before
it reaches 3, the safety gate never engages and nothing looks wrong.
Pull the run-length distribution off tonight's ticks before trusting it.
