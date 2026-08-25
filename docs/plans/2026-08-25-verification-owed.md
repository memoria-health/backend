# What we shipped on 2026-08-25 and have not yet seen work

Everything below is committed, pushed, and covered by tests. None of it
has been watched doing its job on real video, and two items are not
running anywhere yet. This is the list to work through before the next
person-facing claim about any of it.

Branches: `nemara-backend/feat/camera-ownership` (PR #44),
`nemara-bridge/feat/bridge-core-vendor-split` (PR #1),
`nemara-caregiver-app/feat/detector-polish` (PR #32).

---

## 1. The shared-sensor safety gate — never fired in anger

`41d88c5` + `a7f3e26`. A safety detector no longer alerts a resident
when identity has placed a DIFFERENT resident of the same sensor alone
in the room for three ticks running.

Live on the backend container since 02:35 IST. Zero real events have
exercised it — every assertion so far is a unit test.

**Repeat the original test.** Dad alone in the TV Room, acted fall.
Expect: no event on Aryaman's record, and this line in the backend log —

```
[event_promotion] fall fired on camera c51733e6… but identity placed
another resident of this sensor in the room, alone — not alerting
patient 940d4034…
```

**Then the inverse, which matters more.** BOTH of you in the TV Room,
one acts a fall. Expect the alert to still fire — the run length should
be zero for both of you, because every tick has two people in it. This
is the case the review said could silently eat a real fall.

## 2. The three-tick run length is a guess

`SOLO_TICKS_BEFORE_SUPPRESSING = 3` in `camera_ingest.py`. Chosen because
ticks are ~10s and half a minute felt like the shortest defensible run,
not because anything was measured.

The failure we cannot see from here is the gate being a **no-op**: if a
real solo run is broken every few ticks by track churn (and the TV Room
churned six times in three minutes on the night of 08-24), the streak may
never reach 3 and the whole gate never engages. Nothing would look wrong.

Measure the distribution of solo-run lengths from a night's ticks before
trusting the number.

## 3. The running bridge does not have the privacy fix

`29dffaa`. A connect that dies after `_apply_privacy` but before
`publish_track` left the change-guard saying "sent" against a room that
was gone, so the retry's fresh session kept the SFU default — allow-all —
on a camera marked private.

**The process running now started 18:58 on 08-24; the fix committed
02:28 on 08-25.** It is not in the running bridge.

Not currently exposed: the bridge publishes only TV Room, which has
`privacy_enabled = false`, and for an allow-all camera the skipped
re-send changes nothing. The Bedroom, which IS private, is the Tapo via
`nemara-gateway` — a different codebase this fix does not touch.

**But it must be restarted before either of these:** TV Room is ever
marked private, or the bridge is deployed to Railway. Restarting drops
that camera for a few seconds, which is why it was left alone overnight.

## 4. The iOS change compiles and nothing has tapped it

`2d5bbb1`. Live tab's "Add a sensor" now opens `AssignSensorSheet`
instead of `AddCameraSheet`.

`xcodebuild` is clean. No runtime pass. Two things to actually check by
hand:

- The sheet takes no `patientId` and is normally opened from the roster.
  Opened from inside a patient's Live tab, does it behave — list, rename,
  who-it-watches, and the `+` through to pairing?
- Pairing from the Live tab used to silently assign the new sensor to
  whichever patient's page you were on. It should now ask. Confirm it
  does, and that the answer sticks.

## 5. Face harvest has one data point

Fired once, for Aryaman, at 19:35 on 08-24. Never fired for the second
resident. Under the agreed rule — harvest only on a face confirmation —
it will stay rare while body locks out-score faces, which is most of the
time. Worth counting over a full night before deciding it works.

## 6. Pushed code is still not what the node runs

The identity node reads `/persist/state/screen_concept` (and a threshold
override) which shadow the committed defaults. So the display-veto
behaviour verified live on 08-24 is NOT the behaviour in the repo, and a
node rebuild would silently change it. Same for the `.simulcast-off`
rename applied by hand on the mini-PC — in no image, in no script.

Remove the override files, restart the shard, and confirm the committed
defaults reproduce what we saw.

## 7. Body pool is still in memory

`body_pool_embeddings` exists as a table with no code writing to it. The
pool dies on every restart, and the backend was restarted twice during
the night of 08-24/25 — so anything learned before 02:35 is gone. Decide
whether to persist it.

## 8. Test-order pollution: fixed here, unverified in CI

`0a506ac`. 1551 pass in both directions locally. Whether CI runs the
suite in an order that would have caught the original leak is unchecked.

---

## Not verification, but it belongs on tomorrow's list

**A second resident on a shared camera starts with no safety detectors.**
Aryaman turned his father's off deliberately, so this is not a bug here —
but the code cannot distinguish that from an RCFE resident who was added
to a shared lounge camera and has not finished setup. With the new gate,
a confirmed resident with no fall detector means the fall reaches nobody.
Needs a rule at assignment time, not in the gate.

**Two dev `.env` flags must not reach a real household:**
`NEMARA_SENSITIVITY_SCREENING_ENABLED=false` and
`NEMARA_SLEEP_CLIPS_DIR`. The test suite no longer reads the first one
(conftest pins it), so nothing will warn you.
