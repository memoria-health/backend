# IR mode, identity continuity, harvesting — for tomorrow

Written 2026-08-26 ~03:15 IST, Aryaman going to sleep. Plan: day soak
first, then tackle IR.

## 1. IR switch — the actual symptom, narrowed

Aryaman: *"when the camera switches from day to night IR mode, the track
is lost and tough to regain"* and then, importantly:

> **"Just partially when the camera state switches. After and before
> tracks work."**

So this is NOT steady-state IR tracking. It is the transition itself.
Steady state is handled already (`_IR_PERSON_THR`, `_IR_REMAP_IOU 0.10`,
`_IR_GROUND_GRACE`, CLAHE on the whole batch).

He also said IR **loses identity fairly quickly** in general, so treat
"the switch" and "IR steady state" as two problems, not one.

### What shipped tonight (UNVERIFIED — no switch has happened since)

`continuous.py`: the day<->night flip is now DETECTED and LOGGED, and
tracks get a wider miss grace across it.

    _IR_SWITCH_GRACE_TICKS = 6    ticks the window stays open
    _IR_SWITCH_MISS_GRACE  = 5    misses allowed inside it (vs 2 normally)

    continuous.ir_switch cam=... day->night — holding tracks for 6 ticks

That log line did not exist before; the switch was completely invisible,
which is part of why this was hard to pin down. **First real test is at
dawn.** If the id still changes across it, the line tells us the window
opened and was not enough.

Reasoning: the switch changes every pixel in one frame — colour to
grayscale, new exposure, illuminator on — so every mask EdgeTAM carries
was built on pixels that no longer exist. They fail together and a
motionless person is re-numbered.

## 2. Seat continuity — a real gap found live, and fixed

Measured at 21:38 while he sat still:

    21:38:25  seat_locked  T1 @face
    21:38:44  deaths=[1] live=[]     died with NOTHING live
    21:38:44  seat_lost — no single overlapping successor
    21:38:47  births=[2] live=[2]    successor 3 SECONDS later

`carry_seat` only considers tracks alive AT THE INSTANT of death. The
replacement is often a few ticks behind, so the seat was already gone.

Fix: a seat whose track dies with no successor is now PARKED for
`DEAD_BOX_TTL_S` (6s) and re-attaches when exactly one live track
overlaps the box it vacated. Withheld from `seats` while parked, so
nothing downstream reads the resident as located. Same three
conservative rules as carry_seat.

New log lines: `identity.seat_parked`, `identity.seat_reattached`.

## 3. Harvesting — works, but cannot accumulate

Asked directly. Body harvest IS firing:

    21:31:30  harvested pool_size=2  confidence=0.732
    21:40:00  harvested pool_size=1  confidence=0.409

Pool went 2 -> 1: it is being REBUILT, not grown. `body_pool_embeddings`
has **0 rows** — the pool is in-memory only and dies on every restart.
Identity was restarted ~8 times tonight, so it never accumulated.

**This is likely a large part of why re-identification felt bad tonight,
and most of it was self-inflicted by the deploys.** Judge harvesting
after a soak with no restarts, not before.

Persisting the pool is the standing open item (verification-owed #7).

## 4. Two bugs I caused tonight, both the same shape

- **`home.py` split**: I wrote the new module's import list by hand and
  missed `notary_contradiction`, `heal_seat`, `np`. Every identity probe
  crashed with `NameError`, so nothing could ever lock a seat. 159 probe
  failures before the fix, 0 after.
- I ran `ruff --select F401` (unused imports) and reported it clean, but
  never ran **F821** (undefined names), which catches this instantly.
  **Run F821 after any module split.**
- I then saw `seats=0` immediately after deploying and wrote it off as
  "just restarted". That was the evidence.

1,199 tests passed throughout. No test exercises `fold_candidates`.

## 5. Tonight's sleep run

- confirmer ARMED and scoped to `sleep_activity` in
  {entering_bed, leaving_bed, sitting_up} — expect ~10 events/night, not
  the ~245 an edge on `sleep_detected` would give
- `nightly_summaries_enabled=true`, so the night folds automatically
- local backend runs from the `nemara-backend-sleep` worktree on
  `chore/deploy-prep`; the primary checkout is on `main` for the deploy

**Morning numbers to pull:**
1. verdict changes across the night vs 08-23's **739** (that run had the
   confirmer entangled; this one does not)
2. confirm votes actually spent — should be ~10, not ~245
3. did a `sleep_nights` row appear without anyone running anything
4. `continuous.ir_switch` at dawn, and whether the track id survived it

## 6. Do NOT deploy the scope config to prod before the code

`main`'s `parse_eventing` has no knowledge of the `on` key. If main's
code meets the armed config it ignores the scope and falls back to the
`sleep_detected` edge — ~245 confirms/night. Prod DB is untouched; only
the local DB is armed. Code first, config second.
