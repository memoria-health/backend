# Daily Journal Implementation Plan

> Supersedes and DELETES docs/plans/2026-08-15-patient-home-dashboard.md
> (the chart dashboard, torn down in Phase 1). Design + decisions:
> https://claude.ai/code/artifact/6a38f33e-b29a-497b-ae58-03e71a5f9f92
> Process: TDD per task · adjudicated review per commit · phase-end full-diff
> review · build on active branches (backend feat/data-collection-timing-truth,
> iOS fix/door-picker-private-camera).

**Goal:** Patient Home = an automatic daily journal. One page per day
(landing on "Today so far"), chevron/swipe pager back in time, entries
grouped by category, each detector contributing one line whose shape is a
pure function of its taxonomy. Exactly one visualization primitive (the
24-h day-strip), used SPARSELY. No summary sentence. No LLM anywhere.

**Locked decisions (Aryaman, Aug 16):**
- Delete the compiled-widget layer fully (nothing shipped; clean cut).
- Landing page = live "Today so far"; completed days behind it.
- Unreviewed alerts appear in the day with a Review chip → Events.
- NO verdict/summary line of any kind (even templated) — date, entries,
  coverage line only.
- Strips only where timing is the story: sleep, episodes, event days.
  Text-only lines otherwise. Never a strip without a clean map.
- Sleep strip = NOON-TO-NOON axis (contiguous night band); all other
  strips midnight-to-midnight. Exact hour→x mapping fn, snapshot-tested.
- Activity definition (pinned): moving = standing+active minutes from the
  activity summarizer's state vocabulary; a block renders where a 30-min
  bucket has ≥10 moving minutes; solid blocks, no opacity ramp; text
  "about N h up and moving".
- is_sustained on record streams = reporting unit (sustained→duration,
  not→count), judged by the infer prompt with 10-s chunk context (done).
- Customs stay beta; built-ins ship properly. Catalog = 15 displayed
  adult built-ins + hidden setup_test (child cohort deleted).

## Phase 1 — Teardown (both repos)

Backend (each its own commit, review after):
- T1: remove compile hooks + imports (api/detectors.py ×3 sites,
  api/cameras.py, services/onboarding.py, tools/seed_demo_patient.py);
  delete src/services/dashboard/ entirely; delete
  tools/backfill_dashboard_widgets.py.
- T2: catalog.py — remove `widget` field, all 15 templates, WIDGET_TYPES,
  validate_widget_spec/_validate_widget; delete
  test_catalog_dashboard_widgets.py, test_widget_compiler.py,
  test_widget_compile_hook.py.
- T3: db/detector.py + schemas/detector.py — drop dashboard_widget (KEEP
  severity: journal orders sections by it); supabase/schema.sql line;
  delete scripts/apply_dashboard_widget_delta.sql +
  test_dashboard_widget_schema.py; local db:
  `alter table detectors drop column if exists dashboard_widget;`
- KEEP: insights_widgets.py wholesale (web app's engine; detector filter,
  composed responses, detectedTicks all stay), summaries.py endpoints,
  detector-compiler prompt fixes, catalog deletion, seed tool (minus
  apply_widget), event gate.

iOS (one commit):
- T4: delete Features/Patient/Home/{DashboardBuilder,WidgetCharts,
  DashboardWidgetView}.swift, Models/DashboardWidget.swift (JSONValue
  moves to its own file if still referenced — check), dashboardWidget
  property/decode on Detector (KEEP severity), NemaraAPIClient
  widgetData/sleepNights/activityDays + WidgetData/SleepNight/ActivityDay
  types, tests DashboardWidgetCodableTests/DashboardBuilderTests/
  DashboardDataCodableTests/WidgetSnapshotTests. PatientHomeView shrinks
  to sensing line + DashboardEmptyStateView placeholder ("journal coming
  next commit"). xcodegen + full test run.

## Phase 2 — Journal endpoint (backend, TDD)

- J1: `src/services/journal/day.py` — PURE builders, fully unit-tested:
  - `hour_x(dt, tz, axis)` fractional hours on midnight|noon axis
  - line builders per taxonomy from plain rows:
    point events → {kind:"events", line, marks[], review}
    quiet point → {kind:"quiet", "No falls", streak_days}
    episodes (sustained alerting) → spans from event occurred_at (no end
    persisted yet → mark, not span, v1: dot + time; span when duration
    lands later — do NOT fake ends)
    absence → ✓/missed vs schedule_config deadline (event = miss)
    record_duration (sleep) → from sleep_nights row (noon axis)
    record_activity → blocks from analyses activity_state ticks per the
    pinned definition
    record_count (water etc.) → count + dot marks from detected ticks
  - coverage_hours from analyses tick count / 360 per hour watched
- J2: `src/api/journal.py` — GET /v1/patients/{id}/journal?date=
  (CaregiverAuth, patient scope via find_by_id/404, date required
  YYYY-MM-DD, caregiver tz from caregivers.timezone fallback UTC,
  partial=true when date==today-local). Queries: detectors (is_on, not
  hidden), day's events, day's analyses (captured_at range in tz),
  sleep_nights row for night_date=date-1? (night belonging to the
  morning of `date` — night_date = date - 1 day per summaries
  convention; VERIFY against run_daily_summaries), register router.
  API test with stub session per test_summaries_api.py pattern.
- J3: seed check — Margaret journal for yesterday returns all four
  categories + expected lines (integration-ish script or manual psql).

## Phase 3 — iOS journal (TDD where testable)

- I1: Models/Journal.swift decoding (tolerant) + tests.
- I2: client `journal(patientId:date:)` (URLQueryItem!) + fixture test.
- I3: Features/Patient/Home/DayStripView.swift — one renderer: marks
  (dots), spans (rounded), ✓/flag; axis 12a/12p/12a or 12p/12a/12p.
- I4: JournalDayView (sections, rows, chips, coverage) +
  PatientHomeView = pager (chevrons + swipe, landing today, "SO FAR
  TODAY" label, refreshable) + Review chip deep-link → Events tab.
- I5: snapshot tests render a full fixture day (all five line kinds) to
  /tmp/journal-snaps — inspect by eye, iterate.
- Copy rules: no em dashes, one-line captions, less is more.

## Phase 4 — Wrap

- Phase-end full-diff reviews (both repos), fix findings.
- Update memory + this plan's status; artifact already matches.
- Do NOT push / open PRs without Aryaman.

## Verification data
Margaret Ellis (seed tool) has events/sleep/analyses for all categories.
Local DB: supabase docker :54342. Simulator: iPhone 15 Pro
7FBF4460-1165-485B-8ACD-6C5BA3CE0A3A. Backend tests:
.venv/bin/python -m pytest test/unit -q (pre-existing failures were env;
suite was fully green Aug 16).
