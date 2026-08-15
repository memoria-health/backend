# Patient Home Tab — Dashboard Compiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
> Design spec: https://claude.ai/code/artifact/6a38f33e-b29a-497b-ae58-03e71a5f9f92

**Goal:** Per-patient Home tab (new default tab) whose widgets are compiled once per
detector change — pinned catalog mapping for built-ins, rules+LLM for customs — rendered
natively with Swift Charts, fed by the existing `/v1/insights/widget-data` engine.

**Architecture:** A `dashboard_widget` JSONB column on `detectors` holds one compiled
widget spec per detector. Built-ins get their spec from a pinned `widget` field on
`CatalogEntry`; customs are compiled by `src/services/dashboard/widget_compiler.py`
(deterministic trigger-shape rules, LLM pick within those constraints, rules as
fallback). The spec rides the existing `DetectorResponse` — **no new dashboard
endpoint**; iOS derives sections client-side from its live detectors store and fetches
chart data per widget via `widget-data` (which gains a `detector` filter) and two new
read endpoints for `sleep_nights` / `activity_days`.

**Tech Stack:** FastAPI/SQLAlchemy/pytest (backend, branch
`feat/data-collection-timing-truth`); SwiftUI + Swift Charts + XcodeGen (iOS, branch
`fix/door-picker-private-camera`).

**Invariants (from the design spec):**
- Widget = f(detector). Compile only on detector create/from-library/update/refine —
  never per open, never at read time.
- Only detectors on the patient compile. `hidden` and child-cohort entries → no spec.
- Insights only: no alert faces. Two client faces: chart / collecting (`min_days`).
- Less is more: captions are typed slots, one line max. LLM never writes dashboard text.
- Non-additive: reuse widget-data + DetectorResponse + iOS detectors store; remove the
  roster `streakChip` (HomeStatusView) which the Falls tile supersedes.

**Review cadence (per Aryaman's standing rule):** after every task commit, an
adjudicated review (correctness/efficiency/memory/security/goal-adherence) via a
code-reviewer subagent on that commit's diff; full-diff review at each phase end.

**Widget spec shape (persisted, one per detector):**
```json
{
  "widget": "streak_tile|dot_calendar|hour_clock|trend_band|sleep_bands|week_heatmap|delta_tile|duration_bars|count_bars|ratio_tile|room_bars",
  "size": "tile|wide",
  "title": "Night exits",
  "params": {"metric": "events", "detector": "<uuid>", "byHour": true, "days": 14,
              "domain": ["22:00","06:00"], "ratio": "nightsWith"},
  "caption_fields": ["count", "ratio"],
  "ranges": ["W", "M", "3M"],
  "min_days": 3,
  "compiler": "catalog|rules|llm", "compiled_at": "<iso>"
}
```
`params.metric` values are existing widget-data metrics; `params.detector` is filled at
compile time with the detector's own id (catalog templates use `"$self"`).

---

## Phase A — backend (`nemara-backend`, branch `feat/data-collection-timing-truth`)

### Task A1: `dashboard_widget` column

**Files:**
- Modify: `src/db/detector.py` (after `degradation_warning`, ~line 121)
- Modify: `src/schemas/detector.py` (`DetectorResponse` gains `dashboard_widget`)
- Create: `scripts/apply_dashboard_widget_delta.sql`
- Modify: `supabase/schema.sql` (detectors table, ~line 558-641)
- Test: `test/unit/test_dashboard_widget_schema.py`

**Steps:**
1. Failing test: `DetectorResponse` round-trips a `dashboard_widget` dict and defaults
   to None; `Detector` model has the column.
2. Run `make test-unit` (or `.venv/bin/python -m pytest test/unit/test_dashboard_widget_schema.py -v`) → FAIL.
3. Add `dashboard_widget: Mapped[dict | None] = mapped_column(JSONB, default=None)` with
   a comment stating the compile-on-change contract; add the schema field; write the
   delta SQL (`alter table detectors add column if not exists dashboard_widget jsonb`);
   mirror in `supabase/schema.sql`.
4. Test passes. Commit: `feat(dashboard): dashboard_widget spec column on detectors`.
5. Reviewer subagent on the commit diff.

### Task A2: pinned widgets for the 15 adult/both built-ins

**Files:**
- Modify: `src/services/detection/catalog.py` (`CatalogEntry` ~919-1000; entries
  ~1010-1223; `_validate_catalog` ~1230)
- Test: `test/unit/test_catalog_dashboard_widgets.py`

Mapping (from the design spec): fall→streak_tile · wandering→hour_clock(+nightsWith,
domain from when_scope) · calling_for_help/choking/emergency/fire_smoke→streak_tile ·
stove_left_on→dot_calendar · possible_intruder→hour_clock ·
env_hazard_glass/water→room_bars · sleep→sleep_bands · activity→trend_band ·
medication→dot_calendar(adherence) · meal_eaten→dot_calendar+ratio ·
agitation→week_heatmap(+comparePrev). `setup_test` and all `cohort="child"` entries:
`widget=None`.

**Steps:**
1. Failing tests: every non-hidden adult/both entry has a valid widget; hidden and
   child entries have none; widget types ∈ the 11-primitive enum; `_validate_catalog`
   rejects a bad type.
2. FAIL → implement `widget: dict | None = None` on `CatalogEntry`, fill entries,
   extend `_validate_catalog`.
3. PASS. Commit: `feat(dashboard): pin built-in widget specs in catalog`. Review.

### Task A3: widget compiler for customs

**Files:**
- Create: `src/services/dashboard/__init__.py`, `src/services/dashboard/widget_compiler.py`
- Test: `test/unit/test_widget_compiler.py`

`compile_widget(detector) -> dict | None`:
- built-in identifier in catalog → catalog spec with `$self` resolved (title from
  entry name; wandering domain parsed from `when_scope` via existing
  `when_scope.parse_windows`).
- custom → deterministic rules (design spec §5): leading+severity≥100→streak_tile;
  absence+deadline→dot_calendar(adherence); duration|is_sustained→duration_bars
  (week_heatmap if behaviour); record_only→trend_band; environment+multi-room→room_bars;
  leading recurring→dot_calendar, hour_clock when `when_scope` is a night window;
  fallback dot_calendar(occurrence).
- optional `llm_pick` hook (same client pattern as the `user_criteria` summarizer):
  receives detector + the rule-derived candidate + allowed primitives, may swap within
  constraints; any exception or invalid output → rule result. Unit tests mock it.
- hidden identifiers (`catalog.hidden_keys()`) → None.

TDD: one test per rule row, fallback test, LLM-invalid-output test, `$self` resolution
test. Commit: `feat(dashboard): widget compiler — catalog for built-ins, rules+LLM for customs`. Review.

### Task A4: compile hooks in detector write paths

**Files:**
- Modify: `src/api/detectors.py` — from-library (~350), custom create (~611),
  patch (~821), refine (~948). Delete needs nothing.
- Test: `test/unit/test_widget_compile_hook.py`

`should_recompile(old, new)` pure predicate: category, alert_trigger, is_sustained,
duration_seconds, schedule_config, when_scope, where_rooms, subject_scope, severity —
`is_on` and cosmetic fields do NOT recompile (spec stability rule). Hooks set
`detector.dashboard_widget = compile_widget(detector)` on create/from-library/refine and
on patch only when the predicate fires. TDD the predicate + a hook test with a stub
session. Commit: `feat(dashboard): compile widget specs on detector writes`. Review.

### Task A5: `detector` filter in widget-data

**Files:**
- Modify: `src/api/insights_widgets.py` (events workhorse + daysSince + heatmap/byHour
  paths; scope check: detector must belong to a scoped patient)
- Test: `test/unit/test_widget_data_detector_filter.py` (follow the file's existing
  test pattern; if only integration exists, test the filter-builder helper)

`source.detector` (uuid) → `and detector_id = :det` added to the event queries; invalid
or out-of-scope id → 404 (mirror `_narrow`). Commit:
`feat(insights): per-detector filter in widget-data`. Review.

### Task A6: sleep/activity read endpoints

**Files:**
- Create: `src/api/summaries.py` — `GET /v1/patients/{id}/sleep-nights?days=` and
  `GET /v1/patients/{id}/activity-days?days=` (CaregiverAuth, scope via existing
  patient-scope dependency; days clamped ≤90 like `_days` in insights.py)
- Modify: `src/main.py` (router registration ~254-296)
- Test: `test/unit/test_summaries_api.py` (existing API-test pattern)

Rows straight from `sleep_nights` / `activity_days` (written by
`tools/run_daily_summaries.py`; shapes in `scripts/apply_sleep_activity_tables_delta.sql`).
Commit: `feat(api): serve sleep_nights and activity_days`. Review.

### Task A7: backfill + phase review

**Files:**
- Create: `tools/backfill_dashboard_widgets.py` (iterate detectors, `compile_widget`,
  update; `--dry-run` default like other tools)

Smoke-run dry against local db. Commit: `chore(dashboard): backfill widget specs`.
**Phase-end full-diff review** (all A commits) before starting Phase B.

---

## Phase B — iOS (`nemara-caregiver-app`, branch `fix/door-picker-private-camera`)

Run `xcodegen generate` after adding files (project.yml is source of truth).

### Task B1: model + decoding
- Modify: `Models/Detector.swift` — `DashboardWidget` Codable (widget/size/title/params/
  captionFields/ranges/minDays) + `dashboardWidget` on `Detector`.
- Test: `NemaraCaregiverTests/DashboardWidgetDecodingTests.swift` — decode fixture JSON,
  unknown widget type → nil (forward compat), absent field → nil.
- Commit: `feat(home): DashboardWidget model`. Review.

### Task B2: section builder (pure logic, fully TDD-able)
- Create: `Features/Patient/Home/DashboardBuilder.swift` — `[Detector] → [DashboardSection]`:
  categories in fixed order safety→routine→behaviour→care, only categories with ≥1
  `isOn` detector carrying a spec; within section severity desc then createdAt;
  collecting gate (`minDays` vs detector age).
- Test: `DashboardBuilderTests.swift` — ordering, empty-category omission, off-detector
  exclusion, collecting flag.
- Commit: `feat(home): dashboard section builder`. Review.

### Task B3: data plumbing
- Modify: `Services/NemaraAPIClient.swift` — `widgetData(source:)` POST, `sleepNights`,
  `activityDays`; responses as lightweight Codables.
- Test: decoding tests with recorded fixtures.
- Commit: `feat(home): widget-data + summaries client`. Review.

### Task B4: widget views (Swift Charts)
- Create: `Features/Patient/Home/Widgets/*.swift` — card chrome (eyebrow/title/seg
  picker/caption slot) + the 11 primitives; D/W/M/3M segmented control on ranged
  widgets; dashed average RuleMark; `chartXSelection` scrubbing; previews with fixture
  data per widget. Colors from `CategoryPalette`, components from DesignSystem.
- Verify by build + previews (no meaningful unit tests for chart bodies).
- Commit: `feat(home): Swift Charts widget library`. Review.

### Task B5: the Home tab
- Modify: `Features/Patient/PatientShell.swift` — `PatientTab.home` first, default
  `.home`; APNs deep-link to `.events` unchanged (:142-146).
- Create: `Features/Patient/Home/PatientHomeView.swift` — sensing line, sections,
  collecting faces, pull-to-refresh; `.trackScreen("patient.home")`.
- Modify: `Features/Tutorial/TutorialOverlay.swift` — tab-index map (:178-180) shifts
  by one; tour's liveTab step now targets index 1.
- Commit: `feat(home): patient Home tab, default landing`. Review.

### Task B6: non-additive cleanup
- Modify: `Features/Home/HomeStatusView.swift` — remove `streakChip` (:333-346) and its
  single-patient gate (:264-269) + quiet-nights computation now unused; the Falls tile
  on the patient Home tab supersedes it.
- Commit: `refactor(roster): drop streak chip — superseded by patient Home tab`. Review.

### Task B7: build + phase review
- `xcodegen generate && xcodebuild -scheme NemaraCaregiver -destination 'platform=iOS Simulator,name=iPhone 16' test`
- Phase-end full-diff review; then PRs for both repos (deployed code → PR, never
  straight to main).

---

## Deploy notes (from the prod runbook)
- DB delta (`apply_dashboard_widget_delta.sql`) BEFORE backend deploy; then backfill.
- `sleep_nights`/`activity_days` need `run_daily_summaries.py` scheduled in prod or the
  routine widgets sit on empty data — flag at PR time.
- iOS ships behind the same PR review; tutorial change needs a manual tour run-through.
