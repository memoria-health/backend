"""Home-tab widget compiler — widget = f(detector), decided once.

Called ONLY from detector write paths (create / from-library / refine /
shape-changing update) and the backfill tool; never at read time. The
result is persisted on `detectors.dashboard_widget` and rendered as-is
by the iOS Home tab.

Built-ins resolve their pinned template from the catalog. Customs go
through deterministic trigger-shape rules (design spec section 5); an
optional `llm_pick` hook may swap the choice WITHIN the primitive
library — its output is validated with the same check the catalog uses
at import, and anything invalid (or any exception) falls back to the
rules result, so a flaky provider can never blank a dashboard.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from src.services.detection import catalog
from src.services.detection.scheduling.when_scope import parse_windows

log = logging.getLogger("nemara.dashboard")

# An llm_pick receives (detector, rules_candidate) and returns a spec.
LlmPick = Callable[[Any, dict], dict]

# Leading detectors at or above this severity are expected-rare alarms:
# the caregiver-relevant number is "days since", not a frequency chart.
# Must match scheduler._EMERGENCY_SEVERITY (=80), where the cost-gate
# exemption band begins — real loadout: choking 100, fall 95,
# calling_for_help 88, and custom severities clamp to [0,100], so a
# higher bar here would silently exclude the very detectors it's for.
_RARE_SEVERITY = 80

_RANGED_DEFAULT = ["W", "M", "3M"]

# PATCH fields whose change alters the compiled widget. `is_on`,
# sensitivity, icon and description are display/behaviour-only —
# recompiling on them would churn the spec for nothing (custom_prompt
# edits recompile through the background detector compile instead).
# `name` is here because custom widget titles derive from it.
SHAPE_FIELDS = frozenset({
    "name", "category", "alert_trigger", "is_sustained",
    "duration_seconds", "schedule_config", "when_scope", "where_rooms",
    "subject_scope", "severity",
})


def should_recompile(patched_fields) -> bool:
    """True when a PATCH payload touches any compiler input."""
    return bool(SHAPE_FIELDS & set(patched_fields))


def apply_widget(det) -> None:
    """Best-effort `det.dashboard_widget = compile_widget(det)`.

    The guard exists because every call site is a write path that
    matters more than the dashboard: a compiler bug must never 500 a
    caregiver's detector edit, roll back a successful custom compile,
    or abort onboarding seeding. On failure the old spec (or NULL)
    stays and the Home tab renders what it already had."""
    try:
        det.dashboard_widget = compile_widget(det)
    except Exception:
        log.warning(
            "[dashboard] widget compile failed for detector %s — keeping "
            "previous spec", getattr(det, "id", None), exc_info=True,
        )


class _DetectorLike(Protocol):
    id: Any
    name: str
    identifier: str | None
    source: str
    category: str
    alert_trigger: str
    is_sustained: bool
    schedule_config: dict | None
    when_scope: str | None
    where_rooms: list[str] | None
    subject_scope: str | None
    severity: int | None


def _stamp(spec: dict, compiler: str) -> dict:
    spec["compiler"] = compiler
    spec["compiled_at"] = datetime.now(timezone.utc).isoformat()
    return spec


def _mmss(minutes: float) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def _domain(when_scope: str | None) -> list[str] | None:
    """First when_scope window as an ["HH:MM","HH:MM"] axis domain, or
    None for always-eligible scopes (24 h axis)."""
    windows = parse_windows(when_scope)
    if not windows:
        return None
    start, end = windows[0]
    return [_mmss(start), _mmss(end)]


def _is_night(when_scope: str | None) -> bool:
    windows = parse_windows(when_scope)
    if not windows:
        return False
    start, end = windows[0]
    return start > end or start >= 21 * 60  # crosses midnight or evening-anchored


def _title(name: str) -> str:
    return name if len(name) <= 24 else name[:23].rstrip() + "…"


def _resolve_template(det: _DetectorLike, template: dict) -> dict:
    spec = copy.deepcopy(template)
    params = spec.get("params", {})
    if params.get("detector") == "$self":
        params["detector"] = str(det.id)
    if params.get("domain") == "$when_scope":
        dom = _domain(det.when_scope)
        if dom is None:
            params.pop("domain")
        else:
            params["domain"] = dom
    if params.get("windows") == "$schedule":
        if det.schedule_config:
            params["windows"] = det.schedule_config
        else:
            params.pop("windows")
    return _stamp(spec, "catalog")


def _rules_spec(det: _DetectorLike) -> dict:
    """Design spec section 5 — trigger shape anchors the decision, the
    detector's details fill in the axes."""
    base_params: dict[str, Any] = {"metric": "events", "detector": str(det.id)}
    spec: dict[str, Any] = {"size": "wide", "title": _title(det.name),
                            "params": base_params, "min_days": 3}

    # Severity outranks shape (design §5 row 1): a severe leading alarm
    # is a streak tile even if it's also sustained or environment-scoped
    # — "days since" is the read a caregiver wants for a rare alarm.
    if det.alert_trigger == "leading" and (det.severity or 0) >= _RARE_SEVERITY:
        spec.update(widget="streak_tile", size="tile", min_days=1)
        spec["params"] = {"metric": "daysSince", "detector": str(det.id)}
        spec["caption_fields"] = ["longest_gap"]
    elif det.alert_trigger == "record_only":
        # No "D" on trends: one bucket is a dot, not a trend.
        spec.update(widget="trend_band", min_days=7, ranges=_RANGED_DEFAULT)
        base_params.update(days=30, baseline=True)
    elif det.alert_trigger == "absence" and det.schedule_config:
        spec.update(widget="dot_calendar")
        base_params.update(mode="adherence", days=14,
                           windows=det.schedule_config)
        spec["caption_fields"] = ["adherence"]
    elif det.alert_trigger == "duration" or det.is_sustained:
        if det.category == "behaviour":
            spec.update(widget="week_heatmap", min_days=7)
            base_params.update(heatmap=True, days=14, comparePrev=True)
            spec["caption_fields"] = ["delta"]
        else:
            spec.update(widget="duration_bars", ranges=_RANGED_DEFAULT)
            base_params.update(days=7, instances=True)
    elif det.subject_scope == "environment" and (
        det.where_rooms is None or len(det.where_rooms) != 1
    ):
        spec.update(widget="room_bars", size="tile", min_days=7)
        base_params.update(byRoom=True, days=30)
        spec["caption_fields"] = ["count"]
    elif _is_night(det.when_scope):
        spec.update(widget="hour_clock")
        base_params.update(byHour=True, days=14, domain=_domain(det.when_scope),
                           ratio="nightsWith")
        spec["caption_fields"] = ["ratio"]
    else:
        spec.update(widget="dot_calendar")
        base_params.update(mode="occurrence", days=14, ratio="daysWith")
        spec["caption_fields"] = ["ratio"]
    return spec


def compile_widget(det: _DetectorLike, llm_pick: LlmPick | None = None) -> dict | None:
    """The one-time dashboard visualization determiner for one detector.
    Returns the spec to persist, or None for detectors that get no
    widget (hidden/internal identifiers)."""
    if det.identifier and det.identifier in catalog.hidden_keys():
        return None

    if det.source == "preset" and det.identifier:
        entry = next((e for e in catalog.CATALOG if e.key == det.identifier), None)
        if entry is not None:
            if entry.widget is None:
                return None
            return _resolve_template(det, entry.widget)

    candidate = _rules_spec(det)
    if llm_pick is not None:
        try:
            picked = llm_pick(det, copy.deepcopy(candidate))
            catalog.validate_widget_spec(det.identifier or "custom", picked)
            return _stamp(picked, "llm")
        except Exception:
            log.warning("[dashboard] llm_pick failed for %s — using rules",
                        det.identifier, exc_info=True)
    return _stamp(candidate, "rules")
