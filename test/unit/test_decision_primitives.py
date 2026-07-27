"""Unit: P4 decision primitives — eventing parse, cooldown resolution,
and the confirm-config passthrough into EffectiveDims. All parity
defaults pinned: NULL eventing must behave exactly like today.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.services.detection.scheduling.decision import (
    EventingConfig,
    cooldown_seconds_for,
    parse_eventing,
)
from src.services.detection.scheduling.dims import resolve_dims


def test_parse_null_and_junk_yield_parity_defaults() -> None:
    for raw in (None, {}, [], "x", {"entry": "junk", "confirm": 7},
                {"entry": {"m": "3"}, "confirm": {"k": True, "n": 2.5}}):
        cfg = parse_eventing(raw)
        assert cfg == EventingConfig()  # m=1, t=0, k=0, n=0


def test_parse_valid_config() -> None:
    cfg = parse_eventing({"entry": {"m": 3, "t_seconds": 600},
                          "confirm": {"k": 2, "n": 3}})
    assert (cfg.entry_m, cfg.entry_t_seconds, cfg.confirm_k, cfg.confirm_n) \
        == (3, 600, 2, 3)


def test_parse_unsatisfiable_confirm_disables_it() -> None:
    cfg = parse_eventing({"confirm": {"k": 3, "n": 1}})
    assert cfg.confirm_k == 0 and cfg.confirm_n == 0


def test_parse_m_floor_is_one() -> None:
    assert parse_eventing({"entry": {"m": 0}}).entry_m == 1


def test_parse_m_without_window_degrades_to_parity() -> None:
    """M>1 with no/zero T can never accumulate at tick cadence — the
    detector would be silently dead. Must degrade to on-sight (P4
    review finding 1), never to detection loss."""
    cfg = parse_eventing({"entry": {"m": 3}})
    assert cfg.entry_m == 1 and cfg.entry_t_seconds == 0
    cfg2 = parse_eventing({"entry": {"m": 3, "t_seconds": 0}})
    assert cfg2.entry_m == 1


def test_cooldown_row_wins_null_and_zero_fall_back() -> None:
    assert cooldown_seconds_for(SimpleNamespace(cooldown_seconds=5), 30) == 5
    assert cooldown_seconds_for(SimpleNamespace(cooldown_seconds=None), 30) == 30
    assert cooldown_seconds_for(SimpleNamespace(cooldown_seconds=0), 30) == 30
    assert cooldown_seconds_for(SimpleNamespace(cooldown_seconds=True), 30) == 30
    assert cooldown_seconds_for(SimpleNamespace(), 30) == 30


def test_dims_carry_confirm_config_from_row_eventing() -> None:
    row = SimpleNamespace(
        identifier="fall", alert_trigger="leading",
        requires_motion=None, subject_scope=None, bucket=None,
        duty_cycle=None, needs_audio=False, severity=None,
        eventing={"confirm": {"k": 2, "n": 3}},
    )
    d = resolve_dims(row)
    assert (d.confirm_k, d.confirm_n) == (2, 3)
    # And absent eventing = no confirmation (parity).
    row.eventing = None
    d2 = resolve_dims(row)
    assert (d2.confirm_k, d2.confirm_n) == (0, 0)
