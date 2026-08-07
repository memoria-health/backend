"""Evidence must be GENERATED before verdicts, not merely listed first.

`BASE_PROMPT` opens with "FILLING THEM IN THE ORDER GIVEN", and the
2026-08-06 rework rests entirely on that: `heard` and `seen` are the
only places speech and objects may be established, and a detector that
needs a cup at a mouth must find it written above or answer FALSE.

That is a claim about GENERATION ORDER. A JSON schema's `properties` is
a map, so without `property_ordering` the model may emit fields in any
order and the rule has no mechanism under it. Measured live on
2026-08-07: a tick whose `seen` read "holding a smartphone in both
hands while looking down at it", whose description agreed, and whose
drinking-water verdict claimed the subject "picked up a blue plastic
bottle, raised it to his mouth" — at confidence 1.0. Correct
perception, correct description, contradictory verdict, because the
verdict was downstream of nothing.
"""

from __future__ import annotations

from src.services.detection.prompt import DetectorSpec, build_prompt_and_schema
from src.services.detection.providers.gemini import (
    _EVIDENCE_ORDER,
    _SPEECH_FIELDS,
    _evidence_first_ordering,
)


def _props(*identifiers: str) -> dict:
    specs = [
        DetectorSpec(
            identifier=i, name=i, prompt_fragment="x",
            schema_fields={f"{i}_detected": {"type": "boolean"},
                           f"{i}_confidence": {"type": "number"}},
        )
        for i in identifiers
    ]
    built = build_prompt_and_schema(specs)
    return {**built.schema_properties, **_SPEECH_FIELDS}


def test_every_evidence_field_precedes_every_verdict() -> None:
    props = _props("custom_aaa", "fall", "calling_for_help")
    order = _evidence_first_ordering(props)
    evidence = set(_EVIDENCE_ORDER)
    last_evidence = max(i for i, k in enumerate(order) if k in evidence)
    first_other = min(i for i, k in enumerate(order) if k not in evidence)
    assert last_evidence < first_other, order


def test_the_transcript_is_not_stranded_after_the_verdicts() -> None:
    """`_SPEECH_FIELDS` is merged in AFTER the detector fields, so plain
    dict insertion order emits the transcript LAST — the audio half of
    the same bug, for exactly the detectors that read speech."""
    props = _props("calling_for_help")
    insertion = list(props)
    assert insertion.index("speech_transcript") > insertion.index(
        "calling_for_help_detected"
    ), "fixture no longer reproduces the stranding this guards"

    order = _evidence_first_ordering(props)
    assert order.index("speech_transcript") < order.index(
        "calling_for_help_detected"
    )


def test_ordering_is_a_permutation_of_the_properties() -> None:
    """A property missing from `property_ordering` — or named twice —
    is a malformed request, and the failure would be a silently
    degraded schema rather than an error."""
    props = _props("custom_aaa", "custom_bbb")
    order = _evidence_first_ordering(props)
    assert len(order) == len(props)
    assert set(order) == set(props)


def test_description_lands_after_the_evidence_it_summarises() -> None:
    """It justifies what was seen, so it wants the evidence above it —
    but still before any verdict, or it becomes a rationalisation of
    labels already chosen."""
    props = _props("custom_aaa")
    order = _evidence_first_ordering(props)
    assert order.index("seen") < order.index("description")
    assert order.index("description") < order.index("custom_aaa_detected")


def test_reasoning_precedes_verdict_within_each_detector():
    """Field-level evidence-before-verdict (2026-08-08): the reasoning
    text must be GENERATED before the boolean it justifies, or the
    boolean is committed first and the reasoning is post-hoc defense
    of a locked token (arXiv:2408.02442)."""
    from src.services.detection.providers.gemini import (
        _evidence_first_ordering,
    )
    props = {
        "heard": {}, "seen": {}, "description": {},
        "people_present": {}, "people_count": {},
        "fall_detected": {}, "fall_confidence": {}, "fall_reasoning": {},
        "custom_ab12_detected": {}, "custom_ab12_confidence": {},
        "custom_ab12_reasoning": {},
        "medication_taken": {}, "medication_confidence": {},
        "medication_reasoning": {},
    }
    order = _evidence_first_ordering(props)
    assert set(order) == set(props)          # nothing dropped or added
    # Evidence block still leads.
    assert order.index("heard") < order.index("seen") < order.index("description")
    assert order.index("description") < order.index("fall_detected")
    # Within every detector group: reasoning first.
    for ident, verdict in (
        ("fall", "fall_detected"),
        ("custom_ab12", "custom_ab12_detected"),
        ("medication", "medication_taken"),
    ):
        r = order.index(f"{ident}_reasoning")
        assert r < order.index(verdict), f"{ident}: verdict before reasoning"
        assert r < order.index(f"{ident}_confidence")
    # Groups without a reasoning field keep declared order.
    assert order.index("people_present") < order.index("people_count")
