"""``resolution_outcome`` on real gen-2 cost halves, and on gen-3 effect halves."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.event_schema import Event, EventType
from effects.domain.records import ActivationPayload, Costs, ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.resolution_outcome import FAMILY


def _resolved(by_kind):
    return next(r for r in by_kind("resolution") if r.moment.value == "activation"
                and r.payload.outcome is ResolutionOutcome.RESOLVED)


def test_a_resolved_spell_is_the_near_miss_of_a_fizzle_rule(by_kind):
    found = FAMILY.match(_resolved(by_kind), None, {"outcome": "fizzled"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_the_same_spell_edited_to_fizzle_fires(by_kind):
    record = _resolved(by_kind)
    fizzled = replace(record, link_id=None,
                      payload=replace(record.payload, outcome=ResolutionOutcome.FIZZLED))
    found = FAMILY.match(fizzled, None, {"outcome": "fizzled"})
    assert found.polarity is Polarity.FIRES and found.observed
    refs = record.state.refs
    assert found.outcome == (refs.source,)
    assert found.condition == tuple(dict.fromkeys((refs.source, *refs.targets)))


def test_targeted_counts_only_stack_objects_that_had_a_target(by_kind):
    record = _resolved(by_kind)
    untargeted = replace(record, state=replace(
        record.state, refs=replace(record.state.refs, targets=())))
    assert FAMILY.match(untargeted, None, {"outcome": "fizzled", "targeted": True}) is None


def test_an_outcome_outside_both_sides_shows_nothing(by_kind):
    record = _resolved(by_kind)
    declined = replace(record, link_id=None,
                       payload=replace(record.payload, outcome=ResolutionOutcome.DECLINED))
    assert FAMILY.match(declined, None, {"outcome": "countered"}) is None


def test_effect_halves_show_nothing(by_kind):
    effect = next(r for r in by_kind("resolution") if r.moment.value == "resolution")
    assert FAMILY.match(effect, None, {"outcome": "fizzled"}) is None


# -- gen-3: the outcome is the effect half's (FR-042b) ------------------------


def _effect_half(by_kind, **payload):
    effect = next(r for r in by_kind("resolution") if r.moment.value == "resolution")
    return replace(effect, payload=replace(effect.payload, **payload))


def test_a_gen3_effect_half_carries_its_outcome(by_kind):
    fizzled = _effect_half(by_kind, events=(), outcome=ResolutionOutcome.FIZZLED)
    found = FAMILY.match(fizzled, None, {"outcome": "fizzled", "targeted": True})
    assert found.polarity is Polarity.FIRES and found.observed
    resolved = _effect_half(by_kind, outcome=ResolutionOutcome.RESOLVED)
    assert FAMILY.match(resolved, None, {"outcome": "fizzled"}).polarity is (
        Polarity.DOES_NOT_FIRE)


def test_a_countering_half_shows_the_countered_object(by_kind):
    """A countered item writes no effect half: the countering clause's holds
    its ``spell_countered``, whose subject is the object that left the stack."""
    countering = _effect_half(
        by_kind, outcome=ResolutionOutcome.RESOLVED,
        events=(Event(EventType.SPELL_COUNTERED, subjects=("E99",)),))
    found = FAMILY.match(countering, None, {"outcome": "countered"})
    assert found.polarity is Polarity.FIRES
    assert found.condition == ("E99",) and found.outcome == ("E99",)
    near = FAMILY.match(countering, None, {"outcome": "resolved", "near": ["countered"]})
    assert near.polarity is Polarity.DOES_NOT_FIRE and near.outcome == ("E99",)


def test_a_gen3_cost_half_shows_nothing(by_kind):
    cost = next(r for r in by_kind("resolution") if r.moment.value == "activation")
    gen3 = replace(cost, payload=ActivationPayload(costs=Costs(), events=()))
    assert FAMILY.match(gen3, None, {"outcome": "resolved", "near": ["fizzled"]}) is None
