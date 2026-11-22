"""``resolution_outcome`` on real gen-3 effect halves, which carry the outcome (FR-042b)."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.event_schema import Event, EventType
from effects.domain.records import ActivationPayload, Costs, ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.resolution_outcome import FAMILY


def _resolved(by_kind):
    """A real effect half of a spell or ability that resolved with a target."""
    return next(r for r in by_kind("resolution") if r.moment.value == "resolution"
                and r.ability and r.payload.outcome is ResolutionOutcome.RESOLVED
                and r.state.refs.targets_at_cast)


def test_a_resolved_spell_is_the_near_miss_of_a_targeted_fizzle_rule(by_kind):
    found = FAMILY.match(_resolved(by_kind), None, {"outcome": "fizzled", "targeted": True})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    # Unscoped, every resolution would be a near miss: none is (finding 7.1).
    assert FAMILY.match(_resolved(by_kind), None, {"outcome": "fizzled"}) is None


def test_the_same_spell_edited_to_fizzle_fires(by_kind):
    record = _resolved(by_kind)
    fizzled = replace(record,
                      payload=replace(record.payload, outcome=ResolutionOutcome.FIZZLED))
    found = FAMILY.match(fizzled, None, {"outcome": "fizzled"})
    assert found.polarity is Polarity.FIRES and found.observed
    refs = record.state.refs
    assert found.outcome == (refs.source,)
    assert found.condition == tuple(dict.fromkeys((refs.source, *refs.targets)))


def test_targeted_counts_only_stack_objects_that_had_a_target(by_kind):
    record = _resolved(by_kind)
    untargeted = replace(record, state=replace(
        record.state, refs=replace(record.state.refs, targets=(), targets_at_cast=())))
    assert FAMILY.match(untargeted, None, {"outcome": "fizzled", "targeted": True}) is None


def test_an_outcome_outside_both_sides_shows_nothing(by_kind):
    record = _resolved(by_kind)
    declined = replace(record,
                       payload=replace(record.payload, outcome=ResolutionOutcome.DECLINED))
    assert FAMILY.match(declined, None, {"outcome": "countered"}) is None


def _effect_half(by_kind, **payload):
    effect = _resolved(by_kind)
    return replace(effect, payload=replace(effect.payload, **payload))


def test_an_effect_half_carries_its_outcome(by_kind):
    fizzled = _effect_half(by_kind, events=(), outcome=ResolutionOutcome.FIZZLED)
    found = FAMILY.match(fizzled, None, {"outcome": "fizzled", "targeted": True})
    assert found.polarity is Polarity.FIRES and found.observed
    resolved = _effect_half(by_kind, outcome=ResolutionOutcome.RESOLVED)
    assert FAMILY.match(resolved, None, {"outcome": "fizzled", "targeted": True}).polarity is (
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
    near = FAMILY.match(countering, None, {"outcome": "resolved", "near": ["countered"],
                                           "targeted": True})
    assert near.polarity is Polarity.DOES_NOT_FIRE and near.outcome == ("E99",)


def test_a_countering_half_shows_the_countered_ability_by_its_host(by_kind):
    """A countered ability (701.6a) is the countering half's
    ``ability_countered``, naming it by its line on its host: the host is the
    object the outcome reads, though no subject of the event."""
    line = {"entity": "E98", "key": {"script_file": "cardsfolder/p/prodigal_sorcerer.txt",
                                     "face": 0, "trait_kind": "spell", "index_within_kind": 1}}
    countering = _effect_half(
        by_kind, outcome=ResolutionOutcome.RESOLVED,
        events=(Event(EventType.ABILITY_COUNTERED, params={"ability": line}),))
    found = FAMILY.match(countering, None, {"outcome": "countered"})
    assert found.polarity is Polarity.FIRES
    assert found.condition == ("E98",) and found.outcome == ("E98",)


def test_a_cost_half_shows_nothing(by_kind):
    cost = next(r for r in by_kind("resolution") if r.moment.value == "activation")
    stripped = replace(cost, payload=ActivationPayload(costs=Costs(), events=()))
    assert FAMILY.match(stripped, None, {"outcome": "resolved", "near": ["fizzled"]}) is None


def test_a_fizzled_half_reads_its_targets_as_cast():
    """V5 #25: an effect half's ``targets`` lose the illegal targets; the
    targets it was cast with are ``targets_at_cast`` (``Refs.as_cast``)."""
    from effects.domain.records import ResolutionPayload
    from effects.domain.state_snapshot import Refs
    from tests.unit.effects.builders import make_record, make_snapshot

    def half(outcome, targets, at_cast):
        state = make_snapshot(refs=Refs(source="E1", targets=targets, targets_at_cast=at_cast))
        return make_record(state=state, payload=ResolutionPayload(outcome=outcome),
                           generation=3)

    params = {"outcome": "fizzled", "targeted": True}
    fizzled = FAMILY.match(half(ResolutionOutcome.FIZZLED, (), ("E2",)), None, params)
    assert fizzled.polarity is Polarity.FIRES and fizzled.condition == ("E1", "E2")
    untargeted = half(ResolutionOutcome.RESOLVED, (), ())
    assert FAMILY.match(untargeted, None, params) is None
    resolved = FAMILY.match(half(ResolutionOutcome.RESOLVED, ("E2",), ("E2",)), None, params)
    assert resolved.polarity is Polarity.DOES_NOT_FIRE


class _Keyworded:
    """A sidecar lookup whose every card prints one keyword line, the acting
    line among them."""

    def __init__(self, keyword):
        from types import SimpleNamespace

        from effects.domain.provenance import KEYWORD_API_TYPE, SidecarLine

        self.line = SidecarLine(line_index=0, line_kind="trigger", provenance=(),
                                script_api_type=KEYWORD_API_TYPE, script_text=keyword,
                                prose_offset=0)
        self.sidecar = SimpleNamespace(lines=(self.line,))

    def first_line(self, keys):
        return self.line

    def get(self, script_file):
        return self.sidecar


def test_keyword_and_source_keyword_read_the_acting_card_s_line(by_kind):
    """Ward's counter (702.21a) acts through ward's line; a gift spell's
    (702.174) card prints gift."""
    record = _resolved(by_kind)
    for param in ("keyword", "source_keyword"):
        declared = {"outcome": "fizzled", param: "Ward"}
        assert FAMILY.match(record, _Keyworded("Ward"), declared) is not None
        assert FAMILY.match(record, _Keyworded("Gift"), declared) is None
        assert FAMILY.match(record, None, declared) is None
