"""A near miss is counted only where the declaration scopes it (finding 7.1).

A near miss that is "any record of the kind without the fact" says nothing
of the rule: every family counts one only where a parameter of the
declaration narrows the records to the rule's own situation. Each case reads
a real gen-3 fixture record.
"""

from __future__ import annotations

import pytest

from effects.domain.records import ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import source_entities
from effects.domain.rule_detectors.combat_keyword_effect import FAMILY as COMBAT_KEYWORD
from effects.domain.rule_detectors.continuous_contribution import FAMILY as CONTINUOUS
from effects.domain.rule_detectors.legality_membership import FAMILY as LEGALITY
from effects.domain.rule_detectors.outcome_event import FAMILY as OUTCOME
from effects.domain.rule_detectors.outcome_event import acting_api
from effects.domain.rule_detectors.resolution_outcome import FAMILY as RESOLUTION
from effects.domain.rule_detectors.verdict_bit import FAMILY as VERDICT

DOES_NOT_FIRE = Polarity.DOES_NOT_FIRE


def _polarity(found):
    return None if found is None else found.polarity


class TestContinuousContribution:
    @staticmethod
    def _colourless_token_static(by_kind):
        """A token's static contributing no colour: a near miss of any colour rule."""
        return next(r for r in by_kind("continuous")
                    if r.ability[0].script_file.startswith("tokenscripts/")
                    and not any(c.colors for c in r.payload.contributions))

    def test_an_unscoped_declaration_counts_no_near_miss(self, by_kind):
        record = self._colourless_token_static(by_kind)
        assert CONTINUOUS.match(record, None, {"field": "colors"}) is None

    def test_a_declaration_naming_the_static_s_source_counts_its_near_miss(self, by_kind):
        record = self._colourless_token_static(by_kind)
        host = source_entities(record.state, record.ability)[0]
        kind = (host.subtypes or host.name.split())[0].lower()
        found = CONTINUOUS.match(record, None, {"field": "colors", "source_token_kind": kind})
        assert _polarity(found) is DOES_NOT_FIRE


class TestResolutionOutcome:
    @staticmethod
    def _resolved_targeted(by_kind):
        return next(r for r in by_kind("resolution")
                    if r.moment.value == "resolution"
                    and r.payload.outcome is ResolutionOutcome.RESOLVED
                    and r.state.refs.as_cast)

    def test_a_resolved_half_is_no_near_miss_of_an_unscoped_counter_rule(self, by_kind):
        record = self._resolved_targeted(by_kind)
        assert RESOLUTION.match(record, None, {"outcome": "countered"}) is None

    def test_a_targeted_resolved_half_is_the_near_miss_of_a_targeted_rule(self, by_kind):
        record = self._resolved_targeted(by_kind)
        found = RESOLUTION.match(record, None, {"outcome": "fizzled", "targeted": True})
        assert _polarity(found) is DOES_NOT_FIRE


def _is_card(record, entity_id) -> bool:
    entity = record.state.entity(entity_id)
    return entity is not None and entity.token_script_id is None


def _cards_leave(record) -> bool:
    """Whether the record moves snapshot cards, no token among them, off the battlefield."""
    return any(e.type.value == "zone_change" and e.params.get("from_zone") == "battlefield"
               and all(_is_card(record, s) for s in e.subjects)
               for e in record.payload.events)


class TestOutcomeEventQualifier:
    DIES = {"event": "zone_change", "from_zone": "battlefield", "to_zone": "graveyard",
            "subject_is_token": True}

    @staticmethod
    def _cards_dying(by_kind, gen3_sidecars):
        """A DamageAll resolution nontoken creatures die in."""
        return next(r for r in by_kind("resolution")
                    if r.moment.value == "resolution"
                    and acting_api(r, gen3_sidecars) == "DamageAll" and _cards_leave(r))

    def test_an_unscoped_qualifier_counts_no_near_miss(self, by_kind, gen3_sidecars):
        record = self._cards_dying(by_kind, gen3_sidecars)
        assert OUTCOME.match(record, gen3_sidecars, self.DIES) is None

    def test_a_qualifier_scoped_by_api_counts_its_near_miss(self, by_kind, gen3_sidecars):
        record = self._cards_dying(by_kind, gen3_sidecars)
        found = OUTCOME.match(record, gen3_sidecars, {**self.DIES, "api": "DamageAll"})
        assert _polarity(found) is DOES_NOT_FIRE

    def test_a_combat_damage_qualifier_counts_no_near_miss(self, by_kind):
        """No combat record has an acting line, so 510.1b's qualifier has no
        near miss: a blocker's combat damage to the creature it blocks."""
        def to_a_combatant(record, event):
            subject = record.state.entity(event.subjects[0]) if event.subjects else None
            return subject is not None and subject.combat is not None
        record = next(r for r in by_kind("combat") if not r.is_probe and any(
            e.type.value == "damage_dealt" and to_a_combatant(r, e)
            for e in r.payload.listed_events()))
        assert OUTCOME.match(record, None, {"event": "damage_dealt", "combat": True,
                                            "subject_in_combat": False}) is None


class TestLegality:
    @staticmethod
    def _legal_untapped_anchor(by_kind):
        return next(r for r in by_kind("playability", "attackers")
                    if r.payload.anchor_attacker in r.payload.legal_attackers
                    and not r.state.entity(r.payload.anchor_attacker).tapped
                    and not r.payload.must_attack)

    def test_an_untapped_legal_attacker_is_no_near_miss_of_the_tapped_rule(self, by_kind):
        record = self._legal_untapped_anchor(by_kind)
        assert LEGALITY.match(record, None, {"list": "legal_attackers",
                                             "restriction": "tapped"}) is None

    def test_an_untapped_legal_blocker_is_no_near_miss_of_the_tapped_rule(self, by_kind):
        record = next(r for r in by_kind("playability", "blockers")
                      if r.payload.legal_blockers
                      and not any(r.state.entity(e).tapped for e in r.payload.legal_blockers)
                      and not any(r.state.entity(f.entity).tapped for f in r.payload.forbidden))
        assert LEGALITY.match(record, None, {"list": "legal_blockers",
                                             "restriction": "tapped"}) is None

    @pytest.mark.parametrize("params", [
        {"list": "must_attack", "fires_when": "inside"},
        {"list": "legal_attackers", "fires_when": "outside", "responsible_static": True},
    ])
    def test_an_unscoped_list_counts_no_near_miss(self, by_kind, params):
        assert LEGALITY.match(self._legal_untapped_anchor(by_kind), None, params) is None

    def test_a_list_scoped_by_the_judged_entity_counts_its_near_miss(self, by_kind):
        found = LEGALITY.match(self._legal_untapped_anchor(by_kind), None, {
            "list": "must_attack", "fires_when": "inside", "judged_type": "creature"})
        assert _polarity(found) is DOES_NOT_FIRE


class TestVigilance:
    @staticmethod
    def _attack_without_vigilance(by_kind):
        return next(r for r in by_kind("combat") if not r.is_probe and any(
            c.attacking for c in (e.combat for e in r.state.entities) if c is not None))

    def test_an_attacker_without_vigilance_is_no_near_miss_of_vigilance(self, by_kind):
        record = self._attack_without_vigilance(by_kind)
        assert COMBAT_KEYWORD.match(record, None, {"keyword": "vigilance"}) is None

    def test_it_is_the_fires_side_of_attackers_tapping(self, by_kind):
        record = self._attack_without_vigilance(by_kind)
        found = COMBAT_KEYWORD.match(record, None, {"keyword": "vigilance",
                                                    "fires_when": "absent"})
        assert _polarity(found) is Polarity.FIRES


class TestAffordable:
    @staticmethod
    def _decision(by_kind, affordable: bool, failed: set[str]):
        return next(r for r in by_kind("playability", "decision")
                    if len(r.payload.candidates) == 1
                    and r.payload.candidates[0].affordable is affordable
                    and {f.check for f in r.payload.candidates[0].failed_checks} == failed)

    def test_an_unaffordable_candidate_failing_another_check_is_no_near_miss(self, by_kind):
        record = self._decision(by_kind, False, {"casting_speed", "cost"})
        assert VERDICT.match(record, None, {"bit": "affordable", "fires_when": True}) is None

    def test_an_unaffordable_candidate_failing_cost_alone_is_the_near_miss(self, by_kind):
        record = self._decision(by_kind, False, {"cost"})
        found = VERDICT.match(record, None, {"bit": "affordable", "fires_when": True})
        assert _polarity(found) is DOES_NOT_FIRE

    def test_an_affordable_candidate_failing_another_check_is_no_near_miss(self, by_kind):
        record = self._decision(by_kind, True, {"land_play_limit"})
        assert VERDICT.match(record, None, {"bit": "affordable"}) is None

    def test_an_affordable_candidate_failing_nothing_is_the_near_miss(self, by_kind):
        record = self._decision(by_kind, True, set())
        assert _polarity(VERDICT.match(record, None, {"bit": "affordable"})) is DOES_NOT_FIRE


class TestStackEmpty:
    SPLIT_SECOND = {"bit": "failed_check", "check": "split_second", "land_play": False,
                    "stack_empty": False}

    def test_a_candidate_on_an_empty_stack_is_no_near_miss_of_split_second(self, by_kind):
        record = next(r for r in by_kind("playability", "decision")
                      if r.state.global_.stack_size == 0
                      and not r.payload.candidates[0].land_play)
        assert VERDICT.match(record, None, self.SPLIT_SECOND) is None

    def test_a_candidate_failing_split_second_on_a_held_stack_fires(self, by_kind):
        record = next(r for r in by_kind("playability", "decision")
                      if r.state.global_.stack_size > 0
                      and not r.payload.candidates[0].land_play)
        assert _polarity(VERDICT.match(record, None, self.SPLIT_SECOND)) is Polarity.FIRES

    def test_stack_empty_is_a_boolean(self):
        with pytest.raises(ValueError):
            VERDICT.validate({"bit": "can_play", "stack_empty": "yes"})
