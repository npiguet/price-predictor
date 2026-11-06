"""``legality_membership`` on real gen-2 legality records (T029)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.records import ForbiddenEntity
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.legality_membership import FAMILY


def _forbidden_block(by_kind):
    return next(r for r in by_kind("playability", "blockers") if r.payload.forbidden)


def _judged(found, record) -> tuple[str, ...]:
    """The judged entities: the condition entities but the anchor attacker."""
    anchor = getattr(record.payload, "anchor_attacker", None)
    return tuple(e for e in found.condition if e != anchor)


def test_a_creature_forbidden_to_block_fires_a_cant_block_rule(by_kind):
    record = _forbidden_block(by_kind)
    found = FAMILY.match(record, None, {"list": "legal_blockers"})
    assert found.polarity is Polarity.FIRES
    forbidden = tuple(f.entity for f in record.payload.forbidden)
    assert _judged(found, record) == forbidden
    # Each forbidden blocker's blocker_legal is written, and min_blockers at the anchor.
    assert found.outcome == (*forbidden, record.payload.anchor_attacker)
    assert record.payload.anchor_attacker in found.condition


def test_the_same_creature_edited_into_the_legal_list_is_the_near_miss(by_kind):
    record = _forbidden_block(by_kind)
    moved = tuple(f.entity for f in record.payload.forbidden)
    edited = replace(record, payload=replace(record.payload, forbidden=(),
                                             legal_blockers=moved))
    found = FAMILY.match(edited, None, {"list": "legal_blockers"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and _judged(found, edited) == moved


def test_responsible_static_counts_only_what_a_static_forbids(by_kind):
    record = _forbidden_block(by_kind)
    unattributed = replace(record, payload=replace(record.payload, forbidden=tuple(
        ForbiddenEntity(f.entity) for f in record.payload.forbidden)))
    found = FAMILY.match(unattributed, None,
                         {"list": "legal_blockers", "responsible_static": True})
    assert found is None or found.polarity is Polarity.DOES_NOT_FIRE


def test_inside_reads_the_legal_list(by_kind):
    record = next(r for r in by_kind("playability", "attackers") if r.payload.legal_attackers)
    found = FAMILY.match(record, None, {"list": "legal_attackers", "fires_when": "inside"})
    assert found.polarity is Polarity.FIRES
    assert found.condition == found.outcome == record.payload.legal_attackers


class TestUniverse:
    def test_a_noncreature_the_attackers_record_never_lists_is_outside(self, by_kind):
        record = next(r for r in by_kind("playability", "attackers") if any(
            e.on_battlefield and not e.is_creature and e.controller == r.state.global_.active
            for e in r.state.entities))
        found = FAMILY.match(record, None, {"list": "legal_attackers",
                                            "universe": "battlefield"})
        assert found.polarity is Polarity.FIRES
        outside = [record.state.entity(e) for e in found.condition]
        assert any(not e.is_creature for e in outside)
        assert all(e.controller == record.state.global_.active for e in outside)
        assert not set(found.entities) & set(record.payload.legal_attackers)
        # A permanent no attackers record lists has no attacker_legal target:
        # the outside is the condition alone, the forbidden ones its outcome.
        forbidden = {f.entity for f in record.payload.forbidden}
        assert set(found.outcome) == set(found.condition) & forbidden

    def test_the_blockers_universe_is_the_defending_players(self, by_kind):
        record = by_kind("playability", "blockers")[0]
        found = FAMILY.match(record, None, {"list": "legal_blockers",
                                            "universe": "battlefield"})
        active = record.state.global_.active
        assert all(record.state.entity(e).controller != active for e in _judged(found, record))

    def _targeting(self, by_kind):
        return next(r for r in by_kind("playability", "decision")
                    if 0 < len(r.payload.candidates[0].legal_targets) < 32)

    def test_what_a_targeting_candidate_does_not_list_is_outside(self, by_kind):
        record = self._targeting(by_kind)
        legal = set(record.payload.candidates[0].legal_targets)
        found = FAMILY.match(record, None, {"list": "legal_targets",
                                            "universe": "battlefield_and_players"})
        assert found.polarity is Polarity.FIRES and not set(found.entities) & legal
        players = {p.id for p in record.state.players}
        assert players - legal <= set(found.entities)
        # target_legal is written at every card and player of the decision.
        assert found.outcome == found.condition
        inside = FAMILY.match(record, None, {"list": "legal_targets", "fires_when": "inside",
                                             "universe": "battlefield"})
        assert inside.polarity is Polarity.FIRES and set(inside.entities) == legal

    def test_a_capped_or_untargeted_candidate_speaks_for_no_outside(self, by_kind):
        record = self._targeting(by_kind)
        candidate = record.payload.candidates[0]
        declared = {"list": "legal_targets", "universe": "battlefield"}
        capped = replace(candidate, legal_targets=tuple(f"E{i}" for i in range(32)))
        untargeted = replace(candidate, legal_targets=(), has_legal_target=True)
        for edited in (capped, untargeted):
            payload = replace(record.payload, candidates=(edited,))
            assert FAMILY.match(replace(record, payload=payload), None, declared) is None

    def test_an_unknown_universe_is_refused(self, by_kind):
        with pytest.raises(ValueError, match="universe"):
            FAMILY.match(by_kind("playability", "attackers")[0], None,
                         {"list": "legal_attackers", "universe": "graveyard"})


def test_the_list_must_match_the_subkind(by_kind):
    record = _forbidden_block(by_kind)
    assert FAMILY.match(record, None, {"list": "legal_attackers"}) is None


class TestTheGen3Lists:
    """``legal_defenders`` and ``must_block`` on the Java goldens' gen-3 records."""

    @staticmethod
    def _golden(subkind: str):
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        return [record_from_dict(raw) for raw in java_golden_records()
                if raw["kind"] == "playability" and raw["subkind"] == subkind]

    def _anchor(self, anchor: str):
        return next(r for r in self._golden("attackers") if r.payload.anchor_attacker == anchor)

    def test_a_defender_the_anchor_may_attack_is_inside(self):
        found = FAMILY.match(self._anchor("E2"), None,
                             {"list": "legal_defenders", "fires_when": "inside"})
        assert found.polarity is Polarity.FIRES and found.condition == ("P1", "E2")
        assert found.outcome == ("P1",)

    def test_a_creature_that_may_attack_nothing_lists_no_inside(self):
        assert FAMILY.match(self._anchor("E3"), None,
                            {"list": "legal_defenders", "fires_when": "inside"}) is None

    def test_the_defender_universe_is_every_other_permanent_and_the_opponent(self):
        record = self._anchor("E2")
        found = FAMILY.match(record, None, {"list": "legal_defenders",
                                            "universe": "battlefield_and_players"})
        assert found.polarity is Polarity.FIRES
        judged = _judged(found, record)
        assert "E2" not in judged and not {"P0", "P1"} & set(judged)
        assert set(judged) == {e.id for e in record.state.entities
                               if e.on_battlefield and e.id != "E2"}

    def test_a_creature_no_requirement_binds_is_outside_must_block(self):
        record = self._golden("blockers")[0]
        found = FAMILY.match(record, None, {"list": "must_block", "fires_when": "inside"})
        assert found.polarity is Polarity.DOES_NOT_FIRE
        assert _judged(found, record) == (*record.payload.legal_blockers,
                                          *(f.entity for f in record.payload.forbidden))

    def test_a_creature_that_must_block_is_inside(self):
        record = self._golden("blockers")[0]
        bound = record.payload.legal_blockers[:1]
        edited = replace(record, payload=replace(record.payload, must_block=bound))
        found = FAMILY.match(edited, None, {"list": "must_block", "fires_when": "inside"})
        assert found.polarity is Polarity.FIRES and _judged(found, edited) == bound
        assert found.outcome == (*bound, record.payload.anchor_attacker)

    def test_a_gen2_record_shows_neither_list(self, by_kind):
        assert FAMILY.match(by_kind("playability", "attackers")[0], None,
                            {"list": "legal_defenders", "universe": "battlefield"}) is None
        assert FAMILY.match(by_kind("playability", "blockers")[0], None,
                            {"list": "must_block"}) is None

    def test_both_lists_need_their_gen3_field(self):
        assert FAMILY.needs({"list": "legal_defenders"}) == {"legal_defenders"}
        assert FAMILY.needs({"list": "must_block"}) == {"must_block"}
        assert FAMILY.needs({"list": "legal_blockers"}) == frozenset()


class TestKeywordAndCounter:
    """A keyword's restriction while a counter is on its carrier (unleash, 702.98a)."""

    @staticmethod
    def _carrier(by_kind, sidecars):
        """A blockers record, one legal blocker whose card prints a keyword, and the keyword."""
        for record in by_kind("playability", "blockers"):
            for entity_id in record.payload.legal_blockers:
                entity = record.state.entity(entity_id)
                if entity is None or not entity.printed:
                    continue
                try:
                    lines = sidecars.get(entity.printed[0].script_file).lines
                except KeyError:
                    continue
                keyword = next((line.keyword_display_name for line in lines
                                if line.keyword_display_name), None)
                if keyword is not None:
                    return record, entity_id, keyword
        raise AssertionError("no legal blocker printing a keyword")

    @staticmethod
    def _edited(record, entity_id, *, counters, forbidden):
        entities = tuple(replace(e, counters=counters) if e.id == entity_id else e
                         for e in record.state.entities)
        payload = record.payload
        legal = tuple(e for e in payload.legal_blockers if e != entity_id)
        if forbidden:
            payload = replace(payload, legal_blockers=legal, forbidden=(
                *payload.forbidden, ForbiddenEntity(entity_id)))
        return replace(record, state=replace(record.state, entities=entities), payload=payload)

    def _declared(self, keyword):
        return {"list": "legal_blockers", "keyword": keyword, "counter_type": "+1/+1"}

    def test_a_carrier_kept_out_while_holding_the_counter_fires(self, by_kind, gen2_sidecars):
        record, entity_id, keyword = self._carrier(by_kind, gen2_sidecars)
        edited = self._edited(record, entity_id, counters={"P1P1": 1}, forbidden=True)
        found = FAMILY.match(edited, gen2_sidecars, self._declared(keyword))
        assert found.polarity is Polarity.FIRES and entity_id in found.entities

    def test_a_legal_carrier_without_the_counter_is_the_near_miss(self, by_kind, gen2_sidecars):
        record, entity_id, keyword = self._carrier(by_kind, gen2_sidecars)
        edited = self._edited(record, entity_id, counters={}, forbidden=False)
        found = FAMILY.match(edited, gen2_sidecars, self._declared(keyword))
        assert found.polarity is Polarity.DOES_NOT_FIRE and entity_id in found.entities

    def test_a_carrier_on_the_side_its_counter_does_not_explain_is_not_counted(
            self, by_kind, gen2_sidecars):
        record, entity_id, keyword = self._carrier(by_kind, gen2_sidecars)
        for counters, forbidden in (({}, True), ({"+1/+1": 2}, False)):
            edited = self._edited(record, entity_id, counters=counters, forbidden=forbidden)
            found = FAMILY.match(edited, gen2_sidecars, self._declared(keyword))
            assert found is None or entity_id not in found.entities

    def test_an_entity_whose_card_lacks_the_keyword_is_not_counted(
            self, by_kind, gen2_sidecars):
        record, entity_id, _ = self._carrier(by_kind, gen2_sidecars)
        edited = self._edited(record, entity_id, counters={"P1P1": 1}, forbidden=True)
        assert FAMILY.match(edited, gen2_sidecars, self._declared("No Such Keyword")) is None


class TestAnchorLists:
    """``must_attack`` and ``min_blockers`` judge the anchor attacker (T030 follow-up)."""

    @staticmethod
    def _attackers():
        return TestTheGen3Lists._golden("attackers")[0]

    def test_an_anchor_an_attack_requirement_binds_is_inside_must_attack(self):
        record = self._attackers()
        bound = replace(record, payload=replace(record.payload, must_attack=True))
        found = FAMILY.match(bound, None, {"list": "must_attack", "fires_when": "inside"})
        anchor = record.payload.anchor_attacker
        assert found.polarity is Polarity.FIRES and found.condition == (anchor,)
        # The requirement is written at the anchor.
        assert found.outcome == (anchor,)
        free = replace(record, payload=replace(record.payload, must_attack=False))
        found = FAMILY.match(free, None, {"list": "must_attack", "fires_when": "inside"})
        assert found.polarity is Polarity.DOES_NOT_FIRE and found.outcome == (anchor,)

    def test_must_attack_needs_the_gen3_record(self, by_kind):
        assert FAMILY.needs({"list": "must_attack"}) == {"must_attack"}
        assert FAMILY.match(by_kind("playability", "attackers")[0], None,
                            {"list": "must_attack"}) is None

    def test_an_anchor_needing_two_blockers_is_inside_min_blockers(self, by_kind):
        record = by_kind("playability", "blockers")[0]
        anchor = record.payload.anchor_attacker
        declared = {"list": "min_blockers", "fires_when": "inside"}
        menace = replace(record, payload=replace(record.payload, min_blockers=2))
        found = FAMILY.match(menace, None, declared)
        assert found.polarity is Polarity.FIRES
        assert found.condition == (anchor,) and found.outcome == (anchor,)
        lone = replace(record, payload=replace(record.payload, min_blockers=1))
        assert FAMILY.match(lone, None, declared).polarity is Polarity.DOES_NOT_FIRE
        assert FAMILY.match(menace, None, {**declared, "at_least": 3}).polarity is (
            Polarity.DOES_NOT_FIRE)
        assert FAMILY.needs(declared) == frozenset()

    def test_at_least_reads_only_min_blockers(self, by_kind):
        with pytest.raises(ValueError, match="at_least"):
            FAMILY.match(by_kind("playability", "blockers")[0], None,
                         {"list": "legal_blockers", "at_least": 2})


class TestJudgedKind:
    """``judged_type`` and ``min_types`` narrow the judged entities (T030 follow-up)."""

    def test_only_a_judged_battle_counts(self):
        record = next(r for r in TestTheGen3Lists._golden("attackers")
                      if r.payload.anchor_attacker == "E2")
        declared = {"list": "legal_defenders", "fires_when": "inside",
                    "universe": "battlefield", "judged_type": "Battle"}
        assert FAMILY.match(record, None, declared) is None
        battle = next(e for e in record.state.entities if e.on_battlefield and e.id != "E2")
        edited = replace(record, state=replace(record.state, entities=tuple(
            replace(e, types=("battle",)) if e.id == battle.id else e
            for e in record.state.entities)))
        found = FAMILY.match(edited, None, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE
        assert _judged(found, edited) == (battle.id,)

    def test_min_types_keeps_a_judged_entity_of_two_card_types(self, by_kind):
        record = _forbidden_block(by_kind)
        judged = record.payload.forbidden[0].entity
        declared = {"list": "legal_blockers", "min_types": 2}
        two = replace(record, state=replace(record.state, entities=tuple(
            replace(e, types=("artifact", "creature")) if e.id == judged
            else replace(e, types=("creature",)) for e in record.state.entities)))
        found = FAMILY.match(two, None, declared)
        assert found.polarity is Polarity.FIRES and _judged(found, two) == (judged,)
        one = replace(two, state=replace(two.state, entities=tuple(
            replace(e, types=("creature",)) for e in two.state.entities)))
        found = FAMILY.match(one, None, declared)
        assert found is None or judged not in _judged(found, one)
