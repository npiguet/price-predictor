"""The real catalogue's declarations tagging the test fixtures' real records.

Two reports a whole-catalogue reading needs and no single detector test can
give: a declaration that reads no situation at all tags most records of its
kind ("unscoped"), and two scenarios of different rules, not linked by
``same_as``, whose ``fires`` and near-miss records are the same records tag
one population (over records of both sides, at least two of the fires side:
a sample showing one rule happening once cannot tell two rules apart). Each
is listed; a declaration that joins either list fails here until it is
scoped or linked, or, for the first, its rule is the kind's own general fact.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from effects.domain.rule_catalogue import DetectorDeclaration, Polarity
from effects.domain.rule_detectors import DetectorIndex
from effects.infrastructure.rules_catalogue_io import load_scenarios

RULES = Path(__file__).parents[5] / "rules"
FIXTURES = Path(__file__).parents[4] / "fixtures" / "effects"

#: A side of a declaration (its ``fires`` records, or its near-miss records)
#: holding more than this share of its record kind's records, over at least
#: ``_MIN_KIND`` records of the kind, reads no situation of its own: a rule's
#: condition holds on a minority of what a kind records, and a near miss is
#: counted only where the declaration scopes it.
UNSCOPED_SHARE = 0.5
_MIN_KIND = 20
#: Declarations above the share whose rule is the kind's own general fact.
GENERAL = {
    # every combat damage step deals damage equal to power (510.1a, 120.2a)
    "510.1a-1",
    # an unblocked attacker's damage goes to the player it attacks (510.1b):
    # every unblocked attack on a player
    "510.1b-1",
    # combat damage to a player is life lost, unless the source has infect
    "119.2-1", "120.3a-1",
    # combat damage to a creature is marked on it, unless wither or infect
    "120.3e-1",
    # a player losing life to combat damage is what 704.5a's check reads: its
    # near miss is every combat damage step to a player that leaves them alive
    "704.5a-2",
    # combat damage to a creature in combat is a blocked attacker's to its
    # blockers (510.1c): 120.3e-1's records
    "510.1c-1",
    # an attacking creature without vigilance taps (508.1f): every attack
    "508.1f-1",
    # only creatures attack, and a creature is no attack target (506.3): every
    # attackers record leaves its noncreature permanents out, and its creatures
    # out of the defenders
    "506.3-1", "506.3-3",
    # a static applies while its source is on the battlefield (604.2): every
    # continuous record is one, and its other side writes no record
    "604.2-1",
    # each turn-action kind's own action: a draw step draws (504.1), an untap step
    # untaps (703.4c), a cleanup step discards to hand size (514.1)
    "504.1-1", "703.4c-1", "514.1-1",
}


def _label(record) -> str:
    discriminator = record.subkind or record.moment
    return record.kind.value + (f"/{discriminator.value}" if discriminator else "")


@pytest.fixture(scope="module")
def tagged():
    """Per scenario its fires and near-miss record ids, and per kind label its
    records, over the gen-3 fixture and the Java goldens."""
    from effects.infrastructure.converted_trees import SourceTrees
    from effects.infrastructure.record_codec import record_from_dict
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS, java_golden_records

    sets = [
        (read_shard(FIXTURES / "gen3-records.jsonl.gz"),
         SourceTrees.of(list(GEN3_SIDECAR_FOLDERS)).sidecar_cache()),
        ([record_from_dict(raw) for raw in java_golden_records()], None),
    ]
    index = DetectorIndex.of(load_scenarios(RULES))
    matchers = {kind: [(scenario, family.compile(declaration.params))
                       for scenario, declaration, family in entries]
                for kind, entries in index.by_kind.items()}
    sides: dict[str, dict[Polarity, set[str]]] = defaultdict(lambda: defaultdict(set))
    kinds: dict[str, set[str]] = defaultdict(set)
    rules = {}
    for number, (records, lines) in enumerate(sets):
        for record in records:
            rid = f"{number}:{record.record_id}"
            kinds[_label(record)].add(rid)
            for scenario, matcher in matchers.get(record.kind.value, ()):
                found = matcher(record, lines)
                if found is not None:
                    sides[scenario.id][found.polarity].add(rid)
                    sides[scenario.id]["label"] = _label(record)
                    rules[scenario.id] = scenario.rules
    return sides, kinds, rules


def unscoped(tagged) -> list[tuple[str, str, float]]:
    """``(scenario, side, share)`` of every side above ``UNSCOPED_SHARE``."""
    sides, kinds, _ = tagged
    found = []
    for sid, polarity in sides.items():
        kind = kinds[polarity["label"]]
        for side in (Polarity.FIRES, Polarity.DOES_NOT_FIRE):
            hits = polarity[side]
            if len(kind) >= _MIN_KIND and len(hits) > UNSCOPED_SHARE * len(kind):
                found.append((sid, side.value, len(hits) / len(kind)))
    return sorted(found)


def identical(tagged) -> list[tuple[str, str]]:
    sides, _, rules = tagged
    links = {s.id: set(getattr(s.detector, "scenarios", ()))
             for s in load_scenarios(RULES).scenarios}
    by_sets = defaultdict(list)
    for sid, polarity in sides.items():
        key = (frozenset(polarity[Polarity.FIRES]), frozenset(polarity[Polarity.DOES_NOT_FIRE]))
        # Both sides must hold records: a sample showing one rule's fires side
        # on a single record, or no near miss, cannot tell two rules apart.
        if len(key[0]) >= 2 and key[1]:
            by_sets[key].append(sid)
    pairs = []
    for group in by_sets.values():
        for i, a in enumerate(sorted(group)):
            for b in sorted(group)[i + 1:]:
                if set(rules[a]) != set(rules[b]) and not (links[a] & {b} or links[b] & {a}):
                    pairs.append((a, b))
    return pairs


def test_no_side_of_a_declaration_tags_most_of_its_kind_unless_its_rule_is_the_kind_s_own(
        tagged):
    found = unscoped(tagged)
    print("unscoped:", found)
    assert {sid for sid, _, _ in found} <= GENERAL, found


@pytest.mark.parametrize("sid", ["105.3-1", "113.11-1", "113.11-2", "202.2f-1", "205.1a-1",
                                 "305.7-1", "702.114a-1", "702.22b-1"])
def test_an_unscoped_static_rule_has_no_near_miss_on_a_static_that_says_nothing_of_it(
        tagged, sid):
    sides, _, _ = tagged
    assert not sides.get(sid, {}).get(Polarity.DOES_NOT_FIRE)


def test_no_two_rules_tag_the_same_records(tagged):
    found = identical(tagged)
    print("identical tag sets:", found)
    assert not found


def test_the_checks_read_records_of_every_kind(tagged):
    _, kinds, _ = tagged
    assert {"resolution/resolution", "combat", "trigger", "playability/decision"} <= set(kinds)


def test_every_declared_scenario_is_a_declaration():
    assert all(isinstance(s.detector, DetectorDeclaration)
               for s in load_scenarios(RULES).scenarios
               if s.detector is not None and hasattr(s.detector, "family"))


#: Each turn-action kind's own fact (504.1, 703.4c, 514.1): it fires on every record of its
#: kind (pilot 5: all 9,711 draws, 7,273 untaps and 49 cleanup discards), yet the fixtures
#: hold fewer than ``_MIN_KIND`` records of each kind, so the check never reads them.
TURN_ACTION_FACTS = {"504.1-1", "703.4c-1", "514.1-1"}


def test_the_turn_action_kinds_own_facts_are_general():
    assert TURN_ACTION_FACTS <= GENERAL


def test_every_general_declaration_is_a_declared_scenario():
    declared = {s.id for s in load_scenarios(RULES).scenarios}
    assert GENERAL <= declared
