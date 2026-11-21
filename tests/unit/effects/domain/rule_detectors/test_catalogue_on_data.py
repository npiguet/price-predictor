"""The real catalogue's declarations tagging the test fixtures' real records.

Two reports a whole-catalogue reading needs and no single detector test can
give: a declaration that reads no situation at all tags most records of its
kind ("unscoped"), and two scenarios of different rules, not linked by
``same_as``, whose ``fires`` and near-miss records are the same records tag
one population (over records of both sides, at least two of the fires side:
a sample showing one rule happening once cannot tell two rules apart). Each
is listed; a declaration that joins either list fails here until it is
scoped, linked or added to the list below with its reason.
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

#: A declaration tagging more than this share of its record kind's records
#: (fires and near miss together), over at least ``_MIN_KIND`` records of the
#: kind, reads no situation of its own: a rule's condition holds on a minority
#: of what a kind records, and the broadest legitimate conditions measured
#: here (a combat dealing damage, any creature among the legal blockers) stay
#: under it.
UNSCOPED_SHARE = 0.5
_MIN_KIND = 20
#: Declarations above the share whose rule is the kind's own general fact.
GENERAL = {
    # every combat damage step deals damage equal to power (510.1a, 120.2a)
    "510.1a-1",
    # combat damage to a player is life lost, unless the source has infect
    "119.2-1", "120.3a-1",
    # combat damage to a creature is marked on it, unless wither or infect
    "120.3e-1",
    # a player losing life to combat damage is what 704.5a's check reads: its
    # near miss is every combat damage step to a player that leaves them alive
    "704.5a-2",
}
#: Declarations above the share because their near miss, by its declared
#: text, is any record of the kind short of the fires side, so the two sides
#: together cover the kind. The 2026-11-15 fixture is the first whose
#: continuous and attackers records reach ``_MIN_KIND`` and whose effect
#: halves carry their outcome, which is what brought each here; scoping a
#: near miss narrower is a catalogue change, out of this list's reach.
BROAD_NEAR_MISS = {
    # A static that applies to the board and contributes nothing of the
    # declared field (``continuous_contribution``).
    "105.3-1", "113.11-1", "113.11-2", "202.2f-1", "205.1a-1", "208.2a-1", "305.7-1",
    "604.2-1", "613.1d-1", "613.6-1", "613.6-2", "702.114a-1", "702.22b-1", "702.73a-1",
    # A stack object that leaves the stack resolved, or countered
    # (``resolution_outcome``): most effect halves resolve.
    "701.5a-1", "701.6a-1", "701.6a-2",
    # The anchored attacker judged against its record's own lists
    # (``legality_membership``), which every attackers record holds.
    "306.6-1", "506.3-1", "506.3-3", "508.1a-1", "508.1c-2", "508.1d-1",
    # An attacker with or without vigilance: every attack is one or the other.
    "508.1f-1", "702.20b-1",
    # Combat damage to what an unblocked attacker attacks, or to a blocked
    # one's blockers: each one's near miss is the other's fires side.
    "510.1b-1", "510.1c-1",
    # A non-land-play candidate that does not fail the named check
    # (``verdict_bit``): most candidates pass the split-second check.
    "702.61a-1",
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
    sides: dict[str, dict[Polarity, set[str]]] = defaultdict(lambda: defaultdict(set))
    kinds: dict[str, set[str]] = defaultdict(set)
    rules = {}
    for number, (records, lines) in enumerate(sets):
        for record in records:
            rid = f"{number}:{record.record_id}"
            kinds[_label(record)].add(rid)
            for scenario, declaration, family in index.by_kind.get(record.kind.value, ()):
                found = family.match(record, lines, declaration.params)
                if found is not None:
                    sides[scenario.id][found.polarity].add(rid)
                    sides[scenario.id]["label"] = _label(record)
                    rules[scenario.id] = scenario.rules
    return sides, kinds, rules


def unscoped(tagged) -> list[tuple[str, float]]:
    sides, kinds, _ = tagged
    found = []
    for sid, polarity in sides.items():
        kind = kinds[polarity["label"]]
        hits = polarity[Polarity.FIRES] | polarity[Polarity.DOES_NOT_FIRE]
        if len(kind) >= _MIN_KIND and len(hits) > UNSCOPED_SHARE * len(kind):
            found.append((sid, len(hits) / len(kind)))
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


def test_no_declaration_tags_most_of_its_kind_unless_its_rule_is_the_kind_s_own(tagged):
    found = unscoped(tagged)
    print("unscoped:", found)
    assert {sid for sid, _ in found} <= GENERAL | BROAD_NEAR_MISS, found


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
