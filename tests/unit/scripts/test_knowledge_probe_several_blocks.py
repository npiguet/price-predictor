"""The gen-3 combats with several blocked attackers (T163)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from effect_knowledge_probes import (
    combat_grid,
    selection,
)

from tests.unit.scripts.knowledge_probe_support import (
    deaths_by_the_rule,
    one_on_one,
    report_outcomes,
)


def _copy_of_pair_member(entity, attacker, blocker, copy_ids):
    """A copy of the attacker or the blocker under its new id and the new pair's block."""
    a2, b2 = copy_ids
    if entity.id == attacker:
        return replace(entity, id=a2, combat=replace(entity.combat, blocked_by=(b2,)))
    if entity.id == blocker:
        return replace(entity, id=b2, combat=replace(entity.combat, blocking=(a2,)))
    return None


def _two_pairs(records, monkeypatch):
    """A fixture one-on-one block with a second pair beside it (copies of the
    first, under new ids), every outcome the rule's."""

    record, attacker, blocker = one_on_one(records)
    copy_ids = (f"{attacker}~2", f"{blocker}~2")
    copies = [c for e in record.state.entities
              if (c := _copy_of_pair_member(e, attacker, blocker, copy_ids))]
    record = replace(
        record,
        payload=replace(record.payload, blocks={attacker: (blocker,), copy_ids[0]: (copy_ids[1],)}),
        state=replace(record.state, entities=(*record.state.entities, *copies)))
    outcome = {**deaths_by_the_rule(record, attacker, blocker),
               **deaths_by_the_rule(record, *copy_ids)}
    report_outcomes(monkeypatch, outcome)
    return record, (attacker, blocker), copy_ids


def _plain_keywords(entity):
    return set(entity.granted_temporary.keywords)


class TestSeveralBlocks:
    def test_several_blocked_attackers_qualify_for_their_own_grid_only(
        self, records, monkeypatch,
    ):
        record, first, second = _two_pairs(records, monkeypatch)
        assert combat_grid.single_block(record, _plain_keywords) is None
        assert combat_grid.several_blocks(record, _plain_keywords) == min(first, second)

    def test_a_pair_that_breaks_the_rule_disqualifies_the_combat(self, records, monkeypatch):
        import effects.domain.effect_targets as effect_targets

        record, first, second = _two_pairs(records, monkeypatch)
        outcome = dict(effect_targets.derive_targets(record))
        flipped = "zone_outcome" not in outcome[second[1]].fields
        outcome[second[1]] = SimpleNamespace(
            fields={"zone_outcome": 1} if flipped else {}, unknown=set())
        report_outcomes(monkeypatch, outcome)
        assert combat_grid.several_blocks(record, _plain_keywords) is None

    def test_the_grid_edits_one_pair_and_scores_it_by_the_rule(self, records, monkeypatch):
        record, first, second = _two_pairs(records, monkeypatch)
        attacker, blocker = combat_grid.several_blocks(record, _plain_keywords)
        other = second if (attacker, blocker) == first else first
        rows, rule = combat_grid.grid_records(record, attacker, blocker, "blocker", "none")
        assert np.array_equal(rule, np.array(
            [[float(p >= t) for t in range(1, 9)] for p in range(1, 9)]))
        cell = rows[2][4]
        assert cell.state.entity(attacker).power == 3
        assert cell.state.entity(blocker).toughness == 5
        for entity in other:
            assert cell.state.entity(entity) == record.state.entity(entity)

    def test_the_freeze_freezes_the_several_block_grids_beside_the_rest(self):
        names = [grid.name for grid in selection.grid_sources(None, _plain_keywords)]
        several = [n for n in names if n.startswith("combat-several/")]
        assert len(several) == 2 * len(combat_grid.VARIANTS)
        assert len([n for n in names if n.startswith("combat/")]) == len(several)
        assert names[0] == "damage" and names[-1] == "evasion"
        assert len(set(names)) == len(names)

    def test_a_several_block_grid_qualifies_with_the_rule_s_outcomes(
        self, records, monkeypatch,
    ):
        record, first, second = _two_pairs(records, monkeypatch)
        grids = {g.name: g for g in selection.grid_sources(None, _plain_keywords)}
        found = grids["combat-several/none/blocker"].qualify(record)
        assert found.entities == min(first, second)
        assert len(found.outcomes) == 64
