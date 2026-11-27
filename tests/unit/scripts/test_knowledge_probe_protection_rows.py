"""The gen-3 protection rows over every quality."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import pytest
from effect_knowledge_probes import (
    common,
    evasion_grid,
    selection,
)
from effect_knowledge_probes.board_edits import BoardEdit

from effects.domain.records import (
    PlayabilityBlockersPayload,
)
from tests.unit.scripts.knowledge_probe_support import probabilities


def _gen3_block_pair(records):
    """A fixture blockers record made to qualify: a red goblin attacker, a green
    elf blocker of mana value 2, neither with keywords."""
    record = next(r for r in records if isinstance(r.payload, PlayabilityBlockersPayload)
                  and r.payload.legal_blockers)
    attacker, blocker = record.payload.anchor_attacker, record.payload.legal_blockers[0]

    def bare(entity, colours, subtypes):
        return replace(entity, colors=colours, types=("creature",), subtypes=subtypes,
                       mana_value=2, lines=(), granted_temporary=replace(
                           entity.granted_temporary, keywords=()))

    board = (BoardEdit(record)
             .edit_entity(attacker, lambda e: bare(e, ("R",), ("goblin",)))
             .edit_entity(blocker, lambda e: bare(e, ("G",), ("elf",))))
    return board.record, attacker, blocker


class _Case(NamedTuple):
    row: str
    fragment: str | None
    column: str | None
    allowed: bool


def _case_id(case: _Case) -> str:
    return "-".join(str(part) for part in case)


_CASES = [
    _Case("protection-blocker", "Protection from green", None, False),
    _Case("protection-other", None, None, True),
    _Case("protection-type-blocker", "Protection:Elf", None, False),
    _Case("protection-type-blocker", "Protection:Elf", "black", False),
    _Case("protection-type-other", "Protection:Kraken", None, True),
    _Case("protection-creatures", "Protection:Creature", "reach", False),
    _Case("protection-artifacts", "Protection:Artifact", None, True),
    _Case("protection-artifacts", "Protection:Artifact", "artifact", False),
    _Case("protection-monocolored", "Protection:Card.MonoColor:monocolored", None, False),
    _Case("protection-multicolored", "Protection:Card.MultiColor:multicolored", None, True),
    _Case("protection-mana-value-3-or-less", "Protection:Card.cmcLE3:mana value 3 or less",
     None, False),
    _Case("protection-everything", "Protection from everything", "flying", False),
]


class TestProtectionRows:
    @pytest.mark.parametrize("case", _CASES, ids=_case_id)
    def test_each_quality_is_granted_as_forge_writes_it_and_ruled(self, records, case):
        record, attacker, blocker = _gen3_block_pair(records)
        cell, rule = evasion_grid.edited(record, (attacker, blocker), case.row, case.column)
        lines = cell.state.entity(attacker).lines
        granted = [line.fragment for line in lines if line.granted and line.kind == "keyword"]
        assert len(granted) == 1
        if case.fragment is not None:
            assert granted == [case.fragment]
        assert rule is case.allowed

    def test_every_row_is_a_grid_row_and_a_frozen_cell(self, keywords_of):
        assert set(evasion_grid.PROTECTION_ROWS) <= set(evasion_grid.ATTACKER_KEYWORDS)
        grid = selection.EvasionGrid(keywords_of)
        assert len(grid.cells) == (len(evasion_grid.ATTACKER_KEYWORDS)
                                   * len(evasion_grid.BLOCKER_TRAITS))
        assert "protection-everything|none" in grid.cells

    def test_a_blocker_without_a_creature_type_reads_another_type(self, records):
        record, attacker, blocker = _gen3_block_pair(records)
        record = BoardEdit(record).edit_entity(
            blocker, lambda e: replace(e, subtypes=())).record
        cell, rule = evasion_grid.edited(record, (attacker, blocker),
                                         "protection-type-blocker", None)
        assert rule is True

    def test_the_grid_reads_a_gen3_model(self, records, probe_model):
        from effect_knowledge_probes import readout

        record, attacker, blocker = _gen3_block_pair(records)
        cells = [evasion_grid.edited(record, (attacker, blocker), row, None)[0]
                 for row in evasion_grid.PROTECTION_ROWS]
        probabilities(common.predict(probe_model, cells, readout.at_card("may_block", blocker)))
