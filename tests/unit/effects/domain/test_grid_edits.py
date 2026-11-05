"""The edit grids in the package (T159, T160): the control score's grid editor
edits a frozen record into exactly the cells, in exactly the order, the
freeze's grid sources froze them, reading the entity each grid reads; the
damage grid's amount is edited in the encoded text under a key of its own.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effects.domain.grid_edits import (
    ATTACKER_KEYWORDS,
    BLOCKER_TRAITS,
    GRID_PREFIX,
    VALUES,
    VARIANTS,
    CellRead,
    GridEdits,
    damage_cells,
    grid_sources,
)
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import ResolutionPayload
from effects.domain.state_snapshot import GlobalState, PlayerState, PowerToughness, Refs
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

BOLT = ("NumDmg$ 3 | SP$ DealDamage | SpellDescription$ CARDNAME deals 3 damage to "
        "target creature. | ValidTgts$ Creature")
PROSE = "Lightning Bolt deals 3 damage to target creature."
KEY = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)


def _creature(entity_id: str, *, colors=("R",), power=2, toughness=3):
    return make_entity(entity_id, entity_id, controller="P1", types=("creature",),
                       colors=colors, pt=PowerToughness(base=(power, toughness)))


def _record(*entities, **overrides):
    state = make_snapshot(
        global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0", stack_size=1),
        players=(PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0),
                 PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0)),
        entities=(make_entity("S1", "spell", zone="stack"), *entities),
        refs=Refs(targets=("E1",), source="S1"),
    )
    return make_record(**{"record_id": "r.0-0.1", "run_id": "r", "game_id": "r.0-0.1",
                          "state": state, "payload": ResolutionPayload(events=()),
                          "ability": (KEY,), **overrides})


class _Texts:
    """A batcher's two reads, on the script or the prose surface."""

    def __init__(self, text: str, script: str = BOLT, spans=()) -> None:
        self.text = text
        self.line = SimpleNamespace(script_text=script, role_spans=spans)

    def text_of(self, key):
        return self.text

    def line_of(self, key):
        return self.line


class TestTheDamageGrid:
    def test_its_cells_are_amount_by_toughness_read_at_the_target_for_its_death(self):
        record = _record(_creature("E1"))
        cells = damage_cells(record, "E1", _Texts(BOLT))
        assert len(cells) == 64
        assert {(cell.entity, cell.readout) for cell in cells} == {("E1", "dies")}
        toughness = [cell.record.state.entity("E1").pt.total[1] for cell in cells]
        assert toughness == [t for _ in VALUES for t in VALUES]

    def test_each_amount_reads_its_own_key_holding_the_edited_script(self):
        record = _record(_creature("E1"))
        cells = damage_cells(record, "E1", _Texts(BOLT))
        for index, amount in enumerate(VALUES):
            row = cells[index * 8:(index + 1) * 8]
            (override,) = {cell.texts for cell in row}
            (edit,) = override
            assert edit.like == KEY and edit.key != KEY
            assert f"NumDmg$ {amount} " in edit.text and f"deals {amount} damage" in edit.text
            assert {cell.record.ability[0] for cell in row} == {edit.key}
        assert len({cell.texts[0].key for cell in cells}) == len(VALUES)

    def test_on_the_prose_surface_the_prose_amount_is_edited(self):
        cells = damage_cells(_record(_creature("E1")), "E1", _Texts(PROSE))
        assert "deals 5 damage" in cells[4 * 8].texts[0].text

    def test_a_text_that_does_not_state_the_amount_reads_the_real_text(self):
        cells = damage_cells(_record(_creature("E1")), "E1", _Texts("Deal some damage."))
        assert all(cell.texts == () and cell.record.ability == (KEY,) for cell in cells)


class TestTheEditor:
    def test_every_grid_source_has_cells_the_editor_fills_in_order(self):
        """The freeze's cells and the editor's: the same count, so the frozen
        outcomes and the read cells line up (``zip(..., strict=True)``)."""
        sources = {grid.name: grid for grid in grid_sources(None, lambda entity: frozenset())}
        assert len(sources) == 1 + 2 * 2 * len(VARIANTS) + 1
        assert len(sources["evasion"].cells) == len(ATTACKER_KEYWORDS) * len(BLOCKER_TRAITS)
        for name, grid in sources.items():
            if name.startswith(tuple(GRID_PREFIX.values())):
                assert len(grid.cells) == 64

    @pytest.mark.parametrize("which", ["blocker", "attacker"])
    def test_a_combat_grid_reads_the_struck_creatures_death(self, which):
        from effects.domain.records import CombatPayload, RecordKind

        record = _record(_creature("A"), _creature("B"), kind=RecordKind.COMBAT,
                         payload=CombatPayload(blocks={"A": ("B",)}), ability=None)
        cells = GridEdits(_Texts(BOLT)).cells(f"combat/none/{which}", record, ("A", "B"))
        assert len(cells) == 64
        assert {(cell.entity, cell.readout) for cell in cells} == {
            ("B" if which == "blocker" else "A", "dies")}
        striker = "A" if which == "blocker" else "B"
        powers = [cell.record.state.entity(striker).pt.total[0] for cell in cells]
        assert powers == [p for p in VALUES for _ in VALUES]

    def test_the_evasion_grid_reads_the_blockers_legality_in_every_cell(self):
        record = _record(_creature("A", colors=("R",)), _creature("B", colors=("G",)))
        cells = GridEdits(_Texts(BOLT)).cells("evasion", record, ("A", "B"))
        assert len(cells) == len(ATTACKER_KEYWORDS) * len(BLOCKER_TRAITS)
        assert all(isinstance(cell, CellRead) for cell in cells)
        assert {(cell.entity, cell.readout) for cell in cells} == {("B", "may_block")}
        flying = cells[ATTACKER_KEYWORDS.index("flying") * len(BLOCKER_TRAITS)]
        assert "flying" in flying.record.state.entity("A").granted_temporary.keywords

    def test_an_unknown_grid_is_refused(self):
        with pytest.raises(KeyError):
            GridEdits(_Texts(BOLT)).cells("nowhere/none/blocker", _record(), ("A", "B"))
