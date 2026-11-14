"""The edit grids in the package (T159, T160): the control score's grid editor
edits a frozen record into exactly the cells, in exactly the order, the
freeze's grid sources froze them, reading the entity each grid reads; the
damage grid's amount is edited in the encoded text under a key of its own.
"""

from __future__ import annotations

from dataclasses import replace
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
from effects.domain.state_snapshot import (
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
    Refs,
)
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


class _Lines:
    """The damage grid's acting-line read (``SidecarCache.line_or_none``)."""

    def line_or_none(self, key):
        return SimpleNamespace(script_text=BOLT, role_spans=())


def _no_keywords(entity) -> frozenset[str]:
    return frozenset()


class TestTheDamageGridsOutcomes:
    def test_each_cell_s_outcome_is_the_rule_on_the_board_the_cell_reads(self):
        """A target already marked with damage: every cell clears it, so the
        frozen outcome is the rule on the cleared board, never on the real one."""
        from effects.domain.event_schema import Event, EventType
        from effects.domain.grid_edits import DamageGrid
        from effects.domain.rule_outcomes import damage_lethal

        target = make_entity("E1", "E1", controller="P1", types=("creature",),
                             pt=PowerToughness(base=(2, 5)), damage=2)
        events = (Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 3}),
                  Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                        params={"from_zone": "battlefield", "to_zone": "graveyard"}))
        record = _record(target, payload=ResolutionPayload(events=events))
        qualified = DamageGrid(_Lines(), _no_keywords).qualify(record)
        assert qualified is not None and qualified.entities == ("E1",)
        cells = damage_cells(record, "E1", _Texts(BOLT))
        boards = [cell.record.state.entity("E1") for cell in cells]
        amounts = [amount for amount in VALUES for _ in VALUES]
        assert list(qualified.outcomes) == [
            float(damage_lethal(board.pt.total[1], board.damage, amount))
            for board, amount in zip(boards, amounts, strict=True)]
        # Amount 2 on toughness 3: the real board's 2 marked damage would kill.
        assert qualified.outcomes[1 * 8 + 2] == 0.0


def _blockers_record(attacker, blocker):
    from effects.domain.records import (
        PlayabilityBlockersPayload,
        PlayabilitySubkind,
        RecordKind,
    )

    return make_record(
        kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
        state=make_snapshot(entities=(attacker, blocker)), what_if=False,
        payload=PlayabilityBlockersPayload(anchor_attacker=attacker.id,
                                           legal_blockers=(blocker.id,)))


class TestTheEvasionGridsPair:
    def test_a_pair_holding_all_five_colours_is_no_evasion_pair(self):
        """``protection-other`` needs a colour neither creature has: an Esper
        attacker against a red-green blocker leaves none, so the pair is not
        one the grid can fill, rather than a crash that kills the freeze."""
        from effects.domain.grid_edits import EvasionGrid

        record = _blockers_record(_creature("A", colors=("W", "U", "B")),
                                  _creature("B", colors=("R", "G")))
        assert EvasionGrid(_no_keywords).qualify(record) is None

    def test_a_pair_holding_every_spare_creature_type_is_no_evasion_pair(self):
        from effects.domain.grid_edits import OTHER_CREATURE_TYPES, EvasionGrid

        attacker = replace(_creature("A", colors=("W",)), subtypes=OTHER_CREATURE_TYPES)
        record = _blockers_record(attacker, _creature("B", colors=("G",)))
        assert EvasionGrid(_no_keywords).qualify(record) is None

    def test_a_changeling_blocker_is_no_evasion_pair(self):
        """702.73a: a changeling has every creature type, so protection from a
        creature type covers it (702.16f) though its subtypes list none."""
        from effects.domain.grid_edits import EvasionGrid

        blocker = replace(_creature("B", colors=("G",)),
                          granted_temporary=GrantedTemporary(keywords=("changeling",)))
        record = _blockers_record(_creature("A", colors=("W",)), blocker)
        assert EvasionGrid(_no_keywords).qualify(record) is None

    def test_a_phased_out_blocker_or_a_last_known_attacker_is_no_evasion_pair(self):
        from effects.domain.grid_edits import EvasionGrid

        attacker, blocker = _creature("A", colors=("W",)), _creature("B", colors=("G",))
        assert EvasionGrid(_no_keywords).qualify(_blockers_record(attacker, blocker))
        for record in (
                _blockers_record(attacker, replace(blocker, designations=("phased_out",))),
                _blockers_record(replace(attacker, zone=None), blocker)):
            assert EvasionGrid(_no_keywords).qualify(record) is None

    def test_a_pair_with_a_spare_colour_fills_every_cell(self):
        from effects.domain.grid_edits import EvasionGrid

        record = _blockers_record(_creature("A", colors=("W", "U", "B")),
                                  _creature("B", colors=("R",)))
        qualified = EvasionGrid(_no_keywords).qualify(record)
        assert qualified is not None and qualified.entities == ("A", "B")
        assert len(qualified.outcomes) == len(ATTACKER_KEYWORDS) * len(BLOCKER_TRAITS)


class TestTheCombatGridsPair:
    def _combat(self, *extra_events):
        from effects.domain.event_schema import Event, EventType
        from effects.domain.records import CombatPayload, RecordKind
        from effects.domain.state_snapshot import CombatStatus

        attacker = replace(_creature("A", power=2, toughness=3),
                           combat=CombatStatus(attacking="P1", blocked_by=("B",)))
        blocker = replace(_creature("B", power=2, toughness=3),
                          combat=CombatStatus(blocking=("A",)))
        events = (Event(type=EventType.DAMAGE_DEALT, subjects=("B",), params={"amount": 2}),
                  Event(type=EventType.DAMAGE_DEALT, subjects=("A",), params={"amount": 2}),
                  *extra_events)
        state = make_snapshot(
            global_=GlobalState(turn=3, phase="combat_damage", active="P0", priority="P0",
                                stack_size=0, combat_substep="regular"),
            entities=(attacker, blocker))
        return make_record(kind=RecordKind.COMBAT, state=state,
                           payload=CombatPayload(attackers=("A",), blocks={"A": ("B",)},
                                                 events=events))

    def test_a_block_both_creatures_survive_qualifies(self):
        from effects.domain.grid_edits import grid_block

        assert grid_block(self._combat(), _no_keywords, "one") == ("A", "B")

    def test_a_creature_moved_where_no_zone_outcome_names_is_no_grid_pair(self):
        """Its outcome is unknown, never "stayed", so the rule cannot be checked."""
        from effects.domain.event_schema import Event, EventType
        from effects.domain.grid_edits import grid_block

        moved = Event(type=EventType.ZONE_CHANGE, subjects=("B",),
                      params={"from_zone": "battlefield", "to_zone": "stack"})
        assert grid_block(self._combat(moved), _no_keywords, "one") is None
