"""The gen-3 anthem probe (T162)."""

from __future__ import annotations

from types import SimpleNamespace

from effect_knowledge_probes import (
    anthem,
)
from effect_knowledge_probes.board_edits import BoardEdit

from effects.domain.records import Moment, RecordKind
from tests.unit.scripts.knowledge_probe_support import probabilities


def _combat_or_resolution(record) -> bool:
    if record.fork:
        return False
    if record.kind is RecordKind.COMBAT:
        return True
    return record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION


def _kept_alive_board(record, found, sidecars):
    """``record`` with the first creature ``found`` reaches given the damage the
    anthem alone keeps from being lethal, or None when no such damage exists."""
    creature = record.state.entity(found.affected[0])
    damage = creature.toughness - found.boost[1]
    if damage <= 0:
        return None
    edited = BoardEdit(record).set_marked_damage(creature.id, damage).record
    if anthem.chosen(edited, sidecars) is None:
        return None
    return edited, found, creature.id


def _anthem_board(records, sidecars):
    """A fixture board with an anthem, one creature it reaches given damage
    the anthem alone keeps from being lethal."""
    for record in filter(_combat_or_resolution, records):
        for found in anthem.anthems(record, sidecars):
            board = _kept_alive_board(record, found, sidecars)
            if board is not None:
                return board
    raise AssertionError("no fixture board holds an anthem")


class TestAnthem:
    def test_destroying_the_anthem_kills_what_it_kept_alive(self, records, sidecars):
        record, found, creature = _anthem_board(records, sidecars)
        assert anthem.dies_without(record, found, creature)
        after = anthem.destroyed(record, found)
        assert after.state.entity(found.entity) is None
        before, now = record.state.entity(creature), after.state.entity(creature)
        assert now.toughness == before.toughness - found.boost[1]
        assert now.damage == before.damage
        for link in after.state.links or ():
            assert found.entity not in (getattr(link.from_, "entity", link.from_),
                                        getattr(link.to, "entity", link.to))
        assert not any(line.granted and found.key in line.keys for line in now.lines or ())

    def test_each_creature_is_read_as_collected_and_destroyed(
        self, records, sidecars, probe_model,
    ):
        record, found, creature = _anthem_board(records, sidecars)
        rows = anthem.items(*anthem.chosen(record, sidecars))
        assert {row[0] for row in rows} == {anthem.AS_COLLECTED, anthem.DESTROYED}
        assert all(not row[3] for row in rows if row[0] == anthem.AS_COLLECTED)
        assert any(row[3] for row in rows if row[0] == anthem.DESTROYED and row[2] == creature)
        predictions = anthem.predict_items(probe_model, rows)
        probabilities(predictions)
        result = anthem.score(rows, predictions)
        assert result["rule_dies"] >= 1 and result["creatures"] == len(found.affected)
        anthem.report(result, 1)

    def test_a_creature_that_died_disqualifies_the_board(self, records, sidecars, monkeypatch):
        import effects.domain.effect_targets as effect_targets
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, found, creature = _anthem_board(records, sidecars)
        died = SimpleNamespace(fields={"zone_outcome": ZONE_OUTCOMES.index("died")}, unknown=set())
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: {creature: died})
        assert anthem.chosen(record, sidecars) is None
