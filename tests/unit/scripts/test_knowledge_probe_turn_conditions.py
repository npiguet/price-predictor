"""The gen-3 turn and step condition probes (T161) on a tiny random gen-3 model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from effect_knowledge_probes import (
    conditions,
)
from effect_knowledge_probes.board_edits import BoardEdit

from tests.unit.effects.builders import (
    GEN3_SIDECAR_FOLDERS,
)
from tests.unit.scripts.knowledge_probe_support import probabilities


def _assert_restriction_item(item, record, failed) -> None:
    assert item.subset == "restriction" and item.entity is None
    if item.edit == conditions.AS_COLLECTED:
        assert item.rule == (item.readout.removeprefix("failed_") in failed)
    if item.edit == conditions.WHOSE_TURN:
        assert item.record.state.global_.active != record.state.global_.active


def _assert_handing_over_flips_whose_turn(items) -> None:
    """Handing the turn over flips a whose-turn restriction's answer."""
    by = {(i.edit, i.readout): i.rule for i in items}
    if ("as collected", "failed_whose_turn") in by:
        assert by[("whose turn", "failed_whose_turn")] != by[
            ("as collected", "failed_whose_turn")]


def _is_player_turn_static(line) -> bool:
    return "Condition$ PlayerTurn" in (line.script_text or "") and line.line_kind == "static"


def _player_turn_static_key():
    return next(line.provenance[0] for path in GEN3_SIDECAR_FOLDERS
                for line in _sidecar_lines(path) if _is_player_turn_static(line))


class TestConditions:
    def test_the_conditions_hold_as_forge_reads_them(self):
        facts = conditions.TurnFacts(own_turn=True, phase="main1", stack_size=0)
        theirs = replace(facts, own_turn=False)
        assert conditions.static_holds({"Condition": "PlayerTurn"}, facts) is True
        assert conditions.static_holds({"Condition": "PlayerTurn"}, theirs) is False
        assert conditions.static_holds({"Condition": "NotPlayerTurn"}, theirs) is True
        assert conditions.static_holds({"Phases": "Main1,Main2"}, facts) is True
        assert conditions.static_holds({"Phases": "Upkeep"}, facts) is False
        assert conditions.static_holds({"Condition": "ExtraTurn"}, facts) is None
        assert conditions.static_holds({"Mode": "Continuous"}, facts) is None
        assert conditions.clause_holds({"ConditionPlayerTurn": "True"}, theirs) is False
        assert conditions.clause_holds({"ConditionOpponentTurn": "True"}, theirs) is True
        assert conditions.clause_holds({"ConditionPhases": "Main1"}, facts) is True
        assert conditions.clause_holds(
            {"ConditionPlayerTurn": "True", "ConditionFirstCombat": "True"}, facts) is None

    def test_a_restriction_is_read_as_collected_and_on_both_edits(self, records, sidecars):
        found = [items for r in records if (items := conditions.restriction_items(r, sidecars))]
        assert found
        for items in found:
            record = items[0].record
            failed = conditions._failed(record.payload.candidates[0])
            for item in items:
                _assert_restriction_item(item, record, failed)
            _assert_handing_over_flips_whose_turn(items)

    def test_a_casting_speed_item_follows_sorcery_timing(self, records, sidecars, keywords_of):
        found = [items for r in records
                 if (items := conditions.casting_speed_items(r, sidecars, keywords_of))]
        assert found
        for items in found:
            for item in items:
                state = item.record.state.global_
                own = state.active == item.record.actor_player
                timely = own and state.phase in ("main1", "main2") and state.stack_size == 0
                assert item.rule is (not timely)
            phases = {i.edit: i.record.state.global_.phase for i in items}
            assert (phases["phase"] in ("main1", "main2")) != (
                phases["as collected"] in ("main1", "main2"))

    def test_split_second_is_given_and_taken_away(self, records):
        found = [items for r in records if (items := conditions.split_second_items(r))]
        assert found
        edits = {item.edit: item.rule for items in found for item in items}
        assert edits[conditions.SPLIT_SECOND_ADDED] is True
        assert edits[conditions.SPLIT_SECOND_REMOVED] is False
        removed = next(item for items in found for item in items
                       if item.edit == conditions.SPLIT_SECOND_REMOVED)
        assert not any(conditions.has_split_second(e) for e in conditions._stack(removed.record))

    def test_an_edit_resyncs_a_turn_conditioned_static(self, records, sidecars):
        """Flipping whose turn moves the ``active`` flag the snapshot states."""
        from effects.domain.state_snapshot import CardLine

        record = next(r for r in records if r.state.entities and len(r.state.players) == 2)
        key = _player_turn_static_key()
        host = record.state.entities[0]
        static = CardLine(keys=(key,), kind="static", granted=False,
                          active=host.controller == record.state.global_.active)
        record = BoardEdit(record).edit_entity(host.id, lambda e: replace(
            e, lines=(*(e.lines or ()), static))).record
        handed = conditions.resync_statics(conditions.turn_handed_over(record), sidecars)
        line = next(x for x in handed.state.entity(host.id).lines if key in x.keys)
        assert line.active is (host.controller == handed.state.global_.active)
        assert line.active is not static.active

    def test_every_item_reads_a_probability_off_a_gen3_model(
        self, records, sidecars, keywords_of, probe_model,
    ):
        items = [item for r in records
                 for item in conditions.record_items(r, sidecars, keywords_of)]
        predictions = conditions.predict_items(probe_model, items)
        probabilities(predictions)
        scores = conditions.score(items, predictions)
        assert set(scores) == set(conditions.SUBSETS)
        assert scores["casting-speed"]["items"] > 0
        assert np.isfinite(scores["casting-speed"]["auc"])
        conditions.report(scores)


def _sidecar_lines(folder):
    from effects.infrastructure.sidecar_io import read_sidecar

    for path in sorted(folder.rglob("*.provenance.json")):
        yield from read_sidecar(path).lines
