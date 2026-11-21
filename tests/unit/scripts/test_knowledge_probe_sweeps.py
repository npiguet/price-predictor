"""The knowledge probes' sweep editor and affordability edit."""

from __future__ import annotations

from dataclasses import replace

import pytest
from effect_knowledge_probes import (
    sweeps,
)
from effect_knowledge_probes.board_edits import BoardEdit

from tests.unit.effects.builders import player_of
from tests.unit.scripts.knowledge_probe_support import (
    decision_record,
    differences,
    select_sweep_records,
)

# ── the sweep editor ────────────────────────────────────────────────────


class TestSweepEditor:
    def test_toughness_edits_one_entity_only(self, records):
        record = next(r for r in records if any(
            e.pt is not None and e.zone == "battlefield" for e in r.state.entities))
        target = next(e for e in record.state.entities if e.pt is not None)
        edited = BoardEdit(record).set_toughness(target.id, 7).record
        assert edited.state.entity(target.id).pt.total[1] == 7
        assert differences(record, edited) == {"entities"}
        for before, after in zip(record.state.entities, edited.state.entities):
            if before.id != target.id:
                assert before == after

    def test_board_size_sets_the_opposing_creature_count(self, records):
        record = next(r for r in records if sweeps.opposing_creatures(r))
        for count in (0, 1, 5):
            edited = sweeps.set_opposing_creatures(record, count)
            assert len(sweeps.opposing_creatures(edited)) == count
            ids = [e.id for e in edited.state.entities]
            assert len(ids) == len(set(ids))
            assert differences(record, edited) <= {"entities"}

    def test_an_amount_edit_changes_only_the_amount(self):
        text = "NumDmg$ 3 | SP$ DealDamage | ValidTgts$ Creature [SEG] SV1: DB$ Draw"
        assert sweeps.set_amount(text, "NumDmg", 7) == text.replace("NumDmg$ 3", "NumDmg$ 7")
        text = "SP$ Draw | NumCards$ 2 | ValidTgts$ Player"
        assert sweeps.set_amount(text, "NumCards", 5) == text.replace("$ 2", "$ 5")

    def test_cost_pips_read_generic_and_coloured_mana(self):
        assert sweeps.cost_pips("{2}{G}") == ["1", "1", "G"]

    def test_sweep_records_are_chosen_from_real_records(self, records, sidecars):
        chosen = select_sweep_records(records, sidecars)
        assert set(chosen) == {"toughness", "affordability", "board_size", "damage",
                               "activated_damage"}
        assert chosen["affordability"]
        ids = {r.record_id for r in records}
        for rows in chosen.values():
            assert all(row["record_id"] in ids for row in rows)


def _lands(record, player: str) -> list:
    return [e for e in record.state.entities if e.zone == "battlefield"
            and e.controller == player and "land" in e.types]


def _other_players(record, actor: str) -> list:
    return [p for p in record.state.players if p.id != actor]


class TestAffordabilityEdit:
    """The sweep's mana is the player's untapped lands, and the count beside them."""

    def test_the_untapped_lands_make_exactly_the_swept_mana(self, records):
        # The actor's untapped Plains is the board the collector wrote.
        record = decision_record(records, "{U}")
        actor = record.actor_player
        assert [(e.name, e.tapped) for e in _lands(record, actor)] == [("Plains", False)]
        pips = sweeps.candidate_cost(record)
        for amount in range(len(pips) + 3):
            edited = sweeps.set_untapped_lands(record, actor, pips, amount)
            untapped = [e for e in _lands(edited, actor) if not e.tapped]
            assert len(untapped) == amount
            self._assert_untapped_production(edited, actor, untapped)
            self._assert_everything_else_kept(record, edited, actor)

    @staticmethod
    def _assert_untapped_production(edited, actor, untapped) -> None:
        colours = [sweeps.basic_land_colour(e) for e in untapped]
        assert None not in colours
        player = player_of(edited.state, actor)
        assert player.untapped_production == {
            c: colours.count(c) for c in ("W", "U", "B", "R", "G")}
        assert player.floating_mana == {}

    @staticmethod
    def _assert_everything_else_kept(record, edited, actor) -> None:
        ids = [e.id for e in edited.state.entities]
        assert len(ids) == len(set(ids))
        assert _other_players(record, actor) == _other_players(edited, actor)
        kept = {e.id: e for e in edited.state.entities}
        for before in record.state.entities:
            if before not in _lands(record, actor):
                assert kept[before.id] == before

    def test_coloured_pips_are_fed_first_and_the_player_s_basics_reused(self, records):
        record = decision_record(records, "{U}")
        actor = record.actor_player
        pips = sweeps.candidate_cost(record)

        def board(amount):
            edited = sweeps.set_untapped_lands(record, actor, pips, amount)
            production = player_of(edited.state, actor).untapped_production
            plains = next(e for e in _lands(edited, actor) if e.name == "Plains")
            return {c: n for c, n in production.items() if n}, plains.tapped

        assert board(0) == ({}, True)
        assert board(1) == ({"U": 1}, True)
        assert board(2) == ({"U": 1, "W": 1}, False)
        assert board(4) == ({"U": 3, "W": 1}, False)

    def test_every_other_land_of_the_player_is_tapped(self, records):
        record = decision_record(records, "{1}{U}")
        actor = record.actor_player
        lands = _lands(record, actor)
        assert any(e.name == "Island" and not e.tapped for e in lands)
        plains = next(e for e in lands if e.name == "Plains" and not e.tapped)
        edited = sweeps.set_untapped_lands(record, actor, sweeps.candidate_cost(record), 1)
        assert edited.state.entity(plains.id).tapped
        assert [e.name for e in _lands(edited, actor) if not e.tapped] == ["Island"]

    def test_a_fresh_land_is_the_basic_the_collector_writes(self, records):
        fresh = sweeps.basic_land("B", "P9~land0", "P9")
        # A battlefield Swamp first in its board's order and untouched this
        # turn, as a fresh one is.
        swamp = next(e for r in records for e in r.state.entities
                     if e.name == "Swamp" and e.on_battlefield and e.role is None
                     and e.order_rank == fresh.order_rank and e.history == fresh.history)
        assert fresh == replace(swamp, id="P9~land0", controller="P9", owner="P9",
                                zone="battlefield", tapped=False, sick=False)

    @pytest.mark.parametrize(("cost", "payable"), [
        ("{2}{U}", True), ("{1}{G/U}", True), ("{W}{W}", True),
        ("{3}", False), ("{0}", False), ("{X}{R}", False), ("{C}{1}", False),
        ("{S}{G}", False), ("{2/W}", False), ("{U/P}", False),
    ])
    def test_only_a_cost_basic_lands_pay_pip_for_pip_is_swept(self, cost, payable):
        assert sweeps.land_payable(cost) is payable

    def test_an_untapped_mana_creature_keeps_a_record_out(self, records, sidecars):
        record = decision_record(records, "{U}")
        assert sweeps.affordability_ready(record, sidecars)
        elves = next(e for r in records for e in r.state.entities
                     if e.name == "Llanowar Elves")

        def with_elves(tapped):
            dork = replace(elves, id="E-dork", zone="battlefield",
                           controller=record.actor_player, tapped=tapped)
            return replace(record, state=replace(
                record.state, entities=(*record.state.entities, dork)))

        assert not sweeps.affordability_ready(with_elves(False), sidecars)
        assert sweeps.affordability_ready(with_elves(True), sidecars)

    def test_a_candidate_on_the_player_s_own_land_keeps_a_record_out(self, records,
                                                                      sidecars):
        record = decision_record(records, "{U}")
        swamp = _lands(record, record.actor_player)[0]
        candidate = replace(record.payload.candidates[0], ability=(swamp.printed[0],))
        on_land = replace(record, payload=replace(record.payload, candidates=(candidate,)))
        assert not sweeps.affordability_ready(on_land, sidecars)
