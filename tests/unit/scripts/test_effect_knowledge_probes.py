"""The knowledge-probe suite (FR-089): share, labels, sweeps, folds, digest, compare.

``scripts/`` is not part of the installed package, so ``scripts/`` is put on
the path and the suite imports as the package its entry scripts make it, so a
test reads the very modules a run would.

Fixtures are real: the gen-1 records and sidecars cut from the corpus by
``scripts/make_fixture_records.py``, and the real gen-2 chained sidecars.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[3]
_SUITE = _ROOT / "scripts" / "effect_knowledge_probes"
_FIXTURES = _ROOT / "tests" / "fixtures" / "effects"

if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from effect_knowledge_probes import (  # noqa: E402
    board,
    colour_pips,
    combat_grid,
    common,
    compare,
    evasion_grid,
    labels,
    ladder,
    probe_set,
    run,
    sweeps,
    target_restrictions,
    zone_effects,
)
from effect_knowledge_probes.board_edits import BoardEdit  # noqa: E402

from tests.unit.effects.builders import player_of  # noqa: E402


@pytest.fixture(scope="module")
def records():
    from effects.infrastructure.shard_io import read_shard

    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def sidecars():
    return common.sidecar_cache([
        _FIXTURES / "gen1-sidecars" / "cardsfolder",
        _FIXTURES / "gen1-sidecars" / "tokenscripts",
    ])


def _fixture_lines(tree: str):
    from effects.infrastructure.sidecar_io import read_sidecar

    for path in sorted((_FIXTURES / tree).rglob("*.provenance.json")):
        sidecar = read_sidecar(path)
        for line in sidecar.lines:
            if line.script_text:
                yield line


def _line(tree: str, *needles: str):
    for line in _fixture_lines(tree):
        if all(n in line.script_text for n in needles):
            return line
    raise AssertionError(f"no fixture line holds {needles}")


# ── the share and its minimum gap ───────────────────────────────────────


class TestShare:
    def test_it_is_the_vector_s_part_of_the_gap(self):
        assert ladder.share(0.5, 0.7, 0.9) == pytest.approx(0.5)

    def test_it_is_not_reported_under_the_minimum_gap(self):
        assert ladder.share(0.60, 0.70, 0.60 + ladder.MIN_GAP - 1e-6) is None

    def test_it_is_reported_at_the_minimum_gap(self):
        assert ladder.share(0.60, 0.61, 0.60 + ladder.MIN_GAP + 1e-9) is not None

    def test_a_share_above_one_is_kept(self):
        assert ladder.share(0.5, 0.95, 0.8) > 1.0

    def test_an_undefined_rung_gives_no_share(self):
        assert ladder.share(float("nan"), 0.7, 0.9) is None

    def test_the_bootstrap_interval_brackets_the_estimate(self):
        rng = np.random.default_rng(0)
        n = 400
        groups = np.array([f"t{i // 4}" for i in range(n)], dtype=object)
        y = rng.integers(0, 2, n).astype(float)
        oof0 = rng.normal(size=n)
        oof1 = y + rng.normal(scale=1.0, size=n)
        pred3 = y + rng.normal(scale=0.5, size=n)
        point = ladder.share(ladder.score("binary", y, oof0), ladder.score("binary", y, oof1),
                             ladder.score("binary", y, pred3))
        low, high = ladder.bootstrap_share("binary", y, oof0, oof1, pred3, groups,
                                           resamples=200)
        assert low <= point <= high

    def test_run_ladder_reports_each_probe_type_s_own_share(self):
        rng = np.random.default_rng(1)
        n = 300
        y = rng.integers(0, 2, n).astype(float)
        noise = rng.normal(size=(n, 3))
        data = ladder.LadderInput(
            kind="binary", y=y,
            groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            features={"0": noise, "1": np.column_stack([noise, y + rng.normal(size=n)])},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options={"hidden": 8, "epochs": 3, "device": "cpu"},
                                   bootstrap=50)
        assert set(result["rungs"]) == {"0", "1", "3"}
        assert set(result["share"]) == {"linear", "mlp"}
        assert 0.0 < result["share"]["linear"]["value"] < 1.5

    def test_no_share_is_reported_when_the_model_knows_nothing_more(self):
        rng = np.random.default_rng(2)
        n = 1000
        y = rng.integers(0, 2, n).astype(float)
        data = ladder.LadderInput(
            kind="binary", y=y, groups=np.array([f"g{i}" for i in range(n)], dtype=object),
            # The board alone reads the label as well as the model does.
            features={"0": np.column_stack([y + rng.normal(scale=0.3, size=n),
                                            rng.normal(size=n)]),
                      "1": rng.normal(size=(n, 2))},
            rung3=y + rng.normal(scale=0.3, size=n),
        )
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options={"hidden": 16, "epochs": 150, "device": "cpu"},
                                   bootstrap=0)
        assert result["share"] is None


# ── folds ───────────────────────────────────────────────────────────────


class TestFolds:
    def test_no_group_falls_on_both_sides(self):
        groups = [f"text-{i % 37}" for i in range(500)]
        folds = ladder.fold_assignment(groups)
        seen: dict[str, set[int]] = {}
        for group, fold in zip(groups, folds):
            seen.setdefault(group, set()).add(int(fold))
        assert all(len(f) == 1 for f in seen.values())
        assert set(folds.tolist()) == set(range(ladder.FOLDS))

    def test_assignment_is_a_function_of_the_names_alone(self):
        groups = ["b", "a", "c", "a", "d", "e", "f"]
        first = ladder.fold_assignment(groups)
        again = ladder.fold_assignment(list(reversed(groups)))[::-1]
        assert first.tolist() == again.tolist()

    def test_pooled_items_are_grouped_by_card(self, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        by_record = {r.record_id: r for r in records}
        for record in records:
            for item in labels.record_items(record, sidecars, join, acting):
                if item.entity is None:
                    continue
                entity = by_record[item.record_id].state.entity(item.entity)
                assert item.group == labels.card_of(entity)

    def test_the_width_control_is_one_vector_per_text(self):
        a = ladder.width_vector("SP$ Draw | NumCards$ 1", 16)
        assert np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 1", 16))
        assert not np.array_equal(a, ladder.width_vector("SP$ Draw | NumCards$ 2", 16))


# ── the probes ──────────────────────────────────────────────────────────


class TestProbes:
    def test_the_mlp_reads_a_non_linear_label_on_cpu(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(600, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(float)
        folds = ladder.fold_assignment([str(i) for i in range(600)])
        mlp = ladder.mlp_oof(X, y, folds, "binary", hidden=32, epochs=60, device="cpu")
        linear = ladder.linear_oof(X, y, folds, "binary")
        assert ladder.score("binary", y, mlp) > 0.8
        assert ladder.score("binary", y, linear) < 0.65

    def test_an_mlp_amount_stays_within_a_span_of_the_training_range(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(60, 2))
        X[-1] = 1e6
        y = rng.uniform(0, 3, size=60)
        folds = np.array([0] * 59 + [1])
        prediction = ladder.mlp_oof(X, y, folds, "amount", hidden=8, epochs=5, device="cpu")
        low, high = y[:59].min(), y[:59].max()
        span = max(high - low, 1.0)
        assert low - span - 1e-6 <= prediction[-1] <= high + span + 1e-6

    def test_the_linear_probe_reads_an_amount(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(300, 3))
        y = 2 * X[:, 0] + rng.normal(scale=0.1, size=300)
        folds = ladder.fold_assignment([str(i) for i in range(300)])
        assert ladder.score("amount", y, ladder.linear_oof(X, y, folds, "amount")) > 0.95


# ── line labels from real scripts ───────────────────────────────────────


class TestLineLabels:
    def test_an_activated_damage_ability(self):
        line = _line("gen1-sidecars", "AB$ DealDamage", "NumDmg$ 3", "Sac<1/CARDNAME")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["damage"] == 3
        assert found["cost_r"] == 1 and found["cost_generic"] == 4
        assert found["cost_total"] == 5
        assert found["damage_per_mana"] == pytest.approx(3 / 5)
        assert found["sacrifice_cost"] == 1.0 and found["tap_cost"] == 0.0
        assert found["repeatable"] == 0.0
        assert found["instant_speed"] == 1.0

    def test_a_spell_line_has_no_mana_cost_labels(self):
        line = _line("gen1-sidecars", "SP$ DealDamage", "NumDmg$ 3")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type,
                                   "instant")
        assert found["damage"] == 3
        assert "cost_total" not in found and "damage_per_mana" not in found
        assert found["instant_speed"] == 1.0
        assert found["repeatable"] == 0.0

    def test_a_variable_amount_is_left_unlabelled(self):
        line = _line("gen1-sidecars", "NumAtt$ -X")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert "power_change" not in found and "toughness_change" not in found
        assert found["until_end_of_turn"] == 1.0

    def test_a_restricted_any_colour_mana_ability(self):
        line = _line("gen1-sidecars", "Produced$ Any", "RestrictValid$")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["any_colour"] == 1.0 and found["restricted"] == 1.0
        assert found["mana_amount"] == 1.0 and found["tap_cost"] == 1.0
        assert found["produces_g"] == 0.0

    def test_a_two_colour_mana_ability(self):
        line = _line("gen1-sidecars", "Produced$ Combo U R")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["produces_u"] == found["produces_r"] == 1.0
        assert found["produces_w"] == 0.0 and found["any_colour"] == 0.0

    def test_a_sorcery_speed_activation(self):
        line = _line("gen1-sidecars", "SorcerySpeed$ True", "KW$ Flying")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["instant_speed"] == 0.0 and found["repeatable"] == 1.0

    def test_a_counter_is_not_until_end_of_turn(self):
        line = _line("gen1-sidecars", "AB$ PutCounter |")
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        assert found["as_counter"] == 1.0 and found["until_end_of_turn"] == 0.0

    def test_a_gen2_chain_sums_its_segments(self):
        chained = [
            line for line in _fixture_lines("gen2-sidecars")
            if "[SEG]" in line.script_text and "NumDmg$" in line.script_text
        ]
        if not chained:
            pytest.skip("no chained damage text in the gen-2 fixture")
        line = chained[0]
        found = labels.line_labels(line.script_text, line.line_kind, line.script_api_type)
        segments = labels.segments(line.script_text)
        stated = [s["NumDmg"] for s in segments if "NumDmg" in s]
        if all(v.lstrip("+-").isdigit() for v in stated):
            assert found["damage"] == sum(int(v) for v in stated)
        else:
            assert "damage" not in found

    def test_a_charm_root_states_no_amounts(self):
        root = "CharmNum$ 2 | Choices$ SV1,SV2,SV3,SV4 | SP$ Charm"
        found = labels.line_labels(root, "spell", "Charm")
        assert not {"damage", "power_change", "counters_placed"} & set(found)

    # The real scripts, as their sidecars in output/cardsfolder/ carry them.
    @pytest.mark.parametrize(("script", "line_kind", "api", "target", "sign"), [
        pytest.param(
            "Execute$ SV1 | Mode$ Attacks | TriggerDescription$ 10,000 Needles — Whenever "
            "this creature attacks, it gets +9999/+0 until end of turn. | ValidCard$ "
            "Card.Self [SEG] SV1: DB$ Pump | Defined$ Self | NumAtt$ +9999",
            "triggered", "Pump", "power_change", 1, id="jumbo-cactuar"),
        pytest.param(
            "IsCurse$ True | NumDef$ -9999 | SP$ Pump | SpellDescription$ Target creature "
            "gets -0/-9999 until end of turn. | ValidTgts$ Creature",
            "spell", "Pump", "toughness_change", -1, id="overkill"),
        pytest.param(
            "Execute$ SV1 | IsPresent$ Card.Self+counters_GE1000_TIME | Mode$ Always | "
            "TriggerDescription$ When there are 1,000 or more time counters on CARDNAME, "
            "sacrifice it and each opponent loses 1,000 life. | TriggerZones$ Battlefield "
            "[SEG] SV1: DB$ Sacrifice | SubAbility$ SV2 [SEG] SV2: DB$ LoseLife | "
            "Defined$ Opponent | LifeAmount$ 1000",
            "triggered", "Sacrifice", "life_amount", 1, id="the-millennium-calendar"),
        pytest.param(
            "AB$ GainLife | Cost$ SubCounter<8/LOYALTY> | LifeAmount$ 100 | "
            "Planeswalker$ True | SpellDescription$ You gain 100 life. | Ultimate$ True",
            "planeswalker", "GainLife", "life_amount", 1, id="ajani-mentor-of-heroes"),
    ])
    def test_an_amount_is_capped_where_the_model_s_targets_are(
            self, script, line_kind, api, target, sign):
        from effects.domain.effect_head_input import MAGNITUDE_CAP

        assert labels.line_labels(script, line_kind, api)[target] == sign * MAGNITUDE_CAP

    def test_damage_per_mana_is_read_off_the_capped_damage(self):
        from effects.domain.effect_head_input import MAGNITUDE_CAP

        found = labels.line_labels(
            "AB$ DealDamage | Cost$ 2 R T | NumDmg$ 100 | ValidTgts$ Any", "activated",
            "DealDamage")
        assert found["damage"] == MAGNITUDE_CAP
        assert found["damage_per_mana"] == pytest.approx(MAGNITUDE_CAP / 3)

    def test_every_amount_label_is_within_the_cap_and_no_flag_moves(self):
        from effects.domain.effect_head_input import MAGNITUDE_CAP

        for line in _fixture_lines("gen1-sidecars"):
            found = labels.line_labels(line.script_text, line.line_kind,
                                       line.script_api_type)
            for name, value in found.items():
                if labels.TARGETS_BY_NAME[name].kind == labels.AMOUNT:
                    assert abs(value) <= MAGNITUDE_CAP
                else:
                    assert value in (0.0, 1.0)


# ── record labels from real records ─────────────────────────────────────


class TestRecordLabels:
    def _items(self, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        by_id = {r.record_id: r for r in records}
        out = []
        for record in records:
            out += [(by_id[i.record_id], i)
                    for i in labels.record_items(record, sidecars, join, acting)]
        return out

    def test_every_item_names_a_known_target_and_a_finite_label(self, records, sidecars):
        items = self._items(records, sidecars)
        assert items
        for _, item in items:
            assert item.target in labels.TARGETS_BY_NAME
            assert np.isfinite(item.label)

    def test_a_blocker_label_follows_the_legal_blockers(self, records, sidecars):
        blocks = [(r, i) for r, i in self._items(records, sidecars) if i.target == "may_block"]
        assert blocks
        assert any(i.label == 0.0 for _, i in blocks)
        for record, item in blocks:
            assert item.label == float(item.entity in record.payload.legal_blockers)
            assert item.partner_entity == record.payload.anchor_attacker

    def test_a_decision_yields_its_affordable_verdict(self, records, sidecars):
        decisions = [(r, i) for r, i in self._items(records, sidecars)
                     if i.target == "affordable"]
        assert decisions
        for record, item in decisions:
            assert item.label == float(record.payload.candidates[0].affordable)

    def test_a_trigger_yields_whether_it_fired(self, records, sidecars):
        fires = [(r, i) for r, i in self._items(records, sidecars) if i.target == "fires"]
        assert {i.label for _, i in fires} == {0.0, 1.0}
        for record, item in fires:
            assert item.label == float(record.payload.fired)

    def test_affected_items_cover_the_battlefield_of_an_effect_half(self, records, sidecars):
        from effects.domain.effect_targets import derive_targets

        affected = [(r, i) for r, i in self._items(records, sidecars) if i.target == "affected"]
        assert affected
        for record, item in affected:
            entry = derive_targets(record).get(item.entity)
            assert item.label == float(bool(entry and entry.affected))

    def test_damage_taken_is_capped_where_the_model_s_targets_are(self, records, sidecars):
        from dataclasses import replace

        from effects.domain.effect_head_input import MAGNITUDE_CAP
        from effects.domain.event_schema import Event, EventType
        from effects.domain.records import Moment

        # The fixture's resolutions damage no creature, so one is given the
        # damage event a real record carries, at an amount far past the cap.
        record = next(r for r in records if r.moment is Moment.RESOLUTION and any(
            e.zone == "battlefield" and "creature" in e.types for e in r.state.entities))
        creature = next(e for e in record.state.entities
                        if e.zone == "battlefield" and "creature" in e.types)
        event = Event(EventType.DAMAGE_DEALT, subjects=(creature.id,),
                      params={"amount": 9999, "combat": False})
        huge = replace(record, payload=replace(record.payload,
                                               events=(*record.payload.events, event)))
        acting = run._acting_text_function(sidecars, "script")
        found = [i for i in labels.record_items(huge, sidecars, labels.join_triggers([huge]),
                                                acting)
                 if i.target == "damage_taken" and i.entity == creature.id]
        assert [i.label for i in found] == [MAGNITUDE_CAP]

    @staticmethod
    def _producing(records, *mana_by_color):
        """A resolution whose events produce the given ``mana_by_color`` maps.

        The map is what the collector writes: Forge's produced-mana string as a
        count per colour letter, ``{"C": 2}`` for a Sol Ring's ``2``.
        """
        from dataclasses import replace

        from effects.domain.event_schema import Event, EventType
        from effects.domain.records import ResolutionPayload

        record = next(r for r in records if isinstance(r.payload, ResolutionPayload))
        events = tuple(
            Event(EventType.MANA_PRODUCED, subjects=(record.actor_player,),
                  params={"mana_by_color": mana})
            for mana in mana_by_color
        )
        return replace(record, payload=ResolutionPayload(events=events))

    def test_observed_mana_sums_the_counts_each_event_carries(self, records):
        """A Sol Ring makes two mana in one event, a Tinder Farm one of each of
        two colours."""
        assert labels.mana_observed(self._producing(records, {"C": 2})) == 2.0
        assert labels.mana_observed(self._producing(records, {"R": 1, "W": 1})) == 2.0
        assert labels.mana_observed(
            self._producing(records, {"G": 1}, {"G": 1}, {"U": 3})) == 5.0

    def test_observed_mana_is_capped_where_the_model_s_targets_are(self, records):
        from effects.domain.effect_head_input import MAGNITUDE_CAP

        produced = self._producing(records, *({"G": 1},) * (int(MAGNITUDE_CAP) + 10))
        assert labels.mana_observed(produced) == MAGNITUDE_CAP

    def test_an_event_that_names_no_mana_says_nothing(self, records):
        """An empty map is mana the collector could not read, not zero mana."""
        assert labels.mana_observed(self._producing(records, {})) is None
        assert labels.mana_observed(self._producing(records, {}, {"B": 1})) == 1.0

    def test_the_join_finds_causes_in_the_same_game(self, records):
        result = labels.join_triggers(records)
        by_id = {r.record_id: r for r in records}
        assert 0.0 <= result.join_rate <= 1.0 or np.isnan(result.join_rate)
        for record_id, key in result.cause.items():
            game = by_id[record_id].game_id
            assert any(r.game_id == game and r.ability and r.ability[0] == key
                       for r in records if r.kind.value == "resolution")

    def test_state_dependence_is_the_spread_of_a_text_s_outcomes(self, records):
        profiles = [p for p in (labels.effect_profile(r) for r in records) if p is not None]
        assert len(profiles) >= 2
        same = labels.state_dependence({"t": [profiles[0], profiles[0]]})
        assert same["t"] == (0.0, 2)
        mixed = labels.state_dependence({"t": profiles[:2], "once": profiles[:1]})
        assert "once" not in mixed and mixed["t"][1] == 2
        assert labels.state_weight(5) == pytest.approx(0.5)

    def test_the_mined_pairs_join_a_death_trigger_to_a_sacrifice_outlet(self):
        items = common.line_items([_FIXTURES / "gen1-sidecars" / "cardsfolder"], "script")
        pairs = labels.mined_pairs(items)
        by_key = {item.key: item for item in items}
        assert pairs
        for trigger, cause in pairs:
            assert by_key[trigger].line_kind == "triggered"
        assert len({t for t, _ in pairs}) == len(pairs)


# ── the sweep editor ────────────────────────────────────────────────────


def _differences(before, after) -> set[str]:
    """Which top-level parts of a record's state differ."""
    changed = set()
    if before.state.players != after.state.players:
        changed.add("players")
    if before.state.entities != after.state.entities:
        changed.add("entities")
    if before.payload != after.payload or before.ability != after.ability:
        changed.add("record")
    return changed


class TestSweepEditor:
    def test_toughness_edits_one_entity_only(self, records):
        record = next(r for r in records if any(
            e.pt is not None and e.zone == "battlefield" for e in r.state.entities))
        target = next(e for e in record.state.entities if e.pt is not None)
        edited = BoardEdit(record).set_toughness(target.id, 7).record
        assert edited.state.entity(target.id).pt.total[1] == 7
        assert _differences(record, edited) == {"entities"}
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
            assert _differences(record, edited) <= {"entities"}

    def test_damage_edits_only_the_amount(self):
        text = "NumDmg$ 3 | SP$ DealDamage | ValidTgts$ Creature [SEG] SV1: DB$ Draw"
        assert sweeps.set_damage_amount(text, 7) == text.replace("NumDmg$ 3", "NumDmg$ 7")

    def test_cost_pips_read_generic_and_coloured_mana(self):
        assert sweeps.cost_pips("{2}{G}") == ["1", "1", "G"]

    def test_sweep_records_are_chosen_from_real_records(self, records, sidecars):
        chosen = sweeps.select_sweep_records(records, sidecars)
        assert set(chosen) == {"toughness", "affordability", "board_size", "damage"}
        assert chosen["affordability"]
        ids = {r.record_id for r in records}
        for rows in chosen.values():
            assert all(row["record_id"] in ids for row in rows)


def _decision(records, cost: str):
    """The fixture's decision record whose first candidate costs ``cost``."""
    return next(r for r in records if r.subkind is not None
                and r.subkind.value == "decision"
                and sweeps.candidate_mana(r) == cost)


def _lands(record, player: str) -> list:
    return [e for e in record.state.entities if e.zone == "battlefield"
            and e.controller == player and "land" in e.types]


class TestAffordabilityEdit:
    """The sweep's mana is the player's untapped lands, and the count beside them."""

    def test_the_untapped_lands_make_exactly_the_swept_mana(self, records):
        # The actor's untapped Swamp is the board the collector wrote.
        record = _decision(records, "{2}{U}")
        actor = record.actor_player
        assert [(e.name, e.tapped) for e in _lands(record, actor)] == [("Swamp", False)]
        pips = sweeps.candidate_cost(record)
        for amount in range(len(pips) + 3):
            edited = sweeps.set_untapped_lands(record, actor, pips, amount)
            untapped = [e for e in _lands(edited, actor) if not e.tapped]
            assert len(untapped) == amount
            colours = [sweeps.basic_land_colour(e) for e in untapped]
            assert None not in colours
            player = player_of(edited.state, actor)
            assert player.untapped_production == {
                c: colours.count(c) for c in ("W", "U", "B", "R", "G")}
            assert player.floating_mana == {}
            ids = [e.id for e in edited.state.entities]
            assert len(ids) == len(set(ids))
            for before, after in zip(record.state.players, edited.state.players):
                if before.id != actor:
                    assert before == after
            kept = {e.id: e for e in edited.state.entities}
            for before in record.state.entities:
                if before not in _lands(record, actor):
                    assert kept[before.id] == before

    def test_coloured_pips_are_fed_first_and_the_player_s_basics_reused(self, records):
        record = _decision(records, "{2}{U}")
        actor = record.actor_player
        pips = sweeps.candidate_cost(record)

        def board(amount):
            edited = sweeps.set_untapped_lands(record, actor, pips, amount)
            production = player_of(edited.state, actor).untapped_production
            swamp = next(e for e in _lands(edited, actor) if e.name == "Swamp")
            return {c: n for c, n in production.items() if n}, swamp.tapped

        assert board(0) == ({}, True)
        assert board(1) == ({"U": 1}, True)
        assert board(2) == ({"U": 1, "B": 1}, False)
        assert board(4) == ({"U": 3, "B": 1}, False)

    def test_every_other_land_of_the_player_is_tapped(self, records):
        from dataclasses import replace

        record = _decision(records, "{1}{U}")
        actor = record.actor_player
        building = next(e for e in _lands(record, actor) if e.name == "TCRI Building")
        untapped = replace(record, state=replace(record.state, entities=tuple(
            replace(e, tapped=False) if e.id == building.id else e
            for e in record.state.entities)))
        edited = sweeps.set_untapped_lands(untapped, actor, sweeps.candidate_cost(record), 1)
        assert edited.state.entity(building.id).tapped
        assert [e.name for e in _lands(edited, actor) if not e.tapped] == ["Island"]

    def test_a_fresh_land_is_the_basic_the_collector_writes(self, records):
        from dataclasses import replace

        swamp = next(e for r in records for e in r.state.entities if e.name == "Swamp")
        fresh = sweeps.basic_land("B", "P9~land0", "P9")
        assert fresh == replace(swamp, id="P9~land0", controller="P9", zone="battlefield",
                                tapped=False, sick=False)

    @pytest.mark.parametrize(("cost", "payable"), [
        ("{2}{U}", True), ("{1}{G/U}", True), ("{W}{W}", True),
        ("{3}", False), ("{0}", False), ("{X}{R}", False), ("{C}{1}", False),
        ("{S}{G}", False), ("{2/W}", False), ("{U/P}", False),
    ])
    def test_only_a_cost_basic_lands_pay_pip_for_pip_is_swept(self, cost, payable):
        assert sweeps.land_payable(cost) is payable

    def test_an_untapped_mana_creature_keeps_a_record_out(self, records, sidecars):
        from dataclasses import replace

        record = _decision(records, "{2}{U}")
        assert sweeps.affordability_ready(record, sidecars)
        nurturer = next(e for r in records for e in r.state.entities
                        if e.name == "Scaled Nurturer")

        def with_nurturer(tapped):
            dork = replace(nurturer, id="E-dork", zone="battlefield",
                           controller=record.actor_player, tapped=tapped)
            return replace(record, state=replace(
                record.state, entities=(*record.state.entities, dork)))

        assert not sweeps.affordability_ready(with_nurturer(False), sidecars)
        assert sweeps.affordability_ready(with_nurturer(True), sidecars)

    def test_a_candidate_on_the_player_s_own_land_keeps_a_record_out(self, records,
                                                                      sidecars):
        from dataclasses import replace

        record = _decision(records, "{2}{U}")
        swamp = _lands(record, record.actor_player)[0]
        candidate = replace(record.payload.candidates[0], ability=(swamp.printed[1],))
        on_land = replace(record, payload=replace(record.payload, candidates=(candidate,)))
        assert not sweeps.affordability_ready(on_land, sidecars)


# ── the probe set's digest ──────────────────────────────────────────────


def _probe_set(corpus_digest: str):
    """A small sealed set, every field filled the way a freeze fills it."""
    return probe_set.ProbeSet(
        corpus="corpus", corpus_digest=corpus_digest, surface="script",
        games={"card-disjoint": ["g1"]}, sources={"card-disjoint": ["s.jsonl.gz"]},
        items=[probe_set.FrozenLineItem({"script_file": "cardsfolder/s/shock.txt"},
                                        {"amount": 2.0}, {"spread": 0.5})],
        pairs=[{"a": {}, "b": {}, "label": 1.0, "source": "mined"}],
        record_items=[probe_set.FrozenRecordItem(
            record_id="g1.1", stratum="card-disjoint", family="magnitudes",
            target="dies", label=1.0, entity="E1", group="cardsfolder/s/shock.txt")],
        sweeps={"toughness": [{"record_id": "g1.1", "entity": "E1"}]},
        join_rate=0.25, join={"fired": 4, "fired_joined": 1},
        settings={"games_per_stratum": 300},
    ).sealed()



class TestDigest:
    def test_it_is_stable_across_key_order(self):
        a = {"items": [1, 2], "games": {"x": ["g1"]}, "corpus_digest": "c"}
        b = {"corpus_digest": "c", "games": {"x": ["g1"]}, "items": [1, 2]}
        assert probe_set.canonical_digest(a) == probe_set.canonical_digest(b)

    def test_it_excludes_itself(self):
        payload = {"items": [1]}
        payload["digest"] = probe_set.canonical_digest(payload)
        assert probe_set.canonical_digest(payload) == payload["digest"]

    def test_it_changes_with_a_label(self):
        digest = probe_set.canonical_digest
        assert digest({"items": [1]}) != digest({"items": [2]})

    def test_a_probing_run_refuses_a_set_frozen_for_another_corpus(self, tmp_path):
        frozen = _probe_set(corpus_digest="aaa")
        path = tmp_path / "set.json"
        path.write_text(json.dumps(frozen.to_json()), encoding="utf-8")
        with pytest.raises(SystemExit, match="frozen against corpus digest"):
            probe_set.load_probe_set(path, "bbb")
        assert probe_set.load_probe_set(path, "aaa").digest == frozen.digest

    def test_an_edited_set_is_refused(self, tmp_path):
        payload = _probe_set(corpus_digest="aaa").to_json()
        payload["join_rate"] = 0.5
        path = tmp_path / "set.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="does not match its own digest"):
            probe_set.load_probe_set(path, "aaa")

    def test_the_json_round_trips_to_the_same_digest(self):
        frozen = _probe_set(corpus_digest="aaa")
        again = probe_set.ProbeSet.from_json(json.loads(json.dumps(frozen.to_json())))
        assert again.sealed().digest == frozen.digest
        assert again.record_items == frozen.record_items

    def test_a_probing_run_refuses_without_a_set(self, tmp_path):
        with pytest.raises(SystemExit, match="--freeze-probe-set"):
            probe_set.load_probe_set(tmp_path / "missing.json", "aaa")

    def test_the_set_follows_the_reports_dir_by_default(self):
        args = run.parse_args(["--checkpoint", "c.pt", "--reports-dir", "arm/reports"])
        assert run.probe_set_location(args, "abcdef0123456789") == Path(
            "arm/reports/knowledge-probes-set-abcdef012345.json",
        )

    def test_a_probe_set_dir_keeps_the_set_shared_across_arms(self):
        args = run.parse_args([
            "--checkpoint", "c.pt", "--reports-dir", "arm/reports",
            "--probe-set-dir", "shared",
        ])
        assert run.probe_set_location(args, "abcdef0123456789") == Path(
            "shared/knowledge-probes-set-abcdef012345.json",
        )


# ── compare ─────────────────────────────────────────────────────────────


def _card(name: str, one: float, width: float, digest: str = "d"):
    return _scorecard({
        "checkpoint": f"models/effects/runs/{name}/latest.pt",
        "probe_set_digest": digest,
        "families": {"magnitudes": {"targets": {"dies": {
            "scope": "entity", "kind": "binary",
            "strata": {"card-disjoint": {"n": 100, "rungs": {
                "0": {"linear": 0.6, "mlp": 0.6},
                "1": {"linear": one, "mlp": one},
                "1w": {"linear": width, "mlp": width},
                "3": {"model": 0.9},
            }}},
        }}}},
    })


def _scorecard(data: dict):
    from effect_knowledge_probes.scorecard import Scorecard

    return Scorecard.from_json(data)


class TestCompare:
    def test_arms_rank_on_rung_one_over_the_width_control(self):
        ranked = compare.ranking([_card("narrow", 0.70, 0.65), _card("wide", 0.80, 0.62)])
        assert [Path(name).parent.name for name, _ in ranked["mlp AUC"]] == ["wide", "narrow"]
        assert ranked["linear AUC"][0][1] == pytest.approx(0.18)

    def test_auc_and_r2_gains_are_ranked_apart(self):
        """An AUC point and an R² point are not one unit; averaging them would
        let a large R² swing on one amount decide the ranking."""
        def card(name, auc_gain, r2_gain):
            body = _card(name, 0.6 + auc_gain, 0.6)
            body.families["magnitudes"]["targets"]["damage_taken"] = {
                "scope": "entity", "kind": "amount",
                "strata": {"card-disjoint": {"n": 100, "rungs": {
                    "1": {"linear": 0.2 + r2_gain, "mlp": 0.2 + r2_gain},
                    "1w": {"linear": 0.2, "mlp": 0.2},
                }}},
            }
            return body

        ranked = compare.ranking([card("auc", 0.10, -0.50), card("r2", 0.01, 0.30)])
        assert [Path(n).parent.name for n, _ in ranked["mlp AUC"]] == ["auc", "r2"]
        assert [Path(n).parent.name for n, _ in ranked["mlp R²"]] == ["r2", "auc"]
        assert ranked["linear AUC"][0][1] == pytest.approx(0.10)

    def test_mismatched_digests_warn(self):
        assert compare.digest_warning([_card("a", 0.7, 0.6, "x"), _card("b", 0.7, 0.6, "y")])
        assert compare.digest_warning([_card("a", 0.7, 0.6), _card("b", 0.7, 0.6)]) is None

    def test_the_table_sets_checkpoints_side_by_side(self, tmp_path):
        paths = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.json"
            path.write_text(json.dumps(_card(name, 0.7, 0.6).to_json()), encoding="utf-8")
            paths.append(path)
        out = tmp_path / "compare.md"
        assert compare.main([str(p) for p in paths] + ["--output", str(out)]) == 0
        text = out.read_text(encoding="utf-8")
        assert "## magnitudes" in text and "a linear" in text and "b mlp" in text
        assert "Ranking on rung 1" in text

    def test_the_table_prints_to_a_cp1252_stdout(self, tmp_path, monkeypatch):
        path = tmp_path / "a.json"
        path.write_text(json.dumps(_card("a", 0.7, 0.6).to_json()), encoding="utf-8")
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        monkeypatch.setattr(sys, "stdout", stdout)
        assert compare.main([str(path)]) == 0
        stdout.flush()
        assert "rung 1 − rung 1w" in stdout.buffer.getvalue().decode("utf-8")


# ── board abilities ─────────────────────────────────────────────────────


def _key(trait_kind: str, face: int = 0):
    from effects.domain.provenance import ProvenanceKey

    return ProvenanceKey("cardsfolder/x/x.txt", face, trait_kind, 0)


class _Sidecars:
    """A sidecar cache over made-up lines: ``get`` for a card, ``line_or_none`` for a key."""

    def __init__(self, lines, api_by_kind=None):
        self.lines = lines
        self.api_by_kind = api_by_kind or {}

    def get(self, script_file):
        return SimpleNamespace(lines=self.lines)

    def line_or_none(self, key):
        api = self.api_by_kind.get(key.trait_kind)
        return None if api is None else _line_of(api)


def _line_of(api, *keys):
    from effects.domain.provenance import SidecarLine

    # A keyword line's script is its keyword; any other line's, a script.
    text = "Flying" if api == "Keyword" else f"AB$ {api}"
    return SidecarLine(
        line_index=0, line_kind="static", provenance=keys,
        script_api_type=api, script_text=text,
    )


def _legality(*forbidden_kinds):
    from effects.domain.records import PlayabilitySubkind, RecordKind

    return SimpleNamespace(
        kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS,
        payload=SimpleNamespace(forbidden=[
            SimpleNamespace(responsible_static=(_key(kind),)) for kind in forbidden_kinds]),
    )


class TestBoard:
    def test_a_keyword_line_and_the_other_face_are_not_unlisted(self):
        entity = SimpleNamespace(printed=(_key("spell"),), face=0)
        sidecars = _Sidecars([
            _line_of("Keyword", _key("static")),
            _line_of("Continuous", _key("static")),
            _line_of(None, _key("trigger", face=1)),
            _line_of("DealDamage", _key("spell")),
        ])
        assert board.unlisted_kinds(entity, sidecars) == {"static"}

    def test_a_permanent_without_printed_keys_lists_nothing(self):
        entity = SimpleNamespace(printed=(), face=0)
        assert board.unlisted_kinds(entity, _Sidecars([])) == frozenset()

    def test_legality_is_split_by_the_restricting_static(self):
        sidecars = _Sidecars([], {"static": "CantBlock", "keyword": "Keyword"})
        assert board.subset_of(_legality("static", "keyword"), sidecars) == "legality-static"
        assert board.subset_of(_legality("keyword"), sidecars) == "legality-keyword"
        assert board.subset_of(_legality(), sidecars) == "legality-none"
        assert board.subset_of(_legality("trigger"), sidecars) is None

    def test_collect_stops_each_subset_at_its_quota(self):
        sidecars = _Sidecars([], {"static": "CantBlock"})
        chosen = board.collect([_legality()] * 5 + [_legality("static")] * 2,
                               sidecars, per_subset=3)
        assert len(chosen["legality-none"]) == 3
        assert len(chosen["legality-static"]) == 2
        assert set(chosen) == set(board.SUBSETS)

    def test_fixture_records_land_in_subsets_that_fit_them(self, records, sidecars):
        from effects.domain.records import Moment, RecordKind

        seen = set()
        for record in records:
            name = board.subset_of(record, sidecars)
            if name is None:
                continue
            seen.add(name)
            if name.startswith("legality"):
                assert record.kind is RecordKind.PLAYABILITY
            else:
                assert record.kind is RecordKind.RESOLUTION
                assert record.moment is Moment.RESOLUTION
            if name == "resolution-replacement":
                assert any(
                    "replacement" in board.unlisted_kinds(entity, sidecars)
                    for entity in record.state.entities
                    if entity.zone == "battlefield" and entity.id != record.state.refs.source)
        assert {"legality-none", "resolution-none", "resolution-replacement"} <= seen

    def test_the_table_sets_subsets_side_by_side(self):
        body = {"base_loss": 0.5, "noise": {"cards": 0.01, "act": 0.2, "all": 0.2}}
        result = {
            "legality-static": {"n": {"card-disjoint": 3}, "ablation": {"blocker_legal": body}},
            "legality-none": {"n": {"card-disjoint": 4}, "ablation": {"blocker_legal": body}},
        }
        text = board.table(result)
        assert "| field | legality-static | legality-none |" in text
        assert "| blocker_legal | 0.500 | 0.500 |" in text
        assert "| blocker_legal | 0.010 | 0.010 |" in text
        assert "legality-static 3" in text

    def test_a_field_no_subset_measured_is_left_out(self):
        zero = {"base_loss": 0.0, "noise": {"cards": 0.0}}
        result = {"legality-none": {"n": {"card-disjoint": 1},
                                    "ablation": {"colors_lost": zero}}}
        assert "colors_lost" not in board.table(result)


# ── one forward pass through a real checkpoint ──────────────────────────

_GEN1 = _ROOT / "models" / "effects" / "runs" / "2026-09-17-full-textless-corpus" / "latest.pt"


@pytest.mark.skipif(not _GEN1.exists(), reason="the gen-1 checkpoint is not on this machine")
class TestExtraction:
    @pytest.fixture(scope="class")
    def probe_model(self, records):
        checkpoint = common.load_checkpoint(_GEN1)
        if not Path(checkpoint.provenance.vocab_path).exists():
            pytest.skip("the gen-1 vocabulary is not on this machine")
        from effects.infrastructure import effect_model_store

        # The probes hash-check their inference files like every inference
        # command, and both files gen-1 recorded have since been rewritten in
        # place: its vocabulary survives as an archived copy, its keyword
        # table nowhere. This pass checks the extraction's shapes, not the
        # hash check (which has tests of its own), so the archived vocabulary
        # is read where it exists and the check is waived for this load.
        vocab = _ROOT / "models" / "effects" / "gen1-vocab-script.txt"
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(effect_model_store, "verify_hashes", lambda *a, **k: None)
            return common.load_probe_model(_GEN1, [
                _FIXTURES / "gen1-sidecars" / "cardsfolder",
                _FIXTURES / "gen1-sidecars" / "tokenscripts",
            ], records, checkpoint=checkpoint,
                vocab_path=vocab if vocab.exists() else None)

    def test_every_rung_has_one_width_per_target(self, probe_model, records, sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        items: dict[str, list] = {}
        for record in records:
            for item in labels.record_items(record, sidecars, join, acting):
                items.setdefault(record.record_id, []).append(item)
        chosen = [r for r in records if r.record_id in items]
        act_hidden: dict = {}
        extracted = ladder.extract_features(probe_model, chosen, items, batch_size=8,
                                            act_hidden=act_hidden)
        assert extracted and act_hidden
        e_dim = probe_model.e_dim
        for target in {i.target for i in extracted}:
            rows = [i for i in extracted if i.target == target]
            for rung in ladder.PROBED_RUNGS:
                assert len({r.features[rung].shape for r in rows}) == 1
            zero = rows[0].features["0"].shape[0]
            assert rows[0].features["1"].shape[0] == zero + 3 * e_dim
            assert rows[0].features["1w"].shape == rows[0].features["1"].shape
            # Gen-1 never trained its verdict head, so the targets that read it
            # carry no rung 3; every other target reads a trained output.
            trained = ladder.verdict_head_trained(probe_model.checkpoint)
            if target in ladder.VERDICT_TARGETS and not trained:
                assert all(np.isnan(r.rung3) for r in rows)
            else:
                assert all(np.isfinite(r.rung3) for r in rows)

    def test_a_toughness_sweep_reads_one_prediction_per_value(self, probe_model, records,
                                                             sidecars):
        chosen = sweeps.select_sweep_records(records, sidecars)
        if not chosen["toughness"]:
            pytest.skip("no damage record in the fixture")
        row = chosen["toughness"][0]
        record = next(r for r in records if r.record_id == row["record_id"])
        curve = sweeps.run_sweep("toughness", probe_model, record, row)
        assert [p["value"] for p in curve] == list(sweeps.TOUGHNESS_VALUES)
        assert all(0.0 <= p["prediction"] <= 1.0 for p in curve)

    def test_the_per_layer_view_reads_every_trunk_layer(self, probe_model, records,
                                                        sidecars):
        acting = run._acting_text_function(sidecars, "script")
        join = labels.join_triggers(records)
        record = next(r for r in records
                      if labels.record_items(r, sidecars, join, acting))
        items = {record.record_id: labels.record_items(record, sidecars, join, acting)}
        extracted = ladder.extract_features(probe_model, [record], items, per_layer=True)
        depth = len(probe_model.model.trunk.layers)
        assert all(len(item.layers) == depth for item in extracted)
        assert extracted[0].layers[0].shape == extracted[0].features["2"].shape

    def test_method_c_trains_a_shallow_trunk_beside_the_full_model(self, probe_model,
                                                                  records, tmp_path):
        import shutil

        (tmp_path / "training").mkdir()
        shutil.copy(_FIXTURES / "gen1-records.jsonl.gz",
                    tmp_path / "training" / "shard-00001.jsonl.gz")
        args = SimpleNamespace(method_c_layers=0, method_c_steps=2)
        result = run.method_c(args, probe_model, tmp_path, records[:16])
        assert result["steps"] == 2 and result["layers"] == 0
        assert set(result["shallow"]) == set(result["full"])
        assert all(np.isfinite(v) for v in result["shallow"].values())


class TestRung3ReadsTheRightOutput:
    """Rung 3 is the model's own prediction, read from the output that learned it."""

    def _outputs(self, gate_logit: float, legal_logit: float):
        import torch

        from effects.domain.effect_head_layout import FIELD_SLICES, GATE_INDEX, PER_ENTITY_WIDTH

        vector = torch.zeros(1, 1, PER_ENTITY_WIDTH)
        vector[0, 0, GATE_INDEX] = gate_logit
        vector[0, 0, FIELD_SLICES["target_legal"][0]] = legal_logit
        return vector

    def test_target_legality_reads_the_gate_not_the_field(self):
        """Training sets `target_legal` only on legal targets, always to 1, so
        its output on an illegal entity was never supervised; the gate is what
        a decision record raises on its legal targets."""
        import torch

        outputs = self._outputs(gate_logit=-3.0, legal_logit=5.0)
        value = ladder._readout("legal_target", outputs, None, 0, 0)
        assert value == pytest.approx(float(torch.sigmoid(torch.tensor(-3.0))))

    def test_damage_taken_reads_the_amount_alone(self):
        """Every damage_taken item took damage, so the gate's doubt about
        whether the creature was hit must not scale the head's amount."""
        import math

        from effects.domain.effect_head_layout import FIELD_SLICES

        outputs = self._outputs(gate_logit=-3.0, legal_logit=0.0)
        outputs[0, 0, FIELD_SLICES["damage_taken"][0]] = math.log(3.0)
        assert ladder._readout("damage_taken", outputs, None, 0, 0) == pytest.approx(3.0)

    def test_an_untrained_verdict_head_gives_no_rung_3(self):
        assert np.isnan(ladder._readout("affordable", None, None, 0, 0,
                                        verdict_trained=False))
        assert np.isnan(ladder._readout("fires", None, None, 0, 0,
                                        verdict_trained=False))

    def test_a_checkpoint_without_training_settings_never_trained_the_verdict_head(self):
        class Gen1:
            training_settings: dict = {}

        class Gen2:
            training_settings = {"tokenizer_rules": "gen-2", "value_weight": 0.05}

        assert not ladder.verdict_head_trained(Gen1())
        assert ladder.verdict_head_trained(Gen2())

    def test_the_ladder_reports_why_rung_3_is_missing_and_gives_no_share(self):
        rng = np.random.default_rng(4)
        n = 200
        y = rng.integers(0, 2, n).astype(float)
        items = [
            ladder.ExtractedItem(
                target="affordable", label=float(label), group=f"g{i}",
                features={rung: rng.normal(size=3) for rung in ladder.PROBED_RUNGS},
                rung3=float("nan"),
            )
            for i, label in enumerate(y)
        ]
        data = ladder.ladder_inputs(items, {"affordable": "binary"})["affordable"]
        assert data.rung3 is None and data.rung3_note == ladder.UNTRAINED_VERDICT_NOTE
        result = ladder.run_ladder(data, rungs=("0", "1"),
                                   mlp_options={"hidden": 8, "epochs": 2, "device": "cpu"},
                                   bootstrap=0)
        assert "3" not in result["rungs"] and result["share"] is None
        assert result["rung3_unavailable"] == ladder.UNTRAINED_VERDICT_NOTE


# ── the combat grid ─────────────────────────────────────────────────────


def _one_on_one(records):
    """A fixture combat turned into one attacker blocked by one blocker."""
    from dataclasses import replace

    from effects.domain.records import RecordKind

    record = next(r for r in records if r.kind is RecordKind.COMBAT
                  and any(len(b) == 2 for b in r.payload.blocks.values()))
    attacker, (blocker, dropped) = next(
        (a, b) for a, b in record.payload.blocks.items() if len(b) == 2)
    entities = []
    for entity in record.state.entities:
        if entity.id == dropped:
            continue
        if entity.id == attacker:
            entity = replace(entity, combat=replace(entity.combat, blocked_by=(blocker,)))
        entities.append(entity)
    record = replace(
        record, fork=False, interventional=False, mirror_of=None,
        payload=replace(record.payload, blocks={attacker: (blocker,)}, probed_keyword=None),
        state=replace(record.state, entities=tuple(entities),
                      global_=replace(record.state.global_, combat_substep="regular")))
    return record, attacker, blocker


def _keywords(entity):
    return set(entity.granted_temporary.keywords)


class TestCombatGrid:
    @pytest.mark.parametrize("attacker, blocker, keywords, step, expected", [
        ((3, 3, 0), (2, 2, 0), ({}, {}), "regular", (False, True)),
        ((2, 2, 0), (2, 3, 0), ({}, {}), "regular", (True, False)),
        ((2, 2, 0), (1, 3, 1), ({}, {}), "regular", (False, True)),
        ((1, 5, 0), (1, 5, 0), ({"deathtouch"}, {}), "regular", (False, True)),
        ((5, 5, 0), (5, 5, 0), ({}, {"indestructible"}), "regular", (True, False)),
        ((3, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, True)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "first_strike", (False, False)),
        ((2, 3, 0), (5, 3, 0), ({"first_strike"}, {}), "regular", (True, False)),
        ((5, 3, 0), (3, 3, 0), ({}, {"first_strike"}), "first_strike", (True, False)),
        ((2, 2, 0), (1, 4, 0), ({"double_strike"}, {}), "first_strike", (False, False)),
        ((2, 2, 0), (1, 4, 2), ({"double_strike"}, {}), "regular", (False, True)),
        ((2, 2, 0), (0, 4, 0), ({}, {}), "first_strike", (False, False)),
    ])
    def test_the_rule(self, attacker, blocker, keywords, step, expected):
        assert combat_grid.step_deaths(
            attacker, blocker, frozenset(keywords[0]), frozenset(keywords[1]), step) == expected

    def test_power_edits_one_entity_only(self, records):
        record, attacker, _ = _one_on_one(records)
        edited = BoardEdit(record).set_power(attacker, 7).record
        assert edited.state.entity(attacker).pt.total[0] == 7
        assert (edited.state.entity(attacker).pt.total[1]
                == record.state.entity(attacker).pt.total[1])
        assert _differences(record, edited) == {"entities"}

    def test_a_granted_keyword_is_added_once(self, records):
        record, attacker, _ = _one_on_one(records)
        once = BoardEdit(record).grant_keyword(attacker, "deathtouch")
        twice = once.grant_keyword(attacker, "deathtouch").record
        assert twice.state.entity(attacker).granted_temporary.keywords.count("deathtouch") == 1

    def test_a_one_on_one_block_qualifies_when_its_outcome_obeys_the_rule(
        self, records, monkeypatch,
    ):
        import effects.domain.effect_targets as effect_targets
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, attacker, blocker = _one_on_one(records)
        a, _ = combat_grid.side_of(record, attacker)
        b, _ = combat_grid.side_of(record, blocker)
        attacker_dies, blocker_dies = combat_grid.step_deaths(a, b)
        died = ZONE_OUTCOMES.index("died")
        outcome = {e: SimpleNamespace(fields={"zone_outcome": died} if dies else {})
                   for e, dies in ((attacker, attacker_dies), (blocker, blocker_dies))}
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
        assert combat_grid.single_block(record, _keywords) == (attacker, blocker)
        outcome[blocker] = SimpleNamespace(
            fields={} if blocker_dies else {"zone_outcome": died})
        assert combat_grid.single_block(record, _keywords) is None

    def test_a_probe_fork_or_two_blockers_never_qualify(self, records):
        from effects.domain.records import RecordKind

        for record in records:
            if record.kind is RecordKind.COMBAT and (
                    record.fork or any(len(b) > 1 for b in record.payload.blocks.values())):
                assert combat_grid.single_block(record, _keywords) is None

    def test_grid_rules_follow_the_keyword(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, rule = combat_grid.grid_records(record, attacker, blocker, "blocker", "none")
        assert len(rows) == 8 and all(len(row) == 8 for row in rows)
        assert np.array_equal(rule, np.array([[float(p >= t) for t in range(1, 9)]
                                              for p in range(1, 9)]))
        assert rows[2][5].state.entity(attacker).pt.total[0] == 3
        assert rows[2][5].state.entity(blocker).pt.total[1] == 6
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-deathtouch")
        assert rule.min() == 1.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "blocker-indestructible")
        assert rule.max() == 0.0

    def test_a_first_striker_is_read_in_the_first_strike_step(self, records):
        record, attacker, blocker = _one_on_one(records)
        step = np.array([[float(p >= t) for t in range(1, 9)] for p in range(1, 9)])
        rows, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-first-strike")
        assert rows[0][0].state.global_.combat_substep == "first_strike"
        assert np.array_equal(rule, step)
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "attacker", "attacker-first-strike")
        assert rule.max() == 0.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "blocker-first-strike")
        assert rule.max() == 0.0
        _, rule = combat_grid.grid_records(
            record, attacker, blocker, "attacker", "blocker-first-strike")
        assert np.array_equal(rule, step)

    def test_a_double_strikers_second_hit_lands_on_the_first(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, rule = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-double-strike-second-hit")
        assert rows[2][5].state.global_.combat_substep == "regular"
        assert rows[2][5].state.entity(blocker).damage == 3
        assert np.array_equal(rule, np.array([[float(2 * p >= t) for t in range(1, 9)]
                                              for p in range(1, 9)]))

    def test_the_assignment_follows_the_edited_board(self, records):
        record, attacker, blocker = _one_on_one(records)
        rows, _ = combat_grid.grid_records(record, attacker, blocker, "blocker", "none")
        choices = rows[4][1].payload.assignment_choices
        assert choices[attacker] == {blocker: 5}
        assert choices[blocker] == {attacker: rows[4][1].state.entity(blocker).pt.total[0]}
        rows, _ = combat_grid.grid_records(
            record, attacker, blocker, "blocker", "attacker-first-strike")
        choices = rows[2][0].payload.assignment_choices
        assert choices[attacker] == {blocker: 3}
        assert blocker not in choices, "the blocker deals nothing in the first-strike step"

    def test_a_first_strike_step_record_never_qualifies(self, records, monkeypatch):
        import effects.domain.effect_targets as effect_targets
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, attacker, blocker = _one_on_one(records)
        a, _ = combat_grid.side_of(record, attacker)
        b, _ = combat_grid.side_of(record, blocker)
        died = ZONE_OUTCOMES.index("died")
        outcome = {e: SimpleNamespace(fields={"zone_outcome": died} if dies else {})
                   for e, dies in zip((attacker, blocker), combat_grid.step_deaths(a, b))}
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
        assert combat_grid.single_block(record, _keywords) == (attacker, blocker)
        first = BoardEdit(record).set_step("first_strike").record
        assert combat_grid.single_block(first, _keywords) is None

    def test_damage_the_step_already_marked_is_cleared_and_other_damage_disqualifies(
        self, records, monkeypatch,
    ):
        import effects.domain.effect_targets as effect_targets
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, attacker, blocker = _one_on_one(records)
        power = record.state.entity(attacker).pt.total[0]
        marked = BoardEdit(record).set_marked_damage(blocker, power).record
        a, _ = combat_grid.side_of(record, attacker)
        b, _ = combat_grid.side_of(record, blocker)
        attacker_dies, blocker_dies = combat_grid.step_deaths(a, b)
        died = ZONE_OUTCOMES.index("died")
        outcome = {e: SimpleNamespace(fields={"zone_outcome": died} if dies else {})
                   for e, dies in ((attacker, attacker_dies), (blocker, blocker_dies))}
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
        assert combat_grid.single_block(marked, _keywords) == (attacker, blocker)
        earlier = BoardEdit(record).set_marked_damage(blocker, power + 1).record
        assert combat_grid.single_block(earlier, _keywords) is None
        rows, rule = combat_grid.grid_records(marked, attacker, blocker, "attacker", "none")
        assert all(cell.state.entity(blocker).damage == 0 for row in rows for cell in row)
        assert np.array_equal(rule, np.array([[float(p >= t) for t in range(1, 9)]
                                              for p in range(1, 9)]))


# ── the evasion grid ────────────────────────────────────────────────────

def _block_pair(records):
    """A fixture block-legality record made to qualify: a red attacker, a green blocker."""
    from dataclasses import replace

    from effects.domain.records import PlayabilityBlockersPayload

    record = next(r for r in records if isinstance(r.payload, PlayabilityBlockersPayload)
                  and r.payload.legal_blockers)
    attacker, blocker = record.payload.anchor_attacker, record.payload.legal_blockers[0]
    board = (BoardEdit(record).set_colours(attacker, ("R",)).set_colours(blocker, ("G",))
             .edit_entity(blocker, lambda e: replace(e, types=("creature",))))
    for entity in (attacker, blocker):
        board = board.edit_entity(entity, lambda e: replace(
            e, granted_temporary=replace(e.granted_temporary, keywords=())))
    return board.record, attacker, blocker


def _no_keywords(entity):
    return set(entity.granted_temporary.keywords)


class TestEvasionGrid:
    @pytest.mark.parametrize("attacker, blocker, colours, types, expected", [
        ((), (), ("G",), ("creature",), True),
        (("flying",), (), ("G",), ("creature",), False),
        (("flying",), ("reach",), ("G",), ("creature",), True),
        (("horsemanship",), ("flying",), ("G",), ("creature",), False),
        (("fear",), (), ("B",), ("creature",), True),
        (("fear",), (), ("G",), ("artifact", "creature"), True),
        (("fear",), (), ("G",), ("creature",), False),
        (("intimidate",), (), ("R",), ("creature",), True),
        (("intimidate",), (), ("G",), ("creature",), False),
        (("shadow",), (), ("G",), ("creature",), False),
        ((), ("shadow",), ("G",), ("creature",), False),
        (("unblockable",), ("reach",), ("G",), ("creature",), False),
        (("protection_from_green",), (), ("G",), ("creature",), False),
        (("protection_from_blue",), (), ("G",), ("creature",), True),
    ])
    def test_the_rule(self, attacker, blocker, colours, types, expected):
        assert evasion_grid.may_block(attacker, ("R",), blocker, colours, types) is expected

    def test_a_cell_grants_the_keyword_and_the_trait_and_scores_them(self, records):
        record, attacker, blocker = _block_pair(records)
        cell, rule = evasion_grid.edited(record, attacker, blocker, "flying", None)
        assert "flying" in cell.state.entity(attacker).granted_temporary.keywords
        assert rule is False
        cell, rule = evasion_grid.edited(record, attacker, blocker, "flying", "reach")
        assert "reach" in cell.state.entity(blocker).granted_temporary.keywords
        assert rule is True
        _, rule = evasion_grid.edited(record, attacker, blocker, "intimidate", "shares-colour")
        assert rule is True
        _, rule = evasion_grid.edited(record, attacker, blocker, "fear", "artifact")
        assert rule is True

    def test_protection_resolves_against_the_blockers_colour(self, records):
        record, attacker, blocker = _block_pair(records)
        cell, rule = evasion_grid.edited(record, attacker, blocker, "protection-blocker", None)
        assert "protection_from_green" in cell.state.entity(attacker).granted_temporary.keywords
        assert rule is False
        cell, rule = evasion_grid.edited(record, attacker, blocker, "protection-other", None)
        granted = cell.state.entity(attacker).granted_temporary.keywords
        assert any(k.startswith("protection_from_") and k != "protection_from_green"
                   and k != "protection_from_red" for k in granted)
        assert rule is True

    def test_qualification(self, records):
        from dataclasses import replace

        record, attacker, blocker = _block_pair(records)
        assert evasion_grid.qualifying_pair(record, _no_keywords) == (attacker, blocker)
        assert evasion_grid.qualifying_pair(replace(record, fork=True), _no_keywords) is None
        flying = BoardEdit(record).grant_keyword(attacker, "flying").record
        assert evasion_grid.qualifying_pair(flying, _no_keywords) is None
        black = BoardEdit(record).set_colours(blocker, ("B",)).record
        assert evasion_grid.qualifying_pair(black, _no_keywords) is None


class TestTargetRestrictions:
    def test_the_restriction_is_read_from_the_first_segment(self):
        text = "SP$ Destroy | ValidTgts$ Creature.nonLegendary | TgtPrompt$ x [SEG] SV1 | " \
               "ValidTgts$ Player"
        assert target_restrictions.restriction_of(text) == "Creature.nonLegendary"
        assert target_restrictions.restriction_of("SP$ Draw | NumCards$ 2") is None
        assert target_restrictions.masked("Creature.powerLE2") == "Creature.powerLE#"

    @pytest.mark.parametrize("restriction, expected", [
        ("Creature", "card type only"),
        ("Artifact,Enchantment", "card type only"),
        ("Creature.OppCtrl", "controller"),
        ("Creature.nonLegendary", "supertype"),
        ("Creature.nonBlack", "colour"),
        ("Creature.withFlying", "keyword"),
        ("Creature.powerLE2", "number"),
        ("Creature.attacking", "combat or tapped"),
        ("Card.Creature+namedSomething", "other"),
    ])
    def test_family(self, restriction, expected):
        assert target_restrictions.family(restriction) == expected

    def test_flips_break_exactly_the_stated_condition(self, records):
        from effects.domain.records import PlayabilityDecisionPayload

        assert target_restrictions.flip_for("Creature.OppCtrl") == "to the acting player"
        assert target_restrictions.flip_for("Artifact") == "loses artifact"
        assert target_restrictions.flip_for("Artifact,Creature") is None
        assert target_restrictions.flip_for("Creature.OppCtrl+nonLegendary") is None
        record = next(r for r in records if isinstance(r.payload, PlayabilityDecisionPayload)
                      and r.payload.candidates and r.payload.candidates[0].legal_targets)
        entity = next(e for e in record.state.entities if e.zone == "battlefield")
        mine = target_restrictions.flipped(record, entity.id, "to the acting player")
        assert mine.state.entity(entity.id).controller == record.actor_player
        theirs = target_restrictions.flipped(record, entity.id, "to the opponent")
        assert theirs.state.entity(entity.id).controller != record.actor_player
        legendary = target_restrictions.flipped(record, entity.id, "made legendary")
        assert "legendary" in legendary.state.entity(entity.id).supertypes
        lost = target_restrictions.flipped(record, entity.id, "loses creature")
        assert "creature" not in lost.state.entity(entity.id).types

    def test_auc(self):
        assert common.auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
        assert common.auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
        assert np.isnan(common.auc([1, 1], [0.5, 0.6]))

    def test_a_tie_counts_half(self):
        """A score that cannot tell a positive from a negative ranks at chance."""
        assert common.auc([1, 0], [0.5, 0.5]) == 0.5
        assert common.auc([1, 1, 0, 0], [0.9, 0.5, 0.5, 0.1]) == pytest.approx(0.875)




class TestColourPips:
    def test_the_cost_parses_into_coloured_pips_and_generic(self):
        assert colour_pips.parse_cost("{2}{R}{W/U}") == ([("R",), ("W", "U")], 2)
        assert colour_pips.shape([("U",)]) == "one pip"
        assert colour_pips.shape([("U",), ("U",)]) == "repeated pip"
        assert colour_pips.shape([("W",), ("U",)]) == "several colours"
        assert colour_pips.shape([("W", "U")]) == "hybrid"

    @pytest.mark.parametrize("cost, lands, expected", [
        ("{2}{R}", ["R", "G", "G"], True),
        ("{2}{R}", ["G", "G", "G"], False),
        ("{2}{R}", ["R", "G"], False),
        ("{U}{U}", ["U", "G"], False),
        ("{1}{U}", ["U", "U"], True),
        ("{W/U}{W/U}", ["W", "U"], True),
        ("{W/U}{W/U}", ["W", "G"], False),
    ])
    def test_the_rule_matches_pips_to_lands(self, cost, lands, expected):
        assert colour_pips.payable(*colour_pips.parse_cost(cost), lands) is expected

    @pytest.mark.parametrize("cost", ["{2}{R}", "{U}{U}", "{1}{U}{U}", "{W}{U}", "{W/U}",
                                      "{G}"])
    def test_configurations_keep_their_rule_answers(self, cost):
        configs = colour_pips.configurations(cost)
        coloured, generic = colour_pips.parse_cost(cost)
        total = len(coloured) + generic
        assert configs["exact"] == (configs["exact"][0], True)
        assert configs["surplus"][1] is True
        for name in ("one short", "pip missing", "pip missing, +2", "all off colour"):
            assert configs[name][1] is False, name
        assert len(configs["exact"][0]) == total
        assert len(configs["one short"][0]) == total - 1
        assert len(configs["pip missing"][0]) == total
        assert len(configs["pip missing, +2"][0]) == total + 2

    def test_the_missing_colour_is_still_present_only_for_a_repeated_pip(self):
        assert colour_pips.missing_colour_present("{1}{U}{U}")
        assert not colour_pips.missing_colour_present("{1}{U}")
        assert not colour_pips.missing_colour_present("{W}{U}")

    def test_a_configuration_edits_the_lands_to_its_colours(self, records):
        record = _decision(records, "{2}{U}")
        actor = record.actor_player
        lands, _ = colour_pips.configurations("{2}{U}")["pip missing"]
        edited = sweeps.set_untapped_lands(record, actor, list(lands), len(lands))
        production = player_of(edited.state, actor).untapped_production
        assert {c: n for c, n in production.items() if n} == {"W": 3}


class TestZoneEffects:
    def test_chain_apis_read_every_segment(self):
        text = "SP$ Destroy | ValidTgts$ Creature [SEG] SV1 | DB$ Draw | NumCards$ 2"
        assert zone_effects.chain_apis(text) == {"Destroy", "Draw"}
        assert zone_effects.chain_apis(None) == set()

    def test_set_amount(self):
        text = "SP$ Draw | NumCards$ 2 | ValidTgts$ Player"
        assert zone_effects.set_amount(text, 5) == "SP$ Draw | NumCards$ 5 | ValidTgts$ Player"

    def test_retargeting_moves_only_the_player_target(self, records):
        from dataclasses import replace

        record = next(r for r in records if len(r.state.players) == 2)
        one, two = (p.id for p in record.state.players)
        aimed = replace(record, state=replace(record.state, refs=replace(
            record.state.refs, targets=(one, "card-1"))))
        assert zone_effects.player_targets(aimed) == [one]
        moved = zone_effects.retargeted(aimed, one, two)
        assert moved.state.refs.targets == (two, "card-1")
        assert zone_effects.player_targets(moved) == [two]


# ── what a checkpoint's probes read (FR-075) ─────────────────────────────


def _corpus_manifest(**overrides):
    from effects.domain.corpus_manifest import CorpusManifest

    fields = dict(
        seed=1, surface="script", vocab_path="models/effects/vocab-script.txt",
        variant_scripts="", holdout_permille=20, holdout_max_carriers=8,
        text_cap=0, card_disjoint_text_cap=0, training_records=1, class_mix={},
        delivered_mix={}, held_out_cards=(), held_out_texts=(),
        card_disjoint_games=("cg",), game_disjoint_games=("gg",), rarity={},
        sources=(), per_class={}, per_stratum={}, unique_texts={}, shortfall={},
    )
    fields.update(overrides)
    return CorpusManifest(**fields)


class TestProbeContext:
    """Every probe reads the corpus, sidecar trees and variant tree its checkpoint
    trained against, never a corpus or probe set named in the script."""

    def _context(self, tmp_path, monkeypatch, *, recorded_digest=None):
        from effects.domain.split_provenance import SplitProvenance
        from effects.infrastructure.corpus_store import CorpusStore

        variants = tmp_path / "variant-scripts"
        manifest = _corpus_manifest(variant_scripts=str(variants))
        corpus = tmp_path / "corpus"
        CorpusStore(corpus).save(manifest)
        checkpoint = SimpleNamespace(
            provenance=SplitProvenance(
                corpus_path=str(corpus),
                corpus_digest=recorded_digest or manifest.digest(),
            ),
            cards_folders=(str(tmp_path / "cardsfolder"), str(tmp_path / "tokenscripts")),
            epoch=3,
        )
        monkeypatch.setattr(common, "load_checkpoint", lambda path: checkpoint)
        return common.probe_context(tmp_path / "latest.pt"), manifest, corpus, variants

    def test_the_corpus_is_the_one_the_checkpoint_recorded(self, tmp_path, monkeypatch):
        context, manifest, corpus, _ = self._context(tmp_path, monkeypatch)
        assert context.corpus == corpus
        assert context.corpus_digest == manifest.digest()
        assert context.validation_dirs() == [
            corpus / "validation" / "card-disjoint",
            corpus / "validation" / "game-disjoint",
        ]

    def test_the_sidecars_read_the_variant_tree_the_corpus_recorded(
        self, tmp_path, monkeypatch,
    ):
        """Without it a variant card's ability keys resolve to no text."""
        context, _, _, variants = self._context(tmp_path, monkeypatch)
        assert context.variant_scripts == variants
        assert context.sidecars().path_for("variant-scripts/shock_variant.txt").parent == (
            variants
        )
        assert context.cards_folders == (tmp_path / "cardsfolder", tmp_path / "tokenscripts")

    def test_a_corpus_rebuilt_since_training_is_refused(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="rebuilt"):
            self._context(tmp_path, monkeypatch, recorded_digest="an-older-build")

    def test_a_probe_set_frozen_on_another_corpus_is_refused(self, tmp_path, monkeypatch):
        context, *_ = self._context(tmp_path, monkeypatch)
        path = probe_set.probe_set_path(tmp_path, context.corpus_digest)
        payload = {"corpus_digest": "another-corpus", "items": []}
        payload["digest"] = probe_set.canonical_digest(payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match="another-corpus"):
            context.load_probe_set(tmp_path)


class TestPredict:
    def test_it_reads_the_batch_the_batcher_builds(self):
        """``build`` returns inputs, surfaces and encoded texts, not a pair."""
        from effects.application.surface_batching import Materialized

        seen = {}

        class Model:
            def __call__(self, inputs):
                seen["inputs"] = inputs
                return "hidden"

            def per_entity(self, hidden):
                return f"per-entity of {hidden}"

            def verdict(self, hidden):
                return f"verdict of {hidden}"

        batcher = SimpleNamespace(build=lambda records, encoder: Materialized(
            inputs={"x": 1}, surfaces=["s0", "s1"], encoded=None,
        ))
        probe_model = SimpleNamespace(batcher=batcher, encoder=None, model=Model())

        rows = common.predict(probe_model, ["r0", "r1"], lambda o, v, s, r: (o, v, s, r))

        assert seen == {"inputs": {"x": 1}}
        assert rows == [
            ("per-entity of hidden", "verdict of hidden", "s0", 0),
            ("per-entity of hidden", "verdict of hidden", "s1", 1),
        ]


# ── the line ladder's held-out split ────────────────────────────────────


class TestLineSplit:
    """Every game naming a held-out card is kept out of training, so a text is
    untrained exactly when every card printing it is held out."""

    def test_a_trained_text_printed_on_a_held_out_card_is_trained(self):
        assert run.line_split({"Shock", "Lightning Strike"}, {"Shock"}) == "trained"

    def test_a_text_every_carrier_of_which_is_held_out_is_held_out(self):
        assert run.line_split({"Shock", "Spark"}, {"Shock", "Spark", "Bolt"}) == "held-out"

    def test_a_text_on_no_held_out_card_is_trained(self):
        assert run.line_split({"Lightning Strike"}, {"Shock"}) == "trained"
