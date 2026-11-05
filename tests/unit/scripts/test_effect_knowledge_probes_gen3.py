"""The gen-3 probes of FR-074 (T161-T167) on the gen-3 fixture records, read
through a tiny randomly initialized gen-3 model: no trained gen-3 checkpoint
exists yet, so these check that every probe edits a real gen-3 board the way
its rule says and reads a gen-3 model's outputs, not what a trained one scores.

``scripts/`` is put on the path by the conftest, so the suite imports as the
package its entry scripts make it.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from effect_knowledge_probes import (
    ablation,
    anthem,
    board,
    combat_grid,
    common,
    conditions,
    evasion_grid,
    selection,
    sweeps,
    target_restrictions,
)
from effect_knowledge_probes.board_edits import BoardEdit

from effects.domain.records import (
    CombatPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    RecordKind,
)
from tests.unit.effects.builders import GEN3_SIDECAR_FOLDERS, curated_fixture_records

E_DIM = 8


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


@pytest.fixture(scope="module")
def sidecars():
    return common.sidecar_cache(GEN3_SIDECAR_FOLDERS)


@pytest.fixture(scope="module")
def keywords_of(sidecars):
    from effects.domain.damage_step_keywords import KeywordResolver

    return KeywordResolver(sidecars).keywords_of


@pytest.fixture(scope="module")
def probe_model(records, sidecars):
    """A gen-3 encoder, trunk and batcher at random weights over the fixture's
    texts: what a probe reads off a gen-3 checkpoint, without one."""
    import torch

    from effects.application.model_loading import evaluation_fields
    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.ability_tokenizer import SPECIAL_TOKENS, AbilityTokenizer, Surface
    from effects.domain.baselines import baseline_masks
    from effects.domain.effect_head_input import SlotKind, feature_widths
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.infrastructure.tokenizer_loading import load_keyword_definitions
    from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, make_batcher

    definitions = load_keyword_definitions(FIXTURE_KEYWORD_DEFINITIONS)
    base = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    bare = AbilityTokenizer(base, definitions, surface="script")
    widths = feature_widths(records)
    probe = make_batcher(tokenizer=bare, sidecars=sidecars, masks=baseline_masks("full"),
                         surface="script", e_dim=E_DIM, widths=widths,
                         device=torch.device("cpu"))
    vocab = dict(base)
    texts = [*probe.batch_texts(records)]
    texts += [d.reminder_template for d in definitions.values() if d.reminder_template]
    for text in texts:
        for token in bare.tokenize(text):
            vocab.setdefault(token.text, len(vocab))
    tokenizer = AbilityTokenizer(vocab, definitions, surface="script")
    batcher = make_batcher(tokenizer=tokenizer, sidecars=sidecars,
                           masks=baseline_masks("full"), surface="script", e_dim=E_DIM,
                           widths=widths, device=torch.device("cpu"))
    torch.manual_seed(0)
    encoder = AbilityEncoder(AbilityEncoderConfig(
        vocab_size=len(vocab), e_dim=E_DIM, d_model=16, n_layers=1)).eval()
    model = EffectModel(EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
        ability_features=widths[SlotKind.ABILITY], e_dim=E_DIM, vocab_size=len(vocab),
        encoder_d_model=16, d_model=16, n_layers=1, n_heads=2, ff_dim=32,
    )).eval()
    return common.ProbeModel(
        checkpoint=None, encoder=encoder, model=model, batcher=batcher,
        fields=evaluation_fields(records), device=torch.device("cpu"),
        surface=Surface.SCRIPT, e_dim=E_DIM)


def _probabilities(values) -> None:
    values = np.asarray(values, dtype=float)
    assert values.size and np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))


# ── T161: turn and step conditions ──────────────────────────────────────


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
                assert item.subset == "restriction" and item.entity is None
                if item.edit == conditions.AS_COLLECTED:
                    assert item.rule == (item.readout.removeprefix("failed_") in failed)
                if item.edit == conditions.WHOSE_TURN:
                    assert item.record.state.global_.active != record.state.global_.active
            # Handing the turn over flips a whose-turn restriction's answer.
            by = {(i.edit, i.readout): i.rule for i in items}
            if ("as collected", "failed_whose_turn") in by:
                assert by[("whose turn", "failed_whose_turn")] != by[
                    ("as collected", "failed_whose_turn")]

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
        key = next(line.provenance[0] for path in GEN3_SIDECAR_FOLDERS
                   for line in _sidecar_lines(path)
                   if "Condition$ PlayerTurn" in (line.script_text or "")
                   and line.line_kind == "static")
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
        _probabilities(predictions)
        scores = conditions.score(items, predictions)
        assert set(scores) == set(conditions.SUBSETS)
        assert scores["casting-speed"]["items"] > 0
        assert np.isfinite(scores["casting-speed"]["auc"])
        conditions.report(scores)


def _sidecar_lines(folder):
    from effects.infrastructure.sidecar_io import read_sidecar

    for path in sorted(folder.rglob("*.provenance.json")):
        yield from read_sidecar(path).lines


# ── T162: the anthem ────────────────────────────────────────────────────


def _anthem_board(records, sidecars):
    """A fixture board with an anthem, one creature it reaches given damage
    the anthem alone keeps from being lethal."""
    from effects.domain.records import Moment

    for record in records:
        if record.fork or not (record.kind is RecordKind.COMBAT or (
                record.kind is RecordKind.RESOLUTION and record.moment is Moment.RESOLUTION)):
            continue
        for found in anthem.anthems(record, sidecars):
            creature = record.state.entity(found.affected[0])
            damage = creature.toughness - found.boost[1]
            if damage <= 0:
                continue
            edited = BoardEdit(record).set_marked_damage(creature.id, damage).record
            if anthem.chosen(edited, sidecars) is not None:
                return edited, found, creature.id
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
        _probabilities(predictions)
        result = anthem.score(rows, predictions)
        assert result["rule_dies"] >= 1 and result["creatures"] == len(found.affected)
        anthem.report(result, 1)

    def test_a_creature_that_died_disqualifies_the_board(self, records, sidecars, monkeypatch):
        import effects.domain.effect_targets as effect_targets
        from effects.domain.effect_head_layout import ZONE_OUTCOMES

        record, found, creature = _anthem_board(records, sidecars)
        died = SimpleNamespace(fields={"zone_outcome": ZONE_OUTCOMES.index("died")})
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: {creature: died})
        assert anthem.chosen(record, sidecars) is None


# ── T163: combats with several blocked attackers ────────────────────────


def _two_pairs(records, monkeypatch):
    """A fixture one-on-one block with a second pair beside it (copies of the
    first, under new ids), every outcome the rule's."""
    import effects.domain.effect_targets as effect_targets
    from effects.domain.effect_head_layout import ZONE_OUTCOMES

    record = next(r for r in records if r.kind is RecordKind.COMBAT
                  and isinstance(r.payload, CombatPayload)
                  and sum(1 for b in r.payload.blocks.values() if b) == 1)
    attacker, (blocker, *_) = next((a, b) for a, b in record.payload.blocks.items() if b)
    a2, b2 = f"{attacker}~2", f"{blocker}~2"
    copies = []
    for entity in record.state.entities:
        if entity.id == attacker:
            copies.append(replace(entity, id=a2, combat=replace(entity.combat,
                                                                 blocked_by=(b2,))))
        if entity.id == blocker:
            copies.append(replace(entity, id=b2, combat=replace(entity.combat,
                                                                 blocking=(a2,))))
    entities = [replace(e, damage=0, granted_temporary=replace(e.granted_temporary, keywords=()))
                if e.id in (attacker, blocker) else e for e in record.state.entities]
    entities = [replace(e, combat=replace(e.combat, blocked_by=(blocker,)))
                if e.id == attacker else e for e in entities]
    entities = [replace(e, combat=replace(e.combat, blocking=(attacker,)))
                if e.id == blocker else e for e in entities]
    entities += [replace(e, damage=0) for e in copies]
    record = replace(
        record, fork=False, interventional=False, mirror_of=None,
        payload=replace(record.payload, blocks={attacker: (blocker,), a2: (b2,)},
                        probed_keyword=None),
        state=replace(record.state, entities=tuple(entities),
                      global_=replace(record.state.global_, combat_substep="regular")))
    died = ZONE_OUTCOMES.index("died")
    outcome = {}
    for a, b in ((attacker, blocker), (a2, b2)):
        deaths = combat_grid.step_deaths(combat_grid.side_of(record, a)[0],
                                         combat_grid.side_of(record, b)[0])
        for entity, dies in zip((a, b), deaths):
            outcome[entity] = SimpleNamespace(fields={"zone_outcome": died} if dies else {})
    monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
    return record, (attacker, blocker), (a2, b2)


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
        outcome[second[1]] = SimpleNamespace(fields={"zone_outcome": 1} if flipped else {})
        monkeypatch.setattr(effect_targets, "derive_targets", lambda r: outcome)
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
        assert cell.payload.assignment_choices[attacker] == {blocker: 3}

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
        assert grids["combat/none/blocker"].qualify(record) is None


# ── T164: protection rows over every quality ────────────────────────────


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


class TestProtectionRows:
    @pytest.mark.parametrize("row, fragment, column, allowed", [
        ("protection-blocker", "Protection from green", None, False),
        ("protection-other", None, None, True),
        ("protection-type-blocker", "Protection:Elf", None, False),
        ("protection-type-blocker", "Protection:Elf", "black", False),
        ("protection-type-other", "Protection:Kraken", None, True),
        ("protection-creatures", "Protection:Creature", "reach", False),
        ("protection-artifacts", "Protection:Artifact", None, True),
        ("protection-artifacts", "Protection:Artifact", "artifact", False),
        ("protection-monocolored", "Protection:Card.MonoColor:monocolored", None, False),
        ("protection-multicolored", "Protection:Card.MultiColor:multicolored", None, True),
        ("protection-mana-value-3-or-less", "Protection:Card.cmcLE3:mana value 3 or less",
         None, False),
        ("protection-everything", "Protection from everything", "flying", False),
    ])
    def test_each_quality_is_granted_as_forge_writes_it_and_ruled(
        self, records, row, fragment, column, allowed,
    ):
        record, attacker, blocker = _gen3_block_pair(records)
        cell, rule = evasion_grid.edited(record, attacker, blocker, row, column)
        lines = cell.state.entity(attacker).lines
        granted = [line.fragment for line in lines if line.granted and line.kind == "keyword"]
        assert len(granted) == 1
        if fragment is not None:
            assert granted == [fragment]
        assert rule is allowed

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
        cell, rule = evasion_grid.edited(record, attacker, blocker,
                                         "protection-type-blocker", None)
        assert rule is True

    def test_the_grid_reads_a_gen3_model(self, records, probe_model):
        from effect_knowledge_probes import readout

        record, attacker, blocker = _gen3_block_pair(records)
        cells = [evasion_grid.edited(record, attacker, blocker, row, None)[0]
                 for row in evasion_grid.PROTECTION_ROWS]
        _probabilities(common.predict(probe_model, cells, readout.at_card("may_block", blocker)))


# ── T165: the toughness sweep on activated "any target" damage ──────────


class TestActivatedDamageSweep:
    def test_any_target_is_read_as_target_creature(self):
        text = ("AB$ DealDamage | Cost$ T | ValidTgts$ Any | NumDmg$ 1 | "
                "SpellDescription$ CARDNAME deals 1 damage to any target.")
        assert sweeps.target_creature_wording(text) == (
            "AB$ DealDamage | Cost$ T | ValidTgts$ Creature | NumDmg$ 1 | "
            "SpellDescription$ CARDNAME deals 1 damage to target creature.")
        assert sweeps.target_creature_wording("Any target.") == "Target creature."
        assert sweeps.target_creature_wording("ValidTgts$ Anything") == "ValidTgts$ Anything"

    def _ping(self, records, sidecars):
        from effects.domain.records import Moment

        for record in records:
            if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
                continue
            line = sidecars.line_or_none(record.ability[0]) if record.ability else None
            if (line is not None and line.line_kind == "activated"
                    and "ValidTgts$ Any" in (line.script_text or "")):
                return record
        raise AssertionError("no activated any-target ping in the fixture")

    def test_the_selector_takes_an_activated_ping_at_a_damaged_creature(
        self, records, sidecars, monkeypatch,
    ):
        record = self._ping(records, sidecars)
        creature = next(e for e in record.state.entities
                        if e.is_creature and e.on_battlefield)
        monkeypatch.setattr(sweeps, "_damaged_creature", lambda r: creature.id)
        selector = sweeps.SweepSelector(sidecars)
        selector.observe(record)
        rows = selector.result()["activated_damage"]
        assert rows == [{"record_id": record.record_id, "entity": creature.id}]

    def test_both_wordings_are_swept_off_a_gen3_model(self, records, sidecars, probe_model):
        record = self._ping(records, sidecars)
        creature = next(e for e in record.state.entities
                        if e.is_creature and e.on_battlefield)
        curve = sweeps.run_sweep("activated_damage", probe_model, record,
                                 {"record_id": record.record_id, "entity": creature.id})
        assert len(curve) == 2 * len(sweeps.TOUGHNESS_VALUES)
        assert {p["wording"] for p in curve} == set(sweeps.WORDINGS)
        _probabilities([p["prediction"] for p in curve] + [p["affected"] for p in curve])
        summary = sweeps.summarize([curve])
        assert [(row["wording"], row["value"]) for row in summary][:2] == [
            ("any target", 1), ("any target", 2)]
        assert all("affected" in row for row in summary)
        # The other sweeps summarize as before.
        assert sweeps.summarize([[{"value": 1, "prediction": 0.5}]]) == [
            {"value": 1, "n": 1, "prediction": 0.5}]


# ── T166: the two ablations on a gen-3 model ────────────────────────────


def _cache_items(probe_model, records):
    """Line items with ``e`` drawn at random: the cache statistics' input."""
    rng = np.random.default_rng(0)
    texts = list(probe_model.batcher.batch_texts(records))
    return [SimpleNamespace(text=text, api=f"api{i % 3}", e=rng.standard_normal(E_DIM))
            for i, text in enumerate(texts)]


class TestAblationsOnGen3:
    def test_ability_slot_noise_runs_over_gen3_records(self, records, probe_model):
        stats = ablation.cache_statistics(_cache_items(probe_model, records))
        result = ablation.run_ablation(probe_model, records[:24], stats, batch_size=8)
        assert result
        for body in result.values():
            assert np.isfinite(body["base_loss"])
            for replacement in ablation.REPLACEMENTS:
                assert set(body[replacement]) == set(ablation.SCOPES)
                assert all(np.isfinite(v) for v in body[replacement].values())
        # The card blocks' ability slots carry e on gen-3, so their noise moves the loss.
        assert any(body["noise"]["cards"] != 0.0 for body in result.values())

    def test_a_gen3_permanent_lists_its_own_unlisted_lines(self, records, sidecars):
        from effects.domain.state_snapshot import CardLine

        entity = next(e for r in records for e in r.state.entities if e.lines)
        lines = (CardLine(keys=(), kind="static", granted=False),
                 CardLine(keys=(), kind="trigger", granted=True),
                 CardLine(keys=(), kind="replacement", granted=False, active=False),
                 CardLine(keys=(), kind="keyword", granted=False, fragment="Flying"))
        assert board.unlisted_kinds(replace(entity, lines=lines), sidecars) == {"static"}
        assert board.unlisted_kinds(replace(entity, lines=()), sidecars) == frozenset()

    def test_the_board_abilities_ablation_runs_over_gen3_subsets(
        self, records, sidecars, probe_model,
    ):
        chosen = {"card-disjoint": board.collect(records, sidecars, per_subset=6)}
        assert any(chosen["card-disjoint"].values())
        stats = ablation.cache_statistics(_cache_items(probe_model, records))
        result = board.run(probe_model, chosen, stats)
        measured = [name for name in board.SUBSETS if result[name].get("ablation")]
        assert measured
        assert "# Board abilities" in board.table(result)


# ── T167: target legality read off ``target_legal`` ─────────────────────


class TestTargetLegality:
    def _decisions(self, records, sidecars):
        chosen = [(r, *found) for r in records
                  if (found := target_restrictions.decision_items(r, sidecars))]
        assert chosen
        return chosen

    def test_every_card_and_player_is_an_item(self, records, sidecars):
        for record, _, items in self._decisions(records, sidecars):
            refs = {ref for ref, _ in items}
            assert {p.id for p in record.state.players} <= refs
            assert {e.id for e in record.state.entities} <= refs
            legal = set(record.payload.candidates[0].legal_targets)
            assert {ref for ref, is_legal in items if is_legal} == legal & refs

    def test_the_field_and_the_gate_are_read_at_cards_and_players(
        self, records, sidecars, probe_model,
    ):
        chosen = self._decisions(records, sidecars)
        readings = target_restrictions.read_items(probe_model, [c[0] for c in chosen])
        assert set(readings) == {"target_legal", "affected"}
        for (record, _, _), reading in zip(chosen, readings["target_legal"]):
            assert {p.id for p in record.state.players} <= set(reading)
            _probabilities(list(reading.values()))
        tables = target_restrictions.breakdowns(chosen, readings)
        cell = next(iter(tables["family"].values()))
        assert len(cell["target_legal"][0]) == len(cell["affected"][0]) > 0

    def test_flips_touch_battlefield_targets_only(self, records, sidecars):
        for flip, entity, record, edited in target_restrictions.flip_items(
                self._decisions(records, sidecars)):
            assert record.state.entity(entity).zone == "battlefield"
            assert edited is not record


def test_decisions_in_the_fixture_name_players_as_legal_targets(records):
    """The fixture holds what the legality probe reads at a player's slot."""
    assert any(set(r.payload.candidates[0].legal_targets) & {p.id for p in r.state.players}
               for r in records if isinstance(r.payload, PlayabilityDecisionPayload)
               and r.payload.candidates and r.payload.candidates[0].legal_targets)
