"""The sampling mixture and rarity weighting (T041).

Three rules: the mixture renormalizes over the classes present (a stage-one
corpus holds three of eight), rarity weights go as ``effective_games ** -0.5``
capped at 20x the most-observed text's weight, and the two classes with no
acting ability text sample uniformly because rarity has nothing to key on.
"""

from __future__ import annotations

import random

import pytest

from effects.application.train_effect_model import (
    DEFAULT_KIND_MIX,
    RARITY_CAP,
    UNIFORM_CLASSES,
    class_counts,
    effective_games,
    parse_kind_mix,
    plan_batch,
    rarity_weights,
    renormalize_mix,
    sample_weights,
    sampling_class,
)
from effects.domain.effect_model import (
    CLASS_COMBAT,
    CLASS_CONTINUOUS,
    CLASS_PLAYABILITY_DECISION,
    CLASS_PLAYABILITY_LEGALITY,
    CLASS_RESOLUTION_COST,
    CLASS_RESOLUTION_EFFECT,
    CLASS_REWRITE,
    CLASS_TRIGGER,
    SAMPLING_CLASSES,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)
_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


def _record(kind: RecordKind, game_id: str = "g1", **overrides) -> EffectRecord:
    payloads = {
        RecordKind.COMBAT: CombatPayload(),
        RecordKind.CONTINUOUS: ContinuousPayload(),
        RecordKind.TRIGGER: TriggerPayload(event=_EVENT, fired=True),
        RecordKind.REWRITE: RewritePayload(incoming=_EVENT, outgoing=_EVENT),
    }
    defaults: dict = {
        "record_id": f"{game_id}.1", "run_id": "run", "timestamp": "t",
        "game_id": game_id, "kind": kind, "actor_player": "P0",
        "state": _SNAPSHOT,
    }
    if kind is RecordKind.RESOLUTION:
        moment = overrides.pop("moment", Moment.RESOLUTION)
        defaults["moment"] = moment
        defaults["payload"] = (
            ResolutionPayload() if moment is Moment.RESOLUTION
            else ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
            )
        )
    elif kind is RecordKind.PLAYABILITY:
        subkind = overrides.pop("subkind", PlayabilitySubkind.DECISION)
        defaults["subkind"] = subkind
        defaults["payload"] = (
            PlayabilityDecisionPayload()
            if subkind is PlayabilitySubkind.DECISION
            else PlayabilityAttackersPayload()
        )
    else:
        defaults["payload"] = payloads[kind]
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestSamplingClass:
    def test_the_two_resolution_moments_are_different_classes(self):
        assert sampling_class(_record(RecordKind.RESOLUTION)) == (
            CLASS_RESOLUTION_EFFECT
        )
        assert sampling_class(
            _record(RecordKind.RESOLUTION, moment=Moment.ACTIVATION)
        ) == CLASS_RESOLUTION_COST

    def test_the_two_playability_families_are_different_classes(self):
        assert sampling_class(_record(RecordKind.PLAYABILITY)) == (
            CLASS_PLAYABILITY_DECISION
        )
        assert sampling_class(_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
        )) == CLASS_PLAYABILITY_LEGALITY

    def test_the_remaining_kinds_map_one_to_one(self):
        assert sampling_class(_record(RecordKind.COMBAT)) == CLASS_COMBAT
        assert sampling_class(_record(RecordKind.CONTINUOUS)) == CLASS_CONTINUOUS
        assert sampling_class(_record(RecordKind.TRIGGER)) == CLASS_TRIGGER
        assert sampling_class(_record(RecordKind.REWRITE)) == CLASS_REWRITE

    def test_every_class_is_reachable(self):
        assert set(SAMPLING_CLASSES) == set(DEFAULT_KIND_MIX)

    def test_class_counts_tallies_a_corpus(self):
        counts = class_counts([
            _record(RecordKind.COMBAT), _record(RecordKind.COMBAT),
            _record(RecordKind.RESOLUTION),
        ])
        assert counts[CLASS_COMBAT] == 2
        assert counts[CLASS_RESOLUTION_EFFECT] == 1


class TestMixture:
    def test_the_default_mixture_is_the_eight_class_one(self):
        assert sum(DEFAULT_KIND_MIX.values()) == pytest.approx(1.0)
        assert DEFAULT_KIND_MIX[CLASS_RESOLUTION_EFFECT] == 0.30
        assert DEFAULT_KIND_MIX[CLASS_PLAYABILITY_LEGALITY] == 0.05

    def test_it_renormalizes_over_the_classes_present(self):
        """A stage-one corpus has three of eight."""
        stage_one = {CLASS_RESOLUTION_EFFECT, CLASS_RESOLUTION_COST, CLASS_COMBAT}
        mix = renormalize_mix(DEFAULT_KIND_MIX, stage_one)
        assert set(mix) == stage_one
        assert sum(mix.values()) == pytest.approx(1.0)

    def test_renormalizing_keeps_the_relative_shares(self):
        stage_one = {CLASS_RESOLUTION_EFFECT, CLASS_COMBAT}
        mix = renormalize_mix(DEFAULT_KIND_MIX, stage_one)
        assert mix[CLASS_RESOLUTION_EFFECT] / mix[CLASS_COMBAT] == pytest.approx(
            DEFAULT_KIND_MIX[CLASS_RESOLUTION_EFFECT]
            / DEFAULT_KIND_MIX[CLASS_COMBAT]
        )

    def test_a_corpus_with_no_scoring_class_is_rejected(self):
        with pytest.raises(ValueError, match="no sampling class"):
            renormalize_mix(DEFAULT_KIND_MIX, set())

    def test_an_absent_class_takes_no_share(self):
        mix = renormalize_mix(DEFAULT_KIND_MIX, {CLASS_COMBAT})
        assert mix == {CLASS_COMBAT: 1.0}

    def test_the_flag_parses_as_class_equals_share(self):
        mix = parse_kind_mix(f"{CLASS_COMBAT}=0.5,{CLASS_TRIGGER}=0.5")
        assert mix == {CLASS_COMBAT: 0.5, CLASS_TRIGGER: 0.5}

    def test_an_absent_flag_gives_the_default_mixture(self):
        assert parse_kind_mix(None) == DEFAULT_KIND_MIX

    def test_an_unknown_class_is_rejected(self):
        with pytest.raises(ValueError, match="unknown sampling class"):
            parse_kind_mix("resolution-sideways=1.0")


class TestRarityWeighting:
    def test_weight_falls_as_the_inverse_square_root_of_games(self):
        weights = rarity_weights({"rare": 1, "common": 100})
        assert weights["rare"] / weights["common"] == pytest.approx(10.0)

    def test_a_rarer_text_weighs_more(self):
        weights = rarity_weights({"a": 1, "b": 4, "c": 9})
        assert weights["a"] > weights["b"] > weights["c"]

    def test_the_cap_is_twenty_times_the_most_observed_texts_weight(self):
        weights = rarity_weights({"ubiquitous": 10_000, "unique": 1})
        assert weights["unique"] == pytest.approx(
            RARITY_CAP * weights["ubiquitous"]
        )

    def test_below_the_cap_nothing_is_clipped(self):
        weights = rarity_weights({"a": 1, "b": 4})
        assert weights["a"] == pytest.approx(1.0)
        assert weights["b"] == pytest.approx(0.5)

    def test_an_empty_corpus_weighs_nothing(self):
        assert rarity_weights({}) == {}

    def test_effective_games_counts_distinct_games_not_records(self):
        """One long game producing many records is still one game."""
        records = [
            _record(RecordKind.COMBAT, game_id="g1", record_id=f"g1.{i}")
            for i in range(50)
        ] + [_record(RecordKind.COMBAT, game_id="g2")]
        assert effective_games(records, lambda r: "bolt") == {"bolt": 2}

    def test_a_record_with_no_text_is_not_counted(self):
        records = [_record(RecordKind.COMBAT)]
        assert effective_games(records, lambda r: None) == {}


class TestUniformClasses:
    def test_combat_and_playability_legality_sample_uniformly(self):
        assert UNIFORM_CLASSES == {CLASS_COMBAT, CLASS_PLAYABILITY_LEGALITY}

    def test_a_record_with_no_acting_text_weighs_one(self):
        records = [
            _record(RecordKind.COMBAT, game_id=f"g{i}", record_id=f"g{i}.1")
            for i in range(5)
        ]
        assert sample_weights(records, lambda r: None) == [1.0] * 5

    def test_records_with_text_are_weighted_by_rarity(self):
        common = [
            _record(RecordKind.RESOLUTION, game_id=f"g{i}", record_id=f"g{i}.1")
            for i in range(4)
        ]
        rare = [_record(RecordKind.RESOLUTION, game_id="g9", record_id="g9.1")]
        texts = {r.record_id: "common" for r in common} | {"g9.1": "rare"}
        weights = sample_weights(
            [*common, *rare], lambda r: texts[r.record_id],
        )
        assert weights[-1] > weights[0]

    def test_a_decision_records_examples_key_on_the_candidates_text(self):
        """One record, one example per candidate, each keyed on its own text."""
        record = _record(RecordKind.PLAYABILITY)
        assert sampling_class(record) == CLASS_PLAYABILITY_DECISION
        assert CLASS_PLAYABILITY_DECISION not in UNIFORM_CLASSES


class TestSampleWeightsRarityTable:
    """A curated dataset's corpus-wide rarity table, preferred over the
    resident shard's own count (FR-146)."""

    def test_sample_weights_prefer_a_supplied_corpus_wide_rarity_table(self):
        records = [_record(RecordKind.COMBAT, game_id="g1", record_id="r1")]
        # The resident shard shows one game; corpus-wide the text was in a
        # hundred.
        corpus_wide = sample_weights(records, lambda r: "t", rarity={"t": 100})
        shard_only = sample_weights(records, lambda r: "t")
        assert corpus_wide[0] < shard_only[0]

    def test_a_text_the_table_does_not_name_falls_back_to_the_shard(self):
        """A shard collected after the dataset was built still weights sanely."""
        records = [_record(RecordKind.COMBAT, game_id="g1", record_id="r1")]
        assert sample_weights(records, lambda r: "new", rarity={"t": 100}) == (
            sample_weights(records, lambda r: "new")
        )


class TestPoolsKeyByAbilityText:
    """``TrainingLoop._pools`` must key rarity by ability text, not record id.

    A ``record_id`` is unique per record, so keying on it gives
    ``effective_games`` a count of exactly 1 for every key and every record in
    a pool the same weight — rarity weighting silently does nothing. This is
    the live defect task 7 fixes: it fails against the pre-fix ``record_id``
    key and passes once the key is the acting ability's text.
    """

    def test_a_single_game_ability_outweighs_a_many_game_one(self):
        from effects.application.train_effect_model import (
            HeldOutCards,
            TrainEffectModelConfig,
        )
        from effects.application.training_loop import TrainingLoop
        from effects.domain.provenance import ProvenanceKey, SidecarLine
        from tests.unit.effects.application.test_ability_text import (
            FakeSidecarCache,
        )

        common_key = ProvenanceKey("cardsfolder/c/common.txt", 0, "keyword", 0)
        rare_key = ProvenanceKey("cardsfolder/r/rare.txt", 0, "keyword", 0)
        sidecars = FakeSidecarCache(lines={
            common_key: SidecarLine(
                line_index=0, line_kind="keyword", provenance=(common_key,),
                script_text="Common Ability",
            ),
            rare_key: SidecarLine(
                line_index=0, line_kind="keyword", provenance=(rare_key,),
                script_text="Rare Ability",
            ),
        })
        # Carried by fifty games...
        common_records = [
            _record(RecordKind.RESOLUTION, game_id=f"g{i}", ability=(common_key,))
            for i in range(50)
        ]
        # ...against one carried by a single game.
        rare_records = [
            _record(RecordKind.RESOLUTION, game_id="g-rare", ability=(rare_key,)),
        ]
        loop = TrainingLoop(
            TrainEffectModelConfig(),
            held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
            inherited=None, validation_shards=[], training_shards=[],
        )

        _, weights = loop._pools([*common_records, *rare_records], sidecars)

        class_weights = weights[CLASS_RESOLUTION_EFFECT]
        assert class_weights[-1] > class_weights[0]


class TestBatchPlanning:
    def _pools(self):
        return {
            CLASS_RESOLUTION_EFFECT: [
                _record(RecordKind.RESOLUTION, game_id=f"g{i}", record_id=f"g{i}.1")
                for i in range(4)
            ],
            CLASS_COMBAT: [
                _record(RecordKind.COMBAT, game_id=f"h{i}", record_id=f"h{i}.1")
                for i in range(4)
            ],
        }

    def test_a_batch_draws_the_requested_number_of_records(self):
        pools = self._pools()
        mix = renormalize_mix(DEFAULT_KIND_MIX, pools)
        plan = plan_batch(pools, {}, mix, batch_size=16, rng=random.Random(0))
        assert len(plan.records) == 16

    def test_records_are_grouped_by_the_game_they_came_from(self):
        """Grouping is what lets one encode of a text serve many records."""
        pools = self._pools()
        mix = renormalize_mix(DEFAULT_KIND_MIX, pools)
        plan = plan_batch(pools, {}, mix, batch_size=16, rng=random.Random(0))
        for game_id, group in plan.by_game.items():
            assert all(record.game_id == game_id for record in group)

    def test_the_mixture_steers_which_classes_are_drawn(self):
        pools = self._pools()
        heavy = plan_batch(
            pools, {}, {CLASS_COMBAT: 1.0}, batch_size=20, rng=random.Random(0),
        )
        assert {r.kind for r in heavy.records} == {RecordKind.COMBAT}

    def test_an_empty_class_is_skipped_rather_than_stalling_the_draw(self):
        pools = {CLASS_COMBAT: self._pools()[CLASS_COMBAT], CLASS_TRIGGER: []}
        plan = plan_batch(
            pools, {}, {CLASS_COMBAT: 0.5, CLASS_TRIGGER: 0.5},
            batch_size=10, rng=random.Random(0),
        )
        assert len(plan.records) == 10

    def test_an_empty_mixture_draws_nothing(self):
        assert plan_batch({}, {}, {}, batch_size=4, rng=random.Random(0)).records == []

    def test_a_batch_is_reproducible_under_a_seeded_rng(self):
        pools = self._pools()
        mix = renormalize_mix(DEFAULT_KIND_MIX, pools)
        first = plan_batch(pools, {}, mix, batch_size=12, rng=random.Random(5))
        second = plan_batch(pools, {}, mix, batch_size=12, rng=random.Random(5))
        assert [r.record_id for r in first.records] == [
            r.record_id for r in second.records
        ]

    def test_unique_ability_texts_deduplicates_across_the_batch(self):
        pools = self._pools()
        mix = renormalize_mix(DEFAULT_KIND_MIX, pools)
        plan = plan_batch(pools, {}, mix, batch_size=16, rng=random.Random(0))
        assert plan.unique_ability_texts(lambda r: "bolt") == {"bolt"}
