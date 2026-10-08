"""The loss terms beside the per-entity loss, wired into training (T095, T086).

Feature 023 defined the verdict, created-objects, MLM and script-API heads and
called none of their losses, so a decision record's ``[ACT]`` read a zero ``e``
and the verdict head trained on nothing. These tests pin what each record kind
now feeds, on real gen-1 records, and that a head whose weight is zero is not
computed at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from effects.application.surface_batching import EncodedBatch, PreparedBatch
from effects.application.train_effect_model import HeldOutCards, TrainEffectModelConfig
from effects.application.training_loop import (
    NO_TEXT_BUCKET,
    TrainingLoop,
    format_shares,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.effect_targets import (
    created_objects_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.event_schema import EventType
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.records import (
    ActivationPayload,
    Moment,
    PlayabilityDecisionPayload,
    RecordKind,
    TriggerPayload,
    events_of,
)
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


@pytest.fixture(scope="module")
def records():
    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def sidecars():
    return SidecarCache({
        "cardsfolder": _FIXTURES / "gen1-sidecars" / "cardsfolder",
        "tokenscripts": _FIXTURES / "gen1-sidecars" / "tokenscripts",
    })


def _loop(monkeypatch, **config) -> TrainingLoop:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return TrainingLoop(
        TrainEffectModelConfig(corpus="unused", **config),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )


def _model(**overrides) -> EffectModel:
    torch.manual_seed(0)
    config = dict(
        global_features=4, act_features=4, player_features=4, card_features=4,
        e_dim=4, vocab_size=16, n_api_types=3, n_param_keys=5,
        encoder_d_model=8, d_model=8, n_layers=1, n_heads=2, ff_dim=16,
    )
    config.update(overrides)
    return EffectModel(EffectModelConfig(**config))


class _Batcher:
    """What the two halves of a step read off a batcher: the MLM draw on the
    host, then ``encoded`` and the MLM pass on the device."""

    def __init__(self, texts: list[str], lines: dict) -> None:
        self.encoded = EncodedBatch(
            texts=texts, lines=lines, matrix=torch.randn(len(texts), 4),
        )
        self.mlm_calls = 0

    def prepared(self) -> PreparedBatch:
        """The host half ``build`` would have produced for these texts."""
        texts = self.encoded.texts
        return PreparedBatch(
            texts=dict(self.encoded.lines),
            rows={text: row for row, text in enumerate(texts)},
            prepared=[object()] * len(texts), encoder_inputs=None,
            stand_in=None, surfaces=[], inputs={},
        )

    def draw_mlm_mask(self, prepared, mask_prob):
        self.mlm_calls += 1
        return "masked"

    def mlm_forward(self, encoder, masked):
        hidden = torch.randn(len(self.encoded.texts), 3, 8)
        targets = torch.zeros(len(self.encoded.texts), 3, dtype=torch.long)
        mask = torch.zeros(len(self.encoded.texts), 3, dtype=torch.bool)
        mask[:, 1] = True
        return hidden, targets, mask


def _terms(loop, records, batcher, model, hidden, *, training):
    """Both halves of the head terms: targets on the host, losses after.

    The MLM term is computed after the main backward in training, so the step
    is finished here the way the loop finishes it.
    """
    targets = loop._head_targets(
        records, batcher, batcher.prepared(), training=training,
        heads=loop._training_heads(model) if training else frozenset(),
    )
    loop._last_terms = loop._head_terms(targets, batcher, None, model, hidden)
    loop._mlm_backward(None, model)
    return loop._last_terms


def _kinds(records):
    def first(test):
        return next(r for r in records if test(r))

    return [
        first(lambda r: isinstance(r.payload, PlayabilityDecisionPayload)),
        first(lambda r: r.moment is Moment.ACTIVATION),
        first(lambda r: isinstance(r.payload, TriggerPayload)),
        first(lambda r: r.moment is Moment.RESOLUTION),
    ]


class TestTheTargets:
    """Which record supervises which head (FR-060a)."""

    def test_a_decision_record_yields_its_candidates_verdict_bits(self, records):
        record = next(r for r in records if r.subkind is not None
                      and isinstance(r.payload, PlayabilityDecisionPayload))
        target, mask = verdict_targets(record)
        candidate = record.payload.candidates[0]
        assert target[:3] == [float(candidate.can_play), float(candidate.affordable),
                              float(candidate.has_legal_target)]
        assert mask[:3] == [True] * 3 and not any(mask[3:])

    def test_a_cost_half_yields_the_mana_it_paid(self, records):
        from effects.domain.state_snapshot import COLORS

        record = next(r for r in records if isinstance(r.payload, ActivationPayload))
        target, mask = verdict_targets(record)
        paid = record.payload.costs.mana_by_color
        assert target[3:3 + len(COLORS)] == [float(paid.get(c, 0)) for c in COLORS]
        assert mask[3:3 + len(COLORS)] == [True] * len(COLORS)
        assert not mask[0] and not mask[-1]

    def test_a_trigger_record_yields_its_fired_bit(self, records):
        for fired in (True, False):
            record = next(
                r for r in records
                if isinstance(r.payload, TriggerPayload) and r.payload.fired is fired
            )
            target, mask = verdict_targets(record)
            assert target[-1] == float(fired) and mask[-1] and sum(mask) == 1

    def test_combat_supervises_neither_head(self, records):
        record = next(r for r in records if r.kind is RecordKind.COMBAT)
        assert verdict_targets(record) is None
        assert not supervises_created_objects(record)

    def test_an_effect_half_that_makes_a_token_fills_a_created_slot(self, records):
        made = [
            r for r in records if supervises_created_objects(r)
            and any(e.type is EventType.TOKEN_CREATED for e in events_of(r))
        ]
        if not made:
            pytest.skip("the fixture holds no token-making effect half")
        target = created_objects_targets(made[0])
        assert target[0] == 1.0 and target[2] >= 1.0


class TestTheHeadTerms:
    def test_every_head_contributes_on_a_mixed_batch(self, monkeypatch, records):
        """Spec Story 5 scenario 9: decision, cost half, trigger, effect half."""
        loop = _loop(monkeypatch)
        loop.api_types, loop.param_keys = ["DealDamage", "Draw", "Pump"], list("ABCDE")
        batch = _kinds(records)
        model = _model()
        hidden = torch.randn(len(batch), 6, 8)
        batcher = _Batcher(["NumDmg$ 2 | SP$ DealDamage", "SP$ Draw | NumCards$ 1"], {})
        terms = _terms(loop, batch, batcher, model, hidden, training=True)
        assert {"verdict", "value", "api", "mlm"} <= set(terms)
        assert all(torch.isfinite(value) for value in terms.values())
        assert terms["verdict"] > 0

    def test_zero_weights_compute_neither_auxiliary(self, monkeypatch, records):
        loop = _loop(monkeypatch, mlm_weight=0.0, api_weight=0.0)
        batch = _kinds(records)
        batcher = _Batcher(["SP$ Draw | NumCards$ 1"], {})
        terms = _terms(
            loop, batch, batcher, _model(), torch.randn(len(batch), 6, 8),
            training=True,
        )
        assert "mlm" not in terms and "api" not in terms
        assert batcher.mlm_calls == 0

    def test_scoring_computes_only_the_shipped_heads(self, monkeypatch, records):
        loop = _loop(monkeypatch)
        batch = _kinds(records)
        terms = _terms(
            loop, batch, _Batcher(["SP$ Draw"], {}), _model(),
            torch.randn(len(batch), 6, 8), training=False,
        )
        assert set(terms) <= {"verdict", "created_objects"}


class TestEpochShares:
    """FR-055: the share of trained records per rarity bucket and family."""

    def test_every_record_gets_a_bucket_and_a_family(self, monkeypatch, records, sidecars):
        loop = _loop(monkeypatch, vocab_path=Path("models/effects/vocab-script.txt"))
        labels = loop._shard_labels(records, sidecars)
        assert set(labels) == {r.record_id for r in records}
        buckets = {bucket for bucket, _ in labels.values()}
        assert buckets <= {*RARITY_BUCKETS, NO_TEXT_BUCKET}
        assert NO_TEXT_BUCKET in buckets  # combat and legality have no text
        combat = next(r for r in records if r.kind is RecordKind.COMBAT)
        assert labels[combat.record_id][0] == NO_TEXT_BUCKET

    def test_a_texts_bucket_reads_the_corpus_wide_table(self, monkeypatch, records, sidecars):
        loop = _loop(monkeypatch, vocab_path=Path("models/effects/vocab-script.txt"))
        from effects.application.train_effect_model import ability_text_of

        record = next(
            r for r in records if r.moment is Moment.RESOLUTION
            and ability_text_of(r, sidecars, loop.surface) is not None
        )
        loop.rarity = {ability_text_of(record, sidecars, loop.surface): 50}
        assert loop._shard_labels([record], sidecars)[record.record_id][0] == "20+"

    def test_shares_are_reported_in_bucket_order(self):
        line = format_shares(
            {"20+": 1, "1": 3, NO_TEXT_BUCKET: 4}, (*RARITY_BUCKETS, NO_TEXT_BUCKET),
        )
        assert line == "1 37.5%, 20+ 12.5%, no-text 50.0%"


class TestTheBatchersTheLoopBuilds:
    def test_only_a_training_batcher_carries_noise_and_sampled_expansion(self, monkeypatch):
        """FR-056 (training only) and FR-012 (zero outside training)."""
        from effects.domain.ability_tokenizer import INFERENCE_KEYWORD_EXPAND_P

        loop = _loop(monkeypatch, e_noise=0.2, keyword_expand_p=0.25)
        training = loop._batcher(None, SidecarCache({}), {}, training=True)
        scoring = loop._batcher(None, SidecarCache({}), {}, training=False)
        assert training.noise is loop.noise and loop.noise.ratio == 0.2
        assert scoring.noise is None
        assert training.keyword_expand_p == 0.25
        assert scoring.keyword_expand_p == INFERENCE_KEYWORD_EXPAND_P == 0.0

    def test_the_noise_ramps_over_one_epoch_of_steps(self, monkeypatch):
        loop = _loop(monkeypatch, steps_per_epoch=40)
        assert loop.noise.ramp_steps == 40

    def test_the_checkpoint_records_the_sweep_flags(self, monkeypatch):
        loop = _loop(monkeypatch, e_noise=0.05, value_weight=0.1)
        loop.api_types, loop.param_keys = ["Draw"], ["NumCards"]
        settings = loop._training_settings()
        assert settings["e_noise"] == 0.05 and settings["value_weight"] == 0.1
        assert settings["api_types"] == ["Draw"]
        # Every new run reads the gen-2 script rules and says so, so a loader
        # never confuses it with a gen-1 checkpoint that recorded none.
        assert settings["tokenizer_rules"] == "gen-2"
