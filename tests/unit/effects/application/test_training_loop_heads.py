"""The loss terms beside the per-entity loss, wired into training (T095, T086).

Feature 023 defined the verdict, created-objects, MLM and script-API heads and
called none of their losses, so a decision record's ``[ACT]`` read a zero ``e``
and the verdict head trained on nothing. These tests pin what each record kind
now feeds, on real gen-3 records, and that a head whose weight is zero is not
computed at all.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from effects.application.mlm_masking import MlmMasker
from effects.application.run_context import build_run, training_settings
from effects.application.shard_plan import NO_TEXT_BUCKET
from effects.application.step_loss import StepLoss
from effects.application.step_preparation import training_heads
from effects.application.surface_batching import EncodedBatch, PreparedBatch
from effects.application.text_staging import StagedTexts
from effects.application.train_effect_model import (
    CorpusFacts,
    RunSpec,
    TrainEffectModelConfig,
)
from effects.domain.aux_head_targets import (
    created_objects_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.aux_heads import Head
from effects.domain.effect_losses import LossParts
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLine
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.records import (
    ActivationPayload,
    Moment,
    PlayabilityDecisionPayload,
    RecordKind,
    TriggerPayload,
    events_of,
)
from effects.domain.value_targets import ValueLine
from tests.unit.effects.builders import (
    GEN3_SIDECAR_FOLDERS,
    curated_fixture_records,
    make_cpu_training_loop,
    make_preparer,
)


@pytest.fixture(scope="module")
def records():
    return curated_fixture_records()


def _model(**overrides) -> EffectModel:
    torch.manual_seed(0)
    config = dict(
        global_features=4, act_features=4, player_features=4, card_features=4,
        ability_features=4, e_dim=4, vocab_size=16, n_api_types=3, n_param_keys=5,
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
            staged=StagedTexts(
                rows={text: row for row, text in enumerate(texts)},
                prepared=[object()] * len(texts), encoder_inputs=None, taxonomy_vectors=None,
            ),
            surfaces=[], inputs=None,
            value_lines=tuple(ValueLine.of(text, self.encoded.lines.get(text)) for text in texts),
        )

    def draw_mlm_mask(self, inputs, mask_prob):
        self.mlm_calls += 1
        return SimpleNamespace(rows=len(self.encoded.texts))


def _scripted(*scripts: str) -> _Batcher:
    """A batcher over texts that are their own lines' scripts, so the value head
    reads each as a script (a text with no line states nothing)."""
    lines = {
        script: SidecarLine(
            line_index=0, line_kind="spell", provenance=(), script_text=script, prose_offset=0,
        )
        for script in scripts
    }
    return _Batcher(list(scripts), lines)


@dataclass(frozen=True)
class _Rig:
    """What computes the terms: the patcher, the preparer and the losses."""

    monkeypatch: pytest.MonkeyPatch
    preparer: object
    losses: object


@dataclass(frozen=True)
class _StepInputs:
    """What the terms are computed over."""

    records: list
    batcher: _Batcher
    model: EffectModel
    hidden: torch.Tensor


def _terms(rig: _Rig, step: _StepInputs, *, training):
    """Both halves of the head terms: targets on the host, losses after.

    The MLM term is computed after the main backward in training, so the step
    is finished here the way the loop finishes it; the MLM pass's encoder
    forward is stood in for by the token outputs it would read.
    """
    preparer, losses, monkeypatch = rig.preparer, rig.losses, rig.monkeypatch
    records, batcher, model, hidden = step.records, step.batcher, step.model, step.hidden
    targets = preparer._head_targets(
        records, preparer.rule_mask(records), batcher, batcher.prepared(),
        heads=(
            training_heads(preparer.config, api_types=preparer.api_types)
            if training else frozenset()
        ),
    )
    terms = losses.head_terms(targets, batcher.encoded, model, hidden)
    rows = len(batcher.encoded.texts)
    monkeypatch.setattr(MlmMasker, "forward", staticmethod(
        lambda encoder, masked, device: (
            torch.randn(rows, 8), torch.zeros(rows, dtype=torch.long),
        ),
    ))
    loss = StepLoss(
        total=torch.zeros(()), shipped=torch.zeros(()), parts=LossParts(), terms=terms,
        pending_mlm=targets.get(Head.MLM),
    )
    losses.mlm_backward(loss, None, model)
    return loss.terms


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
        from effects.domain.effect_head_layout import VERDICT_BITS, VERDICT_LAYOUT

        target, mask = verdict_targets(record)
        candidate = record.payload.candidates[0]
        assert target[:3] == [float(candidate.can_play), float(candidate.affordable),
                              float(candidate.has_legal_target)]
        supervised = {str(spec.name) for spec, on in zip(VERDICT_LAYOUT.fields, mask) if on}
        assert set(VERDICT_BITS) <= supervised <= {
            *VERDICT_BITS, "timing", "failed_checks", "cost_adjustment"}

    def test_a_cost_half_yields_the_mana_it_paid(self, records):
        from effects.domain.effect_head_layout import VERDICT_LAYOUT
        from effects.domain.state_snapshot import COLORS

        record = next(r for r in records if isinstance(r.payload, ActivationPayload))
        target, mask = verdict_targets(record)
        paid = record.payload.costs.mana_by_color
        start, end = VERDICT_LAYOUT.target_slices["mana_paid"]
        assert target[start:end] == [float(paid.get(c, 0)) for c in COLORS]
        supervised = {str(spec.name) for spec, on in zip(VERDICT_LAYOUT.fields, mask) if on}
        # The resolution outcome is the effect half's (FR-052).
        assert supervised == {"mana_paid"}

    def test_a_trigger_record_yields_its_fired_bit(self, records):
        for fired in (True, False):
            record = next(
                r for r in records
                if isinstance(r.payload, TriggerPayload) and r.payload.fired is fired
            )
            from effects.domain.effect_head_layout import VERDICT_LAYOUT

            target, mask = verdict_targets(record)
            column, _ = VERDICT_LAYOUT.target_slices["trigger_fired"]
            assert target[column] == float(fired) and sum(mask) == 1
            assert mask[VERDICT_LAYOUT.index["trigger_fired"]]

    def test_combat_supervises_neither_head(self, records):
        record = next(r for r in records if r.kind is RecordKind.COMBAT)
        assert verdict_targets(record) is None
        assert not supervises_created_objects(record)

    def test_an_effect_half_that_makes_a_token_fills_a_created_slot(self, records):
        made = [
            r for r in records if supervises_created_objects(r)
            and any(e.type is EventType.TOKEN_CREATED for e in events_of(r))
        ]
        assert made, "the fixture holds no token-making effect half"
        target, known = created_objects_targets(made[0])
        assert target[0] == 1.0 and target[2] >= 1.0
        assert known[0] and known[2]


class TestTheHeadTerms:
    def test_every_head_contributes_on_a_mixed_batch(self, monkeypatch, records, tmp_path):
        """Spec Story 5 scenario 9: decision, cost half, trigger, effect half."""
        losses = make_cpu_training_loop(monkeypatch).losses
        preparer = make_preparer(
            tmp_path, api_types=["DealDamage", "Draw", "Pump"], param_keys=list("ABCDE"),
        )
        batch = _kinds(records)
        model = _model()
        hidden = torch.randn(len(batch), 6, 8)
        batcher = _scripted("NumDmg$ 2 | SP$ DealDamage", "SP$ Draw | NumCards$ 1")
        terms = _terms(
            _Rig(monkeypatch, preparer, losses),
            _StepInputs(batch, batcher, model, hidden), training=True,
        )
        assert {"verdict", "value", "api", "mlm"} <= set(terms)
        assert all(torch.isfinite(value) for value in terms.values())
        assert terms["verdict"] > 0

    def test_zero_weights_compute_neither_auxiliary(self, monkeypatch, records, tmp_path):
        losses = make_cpu_training_loop(monkeypatch, mlm_weight=0.0, api_weight=0.0).losses
        preparer = make_preparer(
            tmp_path, mlm_weight=0.0, api_weight=0.0, api_types=["Draw"],
        )
        batch = _kinds(records)
        batcher = _scripted("SP$ Draw | NumCards$ 1")
        terms = _terms(
            _Rig(monkeypatch, preparer, losses),
            _StepInputs(batch, batcher, _model(), torch.randn(len(batch), 6, 8)),
            training=True,
        )
        assert "mlm" not in terms and "api" not in terms
        assert batcher.mlm_calls == 0

    def test_scoring_computes_only_the_shipped_heads(self, monkeypatch, records, tmp_path):
        losses = make_cpu_training_loop(monkeypatch).losses
        preparer = make_preparer(tmp_path, api_types=["Draw"])
        batch = _kinds(records)
        terms = _terms(
            _Rig(monkeypatch, preparer, losses),
            _StepInputs(
                batch, _scripted("SP$ Draw"), _model(), torch.randn(len(batch), 6, 8),
            ),
            training=False,
        )
        assert set(terms) <= {"verdict", "created_objects", "global"}


class TestTheApiTargets:
    """The script-API head's targets of a text are its batch's line's, whatever
    the preparer prepared before: a run is the same run under any worker count."""

    _TEXT = "Protection from white"

    @staticmethod
    def _line():
        from effects.domain.provenance import ProvenanceKey, SidecarLine

        return SidecarLine(
            line_index=3, line_kind="static",
            provenance=(ProvenanceKey("cardsfolder/w/white_knight.txt", 0, "keyword", 1),),
            script_api_type="Keyword", script_text="Protection from white", prose_offset=0)

    def test_a_text_a_batch_holds_no_line_for_has_no_target_either_way(self, tmp_path):
        """A granted keyword's text, its line in one batch and none in
        another: the lineless batch reads no type, first or second."""
        lined = {self._TEXT: self._line()}
        for name in ("a", "b"):
            (tmp_path / name).mkdir()
        first = make_preparer(tmp_path / "a", api_types=["Keyword"])
        first._api_batch([self._TEXT], lined)
        after = first._api_batch([self._TEXT], {}).types
        fresh = make_preparer(tmp_path / "b", api_types=["Keyword"])._api_batch(
            [self._TEXT], {}).types
        assert after.tolist() == fresh.tolist() == [-1]

    def test_a_lineless_batch_first_leaves_the_line_s_target_whole(self, tmp_path):
        preparer = make_preparer(tmp_path, api_types=["Keyword"])
        preparer._api_batch([self._TEXT], {})
        types = preparer._api_batch([self._TEXT], {self._TEXT: self._line()}).types
        assert types.tolist() == [0]


def _fixture_preparer(directory):
    """A preparer over the gen-3 fixture's sidecars, on their script surface."""
    return make_preparer(directory, cards_folders=GEN3_SIDECAR_FOLDERS)


class TestEpochShares:
    """FR-055: the share of trained records per rarity bucket and family."""

    def test_every_record_gets_a_bucket_and_a_family(self, records, tmp_path):
        preparer = _fixture_preparer(tmp_path)
        texts = preparer._texts(records)
        labels = preparer._shard_plan(records, texts, preparer._games(records, texts)).labels
        assert set(labels) == {r.record_id for r in records}
        buckets = {bucket for bucket, _ in labels.values()}
        assert buckets <= {*RARITY_BUCKETS, NO_TEXT_BUCKET}
        assert NO_TEXT_BUCKET in buckets  # combat and legality have no text
        combat = next(r for r in records if r.kind is RecordKind.COMBAT)
        assert labels[combat.record_id][0] == NO_TEXT_BUCKET

    def test_a_texts_bucket_reads_the_corpus_wide_table(self, records, tmp_path):
        preparer = _fixture_preparer(tmp_path)
        record = next(
            r for r in records if r.moment is Moment.RESOLUTION
            and preparer.resolver.acting_text(r) is not None
        )
        preparer.rarity = {preparer.resolver.acting_text(record): 50}
        texts = preparer._texts([record])
        labels = preparer._shard_plan([record], texts, preparer._games([record], texts)).labels
        assert labels[record.record_id][0] == "20+"


def _built_run(monkeypatch, **config):
    """The run a loop builds over a stand-in preparer, at the smallest encoder."""
    from effects.domain.input_layout import SlotKind

    loop = make_cpu_training_loop(
        monkeypatch, encoder_layers=1, encoder_d_model=8, **config,
    )
    return build_run(loop.spec, SimpleNamespace(
        tokenizer=SimpleNamespace(vocab_size=8),
        surface_spec=SimpleNamespace(widths=dict.fromkeys(SlotKind, 4)),
        api_types=[], param_keys=[],
    ), device=loop.device)


class TestTheBatchersTheLoopBuilds:
    def test_only_a_training_batch_carries_noise_and_sampled_expansion(
        self, monkeypatch, tmp_path,
    ):
        """FR-056 (training only) and FR-012 (zero outside training)."""
        from effects.domain.ability_tokenizer import INFERENCE_KEYWORD_EXPAND_P

        run = _built_run(monkeypatch, e_noise=0.2)
        assert run.training.noise is run.noise and run.noise.ratio == 0.2
        assert run.scoring.noise is None
        preparer = make_preparer(tmp_path, keyword_expand_p=0.25)
        assert preparer.training_batcher(random.Random(0)).stager.keyword_expand_p == 0.25
        assert preparer.scoring_batcher().stager.keyword_expand_p == (
            INFERENCE_KEYWORD_EXPAND_P
        ) == 0.0

    def test_the_noise_ramps_over_one_epoch_of_steps(self, monkeypatch):
        assert _built_run(monkeypatch, steps_per_epoch=40).noise.ramp_steps == 40

    def test_the_checkpoint_records_the_sweep_flags(self, monkeypatch):
        settings = training_settings(
            RunSpec(
                TrainEffectModelConfig.unpathed(corpus="unused", e_noise=0.05, value_weight=0.1),
                CorpusFacts(), seed=7,
            ),
            api_types=["Draw"], param_keys=["NumCards"],
        )
        assert settings["e_noise"] == 0.05 and settings["value_weight"] == 0.1
        assert settings["api_types"] == ["Draw"]
        # Every new run reads the gen-3 script rules and says so, so a loader
        # never confuses it with a gen-1 checkpoint that recorded none.
        assert settings["tokenizer_rules"] == "gen-3"
        # Likewise the input layout (FR-045): a loader refuses a checkpoint
        # recording any other.
        assert settings["input_layout"] == "gen-3"

    def test_the_recorded_settings_keep_their_keys_and_order(self):
        """A checkpoint's ``training_settings`` is part of its bytes, and
        ``lethality_copies`` appears only when set."""
        def keys(**flags):
            return list(training_settings(
                RunSpec(
                    TrainEffectModelConfig.unpathed(corpus="unused", **flags),
                    CorpusFacts(), seed=7,
                ),
                api_types=[], param_keys=[],
            ))

        from effects.domain.checkpoint_layout import LAYOUT_VOCABULARIES

        before = [
            "settings_shape", "variant_scripts", "seed", "variant", "e_dim",
            "relation_values", "rule_mask", "deviance_floor", "ema_epochs",
            "stratum_weight", "e_noise", "value_weight", "encoder_layers",
            "encoder_d_model", "keyword_expand_p", "context_dropout",
        ]
        after = [
            "lr_scale", "mlm_weight", "mlm_mask_prob", "api_weight", "curriculum_epoch",
            "curriculum_rewarm_steps", "batch_size", "grad_accum", "steps_per_epoch",
            "shards_per_epoch", "epochs", "patience", "withhold_keyword",
            "corpus_digest", "selection_set_digest", "corpus_shards_digest",
            "api_types", "param_keys", "tokenizer_rules", "input_layout",
            "relation_target_indices", *(f"{name}_digest" for name, _ in LAYOUT_VOCABULARIES),
        ]
        assert keys() == [*before, *after]
        assert keys(lethality_copies=2) == [*before, "lethality_copies", *after]

    def test_the_checkpoint_records_the_seed_the_run_drew(self, monkeypatch):
        """A run with no --seed is reproducible only through the one it drew."""
        run = _built_run(monkeypatch)
        assert run.training_settings["seed"] == run.spec.seed
