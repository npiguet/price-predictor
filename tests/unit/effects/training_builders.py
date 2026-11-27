"""Training loops for the trainer's tests: over a described corpus, on the
CPU, with the losses, the validator or what the loop builds beneath itself
stood in for, and a short real run over the gen-3 fixture."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from tests.unit.effects.builders import (
    FIXTURE_KEYWORD_DEFINITIONS,
    FIXTURE_TRAINING_SEED,
    FIXTURE_TRAINING_SHARDS,
    GEN3_SIDECAR_FOLDERS,
)
from tests.unit.effects.selection_builders import control_score_of


@dataclass(frozen=True)
class LoopCorpus:
    """The corpus files :func:`make_training_loop` hands its loop."""

    training_shards: Iterable[Path] = ()
    #: Each validation stratum's sample shard; None for none.
    validation_samples: Mapping[str, Path] | None = None
    #: The control score's ``FrozenSelection`` (:func:`fixture_selection`); a
    #: loop without one runs only under :func:`stub_validator`.
    selection: object = None


def make_training_loop(config, corpus: LoopCorpus = LoopCorpus(), **facts):
    """A ``TrainingLoop`` over ``config`` and a corpus described by ``facts``.

    ``facts`` override ``CorpusFacts`` fields; by default the fixture holdout
    rule (20 permille, at most 8 carriers) and an empty rarity table.
    """
    from effects.application.train_effect_model import CorpusFacts
    from effects.application.training_loop import CorpusInputs, TrainingLoop
    from effects.domain.split import HoldoutRule
    from effects.infrastructure.effect_model_store import tokenizer_hashes

    return TrainingLoop(config, corpus=CorpusInputs(
        facts=CorpusFacts(**{
            "holdout": HoldoutRule(), "rarity": {}, **facts,
        }),
        training_shards=list(corpus.training_shards),
        validation_samples=corpus.validation_samples or {},
        selection=corpus.selection,
        tokenizer_hashes=tokenizer_hashes(config.tokenizer_files),
    ))


def make_cpu_training_loop(monkeypatch, **config):
    """A ``TrainingLoop`` over no corpus, on a real ``cpu`` device.

    Forced whatever ``torch.cuda.is_available()`` says: on this box it can
    read ``True`` even under ``CUDA_VISIBLE_DEVICES=""`` (its raw device count
    ignores the mask — the same disagreement ``autocast_enabled`` guards
    against), which would move a test's tensors onto a real GPU by accident.
    """
    import torch

    from effects.application.train_effect_model import TrainEffectModelConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return make_training_loop(TrainEffectModelConfig.unpathed(corpus="unused", **config))


def one_parameter_module():
    """A module with one parameter, so backward and the optimizer have work."""
    import torch

    module = torch.nn.Module()
    module.p = torch.nn.Parameter(torch.ones(2))
    return module


class StubLosses:
    """Stands in for a loop's ``StepLossComputer``: a step's loss is
    ``loss(prepared)``, a ``StepLoss`` or a scalar tensor, and no MLM pass runs."""

    def __init__(self, loss) -> None:
        self.loss = loss
        self.denominators = None

    def use_denominators(self, denominators) -> None:
        self.denominators = denominators

    def device_loss(self, prepared, modules, materializer, request=None):
        from effects.application.step_loss import StepLoss
        from effects.domain.effect_losses import LossParts

        computed = self.loss(prepared)
        if isinstance(computed, StepLoss):
            return computed
        return StepLoss(total=computed, shipped=computed.detach(), parts=LossParts(), terms={})

    def mlm_backward(self, loss, encoder, model) -> None:
        return None


def stub_validator(loop, on_score=None, controls=None) -> None:
    """Swap ``loop``'s validator for one that scores every pass 1.0 without
    running the model, calling ``on_score(fields, margins)`` for each pass,
    and whose control score is ``controls[n]`` at its ``n``-th epoch (the
    last one repeated; 0.0 every epoch by default), with no cache prepared.

    Only ``score`` and the control score are stood in for, so the epoch's own
    choice of which pass feeds the margins and the floor is still the real one.
    """
    from effects.application.validation_pass import ProgressMargins, Validator

    scores = list(controls or [0.0])

    class _Scored(Validator):
        epochs = 0

        def prepare(self, run, selection, *, rule_outputs):
            return None

        def control(self, run, fields):
            value = scores[min(self.epochs, len(scores) - 1)]
            self.epochs += 1
            return control_score_of(value)

        def score(self, records, run, fields, collectors=()):
            if on_score is not None:
                on_score(
                    fields, any(isinstance(c, ProgressMargins) for c in collectors),
                )
            return 1.0

    loop.validator = _Scored(loop.config, loop.device, loop.losses)


def stub_below_the_loop(monkeypatch):
    """Replace what a ``TrainingLoop`` builds beneath itself — the encoder, the
    head, the tokenizer and the script-API vocabulary — so ``execute`` runs
    over records no sidecar describes.

    Returns the ``(model, encoder)`` stand-ins, from :func:`one_parameter_module`.
    """
    import torch

    from effects.application import run_context, step_preparation
    from effects.application import training_loop as loop_module
    from effects.domain.effect_head_layout import DISPERSED_FIELDS

    model, encoder = one_parameter_module(), one_parameter_module()
    # What validation reads off the model for gate 1's deviance margin.
    model.per_entity_dispersion = lambda: torch.zeros(len(DISPERSED_FIELDS))
    monkeypatch.setattr(run_context, "AbilityEncoder", lambda config: encoder)
    monkeypatch.setattr(run_context, "EffectModel", lambda config: model)
    monkeypatch.setattr(
        step_preparation, "load_tokenizer",
        lambda *a, **k: type("T", (), {"vocab_size": 8, "surface": "script"})(),
    )
    # The script-API vocabulary is read off every real sidecar; none here.
    monkeypatch.setattr(run_context, "script_vocabularies", lambda folders: ([], []))
    # No record here is a real one to prepare, so no pre-pass reads them.
    from effects.domain.loss_denominators import LossDenominators

    monkeypatch.setattr(
        loop_module, "loss_denominators",
        lambda *a, floor, **k: LossDenominators.of({}, floor=floor),
    )
    return model, encoder


def fixture_training_config(corpus, out, **overrides):
    """A short real run's config over the application conftest's ``corpus``
    fixture, every draw the host half takes switched on."""
    from effects.application.train_effect_model import TrainEffectModelConfig

    settings = dict(
        corpus=str(corpus.root), cards_folders=GEN3_SIDECAR_FOLDERS,
        keyword_definitions=FIXTURE_KEYWORD_DEFINITIONS, vocab_path=corpus.vocab,
        model_output=out, seed=FIXTURE_TRAINING_SEED, batch_size=8, steps_per_epoch=7,
        shards_per_epoch=FIXTURE_TRAINING_SHARDS, epochs=2, curriculum_epoch=2, patience=5,
        e_dim=16, encoder_layers=1, encoder_d_model=32,
        # Every draw the host half takes, at rates that make each one fire.
        keyword_expand_p=0.5, context_dropout=0.2, mlm_weight=0.1,
        mlm_mask_prob=0.3, value_weight=0.05, api_weight=0.05, e_noise=0.1,
    )
    settings.update(overrides)
    return TrainEffectModelConfig.unpathed(**settings)


def fixture_run(corpus, records, out, **overrides):
    """A real run's context over the application conftest's ``corpus``, on
    the CPU at a tiny trunk: the encoder, model, scoring materializer and
    preparer a training loop builds, trained on ``records``' classes."""
    from dataclasses import replace
    from unittest.mock import patch

    import torch

    from effects.application import run_context
    from effects.application.step_preparation import StepPreparer
    from effects.domain.effect_model import EffectModel
    from effects.domain.sampling import sampling_class

    loop = make_training_loop(
        fixture_training_config(corpus, out, prefetch_workers=0, **overrides),
        trained_classes=frozenset(sampling_class(r) for r in records))
    preparer = StepPreparer(run_context.preparer_settings(loop.spec))
    tiny = lambda config: EffectModel(replace(  # noqa: E731
        config, d_model=32, n_layers=1, n_heads=2, ff_dim=64))
    with patch.object(run_context, "EffectModel", tiny):
        torch.manual_seed(0)
        return run_context.build_run(loop.spec, preparer, device=torch.device("cpu"))
