"""One training run's state, and how a run is built from its config.

Everything an epoch, the resume file and a checkpoint read lives in one
:class:`RunContext`, so none of them is handed the run piece by piece.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from effects.application.ability_text import known_texts
from effects.application.step_preparation import (
    PreparerSettings,
    StepPreparer,
    training_heads,
)
from effects.application.step_workers import StepWorkerPool
from effects.application.surface_batching import (
    IdentityTable,
    Materializer,
    NoiseState,
)
from effects.application.train_effect_model import (
    LEARNING_RATE,
    SWEEP_FLAGS,
    WEIGHT_DECAY,
    EarlyStopper,
    RunSpec,
    TrainEffectModelConfig,
    fields_for_epoch,
    warmup_steps,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    surface_of,
)
from effects.domain.ability_tokenizer import TOKENIZER_RULES, TOKENIZER_RULES_KEY
from effects.domain.baselines import baseline_masks
from effects.domain.effect_head_input import INPUT_LAYOUT, INPUT_LAYOUT_KEY, SlotKind
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.infrastructure.sidecar_io import script_vocabularies, sidecar_roots

logger = logging.getLogger(__name__)


def parameter_groups(encoder, model, identity_table=None) -> list[dict]:
    """One optimizer group per module, named for ``clip_per_group``.

    Clipped apart (FR-095): the head's gradient norm runs several times the
    encoder's, and a joint clip would shrink the encoder's update by it.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters())},
        {"name": "head", "params": list(model.parameters())},
    ]
    if identity_table is not None:
        groups.append({"name": "identity", "params": list(identity_table.parameters())})
    return groups


@dataclass
class RunContext:
    """What every step of a run trains with, and where the run has got to."""

    spec: RunSpec
    device: torch.device
    encoder_config: AbilityEncoderConfig
    model_config: EffectModelConfig
    encoder: AbilityEncoder
    model: EffectModel
    optimizer: torch.optim.Optimizer
    warmup: int
    #: The host half of the steps this process prepares, and of validation.
    preparer: StepPreparer
    #: The device halves: training's carries the noise on ``e``, scoring's none.
    training: Materializer
    scoring: Materializer
    #: Noise on every training batch's ``e`` (FR-056); never on a scoring
    #: batch.
    noise: NoiseState
    #: The identity baseline's table; None for every other variant.
    identity_table: IdentityTable | None
    #: What the checkpoint and the resume file record of the run (FR-063).
    training_settings: dict
    #: Each epoch's field set, fixed up front so tasks queued for the workers
    #: before an epoch began read the same one.
    fields: Mapping[int, tuple[FieldSpec, ...]]
    stopper: EarlyStopper
    #: The processes preparing the run's steps; None prepares them here.
    pool: StepWorkerPool | None = None
    #: Training steps taken over the whole run.
    step: int = 0


def api_vocabularies(config: TrainEffectModelConfig) -> tuple[list[str], list[str]]:
    """The script-API head's two vocabularies; read only when the head trains,
    since the read is every sidecar of every tree."""
    if config.api_weight <= 0.0:
        return [], []
    folders = list(config.cards_folders)
    if config.variant_scripts:
        folders.append(Path(config.variant_scripts))
    return script_vocabularies(folders)


def preparer_settings(spec: RunSpec, *, widths: dict[SlotKind, int]) -> PreparerSettings:
    """What every preparer of this run is built from, here or in a worker."""
    config, facts = spec.config, spec.facts
    api_types, param_keys = api_vocabularies(config)
    if api_types:
        logger.info(
            "Script-API head over %d API types and %d parameter keys.",
            len(api_types), len(param_keys),
        )
    return PreparerSettings(
        config=config, seed=spec.seed, split=facts.split,
        rarity=facts.rarity, api_types=tuple(api_types),
        param_keys=tuple(param_keys), widths=dict(widths),
        heads=training_heads(config, api_types=api_types),
        legality_mode_counts=facts.legality_mode_counts,
    )


def training_settings(spec: RunSpec, *, api_types, param_keys) -> dict:
    """The flags a sweep arm differs by, the seed, and the script-API
    vocabularies (FR-063). ``lethality_copies`` is written only when set, so a
    run without copies resumes against the settings it always had."""
    config = spec.config
    copies = (
        {"lethality_copies": config.lethality_copies}
        if config.lethality_copies else {}
    )
    return {
        **copies,
        "seed": spec.seed,
        **{name: getattr(config, name) for name in SWEEP_FLAGS},
        "api_types": list(api_types),
        "param_keys": list(param_keys),
        TOKENIZER_RULES_KEY: TOKENIZER_RULES,
        INPUT_LAYOUT_KEY: INPUT_LAYOUT.name,
    }


def build_run(
    spec: RunSpec, preparer: StepPreparer, present: frozenset[str], *,
    device: torch.device,
) -> RunContext:
    """The encoder, the model, the optimizer and both device halves, sized by
    ``preparer``."""
    config = spec.config
    vocab_size = preparer.tokenizer.vocab_size
    widths = preparer.surface_spec.widths
    encoder_config = AbilityEncoderConfig(
        vocab_size=vocab_size,
        e_dim=config.e_dim,
        d_model=config.encoder_d_model,
        n_layers=config.encoder_layers,
    )
    model_config = EffectModelConfig(
        global_features=widths[SlotKind.GLOBAL],
        act_features=widths[SlotKind.ACT],
        player_features=widths[SlotKind.PLAYER],
        card_features=widths[SlotKind.CARD],
        e_dim=config.e_dim,
        vocab_size=vocab_size,
        n_api_types=len(preparer.api_types),
        n_param_keys=len(preparer.param_keys),
        encoder_d_model=encoder_config.d_model,
    )
    encoder = AbilityEncoder(encoder_config).to(device)
    model = EffectModel(model_config).to(device)
    masks = baseline_masks(config.variant)
    # The identity baseline's table replaces the encoder's output, so the
    # optimizer has to see it.
    identity_table = (
        _identity_table(config, device) if masks.identity_embedding else None
    )
    noise = NoiseState(config.e_noise, config.steps_per_epoch)
    return RunContext(
        spec=spec, device=device,
        encoder_config=encoder_config, model_config=model_config,
        encoder=encoder, model=model,
        optimizer=torch.optim.AdamW(
            parameter_groups(encoder, model, identity_table),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        ),
        warmup=warmup_steps(
            epochs=config.epochs, steps_per_epoch=config.steps_per_epoch,
        ),
        preparer=preparer,
        training=Materializer(device, masks, identity_table, noise=noise),
        scoring=Materializer(device, masks, identity_table),
        noise=noise,
        identity_table=identity_table,
        training_settings=training_settings(
            spec, api_types=preparer.api_types, param_keys=preparer.param_keys,
        ),
        fields={
            epoch: fields_for_epoch(config, present=present, epoch=epoch)
            for epoch in range(1, config.epochs + 1)
        },
        stopper=EarlyStopper(config.patience),
    )


def _identity_table(config: TrainEffectModelConfig, device: torch.device) -> IdentityTable:
    """The identity baseline's table, one row per text the sidecars hold."""
    texts = known_texts(
        sidecar_roots(config.cards_folders, config.variant_scripts),
        surface_of(config.vocab_path),
    )
    logger.info("Identity table over %d ability texts.", len(texts))
    return IdentityTable(texts, config.e_dim).to(device)
