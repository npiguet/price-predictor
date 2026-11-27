"""Batchers, step preparers and the targets a loss test feeds them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING

from tests.unit.effects.file_builders import write_vocabulary

if TYPE_CHECKING:
    import torch

    from effects.domain.baselines import BaselineMasks


@dataclass(frozen=True)
class PreparerOptions:
    """What :func:`make_preparer` hands the preparer besides its config."""

    seed: int = 0
    api_types: Sequence[str] = ()
    param_keys: Sequence[str] = ()
    #: The rarity table; None for an empty one.
    rarity: Mapping[str, int] | None = None


def _refuse_option_names(config) -> None:
    taken = sorted(config.keys() & {field.name for field in fields(PreparerOptions)})
    if taken:
        raise TypeError(f"make_preparer: pass {taken} in PreparerOptions, not as config")


def make_preparer(directory, options: PreparerOptions = PreparerOptions(), **config):
    """A ``StepPreparer`` over a vocabulary of the specials alone, for tests
    about how it builds batchers rather than about what they read; ``config``
    overrides ``TrainEffectModelConfig`` fields, except a name
    ``PreparerOptions`` holds: ``seed`` is both, and a config seed passed
    here would silently leave the preparer's at its default."""
    _refuse_option_names(config)
    from effects.application.step_preparation import PreparerSettings, StepPreparer
    from effects.application.train_effect_model import TrainEffectModelConfig
    from effects.domain.split import CorpusSplit

    vocab = write_vocabulary(directory / "vocab-script.txt", "script")
    keywords = Path(directory) / "keyword-definitions.json"
    keywords.write_text("{}", encoding="utf-8")
    settings = dict(
        corpus="unused", vocab_path=vocab, keyword_definitions=keywords,
        cards_folders=(directory / "cardsfolder",),
    )
    settings.update(config)
    return StepPreparer(PreparerSettings(
        config=TrainEffectModelConfig.unpathed(**settings), seed=options.seed,
        split=CorpusSplit(), rarity={} if options.rarity is None else options.rarity,
        api_types=tuple(options.api_types), param_keys=tuple(options.param_keys),
        heads=frozenset(),
    ))


@dataclass(frozen=True, kw_only=True)
class BatcherSettings:
    """Everything :func:`make_batcher` builds a ``SurfaceBatcher`` from, each
    defaulted: no masks are ``BaselineMasks()``, no widths ``{}``, no device
    the CPU, and no resolver a ``KeyResolver`` over ``sidecars``."""

    tokenizer: object = None
    sidecars: object = None
    masks: BaselineMasks | None = None
    surface: str = "script"
    e_dim: int = 4
    widths: Mapping[str, int] | None = None
    device: torch.device | None = None
    withhold_keyword: str | None = None
    keyword_expand_p: float = 0.0
    context_dropout: float = 0.0
    rng: object = None
    identity_table: object = None
    noise: object = None
    truncations: object = None
    resolver: object = None


def make_batcher(settings: BatcherSettings = BatcherSettings()):
    """A ``SurfaceBatcher`` and its collaborators from one flat set of settings."""
    import torch

    from effects.application.ability_text import KeyResolver
    from effects.application.surface_batching import (
        Materializer,
        StepDraws,
        SurfaceBatcher,
        SurfaceSpec,
    )
    from effects.application.text_resolution import TextResolution
    from effects.application.text_staging import TextStager
    from effects.domain.baselines import BaselineMasks

    masks = settings.masks or BaselineMasks()
    stager = TextStager(
        tokenizer=settings.tokenizer, surface=settings.surface,
        withhold_keyword=settings.withhold_keyword,
        keyword_expand_p=settings.keyword_expand_p, truncations=settings.truncations,
    )
    resolver = settings.resolver or KeyResolver(settings.sidecars, settings.surface)
    return SurfaceBatcher(
        text=TextResolution(resolver, stager),
        spec=SurfaceSpec(settings.e_dim, settings.widths or {}, masks),
        materializer=Materializer(
            settings.device or torch.device("cpu"), masks, settings.identity_table,
            settings.noise,
        ),
        draws=StepDraws(settings.context_dropout, settings.rng),
    )


def slots_of_kind(surface, kind) -> tuple:
    """A token surface's slots of one ``SlotKind``, in surface order."""
    return tuple(slot for slot in surface.slots if slot.kind is kind)


def fields_everywhere(values):
    """``EntityFieldTargets`` holding ``values``, each present at every entity."""
    import torch

    from effects.domain.entity_field_targets import EntityFieldTargets

    return EntityFieldTargets(
        dict(values),
        {name: torch.ones(value.shape[:2], dtype=torch.bool) for name, value in values.items()},
    )


def stated_targets(text: str) -> dict[str, float | None]:
    """The value head's targets for ``text`` by name, None where masked."""
    from effects.domain.value_targets import VALUE_TARGET_NAMES, value_targets

    targets = value_targets(text)
    return {
        name: value if stated else None
        for name, value, stated in zip(VALUE_TARGET_NAMES, targets.values, targets.mask)
    }


def make_entity_targets(gate, fields=None, mask=None, index=None):
    """An ``EntityTargetBatch`` for a loss test. A mapping of ``fields`` is
    present everywhere; ``mask`` defaults to every entity real and ``index``
    to column zero, which no loss reads."""
    import torch

    from effects.domain.entity_field_targets import EntityFieldTargets, EntityTargetBatch

    if not isinstance(fields, EntityFieldTargets):
        fields = fields_everywhere(fields or {})
    if mask is None:
        mask = torch.ones_like(gate)
    if index is None:
        index = torch.zeros(gate.shape, dtype=torch.long)
    return EntityTargetBatch(gate, fields, mask, index)
