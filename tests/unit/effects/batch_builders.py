"""Batchers, step preparers and the targets a loss test feeds them."""

from __future__ import annotations

from pathlib import Path

from tests.unit.effects.file_builders import write_vocabulary


def make_preparer(
    directory, *, seed: int = 0, api_types=(), param_keys=(), rarity=None, **config,
):
    """A ``StepPreparer`` over a vocabulary of the specials alone, for tests
    about how it builds batchers rather than about what they read."""
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
        config=TrainEffectModelConfig.unpathed(**settings), seed=seed, split=CorpusSplit(),
        rarity={} if rarity is None else rarity, api_types=tuple(api_types),
        param_keys=tuple(param_keys), heads=frozenset(),
    ))


def make_batcher(
    *, tokenizer=None, sidecars=None, masks=None, surface: str = "script",
    e_dim: int = 4, widths=None, device=None, withhold_keyword=None,
    keyword_expand_p: float = 0.0, context_dropout: float = 0.0, rng=None,
    identity_table=None, noise=None, truncations=None,
    resolver=None,
):
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

    masks = masks or BaselineMasks()
    stager = TextStager(
        tokenizer=tokenizer, surface=surface, withhold_keyword=withhold_keyword,
        keyword_expand_p=keyword_expand_p, truncations=truncations,
    )
    return SurfaceBatcher(
        text=TextResolution(resolver or KeyResolver(sidecars, surface), stager),
        spec=SurfaceSpec(e_dim, widths or {}, masks),
        materializer=Materializer(
            device or torch.device("cpu"), masks, identity_table, noise,
        ),
        draws=StepDraws(context_dropout, rng),
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
