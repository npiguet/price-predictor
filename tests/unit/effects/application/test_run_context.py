"""The encoder and the head are clipped apart (FR-095)."""

from __future__ import annotations

import torch

from effects.application.run_context import parameter_groups


def test_groups_are_named_and_cover_every_parameter():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    groups = parameter_groups(encoder, head)
    assert [g["name"] for g in groups] == ["encoder", "head"]
    total_params = len(list(encoder.parameters())) + len(list(head.parameters()))
    assert sum(len(g["params"]) for g in groups) == total_params
    identity = torch.nn.Embedding(4, 2)
    assert [g["name"] for g in parameter_groups(encoder, head, identity)][-1] == "identity"




def test_a_run_records_a_digest_of_every_layout_vocabulary():
    """What the run writes is what ``require_input_layout`` reads back
    (finding 11.6): one digest per registered vocabulary."""
    from effects.application.run_context import training_settings
    from effects.application.train_effect_model import (
        CorpusFacts,
        RunSpec,
        TrainEffectModelConfig,
    )
    from effects.domain.checkpoint_layout import LAYOUT_VOCABULARIES

    settings = training_settings(
        RunSpec(TrainEffectModelConfig.unpathed(corpus="unused"), CorpusFacts(), seed=1),
        api_types=[], param_keys=[],
    )
    for name, _ in LAYOUT_VOCABULARIES:
        assert len(settings[f"{name}_digest"]) == 64


def test_a_run_and_a_saved_model_are_laid_out_by_the_layouts_widths():
    """One source for the widths (finding 19.1): the trunk a run builds and
    the batcher a loaded checkpoint runs under read the layout's, never a
    record's."""
    from types import SimpleNamespace

    from effects.application.surface_batching import SurfaceSpec
    from effects.application.train_effect_model import TrainEffectModelConfig
    from effects.domain.slot_widths import layout_widths

    checkpoint = SimpleNamespace(encoder_config=SimpleNamespace(e_dim=4), variant="full")
    training = SurfaceSpec.for_training(TrainEffectModelConfig.unpathed(corpus="unused"))
    assert training.widths == SurfaceSpec.for_saved_model(checkpoint).widths == layout_widths()
