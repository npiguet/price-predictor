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
