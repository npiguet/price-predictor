"""The encoder and the head are clipped apart (FR-095)."""

from __future__ import annotations

import torch

from effects.application.run_context import parameter_groups


def _modules():
    from effects.application.surface_batching import IdentityTable
    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.effect_model import EffectModel, EffectModelConfig

    torch.manual_seed(0)
    encoder = AbilityEncoder(AbilityEncoderConfig(
        vocab_size=16, d_model=8, n_layers=1, n_heads=2))
    model = EffectModel(EffectModelConfig(
        global_features=4, act_features=4, player_features=4, card_features=4,
        ability_features=4, e_dim=4, vocab_size=16, n_api_types=3, n_param_keys=5,
        encoder_d_model=8, d_model=8, n_layers=1, n_heads=2, ff_dim=16))
    return encoder, model, IdentityTable(["a", "b"], 4)


def _decay_by_name(groups, modules):
    """Each parameter's group name and weight decay, keyed ``module.param``."""
    owner = {id(p): f"{label}.{name}" for label, module in zip(
        ("encoder", "head", "identity"), modules, strict=False)
        for name, p in module.named_parameters()}
    return {owner[id(p)]: (g["name"], g["weight_decay"])
            for g in groups for p in g["params"]}


def test_groups_are_named_and_hold_every_parameter_once():
    modules = _modules()
    groups = parameter_groups(*modules)
    assert {g["name"] for g in groups} == {"encoder", "head", "identity"}
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(len(list(m.parameters())) for m in modules)
    assert all(g["params"] for g in groups)


def test_no_decay_reaches_a_bias_a_gain_a_table_or_a_learned_vector():
    """Finding 15.3: decay only on the matrices."""
    modules = _modules()
    found = _decay_by_name(parameter_groups(*modules), modules)
    for name, (_, decay) in found.items():
        tail = name.rsplit(".", 1)[-1]
        no_decay = (
            tail in {"bias", "in_proj_bias", "log_dispersion", "real_face_subtype_offset",
                     "rel_k", "rel_v", "base", "direction"}
            or "embedding" in name or name.startswith("identity.")
            or ".norm" in name
        )
        assert decay == (0.0 if no_decay else 0.01), name
    assert found["head.verdict_dispersion.log_dispersion"][1] == 0.0
    assert found["head.real_face_subtype_offset"][1] == 0.0
    assert found["head.counter_embedding.weight"][1] == 0.0
    assert found["identity.embedding.weight"][1] == 0.0
    assert 0.01 in {decay for _, decay in found.values()}


def test_each_module_keeps_one_name_over_both_its_halves():
    encoder, head, identity = modules = _modules()
    names = [g["name"] for g in parameter_groups(*modules)]
    assert set(names) == {"encoder", "head", "identity"}
    assert names.count("encoder") == names.count("head") == 2
    assert [g["name"] for g in parameter_groups(encoder, head)][-1] == "head"


def test_a_resume_file_with_other_optimizer_groups_is_refused_by_name():
    import pytest

    from effects.application.resume_state import require_optimizer_groups
    from effects.infrastructure.effect_model_store import ResumeMismatchError

    modules = _modules()
    optimizer = torch.optim.AdamW(parameter_groups(*modules))
    saved = optimizer.state_dict()
    require_optimizer_groups(saved, optimizer)
    old = torch.optim.AdamW([
        {"name": "encoder", "params": list(modules[0].parameters())},
        {"name": "head", "params": list(modules[1].parameters())},
    ]).state_dict()
    with pytest.raises(ResumeMismatchError, match="optimizer groups"):
        require_optimizer_groups(old, optimizer)


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
