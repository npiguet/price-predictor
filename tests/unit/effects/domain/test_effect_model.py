"""The effect model: its trunk, its heads and its training-only auxiliaries (T069)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from effects.domain.aux_heads import AUX_HEADS, SHIPPED_HEADS, TRAINING_ONLY_MODULES
from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.surface_collation import TrunkInputs

_CONFIG = EffectModelConfig(
    global_features=8, act_features=6, player_features=10, card_features=12,
    e_dim=4, vocab_size=32, n_api_types=5, n_param_keys=7,
    d_model=16, n_layers=1, n_heads=2, ff_dim=32,
)


@pytest.fixture
def model() -> EffectModel:
    torch.manual_seed(0)
    return EffectModel(_CONFIG)


def _surface(batch: int = 2, slots: int = 6):
    """A padded token surface: GLOBAL, ACT, PLAYER, PLAYER, CARD, ABILITY."""
    kinds = torch.tensor([[
        SlotKind.GLOBAL, SlotKind.ACT, SlotKind.PLAYER, SlotKind.PLAYER,
        SlotKind.CARD, SlotKind.ABILITY,
    ]] * batch)
    features = {
        kind: torch.zeros(batch, slots, width)
        for kind, width in (
            (SlotKind.GLOBAL, _CONFIG.global_features),
            (SlotKind.ACT, _CONFIG.act_features),
            (SlotKind.PLAYER, _CONFIG.player_features),
            (SlotKind.CARD, _CONFIG.card_features),
        )
    }
    return TrunkInputs(
        slot_features=features,
        slot_kinds=kinds,
        positions=torch.tensor([[0, 1, 2, 3, 0, 1]] * batch),
        e_vectors=torch.zeros(batch, slots, _CONFIG.e_dim),
        attention_mask=torch.ones(batch, slots, dtype=torch.long),
    )


class TestTrainingOnlyHeads:
    def test_every_auxiliary_head_names_modules_the_model_has(self, model):
        for spec in AUX_HEADS:
            for name in spec.modules:
                assert hasattr(model, name), (spec.head, name)

    def test_no_shipped_head_is_in_the_filter_list(self):
        shipped = {"per_entity_head", "created_objects_head", "verdict_head"}
        assert not shipped & set(TRAINING_ONLY_MODULES)
        assert {m for spec in SHIPPED_HEADS for m in spec.modules} <= shipped

    def test_the_auxiliaries_are_absent_when_unconfigured(self):
        bare = EffectModel(EffectModelConfig(
            global_features=8, act_features=6, player_features=10,
            card_features=12, e_dim=4, d_model=16, n_layers=1, n_heads=2,
            ff_dim=32,
        ))
        assert bare.mlm_head is None
        assert bare.api_type_head is None


class TestConfig:
    def test_a_config_with_indivisible_heads_is_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            EffectModelConfig(
                global_features=8, act_features=6, player_features=10,
                card_features=12, d_model=10, n_heads=4,
            )


class TestGen2Heads:
    """The value head and the option-kind embedding."""

    def test_the_value_head_reads_e_and_is_training_only(self, model):
        from effects.domain.effect_head_layout import VALUE_WIDTH

        assert model.value_head(torch.randn(3, _CONFIG.e_dim)).shape == (3, VALUE_WIDTH)
        assert "value_head" in TRAINING_ONLY_MODULES

    def test_the_mlm_head_is_sized_from_the_encoders_width(self):
        model = EffectModel(EffectModelConfig(
            global_features=8, act_features=6, player_features=10, card_features=12,
            e_dim=4, vocab_size=32, encoder_d_model=48,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        ))
        assert model.mlm_head.in_features == 48

    def test_the_option_kind_embedding_starts_at_zero(self, model):
        assert torch.count_nonzero(model.option_kind_embedding.weight) == 0

    def test_an_untrained_option_flag_changes_no_output(self, model):
        model.eval()
        batch = _surface()
        flagged = replace(batch, option_kinds=torch.ones_like(batch.slot_kinds))
        with torch.no_grad():
            assert torch.equal(model(batch), model(flagged))


class TestRelationTrunk:
    """T101: the trunk is relation layers; with empty tables the relations change nothing."""

    def test_the_trunk_is_relation_layers_with_one_table_row_per_type(self, model):
        from effects.domain.relation_attention import RelationEncoderLayer
        from effects.domain.relation_types import RELATION_TYPES

        assert len(model.trunk.layers) == _CONFIG.n_layers
        layer = model.trunk.layers[0]
        assert isinstance(layer, RelationEncoderLayer)
        assert layer.rel_k.shape == (_CONFIG.n_heads, len(RELATION_TYPES),
                                     _CONFIG.d_model // _CONFIG.n_heads)

    def test_the_tables_train_with_the_head_group(self, model):
        names = dict(model.named_parameters())
        assert "trunk.layers.0.rel_k" in names and "trunk.layers.0.rel_v" in names

    def test_relations_change_nothing_while_the_tables_are_zero(self, model):
        model.eval()
        surface = _surface()
        related = replace(surface, relations=torch.tensor([[0, 4, 5, 0], [1, 2, 4, 3]]))
        with torch.no_grad():
            assert torch.allclose(model(surface), model(related))

    def test_a_trained_table_makes_the_relation_matter(self, model):
        model.eval()
        surface = _surface()
        related = replace(surface, relations=torch.tensor([[0, 4, 5, 0]]))
        with torch.no_grad():
            model.trunk.layers[0].rel_k.normal_()
            model.trunk.layers[0].rel_v.normal_()
            assert not torch.allclose(model(surface)[0, 4], model(related)[0, 4])
            assert torch.allclose(model(surface)[1], model(related)[1], atol=1e-6)

    def test_relation_values_off_leaves_no_value_parameter(self):
        torch.manual_seed(0)
        off = EffectModel(replace(_CONFIG, relation_values=False))
        assert "trunk.layers.0.rel_v" not in dict(off.named_parameters())
