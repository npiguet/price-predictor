"""Unit tests for PickerModel, PickerConfig, and the deterministic walk."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND
from sealed.domain.picker_model import PickerConfig, PickerModel, decompose_picks

EMB_DIM = 34  # >= FEATURE_COUNT so the IS_LAND slot is addressable


def _make_model(**overrides) -> PickerModel:
    config = PickerConfig(embedding_dim=EMB_DIM, **overrides)
    model = PickerModel(config)
    model.eval()
    return model


def _land_embedding(width: int = EMB_DIM, is_land: bool = False) -> np.ndarray:
    emb = np.zeros(width, dtype=np.float32)
    emb[-FEATURE_COUNT + IS_LAND] = 1.0 if is_land else 0.0
    return emb


class TestForwardShapes:
    def test_logits_and_aux_shapes(self):
        model = _make_model(d_model=EMB_DIM, n_heads=2)
        cards = torch.randn(2, 10, EMB_DIM)
        mask = torch.ones(2, 10, dtype=torch.bool)
        logits, aux = model(cards, mask)
        assert logits.shape == (2, 10)
        assert aux.shape == (2,)

    def test_projection_inserted_only_on_width_mismatch(self):
        same = _make_model(d_model=EMB_DIM, n_heads=2)
        assert isinstance(same.input_projection, torch.nn.Identity)

        wider = _make_model(d_model=16, n_heads=4)
        assert isinstance(wider.input_projection, torch.nn.Linear)
        assert wider.input_projection.in_features == EMB_DIM
        assert wider.input_projection.out_features == 16
        cards = torch.randn(1, 5, EMB_DIM)
        mask = torch.ones(1, 5, dtype=torch.bool)
        logits, aux = wider(cards, mask)
        assert logits.shape == (1, 5)
        assert aux.shape == (1,)

    def test_aux_meanpool_ignores_padding(self):
        model = _make_model(d_model=EMB_DIM, n_heads=2)
        cards = torch.randn(1, 6, EMB_DIM)
        mask = torch.tensor([[True, True, True, True, False, False]])
        with torch.no_grad():
            _, aux_before = model(cards, mask)
            # Mutate the padded positions only; aux must not change.
            cards[:, 4:, :] = torch.randn(1, 2, EMB_DIM)
            _, aux_after = model(cards, mask)
        torch.testing.assert_close(aux_before, aux_after)

    def test_state_dict_has_both_heads(self):
        model = _make_model(d_model=EMB_DIM, n_heads=2)
        keys = list(model.state_dict().keys())
        assert any(k.startswith("per_card_head") for k in keys)
        assert any(k.startswith("aux_head") for k in keys)


class TestConfigValidation:
    def test_non_divisible_heads_raises(self):
        with pytest.raises(ValueError, match="divisible by n_heads"):
            PickerConfig(embedding_dim=30, d_model=30, n_heads=4)

    def test_d_ff_defaults_to_4x(self):
        config = PickerConfig(embedding_dim=EMB_DIM, d_model=16, n_heads=4)
        assert config.d_ff == 64

    def test_d_model_defaults_to_embedding_dim(self):
        config = PickerConfig(embedding_dim=EMB_DIM, n_heads=2)
        assert config.d_model == EMB_DIM


class TestDeterministicWalk:
    def test_walk_matches_pseudocode(self):
        # 28 cards: lands at ranks 0, 1 (high logits) and 27 (lowest); the rest
        # spells. Logits descend with position so argsort order is 0..27.
        n = 28
        land_positions = {0, 1, 27}
        embeddings = [
            _land_embedding(is_land=(i in land_positions)) for i in range(n)
        ]
        names = [f"card{i}" for i in range(n)]
        logits = torch.tensor([float(n - i) for i in range(n)])

        chosen = decompose_picks(logits, embeddings, names)

        # 23 spells (positions 2..24) + 2 nonbasic lands (positions 0, 1) = 25.
        assert len(chosen) == 25
        assert len(set(chosen)) == 25  # no card picked twice
        spells = [c for c in chosen if c not in {"card0", "card1", "card27"}]
        lands = [c for c in chosen if c in {"card0", "card1", "card27"}]
        assert len(spells) == 23
        assert set(lands) == {"card0", "card1"}  # the bottom land never reached
        # Spells precede lands in the returned list (FR-006 ordering).
        assert chosen[:23] == spells
        assert chosen[23:] == lands
