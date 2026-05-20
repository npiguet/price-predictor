"""Unit tests for picker checkpoint save/load round-trip."""

from __future__ import annotations

import pytest
import torch

from sealed.domain.picker_model import PickerConfig, PickerModel
from sealed.infrastructure.picker_store import PickerStore

EMB_DIM = 34


def _make_model() -> PickerModel:
    return PickerModel(PickerConfig(embedding_dim=EMB_DIM, d_model=EMB_DIM, n_heads=2))


class TestCheckpointRoundTrip:
    def test_save_and_load_all_fields(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = model.config

        store = PickerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=5, best_val_reward=4.2, config=config, path=path,
            train_config={"lr": 3e-4, "epochs": 5},
        )

        loaded = store.load_checkpoint(path)
        assert loaded.epoch == 5
        assert loaded.best_val_reward == pytest.approx(4.2)
        assert loaded.config == config
        assert loaded.model_state_dict
        assert loaded.optimizer_state_dict
        assert loaded.train_config == {"lr": 3e-4, "epochs": 5}

    def test_config_reconstructed_as_dataclass(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        store = PickerStore()
        path = tmp_path / "cfg.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_reward=0.0, config=model.config, path=path,
        )
        loaded = store.load_checkpoint(path)
        assert isinstance(loaded.config, PickerConfig)
        assert loaded.config.embedding_dim == EMB_DIM
        assert loaded.config.d_model == EMB_DIM

    def test_weights_survive_roundtrip(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        store = PickerStore()
        path = tmp_path / "w.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_reward=1.0, config=model.config, path=path,
        )
        loaded = store.load_checkpoint(path)
        model2 = _make_model()
        model2.load_state_dict(loaded.model_state_dict)
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_missing_best_val_reward_defaults_to_neg_inf(self, tmp_path):
        path = tmp_path / "legacy.pt"
        torch.save(
            {
                "model_state_dict": _make_model().state_dict(),
                "optimizer_state_dict": {},
                "epoch": 3,
                "config": {"embedding_dim": EMB_DIM, "d_model": EMB_DIM,
                           "n_layers": 4, "n_heads": 2, "d_ff": 136,
                           "dropout": 0.0},
            },
            path,
        )
        loaded = PickerStore().load_checkpoint(path)
        assert loaded.best_val_reward == float("-inf")


class TestFileNaming:
    def test_latest_and_best_share_schema(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        store = PickerStore()

        latest = tmp_path / "latest.pt"
        best = tmp_path / "best_20260520.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_reward=2.0, config=model.config, path=latest,
            train_config={"lr": 3e-4},
        )
        store.save_checkpoint(
            model, optimizer,
            epoch=2, best_val_reward=3.5, config=model.config, path=best,
            train_config={"lr": 3e-4},
        )
        assert latest.exists()
        assert best.exists()
        loaded_latest = store.load_checkpoint(latest)
        loaded_best = store.load_checkpoint(best)
        assert loaded_latest.epoch == 1
        assert loaded_best.epoch == 2
        assert loaded_latest.best_val_reward == pytest.approx(2.0)
        assert loaded_best.best_val_reward == pytest.approx(3.5)
