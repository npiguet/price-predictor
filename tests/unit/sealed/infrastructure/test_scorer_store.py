"""Unit tests for scorer checkpoint save/load round-trip."""

from __future__ import annotations

import pytest
import torch

from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore


def _make_model() -> SetTransformerScorer:
    return SetTransformerScorer(ScorerConfig())


class TestCheckpointRoundTrip:
    def test_save_and_load_all_fields(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = ScorerConfig()

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=5, best_val_accuracy=0.42, config=config, path=path,
            train_config={"lr": 1e-3, "epochs": 5},
        )

        loaded = store.load_checkpoint(path)
        assert loaded.epoch == 5
        assert loaded.best_val_accuracy == pytest.approx(0.42)
        assert loaded.config == config
        assert loaded.model_state_dict
        assert loaded.optimizer_state_dict
        assert loaded.train_config == {"lr": 1e-3, "epochs": 5}
        assert loaded.encoder_state_dict is None
        assert loaded.encoder_config is None

    def test_phase_a_omits_encoder_keys(self, tmp_path):
        """Phase A: encoder fields absent in ``LoadedScorerCheckpoint``."""
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        store = ScorerStore()
        path = tmp_path / "phase_a.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_accuracy=0.5, config=ScorerConfig(), path=path,
        )

        loaded = store.load_checkpoint(path)
        assert loaded.encoder_state_dict is None
        assert loaded.encoder_config is None

    def test_phase_b_round_trip(self, tmp_path):
        """Phase B: encoder fields persist (T021)."""
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        store = ScorerStore()
        path = tmp_path / "phase_b.pt"

        encoder_state_dict = {"layer.weight": torch.randn(3, 4)}
        encoder_config = {"d_model": 256, "n_layers": 2}
        train_config = {"lr": 1e-5, "embedding_lr": 1e-7, "patience": 5}

        store.save_checkpoint(
            model, optimizer,
            epoch=2, best_val_accuracy=0.6, config=ScorerConfig(), path=path,
            encoder_state_dict=encoder_state_dict,
            encoder_config=encoder_config,
            train_config=train_config,
        )

        loaded = store.load_checkpoint(path)
        assert loaded.encoder_state_dict is not None
        torch.testing.assert_close(
            loaded.encoder_state_dict["layer.weight"],
            encoder_state_dict["layer.weight"],
        )
        assert loaded.encoder_config == encoder_config
        assert loaded.train_config == train_config

    def test_model_weights_survive_roundtrip(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_accuracy=1.0, config=ScorerConfig(), path=path,
        )

        loaded = store.load_checkpoint(path)
        model2 = _make_model()
        model2.load_state_dict(loaded.model_state_dict)

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_feat_mean_std_buffers_survive_roundtrip(self, tmp_path):
        model = _make_model()
        model.feat_mean.copy_(torch.full((32,), 3.14))
        model.feat_std.copy_(torch.full((32,), 2.72))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_accuracy=1.0, config=ScorerConfig(), path=path,
        )

        loaded = store.load_checkpoint(path)
        model2 = _make_model()
        model2.load_state_dict(loaded.model_state_dict)

        torch.testing.assert_close(model2.feat_mean, torch.full((32,), 3.14))
        torch.testing.assert_close(model2.feat_std, torch.full((32,), 2.72))

    def test_legacy_dict_config_loads_as_dataclass(self, tmp_path):
        """Pre-Phase-6 checkpoints stored config as a plain dict."""
        path = tmp_path / "legacy.pt"
        torch.save(
            {
                "model_state_dict": _make_model().state_dict(),
                "optimizer_state_dict": {},
                "epoch": 7,
                "best_val_accuracy": 0.6,
                "config": {"d_model": 544, "n_layers": 2, "n_heads": 4,
                           "n_seeds": 4, "d_ff": 1088, "mlp_hidden": 256},
            },
            path,
        )
        loaded = ScorerStore().load_checkpoint(path)
        assert isinstance(loaded.config, ScorerConfig)
        assert loaded.config.n_layers == 2


class TestFileNaming:
    def test_latest_and_best_paths(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        store = ScorerStore()

        latest = tmp_path / "latest.pt"
        best = tmp_path / "best.pt"
        store.save_checkpoint(
            model, optimizer,
            epoch=1, best_val_accuracy=0.5, config=ScorerConfig(), path=latest,
        )
        store.save_checkpoint(
            model, optimizer,
            epoch=2, best_val_accuracy=0.8, config=ScorerConfig(), path=best,
        )

        assert latest.exists()
        assert best.exists()

        loaded_latest = store.load_checkpoint(latest)
        loaded_best = store.load_checkpoint(best)
        assert loaded_latest.epoch == 1
        assert loaded_best.epoch == 2
