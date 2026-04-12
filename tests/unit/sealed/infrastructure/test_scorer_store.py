"""Unit tests for scorer checkpoint save/load round-trip."""

from __future__ import annotations

from pathlib import Path

import torch
import pytest

from sealed.infrastructure.scorer_store import ScorerStore
from sealed.domain.scorer_model import SetTransformerScorer


def _make_model():
    return SetTransformerScorer(d_model=544, n_layers=2, n_heads=4, n_seeds=4, d_ff=1088, mlp_hidden=256)


class TestCheckpointRoundTrip:
    def test_save_and_load_all_fields(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = {"d_model": 544, "n_layers": 2, "n_heads": 4, "n_seeds": 4, "d_ff": 1088, "mlp_hidden": 256}

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(model, optimizer, epoch=5, best_val_accuracy=0.42, config=config, path=path)

        loaded = store.load_checkpoint(path)
        assert loaded["epoch"] == 5
        assert loaded["best_val_accuracy"] == pytest.approx(0.42)
        assert loaded["config"] == config
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_model_weights_survive_roundtrip(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = {"d_model": 544}

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(model, optimizer, epoch=1, best_val_accuracy=1.0, config=config, path=path)

        loaded = store.load_checkpoint(path)
        model2 = _make_model()
        model2.load_state_dict(loaded["model_state_dict"])

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            torch.testing.assert_close(p1, p2)

    def test_feat_mean_std_buffers_survive_roundtrip(self, tmp_path):
        model = _make_model()
        model.feat_mean.copy_(torch.full((32,), 3.14))
        model.feat_std.copy_(torch.full((32,), 2.72))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        store = ScorerStore()
        path = tmp_path / "test.pt"
        store.save_checkpoint(model, optimizer, epoch=1, best_val_accuracy=1.0, config={}, path=path)

        loaded = store.load_checkpoint(path)
        model2 = _make_model()
        model2.load_state_dict(loaded["model_state_dict"])

        torch.testing.assert_close(model2.feat_mean, torch.full((32,), 3.14))
        torch.testing.assert_close(model2.feat_std, torch.full((32,), 2.72))


class TestFileNaming:
    def test_latest_and_best_paths(self, tmp_path):
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        store = ScorerStore()

        latest = tmp_path / "latest.pt"
        best = tmp_path / "best.pt"
        store.save_checkpoint(model, optimizer, epoch=1, best_val_accuracy=0.5, config={}, path=latest)
        store.save_checkpoint(model, optimizer, epoch=2, best_val_accuracy=0.8, config={}, path=best)

        assert latest.exists()
        assert best.exists()

        loaded_latest = store.load_checkpoint(latest)
        loaded_best = store.load_checkpoint(best)
        assert loaded_latest["epoch"] == 1
        assert loaded_best["epoch"] == 2
