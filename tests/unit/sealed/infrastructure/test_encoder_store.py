"""Unit tests for ``SealedEncoderStore`` save/load round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.infrastructure.encoder_store import SealedEncoderStore


def _config() -> SealedEncoderConfig:
    return SealedEncoderConfig(
        vocab_size=64,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ff_dim=32,
        max_seq_len=8,
        dropout=0.1,
        n_pool_queries=2,
    )


def _ids_mask(batch: int, seq_len: int, vocab: int):
    ids = torch.randint(low=2, high=vocab, size=(batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    return ids, mask


class TestSaveLoadRoundTrip:
    def test_load_reconstructs_identical_encode_output(self, tmp_path: Path):
        cfg = _config()
        torch.manual_seed(7)
        model = SealedEncoderModel(cfg)
        model.eval()

        store = SealedEncoderStore()
        ckpt = store.save_encoder(model, cfg, tmp_path, version="v1")

        loaded_model, loaded_cfg = store.load_encoder(ckpt)
        loaded_model.eval()
        assert loaded_cfg == cfg

        ids, mask = _ids_mask(2, cfg.max_seq_len, cfg.vocab_size)
        before = model.encode(ids, mask)
        after = loaded_model.encode(ids, mask)
        assert torch.allclose(before, after, atol=1e-6)

    def test_saved_state_dict_excludes_regression_heads_and_mlm_head(
        self, tmp_path: Path,
    ):
        cfg = _config()
        store = SealedEncoderStore()
        # Use a freshly-initialized model so every head module has weights
        # in the live state_dict; the save filter must drop them.
        live = SealedEncoderModel(cfg)
        live_keys = set(live.state_dict().keys())
        assert any(k.startswith("regression_heads.") for k in live_keys)
        assert any(k.startswith("mlm_head.") for k in live_keys)

        store.save_encoder(live, cfg, tmp_path, version="v2")
        payload = torch.load(tmp_path / "v2.pt", weights_only=False)
        saved_keys = list(payload["model_state_dict"].keys())
        assert saved_keys, "saved state dict must not be empty"
        assert all(
            k.startswith("token_encoder.") or k.startswith("card_encoder.")
            for k in saved_keys
        )
        assert not any(k.startswith("regression_heads.") for k in saved_keys)
        assert not any(k.startswith("mlm_head.") for k in saved_keys)

    def test_latest_pt_is_byte_for_byte_copy(self, tmp_path: Path):
        cfg = _config()
        store = SealedEncoderStore()
        target = store.save_encoder(SealedEncoderModel(cfg), cfg, tmp_path, version="v3")
        latest = tmp_path / "latest.pt"
        assert latest.exists()
        assert target.read_bytes() == latest.read_bytes()


class TestLoadStrictness:
    def test_raises_on_extra_regression_head_key(self, tmp_path: Path):
        cfg = _config()
        store = SealedEncoderStore()
        ckpt = store.save_encoder(SealedEncoderModel(cfg), cfg, tmp_path, version="v4")

        # Tamper: re-save with a regression-head key sneaked in.
        from dataclasses import asdict
        raw = torch.load(ckpt, weights_only=False)
        raw["model_state_dict"]["regression_heads.score_play.0.weight"] = (
            torch.zeros(1, 2 * cfg.d_model)
        )
        raw["config"] = asdict(cfg)
        torch.save(raw, ckpt)

        with pytest.raises(RuntimeError, match="non-encoder keys"):
            store.load_encoder(ckpt)

    def test_raises_on_extra_mlm_head_key(self, tmp_path: Path):
        cfg = _config()
        store = SealedEncoderStore()
        ckpt = store.save_encoder(SealedEncoderModel(cfg), cfg, tmp_path, version="v5")
        from dataclasses import asdict
        raw = torch.load(ckpt, weights_only=False)
        raw["model_state_dict"]["mlm_head.weight"] = torch.zeros(
            cfg.vocab_size, cfg.d_model,
        )
        raw["config"] = asdict(cfg)
        torch.save(raw, ckpt)
        with pytest.raises(RuntimeError, match="non-encoder keys"):
            store.load_encoder(ckpt)

    def test_raises_on_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SealedEncoderStore().load_encoder(tmp_path / "nope.pt")
