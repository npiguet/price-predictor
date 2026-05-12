"""Integration: train-scorer end-to-end with an attention-pool sealed encoder.

Covers the new path where the encoder produces a `d_model`-wide text vector
(`--pool-mode attn`) instead of the legacy `2 * d_model`:
  - `encode-cards` writes `.npz` of width `d_model + FEATURE_COUNT`,
  - a fresh Phase A `train-scorer` sizes the scorer to that width (not 544),
  - Phase B fine-tuning runs (the encoder's `_encode_and_pool` resolves and
    `EmbeddingTable.set_text_vectors` splices the `d_model`-wide text slice).
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase
from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.infrastructure.cli import run_encode_cards
from sealed.infrastructure.encoder_store import SealedEncoderStore
from sealed.infrastructure.scorer_store import ScorerStore

_ENCODER_D_MODEL = 16
_EXPECTED_EMB_DIM = _ENCODER_D_MODEL + FEATURE_COUNT  # attn pool: pooled_dim == d_model


@pytest.mark.integration
def test_train_scorer_with_attn_sealed_encoder(tmp_path: Path):
    # --- tiny attention-pool sealed encoder ---------------------------------
    encoder_dir = tmp_path / "encoder"
    encoder_dir.mkdir()
    save_vocabulary(
        {tok: i for i, tok in enumerate([
            "[PAD]", "[UNK]", "cardname", "[MASK]", "name", "type", "types",
            "mana", "cost", "creature", "instant", "none",
        ])},
        encoder_dir / "vocab.txt",
    )
    enc_cfg = SealedEncoderConfig(
        vocab_size=16, d_model=_ENCODER_D_MODEL, n_layers=1, n_heads=2,
        ff_dim=32, max_seq_len=32, dropout=0.0, n_pool_queries=2,
        pool_mode="attn",
    )
    torch.manual_seed(7)
    SealedEncoderStore().save_encoder(
        SealedEncoderModel(enc_cfg), enc_cfg, encoder_dir, version="v1",
    )
    encoder_ckpt = encoder_dir / "latest.pt"

    # --- tiny card corpus ---------------------------------------------------
    cards_dir = tmp_path / "cards"
    (cards_dir / "c").mkdir(parents=True)
    card_names = [f"card_{i}" for i in range(10)]
    for name in card_names:
        (cards_dir / "c" / f"{name}.txt").write_text(
            f"name: {name}\ntype: creature\nmana cost: none\n", encoding="utf-8",
        )

    # --- encode-cards: should produce d_model + FEATURE_COUNT wide .npz ------
    rc = run_encode_cards(Namespace(
        encoder_checkpoint=str(encoder_ckpt), scorer_checkpoint=None,
        vocab_path=str(encoder_dir / "vocab.txt"), cards_path=str(cards_dir),
        clean=True,
    ))
    assert rc == 0
    with np.load(cards_dir / "c" / "card_0.npz") as f:
        assert f["embedding"].shape == (_EXPECTED_EMB_DIM,)

    # --- tiny outcomes ------------------------------------------------------
    outcomes = tmp_path / "outcomes.txt"
    rng = np.random.default_rng(5)
    outcomes.write_text("\n".join(
        "2026-05-12T00:00:00Z;fix;RVR;forge-best;forge-3sub;"
        f"{'|'.join(rng.choice(card_names, 4, replace=False))};"
        f"{'|'.join(rng.choice(card_names, 4, replace=False))};AA;BA;9"
        for _ in range(20)
    ) + "\n", encoding="utf-8")

    # --- Phase A: fresh scorer must be sized to the .npz width, not 544 -----
    phase_a_dir = tmp_path / "scorer_a"
    cfg_a = TrainScorerConfig(
        outcomes_path=outcomes, cards_path=cards_dir, checkpoint_dir=phase_a_dir,
        epochs=1, batch_size=4, lr=1e-4, patience=5,
        n_layers=1, n_heads=2, n_seeds=2, d_ff=32, mlp_hidden=16, dropout=0.0,
    )
    result_a = TrainScorerUseCase().execute(cfg_a)
    assert result_a.model.config.d_model == _EXPECTED_EMB_DIM
    assert result_a.model.det_feature_offset == _ENCODER_D_MODEL
    phase_a_best = phase_a_dir / cfg_a.best_checkpoint_name()
    assert phase_a_best.exists()

    # --- Phase B: bootstrap from Phase A with the sealed encoder ------------
    phase_b_dir = tmp_path / "scorer_b"
    cfg_b = TrainScorerConfig(
        outcomes_path=outcomes, cards_path=cards_dir, checkpoint_dir=phase_b_dir,
        scorer_checkpoint=phase_a_best, encoder_checkpoint=encoder_ckpt,
        epochs=1, batch_size=4, lr=1e-5, embedding_lr=1e-4, patience=5,
    )
    result_b = TrainScorerUseCase().execute(cfg_b)
    assert len(result_b.metrics.embedding_drifts) == 1
    loaded = ScorerStore().load_checkpoint(phase_b_dir / cfg_b.best_checkpoint_name())
    assert loaded.encoder_state_dict is not None
    assert loaded.config.d_model == _EXPECTED_EMB_DIM
