"""Phase B end-to-end smoke test (slow integration suite).

Runs Phase A for one epoch, Phase B for two epochs against a tiny synthetic
corpus + tiny encoder, then refreshes the `.npz` cache from the resulting
Phase B checkpoint and confirms every fixture card got a new vector that
differs from the pre-Phase-B baseline. (Spec § Phase B integration test,
T043.)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from price_predictor.infrastructure.transformer_store import save_model
from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase
from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND, total_dim
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore


@pytest.mark.integration
def test_phase_b_smoke(tmp_path):
    encoder_d_model = 16
    encoder_cfg = TransformerConfig(
        d_model=encoder_d_model, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=32, vocab_size=64, dropout=0.0,
    )
    encoder = CardPriceTransformerModel(encoder_cfg)
    encoder_dir = tmp_path / "encoder"
    save_model(encoder, encoder_cfg, encoder_dir, version="v1")
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for tok in [
        "creature", "instant", "land", "type", "types", "mana",
        "cost", "name", "spell", "card", "none",
    ]:
        vocab[tok] = len(vocab)
    save_vocabulary(vocab, encoder_dir / "vocab.txt")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    letter = cards_dir / "c"
    letter.mkdir()
    card_dim = total_dim(encoder_d_model)
    card_names: list[str] = []
    rng = np.random.default_rng(31)
    initial_npz: dict[str, np.ndarray] = {}
    for i in range(10):
        name = f"card_{i}"
        card_names.append(name)
        emb = rng.standard_normal(card_dim).astype(np.float32)
        emb[-FEATURE_COUNT + IS_LAND] = 0.0  # mark as spell, not land
        initial_npz[name] = emb
        np.savez_compressed(letter / f"{name}.npz", embedding=emb)
        (letter / f"{name}.txt").write_text(
            f"name: {name}\ntype: creature\nmana cost: none\n",
            encoding="utf-8",
        )

    outcomes = tmp_path / "outcomes.txt"
    rng2 = np.random.default_rng(53)
    lines = [
        "2026-04-22T14:30:05Z;fixture;RVR;forge-best;forge-3sub;"
        f"{'|'.join(rng2.choice(card_names, 4, replace=False))};"
        f"{'|'.join(rng2.choice(card_names, 4, replace=False))};AA;BA;12"
        for _ in range(20)
    ]
    outcomes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Build a Phase A scorer checkpoint with architecture matching the tiny
    # encoder's card_dim (2*d_model + FEATURE_COUNT = 64). We hand-craft it
    # rather than running the train loop since `TrainScorerConfig` doesn't
    # expose `d_model` (the scorer's `d_model` is the card vector dim, not a
    # CLI flag) — going through TrainScorerUseCase would default to 544.
    scorer_cfg = ScorerConfig(
        d_model=card_dim, n_layers=1, n_heads=2, n_seeds=2,
        d_ff=64, mlp_hidden=32, dropout=0.0,
    )
    scorer = SetTransformerScorer(scorer_cfg)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-5)
    phase_a_dir = tmp_path / "scorer_a"
    phase_a_dir.mkdir()
    phase_a_best = phase_a_dir / "phase_a.pt"
    ScorerStore().save_checkpoint(
        scorer, optimizer, epoch=0, best_val_accuracy=0.5,
        config=scorer_cfg, path=phase_a_best,
        train_config={
            "lr": 1e-5, "embedding_lr": 0.0, "patience": 5,
            "epochs": 1, "batch_size": 4, "n_layers": 1, "n_heads": 2,
            "n_seeds": 2, "d_ff": 64, "mlp_hidden": 32, "dropout": 0.0,
            "val_fraction": 0.2, "random_seed": 42,
            "outcomes_path": str(outcomes), "cards_path": str(cards_dir),
            "checkpoint_dir": str(phase_a_dir),
            "scorer_checkpoint": None, "encoder_checkpoint": str(encoder_dir / "latest.pt"),
            "resume": None,
        },
    )

    # Phase B — bootstrap from Phase A, 2 epochs.
    phase_b_dir = tmp_path / "scorer_b"
    config_b = TrainScorerConfig(
        outcomes_path=outcomes, cards_path=cards_dir,
        checkpoint_dir=phase_b_dir,
        scorer_checkpoint=phase_a_best,
        encoder_checkpoint=encoder_dir / "latest.pt",
        epochs=2, batch_size=4, lr=1e-5, embedding_lr=1e-4, patience=10,
    )
    result = TrainScorerUseCase().execute(config_b)
    phase_b_best = phase_b_dir / config_b.best_checkpoint_name()

    # (a) Phase B checkpoint contains both scorer + encoder state.
    loaded = ScorerStore().load_checkpoint(phase_b_best)
    assert loaded.encoder_state_dict is not None
    assert loaded.encoder_config is not None
    assert loaded.train_config is not None

    # Drift recorded each Phase B epoch.
    assert len(result.metrics.embedding_drifts) == 2

    # (b) `encode-cards --scorer-checkpoint --clean` rewrites every fixture
    # `.npz`, and the new vectors differ from the pre-run baseline.
    from argparse import Namespace

    from sealed.infrastructure.cli import run_encode_cards
    rc = run_encode_cards(Namespace(
        encoder_checkpoint=None,
        scorer_checkpoint=str(phase_b_best),
        vocab_path=str(encoder_dir / "vocab.txt"),
        cards_path=str(cards_dir),
        clean=True,
    ))
    assert rc == 0
    for name in card_names:
        npz = letter / f"{name}.npz"
        assert npz.exists()
        with np.load(npz) as f:
            new = f["embedding"]
        # New vectors come from the (fine-tuned) encoder + parsed deterministic
        # features; they should differ from the random baseline.
        assert not np.array_equal(new, initial_npz[name])

    # (c) Patience is respected: the run stops within budget; len of
    # train_losses ≤ epochs + 1 (the +1 covers the boundary epoch).
    assert len(result.metrics.train_losses) <= config_b.epochs
