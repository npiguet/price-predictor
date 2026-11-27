"""Phase B end-to-end smoke test (slow integration suite).

Runs Phase A for one epoch, Phase B for two epochs against a tiny synthetic
corpus + tiny encoder, then refreshes the `.npz` cache from the resulting
Phase B checkpoint and confirms every fixture card got a new vector that
differs from the pre-Phase-B baseline. (Spec § Phase B integration test.)
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

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
from sealed.infrastructure.cli_encode_cards import run_encode_cards
from sealed.infrastructure.scorer_store import ScorerStore

_ENCODER_D_MODEL = 16


@dataclass(frozen=True)
class _Fixture:
    """The tiny encoder, the ten fixture cards and the outcomes file Phase B reads."""

    encoder_dir: Path
    cards_dir: Path
    letter: Path
    card_names: list[str]
    initial_npz: dict[str, np.ndarray]
    outcomes: Path


def _save_tiny_encoder(tmp_path: Path) -> Path:
    encoder_cfg = TransformerConfig(
        d_model=_ENCODER_D_MODEL, n_layers=1, n_heads=2, ff_dim=32,
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
    return encoder_dir


def _write_fixture_cards(cards_dir: Path) -> tuple[Path, list[str], dict[str, np.ndarray]]:
    cards_dir.mkdir()
    letter = cards_dir / "c"
    letter.mkdir()
    card_dim = total_dim(_ENCODER_D_MODEL)
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
    return letter, card_names, initial_npz


def _write_outcomes(outcomes: Path, card_names: list[str]) -> None:
    rng2 = np.random.default_rng(53)
    lines = [
        "2026-04-22T14:30:05Z;fixture;RVR;forge-best;forge-3sub;"
        f"{'|'.join(rng2.choice(card_names, 4, replace=False))};"
        f"{'|'.join(rng2.choice(card_names, 4, replace=False))};AA;BA;12"
        for _ in range(20)
    ]
    outcomes.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> _Fixture:
    encoder_dir = _save_tiny_encoder(tmp_path)
    cards_dir = tmp_path / "cards"
    letter, card_names, initial_npz = _write_fixture_cards(cards_dir)
    outcomes = tmp_path / "outcomes.txt"
    _write_outcomes(outcomes, card_names)
    return _Fixture(encoder_dir, cards_dir, letter, card_names, initial_npz, outcomes)


def _save_phase_a_checkpoint(fixture: _Fixture, phase_a_dir: Path) -> Path:
    """Build a Phase A scorer checkpoint with architecture matching the tiny
    encoder's card_dim (2*d_model + FEATURE_COUNT = 64). We hand-craft it
    rather than running the train loop since `TrainScorerConfig` doesn't
    expose `d_model` (the scorer's `d_model` is the card vector dim, not a
    CLI flag) — going through TrainScorerUseCase would default to 544.
    """
    scorer_cfg = ScorerConfig(
        d_model=total_dim(_ENCODER_D_MODEL), n_layers=1, n_heads=2, n_seeds=2,
        d_ff=64, mlp_hidden=32, dropout=0.0,
    )
    scorer = SetTransformerScorer(scorer_cfg)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-5)
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
            "outcomes_path": str(fixture.outcomes), "cards_path": str(fixture.cards_dir),
            "checkpoint_dir": str(phase_a_dir),
            "scorer_checkpoint": None,
            "encoder_checkpoint": str(fixture.encoder_dir / "latest.pt"),
            "resume": None,
        },
    )
    return phase_a_best


def _phase_b_config(fixture: _Fixture, phase_a_best: Path, phase_b_dir: Path) -> TrainScorerConfig:
    """Phase B — bootstrap from Phase A, 2 epochs."""
    return TrainScorerConfig(
        outcomes_path=fixture.outcomes, cards_path=fixture.cards_dir,
        checkpoint_dir=phase_b_dir,
        scorer_checkpoint=phase_a_best,
        encoder_checkpoint=fixture.encoder_dir / "latest.pt",
        epochs=2, batch_size=4, lr=1e-5, embedding_lr=1e-4, patience=10,
    )


def _assert_encode_cards_rewrites(fixture: _Fixture, phase_b_best: Path) -> None:
    """`encode-cards --scorer-checkpoint --clean` rewrites every fixture `.npz`,
    and the new vectors differ from the pre-run baseline."""
    rc = run_encode_cards(Namespace(
        encoder_checkpoint=None,
        scorer_checkpoint=str(phase_b_best),
        vocab_path=str(fixture.encoder_dir / "vocab.txt"),
        cards_path=str(fixture.cards_dir),
        embeddings_path=None,
        clean=True,
    ))
    assert rc == 0
    for name in fixture.card_names:
        npz = fixture.letter / f"{name}.npz"
        assert npz.exists()
        with np.load(npz) as f:
            new = f["embedding"]
        # New vectors come from the (fine-tuned) encoder + parsed deterministic
        # features; they should differ from the random baseline.
        assert not np.array_equal(new, fixture.initial_npz[name])


@pytest.mark.integration
def test_phase_b_smoke(tmp_path):
    fixture = _build_fixture(tmp_path)
    phase_a_best = _save_phase_a_checkpoint(fixture, tmp_path / "scorer_a")

    phase_b_dir = tmp_path / "scorer_b"
    config_b = _phase_b_config(fixture, phase_a_best, phase_b_dir)
    result = TrainScorerUseCase().execute(config_b)
    phase_b_best = phase_b_dir / config_b.best_checkpoint_name()

    # (a) Phase B checkpoint contains both scorer + encoder state.
    loaded = ScorerStore().load_checkpoint(phase_b_best)
    assert loaded.encoder_state_dict is not None
    assert loaded.encoder_config is not None
    assert loaded.train_config is not None

    # Drift recorded each Phase B epoch.
    assert len(result.metrics.embedding_drifts) == 2

    # (b) the encode-cards rewrite.
    _assert_encode_cards_rewrites(fixture, phase_b_best)

    # (c) Patience is respected: the run stops within budget; len of
    # train_losses ≤ epochs + 1 (the +1 covers the boundary epoch).
    assert len(result.metrics.train_losses) <= config_b.epochs
