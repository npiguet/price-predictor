"""Helpers shared by the ``train-scorer`` unit tests: the training config and
the tiny price-predictor encoder config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from price_predictor.domain.entities import TransformerConfig
from sealed.application.train_scorer import TrainScorerConfig


@dataclass(frozen=True)
class ScorerCorpus:
    """The match-outcomes file and the converted-cards tree a scorer run reads."""

    outcomes_file: Path
    cards_dir: Path


def _config(
    corpus: ScorerCorpus, checkpoint_dir: Path, *, epochs: int = 1, **overrides,
) -> TrainScorerConfig:
    kwargs = {"batch_size": 8, "lr": 1e-3}
    kwargs.update(overrides)
    return TrainScorerConfig(
        outcomes_path=corpus.outcomes_file,
        cards_path=corpus.cards_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        **kwargs,
    )


def _tiny_transformer_config() -> TransformerConfig:
    return TransformerConfig(
        d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=8, vocab_size=32, dropout=0.0,
    )
