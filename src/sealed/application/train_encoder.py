"""``train-encoder``: train a sealed encoder + nine regression heads + an
MLM head on per-card winnability and an auxiliary mask-prediction
objective.

The pipeline streams ``cards-played.txt`` once to aggregate per-card
counters (primary, ``@play``, and per-color slices), computes nine raw
and shrunk per-card labels per FR-011, splits cards into train/val with
FR-018 stratification, tokenizes the corpus once, then trains the
dual-pool encoder + regression heads + MLM head jointly under FR-017's
weighted MSE + cross-entropy objective and saves only the encoder
weights to its ``--model-output`` run directory.

Per FR-013, aggregation is inline: there is no separate
``aggregate-labels`` subcommand. The human-readable label snapshot goes
to ``TrainEncoderConfig.win_rates_path`` (``--win-rates-output``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from price_predictor.infrastructure.tokenizer_store import load_vocabulary
from sealed.application.encoder_counters import _aggregate, _drop_missing_cards
from sealed.application.encoder_evaluation import (
    _eval_loss,
    _EvalMetrics,
    _format_head_corrs,
)
from sealed.application.encoder_labels import CardLabelMap, _build_label_map
from sealed.application.encoder_mlm import _MASK_TOKEN
from sealed.application.encoder_split import CardSplit, _split_cards
from sealed.application.encoder_training_common import (
    D_MODEL,
    FF_DIM,
    RANDOM_SEED,
    VAL_FRACTION,
    _log,
)
from sealed.application.encoder_training_loop import (
    _BestCheckpoint,
    _EpochTrainMetrics,
    _format_trunk_grads,
    _train_epoch,
)
from sealed.application.encoder_training_setup import (
    _build_training_setup,
    _TrainingSetup,
)
from sealed.application.encoder_win_rates import _write_win_rates
from sealed.infrastructure.cards_played_reader import iter_rows
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.encoder_store import SealedEncoderStore

# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class TrainEncoderConfig:
    cards_played_path: Path
    cards_folder: Path
    vocab_path: Path
    model_output_dir: Path
    #: The per-card label snapshot, a table for inspection that the effects
    #: package's decodability battery also reads (``--win-rates``).
    win_rates_path: Path
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    patience: int = 20
    dropout: float = 0.1
    n_layers: int = 6
    n_heads: int = 4
    n_pool_queries: int = 4
    pool_mode: str = "dual"
    shrinkage_k: float = 20.0
    mlm_weight: float = 0.1
    mlm_mask_prob: float = 0.15

    # d_model / ff_dim default to the FR-022 constants but are tunable via
    # --d-model / --ff-dim for architecture experiments; val_fraction and
    # random_seed stay hardcoded.
    d_model: int = D_MODEL
    ff_dim: int = FF_DIM
    val_fraction: float = VAL_FRACTION
    random_seed: int = RANDOM_SEED


# ── Pre-flight ──────────────────────────────────────────────────────────


class _PreFlightError(RuntimeError):
    """Raised when a pre-flight check fails. Carries an exit code so the
    CLI handler can map directly to ``contracts/cli.md`` § Exit codes.
    """

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _check_vocabulary(config: TrainEncoderConfig) -> None:
    if not config.vocab_path.exists():
        raise _PreFlightError(
            f"Vocabulary file not found: {config.vocab_path}. "
            "Run python -m sealed build-vocab first.",
            exit_code=2,
        )
    vocab = load_vocabulary(config.vocab_path)
    if _MASK_TOKEN not in vocab:
        raise _PreFlightError(
            f"Vocabulary at {config.vocab_path} is missing the reserved "
            f"{_MASK_TOKEN!r} token required by the MLM auxiliary loss "
            "(FR-009a). Re-run python -m sealed build-vocab to refresh.",
            exit_code=2,
        )


def _check_cards_played(config: TrainEncoderConfig) -> None:
    if (
        not config.cards_played_path.exists()
        or config.cards_played_path.stat().st_size == 0
    ):
        raise _PreFlightError(
            f"cards-played.txt missing or empty: {config.cards_played_path}. "
            "Run python -m sealed match-outcomes to collect data first.",
            exit_code=3,
        )


def _check_cards_folder(config: TrainEncoderConfig) -> None:
    if not config.cards_folder.is_dir() or not any(
        config.cards_folder.rglob("*.txt")
    ):
        raise _PreFlightError(
            f"Cards folder is empty or missing: {config.cards_folder}. "
            "Run python -m price_predictor convert to build the corpus.",
            exit_code=4,
        )


def _check_divisibility(config: TrainEncoderConfig) -> None:
    # d_model / n_heads / n_pool_queries divisibility is validated again
    # inside SealedEncoderConfig.__post_init__, but we surface it as exit
    # code 6 at the CLI handler level (the dataclass raises ValueError).
    if config.d_model % config.n_pool_queries != 0:
        raise _PreFlightError(
            f"d_model ({config.d_model}) must be divisible by n_pool_queries "
            f"({config.n_pool_queries})",
            exit_code=6,
        )
    if config.d_model % config.n_heads != 0:
        raise _PreFlightError(
            f"d_model ({config.d_model}) must be divisible by n_heads "
            f"({config.n_heads})",
            exit_code=6,
        )


def _run_preflight(config: TrainEncoderConfig) -> None:
    _check_vocabulary(config)
    _check_cards_played(config)
    _check_cards_folder(config)
    _check_divisibility(config)


# ── Pipeline stages ─────────────────────────────────────────────────────


def _labels_from_corpus(
    config: TrainEncoderConfig, locator: ConvertedCardLocator,
) -> CardLabelMap:
    """Aggregate ``cards-played.txt``, drop cards with no converted text,
    build the label map and write its snapshot.
    """
    _log(f"Reading {config.cards_played_path}")
    counters = _aggregate(iter_rows(config.cards_played_path), locator)
    _log(f"Aggregated counters for {len(counters)} card(s)")

    n_dropped = _drop_missing_cards(counters, locator)
    if n_dropped:
        _log(f"Continuing with {len(counters)} card(s) after dropping {n_dropped}")

    labels = _build_label_map(counters, config.shrinkage_k)
    _log(
        f"Built label map: {len(labels)} card(s) with "
        f"wins_when_in_deck + losses_when_in_deck > 0",
    )

    win_rates_path = Path(config.win_rates_path)
    _write_win_rates(labels, win_rates_path)
    _log(f"Wrote per-card label snapshot to {win_rates_path}")
    return labels


def _card_split(config: TrainEncoderConfig, labels: CardLabelMap) -> CardSplit:
    """Split the labelled cards, refusing a split with an empty side."""
    train_names, val_names = _split_cards(
        labels, val_fraction=config.val_fraction, seed=config.random_seed,
    )
    if not train_names or not val_names:
        raise RuntimeError(
            f"Not enough cards for a train/val split: "
            f"train={len(train_names)}, val={len(val_names)}"
        )
    _log(f"Card-level split: {len(train_names)} train / {len(val_names)} val")
    return CardSplit(train_names, val_names)


def _epoch_line(
    progress: str, train: _EpochTrainMetrics, val: _EvalMetrics, improved: bool,
) -> str:
    """One epoch's loss summary line; ``progress`` reads ``Epoch <e>/<n>``."""
    marker = "*" if improved else " "
    return (
        f"{progress}  "
        f"train_loss={train.total:.4f} "
        f"(reg={train.reg:.4f}, mlm={train.mlm:.4f} ppl={math.exp(train.mlm):.1f})  "
        f"val_loss={val.total:.4f} "
        f"(reg={val.reg:.4f}, mlm={val.mlm:.4f} ppl={val.mlm_perplexity:.1f} "
        f"acc={val.mlm_top1_accuracy * 100:.1f}%)  "
        f"best={marker}"
    )


def _train_epochs(
    config: TrainEncoderConfig, setup: _TrainingSetup,
) -> _BestCheckpoint:
    """Run the epoch loop, stopping after ``patience`` epochs without a
    better val loss, and return the best checkpoint seen.
    """
    model, ctx = setup.model, setup.ctx
    best = _BestCheckpoint()
    epochs_since_best = 0
    for epoch in range(1, config.epochs + 1):
        train = _train_epoch(model, setup.train_loader, setup.stepper, ctx)
        val = _eval_loss(model, setup.val_loader, ctx)
        improved = best.update(model, epoch, val.total)
        _log(_epoch_line(f"Epoch {epoch}/{config.epochs}", train, val, improved))
        _log("  " + _format_head_corrs(val.head_correlations))
        _log("  " + _format_trunk_grads(train, ctx.mlm_weight))
        epochs_since_best = 0 if improved else epochs_since_best + 1
        if not improved and epochs_since_best >= config.patience:
            _log(
                f"Early stop after {epochs_since_best} epochs "
                f"without improvement.",
            )
            break
    return best


def _save_best(
    config: TrainEncoderConfig, setup: _TrainingSetup, best: _BestCheckpoint,
) -> Path:
    """Restore the best epoch's weights, save the encoder, return ``latest.pt``."""
    model = setup.model
    if best.best_state_dict is not None:
        model.load_state_dict(best.best_state_dict)
        _log(
            f"Restored best epoch {best.best_epoch} "
            f"(val_loss={best.best_val_loss:.4f})",
        )

    store = SealedEncoderStore()
    saved = store.save_encoder(model, setup.encoder_config, config.model_output_dir)
    latest = config.model_output_dir / "latest.pt"
    _log(
        f"Best epoch: {best.best_epoch}, val_loss: {best.best_val_loss:.4f}. "
        f"Saved {saved}",
    )
    return latest


def run(config: TrainEncoderConfig) -> Path:
    """Execute the train-encoder pipeline.

    Returns the path of the saved ``latest.pt``. Raises ``_PreFlightError``
    on pre-flight failure (CLI handler maps to exit codes). Cards
    referenced by ``cards-played.txt`` that lack a converted ``.txt`` are
    reported via a log warning and dropped — they do not block the run.
    """
    _run_preflight(config)
    locator = ConvertedCardLocator(config.cards_folder)
    labels = _labels_from_corpus(config, locator)
    split = _card_split(config, labels)
    setup = _build_training_setup(config, labels, split, locator)
    best = _train_epochs(config, setup)
    return _save_best(config, setup, best)
