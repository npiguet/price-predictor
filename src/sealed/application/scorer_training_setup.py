"""A ``train-scorer`` run's setup: the dataset, the scorer it starts from,
the normalization statistics, the device, the Phase B encoder, the
optimizer and the loaders, gathered into the run's training context."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from price_predictor.domain.tokenizer import MtgTokenizer
from sealed.application.scorer_checkpoints import _build_train_config
from sealed.application.scorer_dataset import (
    _load_dataset,
    _make_loaders,
    _set_normalization_stats,
)
from sealed.application.scorer_encoder_loading import (
    _build_phase_b_encoder,
    _check_encoder_width,
)
from sealed.application.scorer_log import _log
from sealed.application.scorer_resume import (
    _build_optimizer,
    _move_optimizer_state,
    _resume_or_build_model,
)
from sealed.application.scorer_training_context import _TrainingContext
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.scorer_store import ScorerStore

if TYPE_CHECKING:
    from sealed.application.train_scorer import TrainScorerConfig


def _prepare(
    config: TrainScorerConfig, store: ScorerStore,
) -> _TrainingContext:
    t0 = time.monotonic()
    _log(f"Loading dataset from {config.outcomes_path}")
    train_examples, val_examples, embedding_table = _load_dataset(config)
    _log(
        f"Dataset ready: {len(train_examples)} train + {len(val_examples)} val "
        f"examples, {embedding_table.num_cards} unique cards "
        f"({time.monotonic() - t0:.1f}s)",
    )

    resume = _resume_or_build_model(config, store, embedding_table.embedding_dim)
    if config.resume is not None:
        _log(f"Resumed from checkpoint at epoch {resume.start_epoch}")
    elif config.scorer_checkpoint is not None:
        _log(f"Bootstrapped scorer from {config.scorer_checkpoint}")
    else:
        _log(
            f"Built fresh scorer model (d_model={resume.model.config.d_model}, "
            f"n_layers={config.n_layers}, dropout={config.dropout})"
        )

    _log("Computing per-card feature normalization stats")
    _set_normalization_stats(resume.model, train_examples, embedding_table)

    device = _select_device()
    resume.model.to(device)
    embedding_table.to(device)

    encoder: torch.nn.Module | None = None
    tokenizer: MtgTokenizer | None = None
    max_seq_len: int | None = None
    locator: ConvertedCardLocator | None = None
    idx_to_name: dict[int, str] | None = None
    if config.phase == "B":
        encoder, tokenizer, max_seq_len = _build_phase_b_encoder(config, resume)
        _check_encoder_width(encoder, embedding_table.embedding_dim, config)
        encoder.to(device)
        locator = ConvertedCardLocator(
            config.cards_path, embeddings_path=config.embeddings_path,
        )
        idx_to_name = {
            idx: name for name, idx in embedding_table.name_to_idx.items()
        }
        _log(
            f"Phase B encoder loaded (d_model={encoder.config.d_model}, "  # type: ignore[attr-defined]
            f"max_seq_len={max_seq_len}); {len(idx_to_name)} cards available "
            "for tokenization"
        )
    _log(f"Model and embedding table moved to {device}")

    optimizer = _build_optimizer(resume.model, config, encoder=encoder)
    if resume.optimizer_state:
        optimizer.load_state_dict(resume.optimizer_state)
        _move_optimizer_state(optimizer, device)

    train_loader, val_loader = _make_loaders(
        train_examples, val_examples, config.batch_size,
    )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_config = _build_train_config(config)
    _log(
        f"Setup complete in {time.monotonic() - t0:.1f}s "
        f"(batch_size={config.batch_size}, "
        f"~{len(train_examples) // max(config.batch_size, 1)} train batches/epoch)",
    )

    return _TrainingContext(
        model=resume.model,
        optimizer=optimizer,
        embedding_table=embedding_table,
        train_loader=train_loader,
        val_loader=val_loader,
        latest_path=config.checkpoint_dir / config.latest_checkpoint_name(),
        best_path=config.checkpoint_dir / config.best_checkpoint_name(),
        start_epoch=resume.start_epoch,
        best_val_accuracy=resume.best_val_accuracy,
        device=device,
        train_config=train_config,
        encoder=encoder,
        tokenizer=tokenizer,
        card_token_cache={} if encoder is not None else None,
        locator=locator,
        idx_to_name=idx_to_name,
        reference_batch=None,
        max_seq_len=max_seq_len,
        encoder_chunk_size=config.encoder_chunk_size,
        max_grad_norm=config.max_grad_norm,
        margin_weighting=config.margin_weighting,
    )


def _select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Training scorer on device: {device}")
    return device
