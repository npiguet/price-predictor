"""What the ``train-encoder`` epoch loop runs on, built once after the split:
the tokenized corpus, the model, the datasets and loaders, the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from price_predictor.infrastructure.tokenizer_store import (
    load_tokenizer,
    load_vocabulary,
)
from sealed.application.encoder_batches import (
    _LengthBucketBatchSampler,
    _pad_collate,
    _tokenize_corpus,
    _WinnabilityDataset,
)
from sealed.application.encoder_labels import CardLabelMap
from sealed.application.encoder_mlm import MlmMaskSpec, _resolve_special_token_ids
from sealed.application.encoder_split import CardSplit
from sealed.application.encoder_training_common import _log
from sealed.application.encoder_training_loop import (
    _make_optimizer,
    _select_device,
    _Stepper,
    _TrainContext,
)
from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator

if TYPE_CHECKING:
    from sealed.application.train_encoder import TrainEncoderConfig


@dataclass
class _TrainingSetup:
    """Everything the epoch loop needs, built once after the train/val split."""

    model: SealedEncoderModel
    encoder_config: SealedEncoderConfig
    train_loader: DataLoader
    val_loader: DataLoader
    stepper: _Stepper
    ctx: _TrainContext


def _build_training_setup(
    config: TrainEncoderConfig,
    labels: CardLabelMap,
    split: CardSplit,
    locator: ConvertedCardLocator,
) -> _TrainingSetup:
    """Tokenize the corpus once, build the model + datasets + loaders +
    optimizer for the epoch loop. ``max_seq_len`` (the position-embedding
    table size, FR-022) falls out of the one-time tokenization.
    """
    tokenizer = load_tokenizer(config.vocab_path)
    vocab = load_vocabulary(config.vocab_path)
    mask_token_id, special_token_ids = _resolve_special_token_ids(vocab)

    tokenized_by_name, max_seq_len = _tokenize_corpus(labels, tokenizer, locator)
    _log(f"Tokenized {len(tokenized_by_name)} card(s); max_seq_len = {max_seq_len}")

    encoder_config = SealedEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_dim=config.ff_dim,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
        n_pool_queries=config.n_pool_queries,
        pool_mode=config.pool_mode,
    )

    torch.manual_seed(config.random_seed)
    model = SealedEncoderModel(encoder_config)
    device = _select_device()
    model.to(device)
    _log(f"Sealed encoder on {device}")

    train_ds = _WinnabilityDataset(
        split.train, labels, tokenized_by_name, config.shrinkage_k,
    )
    val_ds = _WinnabilityDataset(
        split.val, labels, tokenized_by_name, config.shrinkage_k,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=_LengthBucketBatchSampler(
            train_ds.lengths, config.batch_size, shuffle=True,
        ),
        collate_fn=_pad_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=_LengthBucketBatchSampler(
            val_ds.lengths, config.batch_size, shuffle=False,
        ),
        collate_fn=_pad_collate,
    )

    total_steps = max(1, config.epochs * len(train_loader))
    optimizer, scheduler = _make_optimizer(model, config.lr, total_steps)

    ctx = _TrainContext(
        mlm_weight=config.mlm_weight,
        mlm_mask=MlmMaskSpec(
            mask_prob=config.mlm_mask_prob,
            mask_token_id=mask_token_id,
            special_token_ids=special_token_ids,
        ),
        device=device,
    )
    return _TrainingSetup(
        model, encoder_config, train_loader, val_loader,
        _Stepper(optimizer, scheduler), ctx,
    )
