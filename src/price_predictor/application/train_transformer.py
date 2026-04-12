"""Train transformer use case: tokenize cards, train model, save artifact."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from price_predictor.application.converted_card_dataset import load_training_samples
from price_predictor.application.training_sample import TrainingSample
from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from price_predictor.infrastructure.transformer_store import save_model

logger = logging.getLogger(__name__)


@dataclass
class TransformerTrainResult:
    """Result of a transformer training run."""

    model_path: Path
    best_epoch: int
    best_val_loss: float
    stopped_early: bool
    final_epoch: int
    card_count: int
    cards_skipped: int
    price_range_min_eur: float
    price_range_max_eur: float
    max_seq_len: int


def analyze_sequence_lengths(card_texts: list[str], tokenizer: MtgTokenizer) -> tuple[int, dict]:
    """Analyze token length distribution and compute max_seq_len.

    Returns (max_seq_len, stats_dict) where max_seq_len is the maximum token
    length across all cards, rounded up to the nearest multiple of 8, clamped
    to a minimum of 64. This ensures zero truncation — every card is fully
    represented.
    """
    lengths = [len(tokenizer.tokenize(text)) for text in card_texts]

    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    max_len = max(lengths)

    # Use full max so no cards are truncated; round up to multiple of 8
    max_seq_len = math.ceil(max_len / 8) * 8
    max_seq_len = max(max_seq_len, 64)

    stats = {
        "p95": p95,
        "p99": p99,
        "max": max_len,
    }
    return max_seq_len, stats


def _price_bucket(price_eur: float) -> int:
    """Assign a price to a bucket index for stratification/weighting."""
    if price_eur < 2.0:
        return 0
    if price_eur < 10.0:
        return 1
    if price_eur < 50.0:
        return 2
    return 3


def _make_weighted_sampler(prices: list[float], exponent: float = 0.5) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that upsamples expensive cards.

    weight_i = 1 / (bucket_count_i ^ exponent)

    exponent=0.0  — uniform (no oversampling, cheap cards dominate)
    exponent=0.5  — sqrt weighting (~17× oversample expensive vs cheap) [default]
    exponent=1.0  — full inverse weighting (~260× oversample, too aggressive)

    Without any weighting, ~90% of cards are <€2 so the model never learns what
    makes a card expensive. Full inverse overcorrects and makes cheap cards wrong.
    Square-root weighting is the standard compromise.
    """
    buckets = [_price_bucket(p) for p in prices]
    bucket_counts = [0, 0, 0, 0]
    for b in buckets:
        bucket_counts[b] += 1
    weights = [1.0 / (bucket_counts[b] ** exponent) for b in buckets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _batch_to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["target"].to(device),
        batch["meta"].to(device),
    )


def _run_training_epoch(
    model: CardPriceTransformerModel,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Run one training epoch and return the mean batch loss."""
    model.train(True)
    losses: list[float] = []
    for batch in loader:
        input_ids, attention_mask, targets, meta = _batch_to_device(batch, device)
        predictions = model(input_ids, attention_mask, meta)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
    return sum(losses) / len(losses)


def _run_eval_epoch(
    model: CardPriceTransformerModel,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one evaluation epoch (no grad, no optimizer) and return the mean batch loss."""
    model.train(False)
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, targets, meta = _batch_to_device(batch, device)
            predictions = model(input_ids, attention_mask, meta)
            loss = loss_fn(predictions, targets)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def _run_epoch(
    model: CardPriceTransformerModel,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Backwards-compatible wrapper that dispatches to the train or eval helper."""
    if optimizer is not None:
        return _run_training_epoch(model, loader, loss_fn, device, optimizer, scheduler)
    return _run_eval_epoch(model, loader, loss_fn, device)


class _BestCheckpoint:
    """Tracks the lowest val_loss seen and the model weights at that epoch."""

    def __init__(self) -> None:
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self._state_dict: dict[str, torch.Tensor] | None = None

    def update(self, epoch: int, val_loss: float, model: torch.nn.Module) -> bool:
        """Snapshot model weights when val_loss improves. Returns True on new best."""
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        self.best_epoch = epoch
        self._state_dict = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        return True

    def restore(self, model: torch.nn.Module) -> None:
        """Reload the best snapshot into the model (no-op if never updated)."""
        if self._state_dict is not None:
            model.load_state_dict(self._state_dict)


def _train_loop(
    model: CardPriceTransformerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    warmup_epochs: int = 2,
) -> tuple[int, float, bool, int]:
    """Run the training loop with early stopping.

    Returns (best_epoch, best_val_loss, stopped_early, final_epoch).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.HuberLoss(delta=1.0)
    best = _BestCheckpoint()
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _run_training_epoch(
            model, train_loader, loss_fn, device, optimizer, scheduler,
        )
        val_loss = _run_eval_epoch(model, val_loader, loss_fn, device)
        elapsed = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, "
            f"{elapsed:.1f}s"
        )

        if best.update(epoch, val_loss, model):
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                best.restore(model)
                return best.best_epoch, best.best_val_loss, True, epoch

    best.restore(model)
    return best.best_epoch, best.best_val_loss, False, epochs


@dataclass
class _LoadedSamples:
    samples: list[TrainingSample]
    cards_skipped: int


def _load_samples(
    output_dir: Path, prices_path: Path, printings_path: Path
) -> _LoadedSamples:
    """Resolve cards against price + metadata maps and report skip count."""
    metadata_map, price_map = build_metadata_map(printings_path, prices_path)
    txt_file_count = len(list(output_dir.rglob("*.txt")))
    samples = load_training_samples(output_dir, price_map, metadata_map)
    logger.info("Matched %d cards to texts and prices", len(samples))

    if len(samples) < 5:
        raise ValueError(
            f"Insufficient training data: only {len(samples)} matched cards. Need at least 5."
        )

    return _LoadedSamples(samples=samples, cards_skipped=txt_file_count - len(samples))


def _build_dataset(
    samples: list[TrainingSample],
    max_seq_len: int,
    tokenizer: MtgTokenizer,
    log_offset: float,
) -> TransformerTrainingDataset:
    """Build a TransformerTrainingDataset from a list of TrainingSamples."""
    return TransformerTrainingDataset(
        [(s.name, s.text, s.price_eur) for s in samples],
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        log_offset=log_offset,
        printing_data_list=[s.printing_data for s in samples],
    )


def _build_loaders(
    samples: list[TrainingSample],
    tokenizer: MtgTokenizer,
    max_seq_len: int,
    batch_size: int,
    log_offset: float,
    sampler_exponent: float,
    random_seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Stratified split + oversampled train loader + plain val loader."""
    buckets_all = [_price_bucket(s.price_eur) for s in samples]
    train_data, val_data = train_test_split(
        samples, test_size=0.2, random_state=random_seed, stratify=buckets_all,
    )
    logger.info("Split: %d train, %d validation", len(train_data), len(val_data))

    train_dataset = _build_dataset(train_data, max_seq_len, tokenizer, log_offset)
    val_dataset = _build_dataset(val_data, max_seq_len, tokenizer, log_offset)

    sampler = _make_weighted_sampler(
        [s.price_eur for s in train_data], exponent=sampler_exponent,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _build_model(
    config: TransformerConfig, device: torch.device
) -> CardPriceTransformerModel:
    """Instantiate the model on the chosen device."""
    model = CardPriceTransformerModel(config)
    model.to(device)
    return model


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training. No GPU detected.")
    return torch.device("cuda")


def _log_sequence_length_stats(max_seq_len: int, stats: dict) -> None:
    logger.info(
        "Token length stats: p95=%d, p99=%d, max=%d",
        stats["p95"], stats["p99"], stats["max"],
    )
    logger.info(
        "Chosen max_seq_len=%d (max rounded up to multiple of 8). "
        "No cards will be truncated.",
        max_seq_len,
    )


def _finalise_and_save(
    model: CardPriceTransformerModel,
    config: TransformerConfig,
    model_output: Path,
    samples: list[TrainingSample],
    cards_skipped: int,
    best_epoch: int,
    best_val_loss: float,
    stopped_early: bool,
    final_epoch: int,
    epochs: int,
) -> TransformerTrainResult:
    """Move model to CPU, persist it, and assemble the train result."""
    model.cpu()
    _version, model_path = save_model(model, config, model_output)

    prices = [s.price_eur for s in samples]
    print(
        f"\nTraining complete. Best epoch: {best_epoch}/{epochs}, "
        f"val_loss: {best_val_loss:.3f}. "
        + (f"Early stopping triggered at epoch {final_epoch}." if stopped_early else "")
    )
    print(f"Model saved to {model_path}")

    return TransformerTrainResult(
        model_path=model_path,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_early=stopped_early,
        final_epoch=final_epoch,
        card_count=len(samples),
        cards_skipped=cards_skipped,
        price_range_min_eur=min(prices),
        price_range_max_eur=max(prices),
        max_seq_len=config.max_seq_len,
    )


def train_transformer(
    output_dir: Path,
    prices_path: Path,
    printings_path: Path,
    model_output: Path = Path("models/transformer/"),
    vocab_path: Path = Path("models/transformer/vocab.txt"),
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-4,
    patience: int = 5,
    random_seed: int = 42,
    log_offset: float = 2.0,
    dropout: float = 0.1,
    sampler_exponent: float = 1.0,
    d_model: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    ff_dim: int = 512,
) -> TransformerTrainResult:
    """Train a transformer model on converted card texts."""
    torch.manual_seed(random_seed)
    tokenizer = load_tokenizer(vocab_path)

    loaded = _load_samples(output_dir, prices_path, printings_path)

    max_seq_len, stats = analyze_sequence_lengths(
        [s.text for s in loaded.samples], tokenizer,
    )
    _log_sequence_length_stats(max_seq_len, stats)

    train_loader, val_loader = _build_loaders(
        loaded.samples, tokenizer, max_seq_len, batch_size,
        log_offset, sampler_exponent, random_seed,
    )

    config = TransformerConfig(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, ff_dim=ff_dim,
        max_seq_len=max_seq_len, vocab_size=tokenizer.vocab_size,
        dropout=dropout, log_offset=log_offset,
    )

    device = _require_cuda_device()
    model = _build_model(config, device)

    best_epoch, best_val_loss, stopped_early, final_epoch = _train_loop(
        model, train_loader, val_loader, device, epochs, lr, patience,
    )

    return _finalise_and_save(
        model, config, model_output, loaded.samples, loaded.cards_skipped,
        best_epoch, best_val_loss, stopped_early, final_epoch, epochs,
    )
