"""Train transformer use case: tokenize cards, train model, save artifact."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.mtgjson_loader import (
    build_metadata_map,
    build_name_to_uuids,
)
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_model import (
    AuxiliaryTrainingModel,
    CardPriceTransformerModel,
)
from price_predictor.infrastructure.transformer_store import save_model

logger = logging.getLogger(__name__)

# ─── Auxiliary label indices (data-model.md) ──────────────────────────────────
# Classification heads: 0,1-6,14-19 (13 total)
_AUX_CLASSIFICATION_INDICES = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19]
# Regression heads: 7-13 (7 total)
_AUX_REGRESSION_INDICES = [7, 8, 9, 10, 11, 12, 13]


def _compute_pos_weight(labels: np.ndarray) -> float:
    """Compute pos_weight = num_negatives / num_positives for a binary label column."""
    n_pos = float(np.sum(labels > 0.5))
    n_neg = float(np.sum(labels <= 0.5))
    if n_pos == 0:
        return float(n_neg) if n_neg > 0 else 1.0
    return n_neg / n_pos


def _compute_reg_stats(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) with a minimum std floor of 1.0."""
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = max(std, 1.0)
    return mean, std


def _combined_loss(
    l_price: torch.Tensor,
    aux_losses: list[torch.Tensor],
    aux_lambda: float,
) -> torch.Tensor:
    """Combine price loss with auxiliary losses: L_total = L_price + lambda * sum(L_aux_i)."""
    if aux_lambda == 0.0 or not aux_losses:
        return l_price
    return l_price + aux_lambda * sum(aux_losses)


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


def _match_cards_to_texts(
    output_dir: Path,
    name_to_uuids: dict,
    price_map: dict,
    metadata_map: dict | None = None,
) -> list[tuple[str, str, float, PrintingData]]:
    """Read converted text files from output_dir, match to prices and PrintingData.

    Returns list of (card_name, text, price_eur, printing_data) tuples.
    PrintingData is looked up from metadata_map (or defaults if not found).
    """
    lower_to_canonical: dict[str, str] = {k.lower(): k for k in name_to_uuids}
    matched = []
    skipped = 0
    txt_files = sorted(output_dir.rglob("*.txt"))
    for txt_file in txt_files:
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            skipped += 1
            continue

        # Extract card name from the name: line
        card_name = None
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("name:"):
                card_name = line[len("name:"):].strip()
                break

        if not card_name:
            skipped += 1
            continue

        # Case-insensitive matching (converted cards are lowercase)
        card_name_lower = card_name.lower()
        canonical = lower_to_canonical.get(card_name_lower)
        if canonical is None:
            for full_lower, full_canonical in lower_to_canonical.items():
                if full_lower.startswith(card_name_lower + " // "):
                    canonical = full_canonical
                    break

        if canonical is None or canonical not in price_map:
            continue

        # Look up PrintingData from metadata_map (no longer embedded in text)
        printing_data = PrintingData.defaults()
        if metadata_map and canonical in metadata_map:
            printing_data = metadata_map[canonical]

        matched.append((card_name, text, price_map[canonical], printing_data))

    if skipped > 0:
        logger.info("Skipped %d text files (unreadable or missing name)", skipped)
    return matched


def analyze_sequence_lengths(card_texts: list[str], tokenizer: MtgTokenizer) -> tuple[int, dict]:
    """Analyze token length distribution and compute max_seq_len.

    Returns (max_seq_len, stats_dict) where max_seq_len is the maximum token
    length across all cards, rounded up to the nearest multiple of 8, clamped
    to a minimum of 64. This ensures zero truncation — every card is fully
    represented.
    """
    lengths = [len(tokenizer._tokenize(text)) for text in card_texts]

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


def _train_loop(
    model: CardPriceTransformerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    warmup_epochs: int = 2,
    aux_lambda: float = 0.0,
    aux_cls_loss_fns: list | None = None,
    aux_reg_loss_fn: torch.nn.Module | None = None,
    reg_means: dict[int, float] | None = None,
    reg_stds: dict[int, float] | None = None,
) -> tuple[int, float, bool, int]:
    """Run the training loop with early stopping.

    When aux_lambda > 0, wraps model in AuxiliaryTrainingModel and uses
    combined loss. Returns (best_epoch, best_val_loss, stopped_early, final_epoch).
    Early stopping is always based on price validation loss only.
    """
    use_aux = aux_lambda > 0.0 and aux_cls_loss_fns is not None

    if use_aux:
        training_model: nn.Module = AuxiliaryTrainingModel(model, n_aux=20)
    else:
        training_model = model

    training_model.to(device)

    if use_aux and aux_cls_loss_fns is not None:
        for loss_fn in aux_cls_loss_fns:
            loss_fn.to(device)

    optimizer = torch.optim.AdamW(training_model.parameters(), lr=lr)

    # Linear warmup scheduler
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    price_loss_fn = torch.nn.HuberLoss(delta=1.0)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Train
        training_model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)
            meta = batch["meta"].to(device)

            optimizer.zero_grad()

            if use_aux:
                price_pred, aux_preds = training_model(input_ids, attention_mask, meta)
                l_price = price_loss_fn(price_pred, targets)
                aux_labels_batch = batch["aux_labels"].to(device)
                aux_losses = _compute_aux_losses(
                    aux_preds, aux_labels_batch,
                    aux_cls_loss_fns, aux_reg_loss_fn,
                    reg_means, reg_stds, device,
                )
                loss = _combined_loss(l_price, aux_losses, aux_lambda)
            else:
                predictions = training_model(input_ids, attention_mask, meta)
                loss = price_loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validate
        training_model.eval()
        val_losses = []
        val_aux_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)
                meta = batch["meta"].to(device)

                if use_aux:
                    price_pred, aux_preds = training_model(input_ids, attention_mask, meta)
                    l_price = price_loss_fn(price_pred, targets)
                    aux_labels_batch = batch["aux_labels"].to(device)
                    aux_losses = _compute_aux_losses(
                        aux_preds, aux_labels_batch,
                        aux_cls_loss_fns, aux_reg_loss_fn,
                        reg_means, reg_stds, device,
                    )
                    val_losses.append(l_price.item())
                    val_aux_losses.append(sum(a.item() for a in aux_losses))
                else:
                    predictions = training_model(input_ids, attention_mask, meta)
                    loss = price_loss_fn(predictions, targets)
                    val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        elapsed = time.perf_counter() - epoch_start

        if use_aux and val_aux_losses:
            avg_aux = sum(val_aux_losses) / len(val_aux_losses)
            print(
                f"Epoch {epoch}/{epochs} — "
                f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, "
                f"aux_loss: {avg_aux:.3f}, {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} — "
                f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, "
                f"{elapsed:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Always snapshot the base model's state (not the wrapper)
            state_src = training_model.base if use_aux else training_model
            best_state_dict = {k: v.cpu().clone() for k, v in state_src.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_state_dict)
                return best_epoch, best_val_loss, True, epoch

    # Restore best weights to base model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return best_epoch, best_val_loss, False, epochs


def _compute_aux_losses(
    aux_preds: list[torch.Tensor],
    aux_labels: torch.Tensor,
    cls_loss_fns: list,
    reg_loss_fn: torch.nn.Module,
    reg_means: dict[int, float],
    reg_stds: dict[int, float],
    device: torch.device,
) -> list[torch.Tensor]:
    """Compute per-head auxiliary losses."""
    losses = []
    cls_idx = 0
    for head_idx in range(20):
        if head_idx in _AUX_CLASSIFICATION_INDICES:
            targets = aux_labels[:, head_idx]
            loss = cls_loss_fns[cls_idx](aux_preds[head_idx], targets)
            losses.append(loss)
            cls_idx += 1
        else:
            # Regression: standardize targets on-the-fly using pre-computed stats
            mean = reg_means[head_idx]
            std = reg_stds[head_idx]
            targets = (aux_labels[:, head_idx] - mean) / std
            loss = reg_loss_fn(aux_preds[head_idx], targets)
            losses.append(loss)
    return losses


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
    aux_lambda: float = 0.0,
) -> TransformerTrainResult:
    """Train a transformer model on converted card texts."""
    torch.manual_seed(random_seed)

    # 0. Load tokenizer
    tokenizer = load_tokenizer(vocab_path)

    # 1. Read converted text files and match to prices and PrintingData
    metadata_map, price_map = build_metadata_map(printings_path, prices_path)
    name_to_uuids, _uuid_meta = build_name_to_uuids(printings_path)

    txt_file_count = len(list(output_dir.rglob("*.txt")))
    matched = _match_cards_to_texts(output_dir, name_to_uuids, price_map, metadata_map)
    logger.info("Matched %d cards to texts and prices", len(matched))

    if len(matched) < 5:
        raise ValueError(
            f"Insufficient training data: only {len(matched)} matched cards. Need at least 5."
        )

    cards_skipped = txt_file_count - len(matched)

    # 2. Analyze sequence lengths
    texts = [text for _, text, _, _ in matched]
    max_seq_len, stats = analyze_sequence_lengths(texts, tokenizer)
    logger.info(
        "Token length stats: p95=%d, p99=%d, max=%d",
        stats["p95"], stats["p99"], stats["max"],
    )
    logger.info(
        "Chosen max_seq_len=%d (max rounded up to multiple of 8). No cards will be truncated.",
        max_seq_len,
    )

    # 3. Split data 80/20, stratified by price bucket so expensive cards land in both sets
    prices_all = [p for _, _, p, _ in matched]
    buckets_all = [_price_bucket(p) for p in prices_all]
    train_data, val_data = train_test_split(
        matched, test_size=0.2, random_state=random_seed, stratify=buckets_all
    )
    logger.info("Split: %d train, %d validation", len(train_data), len(val_data))

    # 4. Build datasets (strip PrintingData into separate list for dataset)
    train_tuples = [(n, t, p) for n, t, p, _ in train_data]
    val_tuples = [(n, t, p) for n, t, p, _ in val_data]
    train_pd = [pd for _, _, _, pd in train_data]
    val_pd = [pd for _, _, _, pd in val_data]

    # 4a. Pre-compute auxiliary labels when aux_lambda > 0 (FR-011, T013)
    train_aux_tensor: torch.Tensor | None = None
    val_aux_tensor: torch.Tensor | None = None
    aux_cls_loss_fns: list | None = None
    aux_reg_loss_fn: torch.nn.Module | None = None
    reg_means: dict[int, float] | None = None
    reg_stds: dict[int, float] | None = None

    if aux_lambda > 0.0:
        from sealed.domain.embedding_probe import compute_aux_labels

        print(f"Computing auxiliary labels for ~{len(matched)} cards...")
        train_texts = [t for _, t, _, _ in train_data]
        val_texts = [t for _, t, _, _ in val_data]

        train_labels_np = np.stack([compute_aux_labels(t) for t in train_texts])
        val_labels_np = np.stack([compute_aux_labels(t) for t in val_texts])

        print("Computing class weights and target statistics...")
        # Build per-classification-head pos_weight
        pos_weights = {}
        for head_idx in _AUX_CLASSIFICATION_INDICES:
            pw = _compute_pos_weight(train_labels_np[:, head_idx])
            pos_weights[head_idx] = pw

        # Build per-regression-head standardization stats
        reg_means = {}
        reg_stds = {}
        for head_idx in _AUX_REGRESSION_INDICES:
            mean, std = _compute_reg_stats(train_labels_np[:, head_idx])
            reg_means[head_idx] = mean
            reg_stds[head_idx] = std

        # Print statistics
        colors = ["W", "U", "B", "R", "G", "C"]
        head_names = (
            ["is_land"]
            + [f"card_color_{c}" for c in colors]
            + [f"pip_count_{c}" for c in colors]
            + ["mana_value"]
            + [f"mana_produced_{c}" for c in colors]
        )
        for head_idx in _AUX_CLASSIFICATION_INDICES:
            name = head_names[head_idx]
            n_pos = int(np.sum(train_labels_np[:, head_idx] > 0.5))
            n_neg = len(train_labels_np) - n_pos
            print(f"  {name}: pos_weight={pos_weights[head_idx]:.1f} "
                  f"({n_pos} positive / {n_neg} negative)")
        for head_idx in _AUX_REGRESSION_INDICES:
            name = head_names[head_idx]
            print(f"  {name}: mean={reg_means[head_idx]:.2f}, std={reg_stds[head_idx]:.2f}")

        train_aux_tensor = torch.tensor(train_labels_np, dtype=torch.float32)
        val_aux_tensor = torch.tensor(val_labels_np, dtype=torch.float32)

        # Build loss functions
        aux_cls_loss_fns = []
        for head_idx in _AUX_CLASSIFICATION_INDICES:
            pw_tensor = torch.tensor([pos_weights[head_idx]], dtype=torch.float32)
            aux_cls_loss_fns.append(
                torch.nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
            )
        aux_reg_loss_fn = torch.nn.MSELoss()

    train_dataset = TransformerTrainingDataset(
        train_tuples, max_seq_len=max_seq_len, tokenizer=tokenizer,
        log_offset=log_offset, printing_data_list=train_pd,
        aux_labels=train_aux_tensor,
    )
    val_dataset = TransformerTrainingDataset(
        val_tuples, max_seq_len=max_seq_len, tokenizer=tokenizer,
        log_offset=log_offset, printing_data_list=val_pd,
        aux_labels=val_aux_tensor,
    )

    # Oversample expensive cards during training so their gradient signal isn't
    # drowned out by the ~90% cheap-card majority.
    train_prices = [p for _, _, p, _ in train_data]
    sampler = _make_weighted_sampler(train_prices, exponent=sampler_exponent)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Build model
    config = TransformerConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        vocab_size=tokenizer.vocab_size,
        dropout=dropout,
        log_offset=log_offset,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training. No GPU detected.")
    device = torch.device("cuda")

    model = CardPriceTransformerModel(config)

    # 6. Train
    best_epoch, best_val_loss, stopped_early, final_epoch = _train_loop(
        model, train_loader, val_loader, device, epochs, lr, patience,
        aux_lambda=aux_lambda,
        aux_cls_loss_fns=aux_cls_loss_fns,
        aux_reg_loss_fn=aux_reg_loss_fn,
        reg_means=reg_means,
        reg_stds=reg_stds,
    )

    # 7. Save (always save the base model, not the aux wrapper)
    model.cpu()
    version, model_path = save_model(model, config, model_output)

    prices = [price for _, _, price, _ in matched]
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
        card_count=len(matched),
        cards_skipped=cards_skipped,
        price_range_min_eur=min(prices),
        price_range_max_eur=max(prices),
        max_seq_len=max_seq_len,
    )
