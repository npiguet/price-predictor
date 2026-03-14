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
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from price_predictor.application.evaluate_transformer import evaluate_transformer
from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.forge_parser import parse_forge_cards
from price_predictor.infrastructure.mtgjson_loader import build_name_to_uuids, build_price_map
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


def _match_cards_to_texts(
    cards: list,
    name_to_uuids: dict,
    price_map: dict,
    output_dir: Path,
) -> list[tuple[str, str, float]]:
    """Match cards to their converted text files and prices."""
    matched = []
    skipped = 0
    for card in cards:
        card_name = card.name
        if card_name not in name_to_uuids:
            for full_name in name_to_uuids:
                if full_name.startswith(card_name + " // "):
                    card_name = full_name
                    break
        if card_name not in price_map:
            continue

        slug = card.name.lower().replace(" ", "_").replace(",", "").replace("'", "")
        first_letter = slug[0] if slug else "_"
        text_path = output_dir / first_letter / f"{slug}.txt"
        if not text_path.exists():
            skipped += 1
            continue

        text = text_path.read_text(encoding="utf-8")
        matched.append((card.name, text, price_map[card_name]))

    if skipped > 0:
        logger.info("Skipped %d cards with no matching text file", skipped)
    return matched


def analyze_sequence_lengths(card_texts: list[str]) -> tuple[int, dict]:
    """Analyze token length distribution and compute max_seq_len.

    Returns (max_seq_len, stats_dict) where max_seq_len is the 95th percentile
    rounded up to the nearest multiple of 8, clamped to a minimum of 64.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lengths = [len(tokenizer.encode(text)) for text in card_texts]

    p95 = int(np.percentile(lengths, 95))
    p99 = int(np.percentile(lengths, 99))
    max_len = max(lengths)

    # Round up to nearest multiple of 8
    max_seq_len = math.ceil(p95 / 8) * 8
    max_seq_len = max(max_seq_len, 64)

    truncation_pct = sum(1 for n in lengths if n > max_seq_len) / len(lengths) * 100

    stats = {
        "p95": p95,
        "p99": p99,
        "max": max_len,
        "truncation_pct": round(truncation_pct, 1),
    }
    return max_seq_len, stats


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

    # Linear warmup scheduler
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["target"].to(device)

                predictions = model(input_ids, attention_mask)
                loss = loss_fn(predictions, targets)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        elapsed = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch}/{epochs} — "
            f"train_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}, "
            f"{elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                # Restore best weights
                model.load_state_dict(best_state_dict)
                return best_epoch, best_val_loss, True, epoch

    # Restore best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return best_epoch, best_val_loss, False, epochs


def train_transformer(
    output_dir: Path,
    forge_cards_path: Path,
    prices_path: Path,
    printings_path: Path,
    model_output: Path = Path("models/transformer/"),
    batch_size: int = 64,
    epochs: int = 20,
    lr: float = 1e-4,
    patience: int = 5,
    random_seed: int = 42,
) -> TransformerTrainResult:
    """Train a transformer model on converted card texts."""
    torch.manual_seed(random_seed)

    # 1. Load cards and match to prices + text files
    cards, parse_errors = parse_forge_cards(forge_cards_path)
    logger.info("Parsed %d cards (%d parse errors)", len(cards), len(parse_errors))

    name_to_uuids = build_name_to_uuids(printings_path)
    price_map = build_price_map(prices_path, name_to_uuids)

    matched = _match_cards_to_texts(cards, name_to_uuids, price_map, output_dir)
    logger.info("Matched %d cards to texts and prices", len(matched))

    if len(matched) < 5:
        raise ValueError(
            f"Insufficient training data: only {len(matched)} matched cards. Need at least 5."
        )

    cards_skipped = len(cards) - len(matched) + len(parse_errors)

    # 2. Analyze sequence lengths
    texts = [text for _, text, _ in matched]
    max_seq_len, stats = analyze_sequence_lengths(texts)
    logger.info(
        "Sequence length analysis: max_seq_len=%d, p95=%d, p99=%d, max=%d, truncation=%.1f%%",
        max_seq_len, stats["p95"], stats["p99"], stats["max"], stats["truncation_pct"],
    )

    # 3. Split data 80/20
    train_data, val_data = train_test_split(
        matched, test_size=0.2, random_state=random_seed
    )
    logger.info("Split: %d train, %d validation", len(train_data), len(val_data))

    # 4. Build datasets
    train_dataset = TransformerTrainingDataset(train_data, max_seq_len=max_seq_len)
    val_dataset = TransformerTrainingDataset(val_data, max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Build model
    config = TransformerConfig(
        d_model=128,
        n_layers=4,
        n_heads=4,
        ff_dim=512,
        max_seq_len=max_seq_len,
        vocab_size=30522,
        dropout=0.1,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for training. No GPU detected.")
    device = torch.device("cuda")

    model = CardPriceTransformerModel(config)
    model.to(device)

    # 6. Train
    best_epoch, best_val_loss, stopped_early, final_epoch = _train_loop(
        model, train_loader, val_loader, device, epochs, lr, patience,
    )

    # 7. Save
    model.cpu()
    model_path = save_model(model, config, model_output)

    prices = [price for _, _, price in matched]
    print(
        f"\nTraining complete. Best epoch: {best_epoch}/{epochs}, "
        f"val_loss: {best_val_loss:.3f}. "
        + (f"Early stopping triggered at epoch {final_epoch}." if stopped_early else "")
    )
    print(f"Model saved to {model_path}")

    # 8. Auto-evaluate
    try:
        eval_result = evaluate_transformer(
            model_dir=model_output,
            output_dir=output_dir,
            forge_cards_path=forge_cards_path,
            prices_path=prices_path,
            printings_path=printings_path,
            random_seed=random_seed,
        )
        print(
            f"\nEvaluation on validation set ({eval_result.sample_count} cards):\n"
            f"  Mean absolute error: \u20ac{eval_result.mean_absolute_error_eur}\n"
            f"  Median percentage error: {eval_result.median_percentage_error}%"
        )
    except Exception as e:
        logger.warning("Auto-evaluation failed: %s", e)

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
