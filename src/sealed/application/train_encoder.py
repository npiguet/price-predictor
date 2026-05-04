"""``train-encoder``: train a sealed encoder + regression head on the
per-card winnability target.

The training loop reads ``cards-played.txt`` once, aggregates labels with
Bayesian shrinkage, splits cards into train/val by winnability quartile,
trains the dual-pool encoder + regression head from random init on MSE, and
saves only encoder weights to ``models/sealed/encoder/``.

Per FR-013, aggregation is inline: there is no separate
``aggregate-labels`` subcommand.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.infrastructure.cards_played_reader import (
    CardsPlayedRow,
    iter_rows,
)
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.encoder_store import SealedEncoderStore

# ── Hardcoded constants (FR-022) ────────────────────────────────────────
D_MODEL: int = 256
FF_DIM: int = 1024
VAL_FRACTION: float = 0.2
RANDOM_SEED: int = 42
_MAX_SEQ_LEN_MULTIPLE: int = 8
_MIN_MAX_SEQ_LEN: int = 64
_WIN_RATES_FILENAME: str = "cards-win-rates.txt"


def _log(message: str) -> None:
    """Print a timestamped progress line to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# ── Errors ──────────────────────────────────────────────────────────────


class CorpusInconsistencyError(RuntimeError):
    """Raised when the cards-played corpus references cards absent from
    ``output/cardsfolder/`` (FR-023d).
    """


# ── Aggregation ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CardLabel:
    card_name: str
    wins_when_played: int
    wins_when_in_deck: int
    shrunk_label: float

    @property
    def raw_ratio(self) -> float:
        if self.wins_when_in_deck == 0:
            return 0.0
        return self.wins_when_played / self.wins_when_in_deck


WinnabilityMap = dict[str, CardLabel]


def _aggregate_counts(
    rows: Iterable[CardsPlayedRow],
) -> dict[str, tuple[int, int]]:
    """Single pass over the cards-played stream.

    For each card, returns ``(wins_when_played, wins_when_in_deck)``
    counted only on games where that card's side won (FR-010). A card is
    counted at most once per game per side (FR-003 edge case).
    """
    counts: dict[str, list[int]] = {}
    for row in rows:
        if row.winner == "A":
            played, not_played = row.cards_played_a, row.cards_not_played_a
        else:
            played, not_played = row.cards_played_b, row.cards_not_played_b
        played_set = set(played)
        in_deck_set = played_set | set(not_played)
        for name in in_deck_set:
            entry = counts.setdefault(name, [0, 0])
            if name in played_set:
                entry[0] += 1
            entry[1] += 1
    return {name: (wp, wd) for name, (wp, wd) in counts.items()}


def _shrink(wins_when_played: int, wins_when_in_deck: int, k: float) -> float:
    """Bayesian shrinkage toward 0.5 (FR-011)."""
    return (wins_when_played + k / 2.0) / (wins_when_in_deck + k)


def _build_winnability_map(
    counts: dict[str, tuple[int, int]], shrinkage_k: float,
) -> WinnabilityMap:
    """Build the full label map, dropping cards with zero ``wins_when_in_deck``."""
    labels: WinnabilityMap = {}
    for name, (wp, wd) in counts.items():
        if wd == 0:
            continue
        labels[name] = CardLabel(
            card_name=name,
            wins_when_played=wp,
            wins_when_in_deck=wd,
            shrunk_label=_shrink(wp, wd, shrinkage_k),
        )
    return labels


# ── cards-win-rates.txt writer ──────────────────────────────────────────

_WIN_RATES_HEADER: str = (
    "card_name;wins_when_played;wins_when_in_deck;raw_ratio;shrunk_label"
)


def _write_win_rates(labels: WinnabilityMap, path: Path) -> None:
    """Write the per-card label snapshot, sorted by raw ratio descending."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(labels.values(), key=lambda lbl: lbl.raw_ratio, reverse=True)
    lines = [_WIN_RATES_HEADER]
    for lbl in rows:
        lines.append(
            f"{lbl.card_name};{lbl.wins_when_played};{lbl.wins_when_in_deck}"
            f";{lbl.raw_ratio:.5f};{lbl.shrunk_label:.5f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Corpus consistency check ────────────────────────────────────────────


def _check_corpus_consistency(
    labels: WinnabilityMap, locator: ConvertedCardLocator,
) -> None:
    """FR-023d. Raise ``CorpusInconsistencyError`` if any card is missing
    from ``output/cardsfolder/``.
    """
    missing: list[str] = []
    for name in labels:
        if locator.text_path(name) is None:
            missing.append(name)
    if not missing:
        return
    missing.sort()
    head = missing[:20]
    suffix = f" (and {len(missing) - 20} more)" if len(missing) > 20 else ""
    raise CorpusInconsistencyError(
        f"{len(missing)} card(s) referenced by cards-played.txt have no "
        f"converted .txt under the cards folder. First {len(head)}: "
        + ", ".join(head)
        + suffix
        + ". Run python -m price_predictor convert to rebuild the corpus."
    )


# ── Train/val split ─────────────────────────────────────────────────────


def _split_cards(
    labels: WinnabilityMap,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[str], list[str]]:
    """Card-level disjoint split, stratified on winnability quartile (FR-018).

    Falls back to fewer strata if the label distribution is too narrow to
    yield four distinct quantile bins (Edge Cases).
    """
    if not labels:
        return [], []
    names = sorted(labels.keys())
    values = np.array([labels[n].shrunk_label for n in names])

    quartile_edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
    bin_assignments = np.digitize(values, quartile_edges)

    rng = np.random.default_rng(seed)
    train: list[str] = []
    val: list[str] = []
    unique_bins = np.unique(bin_assignments)
    for b in unique_bins:
        bucket_idxs = np.where(bin_assignments == b)[0]
        rng.shuffle(bucket_idxs)
        n_val = int(round(len(bucket_idxs) * val_fraction))
        val_set = set(bucket_idxs[:n_val].tolist())
        for idx in bucket_idxs:
            (val if idx in val_set else train).append(names[idx])
    train.sort()
    val.sort()
    return train, val


# ── Dataset ─────────────────────────────────────────────────────────────


class _WinnabilityDataset(Dataset):
    """``(input_ids, attention_mask, label)`` triples for the encoder.

    Strips the ``name:`` line per FR-014a so the encoder sees the same
    shape as the inference path in ``CardEncoder``.
    """

    def __init__(
        self,
        card_names: Sequence[str],
        labels: WinnabilityMap,
        tokenizer: MtgTokenizer,
        locator: ConvertedCardLocator,
        max_seq_len: int,
    ) -> None:
        self._names = list(card_names)
        self._labels = labels
        self._tokenizer = tokenizer
        self._locator = locator
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self._names)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        name = self._names[idx]
        text = self._read_card_text(name)
        ids, mask = self._tokenizer.encode(text, self._max_seq_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(self._labels[name].shrunk_label, dtype=torch.float32),
        )

    def _read_card_text(self, name: str) -> str:
        converted = self._locator.load_text(name)
        if converted is None:
            raise FileNotFoundError(
                f"No converted card text on disk for {name!r}",
            )
        return converted.without_name_line()


def _measure_max_seq_len(
    card_names: Iterable[str],
    locator: ConvertedCardLocator,
    tokenizer: MtgTokenizer,
) -> int:
    """Longest tokenized card length, rounded up to a multiple of 8 (FR-022)."""
    max_len = 1
    for name in card_names:
        converted = locator.load_text(name)
        if converted is None:
            continue
        tokens = tokenizer.tokenize(converted.without_name_line())
        if len(tokens) > max_len:
            max_len = len(tokens)
    rounded = math.ceil(max_len / _MAX_SEQ_LEN_MULTIPLE) * _MAX_SEQ_LEN_MULTIPLE
    return max(rounded, _MIN_MAX_SEQ_LEN)


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class TrainEncoderConfig:
    cards_played_path: Path = field(
        default_factory=lambda: Path("output/sealed/cards-played.txt"),
    )
    cards_folder: Path = field(
        default_factory=lambda: Path("output/cardsfolder/"),
    )
    vocab_path: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/vocab.txt"),
    )
    model_output_dir: Path = field(
        default_factory=lambda: Path("models/sealed/encoder/"),
    )
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    patience: int = 20
    dropout: float = 0.1
    n_layers: int = 6
    n_heads: int = 4
    n_pool_queries: int = 4
    shrinkage_k: float = 20.0

    # FR-022 hardcoded constants — kept on the dataclass for traceability.
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


def _run_preflight(config: TrainEncoderConfig) -> None:
    if not config.vocab_path.exists():
        raise _PreFlightError(
            f"Vocabulary file not found: {config.vocab_path}. "
            "Run python -m sealed build-vocab first.",
            exit_code=2,
        )
    if not config.cards_played_path.exists() or config.cards_played_path.stat().st_size == 0:
        raise _PreFlightError(
            f"cards-played.txt missing or empty: {config.cards_played_path}. "
            "Run python -m sealed match-outcomes to collect data first.",
            exit_code=3,
        )
    if not config.cards_folder.is_dir() or not any(config.cards_folder.rglob("*.txt")):
        raise _PreFlightError(
            f"Cards folder is empty or missing: {config.cards_folder}. "
            "Run python -m price_predictor convert to build the corpus.",
            exit_code=4,
        )
    # n_pool_queries / d_model relationship is validated again inside
    # SealedEncoderConfig.__post_init__, but we surface it as exit code 6
    # at the CLI handler level (the dataclass raises ValueError).
    if D_MODEL % config.n_pool_queries != 0:
        raise _PreFlightError(
            f"d_model ({D_MODEL}) must be divisible by n_pool_queries "
            f"({config.n_pool_queries})",
            exit_code=6,
        )


# ── Best-checkpoint helper ──────────────────────────────────────────────


@dataclass
class _BestCheckpoint:
    """Track best-by-val-loss state through training. Mirrors the shape of
    ``price_predictor.application.train_transformer._BestCheckpoint``.

    Follow-up: when a third training entry point arrives, extract this to
    a shared utility (research.md "Third-instance check").
    """

    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_state_dict: dict | None = None

    def update(self, model: SealedEncoderModel, epoch: int, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return True
        return False


# ── Training loop ──────────────────────────────────────────────────────


def _make_optimizer(
    model: SealedEncoderModel, lr: float, total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, total_steps // 20)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _train_epoch(
    model: SealedEncoderModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    loss_fn = torch.nn.MSELoss()
    for ids, mask, label in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        prediction = model(ids, mask)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_loss(
    model: SealedEncoderModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    loss_fn = torch.nn.MSELoss()
    for ids, mask, label in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        label = label.to(device)
        prediction = model(ids, mask)
        total_loss += float(loss_fn(prediction, label).item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(config: TrainEncoderConfig) -> Path:
    """Execute the train-encoder pipeline.

    Returns the path of the saved ``latest.pt``. Raises ``_PreFlightError``
    on pre-flight failure (CLI handler maps to exit codes) and
    ``CorpusInconsistencyError`` on missing cards.
    """
    _run_preflight(config)

    _log(f"Reading {config.cards_played_path}")
    counts = _aggregate_counts(iter_rows(config.cards_played_path))
    _log(f"Aggregated counts for {len(counts)} card(s)")

    labels = _build_winnability_map(counts, config.shrinkage_k)
    _log(
        f"Built winnability map: {len(labels)} card(s) with wins_when_in_deck > 0",
    )

    win_rates_path = Path("output/sealed") / _WIN_RATES_FILENAME
    _write_win_rates(labels, win_rates_path)
    _log(f"Wrote per-card win-rates snapshot to {win_rates_path}")

    locator = ConvertedCardLocator(config.cards_folder)
    _check_corpus_consistency(labels, locator)

    train_names, val_names = _split_cards(
        labels, val_fraction=config.val_fraction, seed=config.random_seed,
    )
    if not train_names or not val_names:
        raise RuntimeError(
            f"Not enough cards for a train/val split: "
            f"train={len(train_names)}, val={len(val_names)}"
        )
    _log(f"Card-level split: {len(train_names)} train / {len(val_names)} val")

    tokenizer = load_tokenizer(config.vocab_path)
    max_seq_len = _measure_max_seq_len(labels.keys(), locator, tokenizer)
    _log(f"max_seq_len = {max_seq_len} (rounded to multiple of 8)")

    encoder_config = SealedEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        ff_dim=FF_DIM,
        max_seq_len=max_seq_len,
        dropout=config.dropout,
        n_pool_queries=config.n_pool_queries,
    )

    torch.manual_seed(config.random_seed)
    model = SealedEncoderModel(encoder_config)
    device = _select_device()
    model.to(device)
    _log(f"Sealed encoder on {device}")

    train_ds = _WinnabilityDataset(
        train_names, labels, tokenizer, locator, max_seq_len,
    )
    val_ds = _WinnabilityDataset(
        val_names, labels, tokenizer, locator, max_seq_len,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    total_steps = max(1, config.epochs * len(train_loader))
    optimizer, scheduler = _make_optimizer(model, config.lr, total_steps)

    best = _BestCheckpoint()
    epochs_since_best = 0
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = _eval_loss(model, val_loader, device)
        improved = best.update(model, epoch, val_loss)
        marker = "*" if improved else " "
        _log(
            f"Epoch {epoch}/{config.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  best={marker}"
        )
        if improved:
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= config.patience:
                _log(f"Early stop after {epochs_since_best} epochs without improvement.")
                break

    if best.best_state_dict is not None:
        model.load_state_dict(best.best_state_dict)
        _log(f"Restored best epoch {best.best_epoch} (val_loss={best.best_val_loss:.4f})")

    store = SealedEncoderStore()
    saved = store.save_encoder(model, encoder_config, config.model_output_dir)
    latest = config.model_output_dir / "latest.pt"
    _log(f"Best epoch: {best.best_epoch}, val_loss: {best.best_val_loss:.4f}. Saved {saved}")
    return latest
