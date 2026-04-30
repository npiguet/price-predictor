"""Training use case for the deck scorer model."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from price_predictor.domain.entities import TransformerConfig
from price_predictor.domain.tokenizer import MtgTokenizer
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.match_data_loader import (
    EmbeddingTable,
    MatchTrainingExample,
    TrainingBatch,
    build_training_examples,
    collate_training_examples,
    load_match_outcomes,
)
from sealed.infrastructure.scorer_store import ScorerStore


def _log(message: str) -> None:
    """Print a timestamped progress line to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@dataclass
class TrainScorerConfig:
    outcomes_path: Path
    cards_path: Path
    checkpoint_dir: Path
    resume: Path | None = None
    scorer_checkpoint: Path | None = None
    encoder_checkpoint: Path = field(
        default_factory=lambda: Path("models/price-predictor/transformer/latest.pt"),
    )
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-5
    n_layers: int = 6
    n_heads: int = 4
    n_seeds: int = 4
    d_ff: int = 1088
    mlp_hidden: int = 256
    dropout: float = 0.2
    embedding_lr: float = 0.0
    patience: int = 5
    val_fraction: float = 0.2
    random_seed: int = 42
    encoder_chunk_size: int = 128
    max_grad_norm: float = 100.0

    @property
    def phase(self) -> Literal["A", "B"]:
        return "A" if self.embedding_lr == 0 else "B"

    def best_checkpoint_name(self) -> str:
        if self.phase == "B":
            return (
                f"best_phaseB_l{self.n_layers}_h{self.n_heads}_s{self.n_seeds}"
                f"_ff{self.d_ff}_mlp{self.mlp_hidden}"
                f"_lr{self.lr}_emblr{self.embedding_lr}.pt"
            )
        return (
            f"best_l{self.n_layers}_h{self.n_heads}_s{self.n_seeds}"
            f"_ff{self.d_ff}_mlp{self.mlp_hidden}_lr{self.lr}.pt"
        )

    def latest_checkpoint_name(self) -> str:
        return "latest_phaseB.pt" if self.phase == "B" else "latest.pt"

    def scorer_config(self) -> ScorerConfig:
        return ScorerConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_seeds=self.n_seeds,
            d_ff=self.d_ff,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )


@dataclass
class TrainingMetrics:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    embedding_drifts: list[float] = field(default_factory=list)


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    grad_norms_mean: dict[str, float]
    grad_norms_max: dict[str, float]


@dataclass
class ValidationResult:
    loss: float
    accuracy: float
    score_winner_mean: float
    score_winner_std: float
    score_loser_mean: float
    score_loser_std: float


@dataclass
class ResumeState:
    model: SetTransformerScorer
    start_epoch: int
    best_val_accuracy: float
    optimizer_state: dict | None
    encoder_state_dict: dict | None = None
    encoder_config: dict | None = None
    phase: Literal["A", "B"] = "A"


@dataclass
class TrainScorerResult:
    model: SetTransformerScorer
    metrics: TrainingMetrics
    embedding_table: EmbeddingTable


@dataclass
class ReferenceBatch:
    """The fixed set of unique cards from Phase B step 0 used for the
    ``embedding_drift`` metric (Decision §5, FR-012)."""

    card_indices: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    step0_text_vectors: torch.Tensor


@dataclass
class _TrainingContext:
    model: SetTransformerScorer
    optimizer: torch.optim.Optimizer
    embedding_table: EmbeddingTable
    train_loader: DataLoader
    val_loader: DataLoader
    latest_path: Path
    best_path: Path
    start_epoch: int
    best_val_accuracy: float
    device: torch.device
    train_config: dict[str, Any]
    encoder: CardPriceTransformerModel | None = None
    tokenizer: MtgTokenizer | None = None
    card_token_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    locator: ConvertedCardLocator | None = None
    idx_to_name: dict[int, str] | None = None
    reference_batch: ReferenceBatch | None = None
    max_seq_len: int | None = None
    encoder_chunk_size: int = 128
    max_grad_norm: float = 100.0


class _BatchStats:
    """Accumulate pairwise-BCE batch statistics across a training or validation epoch."""

    def __init__(self) -> None:
        self._total_loss = 0.0
        self._n_batches = 0
        self._correct = 0
        self._total = 0
        self._winner_scores: list[torch.Tensor] = []
        self._loser_scores: list[torch.Tensor] = []

    def add(
        self,
        loss: torch.Tensor,
        score_winner: torch.Tensor,
        score_loser: torch.Tensor,
    ) -> None:
        self._total_loss += loss.item()
        self._n_batches += 1
        self._correct += (score_winner > score_loser).sum().item()
        self._total += score_winner.size(0)
        self._winner_scores.append(score_winner.detach().squeeze(1))
        self._loser_scores.append(score_loser.detach().squeeze(1))

    @property
    def mean_loss(self) -> float:
        return self._total_loss / max(self._n_batches, 1)

    @property
    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def score_summary(self) -> tuple[float, float, float, float]:
        winners = torch.cat(self._winner_scores) if self._winner_scores else torch.zeros(1)
        losers = torch.cat(self._loser_scores) if self._loser_scores else torch.zeros(1)
        return (
            winners.mean().item(), winners.std().item(),
            losers.mean().item(), losers.std().item(),
        )


class TrainScorerUseCase:
    """Train the deck scorer on match outcome data."""

    def execute(self, config: TrainScorerConfig) -> TrainScorerResult:
        store = ScorerStore()
        ctx = self._prepare(config, store)

        metrics = TrainingMetrics()
        best_val_accuracy = ctx.best_val_accuracy
        epochs_since_peak = 0

        _log(f"Starting training loop ({config.epochs} epochs)")
        for epoch in range(ctx.start_epoch, ctx.start_epoch + config.epochs):
            train_stats = _train_one_epoch(ctx)
            metrics.train_losses.append(train_stats.loss)

            val = _validate(
                ctx.model, ctx.embedding_table, ctx.val_loader, ctx.device,
            )
            metrics.val_losses.append(val.loss)
            metrics.val_accuracies.append(val.accuracy)
            if ctx.encoder is not None and ctx.reference_batch is not None:
                metrics.embedding_drifts.append(_embedding_drift(ctx))
            _print_epoch_report(epoch, train_stats, val, metrics)
            new_best, best_val_accuracy = self._persist_checkpoint(
                ctx, store, epoch, val.accuracy, best_val_accuracy,
            )
            if new_best:
                epochs_since_peak = 0
            else:
                epochs_since_peak += 1
                if epochs_since_peak >= config.patience:
                    _log(
                        f"Early stop: {epochs_since_peak} epochs without "
                        f"new peak val_acc (--patience={config.patience})"
                    )
                    break

        return TrainScorerResult(
            model=ctx.model,
            metrics=metrics,
            embedding_table=ctx.embedding_table,
        )

    def _prepare(
        self, config: TrainScorerConfig, store: ScorerStore,
    ) -> _TrainingContext:
        t0 = time.monotonic()
        _log(f"Loading dataset from {config.outcomes_path}")
        train_examples, val_examples, embedding_table = _load_dataset(config)
        _log(
            f"Dataset ready: {len(train_examples)} train + {len(val_examples)} val "
            f"examples, {embedding_table.num_cards} unique cards "
            f"({time.monotonic() - t0:.1f}s)",
        )

        resume = _resume_or_build_model(config, store)
        if config.resume is not None:
            _log(f"Resumed from checkpoint at epoch {resume.start_epoch}")
        elif config.scorer_checkpoint is not None:
            _log(f"Bootstrapped scorer from {config.scorer_checkpoint}")
        else:
            _log(f"Built fresh scorer model (n_layers={config.n_layers}, dropout={config.dropout})")

        _log("Computing per-card feature normalization stats")
        _set_normalization_stats(resume.model, train_examples, embedding_table)

        device = _select_device()
        resume.model.to(device)
        embedding_table.to(device)

        encoder: CardPriceTransformerModel | None = None
        tokenizer: MtgTokenizer | None = None
        max_seq_len: int | None = None
        locator: ConvertedCardLocator | None = None
        idx_to_name: dict[int, str] | None = None
        if config.phase == "B":
            encoder, tokenizer, max_seq_len = _build_phase_b_encoder(config, resume)
            encoder.to(device)
            locator = ConvertedCardLocator(config.cards_path)
            idx_to_name = {
                idx: name for name, idx in embedding_table.name_to_idx.items()
            }
            _log(
                f"Phase B encoder loaded (d_model={encoder.config.d_model}, "
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
        )

    def _persist_checkpoint(
        self,
        ctx: _TrainingContext,
        store: ScorerStore,
        epoch: int,
        val_accuracy: float,
        current_best: float,
    ) -> tuple[bool, float]:
        encoder_state_dict = (
            ctx.encoder.state_dict() if ctx.encoder is not None else None
        )
        encoder_config = (
            asdict(ctx.encoder.config) if ctx.encoder is not None else None
        )
        store.save_checkpoint(
            ctx.model, ctx.optimizer, epoch, current_best,
            ctx.model.config, ctx.latest_path,
            train_config=ctx.train_config,
            encoder_state_dict=encoder_state_dict,
            encoder_config=encoder_config,
        )
        new_best = val_accuracy > current_best
        if new_best:
            current_best = val_accuracy
            store.save_checkpoint(
                ctx.model, ctx.optimizer, epoch, current_best,
                ctx.model.config, ctx.best_path,
                train_config=ctx.train_config,
                encoder_state_dict=encoder_state_dict,
                encoder_config=encoder_config,
            )
        return new_best, current_best


def _load_dataset(
    config: TrainScorerConfig,
) -> tuple[list[MatchTrainingExample], list[MatchTrainingExample], EmbeddingTable]:
    """Load all outcomes, build a shared EmbeddingTable, and randomly split examples."""
    t0 = time.monotonic()
    outcomes = load_match_outcomes(config.outcomes_path)
    _log(f"  parsed {len(outcomes)} match outcomes ({time.monotonic() - t0:.1f}s)")

    t1 = time.monotonic()
    examples, embedding_table = build_training_examples(outcomes, config.cards_path)
    _log(
        f"  built {len(examples)} training examples "
        f"(loaded card embeddings, {time.monotonic() - t1:.1f}s)",
    )
    if len(examples) < 2:
        return examples, [], embedding_table
    train_examples, val_examples = train_test_split(
        examples,
        test_size=config.val_fraction,
        random_state=config.random_seed,
        shuffle=True,
    )
    return train_examples, val_examples, embedding_table


def _resume_or_build_model(
    config: TrainScorerConfig, store: ScorerStore,
) -> ResumeState:
    """Build the scorer model and decide what to seed it (and the encoder)
    from. Cross-phase resume MUST raise per FR-004 (the CLI catches that
    earlier; this is the second layer of defense)."""
    requested_phase = config.phase

    if config.resume is not None:
        checkpoint = store.load_checkpoint(config.resume)
        ckpt_phase: Literal["A", "B"] = (
            "B" if checkpoint.encoder_state_dict is not None else "A"
        )
        if ckpt_phase != requested_phase:
            raise ValueError(
                f"--resume {config.resume} is a Phase {ckpt_phase} checkpoint "
                f"but --embedding-lr {config.embedding_lr} requests Phase "
                f"{requested_phase}. Use --scorer-checkpoint to start a fresh "
                "Phase B run from a Phase A checkpoint."
            )
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        return ResumeState(
            model=model,
            start_epoch=checkpoint.epoch + 1,
            best_val_accuracy=checkpoint.best_val_accuracy,
            optimizer_state=checkpoint.optimizer_state_dict or None,
            encoder_state_dict=checkpoint.encoder_state_dict,
            encoder_config=checkpoint.encoder_config,
            phase=ckpt_phase,
        )

    if config.scorer_checkpoint is not None:
        checkpoint = store.load_checkpoint(config.scorer_checkpoint)
        model = SetTransformerScorer(checkpoint.config)
        model.load_state_dict(checkpoint.model_state_dict)
        return ResumeState(
            model=model,
            start_epoch=0,
            best_val_accuracy=-1.0,
            optimizer_state=None,  # scorer-checkpoint bootstrap resets training-loop state
            encoder_state_dict=None,  # encoder weights come from --encoder-checkpoint
            encoder_config=None,
            phase=requested_phase,
        )

    return ResumeState(
        model=SetTransformerScorer(config.scorer_config()),
        start_epoch=0,
        best_val_accuracy=-1.0,
        optimizer_state=None,
        phase=requested_phase,
    )


def _build_phase_b_encoder(
    config: TrainScorerConfig, resume: ResumeState,
) -> tuple[CardPriceTransformerModel, MtgTokenizer, int]:
    """Construct the Phase B encoder + tokenizer + max-seq-len.

    On `--resume <phaseB>.pt`, the resumed checkpoint is self-contained — the
    encoder is rebuilt from `encoder_config` and weights from
    `encoder_state_dict`, no dependency on the price-predictor `latest.pt`
    being unchanged. Otherwise (`--scorer-checkpoint` bootstrap), encoder
    weights come from `config.encoder_checkpoint`.
    """
    from price_predictor.infrastructure.tokenizer_store import load_tokenizer
    from price_predictor.infrastructure.transformer_store import load_model

    if resume.encoder_state_dict is not None and resume.encoder_config is not None:
        cfg = TransformerConfig(**resume.encoder_config)
        encoder = CardPriceTransformerModel(cfg)
        encoder.load_state_dict(resume.encoder_state_dict)
    else:
        encoder, cfg = load_model(config.encoder_checkpoint)

    vocab_path = config.encoder_checkpoint.parent / "vocab.txt"
    tokenizer = load_tokenizer(vocab_path)
    return encoder, tokenizer, cfg.max_seq_len


def _build_optimizer(
    model: SetTransformerScorer,
    config: TrainScorerConfig,
    encoder: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer.

    Phase A: a single ``"scorer"`` parameter group at ``config.lr``.
    Phase B (``encoder`` supplied): two groups — scorer at ``config.lr`` and
    encoder at ``config.embedding_lr`` — for per-group clipping (FR-005a, FR-008).
    The ``"name"`` key on each group is ignored by AdamW but read by
    ``_train_one_epoch`` to label the per-group gradient norm.
    """
    groups: list[dict[str, Any]] = [
        {"params": list(model.parameters()), "lr": config.lr, "name": "scorer"},
    ]
    if encoder is not None:
        groups.append(
            {"params": list(encoder.parameters()), "lr": config.embedding_lr,
             "name": "encoder"},
        )
    return torch.optim.AdamW(groups)


def _make_loaders(
    train_examples: list[MatchTrainingExample],
    val_examples: list[MatchTrainingExample],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_training_examples,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_training_examples,
    )
    return train_loader, val_loader


def _select_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Training scorer on device: {device}")
    return device


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> None:
    """Move Adam momentum/variance buffers to ``device`` after ``load_state_dict``."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def _batch_to_device(batch: TrainingBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        winner_indices=batch.winner_indices.to(device),
        loser_indices=batch.loser_indices.to(device),
        winner_mask=batch.winner_mask.to(device),
        loser_mask=batch.loser_mask.to(device),
    )


def _train_one_epoch(ctx: _TrainingContext) -> EpochStats:
    ctx.model.train()
    if ctx.encoder is not None:
        # Phase B runs the encoder in eval mode so dropout doesn't add
        # stochasticity. The cached `.npz` files the Phase A scorer was tuned
        # against were produced under `encoder.eval()`; turning dropout on
        # here would feed the scorer noisier vectors than it learned to
        # score against, dragging early Phase B val_acc down independently
        # of any encoder weight movement. `eval()` no-ops dropout/BatchNorm
        # only — it does not disable autograd, so the encoder still trains.
        ctx.encoder.eval()
    stats = _BatchStats()
    norms_per_batch: dict[str, list[float]] = {}

    for batch in ctx.train_loader:
        batch = _batch_to_device(batch, ctx.device)
        ctx.optimizer.zero_grad()
        if ctx.encoder is not None:
            _phase_b_inject_encoder(ctx, batch)
        score_winner, score_loser = _score_batch(ctx.model, ctx.embedding_table, batch)
        loss = _pairwise_bce(score_winner, score_loser)
        loss.backward()
        per_batch = _clip_per_group(ctx.optimizer, max_norm=ctx.max_grad_norm)
        for name, value in per_batch.items():
            norms_per_batch.setdefault(name, []).append(value)
        ctx.optimizer.step()
        ctx.embedding_table.clear_live_weight()
        stats.add(loss, score_winner, score_loser)

    grad_norms_mean = {
        name: sum(values) / len(values) for name, values in norms_per_batch.items()
    }
    grad_norms_max = {
        name: max(values) for name, values in norms_per_batch.items()
    }

    return EpochStats(
        loss=stats.mean_loss,
        accuracy=stats.accuracy,
        grad_norms_mean=grad_norms_mean,
        grad_norms_max=grad_norms_max,
    )


def _phase_b_inject_encoder(ctx: _TrainingContext, batch: TrainingBatch) -> None:
    """Run the encoder over the batch's unique cards and splice the resulting
    text vectors into the embedding-table rows referenced by this batch
    (FR-007, Decision §4). The encoder forward runs once per unique card; the
    autograd graph released after ``optimizer.step()`` ensures the within-batch
    cache does not survive between training steps."""
    assert ctx.encoder is not None
    assert ctx.tokenizer is not None
    assert ctx.card_token_cache is not None
    assert ctx.locator is not None
    assert ctx.idx_to_name is not None
    assert ctx.max_seq_len is not None

    unique = torch.unique(
        torch.cat([batch.winner_indices.flatten(), batch.loser_indices.flatten()]),
    )
    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    for idx_t in unique:
        idx = int(idx_t.item())
        cached = ctx.card_token_cache.get(idx)
        if cached is None:
            cached = _tokenize_card(ctx, idx)
            ctx.card_token_cache[idx] = cached
        input_ids_list.append(cached[0])
        attention_mask_list.append(cached[1])

    input_ids = torch.stack(input_ids_list).to(ctx.device)
    attention_mask = torch.stack(attention_mask_list).to(ctx.device)
    text_vectors = _encode_chunked(
        ctx.encoder, input_ids, attention_mask,
        chunk_size=ctx.encoder_chunk_size, training=True,
    )
    ctx.embedding_table.set_text_vectors(unique.to(ctx.device), text_vectors)

    if ctx.reference_batch is None:
        # The encoder is already in eval mode for the entire Phase B training
        # loop (no dropout), so the forward we just computed for the scorer's
        # loss is identical to the forward `_embedding_drift` will run each
        # epoch on the same inputs. Snapshot it directly — no extra forward
        # needed.
        ctx.reference_batch = ReferenceBatch(
            card_indices=unique.detach().clone().to(ctx.device),
            input_ids=input_ids.detach().clone(),
            attention_mask=attention_mask.detach().clone(),
            step0_text_vectors=text_vectors.detach().clone(),
        )
        _log(f"Reference batch captured: {unique.numel()} unique cards (Phase B step 0)")


def _encode_chunked(
    encoder: CardPriceTransformerModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    chunk_size: int,
    training: bool,
) -> torch.Tensor:
    """Run the encoder forward in fixed-size chunks, applying gradient
    checkpointing per chunk during training so peak activation memory is
    bounded by ``chunk_size`` cards instead of the full unique-card count.

    A typical Phase B step sees ~1000–3000 unique cards in a single batch
    (`batch_size * cards_per_deck * 2` minus duplicates). One forward pass at
    that size on a 6-layer transformer overflows commodity GPUs; chunking
    keeps each forward bounded by ``chunk_size``. Gradient checkpointing
    drops activations after each chunk and recomputes them during backward,
    trading ~1.5× compute for ~10× less memory.
    """
    n = input_ids.size(0)
    if n <= chunk_size or not training:
        return encoder._encode_and_pool(input_ids, attention_mask)
    outputs: list[torch.Tensor] = []
    for start in range(0, n, chunk_size):
        end = start + chunk_size
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        chunk_out = torch.utils.checkpoint.checkpoint(
            encoder._encode_and_pool, chunk_ids, chunk_mask,
            use_reentrant=False,
        )
        outputs.append(chunk_out)
    return torch.cat(outputs, dim=0)


def _tokenize_card(
    ctx: _TrainingContext, idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert ctx.tokenizer is not None
    assert ctx.locator is not None
    assert ctx.idx_to_name is not None
    assert ctx.max_seq_len is not None
    name = ctx.idx_to_name[idx]
    converted = ctx.locator.load_text(name)
    if converted is None:
        raise FileNotFoundError(
            f"Phase B: no converted card .txt for {name!r} under "
            f"{ctx.locator._cards_path}",
        )
    text = converted.without_name_line()
    input_ids, attention_mask = ctx.tokenizer.encode(text, ctx.max_seq_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


def _clip_per_group(
    optimizer: torch.optim.Optimizer, *, max_norm: float,
) -> dict[str, float]:
    """Clip each parameter group at ``max_norm`` and return the pre-clip
    L2 norms. The returned values are the diagnostic signal — post-clip
    norms are bounded by ``max_norm`` and carry no shape information per
    FR-012."""
    norms: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("name", "group")
        pre_clip = torch.nn.utils.clip_grad_norm_(group["params"], max_norm=max_norm)
        norms[name] = float(pre_clip)
    return norms


def _validate(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    val_loader: DataLoader,
    device: torch.device,
) -> ValidationResult:
    """Compute validation loss, accuracy, and score statistics."""
    model.eval()
    stats = _BatchStats()
    with torch.no_grad():
        for batch in val_loader:
            batch = _batch_to_device(batch, device)
            score_winner, score_loser = _score_batch(model, embedding_table, batch)
            loss = _pairwise_bce(score_winner, score_loser)
            stats.add(loss, score_winner, score_loser)

    w_mean, w_std, l_mean, l_std = stats.score_summary()
    return ValidationResult(
        loss=stats.mean_loss,
        accuracy=stats.accuracy,
        score_winner_mean=w_mean,
        score_winner_std=w_std,
        score_loser_mean=l_mean,
        score_loser_std=l_std,
    )


def _score_batch(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    batch: TrainingBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    winner_cards = embedding_table(batch.winner_indices)
    loser_cards = embedding_table(batch.loser_indices)
    return (
        model(winner_cards, batch.winner_mask),
        model(loser_cards, batch.loser_mask),
    )


def _pairwise_bce(
    score_winner: torch.Tensor, score_loser: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        score_winner - score_loser,
        torch.ones_like(score_winner),
    )


def _set_normalization_stats(
    model: SetTransformerScorer,
    examples: list[MatchTrainingExample],
    embedding_table: EmbeddingTable,
) -> None:
    """Compute per-feature mean and std for the deterministic-feature slice."""
    if not examples:
        return
    flat = torch.cat(
        [ex.winner_indices for ex in examples]
        + [ex.loser_indices for ex in examples]
    )
    mean, std = embedding_table.deterministic_feature_stats(flat)
    model.feat_mean.copy_(mean)
    model.feat_std.copy_(std)


def _embedding_drift(ctx: _TrainingContext) -> float:
    """Mean L2 distance between current encoder text vectors on the reference
    batch and their step-0 values (Decision §5, FR-012). Computed under
    ``model.eval()`` so dropout doesn't muddle the signal — the step-0
    reference is captured the same way."""
    assert ctx.encoder is not None
    assert ctx.reference_batch is not None
    was_training = ctx.encoder.training
    ctx.encoder.eval()
    try:
        with torch.no_grad():
            current = _encode_chunked(
                ctx.encoder,
                ctx.reference_batch.input_ids,
                ctx.reference_batch.attention_mask,
                chunk_size=ctx.encoder_chunk_size,
                training=False,
            )
        diff = current - ctx.reference_batch.step0_text_vectors
        return float(diff.norm(dim=-1).mean().item())
    finally:
        if was_training:
            ctx.encoder.train()


def _build_train_config(config: TrainScorerConfig) -> dict[str, Any]:
    """Flatten ``TrainScorerConfig`` to a JSON-friendly dict for checkpoint persistence.

    ``Path``-typed values become strings; ``None`` is preserved (FR-009,
    Decision §1, contracts/checkpoint-format.md §train_config Schema).
    """
    raw = asdict(config)
    return {k: str(v) if isinstance(v, Path) else v for k, v in raw.items()}


def _print_epoch_report(
    epoch: int,
    train_stats: EpochStats,
    val: ValidationResult,
    metrics: TrainingMetrics,
) -> None:
    drift_str = ""
    if metrics.embedding_drifts:
        drift_str = f"  embedding_drift={metrics.embedding_drifts[-1]:.6f}"
    grad_str = "  ".join(
        f"{k}=mean({train_stats.grad_norms_mean[k]:.4f})/max({train_stats.grad_norms_max[k]:.4f})"
        for k in train_stats.grad_norms_mean
    )
    _log(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_stats.loss:.4f}  "
        f"train_acc={train_stats.accuracy:.4f}  "
        f"val_loss={val.loss:.4f}  "
        f"val_acc={val.accuracy:.4f}"
        f"{drift_str}"
    )
    _log(
        f"  scores: "
        f"winner={val.score_winner_mean:.4f}±{val.score_winner_std:.4f}  "
        f"loser={val.score_loser_mean:.4f}±{val.score_loser_std:.4f}"
    )
    _log(f"  grad_norms: {grad_str}")
