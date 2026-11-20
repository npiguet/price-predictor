"""Training use case for the one-shot sealed deck picker.

REINFORCE-from-random-init against a frozen scorer, with a per-pool
empirical-mean baseline, an entropy bonus on a val-reward-driven decay
schedule, and an auxiliary pool-quality head (spec 017).

shared-trainer (resolved, spec 018): four trainers now share the resume /
bootstrap / best-vs-latest persistence / fail-fast width-check / early-stop
scaffolding — ``train_encoder``, ``train_scorer``, ``train_picker``, and
``draft.application.train_draft_agent`` (supervised CE + MC-regression MSE).
The fourth arrival was *not* REINFORCE-style (the original extraction trigger),
and it diverges yet again on dataset shape (per-(draft,seat,pick) typed-token
states) and on carrying two heads over two seat-subsets. The genuinely
identical pieces remain small and stable — the warmup ``LambdaLR`` lambda and
the per-group ``clip_grad_norm_`` — while loss/metric/val bodies differ across
all four. Decision (re-evaluated in specs/018-draft-agent, research §Third-
instance check): **defer extraction**; a generic loop over four loss/metric/
shape variants would be speculative coupling (Simplicity-First). Each trainer
copies the two small helpers by following the pattern.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sealed.application.picker_pools import PreparedPool, _prepare_pools, _split_pools
from sealed.application.picker_resume import (
    _check_picker_width,
    _reference_picker,
    _restored_optimizer,
    _resume_or_build_picker,
)
from sealed.application.picker_scoring import _load_scorer, _select_device
from sealed.application.picker_training_loop import PickerTrainingLoop, _log
from sealed.application.picker_training_run import (
    BestSoFar,
    CheckpointTargets,
    TrainingModels,
    TrainingRun,
)
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.domain.picker_model import PickerConfig, PickerModel
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.picker_store import PickerStore
from sealed.infrastructure.pool_file_reader import parse_pools

RANDOM_SEED = 42  # hardcoded (FR-018); governs init, shuffle, sampling, split.


@dataclass
class TrainPickerConfig:
    pools_path: Path
    scorer_checkpoint: Path
    cards_path: Path
    checkpoint_dir: Path
    auditor_scorer_checkpoint: Path | None = None
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    resume: Path | None = None
    picker_checkpoint: Path | None = None
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.0
    aux_weight: float = 0.1
    normalize_advantage: bool = False
    objective: str = "reinforce"
    topk: int = 16
    batch_size: int = 16
    n_samples: int = 64
    temperature: float = 1.0
    entropy_coef: float = 0.01
    entropy_decay_after: int = 5
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    epochs: int = 100
    val_fraction: float = 0.2
    patience: int = 10
    evals_per_epoch: int = 1
    kl_coef: float = 0.0

    def picker_config(self, embedding_dim: int) -> PickerConfig:
        """Build the architecture config for a fresh picker sized to the cache."""
        return PickerConfig(
            embedding_dim=embedding_dim,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )


def _build_train_config(config: TrainPickerConfig) -> dict[str, Any]:
    """Flatten ``TrainPickerConfig`` to a JSON-friendly dict (Path -> str)."""
    raw = asdict(config)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in raw.items()}


@dataclass
class TrainPickerResult:
    model: PickerModel
    best_val_reward: float
    best_epoch: int
    best_path: Path


def _prepare_training_pools(
    config: TrainPickerConfig, scorer: SetTransformerScorer,
) -> tuple[list[PreparedPool], Path]:
    """Every usable pool, and the embeddings root they were read from; refuse
    a pools file with none, or a cache the scorer's width does not match."""
    locator = ConvertedCardLocator(
        config.cards_path, embeddings_path=config.embeddings_path,
    )
    prepared = _prepare_pools(parse_pools(config.pools_path), locator)
    if not prepared:
        raise ValueError(
            f"No usable pools in {config.pools_path} (every pool had fewer "
            f"than {NONLAND_DECK_SIZE} embeddable cards under {locator.embeddings_root})."
        )
    cache_width = prepared[0].dim
    _check_picker_width(
        scorer.config.d_model, cache_width,
        f"--scorer-checkpoint {config.scorer_checkpoint}", locator.embeddings_root,
    )
    return prepared, locator.embeddings_root


def _load_auditor(
    config: TrainPickerConfig, device: torch.device, cache_width: int,
    embeddings_root: Path,
) -> SetTransformerScorer | None:
    """The cross-scorer auditor when one is configured, checked against the
    cache's width; else None."""
    if config.auditor_scorer_checkpoint is None:
        return None
    auditor = _load_scorer(config.auditor_scorer_checkpoint, device)
    _check_picker_width(
        auditor.config.d_model, cache_width,
        f"--auditor-scorer-checkpoint {config.auditor_scorer_checkpoint}",
        embeddings_root,
    )
    _log("Auditor scorer loaded; cross-scorer audit enabled")
    return auditor


def _checkpoint_targets(config: TrainPickerConfig, store: PickerStore) -> CheckpointTargets:
    """The run directory, created, with its ``latest`` and timestamped best."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = config.checkpoint_dir / "latest.pt"
    best_path = config.checkpoint_dir / f"best_{timestamp}.pt"
    return CheckpointTargets(store, latest_path, best_path, _build_train_config(config))


class TrainPickerUseCase:
    """Train a one-shot picker via REINFORCE against a frozen scorer."""

    def execute(self, config: TrainPickerConfig) -> TrainPickerResult:
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        device = _select_device()
        store = PickerStore()

        t0 = time.monotonic()
        scorer = _load_scorer(config.scorer_checkpoint, device)
        embedding_dim = scorer.config.d_model
        _log(f"Loaded training scorer (d_model={embedding_dim}) on {device}")

        prepared, embeddings_root = _prepare_training_pools(config, scorer)
        auditor = _load_auditor(config, device, prepared[0].dim, embeddings_root)

        val_pools, train_pools = _split_pools(prepared, config.val_fraction)
        _log(
            f"Pools ready: {len(train_pools)} train + {len(val_pools)} val "
            f"({time.monotonic() - t0:.1f}s)"
        )

        resume = _resume_or_build_picker(config, store, embedding_dim)
        picker = resume.model.to(device)
        optimizer = _restored_optimizer(picker, config, resume, device)
        reference = _reference_picker(config, resume, picker, device)
        run = TrainingRun(
            models=TrainingModels(picker, scorer, auditor, reference, optimizer),
            config=config, device=device,
            train_pools=train_pools, val_pools=val_pools,
            checkpoints=_checkpoint_targets(config, store),
            rng=random.Random(RANDOM_SEED), start_epoch=resume.start_epoch,
        )
        best = BestSoFar(resume.best_val_reward, resume.start_epoch)
        last_epoch = PickerTrainingLoop(run, best).train()

        best_path = run.checkpoints.best_path
        _log(
            f"Done (last epoch {last_epoch}). Best val_reward={best.reward:.4f} "
            f"at epoch {best.epoch}; best checkpoint: {best_path}"
        )
        return TrainPickerResult(picker, best.reward, best.epoch, best_path)
